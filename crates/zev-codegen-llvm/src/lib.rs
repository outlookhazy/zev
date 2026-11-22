//! Direct LLVM 23 IR + Freestanding Bare-Metal ARM Cortex-M3 & ESP32-C6 (`rv32imac`) Backend (`zev-codegen-llvm`).
//!
//! Lowers [`zev_mir::MirModule`] directly into modern LLVM 23 opaque-pointer SSA IR (`.ll`) with
//! zero C, zero `.s` files, and zero libc:
//! - **ARM Cortex-M3 (`thumbv7m-none-eabi`, `lm3s6965evb`)**: synthesizes `.isr_vector`,
//!   `@Reset_Handler`, `@SysTick_Handler`, linker script, `.elf`, and runs in `qemu-system-arm`.
//! - **Espressif ESP32-C6 (`riscv32-unknown-none-elf`, `rv32imac` Direct-Boot)**: synthesizes
//!   the `0xAEDB041D` x2 Direct-Boot header, `@_start` at `0x42000008`, `TIMG0`/`TIMG1`/`LP_WDT`/`SWD`
//!   watchdog disarm, 16 MHz `SYSTIMER`, `USB_DEVICE` (`/dev/ttyACM0`) serial console, HP-SRAM
//!   `.iram1` WS2812B RGB driver (`GPIO8`), and `BOOT` button (`GPIO9`) into a flashable `.bin`!

pub mod avr;
pub mod esp32c6;
pub mod mir_codegen;
pub mod nvic;
pub mod runtime_ir;
pub mod stack_sizes;
pub mod toolchain;

use std::path::Path;

pub use esp32c6::ESP32C6_DIRECT_BOOT_MAGIC;
pub use mir_codegen::emit_llvm_ir_with_options;
pub use nvic::NvicMap;
pub use stack_sizes::{HardwareStackProof, analyze_hardware_stack_high_water};
pub use toolchain::{
    AvrImageResult, CodegenError, CodegenOptions, ElfBuildResult, ElfSectionSizes, ElfSymbolAudit,
    Esp32C6ImageResult, QemuExecOutput, QemuRunOptions, QemuRunResult, TargetArch,
    compile_ll_to_elf, compile_ll_to_obj, flash_avr_uno_bin, flash_esp32c6_bin,
    inspect_elf_sizes, inspect_elf_symbols, link_obj_to_elf, objcopy_elf_to_avr_bin,
    objcopy_elf_to_esp32c6_bin, optimize_ll_with_opt, run_elf_in_qemu, synthesize_linker_script,
    verify_ll_with_opt,
};

/// Emit freestanding ARM Cortex-M3 (`thumbv7m-none-eabi`) LLVM 23 `.ll` IR from `MirModule`.
pub fn emit_llvm_ir(module: &zev_mir::MirModule) -> Result<String, CodegenError> {
    emit_llvm_ir_with_options(module, &CodegenOptions::default())
}

/// Emit LLVM 23 `.ll` IR, synthesize `link.ld`, compile via `/bin/llc`, link via `/bin/ld.lld`,
/// and verify `/bin/llvm-size` + zero-heap freestanding ELF32 symbol compliance.
pub fn build_elf(
    module: &zev_mir::MirModule,
    out_elf: &Path,
    opts: &CodegenOptions,
) -> Result<ElfBuildResult, CodegenError> {
    let ll = emit_llvm_ir_with_options(module, opts)?;
    compile_ll_to_elf(&ll, out_elf, opts)
}

/// Convenience wrapper for `build_elf` using default Cortex-M3 (`lm3s6965evb`) options.
pub fn build_elf_default(
    module: &zev_mir::MirModule,
    out_elf: &Path,
) -> Result<ElfBuildResult, CodegenError> {
    build_elf(module, out_elf, &CodegenOptions::default())
}

/// Build a freestanding ESP32-C6 (`riscv32imac`) `.elf` and Direct-Boot `.bin` flash image
/// (verifying zero undefined/heap symbols and `0xAEDB041D` x2 Direct-Boot magic at offset `0x0`).
pub fn build_esp32c6_image(
    module: &zev_mir::MirModule,
    out_elf: &Path,
    out_bin: &Path,
) -> Result<Esp32C6ImageResult, CodegenError> {
    build_esp32c6_image_with_options(module, out_elf, out_bin, &CodegenOptions::esp32c6())
}

/// Build a freestanding ESP32-C6 (`riscv32imac`) `.elf` and Direct-Boot `.bin` with custom [`CodegenOptions`].
pub fn build_esp32c6_image_with_options(
    module: &zev_mir::MirModule,
    out_elf: &Path,
    out_bin: &Path,
    opts: &CodegenOptions,
) -> Result<Esp32C6ImageResult, CodegenError> {
    let elf = build_elf(module, out_elf, opts)?;
    let (bin_size, magic_header) = objcopy_elf_to_esp32c6_bin(&elf.elf_path, out_bin)?;
    Ok(Esp32C6ImageResult {
        elf,
        bin_path: out_bin.to_path_buf(),
        bin_size,
        magic_header,
    })
}

/// Build a freestanding Microchip ATmega328P (`avr`, Arduino Uno R3 Optiboot) `.elf` and Flash `.bin`
/// image (verifying zero undefined/heap symbols, reset `jmp` vector, and `< 1,792 B` static SRAM).
pub fn build_avr_uno_image(
    module: &zev_mir::MirModule,
    out_elf: &Path,
    out_bin: &Path,
) -> Result<AvrImageResult, CodegenError> {
    build_avr_uno_image_with_options(module, out_elf, out_bin, &CodegenOptions::avr_uno())
}

/// Build a freestanding Microchip ATmega328P (`avr`) `.elf` and Flash `.bin` with custom [`CodegenOptions`].
pub fn build_avr_uno_image_with_options(
    module: &zev_mir::MirModule,
    out_elf: &Path,
    out_bin: &Path,
    opts: &CodegenOptions,
) -> Result<AvrImageResult, CodegenError> {
    let elf = build_elf(module, out_elf, opts)?;
    let (bin_size, sram_static_bytes) = objcopy_elf_to_avr_bin(&elf.elf_path, out_bin)?;
    Ok(AvrImageResult {
        elf,
        bin_path: out_bin.to_path_buf(),
        bin_size,
        sram_static_bytes,
    })
}

/// Build a freestanding Cortex-M3 `.elf` from `module` inside `out_elf` and execute it
/// live in `/bin/qemu-system-arm` (`lm3s6965evb`), returning binary sizes and UART0 output.
pub fn run_qemu_cortex_m(
    module: &zev_mir::MirModule,
    out_elf: &Path,
    opts: &CodegenOptions,
    qemu_opts: &QemuRunOptions,
) -> Result<QemuRunResult, CodegenError> {
    let build = build_elf(module, out_elf, opts)?;
    let exec = run_elf_in_qemu(&build.elf_path, qemu_opts)?;
    Ok(QemuRunResult { build, exec })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    fn test_out_dir() -> PathBuf {
        PathBuf::from(format!("/tmp/zev-llvm-tests-{}", std::process::id()))
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    fn compile_example_mir(paths: &[PathBuf]) -> zev_mir::MirModule {
        let mut sources = Vec::with_capacity(paths.len());
        for path in paths {
            let raw = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
            sources.push((path.clone(), path.display().to_string(), raw));
        }
        let roots: Vec<(&str, Option<&Path>)> = sources
            .iter()
            .map(|(p, _, raw)| (raw.as_str(), Some(p.as_path())))
            .collect();
        let all_generics = zev_sema::collect_import_aware_generic_names(&roots, &[]);
        let mut named_progs = Vec::with_capacity(sources.len());
        for (_, name, src) in &sources {
            let prog = zev_parser::parse_str_named_with_generics(src, name, Some(&all_generics))
                .unwrap_or_else(|errs| panic!("failed to parse {name}: {errs:#?}"));
            named_progs.push((prog, name.as_str()));
        }
        let pairs: Vec<(&zev_ast::Program, Option<&str>)> = named_progs
            .iter()
            .map(|(p, n)| (p, Some(*n)))
            .collect();
        let (mono_progs, _) = zev_sema::analyze_and_monomorphize_named_programs(
            &pairs,
            &zev_sema::AnalyzeOptions::default(),
        )
        .into_result()
        .unwrap_or_else(|diags| panic!("zev-sema rejected examples {paths:?}: {diags:#?}"));
        zev_mir::lower_programs(&mono_progs)
            .unwrap_or_else(|e| panic!("failed to lower examples {paths:?} to MIR: {e:?}"))
    }

    #[test]
    fn esp32c6_demo_builds_direct_boot_bin_and_passes_disasm_audit() {
        let demo_path = repo_root().join("examples/esp32c6_demo.zev");
        let mir = compile_example_mir(&[demo_path]);

        // Verify zero heap and disjoint child coroutine union savings in `main` supervisor!
        assert_eq!(mir.ram_report.mcu32.heap_bytes, 0);
        assert!(
            mir.ram_report.mcu32.union_bytes_saved > 0,
            "expected disjoint child frame union savings > 0 in esp32c6_demo.zev, got {:?}",
            mir.ram_report.mcu32
        );

        let dir = test_out_dir();
        let out_elf = dir.join("esp32c6_demo.elf");
        let out_bin = dir.join("esp32c6_demo.bin");

        let img = build_esp32c6_image(&mir, &out_elf, &out_bin)
            .unwrap_or_else(|e| panic!("ESP32-C6 image build failed: {e}"));

        // 1. Verify LLVM 23 `opt -passes=verify` passes cleanly on the generated RISC-V `.ll`
        verify_ll_with_opt(&img.elf.ll_path).expect("LLVM 23 opt -passes=verify on esp32c6_demo.ll");

        // 2. Verify ELF32 freestanding zero-heap compliance and compact Direct-Boot `.bin`
        assert!(img.elf.audit.is_zero_heap_freestanding());
        assert_eq!(img.magic_header, ESP32C6_DIRECT_BOOT_MAGIC);
        let bin_bytes = std::fs::read(&img.bin_path).expect("read esp32c6_demo.bin");
        assert_eq!(
            &bin_bytes[0..8],
            &[0x1d, 0x04, 0xdb, 0xae, 0x1d, 0x04, 0xdb, 0xae]
        );
        assert!(
            img.bin_size > 512 && img.bin_size < 65_536,
            "expected compact contiguous Direct-Boot .bin (<64KB), got {} bytes",
            img.bin_size
        );

        // 3. Inspect RISC-V disassembly via `/bin/llvm-objdump -d` to verify `@_start` at `0x42000008`
        //    and `.iram1` WS2812B waveform driver at `0x40800000`!
        let objdump = Command::new("/bin/llvm-objdump")
            .arg("-d")
            .arg(&img.elf.elf_path)
            .output()
            .expect("invoke llvm-objdump");
        assert!(objdump.status.success());
        let disasm = String::from_utf8_lossy(&objdump.stdout);
        println!(
            "ESP32-C6 ELF sizes: {} | .bin={}B",
            img.elf.sizes, img.bin_size
        );
        assert!(
            disasm.contains("42000008 <_start>:"),
            "expected @_start at exact Direct-Boot entry 0x42000008:\n{disasm}"
        );
        assert!(
            disasm.contains("csrci\tmstatus, 0x8") || disasm.contains("csrci mstatus, 8"),
            "expected mstatus interrupt mask at @_start:\n{disasm}"
        );
        assert!(
            disasm.contains("<__syn_esp32c6_disarm_watchdogs>:")
                || disasm.contains("__syn_esp32c6_reset"),
            "expected watchdog disarm routine in .text:\n{disasm}"
        );
        assert!(
            disasm.contains("40800000 <__syn_rv_vector_table>:")
                && disasm.contains("<__syn_fn_drive_palette_slot>:")
                && disasm.contains("60006"),
            "expected RV vector table at 0x40800000 and pure-.zev RMT (0x60006...) WS2812B driver in disasm:\n{disasm}"
        );
    }

    #[test]
    fn cortex_m_demo_end_to_end_elf_and_qemu() {
        let t0 = Instant::now();
        let demo_path = repo_root().join("examples/cortex_m_demo.zev");
        let mir = compile_example_mir(&[demo_path]);

        // Verify disjoint child coroutine union-aliasing saved SRAM in `main` supervisor!
        assert!(
            mir.ram_report.mcu32.union_bytes_saved > 0,
            "expected disjoint child frame union savings > 0, got {:?}",
            mir.ram_report.mcu32
        );
        assert_eq!(mir.ram_report.mcu32.heap_bytes, 0);

        let out_elf = test_out_dir().join("cortex_m_demo.elf");
        let opts = CodegenOptions::default();
        let qemu_opts = QemuRunOptions::default();

        let res = run_qemu_cortex_m(&mir, &out_elf, &opts, &qemu_opts)
            .unwrap_or_else(|e| panic!("build/QEMU execution failed: {e}"));

        let total_elapsed = t0.elapsed();
        println!("ELF sizes:\n{}", res.build.sizes.raw_output);
        println!("QEMU UART0 output ({:?}):\n{}", res.exec.elapsed, res.exec.stdout);

        assert_eq!(
            res.exec.exit_code, 0,
            "QEMU exited with non-zero status {} (stderr={})",
            res.exec.exit_code, res.exec.stderr
        );
        assert!(
            total_elapsed.as_secs_f64() < 5.0,
            "end-to-end compile+link+QEMU took {:?} (> 5s budget)",
            total_elapsed
        );
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert!(res.build.sizes.text > 0);
        assert!(res.build.sizes.bss > 0);

        let out = &res.exec.stdout;
        assert!(out.contains("[BOOT] Cortex-M3 Language-OS booted"), "out={out}");
        assert!(out.contains("[TELEM] motor1_rpm=1800"), "out={out}");
        assert!(!out.contains("[TELEM] pwm_duty=0\n"), "expected non-zero PI duty, out={out}");
        assert!(out.contains("[TELEM] sleep_depth_dma_active=1"), "out={out}");
        assert!(out.contains("[TELEM] bad_packets=1"), "out={out}");
        assert!(out.contains("[TELEM] sleep_depth_dma_idle=2"), "out={out}");
        assert!(out.contains("[TELEM] sup_boosted_prio=0"), "out={out}");
        assert!(out.contains("[TELEM] gpio_safe_off=0"), "out={out}");
        assert!(out.contains("[TELEM] sup_restored_prio=3"), "out={out}");
        assert!(out.contains("[PASS] CORTEX-M3 BARE-METAL DEMO OK"), "out={out}");
    }

    #[test]
    fn north_star_motor_controller_builds_freestanding_cortex_m_elf() {
        let root = repo_root();
        let mir = compile_example_mir(&[
            root.join("examples/motor_controller.zev"),
            root.join("examples/motor_support.zev"),
        ]);

        let out_elf = test_out_dir().join("motor_controller.elf");
        let mut opts = CodegenOptions::default();
        opts.max_ticks = 8;

        let res = run_qemu_cortex_m(&mir, &out_elf, &opts, &QemuRunOptions::default())
            .expect("motor_controller Cortex-M3 ELF build + QEMU run");

        assert_eq!(res.exec.exit_code, 0);
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert!(res.build.sizes.text > 0);
        assert!(res.build.sizes.bss > 0);
        assert!(
            res.build
                .audit
                .defined_symbols
                .iter()
                .any(|s| s == "__syn_vec_irq_6")
        );
        assert!(
            res.build
                .audit
                .defined_symbols
                .iter()
                .any(|s| s == "__syn_vec_irq_15")
        );
        assert!(
            res.build
                .audit
                .defined_symbols
                .iter()
                .any(|s| s == "__syn_vec_irq_16")
        );
    }

    #[test]
    fn host_x86_64_ir_compiles_cleanly_via_llc() {
        let demo_path = repo_root().join("examples/cortex_m_demo.zev");
        let mir = compile_example_mir(&[demo_path]);

        let mut host_opts = CodegenOptions::default();
        host_opts.target = TargetArch::HostX86_64Linux;
        let ll = emit_llvm_ir_with_options(&mir, &host_opts).expect("emit x86_64 LLVM IR");

        let dir = test_out_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let ll_path = dir.join("host_demo.ll");
        let obj_path = dir.join("host_demo.o");
        std::fs::write(&ll_path, &ll).unwrap();

        compile_ll_to_obj(&ll_path, &obj_path, &host_opts)
            .expect("llc x86_64-unknown-linux-gnu object compilation");
    }

    #[test]
    fn dual_exit_partial_progress_and_register_return_in_qemu() {
        let src = r#"
            enum IoErr { Timeout, BusFault }

            u32 read_burst(u32 count) error IoErr {
                if (count > 4) {
                    error Timeout, 4;
                }
                return count * 10;
            }

            fn main() {
                u32 ok_val = read_burst(3) catch (IoErr e, u32 partial) {
                    sys.exit(1);
                };
                u32 saved_partial = 0;
                u32 recov = read_burst(9) catch (IoErr e, u32 partial) {
                    saved_partial = partial;
                    partial + 100;
                };
                u32 fallback = read_burst(9) or 77;
                sys.log_kv("partial=", saved_partial);
                sys.log_kv("ok_val=", ok_val);
                sys.log_kv("recov=", recov);
                sys.log_kv("fallback=", fallback);
                sys.exit(0);
            }
        "#;
        let prog = zev_parser::parse_str(src).expect("parse partial error test");
        let mir = zev_mir::lower_program(&prog).expect("lower partial error test");
        let out_elf = test_out_dir().join("partial_err.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run partial error test in QEMU");

        assert_eq!(res.exec.exit_code, 0);
        assert!(res.exec.stdout.contains("partial=4"), "out={}", res.exec.stdout);
        assert!(res.exec.stdout.contains("ok_val=30"), "out={}", res.exec.stdout);
        assert!(res.exec.stdout.contains("recov=104"), "out={}", res.exec.stdout);
        assert!(res.exec.stdout.contains("fallback=77"), "out={}", res.exec.stdout);
    }

    #[test]
    fn portable_io_const_switch_signed_arith_and_clamped_subslice_parity() {
        let src = r#"
            const u8 CMD_PING    = 1;
            const u8 CMD_SPECIAL = 170;

            struct DecodedPacket {
                u8  cmd;
                i16 temp;
            }

            enum PacketErr { Corrupt }

            DecodedPacket decode_pkt(bool bad) error PacketErr {
                if (bad) {
                    error Corrupt;
                }
                return DecodedPacket{ cmd: CMD_SPECIAL, temp: 25 };
            }

            DecodedPacket decode_yielding(bool bad) error PacketErr {
                checkpoint();
                if (bad) {
                    error Corrupt;
                }
                return DecodedPacket{ cmd: 2, temp: 200 };
            }

            i32 match_cmd(u8 cmd) {
                switch (cmd) {
                    CMD_PING => {
                        return 11;
                    }
                    CMD_SPECIAL => {
                        return 222;
                    }
                    _ => {
                        return -1;
                    }
                }
            }

            task coro_verifier() {
                u8 c = 170;
                switch (c) {
                    CMD_PING => {
                        sys.log_kv("coro_sw=", 1);
                    }
                    CMD_SPECIAL => {
                        sys.log_kv("coro_sw=", 2);
                    }
                    _ => {
                        sys.log_kv("coro_sw=", 0);
                    }
                }
                u32 c_errs = 0;
                DecodedPacket cp = decode_pkt(true) catch (PacketErr e) {
                    c_errs += 1;
                    checkpoint();
                    DecodedPacket{ cmd: 7, temp: -15 }
                };
                DecodedPacket py_err = decode_yielding(true) catch (PacketErr e2) {
                    c_errs += 10;
                    DecodedPacket{ cmd: 5, temp: 77 };
                };
                DecodedPacket py_ok = decode_yielding(false) catch (PacketErr e3) {
                    c_errs += 100;
                    DecodedPacket{ cmd: 0, temp: 0 }
                };
                sys.log_kv("coro_errs=", c_errs as i32);
                sys.log_kv("coro_temp=", cp.temp as i32);
                sys.log_kv("py_err_cmd=", py_err.cmd as i32);
                sys.log_kv("py_err_temp=", py_err.temp as i32);
                sys.log_kv("py_ok_temp=", py_ok.temp as i32);
                sys.exit(0);
            }

            fn main() {
                print("hdr:");
                println("hello");
                sys.log("syslog_ok");
                sys.log_kv("k=", -5);
                sys.log_hex("hex=", 0xDEADBEEF);

                u16 cur = 500;
                i32 delta = 300 - (cur as i32);
                sys.log_kv("delta=", delta);

                sys.log_kv("sw_ping=", match_cmd(1));
                sys.log_kv("sw_spec=", match_cmd(170));
                sys.log_kv("sw_wild=", match_cmd(99));

                u32 err_cnt = 0;
                DecodedPacket p = decode_pkt(true) catch (PacketErr e) {
                    err_cnt += 1;
                    DecodedPacket{ cmd: 9, temp: 42 }
                };
                sys.log_kv("err_cnt=", err_cnt as i32);
                sys.log_kv("pkt_cmd=", p.cmd as i32);
                sys.log_kv("pkt_temp=", p.temp as i32);

                [5]u8 raw = [10, 20, 30, 40, 50];
                []u8 full = raw[0..5];
                []u8 clamped = full[0..99];
                []u8 tail = full[3..99];
                []u8 empty_inv = full[4..2];
                sys.log_kv("clamped_len=", clamped.len as i32);
                sys.log_kv("tail_len=", tail.len as i32);
                sys.log_kv("tail0=", tail[0] as i32);
                sys.log_kv("empty_len=", empty_inv.len as i32);
                sys.log_kv("now_ok=", (sys.now_ms() == 0) as i32);

                spawn coro_verifier() @high;
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse parity test");
        zev_sema::analyze_program(&prog).expect("sema parity test");
        let mir = zev_mir::lower_program(&prog).expect("mir parity test");

        // 1. Verify ESP32-C6 Direct-Boot build + LLVM 23 `opt -passes=verify`
        let dir = test_out_dir();
        let esp_img = build_esp32c6_image(
            &mir,
            &dir.join("parity_esp32c6.elf"),
            &dir.join("parity_esp32c6.bin"),
        )
        .expect("ESP32-C6 parity image build");
        verify_ll_with_opt(&esp_img.elf.ll_path).expect("opt verify ESP32-C6 parity .ll");
        assert!(esp_img.elf.audit.is_zero_heap_freestanding());

        // 2. Verify Cortex-M3 build + LLVM 23 `opt -passes=verify` + live QEMU execution
        let cm3_elf = dir.join("parity_cm3.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &cm3_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("Cortex-M3 QEMU parity run");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify Cortex-M3 parity .ll");

        assert_eq!(res.exec.exit_code, 0);
        let out = &res.exec.stdout;
        assert!(out.contains("hdr:hello\n"), "out={out}");
        assert!(out.contains("syslog_ok\n"), "out={out}");
        assert!(out.contains("k=-5\n"), "out={out}");
        assert!(out.contains("hex=0xDEADBEEF\n"), "out={out}");
        assert!(out.contains("delta=-200\n"), "out={out}");
        assert!(out.contains("sw_ping=11\n"), "out={out}");
        assert!(out.contains("sw_spec=222\n"), "out={out}");
        assert!(out.contains("sw_wild=-1\n"), "out={out}");
        assert!(out.contains("err_cnt=1\n"), "out={out}");
        assert!(out.contains("pkt_cmd=9\n"), "out={out}");
        assert!(out.contains("pkt_temp=42\n"), "out={out}");
        assert!(out.contains("clamped_len=5\n"), "out={out}");
        assert!(out.contains("tail_len=2\n"), "out={out}");
        assert!(out.contains("tail0=40\n"), "out={out}");
        assert!(out.contains("empty_len=0\n"), "out={out}");
        assert!(out.contains("now_ok=1\n"), "out={out}");
        assert!(out.contains("coro_sw=2\n"), "out={out}");
        assert!(out.contains("coro_errs=11\n"), "out={out}");
        assert!(out.contains("coro_temp=-15\n"), "out={out}");
        assert!(out.contains("py_err_cmd=5\n"), "out={out}");
        assert!(out.contains("py_err_temp=77\n"), "out={out}");
        assert!(out.contains("py_ok_temp=200\n"), "out={out}");
    }

    #[test]
    fn yielding_struct_error_fn_with_multistmt_catch_in_main_coroutine() {
        let src = r#"
            enum PacketErr { BadCrc }

            struct DecodedPacket {
                u8 cmd;
                u16 temp;
            }

            DecodedPacket decode_direct(bool fail) error PacketErr {
                if (fail) {
                    error BadCrc;
                }
                return DecodedPacket{ cmd: 1, temp: 100 };
            }

            DecodedPacket decode_yielding(bool fail) error PacketErr {
                checkpoint();
                if (fail) {
                    error BadCrc;
                }
                return DecodedPacket{ cmd: 2, temp: 200 };
            }

            fn main() {
                i32 count = 0;

                DecodedPacket p1 = decode_direct(true) catch (PacketErr e) {
                    count += 1;
                    DecodedPacket{ cmd: 9, temp: 42 }
                };

                DecodedPacket p2 = decode_yielding(true) catch (PacketErr e2) {
                    count += 10;
                    DecodedPacket{ cmd: 5, temp: 77 };
                };

                sys.log_kv("p1_cmd=", p1.cmd as i32);
                sys.log_kv("p1_temp=", p1.temp as i32);
                sys.log_kv("p2_cmd=", p2.cmd as i32);
                sys.log_kv("p2_temp=", p2.temp as i32);
                sys.log_kv("count=", count);
                sys.exit(0);
            }
        "#;
        let prog = zev_parser::parse_str(src).expect("parse yielding struct catch test");
        zev_sema::analyze_program(&prog).expect("sema yielding struct catch test");
        let mir = zev_mir::lower_program(&prog).expect("mir yielding struct catch test");

        let mut opts = CodegenOptions::default();
        opts.max_ticks = 20;
        let out_elf = test_out_dir().join("yielding_struct_catch.elf");
        let res = run_qemu_cortex_m(&mir, &out_elf, &opts, &QemuRunOptions::default())
            .expect("run yielding struct catch test in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify yielding struct catch .ll");

        assert_eq!(res.exec.exit_code, 0);
        let out = &res.exec.stdout;
        assert!(out.contains("p1_cmd=9\n"), "out={out}");
        assert!(out.contains("p1_temp=42\n"), "out={out}");
        assert!(out.contains("p2_cmd=5\n"), "out={out}");
        assert!(out.contains("p2_temp=77\n"), "out={out}");
        assert!(out.contains("count=11\n"), "out={out}");
    }

    #[test]
    fn max_ticks_terminates_infinite_every_and_busy_checkpoint_in_qemu() {
        // 1. Infinite `loop every 10ms` firmware WITHOUT explicit `sys.use_systick`
        let every_src = r#"
            task periodic() {
                u32 beats = 0;
                loop every 10ms {
                    beats += 1;
                    sys.log_kv("beat=", beats as i32);
                }
            }

            fn main() {
                spawn periodic() @normal;
            }
        "#;
        let p1 = zev_parser::parse_str(every_src).expect("parse every test");
        zev_sema::analyze_program(&p1).expect("sema every test");
        let m1 = zev_mir::lower_program(&p1).expect("mir every test");

        let mut opts1 = CodegenOptions::default();
        opts1.max_ticks = 35;
        let r1 = run_qemu_cortex_m(
            &m1,
            &test_out_dir().join("every_max_ticks.elf"),
            &opts1,
            &QemuRunOptions::default(),
        )
        .expect("run infinite every 10ms with max_ticks=35");
        assert_eq!(r1.exec.exit_code, 0);
        assert!(r1.exec.stdout.contains("beat=1\n"), "out={}", r1.exec.stdout);
        assert!(r1.exec.stdout.contains("beat=3\n"), "out={}", r1.exec.stdout);

        // 2. Tight CPU-busy `loop { checkpoint(); }` WITHOUT peer tasks or `sys.use_systick`
        let spin_src = r#"
            task spinner() {
                println("spin_start");
                loop {
                    checkpoint();
                }
            }

            fn main() {
                spawn spinner() @normal;
            }
        "#;
        let p2 = zev_parser::parse_str(spin_src).expect("parse spin test");
        zev_sema::analyze_program(&p2).expect("sema spin test");
        let m2 = zev_mir::lower_program(&p2).expect("mir spin test");

        let mut opts2 = CodegenOptions::default();
        opts2.max_ticks = 10;
        let r2 = run_qemu_cortex_m(
            &m2,
            &test_out_dir().join("spin_max_ticks.elf"),
            &opts2,
            &QemuRunOptions::default(),
        )
        .expect("run busy checkpoint spinner with max_ticks=10");
        assert_eq!(r2.exec.exit_code, 0);
        assert!(r2.exec.stdout.contains("spin_start\n"), "out={}", r2.exec.stdout);
    }

    #[test]
    fn signal_chan_and_should_share_on_cortex_m_qemu() {
        let src = r#"
            signal fault_sig;
            signal idle_sig;
            inbox<u32, 4> telemetry_ch;
            irq IRQ_EXTI1 = 1;

            isr EXTI1 {
                fault_sig.notify(boost: @critical);
            }

            task worker_producer() {
                u32 i = 0;
                loop {
                    if (i >= 3) {
                        break;
                    }
                    i += 1;
                    telemetry_ch.send(i * 11);
                }
                loop {
                    idle_sig.wait();
                }
            }

            fn main() {
                sys.use_systick(12_000_000);
                IRQ_EXTI1.attach(EXTI1);
                spawn worker_producer() @normal;
                set_priority(@low);

                u32 a = telemetry_ch.recv();
                u32 b = telemetry_ch.recv();
                u32 c = telemetry_ch.recv();
                sys.log_kv("chan_sum=", a + b + c);

                sys.pend_irq(1);
                fault_sig.wait();
                sys.log_kv("sig_boosted_prio=", sys.current_prio());
                sys.exit(0);
            }
        "#;
        let prog = zev_parser::parse_str(src).expect("parse signal+chan test");
        zev_sema::analyze_program(&prog).expect("sema signal+chan test");
        let mir = zev_mir::lower_program(&prog).expect("mir signal+chan test");

        let out_elf = test_out_dir().join("sig_chan.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run signal+chan test in QEMU");

        assert_eq!(res.exec.exit_code, 0);
        assert!(res.exec.stdout.contains("chan_sum=66"), "out={}", res.exec.stdout);
        assert!(res.exec.stdout.contains("sig_boosted_prio=0"), "out={}", res.exec.stdout);
    }

    #[test]
    fn sim_demo_also_runs_on_bare_metal_cortex_m_qemu() {
        let sim_path = repo_root().join("examples/sim_demo.zev");
        let mir = compile_example_mir(&[sim_path]);

        let out_elf = test_out_dir().join("sim_demo_cm3.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run sim_demo on Cortex-M3 QEMU");

        assert_eq!(res.exec.exit_code, 0);
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert!(res.exec.stdout.contains("milestone=2 a=2400 b=1800"), "out={}", res.exec.stdout);
        assert!(res.exec.stdout.contains("milestone=4 a=0 b=0"), "out={}", res.exec.stdout);
    }

    #[test]
    fn opt_levels_o0_o1_o2_o3_os_oz_build_zero_heap_elfs_and_shrink_flash_in_qemu() {
        let root = repo_root();
        let cm3_mir = compile_example_mir(&[root.join("examples/cortex_m_demo.zev")]);

        let dir = test_out_dir();
        let mut cm3_text_by_level = std::collections::BTreeMap::new();

        for level in ["-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"] {
            let mut opts = CodegenOptions::default();
            opts.opt_level = level.to_string();
            let tag = level.trim_start_matches('-');
            let out_elf = dir.join(format!("cm3_{tag}.elf"));
            let res = run_qemu_cortex_m(&cm3_mir, &out_elf, &opts, &QemuRunOptions::default())
                .unwrap_or_else(|e| panic!("cortex_m_demo failed under {level}: {e}"));

            assert_eq!(res.exec.exit_code, 0, "QEMU failed at {level}");
            assert!(
                res.build.audit.is_zero_heap_freestanding(),
                "zero-heap audit failed at {level}: {:?}",
                res.build.audit
            );
            assert!(
                res.exec.stdout.contains("[PASS] CORTEX-M3 BARE-METAL DEMO OK"),
                "missing PASS banner at {level}: {}",
                res.exec.stdout
            );
            println!(
                "[cortex_m_demo {level:>3}] text={}B data={}B bss={}B total={}B",
                res.build.sizes.text,
                res.build.sizes.data,
                res.build.sizes.bss,
                res.build.sizes.dec
            );
            cm3_text_by_level.insert(level, res.build.sizes.text);
        }

        let text_o0 = cm3_text_by_level["-O0"];
        let text_o2 = cm3_text_by_level["-O2"];
        let text_oz = cm3_text_by_level["-Oz"];
        assert!(
            text_o2 < text_o0,
            "expected -O2 .text ({text_o2}B) < -O0 .text ({text_o0}B)"
        );
        assert!(
            text_oz < text_o0,
            "expected -Oz .text ({text_oz}B) < -O0 .text ({text_o0}B)"
        );

        // Also verify ESP32-C6 Direct-Boot image builds cleanly across -O0, -O2, -Os, -Oz
        let esp_mir = compile_example_mir(&[root.join("examples/esp32c6_demo.zev")]);

        let mut esp_bin_o0 = 0u64;
        let mut esp_bin_oz = 0u64;
        for level in ["-O0", "-O2", "-Os", "-Oz"] {
            let mut opts = CodegenOptions::esp32c6();
            opts.opt_level = level.to_string();
            let tag = level.trim_start_matches('-');
            let img = build_esp32c6_image_with_options(
                &esp_mir,
                &dir.join(format!("esp32c6_{tag}.elf")),
                &dir.join(format!("esp32c6_{tag}.bin")),
                &opts,
            )
            .unwrap_or_else(|e| panic!("esp32c6_demo failed under {level}: {e}"));
            assert!(img.elf.audit.is_zero_heap_freestanding());
            assert_eq!(img.magic_header, ESP32C6_DIRECT_BOOT_MAGIC);
            println!(
                "[esp32c6_demo  {level:>3}] text={}B data={}B bss={}B .bin={}B",
                img.elf.sizes.text, img.elf.sizes.data, img.elf.sizes.bss, img.bin_size
            );
            if level == "-O0" {
                esp_bin_o0 = img.bin_size;
            }
            if level == "-Oz" {
                esp_bin_oz = img.bin_size;
            }
        }
        assert!(
            esp_bin_oz < esp_bin_o0,
            "expected ESP32-C6 -Oz .bin ({esp_bin_oz}B) < -O0 .bin ({esp_bin_o0}B)"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 6.2 Wave 1 — Bare-Metal ARM Cortex-M3 QEMU Execution Suite
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn wave1_boot_tree_spawn_and_conditional_activation_in_qemu() {
        let src = r#"
            watch<u32> w_a = 0;
            watch<u32> w_b = 0;
            watch<u32> w_diag = 0;
            watch<u32> w_opt = 0;
            watch<u32> w_never = 0;

            task worker(u32 val, *watch<u32> out) {
                loop every 5ms {
                    out.publish(val);
                }
            }

            task diag_task(u32 tag, *watch<u32> out) {
                loop every 5ms {
                    out.publish(tag);
                }
            }

            task opt_task(u32 tag, *watch<u32> out) {
                loop every 5ms {
                    out.publish(tag);
                }
            }

            task verifier() {
                delay(8ms);
                sys.log_kv("w_a=", w_a.read() as i32);
                sys.log_kv("w_b=", w_b.read() as i32);
                sys.log_kv("w_diag=", w_diag.read() as i32);
                sys.log_kv("w_opt=", w_opt.read() as i32);
                sys.log_kv("w_never=", w_never.read() as i32);
                assert(w_a.read() == 101);
                assert(w_b.read() == 202);
                assert(w_diag.read() == 111);
                assert(w_opt.read() == 77);
                assert(w_never.read() == 0);
                sys.exit(0);
            }

            fn init_workers(*watch<u32> a, *watch<u32> b) {
                spawn worker(101, a) @high;
                spawn worker(202, b) @high;
            }

            fn init_optional(bool early_stop) {
                spawn opt_task(77, &w_opt) @normal;
                if (early_stop) {
                    return;
                }
                spawn opt_task(88, &w_never) @normal;
            }

            fn setup_system(bool en_diag, bool en_aux) {
                init_workers(&w_a, &w_b);
                if (en_diag) {
                    spawn diag_task(111, &w_diag) @normal;
                }
                if (en_aux) {
                    spawn diag_task(222, &w_never) @normal;
                }
                init_optional(true);
                spawn verifier() @low;
            }

            fn main() {
                sys.use_systick(12_000_000);
                setup_system(true, false);
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave1 boot tree");
        zev_sema::analyze_program(&prog).expect("sema wave1 boot tree");
        let mir = zev_mir::lower_program(&prog).expect("mir wave1 boot tree");

        assert_eq!(mir.spawned_tasks.len(), 7);
        assert_eq!(
            mir.spawned_tasks[0].provenance_path,
            vec!["main", "setup_system", "init_workers", "worker"]
        );
        assert!(!mir.spawned_tasks[0].conditional);
        assert!(mir.spawned_tasks[2].conditional);
        assert!(mir.spawned_tasks[3].conditional);

        let table = mir.ram_report.format_table();
        assert!(
            table.contains("boot provenance  : main -> setup_system -> init_workers -> worker"),
            "table:\n{table}"
        );
        assert!(table.contains("[conditional]"), "table:\n{table}");

        let out_elf = test_out_dir().join("wave1_boot_tree.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run wave1 boot tree in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify wave1 boot tree .ll");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0, "stdout={}", res.exec.stdout);
        let out = &res.exec.stdout;
        assert!(out.contains("w_a=101\n"), "out={out}");
        assert!(out.contains("w_b=202\n"), "out={out}");
        assert!(out.contains("w_diag=111\n"), "out={out}");
        assert!(out.contains("w_opt=77\n"), "out={out}");
        assert!(out.contains("w_never=0\n"), "out={out}");
    }

    #[test]
    fn wave1_loop_every_runtime_period_and_overrun_skip_resync_in_qemu() {
        let src = r#"
            watch<u32> inbody_runs = 0;
            watch<u32> sleep_runs = 0;

            u32 compute_period_ms(u32 mult) {
                return mult * 5;
            }

            task inbody_periodic() {
                u32 p = compute_period_ms(4);
                loop every p {
                    u32 next = inbody_runs.read() + 1;
                    inbody_runs.publish(next);
                    if (next == 1) {
                        /* Stall in-body past 4 full 20ms periods (t=0 -> t=85ms)! */
                        noyield {
                            while (sys.now_ms() < 85) {}
                        }
                    }
                }
            }

            task sleep_periodic() {
                loop every 20ms {
                    sleep_runs.publish(sleep_runs.read() + 1);
                }
            }

            task critical_staller() {
                /* Wait at @low until inbody_periodic completes:
                 * iter 1 (t=0..85ms) + iter 2 resynced (t=85ms) + iter 3 on grid (t=100ms) == 3! */
                while (inbody_runs.read() < 3) {
                    delay(2ms);
                }
                u32 ib_before = inbody_runs.read();
                u32 sb_before = sleep_runs.read();
                u32 t_start = sys.now_ms();
                u32 stall_until = t_start + 85;

                /* Elevate to @critical and stall CPU past 4 full 20ms periods while peers sleep! */
                set_priority(@critical);
                noyield {
                    while (sys.now_ms() < stall_until) {}
                }

                /* Drop back to @low and yield immediately: each 20ms peer runs 1 resynced pass (zero bursts)! */
                set_priority(@low);
                checkpoint();

                u32 ib_after = inbody_runs.read();
                u32 sb_after = sleep_runs.read();

                sys.log_kv("ib_before=", ib_before as i32);
                sys.log_kv("ib_after=", ib_after as i32);
                sys.log_kv("sb_delta=", (sb_after - sb_before) as i32);

                assert(ib_before == 3);
                assert(ib_after == 4);
                assert(sb_after == sb_before + 1);

                sys.exit(0);
            }

            fn main() {
                sys.use_systick(12_000_000);
                spawn critical_staller() @low;
                spawn inbody_periodic() @high;
                spawn sleep_periodic() @normal;
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave1 loop every overrun");
        zev_sema::analyze_program(&prog).expect("sema wave1 loop every overrun");
        let mir = zev_mir::lower_program(&prog).expect("mir wave1 loop every overrun");

        let out_elf = test_out_dir().join("wave1_every_overrun.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run wave1 loop every overrun in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify wave1 loop every .ll");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0, "stdout={}", res.exec.stdout);
        let out = &res.exec.stdout;
        assert!(out.contains("ib_before=3\n"), "out={out}");
        assert!(out.contains("ib_after=4\n"), "out={out}");
        assert!(out.contains("sb_delta=1\n"), "out={out}");
    }

    #[test]
    fn wave1_restart_preserves_task_params_and_resets_locals_in_qemu() {
        let src = r#"
            enum ActorCmd { AddWork, TriggerRestart }

            inbox<ActorCmd, 4> actor_q;
            watch<u32> actor_state = 0;
            watch<u32> actor_defers = 0;
            watch<u32> last_turn_prio = 99;
            signal restart_ack;
            irq IRQ_EXTI2 = 2;

            isr EXTI2 {
                actor_q.post(TriggerRestart, boost: @critical);
            }

            task supervisor_actor(u32 base_id, *watch<u32> out_state) {
                u32 local_acc = 10;
                defer {
                    actor_defers.publish(actor_defers.read() + 1);
                }
                out_state.publish(base_id + local_acc);
                loop {
                    ActorCmd cmd = actor_q.recv();
                    last_turn_prio.publish(sys.current_prio());
                    switch (cmd) {
                        AddWork => {
                            local_acc += 25;
                            out_state.publish(base_id + local_acc);
                        }
                        TriggerRestart => {
                            local_acc += 900;
                            restart_ack.notify();
                            restart;
                        }
                    }
                }
            }

            task driver() {
                while (actor_state.read() == 0) {
                    delay(2ms);
                }
                u32 init_st = actor_state.read();
                actor_q.send(AddWork);
                while (actor_state.read() == 510) {
                    delay(2ms);
                }
                u32 work1_st = actor_state.read();
                u32 work1_prio = last_turn_prio.read();

                /* Fire NVIC EXTI2 -> posts TriggerRestart with boost: @critical */
                sys.pend_irq(2);
                restart_ack.wait();
                while (actor_state.read() != 510) {
                    delay(2ms);
                }

                u32 boosted_prio = last_turn_prio.read();
                u32 defers_after = actor_defers.read();
                u32 restarted_st = actor_state.read();

                /* Send another unboosted AddWork to prove @low priority (3) was restored on restart! */
                actor_q.send(AddWork);
                while (actor_state.read() == 510) {
                    delay(2ms);
                }
                u32 restored_prio = last_turn_prio.read();
                u32 work2_st = actor_state.read();

                sys.log_kv("init_st=", init_st as i32);
                sys.log_kv("work1_st=", work1_st as i32);
                sys.log_kv("work1_prio=", work1_prio as i32);
                sys.log_kv("boosted_prio=", boosted_prio as i32);
                sys.log_kv("defers=", defers_after as i32);
                sys.log_kv("restarted_st=", restarted_st as i32);
                sys.log_kv("restored_prio=", restored_prio as i32);
                sys.log_kv("work2_st=", work2_st as i32);

                assert(init_st == 510);
                assert(work1_st == 535);
                assert(work1_prio == 3);
                assert(boosted_prio == 0);
                assert(defers_after == 1);
                assert(restarted_st == 510);
                assert(restored_prio == 3);
                assert(work2_st == 535);

                sys.exit(0);
            }

            fn main() {
                sys.use_systick(12_000_000);
                IRQ_EXTI2.attach(EXTI2);
                spawn supervisor_actor(500, &actor_state) @low;
                spawn driver() @normal;
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave1 restart");
        zev_sema::analyze_program(&prog).expect("sema wave1 restart");
        let mir = zev_mir::lower_program(&prog).expect("mir wave1 restart");

        let out_elf = test_out_dir().join("wave1_restart.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run wave1 restart in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify wave1 restart .ll");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0, "stdout={}", res.exec.stdout);
        let out = &res.exec.stdout;
        assert!(out.contains("init_st=510\n"), "out={out}");
        assert!(out.contains("boosted_prio=0\n"), "out={out}");
        assert!(out.contains("defers=1\n"), "out={out}");
        assert!(out.contains("restarted_st=510\n"), "out={out}");
        assert!(out.contains("restored_prio=3\n"), "out={out}");
        assert!(out.contains("work2_st=535\n"), "out={out}");
    }

    #[test]
    fn wave1_lifo_defer_on_return_and_error_in_qemu() {
        let src = r#"
            enum OpErr { Failed }

            watch<u32> trace_log = 0;

            fn append_trace(u32 digit) {
                trace_log.publish(trace_log.read() * 10 + digit);
            }

            u32 direct_op(bool should_fail) error OpErr {
                u32 ret_val = 42;
                defer {
                    append_trace(1);
                }
                {
                    defer {
                        append_trace(2);
                    }
                }
                for (u8 i in 0..3) {
                    defer {
                        append_trace(3);
                    }
                    if (i == 1) {
                        continue;
                    }
                    if (i == 2) {
                        break;
                    }
                }
                defer {
                    ret_val = 999;
                    append_trace(4);
                }
                if (should_fail) {
                    error Failed, ret_val + 5;
                }
                return ret_val + 8;
            }

            u32 coro_op(bool should_fail) error OpErr {
                u32 ret_val = 100;
                defer {
                    append_trace(7);
                }
                delay(1ms);
                defer {
                    ret_val = 888;
                    append_trace(8);
                }
                if (should_fail) {
                    error Failed, ret_val + 11;
                }
                return ret_val + 22;
            }

            task tester() {
                trace_log.publish(0);
                u32 ok1 = direct_op(false) catch { 0 };
                u32 t1 = trace_log.read();

                u32 part1 = 0;
                trace_log.publish(0);
                u32 err1 = direct_op(true) catch (OpErr e, u32 part) {
                    part1 = part;
                    0
                };
                u32 t2 = trace_log.read();

                trace_log.publish(0);
                u32 ok2 = coro_op(false) catch { 0 };
                u32 t3 = trace_log.read();

                u32 part2 = 0;
                trace_log.publish(0);
                u32 err2 = coro_op(true) catch (OpErr e, u32 part) {
                    part2 = part;
                    0
                };
                u32 t4 = trace_log.read();

                sys.log_kv("ok1=", ok1 as i32);
                sys.log_kv("part1=", part1 as i32);
                sys.log_kv("t1=", t1 as i32);
                sys.log_kv("t2=", t2 as i32);
                sys.log_kv("ok2=", ok2 as i32);
                sys.log_kv("part2=", part2 as i32);
                sys.log_kv("t3=", t3 as i32);
                sys.log_kv("t4=", t4 as i32);

                assert(ok1 == 50);
                assert(part1 == 47);
                assert(t1 == 233341);
                assert(t2 == 233341);
                assert(ok2 == 122);
                assert(part2 == 111);
                assert(t3 == 87);
                assert(t4 == 87);
                assert(err1 == 0 && err2 == 0);

                sys.exit(0);
            }

            fn main() {
                sys.use_systick(12_000_000);
                spawn tester() @normal;
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave1 lifo defer");
        zev_sema::analyze_program(&prog).expect("sema wave1 lifo defer");
        let mir = zev_mir::lower_program(&prog).expect("mir wave1 lifo defer");

        let out_elf = test_out_dir().join("wave1_lifo_defer.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run wave1 lifo defer in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify wave1 lifo defer .ll");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0, "stdout={}", res.exec.stdout);
        let out = &res.exec.stdout;
        assert!(out.contains("ok1=50\n"), "out={out}");
        assert!(out.contains("part1=47\n"), "out={out}");
        assert!(out.contains("t1=233341\n"), "out={out}");
        assert!(out.contains("ok2=122\n"), "out={out}");
        assert!(out.contains("part2=111\n"), "out={out}");
        assert!(out.contains("t4=87\n"), "out={out}");
    }

    #[test]
    fn wave1_bounded_for_range_inclusive_and_foreach_slice_in_qemu() {
        let src = r#"
            u32 sum_direct([]u8 sl) {
                u32 total = 0;
                for (u8 i in 0..=4) {
                    if (i == 2) {
                        continue;
                    }
                    total += (i as u32);
                }
                for (u8 b in sl) {
                    if (b == 99) {
                        continue;
                    }
                    total += (b as u32);
                }
                [3]u16 fixed = [300 as u16, 400 as u16, 500 as u16];
                for (u16 x in fixed) {
                    if (x == 400) {
                        continue;
                    }
                    total += (x as u32);
                }
                return total;
            }

            task range_task() {
                [4]u8 bytes = [10, 99, 20, 30];
                u32 d_total = sum_direct(bytes[0..4]);

                u32 c_total = 0;
                for (u8 i in 0..=4) {
                    checkpoint();
                    if (i == 2) {
                        continue;
                    }
                    c_total += (i as u32);
                }
                []u8 sl = bytes[0..4];
                for (u8 b in sl) {
                    checkpoint();
                    if (b == 99) {
                        continue;
                    }
                    c_total += (b as u32);
                }
                [3]u16 fixed = [300 as u16, 400 as u16, 500 as u16];
                for (u16 x in fixed) {
                    checkpoint();
                    if (x == 400) {
                        continue;
                    }
                    c_total += (x as u32);
                }

                /* Inclusive T::MAX overflow safety: u8 0..=255 and i8 125..=127 */
                u32 u8_iters = 0;
                for (u8 k in 0..=255) {
                    u8_iters += 1;
                    if (k == 254) {
                        continue;
                    }
                }

                i32 i8_acc = 0;
                for (i8 s in 125..=127) {
                    i8_acc += (s as i32);
                }

                sys.log_kv("d_total=", d_total as i32);
                sys.log_kv("c_total=", c_total as i32);
                sys.log_kv("u8_iters=", u8_iters as i32);
                sys.log_kv("i8_acc=", i8_acc);

                assert(d_total == 868);
                assert(c_total == 868);
                assert(u8_iters == 256);
                assert(i8_acc == 378);

                sys.exit(0);
            }

            fn main() {
                sys.use_systick(12_000_000);
                spawn range_task() @normal;
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave1 bounded for");
        zev_sema::analyze_program(&prog).expect("sema wave1 bounded for");
        let mir = zev_mir::lower_program(&prog).expect("mir wave1 bounded for");

        let out_elf = test_out_dir().join("wave1_bounded_for.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run wave1 bounded for in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify wave1 bounded for .ll");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0, "stdout={}", res.exec.stdout);
        let out = &res.exec.stdout;
        assert!(out.contains("d_total=868\n"), "out={out}");
        assert!(out.contains("c_total=868\n"), "out={out}");
        assert!(out.contains("u8_iters=256\n"), "out={out}");
        assert!(out.contains("i8_acc=378\n"), "out={out}");
    }

    #[test]
    fn wave1_unified_inbox_backpressure_send_try_send_and_queries_in_qemu() {
        let src = r#"
            inbox<u32, 2> pipe;
            signal probe_sig;
            watch<u32> prod_stage = 0;
            watch<u32> send_timeout_fired = 0;

            task producer() {
                assert(pipe.is_empty());
                assert(!pipe.is_full());
                assert(pipe.len() == 0);
                u32 empty_probe = pipe.try_recv() or 777;
                assert(empty_probe == 777);

                /* Verify non-blocking signal.try_wait() + is_set() */
                assert(!probe_sig.try_wait());
                probe_sig.notify();
                assert(probe_sig.is_set());
                assert(probe_sig.try_wait());
                assert(!probe_sig.is_set());
                assert(!probe_sig.try_wait());

                bool ok1 = pipe.try_send(10);
                assert(ok1);
                pipe.send(20);
                assert(pipe.is_full());
                assert(pipe.len() == 2);
                bool ok_full = pipe.try_send(999);
                assert(!ok_full);

                prod_stage.publish(1);

                /* Full queue (cap=2): `.send(99, timeout: 3ms)` MUST time out while consumer delays 10ms */
                pipe.send(99, timeout: 3ms) catch {
                    send_timeout_fired.publish(1);
                };
                prod_stage.publish(2);

                /* Blocking `.send(30)` suspends on full queue and wakes as soon as consumer calls `.recv()` */
                pipe.send(30);
                prod_stage.publish(3);

                loop {
                    delay(1s);
                }
            }

            task consumer() {
                delay(10ms);
                assert(send_timeout_fired.read() == 1);
                assert(prod_stage.read() == 2);

                u32 a = pipe.recv();
                u32 b = pipe.recv();
                u32 c = pipe.recv();
                u32 d = pipe.try_recv() or 5;
                u32 sum = a * 1000 + b * 10 + c + d;

                /* Allow producer wakeup to publish prod_stage=3 */
                delay(1ms);
                u32 final_stage = prod_stage.read();

                sys.log_kv("cons_sum=", sum as i32);
                sys.log_kv("prod_stage=", final_stage as i32);
                sys.log_kv("to_fired=", send_timeout_fired.read() as i32);

                assert(sum == 10235);
                assert(final_stage == 3);

                sys.exit(0);
            }

            fn main() {
                sys.use_systick(12_000_000);
                spawn producer() @high;
                spawn consumer() @normal;
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave1 unified inbox");
        zev_sema::analyze_program(&prog).expect("sema wave1 unified inbox");
        let mir = zev_mir::lower_program(&prog).expect("mir wave1 unified inbox");

        let out_elf = test_out_dir().join("wave1_unified_inbox.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run wave1 unified inbox in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify wave1 unified inbox .ll");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0, "stdout={}", res.exec.stdout);
        let out = &res.exec.stdout;
        assert!(out.contains("cons_sum=10235\n"), "out={out}");
        assert!(out.contains("prod_stage=3\n"), "out={out}");
        assert!(out.contains("to_fired=1\n"), "out={out}");
    }

    #[test]
    fn wave1_dual_specialized_soft_yielding_fn_o2_and_oz_in_qemu() {
        let src = r#"
            u32 mix_byte(u32 acc, u8 b) {
                checkpoint();
                u32 v = acc ^ (b as u32);
                u8 step = 0;
                while (step < 4) {
                    switch (step) {
                        3 => {
                            v = (v * 3) + 1;
                            break;
                        }
                        _ => {
                            v = v + 7;
                        }
                    }
                    step += 1;
                }
                return v;
            }

            u32 compute_crc_soft([]u8 buf) {
                u32 acc = 0x1234;
                for (u8 b in buf) {
                    checkpoint();
                    acc = mix_byte(acc, b);
                }
                return acc;
            }

            task verifier() {
                [5]u8 frame = [0x11, 0x22, 0x33, 0x44, 0x55];
                u32 c1 = compute_crc_soft(frame[0..5]);
                u32 c2 = 0;
                noyield {
                    c2 = compute_crc_soft(frame[0..5]);
                }
                sys.log_kv("coro_crc=", c1 as i32);
                sys.log_kv("noyield_crc=", c2 as i32);
                assert(c1 != 0);
                assert(c1 == c2);
                sys.exit(0);
            }

            fn main() {
                sys.use_systick(12_000_000);
                spawn verifier() @normal;
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave1 soft specialization");
        zev_sema::analyze_program(&prog).expect("sema wave1 soft specialization");
        let mir = zev_mir::lower_program(&prog).expect("mir wave1 soft specialization");

        let dual_names: Vec<&str> = mir
            .ram_report
            .dual_specialized_fns
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        assert!(dual_names.contains(&"compute_crc_soft"), "dual_names: {dual_names:?}");
        assert!(dual_names.contains(&"mix_byte"), "dual_names: {dual_names:?}");

        let table = mir.ram_report.format_table();
        assert!(
            table.contains("Dual-Specialized Functions")
                && table.contains("compute_crc_soft__noyield")
                && table.contains("mix_byte__noyield"),
            "table:\n{table}"
        );

        let dir = test_out_dir();
        let mut text_o0 = 0u64;
        let mut text_oz = 0u64;
        for level in ["-O0", "-O2", "-Oz"] {
            let mut opts = CodegenOptions::default();
            opts.opt_level = level.to_string();
            let tag = level.trim_start_matches('-');
            let out_elf = dir.join(format!("wave1_soft_spec_{tag}.elf"));
            let res = run_qemu_cortex_m(&mir, &out_elf, &opts, &QemuRunOptions::default())
                .unwrap_or_else(|e| panic!("wave1 soft specialization failed at {level}: {e}"));
            verify_ll_with_opt(&res.build.ll_path)
                .unwrap_or_else(|e| panic!("opt verify failed at {level}: {e}"));
            assert!(res.build.audit.is_zero_heap_freestanding());
            assert_eq!(res.exec.exit_code, 0, "failed at {level}: {}", res.exec.stdout);
            assert!(res.exec.stdout.contains("coro_crc="), "out={}", res.exec.stdout);
            assert!(res.exec.stdout.contains("noyield_crc="), "out={}", res.exec.stdout);
            if level == "-O0" {
                text_o0 = res.build.sizes.text;
            }
            if level == "-Oz" {
                text_oz = res.build.sizes.text;
            }
        }
        assert!(
            text_oz < text_o0,
            "expected -Oz .text ({text_oz}B) < -O0 .text ({text_o0}B)"
        );
    }

    #[test]
    fn wave1_restart_boost_restore_yields_to_woken_higher_prio_peer_in_qemu() {
        let src = r#"
            enum Cmd { TriggerReset }

            inbox<Cmd, 4> actor_q;
            signal mid_sig;
            watch<u32> seq_log = 0;
            watch<u32> restart_count = 0;
            watch<u32> defer_ran = 0;

            fn append_step(u32 digit) {
                seq_log.publish((seq_log.read() * 10) + digit);
            }

            task mid_observer() {
                loop {
                    mid_sig.wait();
                    append_step(2);
                }
            }

            task resilient_actor(u32 param_seed) {
                u32 local_acc = 0;
                u32 boots = restart_count.read() + 1;
                restart_count.publish(boots);

                if (boots == 2) {
                    append_step(3);
                    sys.log_kv("seq=", seq_log.read() as i32);
                    sys.log_kv("after_seed=", param_seed as i32);
                    sys.log_kv("after_local=", local_acc as i32);
                    sys.log_kv("defer_ran=", defer_ran.read() as i32);
                    assert(seq_log.read() == 123);
                    assert(param_seed == 77);
                    assert(local_acc == 0);
                    assert(defer_ran.read() == 10);
                    sys.exit(0);
                }

                loop {
                    Cmd c = actor_q.recv();
                    defer {
                        defer_ran.publish(defer_ran.read() + 10);
                    }
                    switch (c) {
                        TriggerReset => {
                            local_acc += 500;
                            mid_sig.notify();
                            checkpoint();
                            append_step(1);
                            restart;
                        }
                    }
                }
            }

            fn main() {
                sys.use_systick(12_000_000);
                spawn mid_observer() @normal;
                spawn resilient_actor(77) @low;
                set_priority(@low);
                delay(1ms);
                actor_q.post(TriggerReset, boost: @critical);
                loop {
                    delay(10ms);
                }
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse restart boost restore");
        zev_sema::analyze_program(&prog).expect("sema restart boost restore");
        let mir = zev_mir::lower_program(&prog).expect("mir restart boost restore");

        let mut opts = CodegenOptions::default();
        opts.max_ticks = 100;
        let out_elf = test_out_dir().join("wave1_restart_boost_seq.elf");
        let res = run_qemu_cortex_m(&mir, &out_elf, &opts, &QemuRunOptions::default())
            .expect("run restart boost restore in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify restart boost restore .ll");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0, "stdout={}", res.exec.stdout);
        let out = &res.exec.stdout;
        assert!(out.contains("seq=123\n"), "out={out}");
        assert!(out.contains("after_seed=77\n"), "out={out}");
        assert!(out.contains("after_local=0\n"), "out={out}");
        assert!(out.contains("defer_ran=10\n"), "out={out}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 6.2 Wave 2 — Bare-Metal Floats, Generics & Zero-Heap Formatting
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn wave2_f32_f64_arithmetic_remainder_casts_and_yield_hoisting_in_qemu() {
        let src = r#"
            watch<f32> pid_out_f32 = 0.0f32;
            watch<f64> rem_out_f64 = 0.0;
            watch<i32> cast_out_i32 = 0;

            f32 clamp_f32(f32 x, f32 lo, f32 hi) {
                if (x < lo) { return lo; }
                if (x > hi) { return hi; }
                return x;
            }

            task pid_worker() {
                f32 integral = 1.25f32;
                f64 accum64 = 10.5;
                u32 step = 0;

                loop every 1ms {
                    f32 err = ((step + 1) as f32) * 1.5f32;
                    integral += err * 0.5f32;
                    integral = clamp_f32(integral, -100.0f32, 100.0f32);

                    f32 mod32 = integral % 1.0f32;
                    f64 wide = (integral as f64) + accum64;
                    wide %= 3.25;
                    f64 neg_rem = (-7.5) % 2.0;

                    delay(1ms);

                    pid_out_f32.publish(integral + mod32);
                    rem_out_f64.publish(wide + neg_rem);
                    cast_out_i32.publish((integral * 4.0f32) as i32);

                    step += 1;
                    if (step >= 2) {
                        println(f"PID:pid={pid_out_f32.read():.2} rem={rem_out_f64.read():.2} ci={cast_out_i32.read()}");
                        sys.exit(0);
                    }
                }
            }

            fn main() {
                sys.use_systick(12_000_000);
                spawn pid_worker() @critical;
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave2 floats");
        let mir = zev_mir::lower_program(&prog).expect("lower wave2 floats");

        let bp = mir
            .task_blueprints
            .iter()
            .find(|t| t.name == "pid_worker")
            .expect("pid_worker blueprint");
        assert!(
            bp.vars
                .iter()
                .any(|v| v.name == "integral" && v.ty == zev_mir::MirType::F32 && v.hoisted),
            "f32 `integral` must be hoisted across yield"
        );
        assert!(
            bp.vars
                .iter()
                .any(|v| v.name == "wide" && v.ty == zev_mir::MirType::F64 && v.hoisted),
            "f64 `wide` must be hoisted across yield"
        );

        let dir = test_out_dir();
        let esp_img = build_esp32c6_image(
            &mir,
            &dir.join("wave2_floats_esp32c6.elf"),
            &dir.join("wave2_floats_esp32c6.bin"),
        )
        .expect("build ESP32-C6 wave2 floats image");
        verify_ll_with_opt(&esp_img.elf.ll_path).expect("opt verify ESP32-C6 wave2 floats");
        assert!(esp_img.elf.audit.is_zero_heap_freestanding());

        let out_elf = dir.join("wave2_floats_cm3.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run wave2 floats in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify Cortex-M3 wave2 floats");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0, "stdout={}", res.exec.stdout);
        assert!(
            res.exec.stdout.contains("PID:pid=4.00 rem=-0.50 ci=14\n"),
            "stdout={}",
            res.exec.stdout
        );
    }

    #[test]
    fn wave2_zero_heap_format_strings_gc_sections_sci_debug_and_enums_in_qemu() {
        let dir = test_out_dir();

        // 1. Verify `:e` scientific float symbols are dead-stripped to 0B by `ld.lld --gc-sections`
        //    when `:e` is NOT used, and unformatted enums emit zero `.rodata` lookup helpers!
        let src_no_sci = r#"
            enum Mode { Idle, Armed, Fault }
            enum UnformattedSecret { HiddenA, HiddenB }

            fn main() {
                Mode m = Mode::Armed;
                f32 v = 3.14159f32;
                println(f"mode={m} v={v}");
                sys.exit(0);
            }
        "#;
        let prog_no_sci = zev_parser::parse_str(src_no_sci).expect("parse no_sci");
        let mir_no_sci = zev_mir::lower_program(&prog_no_sci).expect("lower no_sci");
        assert_eq!(mir_no_sci.formatted_enums, vec!["Mode".to_string()]);

        let ll_no_sci = emit_llvm_ir(&mir_no_sci).expect("emit no_sci ll");
        assert!(ll_no_sci.contains("@__syn_enum_name_Mode"));
        assert!(
            !ll_no_sci.contains("@__syn_enum_name_UnformattedSecret"),
            "Unformatted enum must NOT emit `.rodata` name table"
        );
        assert!(
            !ll_no_sci.contains("@__syn_fmt_f64_sci")
                && !ll_no_sci.contains("@__syn_fmt_f32_sci"),
            "Fixed-only float formatting must not emit `:e` IR helpers"
        );

        let res_no_sci = run_qemu_cortex_m(
            &mir_no_sci,
            &dir.join("wave2_no_sci_cm3.elf"),
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run no_sci in QEMU");
        assert_eq!(res_no_sci.exec.exit_code, 0);
        assert!(res_no_sci.build.audit.is_zero_heap_freestanding());
        assert_eq!(res_no_sci.exec.stdout, "mode=Armed v=3.14\n");
        assert!(
            !res_no_sci
                .build
                .audit
                .defined_symbols
                .iter()
                .any(|s| s.contains("fmt_f64_sci") || s.contains("fmt_f32_sci")),
            "expected `ld.lld --gc-sections` to strip `:e` formatters completely when unused; syms={:?}",
            res_no_sci.build.audit.defined_symbols
        );

        let esp_no_sci = build_esp32c6_image(
            &mir_no_sci,
            &dir.join("wave2_no_sci_esp32c6.elf"),
            &dir.join("wave2_no_sci_esp32c6.bin"),
        )
        .expect("build ESP32-C6 no_sci");
        assert!(esp_no_sci.elf.audit.is_zero_heap_freestanding());
        assert!(
            !esp_no_sci
                .elf
                .audit
                .defined_symbols
                .iter()
                .any(|s| s.contains("fmt_f64_sci") || s.contains("fmt_f32_sci")),
            "expected ESP32-C6 `ld.lld --gc-sections` to strip `:e` formatters completely when unused"
        );

        // 2. Full live execution of fixed floats, Arduino edge cases, `:e`, hex/bin/pad, `{pkt:?}`, `{arr:?}`, and enum names!
        let src_full = r#"
            enum Mode { Idle, Armed, Fault }

            struct Vec2 {
                f32 x;
                f32 y;
            }

            struct TelemetryPkt {
                u16 id;
                Mode mode;
                Vec2 pos;
            }

            fn main() {
                f64 huge = 1.0e200;
                f64 pos_inf = huge * huge;
                f64 neg_inf = -pos_inf;
                f64 nan_val = pos_inf - pos_inf;
                f64 big_ovf = 1.0e12;
                f64 neg_ovf = -1.0e12;
                f64 neg_zero = -0.0;

                println(f"FIXED:{3.14159}:{2.5:.0}:{-2.5:.0}:{1.996:.2}:{neg_zero}");
                println(f"EDGE:{nan_val}:{pos_inf}:{neg_inf}:{big_ovf}:{neg_ovf}");
                println(f"SCI:{1234.5:e}:{-0.0045:.3e}:{9.999:.2e}:{0.0:e}");

                u16 reg = 0x00AB;
                i8 neg_byte = -1;
                u8 bits = 0b0010_1101;
                bool ok = true;
                println(f"INT:0x{reg:04x}:0x{reg:04X}:0x{neg_byte:02x}:0b{bits:08b}:{ok}:{!ok}");

                TelemetryPkt pkt = TelemetryPkt {
                    id: 7,
                    mode: Mode::Armed,
                    pos: Vec2 { x: 1.5f32, y: -2.25f32 },
                };
                [3]i32 samples = [-10, 0, 25];
                println(f"DBG:{pkt:?} arr={samples:?}");
                sys.exit(0);
            }
        "#;
        let prog_full = zev_parser::parse_str(src_full).expect("parse wave2 fmt full");
        let mir_full = zev_mir::lower_program(&prog_full).expect("lower wave2 fmt full");

        let esp_full = build_esp32c6_image(
            &mir_full,
            &dir.join("wave2_fmt_full_esp32c6.elf"),
            &dir.join("wave2_fmt_full_esp32c6.bin"),
        )
        .expect("build ESP32-C6 wave2 fmt full");
        verify_ll_with_opt(&esp_full.elf.ll_path).expect("opt verify ESP32-C6 wave2 fmt full");
        assert!(esp_full.elf.audit.is_zero_heap_freestanding());
        assert!(
            esp_full
                .elf
                .audit
                .defined_symbols
                .iter()
                .any(|s| s == "__syn_fmt_f64_sci"),
            "expected `__syn_fmt_f64_sci` symbol present when `:e` IS used"
        );

        let res_full = run_qemu_cortex_m(
            &mir_full,
            &dir.join("wave2_fmt_full_cm3.elf"),
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run wave2 fmt full in QEMU");
        verify_ll_with_opt(&res_full.build.ll_path).expect("opt verify Cortex-M3 wave2 fmt full");
        assert!(res_full.build.audit.is_zero_heap_freestanding());
        assert!(
            res_full
                .build
                .audit
                .defined_symbols
                .iter()
                .any(|s| s == "__syn_fmt_f64_sci"),
            "expected `__syn_fmt_f64_sci` symbol present on Cortex-M3 when `:e` IS used"
        );
        assert_eq!(res_full.exec.exit_code, 0, "stdout={}", res_full.exec.stdout);
        let out = &res_full.exec.stdout;
        assert!(out.contains("FIXED:3.14:3:-3:2.00:0.00\n"), "out={out}");
        assert!(out.contains("EDGE:nan:inf:-inf:ovf:-ovf\n"), "out={out}");
        assert!(out.contains("SCI:1.23e+03:-4.500e-03:1.00e+01:0.00e+00\n"), "out={out}");
        assert!(out.contains("INT:0x00ab:0x00AB:0xff:0b00101101:true:false\n"), "out={out}");
        assert!(
            out.contains("DBG:TelemetryPkt { id: 7, mode: Armed, pos: Vec2 { x: 1.50, y: -2.25 } } arr=[-10, 0, 25]\n"),
            "out={out}"
        );
    }

    #[test]
    fn wave2_fmt_buf_exact_fit_yield_hoisting_and_clamped_truncation_in_qemu() {
        let src = r#"
            enum Mode { Idle, Active }

            task logger() {
                [32]u8 ok_buf;
                Mode m = Mode::Active;
                f32 v = 12.34f32;
                []u8 msg = fmt_buf(ok_buf, f"m={m} v={v:.1}") catch {
                    sys.exit(2);
                };

                delay(1ms);
                println(f"OK:{msg}");

                [16]u8 small_backing;
                []u8 small = small_backing[0..6];
                []u8 res = fmt_buf(small, f"voltage={v:.2}") catch (FmtErr err, []u8 clamped) {
                    println(f"ERR:{err}:{clamped}:{clamped.len}");
                    sys.exit(0);
                };
                println(f"UNEXPECTED:{res}");
                sys.exit(3);
            }

            fn main() {
                sys.use_systick(12_000_000);
                spawn logger() @normal;
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave2 fmt_buf");
        let mir = zev_mir::lower_program(&prog).expect("lower wave2 fmt_buf");

        let bp = mir
            .task_blueprints
            .iter()
            .find(|t| t.name == "logger")
            .expect("logger blueprint");
        assert!(
            bp.vars.iter().any(|v| v.name == "ok_buf" && v.hoisted),
            "`ok_buf` backing `msg` slice across `delay(1ms)` MUST be hoisted into task frame"
        );

        let out_elf = test_out_dir().join("wave2_fmt_buf_cm3.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &out_elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run wave2 fmt_buf in QEMU");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify wave2 fmt_buf");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0, "stdout={}", res.exec.stdout);
        let out = &res.exec.stdout;
        assert!(out.contains("OK:m=Active v=12.3\n"), "out={out}");
        assert!(out.contains("ERR:Truncated:voltag:6\n"), "out={out}");
    }

    #[test]
    fn wave2_monomorphized_structs_fns_and_ram_report_in_qemu() {
        let src = r#"
            const usize VEC_CAP = 4;

            struct FixedVec<T, usize N> {
                [N]T data;
                usize len;
            }

            T clamp<T>(T x, T lo, T hi) {
                if (x < lo) { return lo; }
                if (x > hi) { return hi; }
                return x;
            }

            fn main() {
                FixedVec<f32, VEC_CAP> v = FixedVec<f32, VEC_CAP> {
                    data: [1.0f32, 2.0f32, 3.0f32, 4.0f32],
                    len: 4,
                };
                FixedVec<i32, 2> iv = FixedVec<i32, 2> {
                    data: [-50, 120],
                    len: 2,
                };
                f32 cf = clamp(v.data[3] * 3.0f32, 0.0f32, 9.5f32);
                i32 ci = clamp::<i32>(iv.data[0], -20, 20);
                println(f"MONO:len={v.len} cf={cf:.2} ci={ci}");
                sys.exit(0);
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave2 generics");
        let (mono_progs, sema_model) = zev_sema::analyze_and_monomorphize_programs(
            &[prog],
            &zev_sema::AnalyzeOptions::default(),
        )
        .into_result()
        .expect("sema + monomorphize wave2 generics");

        assert_eq!(sema_model.monomorphized_instances.len(), 4);
        let mir = zev_mir::lower_programs(&mono_progs).expect("lower monomorphized wave2 generics");
        assert_eq!(mir.monomorphized_instances.len(), 4);

        let table = mir.ram_report.format_table();
        assert!(
            table.contains("7. Monomorphized Generic Instantiations")
                && table.contains("FixedVec<f32, 4>")
                && table.contains("FixedVec__f32__4")
                && table.contains("FixedVec<i32, 2>")
                && table.contains("FixedVec__i32__2")
                && table.contains("clamp<f32>")
                && table.contains("clamp__f32")
                && table.contains("clamp<i32>")
                && table.contains("clamp__i32"),
            "table:\n{table}"
        );

        let dir = test_out_dir();
        let esp_img = build_esp32c6_image(
            &mir,
            &dir.join("wave2_mono_esp32c6.elf"),
            &dir.join("wave2_mono_esp32c6.bin"),
        )
        .expect("build ESP32-C6 wave2 mono");
        assert!(esp_img.elf.audit.is_zero_heap_freestanding());

        for level in ["-O2", "-Oz"] {
            let mut opts = CodegenOptions::default();
            opts.opt_level = level.to_string();
            let tag = level.trim_start_matches('-');
            let res = run_qemu_cortex_m(
                &mir,
                &dir.join(format!("wave2_mono_{tag}.elf")),
                &opts,
                &QemuRunOptions::default(),
            )
            .unwrap_or_else(|e| panic!("wave2 mono failed at {level}: {e}"));
            verify_ll_with_opt(&res.build.ll_path).expect("opt verify wave2 mono");
            assert!(res.build.audit.is_zero_heap_freestanding());
            assert_eq!(res.exec.exit_code, 0);
            assert_eq!(res.exec.stdout, "MONO:len=4 cf=9.50 ci=-20\n");
        }
    }

    #[test]
    fn wave2_review_fixes_sat_casts_const_float_raw_bytes_empty_fmt_buf_and_place_scan_in_qemu() {
        let dir = test_out_dir();

        // Part 1 (M1): Float expression appearing ONLY inside `MirPlace::Index` must pull in soft-float shims.
        let src_place_only = r#"
            fn main() {
                u32 arr[4] = [0, 0, 0, 0];
                arr[((sys.now_ms() as f32) + 1.5f32) as usize] = 99;
                if (arr[1] == 99) {
                    sys.exit(0);
                }
                sys.exit(1);
            }
        "#;
        let prog_p = zev_parser::parse_str(src_place_only).expect("parse place_only");
        zev_sema::analyze_program(&prog_p).expect("sema place_only");
        let mir_p = zev_mir::lower_program(&prog_p).expect("mir place_only");

        let esp_p = build_esp32c6_image(
            &mir_p,
            &dir.join("wave2_r2_place_esp32c6.elf"),
            &dir.join("wave2_r2_place_esp32c6.bin"),
        )
        .expect("build ESP32-C6 place_only");
        assert!(esp_p.elf.audit.is_zero_heap_freestanding());

        let res_p = run_qemu_cortex_m(
            &mir_p,
            &dir.join("wave2_r2_place_cm3.elf"),
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run Cortex-M3 place_only");
        assert!(res_p.build.audit.is_zero_heap_freestanding());
        assert_eq!(res_p.exec.exit_code, 0);

        // Part 2 (H6, H7, H8, H9, H13, H2/M3, L-Fmt): Full sema -> mir -> QEMU (-O0 & -O2) + ESP32-C6 parity!
        let src = r#"
            const f64 CF_A = 1.5;
            const f64 CF_B = 2.5;
            const bool C_LT = CF_A < CF_B;
            const bool C_EQ = CF_A == CF_B;
            const i32 C_I32 = 7.9 as i32;
            const u32 C_U32_HI = 3000000000.0 as u32;

            watch<i32> w_cast = (9.75 as i32);

            fn main() {
                sys.use_systick(12_000_000);
                f64 rt0 = sys.now_ms() as f64;
                f32 rt0_f32 = sys.now_ms() as f32;

                // H6: Unsigned float-to-int casts in [2^31..2^32-1] and [2^63..2^64-1]
                u32 u32_rt = (3000000000.0 + rt0) as u32;
                u64 u64_rt = (10000000000000000000.0 + rt0) as u64;
                u32 u32_ct = (3000000000.0) as u32;

                // H7: Saturating out-of-range, narrow, negative-to-unsigned, and NaN casts (0 poison!)
                i32 sat_hi = (1.0e20 + rt0) as i32;
                i32 sat_lo = (-1.0e20 + rt0) as i32;
                i32 sat_ct = (1.0e20) as i32;
                u8  sat_u8 = (300.0f32 + rt0_f32) as u8;
                i8  sat_i8 = (-200.0f32 + rt0_f32) as i8;
                u32 neg_u32 = (-1.5f32 + rt0_f32) as u32;
                f64 nan_v = rt0 / rt0;
                i32 nan_i32 = nan_v as i32;
                u64 nan_u64 = nan_v as u64;

                println(f"H6:{u32_rt}:{u64_rt}:{u32_ct}");
                println(f"H7:{sat_hi}:{sat_lo}:{sat_ct}:{sat_u8}:{sat_i8}:{neg_u32}:{nan_i32}:{nan_u64}");
                println(f"H8:{C_LT}:{C_EQ}:{C_I32}:{C_U32_HI}:{w_cast.read()}");

                // H9: Raw byte escapes \x80..\xFF (`len == 2`, exact bytes 0xFF, 0xAA)
                [2]u8 raw_buf;
                []u8 raw_sl = fmt_buf(raw_buf, f"\xFF\xAA");
                println(f"H9:{raw_sl.len}:0x{raw_sl[0]:02x}:0x{raw_sl[1]:02x}");

                // H13: Zero-init empty []u8 (`ptr == null, len == 0`) in fmt_buf must NEVER write to UART0
                []u8 empty_dst;
                []u8 empty_res = fmt_buf(empty_dst, f"LEAK_SECRET") catch (FmtErr err, []u8 clamped) {
                    println(f"H13:{err}:{clamped.len}");
                    clamped
                };
                assert(empty_res.len == 0);

                // H2/M3: DebugSlice of pointers formats uniformly as 0x<lowercase hex>
                u32 target_val = 42;
                [2]*u32 ptrs = [0 as *u32, &target_val];
                println(f"PTRS:{ptrs:?}");

                // L-Fmt + M6: Negative values rounding to zero suppress leading '-', and deep subnormal/extreme `:e` matches C99
                f64 tiny_neg = -0.0001;
                f64 subhalf_neg = -0.4;
                println(f"NEGZ:{tiny_neg:.2}:{subhalf_neg:.0}:{9.95e-306:.1e}:{1.25e-307:.1e}");

                sys.exit(0);
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse wave2 r2");
        zev_sema::analyze_program(&prog).expect("sema wave2 r2");
        let mir = zev_mir::lower_program(&prog).expect("mir wave2 r2");

        let esp_img = build_esp32c6_image(
            &mir,
            &dir.join("wave2_r2_esp32c6.elf"),
            &dir.join("wave2_r2_esp32c6.bin"),
        )
        .expect("build ESP32-C6 wave2 r2");
        verify_ll_with_opt(&esp_img.elf.ll_path).expect("opt verify ESP32-C6 wave2 r2");
        assert!(esp_img.elf.audit.is_zero_heap_freestanding());

        for level in ["-O0", "-O2"] {
            let mut opts = CodegenOptions::default();
            opts.opt_level = level.to_string();
            let tag = level.trim_start_matches('-');
            let res = run_qemu_cortex_m(
                &mir,
                &dir.join(format!("wave2_r2_cm3_{tag}.elf")),
                &opts,
                &QemuRunOptions::default(),
            )
            .unwrap_or_else(|e| panic!("wave2 r2 failed at {level}: {e}"));
            verify_ll_with_opt(&res.build.ll_path).expect("opt verify Cortex-M3 wave2 r2");
            assert!(res.build.audit.is_zero_heap_freestanding());
            assert_eq!(res.exec.exit_code, 0, "level={level} out={}", res.exec.stdout);

            let out = &res.exec.stdout;
            assert!(
                !out.contains("LEAK_SECRET"),
                "H13 regression: zero-init `[]u8` leaked to UART0 at {level}: {out}"
            );
            assert!(
                out.contains("H6:3000000000:10000000000000000000:3000000000\n"),
                "level={level} out={out}"
            );
            assert!(
                out.contains("H7:2147483647:-2147483648:2147483647:255:-128:0:0:0\n"),
                "level={level} out={out}"
            );
            assert!(
                out.contains("H8:true:false:7:3000000000:9\n"),
                "level={level} out={out}"
            );
            assert!(
                out.contains("H9:2:0xff:0xaa\n"),
                "level={level} out={out}"
            );
            assert!(
                out.contains("H13:Truncated:0\n"),
                "level={level} out={out}"
            );
            assert!(
                out.contains("PTRS:[0x0, 0x200"),
                "expected 0x-prefixed lowercase hex pointers in DebugSlice at {level}: {out}"
            );
            assert!(
                out.contains("NEGZ:0.00:0:1.0e-305:1.3e-307\n"),
                "level={level} out={out}"
            );
        }
    }

    #[test]
    fn wave2_fnptr_indirect_calls_and_thumb2_large_text_no_function_sections_in_qemu() {
        let dir = test_out_dir();
        std::fs::create_dir_all(&dir).unwrap();

        let src = r#"
            watch<u32> hook_acc = 0;

            struct BoxFn<F> {
                F cb;
            }

            u32 add10(u32 x) {
                return x + 10;
            }

            f32 scale2(f32 x) {
                return x * 2.0f32;
            }

            fn bump_hook(u32 delta) {
                hook_acc.publish(hook_acc.read() + delta);
            }

            f32 apply_f32(fn(f32) -> f32 op, f32 v) {
                return op(v);
            }

            fn main() {
                sys.use_systick(12_000_000);
                BoxFn<fn(u32) -> u32> bf = BoxFn<fn(u32) -> u32> { cb: add10 };
                BoxFn<fn(u32) -> void> vf = BoxFn<fn(u32) -> void> { cb: bump_hook };

                u32 r_paren = (bf.cb)(32);
                u32 r_field = bf.cb(32);
                (vf.cb)(5);
                vf.cb(7);
                f32 rf = apply_f32(scale2, 3.5f32);

                println(f"FNPTR:{r_paren}:{r_field}:{hook_acc.read()}:{rf:.2}");
                sys.exit(0);
            }
        "#;

        let prog = zev_parser::parse_str(src).expect("parse fnptr indirect test");
        let (mono_progs, _) = zev_sema::analyze_and_monomorphize_programs(
            &[prog],
            &zev_sema::AnalyzeOptions::default(),
        )
        .into_result()
        .expect("sema + monomorphize fnptr indirect test");
        let mir = zev_mir::lower_programs(&mono_progs).expect("mir fnptr indirect test");

        // 1. Verify raw `emit_llvm_ir` (> 4 KB `.text` with soft-float + fmt) assembles cleanly via
        //    `/bin/llc -mtriple=thumbv7m-none-eabi -mcpu=cortex-m3 -filetype=obj` WITHOUT `--function-sections`!
        let raw_ll = emit_llvm_ir(&mir).expect("emit raw thumbv7m IR");
        let raw_ll_path = dir.join("fnptr_no_fsec.ll");
        let raw_obj_path = dir.join("fnptr_no_fsec.o");
        std::fs::write(&raw_ll_path, &raw_ll).unwrap();
        let llc_out = Command::new("/bin/llc")
            .arg("-mtriple=thumbv7m-none-eabi")
            .arg("-mcpu=cortex-m3")
            .arg("-filetype=obj")
            .arg(&raw_ll_path)
            .arg("-o")
            .arg(&raw_obj_path)
            .output()
            .expect("invoke /bin/llc without --function-sections");
        assert!(
            llc_out.status.success(),
            "llc without --function-sections failed:\n{}",
            String::from_utf8_lossy(&llc_out.stderr)
        );

        // 2. Verify ESP32-C6 Direct-Boot build & Cortex-M3 QEMU execution at `-O0` and `-O2`
        let esp_img = build_esp32c6_image(
            &mir,
            &dir.join("wave2_fnptr_esp32c6.elf"),
            &dir.join("wave2_fnptr_esp32c6.bin"),
        )
        .expect("build ESP32-C6 fnptr image");
        assert!(esp_img.elf.audit.is_zero_heap_freestanding());

        for level in ["-O0", "-O2"] {
            let mut opts = CodegenOptions::default();
            opts.opt_level = level.to_string();
            let tag = level.trim_start_matches('-');
            let res = run_qemu_cortex_m(
                &mir,
                &dir.join(format!("wave2_fnptr_cm3_{tag}.elf")),
                &opts,
                &QemuRunOptions::default(),
            )
            .unwrap_or_else(|e| panic!("fnptr QEMU run failed at {level}: {e}"));
            verify_ll_with_opt(&res.build.ll_path).expect("opt verify fnptr .ll");
            assert!(res.build.audit.is_zero_heap_freestanding());
            assert_eq!(res.exec.exit_code, 0, "level={level} out={}", res.exec.stdout);
            assert_eq!(res.exec.stdout, "FNPTR:42:42:12:7.00\n", "level={level}");
        }
    }

    fn build_wave1_mmio_irq_test_module() -> zev_mir::MirModule {
        use zev_mir::{
            MirBakedArg, MirBinOp, MirBootSequence, MirBootStep, MirExpr, MirExprKind,
            MirFmtChunk, MirIntFmt, MirIntrinsicOp, MirIrqBinding, MirIrqVector, MirIsr,
            MirIsrStateDecl, MirIsrStateVar, MirMmioAccess, MirMmioInstance, MirMmioLayout,
            MirMmioReg, MirPrimKind, MirPrimNonBlockingOp, MirPrimReadOp, MirPrimTarget,
            MirPrimitive, MirStmt, MirType, MirVar, VarId, VarRole,
        };

        let scratch_layout = MirMmioLayout {
            name: "ScratchRegs".to_string(),
            byte_size: 24,
            regs: vec![
                MirMmioReg {
                    name: "ctrl".to_string(),
                    access: MirMmioAccess::Rw,
                    elem_bits: 32,
                    array_len: None,
                    offset: 0,
                },
                MirMmioReg {
                    name: "half".to_string(),
                    access: MirMmioAccess::Rw,
                    elem_bits: 16,
                    array_len: None,
                    offset: 4,
                },
                MirMmioReg {
                    name: "byte_reg".to_string(),
                    access: MirMmioAccess::Rw,
                    elem_bits: 8,
                    array_len: None,
                    offset: 6,
                },
                MirMmioReg {
                    name: "slots".to_string(),
                    access: MirMmioAccess::Rw,
                    elem_bits: 32,
                    array_len: Some(4),
                    offset: 8,
                },
            ],
        };
        let uart_layout = MirMmioLayout {
            name: "Uart0Regs".to_string(),
            byte_size: 4,
            regs: vec![MirMmioReg {
                name: "dr".to_string(),
                access: MirMmioAccess::Rw,
                elem_bits: 32,
                array_len: None,
                offset: 0,
            }],
        };

        let scratch_a = MirMmioInstance {
            name: "SCRATCH_A".to_string(),
            layout: "ScratchRegs".to_string(),
            base_addr: 0x2000_E000,
            byte_size: 24,
        };
        let scratch_b = MirMmioInstance {
            name: "SCRATCH_B".to_string(),
            layout: "ScratchRegs".to_string(),
            base_addr: 0x2000_E100,
            byte_size: 24,
        };
        let uart0 = MirMmioInstance {
            name: "UART0".to_string(),
            layout: "Uart0Regs".to_string(),
            base_addr: 0x4000_C000,
            byte_size: 4,
        };

        let inbox_elem = MirType::U32;
        let primitives = vec![
            MirPrimitive {
                name: "uart_log".to_string(),
                kind: MirPrimKind::Inbox {
                    elem: Box::new(inbox_elem.clone()),
                    cap: 8,
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "shared_hits".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: Some(MirExpr::new(MirExprKind::IntLit(0), MirType::U32)),
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "sel_val".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: Some(MirExpr::new(MirExprKind::IntLit(0), MirType::U32)),
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
        ];

        let mk_var = |id: usize, name: &str, ty: MirType, role: VarRole| MirVar {
            id: VarId(id),
            name: name.to_string(),
            ty,
            role,
            in_noyield_scope: true,
            addr_taken: false,
            hoisted: false,
        };

        let u32_expr = |v: i128| MirExpr::new(MirExprKind::IntLit(v), MirType::U32);
        let var_expr = |id: usize, ty: MirType| MirExpr::new(MirExprKind::Var(VarId(id)), ty);

        // Parameterized ISR `mmio_worker(*ScratchRegs regs, *Uart0Regs uart, u32 tag)` + `state u32 count = 10;`
        let isr_worker = MirIsr {
            name: "mmio_worker".to_string(),
            params: vec![VarId(0), VarId(1), VarId(2)],
            state_vars: vec![MirIsrStateDecl {
                var_id: VarId(3),
                name: "count".to_string(),
                ty: MirType::U32,
                init: u32_expr(10),
            }],
            vars: vec![
                mk_var(0, "regs", MirType::MmioPtr("ScratchRegs".to_string()), VarRole::Param),
                mk_var(1, "uart", MirType::MmioPtr("Uart0Regs".to_string()), VarRole::Param),
                mk_var(2, "tag", MirType::U32, VarRole::Param),
                mk_var(3, "count", MirType::U32, VarRole::IsrState),
            ],
            body: vec![
                MirStmt::MmioWrite {
                    base: Box::new(var_expr(0, MirType::MmioPtr("ScratchRegs".to_string()))),
                    offset: 0,
                    elem_bits: 32,
                    index: None,
                    array_len: None,
                    value: var_expr(2, MirType::U32),
                },
                MirStmt::MmioWrite {
                    base: Box::new(var_expr(0, MirType::MmioPtr("ScratchRegs".to_string()))),
                    offset: 4,
                    elem_bits: 16,
                    index: None,
                    array_len: None,
                    value: MirExpr::new(MirExprKind::IntLit(0x1234), MirType::U16),
                },
                MirStmt::MmioWrite {
                    base: Box::new(var_expr(0, MirType::MmioPtr("ScratchRegs".to_string()))),
                    offset: 6,
                    elem_bits: 8,
                    index: None,
                    array_len: None,
                    value: MirExpr::new(MirExprKind::IntLit(0x5A), MirType::U8),
                },
                MirStmt::MmioWrite {
                    base: Box::new(var_expr(0, MirType::MmioPtr("ScratchRegs".to_string()))),
                    offset: 8,
                    elem_bits: 32,
                    index: Some(Box::new(u32_expr(2))),
                    array_len: Some(4),
                    value: MirExpr::new(
                        MirExprKind::Binary {
                            op: MirBinOp::Add,
                            lhs: Box::new(var_expr(2, MirType::U32)),
                            rhs: Box::new(var_expr(3, MirType::U32)),
                        },
                        MirType::U32,
                    ),
                },
                MirStmt::Assign {
                    target: zev_mir::MirPlace::Var(VarId(3)),
                    op: zev_mir::MirAssignOp::AddAssign,
                    value: u32_expr(1),
                },
                MirStmt::MmioWrite {
                    base: Box::new(var_expr(1, MirType::MmioPtr("Uart0Regs".to_string()))),
                    offset: 0,
                    elem_bits: 32,
                    index: None,
                    array_len: None,
                    value: u32_expr(0x4D), // 'M'
                },
                MirStmt::PrimOp(MirPrimNonBlockingOp::InboxPost {
                    target: MirPrimTarget::Global("uart_log".to_string()),
                    elem_ty: MirType::U32,
                    cap: 8,
                    value: MirExpr::new(
                        MirExprKind::Binary {
                            op: MirBinOp::Add,
                            lhs: Box::new(MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Mul,
                                    lhs: Box::new(var_expr(2, MirType::U32)),
                                    rhs: Box::new(u32_expr(100)),
                                },
                                MirType::U32,
                            )),
                            rhs: Box::new(var_expr(3, MirType::U32)),
                        },
                        MirType::U32,
                    ),
                    boost: None,
                }),
            ],
        };

        let isr_shared_a = MirIsr {
            name: "shared_a".to_string(),
            params: vec![VarId(0)],
            state_vars: vec![MirIsrStateDecl {
                var_id: VarId(1),
                name: "acc".to_string(),
                ty: MirType::U32,
                init: u32_expr(1),
            }],
            vars: vec![
                mk_var(0, "delta", MirType::U32, VarRole::Param),
                mk_var(1, "acc", MirType::U32, VarRole::IsrState),
            ],
            body: vec![
                MirStmt::Assign {
                    target: zev_mir::MirPlace::Var(VarId(1)),
                    op: zev_mir::MirAssignOp::AddAssign,
                    value: var_expr(0, MirType::U32),
                },
                MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                    target: MirPrimTarget::Global("shared_hits".to_string()),
                    elem_ty: MirType::U32,
                    value: MirExpr::new(
                        MirExprKind::Binary {
                            op: MirBinOp::Add,
                            lhs: Box::new(MirExpr::new(
                                MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                                    target: MirPrimTarget::Global("shared_hits".to_string()),
                                    elem_ty: MirType::U32,
                                })),
                                MirType::U32,
                            )),
                            rhs: Box::new(var_expr(1, MirType::U32)),
                        },
                        MirType::U32,
                    ),
                }),
            ],
        };

        let isr_shared_b = MirIsr {
            name: "shared_b".to_string(),
            params: vec![VarId(0)],
            state_vars: vec![MirIsrStateDecl {
                var_id: VarId(1),
                name: "calls".to_string(),
                ty: MirType::U32,
                init: u32_expr(0),
            }],
            vars: vec![
                mk_var(0, "mult", MirType::U32, VarRole::Param),
                mk_var(1, "calls", MirType::U32, VarRole::IsrState),
            ],
            body: vec![
                MirStmt::Assign {
                    target: zev_mir::MirPlace::Var(VarId(1)),
                    op: zev_mir::MirAssignOp::AddAssign,
                    value: u32_expr(1),
                },
                MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                    target: MirPrimTarget::Global("shared_hits".to_string()),
                    elem_ty: MirType::U32,
                    value: MirExpr::new(
                        MirExprKind::Binary {
                            op: MirBinOp::Add,
                            lhs: Box::new(MirExpr::new(
                                MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                                    target: MirPrimTarget::Global("shared_hits".to_string()),
                                    elem_ty: MirType::U32,
                                })),
                                MirType::U32,
                            )),
                            rhs: Box::new(MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Mul,
                                    lhs: Box::new(var_expr(1, MirType::U32)),
                                    rhs: Box::new(var_expr(0, MirType::U32)),
                                },
                                MirType::U32,
                            )),
                        },
                        MirType::U32,
                    ),
                }),
            ],
        };

        let isr_cond_1 = MirIsr {
            name: "cond_isr_1".to_string(),
            params: vec![],
            state_vars: vec![],
            vars: vec![],
            body: vec![MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                target: MirPrimTarget::Global("sel_val".to_string()),
                elem_ty: MirType::U32,
                value: u32_expr(111),
            })],
        };

        let isr_cond_2 = MirIsr {
            name: "cond_isr_2".to_string(),
            params: vec![],
            state_vars: vec![MirIsrStateDecl {
                var_id: VarId(0),
                name: "v".to_string(),
                ty: MirType::U32,
                init: u32_expr(222),
            }],
            vars: vec![mk_var(0, "v", MirType::U32, VarRole::IsrState)],
            body: vec![MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                target: MirPrimTarget::Global("sel_val".to_string()),
                elem_ty: MirType::U32,
                value: var_expr(0, MirType::U32),
            })],
        };

        let irq_vectors = vec![
            MirIrqVector {
                irq_name: "IRQ_5".to_string(),
                irq_num: 5,
                is_shared: false,
                selector_symbol: None,
                bindings: vec![MirIrqBinding {
                    binding_id: 0,
                    isr_name: "mmio_worker".to_string(),
                    baked_args: vec![
                        MirBakedArg::MmioAddr("SCRATCH_A".to_string()),
                        MirBakedArg::MmioAddr("UART0".to_string()),
                        MirBakedArg::Const(u32_expr(100)),
                    ],
                    state_vars: vec![MirIsrStateVar {
                        storage_symbol: "__zev_isr_state_0_count".to_string(),
                        var_name: "count".to_string(),
                        ty: MirType::U32,
                        init: u32_expr(10),
                    }],
                    cond_selector: None,
                }],
            },
            MirIrqVector {
                irq_name: "IRQ_37".to_string(),
                irq_num: 37,
                is_shared: false,
                selector_symbol: None,
                bindings: vec![MirIrqBinding {
                    binding_id: 1,
                    isr_name: "mmio_worker".to_string(),
                    baked_args: vec![
                        MirBakedArg::MmioAddr("SCRATCH_B".to_string()),
                        MirBakedArg::MmioAddr("UART0".to_string()),
                        MirBakedArg::Const(u32_expr(200)),
                    ],
                    state_vars: vec![MirIsrStateVar {
                        storage_symbol: "__zev_isr_state_1_count".to_string(),
                        var_name: "count".to_string(),
                        ty: MirType::U32,
                        init: u32_expr(10),
                    }],
                    cond_selector: None,
                }],
            },
            MirIrqVector {
                irq_name: "IRQ_SHARED".to_string(),
                irq_num: 12,
                is_shared: true,
                selector_symbol: None,
                bindings: vec![
                    MirIrqBinding {
                        binding_id: 2,
                        isr_name: "shared_a".to_string(),
                        baked_args: vec![MirBakedArg::Const(u32_expr(10))],
                        state_vars: vec![MirIsrStateVar {
                            storage_symbol: "__zev_isr_state_2_acc".to_string(),
                            var_name: "acc".to_string(),
                            ty: MirType::U32,
                            init: u32_expr(1),
                        }],
                        cond_selector: None,
                    },
                    MirIrqBinding {
                        binding_id: 3,
                        isr_name: "shared_b".to_string(),
                        baked_args: vec![MirBakedArg::Const(u32_expr(1000))],
                        state_vars: vec![MirIsrStateVar {
                            storage_symbol: "__zev_isr_state_3_calls".to_string(),
                            var_name: "calls".to_string(),
                            ty: MirType::U32,
                            init: u32_expr(0),
                        }],
                        cond_selector: None,
                    },
                ],
            },
            MirIrqVector {
                irq_name: "IRQ_COND".to_string(),
                irq_num: 19,
                is_shared: false,
                selector_symbol: Some("__zev_irq_sel_19".to_string()),
                bindings: vec![
                    MirIrqBinding {
                        binding_id: 4,
                        isr_name: "cond_isr_1".to_string(),
                        baked_args: vec![],
                        state_vars: vec![],
                        cond_selector: Some(("__zev_irq_sel_19".to_string(), 1)),
                    },
                    MirIrqBinding {
                        binding_id: 5,
                        isr_name: "cond_isr_2".to_string(),
                        baked_args: vec![],
                        state_vars: vec![MirIsrStateVar {
                            storage_symbol: "__zev_isr_state_5_v".to_string(),
                            var_name: "v".to_string(),
                            ty: MirType::U32,
                            init: u32_expr(222),
                        }],
                        cond_selector: Some(("__zev_irq_sel_19".to_string(), 2)),
                    },
                ],
            },
        ];

        let try_recv_res_ty = MirType::Result {
            ok: Box::new(MirType::U32),
            err: None,
        };
        let boot_vars = vec![
            mk_var(0, "r0", try_recv_res_ty.clone(), VarRole::CompilerTemp),
            mk_var(1, "r1", try_recv_res_ty.clone(), VarRole::CompilerTemp),
            mk_var(2, "r2", try_recv_res_ty.clone(), VarRole::CompilerTemp),
        ];

        let try_recv_expr = MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxTryRecv {
                target: MirPrimTarget::Global("uart_log".to_string()),
                elem_ty: MirType::U32,
                cap: 8,
            })),
            try_recv_res_ty,
        );
        let mmio_a = MirExpr::new(
            MirExprKind::MmioAddr("SCRATCH_A".to_string()),
            MirType::MmioPtr("ScratchRegs".to_string()),
        );
        let mmio_b = MirExpr::new(
            MirExprKind::MmioAddr("SCRATCH_B".to_string()),
            MirType::MmioPtr("ScratchRegs".to_string()),
        );

        let mmio_read = |base: MirExpr, offset: u32, elem_bits: u8, idx: Option<i128>, ty: MirType| {
            MirExpr::new(
                MirExprKind::MmioRead {
                    base: Box::new(base),
                    offset,
                    elem_bits,
                    index: idx.map(|i| Box::new(u32_expr(i))),
                    array_len: idx.map(|_| 4),
                },
                ty,
            )
        };
        let pend = |irq: i128| {
            MirBootStep::Stmt(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                method: "pend_irq".to_string(),
                args: vec![u32_expr(irq)],
            }))
        };
        let dec = |value: MirExpr| MirFmtChunk::Int {
            value,
            signed: false,
            bits: 32,
            fmt: MirIntFmt::Dec,
        };

        let boot_steps = vec![
            MirBootStep::Stmt(MirStmt::SetIrqSelector {
                symbol: "__zev_irq_sel_19".to_string(),
                value: 2,
            }),
            pend(5),
            pend(5),
            pend(37),
            pend(12),
            pend(19),
            MirBootStep::Stmt(MirStmt::Assign {
                target: zev_mir::MirPlace::Var(VarId(0)),
                op: zev_mir::MirAssignOp::Assign,
                value: try_recv_expr.clone(),
            }),
            MirBootStep::Stmt(MirStmt::Assign {
                target: zev_mir::MirPlace::Var(VarId(1)),
                op: zev_mir::MirAssignOp::Assign,
                value: try_recv_expr.clone(),
            }),
            MirBootStep::Stmt(MirStmt::Assign {
                target: zev_mir::MirPlace::Var(VarId(2)),
                op: zev_mir::MirAssignOp::Assign,
                value: try_recv_expr,
            }),
            MirBootStep::Stmt(MirStmt::PrintFmt {
                chunks: vec![
                    MirFmtChunk::Literal("\nLOG=".to_string()),
                    dec(MirExpr::new(MirExprKind::ResultVal(VarId(0)), MirType::U32)),
                    MirFmtChunk::Literal(",".to_string()),
                    dec(MirExpr::new(MirExprKind::ResultVal(VarId(1)), MirType::U32)),
                    MirFmtChunk::Literal(",".to_string()),
                    dec(MirExpr::new(MirExprKind::ResultVal(VarId(2)), MirType::U32)),
                ],
                newline: true,
            }),
            MirBootStep::Stmt(MirStmt::PrintFmt {
                chunks: vec![
                    MirFmtChunk::Literal("MMIO_A=".to_string()),
                    dec(mmio_read(mmio_a.clone(), 0, 32, None, MirType::U32)),
                    MirFmtChunk::Literal(":".to_string()),
                    dec(MirExpr::new(
                        MirExprKind::Cast {
                            expr: Box::new(mmio_read(mmio_a.clone(), 4, 16, None, MirType::U16)),
                            target_ty: MirType::U32,
                        },
                        MirType::U32,
                    )),
                    MirFmtChunk::Literal(":".to_string()),
                    dec(MirExpr::new(
                        MirExprKind::Cast {
                            expr: Box::new(mmio_read(mmio_a.clone(), 6, 8, None, MirType::U8)),
                            target_ty: MirType::U32,
                        },
                        MirType::U32,
                    )),
                    MirFmtChunk::Literal(":".to_string()),
                    dec(mmio_read(mmio_a, 8, 32, Some(2), MirType::U32)),
                ],
                newline: true,
            }),
            MirBootStep::Stmt(MirStmt::PrintFmt {
                chunks: vec![
                    MirFmtChunk::Literal("MMIO_B=".to_string()),
                    dec(mmio_read(mmio_b.clone(), 0, 32, None, MirType::U32)),
                    MirFmtChunk::Literal(":".to_string()),
                    dec(mmio_read(mmio_b, 8, 32, Some(2), MirType::U32)),
                ],
                newline: true,
            }),
            MirBootStep::Stmt(MirStmt::PrintFmt {
                chunks: vec![
                    MirFmtChunk::Literal("SHARED=".to_string()),
                    dec(MirExpr::new(
                        MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                            target: MirPrimTarget::Global("shared_hits".to_string()),
                            elem_ty: MirType::U32,
                        })),
                        MirType::U32,
                    )),
                    MirFmtChunk::Literal(" SEL=".to_string()),
                    dec(MirExpr::new(
                        MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                            target: MirPrimTarget::Global("sel_val".to_string()),
                            elem_ty: MirType::U32,
                        })),
                        MirType::U32,
                    )),
                ],
                newline: true,
            }),
            MirBootStep::Stmt(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                method: "exit".to_string(),
                args: vec![u32_expr(0)],
            })),
        ];

        let dummy_prog = zev_parser::parse_str("fn main() {}").unwrap();
        let mut mir = zev_mir::lower_program(&dummy_prog).unwrap();
        mir.mmio_layouts = vec![scratch_layout, uart_layout];
        mir.mmio_instances = vec![scratch_a, scratch_b, uart0];
        mir.primitives = primitives;
        mir.isrs = vec![
            isr_worker,
            isr_shared_a,
            isr_shared_b,
            isr_cond_1,
            isr_cond_2,
        ];
        mir.irq_vectors = irq_vectors;
        mir.boot = MirBootSequence {
            present: true,
            vars: boot_vars,
            steps: boot_steps,
            boot_yielding_callees: vec![],
        };
        mir
    }

    #[test]
    fn wave1_mmio_parameterized_isr_state_and_nvic_iser_in_qemu() {
        let dir = test_out_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let mir = build_wave1_mmio_irq_test_module();

        // 1. Verify Thumb-2 LLVM IR structure (`.isr_vector` > 32, `ISER0` + `ISER1`, volatile MMIO, zero-alloca state)
        let raw_ll = emit_llvm_ir(&mir).expect("emit Wave 1 Thumb-2 LLVM IR");
        assert!(
            raw_ll.contains("@__isr_vector = constant [54 x ptr] ["),
            "expected 16 + max(32, 37 + 1) = 54 vector slots:\n{raw_ll}"
        );
        assert!(raw_ll.contains("ptr @__syn_vec_irq_5"));
        assert!(raw_ll.contains("ptr @__syn_vec_irq_12"));
        assert!(raw_ll.contains("ptr @__syn_vec_irq_19"));
        assert!(raw_ll.contains("ptr @__syn_vec_irq_37"));
        assert!(raw_ll.contains("NVIC_ISER0 (0xE000E100)"));
        assert!(raw_ll.contains("NVIC_ISER1 (0xE000E104)"));
        assert!(raw_ll.contains("load volatile i8, ptr"));
        assert!(raw_ll.contains("load volatile i16, ptr"));
        assert!(raw_ll.contains("load volatile i32, ptr"));
        assert!(raw_ll.contains("store volatile i8 "));
        assert!(raw_ll.contains("store volatile i16 "));
        assert!(raw_ll.contains("store volatile i32 "));
        assert!(raw_ll.contains("getelementptr i8, ptr "));
        assert!(raw_ll.contains("@__zev_isr_state_0_count = internal global i32 10, align 4"));
        assert!(raw_ll.contains("@__zev_isr_state_1_count = internal global i32 10, align 4"));
        assert!(raw_ll.contains("@__zev_irq_sel_19 = internal global i8 0, align 1"));
        assert!(
            raw_ll.contains("define void @__syn_isr_mmio_worker(ptr %arg0, ptr %arg1, i32 %arg2, ptr %__state_0)"),
            "expected parameterized ISR signature:\n{raw_ll}"
        );
        assert!(
            raw_ll.contains("switch i8 %sel, label %done"),
            "expected selector switch in @__syn_vec_irq_19:\n{raw_ll}"
        );

        // Verify `VarRole::IsrState` has NO stack alloca inside `@__syn_isr_mmio_worker`
        let worker_ir = raw_ll
            .split("define void @__syn_isr_mmio_worker(")
            .nth(1)
            .unwrap()
            .split("\n}\n")
            .next()
            .unwrap();
        assert!(!worker_ir.contains("%v_3 = alloca"), "worker_ir={worker_ir}");
        assert!(worker_ir.contains("load i32, ptr %__state_0"), "worker_ir={worker_ir}");
        assert!(worker_ir.contains("store i32 "), "worker_ir={worker_ir}");

        // 2. Build & run live in QEMU Cortex-M3 (`lm3s6965evb`) at `-O0` and `-O2`!
        for level in ["-O0", "-O2"] {
            let mut opts = CodegenOptions::default();
            opts.opt_level = level.to_string();
            let tag = level.trim_start_matches('-');
            let res = run_qemu_cortex_m(
                &mir,
                &dir.join(format!("wave1_mmio_irq_{tag}.elf")),
                &opts,
                &QemuRunOptions::default(),
            )
            .unwrap_or_else(|e| panic!("Wave 1 MMIO/IRQ QEMU run failed at {level}: {e}"));
            verify_ll_with_opt(&res.build.ll_path).expect("opt -passes=verify on Wave 1 IR");
            assert!(res.build.audit.is_zero_heap_freestanding());
            assert_eq!(res.exec.exit_code, 0, "level={level} out={}", res.exec.stdout);
            let out = &res.exec.stdout;
            assert!(out.contains("MMM\nLOG=10011,10012,20011\n"), "level={level} out={out}");
            assert!(out.contains("MMIO_A=100:4660:90:111\n"), "level={level} out={out}");
            assert!(out.contains("MMIO_B=200:210\n"), "level={level} out={out}");
            assert!(out.contains("SHARED=1011 SEL=222\n"), "level={level} out={out}");
        }
    }

    #[test]
    fn wave1_mmio_irq_vectors_esp32c6_and_host_verify_link() {
        let dir = test_out_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let mir = build_wave1_mmio_irq_test_module();

        // 1. ESP32-C6 (`rv32imac`) IR, `opt -passes=verify`, `llc`, `ld.lld --gc-sections`, and ELF audit
        let esp_opts = CodegenOptions::esp32c6();
        let esp_ll = emit_llvm_ir_with_options(&mir, &esp_opts).expect("emit ESP32-C6 Wave 1 IR");
        assert!(esp_ll.contains("section \".iram1\", align 256"));
        assert!(esp_ll.contains(".balign 256"));
        assert!(esp_ll.contains("__syn_rv_vector_table:"));
        assert!(esp_ll.contains("j __syn_rv_irq_tramp_1"));
        assert!(esp_ll.contains("j __syn_rv_irq_tramp_2"));
        assert!(!esp_ll.contains("j __syn_rv_irq_tramp_3"));
        assert!(!esp_ll.contains("j __syn_rv_irq_tramp_4"));
        assert!(esp_ll.contains("j __syn_rv_irq_tramp_5"));
        assert!(esp_ll.contains("j __syn_rv_irq_tramp_6"));
        assert!(esp_ll.contains("call __syn_vec_irq_5"));
        assert!(esp_ll.contains("call __syn_vec_irq_12"));
        assert!(esp_ll.contains("call __syn_vec_irq_19"));
        assert!(esp_ll.contains("call __syn_vec_irq_37"));
        assert!(esp_ll.contains("0x20001008"));
        assert!(esp_ll.contains("INTMTX src_id=5 -> cpu_int=1 (0x60010014)"));
        assert!(esp_ll.contains("INTMTX src_id=37 -> cpu_int=6 (0x60010094)"));
        assert!(esp_ll.contains("PLIC_MXINT1_PRI_REG (0x20001014) = 1"));
        assert!(esp_ll.contains("PLIC_MXINT_TYPE_REG (0x20001004)"));
        assert!(esp_ll.contains("PLIC_MXINT_THRESH_REG (0x20001090) = 1"));
        assert!(esp_ll.contains("PLIC_MXINT_ENABLE_REG (0x20001000)"));
        assert!(esp_ll.contains("csrw mtvec, $0"));
        assert!(esp_ll.contains("csrw mie, $0"));
        assert!(esp_ll.contains("csrsi mstatus, 8"));
        assert!(esp_ll.contains("call void @__syn_vec_irq_37()"));

        let esp_elf = dir.join("wave1_esp32c6.elf");
        let esp_bin = dir.join("wave1_esp32c6.bin");
        let img = build_esp32c6_image(&mir, &esp_elf, &esp_bin)
            .expect("build Wave 1 ESP32-C6 Direct-Boot image");
        verify_ll_with_opt(&img.elf.ll_path).expect("opt verify ESP32-C6 Wave 1 IR");
        assert!(img.elf.audit.is_zero_heap_freestanding());
        assert_eq!(img.magic_header, ESP32C6_DIRECT_BOOT_MAGIC);
        assert!(
            img.elf
                .audit
                .defined_symbols
                .iter()
                .any(|s| s == "__syn_rv_vector_table")
        );
        assert!(
            img.elf
                .audit
                .defined_symbols
                .iter()
                .any(|s| s == "__syn_vec_irq_37")
        );

        let nm = Command::new("/bin/llvm-nm")
            .arg(&img.elf.elf_path)
            .output()
            .expect("invoke /bin/llvm-nm");
        assert!(nm.status.success());
        let nm_out = String::from_utf8_lossy(&nm.stdout);
        let vt_line = nm_out
            .lines()
            .find(|l| l.ends_with(" __syn_rv_vector_table"))
            .unwrap_or_else(|| panic!("missing __syn_rv_vector_table in llvm-nm:\n{nm_out}"));
        let vt_addr = u32::from_str_radix(vt_line.split_whitespace().next().unwrap(), 16).unwrap();
        assert_eq!(
            vt_addr % 256,
            0,
            "expected __syn_rv_vector_table to be 256-byte aligned, got 0x{vt_addr:08X}"
        );
        assert!(
            vt_addr >= 0x4080_0000,
            "expected __syn_rv_vector_table in HP-SRAM .iram1, got 0x{vt_addr:08X}"
        );

        // 2. Host x86_64 LLVM IR (`@__zev_mmio_buf_*`, bounds clamping, `opt -passes=verify` & `llc`)
        let mut host_opts = CodegenOptions::default();
        host_opts.target = TargetArch::HostX86_64Linux;
        let host_ll = emit_llvm_ir_with_options(&mir, &host_opts).expect("emit Host Wave 1 IR");
        assert!(
            host_ll.contains("@__zev_mmio_buf_SCRATCH_A = internal global [24 x i8] zeroinitializer, align 4"),
            "host_ll={host_ll}"
        );
        assert!(
            host_ll.contains("@__zev_mmio_buf_UART0 = internal global [4 x i8] zeroinitializer, align 4"),
            "host_ll={host_ll}"
        );
        assert!(host_ll.contains("icmp ult i64 "));
        assert!(host_ll.contains("select i1 "));

        let host_ll_path = dir.join("wave1_host.ll");
        let host_obj_path = dir.join("wave1_host.o");
        std::fs::write(&host_ll_path, &host_ll).unwrap();
        verify_ll_with_opt(&host_ll_path).expect("opt verify Host Wave 1 IR");
        compile_ll_to_obj(&host_ll_path, &host_obj_path, &host_opts)
            .expect("llc compile Host Wave 1 object");
    }

    #[test]
    fn wave1_isr_prim_addr_and_state_addrof_in_qemu() {
        use zev_mir::{
            MirAssignOp, MirBakedArg, MirBinOp, MirBootSequence, MirBootStep, MirExpr,
            MirExprKind, MirFmtChunk, MirIntFmt, MirIntrinsicOp, MirIrqBinding, MirIrqVector,
            MirIsr, MirIsrStateDecl, MirIsrStateVar, MirPlace, MirPrimKind, MirPrimNonBlockingOp,
            MirPrimReadOp, MirPrimTarget, MirPrimitive, MirStmt, MirType, MirUnaryOp, MirVar,
            VarId, VarRole,
        };

        let dir = test_out_dir();
        std::fs::create_dir_all(&dir).unwrap();

        let mk_var = |id: usize, name: &str, ty: MirType, role: VarRole| MirVar {
            id: VarId(id),
            name: name.to_string(),
            ty,
            role,
            in_noyield_scope: true,
            addr_taken: false,
            hoisted: false,
        };
        let u32_expr = |v: i128| MirExpr::new(MirExprKind::IntLit(v), MirType::U32);
        let var_expr = |id: usize, ty: MirType| MirExpr::new(MirExprKind::Var(VarId(id)), ty);

        let inbox_ty = MirType::Primitive(MirPrimKind::Inbox {
            elem: Box::new(MirType::U32),
            cap: 4,
        });
        let ptr_inbox_ty = MirType::Ptr(Box::new(inbox_ty));

        // ISR taking `*inbox<u32, 4> q` (`MirBakedArg::PrimAddr`) and mutating `state u32 acc = 5` via `&acc` (`AddrOfPlace`)!
        let mut state_var = mk_var(1, "acc", MirType::U32, VarRole::IsrState);
        state_var.addr_taken = true;
        let ptr_u32_ty = MirType::Ptr(Box::new(MirType::U32));
        let isr_handler = MirIsr {
            name: "prim_state_worker".to_string(),
            params: vec![VarId(0)],
            state_vars: vec![MirIsrStateDecl {
                var_id: VarId(1),
                name: "acc".to_string(),
                ty: MirType::U32,
                init: u32_expr(5),
            }],
            vars: vec![
                mk_var(0, "q", ptr_inbox_ty.clone(), VarRole::Param),
                state_var,
                mk_var(2, "p_acc", ptr_u32_ty.clone(), VarRole::UserLocal),
            ],
            body: vec![
                MirStmt::Assign {
                    target: MirPlace::Var(VarId(2)),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(
                        MirExprKind::AddrOfPlace(Box::new(MirPlace::Var(VarId(1)))),
                        ptr_u32_ty.clone(),
                    ),
                },
                MirStmt::Assign {
                    target: MirPlace::Deref(
                        Box::new(var_expr(2, ptr_u32_ty.clone())),
                        MirType::U32,
                    ),
                    op: MirAssignOp::AddAssign,
                    value: u32_expr(7),
                },
                MirStmt::PrimOp(MirPrimNonBlockingOp::InboxPost {
                    target: MirPrimTarget::PtrExpr(var_expr(0, ptr_inbox_ty)),
                    elem_ty: MirType::U32,
                    cap: 4,
                    value: MirExpr::new(
                        MirExprKind::Unary {
                            op: MirUnaryOp::Deref,
                            expr: Box::new(var_expr(2, ptr_u32_ty)),
                        },
                        MirType::U32,
                    ),
                    boost: None,
                }),
            ],
        };

        let try_recv_res_ty = MirType::Result {
            ok: Box::new(MirType::U32),
            err: None,
        };
        let try_recv_expr = MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxTryRecv {
                target: MirPrimTarget::Global("ev_q".to_string()),
                elem_ty: MirType::U32,
                cap: 4,
            })),
            try_recv_res_ty.clone(),
        );

        let dummy_prog = zev_parser::parse_str("fn main() {}").unwrap();
        let mut mir = zev_mir::lower_program(&dummy_prog).unwrap();
        mir.primitives = vec![MirPrimitive {
            name: "ev_q".to_string(),
            kind: MirPrimKind::Inbox {
                elem: Box::new(MirType::U32),
                cap: 4,
            },
            dma_safe: false,
            init: None,
            sleep_lock_bit: None,
            dead_stripped: false,
            irq_mask_elided: false,
        }];
        mir.isrs = vec![isr_handler];
        mir.irq_vectors = vec![MirIrqVector {
            irq_name: "IRQ_3".to_string(),
            irq_num: 3,
            is_shared: false,
            selector_symbol: None,
            bindings: vec![MirIrqBinding {
                binding_id: 0,
                isr_name: "prim_state_worker".to_string(),
                baked_args: vec![MirBakedArg::PrimAddr("ev_q".to_string())],
                state_vars: vec![MirIsrStateVar {
                    storage_symbol: "__zev_isr_state_99_acc".to_string(),
                    var_name: "acc".to_string(),
                    ty: MirType::U32,
                    init: u32_expr(5),
                }],
                cond_selector: None,
            }],
        }];
        mir.boot = MirBootSequence {
            present: true,
            vars: vec![
                mk_var(0, "r0", try_recv_res_ty.clone(), VarRole::CompilerTemp),
                mk_var(1, "r1", try_recv_res_ty, VarRole::CompilerTemp),
            ],
            steps: vec![
                MirBootStep::Stmt(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                    method: "pend_irq".to_string(),
                    args: vec![u32_expr(3)],
                })),
                MirBootStep::Stmt(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                    method: "pend_irq".to_string(),
                    args: vec![u32_expr(3)],
                })),
                MirBootStep::Stmt(MirStmt::Assign {
                    target: MirPlace::Var(VarId(0)),
                    op: MirAssignOp::Assign,
                    value: try_recv_expr.clone(),
                }),
                MirBootStep::Stmt(MirStmt::Assign {
                    target: MirPlace::Var(VarId(1)),
                    op: MirAssignOp::Assign,
                    value: try_recv_expr,
                }),
                MirBootStep::Stmt(MirStmt::PrintFmt {
                    chunks: vec![
                        MirFmtChunk::Literal("PRIM_ADDR=".to_string()),
                        MirFmtChunk::Int {
                            value: MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Add,
                                    lhs: Box::new(MirExpr::new(
                                        MirExprKind::ResultVal(VarId(0)),
                                        MirType::U32,
                                    )),
                                    rhs: Box::new(MirExpr::new(
                                        MirExprKind::ResultVal(VarId(1)),
                                        MirType::U32,
                                    )),
                                },
                                MirType::U32,
                            ),
                            signed: false,
                            bits: 32,
                            fmt: MirIntFmt::Dec,
                        },
                    ],
                    newline: true,
                }),
                MirBootStep::Stmt(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                    method: "exit".to_string(),
                    args: vec![u32_expr(0)],
                })),
            ],
            boot_yielding_callees: vec![],
        };

        let res = run_qemu_cortex_m(
            &mir,
            &dir.join("wave1_prim_addr_state_addrof.elf"),
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("QEMU prim_addr + state addrof run");
        verify_ll_with_opt(&res.build.ll_path).expect("opt verify");
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert_eq!(res.exec.exit_code, 0);
        // 1st fire: 5 + 7 = 12; 2nd fire: 12 + 7 = 19; sum = 31!
        assert_eq!(res.exec.stdout, "PRIM_ADDR=31\n");
    }

    #[test]
    fn wave1_audit_backend_fixes_and_qemu_parity() {
        use std::process::Command;
        use zev_mir::{MirIrqBinding, MirIrqVector, MirIsr};

        let dir = test_out_dir();
        std::fs::create_dir_all(&dir).unwrap();

        // 1. ESP32-C6 CLINT line skipping (0, 3, 4, 7), src_id >= 78 error, and > 28 vectors error.
        let mk_vec = |irq_num: u32| MirIrqVector {
            irq_name: format!("IRQ_{irq_num}"),
            irq_num,
            is_shared: false,
            selector_symbol: None,
            bindings: vec![MirIrqBinding {
                binding_id: 0,
                isr_name: "isr_h".to_string(),
                baked_args: vec![],
                state_vars: vec![],
                cond_selector: None,
            }],
        };
        let six_routes = esp32c6::assign_esp32c6_irq_routes(&(0..6u32).map(mk_vec).collect::<Vec<_>>())
            .expect("6 routes");
        let assigned_cpus: Vec<u8> = six_routes.iter().map(|r| r.cpu_int).collect();
        assert_eq!(assigned_cpus, vec![1, 2, 5, 6, 8, 9]);

        let oob_src = [mk_vec(77)];
        assert!(
            esp32c6::assign_esp32c6_irq_routes(&oob_src).is_err(),
            "ESP32-C6 INTMTX src_id=77 must fail"
        );
        let too_many: Vec<_> = (0..29u32).map(mk_vec).collect();
        assert!(
            esp32c6::assign_esp32c6_irq_routes(&too_many).is_err(),
            ">28 ESP32-C6 external IRQ sources must fail"
        );

        // Cortex-M3 / Host compilation MUST still allow irq_num >= 77!
        let cm_prog = zev_parser::parse_str("fn main() {}").unwrap();
        let mut cm_mod = zev_mir::lower_program(&cm_prog).unwrap();
        cm_mod.isrs.push(MirIsr {
            name: "isr_h".to_string(),
            params: vec![],
            vars: vec![],
            state_vars: vec![],
            body: vec![],
        });
        cm_mod.irq_vectors.push(mk_vec(77));
        assert!(emit_llvm_ir_with_options(&cm_mod, &CodegenOptions::default()).is_ok());

        // 2. Shared IRQ vector dispatch preserves exact source attachment order across unconditional
        //    and conditional bindings in both LLVM IR and QEMU execution!
        let shared_src = r#"
irq IRQ_10 = 10;
inbox<u32, 8> log_q;

isr step_1() { log_q.post(1); }
isr step_2() { log_q.post(2); }
isr step_3() { log_q.post(3); }

bool get_cond() { return true; }

fn main() {
    IRQ_10.attach_shared(step_1);
    if (get_cond()) {
        IRQ_10.attach_shared(step_2);
    }
    IRQ_10.attach_shared(step_3);
    sys.pend_irq(10);
    u32 a = log_q.try_recv() or 0;
    u32 b = log_q.try_recv() or 0;
    u32 c = log_q.try_recv() or 0;
    println(f"ORDER={a},{b},{c}");
    sys.exit(0);
}
"#;
        let shared_prog = zev_parser::parse_str(shared_src).unwrap();
        let shared_mir = zev_mir::lower_program(&shared_prog).unwrap();
        let qres = run_qemu_cortex_m(
            &shared_mir,
            &dir.join("wave1_shared_order.elf"),
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("QEMU shared IRQ order run");
        verify_ll_with_opt(&qres.build.ll_path).expect("opt verify shared order");
        let shared_ll = std::fs::read_to_string(&qres.build.ll_path).unwrap();
        let pos_1 = shared_ll
            .find("call void @__syn_isr_step_1()")
            .expect("step_1 call");
        let pos_2 = shared_ll
            .find("call void @__syn_isr_step_2()")
            .expect("step_2 call");
        let pos_3 = shared_ll
            .find("call void @__syn_isr_step_3()")
            .expect("step_3 call");
        assert!(pos_1 < pos_2 && pos_2 < pos_3, "shared IRQ bindings out of order:\n{shared_ll}");
        assert_eq!(qres.exec.stdout, "ORDER=1,2,3\n");

        // 3. Host x86_64 64-bit dynamic MMIO index OOB check (`0x1_0000_0001` clamps to 0, not 1!).
        let host_src = r#"
mmio ArrRegs {
    @rw [4]u32 arr @ 0x0;
}
mmio DEV : ArrRegs @ 0x4000_0000;

usize make_idx(u64 hi, u64 lo) {
    u64 full = (hi << 32) | lo;
    return full as usize;
}

fn main() {
    DEV.arr[0] = 10;
    DEV.arr[1] = 20;
    usize oob = make_idx(1, 1);
    DEV.arr[oob] = 99;
    println(f"HOST={DEV.arr[0]},{DEV.arr[1]}");
}
"#;
        let host_prog = zev_parser::parse_str(host_src).unwrap();
        let host_mir = zev_mir::lower_program(&host_prog).unwrap();
        let mut h_opts = CodegenOptions::default();
        h_opts.target = TargetArch::HostX86_64Linux;
        h_opts.opt_level = "-O0".to_string();
        let host_ll_64 = emit_llvm_ir_with_options(&host_mir, &h_opts).expect("emit host 64 IR");
        let h_ll_path = dir.join("wave1_host_64idx.ll");
        let h_obj_path = dir.join("wave1_host_64idx.o");
        let h_exe_path = dir.join("wave1_host_64idx");
        std::fs::write(&h_ll_path, &host_ll_64).unwrap();
        verify_ll_with_opt(&h_ll_path).expect("opt verify host 64 idx");
        compile_ll_to_obj(&h_ll_path, &h_obj_path, &h_opts).expect("llc host 64 idx");
        let link = Command::new("/bin/clang")
            .args([
                h_obj_path.to_str().unwrap(),
                "-o",
                h_exe_path.to_str().unwrap(),
            ])
            .output()
            .expect("link host exe");
        assert!(
            link.status.success(),
            "clang link failed: {}",
            String::from_utf8_lossy(&link.stderr)
        );
        let run = Command::new(&h_exe_path).output().expect("run host exe");
        assert!(run.status.success());
        assert_eq!(String::from_utf8_lossy(&run.stdout), "HOST=99,20\n");
    }

    #[test]
    fn arduino_uno_demo_builds_avr_bin_and_passes_audit() {
        let demo_path = repo_root().join("examples/arduino_uno_demo.zev");
        let mir = compile_example_mir(&[demo_path]);

        assert_eq!(mir.ram_report.mcu32.heap_bytes, 0);

        let dir = test_out_dir();
        let out_elf = dir.join("arduino_uno_demo.elf");
        let out_bin = dir.join("arduino_uno_demo.bin");

        let img = build_avr_uno_image(&mir, &out_elf, &out_bin)
            .unwrap_or_else(|e| panic!("ATmega328P AVR image build failed: {e}"));

        verify_ll_with_opt(&img.elf.ll_path).expect("LLVM 23 opt -passes=verify on arduino_uno_demo.ll");

        assert!(img.elf.audit.is_zero_heap_freestanding());
        for req_sym in [
            "__syn_avr_reset",
            "__syn_avr_systick_isr",
            "__syn_avr_vec_3",
            "__syn_avr_vec_11",
            "__syn_avr_vec_18",
            "__syn_avr_vec_19",
        ] {
            assert!(
                img.elf.audit.defined_symbols.iter().any(|s| s == req_sym),
                "missing AVR symbol {req_sym} in {:?}",
                img.elf.audit.defined_symbols
            );
        }
        assert!(
            img.bin_size > 0 && img.bin_size <= 32_256,
            "expected AVR Flash image <= 32256 B, got {}",
            img.bin_size
        );
        assert!(
            img.sram_static_bytes > 0 && img.sram_static_bytes < 1_792,
            "expected AVR static SRAM < 1792 B, got {}",
            img.sram_static_bytes
        );

        let objdump = Command::new("/bin/llvm-objdump")
            .args(["-d", "--start-address=0x0000", "--stop-address=0x0068"])
            .arg(&img.elf.elf_path)
            .output()
            .expect("invoke llvm-objdump on AVR ELF");
        assert!(objdump.status.success());
        let disasm = String::from_utf8_lossy(&objdump.stdout);
        assert!(disasm.contains("00000000 <_start>:"), "disasm:\n{disasm}");
        assert!(disasm.contains("__syn_avr_reset") || disasm.contains("jmp"), "disasm:\n{disasm}");

        let full_objdump = Command::new("/bin/llvm-objdump")
            .arg("-d")
            .arg(&img.elf.elf_path)
            .output()
            .expect("invoke full llvm-objdump on AVR ELF");
        assert!(full_objdump.status.success());
        let full_disasm = String::from_utf8_lossy(&full_objdump.stdout);
        assert!(
            !full_disasm.contains("in\tr26, 0x3d"),
            "found forbidden AVR dynamic stack alloca in ELF disassembly"
        );
    }

    #[test]
    fn avr_irq_vector_validation_rejects_reserved_and_out_of_range_slots() {
        assert!(avr::validate_avr_irq_vectors(&[]).expect("empty ok").is_empty());

        let make_vec = |irq_num: u32, irq_name: &str| zev_mir::MirIrqVector {
            irq_name: irq_name.to_string(),
            irq_num,
            is_shared: false,
            selector_symbol: None,
            bindings: vec![zev_mir::MirIrqBinding {
                binding_id: 0,
                isr_name: "dummy_isr".to_string(),
                baked_args: Vec::new(),
                state_vars: Vec::new(),
                cond_selector: None,
            }],
        };

        let attached = avr::validate_avr_irq_vectors(&[make_vec(11, "IRQ_TIMER1_COMPA")])
            .expect("slot 11 is valid on ATmega328P");
        assert!(attached.contains(&11));

        for (bad_num, name) in [(0, "RESET"), (14, "TIMER0_COMPA"), (26, "OOB")] {
            let err = avr::validate_avr_irq_vectors(&[make_vec(bad_num, name)])
                .expect_err("reserved/out-of-range vector must be rejected");
            assert!(
                !err.message.is_empty(),
                "expected descriptive validation error for vector {bad_num}"
            );
        }
    }

    #[test]
    fn byte_ring_isr_push_bipbuffer_and_wait_read_in_qemu() {
        let src = r#"
byte_ring<16> uart_rx;
signal done_sig;
irq IRQ_EXTI4 = 4;

isr uart_isr(*byte_ring<16> rx) {
    state u8 byte_seq = 65;
    rx.push_byte(byte_seq, boost: @critical);
    byte_seq += 1;
}

task parser_task(*byte_ring<16> rx, *signal done) {
    u32 got_bytes = 0;
    loop {
        []u8 chunk = rx.wait_read();
        usize n = chunk.len;
        if (n > 0) {
            u32 prio = sys.current_prio();
            u8 first = chunk[0];
            println(f"RX len={n} first={first} prio={prio}");
            got_bytes += n as u32;
            rx.consume(n);
        }
        if (got_bytes >= 11) {
            done.notify();
        }
    }
}

fn main() {
    sys.use_systick(12_000_000);
    IRQ_EXTI4.attach(uart_isr(&uart_rx));

    // 1. Wrap-around + BipBuffer early wrap in pure LLVM IR
    assert(uart_rx.push("0123456789AB") == 12);
    u8 pre[8];
    assert(uart_rx.pop(pre) == 8);
    assert(pre[0] == '0' && pre[7] == '7');

    // Back free=4 (12..16), front free=8 (0..8) -> grant_write(6) wraps to index 0!
    []u8 wr = uart_rx.grant_write(6) catch {
        sys.exit(1);
    };
    wr[0] = 'X';
    wr[1] = 'Y';
    wr[2] = 'Z';
    uart_rx.commit_write(3);

    u8 tail_a[4];
    assert(uart_rx.pop(tail_a) == 4);
    assert(tail_a[0] == '8' && tail_a[3] == 'B');

    assert((uart_rx.try_pop_byte() or 0) == 'X');
    assert((uart_rx.try_pop_byte() or 0) == 'Y');
    assert((uart_rx.try_pop_byte() or 0) == 'Z');
    assert(uart_rx.is_empty());

    // 2. Spawn parser_task and drive it from NVIC ISR + slice push
    spawn parser_task(&uart_rx, &done_sig) @low;
    set_priority(@low);
    sys.pend_irq(4);
    delay(2ms);
    assert(uart_rx.push("HELLO_RING") == 10);
    done_sig.wait();
    println("[PASS] BYTE_RING QEMU OK");
    sys.exit(0);
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse byte_ring QEMU test");
        zev_sema::analyze_program(&prog).expect("sema byte_ring QEMU test");
        let mir = zev_mir::lower_program(&prog).expect("lower byte_ring QEMU test");
        let dir = test_out_dir();
        let elf = dir.join("byte_ring_qemu.elf");
        let res = run_qemu_cortex_m(
            &mir,
            &elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .expect("run byte_ring Cortex-M3 QEMU");
        assert_eq!(res.exec.exit_code, 0);
        assert!(res.build.audit.is_zero_heap_freestanding());
        assert!(
            res.exec.stdout.contains("RX len=1 first=65 prio=0"),
            "stdout={}",
            res.exec.stdout
        );
        assert!(
            res.exec.stdout.contains("RX len=10 first=72 prio=3"),
            "stdout={}",
            res.exec.stdout
        );
        assert!(
            res.exec.stdout.contains("[PASS] BYTE_RING QEMU OK"),
            "stdout={}",
            res.exec.stdout
        );
    }

    fn run_zev_cortex_m_qemu(
        test_name: &str,
        src: &str,
    ) -> (zev_mir::MirModule, String, crate::QemuRunResult) {
        let prog = zev_parser::parse_str(src).unwrap_or_else(|e| panic!("parse {test_name}: {e:?}"));
        zev_sema::analyze_program(&prog)
            .unwrap_or_else(|e| panic!("sema {test_name}: {e:?}"));
        let mir = zev_mir::lower_program(&prog).unwrap_or_else(|e| panic!("lower {test_name}: {e:?}"));
        let dir = test_out_dir();
        let elf = dir.join(format!("{test_name}.elf"));
        let res = run_qemu_cortex_m(
            &mir,
            &elf,
            &CodegenOptions::default(),
            &QemuRunOptions::default(),
        )
        .unwrap_or_else(|e| panic!("run {test_name} QEMU: {e}"));
        assert_eq!(res.exec.exit_code, 0, "{test_name} exit != 0; stdout={}", res.exec.stdout);
        assert!(res.build.audit.is_zero_heap_freestanding());
        let ll = std::fs::read_to_string(&res.build.ll_path).expect("read generated .ll");
        (mir, ll, res)
    }

    fn assert_all_allocas_in_entry(ll: &str) {
        let mut in_fn = false;
        let mut in_entry = false;
        for line in ll.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("define ") {
                in_fn = true;
                in_entry = false;
            } else if trimmed == "}" {
                in_fn = false;
                in_entry = false;
            } else if in_fn && trimmed == "entry:" {
                in_entry = true;
            } else if in_fn && trimmed.ends_with(':') && !trimmed.contains(' ') {
                in_entry = false;
            } else if in_fn && trimmed.contains("= alloca ") && !in_entry {
                panic!("non-entry alloca found in LLVM IR: {trimmed}");
            }
        }
    }

    #[test]
    fn unsized_views_byte_ring_and_inbox_multi_cap_in_qemu_and_avr() {
        let src = r#"
struct ViewBundle {
    *byte_ring ring;
    *inbox<u32> q;
}

byte_ring<12> ring_a;
byte_ring<24> ring_b;
inbox<u32, 3> q_a;
inbox<u32, 6> q_b;

usize exercise_ring(*byte_ring r, u8 seed) {
    usize cap = r.cap();
    assert(r.is_empty());
    assert(!r.is_full());
    assert(r.free_space() == cap);
    assert(r.push("ABC", boost: @high) == 3);
    assert(r.push_byte(seed));
    assert(r.len() == 4);

    []u8 chunk = r.read_slice();
    assert(chunk.len == 4);
    assert(chunk[0] == 'A');
    assert(chunk[3] == seed);
    r.consume(2);
    assert(r.len() == 2);

    u8 popped[4];
    usize npop = r.pop(popped);
    assert(npop == 2);
    assert(popped[0] == 'C');
    assert(popped[1] == seed);
    u8 pop_sentinel = r.try_pop_byte() catch { 0xEE };
    assert(pop_sentinel == 0xEE);

    []u8 dma = r.grant_write(3) catch { return 0; };
    dma[0] = 'X';
    dma[1] = 'Y';
    dma[2] = 'Z';
    r.commit_write(2, boost: @low);
    assert(r.len() == 2);
    u8 b0 = r.try_pop_byte() catch { return 0; };
    assert(b0 == 'X');
    r.clear();
    assert(r.is_empty());
    return cap;
}

usize exercise_inbox(*inbox<u32> q) {
    usize cap = q.cap();
    assert(q.is_empty());
    assert(!q.is_full());
    q.post(10);
    assert(q.try_send(20, boost: @high));
    assert(q.try_send(30));
    assert(q.len() == 3);
    if (cap == 3) {
        assert(q.is_full());
        assert(!q.try_send(40));
    } else {
        assert(!q.is_full());
    }
    u32 v1 = q.try_recv() catch { return 0; };
    u32 v2 = q.try_recv() catch { return 0; };
    u32 v3 = q.try_recv() catch { return 0; };
    assert(v1 == 10 && v2 == 20 && v3 == 30);
    u32 empty_sentinel = q.try_recv() catch { 9999 };
    assert(empty_sentinel == 9999);
    return cap;
}

task consumer_task(ViewBundle vb) {
    set_priority(@low);
    u8 i = 0;
    while (i < 2) {
        i += 1;
        []u8 chunk = vb.ring.wait_read(timeout: 20ms) catch {
            println("RING_TIMEOUT");
            sys.exit(0);
        };
        println(f"QEMU_VIEW_RX len={chunk.len} b0={chunk[0]} prio={sys.current_prio()}");
        vb.ring.consume(chunk.len);
        u32 msg = vb.q.recv(timeout: 20ms) catch {
            println("INBOX_TIMEOUT");
            sys.exit(0);
        };
        println(f"QEMU_VIEW_MSG={msg}");
    }
    println("[PASS] VIEWS_QEMU_OK");
    sys.exit(0);
}

task producer_task(*byte_ring r, *inbox<u32> q) {
    sys.delay(2ms);
    r.push("HI", boost: @critical);
    q.send(77, timeout: 10ms) catch { loop { sys.delay(10ms); } };
    sys.delay(3ms);
    r.push("OK", boost: @high);
    q.send(88, timeout: 10ms) catch { loop { sys.delay(10ms); } };
    loop { sys.delay(10ms); }
}

fn main() {
    sys.use_systick(12_000_000);
    assert(exercise_ring(&ring_a, 99) == 12);
    assert(exercise_ring(&ring_b, 42) == 24);
    assert(exercise_inbox(&q_a) == 3);
    assert(exercise_inbox(&q_b) == 6);
    spawn consumer_task(ViewBundle{ ring: &ring_a, q: &q_a }) @low;
    spawn producer_task(&ring_a, &q_a) @normal;
}
"#;
        let (mir, ll, res) =
            run_zev_cortex_m_qemu("unsized_views_multi_cap_qemu", src);
        assert_all_allocas_in_entry(&ll);
        assert!(
            res.exec.stdout.contains("QEMU_VIEW_RX len=2 b0=72 prio=0"),
            "stdout={}",
            res.exec.stdout
        );
        assert!(
            res.exec.stdout.contains("QEMU_VIEW_MSG=77")
                && res.exec.stdout.contains("QEMU_VIEW_MSG=88")
                && res.exec.stdout.contains("[PASS] VIEWS_QEMU_OK"),
            "stdout={}",
            res.exec.stdout
        );

        // Verify ATmega328P AVR build + opt verify passes with zero non-entry allocas
        let dir = test_out_dir();
        let avr_img = build_avr_uno_image(
            &mir,
            &dir.join("unsized_views_avr.elf"),
            &dir.join("unsized_views_avr.bin"),
        )
        .expect("AVR build unsized views");
        verify_ll_with_opt(&avr_img.elf.ll_path).expect("LLVM verify AVR unsized views");
        let avr_ll = std::fs::read_to_string(&avr_img.elf.ll_path).expect("read avr ll");
        assert_all_allocas_in_entry(&avr_ll);
        assert!(avr_img.elf.audit.is_zero_heap_freestanding());
    }

    #[test]
    fn subsystem_structs_nested_isr_and_view_coercion_in_qemu() {
        let src = r#"
irq UartIrq = 4;

struct Channel {
    byte_ring<16> tx;
    inbox<u16, 4> q;
}

struct UartSubsys {
    byte_ring<12> rx;
    Channel ch0;
}

struct SyncSubsys {
    signal ready;
}

UartSubsys uart0;
UartSubsys uart1;
SyncSubsys sync_box;

isr on_rx(*byte_ring r, *inbox<u16> q) {
    r.push("AB", boost: @high);
    q.post(1234, boost: @high);
}

fn check_channel(*Channel ch, u16 tag) {
    assert(ch.tx.cap() == 16);
    assert(ch.q.cap() == 4);
    usize n = fmt_ring(&ch.tx, f"tag={tag}") catch { 0 };
    assert(n > 0);
}

task subsys_worker(*UartSubsys drv) {
    set_priority(@low);
    sync_box.ready.wait();
    assert(sys.current_prio() == 3);
    check_channel(&drv.ch0, 7);
    []u8 s = drv.rx.wait_read(timeout: 20ms) catch {
        println("SUBSYS_RX_TO");
        sys.exit(0);
    };
    println(f"QEMU_SUBSYS_RX len={s.len} b0={s[0]} prio={sys.current_prio()}");
    drv.rx.consume(s.len);
    u16 code = drv.ch0.q.recv(timeout: 20ms) catch {
        println("SUBSYS_Q_TO");
        sys.exit(0);
    };
    println(f"QEMU_SUBSYS_Q code={code}");
    []u8 txs = drv.ch0.tx.read_slice();
    println(f"QEMU_SUBSYS_TX len={txs.len} b0={txs[0]}");
    drv.ch0.tx.consume(txs.len);
    sys.exit(0);
}

task irq_driver() {
    sys.delay(1ms);
    sync_box.ready.notify();
    sys.delay(2ms);
    sys.pend_irq(4);
    loop { sys.delay(10ms); }
}

fn main() {
    sys.use_systick(12_000_000);
    check_channel(&uart1.ch0, 9);
    assert(uart1.ch0.tx.len() == 5);
    UartIrq.attach(on_rx(&uart0.rx, &uart0.ch0.q));
    spawn subsys_worker(&uart0) @low;
    spawn irq_driver() @normal;
}
"#;
        let (_mir, ll, res) =
            run_zev_cortex_m_qemu("subsystem_structs_nested_isr_qemu", src);
        assert_all_allocas_in_entry(&ll);
        assert!(
            ll.contains("@__zev_subsys_uart0 = internal global %struct.UartSubsys zeroinitializer")
                && ll.contains("@__zev_subsys_uart1 = internal global %struct.UartSubsys zeroinitializer"),
            "expected static subsystem globals in LLVM IR"
        );
        assert!(
            !ll.contains("@__syn_from_bytes_UartSubsys") && !ll.contains("@__syn_from_bytes_Channel"),
            "subsystem structs must not emit from_bytes helpers"
        );
        assert!(
            res.exec.stdout.contains("QEMU_SUBSYS_RX len=2 b0=65 prio=1")
                && res.exec.stdout.contains("QEMU_SUBSYS_Q code=1234")
                && res.exec.stdout.contains("QEMU_SUBSYS_TX len=5 b0=116"),
            "stdout={}",
            res.exec.stdout
        );
    }

    #[test]
    fn fmt_ring_wrap_single_wake_boost_and_truncation_in_qemu() {
        let src = r#"
byte_ring<16> tx;
signal sync;

usize write_fmt_view(*byte_ring r, u16 v) error FmtErr {
    return fmt_ring(r, f"V={v:04X}", boost: @high) catch (FmtErr e, usize pw) {
        error e, pw;
    };
}

task reader() {
    set_priority(@low);
    sync.notify();
    []u8 cw = tx.wait_read(timeout: 20ms) catch {
        println("FAIL_WAIT1");
        sys.exit(0);
    };
    u32 prio = sys.current_prio();
    usize cw_len = cw.len;
    u8 first_b = cw[0];
    assert(cw_len == 8);
    tx.consume(cw_len);
    println(f"QEMU_FMT_WAKE len={cw_len} b0={first_b} prio={prio}");

    assert(tx.push("0123456789A") == 11);
    u8 trash[10];
    assert(tx.pop(trash) == 10);
    usize w = write_fmt_view(&tx, 0x1A2B) catch { 0 };
    assert(w == 6);

    []u8 c1 = tx.read_slice();
    usize c1_len = c1.len;
    assert(c1_len == 6);
    tx.consume(c1_len);
    []u8 c2 = tx.read_slice();
    assert(c2.len == 1);
    u8 last_b = c2[0];
    tx.consume(c2.len);
    println(f"QEMU_FMT_WRAP c1len={c1_len} last={last_b}");

    assert(tx.push("123456789012") == 12);
    usize partial_w = fmt_ring(&tx, f"ABCD1234") catch (FmtErr e, usize pw) {
        println(f"QEMU_FMT_TRUNC partial={pw} total_len={tx.len()}");
        sys.exit(0);
    };
    println(f"UNEXPECTED_OK={partial_w}");
    sys.exit(0);
}

task writer() {
    sync.wait();
    sys.delay(2ms);
    u8 x = 0x2A;
    u8 y = 7;
    usize w = fmt_ring(&tx, f"X={x:02X} Y={y}", boost: @high) catch { 0 };
    assert(w == 8);
    loop { sys.delay(10ms); }
}

fn main() {
    sys.use_systick(12_000_000);
    spawn reader() @low;
    spawn writer() @normal;
}
"#;
        let (_mir, ll, res) =
            run_zev_cortex_m_qemu("fmt_ring_wrap_trunc_qemu", src);
        assert_all_allocas_in_entry(&ll);
        assert!(
            res.exec.stdout.contains("QEMU_FMT_WAKE len=8 b0=88 prio=1")
                && res.exec.stdout.contains("QEMU_FMT_WRAP c1len=6 last=66")
                && res.exec.stdout.contains("QEMU_FMT_TRUNC partial=4 total_len=16"),
            "stdout={}",
            res.exec.stdout
        );
    }

    #[test]
    fn hardware_stack_high_water_proof_passes_on_cortex_m3_esp32c6_and_avr() {
        // 1. ARM Cortex-M3 (`examples/cortex_m_demo.zev`, 65,536 B SRAM, 32 B NVIC frame)
        let cm3_mir = compile_example_mir(&[repo_root().join("examples/cortex_m_demo.zev")]);
        let cm3_proof =
            analyze_hardware_stack_high_water(&cm3_mir, &CodegenOptions::default())
                .expect("Cortex-M3 stack high-water proof");
        println!("{}", cm3_proof.format_table());
        assert_eq!(cm3_proof.sram_budget_bytes, 65_536);
        assert_eq!(cm3_proof.isr_hw_overhead_bytes, 32);
        assert!(cm3_proof.text_bytes > 0);
        assert!(cm3_proof.static_sram_bytes > 0);
        assert!(cm3_proof.boot_stack_bytes > 0);
        assert!(cm3_proof.sched_task_stack_bytes > 0);
        assert!(cm3_proof.worst_isr_stack_bytes >= 32);
        assert_eq!(
            cm3_proof.peak_stack_bytes,
            cm3_proof.boot_stack_bytes.max(cm3_proof.sched_task_stack_bytes)
                + cm3_proof.worst_isr_stack_bytes
        );
        assert_eq!(
            cm3_proof.total_high_water_bytes,
            cm3_proof.static_sram_bytes + cm3_proof.peak_stack_bytes
        );
        assert!(cm3_proof.within_budget);
        assert!(!cm3_proof.has_cycle);
        assert!(cm3_proof.format_table().contains("PASS"));

        // 2. Espressif ESP32-C6 (`examples/esp32c6_demo.zev`, 524,288 B HP-SRAM, 64 B trap frame)
        let esp_mir = compile_example_mir(&[repo_root().join("examples/esp32c6_demo.zev")]);
        let esp_proof =
            analyze_hardware_stack_high_water(&esp_mir, &CodegenOptions::esp32c6())
                .expect("ESP32-C6 stack high-water proof");
        println!("{}", esp_proof.format_table());
        assert_eq!(esp_proof.sram_budget_bytes, 524_288);
        assert_eq!(esp_proof.isr_hw_overhead_bytes, 64);
        assert!(esp_proof.text_bytes > 0);
        assert!(esp_proof.static_sram_bytes > 0);
        assert!(esp_proof.boot_stack_bytes > 0);
        assert!(esp_proof.sched_task_stack_bytes > 0);
        assert!(esp_proof.worst_isr_stack_bytes >= 64);
        assert_eq!(
            esp_proof.total_high_water_bytes,
            esp_proof.static_sram_bytes + esp_proof.peak_stack_bytes
        );
        assert!(esp_proof.within_budget);
        assert!(!esp_proof.has_cycle);
        assert!(esp_proof.format_table().contains("PASS"));

        // 3. Microchip ATmega328P AVR (`examples/arduino_uno_demo.zev`, 2,048 B SRAM, 2 B IRQ vector push)
        let uno_mir = compile_example_mir(&[repo_root().join("examples/arduino_uno_demo.zev")]);
        let uno_proof =
            analyze_hardware_stack_high_water(&uno_mir, &CodegenOptions::avr_uno())
                .expect("ATmega328P AVR stack high-water proof");
        println!("{}", uno_proof.format_table());
        assert_eq!(uno_proof.sram_budget_bytes, 2_048);
        assert_eq!(uno_proof.isr_hw_overhead_bytes, 2);
        assert!(uno_proof.text_bytes > 0);
        assert!(uno_proof.static_sram_bytes > 0);
        assert!(uno_proof.boot_stack_bytes > 0);
        assert!(uno_proof.sched_task_stack_bytes > 0);
        assert!(uno_proof.worst_isr_stack_bytes > 2);
        assert_eq!(
            uno_proof.total_high_water_bytes,
            uno_proof.static_sram_bytes + uno_proof.peak_stack_bytes
        );
        assert!(uno_proof.within_budget);
        assert!(!uno_proof.has_cycle);
        assert!(uno_proof.format_table().contains("PASS"));

        // 4. Host target rejects hardware stack proof cleanly, and AVR >2048 B high-water fails `within_budget`
        let mut host_opts = CodegenOptions::default();
        host_opts.target = TargetArch::HostX86_64Linux;
        assert!(analyze_hardware_stack_high_water(&cm3_mir, &host_opts).is_err());

        let over_avr_src = r#"
byte_ring<1900> huge_ring;
task printer() {
    loop every 50ms {
        println(f"len={huge_ring.len()}");
    }
}
fn main() {
    huge_ring.push_byte(1);
    spawn printer() @normal;
}
"#;
        let over_prog = zev_parser::parse_str(over_avr_src).unwrap();
        let over_mir = zev_mir::lower_program(&over_prog).unwrap();
        let over_proof = analyze_hardware_stack_high_water(&over_mir, &CodegenOptions::avr_uno())
            .expect("over-budget AVR ELF should still link and produce proof");
        assert_eq!(over_proof.sram_budget_bytes, 2_048);
        assert!(over_proof.total_high_water_bytes > 2_048);
        assert!(!over_proof.within_budget);
        assert!(over_proof.format_table().contains("FAIL"));
    }

    #[test]
    fn visibility_strips_dead_watch_and_elides_task_only_irq_masks() {
        let src = r#"
watch<u64> dead_diag = 0;
watch<u32> task_metric = 7;
watch<u16> isr_hits = 0;
byte_ring<16> task_ring;
irq IRQ_TEST = 2;

isr on_test(*watch<u16> hits) {
    hits.publish(hits.read() + 1);
}

task worker() {
    task_metric.publish(42);
    task_ring.push_byte(9);
    loop {
        u32 m = task_metric.read();
        u16 h = isr_hits.read();
        u8 b = task_ring.try_pop_byte() or 0;
        if (m > 0 && h == 999 && b == 99) {
            sys.exit(0);
        }
        sys.delay(5ms);
    }
}

fn main() {
    dead_diag.publish(12345);
    IRQ_TEST.attach(on_test(&isr_hits));
    spawn worker() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse visibility LLVM test");
        zev_sema::analyze_program(&prog).expect("sema visibility LLVM test");
        let mir = zev_mir::lower_program(&prog).expect("lower visibility LLVM test");
        let ram_txt = mir.ram_report.format_table();
        assert!(
            ram_txt.contains("dead watch stripped -> 0 B"),
            "expected dead watch stripped in RAM report:\n{ram_txt}"
        );
        assert!(
            ram_txt.contains("irq-mask elided"),
            "expected irq-mask elided in RAM report:\n{ram_txt}"
        );
        let ll = emit_llvm_ir_with_options(&mir, &CodegenOptions::default()).expect("emit IR");
        assert!(
            !ll.contains("@__syn_prim_dead_diag ="),
            "dead watch global should be stripped from LLVM IR .bss"
        );
        // Task-only watch<u32> helper must elide @__syn_crit_enter, while ISR watch<u16> retains it
        let u32_fn_start = ll.find("define void @__syn_watch_pub_watch_u32").unwrap();
        let u32_fn_end = ll[u32_fn_start..].find("\n}\n").unwrap() + u32_fn_start;
        assert!(
            ll[u32_fn_start..u32_fn_end].contains("; irq-mask elided"),
            "task-only watch<u32> should elide critical section:\n{}",
            &ll[u32_fn_start..u32_fn_end]
        );
        assert!(!ll[u32_fn_start..u32_fn_end].contains("@__syn_crit_enter"));

        let ring_fn_start = ll.find("define i8 @__syn_ring_push_byte_byte_ring_16").unwrap();
        let ring_fn_end = ll[ring_fn_start..].find("\n}\n").unwrap() + ring_fn_start;
        assert!(
            ll[ring_fn_start..ring_fn_end].contains("; irq-mask elided"),
            "task-only byte_ring<16> should elide critical section:\n{}",
            &ll[ring_fn_start..ring_fn_end]
        );

        let u16_fn_start = ll.find("define void @__syn_watch_pub_watch_u16").unwrap();
        let u16_fn_end = ll[u16_fn_start..].find("\n}\n").unwrap() + u16_fn_start;
        assert!(
            ll[u16_fn_start..u16_fn_end].contains("@__syn_crit_enter"),
            "ISR-contended watch<u16> must keep critical section"
        );
    }
}
