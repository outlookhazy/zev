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

pub mod esp32c6;
pub mod mir_codegen;
pub mod nvic;
pub mod runtime_ir;
pub mod toolchain;

use std::path::Path;

pub use esp32c6::ESP32C6_DIRECT_BOOT_MAGIC;
pub use mir_codegen::emit_llvm_ir_with_options;
pub use nvic::{NvicMap, preferred_nvic_irq};
pub use toolchain::{
    CodegenError, CodegenOptions, ElfBuildResult, ElfSectionSizes, ElfSymbolAudit,
    Esp32C6ImageResult, QemuExecOutput, QemuRunOptions, QemuRunResult, TargetArch,
    compile_ll_to_elf, compile_ll_to_obj, flash_esp32c6_bin, inspect_elf_sizes,
    inspect_elf_symbols, link_obj_to_elf, objcopy_elf_to_esp32c6_bin, optimize_ll_with_opt,
    run_elf_in_qemu, synthesize_linker_script, verify_ll_with_opt,
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

    fn load_example_prog(path: &Path) -> zev_ast::Program {
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
        let normalized = raw
            .replace("\n    every ", "\n    loop every ")
            .replace("\n        every ", "\n        loop every ");
        zev_parser::parse_str(&normalized)
            .unwrap_or_else(|e| panic!("failed to parse {}:\n{e}", path.display()))
    }

    #[test]
    fn esp32c6_demo_builds_direct_boot_bin_and_passes_disasm_audit() {
        let demo_path = repo_root().join("examples/esp32c6_demo.zev");
        let prog = load_example_prog(&demo_path);
        zev_sema::analyze_program(&prog)
            .unwrap_or_else(|diags| panic!("zev-sema rejected esp32c6_demo.zev: {diags:#?}"));
        let mir = zev_mir::lower_program(&prog)
            .unwrap_or_else(|e| panic!("failed to lower esp32c6_demo.zev: {e:?}"));

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
            disasm.contains("40800000 <__syn_esp32c6_iram_delay>:")
                || disasm.contains("<__syn_esp32c6_ws2812_grb>:"),
            "expected WS2812B driver linked into HP-SRAM 0x40800000 (.iram1):\n{disasm}"
        );
    }

    #[test]
    fn cortex_m_demo_end_to_end_elf_and_qemu() {
        let t0 = Instant::now();
        let demo_path = repo_root().join("examples/cortex_m_demo.zev");
        let prog = load_example_prog(&demo_path);
        zev_sema::analyze_program(&prog)
            .unwrap_or_else(|diags| panic!("zev-sema rejected cortex_m_demo.zev: {diags:#?}"));
        let mir = zev_mir::lower_program(&prog)
            .unwrap_or_else(|e| panic!("failed to lower cortex_m_demo.zev to MIR: {e:?}"));

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
            total_elapsed.as_secs_f64() < 2.0,
            "end-to-end compile+link+QEMU took {:?} (> 2s budget)",
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
        let p_ctrl = load_example_prog(&root.join("examples/motor_controller.zev"));
        let p_supp = load_example_prog(&root.join("examples/motor_support.zev"));

        let mir = zev_mir::lower_programs(&[p_ctrl, p_supp])
            .expect("lower motor_controller + motor_support to MIR");

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
                .any(|s| s == "__syn_isr_EXTI0")
        );
        assert!(
            res.build
                .audit
                .defined_symbols
                .iter()
                .any(|s| s == "__syn_isr_DMA1_CH5")
        );
    }

    #[test]
    fn host_x86_64_ir_compiles_cleanly_via_llc() {
        let demo_path = repo_root().join("examples/cortex_m_demo.zev");
        let prog = load_example_prog(&demo_path);
        let mir = zev_mir::lower_program(&prog).expect("lower cortex_m_demo.zev");

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
            extern "C" fn syn_uart_kv(*u8 label, i32 val);
            extern "C" fn syn_semihost_exit(i32 code);

            enum IoErr { Timeout, BusFault }

            u32 read_burst(u32 count) error IoErr {
                if (count > 4) {
                    error Timeout, 4;
                }
                return count * 10;
            }

            fn main() {
                u32 ok_val = read_burst(3) catch (IoErr e, u32 partial) {
                    syn_semihost_exit(1);
                };
                u32 saved_partial = 0;
                u32 recov = read_burst(9) catch (IoErr e, u32 partial) {
                    saved_partial = partial;
                    partial + 100;
                };
                u32 fallback = read_burst(9) or 77;
                syn_uart_kv("partial=", saved_partial);
                syn_uart_kv("ok_val=", ok_val);
                syn_uart_kv("recov=", recov);
                syn_uart_kv("fallback=", fallback);
                syn_semihost_exit(0);
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
            extern "C" fn syn_uart_kv(*u8 label, i32 val);
            extern "C" u32 syn_current_prio();
            extern "C" fn syn_nvic_pend_irq(u32 irq);
            extern "C" fn syn_semihost_exit(i32 code);

            signal fault_sig;
            signal idle_sig;
            inbox<u32, 4> telemetry_ch;

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
                spawn worker_producer() @normal;
                set_priority(@low);

                u32 a = telemetry_ch.recv();
                u32 b = telemetry_ch.recv();
                u32 c = telemetry_ch.recv();
                syn_uart_kv("chan_sum=", a + b + c);

                syn_nvic_pend_irq(1);
                fault_sig.wait();
                syn_uart_kv("sig_boosted_prio=", syn_current_prio());
                syn_semihost_exit(0);
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
        let prog = load_example_prog(&sim_path);
        zev_sema::analyze_program(&prog).expect("sema sim_demo.zev");
        let mir = zev_mir::lower_program(&prog).expect("mir sim_demo.zev");

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
        assert!(res.exec.stdout.contains("[2] 2400,1800"), "out={}", res.exec.stdout);
        assert!(res.exec.stdout.contains("[4] 0,0"), "out={}", res.exec.stdout);
    }

    #[test]
    fn opt_levels_o0_o1_o2_o3_os_oz_build_zero_heap_elfs_and_shrink_flash_in_qemu() {
        let root = repo_root();
        let cm3_prog = load_example_prog(&root.join("examples/cortex_m_demo.zev"));
        zev_sema::analyze_program(&cm3_prog).expect("sema cm3");
        let cm3_mir = zev_mir::lower_program(&cm3_prog).expect("mir cm3");

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
        let esp_prog = load_example_prog(&root.join("examples/esp32c6_demo.zev"));
        zev_sema::analyze_program(&esp_prog).expect("sema esp32c6");
        let esp_mir = zev_mir::lower_program(&esp_prog).expect("mir esp32c6");

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
                u32 p = compute_period_ms(2);
                loop every p {
                    u32 next = inbody_runs.read() + 1;
                    inbody_runs.publish(next);
                    if (next == 1) {
                        /* Stall in-body past 4 full 10ms periods (t=0 -> t=45ms)! */
                        noyield {
                            while (sys.now_ms() < 45) {}
                        }
                    }
                }
            }

            task sleep_periodic() {
                loop every 10ms {
                    sleep_runs.publish(sleep_runs.read() + 1);
                }
            }

            task critical_staller() {
                delay(51ms);
                u32 ib_at_51 = inbody_runs.read();
                u32 sb_before = sleep_runs.read();

                /* Both 10ms tasks are parked sleeping when @critical stalls CPU past 4 periods (51ms -> 92ms)! */
                noyield {
                    while (sys.now_ms() < 92) {}
                }

                /* Yield for 1ms (t=92ms -> t=93ms, well before the next 100ms grid boundary!):
                 * Each 10ms task MUST run EXACTLY 1 resynced pass (zero catch-up bursts!). */
                delay(1ms);
                u32 ib_after = inbody_runs.read();
                u32 sb_after = sleep_runs.read();

                sys.log_kv("ib_at_52=", ib_at_51 as i32);
                sys.log_kv("ib_after=", ib_after as i32);
                sys.log_kv("sb_delta=", (sb_after - sb_before) as i32);

                /* At t=51ms: iter 1 (t=0..45ms) + iter 2 resynced (t=45ms) + iter 3 on grid (t=50ms) == 3 (NOT 6!) */
                assert(ib_at_51 == 3);
                /* After 41ms sleep stall (51ms -> 92ms), both tasks ran 1 resynced pass, NOT 4! */
                assert(ib_after == ib_at_51 + 1);
                assert(sb_after == sb_before + 1);

                sys.exit(0);
            }

            fn main() {
                sys.use_systick(12_000_000);
                spawn critical_staller() @critical;
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
        assert!(out.contains("ib_at_52=3\n"), "out={out}");
        assert!(out.contains("ib_after=4\n"), "out={out}");
        assert!(out.contains("sb_delta=1\n"), "out={out}");
    }

    #[test]
    fn wave1_restart_preserves_task_params_and_resets_locals_in_qemu() {
        let src = r#"
            extern "C" u32 syn_current_prio();
            extern "C" fn syn_nvic_pend_irq(u32 irq);

            enum ActorCmd { AddWork, TriggerRestart }

            inbox<ActorCmd, 4> actor_q;
            watch<u32> actor_state = 0;
            watch<u32> actor_defers = 0;
            watch<u32> last_turn_prio = 99;
            signal restart_ack;

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
                    last_turn_prio.publish(syn_current_prio());
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
                delay(2ms);
                u32 init_st = actor_state.read();
                actor_q.send(AddWork);
                delay(2ms);
                u32 work1_st = actor_state.read();
                u32 work1_prio = last_turn_prio.read();

                /* Fire NVIC EXTI2 -> posts TriggerRestart with boost: @critical */
                syn_nvic_pend_irq(2);
                restart_ack.wait();
                delay(2ms);

                u32 boosted_prio = last_turn_prio.read();
                u32 defers_after = actor_defers.read();
                u32 restarted_st = actor_state.read();

                /* Send another unboosted AddWork to prove @low priority (3) was restored on restart! */
                actor_q.send(AddWork);
                delay(2ms);
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
}
