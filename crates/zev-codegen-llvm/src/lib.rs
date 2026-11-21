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

    #[test]
    fn esp32c6_demo_builds_direct_boot_bin_and_passes_disasm_audit() {
        let demo_path = repo_root().join("examples/esp32c6_demo.zev");
        let prog = zev_parser::parse_file(&demo_path)
            .unwrap_or_else(|e| panic!("failed to parse esp32c6_demo.zev:\n{e}"));
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
        let prog = zev_parser::parse_file(&demo_path)
            .unwrap_or_else(|e| panic!("failed to parse cortex_m_demo.zev:\n{e}"));
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
        let p_ctrl = zev_parser::parse_file(root.join("examples/motor_controller.zev"))
            .expect("parse motor_controller.zev");
        let p_supp = zev_parser::parse_file(root.join("examples/motor_support.zev"))
            .expect("parse motor_support.zev");

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
        let prog = zev_parser::parse_file(&demo_path).expect("parse cortex_m_demo.zev");
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
        // 1. Infinite `every 10ms` firmware WITHOUT explicit `sys.use_systick`
        let every_src = r#"
            task periodic() {
                u32 beats = 0;
                every 10ms {
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
            chan<u32, 4> telemetry_ch;

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
        let prog = zev_parser::parse_file(&sim_path).expect("parse sim_demo.zev");
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
        let cm3_prog =
            zev_parser::parse_file(root.join("examples/cortex_m_demo.zev")).expect("parse cm3");
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
        let esp_prog =
            zev_parser::parse_file(root.join("examples/esp32c6_demo.zev")).expect("parse esp32c6");
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
}
