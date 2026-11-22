//! Bare-Metal ESP32-C6 (`riscv32imac`) Direct-Boot Startup, Interrupt Routing (`INTMTX` / `PLIC_MX`),
//! and Architectural `mstatus` Critical Sections in Pure LLVM 23 IR.
//!
//! Hardware Direct-Boot & interrupt-controller map (zero peripheral drivers live here — watchdogs,
//! APM, `PCR`, `SYSTIMER`, and `USB_DEVICE` console I/O live 100% in `.zev`):
//! - Direct-Boot Flash MMU Cache: `0x4200_0000` (magic `0xAEDB041D` x2 at `0x0`, `@_start` at `0x4200_0008`)
//! - HP-SRAM: `0x4080_0000 .. 0x4088_0000` (initial `sp = 0x4087_0000`, `.iram1` + `.data` + `.bss`)
//! - Interrupt Routing: `INTMTX` (`0x6001_0000`), `PLIC_MX` (`0x2000_1000`), vectored `mtvec` (`@__syn_rv_vector_table`)

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

/// Expected first 8 bytes at offset `0x0` of an ESP32-C6 Direct-Boot flash image (`0xAEDB041D` x2 LE).
pub const ESP32C6_DIRECT_BOOT_MAGIC: [u8; 8] = [0x1d, 0x04, 0xdb, 0xae, 0x1d, 0x04, 0xdb, 0xae];

/// External CPU interrupt lines on ESP32-C6 (`INTMTX` / `INTPRI`), excluding CLINT-reserved
/// core lines `0, 3, 4, 7`.
pub const ESP32C6_EXTERNAL_CPU_INTS: [u8; 28] = [
    1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31,
];

/// Total peripheral interrupt matrix (`INTMTX`) sources on ESP32-C6 (`0..=76`).
pub const ESP32C6_INTMTX_SOURCE_COUNT: u32 = 77;

/// Deterministic ESP32-C6 peripheral interrupt matrix (`INTMTX` `src_id`) -> CPU line (`cpu_int`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Esp32C6IrqRoute {
    pub src_id: u32,
    pub cpu_int: u8,
}

/// Assign deterministic CPU interrupt lines (`1..=31`, skipping CLINT-reserved `0, 3, 4, 7`) for
/// each attached `MirIrqVector` (`src_id = irq_num`).
pub fn assign_esp32c6_irq_routes(
    vectors: &[zev_mir::MirIrqVector],
) -> Result<Vec<Esp32C6IrqRoute>, crate::CodegenError> {
    let mut unique_srcs = BTreeSet::new();
    for v in vectors {
        if v.irq_num >= ESP32C6_INTMTX_SOURCE_COUNT {
            return Err(crate::CodegenError::new(
                "llvm-ir",
                format!(
                    "ESP32-C6 IRQ source ID {} exceeds INTMTX range (0..={})",
                    v.irq_num,
                    ESP32C6_INTMTX_SOURCE_COUNT - 1
                ),
            ));
        }
        unique_srcs.insert(v.irq_num);
    }
    if unique_srcs.len() > ESP32C6_EXTERNAL_CPU_INTS.len() {
        return Err(crate::CodegenError::new(
            "llvm-ir",
            format!(
                "ESP32-C6 supports at most {} distinct external CPU interrupt lines (excluding CLINT lines 0, 3, 4, 7), but program attached {} IRQ sources",
                ESP32C6_EXTERNAL_CPU_INTS.len(),
                unique_srcs.len()
            ),
        ));
    }
    Ok(unique_srcs
        .into_iter()
        .enumerate()
        .map(|(idx, src_id)| Esp32C6IrqRoute {
            src_id,
            cpu_int: ESP32C6_EXTERNAL_CPU_INTS[idx],
        })
        .collect())
}

/// Emit ESP32-C6 `.entry` (`@_start` at `0x4200_0008`), `@__syn_esp32c6_reset`,
/// `.data`/`.iram1` copy + `fence.i`, `.bss` zeroing, and `INTMTX`/`PLIC_MX`/`mie`/vectored `mtvec`.
pub fn emit_esp32c6_startup_ir(
    out: &mut String,
    initial_sp: u32,
    sz: &str,
    routes: &[Esp32C6IrqRoute],
) {
    let irq_init_call = if routes.is_empty() {
        ""
    } else {
        "  call void @__syn_esp32c6_irq_init()\n"
    };
    let _ = writeln!(
        out,
        r#"; ── ESP32-C6 (rv32imac) Direct-Boot Entry Point (@ 0x42000008) ──
module asm ".section .entry, \22ax\22"
module asm ".global _start"
module asm ".type _start, @function"
module asm "_start:"
module asm "  .option push"
module asm "  .option norelax"
module asm "  csrci mstatus, 8"
module asm "  li sp, 0x{initial_sp:08X}"
module asm "  call __syn_esp32c6_reset"
module asm "1: j 1b"
module asm "  .option pop"

@__sdata = external global i8
@__edata = external global i8
@__sidata = external global i8
@__sbss = external global i8
@__ebss = external global i8
@llvm.used = appending global [1 x ptr] [ptr @__syn_esp32c6_reset], section "llvm.metadata"

define void @__syn_esp32c6_reset() #98 {{
entry:
  ; 1. Copy .iram1 + .data from Flash LMA (__sidata) to HP-SRAM VMA [__sdata .. __edata)
  ;    in integer address domain (avoids provenance UB & memcpy idiom before RAM code exists!)
  %sdata_a = ptrtoint ptr @__sdata to i32
  %edata_a = ptrtoint ptr @__edata to i32
  %sidata_a = ptrtoint ptr @__sidata to i32
  br label %data_loop
data_loop:
  %da = phi i32 [ %sdata_a, %entry ], [ %da_next, %data_body ]
  %sa = phi i32 [ %sidata_a, %entry ], [ %sa_next, %data_body ]
  %ddone = icmp uge i32 %da, %edata_a
  br i1 %ddone, label %bss_init, label %data_body
data_body:
  %sp = inttoptr i32 %sa to ptr
  %dp = inttoptr i32 %da to ptr
  %b = load volatile i8, ptr %sp, align 1
  store volatile i8 %b, ptr %dp, align 1
  %da_next = add i32 %da, 1
  %sa_next = add i32 %sa, 1
  br label %data_loop

bss_init:
  ; Flush instruction pipeline after copying .iram1 code to HP-SRAM
  call void asm sideeffect "fence.i", "~{{memory}}"()
  %sbss_a = ptrtoint ptr @__sbss to i32
  %ebss_a = ptrtoint ptr @__ebss to i32
  br label %bss_loop
bss_loop:
  %ba = phi i32 [ %sbss_a, %bss_init ], [ %ba_next, %bss_body ]
  %bdone = icmp uge i32 %ba, %ebss_a
  br i1 %bdone, label %boot, label %bss_body
bss_body:
  %bp = inttoptr i32 %ba to ptr
  store volatile i8 0, ptr %bp, align 1
  %ba_next = add i32 %ba, 1
  br label %bss_loop

boot:
{irq_init_call}  call void @__syn_boot()
  call void @__syn_sched_loop()
  unreachable
}}
"#
    );
    emit_esp32c6_irq_vectors_ir(out, routes, sz);
}

/// Emit 256-byte-aligned `.iram1` vectored trap table `@__syn_rv_vector_table`,
/// naked RV32 GPR-preserving trampolines `@__syn_rv_irq_tramp_<cpu_int>`, and
/// `@__syn_esp32c6_irq_init` (`INTMTX`, `PLIC_MX`, `mie`, and `mtvec = (__syn_rv_vector_table | 1)`).
pub fn emit_esp32c6_irq_vectors_ir(
    out: &mut String,
    routes: &[Esp32C6IrqRoute],
    sz: &str,
) {
    if routes.is_empty() {
        return;
    }

    let mut by_cpu_int: BTreeMap<u8, Vec<u32>> = BTreeMap::new();
    let mut enable_mask: u32 = 0;
    for r in routes {
        by_cpu_int.entry(r.cpu_int).or_default().push(r.src_id);
        enable_mask |= 1u32 << u32::from(r.cpu_int);
    }

    let _ = writeln!(
        out,
        "; ── ESP32-C6 Vectored Trap Table (@__syn_rv_vector_table: section \".iram1\", align 256) ──"
    );
    let _ = writeln!(out, "module asm \".section .iram1, \\22ax\\22, @progbits\"");
    let _ = writeln!(out, "module asm \".balign 256\"");
    let _ = writeln!(out, "module asm \".global __syn_rv_vector_table\"");
    let _ = writeln!(out, "module asm \".type __syn_rv_vector_table, @function\"");
    let _ = writeln!(out, "module asm \"__syn_rv_vector_table:\"");
    let _ = writeln!(out, "module asm \"  .option push\"");
    let _ = writeln!(out, "module asm \"  .option norvc\"");
    let _ = writeln!(out, "module asm \"  j __syn_rv_exc_tramp\"");
    for cpu_int in 1..=31u8 {
        if by_cpu_int.contains_key(&cpu_int) {
            let _ = writeln!(out, "module asm \"  j __syn_rv_irq_tramp_{cpu_int}\"");
        } else {
            let _ = writeln!(out, "module asm \"  j __syn_rv_unhandled_irq\"");
        }
    }
    let _ = writeln!(out, "module asm \"  .option pop\"");
    let _ = writeln!(out, "declare void @__syn_rv_vector_table()");
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        r#"define void @__syn_rv_exc_tramp() naked nounwind section ".iram1" {{
  call void asm sideeffect "1: j 1b", ""()
  unreachable
}}

define void @__syn_rv_unhandled_irq() naked nounwind section ".iram1" {{
  call void asm sideeffect "mret", ""()
  unreachable
}}
"#
    );

    for (cpu_int, src_ids) in &by_cpu_int {
        let clear_mask = 1u32 << u32::from(*cpu_int);
        let mut calls_asm = String::new();
        for src_id in src_ids {
            let _ = writeln!(calls_asm, "    call __syn_vec_irq_{src_id}");
        }
        let _ = writeln!(
            out,
            r#"define void @__syn_rv_irq_tramp_{cpu_int}() naked nounwind section ".iram1" {{
  call void asm sideeffect "
    addi sp, sp, -64
    sw ra, 0(sp)
    sw t0, 4(sp)
    sw t1, 8(sp)
    sw t2, 12(sp)
    sw a0, 16(sp)
    sw a1, 20(sp)
    sw a2, 24(sp)
    sw a3, 28(sp)
    sw a4, 32(sp)
    sw a5, 36(sp)
    sw a6, 40(sp)
    sw a7, 44(sp)
    sw t3, 48(sp)
    sw t4, 52(sp)
    sw t5, 56(sp)
    sw t6, 60(sp)
{calls_asm}    li t0, 0x20001008
    li t1, 0x{clear_mask:X}
    lw t2, 0(t0)
    or t3, t2, t1
    sw t3, 0(t0)
    sw t2, 0(t0)
    lw ra, 0(sp)
    lw t0, 4(sp)
    lw t1, 8(sp)
    lw t2, 12(sp)
    lw a0, 16(sp)
    lw a1, 20(sp)
    lw a2, 24(sp)
    lw a3, 28(sp)
    lw a4, 32(sp)
    lw a5, 36(sp)
    lw a6, 40(sp)
    lw a7, 44(sp)
    lw t3, 48(sp)
    lw t4, 52(sp)
    lw t5, 56(sp)
    lw t6, 60(sp)
    addi sp, sp, 64
    mret
  ", ""()
  unreachable
}}
"#
        );
    }

    let mut cfg_ir = String::new();
    for r in routes {
        let intmtx_addr = 0x6001_0000u32.wrapping_add(4 * r.src_id);
        let intmtx_i32 = intmtx_addr as i32;
        let _ = writeln!(
            cfg_ir,
            "  ; INTMTX src_id={} -> cpu_int={} (0x{intmtx_addr:08X})",
            r.src_id, r.cpu_int
        );
        let _ = writeln!(
            cfg_ir,
            "  store volatile i32 {}, ptr inttoptr (i32 {intmtx_i32} to ptr), align 4",
            r.cpu_int
        );
    }
    for &cpu_int in by_cpu_int.keys() {
        let pri_addr = 0x2000_1010u32 + 4 * u32::from(cpu_int);
        let pri_i32 = pri_addr as i32;
        let _ = writeln!(
            cfg_ir,
            "  ; PLIC_MXINT{cpu_int}_PRI_REG (0x{pri_addr:08X}) = 1"
        );
        let _ = writeln!(
            cfg_ir,
            "  store volatile i32 1, ptr inttoptr (i32 {pri_i32} to ptr), align 4"
        );
    }
    let en_addr = 0x2000_1000u32 as i32;
    let type_addr = 0x2000_1004u32 as i32;
    let thresh_addr = 0x2000_1090u32 as i32;
    let enable_mask_i32 = enable_mask as i32;
    let clear_type_mask_i32 = (!enable_mask) as i32;
    let mie_mask: u32 = enable_mask | (1u32 << 11);
    let mie_mask_i32 = mie_mask as i32;
    let _ = writeln!(
        out,
        r#"define void @__syn_esp32c6_irq_init() nounwind{sz} {{
entry:
{cfg_ir}  ; PLIC_MXINT_TYPE_REG (0x20001004) &= ~0x{enable_mask:08X} (level-triggered)
  %type_old = load volatile i32, ptr inttoptr (i32 {type_addr} to ptr), align 4
  %type_new = and i32 %type_old, {clear_type_mask_i32}
  store volatile i32 %type_new, ptr inttoptr (i32 {type_addr} to ptr), align 4
  ; PLIC_MXINT_THRESH_REG (0x20001090) = 1
  store volatile i32 1, ptr inttoptr (i32 {thresh_addr} to ptr), align 4
  ; Set mtvec = (__syn_rv_vector_table | 1) for vectored HP-SRAM trap dispatch before unmasking
  %vt_a = ptrtoint ptr @__syn_rv_vector_table to i32
  %mtvec_v = or i32 %vt_a, 1
  call void asm sideeffect "csrw mtvec, $0", "r,~{{memory}}"(i32 %mtvec_v)
  ; PLIC_MXINT_ENABLE_REG (0x20001000) = 0x{enable_mask:08X}
  store volatile i32 {enable_mask_i32}, ptr inttoptr (i32 {en_addr} to ptr), align 4
  ; Unmask attached external lines + MEIE (bit 11) in RISC-V mie CSR (0x{mie_mask:08X})
  call void asm sideeffect "csrw mie, $0", "r,~{{memory}}"(i32 {mie_mask_i32})
  ret void
}}
"#
    );
}

/// Emit ESP32-C6 (`rv32imac`) architectural runtime primitives (`mstatus` critical sections,
/// `sys.set_putc` / `.zev` `console_putc(u8)` delegation, and `@__syn_idle_sleep` calling `@__syn_now_ms()`).
pub fn emit_esp32c6_hal_ir(
    out: &mut String,
    sz: &str,
    uses_set_putc: bool,
    console_putc_sym: Option<&str>,
) {
    let putc_body =
        crate::runtime_ir::format_uart_putc_body(uses_set_putc, 4, console_putc_sym, "  ret void");
    let _ = writeln!(
        out,
        r#"; ── ESP32-C6 (rv32imac) Architectural CPU Primitives: mstatus & Idle Poll ──
define i32 @__syn_crit_enter() nounwind alwaysinline {{
entry:
  %pm = call i32 asm sideeffect "csrrci $0, mstatus, 8", "=r,~{{memory}}"()
  ret i32 %pm
}}

define void @__syn_crit_exit(i32 %pm) nounwind alwaysinline {{
entry:
  %mie = and i32 %pm, 8
  call void asm sideeffect "csrrs x0, mstatus, $0", "r,~{{memory}}"(i32 %mie)
  ret void
}}

define void @__syn_uart_putc(i8 %c) nounwind noinline{sz} {{
entry:
{putc_body}
}}

define void @__syn_platform_exit(i32 %code) noreturn nounwind{sz} {{
entry:
  br label %halt
halt:
  call void asm sideeffect "nop", "~{{memory}}"()
  br label %halt
}}

define void @__syn_use_systick(i32 %hz) nounwind{sz} {{
entry:
  store i8 1, ptr @__syn_systick_enabled, align 1
  ret void
}}

define void @__syn_nvic_pend_irq(i32 %irq) nounwind{sz} {{
entry:
  call void @__syn_esp32c6_dispatch_irq(i32 %irq)
  ret void
}}

define void @__syn_idle_sleep() nounwind{sz} {{
entry:
  %locks = load volatile i32, ptr @__syn_sleep_lock_mask, align 4
  %dma_locks = load volatile i32, ptr @__syn_dma_locks, align 4
  %any_locks = or i32 %locks, %dma_locks
  %locked = icmp ne i32 %any_locks, 0
  %depth = select i1 %locked, i32 1, i32 2
  store volatile i32 %depth, ptr @__syn_last_sleep_depth, align 4
  %wc = load volatile i32, ptr @__syn_wfi_count, align 4
  %wc1 = add i32 %wc, 1
  store volatile i32 %wc1, ptr @__syn_wfi_count, align 4
  %ign = call i32 @__syn_now_ms()
  call void asm sideeffect "nop\0Anop\0Anop\0Anop", "~{{memory}}"()
  ret void
}}
"#
    );
}
