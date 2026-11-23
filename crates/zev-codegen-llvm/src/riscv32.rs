//! Pure-ISA 32-Bit RISC-V (`riscv32`) Startup, Vectored `mtvec` Trap Table,
//! GPR Context-Saving Trampolines, and `mstatus` Critical Sections in LLVM 23 IR.
//!
//! Zero chip/board names or vendor interrupt controller (`INTMTX` / `PLIC` / `CLIC` / `PFIC`)
//! MMIO addresses live here: interrupt routing and acknowledgement delegate explicitly through
//! `sys.set_irq_router(route_fn, ack_fn)` (`@__syn_on_rv_irq_route` / `@__syn_on_rv_irq_ack`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::target_spec::{IrqModel, TargetSpec};
use crate::toolchain::CodegenError;

/// Deterministic mapping from an attached peripheral IRQ source ID to a RISC-V CPU interrupt slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RvIrqRoute {
    pub src_id: u32,
    pub cpu_slot: u8,
}

/// Assign deterministic CPU interrupt slots (`0..vector_slots`, skipping `reserved_slots`)
/// for each attached `MirIrqVector` (`src_id = irq_num`).
pub fn assign_rv_irq_routes(
    vectors: &[zev_mir::MirIrqVector],
    max_irq_source: u32,
    vector_slots: u32,
    reserved_slots: &[u32],
) -> Result<Vec<RvIrqRoute>, CodegenError> {
    let mut unique_srcs = BTreeSet::new();
    for v in vectors {
        if v.irq_num > max_irq_source {
            return Err(CodegenError::new(
                "llvm-ir",
                format!(
                    "RISC-V IRQ source ID {} exceeds chip `[irq].max_irq_source` (0..={max_irq_source})",
                    v.irq_num
                ),
            ));
        }
        unique_srcs.insert(v.irq_num);
    }

    let available_slots: Vec<u8> = (0..vector_slots)
        .filter(|s| !reserved_slots.contains(s))
        .filter_map(|s| u8::try_from(s).ok())
        .collect();

    if unique_srcs.len() > available_slots.len() {
        return Err(CodegenError::new(
            "llvm-ir",
            format!(
                "RISC-V target provides {} unreserved vector slot(s) (reserved={:?}), but program attached {} IRQ sources",
                available_slots.len(),
                reserved_slots,
                unique_srcs.len()
            ),
        ));
    }

    Ok(unique_srcs
        .into_iter()
        .enumerate()
        .map(|(idx, src_id)| RvIrqRoute {
            src_id,
            cpu_slot: available_slots[idx],
        })
        .collect())
}

/// Convenience helper deriving `assign_rv_irq_routes` directly from `TargetSpec`.
pub fn assign_rv_irq_routes_for_spec(
    vectors: &[zev_mir::MirIrqVector],
    spec: &TargetSpec,
) -> Result<Vec<RvIrqRoute>, CodegenError> {
    match &spec.irq {
        IrqModel::RiscvVectored {
            vector_slots,
            max_irq_source,
            reserved_slots,
        } => assign_rv_irq_routes(vectors, *max_irq_source, *vector_slots, reserved_slots),
        _ => Ok(Vec::new()),
    }
}

/// Emit pure-ISA RV32 `.entry` (`@_start`), `@__syn_rv_reset` (copies `.data`/`.iram1` + `fence.i`,
/// zeroes `.bss`, runs `@__syn_boot()` FIRST so `sys.set_irq_router` registers routes, then runs
/// `@__syn_rv_irq_init()` SECOND, then `@__syn_sched_loop()`), plus vectored `mtvec` trampolines.
pub fn emit_riscv32_startup_ir(
    out: &mut String,
    spec: &TargetSpec,
    sz: &str,
    routes: &[RvIrqRoute],
) {
    let initial_sp = spec.initial_sp();
    let vector_slots = match spec.irq {
        IrqModel::RiscvVectored { vector_slots, .. } => vector_slots,
        _ => 32,
    };
    let irq_init_call = if routes.is_empty() {
        ""
    } else {
        "  call void @__syn_rv_irq_init()\n"
    };

    let _ = writeln!(
        out,
        r#"; ── RISC-V ({}) Pure-ISA Entry Point (@ 0x{:08X}) ──
module asm ".section .entry, \22ax\22"
module asm ".global _start"
module asm ".type _start, @function"
module asm "_start:"
module asm "  .option push"
module asm "  .option norelax"
module asm "  li sp, 0x{initial_sp:08X}"
module asm "  tail __syn_rv_reset"
module asm "  .option pop"

@__sdata = external global i8
@__edata = external global i8
@__sidata = external global i8
@__sbss = external global i8
@__ebss = external global i8
"#,
        spec.cpu,
        spec.flash_base
    );

    // Build single module-level @llvm.used list keeping inline-asm-referenced hooks & trampolines alive.
    let mut used_syms: Vec<String> = vec![
        "ptr @__syn_rv_reset".to_string(),
        "ptr @__syn_on_fault".to_string(),
    ];
    if !routes.is_empty() {
        used_syms.push("ptr @__syn_on_rv_irq_ack".to_string());
        used_syms.push("ptr @__syn_rv_vector_table".to_string());
        used_syms.push("ptr @__syn_rv_exc_tramp".to_string());
        used_syms.push("ptr @__syn_rv_unhandled_irq".to_string());
        let mut seen_slots = BTreeSet::new();
        for r in routes {
            if seen_slots.insert(r.cpu_slot) {
                used_syms.push(format!("ptr @__syn_rv_irq_tramp_{}", r.cpu_slot));
            }
        }
        for r in routes {
            used_syms.push(format!("ptr @__syn_vec_irq_{}", r.src_id));
        }
    }
    let _ = writeln!(
        out,
        "@llvm.used = appending global [{} x ptr] [{}], section \"llvm.metadata\"\n",
        used_syms.len(),
        used_syms.join(", ")
    );

    let data_copy_ir = if spec.code_in_ram {
        "  br label %bss_init\n".to_string()
    } else {
        let fence_ir = if spec.iram_in_data {
            "  call void asm sideeffect \"fence.i\", \"~{memory}\"()\n"
        } else {
            ""
        };
        format!(
            r#"  %sdata_a = ptrtoint ptr @__sdata to i32
  %edata_a = ptrtoint ptr @__edata to i32
  %sidata_a = ptrtoint ptr @__sidata to i32
  br label %data_loop
data_loop:
  %da = phi i32 [ %sdata_a, %entry ], [ %da_next, %data_body ]
  %sa = phi i32 [ %sidata_a, %entry ], [ %sa_next, %data_body ]
  %ddone = icmp uge i32 %da, %edata_a
  br i1 %ddone, label %data_done, label %data_body
data_body:
  %sp_p = inttoptr i32 %sa to ptr
  %dp_p = inttoptr i32 %da to ptr
  %b = load volatile i8, ptr %sp_p, align 1
  store volatile i8 %b, ptr %dp_p, align 1
  %da_next = add i32 %da, 1
  %sa_next = add i32 %sa, 1
  br label %data_loop
data_done:
{fence_ir}  br label %bss_init
"#
        )
    };

    let _ = writeln!(
        out,
        r#"define void @__syn_rv_reset() #98 {{
entry:
  ; Keep mstatus.MIE disabled until __syn_sched_loop()
  call void asm sideeffect "csrrci x0, mstatus, 8", "~{{memory}}"()
{data_copy_ir}bss_init:
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
  ; Run sequential main() boot FIRST so BSP registers sys.set_irq_router(route, ack)
  call void @__syn_boot()
{irq_init_call}  call void @__syn_sched_loop()
  unreachable
}}
"#
    );

    emit_riscv32_irq_vectors_ir(out, spec, vector_slots, sz, routes);
}

fn emit_riscv32_irq_vectors_ir(
    out: &mut String,
    spec: &TargetSpec,
    vector_slots: u32,
    sz: &str,
    routes: &[RvIrqRoute],
) {
    if routes.is_empty() {
        return;
    }

    let mut by_slot: BTreeMap<u8, Vec<u32>> = BTreeMap::new();
    let mut enable_mask: u32 = 0;
    for r in routes {
        by_slot.entry(r.cpu_slot).or_default().push(r.src_id);
        enable_mask |= 1u32 << u32::from(r.cpu_slot);
    }

    let tramp_sec = if spec.iram_in_data { ".iram1" } else { ".text" };
    let tramp_sec_attr = if spec.iram_in_data {
        " section \".iram1\""
    } else {
        ""
    };

    let _ = writeln!(
        out,
        "; ── RISC-V Vectored Trap Table (@__syn_rv_vector_table: section \"{tramp_sec}\", align 256) ──"
    );
    let _ = writeln!(out, "module asm \".section {tramp_sec}, \\22ax\\22, @progbits\"");
    let _ = writeln!(out, "module asm \".balign 256\"");
    let _ = writeln!(out, "module asm \".global __syn_rv_vector_table\"");
    let _ = writeln!(out, "module asm \".type __syn_rv_vector_table, @function\"");
    let _ = writeln!(out, "module asm \"__syn_rv_vector_table:\"");
    let _ = writeln!(out, "module asm \"  .option push\"");
    let _ = writeln!(out, "module asm \"  .option norvc\"");
    let _ = writeln!(out, "module asm \"  j __syn_rv_exc_tramp\"");
    for slot in 1..vector_slots {
        let slot_u8 = slot as u8;
        if by_slot.contains_key(&slot_u8) {
            let _ = writeln!(out, "module asm \"  j __syn_rv_irq_tramp_{slot_u8}\"");
        } else {
            let _ = writeln!(out, "module asm \"  j __syn_rv_unhandled_irq\"");
        }
    }
    let _ = writeln!(out, "module asm \"  .option pop\"");
    let _ = writeln!(out, "declare void @__syn_rv_vector_table()");
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        r#"define void @__syn_rv_exc_tramp() naked nounwind{tramp_sec_attr} {{
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
    call __syn_on_fault
1:  j 1b
  ", ""()
  unreachable
}}

define void @__syn_rv_unhandled_irq() naked nounwind{tramp_sec_attr} {{
  call void asm sideeffect "mret", ""()
  unreachable
}}
"#
    );

    for (cpu_slot, src_ids) in &by_slot {
        let mut calls_asm = String::new();
        for src_id in src_ids {
            let _ = writeln!(calls_asm, "    call __syn_vec_irq_{src_id}");
        }
        let _ = writeln!(
            out,
            r#"define void @__syn_rv_irq_tramp_{cpu_slot}() naked nounwind{tramp_sec_attr} {{
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
{calls_asm}    li a0, {cpu_slot}
    call __syn_on_rv_irq_ack
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

    let mut route_calls = String::new();
    for r in routes {
        let _ = writeln!(
            route_calls,
            "  call void @__syn_on_rv_irq_route(i32 {}, i32 {})",
            r.src_id, r.cpu_slot
        );
    }
    let mie_mask: u32 = enable_mask | (1u32 << 11);
    let mie_mask_i32 = mie_mask as i32;
    let _ = writeln!(
        out,
        r#"define void @__syn_rv_irq_init() nounwind{sz} {{
entry:
{route_calls}  %vt_a = ptrtoint ptr @__syn_rv_vector_table to i32
  %mtvec_v = or i32 %vt_a, 1
  call void asm sideeffect "csrw mtvec, $0", "r,~{{memory}}"(i32 %mtvec_v)
  call void asm sideeffect "csrw mie, $0", "r,~{{memory}}"(i32 {mie_mask_i32})
  ret void
}}
"#
    );
}

/// Emit pure-ISA RV32 architectural CPU primitives (`mstatus`, `@__syn_uart_putc`,
/// `@__syn_platform_exit`, `@__syn_use_systick`, `@__syn_nvic_pend_irq`, `@__syn_idle_sleep`).
pub fn emit_riscv32_hal_ir(
    out: &mut String,
    spec: &TargetSpec,
    sz: &str,
    enable_irqs: bool,
    can_wfi_on_deadline: bool,
) {
    let idle_sleep_ir = if !enable_irqs {
        r#"define void @__syn_idle_sleep() nounwind{sz} {{
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
  call void @__syn_on_idle(i32 0, i32 %depth)
  call void asm sideeffect "nop\0Anop\0Anop\0Anop", "~{{memory}}"()
  ret void
}}"#
        .to_string()
    } else if can_wfi_on_deadline {
        r#"define void @__syn_idle_sleep() nounwind{sz} {{
entry:
  %pm = call i32 @__syn_crit_enter()
  %now = call i32 @__syn_now_ms()
  call void @__syn_wake_timers(i32 %now)
  %any_ready = call i1 @__syn_any_ready_task()
  br i1 %any_ready, label %restore, label %do_sleep
do_sleep:
  %locks = load volatile i32, ptr @__syn_sleep_lock_mask, align 4
  %dma_locks = load volatile i32, ptr @__syn_dma_locks, align 4
  %any_locks = or i32 %locks, %dma_locks
  %locked = icmp ne i32 %any_locks, 0
  %depth = select i1 %locked, i32 1, i32 2
  store volatile i32 %depth, ptr @__syn_last_sleep_depth, align 4
  %wc = load volatile i32, ptr @__syn_wfi_count, align 4
  %wc1 = add i32 %wc, 1
  store volatile i32 %wc1, ptr @__syn_wfi_count, align 4
  call void @__syn_on_idle(i32 0, i32 %depth)
  call void asm sideeffect "wfi\0Acsrsi mstatus, 8\0Acsrci mstatus, 8", "~{{memory}}"()
  br label %restore
restore:
  call void @__syn_crit_exit(i32 %pm)
  ret void
}}"#
        .to_string()
    } else {
        r#"define void @__syn_idle_sleep() nounwind{sz} {{
entry:
  %pm = call i32 @__syn_crit_enter()
  %now = call i32 @__syn_now_ms()
  call void @__syn_wake_timers(i32 %now)
  %any_ready = call i1 @__syn_any_ready_task()
  br i1 %any_ready, label %restore, label %do_sleep
do_sleep:
  %locks = load volatile i32, ptr @__syn_sleep_lock_mask, align 4
  %dma_locks = load volatile i32, ptr @__syn_dma_locks, align 4
  %any_locks = or i32 %locks, %dma_locks
  %locked = icmp ne i32 %any_locks, 0
  %depth = select i1 %locked, i32 1, i32 2
  store volatile i32 %depth, ptr @__syn_last_sleep_depth, align 4
  %wc = load volatile i32, ptr @__syn_wfi_count, align 4
  %wc1 = add i32 %wc, 1
  store volatile i32 %wc1, ptr @__syn_wfi_count, align 4
  call void @__syn_on_idle(i32 0, i32 %depth)
  %has_dl = load volatile i8, ptr @__syn_has_deadline_waiter, align 1
  %no_dl = icmp eq i8 %has_dl, 0
  br i1 %no_dl, label %do_wfi, label %do_poll
do_wfi:
  call void asm sideeffect "wfi\0Acsrsi mstatus, 8\0Acsrci mstatus, 8", "~{{memory}}"()
  br label %restore
do_poll:
  call void @__syn_crit_exit(i32 %pm)
  call void asm sideeffect "nop\0Anop\0Anop\0Anop", "~{{memory}}"()
  ret void
restore:
  call void @__syn_crit_exit(i32 %pm)
  ret void
}}"#
        .to_string()
    };
    let idle_sleep_ir = idle_sleep_ir.replace("{sz}", sz).replace("{{", "{").replace("}}", "}");

    let _ = writeln!(
        out,
        r#"; ── RISC-V ({}) Pure-ISA CPU Primitives: mstatus & Idle Sleep ──
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
  call void @__syn_on_putc(i8 %c)
  ret void
}}

define void @__syn_platform_exit(i32 %code) noreturn nounwind{sz} {{
entry:
  call void @__syn_on_exit(i32 %code)
  br label %halt
halt:
  call void asm sideeffect "nop", "~{{memory}}"()
  br label %halt
}}

define void @__syn_use_systick(i32 %hz) nounwind{sz} {{
entry:
  store volatile i32 %hz, ptr @__syn_cpu_hz, align 4
  store i8 1, ptr @__syn_systick_enabled, align 1
  ret void
}}

define void @__syn_nvic_pend_irq(i32 %irq) nounwind{sz} {{
entry:
  call void @__syn_rv_dispatch_irq(i32 %irq)
  ret void
}}

{idle_sleep_ir}
"#,
        spec.cpu
    );
}
