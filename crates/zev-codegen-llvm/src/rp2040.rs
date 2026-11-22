//! Bare-Metal Raspberry Pi RP2040 (`thumbv6m-none-eabi`, Cortex-M0+ SRAM UF2 CPU Core).
//!
//! Emits ONLY architectural ARMv6-M Cortex-M0+ CPU primitives in LLVM 23 IR:
//! - Unified SRAM `.entry` (`_start` at `0x2000_0000`) setting MSP and `VTOR` (`0xE000_ED08`)
//! - 256-byte-aligned `[48 x ptr]` `.isr_vector` mapped directly from `.zev` `irq.attach(...)`
//! - `@Reset_Handler` (`.bss` zeroing -> `@__syn_boot()` -> `NVIC_ISER` enable -> scheduler)
//! - BootROM `'UB'` (`0x4255`) `reset_usb_boot(0, 0)` fallback in fault/exit handlers
//! - ARMv6-M `PRIMASK`, `SysTick`, `NVIC_ISPR`, `WFI`, and `.zev` `console_putc(u8)` forwarding
//!
//! Zero peripheral drivers live here — clocks (`XOSC`, `PLL_SYS`, `PLL_USB`, `CLOCKS`) and
//! USB 1.1 CDC-ACM (`USBCTRL`, `USB_DPRAM`) are implemented 100% in `.zev`.

use std::fmt::Write;
use zev_mir::MirIrqVector;

use crate::CodegenError;
use crate::nvic::NvicMap;

pub const RP2040_NVIC_IRQS: usize = 32;
pub const RP2040_VECTOR_LEN: usize = 16 + RP2040_NVIC_IRQS;
pub const RP2040_UF2_FAMILY_ID: u32 = 0xE48B_FF56;

/// Verify all attached IRQ vectors fit RP2040's 32 NVIC lines (`IRQ0..IRQ31`).
pub fn validate_rp2040_irq_vectors(vectors: &[MirIrqVector]) -> Result<(), CodegenError> {
    for v in vectors {
        if v.irq_num >= RP2040_NVIC_IRQS as u32 {
            return Err(CodegenError::new(
                "llvm-ir",
                format!(
                    "RP2040 NVIC supports IRQ0..IRQ31, but program attached IRQ {} (`{}`)",
                    v.irq_num, v.irq_name
                ),
            ));
        }
    }
    Ok(())
}

/// Emit RP2040 SRAM `.entry` (`_start` at `0x2000_0000`), `[48 x ptr]` `.isr_vector`,
/// BootROM `'UB'` `reset_usb_boot(0, 0)`, `@Default_Handler`, `@HardFault_Handler`,
/// `@SysTick_Handler`, and `@Reset_Handler`.
pub fn emit_rp2040_startup_ir(
    out: &mut String,
    initial_sp: u32,
    sz: &str,
    nvic: &NvicMap,
) {
    let iser_enable_ir = if nvic.iser0_mask != 0 {
        let iser0_i32 = nvic.iser0_mask as i32;
        format!(
            "  ; Enable Zev-attached NVIC IRQs in NVIC_ISER (0xE000E100 = -536813312)\n  \
             store volatile i32 {iser0_i32}, ptr inttoptr (i32 -536813312 to ptr), align 4\n  \
             call void asm sideeffect \"dsb\\0Aisb\", \"~{{memory}}\"()\n"
        )
    } else {
        String::new()
    };

    let _ = writeln!(
        out,
        r#"; ── Raspberry Pi RP2040 (ARMv6-M SRAM UF2 @ 0x20000000) Startup & Vectors ──
module asm ".section .entry, \22ax\22, %progbits"
module asm ".global _start"
module asm ".type _start, %function"
module asm ".thumb_func"
module asm "_start:"
module asm "  cpsid i"
module asm "  ldr r0, =0x{initial_sp:08X}"
module asm "  msr msp, r0"
module asm "  ldr r0, =__isr_vector"
module asm "  ldr r1, =0xE000ED08"
module asm "  str r0, [r1]"
module asm "  dsb"
module asm "  isb"
module asm "  bl Reset_Handler"
module asm "1: wfi"
module asm "  b 1b"

@__sbss = external global i8
@__ebss = external global i8
"#
    );

    let _ = writeln!(
        out,
        "@__isr_vector = constant [{RP2040_VECTOR_LEN} x ptr] ["
    );
    let _ = writeln!(
        out,
        "  ptr inttoptr (i32 {} to ptr), ; 0: Initial MSP (0x{initial_sp:08X})",
        initial_sp as i32
    );
    let _ = writeln!(out, "  ptr @Reset_Handler,                  ; 1: Reset");
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 2: NMI");
    let _ = writeln!(out, "  ptr @HardFault_Handler,              ; 3: HardFault");
    for exc in 4..=10 {
        let _ = writeln!(
            out,
            "  ptr null,                            ; {exc}: Reserved"
        );
    }
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 11: SVCall");
    let _ = writeln!(out, "  ptr null,                            ; 12: Reserved");
    let _ = writeln!(out, "  ptr null,                            ; 13: Reserved");
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 14: PendSV");
    let _ = writeln!(out, "  ptr @SysTick_Handler,                ; 15: SysTick");

    for irq in 0..RP2040_NVIC_IRQS {
        let handler = nvic.irq_symbols[irq]
            .as_deref()
            .unwrap_or("@Default_Handler");
        let comma = if irq + 1 == RP2040_NVIC_IRQS { "" } else { "," };
        let _ = writeln!(out, "  ptr {handler}{comma} ; {}: IRQ{irq}", 16 + irq);
    }
    let _ = writeln!(out, "], section \".isr_vector\", align 256");
    let _ = writeln!(
        out,
        "@llvm.used = appending global [1 x ptr] [ptr @__isr_vector], section \"llvm.metadata\"\n"
    );

    let _ = writeln!(
        out,
        r#"define void @__syn_rp2040_usb_boot_reset() noreturn nounwind{sz} {{
entry:
  call void asm sideeffect "cpsid i", "~{{memory}}"()
  ; Lookup ARM BootROM `reset_usb_boot` ('U' | ('B' << 8) = 16981 = 0x4255) via 0x14/0x18 ROM halfwords
  %ft_h = load volatile i16, ptr inttoptr (i32 20 to ptr), align 2
  %ft_i = zext i16 %ft_h to i32
  %ft_p = inttoptr i32 %ft_i to ptr
  %lu_h = load volatile i16, ptr inttoptr (i32 24 to ptr), align 2
  %lu_i = zext i16 %lu_h to i32
  %lu_p = inttoptr i32 %lu_i to ptr
  %ub_p = call ptr %lu_p(ptr %ft_p, i32 16981)
  call void %ub_p(i32 0, i32 0)
  br label %halt
halt:
  call void asm sideeffect "wfi", "~{{memory}}"()
  br label %halt
}}

define void @Default_Handler() noreturn nounwind {{
entry:
  call void @__syn_rp2040_usb_boot_reset()
  unreachable
}}

define void @HardFault_Handler() noreturn nounwind {{
entry:
  call void @__syn_rp2040_usb_boot_reset()
  unreachable
}}

define void @SysTick_Handler() nounwind {{
entry:
  %t = load volatile i32, ptr @__syn_tick_ms, align 4
  %next = add i32 %t, 1
  store volatile i32 %next, ptr @__syn_tick_ms, align 4
  ret void
}}

define void @Reset_Handler() #98 {{
entry:
  ; 1. Zero .bss [__sbss .. __ebss) (.text + .data already placed in SRAM by UF2)
  %sbss_a = ptrtoint ptr @__sbss to i32
  %ebss_a = ptrtoint ptr @__ebss to i32
  br label %bss_loop
bss_loop:
  %ba = phi i32 [ %sbss_a, %entry ], [ %ba_next, %bss_body ]
  %bdone = icmp uge i32 %ba, %ebss_a
  br i1 %bdone, label %boot, label %bss_body
bss_body:
  %bp = inttoptr i32 %ba to ptr
  store volatile i8 0, ptr %bp, align 1
  %ba_next = add i32 %ba, 1
  br label %bss_loop

boot:
  ; 2. Enable attached NVIC IRQs (masked by PRIMASK=1), run .zev sequential `fn main()` boot setup, then enter scheduler
{iser_enable_ir}  call void @__syn_boot()
  call void @__syn_sched_loop()
  unreachable
}}
"#
    );
}

/// Emit architectural ARMv6-M Cortex-M0+ runtime primitives (`PRIMASK`, `SysTick`,
/// `NVIC_ISPR`, `WFI` idle sleep, `sys.exit(0x4255)` BootROM reset) plus optional
/// `sys.set_putc` / `.zev` `fn console_putc(u8 b)` delegation.
pub fn emit_rp2040_hal_ir(
    out: &mut String,
    sz: &str,
    uses_set_putc: bool,
    console_putc_sym: Option<&str>,
) {
    let putc_body =
        crate::runtime_ir::format_uart_putc_body(uses_set_putc, 4, console_putc_sym, "  ret void");

    let _ = writeln!(
        out,
        r#"; ── ARMv6-M Cortex-M0+ CPU Kernel Primitives ──
define i32 @__syn_crit_enter() nounwind alwaysinline {{
entry:
  %pm = call i32 asm sideeffect "mrs $0, primask\0Acpsid i", "=r,~{{memory}}"()
  ret i32 %pm
}}

define void @__syn_crit_exit(i32 %pm) nounwind alwaysinline {{
entry:
  call void asm sideeffect "msr primask, $0", "r,~{{memory}}"(i32 %pm)
  ret void
}}

define void @__syn_uart_putc(i8 %c) nounwind noinline{sz} {{
entry:
{putc_body}
}}

define void @__syn_platform_exit(i32 %code) noreturn nounwind{sz} {{
entry:
  %is_boot = icmp eq i32 %code, 16981
  br i1 %is_boot, label %usb_boot, label %halt
usb_boot:
  call void @__syn_rp2040_usb_boot_reset()
  unreachable
halt:
  call void asm sideeffect "wfi", "~{{memory}}"()
  br label %halt
}}

define void @__syn_use_systick(i32 %hz) nounwind{sz} {{
entry:
  ; ARMv6-M SysTick RVR (0xE000E014 = -536813548), CVR (+0x18 = -536813544), CSR (+0x10 = -536813552)
  %khz = udiv i32 %hz, 1000
  %sub = sub i32 %khz, 1
  %gt_zero = icmp sgt i32 %sub, 1
  %min_clamped = select i1 %gt_zero, i32 %sub, i32 11999
  %lt_max = icmp ult i32 %min_clamped, 16777215
  %rvr = select i1 %lt_max, i32 %min_clamped, i32 16777215
  store i8 1, ptr @__syn_systick_enabled, align 1
  store volatile i32 %rvr, ptr inttoptr (i32 -536813548 to ptr), align 4
  store volatile i32 0, ptr inttoptr (i32 -536813544 to ptr), align 4
  store volatile i32 7, ptr inttoptr (i32 -536813552 to ptr), align 4
  ret void
}}

define void @__syn_nvic_pend_irq(i32 %irq) nounwind{sz} {{
entry:
  %in_boot = load volatile i8, ptr @__syn_cm_in_boot, align 1
  %is_boot = icmp ne i8 %in_boot, 0
  br i1 %is_boot, label %boot_sw, label %hw_ispr
boot_sw:
  call void @__syn_cm_dispatch_irq(i32 %irq)
  ret void
hw_ispr:
  ; ARMv6-M NVIC_ISPR (0xE000E200 = -536813056)
  %sh = and i32 %irq, 31
  %bit = shl i32 1, %sh
  store volatile i32 %bit, ptr inttoptr (i32 -536813056 to ptr), align 4
  call void asm sideeffect "dsb\0Aisb", "~{{memory}}"()
  ret void
}}

define void @__syn_idle_sleep() nounwind{sz} {{
entry:
  %pm = call i32 @__syn_crit_enter()
  %now = load volatile i32, ptr @__syn_tick_ms, align 4
  call void @__syn_wake_timers(i32 %now)
  %any_ready = call i1 @__syn_any_ready_task()
  br i1 %any_ready, label %restore, label %do_wfi
do_wfi:
  %locks = load volatile i32, ptr @__syn_sleep_lock_mask, align 4
  %dma_locks = load volatile i32, ptr @__syn_dma_locks, align 4
  %any_locks = or i32 %locks, %dma_locks
  %locked = icmp ne i32 %any_locks, 0
  %depth = select i1 %locked, i32 1, i32 2
  store volatile i32 %depth, ptr @__syn_last_sleep_depth, align 4
  ; Keep SCR.SLEEPDEEP clear on ARMv6-M (0xE000ED10 = -536810224) so SysTick & IRQs stay active
  %scr_old = load volatile i32, ptr inttoptr (i32 -536810224 to ptr), align 4
  %scr_clr = and i32 %scr_old, -5
  store volatile i32 %scr_clr, ptr inttoptr (i32 -536810224 to ptr), align 4
  %wc = load volatile i32, ptr @__syn_wfi_count, align 4
  %wc1 = add i32 %wc, 1
  store volatile i32 %wc1, ptr @__syn_wfi_count, align 4
  call void asm sideeffect "dsb\0Awfi\0Acpsie i\0Aisb\0Acpsid i", "~{{memory}}"()
  br label %restore
restore:
  call void @__syn_crit_exit(i32 %pm)
  call void asm sideeffect "isb", "~{{memory}}"()
  ret void
}}
"#
    );
}
