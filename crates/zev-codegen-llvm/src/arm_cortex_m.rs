//! Pure-ISA ARM Cortex-M Backend (Thumbv6-M / Thumbv7-M / Thumbv7E-M).
//!
//! Synthesizes `.isr_vector`, optional SRAM `.entry` (`_start` + `VTOR`), `@Reset_Handler`,
//! `@SysTick_Handler`, `@HardFault_Handler`, `@Default_Handler`, `PRIMASK` critical sections,
//! `SysTick` (`0xE000_E010`), `NVIC_ISPR` (`0xE000_E200`), and `wfi` idle sleep solely from
//! declarative `TargetSpec` parameters — zero chip/board names or peripheral MMIO addresses.

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::target_spec::{IrqModel, TargetSpec};
use crate::toolchain::CodegenError;

/// Minimum Cortex-M external NVIC IRQ count when `max_irqs` allows at least 32 lines.
pub const CORTEX_M_MIN_NVIC_IRQS: usize = 32;

/// Resolved mapping from user `isr <NAME>` / `MirIrqVector` entries to NVIC vector slots.
#[derive(Debug, Clone, Default)]
pub struct NvicMap {
    /// Ordered map: user ISR vector name -> NVIC IRQ index (`< 256`).
    pub by_name: BTreeMap<String, u8>,
    /// Legacy convenience table: NVIC IRQ index (`0..31`) -> LLVM function symbol.
    pub irq_symbols: [Option<String>; CORTEX_M_MIN_NVIC_IRQS],
    /// 32-bit bitmask written to `NVIC_ISER0` (`0xE000E100`) during `@Reset_Handler`.
    pub iser0_mask: u32,
    /// Full external NVIC vector slots (`max(min_vector_irqs, max_irq + 1)` entries).
    pub vector_slots: Vec<Option<String>>,
    /// Bitmasks written to `NVIC_ISER0..ISER7` (`0xE000_E100 + 4 * word`) for attached IRQs.
    pub iser_masks: [u32; 8],
}

impl NvicMap {
    /// Validate and map `MirIrqVector` entries against generic `[irq]` bounds (`max_irqs`, `min_vector_irqs`).
    pub fn from_spec(
        vectors: &[zev_mir::MirIrqVector],
        max_irqs: u32,
        min_vector_irqs: u32,
    ) -> Result<Self, CodegenError> {
        for v in vectors {
            if v.irq_num >= max_irqs {
                return Err(CodegenError::new(
                    "llvm-ir",
                    format!(
                        "ARM Cortex-M NVIC supports IRQ0..IRQ{}, but program attached IRQ {} (`{}`)",
                        max_irqs.saturating_sub(1),
                        v.irq_num,
                        v.irq_name
                    ),
                ));
            }
        }
        let min_slots = (min_vector_irqs as usize).max(CORTEX_M_MIN_NVIC_IRQS);
        let max_irq = vectors.iter().map(|v| v.irq_num as usize).max().unwrap_or(0);
        let num_slots = (max_irq + 1).max(min_slots);
        let mut map = NvicMap {
            by_name: BTreeMap::new(),
            irq_symbols: std::array::from_fn(|_| None),
            iser0_mask: 0,
            vector_slots: vec![None; num_slots],
            iser_masks: [0u32; 8],
        };

        for v in vectors {
            let irq = v.irq_num as usize;
            let sym = format!("@__syn_vec_irq_{}", v.irq_num);
            if irq < map.vector_slots.len() {
                map.vector_slots[irq] = Some(sym.clone());
            }
            if irq < CORTEX_M_MIN_NVIC_IRQS {
                map.irq_symbols[irq] = Some(sym);
            }
            if let Ok(irq_u8) = u8::try_from(v.irq_num) {
                map.by_name.insert(v.irq_name.clone(), irq_u8);
            }
            let word = irq / 32;
            let bit = v.irq_num % 32;
            if word < 8 {
                map.iser_masks[word] |= 1u32 << bit;
            }
        }
        map.iser0_mask = map.iser_masks[0];
        Ok(map)
    }

    /// Build `.isr_vector` and `NVIC_ISER0..ISER7` masks with generous 240-line default bounds.
    pub fn from_irq_vectors(vectors: &[zev_mir::MirIrqVector]) -> Self {
        Self::from_spec(vectors, 240, 32).unwrap_or_default()
    }
}

/// Validate and build NVIC map from `TargetSpec`.
pub fn build_nvic_map(
    vectors: &[zev_mir::MirIrqVector],
    spec: &TargetSpec,
) -> Result<NvicMap, CodegenError> {
    match spec.irq {
        IrqModel::CortexMNvic {
            max_irqs,
            min_vector_irqs,
        } => NvicMap::from_spec(vectors, max_irqs, min_vector_irqs),
        _ => Ok(NvicMap::from_irq_vectors(vectors)),
    }
}

/// Sanitize a `.zev` identifier for LLVM symbol emission.
pub fn sanitize_ident(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "_anon".to_string()
    } else {
        out
    }
}

/// Emit pure-ISA Cortex-M startup (`.entry` if `code_in_ram`, `.isr_vector`, `@Reset_Handler`,
/// `@SysTick_Handler`, `@HardFault_Handler`, `@Default_Handler`).
pub fn emit_arm_cortex_m_startup_ir(
    out: &mut String,
    spec: &TargetSpec,
    sz: &str,
    nvic: &NvicMap,
) {
    let initial_sp = spec.initial_sp();
    let sp_i32 = initial_sp as i32;
    let ext_irqs = nvic.vector_slots.len().max(CORTEX_M_MIN_NVIC_IRQS);
    let total_vector_len = 16 + ext_irqs;
    let vec_align = if spec.code_in_ram || spec.has_vtor { 256 } else { 4 };

    let mut iser_words = nvic.iser_masks;
    if iser_words[0] == 0 && nvic.iser0_mask != 0 {
        iser_words[0] = nvic.iser0_mask;
    }

    let mut nvic_enable_ir = String::new();
    let mut any_iser = false;
    for (word, &mask) in iser_words.iter().enumerate() {
        if mask != 0 {
            any_iser = true;
            let addr_u32 = 0xE000_E100u32 + (word as u32) * 4;
            let addr_i32 = addr_u32 as i32;
            let mask_i32 = mask as i32;
            let _ = writeln!(
                nvic_enable_ir,
                "  ; Enable NVIC_ISER{word} (0x{addr_u32:08X}) = 0x{mask:08X}"
            );
            let _ = writeln!(
                nvic_enable_ir,
                "  store volatile i32 {mask_i32}, ptr inttoptr (i32 {addr_i32} to ptr), align 4"
            );
        }
    }
    if any_iser {
        let _ = writeln!(
            nvic_enable_ir,
            "  call void asm sideeffect \"dsb\\0Aisb\", \"~{{memory}}\"()"
        );
    }

    let _ = writeln!(
        out,
        "; ── ARM Cortex-M Pure-ISA Freestanding Vector Table & Startup ({}) ──",
        spec.cpu
    );

    if spec.code_in_ram {
        let vtor_ir = if spec.has_vtor {
            "module asm \"  ldr r0, =__isr_vector\"\n\
             module asm \"  ldr r1, =0xE000ED08\"\n\
             module asm \"  str r0, [r1]\"\n"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            r#"module asm ".section .entry, \22ax\22, %progbits"
module asm ".global _start"
module asm ".type _start, %function"
module asm ".thumb_func"
module asm "_start:"
module asm "  cpsid i"
module asm "  ldr r0, =0x{initial_sp:08X}"
module asm "  msr msp, r0"
{vtor_ir}module asm "  dsb"
module asm "  isb"
module asm "  bl Reset_Handler"
module asm ".balign 4"
"#
        );
    }

    let _ = writeln!(out, "@__sbss = external global i8");
    let _ = writeln!(out, "@__ebss = external global i8");
    let _ = writeln!(out, "@__sdata = external global i8");
    let _ = writeln!(out, "@__edata = external global i8");
    let _ = writeln!(out, "@__sidata = external global i8");
    let _ = writeln!(
        out,
        "@.hf_msg = private unnamed_addr constant [11 x i8] c\"HARDFAULT\\0A\\00\", align 1\n"
    );

    let _ = writeln!(
        out,
        "@__isr_vector = constant [{total_vector_len} x ptr] ["
    );
    let _ = writeln!(
        out,
        "  ptr inttoptr (i32 {sp_i32} to ptr), ; 0: Initial Main Stack Pointer (0x{initial_sp:08X})"
    );
    let _ = writeln!(out, "  ptr @Reset_Handler,                  ; 1: Reset");
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 2: NMI");
    let _ = writeln!(out, "  ptr @HardFault_Handler,              ; 3: HardFault");
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 4: MemManage");
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 5: BusFault");
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 6: UsageFault");
    let _ = writeln!(out, "  ptr null,                            ; 7: Reserved");
    let _ = writeln!(out, "  ptr null,                            ; 8: Reserved");
    let _ = writeln!(out, "  ptr null,                            ; 9: Reserved");
    let _ = writeln!(out, "  ptr null,                            ; 10: Reserved");
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 11: SVCall");
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 12: DebugMon");
    let _ = writeln!(out, "  ptr null,                            ; 13: Reserved");
    let _ = writeln!(out, "  ptr @Default_Handler,                ; 14: PendSV");
    let _ = writeln!(out, "  ptr @SysTick_Handler,                ; 15: SysTick");

    for irq in 0..ext_irqs {
        let handler = nvic
            .vector_slots
            .get(irq)
            .and_then(|s| s.as_deref())
            .or_else(|| {
                if irq < CORTEX_M_MIN_NVIC_IRQS {
                    nvic.irq_symbols[irq].as_deref()
                } else {
                    None
                }
            })
            .unwrap_or("@Default_Handler");
        let comma = if irq + 1 == ext_irqs { "" } else { "," };
        let _ = writeln!(out, "  ptr {handler}{comma} ; {}: IRQ{irq}", 16 + irq);
    }
    let _ = writeln!(out, "], section \".isr_vector\", align {vec_align}");
    let _ = writeln!(
        out,
        "@llvm.used = appending global [1 x ptr] [ptr @__isr_vector], section \"llvm.metadata\"\n"
    );

    let fault_putc = if spec.semihosting {
        "  call void @__syn_uart_puts(ptr @.hf_msg)\n"
    } else {
        ""
    };
    let _ = writeln!(
        out,
        r#"define void @Default_Handler() noreturn nounwind{sz} {{
entry:
  call void asm sideeffect "cpsid i", "~{{memory}}"()
  call void @__syn_on_fault()
  call void @__syn_platform_exit(i32 2)
  unreachable
}}

define void @HardFault_Handler() noreturn nounwind{sz} {{
entry:
  call void asm sideeffect "cpsid i", "~{{memory}}"()
{fault_putc}  call void @__syn_on_fault()
  call void @__syn_platform_exit(i32 3)
  unreachable
}}

define void @SysTick_Handler() nounwind{sz} {{
entry:
  %t = load volatile i32, ptr @__syn_tick_ms, align 4
  %next = add i32 %t, 1
  store volatile i32 %next, ptr @__syn_tick_ms, align 4
  ret void
}}
"#
    );

    let data_copy_ir = if spec.code_in_ram {
        "  br label %hw_init\n".to_string()
    } else {
        r#"  ; Copy initialized .data from Flash LMA (__sidata) to SRAM VMA [__sdata .. __edata)
  %sdata_a = ptrtoint ptr @__sdata to i32
  %edata_a = ptrtoint ptr @__edata to i32
  %sidata_a = ptrtoint ptr @__sidata to i32
  br label %data_loop
data_loop:
  %da = phi i32 [ %sdata_a, %data_init ], [ %da_next, %data_body ]
  %sa = phi i32 [ %sidata_a, %data_init ], [ %sa_next, %data_body ]
  %ddone = icmp uge i32 %da, %edata_a
  br i1 %ddone, label %hw_init, label %data_body
data_body:
  %sp = inttoptr i32 %sa to ptr
  %dp = inttoptr i32 %da to ptr
  %b = load volatile i8, ptr %sp, align 1
  store volatile i8 %b, ptr %dp, align 1
  %da_next = add i32 %da, 1
  %sa_next = add i32 %sa, 1
  br label %data_loop
"#
        .to_string()
    };

    let vtor_reset_ir = if !spec.code_in_ram && spec.has_vtor {
        "  %vt_a = ptrtoint ptr @__isr_vector to i32\n  store volatile i32 %vt_a, ptr inttoptr (i32 -536810232 to ptr), align 4\n  call void asm sideeffect \"dsb\\0Aisb\", \"~{memory}\"()\n"
    } else {
        ""
    };

    let fpu_enable_ir = if spec.has_fpu {
        if spec.has_thumb2 {
            "  call void asm sideeffect \"movw r0, #0xed88\\0Amovt r0, #0xe000\\0Aldr r1, [r0]\\0Aorr r1, r1, #0x00f00000\\0Astr r1, [r0]\\0Adsb\\0Aisb\", \"~{r0},~{r1},~{memory}\"()\n"
        } else {
            "  call void asm sideeffect \"ldr r0, =0xE000ED88\\0Aldr r1, [r0]\\0Aorr r1, r1, #0x00F00000\\0Astr r1, [r0]\\0Adsb\\0Aisb\", \"~{r0},~{r1},~{memory}\"()\n"
        }
    } else {
        ""
    };

    let _ = writeln!(
        out,
        r#"define void @Reset_Handler() #97 {{
entry:
  call void asm sideeffect "cpsid i", "~{{memory}}"()
{fpu_enable_ir}  ; 1. Zero .bss [__sbss .. __ebss)
  %sbss_a = ptrtoint ptr @__sbss to i32
  %ebss_a = ptrtoint ptr @__ebss to i32
  br label %bss_loop
bss_loop:
  %ba = phi i32 [ %sbss_a, %entry ], [ %ba_next, %bss_body ]
  %bdone = icmp uge i32 %ba, %ebss_a
  br i1 %bdone, label %data_init, label %bss_body
bss_body:
  %bp = inttoptr i32 %ba to ptr
  store volatile i8 0, ptr %bp, align 1
  %ba_next = add i32 %ba, 1
  br label %bss_loop

data_init:
{data_copy_ir}
hw_init:
{vtor_reset_ir}{nvic_enable_ir}  br label %call_boot

call_boot:
  call void @__syn_boot()
  call void @__syn_sched_loop()
  unreachable
}}
"#
    );
}

/// Emit pure-ISA ARM Cortex-M CPU runtime HAL (`PRIMASK`, `SysTick`, `NVIC_ISPR`,
/// `@__syn_uart_putc`, `@__syn_platform_exit`, `@__syn_idle_sleep`).
pub fn emit_arm_cortex_m_hal_ir(
    out: &mut String,
    spec: &TargetSpec,
    sz: &str,
    uses_clock_hook: bool,
) {
    let exit_tail = if spec.semihosting && spec.has_thumb2 {
        r#"  %ok = icmp eq i32 %code, 0
  br i1 %ok, label %exit_zero, label %exit_nonzero
exit_zero:
  ; ARM semihosting SYS_EXIT (0x18) + ADP_Stopped_ApplicationExit (0x20026)
  call void asm sideeffect "mov r0, #0x18\0Amovw r1, #0x0026\0Amovt r1, #0x0002\0Abkpt #0xab", "~{r0},~{r1},~{memory}"()
  unreachable
exit_nonzero:
  call void asm sideeffect "mov r0, #0x18\0Amov r1, #1\0Abkpt #0xab", "~{r0},~{r1},~{memory}"()
  unreachable"#
    } else {
        r#"  br label %halt
halt:
  call void asm sideeffect "cpsid i\0Awfi", "~{memory}"()
  br label %halt"#
    };

    let auto_st_check = if uses_clock_hook {
        "  br label %check_sleep\n"
    } else {
        r#"  %en = load i8, ptr @__syn_systick_enabled, align 1
  %is_en = icmp ne i8 %en, 0
  br i1 %is_en, label %check_sleep, label %init_default_st
init_default_st:
  call void @__syn_use_systick(i32 12000000)
  br label %check_sleep
"#
    };

    let scr_update_ir = if spec.allow_sleep_deep {
        r#"  %scr_old = load volatile i32, ptr inttoptr (i32 -536810224 to ptr), align 4
  %scr_clr = and i32 %scr_old, -5
  %scr_bit = select i1 %locked, i32 0, i32 4
  %scr_new = or i32 %scr_clr, %scr_bit
  store volatile i32 %scr_new, ptr inttoptr (i32 -536810224 to ptr), align 4"#
    } else {
        r#"  %scr_old = load volatile i32, ptr inttoptr (i32 -536810224 to ptr), align 4
  %scr_clr = and i32 %scr_old, -5
  store volatile i32 %scr_clr, ptr inttoptr (i32 -536810224 to ptr), align 4"#
    };

    let _ = writeln!(
        out,
        r#"; ── ARM Cortex-M Pure-ISA CPU Primitives ({}) ──
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

define void @__syn_uart_putc(i8 %c) nounwind{sz} {{
entry:
  call void @__syn_on_putc(i8 %c)
  ret void
}}

define void @__syn_platform_exit(i32 %code) noreturn nounwind{sz} {{
entry:
  call void @__syn_on_exit(i32 %code)
{exit_tail}
}}

define void @__syn_use_systick(i32 %hz) nounwind{sz} {{
entry:
  store volatile i32 %hz, ptr @__syn_cpu_hz, align 4
  %khz = udiv i32 %hz, 1000
  %sub = sub i32 %khz, 1
  %gt_zero = icmp sgt i32 %sub, 1
  %min_clamped = select i1 %gt_zero, i32 %sub, i32 11999
  %lt_max = icmp ult i32 %min_clamped, 16777215
  %rvr = select i1 %lt_max, i32 %min_clamped, i32 16777215
  store i8 1, ptr @__syn_systick_enabled, align 1
  ; SysTick RVR (0xE000E014 = -536813548), CVR (0xE000E018 = -536813544), CSR (0xE000E010 = -536813552)
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
  %word = lshr i32 %irq, 5
  %wcl = and i32 %word, 7
  %off = shl i32 %wcl, 2
  ; NVIC_ISPR0..ISPR7 at 0xE000E200 (-536813056) + 4 * word
  %addr = add i32 -536813056, %off
  %ispr_p = inttoptr i32 %addr to ptr
  %sh = and i32 %irq, 31
  %bit = shl i32 1, %sh
  store volatile i32 %bit, ptr %ispr_p, align 4
  call void asm sideeffect "dsb\0Aisb", "~{{memory}}"()
  ret void
}}

define void @__syn_idle_sleep() nounwind{sz} {{
entry:
{auto_st_check}check_sleep:
  %pm = call i32 @__syn_crit_enter()
  %now = call i32 @__syn_now_ms()
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
{scr_update_ir}
  %wc = load volatile i32, ptr @__syn_wfi_count, align 4
  %wc1 = add i32 %wc, 1
  store volatile i32 %wc1, ptr @__syn_wfi_count, align 4
  call void @__syn_on_idle(i32 0, i32 %depth)
  call void asm sideeffect "dsb\0Awfi\0Acpsie i\0Aisb\0Acpsid i", "~{{memory}}"()
  br label %restore
restore:
  call void @__syn_crit_exit(i32 %pm)
  call void asm sideeffect "isb", "~{{memory}}"()
  ret void
}}
"#,
        spec.cpu
    );
}
