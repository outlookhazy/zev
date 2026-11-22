//! ARM Cortex-M3 NVIC vector table mapping and bare-metal `.isr_vector` IR synthesis.
//!
//! Maps `.zev` hardware `isr <NAME>` declarations directly into Cortex-M3 NVIC vector slots
//! (`IRQ0..IRQ31`, exceptions `16..47`), computes the `NVIC_ISER0` (`0xE000E100`) bitmask
//! enabled during `@Reset_Handler`, and emits `@Reset_Handler`, `@SysTick_Handler`,
//! `@HardFault_Handler`, and `@Default_Handler` in pure LLVM 23 IR (zero `.s` files).

use std::collections::BTreeMap;
use std::fmt::Write;

/// Number of external NVIC IRQ lines in the minimum Cortex-M3 `.isr_vector` (`IRQ0..IRQ31`).
pub const CORTEX_M3_NVIC_IRQS: usize = 32;

/// Minimum number of entries in `.isr_vector` (16 Cortex-M system exceptions + 32 NVIC IRQs).
pub const CORTEX_M3_VECTOR_LEN: usize = 16 + CORTEX_M3_NVIC_IRQS;

/// Resolved mapping from user `isr <NAME>` / `MirIrqVector` entries to Cortex-M3 NVIC vector slots.
#[derive(Debug, Clone, Default)]
pub struct NvicMap {
    /// Ordered map: user ISR vector name -> NVIC IRQ index (`0..31` legacy or `< 256`).
    pub by_name: BTreeMap<String, u8>,
    /// Reverse table: NVIC IRQ index (`0..31`) -> LLVM function symbol (e.g. `@__syn_isr_EXTI0` or `@__syn_vec_irq_0`).
    pub irq_symbols: [Option<String>; CORTEX_M3_NVIC_IRQS],
    /// 32-bit bitmask written to `NVIC_ISER0` (`0xE000E100`) before `main()` / scheduler start.
    pub iser0_mask: u32,
    /// Full external NVIC vector slots (`max(32, max_irq + 1)` entries).
    pub vector_slots: Vec<Option<String>>,
    /// Bitmasks written to `NVIC_ISER0..ISER7` (`0xE000_E100 + 4 * word`) for all attached IRQs.
    pub iser_masks: [u32; 8],
}

impl NvicMap {
    /// Build `.isr_vector` and `NVIC_ISER0..ISER7` masks from Wave 1 `MirIrqVector` bindings.
    pub fn from_irq_vectors(vectors: &[zev_mir::MirIrqVector]) -> Self {
        let max_irq = vectors.iter().map(|v| v.irq_num as usize).max().unwrap_or(0);
        let num_slots = (max_irq + 1).max(CORTEX_M3_NVIC_IRQS);
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
            if irq < CORTEX_M3_NVIC_IRQS {
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
        map
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

/// Emit the Cortex-M3 `.isr_vector` table, linker boundary globals, `@Reset_Handler`,
/// `@SysTick_Handler`, `@HardFault_Handler`, and `@Default_Handler` in LLVM 23 IR.
pub fn emit_cortex_m_startup_ir(
    out: &mut String,
    initial_sp: u32,
    nvic: &NvicMap,
) {
    let sp_i32 = initial_sp as i32;
    let ext_irqs = nvic.vector_slots.len().max(CORTEX_M3_NVIC_IRQS);
    let total_vector_len = 16 + ext_irqs;

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

    let _ = writeln!(out, "; ── ARM Cortex-M3 Freestanding Vector Table & Startup ──");
    let _ = writeln!(out, "@__sbss = external global i8");
    let _ = writeln!(out, "@__ebss = external global i8");
    let _ = writeln!(out, "@__sdata = external global i8");
    let _ = writeln!(out, "@__edata = external global i8");
    let _ = writeln!(out, "@__sidata = external global i8");
    let _ = writeln!(
        out,
        "@.hf_msg = private unnamed_addr constant [11 x i8] c\"HARDFAULT\\0A\\00\", align 1"
    );
    let _ = writeln!(out);

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
                if irq < CORTEX_M3_NVIC_IRQS {
                    nvic.irq_symbols[irq].as_deref()
                } else {
                    None
                }
            })
            .unwrap_or("@Default_Handler");
        let comma = if irq + 1 == ext_irqs { "" } else { "," };
        let _ = writeln!(out, "  ptr {handler}{comma} ; {}: IRQ{irq}", 16 + irq);
    }
    let _ = writeln!(out, "], section \".isr_vector\", align 4");
    let _ = writeln!(
        out,
        "@llvm.used = appending global [1 x ptr] [ptr @__isr_vector], section \"llvm.metadata\""
    );
    let _ = writeln!(out);

    // Default_Handler & HardFault_Handler
    let _ = writeln!(
        out,
        r#"define void @Default_Handler() nounwind {{
entry:
  call void @__syn_platform_exit(i32 2)
  unreachable
}}

define void @HardFault_Handler() nounwind {{
entry:
  call void @__syn_uart_puts(ptr @.hf_msg)
  call void @__syn_platform_exit(i32 3)
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
  call void asm sideeffect "cpsid i", "~{{memory}}"()
  ; 1. Zero .bss [__sbss .. __ebss) in integer address domain (avoids provenance UB & memset idiom)
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
  ; 2. Copy initialized .data from Flash LMA (__sidata) to SRAM VMA [__sdata .. __edata)
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

hw_init:
  ; 3. Enable attached NVIC IRQs in NVIC_ISER0..ISER7 (0xE000E100..0xE000E11C)
{nvic_enable_ir}  br label %call_boot

call_boot:
  ; 4. Run sequential main() boot phase, then enter cooperative Language-OS scheduler
  call void @__syn_boot()
  call void @__syn_sched_loop()
  unreachable
}}
"#
    );
}
