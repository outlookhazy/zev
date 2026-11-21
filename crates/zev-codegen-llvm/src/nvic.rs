//! ARM Cortex-M3 NVIC vector table mapping and bare-metal `.isr_vector` IR synthesis.
//!
//! Maps `.zev` hardware `isr <NAME>` declarations directly into Cortex-M3 NVIC vector slots
//! (`IRQ0..IRQ31`, exceptions `16..47`), computes the `NVIC_ISER0` (`0xE000E100`) bitmask
//! enabled during `@Reset_Handler`, and emits `@Reset_Handler`, `@SysTick_Handler`,
//! `@HardFault_Handler`, and `@Default_Handler` in pure LLVM 23 IR (zero `.s` files).

use std::collections::BTreeMap;
use std::fmt::Write;

/// Number of external NVIC IRQ lines emitted in `.isr_vector` (`IRQ0..IRQ31`, all in `ISER0`/`ISPR0`).
pub const CORTEX_M3_NVIC_IRQS: usize = 32;

/// Total entries in `.isr_vector` (16 Cortex-M system exceptions + 32 NVIC IRQs).
pub const CORTEX_M3_VECTOR_LEN: usize = 16 + CORTEX_M3_NVIC_IRQS;

/// Resolved mapping from user `isr <NAME>` names to Cortex-M3 NVIC IRQ numbers (`0..31`).
#[derive(Debug, Clone, Default)]
pub struct NvicMap {
    /// Ordered map: user ISR vector name -> NVIC IRQ index (`0..31`).
    pub by_name: BTreeMap<String, u8>,
    /// Reverse table: NVIC IRQ index (`0..31`) -> LLVM function symbol (e.g. `@__syn_isr_EXTI0`).
    pub irq_symbols: [Option<String>; CORTEX_M3_NVIC_IRQS],
    /// 32-bit bitmask written to `NVIC_ISER0` (`0xE000E100`) before `main()` / scheduler start.
    pub iser0_mask: u32,
}

impl NvicMap {
    /// Build a deterministic Cortex-M3 NVIC vector assignment for the given user ISR names.
    ///
    /// Canonical mappings (matching `lm3s6965evb` & STM32-style names used in `.zev` firmware):
    /// - `GPIOA` / `EXTI0` -> `IRQ0`
    /// - `GPIOB` / `EXTI1` -> `IRQ1`
    /// - `GPIOC` / `EXTI2` -> `IRQ2`
    /// - `GPIOD` / `EXTI3` -> `IRQ3`
    /// - `GPIOE` / `EXTI4` -> `IRQ4`
    /// - `UART0` / `USART1` / `DMA1_CH5` -> `IRQ5`
    /// - `UART1` / `USART2` / `DMA1_CH6` -> `IRQ6`
    /// - `SSI0` / `SPI1` / `DMA1_CH1` -> `IRQ7`
    /// - `I2C0` / `I2C1` / `DMA1_CH2` -> `IRQ8`
    /// - `ADC0` / `ADC1` -> `IRQ14`
    /// - `TIM0A` / `TIM1` -> `IRQ19`
    /// - `TIM1A` / `TIM2` -> `IRQ21`
    /// - `IRQ<N>` (`0..=31`) -> `IRQ<N>`
    pub fn from_isr_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        let mut map = NvicMap {
            by_name: BTreeMap::new(),
            irq_symbols: std::array::from_fn(|_| None),
            iser0_mask: 0,
        };

        let collected: Vec<&str> = names.into_iter().collect();

        // First pass: assign preferred canonical slots.
        let mut deferred = Vec::new();
        for &name in &collected {
            if map.by_name.contains_key(name) {
                continue;
            }
            if let Some(pref) = preferred_nvic_irq(name) {
                let idx = pref as usize;
                if idx < CORTEX_M3_NVIC_IRQS && map.irq_symbols[idx].is_none() {
                    let sym = format!("@__syn_isr_{}", sanitize_ident(name));
                    map.by_name.insert(name.to_string(), pref);
                    map.irq_symbols[idx] = Some(sym);
                    map.iser0_mask |= 1u32 << pref;
                    continue;
                }
            }
            deferred.push(name);
        }

        // Second pass: assign any remaining or colliding ISRs to the lowest free IRQ slot.
        for name in deferred {
            if map.by_name.contains_key(name) {
                continue;
            }
            if let Some(free_irq) = (0..CORTEX_M3_NVIC_IRQS).find(|&i| map.irq_symbols[i].is_none())
            {
                let irq = free_irq as u8;
                let sym = format!("@__syn_isr_{}", sanitize_ident(name));
                map.by_name.insert(name.to_string(), irq);
                map.irq_symbols[free_irq] = Some(sym);
                map.iser0_mask |= 1u32 << irq;
            }
        }

        map
    }

    /// Look up the assigned NVIC IRQ (`0..31`) for a user ISR vector name, or fall back to
    /// `preferred_nvic_irq(name)`.
    pub fn irq_for(&self, name: &str) -> Option<u8> {
        self.by_name
            .get(name)
            .copied()
            .or_else(|| preferred_nvic_irq(name))
    }
}

/// Preferred hardware IRQ index (`0..31`) for well-known Cortex-M peripheral names.
pub fn preferred_nvic_irq(name: &str) -> Option<u8> {
    let upper = name.to_ascii_uppercase();
    if let Some(rest) = upper.strip_prefix("IRQ") {
        if let Ok(n) = rest.parse::<u8>() {
            if (n as usize) < CORTEX_M3_NVIC_IRQS {
                return Some(n);
            }
        }
    }
    match upper.as_str() {
        "GPIOA" | "EXTI0" => Some(0),
        "GPIOB" | "EXTI1" => Some(1),
        "GPIOC" | "EXTI2" => Some(2),
        "GPIOD" | "EXTI3" => Some(3),
        "GPIOE" | "EXTI4" => Some(4),
        "UART0" | "USART1" | "DMA1_CH5" => Some(5),
        "UART1" | "USART2" | "DMA1_CH6" => Some(6),
        "SSI0" | "SPI1" | "DMA1_CH1" => Some(7),
        "I2C0" | "I2C1" | "DMA1_CH2" => Some(8),
        "PWM_FAULT" | "DMA1_CH3" => Some(9),
        "PWM0" | "DMA1_CH4" => Some(10),
        "PWM1" | "DMA1_CH7" => Some(11),
        "PWM2" => Some(12),
        "QEI0" => Some(13),
        "ADC0" | "ADC1" | "ADC" => Some(14),
        "ADC0_SS1" | "ADC2" => Some(15),
        "TIM0A" | "TIM1" | "TIM1_UP" => Some(19),
        "TIM0B" => Some(20),
        "TIM1A" | "TIM2" => Some(21),
        "TIM1B" => Some(22),
        "TIM2A" | "TIM3" => Some(23),
        "UART2" | "USART3" => Some(31),
        _ => None,
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
    let iser_i32 = nvic.iser0_mask as i32;

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
        "@__isr_vector = constant [{CORTEX_M3_VECTOR_LEN} x ptr] ["
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

    for irq in 0..CORTEX_M3_NVIC_IRQS {
        let handler = nvic.irq_symbols[irq]
            .as_deref()
            .unwrap_or("@Default_Handler");
        let comma = if irq + 1 == CORTEX_M3_NVIC_IRQS {
            ""
        } else {
            ","
        };
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
  ; 3. Enable user-declared NVIC IRQs in NVIC_ISER0 (0xE000E100 = -536813312)
  %mask = add i32 0, {iser_i32}
  %has_irqs = icmp ne i32 %mask, 0
  br i1 %has_irqs, label %enable_nvic, label %call_boot
enable_nvic:
  store volatile i32 %mask, ptr inttoptr (i32 -536813312 to ptr), align 4
  call void asm sideeffect "dsb\0Aisb", "~{{memory}}"()
  br label %call_boot

call_boot:
  ; 4. Run sequential main() boot phase, then enter cooperative Language-OS scheduler
  call void @__syn_boot()
  call void @__syn_sched_loop()
  unreachable
}}
"#
    );
}
