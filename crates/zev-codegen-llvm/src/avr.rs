//! Bare-Metal Microchip ATmega328P (`avr`, Arduino Uno R3 16 MHz Optiboot) Startup,
//! Watchdog Disarm, 1 kHz `Timer0` Tick, `115200 8N1` `USART0` Console, `avr_signalcc`
//! Vector Table, `avr-libgcc` Integer Div/Mod/Mul ABI Shims, and Program Address-Space Pass.

use std::collections::BTreeSet;
use std::fmt::Write;

/// Total ATmega328P interrupt vectors (`0..=25`).
pub const AVR_ATMEGA328P_VECTOR_COUNT: u32 = 26;

/// Vector index reserved for Zev's 1.000 kHz system tick (`TIMER0 COMPA`, word `0x001C`).
pub const AVR_TIMER0_COMPA_IRQ: u32 = 14;

/// Validate user-attached IRQ vector numbers against ATmega328P vector slots `1..=25`
/// (excluding slot `0` RESET and slot `14` `TIMER0_COMPA` used by the Zev scheduler tick).
pub fn validate_avr_irq_vectors(
    vectors: &[zev_mir::MirIrqVector],
) -> Result<BTreeSet<u32>, crate::CodegenError> {
    let mut attached = BTreeSet::new();
    for v in vectors {
        if v.irq_num == 0 {
            return Err(crate::CodegenError::new(
                "llvm-ir",
                "ATmega328P IRQ vector 0 is the RESET vector and cannot be attached",
            ));
        }
        if v.irq_num == AVR_TIMER0_COMPA_IRQ {
            return Err(crate::CodegenError::new(
                "llvm-ir",
                format!(
                    "ATmega328P IRQ vector {AVR_TIMER0_COMPA_IRQ} (`TIMER0_COMPA`) is reserved by Zev's 1 ms system tick"
                ),
            ));
        }
        if v.irq_num >= AVR_ATMEGA328P_VECTOR_COUNT {
            return Err(crate::CodegenError::new(
                "llvm-ir",
                format!(
                    "ATmega328P IRQ vector {} exceeds vector table range (1..={})",
                    v.irq_num,
                    AVR_ATMEGA328P_VECTOR_COUNT - 1
                ),
            ));
        }
        attached.insert(v.irq_num);
    }
    Ok(attached)
}

/// Emit ATmega328P 26-slot `.vectors` table (`jmp`), `.text.startup` reset/WDT/data/bss init,
/// `@__syn_avr_hw_init` (`USART0` 115200 8N1 + `Timer0` 1 kHz CTC), and `avr_signalcc` ISRs.
pub fn emit_avr_startup_ir(
    out: &mut String,
    initial_sp: u32,
    sz: &str,
    attached_irqs: &BTreeSet<u32>,
) {
    let sp_lo = initial_sp & 0xFF;
    let sp_hi = (initial_sp >> 8) & 0xFF;

    let _ = writeln!(
        out,
        "; ── ATmega328P (Arduino Uno R3 16 MHz) 26-Slot Flash Vector Table (@ 0x0000) ──"
    );
    let _ = writeln!(out, "module asm \".section .vectors, \\22ax\\22, @progbits\"");
    let _ = writeln!(out, "module asm \".global _start\"");
    let _ = writeln!(out, "module asm \"_start:\"");
    let _ = writeln!(out, "module asm \"  jmp __syn_avr_reset\"");
    for vec_idx in 1..AVR_ATMEGA328P_VECTOR_COUNT {
        if vec_idx == AVR_TIMER0_COMPA_IRQ {
            let _ = writeln!(out, "module asm \"  jmp __syn_avr_systick_isr\"");
        } else if attached_irqs.contains(&vec_idx) {
            let _ = writeln!(out, "module asm \"  jmp __syn_avr_vec_{vec_idx}\"");
        } else {
            let _ = writeln!(out, "module asm \"  jmp __syn_avr_unhandled_irq\"");
        }
    }

    let _ = writeln!(
        out,
        r#"
; ── Reset, Watchdog Disarm, Flash->SRAM .data/.rodata Copy & .bss Zeroing ──
module asm ".section .text.startup, \22ax\22, @progbits"
module asm ".global __syn_avr_reset"
module asm ".global __do_copy_data"
module asm ".global __do_clear_bss"
module asm "__syn_avr_reset:"
module asm "  cli"
module asm "  eor r1, r1"
module asm "  out 0x3f, r1"
module asm "  ldi r28, 0x{sp_lo:02X}"
module asm "  ldi r29, 0x{sp_hi:02X}"
module asm "  out 0x3e, r29"
module asm "  out 0x3d, r28"
module asm "  sts 0x0054, r1"
module asm "  ldi r24, 0x18"
module asm "  sts 0x0060, r24"
module asm "  sts 0x0060, r1"
module asm "__do_copy_data:"
module asm "  ldi r30, lo8(__sidata)"
module asm "  ldi r31, hi8(__sidata)"
module asm "  ldi r26, lo8(__sdata)"
module asm "  ldi r27, hi8(__sdata)"
module asm "  ldi r24, lo8(__edata)"
module asm "  ldi r25, hi8(__edata)"
module asm "1:"
module asm "  cp  r26, r24"
module asm "  cpc r27, r25"
module asm "  breq 2f"
module asm "  lpm r0, Z+"
module asm "  st  X+, r0"
module asm "  rjmp 1b"
module asm "__do_clear_bss:"
module asm "2:"
module asm "  ldi r26, lo8(__sbss)"
module asm "  ldi r27, hi8(__sbss)"
module asm "  ldi r24, lo8(__ebss)"
module asm "  ldi r25, hi8(__ebss)"
module asm "3:"
module asm "  cp  r26, r24"
module asm "  cpc r27, r25"
module asm "  breq 4f"
module asm "  st  X+, r1"
module asm "  rjmp 3b"
module asm "4:"
module asm "  call __syn_avr_entry"
module asm "5: rjmp 5b"

@__sdata = external global i8
@__edata = external global i8
@__sidata = external global i8
@__sbss = external global i8
@__ebss = external global i8

define void @__syn_avr_hw_init() nounwind{sz} {{
entry:
  ; 1. Configure USART0 for 115200 8N1 at 16 MHz (`U2X0=1`, `UBRR0=16`):
  ;    UCSR0A (0xC0=192)=0x02, UBRR0L (0xC4=196)=16, UBRR0H (0xC5=197)=0,
  ;    UCSR0C (0xC2=194)=0x06 (8N1), UCSR0B (0xC1=193)=0x18 (RXEN0|TXEN0)
  store volatile i8 2, ptr inttoptr (i16 192 to ptr), align 1
  store volatile i8 16, ptr inttoptr (i16 196 to ptr), align 1
  store volatile i8 0, ptr inttoptr (i16 197 to ptr), align 1
  store volatile i8 6, ptr inttoptr (i16 194 to ptr), align 1
  store volatile i8 24, ptr inttoptr (i16 193 to ptr), align 1

  ; 2. Configure Timer0 for 1.000 kHz CTC tick (16 MHz / 64 = 250 kHz, OCR0A=249):
  ;    TCCR0A (0x44=68)=0x02 (CTC), TCCR0B (0x45=69)=0x03 (/64), TCNT0 (0x46=70)=0,
  ;    OCR0A (0x47=71)=249, TIFR0 (0x35=53)=0x02 (clear OCF0A), TIMSK0 (0x6E=110)=0x02 (OCIE0A)
  store volatile i8 2, ptr inttoptr (i16 68 to ptr), align 1
  store volatile i8 3, ptr inttoptr (i16 69 to ptr), align 1
  store volatile i8 0, ptr inttoptr (i16 70 to ptr), align 1
  store volatile i8 -7, ptr inttoptr (i16 71 to ptr), align 1
  store volatile i8 2, ptr inttoptr (i16 53 to ptr), align 1
  store volatile i8 2, ptr inttoptr (i16 110 to ptr), align 1
  store i8 1, ptr @__syn_systick_enabled, align 1
  ret void
}}

define void @__syn_avr_entry() noreturn nounwind{sz} {{
entry:
  call void @__syn_avr_hw_init()
  call void @__syn_boot()
  call void @__syn_sched_loop()
  unreachable
}}

define avr_signalcc void @__syn_avr_systick_isr() nounwind {{
entry:
  %t = load volatile i32, ptr @__syn_tick_ms, align 4
  %t1 = add i32 %t, 1
  store volatile i32 %t1, ptr @__syn_tick_ms, align 4
  ret void
}}

define avr_signalcc void @__syn_avr_unhandled_irq() nounwind {{
entry:
  ret void
}}
"#
    );

    for &irq in attached_irqs {
        let _ = writeln!(
            out,
            r#"define avr_signalcc void @__syn_avr_vec_{irq}() nounwind {{
entry:
  call void @__syn_vec_irq_{irq}()
  ret void
}}
"#
        );
    }
}

/// Emit ATmega328P hardware runtime primitives (`SREG` critical sections, tear-free
/// `@__syn_avr_read_ms`, `USART0` blocking putc, AVR Idle `sleep`, and per-section
/// weak `avr-libgcc` integer mul/divmod shims backed by pure SSA `#99` helpers).
pub fn emit_avr_hal_ir(out: &mut String, sz: &str) {
    let _ = writeln!(
        out,
        r#"; ── ATmega328P Hardware Primitives: SREG, USART0 (115200), Timer0, Idle Sleep ──
define i32 @__syn_crit_enter() nounwind alwaysinline {{
entry:
  %sr = call i8 asm sideeffect "in $0, 0x3f\0Acli", "=r,~{{memory}}"()
  %z = zext i8 %sr to i32
  ret i32 %z
}}

define void @__syn_crit_exit(i32 %pm) nounwind alwaysinline {{
entry:
  %ib = and i32 %pm, 128
  %en = icmp ne i32 %ib, 0
  br i1 %en, label %do_sei, label %done
do_sei:
  call void asm sideeffect "sei", "~{{memory}}"()
  br label %done
done:
  ret void
}}

define i32 @__syn_avr_read_ms() nounwind noinline{sz} {{
entry:
  %pm = call i32 @__syn_crit_enter()
  %t = load volatile i32, ptr @__syn_tick_ms, align 4
  call void @__syn_crit_exit(i32 %pm)
  ret i32 %t
}}

define void @__syn_uart_putc(i8 %c) nounwind noinline{sz} {{
entry:
  br label %wait_udre
wait_udre:
  %st = load volatile i8, ptr inttoptr (i16 192 to ptr), align 1
  %udre = and i8 %st, 32
  %ready = icmp ne i8 %udre, 0
  br i1 %ready, label %tx, label %wait_udre
tx:
  store volatile i8 %c, ptr inttoptr (i16 198 to ptr), align 1
  ret void
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
  %pm = call i32 @__syn_crit_enter()
  call void @__syn_avr_dispatch_irq(i32 %irq)
  call void @__syn_crit_exit(i32 %pm)
  ret void
}}

define void @__syn_idle_sleep() nounwind{sz} {{
entry:
  %pm = call i32 @__syn_crit_enter()
  %now = load volatile i32, ptr @__syn_tick_ms, align 4
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
  ; SMCR (0x53=83): SE=1, SM=000 (Idle mode keeps Timer0 & USART0 running)
  store volatile i8 1, ptr inttoptr (i16 83 to ptr), align 1
  call void asm sideeffect "sei\0Asleep", "~{{memory}}"()
  store volatile i8 0, ptr inttoptr (i16 83 to ptr), align 1
  br label %restore
restore:
  call void @__syn_crit_exit(i32 %pm)
  ret void
}}
"#
    );
    emit_avr_libgcc_arith_shims_ir(out);
}

/// Emit per-section weak `avr-libgcc` integer mul/divmod ABI shims (`__mulsi3`,
/// `__udivmodsi4`, `__divmodsi4`, `__udivmodhi4`, `__divmodhi4`, `__udivmodqi4`,
/// `__divmodqi4`, `@__muldi3`) backed by pure SSA `#99` helpers (shared across all AVR targets).
pub(crate) fn emit_avr_libgcc_arith_shims_ir(out: &mut String) {
    let _ = writeln!(
        out,
        r#"; ── Freestanding AVR libgcc Arithmetic ABI Shims (Garbage-Collected per Symbol) ──
module asm ".section .text.__mulsi3, \22ax\22, @progbits"
module asm ".weak __mulsi3"
module asm ".type __mulsi3, @function"
module asm "__mulsi3:"
module asm "  push r0"
module asm "  push r18"
module asm "  push r19"
module asm "  push r20"
module asm "  push r21"
module asm "  push r26"
module asm "  push r27"
module asm "  push r30"
module asm "  push r31"
module asm "  call __syn_avr_mul32"
module asm "  pop r31"
module asm "  pop r30"
module asm "  pop r27"
module asm "  pop r26"
module asm "  pop r21"
module asm "  pop r20"
module asm "  pop r19"
module asm "  pop r18"
module asm "  pop r0"
module asm "  clr r1"
module asm "  ret"

module asm ".section .text.__udivmodsi4, \22ax\22, @progbits"
module asm ".weak __udivmodsi4"
module asm ".type __udivmodsi4, @function"
module asm "__udivmodsi4:"
module asm "  push r0"
module asm "  push r26"
module asm "  push r27"
module asm "  push r30"
module asm "  push r31"
module asm "  call __syn_avr_udivrem32"
module asm "  pop r31"
module asm "  pop r30"
module asm "  pop r27"
module asm "  pop r26"
module asm "  pop r0"
module asm "  clr r1"
module asm "  ret"

module asm ".section .text.__divmodsi4, \22ax\22, @progbits"
module asm ".weak __divmodsi4"
module asm ".type __divmodsi4, @function"
module asm "__divmodsi4:"
module asm "  push r0"
module asm "  push r26"
module asm "  push r27"
module asm "  push r30"
module asm "  push r31"
module asm "  call __syn_avr_sdivrem32"
module asm "  pop r31"
module asm "  pop r30"
module asm "  pop r27"
module asm "  pop r26"
module asm "  pop r0"
module asm "  clr r1"
module asm "  ret"

module asm ".section .text.__udivmodhi4, \22ax\22, @progbits"
module asm ".weak __udivmodhi4"
module asm ".type __udivmodhi4, @function"
module asm "__udivmodhi4:"
module asm "  push r0"
module asm "  push r18"
module asm "  push r19"
module asm "  push r20"
module asm "  push r21"
module asm "  push r26"
module asm "  push r27"
module asm "  push r30"
module asm "  push r31"
module asm "  call __syn_avr_udivrem16"
module asm "  pop r31"
module asm "  pop r30"
module asm "  pop r27"
module asm "  pop r26"
module asm "  pop r21"
module asm "  pop r20"
module asm "  pop r19"
module asm "  pop r18"
module asm "  pop r0"
module asm "  clr r1"
module asm "  ret"

module asm ".section .text.__divmodhi4, \22ax\22, @progbits"
module asm ".weak __divmodhi4"
module asm ".type __divmodhi4, @function"
module asm "__divmodhi4:"
module asm "  push r0"
module asm "  push r18"
module asm "  push r19"
module asm "  push r20"
module asm "  push r21"
module asm "  push r26"
module asm "  push r27"
module asm "  push r30"
module asm "  push r31"
module asm "  call __syn_avr_sdivrem16"
module asm "  pop r31"
module asm "  pop r30"
module asm "  pop r27"
module asm "  pop r26"
module asm "  pop r21"
module asm "  pop r20"
module asm "  pop r19"
module asm "  pop r18"
module asm "  pop r0"
module asm "  clr r1"
module asm "  ret"

module asm ".section .text.__udivmodqi4, \22ax\22, @progbits"
module asm ".weak __udivmodqi4"
module asm ".type __udivmodqi4, @function"
module asm "__udivmodqi4:"
module asm "  push r0"
module asm "  push r18"
module asm "  push r19"
module asm "  push r20"
module asm "  push r21"
module asm "  push r22"
module asm "  push r23"
module asm "  push r26"
module asm "  push r27"
module asm "  push r30"
module asm "  push r31"
module asm "  call __syn_avr_udivrem8"
module asm "  pop r31"
module asm "  pop r30"
module asm "  pop r27"
module asm "  pop r26"
module asm "  pop r23"
module asm "  pop r22"
module asm "  pop r21"
module asm "  pop r20"
module asm "  pop r19"
module asm "  pop r18"
module asm "  pop r0"
module asm "  clr r1"
module asm "  ret"

module asm ".section .text.__divmodqi4, \22ax\22, @progbits"
module asm ".weak __divmodqi4"
module asm ".type __divmodqi4, @function"
module asm "__divmodqi4:"
module asm "  push r0"
module asm "  push r18"
module asm "  push r19"
module asm "  push r20"
module asm "  push r21"
module asm "  push r22"
module asm "  push r23"
module asm "  push r26"
module asm "  push r27"
module asm "  push r30"
module asm "  push r31"
module asm "  call __syn_avr_sdivrem8"
module asm "  pop r31"
module asm "  pop r30"
module asm "  pop r27"
module asm "  pop r26"
module asm "  pop r23"
module asm "  pop r22"
module asm "  pop r21"
module asm "  pop r20"
module asm "  pop r19"
module asm "  pop r18"
module asm "  pop r0"
module asm "  clr r1"
module asm "  ret"

define weak i32 @__syn_avr_mul32(i32 %a, i32 %b) #99 {{
entry:
  br label %loop
loop:
  %step = phi i8 [ 0, %entry ], [ %sn, %loop ]
  %aa = phi i32 [ %a, %entry ], [ %a_sh, %loop ]
  %bb = phi i32 [ %b, %entry ], [ %b_sh, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc_next, %loop ]
  %lsb = and i32 %bb, 1
  %odd = icmp ne i32 %lsb, 0
  %sum = add i32 %acc, %aa
  %acc_next = select i1 %odd, i32 %sum, i32 %acc
  %a_sh = shl i32 %aa, 1
  %b_sh = lshr i32 %bb, 1
  %sn = add i8 %step, 1
  %more = icmp ult i8 %sn, 32
  %nz = icmp ne i32 %b_sh, 0
  %cont = and i1 %more, %nz
  br i1 %cont, label %loop, label %done
done:
  ret i32 %acc_next
}}

define weak i64 @__muldi3(i64 %a, i64 %b) #99 {{
entry:
  br label %loop
loop:
  %step = phi i8 [ 0, %entry ], [ %sn, %loop ]
  %aa = phi i64 [ %a, %entry ], [ %a_sh, %loop ]
  %bb = phi i64 [ %b, %entry ], [ %b_sh, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc_next, %loop ]
  %lsb = and i64 %bb, 1
  %odd = icmp ne i64 %lsb, 0
  %sum = add i64 %acc, %aa
  %acc_next = select i1 %odd, i64 %sum, i64 %acc
  %a_sh = shl i64 %aa, 1
  %b_sh = lshr i64 %bb, 1
  %sn = add i8 %step, 1
  %more = icmp ult i8 %sn, 64
  %nz = icmp ne i64 %b_sh, 0
  %cont = and i1 %more, %nz
  br i1 %cont, label %loop, label %done
done:
  ret i64 %acc_next
}}

define weak {{ i32, i32 }} @__syn_avr_udivrem32(i32 %n, i32 %d) #99 {{
entry:
  %dz = icmp eq i32 %d, 0
  br i1 %dz, label %zero, label %loop
zero:
  ret {{ i32, i32 }} {{ i32 -1, i32 -1 }}
loop:
  %step = phi i8 [ 0, %entry ], [ %sn, %loop ]
  %nn = phi i32 [ %n, %entry ], [ %nn_sh, %loop ]
  %q = phi i32 [ 0, %entry ], [ %q_next, %loop ]
  %r = phi i32 [ 0, %entry ], [ %r_next, %loop ]
  %top = lshr i32 %nn, 31
  %r_sh = shl i32 %r, 1
  %r_cur = or i32 %r_sh, %top
  %ge = icmp uge i32 %r_cur, %d
  %r_sub = sub i32 %r_cur, %d
  %r_next = select i1 %ge, i32 %r_sub, i32 %r_cur
  %q_sh = shl i32 %q, 1
  %q_or = or i32 %q_sh, 1
  %q_next = select i1 %ge, i32 %q_or, i32 %q_sh
  %nn_sh = shl i32 %nn, 1
  %sn = add i8 %step, 1
  %cont = icmp ult i8 %sn, 32
  br i1 %cont, label %loop, label %done
done:
  %p0 = insertvalue {{ i32, i32 }} poison, i32 %q_next, 0
  %p1 = insertvalue {{ i32, i32 }} %p0, i32 %r_next, 1
  ret {{ i32, i32 }} %p1
}}

define weak {{ i32, i32 }} @__syn_avr_sdivrem32(i32 %a, i32 %b) #99 {{
entry:
  %sa = icmp slt i32 %a, 0
  %sb = icmp slt i32 %b, 0
  %na = sub i32 0, %a
  %nb = sub i32 0, %b
  %ua = select i1 %sa, i32 %na, i32 %a
  %ub = select i1 %sb, i32 %nb, i32 %b
  %res = call {{ i32, i32 }} @__syn_avr_udivrem32(i32 %ua, i32 %ub)
  %uq = extractvalue {{ i32, i32 }} %res, 0
  %ur = extractvalue {{ i32, i32 }} %res, 1
  %neg_q = xor i1 %sa, %sb
  %nq = sub i32 0, %uq
  %q = select i1 %neg_q, i32 %nq, i32 %uq
  %nr = sub i32 0, %ur
  %r = select i1 %sa, i32 %nr, i32 %ur
  %p0 = insertvalue {{ i32, i32 }} poison, i32 %q, 0
  %p1 = insertvalue {{ i32, i32 }} %p0, i32 %r, 1
  ret {{ i32, i32 }} %p1
}}

define weak {{ i16, i16 }} @__syn_avr_udivrem16(i16 %n, i16 %d) #99 {{
entry:
  %dz = icmp eq i16 %d, 0
  br i1 %dz, label %zero, label %loop
zero:
  ret {{ i16, i16 }} {{ i16 -1, i16 -1 }}
loop:
  %step = phi i8 [ 0, %entry ], [ %sn, %loop ]
  %nn = phi i16 [ %n, %entry ], [ %nn_sh, %loop ]
  %q = phi i16 [ 0, %entry ], [ %q_next, %loop ]
  %r = phi i16 [ 0, %entry ], [ %r_next, %loop ]
  %top = lshr i16 %nn, 15
  %r_sh = shl i16 %r, 1
  %r_cur = or i16 %r_sh, %top
  %ge = icmp uge i16 %r_cur, %d
  %r_sub = sub i16 %r_cur, %d
  %r_next = select i1 %ge, i16 %r_sub, i16 %r_cur
  %q_sh = shl i16 %q, 1
  %q_or = or i16 %q_sh, 1
  %q_next = select i1 %ge, i16 %q_or, i16 %q_sh
  %nn_sh = shl i16 %nn, 1
  %sn = add i8 %step, 1
  %cont = icmp ult i8 %sn, 16
  br i1 %cont, label %loop, label %done
done:
  %p0 = insertvalue {{ i16, i16 }} poison, i16 %q_next, 0
  %p1 = insertvalue {{ i16, i16 }} %p0, i16 %r_next, 1
  ret {{ i16, i16 }} %p1
}}

define weak {{ i16, i16 }} @__syn_avr_sdivrem16(i16 %a, i16 %b) #99 {{
entry:
  %sa = icmp slt i16 %a, 0
  %sb = icmp slt i16 %b, 0
  %na = sub i16 0, %a
  %nb = sub i16 0, %b
  %ua = select i1 %sa, i16 %na, i16 %a
  %ub = select i1 %sb, i16 %nb, i16 %b
  %res = call {{ i16, i16 }} @__syn_avr_udivrem16(i16 %ua, i16 %ub)
  %uq = extractvalue {{ i16, i16 }} %res, 0
  %ur = extractvalue {{ i16, i16 }} %res, 1
  %neg_q = xor i1 %sa, %sb
  %nq = sub i16 0, %uq
  %q = select i1 %neg_q, i16 %nq, i16 %uq
  %nr = sub i16 0, %ur
  %r = select i1 %sa, i16 %nr, i16 %ur
  %p0 = insertvalue {{ i16, i16 }} poison, i16 %q, 0
  %p1 = insertvalue {{ i16, i16 }} %p0, i16 %r, 1
  ret {{ i16, i16 }} %p1
}}

define weak {{ i8, i8 }} @__syn_avr_udivrem8(i8 %n, i8 %d) #99 {{
entry:
  %dz = icmp eq i8 %d, 0
  br i1 %dz, label %zero, label %loop
zero:
  ret {{ i8, i8 }} {{ i8 -1, i8 -1 }}
loop:
  %step = phi i8 [ 0, %entry ], [ %sn, %loop ]
  %nn = phi i8 [ %n, %entry ], [ %nn_sh, %loop ]
  %q = phi i8 [ 0, %entry ], [ %q_next, %loop ]
  %r = phi i8 [ 0, %entry ], [ %r_next, %loop ]
  %top = lshr i8 %nn, 7
  %r_sh = shl i8 %r, 1
  %r_cur = or i8 %r_sh, %top
  %ge = icmp uge i8 %r_cur, %d
  %r_sub = sub i8 %r_cur, %d
  %r_next = select i1 %ge, i8 %r_sub, i8 %r_cur
  %q_sh = shl i8 %q, 1
  %q_or = or i8 %q_sh, 1
  %q_next = select i1 %ge, i8 %q_or, i8 %q_sh
  %nn_sh = shl i8 %nn, 1
  %sn = add i8 %step, 1
  %cont = icmp ult i8 %sn, 8
  br i1 %cont, label %loop, label %done
done:
  %p0 = insertvalue {{ i8, i8 }} poison, i8 %q_next, 0
  %p1 = insertvalue {{ i8, i8 }} %p0, i8 %r_next, 1
  ret {{ i8, i8 }} %p1
}}

define weak {{ i8, i8 }} @__syn_avr_sdivrem8(i8 %a, i8 %b) #99 {{
entry:
  %sa = icmp slt i8 %a, 0
  %sb = icmp slt i8 %b, 0
  %na = sub i8 0, %a
  %nb = sub i8 0, %b
  %ua = select i1 %sa, i8 %na, i8 %a
  %ub = select i1 %sb, i8 %nb, i8 %b
  %res = call {{ i8, i8 }} @__syn_avr_udivrem8(i8 %ua, i8 %ub)
  %uq = extractvalue {{ i8, i8 }} %res, 0
  %ur = extractvalue {{ i8, i8 }} %res, 1
  %neg_q = xor i1 %sa, %sb
  %nq = sub i8 0, %uq
  %q = select i1 %neg_q, i8 %nq, i8 %uq
  %nr = sub i8 0, %ur
  %r = select i1 %sa, i8 %nr, i8 %ur
  %p0 = insertvalue {{ i8, i8 }} poison, i8 %q, 0
  %p1 = insertvalue {{ i8, i8 }} %p0, i8 %r, 1
  ret {{ i8, i8 }} %p1
}}
"#
    );
}

fn insert_fn_addrspace1(line: &str) -> Option<String> {
    let at = line.find('@')?;
    let rel_open = line[at..].find('(')?;
    let open = at + rel_open;
    let mut depth = 0usize;
    let mut close = None;
    for (off, ch) in line[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + off);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let mut out = String::with_capacity(line.len() + 14);
    out.push_str(&line[..=close]);
    out.push_str(" addrspace(1)");
    out.push_str(&line[close + 1..]);
    Some(out)
}

/// Annotate all AVR program-memory functions, declarations, and non-intrinsic/non-asm
/// call sites with `addrspace(1)` required by LLVM's AVR `e-P1-...` data layout.
pub fn apply_avr_program_addrspace(ll: &str) -> String {
    let mut out = String::with_capacity(ll.len() + 4096);
    for line in ll.lines() {
        if ((line.starts_with("define ") && line.ends_with('{')) || line.starts_with("declare "))
            && !line.contains("@llvm.")
            && !line.contains(") addrspace(1)")
        {
            if let Some(rewritten) = insert_fn_addrspace1(line) {
                out.push_str(&rewritten);
                out.push('\n');
                continue;
            }
        } else if line.starts_with("  ") {
            let trimmed = line.trim_start();
            if !trimmed.starts_with(';')
                && (trimmed.starts_with("call ") || line.contains(" = call "))
                && !line.contains(" asm ")
                && !line.contains("@llvm.")
                && !line.contains("call addrspace(1) ")
            {
                if let Some(pos) = line.find("call ") {
                    out.push_str(&line[..pos]);
                    out.push_str("call addrspace(1) ");
                    out.push_str(&line[pos + 5..]);
                    out.push('\n');
                    continue;
                }
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}
