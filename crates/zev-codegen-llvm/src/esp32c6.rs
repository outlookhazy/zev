//! Bare-Metal ESP32-C6 (`riscv32imac`) Direct-Boot Startup, Watchdog Disarm,
//! 16 MHz `SYSTIMER`, `USB_DEVICE` (`/dev/ttyACM0`) Serial Console, and HP-SRAM
//! Hardware-Calibrated `GPIO8` WS2812B RGB LED + `GPIO9` `BOOT` Button HAL in Pure LLVM 23 IR.
//!
//! Hardware register addresses & bitmasks verified against Espressif ESP32-C6 TRM / SVD:
//! - Direct-Boot Flash MMU Cache: `0x4200_0000` (magic `0xAEDB041D` x2 at `0x0`, `@_start` at `0x4200_0008`)
//! - HP-SRAM: `0x4080_0000 .. 0x4088_0000` (initial `sp = 0x4087_0000`, `.iram1` + `.data` + `.bss`)
//! - `TIMG0` (`0x6000_8000`) & `TIMG1` (`0x6000_9000`): `WDTCONFIG0` `+0x48`, `WDTWPROTECT` `+0x64` (`0x50D8_3AA1`)
//! - `LP_WDT` (`0x600B_1C00`): `CONFIG0` `+0x00`, `WPROTECT` `+0x18`, `SWD_CONFIG` `+0x1C` (`BIT(31)|BIT(30)`), `SWD_WPROTECT` `+0x20`
//! - `PCR` (`0x6009_6000`): `PCR_SYSTIMER_CONF_REG` `+0x54`
//! - `SYSTIMER` (`0x6000_A000`, 16 MHz = `16_000` ticks/ms): `CONF` `+0x00`, `UNIT0_OP` `+0x04` (`UPDATE` bit 30, `VALID` bit 29), `VALUE_HI` `+0x40`, `VALUE_LO` `+0x44`
//! - `USB_DEVICE` (`0x6000_F000`): `EP1` `+0x00`, `EP1_CONF` `+0x04` (`WR_DONE` bit 0, `SERIAL_IN_EP_DATA_FREE` bit 1)
//! - `IO_MUX` (`0x6009_0000`): `GPIO8` `+0x24`, `GPIO9` `+0x28` (`MCU_SEL=1` at `[14:12]`, `FUN_IE` bit 9, `FUN_WPU` bit 8)
//! - `GPIO` (`0x6009_1000`): `OUT_W1TS` `+0x08`, `OUT_W1TC` `+0x0C`, `ENABLE_W1TS` `+0x24`, `ENABLE_W1TC` `+0x28`, `IN` `+0x3C`, `FUNC8_OUT_SEL_CFG` `+0x574` (`SIG_GPIO_OUT_IDX = 256`)

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

/// Emit ESP32-C6 `.entry` (`@_start` at `0x4200_0008`), `@__syn_esp32c6_reset`, watchdog disarm,
/// `.data`/`.iram1` copy + `fence.i`, `.bss` zeroing, `SYSTIMER` init, and `GPIO8`/`GPIO9` +
/// APB `store_ns` / `.iram1` cycle calibration (plus `INTMTX`/`PLIC_MX`/`mie`/vectored `mtvec` when `routes` is non-empty).
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

@__syn_ws2812_t0h = global i32 0, align 4
@__syn_ws2812_t0l = global i32 4, align 4
@__syn_ws2812_t1h = global i32 4, align 4
@__syn_ws2812_t1l = global i32 2, align 4
@__syn_ws2812_store_ns = global i32 200, align 4
@__syn_cpu_calib_dt = global i32 38400, align 4
@__syn_pad8_ok = global i32 0, align 4

define void @__syn_esp32c6_disarm_watchdogs() nounwind{sz} {{
entry:
  ; 1. TIMG0 MWDT0 (base 0x60008000 = 1610645504): WDTWPROTECT (+0x64 = 1610645604), WDTCONFIG0 (+0x48 = 1610645576)
  ;    Unlock key 0x50D83AA1 = 1356348065
  store volatile i32 1356348065, ptr inttoptr (i32 1610645604 to ptr), align 4
  store volatile i32 0, ptr inttoptr (i32 1610645576 to ptr), align 4
  store volatile i32 0, ptr inttoptr (i32 1610645604 to ptr), align 4

  ; 2. TIMG1 MWDT1 (base 0x60009000 = 1610649600): WDTWPROTECT (+0x64 = 1610649700), WDTCONFIG0 (+0x48 = 1610649672)
  store volatile i32 1356348065, ptr inttoptr (i32 1610649700 to ptr), align 4
  store volatile i32 0, ptr inttoptr (i32 1610649672 to ptr), align 4
  store volatile i32 0, ptr inttoptr (i32 1610649700 to ptr), align 4

  ; 3. LP_WDT RWDT (base 0x600B1C00 = 1611340800): WPROTECT (+0x18 = 1611340824), CONFIG0 (+0x00 = 1611340800)
  store volatile i32 1356348065, ptr inttoptr (i32 1611340824 to ptr), align 4
  store volatile i32 0, ptr inttoptr (i32 1611340800 to ptr), align 4
  store volatile i32 0, ptr inttoptr (i32 1611340824 to ptr), align 4

  ; 4. LP_WDT Super Watchdog SWD: SWD_WPROTECT (+0x20 = 1611340832), SWD_CONFIG (+0x1C = 1611340828)
  ;    Set BIT(31) SWD_AUTO_FEED_EN | BIT(30) SWD_DISABLE = 0xC0000000 (-1073741824)
  store volatile i32 1356348065, ptr inttoptr (i32 1611340832 to ptr), align 4
  %swd_old = load volatile i32, ptr inttoptr (i32 1611340828 to ptr), align 4
  %swd_new = or i32 %swd_old, -1073741824
  store volatile i32 %swd_new, ptr inttoptr (i32 1611340828 to ptr), align 4
  store volatile i32 0, ptr inttoptr (i32 1611340832 to ptr), align 4
  ret void
}}

define void @__syn_esp32c6_hw_init() nounwind{sz} {{
entry:
  ; 1. Enable SYSTIMER clock gate in PCR_SYSTIMER_CONF_REG (0x60096054 = 1611227220): set bit 0, clear rst bit 1
  %pcr = load volatile i32, ptr inttoptr (i32 1611227220 to ptr), align 4
  %pcr1 = or i32 %pcr, 1
  %pcr2 = and i32 %pcr1, -3
  store volatile i32 %pcr2, ptr inttoptr (i32 1611227220 to ptr), align 4

  ; 2. Enable SYSTIMER_CLK_EN (bit 31) | SYSTIMER_TIMER_UNIT0_WORK_EN (bit 30) = 0xC0000000 at 0x6000A000 (1610653696)
  %st_old = load volatile i32, ptr inttoptr (i32 1610653696 to ptr), align 4
  %st_new = or i32 %st_old, -1073741824
  store volatile i32 %st_new, ptr inttoptr (i32 1610653696 to ptr), align 4
  store i8 1, ptr @__syn_systick_enabled, align 1

  ; 3. Configure GPIO8 (Waveshare ESP32-C6-Pico WS2812B RGB LED):
  ;    - IO_MUX_GPIO8_REG (0x60090024 = 1611202596) = (1 << 12) | (2 << 10) | (1 << 9) = 0x1A00 (6656: FUNC_GPIO, 20mA, FUN_IE)
  ;    - GPIO_FUNC8_OUT_SEL_CFG_REG (0x60091574 = 1611208052) = 128 | (1 << 9) = 0x280 (640: OUT_SEL=128 GPIO_OUT_REG + OEN_SEL=1)
  ;    - GPIO_OUT_W1TC_REG (0x6009100C = 1611206668) = (1 << 8) = 256 (idle LOW)
  ;    - GPIO_ENABLE_W1TS_REG (0x60091024 = 1611206692) = (1 << 8) = 256
  store volatile i32 6656, ptr inttoptr (i32 1611202596 to ptr), align 4
  store volatile i32 640, ptr inttoptr (i32 1611208052 to ptr), align 4
  store volatile i32 256, ptr inttoptr (i32 1611206668 to ptr), align 4
  store volatile i32 256, ptr inttoptr (i32 1611206692 to ptr), align 4

  ; 3b. Physical pad8 hardware loopback self-check (drive HIGH -> read IN[8]==1, drive LOW -> read IN[8]==0):
  store volatile i32 256, ptr inttoptr (i32 1611206664 to ptr), align 4
  call void asm sideeffect "nop\0Anop\0Anop\0Anop", "~{{memory}}"()
  %in_hi_reg = load volatile i32, ptr inttoptr (i32 1611206716 to ptr), align 4
  store volatile i32 256, ptr inttoptr (i32 1611206668 to ptr), align 4
  call void asm sideeffect "nop\0Anop\0Anop\0Anop", "~{{memory}}"()
  %in_lo_reg = load volatile i32, ptr inttoptr (i32 1611206716 to ptr), align 4
  %hi_b = and i32 %in_hi_reg, 256
  %lo_b = and i32 %in_lo_reg, 256
  %hi_ok = icmp ne i32 %hi_b, 0
  %lo_ok = icmp eq i32 %lo_b, 0
  %pad_ok_b = and i1 %hi_ok, %lo_ok
  %pad_ok_v = zext i1 %pad_ok_b to i32
  store i32 %pad_ok_v, ptr @__syn_pad8_ok, align 4

  ; 4. Configure GPIO9 (BOOT button, active-LOW):
  ;    - GPIO_ENABLE_W1TC_REG (0x60091028 = 1611206696) = (1 << 9) = 512 (input mode)
  ;    - IO_MUX_GPIO9_REG (0x60090028 = 1611202600) = (1 << 12) | (1 << 9) | (1 << 8) = 0x1300 (4864: FUNC_GPIO + FUN_IE + FUN_WPU)
  store volatile i32 512, ptr inttoptr (i32 1611206696 to ptr), align 4
  store volatile i32 4864, ptr inttoptr (i32 1611202600 to ptr), align 4

  ; 5a. Measure exact APB peripheral store latency `store_ns`:
  ;     100 loops x 16 unrolled `sw zero, 8/12(gpio)` = 1,600 back-to-back APB stores!
  ;     At 16 MHz SYSTIMER (62.5 ns/tick), `store_ns = (dt_sw * 5) / 128`.
  %s0 = call i32 @__syn_esp32c6_systimer_lo()
  call void @__syn_esp32c6_calib_stores(i32 100)
  %s1 = call i32 @__syn_esp32c6_systimer_lo()
  %raw_sw = sub i32 %s1, %s0
  %sw_ok = icmp sgt i32 %raw_sw, 400
  %dt_sw = select i1 %sw_ok, i32 %raw_sw, i32 5120
  %sw_mul5 = mul i32 %dt_sw, 5
  %store_ns = udiv i32 %sw_mul5, 128
  store i32 %store_ns, ptr @__syn_ws2812_store_ns, align 4

  ; 5b. Measure exact .iram1 `addi + bnez` loop speed (`16,000` iterations -> `iter_ns = dt_loop / 256`):
  %l0 = call i32 @__syn_esp32c6_systimer_lo()
  call void @__syn_esp32c6_iram_delay(i32 16000)
  %l1 = call i32 @__syn_esp32c6_systimer_lo()
  %raw_dt = sub i32 %l1, %l0
  %valid_dt = icmp sgt i32 %raw_dt, 1000
  %dt = select i1 %valid_dt, i32 %raw_dt, i32 38400
  store i32 %dt, ptr @__syn_cpu_calib_dt, align 4
  %half_dt = udiv i32 %dt, 2

  ; 5c. Compute WS2812B pulse loop counts subtracting actual `store_ns` + branch overhead!
  ;     - T0H target = 300ns (<400ns! overhead = store_ns + 75ns; 0 loops if overhead >= 260ns!)
  %ov0 = add i32 %store_ns, 75
  %need_t0h = icmp slt i32 %ov0, 260
  %rem_t0h_raw = sub i32 300, %ov0
  %rem_t0h = select i1 %need_t0h, i32 %rem_t0h_raw, i32 0
  %num_t0h = mul i32 %rem_t0h, 256
  %t0h_calc = udiv i32 %num_t0h, %dt
  %t0h = select i1 %need_t0h, i32 %t0h_calc, i32 0
  store i32 %t0h, ptr @__syn_ws2812_t0h, align 4

  ;     - T0L target = 900ns (overhead = store_ns + 220ns)
  %ov_l = add i32 %store_ns, 220
  %rem_t0l_s = sub i32 900, %ov_l
  %pos_t0l = icmp sgt i32 %rem_t0l_s, 150
  %rem_t0l = select i1 %pos_t0l, i32 %rem_t0l_s, i32 150
  %num_t0l = mul i32 %rem_t0l, 256
  %num_t0l_r = add i32 %num_t0l, %half_dt
  %c_t0l = udiv i32 %num_t0l_r, %dt
  %ok_t0l = icmp ugt i32 %c_t0l, 1
  %t0l = select i1 %ok_t0l, i32 %c_t0l, i32 1
  store i32 %t0l, ptr @__syn_ws2812_t0l, align 4

  ;     - T1H target = 800ns (overhead = store_ns + 60ns)
  %ov1 = add i32 %store_ns, 60
  %rem_t1h_s = sub i32 800, %ov1
  %pos_t1h = icmp sgt i32 %rem_t1h_s, 300
  %rem_t1h = select i1 %pos_t1h, i32 %rem_t1h_s, i32 300
  %num_t1h = mul i32 %rem_t1h, 256
  %num_t1h_r = add i32 %num_t1h, %half_dt
  %c_t1h = udiv i32 %num_t1h_r, %dt
  %ok_t1h = icmp ugt i32 %c_t1h, 2
  %t1h = select i1 %ok_t1h, i32 %c_t1h, i32 2
  store i32 %t1h, ptr @__syn_ws2812_t1h, align 4

  ;     - T1L target = 500ns (overhead = store_ns + 220ns)
  %rem_t1l_s = sub i32 500, %ov_l
  %pos_t1l = icmp sgt i32 %rem_t1l_s, 150
  %rem_t1l = select i1 %pos_t1l, i32 %rem_t1l_s, i32 150
  %num_t1l = mul i32 %rem_t1l, 256
  %num_t1l_r = add i32 %num_t1l, %half_dt
  %c_t1l = udiv i32 %num_t1l_r, %dt
  %ok_t1l = icmp ugt i32 %c_t1l, 1
  %t1l = select i1 %ok_t1l, i32 %c_t1l, i32 1
  store i32 %t1l, ptr @__syn_ws2812_t1l, align 4

  ; Hold GPIO8 LOW >= 100us (1,600 SYSTIMER ticks) so WS2812B resets cleanly before main()
  %r0 = call i32 @__syn_esp32c6_systimer_lo()
  br label %rst_wait
rst_wait:
  %r1 = call i32 @__syn_esp32c6_systimer_lo()
  %rel = sub i32 %r1, %r0
  %rdone = icmp uge i32 %rel, 1600
  br i1 %rdone, label %init_done, label %rst_wait
init_done:
  ret void
}}

define void @__syn_esp32c6_reset() #98 {{
entry:
  ; 1. Immediately disarm TIMG0, TIMG1, LP_WDT, and SWD watchdogs
  call void @__syn_esp32c6_disarm_watchdogs()

  ; 2. Copy .iram1 + .data from Flash LMA (__sidata) to HP-SRAM VMA [__sdata .. __edata)
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
  call void @__syn_esp32c6_hw_init()
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

/// Emit ESP32-C6 (`rv32imac`) hardware runtime primitives (`mstatus` critical sections,
/// 16 MHz `SYSTIMER` `now_ms`, `USB_DEVICE` `/dev/ttyACM0` serial console, single-block `.iram1`
/// WS2812B assembly shifter on `GPIO8`, and `GPIO` input/output helpers).
pub fn emit_esp32c6_hal_ir(out: &mut String, sz: &str) {
    let _ = writeln!(
        out,
        r#"; ── ESP32-C6 (rv32imac) Hardware Primitives: SYSTIMER, USB_DEVICE, GPIO, WS2812B ──
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

define {{ i32, i32 }} @__syn_esp32c6_systimer_snapshot() nounwind{sz} {{
entry:
  ; Latch SYSTIMER UNIT0: write BIT(30) = 0x40000000 (1073741824) to UNIT0_OP (0x6000A004 = 1610653700)
  ; Allow 4 cycles for clock-domain crossing so stale VALUE_VALID is cleared before polling!
  store volatile i32 1073741824, ptr inttoptr (i32 1610653700 to ptr), align 4
  call void asm sideeffect "nop\0Anop\0Anop\0Anop", "~{{memory}}"()
  br label %poll
poll:
  %tries = phi i32 [ 0, %entry ], [ %t1, %poll_next ]
  %op = load volatile i32, ptr inttoptr (i32 1610653700 to ptr), align 4
  ; Poll BIT(29) VALUE_VALID = 0x20000000 (536870912)
  %valid_bits = and i32 %op, 536870912
  %is_valid = icmp ne i32 %valid_bits, 0
  %timed_out = icmp sge i32 %tries, 256
  %done = or i1 %is_valid, %timed_out
  br i1 %done, label %read_val, label %poll_next
poll_next:
  %t1 = add i32 %tries, 1
  br label %poll
read_val:
  ; UNIT0_VALUE_HI (0x6000A040 = 1610653760), UNIT0_VALUE_LO (0x6000A044 = 1610653764)
  %hi_raw = load volatile i32, ptr inttoptr (i32 1610653760 to ptr), align 4
  %lo = load volatile i32, ptr inttoptr (i32 1610653764 to ptr), align 4
  %hi = and i32 %hi_raw, 1048575
  %r0 = insertvalue {{ i32, i32 }} poison, i32 %hi, 0
  %r1 = insertvalue {{ i32, i32 }} %r0, i32 %lo, 1
  ret {{ i32, i32 }} %r1
}}

define i32 @__syn_esp32c6_systimer_lo() nounwind{sz} {{
entry:
  %snap = call {{ i32, i32 }} @__syn_esp32c6_systimer_snapshot()
  %lo = extractvalue {{ i32, i32 }} %snap, 1
  ret i32 %lo
}}

define i32 @__syn_esp32c6_read_ms() nounwind noinline{sz} {{
entry:
  ; Exact 52-bit SYSTIMER (16 MHz -> 16,000 ticks/ms = 128 * 125) division using pure 32-bit RV32M `divu`/`remu`!
  %snap = call {{ i32, i32 }} @__syn_esp32c6_systimer_snapshot()
  %hi = extractvalue {{ i32, i32 }} %snap, 0
  %lo = extractvalue {{ i32, i32 }} %snap, 1
  %h_sh = lshr i32 %hi, 7
  %lo_sh = lshr i32 %lo, 7
  %hi_low7 = shl i32 %hi, 25
  %l_comb = or i32 %lo_sh, %hi_low7
  %l_hi = lshr i32 %l_comb, 16
  %l_lo = and i32 %l_comb, 65535
  %h_top = shl i32 %h_sh, 16
  %n0 = or i32 %h_top, %l_hi
  %q_hi = udiv i32 %n0, 125
  %r0 = urem i32 %n0, 125
  %r0_sh = shl i32 %r0, 16
  %n1 = or i32 %r0_sh, %l_lo
  %q_lo = udiv i32 %n1, 125
  %q_hi_sh = shl i32 %q_hi, 16
  %ms = add i32 %q_hi_sh, %q_lo
  store volatile i32 %ms, ptr @__syn_tick_ms, align 4
  ret i32 %ms
}}

define void @__syn_uart_putc(i8 %c) nounwind noinline{sz} {{
entry:
  ; Built-in USB_DEVICE Serial/JTAG (0x6000F000 = 1610674176, EP1_CONF 0x6000F004 = 1610674180)
  ; Wait up to ~4000 polls for SERIAL_IN_EP_DATA_FREE (bit 1 = 2); drop byte if host monitor disconnected!
  br label %wait_fifo
wait_fifo:
  %tries = phi i32 [ 0, %entry ], [ %t1, %wait_next ]
  %conf = load volatile i32, ptr inttoptr (i32 1610674180 to ptr), align 4
  %free_bit = and i32 %conf, 2
  %is_free = icmp ne i32 %free_bit, 0
  br i1 %is_free, label %write_byte, label %chk_timeout
chk_timeout:
  %expired = icmp sge i32 %tries, 4000
  br i1 %expired, label %done, label %wait_next
wait_next:
  %t1 = add i32 %tries, 1
  br label %wait_fifo

write_byte:
  %z = zext i8 %c to i32
  store volatile i32 %z, ptr inttoptr (i32 1610674176 to ptr), align 4
  %is_nl = icmp eq i8 %c, 10
  br i1 %is_nl, label %flush_pkt, label %done
flush_pkt:
  ; Write WR_DONE (bit 0 = 1) to EP1_CONF so USB host receives every line immediately!
  store volatile i32 1, ptr inttoptr (i32 1610674180 to ptr), align 4
  br label %done
done:
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
  call void @__syn_esp32c6_dispatch_irq(i32 %irq)
  ret void
}}

define void @__syn_idle_sleep() nounwind{sz} {{
entry:
  %locks = load volatile i32, ptr @__syn_sleep_lock_mask, align 4
  %locked = icmp ne i32 %locks, 0
  %depth = select i1 %locked, i32 1, i32 2
  store volatile i32 %depth, ptr @__syn_last_sleep_depth, align 4
  %wc = load volatile i32, ptr @__syn_wfi_count, align 4
  %wc1 = add i32 %wc, 1
  store volatile i32 %wc1, ptr @__syn_wfi_count, align 4
  %ign = call i32 @__syn_esp32c6_read_ms()
  call void @__syn_esp32c6_iram_delay(i32 64)
  ret void
}}

; ── Zero-Wait-State HP-SRAM (.iram1) Calibration & Single-Block WS2812B Shifter ──
define void @__syn_esp32c6_calib_stores(i32 %n) nounwind noinline section ".iram1" {{
entry:
  %gt = icmp sgt i32 %n, 0
  br i1 %gt, label %run, label %done
run:
  ; 16 back-to-back APB stores per loop to GPIO_OUT_W1TS/W1TC (writing 0 = zero pin glitch!)
  %rem = call i32 asm sideeffect "
    mv $0, $2
  1:
    sw zero, 8($1)
    sw zero, 12($1)
    sw zero, 8($1)
    sw zero, 12($1)
    sw zero, 8($1)
    sw zero, 12($1)
    sw zero, 8($1)
    sw zero, 12($1)
    sw zero, 8($1)
    sw zero, 12($1)
    sw zero, 8($1)
    sw zero, 12($1)
    sw zero, 8($1)
    sw zero, 12($1)
    sw zero, 8($1)
    sw zero, 12($1)
    addi $0, $0, -1
    bnez $0, 1b
  ", "=&r,r,r,~{{memory}}"(ptr inttoptr (i32 1611206656 to ptr), i32 %n)
  br label %done
done:
  ret void
}}

define void @__syn_esp32c6_iram_delay(i32 %n) nounwind noinline section ".iram1" {{
entry:
  %gt = icmp sgt i32 %n, 0
  br i1 %gt, label %run, label %done
run:
  %rem = call i32 asm sideeffect "
    mv $0, $1
  1:
    addi $0, $0, -1
    bnez $0, 1b
  ", "=&r,r,~{{memory}}"(i32 %n)
  br label %done
done:
  ret void
}}

define void @__syn_esp32c6_ws2812_grb(i32 %grb24) nounwind noinline section ".iram1" {{
entry:
  %t0h = load i32, ptr @__syn_ws2812_t0h, align 4
  %t0l = load i32, ptr @__syn_ws2812_t0l, align 4
  %t1h = load i32, ptr @__syn_ws2812_t1h, align 4
  %t1l = load i32, ptr @__syn_ws2812_t1l, align 4
  %grb_sh = shl i32 %grb24, 8
  %pm = call i32 @__syn_crit_enter()

  ; Single self-contained RISC-V assembly 24-bit MSB-first WS2812B shifter in 0-wait-state HP-SRAM:
  ;   $0 = bits (MSB in bit 31), $1 = bit_cnt (24..1), $2 = delay scratch (all =&r early-clobber!)
  ;   $3 = GPIO base (0x60091000 = 1611206656), $4 = GPIO8 mask (256)
  ;   $5 = t0h, $6 = t0l, $7 = t1h, $8 = t1l, $9 = grb_sh
  %ign = call {{ i32, i32, i32 }} asm sideeffect "
    mv $0, $9
    li $1, 24
  1:
    bltz $0, 4f
    sw $4, 8($3)
    beqz $5, 3f
    mv $2, $5
  2:
    addi $2, $2, -1
    bnez $2, 2b
  3:
    sw $4, 12($3)
    mv $2, $6
  5:
    addi $2, $2, -1
    bnez $2, 5b
    slli $0, $0, 1
    addi $1, $1, -1
    bnez $1, 1b
    j 8f
  4:
    sw $4, 8($3)
    mv $2, $7
  6:
    addi $2, $2, -1
    bnez $2, 6b
    sw $4, 12($3)
    mv $2, $8
  7:
    addi $2, $2, -1
    bnez $2, 7b
    slli $0, $0, 1
    addi $1, $1, -1
    bnez $1, 1b
  8:
  ", "=&r,=&r,=&r,r,r,r,r,r,r,r,~{{memory}}"(
    ptr inttoptr (i32 1611206656 to ptr),
    i32 256,
    i32 %t0h,
    i32 %t0l,
    i32 %t1h,
    i32 %t1l,
    i32 %grb_sh
  )

  call void @__syn_crit_exit(i32 %pm)
  ; Hold LOW >= 90 us (1,500 ticks at 16 MHz SYSTIMER) to latch WS2812B frame
  %s0 = call i32 @__syn_esp32c6_systimer_lo()
  br label %latch_wait
latch_wait:
  %s1 = call i32 @__syn_esp32c6_systimer_lo()
  %el = sub i32 %s1, %s0
  %l_done = icmp uge i32 %el, 1500
  br i1 %l_done, label %exit, label %latch_wait
exit:
  ret void
}}

define void @__syn_esp32c6_ws2812_rgb(i8 %r, i8 %g, i8 %b) nounwind{sz} {{
entry:
  %g32 = zext i8 %g to i32
  %r32 = zext i8 %r to i32
  %b32 = zext i8 %b to i32
  %g_sh = shl i32 %g32, 16
  %r_sh = shl i32 %r32, 8
  %gr = or i32 %g_sh, %r_sh
  %grb = or i32 %gr, %b32
  call void @__syn_esp32c6_ws2812_grb(i32 %grb)
  ret void
}}
"#
    );
}
