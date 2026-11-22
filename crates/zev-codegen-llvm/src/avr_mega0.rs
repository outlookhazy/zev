//! Pure-ISA Bare-Metal Microchip ATmega4809 (`avrxmega3`, Arduino Nano Every) Startup,
//! 40-Slot `avr_signalcc` Vector Table, `SREG` Critical Sections, `sleep` Idle Hook,
//! Shared `avr-libgcc` Integer Shims, Image Builder, and Pure-Python `MuxTO` (`jtag2updi`)
//! UPDI Flasher.
//!
//! Contains **zero** device peripheral MMIO addresses or hardcoded peripheral IRQ slots:
//! all peripheral initialization (`CPU.CCP`, `WDT`, `CLKCTRL`, `SLPCTRL`, `FUSE.OSCCFG`,
//! `PORTMUX`, `USART3` `console_putc(u8)`, and `TCB3` 1.000 kHz `sys.tick()` ISR) lives
//! 100% in `.zev` (`chip/atmega4809.zev` + `bsp/arduino_nano_every.zev`).

use std::collections::BTreeSet;
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;

use crate::toolchain::{CodegenError, TargetArch, find_tool, inspect_elf_sizes, read_elf32_sym_addr};

/// Total ATmega4809 interrupt vectors (`0..=39`).
pub const AVR_ATMEGA4809_VECTOR_COUNT: u32 = 40;

/// Validate user-attached IRQ vector numbers against ATmega4809 vector slots `1..=39`
/// (excluding only slot `0`, which is the hardware `RESET` vector; all peripheral slots
/// `1..=39` are open to `.zev` applications and BSPs).
pub fn validate_avr_mega0_irq_vectors(
    vectors: &[zev_mir::MirIrqVector],
) -> Result<BTreeSet<u32>, CodegenError> {
    let mut attached = BTreeSet::new();
    for v in vectors {
        if v.irq_num == 0 {
            return Err(CodegenError::new(
                "llvm-ir",
                "ATmega4809 IRQ vector 0 is the RESET vector and cannot be attached",
            ));
        }
        if v.irq_num >= AVR_ATMEGA4809_VECTOR_COUNT {
            return Err(CodegenError::new(
                "llvm-ir",
                format!(
                    "ATmega4809 IRQ vector {} exceeds vector table range (1..={})",
                    v.irq_num,
                    AVR_ATMEGA4809_VECTOR_COUNT - 1
                ),
            ));
        }
        attached.insert(v.irq_num);
    }
    Ok(attached)
}

/// Emit ATmega4809 40-slot `.vectors` table (`jmp`), `.text.startup` stack/data/bss init,
/// `@__syn_avr_mega0_entry`, and per-attached `avr_signalcc` vector wrappers.
pub fn emit_avr_mega0_startup_ir(
    out: &mut String,
    initial_sp: u32,
    sz: &str,
    attached_irqs: &BTreeSet<u32>,
) {
    let fn_as = TargetArch::AvrAtmega4809.fn_addrspace_attr();
    let call_kw = TargetArch::AvrAtmega4809.call_kw();
    let sp_lo = initial_sp & 0xFF;
    let sp_hi = (initial_sp >> 8) & 0xFF;

    let _ = writeln!(
        out,
        "; ── ATmega4809 (avrxmega3) 40-Slot Flash Vector Table (@ 0x0000) ──"
    );
    let _ = writeln!(out, "module asm \".section .vectors, \\22ax\\22, @progbits\"");
    let _ = writeln!(out, "module asm \".global _start\"");
    let _ = writeln!(out, "module asm \"_start:\"");
    let _ = writeln!(out, "module asm \"  jmp __syn_avr_mega0_reset\"");
    for vec_idx in 1..AVR_ATMEGA4809_VECTOR_COUNT {
        if attached_irqs.contains(&vec_idx) {
            let _ = writeln!(out, "module asm \"  jmp __syn_avr_vec_{vec_idx}\"");
        } else {
            let _ = writeln!(out, "module asm \"  jmp __syn_avr_unhandled_irq\"");
        }
    }

    let _ = writeln!(
        out,
        r#"
; ── Reset, Stack Pointer, Flash->SRAM .data/.rodata Copy & .bss Zeroing ──
module asm ".section .text.startup, \22ax\22, @progbits"
module asm ".global __syn_avr_mega0_reset"
module asm ".global __do_copy_data"
module asm ".global __do_clear_bss"
module asm "__syn_avr_mega0_reset:"
module asm "  cli"
module asm "  eor r1, r1"
module asm "  out 0x3f, r1"
module asm "  ldi r28, 0x{sp_lo:02X}"
module asm "  ldi r29, 0x{sp_hi:02X}"
module asm "  out 0x3e, r29"
module asm "  out 0x3d, r28"
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
module asm "  call __syn_avr_mega0_entry"
module asm "5: rjmp 5b"

@__sdata = external global i8
@__edata = external global i8
@__sidata = external global i8
@__sbss = external global i8
@__ebss = external global i8

define void @__syn_avr_mega0_entry(){fn_as} noreturn nounwind{sz} {{
entry:
  {call_kw} void @__syn_boot()
  {call_kw} void @__syn_sched_loop()
  unreachable
}}

define avr_signalcc void @__syn_avr_unhandled_irq(){fn_as} nounwind {{
entry:
  ret void
}}
"#
    );

    for &irq in attached_irqs {
        let _ = writeln!(
            out,
            r#"define avr_signalcc void @__syn_avr_vec_{irq}(){fn_as} nounwind {{
entry:
  {call_kw} void @__syn_vec_irq_{irq}()
  ret void
}}
"#
        );
    }
}

/// Emit pure-ISA `avrxmega3` CPU runtime primitives (`SREG` critical sections, tear-free
/// `@__syn_avr_read_ms`, `sei; sleep; cli` idle hook, and shared per-section weak
/// `avr-libgcc` integer mul/divmod shims).
pub fn emit_avr_mega0_hal_ir(out: &mut String, sz: &str) {
    let fn_as = TargetArch::AvrAtmega4809.fn_addrspace_attr();
    let call_kw = TargetArch::AvrAtmega4809.call_kw();
    let _ = writeln!(
        out,
        r#"; ── AVR avrxmega3 Pure-ISA CPU Primitives: SREG, Tear-Free Tick Read, Sleep ──
define i32 @__syn_crit_enter(){fn_as} nounwind alwaysinline {{
entry:
  %sr = call i8 asm sideeffect "in $0, 0x3f\0Acli", "=r,~{{memory}}"()
  %z = zext i8 %sr to i32
  ret i32 %z
}}

define void @__syn_crit_exit(i32 %pm){fn_as} nounwind alwaysinline {{
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

define i32 @__syn_avr_read_ms(){fn_as} nounwind noinline{sz} {{
entry:
  %pm = {call_kw} i32 @__syn_crit_enter()
  %t = load volatile i32, ptr @__syn_tick_ms, align 4
  {call_kw} void @__syn_crit_exit(i32 %pm)
  ret i32 %t
}}

define void @__syn_platform_exit(i32 %code){fn_as} noreturn nounwind{sz} {{
entry:
  br label %halt
halt:
  call void asm sideeffect "nop", "~{{memory}}"()
  br label %halt
}}

define void @__syn_use_systick(i32 %hz){fn_as} nounwind{sz} {{
entry:
  store i8 1, ptr @__syn_systick_enabled, align 1
  ret void
}}

define void @__syn_nvic_pend_irq(i32 %irq){fn_as} nounwind{sz} {{
entry:
  %pm = {call_kw} i32 @__syn_crit_enter()
  {call_kw} void @__syn_avr_dispatch_irq(i32 %irq)
  {call_kw} void @__syn_crit_exit(i32 %pm)
  ret void
}}

define void @__syn_idle_sleep(){fn_as} nounwind{sz} {{
entry:
  %pm = {call_kw} i32 @__syn_crit_enter()
  %now = load volatile i32, ptr @__syn_tick_ms, align 4
  {call_kw} void @__syn_wake_timers(i32 %now)
  %any_ready = {call_kw} i1 @__syn_any_ready_task()
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
  call void asm sideeffect "sei\0Asleep\0Acli", "~{{memory}}"()
  br label %restore
restore:
  {call_kw} void @__syn_crit_exit(i32 %pm)
  ret void
}}
"#
    );
    crate::avr::emit_avr_libgcc_arith_shims_ir(out);
}

/// Convert a linked ATmega4809 (`avrxmega3`, Arduino Nano Every) `.elf` into a raw Flash `.bin`
/// image via `/bin/llvm-objcopy -O binary`, verifying reset `jmp` vector at `0x0000`, `<= 49,152 B`
/// (48 KB) Flash budget, and `< 5,632 B` static SRAM budget (reserving `>= 512 B` hardware call stack
/// inside the 6,144 B internal SRAM `0x2800..=0x3FFF`).
pub fn objcopy_elf_to_avr_nano_every_bin(
    elf_path: &Path,
    out_bin: &Path,
) -> Result<(u64, u64), CodegenError> {
    if let Some(parent) = out_bin.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                CodegenError::new("io", format!("failed to create {}: {e}", parent.display()))
            })?;
        }
    }
    let objcopy = find_tool("/bin/llvm-objcopy", "llvm-objcopy");
    let out = Command::new(objcopy)
        .arg("-O")
        .arg("binary")
        .arg(elf_path)
        .arg(out_bin)
        .output()
        .map_err(|e| {
            CodegenError::new("llvm-objcopy", format!("failed to invoke `{objcopy}`: {e}"))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "llvm-objcopy",
            format!("llvm-objcopy exited with {}: {}", out.status, stderr.trim()),
        ));
    }

    let bytes = fs::read(out_bin).map_err(|e| {
        CodegenError::new("io", format!("failed to read {}: {e}", out_bin.display()))
    })?;
    if bytes.len() < 4 || bytes.len() > 49_152 {
        return Err(CodegenError::new(
            "avr-image",
            format!(
                "generated AVR .bin size {} B is out of ATmega4809 Flash bounds (4..=49152 B)",
                bytes.len()
            ),
        ));
    }
    let word0 = u16::from_le_bytes([bytes[0], bytes[1]]);
    if (word0 & 0xFE0E) != 0x940C {
        return Err(CodegenError::new(
            "avr-image",
            format!("expected AVR `jmp` instruction at 0x0000, got word 0x{word0:04X}"),
        ));
    }

    let sram_static_bytes = match (
        read_elf32_sym_addr(elf_path, "__sdata"),
        read_elf32_sym_addr(elf_path, "__ebss"),
    ) {
        (Some(sdata), Some(ebss)) if ebss >= sdata => u64::from(ebss - sdata),
        _ => {
            let sizes = inspect_elf_sizes(elf_path)?;
            sizes.data + sizes.bss
        }
    };
    if sram_static_bytes >= 5_632 {
        return Err(CodegenError::new(
            "avr-image",
            format!(
                "static SRAM footprint {sram_static_bytes} B leaves fewer than 512 B of stack in 6,144 B ATmega4809 SRAM"
            ),
        ));
    }

    Ok((bytes.len() as u64, sram_static_bytes))
}

pub(crate) const AVR_MUXTO_JTAG2UPDI_FLASH_SCRIPT: &str = r#"
import struct, sys, time, serial
try:
    import termios
except ImportError:
    termios = None

port, bin_path = sys.argv[1], sys.argv[2]
data = open(bin_path, "rb").read()
if len(data) < 4 or len(data) > 49152:
    raise SystemExit(f"ATmega4809 image size {len(data)} B out of Flash bounds (4..=49152 B)")

page_size = 128
padded_len = ((len(data) + page_size - 1) // page_size) * page_size
padded = data + b"\xFF" * (padded_len - len(data))
pages = padded_len // page_size

def crc16_ccitt(buf: bytes) -> int:
    crc = 0xFFFF
    for b in buf:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc & 0xFFFF

def read_exact(ser, n: int, deadline: float) -> bytes:
    buf = bytearray()
    while len(buf) < n and time.monotonic() < deadline:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
    return bytes(buf)

seq_box = [0]

def jtag2_cmd(ser, body: bytes, what: str, timeout: float = 2.0) -> bytes:
    seq = seq_box[0] & 0xFFFF
    seq_box[0] = (seq + 1) & 0xFFFF
    pkt_core = b"\x1B" + struct.pack("<HI", seq, len(body)) + b"\x0E" + body
    pkt = pkt_core + struct.pack("<H", crc16_ccitt(pkt_core))

    last_err = "timeout"
    for attempt in range(3):
        ser.reset_input_buffer()
        ser.write(pkt)
        ser.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            b0 = ser.read(1)
            if not b0:
                continue
            if b0 != b"\x1B":
                continue
            hdr = read_exact(ser, 6, deadline)
            if len(hdr) != 6:
                last_err = "truncated JTAG2 header"
                break
            r_seq, r_size = struct.unpack("<HI", hdr)
            if r_size > 512:
                last_err = f"oversize JTAG2 frame ({r_size} B)"
                continue
            tok = read_exact(ser, 1, deadline)
            if tok != b"\x0E":
                last_err = f"bad JTAG2 token {tok!r}"
                continue
            r_body = read_exact(ser, r_size, deadline)
            r_crc_bytes = read_exact(ser, 2, deadline)
            if len(r_body) != r_size or len(r_crc_bytes) != 2:
                last_err = "truncated JTAG2 body/CRC"
                break
            got_crc = struct.unpack("<H", r_crc_bytes)[0]
            exp_crc = crc16_ccitt(b"\x1B" + hdr + b"\x0E" + r_body)
            if got_crc != exp_crc:
                last_err = f"CRC mismatch (got 0x{got_crc:04X}, expected 0x{exp_crc:04X})"
                continue
            if r_seq != seq:
                continue
            return r_body
        time.sleep(0.05)
    raise RuntimeError(f"JTAG2UPDI {what} failed: {last_err}")

def expect_ok(ser, body: bytes, what: str, timeout: float = 2.0):
    resp = jtag2_cmd(ser, body, what, timeout=timeout)
    if not resp or resp[0] != 0x80:
        raise RuntimeError(f"JTAG2UPDI {what} rejected: response={resp.hex(' ').upper()}")

# 1. 1200-baud touch so SAMD11 MuxTO enters `updi_mode` (`baud==1200` with `DTR: 1 -> 0`)
touch = serial.Serial(port, 1200, timeout=0.5)
try:
    touch.dtr = True
    time.sleep(0.05)
    touch.dtr = False
    time.sleep(0.15)
finally:
    touch.close()
time.sleep(0.20)

# 2. Reopen at 115200 for JTAGICE mkII (`jtag2updi`) UPDI session
ser = serial.Serial(port, 115200, timeout=0.2)
try:
    sign_on = jtag2_cmd(ser, b"\x01", "CMND_GET_SIGN_ON", timeout=2.0)
    if not sign_on or sign_on[0] != 0x86:
        raise RuntimeError(f"Unexpected SIGN_ON response: {sign_on.hex(' ').upper()}")

    # PARAM_BAUD_RATE (0x05) = baud_115200 (0x07)
    expect_ok(ser, b"\x02\x05\x07", "CMND_SET_PARAMETER(BAUD_115200)")

    # CMND_SET_DEVICE_DESCRIPTOR (250 B: body[244]=128 flash page, body[246]=64 eeprom page)
    desc = bytearray(250)
    desc[0] = 0x0C
    desc[244] = 128
    desc[246] = 64
    expect_ok(ser, bytes(desc), "CMND_SET_DEVICE_DESCRIPTOR")

    expect_ok(ser, b"\x14", "CMND_ENTER_PROGMODE")

    # Verify ATmega4809 SIGROW signature (`1E 96 51`) at UPDI address 0x1100
    sig_resp = jtag2_cmd(
        ser,
        struct.pack("<BBII", 0x05, 0xB4, 3, 0x1100),
        "CMND_READ_MEMORY(SIGROW 0x1100)",
    )
    if len(sig_resp) != 4 or sig_resp[0] != 0x82:
        raise RuntimeError(f"Signature read failed: {sig_resp.hex(' ').upper()}")
    sig = sig_resp[1:4]
    if sig != b"\x1e\x96\x51":
        raise RuntimeError(f"Expected ATmega4809 signature 1E 96 51, got {sig.hex(' ').upper()}")
    print(f"MuxTO jtag2updi synced on {port}: signature={sig.hex(' ').upper()} (ATmega4809)")

    # Chip erase (`CMND_XMEGA_ERASE = 0x34`, mode 0 = chip erase + re-enter progmode)
    expect_ok(ser, b"\x34\x00\x00\x00\x00\x00", "CMND_XMEGA_ERASE(chip)", timeout=5.0)

    # Program 128-byte Flash pages at UPDI mapped Flash base `0x4000 + offset`
    for p in range(pages):
        off = p * page_size
        chunk = padded[off : off + page_size]
        wr_body = struct.pack("<BBII", 0x04, 0xC0, page_size, 0x4000 + off) + chunk
        expect_ok(ser, wr_body, f"CMND_WRITE_MEMORY(page {p} @ 0x{off:04X})", timeout=2.5)

    # Full readback verification across all programmed pages
    for p in range(pages):
        off = p * page_size
        rd_body = struct.pack("<BBII", 0x05, 0xC0, page_size, 0x4000 + off)
        rd_resp = jtag2_cmd(ser, rd_body, f"CMND_READ_MEMORY(page {p} @ 0x{off:04X})", timeout=2.0)
        if len(rd_resp) != 1 + page_size or rd_resp[0] != 0x82:
            raise RuntimeError(f"Flash readback failed at page {p}: {rd_resp[:8].hex(' ').upper()}")
        got = rd_resp[1:]
        expected = padded[off : off + page_size]
        if got != expected:
            raise RuntimeError(f"Flash verify mismatch in page {p} @ 0x{off:04X}")

    expect_ok(ser, b"\x15", "CMND_LEAVE_PROGMODE")
    expect_ok(ser, b"\x00", "CMND_SIGN_OFF")
    print(f"Flashed and verified {pages} page(s) ({len(data)} bytes) at UPDI 0x4000..0x{0x4000 + padded_len:04X}")
finally:
    ser.close()

# 3. Allow MuxTO `updi_mode_end` (>500ms) timeout to exit UPDI mode, then assert DTR at 115200
#    with ~HUPCL so MuxTO starts the 115200 8N1 CDC bridge and pulses UPDI reset into our app!
time.sleep(0.65)
boot = serial.Serial(port, 115200, timeout=0.2)
try:
    if termios is not None:
        attrs = termios.tcgetattr(boot.fileno())
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(boot.fileno(), termios.TCSANOW, attrs)
    boot.dtr = True
    time.sleep(0.08)
finally:
    boot.close()
"#;

/// Flash a raw ATmega4809 Flash `.bin` image onto an Arduino Nano Every (`MuxTO` SAMD11 `jtag2updi`)
/// via pure-Python `pyserial` (`1200`-baud touch -> JTAGICE mkII over UPDI -> `1E 96 51` signature check
/// -> chip erase -> 128 B page write + full readback verify at `0x4000 + off` -> `115200` CDC wakeup).
pub fn flash_avr_nano_every_bin(bin_path: &Path, port: &str) -> Result<String, CodegenError> {
    let out = Command::new("python3")
        .arg("-c")
        .arg(AVR_MUXTO_JTAG2UPDI_FLASH_SCRIPT)
        .arg(port)
        .arg(bin_path)
        .output()
        .map_err(|e| CodegenError::new("avr-flash", format!("failed to invoke `python3`: {e}")))?;

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        return Err(CodegenError::new(
            "avr-flash",
            format!(
                "Arduino Nano Every MuxTO (jtag2updi) flasher exited with {}:\n{}",
                out.status,
                combined.trim()
            ),
        ));
    }
    Ok(combined)
}
