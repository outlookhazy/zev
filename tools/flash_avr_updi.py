#!/usr/bin/env python3
"""Standalone MuxTO JTAG2UPDI Flasher (ATmega4809 / Arduino Nano Every)."""

import struct
import sys
import time
import serial

try:
    import termios
except ImportError:
    termios = None

if len(sys.argv) < 3:
    raise SystemExit("Usage: flash_avr_updi.py <bin_path> <port>")

bin_path = sys.argv[1]
port = sys.argv[2]

data = open(bin_path, "rb").read()
if len(data) < 4 or len(data) > 49152:
    raise SystemExit(
        f"ATmega4809 image size {len(data)} B out of Flash bounds (4..=49152 B)"
    )

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
                last_err = (
                    f"CRC mismatch (got 0x{got_crc:04X}, expected 0x{exp_crc:04X})"
                )
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
        raise RuntimeError(
            f"Expected ATmega4809 signature 1E 96 51, got {sig.hex(' ').upper()}"
        )
    print(f"MuxTO jtag2updi synced on {port}: signature={sig.hex(' ').upper()} (ATmega4809)")

    # Chip erase (`CMND_XMEGA_ERASE = 0x34`, mode 0 = chip erase + re-enter progmode)
    expect_ok(ser, b"\x34\x00\x00\x00\x00\x00", "CMND_XMEGA_ERASE(chip)", timeout=5.0)

    # Program 128-byte Flash pages at UPDI mapped Flash base `0x4000 + offset`
    for p in range(pages):
        off = p * page_size
        chunk = padded[off : off + page_size]
        wr_body = struct.pack("<BBII", 0x04, 0xC0, page_size, 0x4000 + off) + chunk
        expect_ok(
            ser, wr_body, f"CMND_WRITE_MEMORY(page {p} @ 0x{off:04X})", timeout=2.5
        )

    # Full readback verification across all programmed pages
    for p in range(pages):
        off = p * page_size
        rd_body = struct.pack("<BBII", 0x05, 0xC0, page_size, 0x4000 + off)
        rd_resp = jtag2_cmd(
            ser, rd_body, f"CMND_READ_MEMORY(page {p} @ 0x{off:04X})", timeout=2.0
        )
        if len(rd_resp) != 1 + page_size or rd_resp[0] != 0x82:
            raise RuntimeError(
                f"Flash readback failed at page {p}: {rd_resp[:8].hex(' ').upper()}"
            )
        got = rd_resp[1:]
        expected = padded[off : off + page_size]
        if got != expected:
            raise RuntimeError(f"Flash verify mismatch in page {p} @ 0x{off:04X}")

    expect_ok(ser, b"\x15", "CMND_LEAVE_PROGMODE")
    expect_ok(ser, b"\x00", "CMND_SIGN_OFF")
    print(
        f"Flashed and verified {pages} page(s) ({len(data)} bytes) at UPDI 0x4000..0x{0x4000 + padded_len:04X}"
    )
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
