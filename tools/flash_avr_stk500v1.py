#!/usr/bin/env python3
"""Standalone STK500v1 Optiboot Flasher (ATmega328P / Arduino Uno R3)."""

import sys
import time
import serial

if len(sys.argv) < 3:
    raise SystemExit("Usage: flash_avr_stk500v1.py <bin_path> <port> [baud]")

bin_path = sys.argv[1]
port = sys.argv[2]
baud = int(sys.argv[3]) if len(sys.argv) > 3 else 115200

data = open(bin_path, "rb").read()
if len(data) < 4 or len(data) > 32256:
    raise SystemExit(
        f"AVR image size {len(data)} B out of Optiboot bounds (4..=32256 B)"
    )

page_size = 128
padded_len = ((len(data) + page_size - 1) // page_size) * page_size
padded = data + b"\xFF" * (padded_len - len(data))
pages = padded_len // page_size


def read_exact(ser, n, timeout=0.6):
    buf = bytearray()
    deadline = time.monotonic() + timeout
    while len(buf) < n and time.monotonic() < deadline:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
    return bytes(buf)


def cmd(ser, payload, resp_len, what):
    ser.reset_input_buffer()
    ser.write(payload)
    ser.flush()
    resp = read_exact(ser, resp_len)
    if len(resp) != resp_len or resp[0] != 0x14 or resp[-1] != 0x10:
        raise RuntimeError(f"STK500v1 {what} failed: got {resp.hex(' ').upper()}")
    return resp


ser = serial.Serial(port, baud, timeout=0.1)
try:
    ser.dtr = False
    time.sleep(0.05)
    ser.dtr = True
    time.sleep(0.25)
    ser.reset_input_buffer()

    synced = False
    for _ in range(8):
        ser.reset_input_buffer()
        ser.write(b"\x30\x20")
        ser.flush()
        r = read_exact(ser, 2, 0.25)
        if r == b"\x14\x10":
            synced = True
            break
        time.sleep(0.05)
    if not synced:
        raise RuntimeError("STK_GET_SYNC failed after 8 attempts")

    maj = cmd(ser, b"\x41\x81\x20", 3, "GET_PARAMETER SW_MAJOR")[1]
    minr = cmd(ser, b"\x41\x82\x20", 3, "GET_PARAMETER SW_MINOR")[1]
    sign = cmd(ser, b"\x75\x20", 5, "STK_READ_SIGN")[1:4]
    if sign != b"\x1e\x95\x0f":
        raise RuntimeError(
            f"Expected ATmega328P signature 1E 95 0F, got {sign.hex(' ').upper()}"
        )

    print(
        f"Optiboot {maj}.{minr} synced on {port}: signature={sign.hex(' ').upper()} (ATmega328P)"
    )
    cmd(ser, b"\x50\x20", 2, "STK_ENTER_PROGMODE")

    for p in range(pages):
        byte_addr = p * page_size
        word_addr = byte_addr // 2
        addr_cmd = bytes([0x55, word_addr & 0xFF, (word_addr >> 8) & 0xFF, 0x20])
        cmd(ser, addr_cmd, 2, f"STK_LOAD_ADDRESS(write page {p})")
        chunk = padded[byte_addr : byte_addr + page_size]
        prog_cmd = bytes([0x64, 0x00, page_size, 0x46]) + chunk + b"\x20"
        cmd(ser, prog_cmd, 2, f"STK_PROG_PAGE(page {p})")

    for p in range(pages):
        byte_addr = p * page_size
        word_addr = byte_addr // 2
        addr_cmd = bytes([0x55, word_addr & 0xFF, (word_addr >> 8) & 0xFF, 0x20])
        cmd(ser, addr_cmd, 2, f"STK_LOAD_ADDRESS(read page {p})")
        read_cmd = bytes([0x74, 0x00, page_size, 0x46, 0x20])
        resp = cmd(ser, read_cmd, 2 + page_size, f"STK_READ_PAGE(page {p})")
        got = resp[1:-1]
        expected = padded[byte_addr : byte_addr + page_size]
        if got != expected:
            raise RuntimeError(f"Flash verify mismatch in page {p} @ 0x{byte_addr:04X}")

    cmd(ser, b"\x51\x20", 2, "STK_LEAVE_PROGMODE")
    print(
        f"Flashed and verified {pages} page(s) ({len(data)} bytes) at 0x0000..0x{padded_len:04X}"
    )
finally:
    ser.close()
