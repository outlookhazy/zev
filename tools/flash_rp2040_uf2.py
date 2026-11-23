#!/usr/bin/env python3
"""Standalone RP2040 UF2 Flasher with automatic 1200-baud BOOTSEL reset and udisksctl mount."""

import os
import subprocess
import sys
import time

if len(sys.argv) < 2:
    raise SystemExit("Usage: flash_rp2040_uf2.py <uf2_path> [port_or_mount]")

uf2_path = sys.argv[1]
target_arg = sys.argv[2] if len(sys.argv) > 2 else "/dev/ttyACM0"
by_label = "/dev/disk/by-label/RPI-RP2"

with open(uf2_path, "rb") as f:
    uf2_bytes = f.read()

mount_dir = None
port = None

if os.path.isdir(target_arg):
    mount_dir = target_arg
else:
    port = target_arg
    if not os.path.exists(by_label):
        if not os.path.exists(port):
            raise SystemExit(
                f"neither {by_label} nor serial port `{port}` exists; plug in the Pico or hold BOOTSEL"
            )
        try:
            import serial
        except Exception as e:
            raise SystemExit(f"pyserial unavailable for 1200-baud touch: {e}")

        try:
            ser = serial.Serial(port, 1200, timeout=0.1)
            ser.dtr = False
            time.sleep(0.05)
            ser.close()
        except Exception:
            pass

        deadline = time.monotonic() + 6.0
        while not os.path.exists(by_label):
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"RP2040 did not enter BOOTSEL ({by_label}) after 1200-baud touch on `{port}`; "
                    "hold BOOTSEL while plugging USB to flash the initial Zev SRAM UF2"
                )
            time.sleep(0.1)
        time.sleep(0.2)

    def query_mount():
        try:
            res = subprocess.run(
                ["lsblk", "-nr", "-o", "MOUNTPOINT", by_label],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                mp = res.stdout.strip()
                return mp if mp else None
        except Exception:
            pass
        return None

    mount_dir = query_mount()
    if mount_dir is None:
        last_err = ""
        for _ in range(8):
            res = subprocess.run(
                ["udisksctl", "mount", "--no-user-interaction", "-b", by_label],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                time.sleep(0.15)
                mount_dir = query_mount()
                if mount_dir:
                    break
            else:
                last_err = (res.stdout + res.stderr).strip()
            time.sleep(0.15)
        if mount_dir is None:
            raise SystemExit(f"failed to mount {by_label}: {last_err}")

dest = os.path.join(mount_dir, "firmware.uf2")
with open(dest, "wb") as dst:
    dst.write(uf2_bytes)
    dst.flush()
    os.fsync(dst.fileno())

reboot_deadline = time.monotonic() + 6.0
while os.path.exists(by_label) and time.monotonic() < reboot_deadline:
    time.sleep(0.1)

cdc_present = False
if port is not None:
    cdc_deadline = time.monotonic() + 5.0
    while not os.path.exists(port) and time.monotonic() < cdc_deadline:
        time.sleep(0.1)
    cdc_present = os.path.exists(port)

print(
    f"Flashed RP2040 SRAM UF2 ({len(uf2_bytes)} bytes, {len(uf2_bytes) // 512} block(s)) "
    f"via `{mount_dir}`; USB CDC port `{port or mount_dir}` present={cdc_present}"
)
