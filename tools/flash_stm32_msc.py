#!/usr/bin/env python3
"""Standalone ST-LINK USB Mass-Storage (MSC) Flasher via udisksctl."""

import os
import struct
import subprocess
import sys
import time


def query_mount(by_label):
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


def mount_volume(by_label, retries=8):
    mount_dir = query_mount(by_label)
    if mount_dir is not None:
        return mount_dir, False
    last_err = ""
    for _ in range(retries):
        res = subprocess.run(
            ["udisksctl", "mount", "--no-user-interaction", "-b", by_label],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            time.sleep(0.15)
            mount_dir = query_mount(by_label)
            if mount_dir:
                return mount_dir, True
        else:
            last_err = (res.stdout + res.stderr).strip()
            # Handle race where another process mounted between query_mount and udisksctl mount
            mount_dir = query_mount(by_label)
            if mount_dir:
                return mount_dir, False
        time.sleep(0.2)
    raise SystemExit(f"failed to mount {by_label}: {last_err}")


def unmount_volume(by_label):
    if query_mount(by_label) is None:
        return
    subprocess.run(
        ["udisksctl", "unmount", "--no-user-interaction", "-b", by_label],
        capture_output=True,
        text=True,
        check=False,
    )


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: flash_stm32_msc.py <bin_path> [port] [volume_label]"
        )

    bin_path = sys.argv[1]
    port = sys.argv[2] if len(sys.argv) > 2 else "/dev/ttyACM0"
    volume_label = sys.argv[3] if len(sys.argv) > 3 else "NODE_F334R8"
    by_label = f"/dev/disk/by-label/{volume_label}"

    with open(bin_path, "rb") as f:
        bin_bytes = f.read()

    if len(bin_bytes) < 8:
        raise SystemExit(
            f"binary `{bin_path}` is too small ({len(bin_bytes)} B) for Cortex-M vector header"
        )

    sp, reset = struct.unpack("<2I", bin_bytes[:8])
    if not (0x2000_0000 < sp <= 0x2010_0000):
        raise SystemExit(
            f"invalid initial SP 0x{sp:08X} in `{bin_path}` (expected 0x20000000 < sp <= 0x20100000)"
        )
    if (reset & 0xFF00_0001) != 0x0800_0001:
        raise SystemExit(
            f"invalid Thumb Flash Reset_Handler 0x{reset:08X} in `{bin_path}` (expected (reset & 0xFF000001) == 0x08000001)"
        )

    if not os.path.exists(by_label):
        raise SystemExit(
            f"ST-LINK MSC volume `{by_label}` not found; check USB connection to Nucleo ST-LINK"
        )

    mount_dir, _ = mount_volume(by_label)
    dest = os.path.join(mount_dir, "FIRMWARE.BIN")
    with open(dest, "wb") as dst:
        dst.write(bin_bytes)
        dst.flush()
        os.fsync(dst.fileno())
    os.sync()

    # Wait briefly to see if ST-LINK auto-disconnects on FAT commit; otherwise unmount to flush metadata.
    auto_deadline = time.monotonic() + 1.2
    while time.monotonic() < auto_deadline:
        if not os.path.exists(by_label) or query_mount(by_label) is None:
            break
        time.sleep(0.1)

    if os.path.exists(by_label) and query_mount(by_label) is not None:
        unmount_volume(by_label)

    # Allow ST-LINK SWD programming + target reset + USB re-enumeration.
    time.sleep(0.8)
    reappear_deadline = time.monotonic() + 8.0
    while time.monotonic() < reappear_deadline:
        label_ok = os.path.exists(by_label)
        port_ok = True if not port else os.path.exists(port)
        if label_ok and port_ok:
            break
        time.sleep(0.15)

    if not os.path.exists(by_label):
        raise SystemExit(
            f"ST-LINK volume `{by_label}` did not reappear within 8.0s after flashing"
        )

    # Mount briefly to check whether ST-LINK reported a programming failure in FAIL.TXT.
    check_dir, mounted_for_check = mount_volume(by_label, retries=10)
    try:
        fail_txt = os.path.join(check_dir, "FAIL.TXT")
        if os.path.isfile(fail_txt):
            with open(fail_txt, "r", encoding="utf-8", errors="replace") as ff:
                reason = ff.read().strip()
            raise SystemExit(f"ST-LINK MSC reported flashing failure in FAIL.TXT: {reason}")
    finally:
        if mounted_for_check:
            unmount_volume(by_label)

    port_present = bool(port and os.path.exists(port))
    print(
        f"Flashed STM32 MSC image ({len(bin_bytes)} bytes, SP=0x{sp:08X}, Reset=0x{reset:08X}) "
        f"via `{volume_label}`; port `{port}` present={port_present}"
    )


if __name__ == "__main__":
    main()
