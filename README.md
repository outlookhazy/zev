# Zev (`zevc`)

A Stage-0 and Stage-1 compiler (`zevc`) for **Zev**, an experimental bare-metal cooperative language and static runtime targeting microcontrollers and deterministic host simulation.

Zev builds cooperative tasks, stackless coroutine frames, typed MMIO registers, interrupt safety rules, and static DMA buffer ownership checks directly into the compiler, without a heap allocator or per-task stacks.

See [`DESIGN_V0.md`](DESIGN_V0.md) for the language specification and design rationale.

---

## Compiler Pipeline

- **`zev-ast` / `zev-parser`**: Recursive-descent parser (`f32`/`f64`, `<T, usize N>` generics, `f"..."` strings, `mmio`, `irq`, and `P0001`–`P0013` diagnostics for common C/Rust/Go/Zig syntax habits).
- **`zev-sema`**: Generic monomorphization, `import`/`pub` module resolution, 3-tier call-graph yield inference (`NonYielding`, `SoftYielding`, `HardBlocking`), Boot-Tree `spawn` / `irq.attach` checking, ISR isolation (`E0001`/`E0002`/`E0033`), `fmt_buf` bounds checking (`E0028`), and linear `Lease<T>` tracking (`E0034`).
- **`zev-mir`**: Stackless state-machine lowering, liveness hoisting, disjoint child-frame `union` packing, unreachable function pruning, and static SRAM budget reporting.
- **`zev-codegen-c`**: Stage-0 freestanding C99 emission and sanitized host simulation.
- **`zev-codegen-llvm`**: Stage-1 LLVM IR backend driving `opt`, `llc`, and `ld.lld` across ARM Cortex-M, RISC-V (`rv32imac`), and AVR targets configured declaratively via `chips/*.toml` and `boards/*.toml`.
- **`zev-cli` (`zevc`)**: Unified compiler CLI, SVD-to-MMIO importer (`import-svd`), and offline documentation (`zevc doc`).

---

## Targets

Board and chip definitions live in `boards/*.toml` and `chips/*.toml`. A single portable source ([`examples/portable_controller_demo.zev`](examples/portable_controller_demo.zev)) using `import board;` and `hal.*` compiles across five targets:

| Target (`--board`) | MCU / Architecture | Runtime / Artifact |
|---|---|---|
| `host` (default) | Host C99 simulation | Clang ASan+UBSan executable (`hal/sim`, `bsp/sim`) |
| `lm3s6965_evb` | ARM Cortex-M3 (`thumbv7m`) | Freestanding ELF run under `qemu-system-arm` |
| `nucleo_f334r8` | STM32F334R8 (`thumbv7em`) | Freestanding `.bin` (OpenOCD flash) |
| `esp32c6_pico` | ESP32-C6 (`riscv32imac`) | Direct-Boot `.bin` (`esptool` flash) |
| `rpi_pico_w` | RP2040 (`thumbv6m`) | SRAM `.uf2` (`RPI-RP2` copy + 1200-baud reset) |

Additional standalone hardware demos target `arduino_uno` (ATmega328P `avr`, Optiboot) and `arduino_nano_every` (ATmega4809 `avrxmega3`, `MuxTO` UPDI).

---

## Requirements

- **Rust** stable toolchain (`cargo`, `rustc`)
- **LLVM / Clang / LLD** (`clang`, `opt`, `llc`, `ld.lld`, `llvm-size`, `llvm-objcopy`, `llvm-objdump`, `llvm-readobj`, `llvm-nm` — LLVM 23 required; CI installs it from apt.llvm.org. Stock LLVM 18 fails to link AVR images and changes stack-proof results)
- **`qemu-system-arm`** (for `make qemu` and `zevc run-qemu`)
- **`binutils` (`nm`)** on host C99 test runs

Tool lookup checks `ZEV_LLVM_PREFIX` first (pointing at a custom LLVM install prefix), then `PATH`, then `/bin` and versioned `*-23..*-18` binaries. Override the built-in module/TOML search root with `ZEV_HOME` if running `zevc` outside the repository tree.

---

## Quick Start

```bash
# Build compiler and run workspace unit/integration tests
cargo test --workspace

# Verify built-in reference docs and all example checks
make doc
make check

# Built-in offline language & error reference
cargo run --quiet --bin zevc -- doc
cargo run --quiet --bin zevc -- doc --list
cargo run --quiet --bin zevc -- doc E0034

# Host C99 simulation with AddressSanitizer + UBSan
cargo run --quiet --bin zevc -- run --sanitize examples/portable_controller_demo.zev
make demo

# Bare-metal Cortex-M3 build + QEMU run
cargo run --quiet --bin zevc -- run-qemu examples/portable_controller_demo.zev
make qemu

# Compile-time SRAM report (task frames, child unions, ELF .bss, worst-case call stack)
cargo run --quiet --bin zevc -- ram --board nucleo_f334r8 examples/portable_controller_demo.zev
cargo run --quiet --bin zevc -- ram --board esp32c6_pico examples/portable_controller_demo.zev
cargo run --quiet --bin zevc -- ram --board rpi_pico_w examples/portable_controller_demo.zev

# Build or flash board firmware images
cargo run --quiet --bin zevc -- build --board nucleo_f334r8 examples/portable_controller_demo.zev -o build/f334r8.bin
cargo run --quiet --bin zevc -- flash --board nucleo_f334r8 examples/portable_controller_demo.zev
make esp32c6 pico uno nano-every nucleo-f334r8
```

---

## Status & Limitations

Zev is an early **v0** research/engineering compiler under active development:

- **What is verified in this repository:** Host C99 simulations compile under ASan+UBSan with zero heap allocator symbols (`malloc`/`calloc`/`realloc`/`free`); bare-metal ARM Cortex-M3 firmware executes under `qemu-system-arm`; generated `.ll` and `.elf` images for ARM, RISC-V, and AVR pass `opt -passes=verify`, zero-undefined-symbol audits, and linker/budget checks. Selective demos have been flashed to physical STM32F334R8, ESP32-C6-Pico, RP2040 Pico W, Arduino Uno R3, and Arduino Nano Every boards.
- **Limitations:**
  - Not every peripheral driver or corner case across all boards has long-running hardware soak coverage.
  - Hardware stack high-water calculation flags call-graph recursion cycles as `UNBOUNDED` rather than statically bounded.
  - Core compiler modules (`zev-sema`, `zev-mir`, `zev-codegen-llvm`) are undergoing ongoing refactoring and expanded adversarial negative testing.

---

## License

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (`Apache-2.0 WITH LLVM-exception`). See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
