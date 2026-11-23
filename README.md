# Zev (`zevc`)

A compiler (`zevc`) for **Zev**, an experimental bare-metal cooperative language and static runtime targeting microcontrollers and deterministic host LLVM simulation.

Zev builds cooperative tasks, stackless coroutine frames, typed MMIO registers, deterministic multi-source `wait { ... }`, interrupt safety rules, and static DMA buffer ownership checks directly into the compiler, without a heap allocator or per-task stacks.

See [`DESIGN_V0.md`](DESIGN_V0.md) for the language specification and design rationale.

---

## Compiler Pipeline

- **`zev-ast` / `zev-parser`**: Recursive-descent parser (`f32`/`f64`, `<T, usize N>` generics, canonical prefix types `*T` / `[]T` / `[N]T`, `f"..."` strings, `mmio`, `irq`, `wait { ... }`, `wait_queue`, and `P0001`–`P0014` diagnostics for common C/Rust/Go/Zig syntax habits).
- **`zev-sema`**: Generic monomorphization, `import`/`pub` module resolution, 3-tier call-graph yield inference (`NonYielding`, `SoftYielding`, `HardBlocking`), Boot-Tree `spawn` (`MAX_TASKS <= 32`) / `irq.attach` checking, ISR isolation (`E0001`/`E0002`/`E0033`), `fmt_buf` bounds checking (`E0028`), per-arm `wait` lease scoping, and linear `Lease<T>` tracking (`E0034`).
- **`zev-mir`**: Stackless state-machine lowering, liveness hoisting, disjoint child-frame `union` packing, unreachable function pruning, per-task non-yielding run-bound classification (`straight-line`, `bounded-for`, `noyield-spin`), and static SRAM budget reporting.
- **`zev-codegen-llvm`**: Unified LLVM 23 IR backend driving `opt`, `llc`, and `ld.lld` across host `x86_64` sanitized simulation (`zevc run --sanitize` + `sys.fuzz_irqs`) and bare-metal ARM Cortex-M, RISC-V (`rv32imac`), and AVR targets configured declaratively via `chips/*.toml` and `boards/*.toml`.
- **`zev-cli` (`zevc`)**: Unified compiler CLI, SVD-to-MMIO importer (`import-svd`), and offline documentation (`zevc doc`).

---

## Targets

Board and chip definitions live in `boards/*.toml` and `chips/*.toml`. A single portable source ([`examples/portable_controller_demo.zev`](examples/portable_controller_demo.zev)) using `import board;` and `hal.*` compiles across five targets:

| Target (`--board`) | MCU / Architecture | Runtime / Artifact |
|---|---|---|
| `host` (default) | Host `x86_64` LLVM simulation | Clang ASan+UBSan executable (`hal/sim`, `bsp/sim`, `sys.fuzz_irqs`) |
| `lm3s6965_evb` | ARM Cortex-M3 (`thumbv7m`) | Freestanding ELF run under `qemu-system-arm` |
| `nucleo_f334r8` | STM32F334R8 (`thumbv7em`) | Freestanding `.bin` (OpenOCD flash) |
| `esp32c6_pico` | ESP32-C6 (`riscv32imac`) | Direct-Boot `.bin` (`esptool` flash) |
| `rpi_pico_w` | RP2040 (`thumbv6m`) | SRAM `.uf2` (`RPI-RP2` copy + 1200-baud reset) |

Additional standalone hardware demos target `arduino_uno` (ATmega328P `avr`, Optiboot) and `arduino_nano_every` (ATmega4809 `avrxmega3`, `MuxTO` UPDI).

---

## Requirements

- **Rust** stable toolchain (`cargo`, `rustc`)
- **LLVM / Clang / LLD** (`clang`, `opt`, `llc`, `ld.lld`, `llvm-size`, `llvm-objcopy`, `llvm-objdump`, `llvm-readobj`, `llvm-nm` — LLVM 23+ required and verified at build time; CI installs LLVM 23 from apt.llvm.org)
- **`qemu-system-arm`** (for `make qemu` and `zevc run-qemu`)

Tool lookup checks `ZEV_LLVM_PREFIX` first (pointing at a custom LLVM install prefix), then `PATH` and versioned `*-23+` / `/usr/lib/llvm-23+/bin` binaries, rejecting backend toolchains older than LLVM 23 with an explicit diagnostic. Override the built-in module/TOML search root with `ZEV_HOME` if running `zevc` outside the repository tree.

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

# Host x86_64 LLVM simulation with AddressSanitizer + UBSan
cargo run --quiet --bin zevc -- run --sanitize examples/portable_controller_demo.zev
make demo

# Bare-metal Cortex-M3 build + QEMU run
cargo run --quiet --bin zevc -- run-qemu examples/portable_controller_demo.zev
make qemu

# Compile-time SRAM & run-bound report (task frames, child unions, ELF .bss, max non-yield run, stack)
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

- **What is verified in this repository:** Host `x86_64` LLVM simulations compile under Clang ASan+UBSan with zero heap allocator symbols (`malloc`/`calloc`/`realloc`/`free`) and support `sys.fuzz_irqs` interrupt interleaving; bare-metal ARM Cortex-M3 firmware executes under `qemu-system-arm`; generated `.ll` and `.elf` images for ARM, RISC-V, and AVR pass `opt -passes=verify`, zero-undefined-symbol audits, and linker/budget checks. Selective demos have been flashed to physical STM32F334R8, ESP32-C6-Pico, RP2040 Pico W, Arduino Uno R3, and Arduino Nano Every boards.
- **Limitations:**
  - Single-core cooperative execution only in v0 (`@core(N)` is parsed/stored as spawn metadata and reserved for future multicore work); `MAX_TASKS` is capped at `32`.
  - Not every peripheral driver or corner case across all boards has long-running hardware soak coverage.
  - Hardware stack high-water calculation flags call-graph recursion cycles as `UNBOUNDED` rather than statically bounded.
  - Core compiler modules (`zev-sema`, `zev-mir`, `zev-codegen-llvm`) are undergoing ongoing refactoring and expanded adversarial negative testing.

---

## License

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (`Apache-2.0 WITH LLVM-exception`). See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
