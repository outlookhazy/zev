# Zev Language-OS (`zevc` v0)

A Stage-0 & Stage-1 compiler (`zevc`) for **Zev**, a bare-metal cooperative Language-OS targeting resource-constrained microcontrollers (STM32, ARM Cortex-M, Espressif ESP32-C6 RISC-V, RP2040, AVR) and deterministic host simulation.

> **Core Thesis:** Go-style sequential concurrency composability—backed by a **100% static, zero-heap, single-stack cooperative kernel** instead of a garbage collector or per-thread RTOS stacks.

See [`DESIGN_V0.md`](DESIGN_V0.md) for the complete language specification and architectural notes.

---

## Compiler Pipeline

```text
  .zev sources
       │
       ▼
  zev-parser          Recursive-descent parser, `f32`/`f64`, `<T, usize N>`, `f"..."`, traps (P0001..P0012)
       │
       ▼
  zev-sema            `<T, usize N>` monomorphizer, 3-tier yield inference (`SoftYielding` dual-spec),
       │              Boot-Tree `spawn` provenance, immortal `task` + ISR/task isolation, static
       │              `fmt_buf` capacity proofs, zero-`'a` DMA lease & scope tracking (E0001..E0029)
       ▼
  zev-mir             Stackless state-machine lowering, live-across-yield hoisting,
       │              disjoint child coroutine frame `union` packing, `f"..."` / `{pkt:?}` unpacking
       ├──► zev-codegen-c     Stage 0: Freestanding C99 + deterministic host simulation
       └──► zev-codegen-llvm  Stage 1: Direct LLVM 23 IR -> llc -> ld.lld -> Bare-Metal Firmware
                              • ARM Cortex-M3 (`thumbv7m-none-eabi`, QEMU `lm3s6965evb`)
                              • ESP32-C6 (`riscv32imac` Direct-Boot `.bin` + USB-Serial + WS2812B)
```

---

## Quick Start & CLI Usage

### Build & Test Everything

```bash
make test           # Runs unit & integration suites across all 7 workspace crates
make doc            # Smoke-checks built-in `zevc doc` primer, 8 topic guides & representative explainers (all 41 verified in `make test`)
make check          # Runs `zevc check` on all example firmware slices
make ram            # Prints the static zero-heap SRAM budget & Boot-Tree spawn provenance report
make demo           # Runs deterministic ASan+UBSan C99 host simulation (`sim_demo.zev`)
make qemu           # Builds ARM Cortex-M3 `.elf` via LLVM 23 + `ld.lld` & runs in `qemu-system-arm`
make esp32c6        # Synthesizes freestanding ESP32-C6 `rv32imac` Direct-Boot `.bin`
make flash-esp32c6  # Builds & flashes `examples/esp32c6_demo.zev` onto `/dev/ttyACM0`
```

Or invoke `zevc` directly via Cargo from the workspace root:

```bash
cargo run --bin zevc -- --help
```

### 0. Built-In Agent & Engineer Reference (`zevc doc`)

`zevc` ships a progressive-disclosure documentation engine right inside the compiler binary—giving humans and AI coding agents a 1-call language primer, Rosetta Stone mapping from Rust/Zig/Go/C, 8 focused topic guides, and `BAD` vs `GOOD` explainers for every parser trap (`P0001`..`P0012`) and verifier code (`E0001`..`E0029`):

```bash
zevc doc                  # ~80-line Zev Quick Primer + Rosetta Stone + topic/error index
zevc doc --list           # List all 8 topic guides and 41 error codes
zevc doc syntax           # Guides: syntax | errors | tasks | yielding | primitives | dma | runtime | targets
zevc doc runtime          # Zero-heap `f"..."` (`println`, `fmt_buf`, `:.2`/`:e`/`:?`), `sys.*`, `hw.*`, NVIC
zevc doc P0001            # Foreign-idiom trap explainers (P0001..P0012)
zevc doc E0028            # Static verifier explainers with BAD vs GOOD .zev (E0001..E0029)
```

### 1. Static Verification (`zevc check` & `zevc check --json`)

Parses `.zev` sources, monomorphizes `<T>` / `<T, usize N>` generics (`E0029`), and runs full whole-program semantic verification (3-tier call-graph yield inference with `SoftYielding` `noyield` dual-specialization reporting, Boot-Tree `spawn` + immortal `task` rules, static `fmt_buf` worst-case buffer-fit proofs (`E0028`), `isr` / `noyield` / `critical` / `defer` safety, task/ISR global isolation, DMA `ping_pong` buffer lease lifetimes, and `return`/`error`/`catch` checking), emitting rich `= help:` + `= docs:` pointers or structured JSON (`dual_specialized_functions` + `monomorphized_instances`):

```bash
zevc check examples/motor_controller.zev examples/motor_support.zev
zevc check examples/sim_demo.zev
zevc check examples/cortex_m_demo.zev
zevc check examples/esp32c6_demo.zev
zevc check --json examples/motor_controller.zev examples/motor_support.zev
```

### 2. Static Zero-Heap RAM Budget (`zevc ram`)

Computes exact compile-time SRAM usage across spawned task instances (with full Boot-Tree `spawn` provenance paths), hoisted coroutine state frames, **`union`-aliased disjoint child frames** (e.g., overlapping `save_fault_record` and `scrub_flash_crc` inside `main`'s supervisor frame), dual-specialized `SoftYielding` functions, concrete `<T, usize N>` monomorphized instantiations, unified `inbox` / `watch` / DMA primitives, and scheduler descriptors:

```bash
zevc ram examples/motor_controller.zev examples/motor_support.zev
```

### 3. Stage 0: Emit Freestanding C99 & Host Simulation (`zevc emit-c`, `zevc run`)

Lowers tasks and yielding functions into stackless C99 state machines bundled with the synthesized cooperative kernel, or compiles and executes directly via `/bin/clang -std=c99 -O2`:

```bash
zevc emit-c examples/motor_controller.zev examples/motor_support.zev -o build/motor_controller.c
zevc run examples/sim_demo.zev
zevc run --sanitize examples/sim_demo.zev
```

### 4. Stage 1A: Native LLVM 23 IR, Bare-Metal Cortex-M3 ELF & QEMU (`zevc emit-llvm`, `zevc build-elf`, `zevc run-qemu`)

Emits native LLVM 23 opaque-pointer IR directly from `zev-mir`—synthesizing the `.isr_vector` NVIC table, `Reset_Handler`, `SysTick_Handler`, and `.ld` linker script with zero C, zero `.s` files, and zero `libc`—linking via `llc` + `ld.lld` and executing on `qemu-system-arm -M lm3s6965evb`:

```bash
zevc emit-llvm examples/cortex_m_demo.zev -o build/cortex_m_demo.ll
zevc build-elf examples/cortex_m_demo.zev -o build/cortex_m_demo.elf
zevc run-qemu examples/cortex_m_demo.zev
zevc run-qemu --max-ticks 50 examples/motor_controller.zev examples/motor_support.zev
```

### 5. Stage 1B: Bare-Metal ESP32-C6 (`riscv32imac` Direct-Boot) Hardware Target (`zevc build-esp32c6`, `zevc flash-esp32c6`)

Synthesizes a zero-bootloader ESP32-C6 Direct-Boot image (`0xAEDB041D` x2 ROM header, `@_start` at `0x42000008`, `TIMG0`/`TIMG1`/`LP_WDT`/`SWD` watchdog disarm, 16 MHz `SYSTIMER`, USB-Serial/JTAG console, and `.iram1` WS2812B RGB driver) and flashes directly via `esptool`:

```bash
zevc build-esp32c6 examples/esp32c6_demo.zev -o build/esp32c6_demo.bin
zevc flash-esp32c6 examples/esp32c6_demo.zev --port /dev/ttyACM0
```

---

## Example Files

| File | Description |
|---|---|
| [`examples/motor_controller.zev`](examples/motor_controller.zev) | Reference dual-bus STM32G4 motor controller showcasing `main()` phased boot, multi-instance `spawn`, `loop every 1ms` `noyield` PI loops, zero-copy `ping_pong` UART DMA, `u16be` wire frames, `{err}` enum formatting, and `EXTI0` ISR `boost: @critical`. |
| [`examples/motor_support.zev`](examples/motor_support.zev) | Companion hardware setup, monomorphized `clamp<T>`, bounded `for..in` `crc16`, `pi_step`, yielding `adc_read_rpm`, and disjoint yielding flash helpers (`save_fault_record` w/ `defer` & `scrub_flash_crc`). |
| [`examples/sim_demo.zev`](examples/sim_demo.zev) | Self-contained runnable host simulation exercising `loop every 1ms` control, `FixedVec<u16, 4>` & `clamp<T>` generics, `f32` `:.1` + proven-fit `fmt_buf`, `u16be` DMA packets, `error BadCrc` / `{err}` recovery, `EXTI0` boost & restore, and `@dma_safe` sleep-depth clamping. |
| [`examples/cortex_m_demo.zev`](examples/cortex_m_demo.zev) | Zero-C bare-metal ARM Cortex-M3 firmware (`qemu-system-arm`) with `clamp<T>`, `defer`, zero-heap `println(f"...")` (`:08x` + `{err}`), real NVIC interrupts, `SysTick`, and `sys.exit(0)`. |
| [`examples/esp32c6_demo.zev`](examples/esp32c6_demo.zev) | Bare-metal ESP32-C6-Pico (`rv32imac`) Direct-Boot firmware showcasing `FixedVec<f32, 4>` & `clamp<f32>`, proven-fit `fmt_buf`, `{sample:?}` debug + `{ev}` enum `println(f"...")` telemetry, `GPIO8` WS2812B RGB LED, and `GPIO9` `BOOT` button (`boost: @critical`). |

---

## Workspace Layout

- `crates/zev-ast` — Core AST, `f32`/`f64`, `<T, usize N>` generics, `f"..."` format specs, and source spans
- `crates/zev-parser` — Lexer, recursive-descent `.zev` parser, and `P0001`..`P0012` foreign-idiom traps
- `crates/zev-sema` — `<T, usize N>` monomorphizer, 3-tier yield-chain tracer, Boot-Tree `spawn` unroller, static `fmt_buf` fit prover, and `E0001`..`E0029` diagnostics
- `crates/zev-mir` — Stackless coroutine state-machine MIR, `SoftYielding` `noyield` clone specializer, `f"..."` / `{pkt:?}` unpacker & static `union` RAM packer
- `crates/zev-codegen-c` — Stage-0 freestanding C99 code generator & host simulation runtime
- `crates/zev-codegen-llvm` — Stage-1 native LLVM 23 IR + ARM Cortex-M3 & ESP32-C6 (`rv32imac`) bare-metal backend
- `crates/zev-cli` — `zevc` compiler & built-in `zevc doc` reference (`doc`, `check [--json]`, `ram`, `emit-c`, `run`, `emit-llvm`, `build-elf`, `run-qemu`, `build-esp32c6`, `flash-esp32c6`)

---

## License

Licensed under the **Apache License, Version 2.0 with LLVM Exceptions** (`Apache-2.0 WITH LLVM-exception`). See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details.

