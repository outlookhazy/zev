//! Built-in progressive-disclosure documentation engine (`zevc doc`).
//!
//! Designed so an AI coding agent or embedded systems engineer can learn idiomatic Zev
//! in 1 tool call (`zevc doc`), drill into focused ~60-line topic guides (`zevc doc <topic>`),
//! or inspect any parser/verifier error code (`zevc doc P0001`..`P0010`, `E0001`..`E0024`).

use crate::docs_errors::{ErrorDoc, ERROR_DOCS};
use std::fmt::Write;

pub struct TopicDoc {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub summary: &'static str,
    pub body: &'static str,
}

pub const PRIMER_DOC: &str = "\
================================================================================
  ZEV LANGUAGE-OS — QUICK PRIMER & AGENT REFERENCE (`zevc doc`)
================================================================================

CORE MENTAL MODEL (5 RULES):
  1. Explicit C-Style Types: Write `u16 rpm = 1800;` and `[]u8 buf` (never `let`/`var`).
     Value-returning functions put the return type first (`u16 calc(u16 x)`); `fn f()` is `void`.
  2. Phased Boot + Reusable Tasks: Execution starts sequentially in `fn main()`. `task` items
     are parameterized blueprints instantiated statically via `spawn worker(&ep, 1) @high;`.
  3. Automatic Yielding + Explicit Brakes: Blocking calls (`wait_chunk`, `recv`, `wait`, `delay`,
     `checkpoint`) yield transparently on a single stack (no `async`/`.await`). Use `noyield { ... }`
     (task-atomic), `critical { ... }` (IRQ-masked), or `@manual_yield` when atomicity is required.
  4. Strict Task/ISR Isolation: Zero mutable raw globals. Cross-task/ISR state goes exclusively
     through static primitives: `watch<T>`, `inbox<T, N>`, `chan<T, N>`, `signal`, `ping_pong<T, N>`,
     and `lease<T, N>` (with optional ISR priority elevation `boost: @critical`).
  5. Dual-Exit Register Errors: Normal exit is `return val;`; failure exit is `error;`,
     `error Reason;`, or `error Reason, partial;`, handled via `or <val>` or `catch (...) { ... }`.

ROSETTA STONE (FOREIGN HABIT -> IDIOMATIC ZEV):
  Foreign Habit (Rust / Zig / Go / C)     | Idiomatic Zev (`.zev`)
  ----------------------------------------+-----------------------------------------------------
  `let mut x: u32 = 1;` / `var x = 1;`    | `u32 x = 1;`
  `fn add(a: u16, b: u16) -> u16`         | `u16 add(u16 a, u16 b)`   (`fn init()` = `void`)
  `async fn f()` + `f().await`            | `u16 f()` + `f()`         (call-graph inferred)
  `select { a = ch1 => ..., b = ch2 }`    | `Ev e = q.recv(timeout: 50ms) or Idle; switch (e)`
  `match ev { A => ..., B => ... }`       | `switch (ev) { A => { ... } _ => { ... } }`
  `Result<u16, E>` / `!u16` / `try f()`   | `u16 f() error E` + `error Bad;` + `catch (E e)` / `or 0`
  `catch |err| { ... }`                   | `catch (ParseErr err) { ... }`
  `for i in 0..8 { ... }`                 | `u32 i = 0; while (i < 8) { ... i += 1; }` / `every 1ms`
  `struct R<'a> { b: &'a mut [u8] }`      | `struct R { []u8 b; }` + `i32 *ptr` (0 `'a` syntax)
  `static mut G: u32 = 0;`                | `watch<u32> g = 0;` or `inbox<Ev, 8> q;` (`const` for ROM)
  `println!(\"k={}\", v)` / `printf(...)`   | `println(\"ok\");` and `sys.log_kv(\"k=\", v);`

SHARDED TOPIC GUIDES (`zevc doc <topic>`):
  zevc doc syntax       Types, `u16be`/`packed struct`, `enum`, `fn` vs `Ret f()`, `switch` (`enum`/`const`)
  zevc doc errors       Dual-exit `error` / `error ErrType`, partial progress, multi-stmt `catch`, `or`
  zevc doc tasks        `fn main()` phased boot, `task`, `spawn @prio @core(N)`, `every 1ms`
  zevc doc yielding     Call-graph yield inference, `noyield`, `critical`, `isr`, `extern \"C\"`
  zevc doc primitives   `watch`, `inbox`, `chan`, `signal`, `boost: @critical`, global isolation
  zevc doc dma          `@dma_safe ping_pong`, `lease`, zero-`'a` scope/lease rules, sleep locks
  zevc doc runtime      Portable `println` / `sys.*` I/O, `chip.stm32g4` (`hw.*`), NVIC `isr` map
  zevc doc targets      `zevc check [--json]`, `ram`, `run`, `run-qemu [--max-ticks]`, ESP32-C6

ERROR CODE EXPLAINERS (`zevc doc <code>` or `zevc doc --list`):
  Parser Traps   : P0001..P0010  (instant rewrites for `let`, `->`, `async`, `select`, `match`, ...)
  Static Verifier: E0001..E0024  (isolation, `noyield`/`critical`/`isr`, DMA leases, dual-exit errors)
================================================================================
";

pub const TOPIC_DOCS: &[TopicDoc] = &[
    TopicDoc {
        name: "syntax",
        aliases: &[
            "types", "struct", "structs", "enum", "enums", "packed", "endian", "fn", "functions",
            "switch", "loops",
        ],
        summary: "Explicit scalar/endian types, `packed struct`, `enum`, functions, `switch`, and loops",
        body: "\
# Zev Topic Guide: `syntax`

## 1. Scalar, Endian, Pointer, Slice & Array Types
- Unsigned / Signed: `u8`, `u16`, `u32`, `u64`, `usize`, `i8`, `i16`, `i32`, `i64`, `isize`, `bool`, `void`
- Wire-Endian (valid in `packed struct`): `u16be`, `u16le`, `u32be`, `u32le`, `u64be`, `u64le`
- Compound: fat slices `[]u8` (`.len`, `buf[i]`, clamped `buf[0..5]`), pointers `i32 *p` / `*u8`, arrays `[16]u8`
- Casts & Literals: `raw as i32`, integer separators `170_000_000`, durations `500us`, `1ms`, `5s`

## 2. Wire `packed struct`, In-Memory `struct`, `enum` & `const`
```zev
const u8 CMD_SET_RPM = 1;
const u8 CMD_STOP    = 2;

packed struct CmdFrame {
    u8    op;
    u16be setpoint;   // Decoded from wire big-endian automatically
    u16be crc;
}

struct ByteReader {
    []u8  buf;        // Slices/pointers allowed in normal `struct` (forbidden in `packed struct`)
    usize pos;
}

enum Mode { Idle, ClosedLoop, Fault }
```
Decode a `packed struct` directly from a `[]u8` slice with `CmdFrame.from_bytes(buf[0..5]) catch { ... };`.

## 3. Function Signatures (`Ret name(...)` vs `fn name(...)`)
- Value-returning functions put the return type BEFORE the name (C-style):
  ```zev
  u16 pi_step(u16 target, u16 actual, i32 *integral) {
      i32 err = (target as i32) - (actual as i32);
      *integral += err;
      return target;
  }
  ```
- Void functions use `fn`: `fn setup_clocks() { ... }`
- C FFI prototypes: `extern \"C\" fn log_kv(*u8 key, i32 val);` or `extern \"C\" u32 read_csr();`

## 4. `switch` (Enums AND Integer/`const` Values) & Loops
`switch` dispatches on BOTH `enum` variants AND integer/`bool` literals or named `const` values,
supporting `_` or `default` wildcard arms without accidental C fallthrough:
```zev
fn dispatch(u8 cmd, Mode m) {
    switch (cmd) {
        CMD_SET_RPM => { sys.log_kv(\"cmd=\", 1); }
        CMD_STOP    => { sys.log_kv(\"cmd=\", 2); }
        _           => { sys.log_kv(\"unknown=\", cmd as i32); }
    }

    switch (m) {
        Idle       => { return; }
        ClosedLoop => {
            u32 i = 0;
            while (i < 4) { i += 1; }
        }
        Fault      => { return; }
    }
}
```
Note: Zev v0 has no `let`/`var`, no `match` (use `switch`), and no `for` (use `while`, `loop`, or `every`).
",
    },
    TopicDoc {
        name: "errors",
        aliases: &["err", "error", "catch", "or", "fallible", "result"],
        summary: "Dual-exit `return` vs `error` (`0/1/2` args), multi-stmt `catch`, and `or` fallback",
        body: "\
# Zev Topic Guide: `errors`

## 1. Dual-Exit Register Model (`return` vs `error`)
Normal completion (`return val;`) and failure (`error ...;`) are distinct exits returned in two CPU
registers `(r0, r1)` with zero `Result<T, E>` tag-wrapping boilerplate:

- **Boolean failure (`error` suffix)**:
  ```zev
  u16 read_byte([]u8 buf) error {
      if (buf.len == 0) {
          error;
      }
      return buf[0];
  }
  ```
- **Typed enum failure (`error ErrType` suffix)**:
  ```zev
  enum ParseErr { TooShort, BadCrc }

  u16 parse_setpoint([]u8 buf, u16 expected_crc) error ParseErr {
      if (buf.len < 5) {
          error TooShort;
      }
      CmdFrame f = CmdFrame.from_bytes(buf[0..5]) catch {
          error TooShort;
      };
      if (f.crc != expected_crc) {
          error BadCrc;
      }
      return f.setpoint;
  }
  ```
- **Typed failure + partial progress (`error Reason, partial_val;`)**:
  ```zev
  enum StreamErr { Timeout, Framing }

  u32 read_burst(u32 requested) error StreamErr {
      if (requested > 16) {
          error Timeout, 16;   // Returns partial count `16` in `r0` alongside `Timeout` in `r1`!
      }
      return requested;
  }
  ```

## 2. Symmetric Caller Handling (`or` and Multi-Statement `catch`)
Every fallible call (including `recv(timeout: 5s)`) MUST be handled via `or` or `catch`:
```zev
// 1. Inline default fallback (`or`):
u16 safe_rpm = parse_setpoint(half, 0x1EBF) or 0;
SupEvent ev  = sup_inbox.recv(timeout: 5s) or IdleCheck;

// 2. Diverging `catch` (`continue;`, `break;`, `return;`, or `error ...;`):
u16 rpm = parse_setpoint(half, 0x1EBF) catch (ParseErr err) {
    sup_inbox.post(BadPacket);
    continue;
};

// 3. Multi-statement value-recovering `catch` (trailing expression w/ or w/o `;`):
u32 count = read_burst(32) catch (StreamErr err, u32 completed) {
    recoveries += 1;
    completed + 100
};
```
",
    },
    TopicDoc {
        name: "tasks",
        aliases: &[
            "task", "spawn", "main", "every", "sched", "scheduler", "priority", "priorities",
            "core", "multicore",
        ],
        summary: "Phased `fn main()` boot, reusable `task` blueprints, static `spawn`, and `every` loops",
        body: "\
# Zev Topic Guide: `tasks`

## 1. Reusable `task` Blueprints (100% Static Coroutine Frames)
A `task` is a parameterized blueprint—it consumes zero RAM until instantiated via `spawn` in `main()`.
Each `spawn` site allocates a dedicated compile-time task slot and static coroutine frame (zero `malloc`,
zero per-task RTOS stacks). Locals live across yields are hoisted into the frame automatically;
disjoint child yielding calls (e.g. across `switch` arms) `union`-overlap to save SRAM!

```zev
watch<u16> motor1_rpm = 0;
watch<u16> motor2_rpm = 0;

task motor_control(watch<u16> *target_rpm, u8 pwm_ch) {
    i32 integral = 0;                 // Hoisted into this instance's static task frame

    every 1ms {                       // Drift-free periodic deadline (tick resolution checked!)
        u16 target = target_rpm.read();
        u16 actual = adc_read_rpm(pwm_ch);

        noyield {
            u16 duty = pi_step(target, actual, &integral);  // `duty` stays on CPU stack!
            hw.TIM1.set_compare(pwm_ch, duty);
        }
    }
}
```

## 2. Sequential Phased Boot in `fn main()` + Static `spawn`
`spawn` and `set_priority` are only allowed in `fn main()` so the task table is 100% static:
```zev
fn main() {
    // Phase 1: Deterministic hardware & clock setup before IRQs/tasks start
    setup_clocks();
    sys.use_systick(170_000_000);     // Configures 1ms tick & tickless WFI idle sleep

    // Phase 2: Sequential/yielding power-on self-test before spawning control loops
    run_power_on_self_test();

    // Phase 3: Static multi-instance task instantiation (`@critical` > `@high` > `@normal` > `@low`)
    spawn motor_control(&motor1_rpm, 1) @critical @core(0);
    spawn motor_control(&motor2_rpm, 2) @critical @core(0);

    // Phase 4: Trailing loop in `main()` runs as the `@low` supervisor task!
    set_priority(@low);
    loop {
        checkpoint();
    }
}
```
Starvation bounds (`1ms / 5ms / 15ms / 50ms` across `@critical..@low`) and `Overdue -> Work -> Poll`
dispatch guarantee lower-priority tasks are never starved indefinitely.
",
    },
    TopicDoc {
        name: "yielding",
        aliases: &[
            "yield", "noyield", "critical", "isr", "isrs", "interrupt", "interrupts",
            "checkpoint", "manual_yield", "extern", "ffi",
        ],
        summary: "Call-graph yield inference, `noyield`, `critical`, `isr` rules, and C FFI safety",
        body: "\
# Zev Topic Guide: `yielding`

## 1. Automatic Call-Graph Yield Inference
Zev infers `can_yield` across the whole call graph:
- **Direct yield sites**: `wait_chunk()`, `recv()`, `send()`, `wait()`, `delay(...)`, `every ...`,
  `checkpoint()`, and unbounded `loop { ... }` (which auto-inserts `should_share()` chunking unless
  marked `@manual_yield` or enclosed in `noyield`).
- **Transitive**: Any `fn` calling a yielding `fn` automatically lowers to a stackless coroutine.
- **Non-recursive**: Cycles in the call graph containing a yielding function are rejected (`E0012`).

## 2. Explicit Atomicity Brakes (`noyield`, `critical`, `@manual_yield`)
```zev
// 1. `noyield { ... }` — Task-atomic (forbids any direct or transitive yield; IRQs stay enabled)
noyield {
    u16 duty = pi_step(target, actual, &integral);
    hw.TIM1.set_compare(pwm_ch, duty);
}

// 2. `critical { ... }` — Hardware-atomic (masks IRQs + implies `noyield`; forbids unbounded `loop`)
critical {
    hw.RCC.enable_pll_170mhz();
}

// 3. `@manual_yield` + `checkpoint()` — Suppresses auto-yield while SPI CS is asserted low
@manual_yield
fn spi_burst_transfer([]u8 tx) {
    usize i = 0;
    loop {
        if (i >= tx.len) { break; }
        hw.SPI1.assert_cs_and_send(tx[i]);
        hw.SPI1.deassert_cs();
        checkpoint();                 // Yield ONLY at clean transaction boundaries!
        i += 1;
    }
}
```

## 3. Hardware `isr` & `extern \"C\"` Callback Rules
- `isr <VECTOR> { ... }` is implicitly `noyield` and isolated from raw mutable globals (`E0005`).
- `extern \"C\"` functions are implicitly `noyield`, and any Zev `fn` passed as a callback pointer to
  `extern \"C\"` is statically verified as `noyield` (`E0006`) so a yield never unwinds C stack frames.
",
    },
    TopicDoc {
        name: "primitives",
        aliases: &[
            "prim", "signal", "watch", "inbox", "chan", "channel", "channels", "isolation",
            "globals", "boost",
        ],
        summary: "Static OS primitives (`watch`, `inbox`, `chan`, `signal`) and ISR `boost: @critical`",
        body: "\
# Zev Topic Guide: `primitives`

## 1. Zero Mutable Raw Globals — Static OS Endpoints Only
Top-level mutable variables (`u32 g = 0;`) are rejected (`E0001`/`E0002`). Declare immutable `const`
values or static scheduler-aware OS primitives at module scope and pass pointers (`&ep`) to tasks:

| Primitive | Purpose | Non-Blocking (ISR-Safe) | Yielding (Task-Only) |
|---|---|---|---|
| `watch<T> w = init;` | Latest-value cell | `w.read() -> T`, `w.publish(v)` | `w.wait() -> T` |
| `inbox<T, N> q;` | Active-Object queue | `q.post(v [, boost: @prio])`, `q.try_recv()` | `q.recv([timeout: d]) -> T` |
| `chan<T, N> c;` | Backpressured queue | `c.post(v)`, `c.try_send(v) -> bool` | `c.send(v)`, `c.recv() -> T` |
| `signal s;` | Lightweight wakeup | `s.notify([boost: @prio])`, `s.clear()` | `s.wait([timeout: d])` |

Inspection helpers `.len()`, `.is_empty()`, `.is_full()` are non-blocking on `inbox` and `chan`.

## 2. ISR Priority Elevation (`boost: @critical`) & Sequential Dispatch
An `isr` can wake a low-priority supervisor at elevated priority for a single turn; priority
automatically restores to the task's base priority when that turn loops back to `recv()`:

```zev
import chip.stm32g4 as hw;

enum SupEvent { Overcurrent, BadPacket, IdleCheck }

watch<u16>         motor1_rpm = 0;
inbox<SupEvent, 8> sup_inbox;

isr EXTI0 {
    hw.GPIOA.clear_pin(MOTOR_EN);
    sup_inbox.post(Overcurrent, boost: @critical);   // Elevates `@low` receiver to `@critical`!
}

fn main() {
    set_priority(@low);
    loop {
        SupEvent ev = sup_inbox.recv(timeout: 5s) or IdleCheck;
        switch (ev) {
            Overcurrent => { motor1_rpm.publish(0); }
            BadPacket   => {}
            IdleCheck   => {}
        }
    }
}
```
",
    },
    TopicDoc {
        name: "dma",
        aliases: &[
            "ping_pong", "pingpong", "lease", "slices", "lifetimes", "provenance", "sleep",
            "wakelock",
        ],
        summary: "Zero-copy `@dma_safe ping_pong` & `lease`, zero-`'a` scope/lease safety, sleep clamping",
        body: "\
# Zev Topic Guide: `dma`

## 1. Zero-Copy Circular DMA (`@dma_safe ping_pong<T, N>` & `lease<T, N>`)
- `@dma_safe ping_pong<u8, 64> uart_dma;` allocates aligned static double buffers:
  - In `isr`: `uart_dma.flip();` swaps active/ready halves in O(1) and wakes the waiting task.
  - In `task`: `[]u8 half = uart_dma.wait_chunk();` yields until a half-buffer is ready and lends a
    zero-copy `[]u8` slice directly into the quiescent half.
- `@dma_safe lease<u8, 16> spi_buf;`: single-buffer lend via `.wait_chunk()` and `.release()` / `.complete()`.
- **Automatic Sleep Clamping**: While a `@dma_safe` `ping_pong` or `lease` transfer is active, the
  synthesized kernel automatically clamps tickless idle sleep depth from `DEEP` to `LIGHT` so DMA
  and peripheral clocks stay alive!

## 2. Zero-`'a` Scope-Depth & Lease Provenance Rules (`E0008`..`E0011`)
Zev requires **zero `'a` lifetime annotations**—`zev-sema` tracks origin depth and lease generation:
- **ALLOWED**: Bounds-clamped sub-slicing (`half[0..5]`), wrapping in local structs (`ByteReader{ buf: half, pos: 0 }`),
  and passing slices/wrappers **downward** into helper functions.
- **REJECTED (`E0008`)**: Assigning an inner-scope slice/pointer/wrapper into an outer-scope variable.
- **REJECTED (`E0009`)**: Returning `&local` or a local `wait_chunk()` slice upward out of a function.
- **REJECTED (`E0011`)**: Reading `half` (or any wrapper/sub-slice derived from it) AFTER a second
  `uart_dma.wait_chunk()` swaps the double buffer!

```zev
@dma_safe ping_pong<u8, 64> uart1_dma;
watch<u16> motor1_rpm = 0;

isr DMA1_CH5 {
    uart1_dma.flip();
}

task fieldbus_listener(ping_pong<u8, 64> *rx_dma, watch<u16> *out_rpm) {
    loop {
        []u8 half = rx_dma.wait_chunk();
        if (half.len >= 2) {
            u16 val = ((half[0] as u16) * 256) | (half[1] as u16);
            out_rpm.publish(val);     // Copy decoded scalars out before next `wait_chunk()`!
        }
    }
}
```
",
    },
    TopicDoc {
        name: "runtime",
        aliases: &[
            "io", "std", "sys", "hw", "chip", "irq", "nvic", "print", "println", "uart",
        ],
        summary: "Portable `print`/`println`/`sys.*` built-ins, `hw.*` chip peripherals, and NVIC `isr` map",
        body: "\
# Zev Topic Guide: `runtime`

## 1. Portable Built-In Functions & `sys.*` Namespace (Works Across C99 Sim, Cortex-M3 & ESP32-C6)
- **String & Assertion Built-Ins**:
  - `print(\"text\");` / `println(\"line\");` — Writes a string literal/`*u8` to host `stdout`, Cortex-M3
    `UART0` (`0x4000C000`), or ESP32-C6 `USB_DEVICE` (`/dev/ttyACM0`). (String-only; use `sys.log_kv` for numbers!)
  - `assert(cond);` — Verifies a `bool` invariant (`E0019` if non-`bool`).
  - `delay(10ms);` / `checkpoint();` / `set_priority(@low);` — Scheduler timing, cooperative yield & priority.
- **Portable `sys.*` System & Telemetry Methods** (strictly typechecked; unknown `sys.*`/`sim.*` rejected w/ `E0020`):
  - `sys.use_systick(u32 hz);` — Configures core timer (`12_000_000` on QEMU LM3S, `170_000_000` on STM32G4).
  - `sys.now_ms() -> u32` / `sys.now_ticks() -> u32` — Current monotonic millisecond tick counter.
  - `sys.log(*u8 msg);` — Writes raw message string without newline.
  - `sys.log_kv(*u8 label, i32 val);` — Prints `label` followed by signed decimal `val` + newline!
  - `sys.log_hex(*u8 label, u32 val);` — Prints `label` followed by `0x...` hex `val` + newline!
  - `sys.pend_irq(u32 irq);` — Triggers software/NVIC interrupt `irq` (fires `isr` in QEMU & host sim).
  - `sys.exit(i32 code);` — Cleanly exits host simulation or ARM semihosting QEMU with `code`.

## 2. Virtual SVD Hardware Namespace (`import chip.stm32g4 as hw;`)
- `hw.GPIOA.set_pin(pin)` / `hw.GPIOA.clear_pin(pin)` (built-in `MOTOR_EN` pin constant available)
- `hw.TIM1.set_compare(pwm_ch, duty)`
- `hw.UART0.puts(\"...\")` / `hw.USART1.puts(\"...\")` (writes to `stdout` in host sim and `UART0` in QEMU)
- `hw.DMA1.CH5.clear_irq(HALF_TRANSFER | FULL_TRANSFER)` / `hw.DMA1.bind_circular_rx(ch, usart, &ring)`
- `hw.RCC.enable_pll_170mhz()`, `hw.FLASH.*`, `hw.CRC.*`

## 3. ARM Cortex-M3 NVIC `isr <NAME>` Vector Mapping (`zevc build-elf` / `zevc run-qemu`)
User `isr <NAME> { ... }` blocks are wired directly into `.isr_vector` and auto-enabled in `NVIC_ISER0`:
- `EXTI0` / `GPIOA` -> `IRQ0`  |  `EXTI1` / `GPIOB` -> `IRQ1`  |  `EXTI2`..`EXTI4` -> `IRQ2`..`IRQ4`
- `UART0` / `USART1` / `DMA1_CH5` -> `IRQ5`  |  `UART1` / `USART2` / `DMA1_CH6` -> `IRQ6`
- `SSI0` / `SPI1` / `DMA1_CH1` -> `IRQ7`     |  `I2C0` / `I2C1` / `DMA1_CH2` -> `IRQ8`
- `ADC0` / `ADC1` -> `IRQ14`   |  `TIM0A` / `TIM1` -> `IRQ19`  |  `TIM1A` / `TIM2` -> `IRQ21`
- Explicit `IRQ0`..`IRQ31` names map 1-to-1; call `sys.pend_irq(5)` to trigger `IRQ5` in QEMU!
",
    },
    TopicDoc {
        name: "targets",
        aliases: &[
            "target", "cli", "ram", "sim", "c99", "llvm", "elf", "qemu", "cortex-m", "cortex-m3",
            "esp32", "esp32c6", "riscv",
        ],
        summary: "CLI workflow: `check [--json]`, `ram`, `run`, `run-qemu [--max-ticks]`, and ESP32-C6",
        body: "\
# Zev Topic Guide: `targets`

## 1. Static Verification & Machine-Readable JSON (`zevc check`)
```bash
zevc check examples/motor_controller.zev examples/motor_support.zev
zevc check --json app.zev             # Structured JSON diagnostics for AI agent harnesses
```

## 2. Zero-Heap Static SRAM Report (`zevc ram`)
Computes exact 32-bit MCU and 64-bit host byte counts per spawned task slot, hoisted vs stack-local
`[noyield]` variables, disjoint child `union` bytes saved, OS primitives, and `0 B` dynamic heap:
```bash
zevc ram examples/motor_controller.zev examples/motor_support.zev
```

## 3. Stage 0: Freestanding C99 & Sanitized Host Simulation (`emit-c`, `run`)
```bash
zevc emit-c examples/sim_demo.zev -o build/sim_demo.c
zevc run --sanitize examples/sim_demo.zev
zevc run --max-ticks 50 app.zev       # Bounds infinite `every` / `checkpoint()` firmware loops!
```

## 4. Stage 1A: Native LLVM 23 -> Bare-Metal ARM Cortex-M3 ELF & QEMU
Synthesizes `.isr_vector`, `Reset_Handler`, `SysTick_Handler`, and linker script with zero C/libc.
Supports `-O0`, `-O1`, `-O2` (default), `-O3`, `-Os`, and `-Oz` (`--opt-level <level>`):
```bash
zevc emit-llvm examples/cortex_m_demo.zev -o build/cm3.ll       # Raw readable LLVM IR
zevc emit-llvm -Oz examples/cortex_m_demo.zev -o build/cm3.ll   # `opt` + `minsize` optimized IR
zevc build-elf -Oz examples/cortex_m_demo.zev -o build/cm3.elf
zevc run-qemu -Oz examples/cortex_m_demo.zev
zevc run-qemu --max-ticks 50 app.zev  # Deterministic SysTick cap for non-terminating firmware!
```

## 5. Stage 1B: Bare-Metal Espressif ESP32-C6 (`riscv32imac` Direct-Boot)
Synthesizes `0xAEDB041D` x2 ROM Direct-Boot header, `@_start` at `0x42000008`, watchdog disarm
(`TIMG0`/`TIMG1`/`LP_WDT`/`SWD`), 16 MHz `SYSTIMER`, `USB_DEVICE` serial, and `.iram1` WS2812B driver:
```bash
zevc emit-llvm --target esp32c6 examples/esp32c6_demo.zev -o build/esp32c6.ll
zevc build-esp32c6 -Oz examples/esp32c6_demo.zev -o build/esp32c6_demo.bin
zevc flash-esp32c6 -Oz examples/esp32c6_demo.zev --port /dev/ttyACM0
```
",
    },
];

pub fn render_list() -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Available `zevc doc <topic>` guides:");
    for t in TOPIC_DOCS {
        let _ = writeln!(out, "  {:<12} {}", t.name, t.summary);
    }
    let _ = writeln!(out, "\nAvailable `zevc doc <code>` error explainers:");
    for e in ERROR_DOCS {
        let _ = writeln!(out, "  {:<12} [{}] {}", e.code, e.topic, e.summary);
    }
    out
}

pub fn render_error_doc(e: &ErrorDoc) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "================================================================================"
    );
    let _ = writeln!(out, "  {} — {}", e.code, e.summary);
    let _ = writeln!(out, "  Related Topic Guide: `zevc doc {}`", e.topic);
    let _ = writeln!(
        out,
        "================================================================================\n"
    );
    let _ = writeln!(out, "WHY ZEV ENFORCES THIS:");
    for line in e.why.lines() {
        let _ = writeln!(out, "  {line}");
    }
    let _ = writeln!(out, "\n// BAD (.zev) — triggers {}:", e.code);
    let _ = writeln!(out, "```zev\n{}\n```", e.bad_zev.trim_end());
    let _ = writeln!(out, "\n// GOOD (.zev) — idiomatic zero-heap Zev:");
    let _ = writeln!(out, "```zev\n{}\n```", e.good_zev.trim_end());
    out
}

pub fn lookup_doc(query: Option<&str>, list_all: bool) -> Result<String, String> {
    if list_all {
        return Ok(render_list());
    }

    let Some(raw_q) = query else {
        return Ok(PRIMER_DOC.to_string());
    };

    let q = raw_q.trim();
    if q.is_empty() || q.eq_ignore_ascii_case("primer") || q.eq_ignore_ascii_case("overview") {
        return Ok(PRIMER_DOC.to_string());
    }
    if q == "--list" || q == "-l" || q.eq_ignore_ascii_case("list") {
        return Ok(render_list());
    }

    // 1. Exact or alias topic match (case-insensitive)
    let lower = q.to_ascii_lowercase();
    for topic in TOPIC_DOCS {
        if topic.name == lower || topic.aliases.iter().any(|&a| a == lower) {
            return Ok(topic.body.to_string());
        }
    }

    // 2. Error code match (`P0001`..`P0010`, `E0001`..`E0024`, case-insensitive)
    let upper = q.to_ascii_uppercase();
    for err_doc in ERROR_DOCS {
        if err_doc.code == upper {
            return Ok(render_error_doc(err_doc));
        }
    }

    Err(format!(
        "unknown documentation topic or error code `{q}` (run `zevc doc --list` to see all 8 topics and P0001..P0010 / E0001..E0024 codes)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primer_and_list_render_cleanly() {
        let primer = lookup_doc(None, false).unwrap();
        assert!(primer.contains("ROSETTA STONE"));
        assert!(primer.contains("zevc doc syntax"));
        assert!(primer.contains("zevc doc runtime"));

        let list = lookup_doc(None, true).unwrap();
        assert!(list.contains("runtime"));
        assert!(list.contains("P0001"));
        assert!(list.contains("P0010"));
        assert!(list.contains("E0001"));
        assert!(list.contains("E0024"));
    }

    #[test]
    fn all_topics_and_aliases_resolve() {
        assert_eq!(TOPIC_DOCS.len(), 8);
        for topic in [
            "syntax",
            "errors",
            "tasks",
            "yielding",
            "primitives",
            "dma",
            "runtime",
            "targets",
            "io",
            "std",
            "sys",
            "hw",
            "chip",
            "irq",
            "nvic",
            "print",
            "println",
            "uart",
            "isr",
            "watch",
            "inbox",
            "chan",
            "channels",
            "esp32",
            "esp32c6",
            "qemu",
            "err",
            "catch",
        ] {
            let text = lookup_doc(Some(topic), false)
                .unwrap_or_else(|e| panic!("topic `{topic}` failed: {e}"));
            assert!(text.starts_with("# Zev Topic Guide:"));
        }
    }

    #[test]
    fn all_34_error_codes_resolve_and_good_snippets_pass_parser_and_sema() {
        for n in 1..=10 {
            let code = format!("P{n:04}");
            let lower = code.to_ascii_lowercase();
            let doc = lookup_doc(Some(&lower), false)
                .unwrap_or_else(|e| panic!("missing {code}: {e}"));
            assert!(doc.contains(&code));
        }
        for n in 1..=24 {
            let code = format!("E{n:04}");
            let lower = code.to_ascii_lowercase();
            let doc = lookup_doc(Some(&lower), false)
                .unwrap_or_else(|e| panic!("missing {code}: {e}"));
            assert!(doc.contains(&code));
        }

        assert_eq!(ERROR_DOCS.len(), 34);
        for e in ERROR_DOCS {
            let prog = zev_parser::parse_str_named(e.good_zev, e.code).unwrap_or_else(|pe| {
                panic!(
                    "GOOD snippet for {} failed to parse:\n{}\n---\n{}",
                    e.code, pe, e.good_zev
                )
            });
            zev_sema::analyze_program(&prog).unwrap_or_else(|diags| {
                panic!(
                    "GOOD snippet for {} failed zev-sema:\n{:#?}\n---\n{}",
                    e.code, diags, e.good_zev
                )
            });
        }
    }
}
