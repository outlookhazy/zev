//! Built-in progressive-disclosure documentation engine (`zevc doc`).
//!
//! Designed so an AI coding agent or embedded systems engineer can learn idiomatic Zev
//! in 1 tool call (`zevc doc`), drill into focused ~60-line topic guides (`zevc doc <topic>`),
//! or inspect any parser/verifier error code (`zevc doc P0001`..`P0012`, `E0001`..`E0027`).

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
  2. Boot-Tree `spawn` + Immortal Tasks: Execution starts sequentially in `fn main()`. `task` items
     are permanent actors (`loop` / `loop every <period>`, reset via `restart;`) instantiated statically
     in `main()` or boot-helper `fn`s (`spawn worker(&ep, 1) @high;`). Use `sys.exit(0)` in sim/QEMU.
  3. 3-Tier Yielding + Explicit Brakes: `NonYielding` (`0 B` `.bss`), `SoftYielding` (`checkpoint()`
     compute helpers that auto-specialize a direct stack clone inside `noyield { ... }`), and
     `HardBlocking` (`wait_chunk`, `recv`, `send`, `wait`, `delay`). `critical { ... }` masks IRQs.
  4. Strict Task/ISR Isolation: Zero mutable raw globals. Cross-task/ISR state goes exclusively
     through static primitives: `watch<T>`, unified `inbox<T, N>` (`.post`, `.send`, `.recv`),
     `signal`, `ping_pong<T, N>`, and `lease<T, N>` (with optional ISR `boost: @critical`).
  5. Dual-Exit Errors + `defer`: Normal exit is `return val;`; failure exit is `error [Reason [, val]];`,
     handled via `or <val>` or `catch (...) { ... }`. `defer <stmt>;` runs LIFO across both exits.

ROSETTA STONE (FOREIGN HABIT -> IDIOMATIC ZEV):
  Foreign Habit (Rust / Zig / Go / C)     | Idiomatic Zev (`.zev`)
  ----------------------------------------+-----------------------------------------------------
  `let mut x: u32 = 1;` / `var x = 1;`    | `u32 x = 1;`
  `fn add(a: u16, b: u16) -> u16`         | `u16 add(u16 a, u16 b)`   (`fn init()` = `void`)
  `async fn f()` + `f().await`            | `u16 f()` + `f()`         (3-tier call-graph inferred)
  `select { a = ch1 => ..., b = ch2 }`    | `Ev e = q.recv(timeout: 50ms) or Idle; switch (e)`
  `match ev { A => ..., B => ... }`       | `switch (ev) { A => { ... } _ => { ... } }`
  `Result<u16, E>` / `!u16` / `try f()`   | `u16 f() error E` + `error Bad;` + `catch (E e)` / `or 0`
  `catch |err| { ... }`                   | `catch (ParseErr err) { ... }`
  `for (int i=0; i<n; i++)` / `for i in`  | `for (usize i in 0..n) { ... }` / `for (u8 b in slice)`
  `every 1ms { ... }` (v0.1 bare `every`) | `loop every 1ms { ... }` / `loop every period_ms { ... }`
  `chan<T, N>` (retired separate channel) | `inbox<T, N>` (`.post`, `.try_send`, `.send`, `.recv`)
  `struct R<'a> { b: &'a mut [u8] }`      | `struct R { []u8 b; }` + `i32 *ptr` (0 `'a` syntax)
  `static mut G: u32 = 0;`                | `watch<u32> g = 0;` or `inbox<Ev, 8> q;` (`const` for ROM)
  RAII Drop guard                         | `defer cs_high();` or `defer { ... }` (LIFO on return/error)

SHARDED TOPIC GUIDES (`zevc doc <topic>`):
  zevc doc syntax       Types, `u16be`/`packed struct`, `switch`, `defer`, `for (usize i in 0..n)`
  zevc doc errors       Dual-exit `error` / `error ErrType`, partial progress, multi-stmt `catch`, `or`
  zevc doc tasks        Boot-Tree `spawn`, immortal `task` + `restart;`, `loop every <period_expr>`
  zevc doc yielding     3-tier effects (`SoftYielding` dual-spec), `noyield`, `critical`, `isr`, FFI
  zevc doc primitives   Unified `inbox` (`.post`/`.send`/`.recv`), `watch`, `signal`, `boost: @critical`
  zevc doc dma          `@dma_safe ping_pong`, `lease`, zero-`'a` scope/lease rules, sleep locks
  zevc doc runtime      Portable `println` / `sys.*` (`sys.exit`), `chip.stm32g4` (`hw.*`), NVIC map
  zevc doc targets      `zevc check [--json]`, `ram`, `run`, `run-qemu [--max-ticks]`, ESP32-C6

ERROR CODE EXPLAINERS (`zevc doc <code>` or `zevc doc --list`):
  Parser Traps   : P0001..P0012  (instant rewrites for `let`, `->`, `async`, `for(;;)`, bare `every`, `chan`)
  Static Verifier: E0001..E0027  (isolation, 3-tier yields, DMA leases, unreachable/immortal/boot-spawn)
================================================================================
";

pub const TOPIC_DOCS: &[TopicDoc] = &[
    TopicDoc {
        name: "syntax",
        aliases: &[
            "types", "struct", "structs", "enum", "enums", "packed", "endian", "fn", "functions",
            "switch", "loops", "for", "defer",
        ],
        summary: "Explicit scalar/endian types, `packed struct`, `switch`, bounded `for..in`, and `defer`",
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

## 4. `switch`, Bounded `for..in` Loops & Zero-Cost `defer`
- `switch` dispatches on `enum` variants AND integer/`bool`/`const` values (`_` / `default` wildcard, zero fallthrough).
- Bounded `for..in` loops support half-open `0..n`, inclusive `0..=max` (overflow-safe at `T::MAX`), and slices/arrays
  `for (u8 b in slice)` (while C's 3-clause `for (;;)` and untyped `for i in 0..n` are trapped via `P0008`).
- `defer <stmt>;` and `defer { ... }` execute in strict LIFO order across scope exit, `return`, `error`, `break`,
  `continue`, `restart`, and `sys.exit` (`defer` bodies are implicitly non-yielding).
- Any statement after diverging control flow (`return`, `error`, `restart`, `sys.exit`, or break-less `loop`) is rejected (`E0025`).
```zev
u16 checksum_with_cs([]u8 payload) {
    cs_assert_low();
    defer cs_release_high();

    u16 acc = 0;
    for (u8 b in payload) {
        acc = (acc + (b as u16)) & 0xFFFF;
    }
    for (usize pad in 0..4) {
        acc ^= (pad as u16);
    }
    return acc;
}
```
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
registers `(r0, r1)` with zero `Result<T, E>` tag-wrapping boilerplate (and `defer` statements execute
cleanly on both paths after return/error expressions are evaluated):

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
Every fallible call (including `recv(timeout: 5s)` and `send(v, timeout: 10ms)`) MUST be handled via `or` or `catch`:
```zev
// 1. Inline default fallback (`or`):
u16 safe_rpm = parse_setpoint(half, 0x1EBF) or 0;
SupEvent ev  = sup_inbox.recv(timeout: 5s) or IdleCheck;

// 2. Diverging `catch` (`continue;`, `break;`, `return;`, `restart;`, or `error ...;`):
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
            "task", "spawn", "main", "every", "loop_every", "restart", "sched", "scheduler",
            "priority", "priorities", "core", "multicore",
        ],
        summary: "Boot-Tree `spawn`, immortal `task` blueprints + `restart;`, and `loop every <period>`",
        body: "\
# Zev Topic Guide: `tasks`

## 1. Immortal `task` Blueprints, `loop every <period_expr>` & `restart;`
A `task` is a permanent subsystem actor (`return;` and falling off `}` are compile-time errors `E0026`;
move one-shot finite workflows into yielding `fn`s). Each static `spawn` allocates a dedicated `.bss`
coroutine frame (`0 B` heap, single shared CPU stack):
- **`loop every <period_expr> { ... }`**: Accepts compile-time durations (`1ms`, `20ms`) OR runtime tick
  expressions (`period_ms`), and **skips/resyncs missed deadlines on overrun** (zero catch-up bursts).
  (Bare `every 1ms` is trapped via `P0011`.)
- **`restart;`**: Unwinds active `defer`s, restores any temporary priority boost, zeroes task-local state
  (preserving `spawn` parameters), and jumps cleanly back to task entry.

```zev
watch<u16> motor1_rpm = 0;

task motor_control(watch<u16> *target_rpm, u8 pwm_ch, u32 period_ms) {
    i32 integral = 0;                 // Hoisted into this instance's static task frame

    loop every period_ms {            // Runtime or literal period; overrun-resynced!
        u16 target = target_rpm.read();
        if (target == 0xFFFF) {
            restart;                  // Re-initializes `integral = 0`, keeps `target_rpm`/`pwm_ch`
        }
        u16 actual = adc_read_rpm(pwm_ch);
        noyield {
            u16 duty = pi_step(target, actual, &integral);
            hw.TIM1.set_compare(pwm_ch, duty);
        }
    }
}
```

## 2. Boot-Tree `spawn` & Provenance (`fn main()` + Boot Helpers)
`spawn` is permitted inside `fn main()` (including conditional `if` / `switch` boot-mode branches) AND
inside non-loop, non-recursive boot-helper `fn`s called before `main()`'s supervisor loop (`E0027` if
called inside a `task`, `isr`, loop, or recursion). `zevc ram` prints each slot's full boot provenance
path (`main -> init_drive -> motor_control`):
```zev
fn init_drive(watch<u16> *rpm, u8 ch) {
    spawn motor_control(rpm, ch, 1) @critical @core(0);
}

fn main() {
    setup_clocks();
    sys.use_systick(170_000_000);
    run_power_on_self_test();

    init_drive(&motor1_rpm, 1);

    set_priority(@low);
    loop {
        checkpoint();
    }
}
```
",
    },
    TopicDoc {
        name: "yielding",
        aliases: &[
            "yield", "noyield", "critical", "isr", "isrs", "interrupt", "interrupts",
            "checkpoint", "manual_yield", "extern", "ffi", "soft_yielding",
        ],
        summary: "3-tier function effects (`SoftYielding` dual-spec), `noyield`, `critical`, `isr` & FFI",
        body: "\
# Zev Topic Guide: `yielding`

## 1. Three-Tier Inferred Function Effects (Zero `async`/`.await` Coloring)
`zev-sema` infers one of three effects for every `fn` across the whole call graph:
1. **`NonYielding`**: Contains no yields; compiles to a direct stack call (`0 B` `.bss`) and is legal
   everywhere (`noyield`, `critical`, `isr`, `defer`, and `extern \"C\"` callbacks).
2. **`SoftYielding`**: Terminating compute helpers (e.g. `crc_stream`, `fft`, `sha256`) that only yield
   cooperatively via polite `checkpoint()` calls:
   - In normal `task` flow, runs as a union-packed coroutine sharing the CPU smoothly.
   - Inside `noyield { ... }`, **auto-specializes a direct `0 B`-`.bss` stack clone** (`{fn}__noyield`,
     `checkpoint()` stripped) reported transparently in `zevc ram` and `zevc check --json`!
   - Still **forbidden** inside `critical`, `isr`, `defer`, and `extern \"C\"` callbacks (`E0004`–`E0006`).
3. **`HardBlocking`**: Awaits external events or time (`delay`, `inbox.recv`/`send`, `watch.wait`,
   `signal.wait`, `ping_pong.wait_chunk`, `loop every`, or unbounded `loop`); forbidden inside `noyield`,
   `critical`, `isr`, `defer`, and `extern \"C\"` callbacks (`E0003`–`E0006`).
- Cycles in the call graph containing any yielding function are rejected (`E0012`).

## 2. Explicit Atomicity Brakes (`noyield`, `critical`, `@manual_yield`)
```zev
u32 hash_block([]u8 data) {           // Inferred `SoftYielding` (`checkpoint()` only)
    u32 h = 5381;
    for (u8 b in data) {
        h = (h * 33) ^ (b as u32);
        checkpoint();
    }
    return h;
}

// 1. `noyield { ... }` — Task-atomic (forbids `HardBlocking`; auto-clones `SoftYielding` `hash_block`!)
noyield {
    u32 boot_digest = hash_block(rom_hdr);
}

// 2. `critical { ... }` — Hardware-atomic (masks IRQs; forbids ALL yields & unbounded `loop`)
critical {
    hw.RCC.enable_pll_170mhz();
}

// 3. `@manual_yield` + `checkpoint()` — Suppresses auto-yield while SPI CS is asserted low
@manual_yield
fn spi_burst_transfer([]u8 tx) {
    defer hw.SPI1.deassert_cs();
    for (u8 byte in tx) {
        hw.SPI1.assert_cs_and_send(byte);
        hw.SPI1.deassert_cs();
        checkpoint();                 // Yield ONLY at clean transaction boundaries!
    }
}
```
",
    },
    TopicDoc {
        name: "primitives",
        aliases: &[
            "prim", "signal", "watch", "inbox", "chan", "channel", "channels", "isolation",
            "globals", "boost",
        ],
        summary: "Static OS primitives (unified `inbox`, `watch`, `signal`) and ISR `boost: @critical`",
        body: "\
# Zev Topic Guide: `primitives`

## 1. Zero Mutable Raw Globals — Static OS Endpoints Only
Top-level mutable variables (`u32 g = 0;`) are rejected (`E0001`/`E0002`). Declare immutable `const`
values or static scheduler-aware OS primitives at module scope and pass pointers (`&ep`) to tasks
(`chan<T, N>` has been unified into `inbox<T, N>`, trapped via `P0012`):

| Primitive | Purpose | Non-Blocking (ISR & Task Safe) | Yielding (Task-Only) |
|---|---|---|---|
| `watch<T> w = init;` | Latest-value cell | `w.read() -> T`, `w.publish(v)` | `w.wait() -> T` |
| `inbox<T, N> q;` | Unified bounded queue | `q.post(v [, boost: @prio])`, `q.try_send(v) -> bool`, `q.try_recv()` | `q.send(v [, timeout: d])`, `q.recv([timeout: d]) -> T` |
| `signal s;` | Lightweight wakeup | `s.notify([boost: @prio])`, `s.try_wait() -> bool`, `s.clear()` | `s.wait([timeout: d])` |

Inspection helpers `.len()`, `.is_empty()`, `.is_full()` are non-blocking on `inbox<T, N>`.

## 2. ISR Priority Elevation (`boost: @critical`) & Sequential Dispatch
An `isr` can wake a low-priority supervisor at elevated priority for a single turn; priority
automatically restores to the task's base priority when that turn loops back to `recv()` (or `restart;`):

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
- `@dma_safe ping_pong<u8, 64> uart_dma;` allocates aligned contiguous `2 * N` static double buffers:
  - In `isr`: `uart_dma.flip();` swaps active/ready halves in O(1) and wakes the waiting task.
  - In `task`: `[]u8 half = uart_dma.wait_chunk();` yields until a half-buffer is ready and lends a
    zero-copy `[]u8` slice directly into the quiescent half.
- `@dma_safe lease<u8, 16> spi_buf;`: single-buffer lend via `.wait_chunk()` and `.release()` / `.complete()`.
- **Automatic Sleep Clamping**: While a `@dma_safe` `ping_pong` or `lease` transfer is active, the
  synthesized kernel automatically clamps tickless idle sleep depth from `DEEP` to `LIGHT` so DMA
  and peripheral clocks stay alive!

## 2. Zero-`'a` Scope-Depth & Lease Provenance Rules (`E0008`..`E0011`)
Zev requires **zero `'a` lifetime annotations**—`zev-sema` tracks origin depth and lease generation
(including across `for (u8 b in half)` loops):
- **ALLOWED**: Bounds-clamped sub-slicing (`half[0..5]`), wrapping in local structs (`ByteReader{ buf: half, pos: 0 }`),
  bounded `for (u8 b in half)` iteration, and passing slices/wrappers **downward** into helper functions.
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
  - `sys.exit(i32 code);` — Diverging termination call (runs active LIFO `defer`s, then exits host sim or QEMU).

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
Reports task/primitive counts, tick period, and any dual-specialized `SoftYielding` functions
(`dual_specialized=[...]` in human output, `\"dual_specialized_functions\": [...]` in `--json`):
```bash
zevc check examples/motor_controller.zev examples/motor_support.zev
zevc check --json app.zev             # Structured JSON diagnostics + dual-specialization list
```

## 2. Zero-Heap Static SRAM & Provenance Report (`zevc ram`)
Computes exact 32-bit MCU and 64-bit host byte counts per spawned task slot (including full Boot-Tree
`spawn` provenance `main -> helper -> worker` and `[conditional]` tags), hoisted vs stack-local `[noyield]`
variables, disjoint child `union` bytes saved, OS primitives, dual-specialized `SoftYielding` functions,
and `0 B` dynamic heap:
```bash
zevc ram examples/motor_controller.zev examples/motor_support.zev
```

## 3. Stage 0: Freestanding C99 & Sanitized Host Simulation (`emit-c`, `run`)
```bash
zevc emit-c examples/sim_demo.zev -o build/sim_demo.c
zevc run --sanitize examples/sim_demo.zev
zevc run --max-ticks 50 app.zev       # Bounds infinite `loop every` / `checkpoint()` firmware loops!
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

    // 2. Error code match (`P0001`..`P0012`, `E0001`..`E0027`, case-insensitive)
    let upper = q.to_ascii_uppercase();
    for err_doc in ERROR_DOCS {
        if err_doc.code == upper {
            return Ok(render_error_doc(err_doc));
        }
    }

    Err(format!(
        "unknown documentation topic or error code `{q}` (run `zevc doc --list` to see all 8 topics and P0001..P0012 / E0001..E0027 codes)"
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
        assert!(primer.contains("P0001..P0012"));
        assert!(primer.contains("E0001..E0027"));

        let list = lookup_doc(None, true).unwrap();
        assert!(list.contains("runtime"));
        assert!(list.contains("P0001"));
        assert!(list.contains("P0012"));
        assert!(list.contains("E0001"));
        assert!(list.contains("E0027"));
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
            "for",
            "defer",
            "restart",
            "loop_every",
            "soft_yielding",
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
    fn all_39_error_codes_resolve_and_bad_and_good_snippets_verify() {
        for n in 1..=12 {
            let code = format!("P{n:04}");
            let lower = code.to_ascii_lowercase();
            let doc = lookup_doc(Some(&lower), false)
                .unwrap_or_else(|e| panic!("missing {code}: {e}"));
            assert!(doc.contains(&code));
        }
        for n in 1..=27 {
            let code = format!("E{n:04}");
            let lower = code.to_ascii_lowercase();
            let doc = lookup_doc(Some(&lower), false)
                .unwrap_or_else(|e| panic!("missing {code}: {e}"));
            assert!(doc.contains(&code));
        }

        assert_eq!(ERROR_DOCS.len(), 39);
        for e in ERROR_DOCS {
            // 1. Every GOOD snippet must parse AND pass zev-sema cleanly
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

            // 2. Every BAD snippet must trigger its exact documented error code
            if e.code.starts_with('P') {
                let pe = zev_parser::parse_str_named(e.bad_zev, e.code).expect_err(&format!(
                    "BAD snippet for {} unexpectedly parsed cleanly:\n{}",
                    e.code, e.bad_zev
                ));
                assert_eq!(
                    pe.code,
                    Some(e.code),
                    "BAD snippet for {} triggered wrong parser code:\n{pe}",
                    e.code
                );
            } else {
                let bad_prog =
                    zev_parser::parse_str_named(e.bad_zev, e.code).unwrap_or_else(|pe| {
                        panic!(
                            "BAD snippet for {} failed in parser instead of sema:\n{}\n---\n{}",
                            e.code, pe, e.bad_zev
                        )
                    });
                let diags = zev_sema::analyze_program(&bad_prog).expect_err(&format!(
                    "BAD snippet for {} unexpectedly passed zev-sema:\n{}",
                    e.code, e.bad_zev
                ));
                assert!(
                    diags.iter().any(|d| d.code() == e.code),
                    "BAD snippet for {} did not emit {}; got:\n{:#?}",
                    e.code,
                    e.code,
                    diags
                );
            }
        }
    }
}
