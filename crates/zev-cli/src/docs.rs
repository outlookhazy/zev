//! Built-in progressive-disclosure documentation engine (`zevc doc`).
//!
//! Designed so an AI coding agent or embedded systems engineer can learn idiomatic Zev
//! in 1 tool call (`zevc doc`), drill into focused ~60-line topic guides (`zevc doc <topic>`),
//! or inspect any parser/verifier error code (`zevc doc P0001`..`P0013`, `E0001`..`E0034`).

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
  1. Explicit Types + Bounded Generics: Write `u16 rpm = 1800;`, `f32 kp = 0.4;`, and `[]u8 buf`
     (never `let`/`var`). `struct FixedVec<T, usize N>` and `T clamp<T>(...)` monomorphize at compile time.
  2. Boot-Tree `spawn` + Immortal Tasks: Execution starts sequentially in `fn main()`. `task` items
     are permanent actors (`loop` / `loop every <period>`, reset via `restart;`) instantiated statically
     in `main()` or boot-helper `fn`s (`spawn worker(&ep, 1) @high;`). Use `sys.exit(0)` in sim/QEMU.
  3. 3-Tier Yielding + Explicit Brakes: `NonYielding` (`0 B` `.bss`), `SoftYielding` (`checkpoint()`
     compute helpers that auto-specialize a direct stack clone inside `noyield { ... }`), and
     `HardBlocking` (`wait_chunk`, `recv`, `send`, `wait`, `delay`). `critical { ... }` masks IRQs.
  4. Strict Task/ISR Isolation: Zero mutable raw globals. Cross-task/ISR state goes exclusively
     through static primitives: `watch<T>`, unified `inbox<T, N>` (`.post`, `.send`, `.recv`),
     `signal`, `byte_ring<N>`, `@dma_safe ping_pong<T, N>`, and `@dma_safe pool<T, K>` + `Lease<T>`.
  5. Dual-Exit Errors, `defer` & Zero-Heap `f\"...\"`: Normal exit `return val;`, failure `error [E [, v]];`,
     handled via `or <val>` / `catch (...)`. `println(...)`, `fmt_buf`, & `fmt_ring` use 0 heap.

ROSETTA STONE (FOREIGN HABIT -> IDIOMATIC ZEV):
  Foreign Habit (Rust / Zig / Go / C)     | Idiomatic Zev (`.zev`)
  ----------------------------------------+-----------------------------------------------------
  `let mut x: u32 = 1;` / `var x = 1;`    | `u32 x = 1;` / `f32 gain = 0.25f32;`
  `fn add(a: u16, b: u16) -> u16`         | `u16 add(u16 a, u16 b)`   (`fn init()` = `void`)
  `template<class T, size_t N>` / `Vec<T>`| `struct FixedVec<T, usize N> { [N]T data; usize len; }`
  `async fn f()` + `f().await`            | `u16 f()` + `f()`         (3-tier call-graph inferred)
  `select { a = ch1 => ..., b = ch2 }`    | `Ev e = q.recv(timeout: 50ms) or Idle; switch (e)`
  `match ev { A => ..., B => ... }`       | `switch (ev) { A => { ... } _ => { ... } }`
  `Result<u16, E>` / `!u16` / `try f()`   | `u16 f() error E` + `error Bad;` + `catch (E e)` / `or 0`
  `for (int i=0; i<n; i++)` / `for i in`  | `for (usize i in 0..n) { ... }` / `for (u8 b in slice)`
  `every 1ms { ... }` / `chan<T, N>`      | `loop every 1ms { ... }` / unified `inbox<T, N>`
  `struct R<'a> { b: &'a mut [u8] }`      | `struct R { []u8 b; }` + `*byte_ring` / `*inbox<T>` views
  `println!(\"t={:.2}\", t)` / `snprintf`   | `println(f\"t={t:.2}\");` / `fmt_buf` / `fmt_ring`
  RAII Drop guard                         | `defer cs_high();` or `defer { ... }` (LIFO on return/error)

SHARDED TOPIC GUIDES (`zevc doc <topic>`):
  zevc doc syntax       Types (`f32`/`f64`), `<T, usize N>` generics, `packed struct`, `switch`, `for`/`defer`
  zevc doc errors       Dual-exit `error`, partial progress, `fmt_buf -> []u8 error FmtErr`, `catch` / `or`
  zevc doc tasks        Boot-Tree `spawn`, immortal `task` + `restart;`, `loop every <period_expr>`
  zevc doc yielding     3-tier effects (`SoftYielding` dual-spec), `noyield`, `critical`, `isr`, FFI
  zevc doc primitives   `inbox`, `watch`, `signal`, `byte_ring<N>`, `*byte_ring`/`*inbox<T>` views, subsystems
  zevc doc dma          `@dma_safe pool<T, K>`, `Lease<T>`, `dma_slot<T>`, `ping_pong`, sleep locks
  zevc doc runtime      Zero-heap `f\"...\"` (`fmt_buf`/`fmt_ring`), `import`/`pub`, typed `mmio`, `irq.attach`
  zevc doc targets      `zevc check [--json]`, `ram --target`, `import-svd`, `run`, `run-qemu`, ESP32-C6, AVR, RP2040

ERROR CODE EXPLAINERS (`zevc doc <code>` or `zevc doc --list`):
  Parser Traps   : P0001..P0013  (instant rewrites for `let`, `->`, `async`, `for(;;)`, bare `every`, `chan`, `lease`)
  Static Verifier: E0001..E0034  (isolation, 3-tier yields, DMA/linear `Lease<T>`, `modules`, `mmio`, `irq.attach`, `state`)
================================================================================
";

pub const TOPIC_DOCS: &[TopicDoc] = &[
    TopicDoc {
        name: "syntax",
        aliases: &[
            "types", "struct", "structs", "enum", "enums", "packed", "endian", "fn", "functions",
            "switch", "loops", "for", "defer", "generics", "float", "floats", "modules", "pub",
            "import",
        ],
        summary: "Types (`f32`/`f64`), `<T, usize N>` generics, `packed struct`, `import`/`pub`, `for`/`defer`",
        body: "\
# Zev Topic Guide: `syntax`

## 1. Scalar, Float, Endian, Pointer, Slice & Array Types
- Integers & Bool: `u8`, `u16`, `u32`, `u64`, `usize`, `i8`, `i16`, `i32`, `i64`, `isize`, `bool`, `void`
- IEEE-754 Floats: `f32`, `f64` (`0.25f32`, `2.5e-3`, casts `raw as f32` / `val as i32`)
- Wire-Endian (valid in `packed struct`): `u16be`, `u16le`, `u32be`, `u32le`, `u64be`, `u64le`
- Compound: fat slices `[]u8` (`.len`, `buf[i]`, clamped `buf[0..5]`), pointers `i32 *p` / `*u8`, arrays `[16]u8` / `u8 b[16]`

## 2. Wire `packed struct`, In-Memory `struct`, `enum` & `const`
```zev
const u8 CMD_SET_RPM = 1;
const usize MAX_SAMPLES = 8;

packed struct CmdFrame {
    u8    op;
    u16be setpoint;   // Decoded from wire big-endian automatically
    u16be crc;
}

enum Mode { Idle, ClosedLoop, Fault }
```
Decode a `packed struct` directly from a `[]u8` slice with `CmdFrame.from_bytes(buf[0..5]) catch { ... };`.

## 3. Signatures (`Ret name(...)` vs `fn name(...)`) & Monomorphized `<T, usize N>` Generics
- Value-returning functions put the return type first; `fn` is `void`. User `struct`s and `fn`s support
  compile-time monomorphized `<T>` and `<T, usize N>` parameters (concrete Flash instantiations are reported in
  `zevc ram` and `zevc check --json`; C++ ad-hoc overloading and trait hierarchies are omitted; `E0029` on misuse):
```zev
struct FixedVec<T, usize N> {
    [N]T  data;
    usize len;
}

T clamp<T>(T val, T lo, T hi) {
    if (val < lo) { return lo; }
    if (val > hi) { return hi; }
    return val;
}
// Call with inferred `clamp(x, 0.0f32, 1.0f32)` or explicit `clamp<f32>(...)` / `clamp::<f32>(...)`!
```

## 4. `switch`, Bounded `for..in` Loops & Zero-Cost `defer`
- `switch` dispatches on `enum` variants AND integer/`bool`/`const` values (`_` / `default` wildcard, zero fallthrough).
- Bounded `for..in` loops support `for (usize i in 0..n)`, `0..=max`, and `for (u8 b in slice)` (3-clause `for (;;)` -> `P0008`).
- `defer <stmt>;` and `defer { ... }` execute in strict LIFO order across scope exit, `return`, `error`, `break`,
  `continue`, `restart`, and `sys.exit` (`defer` bodies are implicitly non-yielding; dead code after exits -> `E0025`).
- File/directory modules are **private by default**: prefix top-level exports with `pub` and import via
  `import chip.stm32g4 as hw;` or `import bsp.motor_drive as bsp;` (pass `-I <dir>` for custom roots).
",
    },
    TopicDoc {
        name: "errors",
        aliases: &["err", "error", "catch", "or", "fallible", "result", "fmt_buf"],
        summary: "Dual-exit `return` vs `error`, partial progress (`fmt_buf` `Truncated`), `catch`, and `or`",
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
      if (buf.len < 5) { error TooShort; }
      CmdFrame f = CmdFrame.from_bytes(buf[0..5]) catch { error TooShort; };
      if (f.crc != expected_crc) { error BadCrc; }
      return f.setpoint;
  }
  ```
- **Typed failure + partial progress (`error Reason, partial_val;`) & `fmt_buf`**:
  Built-in `fmt_buf(dst, f\"...\") -> []u8 error FmtErr` proves worst-case buffer fit at compile time when all
  holes are statically bounded (`E0028` if `dst` is too small), and returns `error Truncated, clamped_slice;`
  at runtime when dynamic `[]u8` slices exceed `dst.len`:
  ```zev
  u8 buf[24];
  []u8 line = fmt_buf(buf[..], f\"tag={dyn_slice}\") catch (FmtErr err, []u8 clamped) {
      clamped     // Zero-copy clamped prefix written into `buf` before truncation!
  };
  ```

## 2. Symmetric Caller Handling (`or` and Multi-Statement `catch`)
Every dynamically fallible call (including `recv(timeout: 5s)`, `send(v, timeout: 10ms)`, and `fmt_buf`
when not statically proven to fit at compile time; proven-fit fixed-buffer `fmt_buf` calls need neither
`or` nor `catch`) MUST use `or` or `catch`:
```zev
u16 safe_rpm = parse_setpoint(half, 0x1EBF) or 0;
SupEvent ev  = sup_inbox.recv(timeout: 5s) or IdleCheck;

u16 rpm = parse_setpoint(half, 0x1EBF) catch (ParseErr err) {
    sup_inbox.post(BadPacket);
    continue;
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
            hw.TIM1.ccr[(pwm_ch - 1) as usize] = duty as u32;
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
            "checkpoint", "manual_yield", "extern", "ffi", "soft_yielding", "state",
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
    hw.RCC.cr |= 0x0100_0000;
}

// 3. Parameterized `isr` + ISR-private `state` (isolated `.bss` per Boot-Tree `.attach(...)`)
isr overcurrent_isr(*inbox<SupEv, 8> sup, *hw.GpioRegs gpio) {
    state u32 trips = 0;              // Private static counter scoped to this attachment
    trips += 1;
    gpio.brr = 1 << 0;
    sup.post(Fault, boost: @critical);
}

// 4. `@manual_yield` + `checkpoint()` — Suppresses auto-yield while SPI CS is asserted low
@manual_yield
fn spi_burst_transfer([]u8 tx, *hw.GpioRegs cs_port, *hw.SpiRegs spi) {
    defer cs_port.bsrr = 1 << 4;
    for (u8 byte in tx) {
        spi.dr = byte as u32;
        cs_port.bsrr = 1 << 4;
        checkpoint();                 // Yield ONLY at clean transaction boundaries!
    }
}
```
",
    },
    TopicDoc {
        name: "primitives",
        aliases: &[
            "prim", "signal", "watch", "inbox", "byte_ring", "chan", "channel", "channels",
            "isolation", "globals", "boost", "subsystem", "subsystems", "view", "views",
        ],
        summary: "Static OS primitives, subsystem `struct`s, `*byte_ring`/`*inbox<T>` views, and `boost:`",
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
| `byte_ring<N> r;` | Lock-free SPSC stream / BipBuffer | `r.push(s [, boost:])`, `r.push_byte(b [, boost:])`, `r.grant_write(n)`, `r.commit_write(n [, boost:])`, `r.read_slice()`, `r.consume(n)`, `r.pop(dst)`, `r.try_pop_byte()`, `r.clear()` | `r.wait_read([timeout: d]) -> []u8` |
| `signal s;` | Lightweight wakeup | `s.notify([boost: @prio])`, `s.try_wait() -> bool`, `s.clear()` | `s.wait([timeout: d])` |
| `@dma_safe pool<T, K> p;` | Zero-copy `O(1)` slab (`K<=32`) | `p.try_alloc() -> Lease<T> error`, `p.avail()`, `p.in_use()` | `p.alloc([timeout: d]) -> Lease<T>` |
| `dma_slot<T> s;` | Cross-IRQ `Lease<T>` slot | `s.park(l)`, `s.take() -> Lease<T> error`, `s.release([boost:])`, `s.addr()` | *(park across IRQ/DMA transfer)* |

Inspection helpers `.len()`, `.is_empty()`, `.is_full()` work on both `inbox<T, N>` and `byte_ring<N>` (plus `.cap()` and `.free_space()` on `byte_ring<N>`), while stale `read_slice()`, `wait_read()`, `grant_write()`, or `Lease.get()` slices/pointers are caught at compile time (`E0011`).

## 2. Subsystem `struct`s, Views (`*byte_ring`, `*inbox<T>`, `*watch<T>`) & Singleton Folding
Group a driver's RX/TX rings, pools, slots, queues, or flags inside a module-level `struct` instantiated as a static primitive:
```zev
struct SerialPort {
    byte_ring<64> rx;
    byte_ring<128> tx;
    watch<u32> drops;
}
SerialPort usb_serial;

fn flush_tx(*byte_ring tx) { /* ... works for byte_ring<64>, byte_ring<128>, etc. ... */ }
fn push_evt(*inbox<u32> q, u32 ev) { q.try_send(ev); }
```
- `*byte_ring`, `*inbox<T>`, and `*watch<T>` bundle cleanly into reusable non-subsystem helper structs (`DuplexStream`, `StatusCell<T>`) with zero dynamic vtables and zero `<usize N>` code duplication on Flash.
- Whole-program visibility optimization automatically constant-folds singleton subsystem/task pointers (`instance_count == 1`), strips unread `watch<T>` cells (`0 B` `.bss`, keeping side effects in `.publish(expr)`), and elides IRQ critical sections (`irq_mask_elided`) when a primitive or view family is never accessed from an `isr`.

## 3. ISR Priority Elevation (`boost: @critical`) & Sequential Dispatch
An `isr` can wake a low-priority supervisor at elevated priority for a single turn; priority
automatically restores to the task's base priority when that turn loops back to `recv()` (or `restart;`):

```zev
import chip.stm32g4 as hw;

enum SupEvent { Overcurrent, BadPacket, IdleCheck }

watch<u16>         motor1_rpm = 0;
inbox<SupEvent, 8> sup_inbox;

isr overcurrent_isr(*inbox<SupEvent, 8> sup, *hw.GpioRegs gpio) {
    state u32 trips = 0;
    trips += 1;
    gpio.brr = 1 << hw.PIN_0;
    sup.post(Overcurrent, boost: @critical);   // Elevates `@low` receiver to `@critical`!
}

fn main() {
    hw.IRQ_EXTI0.attach(overcurrent_isr(&sup_inbox, hw.GPIOA));
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
            "ping_pong", "pingpong", "lease", "pool", "dma_slot", "slices", "lifetimes",
            "provenance", "sleep", "wakelock",
        ],
        summary: "Zero-copy `@dma_safe pool<T, K>` + linear `Lease<T>`, `dma_slot<T>`, `ping_pong`, sleep clamping",
        body: "\
# Zev Topic Guide: `dma`

## 1. Zero-Copy Pools (`@dma_safe pool<T, K>`, `Lease<T>`, `dma_slot<T>`) & Circular `ping_pong<T, N>`
- `@dma_safe pool<Frame, 4> net_pool;` allocates an IRQ-safe `O(1)` `u32` bitmap slab (`1 <= K <= 32`)
  whose `@dma_safe` slots align to 32-byte cache lines (`DEEP -> LIGHT` sleep clamping while leased):
  - Acquire a move-only handle via `Lease<Frame> b = net_pool.try_alloc() catch { ... };` (or blocking `net_pool.alloc([timeout:])`).
  - Borrow its payload pointer via `b.get()` (`*Frame`) and address via `b.addr()` (`usize`).
  - Linear ownership (`E0034`): `b` must be moved once into `.release([boost:])`, `defer b.release();`, `slot.park(b)`, or `inbox.post(b)`/`send(b)` (full-queue `inbox.post` and busy `dma_slot.park` auto-recycle `b`!).
  - Standalone `lease<T, N>` is retired (`P0013`); keep `@dma_safe ping_pong<u8, 64>` for contiguous `2 * N` circular DMA (`.flip()` in `isr`, `.wait_chunk()` in `task`).

## 2. Zero-`'a` Scope-Depth & Lease Provenance Rules (`E0008`..`E0011`, `E0034`)
Zev requires **zero `'a` lifetime annotations**—`zev-sema` tracks origin depth and lease generation
(including across `for (u8 b in half)` loops):
- **ALLOWED**: Bounds-clamped sub-slicing (`half[0..5]`), wrapping in local structs (`ByteReader{ buf: half, pos: 0 }`),
  bounded `for (u8 b in half)` iteration, and passing slices/wrappers **downward** into helper functions.
- **REJECTED (`E0008`/`E0009`)**: Escaping an inner-scope slice/`b.get()` pointer upward or to an outer scope.
- **REJECTED (`E0011`)**: Reading `half` (or a `b.get()` pointer) AFTER a second `wait_chunk()` or moving/releasing `b`!

```zev
@dma_safe pool<Frame, 4> tx_pool;
dma_slot<Frame>          tx_inflight;
inbox<Lease<Frame>, 4>   done_q;

isr dma_done_isr(*dma_slot<Frame> slot, *inbox<Lease<Frame>, 4> done) {
    Lease<Frame> f = slot.take() catch { return; };
    done.post(f, boost: @critical);     // Auto-recycles if queue is full!
}

task net_worker() {
    Lease<Frame> buf = tx_pool.try_alloc() catch { sys.exit(1); };
    buf.get().len = 32;
    tx_inflight.park(buf) catch { return; };
    Lease<Frame> done = done_q.recv();
    done.release();
}
```
",
    },
    TopicDoc {
        name: "runtime",
        aliases: &[
            "io", "std", "sys", "hw", "chip", "bsp", "mmio", "irq", "attach", "nvic", "print",
            "println", "uart", "fmt", "format", "fstring", "fmt_ring",
        ],
        summary: "Zero-heap `f\"...\"` (`fmt_buf`/`fmt_ring`), typed `mmio`, `irq.attach`, and `sys.*`",
        body: "\
# Zev Topic Guide: `runtime`

## 1. Zero-Heap Streamable `f\"...\"` Formatting (`print`, `println`, `fmt_buf`, `fmt_ring`)
`f\"...\"` unpacks at compile time into direct per-call-site chunk writes (zero `malloc`, zero runtime
format-string parser, zero global handler registry):
- **Console / UART Sinks**: `print(f\"...\");` and `println(f\"...\");` stream directly to host `stdout`,
  Cortex-M3 `UART0`, ESP32-C6 `USB_DEVICE`, ATmega328P `USART0`, ATmega4809 `USART3`, or RP2040 `USBCTRL` (via `.zev` `console_putc` / `sys.set_putc`).
- **Bounded Buffer Sink**: `fmt_buf(dst, f\"...\") -> []u8 error FmtErr` proves worst-case capacity fit at
  compile time when all holes are bounded (`E0028` if `dst` is too small) and returns dual-exit
  `error Truncated, clamped_slice;` when dynamic `[]u8` slices exceed `dst.len`.
- **Ring Stream Sink**: `fmt_ring(tx_ring, f\"...\" [, boost: @prio]) -> usize error FmtErr` accepts
  `*byte_ring<N>` or capacity-erased `*byte_ring`, emits directly into the SPSC ring with single-wake
  notification, and supports `boost:` for interrupt-driven or buffered serial TX.
- **Supported `{...}` Specifiers** (`{{` and `}}` escape literal braces):
  - `{v}` — Decimal integers, `bool`, `[]u8` / string literals, default `.2` fixed float, and automatic
    `enum` variant names (`.rodata` emitted only for formatted enums!).
  - `{v:x}` / `{v:X}` / `{v:04x}` — Hexadecimal with optional zero-padded width; `{v:b}` / `{v:08b}` — Binary.
  - `{t:.2}` — Compact (~150 B Arduino-style) fixed-decimal `f32`/`f64` (`\"nan\"`, `\"inf\"`, `\"ovf\"`).
  - `{t:e}` / `{t:.3e}` — Scientific float (linked via `--gc-sections` *only* if `:e` appears in program!).
  - `{pkt:?}` — Compile-time struct field debug expansion (`\"CmdFrame { op: 1, ... }\"`).

## 2. Portable Built-Ins, `.zev` BSP Hooks & `sys.*` Namespace
- `assert(cond);` / `delay(10ms);` / `checkpoint();` / `set_priority(@low);`
- `sys.use_systick(u32 hz);` — Configures core timer (`12_000_000` on QEMU LM3S, `125_000_000` RP2040, `170_000_000` STM32G4).
- `sys.tick();` / `sys.tick_ms();` — Advances the 1 ms scheduler tick from a pure-`.zev` timer `isr` (e.g., `Timer0` on Uno, `TCB3` on ATmega4809).
- `pub fn console_putc(u8 c)` / `fn sys_now_ms() -> u32` / `sys.set_putc(fn(u8))` — Pure-`.zev` BSP console & timebase hooks automatically wired into `print`/`println`/`sys.now_ms` so zero board peripheral MMIO lives in Rust/LLVM.
- `sys.now_ms() -> u32` / `sys.now_ticks() -> u32` — Current monotonic millisecond tick counter.
- `sys.log(*u8 msg);` / `sys.log_kv(*u8 label, i32 val);` / `sys.log_hex(*u8 label, u32 val);`
- `sys.pend_irq(u32 irq);` — Triggers software/NVIC interrupt `irq` (fires `isr` in QEMU & host sim).
- `sys.exit(i32 code);` — Diverging termination call (runs active LIFO `defer`s, then exits sim/QEMU or resets RP2040 BootROM on `0x4255`).

## 3. Typed `mmio` Register Blocks & Boot-Tree `irq.attach(...)` / `attach_shared(...)`
- **Core-vs-Library Hardware Drivers**: Chip register maps (`chip.stm32g4`, `chip.lm3s6965`, `chip.esp32c6`)
  and board support (`bsp.*`) are plain `.zev` files (`pub mmio`, `pub irq`, `pub const`). Import them with
  `import chip.stm32g4 as hw;` (or pass `-I <dir>` for custom module directories).
- **Access Enforcement (`E0031`)**: `@rw` allows read/write/RMW (`|=`), `@ro` read-only, `@wo` write-only,
  `@w1c` read or direct `=` only (forbids `|=`/`&=` read-modify-write). Widths: `u8`, `u16`, `u32`, arrays `[4]u32`.
```zev
pub mmio DmaRegs {
    @ro  u32 isr  @ 0x00;
    @w1c u32 ifcr @ 0x04;
}
pub mmio DMA1 : DmaRegs @ 0x4002_0000;
pub irq IRQ_DMA1_CH5 = 15;

isr dma_rx_isr(*DmaRegs dma, u32 clr) {
    state u32 count = 0;       // Isolated per-attachment `.bss` static variable
    count += 1;
    dma.ifcr = clr;
}
// Inside boot-tree `fn main()`: `IRQ_DMA1_CH5.attach(dma_rx_isr(DMA1, 1 << 18));`
```
- **Target Vector Mapping**: Boot-Tree `vec.attach(...)` and `vec.attach_shared(...)` synthesize Flash
  `.isr_vector` + `NVIC_ISER0..7` on Cortex-M3 and `INTMTX` + `INTPRI` + `.iram1` `mtvec` on ESP32-C6.
",
    },
    TopicDoc {
        name: "targets",
        aliases: &[
            "target", "cli", "ram", "sim", "c99", "llvm", "elf", "qemu", "cortex-m", "cortex-m3",
            "esp32", "esp32c6", "riscv", "avr", "atmega328p", "uno", "arduino", "atmega4809",
            "nano-every", "nano", "updi", "rp2040", "pico", "pico-w", "cortex-m0plus", "uf2",
            "svd", "import-svd",
        ],
        summary: "CLI workflow: `check [--json]`, `ram`, `import-svd`, `run`, `run-qemu`, ESP32-C6, AVR, and RP2040",
        body: "\
# Zev Topic Guide: `targets`

## 1. Static Verification & Machine-Readable JSON (`zevc check`)
Reports task/primitive counts, tick period, dual-specialized `SoftYielding` functions, and concrete
monomorphized generic instances (`dual_specialized_functions` & `monomorphized_instances` in `--json`):
```bash
zevc check examples/motor_controller.zev examples/motor_support.zev
zevc check --json app.zev             # Structured JSON diagnostics + dual-spec & monomorphized lists
```

## 2. SVD Hardware Generation & Static SRAM + Stack Proofs (`zevc import-svd`, `zevc ram`)
Generate typed `.zev` `pub mmio` layouts, peripheral base addresses, and `pub irq` tokens directly
from any vendor CMSIS-SVD file, inspect exact 32-bit/64-bit `.bss`/ISR `state`/union RAM budgets
(with singleton constant-folding `[singleton -> folded]`, dead-`watch` stripping & IRQ-mask elision telemetry), or run a hardware single-stack +
`.bss` high-water proof (`--target <cortex-m3|esp32c6|avr|nano-every|rp2040>` using LLVM `.stack_sizes` / AVR disassembly):
```bash
zevc import-svd vendor/STM32G474.svd -o chip/stm32g474.zev
zevc check -I chip examples/motor_controller.zev examples/motor_support.zev
zevc ram examples/motor_controller.zev examples/motor_support.zev
zevc ram --target esp32c6 examples/esp32c6_demo.zev
zevc ram --target avr examples/arduino_uno_demo.zev
zevc ram --target nano-every examples/arduino_nano_every_demo.zev
zevc ram --target rp2040 examples/pico_w_demo.zev
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
Synthesizes `0xAEDB041D` x2 ROM Direct-Boot header, `@_start` at `0x42000008`, and `INTMTX`/`mtvec`;
watchdog disarm (`TIMG0`/`TIMG1`/`LP_WDT`/`SWD`), 16 MHz `SYSTIMER` (`sys_now_ms`), `USB_DEVICE` (`console_putc`), and RMT WS2812B (GPIO8) all live in pure `.zev`:
```bash
zevc emit-llvm --target esp32c6 examples/esp32c6_demo.zev -o build/esp32c6.ll
zevc build-esp32c6 -Oz examples/esp32c6_demo.zev -o build/esp32c6_demo.bin
zevc flash-esp32c6 -Oz examples/esp32c6_demo.zev --port /dev/ttyACM0
```

## 6. Stage 1C: Bare-Metal Microchip ATmega328P / Arduino Uno R3 (`avr` Optiboot)
Synthesizes the 26-slot Flash vector table, `.data`/`.rodata`/`.bss` startup, `SREG`, and `avr-libgcc` shims;
WDT disarm, `Timer0` 1 kHz `sys.tick()` ISR, and `115200 8N1` `USART0` (`console_putc`) live in pure `.zev`:
```bash
zevc emit-llvm --target avr examples/arduino_uno_demo.zev -o build/uno.ll
zevc build-avr examples/arduino_uno_demo.zev -o build/uno_demo.bin
zevc flash-avr examples/arduino_uno_demo.zev --port /dev/ttyACM0
```

## 7. Stage 1D: Bare-Metal Microchip ATmega4809 / Arduino Nano Every (`avrxmega3` UPDI)
Synthesizes pure-ISA 40-slot Flash vectors + `.data`/`.bss` startup (zero peripheral MMIO in Rust), while
`bsp.init_board()` configures `CPU.ccp` WDT/clock unlock, `FUSE.OSCCFG` 16/20 MHz auto-calibration,
`SLPCTRL`, `isr systick_isr` (`TCB3` 1 kHz `sys.tick()`), and `pub fn console_putc(u8)` (`115200 8N1` `USART3`) in `.zev`:
```bash
zevc emit-llvm --target nano-every examples/arduino_nano_every_demo.zev -o build/nano.ll
zevc build-nano-every examples/arduino_nano_every_demo.zev -o build/nano_every_demo.bin
zevc flash-nano-every examples/arduino_nano_every_demo.zev --port /dev/ttyACM0
```

## 8. Stage 1E: Bare-Metal Raspberry Pi Pico / Pico W (`rp2040` SRAM UF2)
Synthesizes ARMv6-M `.entry` (`0x2000_0000`), `[48 x ptr]` `.isr_vector`, `SysTick`, and BootROM `0x4255` reset;
125 MHz `PLL_SYS`/48 MHz `PLL_USB` clocks and Full-Speed USB 1.1 CDC-ACM (`VID=0x2E8A`, `PID=0x000A`, `1200`-baud `RPI-RP2` auto-reset, `console_putc`) are written 100% in `.zev`:
```bash
zevc emit-llvm --target rp2040 examples/pico_w_demo.zev -o build/pico_w.ll
zevc build-rp2040 examples/pico_w_demo.zev -o build/pico_w_demo.uf2
zevc flash-rp2040 examples/pico_w_demo.zev --port /dev/ttyACM2
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

    // 2. Error code match (`P0001`..`P0012`, `E0001`..`E0033`, case-insensitive)
    let upper = q.to_ascii_uppercase();
    for err_doc in ERROR_DOCS {
        if err_doc.code == upper {
            return Ok(render_error_doc(err_doc));
        }
    }

    Err(format!(
        "unknown documentation topic or error code `{q}` (run `zevc doc --list` to see all 8 topics and P0001..P0013 / E0001..E0034 codes)"
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
        assert!(primer.contains("P0001..P0013"));
        assert!(primer.contains("E0001..E0034"));

        let list = lookup_doc(None, true).unwrap();
        assert!(list.contains("runtime"));
        assert!(list.contains("P0001"));
        assert!(list.contains("P0013"));
        assert!(list.contains("E0001"));
        assert!(list.contains("E0034"));
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
            "pool",
            "dma_slot",
            "runtime",
            "targets",
            "generics",
            "floats",
            "fmt",
            "fmt_buf",
            "fmt_ring",
            "subsystem",
            "views",
            "ram",
            "avr",
            "nano-every",
            "atmega4809",
            "updi",
            "rp2040",
            "pico",
            "for",
            "defer",
            "restart",
            "loop_every",
            "soft_yielding",
            "modules",
            "pub",
            "import",
            "io",
            "std",
            "sys",
            "hw",
            "chip",
            "bsp",
            "mmio",
            "irq",
            "attach",
            "nvic",
            "state",
            "svd",
            "import-svd",
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
    fn all_47_error_codes_resolve_and_bad_and_good_snippets_verify() {
        for n in 1..=13 {
            let code = format!("P{n:04}");
            let lower = code.to_ascii_lowercase();
            let doc = lookup_doc(Some(&lower), false)
                .unwrap_or_else(|e| panic!("missing {code}: {e}"));
            assert!(doc.contains(&code));
        }
        for n in 1..=34 {
            let code = format!("E{n:04}");
            let lower = code.to_ascii_lowercase();
            let doc = lookup_doc(Some(&lower), false)
                .unwrap_or_else(|e| panic!("missing {code}: {e}"));
            assert!(doc.contains(&code));
        }

        assert_eq!(ERROR_DOCS.len(), 47);
        for e in ERROR_DOCS {
            // 1. Every GOOD snippet must parse, monomorphize & pass zev-sema, AND lower cleanly
            //    through zev-mir, zev-codegen-c (including clang -std=c99 syntax check), and zev-codegen-llvm!
            let prog = zev_parser::parse_str_named(e.good_zev, e.code).unwrap_or_else(|pe| {
                panic!(
                    "GOOD snippet for {} failed to parse:\n{}\n---\n{}",
                    e.code, pe, e.good_zev
                )
            });
            let (mono_progs, _) = zev_sema::analyze_and_monomorphize_programs(
                &[prog],
                &zev_sema::AnalyzeOptions::default(),
            )
            .into_result()
            .unwrap_or_else(|diags| {
                panic!(
                    "GOOD snippet for {} failed zev-sema:\n{:#?}\n---\n{}",
                    e.code, diags, e.good_zev
                )
            });
            let mir = zev_mir::lower_programs(&mono_progs).unwrap_or_else(|me| {
                panic!(
                    "GOOD snippet for {} failed zev-mir lowering:\n{}\n---\n{}",
                    e.code, me, e.good_zev
                )
            });
            let c_src = zev_codegen_c::emit_c(&mir);
            assert!(!c_src.is_empty(), "empty C99 output for {}", e.code);
            let mut clang = std::process::Command::new("/bin/clang")
                .args([
                    "-std=c99",
                    "-fsyntax-only",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-Wno-unused-function",
                    "-Wno-unused-variable",
                    "-Wno-unused-parameter",
                    "-Wno-unused-but-set-variable",
                    "-x",
                    "c",
                    "-",
                ])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .expect("spawn /bin/clang for doc snippet verification");
            {
                use std::io::Write as _;
                clang
                    .stdin
                    .as_mut()
                    .unwrap()
                    .write_all(c_src.as_bytes())
                    .unwrap();
            }
            let clang_out = clang.wait_with_output().unwrap();
            assert!(
                clang_out.status.success(),
                "GOOD snippet for {} emitted invalid C99:\n{}\n---\n{}",
                e.code,
                String::from_utf8_lossy(&clang_out.stderr),
                e.good_zev
            );

            let ll_src = zev_codegen_llvm::emit_llvm_ir(&mir).unwrap_or_else(|le| {
                panic!(
                    "GOOD snippet for {} failed zev-codegen-llvm:\n{}\n---\n{}",
                    e.code, le, e.good_zev
                )
            });
            assert!(!ll_src.is_empty(), "empty LLVM IR output for {}", e.code);

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
