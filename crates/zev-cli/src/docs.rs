//! Built-in progressive-disclosure documentation engine (`zevc doc`).
//!
//! Designed so an AI coding agent or embedded systems engineer can learn idiomatic Zev
//! in 1 tool call (`zevc doc`), drill into focused ~60-line topic guides (`zevc doc <topic>`),
//! or inspect any parser/verifier error code (`zevc doc P0001`..`P0018`, `E0001`..`E0034`).

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
  ZEV — QUICK PRIMER & AGENT REFERENCE (`zevc doc`)
================================================================================

CORE MENTAL MODEL (5 RULES):
  1. Explicit Types + Bounded Generics: Write `u16 rpm = 1800;`, `f32 kp = 0.4;`, `*u8 p`, `[]u8 buf`,
     and `[16]u8 arr` (never `let`/`var` or postfix `*`/`[]`). `<T, usize N>` monomorphizes statically.
  2. Boot-Tree `spawn` + Immortal Tasks (<= 32 slots): Execution starts sequentially in `fn main()`. `task`
     items are permanent actors (`loop` / `loop every <period>`, reset via `restart;`) instantiated
     statically in `main()` or boot helpers (`spawn worker(&ep, 1) @high;`). Use `sys.exit(0)` in sim/QEMU.
  3. 3-Tier Yielding + Explicit Brakes: `NonYielding` (`0 B` `.bss`), `SoftYielding` (`checkpoint()` and
     unguarded `while (cond)` compute loops that auto-specialize direct clones in `noyield { ... }`),
     and `HardBlocking` (`wait { ... }`, `recv`, `send`, `wait`, `wait_change`, `wait_chunk`, `delay`).
  4. Strict Task/ISR Isolation: Zero mutable raw globals. Cross-task/ISR state goes exclusively through
     static primitives: `watch<T>` (enforced single-writer), `guarded<T>` (multi-writer inside `critical`),
     unified `inbox<T, N>` (`.post`, `.send`, `.recv`), `signal`, `byte_ring<N>`, `wait_queue` + `std.sync`
     (`Semaphore`, `EventFlags`, `Latch`, `Barrier<N>`), `@dma_safe ping_pong<T, N>`, and
     `@dma_safe pool<T, K>` + `Lease<T>`.
  5. Dual-Exit Errors, `defer`, Zero-Heap `f\"...\"` & Defined Safety: Normal exit `return val;`, failure
     `error [E [, v]];` (`or`/`catch`). Out-of-bounds indices/subslices/MMIO arrays and div/rem by zero
     trap deterministically (`sys.set_fault_handler`), `INT_MIN / -1` evaluates to `INT_MIN` (`% -1` -> `0`), and
     constant overshifts fail at compile time while variable shifts mask by `(bit_width - 1)`.

ROSETTA STONE (FOREIGN HABIT -> IDIOMATIC ZEV):
  Foreign Habit (Rust / Zig / Go / C)     | Idiomatic Zev (`.zev`)
  ----------------------------------------+-----------------------------------------------------
  `let mut x: u32 = 1;` / `var x = 1;`    | `u32 x = 1;` / `f32 gain = 0.25f32;`
  `fn add(a: u16, b: u16) -> u16`         | `fn add(u16 a, u16 b) -> u16` (`fn init()` = `void`)
  `uint8_t *p;` / `uint8_t buf[16];`      | `*u8 p;` / `[16]u8 buf;`  (canonical prefix syntax)
  `template<class T, size_t N>` / `Vec<T>`| `struct FixedVec<T, usize N> { [N]T data; usize len; }`
  `async fn f()` + `f().await`            | `fn f() -> u16` + `f()`       (3-tier call-graph inferred)
  `select { a = ch1 => ..., b = ch2 }`    | `wait { Ev a = q1.recv() => ..., timeout 50ms => ... }`
  `match ev { A => ..., B => ... }`       | `switch (ev) { A => { ... } _ => { ... } }`
  `Result<u16, E>` / `!u16` / `try f()`   | `fn f() -> u16 error E` + `error Bad;` + `catch (E e)` / `or 0`
  `for (int i=0; i<n; i++)` / `for i in`  | `for (usize i in 0..n) { ... }` / `for (u8 b in slice)`
  `every 1ms { ... }` / `chan<T, N>`      | `loop every 1ms { ... }` / unified `inbox<T, N>`
  `struct R<'a> { b: &'a mut [u8] }`      | `struct R { []u8 b; }` + `*byte_ring` / `*inbox<T>` views
  `println!(\"t={:.2}\", t)` / `snprintf`   | `println(f\"t={t:.2}\");` / `fmt_buf` / `fmt_ring`
  RAII Drop guard                         | `defer cs_high();` or `defer { ... }` (LIFO on return/error)

SHARDED TOPIC GUIDES (`zevc doc <topic>`):
  zevc doc tutorial     Step-by-step walkthrough: `main()`, `task`, `watch`/`inbox`, `isr`, `ram`, boards
  zevc doc syntax       Types (`f32`/`f64`), `<T, usize N>` generics, `packed struct`, `switch`, `for`/`defer`
  zevc doc errors       Dual-exit `error`, partial progress, `fmt_buf -> []u8 error FmtErr`, `catch` / `or`
  zevc doc tasks        Boot-Tree `spawn` (<= 32), immortal `task` + `restart;`, `loop every`, run bounds
  zevc doc yielding     3-tier effects (`while (cond)` auto-checkpoint & `SoftYielding`), `noyield`, `critical`, `isr`
  zevc doc primitives   `wait { ... }`, `inbox`, `watch`, `signal`, `byte_ring`, `wait_queue` / `std.sync`, views
  zevc doc dma          `@dma_safe pool<T, K>`, `Lease<T>`, `dma_slot<T>`, `ping_pong`, sleep locks
  zevc doc runtime      Zero-heap `f\"...\"`, `import`/`pub`, `mmio`, `irq.attach`, runtime traps, `sys.fuzz_irqs`
  zevc doc targets      `zevc check [--json]`, `ram --board`, `import-svd`, `run --sanitize`, `run-qemu`, ESP32-C6, AVR, RP2040

ERROR CODE EXPLAINERS (`zevc doc <code>` or `zevc doc --list`):
  Parser Traps   : P0001..P0018  (rewrites for `let`, `:`, `async`, `select`, `for(;;)`, `every`, `chan`, `lease`, postfix `*`/`[]`, `->`/`:`/`(T)x`/`Ret fn` casts & decls)
  Static Verifier: E0001..E0034  (isolation, 3-tier yields, DMA/linear `Lease<T>`, `modules`, `mmio`, `irq.attach`, `state`)
================================================================================
";

pub const TOPIC_DOCS: &[TopicDoc] = &[
    TopicDoc {
        name: "tutorial",
        aliases: &["guide", "getting-started", "walkthrough", "quickstart"],
        summary: "Step-by-step walkthrough from `fn main()` simulation to `task`, `isr`, `ram`, and hardware builds",
        body: "\
# Zev Topic Guide: `tutorial`

## Step 1 — Minimal Host Simulation (`zevc run --sanitize`)
Every Zev program starts sequentially inside `fn main()`. Test pure computation first using `zevc check hello.zev`
and `zevc run --sanitize hello.zev` (compiles via LLVM 23 with AddressSanitizer + UBSan and exits via `sys.exit(0)`):
```zev
fn scale_mv(u16 raw) -> u32 {
    return ((raw as u32) * 3300u32) / 4095u32;
}

fn main() {
    u32 mv = scale_mv(2048u16);
    println(f\"sample={mv}mV\");
    sys.exit(0);
}
```

## Step 2 — Spawning a Periodic `task` & Reading Telemetry (`watch<T>`)
Move concurrent work into immortal `task` blueprints instantiated from `main()`'s boot tree via `spawn`.
Cross-task state uses scheduler-tracked primitives instead of raw globals: a `watch<T>` has one enforced writer
and any number of readers (`loop every 10ms` stays drift-free without catch-up bursts):
```zev
watch<u32> tick_count = 0;

task blinker() {
    u32 n = 0;
    loop every 10ms {
        n += 1;
        tick_count.publish(n);
    }
}

task monitor() {
    loop every 25ms {
        u32 cur = tick_count.read();
        println(f\"count={cur}\");
        if (cur >= 3u32) {
            sys.exit(0);
        }
    }
}

fn main() {
    spawn blinker() @high;
    spawn monitor() @low;
}
```
Run `zevc ram app.zev` to inspect each spawned task's static `.bss` frame size and worst-case non-yielding run bound.

## Step 3 — Wiring an Interrupt (`isr` + `irq.attach`) & Multi-Source `wait { ... }`
Hardware `isr` handlers stay fast and non-blocking (`E0005` forbids yields), waking consumer tasks via `inbox<T, N>`
or `signal` (`boost: @high` temporarily elevates the waking task for that single turn):
```zev
irq IRQ_BTN = 6;
inbox<u8, 4> btn_q;

isr btn_isr(*inbox<u8, 4> q) {
    state u8 clicks = 0;
    clicks += 1;
    q.post(clicks, boost: @high);
}

task controller() {
    loop {
        wait {
            u8 c = btn_q.recv() => { println(f\"click={c}\"); sys.exit(0); }
            timeout 50ms => { println(\"idle\"); }
        }
    }
}

fn main() {
    IRQ_BTN.attach(btn_isr(&btn_q));
    spawn controller() @normal;
    sys.pend_irq(6);
}
```

## Step 4 — Building for Real Boards & `zev.toml`
Use `import board;` and `hal.*` for portable firmware, then target hardware boards cleanly:
- `zevc ram --board nucleo_f334r8 app.zev` — verify exact flash/RAM `.bss` + static call-stack bounds.
- `zevc build --board nucleo_f334r8 app.zev -o app.bin` — emit `.bin` / `.uf2` / `.elf` with zero heap symbols.
- Put `[project]` defaults (`entry`, `board`, `port`, `opt_level`) in `zev.toml` to run `zevc build` / `zevc flash` directly.
",
    },
    TopicDoc {
        name: "syntax",
        aliases: &[
            "types", "struct", "structs", "enum", "enums", "packed", "endian", "fn", "functions",
            "switch", "loops", "for", "defer", "generics", "float", "floats", "modules", "pub",
            "import", "impl", "methods", "const",
        ],
        summary: "Types (`f32`/`f64`), `<T, usize N>` generics, `impl` methods, `.rodata` `const`, `import`/`pub import`",
        body: "\
# Zev Topic Guide: `syntax`

## 1. Scalar, Float, Endian, Pointer, Slice & Array Types
- Integers & Bool: `u8`, `u16`, `u32`, `u64`, `usize`, `i8`, `i16`, `i32`, `i64`, `isize`, `bool`, `void`
- IEEE-754 Floats: `f32`, `f64` (`0.25f32`, `2.5e-3`, casts `raw as f32` / saturating `val as i32`)
- Wire-Endian (valid in `packed struct`): `u16be`, `u16le`, `u32be`, `u32le`, `u64be`, `u64le`
- Compound: fat slices `[]u8` (`.len`, `buf[i]`, `buf[0..5]`), pointers `*i32 p` / `*u8`, arrays `[16]u8`
- Defined Arithmetic & Conversions: integer `+`/`-`/`*` wrap in two's complement; constant shifts outside
  `0..bit_width` fail at compile time (`E0024`) while variable shifts mask by `(bit_width - 1)`; `/`/`%` by `0`
  and out-of-bounds indexing/subslicing trap deterministically (`sys.set_fault_handler`); uninitialized locals zero-initialize.
- Lossless Integer Coercion & Destination-Typed Arithmetic: implicit integer conversions are strictly
  lossless (unsigned -> wider unsigned / strictly wider signed; signed -> wider signed; wire-endian
  `u16be`/`u32le`... <-> same-width native). Narrowing and signed<->unsigned require explicit `as`.
  `usize`/`isize` conversions with fixed widths stay portable (`u8`/`u16` -> `usize` -> `u64`,
  `i8`/`i16`/`u8` -> `isize` -> `i64`), while `usize`/`isize` literals and constant expressions are
  range-checked against the selected target's pointer width (`16` on AVR, `32` on Cortex-M and RISC-V,
  `64` on host; plain `zevc check` with no `--board`/`--chip` uses `host`). Literals are range-checked
  against their destination (`u8 x = 300;` -> `E0019`), incompatible binary/comparison operands
  (`u32 < i32`) error, and arithmetic expressions widened to a destination compute at that
  destination width (`u16 w = a + b;` with `u8 a = 200, b = 100` -> `300`, whereas `(a + b) as u16` -> `44`).

## 2. Wire `packed struct`, In-Memory `struct`, `enum` & `.rodata` `const`
```zev
const u8 CMD_SET_RPM = 1;
const [4]u16 LUT_DUTY = [150u16, 400u16, 700u16, 900u16]; // Lives in Flash .rodata

packed struct CmdFrame {
    u8    op;
    u16be setpoint;   // Decoded from wire big-endian automatically
    u16be crc;
}

enum Mode { Idle, ClosedLoop, Fault }
```
Decode a `packed struct` directly from a `[]u8` slice with `CmdFrame.from_bytes(buf[0..5]) catch { ... };`.
Aggregate `const` structs/arrays live in `.rodata` and support direct field reads, indexing, and
read-only slicing (`LUT_DUTY[0..2]`); taking `&CONST.field` or calling a `*self` method on a `const`
value is rejected at compile time.

## 3. Same-Module `impl` Blocks, `fn name(...) [-> Ret]` Signatures & `<T, usize N>` Generics
- User `struct`s and `fn`s support compile-time monomorphized `<T>` and `<T, usize N>` parameters (`zevc ram`
  and `zevc check --json` report concrete Flash instantiations; `E0029` on misuse).
- `<...>` disambiguation is purely syntactic: after `ident` or `module.ident` in expression position, `<...>`
  parses as generic arguments only when immediately followed by `(` or `{` (where struct literals are allowed);
  otherwise it parses as `<` comparison (parenthesize multi-arg comparisons such as `f((a < b), (c > (d)))`).
  Constant-generic arguments in expression position accept literals (`4`, `true`), identifiers (`N`), or
  parenthesized expressions (`(N + 1)`); type position accepts full constant expressions (`[N + 1]T`,
  `FixedVec<u32, N + 1>`). At statement start, `Name<...> ident` is always a variable declaration.
- Same-module `impl` blocks attach methods (including partial specializations like `impl<usize N> FixedVec<f32, N>`)
  to types declared in the same file (`pub` goes on individual methods, never `pub impl`):
  - `self`: pass-by-value receiver (valid on both variables and `.rodata` `const` descriptors)
  - `*self`: auto-borrowed mutable pointer receiver (`v.push(x)` borrows `&v`; rejected on `const`)
  - No `self` parameter: associated constructor/helper called as `Type.make(...)`
```zev
struct FixedVec<T, usize N> {
    [N]T  data;
    usize len;
}

impl<T, usize N> FixedVec<T, N> {
    pub fn push(*self, T item) {
        if (self.len < N) { self.data[self.len] = item; self.len += 1; }
    }
}

impl<usize N> FixedVec<f32, N> {
    pub fn first_or_zero(*self) -> f32 {
        if (self.len == 0) { return 0.0f32; }
        return self.data[0];
    }
}

fn clamp<T>(T val, T lo, T hi) -> T {
    if (val < lo) { return lo; }
    if (val > hi) { return hi; }
    return val;
}
```

## 4. `switch`, Bounded `for..in`, `defer`, Grouped `import` & `pub import`
- `switch` dispatches on `enum` variants AND integer/`bool`/`const` values (`_` / `default` wildcard, zero fallthrough).
- Bounded `for..in` loops support `for (usize i in 0..n)`, `0..=max`, and `for (u8 b in slice)` (3-clause `for (;;)` -> `P0008`).
- `defer <stmt>;` and `defer { ... }` execute in strict LIFO order across scope exit, `return`, `error`, `break`,
  `continue`, `restart`, and `sys.exit` (`defer` bodies are implicitly non-yielding; dead code after exits -> `E0025`).
- File/directory modules are **private by default**: prefix exports with `pub`, group imports cleanly via
  `import hal.{time, gpio, pwm};`, and re-export child modules/symbols via `pub import chip.stm32f3.{rcc, gpio};`
  (pass `-I <dir>` for custom roots).
",
    },
    TopicDoc {
        name: "errors",
        aliases: &["err", "error", "catch", "or", "fallible", "result", "fmt_buf"],
        summary: "Dual-exit `return` vs `error`, partial progress (`fmt_buf` `Truncated`), `catch`, and `or`",
        body: "\
# Zev Topic Guide: `errors`

## 1. Dual-Exit Model (`return` vs `error`)
Normal completion (`return val;`) and failure (`error ...;`) are distinct exits with zero `Result<T, E>`
tag-wrapping boilerplate (`NonYielding` functions return `{ OkTy, i32 }` by value; yielding coroutines write
`{ OkTy, i32 }` into their static frame's `_ret` slot, and `defer` statements execute cleanly on both paths):

- **Boolean failure (`error` suffix)**:
  ```zev
  fn read_byte([]u8 buf) -> u16 error {
      if (buf.len == 0) {
          error;
      }
      return buf[0];
  }
  ```
- **Typed enum failure (`error ErrType` suffix)**:
  ```zev
  enum ParseErr { TooShort, BadCrc }

  fn parse_setpoint([]u8 buf, u16 expected_crc) -> u16 error ParseErr {
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
  [24]u8 buf;
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
  expressions (`period_ms`), clamps runtime periods `<= 0` to `1` tick and `> 2147483647` (`i32::MAX`)
  to `2147483647` ticks, and **skips/resyncs missed deadlines on overrun** (zero catch-up bursts).
  (Bare `every 1ms` is trapped via `P0011`.)
- **`restart;`**: Unwinds active `defer`s, restores any temporary priority boost, zeroes task-local state
  (preserving `spawn` parameters), and jumps cleanly back to task entry.

```zev
watch<u16> motor1_rpm = 0;

task motor_control(*watch<u16> target_rpm, u8 pwm_ch, u32 period_ms) {
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

## 2. Boot-Tree `spawn`, `MAX_TASKS = 32` & Provenance (`fn main()` + Boot Helpers)
`spawn` is permitted inside `fn main()` (including conditional `if` / `switch` boot-mode branches) AND
inside non-loop, non-recursive boot-helper `fn`s called before `main()`'s supervisor loop (up to
32 static slots with `O(1)` per-priority `u32` ready bitmasks and fixed `1 / 5 / 15 / 50 ms` starvation
windows; `E0027` if > 32 or inside a `task`/`isr`/loop/recursion). `zevc ram` reports each slot's
boot provenance (`main -> init_drive -> motor_control`) AND static non-yielding run classification
(`straight-line`, `bounded-for`, or `noyield-spin`, plus `max_run_blocks`).
NOTE: v0 schedules all tasks on a single core; `@core(N)` is parsed and stored as spawn metadata.
```zev
fn init_drive(*watch<u16> rpm, u8 ch) {
    spawn motor_control(rpm, ch, 1) @critical;
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
2. **`SoftYielding`**: Terminating compute or polling helpers (e.g. checksums, parsers, or status loops)
   that yield cooperatively via explicit `checkpoint()` calls OR back-edges of `while (cond)` loops outside
   `noyield` / `critical` / `isr` / `defer` (whereas bounded `for..in` loops stay in registers with zero
   auto-yielding; wrap tight register-polling `while` loops in `noyield { while (...) {} }`):
   - In normal `task` flow, runs as a union-packed coroutine sharing the CPU smoothly.
   - Inside `noyield { ... }`, **auto-specializes a direct `0 B`-`.bss` stack clone** (`{fn}__noyield`,
     checkpoints stripped) reported transparently in `zevc ram` and `zevc check --json`!
   - Function pointers referencing a `SoftYielding` function point to `{fn}__noyield` and execute
     without sharing the CPU (`HardBlocking` functions are not yet supported through function pointers).
   - Still **forbidden** inside `critical`, `isr`, `defer`, and `extern \"C\"` callbacks (`E0003`–`E0006`).
3. **`HardBlocking`**: Awaits external events or time (`delay`, `wait { ... }`, `inbox.recv`/`send`,
   `watch.wait_change`, `signal.wait`, `ping_pong.wait_chunk`, `byte_ring.wait_read`, `pool.alloc`,
   `wait_queue.commit_wait`, `loop every`, or unbounded `loop`); forbidden inside `noyield`,
   `critical`, `isr`, `defer`, and `extern \"C\"` callbacks (`E0003`–`E0006`).
- Cycles in the call graph containing any yielding function are rejected (`E0012`).
- **Defined Runtime Traps**: Out-of-bounds `a[i]`, sub-slice `a[s..e]`, MMIO array `regs.arr[i]`, and integer
  `x / 0` or `x % 0` trap deterministically (invoking any registered `sys.set_fault_handler` before halt), while signed
  `INT_MIN / -1` evaluates to `INT_MIN` and `INT_MIN % -1` evaluates to `0`.

## 2. Explicit Atomicity Brakes (`noyield`, `critical`, `@manual_yield`)
```zev
fn hash_block([]u8 data) -> u32 {           // Inferred `SoftYielding` (`checkpoint()` only)
    u32 h = 5381;
    for (u8 b in data) {
        h = (h * 33) ^ (b as u32);
        checkpoint();
    }
    return h;
}

fn boot_init([]u8 rom_hdr) {
    // 1. `noyield { ... }` — Task-atomic (forbids `HardBlocking`; auto-clones `SoftYielding` `hash_block`!)
    noyield {
        u32 boot_digest = hash_block(rom_hdr);
    }

    // 2. `critical { ... }` — Hardware-atomic (masks IRQs; forbids ALL yields & unbounded `loop`)
    critical {
        hw.RCC.cr |= 0x0100_0000;
    }
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
            "prim", "signal", "watch", "guarded", "inbox", "byte_ring", "chan", "channel", "channels",
            "isolation", "globals", "boost", "subsystem", "subsystems", "view", "views",
        ],
        summary: "Static OS primitives, `guarded<T>`, subsystem `struct`s, `*byte_ring`/`*inbox<T>` views, and `boost:`",
        body: "\
# Zev Topic Guide: `primitives`

## 1. Zero Mutable Raw Globals — Static OS Endpoints Only
Top-level mutable variables (`u32 g = 0;`) are rejected (`E0001`/`E0002`). Declare immutable `const`
values or static scheduler-aware OS primitives at module scope and pass pointers (`&ep`) to tasks
(`chan<T, N>` has been unified into `inbox<T, N>`, trapped via `P0012`):

| Primitive | Purpose | Non-Blocking (ISR & Task Safe) | Yielding (Task-Only) |
|---|---|---|---|
| `watch<T> w = init;` | Enforced single-writer cell | `w.read() -> T`, `w.publish(v)` | `w.wait_change([timeout: d]) -> T` |
| `guarded<T> g = init;` | Multi-writer (`critical` only) | `g.get() -> T`, `g.set(v)` (in `critical`) | *(not waitable; pair with `wait_queue`)* |
| `inbox<T, N> q;` | Unified bounded queue | `q.post(v [, boost: @prio])`, `q.try_send(v) -> bool`, `q.try_recv()` | `q.send(v [, timeout: d])`, `q.recv([timeout: d]) -> T` |
| `byte_ring<N> r;` | Contiguous BipBuffer byte stream | `r.push(s [, boost:])`, `r.push_byte(b [, boost:])`, `r.grant_write(n)`, `r.commit_write(n [, boost:])`, `r.read_slice()`, `r.consume(n)`, `r.pop(dst)`, `r.try_pop_byte()`, `r.clear()` | `r.wait_read([timeout: d]) -> []u8` |
| `signal s;` | Lightweight wakeup | `s.notify([boost: @prio])`, `s.try_wait() -> bool`, `s.clear()` | `s.wait([timeout: d])` |
| `@dma_safe pool<T, K> p;` | Zero-copy `O(1)` slab (`K<=32`) | `p.try_alloc() -> Lease<T> error`, `p.avail()`, `p.in_use()` | `p.alloc([timeout: d]) -> Lease<T>` |
| `dma_slot<T> s;` | Cross-IRQ `Lease<T>` slot | `s.park(l)`, `s.take() -> Lease<T> error`, `s.release([boost:])`, `s.addr()` | *(park across IRQ/DMA transfer)* |
| `@dma_safe ping_pong<T, N> p;` | Circular `2 * N` DMA halves | `p.flip()`, `p.complete_half(h)`, `p.start()`, `p.stop()`, `p.release()`, `p.active_buf()`, `p.dma_buf()` | `p.wait_chunk([timeout: d]) -> []T` |
| `wait_queue wq;` | Race-free waiter bitmask | `wq.wake_one([boost:]) -> bool`, `wq.wake_all([boost:])` | `wq.prepare_wait()` (`critical`) + `wq.commit_wait([timeout:])` |

Standard library `import std.sync as sync;` builds **`Semaphore`**, **`EventFlags`**, **`Latch`**, and
**`Barrier<usize N>`** on top of `wait_queue` using pure `.zev` structs and `critical { prepare_wait(); }`!

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

guarded<u16>       motor1_rpm = 0;
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
            Overcurrent => { critical { motor1_rpm.set(0); } }
            BadPacket   => {}
            IdleCheck   => {}
        }
    }
}
```

## 4. Deterministic Multi-Source `wait { ... }` (`P0004` replaces `select`)
A task can wait atomically across multiple consumer primitives + an idle `timeout <dur> =>` or periodic
`deadline <u32_tick> =>` using `wait { ... }`. Arms are checked top-to-bottom under a single critical section,
park atomically via 32-bit primitive waiter bitmasks (zero lost wakeups), allow the six direct zero-argument
consumer ops (`inbox.recv()`, `signal.wait()`, `byte_ring.wait_read()`, `ping_pong.wait_chunk()`,
`watch.wait_change()`, `pool.alloc()`), and scope leased slices per arm (`E0011`/`E0034`; for single-queue
sends with a timeout, use `q.send(v, timeout: d) catch { ... }`):
```zev
u32 next_beat = sys.now_ms() + 100u32;
loop {
    wait {
        SupEvent ev = sup_inbox.recv() => { handle_sup(ev); }
        []u8 chunk = rx_ring.wait_read() => { parse(chunk); rx_ring.consume(chunk.len); }
        sig.wait() => { tick_flag(); }
        deadline next_beat => { heartbeat(); next_beat += 100u32; }
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
- **ALLOWED**: Bounds-checked sub-slicing (`half[0..5]`, trapping cleanly if out of range), wrapping in local structs (`ByteReader{ buf: half, pos: 0 }`),
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
    loop {
        Lease<Frame> buf = tx_pool.try_alloc() catch { sys.exit(1); };
        buf.get().len = 32;
        tx_inflight.park(buf) catch { continue; };
        Lease<Frame> done = done_q.recv();
        done.release();
    }
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
- `sys.tick();` / `sys.tick_ms();` / `sys.tick_by(u32 ms);` — Advances the 1 ms scheduler tick from a pure-`.zev` timer `isr` (e.g., `Timer0` on Uno, `TCB3` on ATmega4809).
- Explicit OS hooks wired directly from `.zev` `bsp.*` / application code (zero hardcoded board MMIO or magic symbol names in Rust/LLVM):
  - `sys.set_putc(fn(u8))` — Console output backend for `print` / `println` / `sys.log`.
  - `sys.set_clock_ms(fn() -> u32)` — Monotonic millisecond hardware clock reader backing `sys.now_ms()`.
  - `sys.set_exit(fn(i32))` — Target shutdown/reboot handler invoked by `sys.exit(code)`.
  - `sys.set_fault_handler(fn())` — Hardware fault hook invoked before halt on fatal traps.
  - `sys.set_idle(fn(u32, u32))` — Idle sleep hook (`(in_sim, sleep_depth)`).
  - `sys.set_irq_router(fn(u32, u32), fn(u32))` — RISC-V interrupt matrix routing and acknowledge hooks (`route(irq, slot)`, `ack(irq)`).
  - `sys.rom_call_u32` / `sys.rom_call_void` / `sys.rom_call_noreturn` — Stack-bounded external ROM ABI calls (`stack_bound` proved statically via `.stack_sizes`).
- `sys.now_ms() -> u32` / `sys.now_ticks() -> u32` — Current monotonic millisecond tick counter.
- `sys.nop();` — Single-cycle CPU no-op barrier for short MMIO settle delays (use `hal.time.spin_us(u32)` for `<50 us` hardware settle delays; DMA sleep-depth clamping is managed automatically by `@dma_safe` `ping_pong` / `pool`).
- `sys.log(*u8 msg);` / `sys.log_kv(*u8 label, i32 val);` / `sys.log_hex(*u8 label, u32 val);`
- `sys.pend_irq(u32 irq);` — Triggers software/NVIC interrupt `irq` (fires `isr` safely on host & Cortex-M).
- `sys.fuzz_irqs(u32 seed, u32 mask);` — Host simulation hook injecting pseudo-random IRQ deliveries at
  scheduler critical-section boundaries to stress-test ISR/task race freedom (`0` runtime overhead on MCUs).
- `sys.sleep_depth() -> u32` / `sys.wfi_count() -> u32` / `sys.trace_sleep(bool on)` — Inspect current
  idle clamp (`1=LIGHT`, `2=DEEP`) and sleep transitions (`trace_sleep` is a zero-overhead compatibility
  diagnostic flag alongside `sys.set_idle`). Note: Cortex-M `allow_sleep_deep` defaults `false`, and
  `SLEEPDEEP` (`SCB->SCR` bit 2) is suppressed whenever DMA locks OR pending timer deadlines exist;
  RISC-V uses `wfi` when IRQs are attached and zero deadlines block a polled clock.
- `sys.exit(i32 code);` — Diverging termination call (runs active LIFO `defer`s, then invokes `sys.set_exit` handler or target exit).

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
            "target", "cli", "ram", "sim", "llvm", "elf", "qemu", "cortex-m", "cortex-m3",
            "esp32", "esp32c6", "riscv", "avr", "atmega328p", "uno", "arduino", "atmega4809",
            "nano-every", "nano", "updi", "rp2040", "pico", "pico-w", "cortex-m0plus", "uf2",
            "svd", "import-svd", "board", "chip", "toml", "build", "flash",
            "stm32", "stm32f334", "fpu",
        ],
        summary: "CLI workflow: `build --board`, `flash --board`, `ram --board`, `chips/*.toml` + `boards/*.toml`, `zev.toml`, `import-svd`",
        body: "\
# Zev Topic Guide: `targets`

## 1. Static Verification & Machine-Readable JSON (`zevc check`)
Reports task/primitive counts, tick period, dual-specialized `SoftYielding` functions, concrete monomorphized
generic instances, AND per-task non-yielding run bounds (`task_runs`: `max_run_kind`, `max_run_blocks`):
```bash
zevc check examples/motor_controller.zev examples/motor_support.zev
zevc check --json app.zev             # Structured JSON diagnostics, task_runs, dual-spec & monomorphized lists
```

## 2. Declarative Target Specs & `zev.toml` (`build`, `flash`, `ram`)
Targets are defined declaratively in `chips/<chip>.toml` + `boards/<board>.toml` (with zero compiler edits for new chips/boards):
- `zevc build --board <board> [-Oz] [--emit-ll] [files...] [-o out]`
- `zevc flash --board <board> [--port <dev>] [-Oz] [files...]`
- `zevc ram --board <board> [files...]` (also supports `--chip <chip>` or legacy `--target <name>`)
- Target-bound portability: `[chip]`/`[board]` `hal_dir` binds `import hal.*;` (default `hal/sim`) and `[board]` `board_module` binds `import board;` (default `bsp/sim`), allowing one portable `.zev` app (`examples/portable_controller_demo.zev`) to run across host, QEMU, STM32F3, ESP32-C6, and RP2040 Pico W unchanged.
- Optional `zev.toml` project manifest sets defaults (`entry`, `board`, `chip`, `port`, `opt_level`) so `zevc build` / `zevc flash` work with zero extra CLI flags.

```bash
zevc ram examples/motor_controller.zev examples/motor_support.zev
zevc ram --board lm3s6965_evb examples/cortex_m_demo.zev
zevc ram --board esp32c6_pico examples/esp32c6_demo.zev
zevc ram --board arduino_uno examples/arduino_uno_demo.zev
zevc ram --board arduino_nano_every examples/arduino_nano_every_demo.zev
zevc ram --board rpi_pico_w examples/pico_w_demo.zev
```

## 3. SVD Hardware Generation (`zevc import-svd`)
Generate typed `.zev` `pub mmio` layouts, peripheral base addresses, and `pub irq` tokens directly
from any vendor CMSIS-SVD file:
```bash
zevc import-svd vendor/STM32G474.svd -o chip/stm32g474.zev
zevc check -I chip examples/motor_controller.zev examples/motor_support.zev
```

## 4. Sanitized Host LLVM Simulation (`run`)
Unified `zev-codegen-llvm` backend emits host `x86_64` LLVM IR linked with Clang `-fsanitize=address,undefined`
and deterministic `sys.fuzz_irqs(seed, mask)` IRQ interleaving — running the exact same scheduler IR as MCUs.
Piping input or redirecting `/dev/null` without `--script` / `--cmd` stops polling stdin at EOF and keeps the
firmware running (tasks continue executing normally); use `--script` / `--cmd`, `--max-ticks`, or `sys.exit(...)`
to bound or terminate non-interactive runs:
```bash
zevc run --sanitize examples/sim_demo.zev
zevc run --max-ticks 50 app.zev       # Bounds infinite `loop every` / `checkpoint()` firmware loops!
zevc run app.zev --cmd help           # Scripted mode: stops after final `ok` / `err` response
```

## 5. ARM Cortex-M3 (`lm3s6965_evb`) ELF & QEMU
Synthesizes `.isr_vector`, `Reset_Handler`, `SysTick_Handler`, and linker script with zero C/libc.
Supports `-O0`, `-O1`, `-O2` (default), `-O3`, `-Os`, and `-Oz` (`--opt-level <level>`):
```bash
zevc emit-llvm --board lm3s6965_evb examples/cortex_m_demo.zev -o build/cm3.ll
zevc build --board lm3s6965_evb -Oz examples/cortex_m_demo.zev -o build/cm3.elf
zevc run-qemu -Oz examples/cortex_m_demo.zev
zevc run-qemu --max-ticks 50 app.zev  # Deterministic SysTick cap for non-terminating firmware!
```

## 6. Espressif ESP32-C6 (`esp32c6_pico`, `riscv32imac` Direct-Boot)
Synthesizes `0xAEDB041D` x2 ROM Direct-Boot header, `@_start` at `0x42000008`, and `mtvec` dispatch;
`route_irq`/`ack_irq`, watchdog disarm (`TIMG0`/`TIMG1`/`LP_WDT`/`SWD`), `SYSTIMER` (`sys.set_clock_ms`), and `USB_DEVICE` (`sys.set_putc`) live in `.zev`:
```bash
zevc emit-llvm --board esp32c6_pico examples/esp32c6_demo.zev -o build/esp32c6.ll
zevc build --board esp32c6_pico -Oz examples/esp32c6_demo.zev -o build/esp32c6_demo.bin
zevc flash --board esp32c6_pico -Oz examples/esp32c6_demo.zev --port /dev/ttyACM0
```

## 7. Microchip ATmega328P / Arduino Uno R3 (`arduino_uno`, `avr` Optiboot)
Synthesizes Flash vector table, `.data`/`.rodata`/`.bss` startup, `SREG`, and `avr-libgcc` shims;
WDT disarm, `Timer0` 1 kHz `sys.tick()` ISR, and `115200 8N1` `USART0` (`sys.set_putc`) live in `.zev`:
```bash
zevc emit-llvm --board arduino_uno examples/arduino_uno_demo.zev -o build/uno.ll
zevc build --board arduino_uno examples/arduino_uno_demo.zev -o build/uno_demo.bin
zevc flash --board arduino_uno examples/arduino_uno_demo.zev --port /dev/ttyACM0
```

## 8. Microchip ATmega4809 / Arduino Nano Every (`arduino_nano_every`, `avrxmega3` UPDI)
Synthesizes Flash vectors + `.data`/`.bss` startup, while
`bsp.init_board()` configures `CPU.ccp` WDT/clock unlock, `FUSE.OSCCFG` 16/20 MHz auto-calibration,
`SLPCTRL`, `isr systick_isr` (`TCB3` 1 kHz `sys.tick()`), and `sys.set_putc` (`115200 8N1` `USART3`) in `.zev`:
```bash
zevc emit-llvm --board arduino_nano_every examples/arduino_nano_every_demo.zev -o build/nano.ll
zevc build --board arduino_nano_every examples/arduino_nano_every_demo.zev -o build/nano_every_demo.bin
zevc flash --board arduino_nano_every examples/arduino_nano_every_demo.zev --port /dev/ttyACM0
```

## 9. Raspberry Pi Pico / Pico W (`rpi_pico_w`, `rp2040` SRAM UF2)
Synthesizes ARMv6-M `.entry` (`0x2000_0000`), `[48 x ptr]` `.isr_vector`, and `SysTick`;
125 MHz `PLL_SYS`/48 MHz `PLL_USB`, BootROM reset (`sys.rom_call_u32` + `sys.rom_call_noreturn`), `sys.set_exit`, and Full-Speed USB 1.1 CDC-ACM live in `.zev`:
```bash
zevc emit-llvm --board rpi_pico_w examples/pico_w_demo.zev -o build/pico_w.ll
zevc build --board rpi_pico_w examples/pico_w_demo.zev -o build/pico_w_demo.uf2
zevc flash --board rpi_pico_w examples/pico_w_demo.zev --port /dev/ttyACM2
```

## 10. STMicroelectronics STM32F334R8 / NUCLEO-F334R8 (`nucleo_f334r8`, Cortex-M4F + FPU)
Targets `thumbv7em-none-eabihf` (`+vfp4d16sp`, `has_fpu = true`): `Reset_Handler` enables `SCB->CPACR` (`CP10`/`CP11` full access) before `main`, `.stack_sizes` accounts for the `104 B` hardware NVIC FPU exception frame on `isr`s, and `bsp.nucleo_f334r8` configures 72 MHz HSE-BYP PLL (64 MHz HSI/2 fallback), `HRTIM1`, `ADC1/2`, `DAC1`, `COMP2`, `bxCAN1`, `SPI1`, `I2C1`, `DMA1`, and Flash NVS:
```bash
zevc ram --board nucleo_f334r8 examples/nucleo_f334r8_demo.zev
zevc build --board nucleo_f334r8 examples/nucleo_f334r8_demo.zev -o build/nucleo_f334r8_demo.elf
zevc flash --board nucleo_f334r8 examples/nucleo_f334r8_demo.zev
```

## 11. WCH CH32V307 / YD-CH32V307VCT6 (`yd_ch32v307`, QingKe V4F `riscv32imac`) — bring-up, untested on hardware
Reuses the `riscv32` backend with `[irq] routing = \"identity\"` and `enable_csr = \"none\"`: the PFIC
interrupt number is the vector slot (104 slots) and the core has no `mie` CSR, so lines are enabled by
`route_irq` in `.zev`. Built soft-float; idle polls (`allow_wfi = false`). The 8 MHz HSI bring-up clock,
core-timer `sys.set_clock_ms`, and `115200 8N1` `USART1` on `PA9` (`sys.set_putc`) live in `.zev`.
Flash over the chip's USB bootloader: hold `BOOT0`, tap `RST`, release, then:
```bash
zevc build --board yd_ch32v307 examples/ch32v307_blink.zev -o build/ch32v307_blink.bin
zevc flash --board yd_ch32v307 examples/ch32v307_blink.zev      # runs `wchisp flash <image>`
zevc flash --board yd_ch32v307 examples/ch32v307_demo.zev       # button interrupt via PFIC
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

    // 2. Error code match (`P0001`..`P0018`, `E0001`..`E0034`, case-insensitive)
    let upper = q.to_ascii_uppercase();
    for err_doc in ERROR_DOCS {
        if err_doc.code == upper {
            return Ok(render_error_doc(err_doc));
        }
    }

    Err(format!(
        "unknown documentation topic or error code `{q}` (run `zevc doc --list` to see all 8 topics and P0001..P0018 / E0001..E0034 codes)"
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
        assert!(primer.contains("P0001..P0018"));
        assert!(primer.contains("E0001..E0034"));

        let list = lookup_doc(None, true).unwrap();
        assert!(list.contains("runtime"));
        assert!(list.contains("P0001"));
        assert!(list.contains("P0018"));
        assert!(list.contains("E0001"));
        assert!(list.contains("E0034"));
    }

    #[test]
    fn dma_topic_net_worker_snippet_verifies_cleanly() {
        let src = r#"
            struct Frame { u32 len; }
            @dma_safe pool<Frame, 4> tx_pool;
            dma_slot<Frame>          tx_inflight;
            inbox<Lease<Frame>, 4>   done_q;

            isr dma_done_isr(*dma_slot<Frame> slot, *inbox<Lease<Frame>, 4> done) {
                Lease<Frame> f = slot.take() catch { return; };
                done.post(f, boost: @critical);
            }

            task net_worker() {
                loop {
                    Lease<Frame> buf = tx_pool.try_alloc() catch { sys.exit(1); };
                    buf.get().len = 32;
                    tx_inflight.park(buf) catch { continue; };
                    Lease<Frame> done = done_q.recv();
                    done.release();
                }
            }

            fn main() {
                spawn net_worker() @normal;
            }
        "#;
        let prog = zev_parser::parse_str_named(src, "dma_doc").expect("dma doc snippet parse");
        zev_sema::analyze_program(&prog).expect("dma doc snippet sema");
    }

    #[test]
    fn wait_topic_snippet_verifies_cleanly() {
        let src = r#"
            enum SupEvent { Tick, Fault }
            inbox<SupEvent, 4> sup_inbox;
            byte_ring<64> rx_ring;
            signal sig;

            fn handle_sup(SupEvent ev) {}
            fn parse([]u8 c) {}
            fn tick_flag() {}
            fn heartbeat() {}

            task worker() {
                u32 next_beat = sys.now_ms() + 100u32;
                loop {
                    wait {
                        SupEvent ev = sup_inbox.recv() => { handle_sup(ev); }
                        []u8 chunk = rx_ring.wait_read() => { parse(chunk); rx_ring.consume(chunk.len); }
                        sig.wait() => { tick_flag(); }
                        deadline next_beat => { heartbeat(); next_beat += 100u32; }
                    }
                }
            }

            fn main() {
                spawn worker() @normal;
            }
        "#;
        let prog = zev_parser::parse_str_named(src, "wait_doc").expect("wait doc snippet parse");
        zev_sema::analyze_program(&prog).expect("wait doc snippet sema");
    }

    #[test]
    fn all_topics_and_aliases_resolve() {
        assert_eq!(TOPIC_DOCS.len(), 9);
        for topic in [
            "tutorial",
            "guide",
            "getting-started",
            "walkthrough",
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
            "stm32",
            "stm32f334",
            "fpu",
            "err",
            "catch",
        ] {
            let text = lookup_doc(Some(topic), false)
                .unwrap_or_else(|e| panic!("topic `{topic}` failed: {e}"));
            assert!(text.starts_with("# Zev Topic Guide:"));
        }
    }

    #[test]
    fn all_error_codes_resolve_and_bad_and_good_snippets_verify() {
        for n in 1..=18 {
            let code = format!("P{n:04}");
            let lower = code.to_ascii_lowercase();
            let doc =
                lookup_doc(Some(&lower), false).unwrap_or_else(|e| panic!("missing {code}: {e}"));
            assert!(doc.contains(&code));
        }
        for n in 1..=34 {
            let code = format!("E{n:04}");
            let lower = code.to_ascii_lowercase();
            let doc =
                lookup_doc(Some(&lower), false).unwrap_or_else(|e| panic!("missing {code}: {e}"));
            assert!(doc.contains(&code));
        }

        assert_eq!(ERROR_DOCS.len(), 52);
        for e in ERROR_DOCS {
            // 1. Every GOOD snippet must parse, monomorphize & pass zev-sema, AND lower cleanly
            //    through zev-mir and zev-codegen-llvm (both Cortex-M and host x86_64 LLVM IR)!
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
            let host_ll = zev_codegen_llvm::emit_llvm_ir_with_options(
                &mir,
                &zev_codegen_llvm::CodegenOptions::host_linux(),
            )
            .unwrap_or_else(|le| {
                panic!(
                    "GOOD snippet for {} failed host zev-codegen-llvm:\n{}\n---\n{}",
                    e.code, le, e.good_zev
                )
            });
            assert!(
                !host_ll.is_empty(),
                "empty host LLVM IR output for {}",
                e.code
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

    fn extract_zev_fences(markdown: &str) -> Vec<(usize, String)> {
        let mut fences = Vec::new();
        let mut in_zev = false;
        let mut start_line = 0usize;
        let mut cur = String::new();
        for (idx, line) in markdown.lines().enumerate() {
            let trimmed = line.trim_start();
            if !in_zev {
                if trimmed == "```zev" {
                    in_zev = true;
                    start_line = idx + 1;
                    cur.clear();
                }
            } else if trimmed == "```" {
                in_zev = false;
                fences.push((start_line, cur.clone()));
            } else {
                cur.push_str(line);
                cur.push('\n');
            }
        }
        assert!(
            !in_zev,
            "unclosed ```zev fence starting at line {start_line}"
        );
        fences
    }

    #[test]
    fn all_fenced_zev_snippets_in_docs_and_design_parse_cleanly() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let design = std::fs::read_to_string(root.join("DESIGN_V0.md")).expect("read DESIGN_V0.md");
        let readme = std::fs::read_to_string(root.join("README.md")).expect("read README.md");

        let mut sources: Vec<(String, String)> = vec![
            ("DESIGN_V0.md".to_string(), design),
            ("README.md".to_string(), readme),
        ];
        for t in TOPIC_DOCS {
            sources.push((format!("zevc doc {}", t.name), t.body.to_string()));
        }

        let mut checked = 0usize;
        for (doc_name, text) in &sources {
            for (line_no, snippet) in extract_zev_fences(text) {
                checked += 1;
                let label = format!("{doc_name}:{line_no}");
                let parsed_top = zev_parser::parse_str_named(&snippet, &label);
                match parsed_top {
                    Ok(prog) => {
                        if doc_name == "DESIGN_V0.md" || doc_name == "README.md" {
                            let with_main = if snippet.contains("fn main(") {
                                prog
                            } else {
                                zev_parser::parse_str_named(
                                    &format!("{snippet}\nfn main() {{}}\n"),
                                    &label,
                                )
                                .expect("parse DESIGN_V0 snippet with main")
                            };
                            let opts = zev_sema::AnalyzeOptions {
                                include_dirs: vec![root.clone()],
                                hal_dir: Some("hal/sim".to_string()),
                                board_module: Some("bsp/sim".to_string()),
                                ..zev_sema::AnalyzeOptions::default()
                            };
                            let (mono_progs, _) = zev_sema::analyze_and_monomorphize_programs(
                                std::slice::from_ref(&with_main),
                                &opts,
                            )
                            .into_result()
                            .unwrap_or_else(|diags| {
                                panic!(
                                    "{doc_name} snippet at {label} failed zev-sema:\n{diags:#?}\n---\n{snippet}"
                                )
                            });
                            if doc_name == "README.md" {
                                let mir = zev_mir::lower_programs(&mono_progs)
                                    .expect("README snippet MIR lower");
                                let ll = zev_codegen_llvm::emit_llvm_ir(&mir)
                                    .expect("README snippet LLVM codegen");
                                assert!(!ll.is_empty());
                            }
                        }
                    }
                    Err(top_err) => {
                        // Statement-level doc excerpts (e.g. `wait { ... }`, `catch`, `fmt_buf`) must
                        // parse cleanly inside a `task` body.
                        let wrapped = format!(
                            "task __doc_excerpt() {{\n{snippet}\n    loop {{ checkpoint(); }}\n}}\n"
                        );
                        if let Err(stmt_err) = zev_parser::parse_str_named(&wrapped, &label) {
                            panic!(
                                "fenced ```zev snippet at {label} failed to parse both as items and as statements!\n\
                                 Item error: {top_err}\nStmt error: {stmt_err}\n---\n{snippet}"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            checked >= 15,
            "expected at least 15 fenced zev snippets, found {checked}"
        );
    }

    #[test]
    fn all_cited_error_codes_in_docs_exist() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let design = std::fs::read_to_string(root.join("DESIGN_V0.md")).expect("read DESIGN_V0.md");
        let readme = std::fs::read_to_string(root.join("README.md")).expect("read README.md");

        let mut corpus: Vec<(String, String)> = vec![
            ("README.md".to_string(), readme),
            ("DESIGN_V0.md".to_string(), design),
            ("PRIMER_DOC".to_string(), PRIMER_DOC.to_string()),
        ];
        for t in TOPIC_DOCS {
            corpus.push((format!("zevc doc {}", t.name), t.body.to_string()));
        }

        let valid_codes: std::collections::BTreeSet<&str> =
            ERROR_DOCS.iter().map(|e| e.code).collect();
        let mut total_cited = 0usize;
        for (doc_name, text) in &corpus {
            let b = text.as_bytes();
            let mut i = 0usize;
            while i + 5 <= b.len() {
                if (b[i] == b'E' || b[i] == b'P')
                    && b[i + 1] == b'0'
                    && b[i + 2].is_ascii_digit()
                    && b[i + 3].is_ascii_digit()
                    && b[i + 4].is_ascii_digit()
                {
                    let prev_ok = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
                    let next_ok =
                        i + 5 == b.len() || !(b[i + 5].is_ascii_alphanumeric() || b[i + 5] == b'_');
                    if prev_ok && next_ok {
                        let code = &text[i..i + 5];
                        assert!(
                            valid_codes.contains(code),
                            "documented error code `{code}` in {doc_name} does not exist in ERROR_DOCS"
                        );
                        total_cited += 1;
                    }
                }
                i += 1;
            }
        }
        assert!(
            total_cited >= 40,
            "expected at least 40 cited codes, found {total_cited}"
        );
    }

    #[test]
    fn user_docs_and_readme_do_not_leak_internal_syn_symbols() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let readme = std::fs::read_to_string(root.join("README.md")).expect("read README.md");
        assert!(
            !readme.contains("@__syn_"),
            "README.md must not leak internal @__syn_* compiler symbols"
        );
        assert!(
            !PRIMER_DOC.contains("@__syn_"),
            "PRIMER_DOC must not leak internal @__syn_* compiler symbols"
        );
        for t in TOPIC_DOCS {
            assert!(
                !t.body.contains("@__syn_"),
                "zevc doc {} must not leak internal @__syn_* compiler symbols",
                t.name
            );
        }
    }
}
