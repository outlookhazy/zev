//! Complete structured error explainers (`P0001`..`P0014` and `E0001`..`E0034`) for `zevc doc`.

pub struct ErrorDoc {
    pub code: &'static str,
    pub summary: &'static str,
    pub topic: &'static str,
    pub why: &'static str,
    pub bad_zev: &'static str,
    pub good_zev: &'static str,
}

pub const ERROR_DOCS: &[ErrorDoc] = &[
    // ── Parser Foreign-Idiom Traps (P0001..P0014) ───────────────────────────
    ErrorDoc {
        code: "P0001",
        summary: "Foreign `let` / `var` / `mut` variable binding instead of explicit C-style `Type name`",
        topic: "syntax",
        why: "In bare-metal firmware, every local variable lives either in CPU registers or inside a compile-time-sized static coroutine frame. Zev requires explicit C-style bit-width types (`u8`, `u16`, `u32`, `i32`, `bool`, `[]u8`) at every declaration so overflow boundaries and static SRAM costs are never hidden behind type inference.",
        bad_zev: "\
fn main() {
    let mut rpm: u16 = 1800;
    var duty = 500;
}",
        good_zev: "\
fn main() {
    u16 rpm = 1800;
    u16 duty = rpm / 4;
    if (duty > 0) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "P0002",
        summary: "Rust/Zig/Go `name: Type` parameter or `-> RetType` function return syntax",
        topic: "syntax",
        why: "Zev uses C-style return-first signatures (`RetType name(Type param)`) for value-returning functions and `fn name(Type param)` strictly for `void`-returning functions. Function pointer types still write `fn(T1) -> Ret`, but function definitions never use `:` or `->`.",
        bad_zev: "\
fn clamp_duty(val: u16, limit: u16) -> u16 {
    if (val > limit) { return limit; }
    return val;
}",
        good_zev: "\
u16 clamp_duty(u16 val, u16 limit) {
    if (val > limit) {
        return limit;
    }
    return val;
}

fn main() {
    u16 d = clamp_duty(1200, 1000);
    if (d == 1000) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "P0003",
        summary: "`async` or `.await` used in a call-graph-inferred cooperative language",
        topic: "yielding",
        why: "Zev eliminates function coloring: any function that calls a blocking primitive (`wait_chunk()`, `recv()`, `wait()`, `delay()`, `checkpoint()`) yields transparently on the single hardware stack, and its live locals pack into the caller's static coroutine frame automatically. Instead of marking async paths, Zev marks atomic paths (`noyield { ... }`, `critical { ... }`, `isr`).",
        bad_zev: "\
async fn read_sensor() -> u16 {
    checkpoint().await;
    return 400;
}",
        good_zev: "\
u16 read_sensor() {
    u16 base = 400;
    checkpoint();
    return base + 12;
}

fn main() {
    u16 v = read_sensor();
    if (v > 0) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "P0004",
        summary: "Go/Rust `select` block used instead of Zev's deterministic `wait { ... }` block",
        topic: "primitives",
        why: "Multi-source waiting in Zev uses contextual `wait { ... }` blocks, which register O(1) static per-primitive waiter bitmasks and evaluate arms in deterministic top-to-bottom priority order (never random selection like Go's `select`). Arms must be direct zero-arg blocking consumer calls (`recv`, `wait`, `wait_read`, `wait_chunk`, `wait_change`, `alloc`) plus an optional `timeout <dur>` or absolute `deadline <u32>` arm.",
        bad_zev: "\
fn main() {
    select {
        ev = fault_ch.recv() => { handle(ev); }
        _ = after(1s) => { heartbeat(); }
    }
}",
        good_zev: "\
signal fault_sig;
inbox<u32, 8> cmd_q;
watch<u32> beats = 0;

task sup() {
    loop {
        wait {
            fault_sig.wait() => { beats.publish(0u32); }
            u32 cmd = cmd_q.recv() => { beats.publish(cmd); }
            timeout 50ms => { beats.publish(beats.read() + 1u32); }
        }
    }
}

fn main() {
    spawn sup() @normal;
}",
    },
    ErrorDoc {
        code: "P0005",
        summary: "`match` keyword used instead of Zev's `switch (expr) { Variant => { ... } }`",
        topic: "syntax",
        why: "Zev uses C-familiar `switch (scrutinee)` paired with clean non-fallthrough `Variant => { ... }` fat-arrow arms (no accidental C fallthrough bugs and no `match` keyword).",
        bad_zev: "\
enum Mode { Idle, Active }
fn main() {
    Mode m = Active;
    match m {
        Idle => {}
        Active => {}
    }
}",
        good_zev: "\
enum Mode { Idle, Active }

fn main() {
    Mode m = Active;
    switch (m) {
        Idle => { return; }
        Active => { return; }
    }
}",
    },
    ErrorDoc {
        code: "P0006",
        summary: "Zig-style `catch |err|` pipe capture instead of `catch (ErrType err)`",
        topic: "errors",
        why: "Zev keeps variable declarations uniform everywhere—including `catch` clauses—using explicit parenthesized `(ErrType err)` or `(ErrType err, ValType partial)` bindings.",
        bad_zev: "\
enum ParseErr { BadCrc }
u16 decode([]u8 b) error ParseErr { error BadCrc; }
fn main() {
    []u8 empty;
    u16 v = decode(empty) catch |err| { return; };
}",
        good_zev: "\
enum ParseErr { BadCrc }

u16 decode(u8 b) error ParseErr {
    if (b == 0) {
        error BadCrc;
    }
    return b;
}

fn main() {
    u16 v = decode(0) catch (ParseErr err) {
        return;
    };
    if (v > 0) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "P0007",
        summary: "Zig/Rust `!Type`, `try`, or `return error.X` / `Err(...)` instead of dual-exit `error`",
        topic: "errors",
        why: "Zev models normal completion (`return val;` -> `r1 == 0`) and failure (`error [Reason [, partial]];` -> `r1 != 0`) as distinct dual exits returned in two CPU registers `(r0, r1)`. Declare fallibility after the parameter list (`u16 f() error ErrEnum`), fail with `error Variant;`, and handle with `catch` or `or`.",
        bad_zev: "\
enum IoErr { Timeout }
!u16 read_reg() {
    return error.Timeout;
}
fn main() {
    u16 x = try read_reg();
}",
        good_zev: "\
enum IoErr { Timeout }

u16 read_reg(bool ok) error IoErr {
    if (!ok) {
        error Timeout;
    }
    return 250;
}

fn main() {
    u16 x = read_reg(false) or 0;
    if (x == 0) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "P0008",
        summary: "C-style `for (;;)` or untyped `for i in 0..n` instead of typed bounded `for..in`",
        topic: "syntax",
        why: "Zev forbids C's arbitrary 3-clause `for (;;)` (which can hide unbounded loops inside atomic blocks) and untyped `for i in 0..n` bindings. Use explicit typed bounded `for (usize i in 0..n) { ... }`, inclusive `for (u8 i in 0..=max) { ... }`, or slice/array iteration `for (u8 b in slice) { ... }`.",
        bad_zev: "\
fn main() {
    for (u32 i = 0; i < 4; i += 1) {
        checkpoint();
    }
}",
        good_zev: "\
fn main() {
    [4]u8 data = [10, 20, 30, 40];
    u32 sum = 0;
    for (u8 b in data) {
        sum += b as u32;
    }
    for (usize i in 0..4) {
        sum += i as u32;
    }
    if (sum > 0) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "P0009",
        summary: "Rust `'a` lifetime annotation or `&mut` reference used instead of `*T` / `&x` / `[]T`",
        topic: "dma",
        why: "Zev achieves memory and DMA lease safety with zero `'a` lifetime syntax: `zev-sema` infers lexical scope depth and `wait_chunk()` lease provenance across pointers `*T`, slices `[]T`, and wrapper structs automatically.",
        bad_zev: "\
struct Reader<'a> {
    buf: &'a [u8],
}
fn step(acc: &mut i32) { *acc += 1; }",
        good_zev: "\
struct ByteReader {
    []u8  buf;
    usize pos;
}

fn step(*i32 acc) {
    *acc += 1;
}

fn main() {
    i32 x = 0;
    step(&x);
}",
    },
    ErrorDoc {
        code: "P0010",
        summary: "C-style `static` variable modifier used instead of Zev task frames, ISR `state`, or OS primitives",
        topic: "primitives",
        why: "Zev forbids C `static` storage specifiers on variables (`pub` is used strictly for top-level module exports): persistent per-task state is a normal local at the top of a `task`, ISR-private persistent counters use explicit `state Type x = init;` inside `isr`, immutable constants use `const`, and cross-context shared state uses `watch<T>` or `inbox<T, N>`.",
        bad_zev: "\
static u32 counter = 0;
task blink() {
    static u32 ticks = 0;
    loop every 10ms { ticks += 1; }
}",
        good_zev: "\
pub const u32 LIMIT = 100;
watch<u32> counter = 0;

pub task blink() {
    u32 ticks = 0;
    loop every 10ms {
        if (ticks < LIMIT) {
            ticks += 1;
            counter.publish(ticks);
        }
    }
}

fn main() {
    spawn blink() @normal;
}",
    },
    ErrorDoc {
        code: "P0011",
        summary: "Deprecated bare `every <period> { ... }` loop instead of explicit `loop every <period_expr> { ... }`",
        topic: "tasks",
        why: "Spelling periodic deadline loops `loop every <period_expr> { ... }` makes sequential loop control flow (`continue;`, `break;`, divergence) visually explicit, supports runtime integer/tick period expressions alongside duration literals (`20ms`), and automatically skips/resyncs missed deadlines on overrun instead of bursting back-to-back catch-up iterations.",
        bad_zev: "\
task ctrl() {
    every 20ms {
        checkpoint();
    }
}
fn main() { spawn ctrl() @high; }",
        good_zev: "\
watch<u32> ticks = 0;

task ctrl(u32 period_ms) {
    u32 count = 0;
    loop every period_ms {
        count += 1;
        ticks.publish(count);
    }
}

fn main() {
    spawn ctrl(20) @high;
}",
    },
    ErrorDoc {
        code: "P0012",
        summary: "Retired `chan<T, N>` type used instead of unified `inbox<T, N>`",
        topic: "primitives",
        why: "`chan<T, N>` and `inbox<T, N>` have been unified into a single bounded static queue `inbox<T, N>` supporting both non-blocking ISR/task `.post(v [, boost: @prio])` / `.try_send(v)` and task-only backpressured yielding `.send(v [, timeout: dur])`, alongside `.recv([timeout: dur])` and `.try_recv()`.",
        bad_zev: "\
chan<u32, 8> work_q;
fn main() {}",
        good_zev: "\
inbox<u32, 8> work_q;
watch<u32> processed = 0;

task producer() {
    u32 seq = 1;
    loop every 10ms {
        work_q.send(seq);
        seq += 1;
    }
}

task consumer() {
    loop {
        u32 item = work_q.recv();
        processed.publish(item);
    }
}

fn main() {
    spawn producer() @normal;
    spawn consumer() @high;
}",
    },
    ErrorDoc {
        code: "P0013",
        summary: "Retired `lease<T, N>` type used instead of `@dma_safe pool<T, K>` + `Lease<T>` + `dma_slot<T>`",
        topic: "dma",
        why: "Standalone `lease<T, N>` has been replaced by static buffer pools `@dma_safe pool<T, K>`, move-only linear `Lease<T>` handles, and single-occupancy `dma_slot<T>` handoff cells (or `@dma_safe ping_pong<T, N>` for continuous alternating circular DMA).",
        bad_zev: "\
lease<u8, 64> spi_buf;
fn main() {}",
        good_zev: "\
@dma_safe pool<[64]u8, 2> spi_pool;
dma_slot<[64]u8> spi_dma;

fn main() {
    Lease<[64]u8> buf = spi_pool.try_alloc() catch { return; };
    *[64]u8 raw = buf.get();
    (*raw)[0] = 0xAAu8;
    spi_dma.park(buf) catch { return; };
    bool freed = spi_dma.release();
    if (freed) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "P0014",
        summary: "Postfix C-style pointer/array/function-pointer syntax used instead of canonical prefix `*T` / `[N]T` / `fn(...)`",
        topic: "syntax",
        why: "Zev uses unambiguous left-to-right prefix type modifiers (`*T` for pointers, `[]T` for fat slices, `[N]T` for fixed arrays, and `fn(Args...) -> Ret` for function pointers) rather than C's inside-out postfix declarators (`T *name`, `T name[N]`, `T*`, `Ret (*cb)(Args)`).",
        bad_zev: "\
fn main() {
    u8 buf[16];
}",
        good_zev: "\
fn step(*u32 x) {
    *x += 1;
}
fn main() {
    [16]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    u32 val = buf[0] as u32;
    step(&val);
}",
    },

    // ── Semantic & Static-Verifier Diagnostics (E0001..E0034) ───────────────
    ErrorDoc {
        code: "E0001",
        summary: "Illegal raw mutable global variable declared at top level",
        topic: "primitives",
        why: "Raw mutable globals create data races between hardware ISRs, cooperative tasks, and symmetric multicore targets. Top-level state in Zev must be either an immutable `const` or a scheduler-aware static OS primitive (`watch<T>`, `inbox<T, N>`, `byte_ring<N>`, `signal`, `ping_pong<T, N>`, `pool<T, K>`, `dma_slot<T>`).",
        bad_zev: "\
u32 latest_rpm = 0;

fn main() {
    latest_rpm = 1500;
}",
        good_zev: "\
const u16 MAX_RPM = 3000;
watch<u16> latest_rpm = 0;

fn main() {
    latest_rpm.publish(MAX_RPM);
}",
    },
    ErrorDoc {
        code: "E0002",
        summary: "Task, ISR, or function reads/writes a raw mutable global (isolation violation)",
        topic: "primitives",
        why: "ISRs can preempt any task between instructions unless masked, and tasks run independently across turns/cores. Replacing raw shared globals with `watch<T>` (latest value) or `inbox<Ev, N>` (event queue with optional `boost: @critical`) makes cross-context dataflow atomic and wakeup-aware.",
        bad_zev: "\
bool overcurrent_latched = false;

isr EXTI0 {
    overcurrent_latched = true;
}
fn main() {}",
        good_zev: "\
enum FaultEv { Overcurrent }
inbox<FaultEv, 4> fault_inbox;
irq IRQ_EXTI0 = 0;

isr exti0_isr() {
    fault_inbox.post(Overcurrent, boost: @critical);
}

fn main() {
    IRQ_EXTI0.attach(exti0_isr);
    FaultEv ev = fault_inbox.recv(timeout: 10ms) or Overcurrent;
    switch (ev) {
        Overcurrent => { return; }
    }
}",
    },
    ErrorDoc {
        code: "E0003",
        summary: "`HardBlocking` call, direct `checkpoint()`, `defer` yield, or auto-yield `loop` inside `noyield`",
        topic: "yielding",
        why: "`noyield { ... }` guarantees task-atomicity: no other cooperative task can interleave mid-update, and transient locals stay in CPU registers. `SoftYielding` helper `fn`s (terminating functions that only call `checkpoint()`) automatically specialize a direct `0 B`-`.bss` stack clone inside `noyield`, whereas `HardBlocking` waits (`delay`, `recv`, `send`, `wait_chunk`), direct `checkpoint()`, and unbounded `loop` are rejected.",
        bad_zev: "\
u16 adc_sample() {
    delay(1ms);
    return 900;
}
fn main() {
    noyield {
        u16 s = adc_sample();
    }
}",
        good_zev: "\
u16 adc_sample() {
    delay(1ms);
    return 900;
}

u16 compute_duty(u16 sample) {
    return sample / 2;
}

fn main() {
    u16 s = adc_sample();
    noyield {
        u16 duty = compute_duty(s);
        if (duty > 0) {
            return;
        }
    }
}",
    },
    ErrorDoc {
        code: "E0004",
        summary: "Yielding call or `checkpoint()` inside hardware-IRQ-masked `critical { ... }`",
        topic: "yielding",
        why: "`critical { ... }` masks hardware interrupts (`cpsid i` / `csrci mstatus, 8`). Yielding to the scheduler while interrupts are masked would either deadlock waiting for an IRQ or corrupt interrupt-mask state across tasks.",
        bad_zev: "\
import chip.stm32g4 as hw;
fn main() {
    critical {
        hw.RCC.cr |= 0x0100_0000;
        delay(1ms);
    }
}",
        good_zev: "\
import chip.stm32g4 as hw;

fn main() {
    critical {
        hw.RCC.cr |= 0x0100_0000;
    }
    delay(1ms);
}",
    },
    ErrorDoc {
        code: "E0005",
        summary: "Yielding call, `checkpoint()`, or auto-yield `loop` inside an `isr` handler",
        topic: "yielding",
        why: "Hardware ISRs run on the interrupt context preempting the active task on the shared hardware stack; an ISR must execute in microseconds and return immediately. Perform minimal hardware safe-off in the `isr`, then wake a task non-blockingly with `inbox.post(Ev, boost: @critical)` or `signal.notify()`.",
        bad_zev: "\
fn write_flash_log() {
    checkpoint();
}
isr EXTI0 {
    write_flash_log();
}
fn main() {}",
        good_zev: "\
enum SupEv { Fault, Idle }
inbox<SupEv, 4> sup_inbox;
irq IRQ_EXTI0 = 0;

fn write_flash_log() {
    checkpoint();
}

isr fault_isr() {
    sup_inbox.post(Fault, boost: @critical);
}

fn main() {
    IRQ_EXTI0.attach(fault_isr);
    SupEv ev = sup_inbox.recv(timeout: 5ms) or Idle;
    switch (ev) {
        Fault => { write_flash_log(); }
        Idle => {}
    }
}",
    },
    ErrorDoc {
        code: "E0006",
        summary: "Passing a yielding function as a callback pointer to an `extern \"C\"` function",
        topic: "yielding",
        why: "Zev tasks share a single hardware stack. If C code invokes a Zev callback while C stack frames are active on the stack, a stackless coroutine yield would unwind right through the active C frames. Callbacks passed to C must be strictly non-yielding.",
        bad_zev: "\
extern \"C\" fn register_hook(fn() -> void cb);
fn bad_hook() {
    checkpoint();
}
fn main() {
    register_hook(bad_hook);
}",
        good_zev: "\
extern \"C\" fn register_hook(fn() -> void cb);
signal hook_fired;

fn fast_nonyielding_hook() {
    hook_fired.notify();
}

fn main() {
    register_hook(fast_nonyielding_hook);
}",
    },
    ErrorDoc {
        code: "E0007",
        summary: "Unbounded `loop { ... }` inside `critical { ... }`",
        topic: "yielding",
        why: "Hardware interrupts are masked inside `critical { ... }`, so execution time must be strictly bounded to protect real-time IRQ latency. Use a bounded `while (retries < N)` condition if polling a short hardware status bit, or move loops outside `critical`.",
        bad_zev: "\
fn main() {
    critical {
        loop {
            break;
        }
    }
}",
        good_zev: "\
fn main() {
    critical {
        u32 spins = 0;
        while (spins < 8) {
            spins += 1;
        }
    }
}",
    },
    ErrorDoc {
        code: "E0008",
        summary: "Pointer, slice, or wrapper struct escapes to an outer lexical scope (`dest_depth < origin_depth`)",
        topic: "dma",
        why: "Without a garbage collector or `'a` lifetime annotations, Zev enforces stack and DMA safety by tracking origin lexical depth (`static = 0`, `fn/task = 1`, nested `{ ... } = 2..N`). A reference or DMA slice created in an inner scope cannot be assigned into a variable declared in a longer-lived outer scope.",
        bad_zev: "\
@dma_safe ping_pong<u8, 64> rx_dma;
task listener() {
    loop {
        []u8 saved = rx_dma.wait_chunk();
        {
            []u8 inner = rx_dma.wait_chunk();
            saved = inner;
        }
    }
}
fn main() { spawn listener() @normal; }",
        good_zev: "\
struct ByteReader {
    []u8  buf;
    usize pos;
}

u8 peek_first(ByteReader r) {
    if (r.buf.len == 0) {
        return 0;
    }
    return r.buf[r.pos];
}

@dma_safe ping_pong<u8, 64> rx_dma;
watch<u8> last_hdr = 0;

task listener() {
    loop {
        []u8 half = rx_dma.wait_chunk();
        ByteReader reader = ByteReader{ buf: half, pos: 0 };
        last_hdr.publish(peek_first(reader));
    }
}

fn main() {
    spawn listener() @normal;
}",
    },
    ErrorDoc {
        code: "E0009",
        summary: "Returning a pointer, slice, or wrapper backed by a callee-local variable or DMA lease",
        topic: "dma",
        why: "Returning `&local` or a local-leased `wait_chunk()` slice upward out of a function would leave the caller holding a dangling pointer to a destroyed stack frame or unguarded DMA buffer. Return value types (scalars/structs) upward, or pass slices downward into helpers.",
        bad_zev: "\
*u32 bad_ptr() {
    u32 tmp = 42;
    return &tmp;
}
fn main() {}",
        good_zev: "\
[]u8 trim_header([]u8 packet) {
    if (packet.len < 2) {
        return packet;
    }
    return packet[2..packet.len];
}

@dma_safe ping_pong<u8, 32> rx_dma;
watch<usize> payload_len = 0;

task worker() {
    loop {
        []u8 raw = rx_dma.wait_chunk();
        []u8 body = trim_header(raw);
        payload_len.publish(body.len);
    }
}

fn main() {
    spawn worker() @normal;
}",
    },
    ErrorDoc {
        code: "E0010",
        summary: "Passing a non-static pointer or slice into `spawn` (directly or via boot helper)",
        topic: "dma",
        why: "Spawned tasks outlive the initial boot setup phase of `main()`. Task endpoint parameters must point to top-level static OS primitives (e.g. `&motor1_rpm`, `&uart1_dma`), never to transient local variables inside `main()` or boot helpers.",
        bad_zev: "\
task worker(*u32 cfg) {
    loop every 10ms {
        if (*cfg > 0) {}
    }
}
fn main() {
    u32 local_cfg = 10;
    spawn worker(&local_cfg) @normal;
}",
        good_zev: "\
watch<u32> shared_cfg = 10;

task worker(*watch<u32> cfg) {
    loop every 10ms {
        u32 v = cfg.read();
        if (v > 0) {}
    }
}

fn main() {
    spawn worker(&shared_cfg) @normal;
}",
    },
    ErrorDoc {
        code: "E0011",
        summary: "Stale DMA lease read after a subsequent `wait_chunk()` swap",
        topic: "dma",
        why: "`ping_pong<T, N>` double-buffers zero-copy DMA: while the CPU reads half A, DMA hardware overwrites half B. Calling `rx_dma.wait_chunk()` a second time swaps ownership so DMA can overwrite half A—making any slice, sub-slice, or wrapper struct derived from the first `wait_chunk()` stale. Copy out any scalar/struct fields you need before calling `wait_chunk()` again.",
        bad_zev: "\
@dma_safe ping_pong<u8, 64> rx_dma;
task bad_reader() {
    loop {
        []u8 first = rx_dma.wait_chunk();
        []u8 second = rx_dma.wait_chunk();
        u8 sum = first[0] + second[0];
    }
}
fn main() { spawn bad_reader() @normal; }",
        good_zev: "\
@dma_safe ping_pong<u8, 64> rx_dma;
watch<u8> combined = 0;

task good_reader() {
    loop {
        []u8 first = rx_dma.wait_chunk();
        u8 b0 = first[0];
        []u8 second = rx_dma.wait_chunk();
        u8 b1 = second[0];
        combined.publish(b0 + b1);
    }
}

fn main() {
    spawn good_reader() @normal;
}",
    },
    ErrorDoc {
        code: "E0012",
        summary: "Recursive call cycle across a yielding path",
        topic: "yielding",
        why: "Zev compiles every task and yielding function into a fixed-size static coroutine frame at compile time with zero heap and zero per-task stacks. A recursive yielding cycle (`a -> b -> a`) would require unbounded coroutine frame nesting. Rewrite yielding recursion as an iterative `while` or `loop`.",
        bad_zev: "\
fn step_a(u32 n) {
    checkpoint();
    if (n > 0) { step_b(n - 1); }
}
fn step_b(u32 n) {
    step_a(n);
}
fn main() { step_a(3); }",
        good_zev: "\
fn step_iterative(u32 count) {
    u32 rem = count;
    while (rem > 0) {
        checkpoint();
        rem -= 1;
    }
}

fn main() {
    step_iterative(3);
}",
    },
    ErrorDoc {
        code: "E0013",
        summary: "Duration literal or period is zero, sub-tick (truncates to 0), or not an exact tick multiple",
        topic: "tasks",
        why: "The cooperative scheduler's default tick period is `1ms` (`1000us`). Writing `loop every 500us` on a `1ms` tick (or `loop every 0`) would silently truncate to `0` ticks or drift by 50%. Use `>= 1ms` multiples of the tick period (or sub-millisecond hardware timers/ISRs for microsecond waveforms).",
        bad_zev: "\
task fast_poll() {
    loop every 500us {
        checkpoint();
    }
}
fn main() { spawn fast_poll() @high; }",
        good_zev: "\
watch<u32> samples = 0;

task fast_poll() {
    u32 cnt = 0;
    loop every 1ms {
        cnt += 1;
        samples.publish(cnt);
    }
}

fn main() {
    spawn fast_poll() @high;
}",
    },
    ErrorDoc {
        code: "E0014",
        summary: "`return` statement missing a value or returning wrong type for signature",
        topic: "errors",
        why: "Value-returning functions (`u16 foo()`) must return an expression assignable to their declared return type on normal exit paths, whereas `fn name()` (`void`) and `isr` items use bare `return;` without a value (and `task` items are immortal actors that use `restart;` instead of `return;`).",
        bad_zev: "\
u16 get_rpm() {
    return;
}
fn main() {}",
        good_zev: "\
u16 get_rpm() {
    return 1500;
}

fn main() {
    u16 r = get_rpm();
    if (r == 1500) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0015",
        summary: "`error` exit statement does not match function's `error` / `error ErrType` signature",
        topic: "errors",
        why: "Zev supports three exact `error` exit forms matched to the function header:\n  1. Untyped `u16 f() error` -> exit with bare `error;` (or `error, partial_val;`).\n  2. Typed `u16 f() error MyErr` -> exit with `error Variant;`.\n  3. Typed + partial progress `usize f() error MyErr` -> exit with `error Variant, partial_val;`.\nFunctions without `error` in their signature cannot use the `error` statement.",
        bad_zev: "\
enum ParseErr { BadCrc }
u16 check_frame() {
    error BadCrc;
}
fn main() {}",
        good_zev: "\
enum ParseErr { TooShort, BadCrc }

u16 check_frame(u8 len, bool crc_ok) error ParseErr {
    if (len < 5) {
        error TooShort;
    }
    if (!crc_ok) {
        error BadCrc;
    }
    return 1800;
}

fn main() {
    u16 rpm = check_frame(5, true) or 0;
    if (rpm > 0) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0016",
        summary: "Fallible call or timeout wait is not handled with `catch` or `or`",
        topic: "errors",
        why: "Ignoring hardware or protocol errors silently is a classic source of embedded field failures. Every call to an `error`-returning function—or primitive wait with `timeout: ...`—must explicitly handle failure with postfix `or <fallback>` or `catch (...) { ... }`.",
        bad_zev: "\
enum BusErr { Nack }
u8 i2c_read() error BusErr {
    error Nack;
}
fn main() {
    u8 val = i2c_read();
}",
        good_zev: "\
enum BusErr { Nack }

u8 i2c_read(bool present) error BusErr {
    if (!present) {
        error Nack;
    }
    return 0x42;
}

fn main() {
    u8 v1 = i2c_read(false) or 0;
    u8 v2 = i2c_read(true) catch (BusErr err) {
        return;
    };
    if (v1 + v2 > 0) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0017",
        summary: "Invalid `catch` parameter count/types or non-diverging `catch` block",
        topic: "errors",
        why: "A `catch` clause binds `(ErrType err)` (or `(ErrType err, RetType partial)` when inspecting partial progress) for typed errors, and takes no typed error binding for untyped `error` functions. When assigning the result (`u16 x = f() catch ...;`), the `catch` block must either diverge (`return;`, `continue;`, `break;`, `error ...;`) or evaluate to a replacement value.",
        bad_zev: "\
enum ParseErr { BadCrc }
u16 parse(u8 b) error ParseErr {
    if (b == 0) { error BadCrc; }
    return b;
}
fn main() {
    u16 x = parse(0) catch (u32 wrong_ty) {
    };
}",
        good_zev: "\
struct DecodedPacket {
    u8  cmd;
    u16 val;
}

enum IoErr { Timeout }

DecodedPacket read_pkt(u32 limit) error IoErr {
    if (limit > 4) {
        error Timeout, DecodedPacket{ cmd: 9, val: 4 };
    }
    return DecodedPacket{ cmd: 1, val: limit as u16 };
}

fn main() {
    u32 recoveries = 0;
    DecodedPacket pkt = read_pkt(8) catch (IoErr err, DecodedPacket partial) {
        recoveries += 1;
        DecodedPacket{ cmd: partial.cmd, val: partial.val + 100 }
    };
    if (recoveries == 1 && pkt.val == 104) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0018",
        summary: "Invalid `or` fallback expression (infallible LHS or mismatched fallback type)",
        topic: "errors",
        why: "Postfix `expr or fallback` unwraps a fallible expression (`error` function call or `recv(timeout: ...)`), substituting `fallback` of the exact same value type when `expr` fails. Do not attach `or` to an infallible call (like `inbox.recv()` without `timeout:`).",
        bad_zev: "\
enum Ev { Ping, Idle }
inbox<Ev, 4> q;
fn main() {
    Ev e = q.recv() or Idle;
}",
        good_zev: "\
enum Ev { Ping, Idle }
inbox<Ev, 4> q;

fn main() {
    Ev e = q.recv(timeout: 10ms) or Idle;
    switch (e) {
        Ping => { return; }
        Idle => { return; }
    }
}",
    },
    ErrorDoc {
        code: "E0019",
        summary: "Type mismatch in variable initialization, assignment, operator, or call argument",
        topic: "syntax",
        why: "Zev enforces strict width and signedness safety across assignments, operators, and function calls. Use matching integer/signed types or an explicit `expr as TargetType` scalar cast when narrowing or crossing signed/unsigned boundaries.",
        bad_zev: "\
fn main() {
    u16 rpm = true;
}",
        good_zev: "\
fn main() {
    u32 raw = 1800;
    u16 rpm = raw as u16;
    if (rpm == 1800) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0020",
        summary: "Undefined identifier, type, field, or invalid OS primitive method name",
        topic: "syntax",
        why: "Every variable, function, enum variant, struct field, and OS primitive method is resolved statically. Check primitive method names carefully: `watch` uses `.read()` / `.publish(v)`, unified `inbox` uses `.post(v [, boost: @prio])` / `.try_send(v)` / `.send(v [, timeout: dur])` / `.recv([timeout: dur])` / `.try_recv()`, `signal` uses `.notify()` / `.wait()`, and `ping_pong` uses `.wait_chunk()` / `.flip()`.",
        bad_zev: "\
watch<u16> rpm = 0;
fn main() {
    rpm.push(1200);
}",
        good_zev: "\
watch<u16> rpm = 0;

fn main() {
    rpm.publish(1200);
    u16 cur = rpm.read();
    if (cur == 1200) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0021",
        summary: "Duplicate definition of a top-level item or same-scope local variable",
        topic: "syntax",
        why: "Each top-level declaration (`fn`, `task`, `isr`, `struct`, `enum`, `const`, OS primitive) and each local variable within the same lexical block must have a unique name.",
        bad_zev: "\
fn main() {
    u16 duty = 100;
    u16 duty = 200;
}",
        good_zev: "\
fn main() {
    u16 duty = 100;
    duty = 200;
    if (duty == 200) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0022",
        summary: "Attempt to mutate an immutable `const` or non-lvalue expression",
        topic: "syntax",
        why: "`const` items live in read-only flash (`.rodata`) and cannot be modified at runtime. Use a local variable for task-private mutable state or `watch<T>` for shared single-writer state.",
        bad_zev: "\
const u32 MAX_DUTY = 1000;
fn main() {
    MAX_DUTY = 500;
}",
        good_zev: "\
const u32 MAX_DUTY = 1000;

fn main() {
    u32 current_duty = MAX_DUTY / 2;
    if (current_duty == 500) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0023",
        summary: "Invalid control flow (`break`/`continue` outside loop, `restart` outside `task`, or jump inside `defer`)",
        topic: "tasks",
        why: "`break;` and `continue;` must appear inside a loop (`loop`, `loop every`, `while`, or `for`), `restart;` is only valid inside a `task` body, `set_priority` belongs in `fn main()`, and `defer` cleanup blocks cannot jump out of scope via `return`, `error`, `break`, `continue`, or `restart`.",
        bad_zev: "\
fn helper() {
    restart;
}
fn main() {
    helper();
}",
        good_zev: "\
watch<u32> ticks = 0;

task worker() {
    u32 n = 0;
    loop every 5ms {
        if (n > 100) {
            restart;
        }
        n += 1;
        ticks.publish(n);
    }
}

fn main() {
    spawn worker() @normal;
    set_priority(@low);
}",
    },
    ErrorDoc {
        code: "E0024",
        summary: "Invalid type layout or operation (e.g. pointer/slice in `packed struct` or `void` variable)",
        topic: "syntax",
        why: "`packed struct` represents wire-exact protocol frames decoded via `Type.from_bytes(slice)`, so all fields must be fixed-width integer/endian scalars (`u8`, `u16be`, `u32le`, etc.—never pointers or fat slices). Use a normal `struct` for in-memory slice wrappers like `ByteReader`.",
        bad_zev: "\
packed struct BadWire {
    u8   op;
    []u8 payload;
}
fn main() {}",
        good_zev: "\
packed struct WireHdr {
    u8    op;
    u16be length;
    u16be crc;
}

struct PacketView {
    WireHdr hdr;
    []u8    payload;
}

fn main() {
    WireHdr h = WireHdr{ op: 1, length: 4, crc: 0x1234 };
    if (h.op == 1) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0025",
        summary: "Unreachable statement after `return`, `error`, `break`, `continue`, `restart`, `sys.exit`, or diverging `loop`",
        topic: "syntax",
        why: "In safety-critical firmware, dead statements after an unconditional exit or after a `break`-less steady-state `loop` / `loop every` almost always indicate a misplaced hardware teardown or missing `break;` condition. Place per-task initialization before `loop { ... }`, or register scope-exit cleanup before the loop/exit via `defer <stmt>;`.",
        bad_zev: "\
task worker() {
    loop every 10ms {
        checkpoint();
    }
    u32 dead_after_loop = 1;
}
fn main() { spawn worker() @normal; }",
        good_zev: "\
watch<u32> status = 0;

task worker() {
    status.publish(1);
    loop every 10ms {
        status.publish(status.read() + 1);
    }
}

fn main() {
    spawn worker() @normal;
}",
    },
    ErrorDoc {
        code: "E0026",
        summary: "Non-diverging `task` (`return;` inside a `task` or falling off a `task`'s closing `}`)",
        topic: "tasks",
        why: "Every `spawn` allocates a permanent static `.bss` task slot; a `task` that returns or falls off `}` silently leaves dead SRAM and can strand producer queues. Tasks model permanent subsystem actors (`loop { ... }` or `loop every <period> { ... }`), reset cleanly to entry via `restart;`, and park at `0%` CPU on `inbox.recv()`/`signal.wait()`. Move finite one-shot sequences (e.g. self-test, calibration) into ordinary yielding `fn`s.",
        bad_zev: "\
task one_shot() {
    checkpoint();
    return;
}
fn main() { spawn one_shot() @normal; }",
        good_zev: "\
enum Cmd { Step, ResetActor }
inbox<Cmd, 4> cmd_q;
watch<u32> boot_ok = 0;

fn run_boot_self_test() {
    checkpoint();
    boot_ok.publish(1);
}

task subsystem_actor() {
    u32 count = 0;
    loop {
        Cmd c = cmd_q.recv();
        switch (c) {
            Step => { count += 1; }
            ResetActor => { restart; }
        }
    }
}

fn main() {
    run_boot_self_test();
    spawn subsystem_actor() @normal;
}",
    },
    ErrorDoc {
        code: "E0027",
        summary: "Invalid `spawn` placement (`spawn` inside a `task`, `isr`, loop, recursion, or post-boot supervisor)",
        topic: "tasks",
        why: "To guarantee 1:1 compile-time `.bss` task slot allocation with zero heap, `spawn` is restricted to **Boot-Tree `spawn`**: `fn main()` (including conditional `if`/`switch` boot branches) and non-loop, non-recursive setup `fn`s called during `main()`'s sequential boot phase before its steady-state supervisor loop (`zevc ram` prints each slot's full provenance path `main -> init_bus -> worker`).",
        bad_zev: "\
task worker(u8 ch) {
    loop every 10ms {
        checkpoint();
    }
}
fn main() {
    for (u8 ch in 0..2) {
        spawn worker(ch) @normal;
    }
}",
        good_zev: "\
watch<u32> bus1_ticks = 0;
watch<u32> bus2_ticks = 0;

task bus_worker(*watch<u32> out, u32 step) {
    u32 acc = 0;
    loop every 10ms {
        acc += step;
        out.publish(acc);
    }
}

fn start_bus(*watch<u32> out, u32 step) {
    spawn bus_worker(out, step) @normal;
}

fn main() {
    start_bus(&bus1_ticks, 1);
    start_bus(&bus2_ticks, 2);
}",
    },
    ErrorDoc {
        code: "E0028",
        summary: "Static worst-case `fmt_buf(dst, f\"...\")` output exceeds known fixed `[N]u8` buffer capacity",
        topic: "runtime",
        why: "When `dst` in `fmt_buf(dst, f\"...\")` has a statically known byte capacity (`[N]u8`, `*[N]u8`, `dst[..]`, or constant `dst[a..b]`) and every `{...}` hole has a bounded maximum width (e.g. `u32` decimal `10 B`, `i32` `11 B`, `f32` `:.2` `14 B`, `enum` max variant length), `zevc` proves worst-case buffer fit at compile time (`worst <= cap` needs no `or`/`catch`!). Either size `dst` to fit the worst-case bound, or pass a runtime-sized `[]u8` slice and handle `FmtErr::Truncated` with `catch (FmtErr err, []u8 clamped)`.",
        bad_zev: "\
fn main() {
    [8]u8 dst;
    u32 rpm = 1800;
    []u8 line = fmt_buf(dst, f\"rpm={rpm}\");
    if (line.len > 0) {
        return;
    }
}",
        good_zev: "\
fn write_tagged([]u8 dyn_tag) {
    // 1. Statically proven fit (\"rpm=\" 4 B + u32 max 10 B = 14 B <= 16 B): zero `catch` needed!
    [16]u8 line_buf;
    u32 rpm = 1800;
    []u8 ok_line = fmt_buf(line_buf, f\"rpm={rpm}\");

    // 2. Runtime `[]u8` slice hole: recovers partial clamped slice on `FmtErr::Truncated`!
    [12]u8 tag_buf;
    []u8 tagged = fmt_buf(tag_buf, f\"tag={dyn_tag}\") catch (FmtErr err, []u8 clamped) {
        clamped
    };
    if (ok_line.len + tagged.len > 0) {
        return;
    }
}

fn main() {
    [4]u8 raw = [65, 66, 67, 68];
    write_tagged(raw[0..4]);
}",
    },
    ErrorDoc {
        code: "E0029",
        summary: "Invalid generic instantiation (arity/kind mismatch, non-const `usize` bound, uninferable type, or cycle)",
        topic: "syntax",
        why: "Zev monomorphizes `<T>` and `<T, usize N>` `struct`s and `fn`s before MIR lowering (reporting concrete Flash instantiations in `zevc ram` and `zevc check --json`) without C++ overloading or vtables. Every instantiation must match declared type vs `usize` const parameter kinds/arity, supply compile-time constant `usize` bounds (integer literals or `const` items, never runtime locals), have inferable or explicit type arguments, and stay non-recursive.",
        bad_zev: "\
struct FixedVec<T, usize N> {
    [N]T data;
    usize len;
}

fn main() {
    usize runtime_cap = 8;
    FixedVec<u16, runtime_cap> bad_vec;
}",
        good_zev: "\
const usize RING_CAP = 4;

struct FixedVec<T, usize N> {
    [N]T data;
    usize len;
}

T clamp<T>(T val, T lo, T hi) {
    if (val < lo) {
        return lo;
    }
    if (val > hi) {
        return hi;
    }
    return val;
}

fn main() {
    FixedVec<u16, RING_CAP> samples = FixedVec<u16, RING_CAP> {
        data: [100, 200, 300, 400],
        len: 4
    };
    f32 duty = clamp<f32>(1.25f32, 0.0f32, 1.0f32);
    if (samples.len == 4 && duty <= 1.0f32) {
        return;
    }
}",
    },
    ErrorDoc {
        code: "E0030",
        summary: "Module resolution error (missing module, circular `import`, or private symbol access)",
        topic: "runtime",
        why: "Top-level items in `.zev` modules are private by default unless prefixed with `pub`. `import path.to.mod as alias;` resolves relative to the importing file, workspace roots, `-I` search directories, and built-in `chip`/`bsp`/`std` modules without circular dependencies.",
        bad_zev: "\
import drivers.nonexistent as drv;
fn main() {}",
        good_zev: "\
import chip.stm32g4 as hw;
import std.math as math;

pub const u32 MAX_DUTY = 1000;

fn main() {
    hw.TIM1.ccr[0] = math.clamp<u32>(500, 0, MAX_DUTY);
}",
    },
    ErrorDoc {
        code: "E0031",
        summary: "Unsafe `mmio` register access (RMW on `@w1c`/`@wo`, write to `@ro`, or alias escape)",
        topic: "runtime",
        why: "`@w1c` (write-1-to-clear) registers must be assigned directly (`dma.ifcr = mask;`), never read-modify-written (`|=` / `&=`), because reading pending bits and writing them back would accidentally clear unrelated active interrupts! Similarly, `@wo` registers cannot be read, `@ro` registers cannot be written, and hardware registers cannot be copied by value or taken via `&reg`.",
        bad_zev: "\
mmio DmaRegs {
    @w1c u32 ifcr @ 0x04;
}
mmio DMA1 : DmaRegs @ 0x4002_0000;

fn main() {
    DMA1.ifcr |= 1 << 17;
}",
        good_zev: "\
mmio DmaRegs {
    @ro  u32 isr  @ 0x00;
    @w1c u32 ifcr @ 0x04;
}
mmio DMA1 : DmaRegs @ 0x4002_0000;

fn main() {
    u32 pending = DMA1.isr;
    DMA1.ifcr = (1 << 17) | (pending & 0);
}",
    },
    ErrorDoc {
        code: "E0032",
        summary: "Invalid Boot-Tree `irq.attach(...)` / `irq.attach_shared(...)` delegation or signature",
        topic: "runtime",
        why: "Hardware `irq` vectors are attached once during sequential boot (`fn main()` or non-loop, non-recursive boot helpers) with baked static pointers/MMIO handles/constants. Duplicate `.attach()` on the same execution path, attaching inside a `task`/`isr`/loop/recursion, signature mismatches, or passing non-static stack pointers are rejected at compile time.",
        bad_zev: "\
irq IRQ_EXTI0 = 6;
isr on_edge(u32 tag) {}

fn main() {
    IRQ_EXTI0.attach(on_edge(1));
    IRQ_EXTI0.attach(on_edge(2));
}",
        good_zev: "\
mmio GpioRegs {
    @wo u32 brr @ 0x28;
}
mmio GPIOA : GpioRegs @ 0x4800_0000;
irq IRQ_EXTI0 = 6;

isr on_edge(*GpioRegs gpio, u32 pin_mask) {
    gpio.brr = pin_mask;
}

fn main() {
    IRQ_EXTI0.attach(on_edge(GPIOA, 1));
}",
    },
    ErrorDoc {
        code: "E0033",
        summary: "Invalid ISR-private `state` declaration (`state` outside `isr`, uninitialized, or `&state` escape)",
        topic: "yielding",
        why: "Persistent per-attachment ISR storage uses `state Type x = init;` directly inside an `isr` body: every `state` variable must have an explicit initializer, cannot appear outside an `isr`, and cannot escape `&state_var` out of its declaring block (each Boot-Tree `vec.attach(...)` gets its own isolated `.bss`/`.data` instance automatically).",
        bad_zev: "\
u32 helper() {
    state u32 count = 0;
    count += 1;
    return count;
}",
        good_zev: "\
watch<u32> irq_hits = 0;
irq IRQ_EXTI0 = 6;

isr edge_isr(*watch<u32> out, u32 step) {
    state u32 count = 0;
    count += step;
    out.publish(count);
}

fn main() {
    IRQ_EXTI0.attach(edge_isr(&irq_hits, 1));
}",
    },
    ErrorDoc {
        code: "E0034",
        summary: "Linear `Lease<T>` ownership violation (leaked, double-moved, or invalidated)",
        topic: "dma",
        why: "`Lease<T>` is a compile-time move-only linear handle to a static `pool<T, K>` slot. Every live `Lease<T>` must be consumed exactly once across every control-flow path (via `.release()`, `slot.park(lease)`, `inbox.post(lease)` / `.try_send(lease)` / `.send(lease)`, `return lease;`, or passing into a function), cannot be read after move or overwritten while live, cannot disagree across `if`/`else` or `switch` arms, and cannot leak out of loops or early exits. Use `defer lease.release();` to keep `.get()` and `.addr()` accessible until scope exit without double-releasing.",
        bad_zev: "\
@dma_safe pool<[64]u8, 2> spi_pool;

fn leak_lease() {
    Lease<[64]u8> buf = spi_pool.try_alloc() catch { return; };
}

fn main() {
    leak_lease();
}",
        good_zev: "\
@dma_safe pool<[64]u8, 2> spi_pool;

fn use_lease() {
    Lease<[64]u8> buf = spi_pool.try_alloc() catch { return; };
    defer buf.release();
    *[64]u8 raw = buf.get();
    (*raw)[0] = 0x55u8;
}

fn main() {
    use_lease();
}",
    },
];
