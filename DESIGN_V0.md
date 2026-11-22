# Zev Language-OS — Design Notes (v0)

## 1. Core Thesis

In desktop and server software, the operating system lives outside the binary behind a syscall boundary. In bare-metal firmware, **Platform + OS + Libraries + Application** always compile into one flat static binary—yet C forces artificial walls between them (RTOS port layers, manual coroutine state structs, callback tables, `#ifdef` config headers, linker scripts) because C only understands functions and structs, not tasks, interrupts, deadlines, or hardware ownership.

**Elevator Pitch:**
> *Go's "batteries-included" concurrency composability—backed by a zero-heap, single-stack cooperative bare-metal scheduler instead of a garbage collector.*

Where [SyntropicOS](../SyntropicOS/README.md) pushes C99 to its limits via Duff's-device protothreads, manual state structs, and custom AST linters, **Zev** (`zevc` / `.zev`) makes the cooperative kernel, static coroutine packing, and ISR/task safety native to the compiler.

---

## 2. Core Semantic Pillars

### 2.1 Entry Point, Boot-Tree `spawn` & Immortal Task Lifecycle
- **Phased Sequential Boot (`fn main()`):** Execution starts sequentially in `fn main()`, preserving top-to-bottom C readability and phased boot (e.g., running PLL/clock setup and self-tests *before* control loops start).
- **Reusable Task Blueprints:** `task` definitions are **parameterized, reusable blueprints** (not auto-running global singletons). Libraries can export or encapsulate internal worker tasks cleanly without hidden background execution, and boot setup can `spawn` multiple instances (e.g., one per UART or SPI bus).
- **Boot-Tree `spawn` (Static Topology Wiring):** `spawn` is permitted inside `fn main()` (including conditional `if` / `switch` boot-mode branches) **and** inside any setup `fn` called during `main()`'s sequential boot phase before the scheduler starts, while statically **forbidden** inside loops, recursion, `task`s, and `isr`s (`zev-sema` enforces that any `fn` transitively calling `spawn` is boot-only). Every static boot-tree call path maps 1:1 to a compile-time-allocated `.bss` task slot and coroutine state frame (`0 B` heap, zero per-thread stacks—all tasks execute on the single hardware stack), and `zevc ram` prints the full static spawn provenance path (e.g., `main -> usb.start -> usb::xfer_task @high`) so libraries can encapsulate internal worker tasks without hiding task counts, SRAM budgets, or priorities.
- **Immortal Task Lifecycle (`E0026`):** Tasks model permanent subsystem actors rather than dynamic threads:
  - Both `return;` and falling off the closing `}` of a `task` body are compile-time errors (`E0026`), preventing silent dead `.bss` slots and producer/consumer channel deadlocks.
  - **`restart;`:** Explicit self-reset statement inside a `task` that unwinds active `defer`s, restores any temporary priority boost, zeroes task-local state (preserving `spawn` parameters), and jumps cleanly back to task entry on fault recovery.
  - **Dormant / Triggered Tasks:** Background or CLI-triggered subsystems park on `inbox.recv()` or `signal.wait()` at `0%` CPU cost until needed.
  - **Finite / Mutually Exclusive Workflows:** Foreground sequences (e.g., power-on self-test, calibration, DFU) are ordinary yielding `fn` calls whose static coroutine frames automatically `union`-overlap in caller SRAM.
  - **Test / Sim Exit:** Deterministic host simulations and QEMU test harnesses terminate explicitly via `sys.exit(code)`.

### 2.2 Built-In Cooperative Scheduler, Loop Semantics & Power Management
- **Ranked Dispatch & Starvation Bounds:** Priorities (`@critical`, `@high`, `@normal`, `@low`) with configurable starvation windows (`1 / 5 / 15 / 50 ms`, `Overdue -> Work -> Poll` dispatch).
- **Demand-Driven Chunking (`should_share`):** Long background loops execute back-to-back at full speed when uncontended, yielding `MORE` only when an equal/higher-priority task wakes or a peer hits its starvation bound.
- **Init vs. Steady-State Loop Structure:** Inside a `task`, statements before `loop { ... }` represent one-time per-instance initialization and persistent hoisted state; any statement placed after a `break`-less diverging `loop` (or after `return`, `error`, `restart`, or `sys.exit`) is rejected as a compile-time unreachable-code error (`E0025`).
- **Periodic Deadline Loops (`loop every <period>`):**
  - Spelling periodic loops `loop every <period> { ... }` (trapping bare `every` via `P0011`) makes them visually unmistakable as imperative loops, accepts both duration literals and runtime period expressions (e.g., configurable control rates), and **skips/resyncs missed deadlines on overrun** (`next_wake += k * period` past `now`) instead of bursting back-to-back catch-up iterations.
- **Automatic Tickless Idle:** When all active tasks are blocked on primitives or timer deadlines, the synthesized runtime enters low-power sleep (`wfi`) until the next hardware IRQ or deadline.

### 2.3 Yielding & Three-Tier Function Effects
- **Call-Graph Inferred (Zero `async`/`.await` Coloring):** Calling a yielding function yields transparently on the shared stack, and live local variables across yields are packed into the caller's static frame via CFG backward-liveness analysis (union-aliasing disjoint child call frames).
- **Three-Tier Inferred Function Effects:**
  1. **Non-Yielding:** Contains no yields; compiles to a direct C/LLVM stack call (`0 B` `.bss`) and is legal everywhere, including inside `noyield`, `critical`, `isr`, `defer`, and `extern "C"` callbacks.
  2. **Soft-Yielding / Shareable:** Terminating compute functions (e.g., `fft`, `x25519`, `sha256`) that only yield cooperatively via polite `checkpoint()` calls (bounded `while` and `for..in` loops stay in CPU registers without auto-yielding):
     - In normal cooperative `task` flow, compiles as a nested coroutine (`union`-packed child frame) so background compute shares the CPU smoothly.
     - Inside `noyield { ... }`, **auto-specializes a direct stack-only clone** (`{fn}__noyield`, `0 B` `.bss`, `checkpoint()` stripped to a no-op) so the exact same library function runs at full register speed without code duplication by the user.
     - Remains **statically forbidden** inside `critical { ... }`, `isr` handlers, `defer`, and `extern "C"` callbacks so long compute routines can never stretch hardware interrupt-masked latency.
  3. **Hard-Blocking:** Awaits external time or events (`delay`, `inbox.recv`/`send`, `watch.wait`, `signal.wait`, `ping_pong.wait_chunk`, `loop every`, or non-terminating `loop`); legal only in cooperative `task` flow (and sequential `main()`).
- **Dual-Specialization Flash Visibility:** Whenever the compiler emits both the coroutine version and the `noyield` stack-specialized version of a soft-yielding function, `zevc ram` and `zevc check --json` explicitly report the dual-specialized function list so `.text` Flash trade-offs remain 100% transparent.
- **Explicit Atomicity Escape Hatches:**
  - `noyield { ... }` *(task-atomic)*: Forbids hard-blocking yields and scheduler interleaving while keeping hardware IRQs enabled; keeps tight DSP/PID temporaries (`duty`) in CPU registers.
  - `critical { ... }` *(hardware-atomic)*: Masks hardware interrupts (`PRIMASK` / `MIE`) and forbids both soft/hard yields and unbounded `loop`s.
  - `@manual_yield` + `checkpoint()`: Suppresses auto-yielding inside loops holding external hardware invariants (e.g., SPI `CS` asserted low) so yields only occur at clean transaction boundaries.

### 2.4 Task/ISR Isolation, Primitives & High-Throughput Driver Model
- **Strict Global Isolation:** Arbitrary shared mutable globals across tasks/ISRs are statically rejected (`E0001`/`E0002`), preserving race-freedom and dual-core portability (RP2040, ESP32).
- **Core Scheduler-Aware Static Primitives & Consolidation:**
  - `signal` — lightweight event wakeup (`wait`, `try_wait`, `notify`).
  - `inbox<T, N>` *(unified bounded queue; retired `chan<T, N>` trapped via `P0012`)* — single bounded static queue supporting non-blocking `.post(v [, boost: @prio])` / `.try_send(v)` (ISR- and task-safe), yielding backpressured `.send(v [, timeout: d])` (task-only), blocking `.recv([timeout: d])`, and non-blocking `.try_recv()`.
  - `watch<T>` — single-writer, zero-lock latest-value cell (`publish`, `read`, `wait`).
  - `@dma_safe ping_pong<T, N>` — dedicated to contiguous `2 * N` hardware circular DMA (`HALF_TRANSFER` / `FULL_TRANSFER` double-buffering across the `isr`/DMA ↔ `task` boundary; standalone v0.1 `lease<T, N>` retires into `pool<T, K>` + `Lease<T>` below).
- **Dataflow + Scheduling Fusion:** An `isr` can post with priority elevation (`sup_inbox.post(Overcurrent, boost: @critical)`), temporarily boosting the handler task until that turn completes before restoring its base priority.
- **Strictly Sequential Control Flow (No `select`):** Multi-source tasks pull typed events from a single `inbox` followed by `switch` (`enum` or `const`/integer arms), and timeouts use sequential deadlines (`recv(timeout: 5s) or IdleCheck`), keeping per-task wait state `O(1)`.
- **Whole-Program Visibility Optimizations *(queued)*:** Because `zevc` analyzes the complete static task/ISR call and primitive graph before codegen, it applies four zero-annotation whole-program optimizations:
  1. **Automatic IRQ-mask elision:** Task-to-task queues and 1-ISR SPSC queues elide `PRIMASK`/`MIE` interrupt masking automatically, emitting IRQ critical sections only when preempting writers/ISRs actually contend.
  2. **Singleton subsystem constant-folding (`instance_count == 1`):** Single-instance driver `struct`s and `task`s constant-fold their instance and MMIO pointers directly into immediates (`0` pointer-indirection cost).
  3. **Dead `watch` stripping (`readers == 0`):** Unread telemetry/debug `watch<T>` cells strip to `0 B` `.bss` and no-op `publish` sites automatically.
  4. **Static single-stack + `.bss` high-water proof:** Combines the non-recursive call graph (`main`, task step functions, and nested ISR preemption levels) with LLVM `.stack_sizes` metadata to prove worst-case total SRAM usage in `zevc ram`.
- **Top-Half / Bottom-Half High-Throughput Driver Model (USB / Wi-Fi / Radio):**
  - Hardware `isr`s act strictly as `< 2 µs` top-halves (acknowledge hardware flags, swap/enqueue descriptor, wake `@critical` driver task with `boost: @critical`), while protocol state machines run in a bottom-half `@critical` `task` with hardware interrupts enabled (interfacing with vendor binary blobs via `noyield`-verified `extern "C"` callbacks).
  - **Next-Stage Hardware & High-Throughput Driver Primitives *(under design)*:**
    1. **Typed `mmio` Register Blocks (`@rw`, `@ro`, `@wo`, `@w1c` + Widths & Arrays):**
       ```zev
       mmio UsbRegs @ 0x6000_F000 {
           @rw  u32 ep1_fifo  @ 0x00;
           @ro  u32 int_st    @ 0x0C;
           @w1c u32 int_clr   @ 0x14;   // Forbids `|=` / `&=` read-modify-write!
           @rw  u16 ep_cfg[8] @ 0x20;   // Sub-word widths (`u8`/`u16`/`u32`) & channel arrays
       }
       ```
       - Statically forbids writes to `@ro` registers and statically forbids read-modify-write (`|=`, `&=`) on `@w1c` / `@wo` interrupt-clear registers (`USB0.int_clr = flags;` only), preventing classic W1C interrupt-loss bugs; auto-generated from CMSIS-SVD via `zevc import-svd <chip.svd>`.
    2. **ISR-Private `state` Variables:**
       ```zev
       isr USB_IRQ {
           state u8 pending_addr = 0;   // Static `.bss` scoped exclusively to `USB_IRQ`
           ...
       }
       ```
       - Mutable static `.bss` storage scoped and visible exclusively to a single `isr` across hardware interrupts (e.g., USB `EP0` `pending_addr` state), preserving zero cross-task sharing.
    3. **Lock-Free SPSC `byte_ring<N>` (Wrap-Around & Grant Invariant):**
       ```zev
       byte_ring<512> cdc_rx;
       // ISR: `[]u8 dst = cdc_rx.grant_write(len) catch { ... };` or `cdc_rx.push(chunk);`
       // Task: `[]u8 view = cdc_rx.read_slice(); ... cdc_rx.consume(view.len);`
       ```
       - Single-producer/single-consumer atomic head/tail ring requiring zero IRQ masking, providing both wrap-safe `push(slice)` / `pop(slice)` AND fallible contiguous `grant_write(len) error` (BipBuffer-style wrap-ahead when space permits, returning `error` on backpressure rather than silently truncating at buffer wrap-around).
       - `zev-sema` automatically invalidates any `read_slice()` view upon `consume(n)` so dangling ring slices are rejected at compile time.
    4. **Bidirectional Zero-Copy `@dma_safe pool<T, K>` + Move-Only `Lease<T>` (Superseding Standalone `lease<T, N>`):**
       ```zev
       @dma_safe pool<EthFrame, 8> net_pool;
       dma_slot<EthFrame>          eth_tx_inflight;
       inbox<Lease<EthFrame>, 8>   rx_inbox;
       ```
       - **In-flight DMA ownership across interrupts:** Because hardware DMA spans from register kickoff until a subsequent completion IRQ, a move-only `Lease<T>` cannot auto-free at the end of a kickoff scope—an explicit static in-flight parking slot (`dma_slot<T>` or ISR-private `state`) holds `Lease<T>` ownership until the completion `isr` releases or recycles it.
       - **Leak-free full-queue backpressure:** When `rx_inbox.post(lease)` fails because the destination queue is full under a packet burst/flood, `.post()` either auto-recycles `lease` back to `net_pool` or rejects via `error` so scope-exit returns the slot cleanly—making permanent pool exhaustion impossible.
       - **IRQ-safe `O(1)` bitmap allocator & cache-line alignment:** Because both `task` and `isr` allocate/free concurrently across preemptible boundaries, `pool<T, K>` uses an atomic/IRQ-masked `u32` bitmap free-list (`O(1)` bounded latency, zero linked-list races), and `@dma_safe` enforces DMA/cache-line alignment on pool slots so adjacent buffers never false-share cache lines with active DMA.

### 2.5 Everyday Types, Bounded Generics, Dual-Exit Errors & Ergonomics
- **Explicit Types Everywhere:** C-style `Type name = ...` declarations (`u8`, `u16`, `i32`, `f32`, `f64`, `[]u8` fat slices)—no `let`/`var` hiding bit-widths, overflow boundaries, or static task RAM costs.
- **Wire-Exact Layout:** `packed struct` with native endian types (`u16be`, `u32le`) decoded safely via `Type.from_bytes(slice)`.
- **Monomorphized `<T>` / `<T, usize N>` Generics (No Ad-Hoc Overloading, No Trait Hierarchies):**
  - User `.zev` `struct`s (`FixedVec<T, usize N>`) and `fn`s (`clamp<T>(...)`) may declare `<T>` and `<T, usize N>` parameters monomorphized before MIR lowering (`E0029` on invalid instantiations), with concrete Flash instantiations reported in `zevc ram` and `zevc check --json`.
  - C++-style ad-hoc function overloading (which risks silent register-width promotion bugs) and user-defined trait hierarchies / `'a` lifetime annotations are intentionally omitted.
- **Distinct `return` vs. `error` Exit Paths:**
  - `return val;` always represents normal completion (`r1 == 0`, `r0 = val`).
  - `error [reason [, partial_val]];` represents failure (`r1 != 0`, optional partial progress `partial_val` in `r0` at zero extra register cost):
    - `error;` — simple boolean failure (no single-use enum boilerplate).
    - `error BadCrc;` — typed failure (error enums contain only real errors, no fake `OK`).
    - `error Timeout, bytes_read;` — typed failure *plus* partial progress in `r0`.
  - Callers handle failure symmetrically via `or <fallback>` or single-/multi-statement `catch ([ErrType err [, ValType partial]]) { ... }`.
- **Zero-Heap Streamable `f"..."` Formatting (Per-Call-Site Dispatch, Zero Global Handler Registry):**
  - **Compile-Time Unpacked Sink Writes & Static Fit Proof:** `f"..."` unpacks at compile time into direct chunk writes across `print`, `println`, and `fmt_buf(dst, f"...") -> []u8 error FmtErr` *(with `*byte_ring` sink support landing alongside `byte_ring<N>` in §6.3)*, proving worst-case buffer fit at compile time (`E0028`) when all arguments have statically bounded widths and returning dual-exit `error Truncated, clamped_slice;` when runtime `[]u8` slices exceed `dst.len`.
  - **Call-Site Tiered Float Formatting (No Global `_printf_float` Swapping):** Because `zevc` resolves each `{...}` specifier at compile time rather than routing through a monolithic runtime `printf` table, `{temp}` / `{temp:.2}` emits a direct call to a compact (~150 B Arduino-style / scaled-integer) fixed-decimal converter while `{val:e}` calls a scientific float converter—so `ld.lld --gc-sections` links scientific float code *only* if `:e` appears in the program, and independent libraries never conflict over global float-handler state.
  - **`enum` Names & `{pkt:?}` Debug Without Custom Format Traits:** App-wide `Display`/`Format` traits and struct formatter registries are omitted (avoiding console-vs-log format conflicts; callers write `f"{pt.x},{pt.y}"` explicitly), while `zevc` provides automatic `enum` variant-name strings (`f"{err}"` -> `"BadCrc"`, `.rodata` emitted only for formatted enums) and optional `{pkt:?}` compile-time struct field debug expansion.
- **Zero-Cost Control-Flow Conveniences:**
  - **`defer <stmt>;` / `defer { ... }`:** Executes deterministic LIFO scope-exit hardware/resource cleanup across `return`, `error`, `break`, `continue`, `restart`, and `sys.exit`.
  - **Bounded `for..in` Loops:** Supports `for (u8 b in slice)` and `for (usize i in 0..n)` / `0..=max` while keeping C's 3-clause `for (;;)` and untyped `for` forbidden (`P0008`).

### 2.6 Agent-First Diagnostics & Progressive Disclosure (`zevc doc`)
- **Foreign-Idiom Parser Traps (`P0001`..`P0012`):** Actively detects Rust/Zig/Go/C muscle-memory and retired syntax (`let`/`var`, `fn f(x: T) -> R`, `async`/`.await`, `select`, `match`, `catch |e|`, `!T`/`try`, C-style `for (;;)`, `'a`/`&mut`, `static`/`pub`, bare `every`, and `chan<T, N>`) and emits 1-shot idiomatic Zev rewrites.
- **Actionable Verifier Diagnostics (`E0001`..`E0029`) & `--json`:** Includes shortest-path transitive yield call chains (`isr -> f1 -> f2 -> checkpoint()`), static `fmt_buf` capacity proofs (`E0028`), generic instantiation checks (`E0029`), primitive method fuzzy suggestions (`watch.write` -> `watch.publish`), `= docs: run 'zevc doc CODE'`, and machine-readable `zevc check --json` (including `dual_specialized_functions` and `monomorphized_instances`).
- **Self-Documenting Binary (`zevc doc`):** Ships a ~80-line quick primer + Rosetta Stone, 8 sharded topic guides (`syntax`, `errors`, `tasks`, `yielding`, `primitives`, `dma`, `runtime`, `targets`), and `BAD` vs `GOOD` `.zev` explainers for all 41 error codes directly inside `zevc`.

### 2.7 Modules, Libraries & Core-vs-Library Boundary
- **Modules & Visibility (`import` + `pub`):** File and directory modules default to module-private declarations with explicit `pub` exports (`pub struct`, `pub fn`, `pub task`, `pub const`, `pub enum`, `pub mmio`, `pub irq`), replacing v0.1's single-file `P0010` placeholder trap on `pub` so subsystems encapsulate internal state and worker tasks cleanly.
- **Compile-Time `irq` Token Delegation (`attach` / `attach_shared`):**
  - Hardware `irq` vectors are compile-time SVD tokens bound inside boot-tree setup (`vec.attach(uart_isr(port, regs))`) with **zero RAM vector tables and zero runtime function pointers**—synthesizing Flash `.isr_vector` (ARM Cortex-M NVIC) and RISC-V `INTMTX` + `mtvec` (ESP32-C6) handlers with constant instance and MMIO pointers baked directly into codegen.
  - Exclusive `attach` catches accidental duplicate IRQ bindings across libraries at compile time, while shared hardware lines (`EXTI9_5`, `GPIO`) opt in explicitly via `attach_shared` to synthesize a static sequential poll chain inside one Flash ISR.
- **Subsystem `struct`s & Unsized Primitive Views (Zero Vtables, Zero Per-Capacity Flash Bloat):**
  ```zev
  pub struct UartDriver {
      byte_ring<256> rx;
      byte_ring<256> tx;
  }
  // `&drv.rx` (`*byte_ring<256>`) coerces to capacity-erased `*byte_ring`
  ```
  - Static `struct`s may embed OS primitives (`inbox`, `watch`, `byte_ring`, `signal`, `pool`), and sized primitive pointers coerce to capacity-erased views (`*byte_ring`, `*inbox<T>`, `*watch<T>`, analogous to `[N]u8` -> `[]u8`).
  - Enables reusable multi-instance drivers (`UartDriver bus1; UartDriver bus2;`) and protocol stacks over differently sized queues with **zero dynamic vtables and zero `<usize N>` protocol code duplication**, verified via static-path alias and SPSC checking in `zev-sema`.
- **Strict Core-vs-Library Boundary (Occam's Razor for Primitives):**
  - **Inside `zevc` (Core Language ONLY):** Strictly constructs that directly touch scheduler wait/wake state, coroutine frame lowering, static ISR/DMA memory provenance, volatile MMIO/IRQ codegen, or compile-time `f"..."` unpacking: `task`, `spawn`, `irq`, `mmio`, `packed struct`, `signal`, `watch`, unified `inbox`, `byte_ring`, `ping_pong` (contiguous circular DMA), and `pool`+`Lease` (retiring standalone v0.1 `lease`).
  - **Outside `zevc` (Plain `.zev` Libraries):**
    - `chip/*.zev` — SVD-generated `mmio` + `irq` maps (replacing hardcoded compiler `chip.*` intrinsics).
    - `bsp/*.zev` — board pinmux, clock tree, and DMA channel wiring.
    - `std/*` (`std/io.zev`, `std/bus.zev`, `std/fmt.zev`) — ordinary `.zev` structs and formatting helpers composed from `*byte_ring` / `*inbox` views (e.g., duplex byte streams, half-duplex RS-485 links, or shared I2C/SPI bus handles—**not** built-in compiler primitives).
    - Protocol & compute libraries (`modbus`, `cli`, `fft`, `x25519`) — pure `.zev` with zero `mmio`/`irq` imports, unit-testable deterministically on the host via in-memory primitives.
- **Explicit Compile-Time Unrolling Trade-Offs:**
  - **Whole-program Zev source compilation:** All `.zev` modules compile together for global isolation proofs, coroutine frame union-packing, and static Flash vector synthesis; closed vendor binary blobs link via `noyield`-verified `extern "C"`.
  - **Static-path subsystem arguments:** Subsystem wiring requires compile-time-resolvable static instance paths (`&bus1`, `&ports[0]`) and rejects runtime-indexed `&ports[runtime_i]` so SPSC, aliasing, and IRQ provenance remain statically decidable.
  - **Conditional boot-branch `irq` binding:** Synthesizes a minimal runtime dispatch flag *only* when mutually exclusive `if`/`else` boot branches bind the same physical `irq`; single-target `attach` emits zero-branch direct Flash dispatch.

---

## 3. Compiler Pipeline (`zevc`)

```mermaid
flowchart LR
    Src["Source (.zev) + Optional SVD"] --> Front["Parser & Idiom Traps (zev-parser)"]
    Front --> Verifier["Static Verifier (zev-sema: ISR Isolation, noyield, No Yield-Recursion, DMA Lease)"]
    Verifier --> MIR["Coroutine & Kernel MIR (zev-mir: Packed Static Frames + Synthesized Scheduler)"]
    MIR --> C99["Stage 0 Backend (zev-codegen-c): Freestanding C99"]
    MIR --> LLVM["Stage 1 Backend (zev-codegen-llvm): LLVM 23 IR -> opt (-O0..-Oz) -> llc + ld.lld -> ELF / .bin"]
```

- **Why a State-Machine MIR in the middle:** Coroutine lowering, backward-liveness variable hoisting, disjoint frame union-packing, and scheduler synthesis happen *above* the backend:
  - **Stage 0 (`zev-codegen-c`):** Emits clean freestanding C99 for deterministic host simulation, Clang ASan/UBSan verification (`zevc run --sanitize`), and legacy MCU toolchains.
  - **Stage 1 (`zev-codegen-llvm`):** Emits opaque-pointer LLVM 23 IR passed through middle-end `/bin/opt` + `/bin/llc` (`-O0`, `-O1`, `-O2` default, `-O3`, `-Os` `optsize`, and `-Oz` `minsize`—achieving ~38–42% `.text` Flash reductions over `-O0` while preserving `.iram1` cycle-exact routines and freestanding zero-libc startup) and linked via `/bin/ld.lld` into bare-metal ARM Cortex-M3 `.elf` and Espressif ESP32-C6 (`rv32imac`) Direct-Boot `.bin` images.

---

## 4. Reference Examples

- [`examples/motor_controller.zev`](examples/motor_controller.zev) + [`examples/motor_support.zev`](examples/motor_support.zev) — North Star dual-bus STM32G4 motor controller demonstrating `main()` + multi-instance `spawn`, periodic `noyield` PI loop, circular `ping_pong` UART DMA, `u16be` wire frames, ISR `boost: @critical`, and `20 B` disjoint child `union` SRAM savings.
- [`examples/sim_demo.zev`](examples/sim_demo.zev) — Deterministic ASan/UBSan host simulation timeline exercising big-endian DMA packets, `BadCrc` recovery, `EXTI0` priority boost/restore, and `LIGHT -> DEEP` sleep-lock clamping.
- [`examples/cortex_m_demo.zev`](examples/cortex_m_demo.zev) — Zero-C bare-metal ARM Cortex-M3 firmware executing live in `qemu-system-arm` with real NVIC preemption and `SysTick`.
- [`examples/esp32c6_demo.zev`](examples/esp32c6_demo.zev) — Zero-bootloader ESP32-C6-Pico (`rv32imac` Direct-Boot) firmware verified on physical silicon (`16 MHz SYSTIMER`, `.iram1` WS2812B RGB driver on `GPIO8`, debounced `GPIO9` button `boost: @critical`, and `USB_DEVICE` serial telemetry).

---

## 5. Edge-Case & Systems Rules

1. **Pointers & Slices Across Yields (Zero Lifetime Syntax):**
   Whole-program scope-depth and lease-origin inference tracks temporary pointers and slices automatically. Local wrapper structs (e.g., `ByteReader` holding a DMA slice) work naturally with zero `'a` annotations, while escaping a reference to an outer scope (`depth(dest) < depth(origin)`) or reading a leased DMA buffer past the next `wait_chunk()` is a compile error.
2. **Task Lifecycle & Operating Modes (No External Task Killing):**
   There is no external `kill`/`stop` API that could yank a task mid-hardware-transaction. Tasks are permanent subsystem actors (`restart;` resets a task cleanly to entry; dormant tasks park on `inbox`/`signal` at zero CPU cost), while mutually exclusive finite workflows (e.g., Calibration vs. DFU) are ordinary yielding `fn`s whose static local frames `union`-overlap automatically.
3. **Timebase & Automatic Sleep Locks:**
   Core-level critical sections and IRQ vector/controller ops are emitted natively by the compiler. For time, projects either initialize the standard core timer in `main()` after PLL setup (`sys.use_systick(170_000_000)`) or supply custom `now_ticks()` + `sleep_until(wake_tick, allowed_depth)` hooks (e.g., `LPTIM`/`RTC`/`SYSTIMER`). Active hardware primitives (`ping_pong`, `pool`/`Lease`) automatically clamp `allowed_depth` to keep peripheral clocks alive during active transfers.
4. **Hybrid Multicore Execution:**
   Because stackless coroutine frames leave nothing on the CPU stack at yield points, unpinned tasks can run dynamically across symmetric cores (e.g., RP2040, ESP32-S3) with zero stack copying (~15–25 cycles spinlock + barrier at yields), while `@core(N)` pins hard-real-time or core-local-IRQ tasks to dedicated silicon (and is required on asymmetric MCUs like Cortex-M7+M4). Single-core and fully-pinned builds compile out all multicore locks/barriers.
5. **C Interop (`extern "C"`):**
   External C functions are implicitly `noyield`. Any function callback passed *into* C (e.g., filesystem block-device or radio-stack hooks) is also statically enforced as `noyield` so a stackless yield can never unwind through active C stack frames on the shared hardware stack.

---

## 6. Implementation Status & Queued Compiler Work

### 6.1 Live in `zevc` v0.1 Today
- [x] Lexer, recursive-descent `.zev` parser, `f32`/`f64` float literals, `<T, usize N>` generics, `f"..."` interpolation, and foreign-idiom traps `P0001`–`P0012` (`crates/zev-parser`).
- [x] Whole-program static verifier `E0001`–`E0029`: `<T, usize N>` monomorphization, 3-tier call-graph yield inference, shortest-path yield traces, Boot-Tree `spawn` provenance, immortal `task` + unreachable-code safety, static `fmt_buf` capacity proofs, global/ISR isolation, `noyield`/`critical`/`defer`/`extern "C"` callback safety, zero-`'a` scope-depth & DMA `ping_pong` lease tracking, and fuzzy primitive/identifier hints (`crates/zev-sema`).
- [x] Built-in `zevc doc` progressive-disclosure primer, 8 topic guides, 41 verified `BAD`/`GOOD` error explainers, and `zevc check --json` (`crates/zev-cli`).
- [x] CFG state-machine MIR lowering, backward-liveness frame variable hoisting, disjoint child coroutine `union` packing, `SoftYielding` `noyield` clone specialization, compile-time `f"..."` / `{pkt:?}` expansion, and static zero-heap `zevc ram` budget & provenance reports (`crates/zev-mir`).
- [x] Stage-0 freestanding C99 emitter + deterministic host simulation harness verified under Clang `-fsanitize=address,undefined` (`crates/zev-codegen-c`).
- [x] Stage-1 native LLVM 23 backend (`crates/zev-codegen-llvm`) with `/bin/opt` + `/bin/llc` (`-O0`..`-O3`, `-Os`, `-Oz`) targeting bare-metal ARM Cortex-M3 `.elf` (`qemu-system-arm`) and physical Espressif ESP32-C6 (`rv32imac` Direct-Boot `.bin`).

### 6.2 Established Semantic Refinements (Wave 1 & Wave 2 Complete)
- [x] Enforce **Boot-Tree `spawn`** (`E0027`): permit `spawn` inside `fn main()` (including conditional `if` / `switch` boot branches) and boot-only setup `fn`s called before scheduler start; forbid `spawn` inside loops, recursion, `task`s, and `isr`s; print full static spawn provenance paths (`main -> usb.start -> usb::xfer_task @high`) in `zevc ram`.
- [x] Compile-time unreachable-code error (`E0025`) on any statement following a `break`-less diverging `loop`, `return`, `error`, `restart`, or `sys.exit`.
- [x] Immortal `task` enforcement (`E0026`: reject `return;` and falling off `}` in `task` bodies) + `restart;` self-reset statement + `sys.exit(code)` termination rules.
- [x] Replace bare `every <lit> { ... }` (`P0011`) with `loop every <period_expr> { ... }` supporting runtime period expressions and overrun deadline skip/resync.
- [x] Three-tier function effect inference (*Non-yielding* / *Soft-yielding* / *Hard-blocking*) with automatic `noyield` stack-clone specialization of `checkpoint()`-only functions and explicit dual-specialization reporting in `zevc ram` / `zevc check --json`.
- [x] Unify `inbox<T, N>` and `chan<T, N>` (`P0012`) into a single bounded `inbox<T, N>` supporting non-blocking `.post(v [, boost: @prio])` / `.try_send(v)` (ISR/task-safe), yielding backpressured `.send(v [, timeout: d])` (task-only), `.recv([timeout: d])`, and `.try_recv()`.
- [x] Zero-cost `defer <stmt>;` / `defer { ... }` LIFO cleanup across `return`, `error`, `restart`, and `sys.exit`, plus bounded `for (u8 b in slice)` and `for (usize i in 0..n)` / `0..=max` loops (keeping C's 3-clause `for (;;)` forbidden via `P0008`).
- [x] Monomorphized `<T>` / `<T, usize N>` generics (`E0029`) on user `struct`s and `fn`s with concrete Flash instantiation reporting in `zevc ram` and `zevc check --json` (omitting C++-style overloading and user trait hierarchies).
- [x] Zero-heap per-call-site `f"..."` formatting across `print`, `println`, and `fmt_buf` *(with `*byte_ring` sink support landing alongside `byte_ring<N>` in §6.3)*: compile-time buffer-fit proof `E0028` + `error Truncated, clamped_slice;`, call-site tiered fixed-decimal vs. `:e` scientific float converters for `--gc-sections`, on-demand `enum` variant names, and `{pkt:?}` struct debug expansion.

### 6.3 Next-Stage Modules, Driver Architecture, Primitives & Optimizations (Under Design)
- [ ] File/directory module system (`import` + private-by-default declarations + explicit `pub` exports, graduating `pub` out of `P0010`) and strict Core-vs-Library separation (`chip/*.zev` SVD maps replacing hardcoded `chip.*` intrinsics, `bsp/*.zev` board wiring, and plain-`.zev` `std/io`, `std/bus`, `std/fmt`).
- [ ] Subsystem `struct`s embedding OS primitives + capacity-erased unsized primitive views (`*byte_ring`, `*inbox<T>`, `*watch<T>`) with static-path alias/SPSC checking in `zev-sema` (zero dynamic vtables, zero `<usize N>` protocol code duplication).
- [ ] Compile-time `irq` token delegation (`vec.attach(handler(inst, regs))` and `attach_shared`) synthesizing zero-RAM Flash `.isr_vector` (ARM Cortex-M NVIC) and RISC-V `INTMTX` + `mtvec` (ESP32-C6) dispatch with baked constant instance pointers and compile-time duplicate-attachment checks.
- [ ] Typed `mmio` register blocks (`@rw`, `@ro`, `@wo`, `@w1c` forbidding RMW on W1C/WO, `u8`/`u16`/`u32` widths, register arrays) + `zevc import-svd <chip.svd>`.
- [ ] ISR-private persistent `state` variables (static `.bss` scoped exclusively to a single `isr`).
- [ ] Lock-free SPSC `byte_ring<N>` with wrap-safe `push`/`pop`, BipBuffer-style `grant_write(len) error`, and compile-time `read_slice()` invalidation on `consume(n)`.
- [ ] Bidirectional zero-copy `@dma_safe pool<T, K>` + move-only `Lease<T>` handles with `dma_slot<T>` in-flight parking across IRQs, leak-free full-queue `inbox.post(lease)` recycling, and IRQ-safe `O(1)` bitmap allocation + cache-line alignment (retiring standalone v0.1 `lease<T, N>`, while keeping `@dma_safe ping_pong<T, N>` for contiguous `2 * N` hardware circular `HALF_TRANSFER` / `FULL_TRANSFER` DMA).
- [ ] Whole-program visibility optimizations: automatic IRQ-mask elision on task-to-task and 1-ISR SPSC queues, singleton subsystem constant-folding (`instance_count == 1`), dead `watch` stripping (`readers == 0`), and static call-graph + LLVM `.stack_sizes` single-stack + `.bss` high-water proof in `zevc ram`.
