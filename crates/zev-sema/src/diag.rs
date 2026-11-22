//! Structured semantic diagnostics with source line/column spans, idiomatic Zev fix-it hints,
//! and progressive-disclosure `zevc doc` pointers.

use std::fmt;
use zev_ast::Span;

/// Categories of semantic and static-verification errors enforced by `zev-sema`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DiagnosticKind {
    /// Raw mutable global variable declared at top level (`E0001`).
    IllegalMutableGlobal,
    /// `task`, `isr`, or `fn` reading or writing an illegal mutable global (`E0002`).
    GlobalIsolationViolation,
    /// Hard-blocking call, `loop every`, `checkpoint`, `defer` yield, or auto-yield `loop` inside `noyield { ... }` (`E0003`).
    YieldInNoYield,
    /// Yielding call, `loop every`, or `checkpoint` inside `critical { ... }` (`E0004`).
    YieldInCritical,
    /// Yielding call, `loop every`, `checkpoint`, or auto-yield `loop` inside `isr` (`E0005`).
    YieldInIsr,
    /// Passing a yielding function as a callback pointer to an `extern "C"` function (`E0006`).
    YieldingCallbackToExternC,
    /// Unbounded `loop { ... }` inside `critical { ... }` (`E0007`).
    UnboundedLoopInCritical,
    /// Escaping a pointer, slice, or ref-bearing struct to an outer lexical scope (`dest_depth < origin_depth`) (`E0008`).
    ScopeDepthEscape,
    /// Returning a pointer, slice, or ref-bearing struct backed by a callee-local variable or DMA lease (`E0009`).
    ReturnLocalReference,
    /// Passing a non-static pointer/slice/lease into `spawn` or a boot-spawning helper (`E0010`).
    SpawnNonStaticReference,
    /// Reading a variable or wrapper struct derived from `wait_chunk()` after a subsequent `wait_chunk()` (`E0011`).
    StaleDmaLease,
    /// Cycle in the call graph containing at least one yielding function (`E0012`).
    YieldingRecursion,
    /// Duration literal is zero, sub-tick (truncates to 0), or not an exact multiple of the tick period (`E0013`).
    InvalidDurationResolution,
    /// `return` value missing or incompatible with function signature (`E0014`).
    ReturnMismatch,
    /// `error` statement arity or types incompatible with function `error` specification (`E0015`).
    ErrorSpecMismatch,
    /// Fallible expression not handled with `catch` or `or` (`E0016`).
    UnhandledFallibleExpr,
    /// Invalid `catch` clause parameter count, types, or non-diverging body (`E0017`).
    InvalidCatchSignature,
    /// Invalid `or` fallback expression (`E0018`).
    InvalidOrFallback,
    /// General expression/statement type mismatch (`E0019`).
    TypeMismatch,
    /// Reference to an undeclared identifier, type, field, or method (`E0020`).
    UndefinedSymbol,
    /// Duplicate top-level or same-scope declaration (`E0021`).
    DuplicateDefinition,
    /// Attempt to mutate an immutable `const` or non-lvalue expression (`E0022`).
    ImmutableAssignment,
    /// `break`/`continue` outside a loop, `restart` outside a task, or control-flow transfer inside `defer` (`E0023`).
    InvalidControlFlow,
    /// Invalid type definition or unsupported operation (e.g. pointer/slice inside `packed struct`) (`E0024`).
    InvalidOperation,
    /// Statement following diverging control flow (`return`, `error`, `break`, `continue`, `restart`, `sys.exit`, or non-breaking `loop`) (`E0025`).
    UnreachableCode,
    /// `return` inside a `task` or a `task` body that can fall off its closing `}` without diverging (`E0026`).
    NonDivergingTask,
    /// `spawn` (or call to a boot-spawning `fn`) inside a `task`, `isr`, loop, recursive cycle, or post-supervisor code (`E0027`).
    InvalidSpawnContext,
    /// `fmt_buf(dst, f"...")` static worst-case formatted length exceeds known destination buffer capacity (`E0028`).
    FormatBufferTooSmall,
    /// Invalid `<T>` / `<T, usize N>` generic instantiation (arity, kind, inference, non-const `usize`, or recursion depth > 32) (`E0029`).
    GenericInstantiationError,
    /// Module resolution failure, import cycle, or cross-module visibility (`pub`) violation (`E0030`).
    ModuleError,
    /// Invalid `mmio` layout, by-value `mmio` usage, raw `&reg`, or `@rw`/`@ro`/`@wo`/`@w1c` access violation (`E0031`).
    MmioViolation,
    /// Invalid `irq.attach` / `irq.attach_shared` context, duplicate exclusive binding, or non-static/non-const handler argument (`E0032`).
    IrqAttachmentError,
    /// Invalid ISR `state` variable outside `isr`, missing/non-const initializer, reference-bearing type, or escaping `&state` (`E0033`).
    InvalidIsrState,
    /// Move-only `Lease<T>` linearity violation (use-after-move, unconsumed lease on exit, overwrite, branch mismatch, or illegal placement) (`E0034`).
    LinearLeaseViolation,
}

impl DiagnosticKind {
    /// All 34 semantic error kinds (`E0001`..`E0034`) in code order.
    pub const ALL: [Self; 34] = [
        Self::IllegalMutableGlobal,
        Self::GlobalIsolationViolation,
        Self::YieldInNoYield,
        Self::YieldInCritical,
        Self::YieldInIsr,
        Self::YieldingCallbackToExternC,
        Self::UnboundedLoopInCritical,
        Self::ScopeDepthEscape,
        Self::ReturnLocalReference,
        Self::SpawnNonStaticReference,
        Self::StaleDmaLease,
        Self::YieldingRecursion,
        Self::InvalidDurationResolution,
        Self::ReturnMismatch,
        Self::ErrorSpecMismatch,
        Self::UnhandledFallibleExpr,
        Self::InvalidCatchSignature,
        Self::InvalidOrFallback,
        Self::TypeMismatch,
        Self::UndefinedSymbol,
        Self::DuplicateDefinition,
        Self::ImmutableAssignment,
        Self::InvalidControlFlow,
        Self::InvalidOperation,
        Self::UnreachableCode,
        Self::NonDivergingTask,
        Self::InvalidSpawnContext,
        Self::FormatBufferTooSmall,
        Self::GenericInstantiationError,
        Self::ModuleError,
        Self::MmioViolation,
        Self::IrqAttachmentError,
        Self::InvalidIsrState,
        Self::LinearLeaseViolation,
    ];

    /// Stable error code (`"E0001"`..`"E0034"`).
    pub const fn code(self) -> &'static str {
        match self {
            Self::IllegalMutableGlobal => "E0001",
            Self::GlobalIsolationViolation => "E0002",
            Self::YieldInNoYield => "E0003",
            Self::YieldInCritical => "E0004",
            Self::YieldInIsr => "E0005",
            Self::YieldingCallbackToExternC => "E0006",
            Self::UnboundedLoopInCritical => "E0007",
            Self::ScopeDepthEscape => "E0008",
            Self::ReturnLocalReference => "E0009",
            Self::SpawnNonStaticReference => "E0010",
            Self::StaleDmaLease => "E0011",
            Self::YieldingRecursion => "E0012",
            Self::InvalidDurationResolution => "E0013",
            Self::ReturnMismatch => "E0014",
            Self::ErrorSpecMismatch => "E0015",
            Self::UnhandledFallibleExpr => "E0016",
            Self::InvalidCatchSignature => "E0017",
            Self::InvalidOrFallback => "E0018",
            Self::TypeMismatch => "E0019",
            Self::UndefinedSymbol => "E0020",
            Self::DuplicateDefinition => "E0021",
            Self::ImmutableAssignment => "E0022",
            Self::InvalidControlFlow => "E0023",
            Self::InvalidOperation => "E0024",
            Self::UnreachableCode => "E0025",
            Self::NonDivergingTask => "E0026",
            Self::InvalidSpawnContext => "E0027",
            Self::FormatBufferTooSmall => "E0028",
            Self::GenericInstantiationError => "E0029",
            Self::ModuleError => "E0030",
            Self::MmioViolation => "E0031",
            Self::IrqAttachmentError => "E0032",
            Self::InvalidIsrState => "E0033",
            Self::LinearLeaseViolation => "E0034",
        }
    }

    /// Progressive-disclosure documentation topic in `zevc doc <topic>` corresponding to this error.
    pub const fn doc_topic(self) -> &'static str {
        match self {
            Self::IllegalMutableGlobal | Self::GlobalIsolationViolation => "primitives",
            Self::YieldInNoYield
            | Self::YieldInCritical
            | Self::YieldInIsr
            | Self::YieldingCallbackToExternC
            | Self::UnboundedLoopInCritical
            | Self::YieldingRecursion => "yielding",
            Self::ScopeDepthEscape
            | Self::ReturnLocalReference
            | Self::SpawnNonStaticReference
            | Self::StaleDmaLease
            | Self::LinearLeaseViolation => "dma",
            Self::InvalidDurationResolution
            | Self::InvalidControlFlow
            | Self::NonDivergingTask
            | Self::InvalidSpawnContext => "tasks",
            Self::IrqAttachmentError
            | Self::ModuleError
            | Self::MmioViolation => "runtime",
            Self::InvalidIsrState => "yielding",
            Self::ReturnMismatch
            | Self::ErrorSpecMismatch
            | Self::UnhandledFallibleExpr
            | Self::InvalidCatchSignature
            | Self::InvalidOrFallback => "errors",
            Self::FormatBufferTooSmall => "runtime",
            Self::TypeMismatch
            | Self::UndefinedSymbol
            | Self::DuplicateDefinition
            | Self::ImmutableAssignment
            | Self::InvalidOperation
            | Self::UnreachableCode
            | Self::GenericInstantiationError => "syntax",
        }
    }

    /// Formatted CLI documentation pointer (`run 'zevc doc E00XX' or 'zevc doc <topic>'`).
    pub fn doc_command(self) -> String {
        format!(
            "run 'zevc doc {}' or 'zevc doc {}'",
            self.code(),
            self.doc_topic()
        )
    }

    /// Default actionable idiomatic Zev fix-it guidance for this error category.
    pub const fn doc_hint(self) -> &'static str {
        match self {
            Self::IllegalMutableGlobal => {
                "replace mutable globals with `const Type NAME = val;` (immutable), `watch<Type> name = init;` (latest value), or `inbox<Event, 8> name;` (queued events)"
            }
            Self::GlobalIsolationViolation => {
                "tasks and ISRs communicate exclusively through static OS primitives (`watch<T>`, `inbox<T, N>`, `signal`, `ping_pong<T, N>`, `pool<T, K>`, `dma_slot<T>`, `byte_ring<N>`)"
            }
            Self::YieldInNoYield => {
                "move hard-blocking calls/waits outside `noyield { ... }`, or use a bounded `for (usize i in 0..n)` / `while (cond)` loop inside `noyield`"
            }
            Self::YieldInCritical => {
                "`critical { ... }` masks hardware IRQs—move yielding calls/waits outside `critical` and keep only short atomic register ops inside"
            }
            Self::YieldInIsr => {
                "ISRs cannot block or yield; post an event (`inbox.post(Ev, boost: @critical);`, `signal.notify();`, or `dma.flip();`) and handle blocking work in a `task`"
            }
            Self::YieldingCallbackToExternC => {
                "stackless coroutine yields cannot unwind through C stack frames; pass a non-yielding `fn` to `extern \"C\"`"
            }
            Self::UnboundedLoopInCritical => {
                "IRQ-masked `critical { ... }` blocks must stay bounded; move `loop { ... }` outside `critical` or use a bounded `for` / `while` outside IRQ-masked code"
            }
            Self::ScopeDepthEscape => {
                "slices/pointers and structs wrapping them cannot escape to an outer lexical scope (`dest_depth < origin_depth`); pass them down into a helper `fn` or copy needed bytes into a value/array"
            }
            Self::ReturnLocalReference => {
                "callee stack locals and `wait_chunk()` DMA leases expire on return; return a decoded value struct (e.g. `CmdFrame`) or write into a caller-passed `[]u8` buffer"
            }
            Self::SpawnNonStaticReference => {
                "spawned tasks outlive local stack frames; pass pointers to top-level static OS primitives (`&motor_rpm`, `&uart_dma`) or scalars"
            }
            Self::StaleDmaLease => {
                "finish reading/decoding the current `wait_chunk()` slice (or copy bytes out) before calling `wait_chunk()` again on the same DMA endpoint"
            }
            Self::YieldingRecursion => {
                "static coroutine frame packing forbids recursion across yielding functions; rewrite the recursive cycle as a `loop { ... }` or `while` state machine"
            }
            Self::InvalidDurationResolution => {
                "use a duration that is an exact non-zero multiple of the tick period (default `1ms` = `1000us`), or set a finer tick via `sys.use_systick(hz, 500us);`"
            }
            Self::ReturnMismatch => {
                "use `return expr;` matching the declared return type (`Ret name(...)`), or `fn name(...)` + `return;` for `void` functions"
            }
            Self::ErrorSpecMismatch => {
                "declare `Ret name(...) error` for `error;` (0 args), or `Ret name(...) error ErrEnum` for `error Reason;` (1 arg) / `error Reason, partial_val;` (2 args)"
            }
            Self::UnhandledFallibleExpr => {
                "unwrap fallible calls with `expr or <fallback>` or `expr catch [(Err e [, Val partial])] { ... }`"
            }
            Self::InvalidCatchSignature => {
                "use `catch { ... }` for bare `error`, `catch (Err e) { ... }` for `error Err`, or `catch (Err e, Val partial) { ... }` for partial progress"
            }
            Self::InvalidOrFallback => {
                "`expr or <fallback>` requires `expr` to be fallible (`error` / `timeout:`) and `<fallback>` to match `expr`'s value type"
            }
            Self::TypeMismatch => {
                "use explicit C-style Zev types (`u8`, `u16`, `u32`, `i32`, `usize`, `bool`, `[]u8`, `*T`) and matching argument/operand types"
            }
            Self::UndefinedSymbol => {
                "check spelling and ensure types, constants, OS primitives, and functions are declared before or at top level"
            }
            Self::DuplicateDefinition => {
                "rename one of the colliding declarations so each symbol in the same namespace/scope is unique"
            }
            Self::ImmutableAssignment => {
                "`const` globals and OS primitive endpoints cannot be reassigned directly; use a local variable or primitive methods (`.publish(val)`, `.post(ev)`)"
            }
            Self::InvalidControlFlow => {
                "`break;` and `continue;` belong inside loops, `restart;` belongs inside a `task`, and `defer` blocks cannot transfer control flow out of scope"
            }
            Self::InvalidOperation => {
                "check Zev type/item constraints (`packed struct` holds wire-exact scalars only; `task` blueprints are instantiated via `spawn task_name(...)`)"
            }
            Self::UnreachableCode => {
                "remove unreachable statements after `return`, `error`, `break`, `continue`, `restart`, `sys.exit`, or a non-breaking `loop` / `loop every`"
            }
            Self::NonDivergingTask => {
                "tasks are permanent subsystem actors: run a steady-state `loop { ... }` or `loop every 10ms { ... }`, park on `inbox.recv()`/`signal.wait()`, use `restart;` to reset to entry, or move finite one-shot workflows into a yielding `fn`"
            }
            Self::InvalidSpawnContext => {
                "`spawn` is only permitted during sequential boot in `fn main()` and non-loop, non-recursive boot-helper `fn`s called before `main()`'s supervisor loop"
            }
            Self::FormatBufferTooSmall => {
                "increase the destination `[N]u8` buffer size to fit the worst-case `f\"...\"` byte count, or pass a `[]u8` slice and handle truncation via `catch (FmtErr err, []u8 clamped) { ... }`"
            }
            Self::GenericInstantiationError => {
                "supply matching `<Type, usize N>` generic arguments (with compile-time constant `usize` bounds and non-recursive type expansion)"
            }
            Self::ModuleError => {
                "verify the `import path.to.module as alias;` path, break any circular `import` cycles, and mark cross-module APIs with `pub`"
            }
            Self::MmioViolation => {
                "use natural alignment and scalar/array integer widths (`u8`, `u16`, `u32`) in `mmio`, pass `*Layout` pointers instead of by-value `mmio`, never take `&regs.reg`, and obey `@rw`/`@ro`/`@wo`/`@w1c` (no RMW on `@w1c`)"
            }
            Self::IrqAttachmentError => {
                "call `IRQ.attach(isr(args...))` or `IRQ.attach_shared(...)` only during sequential boot in `fn main()` (or non-loop boot helpers) with static primitive `&prim`, peripheral handles, or `const` arguments, without conflicting exclusive attachments"
            }
            Self::InvalidIsrState => {
                "declare `state Type name = const_init;` only inside an `isr`, using non-reference-bearing types and without escaping `&name` outside the ISR"
            }
            Self::LinearLeaseViolation => {
                "a move-only `Lease<T>` must be consumed exactly once along every control-flow exit via `.release()`, `dma_slot.park(lease)`, `inbox.post(lease)` / `.send(lease)`, or `return lease;` (or register `defer lease.release();` immediately)"
            }
        }
    }
}

/// A semantic error diagnostic with 1-indexed line/column location and optional context note.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub kind: DiagnosticKind,
    pub span: Span,
    pub file: Option<String>,
    pub message: String,
    pub note: Option<String>,
}

impl Diagnostic {
    pub fn new(kind: DiagnosticKind, span: Span, message: impl Into<String>) -> Self {
        Self {
            kind,
            span,
            file: None,
            message: message.into(),
            note: None,
        }
    }

    pub fn with_file(mut self, file: Option<String>) -> Self {
        self.file = file;
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Stable error code (`"E0001"`..`"E0034"`).
    pub const fn code(&self) -> &'static str {
        self.kind.code()
    }

    /// Documentation topic (`"primitives"`, `"yielding"`, `"dma"`, `"tasks"`, `"errors"`, `"runtime"`, `"syntax"`).
    pub const fn doc_topic(&self) -> &'static str {
        self.kind.doc_topic()
    }

    /// CLI documentation command hint (`"run 'zevc doc E00XX' or 'zevc doc <topic>'"`).
    pub fn doc_command(&self) -> String {
        self.kind.doc_command()
    }

    /// Specific fix-it note if set, or the category's default idiomatic hint.
    pub fn help_text(&self) -> &str {
        self.note
            .as_deref()
            .unwrap_or_else(|| self.kind.doc_hint())
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(file) = &self.file {
            write!(
                f,
                "{}:{}:{}: error[{}]: {}",
                file,
                self.span.line,
                self.span.col,
                self.kind.code(),
                self.message
            )?;
        } else {
            write!(
                f,
                "{}:{}: error[{}]: {}",
                self.span.line,
                self.span.col,
                self.kind.code(),
                self.message
            )?;
        }
        if let Some(note) = &self.note {
            write!(f, "\n  = note: {}", note)?;
        }
        write!(f, "\n  = docs: {}", self.kind.doc_command())?;
        Ok(())
    }
}

impl std::error::Error for Diagnostic {}
