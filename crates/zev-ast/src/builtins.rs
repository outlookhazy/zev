//! Typed identifiers for built-in free functions and `sys.*` runtime namespace methods.
//!
//! Centralizing these symbols in `zev-ast` avoids duplicated raw string matching across
//! semantic checking, MIR lowering, and code generation backends.

/// Reserved runtime namespace identifier (`sys.*`).
pub const SYS_NAMESPACE: &str = "sys";

/// Built-in free functions recognized natively by the Zev compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFn {
    SetPriority,
    Checkpoint,
    Delay,
    DelayMs,
    Sleep,
    NowTicks,
    NowMs,
    Print,
    Println,
    FmtBuf,
    FmtRing,
    Assert,
}

impl BuiltinFn {
    /// Resolve a free identifier name into a known built-in function, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "set_priority" => Some(Self::SetPriority),
            "checkpoint" => Some(Self::Checkpoint),
            "delay" => Some(Self::Delay),
            "delay_ms" => Some(Self::DelayMs),
            "sleep" => Some(Self::Sleep),
            "now_ticks" => Some(Self::NowTicks),
            "now_ms" => Some(Self::NowMs),
            "print" => Some(Self::Print),
            "println" => Some(Self::Println),
            "fmt_buf" => Some(Self::FmtBuf),
            "fmt_ring" => Some(Self::FmtRing),
            "assert" => Some(Self::Assert),
            _ => None,
        }
    }

    /// Return the canonical Zev source spelling for this built-in function.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::SetPriority => "set_priority",
            Self::Checkpoint => "checkpoint",
            Self::Delay => "delay",
            Self::DelayMs => "delay_ms",
            Self::Sleep => "sleep",
            Self::NowTicks => "now_ticks",
            Self::NowMs => "now_ms",
            Self::Print => "print",
            Self::Println => "println",
            Self::FmtBuf => "fmt_buf",
            Self::FmtRing => "fmt_ring",
            Self::Assert => "assert",
        }
    }

    /// Whether this built-in blocks cooperatively (`delay`, `delay_ms`, `sleep`).
    pub const fn is_blocking_delay(self) -> bool {
        matches!(self, Self::Delay | Self::DelayMs | Self::Sleep)
    }
}

/// Built-in methods under the reserved `sys.*` runtime namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SysMethod {
    UseSystick,
    SetTick,
    SetTickPeriod,
    SetTickUs,
    Delay,
    Tick,
    TickMs,
    TickBy,
    SetPutc,
    SetClockMs,
    SetExit,
    SetFaultHandler,
    SetIdle,
    SetIrqRouter,
    RomCallU32,
    RomCallVoid,
    RomCallNoreturn,
    NowTicks,
    NowMs,
    CurrentPrio,
    SleepDepth,
    WfiCount,
    TraceSleep,
    Nop,
    PendIrq,
    FuzzIrqs,
    Exit,
    Log,
    LogKv,
    LogHex,
}

impl SysMethod {
    /// Complete list of valid `sys.*` method names for diagnostics and fuzzy suggestions.
    pub const ALL_NAMES: &'static [&'static str] = &[
        "use_systick",
        "tick",
        "tick_ms",
        "tick_by",
        "set_tick",
        "set_tick_period",
        "set_tick_us",
        "set_putc",
        "set_clock_ms",
        "set_exit",
        "set_fault_handler",
        "set_idle",
        "set_irq_router",
        "rom_call_u32",
        "rom_call_void",
        "rom_call_noreturn",
        "now_ms",
        "now_ticks",
        "current_prio",
        "sleep_depth",
        "wfi_count",
        "trace_sleep",
        "pend_irq",
        "fuzz_irqs",
        "nop",
        "exit",
        "log",
        "log_kv",
        "log_hex",
        "delay",
    ];

    /// Resolve a method identifier name on `sys` into a known [`SysMethod`].
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "use_systick" => Some(Self::UseSystick),
            "set_tick" => Some(Self::SetTick),
            "set_tick_period" => Some(Self::SetTickPeriod),
            "set_tick_us" => Some(Self::SetTickUs),
            "delay" => Some(Self::Delay),
            "tick" => Some(Self::Tick),
            "tick_ms" => Some(Self::TickMs),
            "tick_by" => Some(Self::TickBy),
            "set_putc" => Some(Self::SetPutc),
            "set_clock_ms" => Some(Self::SetClockMs),
            "set_exit" => Some(Self::SetExit),
            "set_fault_handler" => Some(Self::SetFaultHandler),
            "set_idle" => Some(Self::SetIdle),
            "set_irq_router" => Some(Self::SetIrqRouter),
            "rom_call_u32" => Some(Self::RomCallU32),
            "rom_call_void" => Some(Self::RomCallVoid),
            "rom_call_noreturn" => Some(Self::RomCallNoreturn),
            "now_ticks" => Some(Self::NowTicks),
            "now_ms" => Some(Self::NowMs),
            "current_prio" => Some(Self::CurrentPrio),
            "sleep_depth" => Some(Self::SleepDepth),
            "wfi_count" => Some(Self::WfiCount),
            "trace_sleep" => Some(Self::TraceSleep),
            "nop" => Some(Self::Nop),
            "pend_irq" => Some(Self::PendIrq),
            "fuzz_irqs" => Some(Self::FuzzIrqs),
            "exit" => Some(Self::Exit),
            "log" => Some(Self::Log),
            "log_kv" => Some(Self::LogKv),
            "log_hex" => Some(Self::LogHex),
            _ => None,
        }
    }

    /// Return the canonical Zev source method name.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::UseSystick => "use_systick",
            Self::SetTick => "set_tick",
            Self::SetTickPeriod => "set_tick_period",
            Self::SetTickUs => "set_tick_us",
            Self::Delay => "delay",
            Self::Tick => "tick",
            Self::TickMs => "tick_ms",
            Self::TickBy => "tick_by",
            Self::SetPutc => "set_putc",
            Self::SetClockMs => "set_clock_ms",
            Self::SetExit => "set_exit",
            Self::SetFaultHandler => "set_fault_handler",
            Self::SetIdle => "set_idle",
            Self::SetIrqRouter => "set_irq_router",
            Self::RomCallU32 => "rom_call_u32",
            Self::RomCallVoid => "rom_call_void",
            Self::RomCallNoreturn => "rom_call_noreturn",
            Self::NowTicks => "now_ticks",
            Self::NowMs => "now_ms",
            Self::CurrentPrio => "current_prio",
            Self::SleepDepth => "sleep_depth",
            Self::WfiCount => "wfi_count",
            Self::TraceSleep => "trace_sleep",
            Self::Nop => "nop",
            Self::PendIrq => "pend_irq",
            Self::FuzzIrqs => "fuzz_irqs",
            Self::Exit => "exit",
            Self::Log => "log",
            Self::LogKv => "log_kv",
            Self::LogHex => "log_hex",
        }
    }

    /// Whether this `sys.*` call is a pure value-returning query rather than a statement op.
    pub const fn is_expr_query(self) -> bool {
        matches!(
            self,
            Self::NowMs
                | Self::NowTicks
                | Self::CurrentPrio
                | Self::SleepDepth
                | Self::WfiCount
                | Self::RomCallU32
        )
    }

    /// Whether calling this `sys.*` method diverges cleanly (terminating the entry loop).
    pub const fn is_diverging_exit(self) -> bool {
        matches!(self, Self::Exit | Self::RomCallNoreturn)
    }
}
