//! Abstract Syntax Tree (AST) and source location definitions for Zev (`zevc`).
//!
//! Covers all v0 constructs from `DESIGN_V0.md` and `examples/`:
//! - Module imports & re-exports (`import`, `pub import`, `pub`)
//! - Attributes & priorities (`@dma_safe`, `@manual_yield`, `@critical`, `@high`, `@normal`, `@low`, `@core(N)`)
//! - Wire-exact `packed struct`, `struct`, `mmio`, `irq`, and `enum` declarations with endian & float primitives (`u16be`, `u32le`, `f32`, `f64`, etc.)
//! - Monomorphized `<T>` & `<T, usize N>` bounded generics on `struct`s, `impl` blocks, and `fn`s, plus `[N]T` generic array lengths
//! - Static OS & DMA primitives (`signal`, `watch<T>`, `inbox<T, N>`, `byte_ring<N>`, `wait_queue`, `ping_pong<T, N>`, `pool<T, K>`, `Lease<T>`, `dma_slot<T>`)
//! - Constants (`const`) and raw globals (parsed so `zev-sema` can reject illegal mutable globals cleanly)
//! - Execution items: `fn` (with dual-exit `error` / `error ErrType` and `extern "C"`), `task` blueprints, and `isr` handlers (with ISR-private `state`)
//! - Cooperative & atomic control flow: `wait { ... }`, `loop every <period>`, `loop`, `while`, `for (T x in ...)`, `defer`, `restart;`, `noyield`, `critical`, `checkpoint()`, `spawn`, `switch`
//! - Zero-heap `f"..."` format strings with compile-time resolved [`FormatSpec`]s
//! - Dual-exit error handling: `error;`, `error E;`, `error E, partial;`, postfix `or`, and postfix `catch`

use std::fmt;

pub mod builtins;
pub use builtins::{BuiltinFn, SysMethod, SYS_NAMESPACE};

// ─────────────────────────────────────────────────────────────────────────────
// Source Spans & Identifiers
// ─────────────────────────────────────────────────────────────────────────────

/// Source location span carrying both 0-indexed byte offsets (`start..end`)
/// and 1-indexed `line` / `col` coordinates for direct diagnostic reporting.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    /// Inclusive start byte offset in source (0-indexed).
    pub start: u32,
    /// Exclusive end byte offset in source (0-indexed).
    pub end: u32,
    /// 1-indexed start line number.
    pub line: u32,
    /// 1-indexed start column number (in bytes/ASCII columns).
    pub col: u32,
}

impl Span {
    /// Construct a new `Span`.
    pub const fn new(start: u32, end: u32, line: u32, col: u32) -> Self {
        Self {
            start,
            end,
            line,
            col,
        }
    }

    /// Construct a synthetic/dummy span (0..0 at 1:1) for tests or built-ins.
    pub const fn dummy() -> Self {
        Self {
            start: 0,
            end: 0,
            line: 1,
            col: 1,
        }
    }

    /// Merge two spans from the start of `self` to the end of `other`.
    pub const fn merge(self, other: Span) -> Self {
        let end = if other.end > self.end {
            other.end
        } else {
            self.end
        };
        Self {
            start: self.start,
            end,
            line: self.line,
            col: self.col,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// An identifier with its source [`Span`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }

    pub fn dummy(name: impl Into<String>) -> Self {
        Self::new(name, Span::dummy())
    }

    pub fn as_str(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Attributes & Scheduler Priorities
// ─────────────────────────────────────────────────────────────────────────────

/// Cooperative scheduler priority levels (`@critical = 0` highest .. `@low = 3` lowest).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

impl Priority {
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Critical => "@critical",
            Self::High => "@high",
            Self::Normal => "@normal",
            Self::Low => "@low",
        }
    }

    /// Default starvation bound in milliseconds per v0 design (`1 / 5 / 15 / 50 ms`).
    pub const fn starvation_ms(self) -> u32 {
        match self {
            Self::Critical => 1,
            Self::High => 5,
            Self::Normal => 15,
            Self::Low => 50,
        }
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An `@attribute` annotation on an item, statement, or spawn site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attr {
    pub kind: AttrKind,
    pub span: Span,
}

/// Supported v0 attributes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttrKind {
    /// `@dma_safe` — aligns/places static buffer for safe zero-copy DMA transfers.
    DmaSafe,
    /// `@manual_yield` — disables automatic `should_share()` loop yielding.
    ManualYield,
    /// `@critical`, `@high`, `@normal`, `@low`.
    Priority(Priority),
    /// `@core(N)` — pins a task to hardware core `N`.
    Core(u32),
    /// Any additional custom attribute `@name` or `@name(arg)`.
    Custom { name: String, arg: Option<String> },
}

// ─────────────────────────────────────────────────────────────────────────────
// Types & OS Primitive Endpoints
// ─────────────────────────────────────────────────────────────────────────────

/// Built-in scalar, floating-point, and wire-endian primitive types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrimType {
    U8,
    U16,
    U32,
    U64,
    Usize,
    I8,
    I16,
    I32,
    I64,
    Isize,
    F32,
    F64,
    U16Be,
    U16Le,
    U32Be,
    U32Le,
    U64Be,
    U64Le,
    Bool,
    Void,
}

impl PrimType {
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "u8" => Some(Self::U8),
            "u16" => Some(Self::U16),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
            "usize" => Some(Self::Usize),
            "i8" => Some(Self::I8),
            "i16" => Some(Self::I16),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "isize" => Some(Self::Isize),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            "u16be" => Some(Self::U16Be),
            "u16le" => Some(Self::U16Le),
            "u32be" => Some(Self::U32Be),
            "u32le" => Some(Self::U32Le),
            "u64be" => Some(Self::U64Be),
            "u64le" => Some(Self::U64Le),
            "bool" => Some(Self::Bool),
            "void" => Some(Self::Void),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Usize => "usize",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Isize => "isize",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U16Be => "u16be",
            Self::U16Le => "u16le",
            Self::U32Be => "u32be",
            Self::U32Le => "u32le",
            Self::U64Be => "u64be",
            Self::U64Le => "u64le",
            Self::Bool => "bool",
            Self::Void => "void",
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    pub const fn is_integer(self) -> bool {
        !matches!(self, Self::F32 | Self::F64 | Self::Bool | Self::Void)
    }

    pub const fn is_numeric(self) -> bool {
        self.is_integer() || self.is_float()
    }

    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Isize | Self::F32 | Self::F64
        )
    }

    pub const fn is_endian(self) -> bool {
        matches!(
            self,
            Self::U16Be | Self::U16Le | Self::U32Be | Self::U32Le | Self::U64Be | Self::U64Le
        )
    }

    /// Maps explicit integer width literal suffixes (`u8`..`isize`) to [`PrimType`].
    pub fn from_int_suffix(s: &str) -> Option<Self> {
        match s {
            "u8" => Some(Self::U8),
            "u16" => Some(Self::U16),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
            "usize" => Some(Self::Usize),
            "i8" => Some(Self::I8),
            "i16" => Some(Self::I16),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "isize" => Some(Self::Isize),
            _ => None,
        }
    }

    pub const fn is_big_endian(self) -> bool {
        matches!(self, Self::U16Be | Self::U32Be | Self::U64Be)
    }

    /// Size in bytes given `ptr_bytes` (e.g. `4` on 32-bit MCU, `8` on 64-bit host).
    pub const fn byte_size(self, ptr_bytes: usize) -> usize {
        match self {
            Self::Void => 0,
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::U16Be | Self::U16Le => 2,
            Self::U32 | Self::I32 | Self::F32 | Self::U32Be | Self::U32Le => 4,
            Self::U64 | Self::I64 | Self::F64 | Self::U64Be | Self::U64Le => 8,
            Self::Usize | Self::Isize => ptr_bytes,
        }
    }
}

impl fmt::Display for PrimType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Scheduler-aware static OS & DMA endpoint types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OsPrimitiveType {
    /// `signal` — lightweight event wakeup.
    Signal,
    /// `inbox<T, N>` — unified bounded static queue of capacity `cap` (supports `.post()`, `.send()`, `.recv()`, `.try_recv()`).
    Inbox { elem: Box<TypeExpr>, cap: usize },
    /// `*inbox<T>` — unsized pointer view over any `inbox<T, N>`.
    InboxUnsized { elem: Box<TypeExpr> },
    /// `watch<T>` — single-writer latest-value cell.
    Watch { elem: Box<TypeExpr> },
    /// `guarded<T>` — multi-writer shared cell accessible only inside `critical { ... }`.
    Guarded { elem: Box<TypeExpr> },
    /// `ping_pong<T, N>` — double-buffered zero-copy DMA endpoint (`N` = half/buffer size).
    PingPong { elem: Box<TypeExpr>, cap: usize },
    /// `pool<T, K>` — bounded zero-copy DMA/buffer pool of `cap` slots (1..=32).
    Pool { elem: Box<TypeExpr>, cap: usize },
    /// `dma_slot<T>` — single-slot DMA ownership parking cell for in-flight `Lease<T>`.
    DmaSlot { elem: Box<TypeExpr> },
    /// `byte_ring<N>` — lock-free SPSC circular + BipBuffer byte ring of capacity `cap`.
    ByteRing { cap: usize },
    /// `byte_ring` — unsized pointer view over any `byte_ring<N>`.
    ByteRingUnsized,
    /// `wait_queue` — low-level kernel waiter bitmask primitive for user-defined blocking sync types.
    WaitQueue,
}

impl OsPrimitiveType {
    pub const fn kind_name(&self) -> &'static str {
        match self {
            Self::Signal => "signal",
            Self::Inbox { .. } | Self::InboxUnsized { .. } => "inbox",
            Self::Watch { .. } => "watch",
            Self::Guarded { .. } => "guarded",
            Self::PingPong { .. } => "ping_pong",
            Self::Pool { .. } => "pool",
            Self::DmaSlot { .. } => "dma_slot",
            Self::ByteRing { .. } | Self::ByteRingUnsized => "byte_ring",
            Self::WaitQueue => "wait_queue",
        }
    }
}

/// A syntactic type expression with its source [`Span`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeExpr {
    pub kind: TypeKind,
    pub span: Span,
}

impl TypeExpr {
    pub const fn new(kind: TypeKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub const fn void(span: Span) -> Self {
        Self {
            kind: TypeKind::Prim(PrimType::Void),
            span,
        }
    }

    pub const fn is_void(&self) -> bool {
        matches!(self.kind, TypeKind::Prim(PrimType::Void))
    }
}

/// Kind of a compile-time monomorphized generic parameter (`<T>` or `<T, usize N>`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GenericParamKind {
    /// Type parameter (`T` in `struct FixedVec<T, usize N>` or `T clamp<T>(...)`).
    Type,
    /// Constant `usize` bound parameter (`usize N` in `struct FixedVec<T, usize N>`).
    ConstUsize,
}

/// A generic parameter declared on a `struct` or `fn`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericParam {
    pub name: Ident,
    pub kind: GenericParamKind,
    pub span: Span,
}

/// A generic argument supplied to a generic `struct` or `fn` (`<u32, 16>`, `<T, N>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GenericArg {
    Type(TypeExpr),
    Const(Box<Expr>),
}

/// Fixed-size array bound (`[16]u8` or generic/const `[N]T`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrayLen {
    /// Literal `usize` bound (`[16]u8`).
    Fixed(usize),
    /// Symbolic generic `usize` parameter or constant expression (`[N]T`, `[N + 1]u8`).
    Expr(Box<Expr>),
}

impl ArrayLen {
    pub const fn as_fixed(&self) -> Option<usize> {
        match self {
            Self::Fixed(n) => Some(*n),
            Self::Expr(_) => None,
        }
    }
}

/// Variants of [`TypeExpr`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeKind {
    /// Scalar / float / endian built-in (`u8`, `f32`, `u16be`, `i32`, `bool`, `void`, ...).
    Prim(PrimType),
    /// Unqualified user-defined type or generic type parameter (`CmdFrame`, `ParseErr`, `T`).
    Named(Ident),
    /// Module-qualified type path (`hw.Usart`, `chip.stm32g4.DmaChannel`).
    Path(Vec<Ident>),
    /// Generic type instantiation (`FixedVec<u32, 16>`, `FixedVec<T, N>`).
    Generic { base: Ident, args: Vec<GenericArg> },
    /// Module-qualified generic type instantiation (`col.FixedVec<u32, 16>`).
    GenericPath {
        path: Vec<Ident>,
        args: Vec<GenericArg>,
    },
    /// Fat slice `[]T` (`ptr` + `len`).
    Slice(Box<TypeExpr>),
    /// Fixed-size array `[16]u8`, `[N]T`, or `T[N]`.
    Array { elem: Box<TypeExpr>, len: ArrayLen },
    /// Raw / endpoint pointer `*T` or `T*` (e.g. `watch<u16> *target_rpm`, `*u8`).
    Ptr(Box<TypeExpr>),
    /// Scheduler-aware OS primitive (`watch<u16>`, `ping_pong<u8, 64>`, `inbox<E, 8>`, ...).
    OsPrimitive(OsPrimitiveType),
    /// Linear, move-only buffer lease `Lease<T>` allocated from a `pool<T, K>`.
    Lease(Box<TypeExpr>),
    /// Function pointer type `[noyield] fn(T1, T2) [-> Ret] [error [E]]`.
    FnPtr {
        params: Vec<TypeExpr>,
        ret: Box<TypeExpr>,
        error_spec: Option<Box<ErrorSpec>>,
        is_noyield: bool,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Program & Top-Level Items
// ─────────────────────────────────────────────────────────────────────────────

/// Syntactic `<...>(...)` sequence where `<...>` failed speculative generic arg parsing
/// (e.g. an unparenthesized constant expression like `lim<K + 1>(1)`). Purely syntactic:
/// sema checks whether the callee actually names a generic function before diagnosing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BareGenericCallCandidate {
    pub callee_name: String,
    pub display_callee: String,
    pub callee_span: Span,
    pub full_span: Span,
    pub rewrite: String,
}

/// A parsed `.zev` compilation unit (or merged multi-file AST).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub items: Vec<Item>,
    pub bare_generic_call_candidates: Vec<BareGenericCallCandidate>,
    pub span: Span,
}

impl Program {
    pub fn new(items: Vec<Item>, span: Span) -> Self {
        Self {
            items,
            bare_generic_call_candidates: Vec::new(),
            span,
        }
    }

    pub fn with_bare_generic_call_candidates(
        mut self,
        candidates: Vec<BareGenericCallCandidate>,
    ) -> Self {
        self.bare_generic_call_candidates = candidates;
        self
    }

    /// Merge multiple parsed files into a single `Program`.
    pub fn merge_all(programs: impl IntoIterator<Item = Program>) -> Self {
        let mut items = Vec::new();
        let mut bare_generic_call_candidates = Vec::new();
        let mut first_span: Option<Span> = None;
        let mut last_span = Span::dummy();
        for prog in programs {
            if first_span.is_none() {
                first_span = Some(prog.span);
            }
            last_span = prog.span;
            items.extend(prog.items);
            bare_generic_call_candidates.extend(prog.bare_generic_call_candidates);
        }
        let span = first_span.map_or_else(Span::dummy, |s| s.merge(last_span));
        Self {
            items,
            bare_generic_call_candidates,
            span,
        }
    }
}

/// A top-level declaration in a `.zev` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub attrs: Vec<Attr>,
    /// Whether the top-level declaration is exported across module boundaries (`pub`).
    pub is_pub: bool,
    pub kind: ItemKind,
    pub span: Span,
}

impl Item {
    /// Construct a module-private top-level [`Item`].
    pub fn new(attrs: Vec<Attr>, kind: ItemKind, span: Span) -> Self {
        Self {
            attrs,
            is_pub: false,
            kind,
            span,
        }
    }

    /// Construct a top-level [`Item`] with explicit visibility.
    pub fn with_pub(attrs: Vec<Attr>, is_pub: bool, kind: ItemKind, span: Span) -> Self {
        Self {
            attrs,
            is_pub,
            kind,
            span,
        }
    }
}

/// Top-level item kinds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ItemKind {
    Import(ImportDecl),
    Struct(StructDecl),
    Impl(ImplDecl),
    Enum(EnumDecl),
    Mmio(MmioDecl),
    Irq(IrqDecl),
    OsPrimitive(OsPrimitiveDecl),
    Const(ConstDecl),
    GlobalVar(GlobalVarDecl),
    Fn(FnDecl),
    Task(TaskDecl),
    Isr(IsrDecl),
}

/// Access qualifier for a typed `mmio` hardware register (`@rw`, `@ro`, `@wo`, `@w1c`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MmioAccess {
    /// Read-write register (`@rw`); reads, writes, and compound RMW allowed.
    Rw,
    /// Read-only register (`@ro`); writes and compound RMW forbidden.
    Ro,
    /// Write-only register (`@wo`); reads and compound RMW forbidden.
    Wo,
    /// Write-1-to-clear status register (`@w1c`); reads and plain `=` allowed, compound RMW forbidden.
    W1c,
}

impl MmioAccess {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rw => "@rw",
            Self::Ro => "@ro",
            Self::Wo => "@wo",
            Self::W1c => "@w1c",
        }
    }
}

/// A single register entry inside an [`MmioDecl`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmioRegDecl {
    pub access: MmioAccess,
    pub ty: TypeExpr,
    pub name: Ident,
    pub offset: Expr,
    pub span: Span,
}

/// Variants of a typed `mmio` declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MmioDeclKind {
    /// Reusable register layout (`mmio UartRegs { @rw u32 cr1 @ 0x00; ... }`).
    Layout { regs: Vec<MmioRegDecl> },
    /// Peripheral instance bound to an existing layout (`mmio USART1 : UartRegs @ 0x4001_1000;`).
    Instance { layout: TypeExpr, base_addr: Expr },
    /// Single-instance inline block (`mmio Rcc @ 0x4002_1000 { ... }`).
    Inline {
        base_addr: Expr,
        regs: Vec<MmioRegDecl>,
    },
}

/// Typed volatile memory-mapped I/O register layout or peripheral instance (`mmio ...`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmioDecl {
    pub name: Ident,
    pub kind: MmioDeclKind,
    pub span: Span,
}

/// Hardware interrupt token declaration (`irq IRQ_EXTI0 = 0;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrqDecl {
    pub name: Ident,
    pub number: Expr,
    pub span: Span,
}

/// `import a.b.c [as alias];`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportDecl {
    pub path: Vec<Ident>,
    pub alias: Option<Ident>,
    pub span: Span,
}

/// `[packed] struct Name[<T, usize N>] { Type field; ... }`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDecl {
    pub is_packed: bool,
    pub name: Ident,
    pub generic_params: Vec<GenericParam>,
    pub fields: Vec<StructField>,
    pub span: Span,
}

/// A field inside a [`StructDecl`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructField {
    pub attrs: Vec<Attr>,
    pub ty: TypeExpr,
    pub name: Ident,
    pub span: Span,
}

/// Receiver mode for a method declared inside an `impl` block (`self` vs `*self`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SelfReceiver {
    Value,
    Ptr,
}

/// A method or associated function declared inside an [`ImplDecl`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MethodDecl {
    pub is_pub: bool,
    pub receiver: Option<SelfReceiver>,
    pub func: FnDecl,
    pub span: Span,
}

/// Same-module inherent implementation block (`impl[<T, usize N>] StructName[<...>] { ... }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplDecl {
    pub generic_params: Vec<GenericParam>,
    pub target_name: Ident,
    pub target_generic_args: Vec<GenericArg>,
    pub methods: Vec<MethodDecl>,
    pub span: Span,
}

/// `enum Name { Variant1, Variant2, ... }`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumDecl {
    pub name: Ident,
    pub variants: Vec<EnumVariant>,
    pub span: Span,
}

/// A variant inside an [`EnumDecl`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumVariant {
    pub name: Ident,
    pub discriminant: Option<i64>,
    pub span: Span,
}

/// Top-level static OS primitive endpoint declaration
/// (e.g. `@dma_safe ping_pong<u8, 64> uart1_dma;`, `watch<u16> motor1_rpm = 0;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OsPrimitiveDecl {
    pub attrs: Vec<Attr>,
    pub ty: OsPrimitiveType,
    pub name: Ident,
    pub init: Option<Expr>,
    pub span: Span,
}

/// Top-level immutable constant (`const u32 MAX_RPM = 5000;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstDecl {
    pub ty: TypeExpr,
    pub name: Ident,
    pub value: Expr,
    pub span: Span,
}

/// Top-level raw mutable global (`u32 bad_global = 0;`).
///
/// Parsed into the AST so `zev-sema` can reject mutable raw globals with a precise
/// task/ISR isolation diagnostic instead of an opaque syntax error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalVarDecl {
    pub ty: TypeExpr,
    pub name: Ident,
    pub init: Option<Expr>,
    pub span: Span,
}

/// Function declaration or `extern "C"` prototype.
///
/// Supports both `fn main() { ... }` and explicit return-first syntax
/// `fn parse_setpoint([]u8 buf) -> u16 error ParseErr { ... }`, as well as
/// `extern "C" fn ...;` / `extern "C" RetType name(...);`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FnDecl {
    pub attrs: Vec<Attr>,
    pub is_extern_c: bool,
    pub name: Ident,
    pub generic_params: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub is_variadic: bool,
    pub ret_type: TypeExpr,
    pub error_spec: Option<ErrorSpec>,
    /// `Some(block)` for defined functions, `None` for `extern "C"` declarations ending in `;`.
    pub body: Option<Block>,
    pub span: Span,
}

/// Typed parameter in a `fn` or `task` signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub ty: TypeExpr,
    pub name: Ident,
    pub span: Span,
}

/// Function error exit specification (`error` or `error ErrType`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorSpec {
    /// Boolean error exit (`error` without payload type).
    Bare(Span),
    /// Typed error exit (`error ParseErr`).
    Typed(TypeExpr),
}

/// Reusable parameterized `task` blueprint (`task motor_control(watch<u16> *target, u8 ch) { ... }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskDecl {
    pub attrs: Vec<Attr>,
    pub name: Ident,
    pub params: Vec<Param>,
    pub body: Block,
    pub span: Span,
}

/// Hardware interrupt service routine (`isr EXTI0 { ... }` or `isr uart_isr(*UartDriver d, *UartRegs r) { ... }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IsrDecl {
    pub attrs: Vec<Attr>,
    pub vector: Ident,
    pub params: Vec<Param>,
    pub body: Block,
    pub span: Span,
}

// ─────────────────────────────────────────────────────────────────────────────
// Blocks & Statements
// ─────────────────────────────────────────────────────────────────────────────

/// Braced lexical block `{ stmt* }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

impl Block {
    pub fn new(stmts: Vec<Stmt>, span: Span) -> Self {
        Self { stmts, span }
    }
}

/// A statement inside a [`Block`], with optional `@attributes` (e.g. `@manual_yield loop { ... }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stmt {
    pub attrs: Vec<Attr>,
    pub kind: StmtKind,
    pub span: Span,
}

impl Stmt {
    pub fn new(kind: StmtKind, span: Span) -> Self {
        Self {
            attrs: Vec::new(),
            kind,
            span,
        }
    }

    pub fn with_attrs(attrs: Vec<Attr>, kind: StmtKind, span: Span) -> Self {
        Self { attrs, kind, span }
    }
}

/// Statement variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StmtKind {
    /// Explicit C-style local variable declaration (`u16 rpm = ...;`, `i32 integral = 0;`).
    VarDecl {
        ty: TypeExpr,
        name: Ident,
        init: Option<Expr>,
    },
    /// ISR-private persistent static variable (`state u32 count = 0;`).
    StateVar {
        ty: TypeExpr,
        name: Ident,
        init: Option<Expr>,
    },
    /// Assignment or compound assignment (`x = 1;`, `bad_packets += 1;`).
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    /// Normal return exit (`return;` or `return expr;`).
    Return(Option<Expr>),
    /// Dual-exit failure (`error;`, `error Reason;`, or `error Reason, partial_val;`).
    Error {
        reason: Option<Expr>,
        partial: Option<Expr>,
    },
    /// Conditional branch (`if (cond) { ... } [else { ... }]`).
    /// Note: `else if (...) { ... }` is represented as `else_block` containing a single `If` `Stmt`.
    If {
        cond: Expr,
        then_block: Block,
        else_block: Option<Block>,
    },
    /// Bounded conditional loop (`while (cond) { ... }`).
    While { cond: Expr, body: Block },
    /// Bounded integer range loop (`for (usize i in 0..8) { ... }` or `for (u32 i in 0..=n) { ... }`).
    ForRange {
        var_ty: TypeExpr,
        var_name: Ident,
        start: Box<Expr>,
        end: Box<Expr>,
        inclusive: bool,
        body: Block,
    },
    /// Bounded slice/array element loop (`for (u8 b in slice_or_array) { ... }`).
    ForEach {
        elem_ty: TypeExpr,
        elem_name: Ident,
        iter: Box<Expr>,
        body: Block,
    },
    /// Unbounded loop (`loop { ... }`).
    Loop { body: Block },
    /// Drift-free periodic loop (`loop every 20ms { ... }` or `loop every period_expr { ... }`).
    LoopEvery { period: Box<Expr>, body: Block },
    /// Deterministic scope-exit cleanup (`defer expr;` or `defer { ... }`).
    Defer(Box<Stmt>),
    /// Task-atomic non-yielding scope (`noyield { ... }`).
    NoYield { body: Block },
    /// Hardware-IRQ-masked atomic scope (`critical { ... }`).
    Critical { body: Block },
    /// Sequential pattern dispatch (`switch (ev) { Variant => { ... } }`).
    Switch {
        scrutinee: Expr,
        arms: Vec<SwitchArm>,
    },
    /// Bounded O(1) deterministic multi-source wait (`wait { ... }`).
    Wait {
        arms: Vec<WaitArm>,
        timer: Option<WaitTimerArm>,
    },
    /// Static task instantiation (`spawn motor_control(&motor1_rpm, 1) @critical @core(0);`).
    Spawn(SpawnStmt),
    /// `continue;`
    Continue,
    /// `break;`
    Break,
    /// Explicit task self-reset statement (`restart;`).
    Restart,
    /// Explicit cooperative yield point (`checkpoint();`).
    Checkpoint,
    /// Standalone nested lexical scope `{ ... }`.
    Block(Block),
    /// Expression statement (`setup_clocks();`).
    Expr(Expr),
}

/// Assignment operators.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AssignOp {
    Assign,       // =
    AddAssign,    // +=
    SubAssign,    // -=
    MulAssign,    // *=
    DivAssign,    // /=
    RemAssign,    // %=
    BitAndAssign, // &=
    BitOrAssign,  // |=
    BitXorAssign, // ^=
    ShlAssign,    // <<=
    ShrAssign,    // >>=
}

impl AssignOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Assign => "=",
            Self::AddAssign => "+=",
            Self::SubAssign => "-=",
            Self::MulAssign => "*=",
            Self::DivAssign => "/=",
            Self::RemAssign => "%=",
            Self::BitAndAssign => "&=",
            Self::BitOrAssign => "|=",
            Self::BitXorAssign => "^=",
            Self::ShlAssign => "<<=",
            Self::ShrAssign => ">>=",
        }
    }
}

/// A single arm in a `switch` statement (`Overcurrent => { ... }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchArm {
    pub pat: SwitchPat,
    pub body: Block,
    pub span: Span,
}

/// An event arm in a deterministic `wait { ... }` block (`Cmd c = q.recv() => { ... }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaitArm {
    pub bind: Option<(TypeExpr, Ident)>,
    pub call: Expr,
    pub body: Block,
    pub span: Span,
}

/// Kind of optional timer fallback arm in a `wait { ... }` block.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WaitTimerKind {
    Timeout,
    Deadline,
}

/// Optional timer fallback arm in a `wait { ... }` block (`timeout 50ms => { ... }` or `deadline t => { ... }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaitTimerArm {
    pub kind: WaitTimerKind,
    pub expr: Expr,
    pub body: Block,
    pub span: Span,
}

/// Pattern in a [`SwitchArm`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwitchPat {
    /// Enum variant or constant identifier (`Overcurrent`, `SupEvent.Overcurrent`, or `sm.MODE_A`).
    Ident {
        prefix: Option<Ident>,
        name: Ident,
        span: Span,
    },
    /// Module-qualified enum variant pattern (`sm.Kind.Alpha`).
    Qualified { path: Vec<Ident>, span: Span },
    /// Scalar literal (`0`, `1`, `true`, ...).
    Literal(Literal, Span),
    /// Wildcard / default arm (`_` or `default`).
    Wildcard(Span),
}

/// Static task instantiation statement (`spawn [mod.]task_fn(args...) [@prio] [@core(N)];`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpawnStmt {
    pub module: Option<Ident>,
    pub task: Ident,
    pub args: Vec<CallArg>,
    pub priority: Option<Priority>,
    pub core: Option<u32>,
    pub span: Span,
}

// ─────────────────────────────────────────────────────────────────────────────
// Expressions & Literals
// ─────────────────────────────────────────────────────────────────────────────

/// An expression with its source [`Span`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    pub const fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// Expression variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprKind {
    /// Scalar, duration, boolean, or string literal.
    Literal(Literal),
    /// Variable, enum variant, constant, or function identifier.
    Ident(Ident),
    /// Priority literal in expression position (`set_priority(@low)`, `boost: @critical`).
    Priority(Priority),
    /// Prefix unary operation (`&x`, `*p`, `!flag`, `~mask`, `-val`).
    Unary { op: UnaryOp, expr: Box<Expr> },
    /// Infix binary operation (`a + b`, `HALF | FULL`, `crc != frame.crc`, ...).
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Explicit scalar cast (`x as u16` or `(u16)x`).
    Cast { expr: Box<Expr>, ty: TypeExpr },
    /// Field or peripheral property access (`frame.crc`, `buf.len`, `hw.DMA1.CH5`).
    Field { target: Box<Expr>, field: Ident },
    /// Direct function call (`parse_setpoint(half)`, `clamp<f32>(v, 0.0, 1.0)`).
    Call {
        callee: Box<Expr>,
        generic_args: Vec<GenericArg>,
        args: Vec<CallArg>,
    },
    /// Method or associated-function call (`rx_dma.wait_chunk()`, `CmdFrame.from_bytes(buf[0..5])`,
    /// `sup_inbox.post(Overcurrent, boost: @critical)`).
    MethodCall {
        receiver: Box<Expr>,
        method: Ident,
        args: Vec<CallArg>,
    },
    /// Element indexing (`buf[i]`).
    Index { target: Box<Expr>, index: Box<Expr> },
    /// Sub-slice range (`buf[0..5]`, `buf[..5]`, `buf[3..]`).
    Slice {
        target: Box<Expr>,
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
    /// Struct literal (`ByteReader { buf: half, pos: 0 }`, `mod.FixedVec<u32, 4> { ... }`).
    StructLit {
        module: Option<Ident>,
        name: Ident,
        generic_args: Vec<GenericArg>,
        fields: Vec<FieldInit>,
    },
    /// Fixed array literal (`[0x01, 0x00, 0x64]`).
    ArrayLit(Vec<Expr>),
    /// Zero-heap compile-time unpacked `f"..."` format string (`f"temp={t:.2} pkt={pkt:?}"`).
    FormatString(Vec<FormatPart>),
    /// Fallback error/timeout handler (`sup_inbox.recv(timeout: 5s) or IdleCheck`).
    Or {
        expr: Box<Expr>,
        fallback: Box<Expr>,
    },
    /// Block error handler (`parse_setpoint(half) catch (ParseErr err) { ... }`).
    Catch {
        expr: Box<Expr>,
        clause: Box<CatchClause>,
    },
}

/// Alignment direction for padded format holes (`{x:>5}`, `{x:<12}`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FormatAlign {
    Left,
    Right,
}

/// Base/radix for padded format holes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FormatPadBase {
    /// Decimal for integers, plain bytes for text (`[]u8` / `[N]u8` / `str` / `*u8`).
    Default,
    Hex {
        upper: bool,
    },
    Bin,
}

/// Compile-time resolved format specifier inside an `f"..."` interpolation hole.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum FormatSpec {
    /// `{x}` — default type-directed formatting.
    #[default]
    Default,
    /// `{x:x}`, `{x:X}`, `{x:02x}`, `{x:04X}` — lower/upper hexadecimal with optional zero-padded width.
    Hex { upper: bool, width: Option<u8> },
    /// `{x:b}`, `{x:08b}` — binary with optional zero-padded width.
    Bin { width: Option<u8> },
    /// `{x:.2}`, `{x:.3}` — fixed-decimal formatting with `precision` fractional digits.
    Fixed { precision: u8 },
    /// `{x:e}`, `{x:.2e}` — scientific notation formatting (linked on demand for `--gc-sections`).
    Sci { precision: Option<u8> },
    /// `{pkt:?}` — compile-time struct/value debug expansion.
    Debug,
    /// Width/alignment-padded formatting (`{x:>5}`, `{x:<12}`, `{x:05}`, `{x:>05}`, `{x:>6x}`, ...).
    Padded {
        align: Option<FormatAlign>,
        zero_pad: bool,
        width: u8,
        base: FormatPadBase,
    },
}

/// Segment of a zero-heap `f"..."` format string (`ExprKind::FormatString`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatPart {
    /// Static literal chunk (`{{` and `}}` already unescaped to `{` and `}`).
    Literal(String),
    /// `{expr}` or `{expr:spec}` interpolation hole.
    Hole {
        expr: Box<Expr>,
        spec: FormatSpec,
        span: Span,
    },
}

/// Literal values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Literal {
    Int {
        value: u64,
        /// Optional explicit integer width suffix (`u8`, `i32`, `usize`, ...).
        suffix: Option<PrimType>,
    },
    Float {
        /// IEEE-754 `f64::to_bits()` representation of the parsed literal.
        bits: u64,
        /// Optional explicit float type suffix (`PrimType::F32` or `PrimType::F64`).
        suffix: Option<PrimType>,
    },
    Duration(DurationLit),
    Bool(bool),
    String(String),
}

impl Literal {
    pub fn float_from_f64(value: f64, suffix: Option<PrimType>) -> Self {
        Self::Float {
            bits: value.to_bits(),
            suffix,
        }
    }
}

/// Base of the Unicode BMP Private Use Area range (`U+E080..=U+E0FF`) used to preserve raw
/// `\x80`..`\xFF` byte escapes inside `String` / `f"..."` literals without colliding with
/// literal source UTF-8 characters (such as `'µ'` `U+00B5`).
pub const RAW_BYTE_ESCAPE_PUA_BASE: u32 = 0xE000;

/// Append a raw `\xHH` escaped `byte` into a Zev literal `String`.
///
/// - `0x00..=0x7F` are pushed directly as ASCII `char`s.
/// - `0x80..=0xFF` are encoded into the BMP Private Use Area `U+E080..=U+E0FF` so
///   [`decoded_string_bytes`] and [`decoded_string_byte_len`] recover the exact single byte
///   without corrupting multi-byte UTF-8 source text.
pub fn push_raw_byte_escape(s: &mut String, byte: u8) {
    if byte <= 0x7F {
        s.push(byte as char);
    } else if let Some(ch) = char::from_u32(RAW_BYTE_ESCAPE_PUA_BASE + byte as u32) {
        s.push(ch);
    }
}

/// Decode a Zev `String` or `FormatPart::Literal` into its exact target byte sequence,
/// mapping `U+E080..=U+E0FF` raw-byte escapes back to single `0x80..=0xFF` bytes and
/// emitting standard UTF-8 bytes for all other Unicode scalar values.
pub fn decoded_string_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut utf8_buf = [0u8; 4];
    for ch in s.chars() {
        if ('\u{E080}'..='\u{E0FF}').contains(&ch) {
            out.push(((ch as u32) - RAW_BYTE_ESCAPE_PUA_BASE) as u8);
        } else {
            out.extend_from_slice(ch.encode_utf8(&mut utf8_buf).as_bytes());
        }
    }
    out
}

/// Compute the exact decoded byte length of a Zev `String` or `FormatPart::Literal`
/// (`decoded_string_bytes(s).len()` without allocating).
pub fn decoded_string_byte_len(s: &str) -> usize {
    let mut len = 0usize;
    for ch in s.chars() {
        if ('\u{E080}'..='\u{E0FF}').contains(&ch) {
            len += 1;
        } else {
            len += ch.len_utf8();
        }
    }
    len
}

/// Largest scheduler deadline distance, in ticks (`i32::MAX`).
///
/// Scheduler deadlines are 32-bit tick counts compared with wrap-safe signed
/// subtraction (`now - deadline >= 0`), so a deadline `2^31` or more ticks ahead
/// is indistinguishable from one that has already expired.
pub const MAX_DURATION_TICKS: u64 = i32::MAX as u64;

/// Hardware/scheduler duration literal (`500us`, `1ms`, `5s`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DurationLit {
    pub value: u64,
    pub unit: DurationUnit,
    pub span: Span,
}

impl DurationLit {
    /// Convert this duration literal to microseconds (`us`).
    pub const fn as_micros(&self) -> u64 {
        match self.unit {
            DurationUnit::Us => self.value,
            DurationUnit::Ms => self.value.saturating_mul(1_000),
            DurationUnit::S => self.value.saturating_mul(1_000_000),
        }
    }
}

impl fmt::Display for DurationLit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.value, self.unit.as_str())
    }
}

/// Time unit suffix on a [`DurationLit`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DurationUnit {
    Us,
    Ms,
    S,
}

impl DurationUnit {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Us => "us",
            Self::Ms => "ms",
            Self::S => "s",
        }
    }
}

/// Prefix unary operators.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    AddrOf, // &
    Deref,  // *
    Not,    // !
    BitNot, // ~
    Neg,    // -
}

impl UnaryOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AddrOf => "&",
            Self::Deref => "*",
            Self::Not => "!",
            Self::BitNot => "~",
            Self::Neg => "-",
        }
    }
}

/// Infix binary operators.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,    // +
    Sub,    // -
    Mul,    // *
    Div,    // /
    Rem,    // %
    BitOr,  // |
    BitAnd, // &
    BitXor, // ^
    Shl,    // <<
    Shr,    // >>
    Eq,     // ==
    Ne,     // !=
    Lt,     // <
    Le,     // <=
    Gt,     // >
    Ge,     // >=
    And,    // &&
    Or,     // ||
}

impl BinaryOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Rem => "%",
            Self::BitOr => "|",
            Self::BitAnd => "&",
            Self::BitXor => "^",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::And => "&&",
            Self::Or => "||",
        }
    }
}

/// Positional or named call argument (`Overcurrent`, `timeout: 5s`, `boost: @critical`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallArg {
    pub name: Option<Ident>,
    pub value: Expr,
    pub span: Span,
}

impl CallArg {
    pub fn positional(value: Expr) -> Self {
        let span = value.span;
        Self {
            name: None,
            value,
            span,
        }
    }

    pub fn named(name: Ident, value: Expr, span: Span) -> Self {
        Self {
            name: Some(name),
            value,
            span,
        }
    }
}

/// Field initializer inside a [`ExprKind::StructLit`] (`buf: half`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInit {
    pub name: Ident,
    pub value: Expr,
    pub span: Span,
}

/// Postfix `catch [(ErrType err [, ValType partial])] { ... }` handler clause.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatchClause {
    pub err_param: Option<CatchParam>,
    pub partial_param: Option<CatchParam>,
    pub body: Block,
    pub span: Span,
}

/// Typed binding inside a [`CatchClause`] (`ParseErr err` or `usize bytes_read`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatchParam {
    pub ty: TypeExpr,
    pub name: Ident,
    pub span: Span,
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_merge_and_display() {
        let s1 = Span::new(10, 15, 2, 4);
        let s2 = Span::new(20, 28, 3, 12);
        let merged = s1.merge(s2);
        assert_eq!(merged, Span::new(10, 28, 2, 4));
        assert_eq!(merged.to_string(), "2:4");
    }

    #[test]
    fn prim_types_and_endianness() {
        assert_eq!(PrimType::from_name("u16be"), Some(PrimType::U16Be));
        assert_eq!(PrimType::from_name("u32le"), Some(PrimType::U32Le));
        assert_eq!(PrimType::from_name("usize"), Some(PrimType::Usize));
        assert_eq!(PrimType::from_name("unknown"), None);

        assert!(PrimType::U16Be.is_endian());
        assert!(PrimType::U16Be.is_big_endian());
        assert!(PrimType::U32Le.is_endian());
        assert!(!PrimType::U32Le.is_big_endian());
        assert_eq!(PrimType::U16Be.byte_size(4), 2);
        assert_eq!(PrimType::Usize.byte_size(4), 4);
        assert_eq!(PrimType::Usize.byte_size(8), 8);
    }

    #[test]
    fn duration_conversion() {
        let d_us = DurationLit {
            value: 500,
            unit: DurationUnit::Us,
            span: Span::dummy(),
        };
        let d_ms = DurationLit {
            value: 1,
            unit: DurationUnit::Ms,
            span: Span::dummy(),
        };
        let d_s = DurationLit {
            value: 5,
            unit: DurationUnit::S,
            span: Span::dummy(),
        };
        assert_eq!(d_us.as_micros(), 500);
        assert_eq!(d_ms.as_micros(), 1_000);
        assert_eq!(d_s.as_micros(), 5_000_000);
        assert_eq!(d_s.to_string(), "5s");
    }

    #[test]
    fn priority_ordering_and_starvation() {
        assert!(Priority::Critical < Priority::High);
        assert!(Priority::High < Priority::Normal);
        assert!(Priority::Normal < Priority::Low);
        assert_eq!(Priority::Critical.as_u8(), 0);
        assert_eq!(Priority::Low.as_u8(), 3);
        assert_eq!(Priority::Critical.starvation_ms(), 1);
        assert_eq!(Priority::Low.starvation_ms(), 50);
    }

    #[test]
    fn construct_program_and_merge() {
        let p1 = Program::new(
            vec![Item::new(
                vec![],
                ItemKind::Import(ImportDecl {
                    path: vec![Ident::dummy("chip"), Ident::dummy("stm32g4")],
                    alias: Some(Ident::dummy("hw")),
                    span: Span::new(0, 26, 1, 1),
                }),
                Span::new(0, 26, 1, 1),
            )],
            Span::new(0, 26, 1, 1),
        );
        let p2 = Program::new(vec![], Span::new(0, 10, 1, 1));
        let merged = Program::merge_all([p1, p2]);
        assert_eq!(merged.items.len(), 1);
    }

    #[test]
    fn wave1_ast_variants_and_unified_os_primitives() {
        let sp = Span::dummy();
        let u8_ty = TypeExpr::new(TypeKind::Prim(PrimType::U8), sp);
        let inbox = OsPrimitiveType::Inbox {
            elem: Box::new(u8_ty.clone()),
            cap: 16,
        };
        assert_eq!(inbox.kind_name(), "inbox");
        assert_eq!(OsPrimitiveType::Signal.kind_name(), "signal");
        assert_eq!(OsPrimitiveType::WaitQueue.kind_name(), "wait_queue");
        let pool = OsPrimitiveType::Pool {
            elem: Box::new(u8_ty.clone()),
            cap: 4,
        };
        let slot = OsPrimitiveType::DmaSlot {
            elem: Box::new(u8_ty.clone()),
        };
        assert_eq!(pool.kind_name(), "pool");
        assert_eq!(slot.kind_name(), "dma_slot");
        let _lease_ty = TypeExpr::new(TypeKind::Lease(Box::new(u8_ty.clone())), sp);

        let restart_stmt = Stmt::new(StmtKind::Restart, sp);
        let defer_stmt = Stmt::new(StmtKind::Defer(Box::new(restart_stmt.clone())), sp);
        let loop_every = Stmt::new(
            StmtKind::LoopEvery {
                period: Box::new(Expr::new(
                    ExprKind::Literal(Literal::Duration(DurationLit {
                        value: 20,
                        unit: DurationUnit::Ms,
                        span: sp,
                    })),
                    sp,
                )),
                body: Block::new(vec![defer_stmt], sp),
            },
            sp,
        );
        let for_range = Stmt::new(
            StmtKind::ForRange {
                var_ty: TypeExpr::new(TypeKind::Prim(PrimType::Usize), sp),
                var_name: Ident::dummy("i"),
                start: Box::new(Expr::new(
                    ExprKind::Literal(Literal::Int {
                        value: 0,
                        suffix: None,
                    }),
                    sp,
                )),
                end: Box::new(Expr::new(
                    ExprKind::Literal(Literal::Int {
                        value: 8,
                        suffix: None,
                    }),
                    sp,
                )),
                inclusive: true,
                body: Block::new(vec![], sp),
            },
            sp,
        );
        let for_each = Stmt::new(
            StmtKind::ForEach {
                elem_ty: u8_ty,
                elem_name: Ident::dummy("b"),
                iter: Box::new(Expr::new(ExprKind::Ident(Ident::dummy("buf")), sp)),
                body: Block::new(vec![], sp),
            },
            sp,
        );
        assert!(matches!(loop_every.kind, StmtKind::LoopEvery { .. }));
        assert!(matches!(
            for_range.kind,
            StmtKind::ForRange {
                inclusive: true,
                ..
            }
        ));
        assert!(matches!(for_each.kind, StmtKind::ForEach { .. }));
    }

    #[test]
    fn wave2_floats_generics_and_format_strings() {
        let sp = Span::dummy();
        assert_eq!(PrimType::from_name("f32"), Some(PrimType::F32));
        assert_eq!(PrimType::from_name("f64"), Some(PrimType::F64));
        assert!(PrimType::F32.is_float());
        assert!(PrimType::F64.is_float());
        assert!(!PrimType::F32.is_integer());
        assert!(!PrimType::F64.is_integer());
        assert!(PrimType::F32.is_numeric());
        assert!(PrimType::F32.is_signed());
        assert_eq!(PrimType::F32.byte_size(4), 4);
        assert_eq!(PrimType::F64.byte_size(4), 8);

        let lit = Literal::float_from_f64(3.25, Some(PrimType::F32));
        assert_eq!(
            lit,
            Literal::Float {
                bits: 3.25_f64.to_bits(),
                suffix: Some(PrimType::F32)
            }
        );

        let gp_t = GenericParam {
            name: Ident::dummy("T"),
            kind: GenericParamKind::Type,
            span: sp,
        };
        let gp_n = GenericParam {
            name: Ident::dummy("N"),
            kind: GenericParamKind::ConstUsize,
            span: sp,
        };
        assert_eq!(gp_t.kind, GenericParamKind::Type);
        assert_eq!(gp_n.kind, GenericParamKind::ConstUsize);

        let arr_fixed = ArrayLen::Fixed(16);
        let arr_sym = ArrayLen::Expr(Box::new(Expr::new(ExprKind::Ident(Ident::dummy("N")), sp)));
        assert_eq!(arr_fixed.as_fixed(), Some(16));
        assert_eq!(arr_sym.as_fixed(), None);

        let fstr = Expr::new(
            ExprKind::FormatString(vec![
                FormatPart::Literal("temp=".to_owned()),
                FormatPart::Hole {
                    expr: Box::new(Expr::new(ExprKind::Ident(Ident::dummy("t")), sp)),
                    spec: FormatSpec::Fixed { precision: 2 },
                    span: sp,
                },
                FormatPart::Hole {
                    expr: Box::new(Expr::new(ExprKind::Ident(Ident::dummy("pkt")), sp)),
                    spec: FormatSpec::Debug,
                    span: sp,
                },
            ]),
            sp,
        );
        assert!(matches!(fstr.kind, ExprKind::FormatString(_)));
    }

    #[test]
    fn raw_byte_escape_roundtrip_preserves_utf8_and_single_bytes() {
        let mut s = String::new();
        push_raw_byte_escape(&mut s, 0x00);
        push_raw_byte_escape(&mut s, 0x7F);
        s.push_str("5µs");
        push_raw_byte_escape(&mut s, 0x80);
        push_raw_byte_escape(&mut s, 0xAA);
        push_raw_byte_escape(&mut s, 0xFF);

        // 0x00, 0x7F, '5' (0x35), 'µ' ([0xC2, 0xB5]), 's' (0x73), 0x80, 0xAA, 0xFF
        let expected = vec![0x00, 0x7F, b'5', 0xC2, 0xB5, b's', 0x80, 0xAA, 0xFF];
        assert_eq!(decoded_string_bytes(&s), expected);
        assert_eq!(decoded_string_byte_len(&s), expected.len());
    }
}
