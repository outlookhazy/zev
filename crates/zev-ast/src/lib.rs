//! Abstract Syntax Tree (AST) and source location definitions for the Zev Language-OS.
//!
//! Covers all v0 constructs from `DESIGN_V0.md` and `examples/motor_controller.zev`:
//! - Module imports (`import chip.stm32g4 as hw;`)
//! - Attributes & priorities (`@dma_safe`, `@manual_yield`, `@critical`, `@high`, `@normal`, `@low`, `@core(N)`)
//! - Wire-exact `packed struct`, `struct`, and `enum` declarations with endian primitives (`u16be`, `u32le`, etc.)
//! - Static OS & DMA primitives (`signal`, `chan<T, N>`, `watch<T>`, `inbox<T, N>`, `ping_pong<T, N>`, `lease<T, N>`)
//! - Constants (`const`) and raw globals (parsed so `zev-sema` can reject illegal mutable globals cleanly)
//! - Execution items: `fn` (with dual-exit `error` / `error ErrType` and `extern "C"`), `task` blueprints, and `isr` handlers
//! - Cooperative & atomic control flow: `every <dur>`, `loop`, `while`, `noyield`, `critical`, `checkpoint()`, `spawn`, `switch`
//! - Dual-exit error handling: `error;`, `error E;`, `error E, partial;`, postfix `or`, and postfix `catch`

use std::fmt;

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

/// Built-in scalar and wire-endian primitive types.
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

    pub const fn is_integer(self) -> bool {
        !matches!(self, Self::Bool | Self::Void)
    }

    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Isize
        )
    }

    pub const fn is_endian(self) -> bool {
        matches!(
            self,
            Self::U16Be | Self::U16Le | Self::U32Be | Self::U32Le | Self::U64Be | Self::U64Le
        )
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
            Self::U32 | Self::I32 | Self::U32Be | Self::U32Le => 4,
            Self::U64 | Self::I64 | Self::U64Be | Self::U64Le => 8,
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
    /// `chan<T, N>` — bounded static MPSC/SPSC queue of capacity `cap`.
    Chan { elem: Box<TypeExpr>, cap: usize },
    /// `inbox<T, N>` — bounded static Active-Object event queue of capacity `cap`.
    Inbox { elem: Box<TypeExpr>, cap: usize },
    /// `watch<T>` — single-writer, zero-lock latest-value cell.
    Watch { elem: Box<TypeExpr> },
    /// `ping_pong<T, N>` — double-buffered zero-copy DMA endpoint (`N` = half/buffer size).
    PingPong { elem: Box<TypeExpr>, cap: usize },
    /// `lease<T, N>` — static zero-copy buffer lending endpoint of size `cap`.
    Lease { elem: Box<TypeExpr>, cap: usize },
}

impl OsPrimitiveType {
    pub const fn kind_name(&self) -> &'static str {
        match self {
            Self::Signal => "signal",
            Self::Chan { .. } => "chan",
            Self::Inbox { .. } => "inbox",
            Self::Watch { .. } => "watch",
            Self::PingPong { .. } => "ping_pong",
            Self::Lease { .. } => "lease",
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

/// Variants of [`TypeExpr`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeKind {
    /// Scalar / endian built-in (`u8`, `u16be`, `i32`, `bool`, `void`, ...).
    Prim(PrimType),
    /// Unqualified user-defined type (`CmdFrame`, `ParseErr`, `SupEvent`).
    Named(Ident),
    /// Module-qualified type path (`hw.Usart`, `chip.stm32g4.DmaChannel`).
    Path(Vec<Ident>),
    /// Fat slice `[]T` (`ptr` + `len`).
    Slice(Box<TypeExpr>),
    /// Fixed-size array `[N]T` or `T[N]`.
    Array { elem: Box<TypeExpr>, len: usize },
    /// Raw / endpoint pointer `*T` or `T*` (e.g. `watch<u16> *target_rpm`, `*u8`).
    Ptr(Box<TypeExpr>),
    /// Scheduler-aware OS primitive (`watch<u16>`, `ping_pong<u8, 64>`, `inbox<E, 8>`, ...).
    OsPrimitive(OsPrimitiveType),
    /// Function pointer type `fn(T1, T2) -> Ret` (for C interop callbacks, etc.).
    FnPtr {
        params: Vec<TypeExpr>,
        ret: Box<TypeExpr>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Program & Top-Level Items
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed `.zev` compilation unit (or merged multi-file AST).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub items: Vec<Item>,
    pub span: Span,
}

impl Program {
    pub fn new(items: Vec<Item>, span: Span) -> Self {
        Self { items, span }
    }

    /// Merge multiple parsed files into a single `Program`.
    pub fn merge_all(programs: impl IntoIterator<Item = Program>) -> Self {
        let mut items = Vec::new();
        let mut first_span: Option<Span> = None;
        let mut last_span = Span::dummy();
        for prog in programs {
            if first_span.is_none() {
                first_span = Some(prog.span);
            }
            last_span = prog.span;
            items.extend(prog.items);
        }
        let span = first_span.map_or_else(Span::dummy, |s| s.merge(last_span));
        Self { items, span }
    }
}

/// A top-level declaration in a `.zev` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub attrs: Vec<Attr>,
    pub kind: ItemKind,
    pub span: Span,
}

/// Top-level item kinds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ItemKind {
    Import(ImportDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    OsPrimitive(OsPrimitiveDecl),
    Const(ConstDecl),
    GlobalVar(GlobalVarDecl),
    Fn(FnDecl),
    Task(TaskDecl),
    Isr(IsrDecl),
}

/// `import a.b.c [as alias];`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportDecl {
    pub path: Vec<Ident>,
    pub alias: Option<Ident>,
    pub span: Span,
}

/// `[packed] struct Name { Type field; ... }`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDecl {
    pub is_packed: bool,
    pub name: Ident,
    pub fields: Vec<StructField>,
    pub span: Span,
}

/// A field inside a [`StructDecl`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructField {
    pub ty: TypeExpr,
    pub name: Ident,
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
/// `u16 parse_setpoint([]u8 buf) error ParseErr { ... }`, as well as
/// `extern "C" fn ...;` / `extern "C" RetType name(...);`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FnDecl {
    pub attrs: Vec<Attr>,
    pub is_extern_c: bool,
    pub name: Ident,
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

/// Hardware interrupt service routine (`isr EXTI0 { ... }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IsrDecl {
    pub attrs: Vec<Attr>,
    pub vector: Ident,
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
    While {
        cond: Expr,
        body: Block,
    },
    /// Unbounded loop (`loop { ... }`).
    Loop {
        body: Block,
    },
    /// Drift-free periodic loop (`every 1ms { ... }`).
    Every {
        interval: DurationLit,
        body: Block,
    },
    /// Task-atomic non-yielding scope (`noyield { ... }`).
    NoYield {
        body: Block,
    },
    /// Hardware-IRQ-masked atomic scope (`critical { ... }`).
    Critical {
        body: Block,
    },
    /// Sequential pattern dispatch (`switch (ev) { Variant => { ... } }`).
    Switch {
        scrutinee: Expr,
        arms: Vec<SwitchArm>,
    },
    /// Static task instantiation (`spawn motor_control(&motor1_rpm, 1) @critical @core(0);`).
    Spawn(SpawnStmt),
    /// `continue;`
    Continue,
    /// `break;`
    Break,
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

/// Pattern in a [`SwitchArm`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwitchPat {
    /// Enum variant or constant identifier (`Overcurrent` or `SupEvent.Overcurrent`).
    Ident {
        prefix: Option<Ident>,
        name: Ident,
        span: Span,
    },
    /// Scalar literal (`0`, `1`, `true`, ...).
    Literal(Literal, Span),
    /// Wildcard / default arm (`_` or `default`).
    Wildcard(Span),
}

/// Static task instantiation statement (`spawn task_fn(args...) [@prio] [@core(N)];`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpawnStmt {
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
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    /// Infix binary operation (`a + b`, `HALF | FULL`, `crc != frame.crc`, ...).
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Explicit scalar cast (`x as u16` or `(u16)x`).
    Cast {
        expr: Box<Expr>,
        ty: TypeExpr,
    },
    /// Field or peripheral property access (`frame.crc`, `buf.len`, `hw.DMA1.CH5`).
    Field {
        target: Box<Expr>,
        field: Ident,
    },
    /// Direct function call (`parse_setpoint(half)`, `pi_step(target, actual, &integral)`).
    Call {
        callee: Box<Expr>,
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
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },
    /// Sub-slice range (`buf[0..5]`, `buf[..5]`, `buf[3..]`).
    Slice {
        target: Box<Expr>,
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
    /// Struct literal (`ByteReader { buf: half, pos: 0 }`).
    StructLit {
        name: Ident,
        fields: Vec<FieldInit>,
    },
    /// Fixed array literal (`[0x01, 0x00, 0x64]`).
    ArrayLit(Vec<Expr>),
    /// Fallback error/timeout handler (`sup_inbox.recv(timeout: 5s) or IdleCheck`).
    Or {
        expr: Box<Expr>,
        fallback: Box<Expr>,
    },
    /// Block error handler (`parse_setpoint(half) catch (ParseErr err) { ... }`).
    Catch {
        expr: Box<Expr>,
        clause: CatchClause,
    },
}

/// Literal values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Literal {
    Int(u64),
    Duration(DurationLit),
    Bool(bool),
    String(String),
}

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
            vec![Item {
                attrs: vec![],
                kind: ItemKind::Import(ImportDecl {
                    path: vec![Ident::dummy("chip"), Ident::dummy("stm32g4")],
                    alias: Some(Ident::dummy("hw")),
                    span: Span::new(0, 26, 1, 1),
                }),
                span: Span::new(0, 26, 1, 1),
            }],
            Span::new(0, 26, 1, 1),
        );
        let p2 = Program::new(vec![], Span::new(0, 10, 1, 1));
        let merged = Program::merge_all([p1, p2]);
        assert_eq!(merged.items.len(), 1);
    }
}
