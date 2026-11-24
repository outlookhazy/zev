//! Core MIR data structures for coroutine state-machine tasks, yielding functions,
//! OS primitives, hardware ISRs, and static frame layout.

use crate::ram::RamReport;

/// Cooperative scheduler priority levels (`0` = highest urgency).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MirPriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

impl MirPriority {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn c_const_name(self) -> &'static str {
        match self {
            MirPriority::Critical => "SYN_PRIO_CRITICAL",
            MirPriority::High => "SYN_PRIO_HIGH",
            MirPriority::Normal => "SYN_PRIO_NORMAL",
            MirPriority::Low => "SYN_PRIO_LOW",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MirPriority::Critical => "@critical",
            MirPriority::High => "@high",
            MirPriority::Normal => "@normal",
            MirPriority::Low => "@low",
        }
    }

    pub fn default_max_wait_ms(self) -> u16 {
        match self {
            MirPriority::Critical => 1,
            MirPriority::High => 5,
            MirPriority::Normal => 15,
            MirPriority::Low => 50,
        }
    }
}

/// Resolved MIR value or endpoint type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MirType {
    Void,
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Usize,
    Isize,
    F32,
    F64,
    Str,
    /// Wire-endian integer field/value (`u16be`, `u16le`, `u32be`, `u32le`, `u64be`, `u64le`).
    EndianInt {
        bits: u8,
        signed: bool,
        big_endian: bool,
    },
    /// User-defined enum type by name.
    Enum(String),
    /// User-defined struct or `packed struct` type by name.
    Struct(String),
    /// Fat slice `[]T` (`{ T *ptr; size_t len; }`).
    Slice(Box<MirType>),
    /// Fixed-size array `[N]T`.
    Array {
        elem: Box<MirType>,
        len: usize,
    },
    /// Pointer `*T`.
    Ptr(Box<MirType>),
    /// Move-only linear DMA/pool buffer handle `Lease<T>`.
    Lease(Box<MirType>),
    /// OS primitive endpoint type (usually accessed as global or via `*Primitive`).
    Primitive(MirPrimKind),
    /// Unsized fat-pointer view over any `byte_ring<N>` or `inbox<T, N>`.
    PrimView(MirPrimViewKind),
    /// Typed pointer to an `mmio` register layout (`*UartRegs`).
    MmioPtr(String),
    /// Function pointer type (with optional `error [E]` exit spec).
    FnPtr {
        params: Vec<MirType>,
        ret: Box<MirType>,
        error_spec: Box<MirErrorSpec>,
    },
    /// Internal dual-exit return envelope `{ bool is_err; [Err err;] [Val val;] }`.
    Result {
        ok: Box<MirType>,
        err: Option<Box<MirType>>,
    },
}

impl MirType {
    /// Returns the host/CPU decoded type for wire-endian fields (e.g. `u16be` -> `u16`).
    pub fn host_value_type(&self) -> MirType {
        match self {
            MirType::EndianInt { bits, signed, .. } => match (*bits, *signed) {
                (16, false) => MirType::U16,
                (16, true) => MirType::I16,
                (32, false) => MirType::U32,
                (32, true) => MirType::I32,
                (64, false) => MirType::U64,
                (64, true) => MirType::I64,
                _ => MirType::U32,
            },
            other => other.clone(),
        }
    }

    pub fn is_void(&self) -> bool {
        matches!(self, MirType::Void)
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(
            self.host_value_type(),
            MirType::Bool
                | MirType::U8
                | MirType::U16
                | MirType::U32
                | MirType::U64
                | MirType::Usize
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self.host_value_type(), MirType::F32 | MirType::F64)
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self.host_value_type(),
            MirType::U8
                | MirType::U16
                | MirType::U32
                | MirType::U64
                | MirType::Usize
                | MirType::I8
                | MirType::I16
                | MirType::I32
                | MirType::I64
                | MirType::Isize
                | MirType::F32
                | MirType::F64
        )
    }

    /// Short canonical identifier used for mangling helper types (`syn_slice_u8`, `syn_watch_u16`).
    pub fn mangle_name(&self) -> String {
        match self {
            MirType::Void => "void".to_string(),
            MirType::Bool => "bool".to_string(),
            MirType::U8 => "u8".to_string(),
            MirType::U16 => "u16".to_string(),
            MirType::U32 => "u32".to_string(),
            MirType::U64 => "u64".to_string(),
            MirType::I8 => "i8".to_string(),
            MirType::I16 => "i16".to_string(),
            MirType::I32 => "i32".to_string(),
            MirType::I64 => "i64".to_string(),
            MirType::Usize => "usize".to_string(),
            MirType::Isize => "isize".to_string(),
            MirType::F32 => "f32".to_string(),
            MirType::F64 => "f64".to_string(),
            MirType::Str => "str".to_string(),
            MirType::EndianInt {
                bits,
                signed,
                big_endian,
            } => {
                let p = if *signed { 'i' } else { 'u' };
                let e = if *big_endian { "be" } else { "le" };
                format!("{p}{bits}{e}")
            }
            MirType::Enum(n) | MirType::Struct(n) => n.clone(),
            MirType::Slice(inner) => format!("slice_{}", inner.mangle_name()),
            MirType::Array { elem, len } => format!("arr_{}_{}", len, elem.mangle_name()),
            MirType::Ptr(inner) => format!("ptr_{}", inner.mangle_name()),
            MirType::Lease(inner) => format!("lease_{}", inner.mangle_name()),
            MirType::Primitive(pk) => pk.mangle_name(),
            MirType::PrimView(vk) => vk.mangle_name(),
            MirType::MmioPtr(n) => format!("mmio_ptr_{n}"),
            MirType::FnPtr {
                params,
                ret,
                error_spec,
            } => {
                let err_sfx = match error_spec.as_ref() {
                    MirErrorSpec::None => String::new(),
                    MirErrorSpec::Untyped => "_err".to_string(),
                    MirErrorSpec::Typed(e) => format!("_err_{}", e.mangle_name()),
                };
                if params.is_empty() {
                    format!("fnptr_void_ret_{}{}", ret.mangle_name(), err_sfx)
                } else {
                    let ps: Vec<String> = params.iter().map(|p| p.mangle_name()).collect();
                    format!(
                        "fnptr_{}_ret_{}{}",
                        ps.join("_"),
                        ret.mangle_name(),
                        err_sfx
                    )
                }
            }
            MirType::Result { ok, err } => match err {
                Some(e) => format!("res_{}_err_{}", ok.mangle_name(), e.mangle_name()),
                None => format!("res_{}_err_bool", ok.mangle_name()),
            },
        }
    }

    /// Human-readable `.zev` type display string for RAM reports and diagnostics.
    pub fn display_name(&self) -> String {
        match self {
            MirType::Void => "void".to_string(),
            MirType::Bool => "bool".to_string(),
            MirType::U8 => "u8".to_string(),
            MirType::U16 => "u16".to_string(),
            MirType::U32 => "u32".to_string(),
            MirType::U64 => "u64".to_string(),
            MirType::I8 => "i8".to_string(),
            MirType::I16 => "i16".to_string(),
            MirType::I32 => "i32".to_string(),
            MirType::I64 => "i64".to_string(),
            MirType::Usize => "usize".to_string(),
            MirType::Isize => "isize".to_string(),
            MirType::F32 => "f32".to_string(),
            MirType::F64 => "f64".to_string(),
            MirType::Str => "str".to_string(),
            MirType::EndianInt {
                bits,
                signed,
                big_endian,
            } => {
                let p = if *signed { 'i' } else { 'u' };
                let e = if *big_endian { "be" } else { "le" };
                format!("{p}{bits}{e}")
            }
            MirType::Enum(n) | MirType::Struct(n) => n.clone(),
            MirType::Slice(inner) => format!("[]{}", inner.display_name()),
            MirType::Array { elem, len } => format!("[{len}]{}", elem.display_name()),
            MirType::Ptr(inner) => format!("*{}", inner.display_name()),
            MirType::Lease(inner) => format!("Lease<{}>", inner.display_name()),
            MirType::Primitive(pk) => pk.display_name(),
            MirType::PrimView(vk) => vk.display_name(),
            MirType::MmioPtr(n) => format!("*{n}"),
            MirType::FnPtr { .. } => "fn(...)".to_string(),
            MirType::Result { ok, err } => match err {
                Some(e) => format!("{} error {}", ok.display_name(), e.display_name()),
                None => format!("{} error", ok.display_name()),
            },
        }
    }
}

/// Static OS & Hardware endpoint primitive kinds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MirPrimKind {
    Signal,
    WaitQueue,
    Watch { elem: Box<MirType> },
    Guarded { elem: Box<MirType> },
    Inbox { elem: Box<MirType>, cap: usize },
    PingPong { elem: Box<MirType>, cap: usize },
    Pool { elem: Box<MirType>, cap: usize },
    DmaSlot { elem: Box<MirType> },
    ByteRing { cap: usize },
}

impl MirPrimKind {
    pub fn mangle_name(&self) -> String {
        match self {
            MirPrimKind::Signal => "signal".to_string(),
            MirPrimKind::WaitQueue => "wait_queue".to_string(),
            MirPrimKind::Watch { elem } => format!("watch_{}", elem.mangle_name()),
            MirPrimKind::Guarded { elem } => format!("guarded_{}", elem.mangle_name()),
            MirPrimKind::Inbox { elem, cap } => format!("inbox_{}_{}", elem.mangle_name(), cap),
            MirPrimKind::PingPong { elem, cap } => format!("pp_{}_{}", elem.mangle_name(), cap),
            MirPrimKind::Pool { elem, cap } => format!("pool_{}_{}", elem.mangle_name(), cap),
            MirPrimKind::DmaSlot { elem } => format!("dma_slot_{}", elem.mangle_name()),
            MirPrimKind::ByteRing { cap } => format!("byte_ring_{cap}"),
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            MirPrimKind::Signal => "signal".to_string(),
            MirPrimKind::WaitQueue => "wait_queue".to_string(),
            MirPrimKind::Watch { elem } => format!("watch<{}>", elem.display_name()),
            MirPrimKind::Guarded { elem } => format!("guarded<{}>", elem.display_name()),
            MirPrimKind::Inbox { elem, cap } => format!("inbox<{}, {cap}>", elem.display_name()),
            MirPrimKind::PingPong { elem, cap } => {
                format!("ping_pong<{}, {cap}>", elem.display_name())
            }
            MirPrimKind::Pool { elem, cap } => format!("pool<{}, {cap}>", elem.display_name()),
            MirPrimKind::DmaSlot { elem } => format!("dma_slot<{}>", elem.display_name()),
            MirPrimKind::ByteRing { cap } => format!("byte_ring<{cap}>"),
        }
    }

    pub fn holds_dma_sleep_lock(&self, dma_safe: bool) -> bool {
        matches!(self, MirPrimKind::PingPong { .. })
            || (dma_safe && matches!(self, MirPrimKind::Pool { .. }))
    }
}

/// Unsized view endpoint kind (`*byte_ring` or `*inbox<T>`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MirPrimViewKind {
    ByteRing,
    Inbox { elem: Box<MirType> },
}

impl MirPrimViewKind {
    pub fn mangle_name(&self) -> String {
        match self {
            MirPrimViewKind::ByteRing => "view_byte_ring".to_string(),
            MirPrimViewKind::Inbox { elem } => format!("view_inbox_{}", elem.mangle_name()),
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            MirPrimViewKind::ByteRing => "*byte_ring".to_string(),
            MirPrimViewKind::Inbox { elem } => format!("*inbox<{}>", elem.display_name()),
        }
    }
}

/// Imported chip/module descriptor.
#[derive(Debug, Clone)]
pub struct MirImport {
    pub path: Vec<String>,
    pub alias: Option<String>,
}

/// Top-level immutable `const` item.
#[derive(Debug, Clone)]
pub struct MirConst {
    pub name: String,
    pub ty: MirType,
    pub init: MirExpr,
}

/// Top-level `enum` item with resolved discriminants.
#[derive(Debug, Clone)]
pub struct MirEnum {
    pub name: String,
    pub variants: Vec<(String, i64)>,
}

/// Field inside a `struct` or `packed struct`.
#[derive(Debug, Clone)]
pub struct MirField {
    pub name: String,
    pub ty: MirType,
    pub dma_safe: bool,
}

/// Top-level `struct` or `packed struct` item.
#[derive(Debug, Clone)]
pub struct MirStruct {
    pub name: String,
    pub packed: bool,
    pub is_subsystem: bool,
    pub fields: Vec<MirField>,
}

/// Top-level static subsystem struct instance (`UartSubsys uart0;`).
#[derive(Debug, Clone)]
pub struct MirSubsystemGlobal {
    pub name: String,
    pub struct_name: String,
    pub dma_safe: bool,
}

/// Top-level OS primitive endpoint declaration.
#[derive(Debug, Clone)]
pub struct MirPrimitive {
    pub name: String,
    pub kind: MirPrimKind,
    pub dma_safe: bool,
    pub init: Option<MirExpr>,
    /// Assigned wake-lock bit (`0..31`) if this primitive clamps sleep depth (`ping_pong`, `lease`).
    pub sleep_lock_bit: Option<u8>,
    /// True if whole-program visibility proved this `watch<T>` unread and address-free (`.bss` stripped).
    pub dead_stripped: bool,
    /// True if whole-program visibility proved this primitive never contended by any hardware ISR.
    pub irq_mask_elided: bool,
}

/// Function error exit specification (`error` or `error ErrType`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MirErrorSpec {
    /// No `error` exit path.
    None,
    /// Boolean `error;` exit path with no error payload type.
    Untyped,
    /// Typed `error ErrType;` exit path.
    Typed(MirType),
}

impl MirErrorSpec {
    pub fn has_error(&self) -> bool {
        !matches!(self, MirErrorSpec::None)
    }

    pub fn err_type(&self) -> Option<MirType> {
        match self {
            MirErrorSpec::None | MirErrorSpec::Untyped => None,
            MirErrorSpec::Typed(ty) => Some(ty.clone()),
        }
    }
}

/// `extern "C"` function declaration.
#[derive(Debug, Clone)]
pub struct MirExternFn {
    pub name: String,
    pub params: Vec<(String, MirType)>,
    pub ret_ty: MirType,
    pub variadic: bool,
}

/// Variable identifier inside a function, task, or ISR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarId(pub usize);

/// Origin/kind of a variable inside a function or task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarRole {
    /// Parameter passed into the task or function.
    Param,
    /// User-declared local variable (`Type name = ...;`).
    UserLocal,
    /// Compiler-synthesized state or temporary (e.g., `every` deadline, `wait` timeout, call temp).
    CompilerTemp,
    /// ISR-private persistent static variable (`state Type name = init;`), passed as a pointer
    /// to each per-attachment static storage slot.
    IsrState,
}

/// Metadata and computed placement for a variable in a function or task.
#[derive(Debug, Clone)]
pub struct MirVar {
    pub id: VarId,
    pub name: String,
    pub ty: MirType,
    pub role: VarRole,
    /// True if placed inside a `noyield` or `critical` block.
    pub in_noyield_scope: bool,
    /// True if `&var` is ever taken.
    pub addr_taken: bool,
    /// Computed by `liveness`: `true` if hoisted into the static coroutine frame struct,
    /// `false` if kept as a fast transient C stack variable.
    pub hoisted: bool,
}

/// A yielding child call site inside a parent coroutine's `union _child`.
#[derive(Debug, Clone)]
pub struct MirChildCallSlot {
    /// Unique slot name inside `union _child` (e.g. `c0_save_fault_record`).
    pub slot_name: String,
    /// Name of the yielding callee function.
    pub callee: String,
    /// Conflict color group (`0` when all child calls share a single disjoint union).
    pub union_group: usize,
}

/// Basic block identifier inside a state-machine task or function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u16);

/// Basic block in a MIR coroutine CFG.
#[derive(Debug, Clone)]
pub struct MirBlock {
    pub id: BlockId,
    /// True when `_pc` can hold `id.0` and dispatch via the top-level `switch (_f->_pc)`.
    pub is_resume_entry: bool,
    pub stmts: Vec<MirStmt>,
    pub terminator: MirTerminator,
}

/// Non-yielding sequential MIR statements inside a basic block or non-yielding fn/ISR.
#[derive(Debug, Clone)]
pub enum MirStmt {
    /// Assign an expression to an L-value (`place = expr` or compound `+=`, etc.).
    Assign {
        target: MirPlace,
        op: MirAssignOp,
        value: MirExpr,
    },
    /// Evaluate a non-yielding expression for side effects.
    Expr(MirExpr),
    /// Non-blocking primitive operation: `watch.publish(v)`, `inbox.post(v, boost)`,
    /// `ping_pong.flip()`, `ping_pong.start()`, `ping_pong.stop()`, `signal.notify(boost)`.
    PrimOp(MirPrimNonBlockingOp),
    /// Runtime intrinsic operation (`sys.use_systick(...)`, `set_priority(...)`, etc.).
    Intrinsic(MirIntrinsicOp),
    /// Enter hardware critical section (`PRIMASK` / IRQ mask).
    EnterCritical,
    /// Exit hardware critical section.
    ExitCritical,
    /// Restore any temporary inbox priority boost back to `base_priority` at end of event turn.
    EndInboxTurn,
    /// Structured `if` (used only in non-yielding functions/ISRs/boot; yielding bodies use CFG blocks).
    If {
        cond: MirExpr,
        then_body: Vec<MirStmt>,
        else_body: Vec<MirStmt>,
    },
    /// Structured bounded `while`/`loop` (used in non-yielding functions; yielding loops use CFG blocks).
    While {
        cond: MirExpr,
        body: Vec<MirStmt>,
    },
    /// Structured `switch` (used in non-yielding functions).
    Switch {
        discr: MirExpr,
        arms: Vec<(MirSwitchPattern, Vec<MirStmt>)>,
        default: Vec<MirStmt>,
    },
    /// Direct return from a non-yielding function/ISR.
    Return(Option<MirExpr>),
    /// Direct error exit from a non-yielding function.
    Error {
        reason: Option<MirExpr>,
        partial: Option<MirExpr>,
    },
    /// Break/continue inside structured non-yielding loop.
    Break,
    Continue,
    /// Activate static task slot `slot_id` with its spawn arguments (supports conditional boot `if`/`switch`).
    SpawnActivate {
        slot_id: usize,
        args: Vec<MirExpr>,
    },
    /// Zero-heap compile-time unpacked `print(f"...")` / `println(f"...")` stream output.
    PrintFmt {
        chunks: Vec<MirFmtChunk>,
        newline: bool,
    },
    /// Zero-heap bounded buffer formatting `fmt_buf(dst, f"...") -> []u8 error FmtErr`.
    ///
    /// Writes unpacked `chunks` into `buf` (`MirType::Slice(U8)`) and stores dual-exit
    /// `Result { ok: Slice(U8), err: Some(Enum("FmtErr")) }` into `dst_res` (`is_err = false, val = written_slice`
    /// on fit; `is_err = true, err = FmtErr::Truncated (0), val = clamped_slice` on overflow).
    FmtBuf {
        dst_res: VarId,
        buf: MirExpr,
        chunks: Vec<MirFmtChunk>,
    },
    /// Zero-heap bounded ring formatting `fmt_ring(tx, f"..." [, boost: @prio]) -> usize error FmtErr`.
    FmtRing {
        dst_res: VarId,
        ring: MirExpr,
        chunks: Vec<MirFmtChunk>,
        boost: Option<MirPriority>,
    },
    /// Volatile MMIO register write (`regs.cr1 = val;` or `regs.arr[i] = val;`).
    MmioWrite {
        base: Box<MirExpr>,
        offset: u32,
        elem_bits: u8,
        index: Option<Box<MirExpr>>,
        array_len: Option<usize>,
        value: MirExpr,
    },
    /// Record runtime branch selection for conditional boot-branch `irq.attach(...)`.
    SetIrqSelector {
        symbol: String,
        value: u8,
    },
    /// Semantic marker emitted on bounded `for` loop latches for static run-bound classification.
    BoundedForLatch,
    /// Zero non-parameter task locals and child call frames on `restart;` before jumping to `bb_0`.
    ZeroTaskLocals,
}

/// Integer radix and zero-padding format mode for [`MirFmtChunk::Int`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirIntFmt {
    /// Base-10 decimal (`{x}`).
    Dec,
    /// Base-16 hexadecimal (`{x:x}`, `{x:X}`, `{x:04x}`, `{x:08X}`).
    Hex { upper: bool, zero_pad_width: u8 },
    /// Base-2 binary (`{x:b}`, `{x:08b}`).
    Bin { zero_pad_width: u8 },
}

/// Floating-point format mode for [`MirFmtChunk::Float`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirFloatFmt {
    /// Compact Arduino-style fixed-decimal formatter (`{x}` => `precision: 2`, `{x:.3}` => `precision: 3`).
    Fixed { precision: u8 },
    /// On-demand scientific notation formatter (`{x:e}` => `precision: 2`, `{x:.3e}` => `precision: 3`).
    /// Emitted in a separate function so `ld.lld --gc-sections` strips it to `0 B` when unused.
    Sci { precision: u8 },
}

/// Compile-time unpacked zero-heap formatting chunk (`f"..."`).
///
/// Struct `{pkt:?}` holes are flattened into `Literal` + field chunks during MIR lowering
/// so codegen backends require zero runtime struct reflection.
#[derive(Debug, Clone)]
pub enum MirFmtChunk {
    /// Static literal string segment (`"temp="`, `"Packet { op: "`, `" }"`).
    Literal(String),
    /// Integer value (`bits`: `8`, `16`, `32`, `64`, or `0` for target-sized `usize`/`isize`).
    Int {
        value: MirExpr,
        signed: bool,
        bits: u8,
        fmt: MirIntFmt,
    },
    /// Floating-point value (`MirType::F32` when `!is_f64`, `MirType::F64` when `is_f64`).
    Float {
        value: MirExpr,
        is_f64: bool,
        fmt: MirFloatFmt,
    },
    /// Boolean value (`"true"` or `"false"`).
    Bool(MirExpr),
    /// Null-terminated string literal or `*u8` pointer.
    Str(MirExpr),
    /// Raw `[]u8` byte slice written verbatim (`{slice}`).
    Bytes(MirExpr),
    /// Enum variant name (`f"{err}"` -> `"BadCrc"`). `enum_name` is also recorded in `MirModule.formatted_enums`.
    EnumName { enum_name: String, value: MirExpr },
    /// Debug-formatted primitive slice `[e0, e1, ...]` (`{slice:?}` or `{arr:?}`).
    DebugSlice { elem_ty: MirType, slice: MirExpr },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirAssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    BitOrAssign,
    BitAndAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

/// Target of an assignment (`x`, `*p`, `s.field`, `a[i]`).
#[derive(Debug, Clone)]
pub enum MirPlace {
    Var(VarId),
    Deref(Box<MirExpr>, MirType),
    Field {
        base: Box<MirPlace>,
        struct_name: String,
        packed: bool,
        field: String,
        field_ty: MirType,
    },
    Index {
        base: Box<MirPlace>,
        index: Box<MirExpr>,
        elem_ty: MirType,
        is_slice: bool,
    },
}

/// Reference to an OS primitive instance (either a static global by name, or via a pointer expression like `rx_dma`).
#[derive(Debug, Clone)]
pub enum MirPrimTarget {
    Global(String),
    PtrExpr(MirExpr),
}

/// Non-blocking (ISR- and `noyield`-safe) primitive operations.
#[derive(Debug, Clone)]
pub enum MirPrimNonBlockingOp {
    WatchPublish {
        target: MirPrimTarget,
        elem_ty: MirType,
        value: MirExpr,
    },
    GuardedSet {
        target: MirPrimTarget,
        elem_ty: MirType,
        value: MirExpr,
    },
    InboxPost {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
        value: MirExpr,
        boost: Option<MirPriority>,
    },
    SignalNotify {
        target: MirPrimTarget,
        boost: Option<MirPriority>,
    },
    SignalClear {
        target: MirPrimTarget,
    },
    WaitQueuePrepare {
        target: MirPrimTarget,
    },
    WaitQueueCancel {
        target: MirPrimTarget,
    },
    WaitQueueWakeAll {
        target: MirPrimTarget,
        boost: Option<MirPriority>,
    },
    PingPongFlip {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    PingPongCompleteHalf {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
        half: MirExpr,
    },
    PingPongSetActive {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
        active: bool,
    },
    /// Move-consume `Lease<T>` back to its owning pool (`lease.release([boost: @prio])`).
    LeaseRelease {
        value: MirExpr,
        elem_ty: MirType,
        boost: Option<MirPriority>,
    },
    /// Nullify `pool_hdr` on a moved `Lease<T>` place (defense-in-depth).
    LeaseNullify {
        target: MirPlace,
        elem_ty: MirType,
    },
    ByteRingConsume {
        target: MirPrimTarget,
        cap: usize,
        len: MirExpr,
    },
    ByteRingCommitWrite {
        target: MirPrimTarget,
        cap: usize,
        len: MirExpr,
        boost: Option<MirPriority>,
    },
    ByteRingClear {
        target: MirPrimTarget,
        cap: usize,
    },
    InboxViewPost {
        view: MirExpr,
        elem_ty: MirType,
        value: MirExpr,
        boost: Option<MirPriority>,
    },
    ByteRingViewConsume {
        view: MirExpr,
        len: MirExpr,
    },
    ByteRingViewCommitWrite {
        view: MirExpr,
        len: MirExpr,
        boost: Option<MirPriority>,
    },
    ByteRingViewClear {
        view: MirExpr,
    },
}

/// Built-in system, hardware SVD, and simulation intrinsics.
#[derive(Debug, Clone)]
pub enum MirIntrinsicOp {
    UseSystick { hz: MirExpr },
    SetPriority { prio: MirPriority },
    SysOp { method: String, args: Vec<MirExpr> },
}

/// Switch arm pattern (`Variant` or integer literal or wildcard `_`).
#[derive(Debug, Clone)]
pub enum MirSwitchPattern {
    EnumVariant { enum_name: String, variant: String },
    Int(i128),
    Bool(bool),
}

/// Destination of a yielding function call.
#[derive(Debug, Clone)]
pub enum MirCallDst {
    /// Callee returns plain `T` (no `error` exit); store into `VarId`.
    Value(VarId),
    /// Callee has an `error` exit; store `{ is_err, [err,] [val] }` envelope into `VarId`.
    Result(VarId),
}

/// Yielding operations that suspend a coroutine and return control to the scheduler.
#[derive(Debug, Clone)]
pub enum MirYieldKind {
    /// Drift-free periodic timer latch at the end of `loop every <period> { ... }` (with overrun skip/resync).
    EveryLatch {
        next_deadline_var: VarId,
        period_var: VarId,
        resume_bb: BlockId,
    },
    /// Relative delay (`delay(dur)` / `sys.delay(dur)`): sets `deadline_var` before polling in `poll_bb`.
    DelayPoll {
        deadline_var: VarId,
        poll_bb: BlockId,
        resume_bb: BlockId,
    },
    /// Demand-driven cooperative CPU share (`should_share()`) at unbounded loop back-edges, `checkpoint()`, or `restart;`.
    ShareCheckpoint { resume_bb: BlockId },
    /// Blocking receive on unified `inbox<T, N>`, optionally bounded by `timeout_deadline_var`.
    /// Single primitive arm inside a unified atomic `MirYieldKind::Wait`.
    Wait {
        arms: Vec<MirWaitArm>,
        timer: Option<MirWaitTimer>,
        poll_bb: BlockId,
    },
    /// Initialize and invoke a yielding child function via `union _child.<slot_name>`.
    CallYielding {
        callee: String,
        slot_name: String,
        args: Vec<MirExpr>,
        dst: Option<MirCallDst>,
        poll_bb: BlockId,
        resume_bb: BlockId,
    },
    /// Poll an already-initialized yielding child function in `union _child.<slot_name>`.
    PollChild {
        callee: String,
        slot_name: String,
        dst: Option<MirCallDst>,
        poll_bb: BlockId,
        resume_bb: BlockId,
    },
}

/// Single primitive wait operation inside [`MirYieldKind::Wait`].
#[derive(Debug, Clone)]
pub enum MirWaitArmOp {
    InboxRecv {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
        dst_val: Option<VarId>,
        dst_res: Option<VarId>,
    },
    InboxViewRecv {
        view: MirExpr,
        elem_ty: MirType,
        dst_val: Option<VarId>,
        dst_res: Option<VarId>,
    },
    InboxSend {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
        val_var: VarId,
        dst_res: Option<VarId>,
    },
    InboxViewSend {
        view: MirExpr,
        elem_ty: MirType,
        val_var: VarId,
        dst_res: Option<VarId>,
    },
    SignalWait {
        target: MirPrimTarget,
        dst_res: Option<VarId>,
    },
    WaitQueueCommit {
        target: MirPrimTarget,
        dst_res: Option<VarId>,
    },
    WatchWaitChange {
        target: MirPrimTarget,
        elem_ty: MirType,
        last_gen_var: VarId,
        seeded_var: Option<VarId>,
        dst_val: Option<VarId>,
        dst_res: Option<VarId>,
    },
    PingPongWaitChunk {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
        dst_slice: VarId,
    },
    PoolAlloc {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
        dst_val: Option<VarId>,
        dst_res: Option<VarId>,
    },
    ByteRingWaitRead {
        target: MirPrimTarget,
        cap: usize,
        dst_val: Option<VarId>,
        dst_res: Option<VarId>,
    },
    ByteRingViewWaitRead {
        view: MirExpr,
        dst_val: Option<VarId>,
        dst_res: Option<VarId>,
    },
    ByteRingWaitWrite {
        target: MirPrimTarget,
        cap: usize,
        dst_val: Option<VarId>,
        dst_res: Option<VarId>,
    },
    ByteRingViewWaitWrite {
        view: MirExpr,
        dst_val: Option<VarId>,
        dst_res: Option<VarId>,
    },
}

/// One branch of a [`MirYieldKind::Wait`], polled top-to-bottom under one critical section.
#[derive(Debug, Clone)]
pub struct MirWaitArm {
    pub op: MirWaitArmOp,
    pub target_bb: BlockId,
}

/// Optional deadline timer arm of a [`MirYieldKind::Wait`].
#[derive(Debug, Clone)]
pub struct MirWaitTimer {
    pub deadline_var: VarId,
    pub timeout_bb: BlockId,
    pub timeout_err_res: Option<VarId>,
    pub release_lease_on_timeout: Option<(VarId, MirType)>,
}

/// Basic block terminator in a state-machine task or yielding function.
#[derive(Debug, Clone)]
pub enum MirTerminator {
    Goto(BlockId),
    Branch {
        cond: MirExpr,
        then_bb: BlockId,
        else_bb: BlockId,
    },
    Switch {
        discr: MirExpr,
        arms: Vec<(MirSwitchPattern, BlockId)>,
        default_bb: BlockId,
    },
    Return(Option<MirExpr>),
    Error {
        reason: Option<MirExpr>,
        partial: Option<MirExpr>,
    },
    Yield(MirYieldKind),
}

impl MirTerminator {
    /// Control-flow successor blocks within the same coroutine (used by backward liveness analysis).
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            MirTerminator::Goto(b) => vec![*b],
            MirTerminator::Branch {
                then_bb, else_bb, ..
            } => vec![*then_bb, *else_bb],
            MirTerminator::Switch {
                arms, default_bb, ..
            } => {
                let mut out: Vec<BlockId> = arms.iter().map(|(_, b)| *b).collect();
                out.push(*default_bb);
                out
            }
            MirTerminator::Return(_) | MirTerminator::Error { .. } => vec![],
            MirTerminator::Yield(yk) => match yk {
                MirYieldKind::EveryLatch { resume_bb, .. }
                | MirYieldKind::ShareCheckpoint { resume_bb } => vec![*resume_bb],
                MirYieldKind::DelayPoll {
                    poll_bb, resume_bb, ..
                }
                | MirYieldKind::PollChild {
                    poll_bb, resume_bb, ..
                } => vec![*poll_bb, *resume_bb],
                MirYieldKind::Wait {
                    arms,
                    timer,
                    poll_bb,
                } => {
                    let mut out = vec![*poll_bb];
                    for arm in arms {
                        if !out.contains(&arm.target_bb) {
                            out.push(arm.target_bb);
                        }
                    }
                    if let Some(t) = timer {
                        if !out.contains(&t.timeout_bb) {
                            out.push(t.timeout_bb);
                        }
                    }
                    out
                }
                MirYieldKind::CallYielding { poll_bb, .. } => vec![*poll_bb],
            },
        }
    }

    /// Returns the block(s) where execution resumes after returning to the scheduler.
    pub fn yield_resume_blocks(&self) -> Vec<BlockId> {
        match self {
            MirTerminator::Yield(yk) => match yk {
                MirYieldKind::EveryLatch { resume_bb, .. }
                | MirYieldKind::ShareCheckpoint { resume_bb } => vec![*resume_bb],
                MirYieldKind::DelayPoll { poll_bb, .. }
                | MirYieldKind::Wait { poll_bb, .. }
                | MirYieldKind::PollChild { poll_bb, .. } => vec![*poll_bb],
                MirYieldKind::CallYielding { .. } => vec![],
            },
            _ => vec![],
        }
    }
}

/// Typed, pure/non-yielding MIR expression.
#[derive(Debug, Clone)]
pub struct MirExpr {
    pub kind: MirExprKind,
    pub ty: MirType,
}

impl MirExpr {
    pub fn new(kind: MirExprKind, ty: MirType) -> Self {
        Self { kind, ty }
    }
}

#[derive(Debug, Clone)]
pub enum MirExprKind {
    IntLit(i128),
    /// IEEE-754 `f64::to_bits()` representation of a float literal (`MirExpr.ty` is `MirType::F32` or `MirType::F64`).
    FloatLit(u64),
    BoolLit(bool),
    StringLit(String),
    /// Duration already lowered to ticks (`1ms` -> `1`, `5s` -> `5000` on `1ms` tick).
    TicksLit(u64),
    EnumVariant {
        enum_name: String,
        variant: String,
    },
    ConstRef(String),
    GlobalPrimAddr(String),
    /// Base pointer to a static subsystem struct instance (`*Subsys`).
    SubsystemAddr(String),
    /// Construct an unsized primitive view `{ hdr/buf, cap }` from a sized primitive pointer.
    MakePrimView {
        ptr: Box<MirExpr>,
        cap: usize,
        kind: MirPrimViewKind,
    },
    /// Base address of a typed `mmio` peripheral instance (`MirType::MmioPtr(layout)`).
    MmioAddr(String),
    /// Volatile MMIO register read (`regs.isr` or `regs.arr[i]`).
    MmioRead {
        base: Box<MirExpr>,
        offset: u32,
        elem_bits: u8,
        index: Option<Box<MirExpr>>,
        array_len: Option<usize>,
    },
    FnAddr(String),
    Var(VarId),
    Unary {
        op: MirUnaryOp,
        expr: Box<MirExpr>,
    },
    AddrOfPlace(Box<MirPlace>),
    Binary {
        op: MirBinOp,
        lhs: Box<MirExpr>,
        rhs: Box<MirExpr>,
    },
    Cast {
        expr: Box<MirExpr>,
        target_ty: MirType,
    },
    StructLit {
        struct_name: String,
        packed: bool,
        fields: Vec<(String, MirExpr, MirType)>,
    },
    ArrayLit {
        elems: Vec<MirExpr>,
        elem_ty: MirType,
    },
    FieldAccess {
        base: Box<MirExpr>,
        struct_name: String,
        packed: bool,
        is_ptr: bool,
        field: String,
        raw_field_ty: MirType,
    },
    SliceLen(Box<MirExpr>),
    ArrayLen(usize),
    Index {
        base: Box<MirExpr>,
        index: Box<MirExpr>,
        is_slice: bool,
    },
    SubSlice {
        base: Box<MirExpr>,
        start: Box<MirExpr>,
        end: Box<MirExpr>,
        elem_ty: MirType,
        base_is_array: bool,
    },
    ArrayToSlice {
        array_place: Box<MirPlace>,
        len: usize,
        elem_ty: MirType,
    },
    /// Decode wire bytes into a `packed struct`: `Type.from_bytes(slice)` -> `Result { ok: Type, err: None }`.
    FromBytes {
        struct_name: String,
        slice: Box<MirExpr>,
    },
    /// Non-blocking primitive read/query (`watch.read()`, `inbox.try_recv()`, `signal.is_set()`).
    PrimRead(Box<MirPrimReadOp>),
    /// Borrow payload pointer `*T` from a `Lease<T>` (`lease.get()`).
    LeaseGet {
        lease: Box<MirExpr>,
        elem_ty: MirType,
    },
    /// Borrow payload address `usize` from a `Lease<T>` (`lease.addr()`).
    LeaseAddr {
        lease: Box<MirExpr>,
        elem_ty: MirType,
    },
    /// Direct call to a non-yielding function or `extern "C"` function.
    CallDirect {
        callee: String,
        is_extern_c: bool,
        args: Vec<MirExpr>,
    },
    /// Read fields of an internal dual-exit `{ is_err, err, val }` result variable.
    ResultIsErr(VarId),
    ResultErr(VarId),
    ResultVal(VarId),
    /// Current scheduler tick (`syn_now_ms()`).
    NowTicks,
    /// Portable kernel state query (`sys.current_prio()`, `sys.sleep_depth()`, `sys.wfi_count()`).
    SysQuery(MirSysQuery),
    /// Fixed-address / BootROM call returning `u32` with compile-time static stack bound.
    RomCallU32 {
        addr: Box<MirExpr>,
        stack_bound: u32,
        args: Vec<MirExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirSysQuery {
    CurrentPrio,
    SleepDepth,
    WfiCount,
}

#[derive(Debug, Clone)]
pub enum MirPrimReadOp {
    WatchRead {
        target: MirPrimTarget,
        elem_ty: MirType,
    },
    GuardedGet {
        target: MirPrimTarget,
        elem_ty: MirType,
    },
    WatchGen {
        target: MirPrimTarget,
        elem_ty: MirType,
    },
    InboxTryRecv {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    InboxTrySend {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
        value: Box<MirExpr>,
        boost: Option<MirPriority>,
    },
    InboxLen {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    InboxIsEmpty {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    InboxIsFull {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    SignalIsSet {
        target: MirPrimTarget,
    },
    SignalTryWait {
        target: MirPrimTarget,
    },
    WaitQueueWakeOne {
        target: MirPrimTarget,
        boost: Option<MirPriority>,
    },
    PingPongDmaBuf {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    PingPongBaseAddr {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    PoolTryAlloc {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    PoolAvail {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    PoolInUse {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    PoolIsExhausted {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    PoolCap {
        target: MirPrimTarget,
        elem_ty: MirType,
        cap: usize,
    },
    DmaSlotPark {
        target: MirPrimTarget,
        elem_ty: MirType,
        lease: Box<MirExpr>,
    },
    DmaSlotTake {
        target: MirPrimTarget,
        elem_ty: MirType,
    },
    DmaSlotRelease {
        target: MirPrimTarget,
        elem_ty: MirType,
        boost: Option<MirPriority>,
    },
    DmaSlotIsBusy {
        target: MirPrimTarget,
        elem_ty: MirType,
    },
    DmaSlotIsEmpty {
        target: MirPrimTarget,
        elem_ty: MirType,
    },
    DmaSlotAddr {
        target: MirPrimTarget,
        elem_ty: MirType,
    },
    ByteRingPush {
        target: MirPrimTarget,
        cap: usize,
        src: Box<MirExpr>,
        boost: Option<MirPriority>,
    },
    ByteRingPushByte {
        target: MirPrimTarget,
        cap: usize,
        byte: Box<MirExpr>,
        boost: Option<MirPriority>,
    },
    ByteRingGrantWrite {
        target: MirPrimTarget,
        cap: usize,
        len: Box<MirExpr>,
    },
    ByteRingReadSlice {
        target: MirPrimTarget,
        cap: usize,
    },
    ByteRingPop {
        target: MirPrimTarget,
        cap: usize,
        dst: Box<MirExpr>,
    },
    ByteRingTryPopByte {
        target: MirPrimTarget,
        cap: usize,
    },
    ByteRingLen {
        target: MirPrimTarget,
        cap: usize,
    },
    ByteRingFreeSpace {
        target: MirPrimTarget,
        cap: usize,
    },
    ByteRingIsEmpty {
        target: MirPrimTarget,
        cap: usize,
    },
    ByteRingIsFull {
        target: MirPrimTarget,
        cap: usize,
    },
    InboxViewTryRecv {
        view: MirExpr,
        elem_ty: MirType,
    },
    InboxViewTrySend {
        view: MirExpr,
        elem_ty: MirType,
        value: Box<MirExpr>,
        boost: Option<MirPriority>,
    },
    InboxViewLen {
        view: MirExpr,
        elem_ty: MirType,
    },
    InboxViewCap {
        view: MirExpr,
        elem_ty: MirType,
    },
    InboxViewIsEmpty {
        view: MirExpr,
        elem_ty: MirType,
    },
    InboxViewIsFull {
        view: MirExpr,
        elem_ty: MirType,
    },
    ByteRingViewPush {
        view: MirExpr,
        src: Box<MirExpr>,
        boost: Option<MirPriority>,
    },
    ByteRingViewPushByte {
        view: MirExpr,
        byte: Box<MirExpr>,
        boost: Option<MirPriority>,
    },
    ByteRingViewGrantWrite {
        view: MirExpr,
        len: Box<MirExpr>,
    },
    ByteRingViewReadSlice {
        view: MirExpr,
    },
    ByteRingViewPop {
        view: MirExpr,
        dst: Box<MirExpr>,
    },
    ByteRingViewTryPopByte {
        view: MirExpr,
    },
    ByteRingViewLen {
        view: MirExpr,
    },
    ByteRingViewCap {
        view: MirExpr,
    },
    ByteRingViewFreeSpace {
        view: MirExpr,
    },
    ByteRingViewIsEmpty {
        view: MirExpr,
    },
    ByteRingViewIsFull {
        view: MirExpr,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirUnaryOp {
    Neg,
    Not,
    BitNot,
    Deref,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitOr,
    BitAnd,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LogicalAnd,
    LogicalOr,
}

/// ISR-private `state` variable declared inside a [`MirIsr`] body.
#[derive(Debug, Clone)]
pub struct MirIsrStateDecl {
    pub var_id: VarId,
    pub name: String,
    pub ty: MirType,
    pub init: MirExpr,
}

/// Hardware Interrupt Service Routine (`isr NAME(...) { ... }`). Always `noyield`.
#[derive(Debug, Clone)]
pub struct MirIsr {
    pub name: String,
    pub params: Vec<VarId>,
    pub state_vars: Vec<MirIsrStateDecl>,
    pub vars: Vec<MirVar>,
    pub body: Vec<MirStmt>,
}

/// Access qualifier for a typed `mmio` register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirMmioAccess {
    Rw,
    Ro,
    Wo,
    W1c,
}

/// A single register inside a [`MirMmioLayout`].
#[derive(Debug, Clone)]
pub struct MirMmioReg {
    pub name: String,
    pub access: MirMmioAccess,
    pub elem_bits: u8,
    pub array_len: Option<usize>,
    pub offset: u32,
}

/// Reusable typed `mmio` register block layout.
#[derive(Debug, Clone)]
pub struct MirMmioLayout {
    pub name: String,
    pub byte_size: usize,
    pub regs: Vec<MirMmioReg>,
}

/// Physical peripheral instance bound to a [`MirMmioLayout`] at `base_addr`.
#[derive(Debug, Clone)]
pub struct MirMmioInstance {
    pub name: String,
    pub layout: String,
    pub base_addr: u64,
    pub byte_size: usize,
}

/// Per-attachment static `.bss`/`.data` storage backing an ISR's `state` variable.
#[derive(Debug, Clone)]
pub struct MirIsrStateVar {
    pub storage_symbol: String,
    pub var_name: String,
    pub ty: MirType,
    pub init: MirExpr,
}

/// Compile-time baked argument passed to an attached ISR handler.
#[derive(Debug, Clone)]
pub enum MirBakedArg {
    /// Address of a static OS/DMA primitive (`&sup_inbox`).
    PrimAddr(String),
    /// Address of a static subsystem struct instance or nested field (`&uart0`, `&uart0.rx`).
    SubsystemPath {
        root: String,
        root_struct: String,
        field_chain: Vec<(String, String)>,
    },
    /// Base pointer of an `mmio` peripheral instance (`hw.USART1` / `&hw.USART1`).
    MmioAddr(String),
    /// Compile-time constant scalar (`u32`, `bool`, enum variant, etc.).
    Const(MirExpr),
}

/// A single `.attach(...)` or `.attach_shared(...)` binding on a hardware IRQ vector.
#[derive(Debug, Clone)]
pub struct MirIrqBinding {
    pub binding_id: usize,
    pub isr_name: String,
    pub baked_args: Vec<MirBakedArg>,
    pub state_vars: Vec<MirIsrStateVar>,
    /// Present only when mutually exclusive conditional boot branches bind the same IRQ.
    pub cond_selector: Option<(String, u8)>,
}

/// Synthesized hardware IRQ vector entry for Flash `.isr_vector` / `INTMTX` dispatch.
#[derive(Debug, Clone)]
pub struct MirIrqVector {
    pub irq_name: String,
    pub irq_num: u32,
    pub is_shared: bool,
    pub selector_symbol: Option<String>,
    pub bindings: Vec<MirIrqBinding>,
}

/// Body representation of a `MirFunction`.
#[derive(Debug, Clone)]
pub enum MirFnBody {
    /// Non-yielding function: ordinary C stack execution, zero static frame bytes.
    Direct(Vec<MirStmt>),
    /// Yielding function: coroutine state machine with packed static frame & child union.
    Coroutine {
        blocks: Vec<MirBlock>,
        child_slots: Vec<MirChildCallSlot>,
    },
}

/// Lowered function (`fn`, either pure/noyield or yielding state machine).
#[derive(Debug, Clone)]
pub struct MirFunction {
    pub name: String,
    pub params: Vec<VarId>,
    pub ret_ty: MirType,
    pub error_spec: MirErrorSpec,
    pub manual_yield: bool,
    pub is_yielding: bool,
    pub vars: Vec<MirVar>,
    pub body: MirFnBody,
}

impl MirFunction {
    /// Envelope type stored in `_ret` (for coroutine) or returned by value (for direct fn).
    pub fn envelope_return_type(&self) -> MirType {
        if self.error_spec.has_error() {
            MirType::Result {
                ok: Box::new(self.ret_ty.clone()),
                err: self.error_spec.err_type().map(Box::new),
            }
        } else {
            self.ret_ty.clone()
        }
    }
}

/// Reusable `task` blueprint (or synthesized `__main_supervisor` task).
#[derive(Debug, Clone)]
pub struct MirTaskBlueprint {
    pub name: String,
    pub is_main_supervisor: bool,
    pub manual_yield: bool,
    pub params: Vec<VarId>,
    pub vars: Vec<MirVar>,
    pub blocks: Vec<MirBlock>,
    pub child_slots: Vec<MirChildCallSlot>,
}

/// Static task instance created by a `spawn` site (or `main`'s supervisor tail).
#[derive(Debug, Clone)]
pub struct MirSpawnInstance {
    pub slot_id: usize,
    pub instance_name: String,
    pub blueprint_name: String,
    pub priority: MirPriority,
    pub core_pin: Option<u8>,
    pub init_args: Vec<MirExpr>,
    pub is_main_supervisor: bool,
    /// Full static boot-tree call path ending in `blueprint_name` (e.g. `["main", "setup_bus", "worker"]`).
    pub provenance_path: Vec<String>,
    /// True when spawned inside a conditional `if`/`else`/`switch` boot branch.
    pub conditional: bool,
}

/// Sequential boot step in `fn main()` before the cooperative scheduler starts.
#[derive(Debug, Clone)]
pub enum MirBootStep {
    /// Ordinary non-yielding statement executed during sequential boot.
    Stmt(Box<MirStmt>),
    /// Sequential boot call to a yielding function (e.g. `run_power_on_self_test()`),
    /// driven to completion before spawned tasks begin!
    RunYieldingFn {
        callee: String,
        args: Vec<MirExpr>,
        dst: Option<MirCallDst>,
    },
    /// Activate static task slot `slot_id` with its spawn arguments.
    SpawnActivate { slot_id: usize, args: Vec<MirExpr> },
}

/// Sequential entry point (`fn main()`) split into boot initialization + optional supervisor task.
#[derive(Debug, Clone)]
pub struct MirBootSequence {
    pub present: bool,
    pub from_main_entry: bool,
    pub vars: Vec<MirVar>,
    pub steps: Vec<MirBootStep>,
    /// Static frames required only during `main()` sequential boot for `RunYieldingFn` calls.
    pub boot_yielding_callees: Vec<String>,
}

/// Kind of a monomorphized generic declaration (`struct` or `fn`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MonomorphizedItemKind {
    Struct,
    Fn,
}

impl MonomorphizedItemKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            MonomorphizedItemKind::Struct => "struct",
            MonomorphizedItemKind::Fn => "fn",
        }
    }
}

/// Concrete monomorphized generic instantiation recorded in `MirModule` and `RamReport`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonomorphizedMirItem {
    pub kind: MonomorphizedItemKind,
    /// Human-readable instantiation signature (e.g. `"FixedVec<u32, 16>"` or `"clamp<f32>"`).
    pub signature: String,
    /// Concrete mangled symbol name (e.g. `"FixedVec__u32_16"` or `"clamp__f32"`).
    pub mangled_name: String,
    /// Static struct byte size (for `Struct`) or packed coroutine frame byte size (for yielding `Fn`; `0` for direct `Fn`) on 32-bit MCU ABI.
    pub size_32: usize,
    /// Static struct byte size (for `Struct`) or packed coroutine frame byte size (for yielding `Fn`; `0` for direct `Fn`) on 64-bit host ABI.
    pub size_64: usize,
}

/// Complete lowered MIR module ready for RAM reporting and LLVM backend emission.
#[derive(Debug, Clone)]
pub struct MirModule {
    pub tick_period_us: u64,
    pub systick_hz: Option<u64>,
    pub imports: Vec<MirImport>,
    pub consts: Vec<MirConst>,
    pub enums: Vec<MirEnum>,
    pub structs: Vec<MirStruct>,
    pub mmio_layouts: Vec<MirMmioLayout>,
    pub mmio_instances: Vec<MirMmioInstance>,
    pub primitives: Vec<MirPrimitive>,
    pub subsystems: Vec<MirSubsystemGlobal>,
    pub extern_fns: Vec<MirExternFn>,
    pub isrs: Vec<MirIsr>,
    pub irq_vectors: Vec<MirIrqVector>,
    pub functions: Vec<MirFunction>,
    pub task_blueprints: Vec<MirTaskBlueprint>,
    pub spawned_tasks: Vec<MirSpawnInstance>,
    pub dual_specialized_fns: Vec<String>,
    pub monomorphized_instances: Vec<MonomorphizedMirItem>,
    /// Sorted list of enum names referenced by `MirFmtChunk::EnumName` (for on-demand `.rodata` variant-name tables).
    pub formatted_enums: Vec<String>,
    pub boot: MirBootSequence,
    pub visibility: crate::opt::VisibilityReport,
    pub ram_report: RamReport,
}
