//! Resolved semantic types, struct/enum metadata, and OS primitive endpoint descriptors.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use zev_ast::{PrimType, Span};

/// Resolved semantic type representation used across `zev-sema` and `SemanticModel`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    /// Built-in scalar, boolean, floating-point, wire-endian, or `void` primitive.
    Prim(PrimType),
    /// Untyped integer literal (`0`, `5`, `170_000_000`) before coercion to a concrete integer type.
    IntLit,
    /// Unsuffixed floating-point literal (`0.0`, `3.14`, `1e-3`) defaulting to `f64` or coercing to contextual `f32`.
    FloatLit,
    /// Duration literal (`1ms`, `500us`, `5s`).
    Duration,
    /// Scheduler priority literal (`@critical`, `@high`, `@normal`, `@low`).
    Priority,
    /// String literal (used in debug/trace/extern "C" calls).
    Str,
    /// User-defined `struct` or `packed struct`.
    Struct(String),
    /// User-defined `enum`.
    Enum(String),
    /// Fat slice `[]T` (`ptr` + `len`).
    Slice(Box<Ty>),
    /// Fixed-length array `[N]T`.
    Array { elem: Box<Ty>, len: usize },
    /// Raw or endpoint pointer `*T`.
    Ptr(Box<Ty>),
    /// Static scheduler/DMA OS primitive endpoint (`watch<u16>`, `ping_pong<u8, 64>`, etc.).
    OsPrimitive(ResolvedOsPrimitive),
    /// Function pointer or function reference (`fn(T1, T2) -> Ret`).
    FnPtr {
        params: Vec<Ty>,
        ret: Box<Ty>,
    },
    /// Typed MMIO register block layout (`UartRegs`), usable behind pointer `*UartRegs` (`mmio` values cannot be copied by value).
    MmioLayout(String),
    /// Typed physical MMIO peripheral instance (`USART1 : UartRegs @ 0x4001_1000`), coercing implicitly to `*UartRegs`.
    MmioInstance { name: String, layout: String },
    /// Hardware interrupt vector token (`irq IRQ_USART1 = 37;`).
    IrqToken,
    /// Built-in namespace (`sys`, `sim`, imported module alias, or type name for associated calls).
    Namespace(NamespaceKind),
    /// Error recovery sentinel to prevent cascading duplicate diagnostics.
    Error,
}

/// Built-in or user type namespace kinds in receiver position (`sys.use_systick`, `CmdFrame.from_bytes`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NamespaceKind {
    Sys,
    Sim,
    Module(String),
    Type(String),
    Enum(String),
}

/// Kind of scheduler-aware static OS primitive.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimitiveKind {
    Signal,
    Inbox,
    Watch,
    PingPong,
    Lease,
}

impl PrimitiveKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Signal => "signal",
            Self::Inbox => "inbox",
            Self::Watch => "watch",
            Self::PingPong => "ping_pong",
            Self::Lease => "lease",
        }
    }

    /// Active DMA primitives (`ping_pong`, `lease`) clamp tickless idle sleep depth
    /// from `DEEP` to `LIGHT` so peripheral clocks remain active during transfers.
    pub const fn clamps_sleep_depth(self) -> bool {
        matches!(self, Self::PingPong | Self::Lease)
    }
}

impl fmt::Display for PrimitiveKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Resolved OS primitive endpoint type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResolvedOsPrimitive {
    Signal,
    Inbox { elem: Box<Ty>, cap: usize },
    Watch { elem: Box<Ty> },
    PingPong { elem: Box<Ty>, cap: usize },
    Lease { elem: Box<Ty>, cap: usize },
}

impl ResolvedOsPrimitive {
    pub const fn kind(&self) -> PrimitiveKind {
        match self {
            Self::Signal => PrimitiveKind::Signal,
            Self::Inbox { .. } => PrimitiveKind::Inbox,
            Self::Watch { .. } => PrimitiveKind::Watch,
            Self::PingPong { .. } => PrimitiveKind::PingPong,
            Self::Lease { .. } => PrimitiveKind::Lease,
        }
    }

    pub fn elem_ty(&self) -> Option<&Ty> {
        match self {
            Self::Signal => None,
            Self::Inbox { elem, .. }
            | Self::Watch { elem }
            | Self::PingPong { elem, .. }
            | Self::Lease { elem, .. } => Some(elem),
        }
    }

    pub const fn capacity(&self) -> Option<usize> {
        match self {
            Self::Signal | Self::Watch { .. } => None,
            Self::Inbox { cap, .. } | Self::PingPong { cap, .. } | Self::Lease { cap, .. } => {
                Some(*cap)
            }
        }
    }
}

impl fmt::Display for ResolvedOsPrimitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Signal => write!(f, "signal"),
            Self::Inbox { elem, cap } => write!(f, "inbox<{}, {}>", elem, cap),
            Self::Watch { elem } => write!(f, "watch<{}>", elem),
            Self::PingPong { elem, cap } => write!(f, "ping_pong<{}, {}>", elem, cap),
            Self::Lease { elem, cap } => write!(f, "lease<{}, {}>", elem, cap),
        }
    }
}

/// Resolved dual-exit error specification on a function or fallible operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedErrorSpec {
    /// Infallible (`return` only).
    None,
    /// Untyped boolean failure (`error`).
    Untyped,
    /// Typed failure (`error ErrType`).
    Typed(Ty),
}

impl ResolvedErrorSpec {
    pub const fn is_fallible(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Resolved struct field metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInfo {
    pub name: String,
    pub ty: Ty,
    pub span: Span,
}

/// Resolved `struct` or `packed struct` metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructInfo {
    pub name: String,
    pub is_packed: bool,
    pub fields: Vec<FieldInfo>,
    pub span: Span,
}

impl StructInfo {
    pub fn field(&self, name: &str) -> Option<&FieldInfo> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// Resolved `enum` metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumInfo {
    pub name: String,
    pub variants: Vec<String>,
    pub span: Span,
}

impl EnumInfo {
    pub fn has_variant(&self, v: &str) -> bool {
        self.variants.iter().any(|x| x == v)
    }
}

impl Ty {
    pub const fn void() -> Self {
        Self::Prim(PrimType::Void)
    }

    pub const fn bool() -> Self {
        Self::Prim(PrimType::Bool)
    }

    pub const fn usize() -> Self {
        Self::Prim(PrimType::Usize)
    }

    pub const fn u8() -> Self {
        Self::Prim(PrimType::U8)
    }

    pub const fn u16() -> Self {
        Self::Prim(PrimType::U16)
    }

    pub const fn u32() -> Self {
        Self::Prim(PrimType::U32)
    }

    pub const fn f32() -> Self {
        Self::Prim(PrimType::F32)
    }

    pub const fn f64() -> Self {
        Self::Prim(PrimType::F64)
    }

    pub const fn is_void(&self) -> bool {
        matches!(self, Self::Prim(PrimType::Void))
    }

    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error)
    }

    pub const fn is_integer(&self) -> bool {
        match self {
            Self::IntLit => true,
            Self::Prim(p) => p.is_integer(),
            _ => false,
        }
    }

    pub const fn is_float(&self) -> bool {
        match self {
            Self::FloatLit => true,
            Self::Prim(p) => p.is_float(),
            _ => false,
        }
    }

    pub const fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    pub const fn is_bool(&self) -> bool {
        matches!(self, Self::Prim(PrimType::Bool))
    }

    /// Normalize wire-endian scalar types (`u16be`/`u16le` -> `u16`, etc.) to their host scalar counterpart.
    pub const fn host_prim(p: PrimType) -> PrimType {
        match p {
            PrimType::U16Be | PrimType::U16Le => PrimType::U16,
            PrimType::U32Be | PrimType::U32Le => PrimType::U32,
            PrimType::U64Be | PrimType::U64Le => PrimType::U64,
            other => other,
        }
    }

    /// Returns `true` if values of this type contain (directly or inside struct/array fields)
    /// any pointer `*T` or slice `[]T` whose lexical scope depth / DMA lease must be tracked.
    pub fn contains_reference(&self, structs: &BTreeMap<String, StructInfo>) -> bool {
        let mut visited = BTreeSet::new();
        self.contains_ref_inner(structs, &mut visited)
    }

    fn contains_ref_inner(
        &self,
        structs: &BTreeMap<String, StructInfo>,
        visited: &mut BTreeSet<String>,
    ) -> bool {
        match self {
            Self::Ptr(_) | Self::Slice(_) => true,
            Self::Array { elem, .. } => elem.contains_ref_inner(structs, visited),
            Self::Struct(name) => {
                if !visited.insert(name.clone()) {
                    return false;
                }
                if let Some(info) = structs.get(name) {
                    info.fields
                        .iter()
                        .any(|f| f.ty.contains_ref_inner(structs, visited))
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Returns `true` if values of this type contain any function pointer (directly or inside struct fields).
    pub fn contains_fn_ptr(&self, structs: &BTreeMap<String, StructInfo>) -> bool {
        let mut visited = BTreeSet::new();
        self.contains_fn_ptr_inner(structs, &mut visited)
    }

    fn contains_fn_ptr_inner(
        &self,
        structs: &BTreeMap<String, StructInfo>,
        visited: &mut BTreeSet<String>,
    ) -> bool {
        match self {
            Self::FnPtr { .. } => true,
            Self::Ptr(inner) | Self::Slice(inner) => inner.contains_fn_ptr_inner(structs, visited),
            Self::Array { elem, .. } => elem.contains_fn_ptr_inner(structs, visited),
            Self::Struct(name) => {
                if !visited.insert(name.clone()) {
                    return false;
                }
                structs.get(name).is_some_and(|s| {
                    s.fields
                        .iter()
                        .any(|f| f.ty.contains_fn_ptr_inner(structs, visited))
                })
            }
            _ => false,
        }
    }

    /// Checks whether a value of type `actual` can be assigned/passed to `expected`.
    pub fn is_assignable_to(&self, expected: &Ty) -> bool {
        if self.is_error() || expected.is_error() {
            return true;
        }
        if self == expected {
            return true;
        }
        match (self, expected) {
            // Untyped integer literals coerce to any integer type.
            (Ty::IntLit, Ty::Prim(p)) if p.is_integer() => true,
            (Ty::Prim(p), Ty::IntLit) if p.is_integer() => true,

            // Unsuffixed float literals coerce to `f32` or `f64`.
            (Ty::FloatLit, Ty::Prim(PrimType::F32 | PrimType::F64)) => true,
            (Ty::Prim(PrimType::F32 | PrimType::F64), Ty::FloatLit) => true,

            // Wire-endian types and fixed-width integers convert among integer types in C-style
            // firmware expressions, while keeping strict separation from bool, floats, enums, durations, and pointers.
            (Ty::Prim(a), Ty::Prim(b)) if a.is_integer() && b.is_integer() => true,

            // Fixed-array `[N]T` coerces elementwise for literal arrays (`[1, 2]` -> `[2]u32`,
            // `[0.5, 1.0]` -> `[2]f32`) and `[N]T` coerces to slice `[]T` when element types match.
            (Ty::Array { elem: a, len: la }, Ty::Array { elem: b, len: lb }) => {
                la == lb && a.is_assignable_to(b)
            }
            (Ty::Array { elem: a, .. }, Ty::Slice(b)) => {
                a == b || (matches!(**a, Ty::IntLit | Ty::FloatLit) && a.is_assignable_to(b))
            }

            // Any pointer `*T` coerces to opaque `*void` (e.g. for extern "C" or generic DMA helpers),
            // and string literals coerce to `*u8` / `*i8` / `[]u8` / `*void` for trace/C helpers.
            (Ty::Ptr(_), Ty::Ptr(inner)) if inner.is_void() => true,
            (Ty::Str, Ty::Ptr(inner))
                if inner.is_void()
                    || matches!(**inner, Ty::Prim(PrimType::U8 | PrimType::I8)) =>
            {
                true
            }
            (Ty::Str, Ty::Slice(inner)) if matches!(**inner, Ty::Prim(PrimType::U8)) => true,

            // Typed physical MMIO peripheral instance coerces implicitly to `*Layout`.
            (Ty::MmioInstance { layout: a, .. }, Ty::Ptr(inner))
                if matches!(&**inner, Ty::MmioLayout(b) if a == b) =>
            {
                true
            }

            // Function pointers match when parameter/return types are compatible.
            (
                Ty::FnPtr {
                    params: pa,
                    ret: ra,
                },
                Ty::FnPtr {
                    params: pb,
                    ret: rb,
                },
            ) => {
                pa.len() == pb.len()
                    && pa.iter().zip(pb.iter()).all(|(a, b)| a.is_assignable_to(b))
                    && ra.is_assignable_to(rb)
            }
            // Allow passing a function reference to opaque `*void` callback parameter in extern "C".
            (Ty::FnPtr { .. }, Ty::Ptr(inner)) if inner.is_void() => true,

            _ => false,
        }
    }

    /// Estimate byte size of a value given `ptr_bytes` (`4` on 32-bit MCU, `8` on 64-bit host).
    pub fn byte_size(&self, ptr_bytes: usize, structs: &BTreeMap<String, StructInfo>) -> usize {
        match self {
            Self::Prim(p) => p.byte_size(ptr_bytes),
            Self::IntLit => 4,
            Self::FloatLit => 8,
            Self::Duration => 8,
            Self::Priority | Self::Enum(_) => 1,
            Self::Str | Self::Slice(_) => ptr_bytes * 2,
            Self::Ptr(_) | Self::FnPtr { .. } | Self::MmioInstance { .. } => ptr_bytes,
            Self::IrqToken => 4,
            Self::MmioLayout(_) => 0,
            Self::Array { elem, len } => elem.byte_size(ptr_bytes, structs) * len,
            Self::Struct(name) => {
                if let Some(s) = structs.get(name) {
                    s.fields
                        .iter()
                        .map(|f| f.ty.byte_size(ptr_bytes, structs))
                        .sum()
                } else {
                    0
                }
            }
            Self::OsPrimitive(p) => match p {
                ResolvedOsPrimitive::Signal => 4,
                ResolvedOsPrimitive::Watch { elem } => elem.byte_size(ptr_bytes, structs) + 4,
                ResolvedOsPrimitive::Inbox { elem, cap } => {
                    elem.byte_size(ptr_bytes, structs) * cap + 16
                }
                ResolvedOsPrimitive::PingPong { elem, cap } => {
                    elem.byte_size(ptr_bytes, structs) * cap * 2 + 16
                }
                ResolvedOsPrimitive::Lease { elem, cap } => {
                    elem.byte_size(ptr_bytes, structs) * cap + 16
                }
            },
            Self::Namespace(_) | Self::Error => 0,
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prim(p) => write!(f, "{}", p),
            Self::IntLit => write!(f, "{{integer}}"),
            Self::FloatLit => write!(f, "{{float}}"),
            Self::Duration => write!(f, "duration"),
            Self::Priority => write!(f, "priority"),
            Self::Str => write!(f, "str"),
            Self::Struct(name) | Self::Enum(name) => write!(f, "{}", name),
            Self::Slice(elem) => write!(f, "[]{}", elem),
            Self::Array { elem, len } => write!(f, "[{}]{}", len, elem),
            Self::Ptr(inner) => write!(f, "*{}", inner),
            Self::OsPrimitive(p) => write!(f, "{}", p),
            Self::FnPtr { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Self::MmioLayout(name) => write!(f, "mmio {name}"),
            Self::MmioInstance { name, layout } => write!(f, "mmio {name} : {layout}"),
            Self::IrqToken => write!(f, "irq"),
            Self::Namespace(ns) => match ns {
                NamespaceKind::Sys => write!(f, "sys"),
                NamespaceKind::Sim => write!(f, "sim"),
                NamespaceKind::Module(m) => write!(f, "module {}", m),
                NamespaceKind::Type(t) | NamespaceKind::Enum(t) => write!(f, "type {}", t),
            },
            Self::Error => write!(f, "<error>"),
        }
    }
}
