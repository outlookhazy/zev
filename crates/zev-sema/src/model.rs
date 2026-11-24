//! Rich `SemanticModel` exported by `zev-sema` for `zev-mir`, `zev-codegen-llvm`,
//! `zev-codegen-llvm`, and `zev-cli`.

use std::collections::{BTreeMap, BTreeSet};
use zev_ast::{DurationLit, Expr, MmioAccess, PrimType, Priority, Span};

use crate::call_graph::CallGraph;
use crate::types::{
    EnumInfo, PrimitiveKind, ResolvedErrorSpec, ResolvedOsPrimitive, StructInfo, Ty,
};

/// Default scheduler tick period in microseconds (`1ms` = `1000us`).
pub const DEFAULT_TICK_PERIOD_US: u64 = 1_000;

/// Three-tier function/task scheduling effect inferred across the whole-program call graph.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FnEffect {
    /// Contains zero yields; compiles to a direct C/LLVM stack call (`0 B` `.bss`) and is legal
    /// everywhere (`noyield`, `critical`, `isr`, `defer`, `extern "C"` callbacks).
    #[default]
    NonYielding,
    /// Terminating cooperative compute that only yields politely via `checkpoint()` or terminating
    /// `loop` (`should_share()`). Legal in cooperative coroutines AND inside `noyield { ... }`
    /// (via stack-only specialization with `checkpoint()` stripped to a no-op), while still
    /// forbidden inside `critical`, `isr`, `defer`, and `extern "C"` callbacks.
    SoftYielding,
    /// Awaits external time/events (`delay`, `loop every`, `.recv()`, `.send()`, `.wait()`,
    /// `.wait_change()`, `.wait_chunk()`, or non-breaking infinite `loop`/`while (true)`).
    /// Legal only in cooperative `task`s and `main()` outside `noyield`/`critical`/`isr`/`defer`.
    HardBlocking,
}

impl FnEffect {
    /// Returns `true` if this effect may yield in normal cooperative execution (`SoftYielding` or `HardBlocking`).
    pub const fn can_yield(self) -> bool {
        !matches!(self, Self::NonYielding)
    }
}

/// Resolved parameter metadata on a `fn` or `task`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamInfo {
    pub name: String,
    pub ty: Ty,
    pub span: Span,
}

/// Resolved function metadata (`fn` and `extern "C"`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionInfo {
    pub name: String,
    pub params: Vec<ParamInfo>,
    pub is_variadic: bool,
    pub ret_ty: Ty,
    pub error_spec: ResolvedErrorSpec,
    pub is_extern_c: bool,
    pub direct_yield: bool,
    pub can_yield: bool,
    pub effect: FnEffect,
    /// True if this function calls `spawn` directly or transitively during boot setup.
    pub spawns_in_boot: bool,
    pub yield_chain: Vec<String>,
    pub callees: Vec<String>,
    pub yielding_callees: Vec<String>,
    /// Parameter indices whose reference/slice provenance may be returned by this function.
    pub return_param_origins: Vec<usize>,
    pub span: Span,
}

/// Resolved reusable `task` blueprint metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskInfo {
    pub name: String,
    pub params: Vec<ParamInfo>,
    pub direct_yield: bool,
    pub can_yield: bool,
    pub effect: FnEffect,
    pub callees: Vec<String>,
    pub yielding_callees: Vec<String>,
    /// Number of static `spawn` sites instantiating this task blueprint.
    pub instance_count: usize,
    pub span: Span,
}

/// Static task instance created at a `spawn` site (e.g. `spawn motor_control(&motor1_rpm, 1) @critical;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpawnedTaskInstance {
    /// 0-indexed static task slot ID in boot-tree spawn order.
    pub instance_id: usize,
    /// Blueprint task name.
    pub task_name: String,
    /// Base priority assigned at spawn (`@normal` if unspecified).
    pub priority: Priority,
    /// Optional hardware core affinity (`@core(N)`).
    pub core: Option<u32>,
    /// Immediate enclosing function where `spawn` appeared (e.g. `"main"` or `"usb_start"`).
    pub spawned_in: String,
    /// Full static boot-tree call path ending in `task_name` (e.g. `["main", "usb_start", "usb_worker"]`).
    pub provenance_path: Vec<String>,
    /// True when spawned inside a conditional `if`/`else`/`switch` boot branch.
    pub conditional: bool,
    /// Static endpoint names referenced by pointer arguments (e.g. `["motor1_rpm"]`).
    pub endpoint_bindings: Vec<Option<String>>,
    pub span: Span,
}

impl SpawnedTaskInstance {
    /// Formats `provenance_path` as `"main -> usb_start -> usb_worker"`.
    pub fn format_provenance(&self) -> String {
        self.provenance_path.join(" -> ")
    }
}

/// Priority boost event posted by an ISR or task (`inbox.post(ev, boost: @critical)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PriorityBoostSite {
    pub primitive_name: String,
    pub boost_priority: Priority,
    pub span: Span,
}

/// Resolved hardware `isr` metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IsrInfo {
    pub vector: String,
    pub params: Vec<ParamInfo>,
    pub callees: Vec<String>,
    pub priority_boosts: Vec<PriorityBoostSite>,
    pub span: Span,
}

/// Resolved `mmio` register metadata inside an [`MmioLayoutInfo`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmioRegInfo {
    pub name: String,
    pub access: MmioAccess,
    pub elem_ty: PrimType,
    pub array_len: Option<usize>,
    pub offset: u32,
    pub span: Span,
}

/// Resolved reusable `mmio` register block layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmioLayoutInfo {
    pub name: String,
    pub byte_size: usize,
    pub regs: Vec<MmioRegInfo>,
    pub span: Span,
}

/// Resolved physical `mmio` peripheral instance (`mmio USART1 : UartRegs @ 0x4001_1000;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmioInstanceInfo {
    pub name: String,
    pub layout: String,
    pub base_addr: u32,
    pub span: Span,
}

/// Resolved hardware `irq` vector token (`irq IRQ_EXTI0 = 0;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrqInfo {
    pub name: String,
    pub irq_num: u32,
    pub span: Span,
}

/// Boot-time argument passed to a bound hardware ISR (`vec.attach(handler(args...))`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrqBindingArg {
    /// Static OS/DMA primitive handle address (`&sup_inbox`).
    PrimAddr(String),
    /// Static subsystem instance or nested primitive/subsystem field address (`&uart0`, `&uart0.rx`).
    SubsystemPath {
        root: String,
        root_struct: String,
        field_chain: Vec<(String, String)>,
    },
    /// Static `mmio` peripheral instance base pointer (`USART1` / `&USART1`).
    MmioAddr(String),
    /// Compile-time constant scalar (`i128` integer value + resolved scalar type).
    ConstInt { ty: Ty, value: i128 },
    /// Compile-time constant boolean.
    ConstBool(bool),
    /// Other compile-time constant expression (e.g. `const` float or resolved enum literal).
    ConstExpr { ty: Ty, expr: Box<Expr> },
}

/// Resolved boot-time `irq.attach(...)` or `irq.attach_shared(...)` binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrqBindingInfo {
    pub irq_name: String,
    pub irq_num: u32,
    pub is_shared: bool,
    pub isr_name: String,
    pub args: Vec<IrqBindingArg>,
    pub conditional_branch: Option<usize>,
    pub span: Span,
}

/// Resolved ISR-private `state` variable (`state u32 count = 0;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IsrStateInfo {
    pub isr_name: String,
    pub name: String,
    pub ty: Ty,
    pub span: Span,
}

/// Resolved static OS / DMA primitive instance in global SRAM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimitiveInfo {
    pub name: String,
    pub primitive: ResolvedOsPrimitive,
    pub kind: PrimitiveKind,
    pub elem_ty: Option<Ty>,
    pub capacity: Option<usize>,
    /// True if annotated `@dma_safe`.
    pub dma_safe: bool,
    /// True if initialized at declaration (`watch<u16> motor1_rpm = 0;`).
    pub has_init: bool,
    /// True if active transfers on this primitive clamp sleep depth from `DEEP` to `LIGHT`.
    pub clamps_sleep_depth: bool,
    pub span: Span,
}

/// Resolved static subsystem struct instance in global SRAM (`UartSubsys uart0;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubsystemInfo {
    pub name: String,
    pub struct_name: String,
    /// True if annotated `@dma_safe`.
    pub dma_safe: bool,
    pub span: Span,
}

/// Resolved immutable `const` metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstInfo {
    pub name: String,
    pub ty: Ty,
    pub int_value: Option<u64>,
    pub span: Span,
}

/// Kind of monomorphized generic declaration (`struct` or `fn`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MonomorphizedKind {
    Struct,
    Fn,
}

impl MonomorphizedKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Struct => "struct",
            Self::Fn => "fn",
        }
    }
}

impl std::fmt::Display for MonomorphizedKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Concrete instantiation produced by monomorphizing a generic `struct` or `fn`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonomorphizedInstance {
    pub kind: MonomorphizedKind,
    /// Human-readable instantiated signature (e.g. `"FixedVec<u32, 16>"` or `"clamp<f32>"`).
    pub signature: String,
    /// Deterministic mangled identifier (e.g. `"FixedVec__u32_16"` or `"clamp__f32"`).
    pub mangled_name: String,
}

/// Complete validated semantic model of a `.zev` compilation unit (single- or multi-file).
#[derive(Clone, Debug)]
pub struct SemanticModel {
    /// Configured scheduler tick resolution in microseconds (default `1000` = `1ms`).
    pub tick_period_us: u64,
    /// Optional core clock frequency passed to `sys.use_systick(hz)`.
    pub systick_hz: Option<u64>,
    /// User-defined `struct` and `packed struct` types.
    pub structs: BTreeMap<String, StructInfo>,
    /// User-defined `enum` types.
    pub enums: BTreeMap<String, EnumInfo>,
    /// Top-level immutable `const` declarations.
    pub consts: BTreeMap<String, ConstInfo>,
    /// Top-level static OS & DMA primitive endpoints.
    pub primitives: BTreeMap<String, PrimitiveInfo>,
    /// Top-level static subsystem struct instances.
    pub subsystems: BTreeMap<String, SubsystemInfo>,
    /// Declared `fn` and `extern "C"` functions.
    pub functions: BTreeMap<String, FunctionInfo>,
    /// Declared `task` blueprints.
    pub tasks: BTreeMap<String, TaskInfo>,
    /// Declared hardware `isr` handlers.
    pub isrs: BTreeMap<String, IsrInfo>,
    /// Typed MMIO register block layouts (`UartRegs`, `<mod>__UartRegs`).
    pub mmio_layouts: BTreeMap<String, MmioLayoutInfo>,
    /// Typed physical MMIO peripheral instances (`USART1`, `<mod>__USART1`).
    pub mmio_instances: BTreeMap<String, MmioInstanceInfo>,
    /// Hardware IRQ vector tokens (`IRQ_USART1`, `<mod>__IRQ_USART1`).
    pub irqs: BTreeMap<String, IrqInfo>,
    /// Validated boot-tree `vec.attach(...)` and `vec.attach_shared(...)` bindings.
    pub irq_bindings: Vec<IrqBindingInfo>,
    /// Validated ISR-private `state` variables across all `isr` handlers.
    pub isr_states: Vec<IsrStateInfo>,
    /// Human-readable display names for canonicalized module symbols (`mod__Foo` -> `alias.Foo`).
    pub display_names: BTreeMap<String, String>,
    /// Static task instances allocated across all `spawn` sites in boot-tree order.
    pub spawned_tasks: Vec<SpawnedTaskInstance>,
    /// Concrete generic `struct` and `fn` instantiations (deduplicated & sorted).
    pub monomorphized_instances: Vec<MonomorphizedInstance>,
    /// `SoftYielding` functions reachable in normal cooperative coroutine execution (outside `noyield`).
    pub soft_yielding_in_coro: BTreeSet<String>,
    /// `SoftYielding` functions reachable inside `noyield { ... }` (compiled as stack-only clones).
    pub soft_yielding_in_noyield: BTreeSet<String>,
    /// `SoftYielding` functions reachable in BOTH coroutine and `noyield` contexts (`soft_yielding_in_coro ∩ soft_yielding_in_noyield`).
    pub dual_specialized_fns: BTreeSet<String>,
    /// Priority assigned to `main` via `set_priority(@low)` if `main` continues as supervisor task.
    pub main_supervisor_priority: Option<Priority>,
    /// True if `main()` exists and contains trailing yielding/loop supervisor execution.
    pub main_can_yield: bool,
    /// Underlying whole-program call graph.
    pub call_graph: CallGraph,
}

impl SemanticModel {
    /// Returns the inferred [`FnEffect`] of `name` (`fn` or `task`), defaulting to `NonYielding`.
    pub fn fn_effect(&self, name: &str) -> FnEffect {
        if let Some(f) = self.functions.get(name) {
            return f.effect;
        }
        if let Some(t) = self.tasks.get(name) {
            return t.effect;
        }
        FnEffect::NonYielding
    }

    /// Returns `true` if `name` is a function or task that can yield directly or transitively.
    pub fn can_yield(&self, name: &str) -> bool {
        if let Some(f) = self.functions.get(name) {
            return f.can_yield;
        }
        if let Some(t) = self.tasks.get(name) {
            return t.can_yield;
        }
        false
    }

    /// Returns the sorted list of yielding functions called directly by `name`.
    pub fn yielding_callees(&self, name: &str) -> &[String] {
        if let Some(f) = self.functions.get(name) {
            return &f.yielding_callees;
        }
        if let Some(t) = self.tasks.get(name) {
            return &t.yielding_callees;
        }
        &[]
    }

    /// Convert a validated `DurationLit` into exact scheduler ticks (`dur_us / tick_period_us`).
    pub fn duration_to_ticks(&self, dur: &DurationLit) -> u64 {
        dur.as_micros() / self.tick_period_us.max(1)
    }

    /// Convert microseconds into exact scheduler ticks.
    pub fn micros_to_ticks(&self, us: u64) -> u64 {
        us / self.tick_period_us.max(1)
    }

    /// Returns `true` if any task is pinned to a specific core via `@core(N)`.
    pub fn uses_multicore_pinning(&self) -> bool {
        self.spawned_tasks.iter().any(|s| s.core.is_some())
    }

    /// Returns `true` if any declared primitive or subsystem requires DMA sleep-lock clamping
    /// (`ping_pong` or `@dma_safe pool`).
    pub fn has_dma_sleep_locks(&self) -> bool {
        if self.primitives.values().any(|p| p.clamps_sleep_depth) {
            return true;
        }
        fn struct_has_dma_lock(
            st_name: &str,
            parent_dma: bool,
            structs: &BTreeMap<String, StructInfo>,
        ) -> bool {
            let Some(st) = structs.get(st_name) else {
                return false;
            };
            st.fields.iter().any(|f| {
                let eff_dma = parent_dma || f.dma_safe;
                match &f.ty {
                    Ty::OsPrimitive(ResolvedOsPrimitive::PingPong { .. }) => true,
                    Ty::OsPrimitive(ResolvedOsPrimitive::Pool { .. }) => eff_dma,
                    Ty::Struct(inner) => struct_has_dma_lock(inner, eff_dma, structs),
                    _ => false,
                }
            })
        }
        self.subsystems
            .values()
            .any(|sub| struct_has_dma_lock(&sub.struct_name, sub.dma_safe, &self.structs))
    }
}
