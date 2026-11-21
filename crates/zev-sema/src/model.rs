//! Rich `SemanticModel` exported by `zev-sema` for `zev-mir`, `zev-codegen-c`,
//! `zev-codegen-llvm`, and `zev-cli`.

use std::collections::BTreeMap;
use zev_ast::{DurationLit, Priority, Span};

use crate::call_graph::CallGraph;
use crate::types::{
    EnumInfo, PrimitiveKind, ResolvedErrorSpec, ResolvedOsPrimitive, StructInfo, Ty,
};

/// Default scheduler tick period in microseconds (`1ms` = `1000us`).
pub const DEFAULT_TICK_PERIOD_US: u64 = 1_000;

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
    pub callees: Vec<String>,
    pub yielding_callees: Vec<String>,
    /// Number of static `spawn` sites instantiating this task blueprint.
    pub instance_count: usize,
    pub span: Span,
}

/// Static task instance created at a `spawn` site (e.g. `spawn motor_control(&motor1_rpm, 1) @critical;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpawnedTaskInstance {
    /// 0-indexed static task slot ID in declaration/spawn order.
    pub instance_id: usize,
    /// Blueprint task name.
    pub task_name: String,
    /// Base priority assigned at spawn (`@normal` if unspecified).
    pub priority: Priority,
    /// Optional hardware core affinity (`@core(N)`).
    pub core: Option<u32>,
    /// Enclosing function or task where `spawn` appeared (typically `"main"`).
    pub spawned_in: String,
    /// Static endpoint names referenced by pointer arguments (e.g. `["motor1_rpm"]`).
    pub endpoint_bindings: Vec<Option<String>>,
    pub span: Span,
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
    pub callees: Vec<String>,
    pub priority_boosts: Vec<PriorityBoostSite>,
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

/// Resolved immutable `const` metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstInfo {
    pub name: String,
    pub ty: Ty,
    pub int_value: Option<u64>,
    pub span: Span,
}

/// Complete validated semantic model of a `.zev` compilation unit (single- or multi-file).
#[derive(Clone, Debug)]
pub struct SemanticModel {
    /// Configured scheduler tick resolution in microseconds (default `1000` = `1ms`).
    pub tick_period_us: u64,
    /// Optional core clock frequency passed to `sys.use_systick(hz)`.
    pub systick_hz: Option<u64>,
    /// Imported chip/SVD modules (`alias -> "chip.stm32g4"`).
    pub chip_imports: BTreeMap<String, String>,
    /// User-defined `struct` and `packed struct` types.
    pub structs: BTreeMap<String, StructInfo>,
    /// User-defined `enum` types.
    pub enums: BTreeMap<String, EnumInfo>,
    /// Top-level immutable `const` declarations.
    pub consts: BTreeMap<String, ConstInfo>,
    /// Top-level static OS & DMA primitive endpoints.
    pub primitives: BTreeMap<String, PrimitiveInfo>,
    /// Declared `fn` and `extern "C"` functions.
    pub functions: BTreeMap<String, FunctionInfo>,
    /// Declared `task` blueprints.
    pub tasks: BTreeMap<String, TaskInfo>,
    /// Declared hardware `isr` handlers.
    pub isrs: BTreeMap<String, IsrInfo>,
    /// Static task instances allocated across all `spawn` sites in spawn order.
    pub spawned_tasks: Vec<SpawnedTaskInstance>,
    /// Priority assigned to `main` via `set_priority(@low)` if `main` continues as supervisor task.
    pub main_supervisor_priority: Option<Priority>,
    /// True if `main()` exists and contains trailing yielding/loop supervisor execution.
    pub main_can_yield: bool,
    /// Underlying whole-program call graph.
    pub call_graph: CallGraph,
}

impl SemanticModel {
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

    /// Returns `true` if any declared primitive requires DMA sleep-lock clamping (`ping_pong`, `lease`).
    pub fn has_dma_sleep_locks(&self) -> bool {
        self.primitives.values().any(|p| p.clamps_sleep_depth)
    }
}
