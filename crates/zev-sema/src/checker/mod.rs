//! Multi-pass semantic analyzer and static verifier for the Zev Language-OS.

pub(super) use std::collections::{BTreeMap, BTreeSet};
pub(super) use std::path::PathBuf;

pub(super) use zev_ast::{
    ArrayLen, AssignOp, Attr, AttrKind, BinaryOp, Block, BuiltinFn, CallArg, CatchClause,
    ConstDecl, DurationLit, EnumDecl, ErrorSpec, Expr, ExprKind, FnDecl, FormatPart, FormatSpec,
    GlobalVarDecl, Ident, IrqDecl, IsrDecl, Item, ItemKind, Literal, MmioAccess, MmioDecl,
    MmioDeclKind, OsPrimitiveDecl, OsPrimitiveType, Param, PrimType, Priority, Program, Span,
    SpawnStmt, Stmt, StmtKind, StructDecl, SwitchPat, SysMethod, TaskDecl, TypeExpr, TypeKind,
    UnaryOp, SYS_NAMESPACE,
};

pub(super) use crate::call_graph::{
    CallContext, CallEdge, CallGraph, CallableKind, ExternCallbackSite,
};
pub(super) use crate::diag::{Diagnostic, DiagnosticKind};
pub(super) use crate::model::{
    ConstInfo, FnEffect, FunctionInfo, IrqBindingArg, IrqBindingInfo, IrqInfo, IsrInfo,
    IsrStateInfo, MmioInstanceInfo, MmioLayoutInfo, MmioRegInfo, MonomorphizedInstance,
    MonomorphizedKind, ParamInfo, PrimitiveInfo, PriorityBoostSite, SemanticModel,
    SpawnedTaskInstance, SubsystemInfo, TaskInfo, DEFAULT_TICK_PERIOD_US,
};
pub(super) use crate::modules::LinkedModulesOutput;
pub(super) use crate::provenance::{LeaseEnv, Provenance};
pub(super) use crate::suggest::{
    advise_primitive_method, closest_match, suggest_foreign_type_name,
};
pub(super) use crate::types::{
    EnumInfo, FieldInfo, NamespaceKind, PrimitiveKind, ResolvedErrorSpec, ResolvedOsPrimitive,
    StructInfo, Ty,
};

/// Configuration options for semantic analysis.
#[derive(Clone, Debug)]
pub struct AnalyzeOptions {
    /// Default scheduler tick resolution in microseconds (`1000` = `1ms`).
    pub default_tick_period_us: u64,
    /// Explicit module search directories (`-I` include paths) checked by module resolution.
    pub include_dirs: Vec<PathBuf>,
    /// Target-bound HAL directory override (e.g. `"hal/stm32f3"`), defaulting to `"hal/sim"`.
    pub hal_dir: Option<String>,
    /// Target-bound board module override (e.g. `"bsp/nucleo_f334r8"`), defaulting to `"bsp/sim"`.
    pub board_module: Option<String>,
    /// Pointer bit-width of the selected target (`16` on AVR, `32` on Cortex-M/RISC-V, `64` on host).
    pub target_ptr_bits: u32,
    /// Human-readable target name reported in `usize`/`isize` range diagnostics.
    pub target_name: String,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            default_tick_period_us: DEFAULT_TICK_PERIOD_US,
            include_dirs: Vec::new(),
            hal_dir: None,
            board_module: None,
            target_ptr_bits: 64,
            target_name: "host".to_string(),
        }
    }
}

/// Output of running `zev-sema` even when diagnostics are present (useful for IDEs/CLI).
#[derive(Clone, Debug)]
pub struct AnalysisOutput {
    pub model: SemanticModel,
    pub diagnostics: Vec<Diagnostic>,
}

impl AnalysisOutput {
    pub fn is_ok(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn into_result(self) -> Result<SemanticModel, Vec<Diagnostic>> {
        if self.diagnostics.is_empty() {
            Ok(self.model)
        } else {
            Err(self.diagnostics)
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct IllegalGlobalInfo {
    pub(super) ty: Ty,
    pub(super) decl_span: Span,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum LinearLeaseState {
    Live,
    DeferredRelease {
        span: Span,
        scope_depth: u32,
    },
    Moved {
        moved_at: Span,
        reason: &'static str,
    },
    Uninit,
}

#[derive(Clone, Debug)]
pub(super) struct LocalVar {
    pub(super) name: String,
    pub(super) ty: Ty,
    pub(super) decl_depth: u32,
    pub(super) prov: Provenance,
    pub(super) field_prov: BTreeMap<String, Provenance>,
    pub(super) endpoint_alias: Option<String>,
    pub(super) field_endpoints: BTreeMap<String, String>,
    pub(super) is_isr_state: bool,
    pub(super) lease_state: Option<LinearLeaseState>,
    pub(super) lease_gen_key: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct ExprVal {
    pub(super) ty: Ty,
    pub(super) error_spec: ResolvedErrorSpec,
    pub(super) prov: Provenance,
    pub(super) endpoint_name: Option<String>,
    pub(super) field_endpoints: BTreeMap<String, String>,
    pub(super) span: Span,
}

impl ExprVal {
    pub(super) fn error(span: Span) -> Self {
        Self {
            ty: Ty::Error,
            error_spec: ResolvedErrorSpec::None,
            prov: Provenance::pure_static(),
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }

    pub(super) fn scalar(ty: Ty, span: Span) -> Self {
        Self {
            ty,
            error_spec: ResolvedErrorSpec::None,
            prov: Provenance::pure_static(),
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum EndpointSpec {
    Static(String),
    Param {
        index: usize,
        suffix: Option<String>,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum ByteRingOpRole {
    Producer,
    Consumer,
    Both,
}

#[derive(Clone, Debug)]
pub(super) struct RecordedByteRingOp {
    pub(super) caller: String,
    pub(super) role: ByteRingOpRole,
    pub(super) method: String,
    pub(super) target: EndpointSpec,
    pub(super) span: Span,
    pub(super) file: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct RecordedWatchWrite {
    pub(super) caller: String,
    pub(super) target: EndpointSpec,
    pub(super) span: Span,
    pub(super) file: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) enum ActiveBodyKind {
    Fn {
        ret_ty: Ty,
        error_spec: ResolvedErrorSpec,
    },
    Task,
    Isr,
}

#[derive(Clone, Debug)]
pub(super) struct BootArgBinding {
    pub(super) static_endpoint: Option<String>,
    pub(super) endpoint_from_param: Option<(usize, Option<String>)>,
    pub(super) field_endpoints: BTreeMap<String, EndpointSpec>,
    pub(super) ref_param_origins: BTreeSet<usize>,
}

#[derive(Clone, Debug)]
pub(super) struct BootCallArg {
    pub(super) static_endpoint: Option<String>,
    pub(super) endpoint_from_param: Option<(usize, Option<String>)>,
    pub(super) field_endpoints: BTreeMap<String, EndpointSpec>,
    pub(super) has_non_static_local_ref: bool,
    pub(super) has_isr_state_ref: bool,
    pub(super) ref_param_origins: BTreeSet<usize>,
    pub(super) span: Span,
}

#[derive(Clone, Debug)]
pub(super) enum BootStep {
    DirectSpawn {
        task_name: String,
        priority: Priority,
        core: Option<u32>,
        conditional: bool,
        arg_bindings: Vec<BootArgBinding>,
        span: Span,
    },
    HelperCall {
        callee: String,
        conditional: bool,
        args: Vec<BootCallArg>,
    },
}

#[derive(Clone, Debug)]
pub(super) enum BootAttachArgSpec {
    Static(IrqBindingArg),
    Param {
        index: usize,
        field_chain: Vec<(String, String)>,
    },
}

#[derive(Clone, Debug)]
pub(super) struct BootAttachCallArg {
    pub(super) spec: Option<BootAttachArgSpec>,
}

#[derive(Clone, Debug)]
pub(super) struct PendingIrqAttach {
    pub(super) irq_name: String,
    pub(super) irq_num: u32,
    pub(super) is_shared: bool,
    pub(super) isr_name: String,
    pub(super) args: Vec<(Option<BootAttachArgSpec>, Span, String)>,
    pub(super) span: Span,
    pub(super) file: Option<String>,
    pub(super) in_caller: String,
    pub(super) invalid_ctx: bool,
}

#[derive(Clone, Debug)]
pub(super) enum IrqBootNode {
    Attach(PendingIrqAttach),
    Call {
        callee: String,
        args: Vec<BootAttachCallArg>,
    },
    BranchGroup {
        arms: Vec<(bool, Vec<IrqBootNode>)>,
    },
}

#[derive(Clone, Debug)]
pub(super) struct FnPtrValueSite {
    pub(super) fn_name: String,
    pub(super) span: Span,
    pub(super) prog_idx: usize,
    pub(super) file: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct RecordedCallArgs {
    pub(super) caller: String,
    pub(super) callee: String,
    pub(super) args: Vec<BootCallArg>,
    pub(super) in_main_post_boot: bool,
    pub(super) in_main_after_spawn: bool,
    pub(super) in_main_after_irq_attach: bool,
}

pub(crate) struct Analyzer {
    pub(super) tick_period_us: u64,
    pub(super) tick_configured_explicitly: bool,
    pub(super) systick_hz: Option<u64>,
    pub(super) target_ptr_bits: u32,
    pub(super) target_name: String,
    pub(super) display_names: BTreeMap<String, String>,
    pub(super) program_extern_c_scopes: Vec<BTreeSet<String>>,
    pub(super) rewritten_pub_extern_sites: BTreeSet<(usize, u32, u32)>,
    pub(super) non_root_extern_c_names: BTreeSet<String>,
    pub(super) non_root_prog_count: usize,
    pub(super) structs: BTreeMap<String, StructInfo>,
    pub(super) struct_files: BTreeMap<String, Option<String>>,
    pub(super) enums: BTreeMap<String, EnumInfo>,
    pub(super) enum_prog_idx: BTreeMap<String, Option<usize>>,
    /// Maps unqualified enum variant name -> list of enums defining that variant.
    pub(super) variant_to_enums: BTreeMap<String, Vec<String>>,
    pub(super) consts: BTreeMap<String, ConstInfo>,
    pub(super) raw_const_exprs: BTreeMap<String, (Expr, Ty, usize)>,
    pub(super) primitives: BTreeMap<String, PrimitiveInfo>,
    pub(super) subsystems: BTreeMap<String, SubsystemInfo>,
    pub(super) mmio_layout_names: BTreeSet<String>,
    pub(super) mmio_layouts: BTreeMap<String, MmioLayoutInfo>,
    pub(super) mmio_instances: BTreeMap<String, MmioInstanceInfo>,
    pub(super) irqs: BTreeMap<String, IrqInfo>,
    pub(super) irq_bindings: Vec<IrqBindingInfo>,
    pub(super) isr_states: Vec<IsrStateInfo>,
    pub(super) fn_irq_boot_nodes: BTreeMap<String, Vec<IrqBootNode>>,
    pub(super) illegal_globals: BTreeMap<String, IllegalGlobalInfo>,
    pub(super) functions: BTreeMap<String, FunctionInfo>,
    pub(super) fn_decl_order: Vec<String>,
    pub(super) fn_boot_steps: BTreeMap<String, Vec<BootStep>>,
    pub(super) recorded_call_args: Vec<RecordedCallArgs>,
    pub(super) recorded_byte_ring_ops: Vec<RecordedByteRingOp>,
    pub(super) recorded_watch_writes: Vec<RecordedWatchWrite>,
    pub(super) fn_ptr_value_sites: Vec<FnPtrValueSite>,
    pub(super) extern_callback_arg_spans: BTreeSet<(usize, u32, u32)>,
    pub(super) spawned_task_field_endpoints: BTreeMap<usize, Vec<BTreeMap<String, String>>>,
    pub(super) prim_escaping_ref_params: BTreeMap<String, BTreeSet<usize>>,
    pub(super) tasks: BTreeMap<String, TaskInfo>,
    pub(super) isrs: BTreeMap<String, IsrInfo>,
    pub(super) spawned_tasks: Vec<SpawnedTaskInstance>,
    pub(super) soft_yielding_in_coro: BTreeSet<String>,
    pub(super) soft_yielding_in_noyield: BTreeSet<String>,
    pub(super) dual_specialized_fns: BTreeSet<String>,
    pub(super) main_supervisor_priority: Option<Priority>,
    pub(super) call_graph: CallGraph,
    pub(super) current_file: Option<String>,
    pub(super) current_prog_idx: usize,
    pub(super) current_instantiation_note: Option<String>,
    pub(super) mono_instantiation_notes: BTreeMap<String, String>,
    pub(super) defaulted_generic_calls: Vec<crate::monomorphize::DefaultedGenericCall>,
    pub(super) f32_overflow_reported_spans: BTreeSet<(u32, u32)>,
    pub(super) node_files: BTreeMap<String, String>,
    pub(super) diagnostics: Vec<Diagnostic>,
}

mod analyzer;
mod body_exprs;
mod body_stmts;
mod format;
mod helpers;

pub(in crate::checker) use helpers::*;

pub(super) struct BodyChecker<'a> {
    pub(super) sema: &'a mut Analyzer,
    pub(super) caller_key: String,
    pub(super) caller_kind: CallableKind,
    pub(super) body_kind: ActiveBodyKind,
    pub(super) prog_idx: usize,
    /// Scope stack: `scopes.len()` equals lexical scope depth (`1` = fn/task body, `2..N` = nested blocks).
    pub(super) scopes: Vec<BTreeMap<String, LocalVar>>,
    pub(super) leases: LeaseEnv,
    pub(super) loop_depth: u32,
    pub(super) noyield_depth: u32,
    pub(super) critical_depth: u32,
    pub(super) defer_depth: u32,
    pub(super) branch_depth: u32,
    pub(super) after_diverge: bool,
    pub(super) main_supervisor_started: bool,
    pub(super) in_main_supervisor_loop: bool,
    pub(super) has_spawned: bool,
    pub(super) has_attached_irq: bool,
    pub(super) param_names: Vec<String>,
    pub(super) isr_state_names: BTreeSet<String>,
    pub(super) boot_steps: Vec<BootStep>,
    pub(super) boot_block_stack: Vec<Vec<IrqBootNode>>,
    pub(super) return_param_origins: BTreeSet<usize>,
    pub(super) return_endpoint: Option<String>,
    pub(super) return_field_endpoints: BTreeMap<String, String>,
    pub(super) recorded_boosts: Vec<PriorityBoostSite>,
    pub(super) proven_infallible_spans: BTreeSet<(u32, u32)>,
    pub(super) last_struct_lit_fields: BTreeMap<(u32, u32), BTreeMap<String, Provenance>>,
    pub(super) mmio_array_expr_spans: BTreeMap<(u32, u32), (String, String, usize)>,
    /// Tracks lexical scope depths at loop entry and outer variables carrying DMA leases read inside the loop.
    pub(super) loop_entry_depths: Vec<u32>,
    pub(super) loop_outer_reads: Vec<Vec<(String, LeaseEnv, Provenance, Span)>>,
    /// Tracks loop-entry snapshots for outer `Lease<T>` move checking inside loops.
    pub(super) loop_lease_frames: Vec<(u32, Vec<BTreeMap<String, LocalVar>>)>,
    /// When inside a `defer` statement, records the lexical scope depth owning that `defer`.
    pub(super) active_defer_owner_depth: Option<u32>,
    pub(super) next_lease_var_id: u64,
    pub(super) expr_tys: std::collections::HashMap<usize, Ty>,
}
