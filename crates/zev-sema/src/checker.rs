//! Multi-pass semantic analyzer and static verifier for the Zev Language-OS.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use zev_ast::{
    ArrayLen, AssignOp, Attr, AttrKind, BinaryOp, Block, CallArg, CatchClause, ConstDecl,
    DurationLit, EnumDecl, ErrorSpec, Expr, ExprKind, FnDecl, FormatPart, FormatSpec,
    GlobalVarDecl, Ident, IrqDecl, IsrDecl, Item, ItemKind, Literal, MmioAccess,
    MmioDecl, MmioDeclKind, OsPrimitiveDecl, OsPrimitiveType, Param, PrimType, Priority, Program,
    SpawnStmt, Span, Stmt, StmtKind, StructDecl, SwitchPat, TaskDecl, TypeExpr, TypeKind,
    UnaryOp,
};

use crate::call_graph::{CallContext, CallEdge, CallGraph, CallableKind, ExternCallbackSite};
use crate::diag::{Diagnostic, DiagnosticKind};
use crate::model::{
    ConstInfo, DEFAULT_TICK_PERIOD_US, FnEffect, FunctionInfo, IrqBindingArg, IrqBindingInfo,
    IrqInfo, IsrInfo, IsrStateInfo, MmioInstanceInfo, MmioLayoutInfo, MmioRegInfo,
    MonomorphizedInstance, MonomorphizedKind, ParamInfo, PrimitiveInfo, PriorityBoostSite,
    SemanticModel, SpawnedTaskInstance, SubsystemInfo, TaskInfo,
};
use crate::modules::LinkedModulesOutput;
use crate::provenance::{LeaseEnv, Provenance};
use crate::suggest::{
    advise_primitive_method, closest_match, suggest_foreign_type_name,
};
use crate::types::{
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
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            default_tick_period_us: DEFAULT_TICK_PERIOD_US,
            include_dirs: Vec::new(),
            hal_dir: None,
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
struct IllegalGlobalInfo {
    ty: Ty,
    decl_span: Span,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LinearLeaseState {
    Live,
    DeferredRelease { span: Span, scope_depth: u32 },
    Moved { moved_at: Span, reason: &'static str },
    Uninit,
}

#[derive(Clone, Debug)]
struct LocalVar {
    name: String,
    ty: Ty,
    decl_depth: u32,
    prov: Provenance,
    field_prov: BTreeMap<String, Provenance>,
    endpoint_alias: Option<String>,
    field_endpoints: BTreeMap<String, String>,
    is_isr_state: bool,
    lease_state: Option<LinearLeaseState>,
    lease_gen_key: Option<String>,
}

#[derive(Clone, Debug)]
struct ExprVal {
    ty: Ty,
    error_spec: ResolvedErrorSpec,
    prov: Provenance,
    endpoint_name: Option<String>,
    field_endpoints: BTreeMap<String, String>,
    span: Span,
}

impl ExprVal {
    fn error(span: Span) -> Self {
        Self {
            ty: Ty::Error,
            error_spec: ResolvedErrorSpec::None,
            prov: Provenance::pure_static(),
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }

    fn scalar(ty: Ty, span: Span) -> Self {
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
enum EndpointSpec {
    Static(String),
    Param {
        index: usize,
        suffix: Option<String>,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ByteRingOpRole {
    Producer,
    Consumer,
    Both,
}

#[derive(Clone, Debug)]
struct RecordedByteRingOp {
    caller: String,
    role: ByteRingOpRole,
    method: String,
    target: EndpointSpec,
    span: Span,
    file: Option<String>,
}

#[derive(Clone, Debug)]
enum ActiveBodyKind {
    Fn {
        ret_ty: Ty,
        error_spec: ResolvedErrorSpec,
    },
    Task,
    Isr,
}

#[derive(Clone, Debug)]
struct BootArgBinding {
    static_endpoint: Option<String>,
    endpoint_from_param: Option<(usize, Option<String>)>,
    field_endpoints: BTreeMap<String, EndpointSpec>,
    ref_param_origins: BTreeSet<usize>,
}

#[derive(Clone, Debug)]
struct BootCallArg {
    static_endpoint: Option<String>,
    endpoint_from_param: Option<(usize, Option<String>)>,
    field_endpoints: BTreeMap<String, EndpointSpec>,
    has_non_static_local_ref: bool,
    has_isr_state_ref: bool,
    ref_param_origins: BTreeSet<usize>,
    span: Span,
}

#[derive(Clone, Debug)]
enum BootStep {
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
enum BootAttachArgSpec {
    Static(IrqBindingArg),
    Param {
        index: usize,
        field_chain: Vec<(String, String)>,
    },
}

#[derive(Clone, Debug)]
struct BootAttachCallArg {
    spec: Option<BootAttachArgSpec>,
}

#[derive(Clone, Debug)]
struct PendingIrqAttach {
    irq_name: String,
    irq_num: u32,
    is_shared: bool,
    isr_name: String,
    args: Vec<(Option<BootAttachArgSpec>, Span, String)>,
    span: Span,
    file: Option<String>,
    in_caller: String,
    invalid_ctx: bool,
}

#[derive(Clone, Debug)]
enum IrqBootNode {
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
struct RecordedCallArgs {
    caller: String,
    callee: String,
    args: Vec<BootCallArg>,
    in_main_post_boot: bool,
}

pub(crate) struct Analyzer {
    tick_period_us: u64,
    tick_configured_explicitly: bool,
    systick_hz: Option<u64>,
    display_names: BTreeMap<String, String>,
    program_extern_c_scopes: Vec<BTreeSet<String>>,
    rewritten_pub_extern_sites: BTreeSet<(usize, u32, u32)>,
    non_root_extern_c_names: BTreeSet<String>,
    non_root_prog_count: usize,
    structs: BTreeMap<String, StructInfo>,
    struct_files: BTreeMap<String, Option<String>>,
    enums: BTreeMap<String, EnumInfo>,
    enum_prog_idx: BTreeMap<String, Option<usize>>,
    /// Maps unqualified enum variant name -> list of enums defining that variant.
    variant_to_enums: BTreeMap<String, Vec<String>>,
    consts: BTreeMap<String, ConstInfo>,
    raw_const_exprs: BTreeMap<String, (Expr, Ty, usize)>,
    primitives: BTreeMap<String, PrimitiveInfo>,
    subsystems: BTreeMap<String, SubsystemInfo>,
    mmio_layout_names: BTreeSet<String>,
    mmio_layouts: BTreeMap<String, MmioLayoutInfo>,
    mmio_instances: BTreeMap<String, MmioInstanceInfo>,
    irqs: BTreeMap<String, IrqInfo>,
    irq_bindings: Vec<IrqBindingInfo>,
    isr_states: Vec<IsrStateInfo>,
    fn_irq_boot_nodes: BTreeMap<String, Vec<IrqBootNode>>,
    illegal_globals: BTreeMap<String, IllegalGlobalInfo>,
    functions: BTreeMap<String, FunctionInfo>,
    fn_decl_order: Vec<String>,
    fn_boot_steps: BTreeMap<String, Vec<BootStep>>,
    recorded_call_args: Vec<RecordedCallArgs>,
    recorded_byte_ring_ops: Vec<RecordedByteRingOp>,
    spawned_task_field_endpoints: BTreeMap<usize, Vec<BTreeMap<String, String>>>,
    prim_escaping_ref_params: BTreeMap<String, BTreeSet<usize>>,
    tasks: BTreeMap<String, TaskInfo>,
    isrs: BTreeMap<String, IsrInfo>,
    spawned_tasks: Vec<SpawnedTaskInstance>,
    soft_yielding_in_coro: BTreeSet<String>,
    soft_yielding_in_noyield: BTreeSet<String>,
    dual_specialized_fns: BTreeSet<String>,
    main_supervisor_priority: Option<Priority>,
    call_graph: CallGraph,
    current_file: Option<String>,
    current_prog_idx: usize,
    current_instantiation_note: Option<String>,
    mono_instantiation_notes: BTreeMap<String, String>,
    f32_overflow_reported_spans: BTreeSet<(u32, u32)>,
    node_files: BTreeMap<String, String>,
    diagnostics: Vec<Diagnostic>,
}

impl Analyzer {
    pub fn new(opts: &AnalyzeOptions) -> Self {
        Self {
            tick_period_us: opts.default_tick_period_us.max(1),
            tick_configured_explicitly: false,
            systick_hz: None,
            display_names: BTreeMap::new(),
            program_extern_c_scopes: Vec::new(),
            rewritten_pub_extern_sites: BTreeSet::new(),
            non_root_extern_c_names: BTreeSet::new(),
            non_root_prog_count: 0,
            structs: BTreeMap::new(),
            struct_files: BTreeMap::new(),
            enums: BTreeMap::new(),
            enum_prog_idx: BTreeMap::new(),
            variant_to_enums: BTreeMap::new(),
            consts: BTreeMap::new(),
            raw_const_exprs: BTreeMap::new(),
            primitives: BTreeMap::new(),
            subsystems: BTreeMap::new(),
            mmio_layout_names: BTreeSet::new(),
            mmio_layouts: BTreeMap::new(),
            mmio_instances: BTreeMap::new(),
            irqs: BTreeMap::new(),
            irq_bindings: Vec::new(),
            isr_states: Vec::new(),
            fn_irq_boot_nodes: BTreeMap::new(),
            illegal_globals: BTreeMap::new(),
            functions: BTreeMap::new(),
            fn_decl_order: Vec::new(),
            fn_boot_steps: BTreeMap::new(),
            recorded_call_args: Vec::new(),
            recorded_byte_ring_ops: Vec::new(),
            spawned_task_field_endpoints: BTreeMap::new(),
            prim_escaping_ref_params: BTreeMap::new(),
            tasks: BTreeMap::new(),
            isrs: BTreeMap::new(),
            spawned_tasks: Vec::new(),
            soft_yielding_in_coro: BTreeSet::new(),
            soft_yielding_in_noyield: BTreeSet::new(),
            dual_specialized_fns: BTreeSet::new(),
            main_supervisor_priority: None,
            call_graph: CallGraph::new(),
            current_file: None,
            current_prog_idx: 0,
            current_instantiation_note: None,
            mono_instantiation_notes: BTreeMap::new(),
            f32_overflow_reported_spans: BTreeSet::new(),
            node_files: BTreeMap::new(),
            diagnostics: Vec::new(),
        }
    }

    pub(crate) fn with_linked_context(mut self, linked: &LinkedModulesOutput) -> Self {
        self.display_names = linked.display_names.clone();
        self.program_extern_c_scopes = linked.program_extern_c_scopes.clone();
        self.rewritten_pub_extern_sites = linked.rewritten_pub_extern_sites.clone();
        self.non_root_extern_c_names = linked.non_root_extern_c_names.clone();
        self.non_root_prog_count = linked.non_root_program_count;
        self
    }

    fn emit(&mut self, mut diag: Diagnostic) {
        if diag.file.is_none() {
            diag.file = self.current_file.clone();
        }
        if diag.note.is_none() {
            diag.note = Some(diag.kind.doc_hint().to_string());
        }
        if let Some(origin) = &self.current_instantiation_note {
            if let Some(existing) = &mut diag.note {
                if !existing.contains("instantiated from ") {
                    existing.push_str(&format!(" (instantiated from `{}`)", origin));
                }
            } else {
                diag.note = Some(format!("instantiated from `{}`", origin));
            }
        }
        self.diagnostics.push(diag);
    }

    pub fn run(mut self, programs: &[(&Program, Option<&str>)]) -> AnalysisOutput {
        // Merge items while tracking optional file names and program index per item.
        let mut all_items: Vec<(&Item, Option<String>, usize)> = Vec::new();
        for (prog_idx, (prog, file)) in programs.iter().enumerate() {
            for item in &prog.items {
                all_items.push((item, file.map(str::to_string), prog_idx));
            }
        }

        // Pass 1a: Imports, struct names, enum declarations, and MMIO layout type names.
        let mut mono_instances_from_attrs: BTreeSet<MonomorphizedInstance> = BTreeSet::new();
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            match &item.kind {
                ItemKind::Import(imp) => {
                    let dot = imp
                        .path
                        .iter()
                        .map(|id| id.name.as_str())
                        .collect::<Vec<_>>()
                        .join(".");
                    self.emit(Diagnostic::new(
                        DiagnosticKind::ModuleError,
                        imp.span,
                        format!("unresolved module import `{dot}`"),
                    ));
                }
                ItemKind::Struct(st) => {
                    if !st.generic_params.is_empty() {
                        continue;
                    }
                    if let Some(sig) = extract_monomorphized_attr(&item.attrs) {
                        mono_instances_from_attrs.insert(MonomorphizedInstance {
                            kind: MonomorphizedKind::Struct,
                            signature: sig,
                            mangled_name: st.name.name.clone(),
                        });
                    }
                    if let Some(orig) = extract_instantiated_from_attr(&item.attrs) {
                        self.mono_instantiation_notes
                            .insert(st.name.name.clone(), orig);
                    }
                    self.struct_files
                        .insert(st.name.name.clone(), file.clone());
                    self.register_struct_header(st);
                }
                ItemKind::Enum(en) => self.collect_enum(en),
                ItemKind::Mmio(mm)
                    if matches!(
                        mm.kind,
                        MmioDeclKind::Layout { .. } | MmioDeclKind::Inline { .. }
                    ) =>
                {
                    self.register_mmio_layout_name(mm);
                }
                ItemKind::Fn(f) if f.generic_params.is_empty() => {
                    if let Some(sig) = extract_monomorphized_attr(&item.attrs) {
                        mono_instances_from_attrs.insert(MonomorphizedInstance {
                            kind: MonomorphizedKind::Fn,
                            signature: sig,
                            mangled_name: f.name.name.clone(),
                        });
                    }
                    if let Some(orig) = extract_instantiated_from_attr(&item.attrs) {
                        self.mono_instantiation_notes
                            .insert(f.name.name.clone(), orig);
                    }
                }
                _ => {}
            }
        }

        // Pre-register built-in `enum FmtErr { Truncated }` if not user-declared.
        if !self.enums.contains_key("FmtErr")
            && !self.structs.contains_key("FmtErr")
            && !self.mmio_layout_names.contains("FmtErr")
        {
            self.enums.insert(
                "FmtErr".to_string(),
                EnumInfo {
                    name: "FmtErr".to_string(),
                    variants: vec!["Truncated".to_string()],
                    span: Span::dummy(),
                },
            );
            self.enum_prog_idx.insert("FmtErr".to_string(), None);
            if !self.variant_to_enums.contains_key("Truncated") {
                self.variant_to_enums
                    .insert("Truncated".to_string(), vec!["FmtErr".to_string()]);
            }
        }

        // Pass 1b: Resolve struct fields and verify packed struct & recursive value invariants.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            if let ItemKind::Struct(st) = &item.kind {
                if st.generic_params.is_empty() {
                    self.current_instantiation_note =
                        self.mono_instantiation_notes.get(&st.name.name).cloned();
                    self.resolve_struct_fields(st);
                    self.current_instantiation_note = None;
                }
            }
        }
        self.current_file = None;
        self.check_recursive_struct_values();
        self.finalize_subsystem_structs();

        // Pass 1c: Pre-scan for `sys.use_systick` / `sys.set_tick` so custom tick period is known
        // before validating duration literals anywhere in the program.
        for (item, _, _) in &all_items {
            if let ItemKind::Fn(f) = &item.kind {
                if f.generic_params.is_empty() {
                    if let Some(body) = &f.body {
                        self.prescan_tick_config(body);
                    }
                }
            }
        }

        // Pass 1d: Collect and evaluate `const` declarations first so `mmio` offsets/base addresses,
        // `irq` vector numbers, and `[CAP]T` array sizes can reference any top-level `const`.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            if let ItemKind::Const(c) = &item.kind {
                self.collect_const(c);
            }
        }
        self.evaluate_all_const_int_values();

        // Collect `mmio`, `irq`, OS primitives, illegal mutable globals, and callable signatures.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            match &item.kind {
                ItemKind::Mmio(mm) => self.collect_mmio(mm),
                ItemKind::Irq(iq) => self.collect_irq(iq),
                ItemKind::OsPrimitive(p) => self.collect_os_primitive(&item.attrs, p),
                ItemKind::GlobalVar(g) => self.collect_global_var(&item.attrs, g),
                ItemKind::Fn(f) if f.generic_params.is_empty() => {
                    self.current_instantiation_note =
                        self.mono_instantiation_notes.get(&f.name.name).cloned();
                    self.collect_fn_sig(&item.attrs, f);
                    self.current_instantiation_note = None;
                }
                ItemKind::Task(t) => self.collect_task_sig(&item.attrs, t),
                ItemKind::Isr(i) => self.collect_isr_sig(&item.attrs, i),
                _ => {}
            }
        }

        // Validate every MMIO instance's [base_addr .. base_addr + byte_size] fits within 32-bit address space
        // after all layouts have been collected (handles forward-declared layouts).
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            if let ItemKind::Mmio(mm) = &item.kind {
                if matches!(
                    mm.kind,
                    MmioDeclKind::Instance { .. } | MmioDeclKind::Inline { .. }
                ) {
                    if let Some(inst) = self.mmio_instances.get(&mm.name.name) {
                        if let Some(layout_info) = self.mmio_layouts.get(&inst.layout) {
                            let size_u32 =
                                u32::try_from(layout_info.byte_size).unwrap_or(u32::MAX);
                            if inst.base_addr.checked_add(size_u32).is_none() {
                                self.emit(Diagnostic::new(
                                    DiagnosticKind::MmioViolation,
                                    mm.span,
                                    format!(
                                        "MMIO instance '{}' at base 0x{:08X} with layout '{}' ({} bytes) overflows 32-bit address space",
                                        inst.name, inst.base_addr, inst.layout, layout_info.byte_size
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Pass 2a: Analyze non-`main` helper functions first so interprocedural `return_param_origins`
        // summaries are ready before `main` and `task` bodies call them.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            if let ItemKind::Fn(f) = &item.kind {
                if f.generic_params.is_empty() && f.name.name != "main" {
                    self.check_fn_body(&item.attrs, f);
                }
            }
        }
        // Pass 2b: Analyze `main()`, `task` blueprints, and `isr` handlers.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            match &item.kind {
                ItemKind::Fn(f) if f.generic_params.is_empty() && f.name.name == "main" => {
                    self.check_fn_body(&item.attrs, f);
                }
                ItemKind::Task(t) => self.check_task_body(&item.attrs, t),
                ItemKind::Isr(i) => self.check_isr_body(&item.attrs, i),
                _ => {}
            }
        }
        self.current_file = None;

        // Pass 3: Whole-program transitive yield inference, `noyield`/`critical`/`isr` call checks,
        // `extern "C"` callback checks, boot-tree spawn & IRQ attach unrolling, and cycle detection.
        self.run_whole_program_verification();

        let main_can_yield = self
            .functions
            .get("main")
            .is_some_and(|m| m.can_yield);

        let mut monomorphized_instances: Vec<MonomorphizedInstance> =
            mono_instances_from_attrs.into_iter().collect();
        monomorphized_instances.sort();

        let model = SemanticModel {
            tick_period_us: self.tick_period_us,
            systick_hz: self.systick_hz,
            structs: self.structs,
            enums: self.enums,
            consts: self.consts,
            primitives: self.primitives,
            subsystems: self.subsystems,
            functions: self.functions,
            tasks: self.tasks,
            isrs: self.isrs,
            mmio_layouts: self.mmio_layouts,
            mmio_instances: self.mmio_instances,
            irqs: self.irqs,
            irq_bindings: self.irq_bindings,
            isr_states: self.isr_states,
            display_names: self.display_names,
            spawned_tasks: self.spawned_tasks,
            monomorphized_instances,
            soft_yielding_in_coro: self.soft_yielding_in_coro,
            soft_yielding_in_noyield: self.soft_yielding_in_noyield,
            dual_specialized_fns: self.dual_specialized_fns,
            main_supervisor_priority: self.main_supervisor_priority,
            main_can_yield,
            call_graph: self.call_graph,
        };

        AnalysisOutput {
            model,
            diagnostics: self.diagnostics,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass 1 Helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn register_mmio_layout_name(&mut self, mm: &MmioDecl) {
        if self.structs.contains_key(&mm.name.name)
            || self.enums.contains_key(&mm.name.name)
            || !self.mmio_layout_names.insert(mm.name.name.clone())
        {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                mm.name.span,
                format!("duplicate type definition '{}'", mm.name.name),
            ));
        }
    }

    fn register_struct_header(&mut self, st: &StructDecl) {
        if self.structs.contains_key(&st.name.name)
            || self.enums.contains_key(&st.name.name)
            || self.mmio_layout_names.contains(&st.name.name)
        {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                st.name.span,
                format!("duplicate type definition '{}'", st.name.name),
            ));
            return;
        }
        self.structs.insert(
            st.name.name.clone(),
            StructInfo {
                name: st.name.name.clone(),
                is_packed: st.is_packed,
                is_subsystem: false,
                fields: Vec::new(),
                span: st.span,
            },
        );
    }

    fn collect_enum(&mut self, en: &EnumDecl) {
        if self.enums.contains_key(&en.name.name)
            || self.structs.contains_key(&en.name.name)
            || self.mmio_layout_names.contains(&en.name.name)
        {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                en.name.span,
                format!("duplicate type definition '{}'", en.name.name),
            ));
            return;
        }
        let mut seen = BTreeSet::new();
        let mut variants = Vec::new();
        for v in &en.variants {
            if !seen.insert(v.name.name.clone()) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::DuplicateDefinition,
                    v.name.span,
                    format!(
                        "duplicate variant '{}' in enum '{}'",
                        v.name.name, en.name.name
                    ),
                ));
                continue;
            }
            variants.push(v.name.name.clone());
            self.variant_to_enums
                .entry(v.name.name.clone())
                .or_default()
                .push(en.name.name.clone());
        }
        self.enums.insert(
            en.name.name.clone(),
            EnumInfo {
                name: en.name.name.clone(),
                variants,
                span: en.span,
            },
        );
        let scope_idx = if self.current_prog_idx < self.non_root_prog_count {
            Some(self.current_prog_idx)
        } else {
            None
        };
        self.enum_prog_idx.insert(en.name.name.clone(), scope_idx);
    }

    fn visible_enums_for_variant(&self, prog_idx: usize, variant: &str) -> Vec<String> {
        let Some(all) = self.variant_to_enums.get(variant) else {
            return Vec::new();
        };
        all.iter()
            .filter(|en| match self.enum_prog_idx.get(*en).copied().flatten() {
                Some(decl_idx) => decl_idx == prog_idx,
                None => prog_idx >= self.non_root_prog_count,
            })
            .cloned()
            .collect()
    }

    fn check_assignable_or_emit(
        &mut self,
        actual: &Ty,
        expected: &Ty,
        span: Span,
        default_kind: DiagnosticKind,
        default_msg: impl FnOnce() -> String,
    ) -> bool {
        if actual.is_assignable_to(expected) {
            return true;
        }
        if matches!(expected, Ty::Ptr(inner) if inner.is_void()) && actual.contains_mmio() {
            self.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                span,
                format!(
                    "cannot coerce MMIO handle '{}' to raw '*void' pointer",
                    actual
                ),
            ));
        } else {
            self.emit(Diagnostic::new(default_kind, span, default_msg()));
        }
        false
    }

    fn resolve_struct_fields(&mut self, st: &StructDecl) {
        let mut seen = BTreeSet::new();
        let mut fields = Vec::new();
        for f in &st.fields {
            if !seen.insert(f.name.name.clone()) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::DuplicateDefinition,
                    f.name.span,
                    format!(
                        "duplicate field '{}' in struct '{}'",
                        f.name.name, st.name.name
                    ),
                ));
                continue;
            }
            let ty = self.resolve_type(&f.ty);
            if ty.is_void() {
                self.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    f.ty.span,
                    format!(
                        "field '{}' in struct '{}' cannot have type 'void'",
                        f.name.name, st.name.name
                    ),
                ));
            }
            self.check_type_no_byval_mmio_or_unsized(
                &ty,
                f.ty.span,
                &format!("field '{}' of struct '{}'", f.name.name, st.name.name),
            );
            if st.is_packed && matches!(ty, Ty::Ptr(_) | Ty::Slice(_) | Ty::FnPtr { .. }) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    f.span,
                    format!(
                        "packed struct '{}' must have wire-exact layout and cannot contain pointer or slice field '{}'",
                        st.name.name, f.name.name
                    ),
                ));
            }
            let dma_safe = has_attr_dma_safe(&f.attrs);
            for attr in &f.attrs {
                match &attr.kind {
                    AttrKind::DmaSafe => {
                        if !matches!(
                            ty,
                            Ty::OsPrimitive(
                                ResolvedOsPrimitive::Pool { .. }
                                    | ResolvedOsPrimitive::PingPong { .. }
                            )
                        ) {
                            self.emit(Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                attr.span,
                                format!(
                                    "`@dma_safe` on field '{}' of struct '{}' is only allowed on `pool<T, K>` or `ping_pong<T, N>` fields",
                                    f.name.name, st.name.name
                                ),
                            ));
                        }
                    }
                    other => {
                        self.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            attr.span,
                            format!(
                                "attribute `{:?}` is not valid on field '{}' of struct '{}'",
                                other, f.name.name, st.name.name
                            ),
                        ));
                    }
                }
            }
            self.check_disallowed_lease_type(
                &ty,
                f.ty.span,
                &format!("field '{}' of struct '{}'", f.name.name, st.name.name),
                false,
            );
            fields.push(FieldInfo {
                name: f.name.name.clone(),
                ty,
                dma_safe,
                span: f.span,
            });
        }
        if let Some(info) = self.structs.get_mut(&st.name.name) {
            info.fields = fields;
        }
    }

    fn check_recursive_struct_values(&mut self) {
        let names: Vec<String> = self.structs.keys().cloned().collect();
        for name in names {
            let mut visiting = BTreeSet::new();
            if self.has_value_cycle(&name, &name, &mut visiting) {
                let span = self.structs[&name].span;
                let file = self.struct_files.get(&name).cloned().flatten();
                self.current_instantiation_note =
                    self.mono_instantiation_notes.get(&name).cloned();
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        span,
                        format!(
                            "recursive struct '{}' has infinite size without indirection",
                            name
                        ),
                    )
                    .with_file(file),
                );
                self.current_instantiation_note = None;
            }
        }
    }

    fn has_value_cycle(
        &self,
        target: &str,
        curr: &str,
        visiting: &mut BTreeSet<String>,
    ) -> bool {
        let Some(st) = self.structs.get(curr) else {
            return false;
        };
        for f in &st.fields {
            let mut inner = &f.ty;
            while let Ty::Array { elem, .. } = inner {
                inner = elem;
            }
            if let Ty::Struct(next) = inner {
                if next == target {
                    return true;
                }
                if visiting.insert(next.clone()) && self.has_value_cycle(target, next, visiting) {
                    return true;
                }
            }
        }
        false
    }

    fn finalize_subsystem_structs(&mut self) {
        let names: Vec<String> = self.structs.keys().cloned().collect();
        // Fixed-point propagate `is_subsystem` across nested subsystem structs.
        loop {
            let mut changed = false;
            for name in &names {
                if self.structs[name].is_subsystem {
                    continue;
                }
                let has_subsys_content = self.structs[name].fields.iter().any(|f| {
                    ty_embeds_os_primitive(&f.ty)
                        || ty_contains_byval_subsys(&f.ty, &self.structs).is_some()
                });
                if has_subsys_content {
                    if let Some(info) = self.structs.get_mut(name) {
                        info.is_subsystem = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        for name in &names {
            let Some(st) = self.structs.get(name).cloned() else {
                continue;
            };
            if !st.is_subsystem {
                continue;
            }
            let file = self.struct_files.get(name).cloned().flatten();
            self.current_instantiation_note = self.mono_instantiation_notes.get(name).cloned();
            if st.is_packed {
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        st.span,
                        format!(
                            "packed struct '{}' cannot embed OS primitives or act as a subsystem struct",
                            st.name
                        ),
                    )
                    .with_file(file.clone()),
                );
            }
            for f in &st.fields {
                let valid_field = match &f.ty {
                    Ty::OsPrimitive(os) => !os.is_unsized_view(),
                    Ty::Struct(sub) => self.structs.get(sub).is_some_and(|s| s.is_subsystem),
                    _ => false,
                };
                if !valid_field {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            f.span,
                            format!(
                                "subsystem struct '{}' cannot mix non-primitive field '{}' of type '{}' (all fields must be concrete OS primitives or nested subsystem structs)",
                                st.name, f.name, f.ty
                            ),
                        )
                        .with_file(file.clone())
                        .with_note("move raw scalar/buffer fields into `watch<T>`, `inbox<T, N>`, `pool<T, K>`, `dma_slot<T>`, or `byte_ring<N>` fields"),
                    );
                }
                match &f.ty {
                    Ty::OsPrimitive(ResolvedOsPrimitive::Pool { elem, .. }) => {
                        self.validate_pool_slot_elem(
                            elem,
                            f.span,
                            &format!("pool field '{}.{}'", st.name, f.name),
                        );
                    }
                    Ty::OsPrimitive(ResolvedOsPrimitive::DmaSlot { elem }) => {
                        self.validate_pool_slot_elem(
                            elem,
                            f.span,
                            &format!("dma_slot field '{}.{}'", st.name, f.name),
                        );
                    }
                    _ => {}
                }
            }
            self.current_instantiation_note = None;
        }
    }

    fn prescan_tick_config(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.prescan_tick_stmt(stmt);
        }
    }

    fn prescan_tick_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Expr(expr) => self.prescan_tick_expr(expr),
            StmtKind::Block(b) | StmtKind::NoYield { body: b } | StmtKind::Critical { body: b } => {
                self.prescan_tick_config(b);
            }
            StmtKind::Defer(inner) => self.prescan_tick_stmt(inner),
            StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                self.prescan_tick_config(then_block);
                if let Some(eb) = else_block {
                    self.prescan_tick_config(eb);
                }
            }
            _ => {}
        }
    }

    fn prescan_tick_expr(&mut self, expr: &Expr) {
        if let ExprKind::MethodCall {
            receiver,
            method,
            args,
        } = &expr.kind
        {
            if matches!(&receiver.kind, ExprKind::Ident(id) if id.name == "sys") {
                if method.name == "use_systick" {
                    if let Some(first) = args.first() {
                        if let ExprKind::Literal(Literal::Int(hz)) = &first.value.kind {
                            self.systick_hz = Some(*hz);
                        }
                    }
                    if let Some(second) = args.get(1) {
                        if let ExprKind::Literal(Literal::Duration(dur)) = &second.value.kind {
                            let us = dur.as_micros();
                            if us > 0 {
                                self.tick_period_us = us;
                                self.tick_configured_explicitly = true;
                            }
                        }
                    }
                } else if method.name == "set_tick" || method.name == "set_tick_period" {
                    if let Some(first) = args.first() {
                        if let ExprKind::Literal(Literal::Duration(dur)) = &first.value.kind {
                            let us = dur.as_micros();
                            if us > 0 {
                                self.tick_period_us = us;
                                self.tick_configured_explicitly = true;
                            }
                        }
                    }
                } else if method.name == "set_tick_us" {
                    if let Some(first) = args.first() {
                        if let ExprKind::Literal(Literal::Int(us)) = &first.value.kind {
                            if *us > 0 {
                                self.tick_period_us = *us;
                                self.tick_configured_explicitly = true;
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_top_level_name_unique(&mut self, name: &Ident) -> bool {
        if self.consts.contains_key(&name.name)
            || self.primitives.contains_key(&name.name)
            || self.subsystems.contains_key(&name.name)
            || self.mmio_instances.contains_key(&name.name)
            || self.irqs.contains_key(&name.name)
            || self.illegal_globals.contains_key(&name.name)
            || self.functions.contains_key(&name.name)
            || self.tasks.contains_key(&name.name)
        {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                name.span,
                format!("duplicate top-level symbol '{}'", name.name),
            ));
            return false;
        }
        true
    }

    fn collect_mmio(&mut self, mm: &MmioDecl) {
        match &mm.kind {
            MmioDeclKind::Layout { regs } => {
                self.collect_mmio_layout(&mm.name, regs, mm.span);
            }
            MmioDeclKind::Instance { layout, base_addr } => {
                if !self.check_top_level_name_unique(&mm.name) {
                    return;
                }
                let resolved_layout = self.resolve_type(layout);
                let layout_name = match resolved_layout {
                    Ty::MmioLayout(l) => l,
                    Ty::Error => return,
                    other => {
                        self.emit(Diagnostic::new(
                            DiagnosticKind::MmioViolation,
                            layout.span,
                            format!(
                                "MMIO instance '{}' must use a `mmio` register layout type, found '{}'",
                                mm.name.name, other
                            ),
                        ));
                        return;
                    }
                };
                let base_u32 = self.eval_u32_const(
                    base_addr,
                    DiagnosticKind::MmioViolation,
                    &format!("MMIO instance '{}' base address", mm.name.name),
                );
                self.mmio_instances.insert(
                    mm.name.name.clone(),
                    MmioInstanceInfo {
                        name: mm.name.name.clone(),
                        layout: layout_name,
                        base_addr: base_u32,
                        span: mm.span,
                    },
                );
            }
            MmioDeclKind::Inline { base_addr, regs } => {
                if !self.check_top_level_name_unique(&mm.name) {
                    return;
                }
                let layout_id = Ident::new(format!("{}__Layout", mm.name.name), mm.name.span);
                self.collect_mmio_layout(&layout_id, regs, mm.span);
                let base_u32 = self.eval_u32_const(
                    base_addr,
                    DiagnosticKind::MmioViolation,
                    &format!("MMIO instance '{}' base address", mm.name.name),
                );
                self.mmio_instances.insert(
                    mm.name.name.clone(),
                    MmioInstanceInfo {
                        name: mm.name.name.clone(),
                        layout: layout_id.name,
                        base_addr: base_u32,
                        span: mm.span,
                    },
                );
            }
        }
    }

    fn collect_mmio_layout(
        &mut self,
        name: &Ident,
        regs: &[zev_ast::MmioRegDecl],
        span: Span,
    ) {
        if self.mmio_layouts.contains_key(&name.name) {
            return;
        }
        let mut seen_regs = BTreeSet::new();
        let mut resolved_regs = Vec::new();
        let mut ranges: Vec<(u64, u64, String)> = Vec::new();
        let mut max_end: u64 = 0;

        for reg in regs {
            if !seen_regs.insert(reg.name.name.clone()) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.name.span,
                    format!(
                        "duplicate MMIO register '{}' in layout '{}'",
                        reg.name.name, name.name
                    ),
                ));
                continue;
            }

            let reg_ty = self.resolve_type(&reg.ty);
            let (elem_ty, array_len, elem_bytes) = match &reg_ty {
                Ty::Prim(p @ (PrimType::U8 | PrimType::U16 | PrimType::U32)) => {
                    (*p, None, p.byte_size(4) as u64)
                }
                Ty::Array { elem, len } => match &**elem {
                    Ty::Prim(p @ (PrimType::U8 | PrimType::U16 | PrimType::U32)) => {
                        if *len == 0 {
                            self.emit(Diagnostic::new(
                                DiagnosticKind::MmioViolation,
                                reg.ty.span,
                                format!(
                                    "MMIO register array '{}.{}' must have length >= 1",
                                    name.name, reg.name.name
                                ),
                            ));
                        }
                        (*p, Some((*len).max(1)), p.byte_size(4) as u64)
                    }
                    _ => {
                        self.emit(Diagnostic::new(
                            DiagnosticKind::MmioViolation,
                            reg.ty.span,
                            format!(
                                "MMIO register '{}.{}' has unsupported type '{}'; only `u8`, `u16`, `u32`, and fixed arrays `[N]u8`/`[N]u16`/`[N]u32` are supported",
                                name.name, reg.name.name, reg_ty
                            ),
                        ));
                        continue;
                    }
                },
                Ty::Error => continue,
                other => {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        reg.ty.span,
                        format!(
                            "MMIO register '{}.{}' has unsupported type '{}'; only `u8`, `u16`, `u32`, and fixed arrays `[N]u8`/`[N]u16`/`[N]u32` are supported",
                            name.name, reg.name.name, other
                        ),
                    ));
                    continue;
                }
            };

            let offset_u32 = self.eval_u32_const(
                &reg.offset,
                DiagnosticKind::MmioViolation,
                &format!("MMIO register '{}.{}' offset", name.name, reg.name.name),
            );
            let offset_u64 = u64::from(offset_u32);

            if offset_u64 % elem_bytes != 0 {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.offset.span,
                    format!(
                        "MMIO register '{}.{}' at offset 0x{:X} is not naturally aligned to {}-byte `{}` boundary",
                        name.name, reg.name.name, offset_u64, elem_bytes, elem_ty
                    ),
                ));
            }

            let Ok(count_u32) = u32::try_from(array_len.unwrap_or(1)) else {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.ty.span,
                    format!(
                        "MMIO register array '{}.{}' length {} exceeds `u32` address space",
                        name.name,
                        reg.name.name,
                        array_len.unwrap_or(1)
                    ),
                ));
                continue;
            };
            let elem_bytes_u32 = elem_bytes as u32;
            let Some(reg_bytes_u32) = elem_bytes_u32.checked_mul(count_u32) else {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.ty.span,
                    format!(
                        "MMIO register '{}.{}' byte size overflows 32-bit MMIO address space",
                        name.name, reg.name.name
                    ),
                ));
                continue;
            };
            let Some(end_u32) = offset_u32.checked_add(reg_bytes_u32) else {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.span,
                    format!(
                        "MMIO register '{}.{}' at offset 0x{:X} (size {} bytes) overflows 32-bit MMIO address space",
                        name.name, reg.name.name, offset_u32, reg_bytes_u32
                    ),
                ));
                continue;
            };
            let end = u64::from(end_u32);

            for (prev_start, prev_end, prev_name) in &ranges {
                if offset_u64 < *prev_end && *prev_start < end {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        reg.span,
                        format!(
                            "MMIO register '{}.{}' (0x{:X}..0x{:X}) overlaps register '{}' (0x{:X}..0x{:X})",
                            name.name, reg.name.name, offset_u64, end, prev_name, prev_start, prev_end
                        ),
                    ));
                    break;
                }
            }

            ranges.push((offset_u64, end, reg.name.name.clone()));
            max_end = max_end.max(end);
            resolved_regs.push(MmioRegInfo {
                name: reg.name.name.clone(),
                access: reg.access,
                elem_ty,
                array_len,
                offset: offset_u32,
                span: reg.span,
            });
        }

        self.mmio_layouts.insert(
            name.name.clone(),
            MmioLayoutInfo {
                name: name.name.clone(),
                byte_size: usize::try_from(max_end).unwrap_or(usize::MAX),
                regs: resolved_regs,
                span,
            },
        );
    }

    fn collect_irq(&mut self, iq: &IrqDecl) {
        if !self.check_top_level_name_unique(&iq.name) {
            return;
        }
        let irq_num = self.eval_u32_const(
            &iq.number,
            DiagnosticKind::IrqAttachmentError,
            &format!("IRQ '{}' vector number", iq.name.name),
        );
        self.irqs.insert(
            iq.name.name.clone(),
            IrqInfo {
                name: iq.name.name.clone(),
                irq_num,
                span: iq.span,
            },
        );
    }

    fn eval_u32_const(&mut self, expr: &Expr, diag_kind: DiagnosticKind, what: &str) -> u32 {
        let mut visiting = BTreeSet::new();
        match self.eval_const_expr_i128(expr, &mut visiting) {
            Some(v) => match u32::try_from(v) {
                Ok(u) => u,
                Err(_) => {
                    self.emit(Diagnostic::new(
                        diag_kind,
                        expr.span,
                        format!("{what} value {v} is outside valid `u32` range (0..=0xFFFF_FFFF)"),
                    ));
                    0
                }
            },
            None => {
                self.emit(Diagnostic::new(
                    diag_kind,
                    expr.span,
                    format!("{what} must be a compile-time `u32` constant expression"),
                ));
                0
            }
        }
    }

    fn collect_const(&mut self, c: &ConstDecl) {
        if !self.check_top_level_name_unique(&c.name) {
            return;
        }

        let ty = self.resolve_type(&c.ty);
        self.raw_const_exprs
            .insert(c.name.name.clone(), (c.value.clone(), ty.clone(), self.current_prog_idx));
        let int_value = match &c.value.kind {
            ExprKind::Literal(Literal::Int(v)) => Some(*v),
            _ => None,
        };
        let mut body_checker = BodyChecker::new(
            self,
            "<const>",
            CallableKind::Fn,
            ActiveBodyKind::Fn {
                ret_ty: Ty::void(),
                error_spec: ResolvedErrorSpec::None,
            },
        );
        let val = body_checker.check_expr_infallible(&c.value);
        if !val.ty.is_assignable_to(&ty) {
            self.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                c.value.span,
                format!(
                    "const '{}' expected initializer of type '{}', found '{}'",
                    c.name.name, ty, val.ty
                ),
            ));
        } else {
            self.check_f32_coercion_expr(&c.value, &ty);
        }

        self.consts.insert(
            c.name.name.clone(),
            ConstInfo {
                name: c.name.name.clone(),
                ty,
                int_value,
                span: c.span,
            },
        );
    }

    fn evaluate_all_const_int_values(&mut self) {
        let names: Vec<String> = self.consts.keys().cloned().collect();
        for name in names {
            let mut visiting = BTreeSet::new();
            if let Some(v) = self.eval_named_const_i128(&name, &mut visiting) {
                if let Ok(u) = u64::try_from(v) {
                    if let Some(info) = self.consts.get_mut(&name) {
                        info.int_value = Some(u);
                    }
                }
            }
        }
    }

    fn eval_named_const_i128(
        &self,
        name: &str,
        visiting: &mut BTreeSet<String>,
    ) -> Option<i128> {
        let (expr, ty, _) = self.raw_const_exprs.get(name)?;
        if !visiting.insert(name.to_string()) {
            return None;
        }
        let raw = self.eval_const_expr_i128(expr, visiting);
        visiting.remove(name);
        let val = raw?;
        if let Ty::Prim(p) = ty {
            if p.is_integer() {
                return Some(crate::monomorphize::truncate_int_prim(val, *p));
            }
        }
        Some(val)
    }

    fn infer_const_prim_ty(&self, expr: &Expr) -> Option<PrimType> {
        match &expr.kind {
            ExprKind::Ident(id) => match self.consts.get(&id.name).map(|c| &c.ty) {
                Some(Ty::Prim(p)) if p.is_integer() => Some(*p),
                _ => None,
            },
            ExprKind::Cast { ty, .. } => match &ty.kind {
                TypeKind::Prim(p) if p.is_integer() => Some(*p),
                TypeKind::Named(id) => PrimType::from_name(&id.name).filter(|p| p.is_integer()),
                _ => None,
            },
            ExprKind::Unary { expr: inner, .. } => self.infer_const_prim_ty(inner),
            ExprKind::Binary { lhs, rhs, .. } => self
                .infer_const_prim_ty(lhs)
                .or_else(|| self.infer_const_prim_ty(rhs)),
            _ => None,
        }
    }

    fn eval_const_expr_i128(
        &self,
        expr: &Expr,
        visiting: &mut BTreeSet<String>,
    ) -> Option<i128> {
        self.eval_const_expr_i128_shadowed(expr, visiting, &BTreeSet::new())
    }

    fn eval_const_expr_i128_shadowed(
        &self,
        expr: &Expr,
        visiting: &mut BTreeSet<String>,
        shadowed: &BTreeSet<String>,
    ) -> Option<i128> {
        match &expr.kind {
            ExprKind::Literal(Literal::Int(v)) => Some(*v as i128),
            ExprKind::Ident(id) => {
                if shadowed.contains(&id.name) {
                    return None;
                }
                self.eval_named_const_i128(&id.name, visiting).or_else(|| {
                    self.consts
                        .get(&id.name)
                        .and_then(|c| c.int_value)
                        .map(|u| u as i128)
                })
            }
            ExprKind::Cast { expr: inner, ty } => {
                let v = self.eval_const_expr_i128_shadowed(inner, visiting, shadowed)?;
                let target_prim = match &ty.kind {
                    TypeKind::Prim(p) if p.is_integer() => Some(*p),
                    TypeKind::Named(id) => PrimType::from_name(&id.name).filter(|p| p.is_integer()),
                    _ => None,
                }?;
                Some(crate::monomorphize::truncate_int_prim(v, target_prim))
            }
            ExprKind::Unary { op, expr: inner } => {
                let v = self.eval_const_expr_i128_shadowed(inner, visiting, shadowed)?;
                match op {
                    UnaryOp::Neg => {
                        let neg = v.checked_neg()?;
                        if let Some(p) = self.infer_const_prim_ty(inner) {
                            if p.is_signed() {
                                return Some(crate::monomorphize::truncate_int_prim(neg, p));
                            }
                        }
                        Some(neg)
                    }
                    UnaryOp::BitNot => {
                        let inv = !v;
                        if let Some(p) = self.infer_const_prim_ty(inner) {
                            Some(crate::monomorphize::truncate_int_prim(inv, p))
                        } else {
                            Some(inv)
                        }
                    }
                    _ => None,
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.eval_const_expr_i128_shadowed(lhs, visiting, shadowed)?;
                let r = self.eval_const_expr_i128_shadowed(rhs, visiting, shadowed)?;
                match op {
                    BinaryOp::Add => l.checked_add(r),
                    BinaryOp::Sub => l.checked_sub(r),
                    BinaryOp::Mul => l.checked_mul(r),
                    BinaryOp::Div if r != 0 => l.checked_div(r),
                    BinaryOp::Rem if r != 0 => l.checked_rem(r),
                    BinaryOp::BitOr => Some(l | r),
                    BinaryOp::BitAnd => Some(l & r),
                    BinaryOp::BitXor => Some(l ^ r),
                    BinaryOp::Shl => u32::try_from(r).ok().and_then(|sh| l.checked_shl(sh)),
                    BinaryOp::Shr => u32::try_from(r).ok().and_then(|sh| l.checked_shr(sh)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn check_f32_coercion_expr(&mut self, expr: &Expr, target_ty: &Ty) {
        match target_ty {
            Ty::Prim(PrimType::F32) => match &expr.kind {
                ExprKind::Literal(Literal::Float { bits, suffix: None }) => {
                    if !(f64::from_bits(*bits) as f32).is_finite()
                        && self
                            .f32_overflow_reported_spans
                            .insert((expr.span.start, expr.span.end))
                    {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                expr.span,
                                "float literal out of finite 'f32' range",
                            )
                            .with_note(
                                "use `f64` for double-precision exponents above ~3.40e38, or write an explicit `(f32)` cast if saturation to infinity is intended",
                            ),
                        );
                    }
                }
                ExprKind::Unary {
                    op: UnaryOp::Neg,
                    expr: inner,
                } => self.check_f32_coercion_expr(inner, target_ty),
                ExprKind::Binary {
                    op:
                        BinaryOp::Add
                        | BinaryOp::Sub
                        | BinaryOp::Mul
                        | BinaryOp::Div
                        | BinaryOp::Rem,
                    lhs,
                    rhs,
                } => {
                    self.check_f32_coercion_expr(lhs, target_ty);
                    self.check_f32_coercion_expr(rhs, target_ty);
                }
                ExprKind::Or {
                    expr: a,
                    fallback: b,
                } => {
                    self.check_f32_coercion_expr(a, target_ty);
                    self.check_f32_coercion_expr(b, target_ty);
                }
                ExprKind::Catch { expr: a, clause } => {
                    self.check_f32_coercion_expr(a, target_ty);
                    if let Some(Stmt {
                        kind: StmtKind::Expr(tail),
                        ..
                    }) = clause.body.stmts.last()
                    {
                        self.check_f32_coercion_expr(tail, target_ty);
                    }
                }
                _ => {}
            },
            Ty::Array { elem, .. } | Ty::Slice(elem) => {
                if let ExprKind::ArrayLit(elems) = &expr.kind {
                    for e in elems {
                        self.check_f32_coercion_expr(e, elem);
                    }
                }
            }
            _ => {}
        }
    }

    fn collect_os_primitive(&mut self, item_attrs: &[Attr], p: &OsPrimitiveDecl) {
        if !self.check_top_level_name_unique(&p.name) {
            return;
        }
        let dma_safe = has_attr_dma_safe(item_attrs) || has_attr_dma_safe(&p.attrs);
        let resolved = self.resolve_os_primitive(&p.ty, p.span);
        if resolved.is_unsized_view() {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                p.span,
                format!(
                    "top-level OS primitive '{}' requires a concrete capacity `<N>` (use `{}` view only behind pointer `*`)",
                    resolved, resolved
                ),
            ));
            return;
        }

        if let Some(init_expr) = &p.init {
            let mut body_checker = BodyChecker::new(
                self,
                "<primitive-init>",
                CallableKind::Fn,
                ActiveBodyKind::Fn {
                    ret_ty: Ty::void(),
                    error_spec: ResolvedErrorSpec::None,
                },
            );
            let val = body_checker.check_expr_infallible(init_expr);
            match &resolved {
                ResolvedOsPrimitive::Watch { elem } => {
                    if !val.ty.is_assignable_to(elem) {
                        self.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            init_expr.span,
                            format!(
                                "watch '{}' initializer has type '{}', expected '{}'",
                                p.name.name, val.ty, elem
                            ),
                        ));
                    } else {
                        self.check_f32_coercion_expr(init_expr, elem);
                    }
                }
                other => {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        init_expr.span,
                        format!(
                            "OS primitive '{}' of kind '{}' does not take a scalar initializer",
                            p.name.name,
                            other.kind()
                        ),
                    ));
                }
            }
        }

        let kind = resolved.kind();
        let elem_ty = resolved.elem_ty().cloned();
        if let Some(et) = &elem_ty {
            self.check_type_no_byval_mmio(
                et,
                p.span,
                &format!("element type of OS primitive '{}'", p.name.name),
            );
            match &resolved {
                ResolvedOsPrimitive::Pool { .. } => {
                    self.validate_pool_slot_elem(
                        et,
                        p.span,
                        &format!("pool '{}'", p.name.name),
                    );
                }
                ResolvedOsPrimitive::DmaSlot { .. } => {
                    self.validate_pool_slot_elem(
                        et,
                        p.span,
                        &format!("dma_slot '{}'", p.name.name),
                    );
                }
                ResolvedOsPrimitive::Inbox { elem, .. }
                    if matches!(&**elem, Ty::Lease(_)) =>
                {
                    if let Ty::Lease(inner) = &**elem {
                        self.validate_pool_slot_elem(
                            inner,
                            p.span,
                            &format!("Lease<T> in inbox '{}'", p.name.name),
                        );
                    }
                }
                _ => {
                    self.check_disallowed_lease_type(
                        et,
                        p.span,
                        &format!("OS primitive '{}'", p.name.name),
                        false,
                    );
                }
            }
        }
        let capacity = resolved.capacity();
        let clamps_sleep_depth = matches!(kind, PrimitiveKind::PingPong)
            || (dma_safe && matches!(kind, PrimitiveKind::Pool));
        self.primitives.insert(
            p.name.name.clone(),
            PrimitiveInfo {
                name: p.name.name.clone(),
                primitive: resolved,
                kind,
                elem_ty,
                capacity,
                dma_safe,
                has_init: p.init.is_some(),
                clamps_sleep_depth,
                span: p.span,
            },
        );
    }

    fn collect_global_var(&mut self, item_attrs: &[Attr], g: &GlobalVarDecl) {
        // Resilient fallback if a parser ever produces GlobalVar with an OsPrimitive TypeKind.
        if let TypeKind::OsPrimitive(os_ty) = &g.ty.kind {
            let synth = OsPrimitiveDecl {
                attrs: Vec::new(),
                ty: os_ty.clone(),
                name: g.name.clone(),
                init: g.init.clone(),
                span: g.span,
            };
            self.collect_os_primitive(item_attrs, &synth);
            return;
        }

        if !self.check_top_level_name_unique(&g.name) {
            return;
        }
        let ty = self.resolve_type(&g.ty);
        if let Ty::Struct(struct_name) = &ty {
            if self.structs.get(struct_name).is_some_and(|s| s.is_subsystem) {
                if let Some(init_expr) = &g.init {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        init_expr.span,
                        format!(
                            "subsystem instance '{}' of type '{}' cannot have a runtime initializer; embedded OS primitives are statically zero-initialized",
                            g.name.name, struct_name
                        ),
                    ));
                }
                self.subsystems.insert(
                    g.name.name.clone(),
                    SubsystemInfo {
                        name: g.name.name.clone(),
                        struct_name: struct_name.clone(),
                        dma_safe: has_attr_dma_safe(item_attrs),
                        span: g.span,
                    },
                );
                return;
            }
        }
        let ty_str = if ty.is_error() {
            "u32".to_string()
        } else {
            ty.to_string()
        };
        self.emit(
            Diagnostic::new(
                DiagnosticKind::IllegalMutableGlobal,
                g.span,
                format!(
                    "mutable raw global variable '{}' is forbidden; top-level declarations must be 'const' or an OS primitive (signal, inbox, watch, ping_pong, pool, dma_slot, byte_ring)",
                    g.name.name
                ),
            )
            .with_note(format!(
                "replace with `const {ty} {name} = ...;` (immutable), `watch<{ty}> {name} = 0;` (single-writer latest value via `.read()`/`.publish(val)`), or `inbox<{ty}, 8> {name};` (queued events via `.post(ev)`/`.recv()`)",
                ty = ty_str,
                name = g.name.name
            )),
        );
        self.check_disallowed_lease_type(
            &ty,
            g.ty.span,
            &format!("global declaration '{}'", g.name.name),
            false,
        );

        self.illegal_globals.insert(
            g.name.name.clone(),
            IllegalGlobalInfo {
                ty,
                decl_span: g.span,
            },
        );
    }

    fn collect_fn_sig(&mut self, _item_attrs: &[Attr], f: &FnDecl) {
        if !self.check_top_level_name_unique(&f.name) {
            return;
        }
        let ret_ty = self.resolve_type(&f.ret_type);
        self.check_type_no_byval_mmio(
            &ret_ty,
            f.ret_type.span,
            &format!("return type of function '{}'", f.name.name),
        );
        if f.is_extern_c {
            self.check_disallowed_lease_type(
                &ret_ty,
                f.ret_type.span,
                &format!("return type of extern \"C\" function '{}'", f.name.name),
                false,
            );
        } else {
            self.check_disallowed_lease_type(
                &ret_ty,
                f.ret_type.span,
                &format!("return type of function '{}'", f.name.name),
                true,
            );
        }
        let error_spec = match &f.error_spec {
            None => ResolvedErrorSpec::None,
            Some(ErrorSpec::Bare(_)) => ResolvedErrorSpec::Untyped,
            Some(ErrorSpec::Typed(err_ty_expr)) => {
                let err_ty = self.resolve_type(err_ty_expr);
                self.check_type_no_byval_mmio(
                    &err_ty,
                    err_ty_expr.span,
                    &format!("error specification of function '{}'", f.name.name),
                );
                self.check_disallowed_lease_type(
                    &err_ty,
                    err_ty_expr.span,
                    &format!("error specification of function '{}'", f.name.name),
                    false,
                );
                if err_ty.is_void() {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::ErrorSpecMismatch,
                        err_ty_expr.span,
                        "typed 'error' specification cannot use 'void'; use bare 'error' instead",
                    ));
                }
                ResolvedErrorSpec::Typed(err_ty)
            }
        };

        if !f.is_extern_c && f.body.is_none() {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                f.span,
                format!(
                    "function '{}' has no body (only 'extern \"C\"' declarations may omit a body)",
                    f.name.name
                ),
            ));
        }

        let params = self.resolve_params(&f.params, &f.name.name, !f.is_extern_c);
        let callable_kind = if f.is_extern_c {
            CallableKind::ExternCFn
        } else {
            CallableKind::Fn
        };
        self.call_graph
            .add_node(f.name.name.clone(), callable_kind, f.span);
        if let Some(cf) = &self.current_file {
            self.node_files.insert(f.name.name.clone(), cf.clone());
        }

        self.fn_decl_order.push(f.name.name.clone());
        self.functions.insert(
            f.name.name.clone(),
            FunctionInfo {
                name: f.name.name.clone(),
                params,
                is_variadic: f.is_variadic,
                ret_ty,
                error_spec,
                is_extern_c: f.is_extern_c,
                direct_yield: false,
                can_yield: false,
                effect: FnEffect::NonYielding,
                spawns_in_boot: false,
                yield_chain: Vec::new(),
                callees: Vec::new(),
                yielding_callees: Vec::new(),
                return_param_origins: Vec::new(),
                span: f.span,
            },
        );
    }

    fn collect_task_sig(&mut self, _item_attrs: &[Attr], t: &TaskDecl) {
        if !self.check_top_level_name_unique(&t.name) {
            return;
        }
        let params = self.resolve_params(&t.params, &t.name.name, false);
        let node_key = format!("task::{}", t.name.name);
        if let Some(cf) = &self.current_file {
            self.node_files.insert(node_key.clone(), cf.clone());
        }
        self.call_graph
            .add_node(node_key, CallableKind::Task, t.span);

        self.tasks.insert(
            t.name.name.clone(),
            TaskInfo {
                name: t.name.name.clone(),
                params,
                direct_yield: false,
                can_yield: false,
                effect: FnEffect::NonYielding,
                callees: Vec::new(),
                yielding_callees: Vec::new(),
                instance_count: 0,
                span: t.span,
            },
        );
    }

    fn collect_isr_sig(&mut self, _item_attrs: &[Attr], i: &IsrDecl) {
        if self.isrs.contains_key(&i.vector.name) {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                i.vector.span,
                format!("duplicate ISR handler for vector '{}'", i.vector.name),
            ));
            return;
        }
        let params = self.resolve_params(&i.params, &format!("isr {}", i.vector.name), false);
        let node_key = format!("isr::{}", i.vector.name);
        if let Some(cf) = &self.current_file {
            self.node_files.insert(node_key.clone(), cf.clone());
        }
        self.call_graph
            .add_node(node_key, CallableKind::Isr, i.span);

        self.isrs.insert(
            i.vector.name.clone(),
            IsrInfo {
                vector: i.vector.name.clone(),
                params,
                callees: Vec::new(),
                priority_boosts: Vec::new(),
                span: i.span,
            },
        );
    }

    fn resolve_params(
        &mut self,
        params: &[Param],
        owner: &str,
        allow_top_level_lease: bool,
    ) -> Vec<ParamInfo> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for p in params {
            if !seen.insert(p.name.name.clone()) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::DuplicateDefinition,
                    p.name.span,
                    format!("duplicate parameter '{}' in '{}'", p.name.name, owner),
                ));
            }
            let ty = self.resolve_type(&p.ty);
            if ty.is_void() {
                self.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    p.ty.span,
                    format!("parameter '{}' in '{}' cannot have type 'void'", p.name.name, owner),
                ));
            }
            self.check_type_no_byval_mmio(
                &ty,
                p.ty.span,
                &format!("parameter '{}' of '{}'", p.name.name, owner),
            );
            self.check_disallowed_lease_type(
                &ty,
                p.ty.span,
                &format!("parameter '{}' of '{}'", p.name.name, owner),
                allow_top_level_lease,
            );
            if matches!(
                &ty,
                Ty::OsPrimitive(
                    ResolvedOsPrimitive::Pool { .. } | ResolvedOsPrimitive::DmaSlot { .. }
                )
            ) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    p.ty.span,
                    format!(
                        "cannot pass static OS primitive '{}' by value in parameter '{}' of '{}' (pass a pointer `*{}`)",
                        ty, p.name.name, owner, ty
                    ),
                ));
            }
            out.push(ParamInfo {
                name: p.name.name.clone(),
                ty,
                span: p.span,
            });
        }
        out
    }

    fn check_type_no_byval_mmio_or_unsized(&mut self, ty: &Ty, span: Span, context: &str) {
        if let Some(layout_name) = ty_contains_byval_mmio(ty) {
            self.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                span,
                format!(
                    "cannot pass or store MMIO register layout '{}' by value in {} (use pointer `*{}`)",
                    layout_name, context, layout_name
                ),
            ));
        }
        if let Some(unsized_ty) = ty_contains_bare_unsized_prim(ty) {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                span,
                format!(
                    "unsized primitive type '{}' in {} requires a concrete capacity `<N>` (or pointer view `*{}`)",
                    unsized_ty, context, unsized_ty
                ),
            ));
        }
    }

    fn check_type_no_byval_mmio(&mut self, ty: &Ty, span: Span, context: &str) {
        self.check_type_no_byval_mmio_or_unsized(ty, span, context);
        if let Some(subsys_name) = ty_contains_byval_subsys(ty, &self.structs) {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                span,
                format!(
                    "cannot pass, store, or return subsystem struct '{}' by value in {} (use pointer `*{}`)",
                    subsys_name, context, subsys_name
                ),
            ));
        }
    }

    fn resolve_type(&mut self, te: &TypeExpr) -> Ty {
        match &te.kind {
            TypeKind::Prim(p) => Ty::Prim(*p),
            TypeKind::Named(ident) => {
                if let Some(p) = PrimType::from_name(&ident.name) {
                    return Ty::Prim(p);
                }
                if self.structs.contains_key(&ident.name) {
                    return Ty::Struct(ident.name.clone());
                }
                if self.enums.contains_key(&ident.name) {
                    return Ty::Enum(ident.name.clone());
                }
                if self.mmio_layouts.contains_key(&ident.name)
                    || self.mmio_layout_names.contains(&ident.name)
                {
                    return Ty::MmioLayout(ident.name.clone());
                }
                let hint = if let Some(z_ty) = suggest_foreign_type_name(&ident.name) {
                    format!(
                        "did you mean Zev built-in type `{}`? (Zev uses explicit fixed-width types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `usize`, `bool`, `u16be`, `u32le`, `[]u8`)",
                        z_ty
                    )
                } else {
                    let prim_names = [
                        "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize",
                        "f32", "f64", "u16be", "u16le", "u32be", "u32le", "u64be", "u64le",
                        "bool", "void",
                    ];
                    let candidates = prim_names
                        .into_iter()
                        .chain(self.structs.keys().map(String::as_str))
                        .chain(self.enums.keys().map(String::as_str));
                    if let Some(best) = closest_match(&ident.name, candidates, 2) {
                        format!("did you mean type `{}`?", best)
                    } else {
                        "built-in Zev types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `usize`, `bool`, `void`, `u16be`, `u32le`, `[]T`, `*T`".to_string()
                    }
                };
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        ident.span,
                        format!("unknown type '{}'", ident.name),
                    )
                    .with_note(hint),
                );
                Ty::Error
            }
            TypeKind::Path(segments) => {
                let joined = segments
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                self.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    te.span,
                    format!("unknown qualified type '{}'", joined),
                ));
                Ty::Error
            }
            TypeKind::Generic { base, .. } => {
                self.emit(Diagnostic::new(
                    DiagnosticKind::GenericInstantiationError,
                    te.span,
                    format!("unmonomorphized generic type '{}'", base.name),
                ));
                Ty::Error
            }
            TypeKind::GenericPath { path, .. } => {
                let joined = path
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                self.emit(Diagnostic::new(
                    DiagnosticKind::GenericInstantiationError,
                    te.span,
                    format!("unmonomorphized generic type '{}'", joined),
                ));
                Ty::Error
            }
            TypeKind::Slice(elem) => Ty::Slice(Box::new(self.resolve_type(elem))),
            TypeKind::Array { elem, len } => {
                let resolved_len = match len {
                    ArrayLen::Fixed(n) => *n,
                    ArrayLen::Expr(expr) => {
                        let mut visiting = BTreeSet::new();
                        if let Some(v) = self.eval_const_expr_i128(expr, &mut visiting) {
                            usize::try_from(v).unwrap_or_else(|_| {
                                self.emit(Diagnostic::new(
                                    DiagnosticKind::GenericInstantiationError,
                                    expr.span,
                                    "array length must be a non-negative compile-time constant 'usize'",
                                ));
                                1
                            })
                        } else {
                            self.emit(Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                expr.span,
                                "array length must be a compile-time constant 'usize'",
                            ));
                            1
                        }
                    }
                };
                if resolved_len == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        te.span,
                        "fixed array length must be greater than 0",
                    ));
                }
                Ty::Array {
                    elem: Box::new(self.resolve_type(elem)),
                    len: resolved_len,
                }
            }
            TypeKind::Ptr(inner) => Ty::Ptr(Box::new(self.resolve_type(inner))),
            TypeKind::OsPrimitive(os) => Ty::OsPrimitive(self.resolve_os_primitive(os, te.span)),
            TypeKind::Lease(inner) => {
                let elem = self.resolve_non_void_elem(inner);
                Ty::Lease(Box::new(elem))
            }
            TypeKind::FnPtr { params, ret } => {
                let ps = params.iter().map(|p| self.resolve_type(p)).collect();
                let r = self.resolve_type(ret);
                Ty::FnPtr {
                    params: ps,
                    ret: Box::new(r),
                }
            }
        }
    }

    fn resolve_os_primitive(&mut self, os: &OsPrimitiveType, span: Span) -> ResolvedOsPrimitive {
        match os {
            OsPrimitiveType::Signal => ResolvedOsPrimitive::Signal,
            OsPrimitiveType::Inbox { elem, cap } => {
                if *cap == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        span,
                        "inbox capacity must be greater than 0",
                    ));
                }
                ResolvedOsPrimitive::Inbox {
                    elem: Box::new(self.resolve_non_void_elem(elem)),
                    cap: *cap,
                }
            }
            OsPrimitiveType::InboxUnsized { elem } => ResolvedOsPrimitive::InboxUnsized {
                elem: Box::new(self.resolve_non_void_elem(elem)),
            },
            OsPrimitiveType::Watch { elem } => ResolvedOsPrimitive::Watch {
                elem: Box::new(self.resolve_non_void_elem(elem)),
            },
            OsPrimitiveType::PingPong { elem, cap } => {
                if *cap == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        span,
                        "ping_pong buffer capacity must be greater than 0",
                    ));
                }
                ResolvedOsPrimitive::PingPong {
                    elem: Box::new(self.resolve_non_void_elem(elem)),
                    cap: *cap,
                }
            }
            OsPrimitiveType::Pool { elem, cap } => {
                if *cap == 0 || *cap > 32 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        span,
                        format!(
                            "pool capacity must be between 1 and 32 slots (32-bit bitmask), found {}",
                            cap
                        ),
                    ));
                }
                ResolvedOsPrimitive::Pool {
                    elem: Box::new(self.resolve_non_void_elem(elem)),
                    cap: *cap,
                }
            }
            OsPrimitiveType::DmaSlot { elem } => ResolvedOsPrimitive::DmaSlot {
                elem: Box::new(self.resolve_non_void_elem(elem)),
            },
            OsPrimitiveType::ByteRing { cap } => {
                if *cap == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        span,
                        "byte_ring capacity must be greater than 0",
                    ));
                }
                ResolvedOsPrimitive::ByteRing { cap: *cap }
            }
            OsPrimitiveType::ByteRingUnsized => ResolvedOsPrimitive::ByteRingUnsized,
        }
    }

    fn resolve_non_void_elem(&mut self, elem: &TypeExpr) -> Ty {
        let t = self.resolve_type(elem);
        if t.is_void() {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                elem.span,
                "OS primitive element type cannot be 'void'",
            ));
        }
        t
    }

    fn validate_duration(&mut self, dur: &DurationLit) {
        let us = dur.as_micros();
        if us == 0 {
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::InvalidDurationResolution,
                    dur.span,
                    format!("duration '{}' must be greater than 0", dur),
                )
                .with_note(format!(
                    "scheduler tick period is {}us; use at least 1 tick (e.g. `1ms`)",
                    self.tick_period_us
                )),
            );
            return;
        }
        if us < self.tick_period_us {
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::InvalidDurationResolution,
                    dur.span,
                    format!(
                        "sub-tick duration '{}' ({}us) is smaller than configured tick resolution {}us and would truncate to 0 ticks",
                        dur, us, self.tick_period_us
                    ),
                )
                .with_note(format!(
                    "increase the duration to at least `{}us` (`1ms` by default) or configure a finer tick in `main()` via `sys.use_systick(hz, {});`",
                    self.tick_period_us, dur
                )),
            );
            return;
        }
        if us % self.tick_period_us != 0 {
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::InvalidDurationResolution,
                    dur.span,
                    format!(
                        "duration '{}' ({}us) is not an exact multiple of configured tick resolution {}us",
                        dur, us, self.tick_period_us
                    ),
                )
                .with_note(format!(
                    "use an exact multiple of {}us or configure a finer tick period via `sys.use_systick(hz, 500us);`",
                    self.tick_period_us
                )),
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass 2 Entry Points
    // ─────────────────────────────────────────────────────────────────────────

    fn check_fn_body(&mut self, item_attrs: &[Attr], f: &FnDecl) {
        let Some(body) = &f.body else {
            return;
        };
        let Some(fn_info) = self.functions.get(&f.name.name).cloned() else {
            return;
        };
        let prev_note = self.current_instantiation_note.take();
        self.current_instantiation_note =
            self.mono_instantiation_notes.get(&f.name.name).cloned();
        let force_noyield = has_attr_noyield(item_attrs) || has_attr_noyield(&f.attrs);
        let mut checker =
            BodyChecker::new(self, &f.name.name, CallableKind::Fn, ActiveBodyKind::Fn {
                ret_ty: fn_info.ret_ty.clone(),
                error_spec: fn_info.error_spec.clone(),
            });
        if force_noyield {
            checker.noyield_depth += 1;
        }
        checker.bind_params(&fn_info.params);
        let diverges = checker.check_block_stmts(&body.stmts);
        if !diverges {
            checker.finish_lexical_block(false, f.name.span);
        }
        let ret_origins: Vec<usize> = checker.return_param_origins.into_iter().collect();
        let boot_steps = checker.boot_steps;
        let irq_nodes = checker
            .boot_block_stack
            .into_iter()
            .next()
            .unwrap_or_default();

        if !fn_info.ret_ty.is_void() && !fn_info.ret_ty.is_error() && !diverges {
            let ret_ty = fn_info.ret_ty.clone();
            self.emit(Diagnostic::new(
                DiagnosticKind::ReturnMismatch,
                f.name.span,
                format!(
                    "function '{}' declared with return type '{}' may exit without returning a value",
                    f.name.name, ret_ty
                ),
            ));
        }
        self.current_instantiation_note = prev_note;

        self.fn_boot_steps.insert(f.name.name.clone(), boot_steps);
        self.fn_irq_boot_nodes
            .insert(f.name.name.clone(), irq_nodes);
        if let Some(info) = self.functions.get_mut(&f.name.name) {
            info.return_param_origins = ret_origins;
        }
    }

    fn check_task_body(&mut self, item_attrs: &[Attr], t: &TaskDecl) {
        let Some(task_info) = self.tasks.get(&t.name.name).cloned() else {
            return;
        };
        let node_key = format!("task::{}", t.name.name);
        let force_noyield = has_attr_noyield(item_attrs) || has_attr_noyield(&t.attrs);
        let mut checker =
            BodyChecker::new(self, &node_key, CallableKind::Task, ActiveBodyKind::Task);
        if force_noyield {
            checker.noyield_depth += 1;
        }
        checker.bind_params(&task_info.params);
        let diverges = checker.check_block_stmts(&t.body.stmts);
        if !diverges {
            self.emit(Diagnostic::new(
                DiagnosticKind::NonDivergingTask,
                t.name.span,
                format!(
                    "task '{}' may exit without diverging; tasks are permanent actors and must run inside 'loop' / 'loop every', park on 'inbox.recv()' / 'signal.wait()', reset via 'restart;', or move finite workflows into yielding 'fn's",
                    t.name.name
                ),
            ));
        }
    }

    fn check_isr_body(&mut self, _item_attrs: &[Attr], i: &IsrDecl) {
        let Some(isr_info) = self.isrs.get(&i.vector.name).cloned() else {
            return;
        };
        let node_key = format!("isr::{}", i.vector.name);
        let mut checker =
            BodyChecker::new(self, &node_key, CallableKind::Isr, ActiveBodyKind::Isr);
        checker.bind_params(&isr_info.params);
        let diverges = checker.check_block_stmts(&i.body.stmts);
        if !diverges {
            checker.finish_lexical_block(false, i.vector.span);
        }
        let boosts = checker.recorded_boosts;
        if let Some(info) = self.isrs.get_mut(&i.vector.name) {
            info.priority_boosts = boosts;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass 3: Whole-Program Call-Graph Verification & Boot-Tree Unrolling
    // ─────────────────────────────────────────────────────────────────────────

    fn run_whole_program_verification(&mut self) {
        self.call_graph.infer_yields();

        // 1. Check all recorded call edges against `isr`, `critical`, `defer`, `noyield`, and boot-spawn contexts.
        let edges = self.call_graph.edges.clone();
        for edge in &edges {
            let Some(callee_node) = self.call_graph.nodes.get(&edge.callee).cloned() else {
                continue;
            };
            let caller_file = self.node_files.get(&edge.caller).cloned();

            // 1a. Yielding call checks (3-tier: `SoftYielding` allowed inside `noyield`, forbidden in `critical`/`isr`/`defer`).
            if edge.ctx.in_isr {
                if callee_node.any_yield_effect.can_yield() {
                    let isr_name = edge.caller.strip_prefix("isr::").unwrap_or(&edge.caller);
                    let sub_chain = self
                        .call_graph
                        .shortest_any_yield_chain(&edge.callee)
                        .join(" -> ");
                    let note = format!(
                        "yield chain: isr {} -> {}; ISRs cannot block or yield—signal a task via `inbox.post(Ev, boost: @critical);`, `signal.notify();`, or `dma.flip();`",
                        isr_name, sub_chain
                    );
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::YieldInIsr,
                            edge.span,
                            format!(
                                "cannot call yielding function '{}' inside ISR '{}'",
                                edge.callee, isr_name
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(note),
                    );
                }
            } else if edge.ctx.in_critical {
                if callee_node.any_yield_effect.can_yield() {
                    let caller_disp = edge
                        .caller
                        .strip_prefix("task::")
                        .unwrap_or(&edge.caller);
                    let sub_chain = self
                        .call_graph
                        .shortest_any_yield_chain(&edge.callee)
                        .join(" -> ");
                    let note = format!(
                        "yield chain: critical block in {} -> {}; `critical {{ ... }}` masks hardware IRQs—move yielding calls outside `critical`",
                        caller_disp, sub_chain
                    );
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::YieldInCritical,
                            edge.span,
                            format!(
                                "cannot call yielding function '{}' inside 'critical' block (IRQs are masked)",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(note),
                    );
                }
            } else if edge.ctx.in_defer {
                if callee_node.any_yield_effect.can_yield() {
                    let caller_disp = edge
                        .caller
                        .strip_prefix("task::")
                        .unwrap_or(&edge.caller);
                    let sub_chain = self
                        .call_graph
                        .shortest_any_yield_chain(&edge.callee)
                        .join(" -> ");
                    let note = format!(
                        "yield chain: defer cleanup in {} -> {}; `defer` statements execute synchronously on scope exit and cannot yield",
                        caller_disp, sub_chain
                    );
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::YieldInNoYield,
                            edge.span,
                            format!(
                                "cannot call yielding function '{}' inside 'defer' cleanup statement",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(note),
                    );
                }
            } else if edge.ctx.in_noyield && callee_node.effect == FnEffect::HardBlocking {
                let caller_disp = edge
                    .caller
                    .strip_prefix("task::")
                    .unwrap_or(&edge.caller);
                let sub_chain = self
                    .call_graph
                    .shortest_hard_yield_chain(&edge.callee)
                    .join(" -> ");
                let note = format!(
                    "yield chain: noyield block in {} -> {}; move hard-blocking calls outside `noyield {{ ... }}` (`SoftYielding` functions that only call `checkpoint()` ARE allowed inside `noyield`)",
                    caller_disp, sub_chain
                );
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::YieldInNoYield,
                        edge.span,
                        format!(
                            "cannot call hard-blocking function '{}' inside 'noyield' scope",
                            edge.callee
                        ),
                    )
                    .with_file(caller_file.clone())
                    .with_note(note),
                );
            }

            // 1b. Boot-only `spawns_in_boot` call-site checks.
            if callee_node.spawns_in_boot {
                let spawn_chain = callee_node.spawn_chain.join(" -> ");
                if edge.ctx.in_defer {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidControlFlow,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' inside 'defer' cleanup statement",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                } else if edge.caller_kind == CallableKind::Task {
                    let task_name = edge.caller.strip_prefix("task::").unwrap_or(&edge.caller);
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' inside task '{}'; tasks must be statically spawned in 'main()' boot setup",
                                edge.callee, task_name
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                } else if edge.caller_kind == CallableKind::Isr || edge.ctx.in_isr {
                    let isr_name = edge.caller.strip_prefix("isr::").unwrap_or(&edge.caller);
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' inside ISR '{}'",
                                edge.callee, isr_name
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                } else if edge.ctx.in_loop {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' inside a loop; every static boot-tree path must map 1:1 to a task slot",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                } else if edge.ctx.after_diverge || edge.ctx.after_main_supervisor {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' after 'main()' supervisor loop or diverging control flow",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                }
            }

            // 1c. Boot-only `attaches_in_boot` call-site checks (`E0032`).
            if callee_node.attaches_in_boot {
                let attach_chain = callee_node.attach_chain.join(" -> ");
                if edge.ctx.in_defer {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' inside 'defer' cleanup statement",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                } else if edge.caller_kind == CallableKind::Task {
                    let task_name = edge.caller.strip_prefix("task::").unwrap_or(&edge.caller);
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' inside task '{}'; IRQs must be attached during sequential boot",
                                edge.callee, task_name
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                } else if edge.caller_kind == CallableKind::Isr || edge.ctx.in_isr {
                    let isr_name = edge.caller.strip_prefix("isr::").unwrap_or(&edge.caller);
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' inside ISR '{}'",
                                edge.callee, isr_name
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                } else if edge.ctx.in_loop {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' inside a loop; IRQ attachments must be deterministic in boot",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                } else if edge.ctx.after_diverge || edge.ctx.after_main_supervisor {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' after 'main()' supervisor loop or diverging control flow",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                }
            }
        }

        // 2. Check all `extern "C"` and `sys.*` hook callback arguments for yielding, boot-spawning, or boot-attaching functions.
        let cb_sites = self.call_graph.extern_callbacks.clone();
        for site in &cb_sites {
            if let Some(cb_node) = self.call_graph.nodes.get(&site.callback_fn).cloned() {
                let is_sys_hook = site.extern_fn.starts_with("sys.");
                if cb_node.any_yield_effect.can_yield() {
                    let cb_file = self.node_files.get(&site.callback_fn).cloned();
                    let sub_chain = self
                        .call_graph
                        .shortest_any_yield_chain(&site.callback_fn)
                        .join(" -> ");
                    let (msg, note) = if site.extern_fn == "sys.set_putc" {
                        (
                            format!(
                                "cannot pass yielding function '{}' to 'sys.set_putc': console byte-sink callbacks must be non-yielding",
                                site.callback_fn
                            ),
                            format!(
                                "yield chain: sys.set_putc(...) callback -> {}; pass a non-yielding `fn(u8)` callback",
                                sub_chain
                            ),
                        )
                    } else if is_sys_hook {
                        (
                            format!(
                                "cannot pass yielding function '{}' to '{}': system hook callbacks must be non-yielding",
                                site.callback_fn, site.extern_fn
                            ),
                            format!(
                                "yield chain: {}(...) callback -> {}; pass a non-yielding `fn` callback",
                                site.extern_fn, sub_chain
                            ),
                        )
                    } else {
                        (
                            format!(
                                "cannot pass yielding function '{}' as callback to extern \"C\" function '{}': stackless coroutine yields cannot unwind through C stack frames",
                                site.callback_fn, site.extern_fn
                            ),
                            format!(
                                "yield chain: extern \"C\" {}(...) callback -> {}; stackless coroutines cannot suspend across C stack frames—pass a non-yielding `fn` callback",
                                site.extern_fn, sub_chain
                            ),
                        )
                    };
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::YieldingCallbackToExternC,
                            site.span,
                            msg,
                        )
                        .with_file(cb_file)
                        .with_note(note),
                    );
                }
                if cb_node.spawns_in_boot {
                    let cb_file = self.node_files.get(&site.callback_fn).cloned();
                    let target_desc = if is_sys_hook {
                        format!("'{}'", site.extern_fn)
                    } else {
                        format!("extern \"C\" function '{}'", site.extern_fn)
                    };
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            site.span,
                            format!(
                                "cannot pass boot-spawning function '{}' as callback to {}",
                                site.callback_fn, target_desc
                            ),
                        )
                        .with_file(cb_file),
                    );
                }
                if cb_node.attaches_in_boot {
                    let cb_file = self.node_files.get(&site.callback_fn).cloned();
                    let target_desc = if is_sys_hook {
                        format!("'{}'", site.extern_fn)
                    } else {
                        format!("extern \"C\" function '{}'", site.extern_fn)
                    };
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            site.span,
                            format!(
                                "cannot pass IRQ-attaching function '{}' as callback to {}",
                                site.callback_fn, target_desc
                            ),
                        )
                        .with_file(cb_file),
                    );
                }
            }
        }

        // 3. Detect recursive call cycles across boot-spawning (`E0027`), boot-attaching (`E0032`), and yielding (`E0012`) functions.
        let spawning_cycles = self.call_graph.find_spawning_cycles();
        for (cycle, span) in &spawning_cycles {
            let cycle_file = cycle
                .first()
                .and_then(|f| self.node_files.get(f))
                .cloned();
            let cycle_str = cycle.join(" -> ");
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::InvalidSpawnContext,
                    *span,
                    format!(
                        "recursive call cycle across boot-spawning function: {}",
                        cycle_str
                    ),
                )
                .with_file(cycle_file)
                .with_note(
                    "boot-tree `spawn` functions must form a static DAG so task slots are statically bounded at compile time",
                ),
            );
        }
        let attaching_cycles = self.call_graph.find_attaching_cycles();
        for (cycle, span) in &attaching_cycles {
            let cycle_file = cycle
                .first()
                .and_then(|f| self.node_files.get(f))
                .cloned();
            let cycle_str = cycle.join(" -> ");
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    *span,
                    format!(
                        "recursive call cycle across IRQ-attaching function: {}",
                        cycle_str
                    ),
                )
                .with_file(cycle_file)
                .with_note(
                    "boot-tree `irq.attach(...)` functions must form a static acyclic boot tree",
                ),
            );
        }

        for (cycle, span) in self.call_graph.find_yielding_cycles() {
            let cycle_file = cycle
                .first()
                .and_then(|f| self.node_files.get(f))
                .cloned();
            let cycle_str = cycle.join(" -> ");
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::YieldingRecursion,
                    span,
                    format!("recursive yielding call cycle detected: {}", cycle_str),
                )
                .with_file(cycle_file)
                .with_note(format!(
                    "100% static coroutine frame allocation requires an acyclic call graph across yielding functions; rewrite `{}` as an iterative `loop {{ ... }}` or `while` state machine",
                    cycle_str
                )),
            );
        }

        // 4. Verify interprocedural reference provenance for `spawn` (`E0010`) and ISR `state`
        //    escapes into OS primitives (`E0033`), then unroll `main()`'s static boot tree into
        //    `self.spawned_tasks` and `self.irq_bindings` (`E0032`), and verify `byte_ring` SPSC (`E0002`).
        self.verify_boot_spawn_arg_provenance();
        self.verify_isr_state_escape_provenance();
        self.unroll_boot_spawn_tree();
        self.unroll_irq_boot_tree();
        self.verify_byte_ring_spsc();

        // 5. Compute 3-tier `SoftYielding` coroutine vs `noyield` dual specializations.
        let (soft_coro, soft_ny, dual_spec) = self.call_graph.compute_soft_specializations();
        self.soft_yielding_in_coro = soft_coro;
        self.soft_yielding_in_noyield = soft_ny;
        self.dual_specialized_fns = dual_spec;

        // 6. Synchronize resolved `effect`, `can_yield`, `spawns_in_boot`, and callee lists into `FunctionInfo`, `TaskInfo`, and `IsrInfo`.
        let fn_names: Vec<String> = self.functions.keys().cloned().collect();
        for name in fn_names {
            let callees = self.call_graph.callees_of(&name);
            let yielding_callees = self.call_graph.yielding_callees_of(&name);
            if let (Some(info), Some(node)) = (
                self.functions.get_mut(&name),
                self.call_graph.nodes.get(&name),
            ) {
                info.direct_yield = node.direct_yield;
                info.can_yield = node.can_yield;
                info.effect = node.effect;
                info.spawns_in_boot = node.spawns_in_boot;
                info.yield_chain = node.yield_chain.clone();
                info.callees = callees;
                info.yielding_callees = yielding_callees;
            }
        }

        let task_names: Vec<String> = self.tasks.keys().cloned().collect();
        for name in task_names {
            let key = format!("task::{}", name);
            let callees = self.call_graph.callees_of(&key);
            let yielding_callees = self.call_graph.yielding_callees_of(&key);
            if let (Some(info), Some(node)) =
                (self.tasks.get_mut(&name), self.call_graph.nodes.get(&key))
            {
                info.direct_yield = node.direct_yield;
                info.can_yield = node.can_yield;
                info.effect = node.effect;
                info.callees = callees;
                info.yielding_callees = yielding_callees;
            }
        }

        let isr_names: Vec<String> = self.isrs.keys().cloned().collect();
        for name in isr_names {
            let key = format!("isr::{}", name);
            let callees = self.call_graph.callees_of(&key);
            if let Some(info) = self.isrs.get_mut(&name) {
                info.callees = callees;
            }
        }
    }

    fn propagate_escaping_ref_params(
        escaping: &mut BTreeMap<String, BTreeSet<usize>>,
        recorded_call_args: &[RecordedCallArgs],
    ) {
        loop {
            let mut changed = false;
            for call in recorded_call_args {
                let Some(callee_params) = escaping.get(&call.callee).cloned() else {
                    continue;
                };
                for p_idx in callee_params {
                    if let Some(arg) = call.args.get(p_idx) {
                        for &caller_p in &arg.ref_param_origins {
                            if escaping
                                .entry(call.caller.clone())
                                .or_default()
                                .insert(caller_p)
                            {
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn verify_boot_spawn_arg_provenance(&mut self) {
        let mut spawning_ref_params: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
        for (fn_name, steps) in &self.fn_boot_steps {
            for step in steps {
                if let BootStep::DirectSpawn { arg_bindings, .. } = step {
                    for b in arg_bindings {
                        if !b.ref_param_origins.is_empty() {
                            spawning_ref_params
                                .entry(fn_name.clone())
                                .or_default()
                                .extend(b.ref_param_origins.iter().copied());
                        }
                    }
                }
            }
        }

        Self::propagate_escaping_ref_params(&mut spawning_ref_params, &self.recorded_call_args);

        let calls = self.recorded_call_args.clone();
        for call in &calls {
            let Some(callee_params) = spawning_ref_params.get(&call.callee) else {
                continue;
            };
            let caller_file = self.node_files.get(&call.caller).cloned();
            for &p_idx in callee_params {
                if let Some(arg) = call.args.get(p_idx) {
                    if arg.has_non_static_local_ref {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::SpawnNonStaticReference,
                                arg.span,
                                format!(
                                    "cannot pass non-static reference/slice into boot helper '{}' (argument {}): parameter is forwarded into 'spawn' and outlives local stack variables",
                                    call.callee,
                                    p_idx + 1
                                ),
                            )
                            .with_file(caller_file.clone()),
                        );
                    }
                }
            }
        }
    }

    fn verify_isr_state_escape_provenance(&mut self) {
        let mut escaping = self.prim_escaping_ref_params.clone();
        Self::propagate_escaping_ref_params(&mut escaping, &self.recorded_call_args);

        let calls = self.recorded_call_args.clone();
        for call in &calls {
            let Some(callee_params) = escaping.get(&call.callee) else {
                continue;
            };
            let caller_file = self.node_files.get(&call.caller).cloned();
            for &p_idx in callee_params {
                if let Some(arg) = call.args.get(p_idx) {
                    if arg.has_isr_state_ref {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::InvalidIsrState,
                                arg.span,
                                format!(
                                    "cannot pass `&` of ISR `state` variable into helper '{}' (argument {}): parameter escapes into an OS primitive",
                                    call.callee,
                                    p_idx + 1
                                ),
                            )
                            .with_file(caller_file.clone()),
                        );
                    }
                }
            }
        }
    }

    fn fn_boot_flags(&self, fn_name: &str) -> (bool, bool) {
        let mut visiting = BTreeSet::new();
        let mut stack = vec![fn_name.to_string()];
        visiting.insert(fn_name.to_string());
        let mut spawns = false;
        let mut attaches = false;
        while let Some(cur) = stack.pop() {
            if let Some(node) = self.call_graph.nodes.get(&cur) {
                spawns |= node.spawns_in_boot;
                attaches |= node.attaches_in_boot;
            }
            for e in &self.call_graph.edges {
                if e.caller == cur && visiting.insert(e.callee.clone()) {
                    stack.push(e.callee.clone());
                }
            }
        }
        (spawns, attaches)
    }

    fn resolve_field_chain_map(
        fields: &BTreeMap<String, String>,
        chain: &str,
    ) -> Option<String> {
        if let Some(exact) = fields.get(chain) {
            return Some(exact.clone());
        }
        let mut prefix_len = 0;
        let mut best: Option<(usize, String)> = None;
        for (k, v) in fields {
            if let Some(rest) = chain.strip_prefix(k).and_then(|r| r.strip_prefix('.')) {
                if k.len() > prefix_len {
                    prefix_len = k.len();
                    best = Some((k.len(), format!("{v}.{rest}")));
                }
            }
        }
        best.map(|(_, s)| s)
    }

    fn resolve_endpoint_spec_with_bindings(
        spec: &EndpointSpec,
        param_endpoints: &[Option<String>],
        param_field_endpoints: &[BTreeMap<String, String>],
    ) -> Option<String> {
        match spec {
            EndpointSpec::Static(s) => Some(s.clone()),
            EndpointSpec::Param { index, suffix } => match suffix {
                Some(suf) => param_field_endpoints
                    .get(*index)
                    .and_then(|m| Self::resolve_field_chain_map(m, suf))
                    .or_else(|| {
                        param_endpoints
                            .get(*index)
                            .cloned()
                            .flatten()
                            .map(|base| format!("{base}.{suf}"))
                    }),
                None => param_endpoints.get(*index).cloned().flatten(),
            },
        }
    }

    fn resolve_boot_arg_endpoint_and_fields(
        static_endpoint: &Option<String>,
        endpoint_from_param: &Option<(usize, Option<String>)>,
        field_endpoints: &BTreeMap<String, EndpointSpec>,
        param_endpoints: &[Option<String>],
        param_field_endpoints: &[BTreeMap<String, String>],
    ) -> (Option<String>, BTreeMap<String, String>) {
        let ep = static_endpoint.clone().or_else(|| {
            endpoint_from_param.as_ref().and_then(|(idx, suffix)| {
                Self::resolve_endpoint_spec_with_bindings(
                    &EndpointSpec::Param {
                        index: *idx,
                        suffix: suffix.clone(),
                    },
                    param_endpoints,
                    param_field_endpoints,
                )
            })
        });
        let mut f_map = BTreeMap::new();
        if let Some((idx, suffix)) = endpoint_from_param {
            if let Some(src_fields) = param_field_endpoints.get(*idx) {
                match suffix {
                    None => {
                        for (k, v) in src_fields {
                            f_map.insert(k.clone(), v.clone());
                        }
                    }
                    Some(suf) => {
                        let prefix = format!("{suf}.");
                        for (k, v) in src_fields {
                            if let Some(rest) = k.strip_prefix(&prefix) {
                                f_map.insert(rest.to_string(), v.clone());
                            }
                        }
                    }
                }
            }
        }
        for (fname, fspec) in field_endpoints {
            if let Some(resolved_f) = Self::resolve_endpoint_spec_with_bindings(
                fspec,
                param_endpoints,
                param_field_endpoints,
            ) {
                f_map.insert(fname.clone(), resolved_f);
            }
        }
        (ep, f_map)
    }

    fn verify_byte_ring_spsc(&mut self) {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        enum ActorId {
            Task(usize),
            Isr(usize),
            MainPostBoot,
        }

        #[derive(Clone, Debug)]
        struct ActorAccess {
            actor_id: ActorId,
            actor_desc: String,
            method: String,
            span: Span,
            file: Option<String>,
        }

        let mut ops_by_caller: BTreeMap<String, Vec<RecordedByteRingOp>> = BTreeMap::new();
        for op in &self.recorded_byte_ring_ops {
            ops_by_caller
                .entry(op.caller.clone())
                .or_default()
                .push(op.clone());
        }
        let mut calls_by_caller: BTreeMap<String, Vec<RecordedCallArgs>> = BTreeMap::new();
        for call in &self.recorded_call_args {
            calls_by_caller
                .entry(call.caller.clone())
                .or_default()
                .push(call.clone());
        }

        type RingActorsMap = BTreeMap<String, Vec<ActorAccess>>;
        let mut producers_by_ring: RingActorsMap = BTreeMap::new();
        let mut consumers_by_ring: RingActorsMap = BTreeMap::new();

        let record_access =
            |map: &mut RingActorsMap, ring: String, access: ActorAccess| {
                let list = map.entry(ring).or_default();
                if !list.iter().any(|a| a.actor_id == access.actor_id) {
                    list.push(access);
                }
            };

        let mut actors: Vec<(
            ActorId,
            String,
            String,
            Vec<Option<String>>,
            Vec<BTreeMap<String, String>>,
            bool,
        )> = Vec::new();

        for inst in &self.spawned_tasks {
            let caller_key = format!("task::{}", inst.task_name);
            let field_eps = self
                .spawned_task_field_endpoints
                .get(&inst.instance_id)
                .cloned()
                .unwrap_or_else(|| vec![BTreeMap::new(); inst.endpoint_bindings.len()]);
            actors.push((
                ActorId::Task(inst.instance_id),
                format!("task '{}' (spawned via {})", inst.task_name, inst.format_provenance()),
                caller_key,
                inst.endpoint_bindings.clone(),
                field_eps,
                false,
            ));
        }

        for (idx, binding) in self.irq_bindings.iter().enumerate() {
            let caller_key = format!("isr::{}", binding.isr_name);
            let mut param_eps = Vec::with_capacity(binding.args.len());
            for arg in &binding.args {
                let ep = match arg {
                    IrqBindingArg::PrimAddr(name) => Some(name.clone()),
                    IrqBindingArg::SubsystemPath {
                        root, field_chain, ..
                    } => {
                        if field_chain.is_empty() {
                            Some(root.clone())
                        } else {
                            let chain = field_chain
                                .iter()
                                .map(|(f, _)| f.as_str())
                                .collect::<Vec<_>>()
                                .join(".");
                            Some(format!("{root}.{chain}"))
                        }
                    }
                    _ => None,
                };
                param_eps.push(ep);
            }
            let field_eps = vec![BTreeMap::new(); param_eps.len()];
            actors.push((
                ActorId::Isr(idx),
                format!("ISR '{}' (IRQ '{}')", binding.isr_name, binding.irq_name),
                caller_key,
                param_eps,
                field_eps,
                false,
            ));
        }

        let has_main_post_boot = ops_by_caller.contains_key("main::post_boot")
            || self
                .recorded_call_args
                .iter()
                .any(|c| c.caller == "main" && c.in_main_post_boot);
        if has_main_post_boot {
            actors.push((
                ActorId::MainPostBoot,
                "main() (post-boot)".to_string(),
                "main".to_string(),
                Vec::new(),
                Vec::new(),
                true,
            ));
        }

        for (actor_id, actor_desc, root_caller, root_params, root_fields, is_main_post) in actors {
            let mut visited: BTreeSet<(
                String,
                Vec<Option<String>>,
                Vec<BTreeMap<String, String>>,
            )> = BTreeSet::new();
            let mut worklist = vec![(root_caller.clone(), root_params, root_fields, is_main_post)];
            visited.insert((worklist[0].0.clone(), worklist[0].1.clone(), worklist[0].2.clone()));

            while let Some((caller, param_eps, param_fields, post_boot_only)) = worklist.pop() {
                let op_lookup_key = if post_boot_only {
                    "main::post_boot"
                } else {
                    caller.as_str()
                };
                if let Some(ops) = ops_by_caller.get(op_lookup_key) {
                    for op in ops {
                        let Some(concrete_ring) = Self::resolve_endpoint_spec_with_bindings(
                            &op.target,
                            &param_eps,
                            &param_fields,
                        ) else {
                            continue;
                        };
                        if concrete_ring == "<anon_endpoint>" {
                            continue;
                        }
                        let access = ActorAccess {
                            actor_id: actor_id.clone(),
                            actor_desc: actor_desc.clone(),
                            method: op.method.clone(),
                            span: op.span,
                            file: op.file.clone(),
                        };
                        match op.role {
                            ByteRingOpRole::Producer => {
                                record_access(&mut producers_by_ring, concrete_ring, access);
                            }
                            ByteRingOpRole::Consumer => {
                                record_access(&mut consumers_by_ring, concrete_ring, access);
                            }
                            ByteRingOpRole::Both => {
                                record_access(
                                    &mut producers_by_ring,
                                    concrete_ring.clone(),
                                    access.clone(),
                                );
                                record_access(&mut consumers_by_ring, concrete_ring, access);
                            }
                        }
                    }
                }

                if visited.len() >= 4096 {
                    continue;
                }
                if let Some(calls) = calls_by_caller.get(&caller) {
                    for call in calls {
                        if post_boot_only && !call.in_main_post_boot {
                            continue;
                        }
                        let mut child_eps = Vec::with_capacity(call.args.len());
                        let mut child_fields = Vec::with_capacity(call.args.len());
                        for arg in &call.args {
                            let (ep, f_map) = Self::resolve_boot_arg_endpoint_and_fields(
                                &arg.static_endpoint,
                                &arg.endpoint_from_param,
                                &arg.field_endpoints,
                                &param_eps,
                                &param_fields,
                            );
                            child_eps.push(ep);
                            child_fields.push(f_map);
                        }
                        let state = (call.callee.clone(), child_eps.clone(), child_fields.clone());
                        if visited.insert(state) {
                            worklist.push((call.callee.clone(), child_eps, child_fields, false));
                        }
                    }
                }
            }
        }

        let mut all_rings: BTreeSet<String> = producers_by_ring.keys().cloned().collect();
        all_rings.extend(consumers_by_ring.keys().cloned());

        for ring in all_rings {
            let disp_ring = self
                .display_names
                .get(&ring)
                .cloned()
                .unwrap_or_else(|| ring.clone());
            if let Some(prods) = producers_by_ring.get(&ring) {
                if prods.len() > 1 {
                    let first = &prods[0];
                    for extra in &prods[1..] {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::GlobalIsolationViolation,
                                extra.span,
                                format!(
                                    "byte_ring '{}' SPSC violation: multiple concurrent producers ('{}' and '{}' both call `{}`)",
                                    disp_ring, first.actor_desc, extra.actor_desc, extra.method
                                ),
                            )
                            .with_file(extra.file.clone())
                            .with_note(format!(
                                "first producer '{}' called `{}` at {}; `byte_ring` is strictly single-producer single-consumer (SPSC) — use a separate `byte_ring` per producer or an `inbox<T, N>` for MPSC",
                                first.actor_desc, first.method, first.span
                            )),
                        );
                    }
                }
            }
            if let Some(cons) = consumers_by_ring.get(&ring) {
                if cons.len() > 1 {
                    let first = &cons[0];
                    for extra in &cons[1..] {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::GlobalIsolationViolation,
                                extra.span,
                                format!(
                                    "byte_ring '{}' SPSC violation: multiple concurrent consumers ('{}' and '{}' both call `{}`)",
                                    disp_ring, first.actor_desc, extra.actor_desc, extra.method
                                ),
                            )
                            .with_file(extra.file.clone())
                            .with_note(format!(
                                "first consumer '{}' called `{}` at {}; `byte_ring` is strictly single-producer single-consumer (SPSC) — use a separate `byte_ring` per consumer",
                                first.actor_desc, first.method, first.span
                            )),
                        );
                    }
                }
            }
        }
    }

    fn unroll_boot_spawn_tree(&mut self) {
        self.spawned_tasks.clear();
        for t in self.tasks.values_mut() {
            t.instance_count = 0;
        }

        let has_main = self.functions.contains_key("main");
        let roots: Vec<String> = if has_main {
            vec!["main".to_string()]
        } else {
            let mut called_by_fn = BTreeSet::new();
            for e in &self.call_graph.edges {
                if e.caller_kind == CallableKind::Fn {
                    called_by_fn.insert(e.callee.clone());
                }
            }
            self.fn_decl_order
                .iter()
                .filter(|name| {
                    !called_by_fn.contains(*name)
                        && self
                            .call_graph
                            .nodes
                            .get(*name)
                            .is_some_and(|n| n.spawns_in_boot)
                })
                .cloned()
                .collect()
        };

        for root in roots {
            let mut path = vec![root.clone()];
            let mut visiting = BTreeSet::new();
            visiting.insert(root.clone());
            self.unroll_boot_fn(&root, &mut path, false, &[], &[], &mut visiting);
        }
    }

    fn unroll_irq_boot_tree(&mut self) {
        let mut called_by_fn = BTreeSet::new();
        for e in &self.call_graph.edges {
            if e.caller_kind == CallableKind::Fn {
                called_by_fn.insert(e.callee.clone());
            }
        }
        let has_main = self.functions.contains_key("main");
        let roots: Vec<String> = if has_main {
            vec!["main".to_string()]
        } else {
            self.fn_decl_order
                .iter()
                .filter(|name| {
                    !called_by_fn.contains(*name)
                        && self
                            .call_graph
                            .nodes
                            .get(*name)
                            .is_some_and(|n| n.attaches_in_boot)
                })
                .cloned()
                .collect()
        };

        let mut irq_mode_seen: BTreeMap<u32, (bool, String, Span)> = BTreeMap::new();
        let mut mixed_warned: BTreeSet<u32> = BTreeSet::new();
        let mut next_branch_id: usize = 0;
        let mut reached_fns: BTreeSet<String> = BTreeSet::new();

        for root in roots {
            let Some(nodes) = self.fn_irq_boot_nodes.get(&root).cloned() else {
                continue;
            };
            let mut visiting = BTreeSet::new();
            visiting.insert(root.clone());
            reached_fns.insert(root.clone());
            let initial_excl = BTreeSet::new();
            let _ = self.unroll_irq_nodes(
                &nodes,
                &[],
                None,
                &initial_excl,
                &mut visiting,
                &mut reached_fns,
                &mut irq_mode_seen,
                &mut mixed_warned,
                &mut next_branch_id,
            );
        }

        // Also validate any uncalled helper functions that directly/transitively call `attach`
        // (for instance `fn bad_helper(u32 x) { IRQ.attach(h(x)); } fn main() {}`).
        if has_main {
            let unreached: Vec<String> = self
                .fn_decl_order
                .iter()
                .filter(|name| {
                    !reached_fns.contains(*name)
                        && !called_by_fn.contains(*name)
                        && self
                            .call_graph
                            .nodes
                            .get(*name)
                            .is_some_and(|n| n.attaches_in_boot)
                })
                .cloned()
                .collect();
            for fname in unreached {
                if let Some(nodes) = self.fn_irq_boot_nodes.get(&fname).cloned() {
                    let mut visiting = BTreeSet::new();
                    visiting.insert(fname.clone());
                    let initial_excl = BTreeSet::new();
                    let _ = self.unroll_irq_nodes(
                        &nodes,
                        &[],
                        None,
                        &initial_excl,
                        &mut visiting,
                        &mut reached_fns,
                        &mut irq_mode_seen,
                        &mut mixed_warned,
                        &mut next_branch_id,
                    );
                }
            }
        }
    }

    fn irq_nodes_contain_attach(&self, nodes: &[IrqBootNode]) -> bool {
        for n in nodes {
            match n {
                IrqBootNode::Attach(_) => return true,
                IrqBootNode::Call { callee, .. } => {
                    if self
                        .call_graph
                        .nodes
                        .get(callee)
                        .is_some_and(|node| node.attaches_in_boot)
                    {
                        return true;
                    }
                }
                IrqBootNode::BranchGroup { arms } => {
                    if arms
                        .iter()
                        .any(|(_, sub)| self.irq_nodes_contain_attach(sub))
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn unroll_irq_nodes(
        &mut self,
        nodes: &[IrqBootNode],
        param_args: &[Option<IrqBindingArg>],
        cond_branch: Option<usize>,
        active_exclusive: &BTreeSet<u32>,
        visiting: &mut BTreeSet<String>,
        reached_fns: &mut BTreeSet<String>,
        irq_mode_seen: &mut BTreeMap<u32, (bool, String, Span)>,
        mixed_warned: &mut BTreeSet<u32>,
        next_branch_id: &mut usize,
    ) -> (bool, BTreeSet<u32>, BTreeSet<u32>) {
        let mut cur_excl = active_exclusive.clone();
        let mut ret_union = BTreeSet::new();
        for node in nodes {
            match node {
                IrqBootNode::Attach(att) => {
                    let mut resolved_args = Vec::with_capacity(att.args.len());
                    let mut args_ok = true;
                    for (spec_opt, arg_span, pname) in &att.args {
                        match spec_opt {
                            Some(BootAttachArgSpec::Static(arg)) => {
                                resolved_args.push(arg.clone());
                            }
                            Some(BootAttachArgSpec::Param { index, field_chain }) => {
                                if let Some(Some(forwarded)) = param_args.get(*index) {
                                    if field_chain.is_empty() {
                                        resolved_args.push(forwarded.clone());
                                    } else if let IrqBindingArg::SubsystemPath {
                                        root,
                                        root_struct,
                                        field_chain: mut base_chain,
                                    } = forwarded.clone()
                                    {
                                        base_chain.extend(field_chain.iter().cloned());
                                        resolved_args.push(IrqBindingArg::SubsystemPath {
                                            root,
                                            root_struct,
                                            field_chain: base_chain,
                                        });
                                    } else {
                                        self.emit(
                                            Diagnostic::new(
                                                DiagnosticKind::IrqAttachmentError,
                                                *arg_span,
                                                format!(
                                                    "parameter '{}' forwarded into '{}.attach({}(...))' in '{}' must resolve to a static subsystem `&inst`",
                                                    pname, att.irq_name, att.isr_name, att.in_caller
                                                ),
                                            )
                                            .with_file(att.file.clone()),
                                        );
                                        args_ok = false;
                                    }
                                } else {
                                    self.emit(
                                        Diagnostic::new(
                                            DiagnosticKind::IrqAttachmentError,
                                            *arg_span,
                                            format!(
                                                "parameter '{}' forwarded into '{}.attach({}(...))' in '{}' must be bound to a static primitive `&prim`, MMIO handle, or compile-time constant at every boot call site",
                                                pname, att.irq_name, att.isr_name, att.in_caller
                                            ),
                                        )
                                        .with_file(att.file.clone()),
                                    );
                                    args_ok = false;
                                }
                            }
                            None => {
                                args_ok = false;
                            }
                        }
                    }

                    if !att.invalid_ctx {
                        if let Some((prev_shared, prev_name, prev_span)) =
                            irq_mode_seen.get(&att.irq_num).cloned()
                        {
                            if prev_shared != att.is_shared && mixed_warned.insert(att.irq_num) {
                                self.emit(
                                    Diagnostic::new(
                                        DiagnosticKind::IrqAttachmentError,
                                        att.span,
                                        format!(
                                            "IRQ '{}' (#{}) cannot mix exclusive `.attach(...)` and shared `.attach_shared(...)`",
                                            att.irq_name, att.irq_num
                                        ),
                                    )
                                    .with_file(att.file.clone())
                                    .with_note(format!(
                                        "previously bound via `{}.{}(...)` at {}",
                                        prev_name,
                                        if prev_shared { "attach_shared" } else { "attach" },
                                        prev_span
                                    )),
                                );
                            }
                        } else {
                            irq_mode_seen.insert(
                                att.irq_num,
                                (att.is_shared, att.irq_name.clone(), att.span),
                            );
                        }

                        if !att.is_shared && !cur_excl.insert(att.irq_num) {
                            self.emit(
                                Diagnostic::new(
                                    DiagnosticKind::IrqAttachmentError,
                                    att.span,
                                    format!(
                                        "IRQ '{}' (#{}) already has an exclusive `.attach(...)` handler on this boot path",
                                        att.irq_name, att.irq_num
                                    ),
                                )
                                .with_file(att.file.clone())
                                .with_note(
                                    "use `.attach_shared(...)` for multi-handler interrupts or guard mutually exclusive variants in `if`/`else`/`switch` branches",
                                ),
                            );
                        } else if args_ok {
                            self.irq_bindings.push(IrqBindingInfo {
                                irq_name: att.irq_name.clone(),
                                irq_num: att.irq_num,
                                is_shared: att.is_shared,
                                isr_name: att.isr_name.clone(),
                                args: resolved_args,
                                conditional_branch: cond_branch,
                                span: att.span,
                            });
                        }
                    }
                }
                IrqBootNode::Call {
                    callee,
                    args,
                } => {
                    let attaches = self
                        .call_graph
                        .nodes
                        .get(callee)
                        .is_some_and(|n| n.attaches_in_boot);
                    if !attaches || !visiting.insert(callee.clone()) {
                        continue;
                    }
                    reached_fns.insert(callee.clone());
                    let child_params: Vec<Option<IrqBindingArg>> = args
                        .iter()
                        .map(|a| match &a.spec {
                            Some(BootAttachArgSpec::Static(s)) => Some(s.clone()),
                            Some(BootAttachArgSpec::Param { index, field_chain }) => {
                                let base = param_args.get(*index).cloned().flatten()?;
                                if field_chain.is_empty() {
                                    Some(base)
                                } else if let IrqBindingArg::SubsystemPath {
                                    root,
                                    root_struct,
                                    field_chain: mut base_chain,
                                } = base
                                {
                                    base_chain.extend(field_chain.iter().cloned());
                                    Some(IrqBindingArg::SubsystemPath {
                                        root,
                                        root_struct,
                                        field_chain: base_chain,
                                    })
                                } else {
                                    None
                                }
                            }
                            None => None,
                        })
                        .collect();
                    if let Some(child_nodes) = self.fn_irq_boot_nodes.get(callee).cloned() {
                        let (_child_div, mut child_fall, child_ret) = self.unroll_irq_nodes(
                            &child_nodes,
                            &child_params,
                            cond_branch,
                            &cur_excl,
                            visiting,
                            reached_fns,
                            irq_mode_seen,
                            mixed_warned,
                            next_branch_id,
                        );
                        // Returning inside a helper exits only that helper; execution resumes
                        // in this caller with the union of IRQs attached on all returning paths.
                        child_fall.extend(child_ret);
                        cur_excl = child_fall;
                        visiting.remove(callee);
                    } else {
                        visiting.remove(callee);
                    }
                }
                IrqBootNode::BranchGroup { arms } => {
                    let group_has_attach = arms
                        .iter()
                        .any(|(_, sub)| self.irq_nodes_contain_attach(sub));
                    if !group_has_attach {
                        continue;
                    }
                    let mut post_union = cur_excl.clone();
                    let mut all_arms_div = true;
                    let mut any_non_div = false;
                    for (arm_div, arm_nodes) in arms {
                        let arm_branch = if cond_branch.is_some() {
                            cond_branch
                        } else {
                            let id = *next_branch_id;
                            *next_branch_id += 1;
                            Some(id)
                        };
                        let (sub_div, arm_fall, arm_ret) = self.unroll_irq_nodes(
                            arm_nodes,
                            param_args,
                            arm_branch,
                            &cur_excl,
                            visiting,
                            reached_fns,
                            irq_mode_seen,
                            mixed_warned,
                            next_branch_id,
                        );
                        ret_union.extend(arm_ret);
                        if *arm_div || sub_div {
                            ret_union.extend(arm_fall);
                        } else {
                            all_arms_div = false;
                            if !any_non_div {
                                post_union = arm_fall;
                                any_non_div = true;
                            } else {
                                post_union.extend(arm_fall);
                            }
                        }
                    }
                    if all_arms_div {
                        return (true, BTreeSet::new(), ret_union);
                    }
                    cur_excl = post_union;
                }
            }
        }
        (false, cur_excl, ret_union)
    }

    fn unroll_boot_fn(
        &mut self,
        fn_name: &str,
        path: &mut Vec<String>,
        cond_active: bool,
        param_endpoints: &[Option<String>],
        param_field_endpoints: &[BTreeMap<String, String>],
        visiting: &mut BTreeSet<String>,
    ) {
        let Some(steps) = self.fn_boot_steps.get(fn_name).cloned() else {
            return;
        };
        for step in steps {
            match step {
                BootStep::DirectSpawn {
                    task_name,
                    priority,
                    core,
                    conditional,
                    arg_bindings,
                    span,
                } => {
                    let mut endpoint_bindings = Vec::with_capacity(arg_bindings.len());
                    let mut resolved_arg_fields = Vec::with_capacity(arg_bindings.len());
                    for b in &arg_bindings {
                        let (resolved, f_map) = Self::resolve_boot_arg_endpoint_and_fields(
                            &b.static_endpoint,
                            &b.endpoint_from_param,
                            &b.field_endpoints,
                            param_endpoints,
                            param_field_endpoints,
                        );
                        endpoint_bindings.push(resolved);
                        resolved_arg_fields.push(f_map);
                    }
                    let mut provenance_path = path.clone();
                    provenance_path.push(task_name.clone());
                    let instance_id = self.spawned_tasks.len();
                    if resolved_arg_fields.iter().any(|m| !m.is_empty()) {
                        self.spawned_task_field_endpoints
                            .insert(instance_id, resolved_arg_fields);
                    }
                    self.spawned_tasks.push(SpawnedTaskInstance {
                        instance_id,
                        task_name: task_name.clone(),
                        priority,
                        core,
                        spawned_in: fn_name.to_string(),
                        provenance_path,
                        conditional: cond_active || conditional,
                        endpoint_bindings,
                        span,
                    });
                    if let Some(t) = self.tasks.get_mut(&task_name) {
                        t.instance_count += 1;
                    }
                }
                BootStep::HelperCall {
                    callee,
                    conditional,
                    args,
                } => {
                    let spawns = self
                        .call_graph
                        .nodes
                        .get(&callee)
                        .is_some_and(|n| n.spawns_in_boot);
                    if !spawns || !visiting.insert(callee.clone()) {
                        continue;
                    }
                    let mut child_endpoints = Vec::with_capacity(args.len());
                    let mut child_field_endpoints = Vec::with_capacity(args.len());
                    for a in &args {
                        let (resolved, f_map) = Self::resolve_boot_arg_endpoint_and_fields(
                            &a.static_endpoint,
                            &a.endpoint_from_param,
                            &a.field_endpoints,
                            param_endpoints,
                            param_field_endpoints,
                        );
                        child_endpoints.push(resolved);
                        child_field_endpoints.push(f_map);
                    }
                    path.push(callee.clone());
                    self.unroll_boot_fn(
                        &callee,
                        path,
                        cond_active || conditional,
                        &child_endpoints,
                        &child_field_endpoints,
                        visiting,
                    );
                    path.pop();
                    visiting.remove(&callee);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-Function / Task / ISR Body Checker
// ─────────────────────────────────────────────────────────────────────────────

struct BodyChecker<'a> {
    sema: &'a mut Analyzer,
    caller_key: String,
    caller_kind: CallableKind,
    body_kind: ActiveBodyKind,
    prog_idx: usize,
    /// Scope stack: `scopes.len()` equals lexical scope depth (`1` = fn/task body, `2..N` = nested blocks).
    scopes: Vec<BTreeMap<String, LocalVar>>,
    leases: LeaseEnv,
    loop_depth: u32,
    noyield_depth: u32,
    critical_depth: u32,
    defer_depth: u32,
    branch_depth: u32,
    after_diverge: bool,
    main_supervisor_started: bool,
    has_spawned: bool,
    has_attached_irq: bool,
    param_names: Vec<String>,
    isr_state_names: BTreeSet<String>,
    boot_steps: Vec<BootStep>,
    boot_block_stack: Vec<Vec<IrqBootNode>>,
    return_param_origins: BTreeSet<usize>,
    recorded_boosts: Vec<PriorityBoostSite>,
    proven_infallible_spans: BTreeSet<(u32, u32)>,
    last_struct_lit_fields: BTreeMap<(u32, u32), BTreeMap<String, Provenance>>,
    mmio_array_expr_spans: BTreeMap<(u32, u32), (String, String, usize)>,
    /// Tracks lexical scope depths at loop entry and outer variables carrying DMA leases read inside the loop.
    loop_entry_depths: Vec<u32>,
    loop_outer_reads: Vec<Vec<(String, LeaseEnv, Provenance, Span)>>,
    /// Tracks loop-entry snapshots for outer `Lease<T>` move checking inside loops.
    loop_lease_frames: Vec<(u32, Vec<BTreeMap<String, LocalVar>>)>,
    /// When inside a `defer` statement, records the lexical scope depth owning that `defer`.
    active_defer_owner_depth: Option<u32>,
    next_lease_var_id: u64,
}

impl<'a> BodyChecker<'a> {
    fn new(
        sema: &'a mut Analyzer,
        caller_key: &str,
        caller_kind: CallableKind,
        body_kind: ActiveBodyKind,
    ) -> Self {
        let prog_idx = sema.current_prog_idx;
        Self {
            sema,
            caller_key: caller_key.to_string(),
            caller_kind,
            body_kind,
            prog_idx,
            scopes: vec![BTreeMap::new()], // depth 1 = top-level fn/task/isr body
            leases: LeaseEnv::new(),
            loop_depth: 0,
            noyield_depth: 0,
            critical_depth: 0,
            defer_depth: 0,
            branch_depth: 0,
            after_diverge: false,
            main_supervisor_started: false,
            has_spawned: false,
            has_attached_irq: false,
            param_names: Vec::new(),
            isr_state_names: BTreeSet::new(),
            boot_steps: Vec::new(),
            boot_block_stack: vec![Vec::new()],
            return_param_origins: BTreeSet::new(),
            recorded_boosts: Vec::new(),
            proven_infallible_spans: BTreeSet::new(),
            last_struct_lit_fields: BTreeMap::new(),
            mmio_array_expr_spans: BTreeMap::new(),
            loop_entry_depths: Vec::new(),
            loop_outer_reads: Vec::new(),
            loop_lease_frames: Vec::new(),
            active_defer_owner_depth: None,
            next_lease_var_id: 0,
        }
    }

    fn current_depth(&self) -> u32 {
        self.scopes.len() as u32
    }

    fn active_local_names(&self) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for scope in &self.scopes {
            names.extend(scope.keys().cloned());
        }
        names
    }

    fn in_isr(&self) -> bool {
        self.caller_kind == CallableKind::Isr
    }

    fn in_critical(&self) -> bool {
        self.critical_depth > 0
    }

    fn in_noyield(&self) -> bool {
        self.noyield_depth > 0
    }

    fn in_defer(&self) -> bool {
        self.defer_depth > 0
    }

    fn can_record_boot_step(&self) -> bool {
        self.caller_kind == CallableKind::Fn
            && self.loop_depth == 0
            && self.defer_depth == 0
            && !self.after_diverge
            && !self.main_supervisor_started
    }

    fn in_main_post_boot(&self) -> bool {
        self.caller_key == "main" && (self.has_spawned || self.has_attached_irq)
    }

    fn resolve_endpoint_spec(&self, ep: &str) -> EndpointSpec {
        if let Some(rest) = ep.strip_prefix("param:") {
            let (pname, suffix) = match rest.split_once('.') {
                Some((head, tail)) => (head, Some(tail.to_string())),
                None => (rest, None),
            };
            if let Some(idx) = self.param_names.iter().position(|n| n == pname) {
                return EndpointSpec::Param {
                    index: idx,
                    suffix,
                };
            }
        }
        EndpointSpec::Static(ep.to_string())
    }

    fn extract_field_endpoint_specs(
        &self,
        val: &ExprVal,
    ) -> BTreeMap<String, EndpointSpec> {
        let mut out = BTreeMap::new();
        for (fname, ep) in &val.field_endpoints {
            if ep != "<anon_endpoint>" {
                out.insert(fname.clone(), self.resolve_endpoint_spec(ep));
            }
        }
        out
    }

    fn record_byte_ring_op(
        &mut self,
        role: ByteRingOpRole,
        method: &str,
        source_key: &str,
        span: Span,
    ) {
        if source_key == "<anon_endpoint>" {
            return;
        }
        let caller = if self.caller_key == "main" {
            if !self.in_main_post_boot() {
                return;
            }
            "main::post_boot".to_string()
        } else {
            self.caller_key.clone()
        };
        let target = self.resolve_endpoint_spec(source_key);
        self.sema.recorded_byte_ring_ops.push(RecordedByteRingOp {
            caller,
            role,
            method: method.to_string(),
            target,
            span,
            file: self.sema.current_file.clone(),
        });
    }

    fn resolve_param_endpoint_index(
        &self,
        endpoint_name: &Option<String>,
    ) -> Option<(usize, Option<String>)> {
        let ep = endpoint_name.as_deref()?;
        let rest = ep.strip_prefix("param:")?;
        let (pname, suffix) = match rest.split_once('.') {
            Some((head, tail)) => (head, Some(tail.to_string())),
            None => (rest, None),
        };
        self.param_names
            .iter()
            .position(|n| n == pname)
            .map(|idx| (idx, suffix))
    }

    fn call_context(&self) -> CallContext {
        CallContext {
            in_noyield: self.in_noyield(),
            in_critical: self.in_critical(),
            in_isr: self.in_isr(),
            in_defer: self.in_defer(),
            in_loop: self.loop_depth > 0,
            after_diverge: self.after_diverge,
            after_main_supervisor: self.main_supervisor_started,
        }
    }

    fn display_caller_name(&self) -> &str {
        self.caller_key
            .strip_prefix("task::")
            .or_else(|| self.caller_key.strip_prefix("isr::"))
            .unwrap_or(&self.caller_key)
    }

    fn bind_params(&mut self, params: &[ParamInfo]) {
        self.param_names = params.iter().map(|p| p.name.clone()).collect();
        for (idx, p) in params.iter().enumerate() {
            let is_ptr = matches!(p.ty, Ty::Ptr(_));
            let endpoint_alias = if ty_contains_os_prim_view(&p.ty, &self.sema.structs, &mut BTreeSet::new()) {
                Some(format!("param:{}", p.name))
            } else {
                None
            };
            let prov = if p.ty.contains_reference(&self.sema.structs) {
                Provenance::for_param(idx, is_ptr, endpoint_alias.clone())
            } else if matches!(p.ty, Ty::FnPtr { .. }) {
                Provenance::pure_static()
            } else {
                Provenance::pure_static()
            };
            let (lease_state, lease_gen_key) = if p.ty.is_linear_lease() {
                if self.caller_kind == CallableKind::Fn {
                    let key = self.fresh_lease_gen_key(&p.name);
                    (Some(LinearLeaseState::Live), Some(key))
                } else {
                    (
                        Some(LinearLeaseState::Moved {
                            moved_at: p.span,
                            reason: "invalid param",
                        }),
                        None,
                    )
                }
            } else {
                (None, None)
            };
            self.scopes[0].insert(
                p.name.clone(),
                LocalVar {
                    name: p.name.clone(),
                    ty: p.ty.clone(),
                    decl_depth: 1,
                    prov,
                    field_prov: BTreeMap::new(),
                    endpoint_alias,
                    field_endpoints: BTreeMap::new(),
                    is_isr_state: false,
                    lease_state,
                    lease_gen_key,
                },
            );
        }
    }

    fn lookup_local(&self, name: &str) -> Option<&LocalVar> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v);
            }
        }
        None
    }

    fn lookup_local_mut(&mut self, name: &str) -> Option<&mut LocalVar> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(v) = scope.get_mut(name) {
                return Some(v);
            }
        }
        None
    }

    fn suggest_ident_hint(&self, needle: &str) -> Option<String> {
        match needle {
            "sleep_ms" | "usleep" | "msleep" => {
                return Some("did you mean `delay(10ms)`? (Zev uses typed duration literals like `1ms`, `500us`, `5s`)".to_string());
            }
            "yield_now" | "task_yield" | "co_yield" | "sched_yield" => {
                return Some("did you mean `checkpoint()`?".to_string());
            }
            "printf" | "log" => {
                return Some(
                    "did you mean `println(\"...\")`, `sys.log(\"...\")`, or `sys.log_kv(\"label=\", val)`?"
                        .to_string(),
                );
            }
            _ => {}
        }
        let mut pool: Vec<&str> = Vec::new();
        for scope in &self.scopes {
            pool.extend(scope.keys().map(String::as_str));
        }
        pool.extend(self.sema.primitives.keys().map(String::as_str));
        pool.extend(self.sema.consts.keys().map(String::as_str));
        pool.extend(
            self.sema
                .variant_to_enums
                .keys()
                .filter(|v| !self.sema.visible_enums_for_variant(self.prog_idx, v).is_empty())
                .map(String::as_str),
        );
        pool.extend(self.sema.functions.keys().map(String::as_str));
        pool.extend(["set_priority", "checkpoint", "delay", "sys", "sim"]);
        closest_match(needle, pool, 2).map(|m| format!("did you mean `{}`?", m))
    }

    fn suggest_struct_field_hint(&self, struct_name: &str, needle: &str) -> String {
        if let Some(st) = self.sema.structs.get(struct_name) {
            let names: Vec<&str> = st.fields.iter().map(|f| f.name.as_str()).collect();
            let avail = names.join(", ");
            if let Some(best) = closest_match(needle, names, 2) {
                return format!(
                    "did you mean `{}.{}`? (available fields on `{}`: {})",
                    struct_name, best, struct_name, avail
                );
            }
            return format!("available fields on `{}`: {}", struct_name, avail);
        }
        format!("check field definitions on struct `{}`", struct_name)
    }

    fn suggest_enum_variant_hint(&self, enum_name: &str, needle: &str) -> String {
        if let Some(en) = self.sema.enums.get(enum_name) {
            let names: Vec<&str> = en.variants.iter().map(String::as_str).collect();
            let avail = names.join(", ");
            if let Some(best) = closest_match(needle, names, 2) {
                return format!(
                    "did you mean `{}`? (variants of `{}`: {})",
                    best, enum_name, avail
                );
            }
            return format!("valid variants of `{}`: {}", enum_name, avail);
        }
        format!("check variant definitions on enum `{}`", enum_name)
    }

    fn record_direct_yield(&mut self, span: Span, tier: FnEffect, construct_desc: &str) {
        if self.in_isr() {
            let isr_name = self.display_caller_name().to_string();
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::YieldInIsr,
                span,
                format!("cannot yield via {} inside ISR '{}'", construct_desc, isr_name),
            ));
        } else if self.in_critical() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::YieldInCritical,
                span,
                format!(
                    "cannot yield via {} inside 'critical' block while hardware IRQs are masked",
                    construct_desc
                ),
            ));
        } else if self.in_defer() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::YieldInNoYield,
                span,
                format!(
                    "cannot yield via {} inside 'defer' cleanup statement",
                    construct_desc
                ),
            ));
        } else if self.in_noyield() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::YieldInNoYield,
                span,
                format!("cannot yield via {} inside 'noyield' block", construct_desc),
            ));
        } else {
            self.sema
                .call_graph
                .mark_direct_yield(&self.caller_key, tier, construct_desc);
        }
    }

    fn fresh_lease_gen_key(&mut self, var_name: &str) -> String {
        self.next_lease_var_id += 1;
        format!("{}#{}", var_name, self.next_lease_var_id)
    }

    fn display_lease_source(source: &str) -> &str {
        source.split('#').next().unwrap_or(source)
    }

    fn finish_lexical_block(&mut self, diverged: bool, block_span: Span) {
        let closing_depth = self.current_depth();
        if diverged {
            return;
        }
        if let Some(closing_scope) = self.scopes.last() {
            let unconsumed: Vec<String> = closing_scope
                .values()
                .filter(|v| matches!(v.lease_state, Some(LinearLeaseState::Live)))
                .map(|v| v.name.clone())
                .collect();
            for vname in unconsumed {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::LinearLeaseViolation,
                    block_span,
                    format!(
                        "linear Lease<T> variable '{}' was not consumed before end of scope",
                        vname
                    ),
                ));
            }
        }
        // Any outer variable whose `defer` was registered at `closing_depth` is released right here!
        let mut to_invalidate = Vec::new();
        for scope in &mut self.scopes {
            for var in scope.values_mut() {
                if let Some(LinearLeaseState::DeferredRelease { span, scope_depth }) =
                    var.lease_state
                {
                    if scope_depth == closing_depth {
                        var.lease_state = Some(LinearLeaseState::Moved {
                            moved_at: span,
                            reason: "defer release",
                        });
                        if let Some(ref key) = var.lease_gen_key {
                            to_invalidate.push((key.clone(), span));
                        }
                    }
                }
            }
        }
        for (key, sp) in to_invalidate {
            let _ = self.leases.invalidate(&key, sp, "defer release");
        }
    }

    fn check_unconsumed_leases_on_exit(&mut self, exit_span: Span, exit_kind: &str, min_depth_exclusive: u32) {
        let mut offenders = Vec::new();
        for scope in &self.scopes {
            for var in scope.values() {
                if var.decl_depth > min_depth_exclusive
                    && matches!(var.lease_state, Some(LinearLeaseState::Live))
                {
                    offenders.push(var.name.clone());
                }
            }
        }
        for name in offenders {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                exit_span,
                format!(
                    "cannot '{}' while linear Lease<T> variable '{}' is still live and unconsumed",
                    exit_kind, name
                ),
            ));
        }
    }

    fn consume_lease_value(
        &mut self,
        expr: &Expr,
        val: &ExprVal,
        reason: &'static str,
        use_span: Span,
    ) {
        if !val.ty.is_linear_lease() {
            return;
        }
        let ExprKind::Ident(ident) = &expr.kind else {
            return;
        };
        let Some(var) = self.lookup_local(&ident.name).cloned() else {
            return;
        };
        if !var.ty.is_linear_lease() {
            return;
        }
        match var.lease_state {
            Some(LinearLeaseState::Live) => {
                // Reject moving/releasing an outer variable that was already Live/Deferred at loop entry.
                for (entry_depth, entry_vars) in &self.loop_lease_frames {
                    if var.decl_depth <= *entry_depth {
                        let was_outer_live = entry_vars
                            .iter()
                            .rev()
                            .find_map(|s| s.get(&var.name))
                            .is_some_and(|ev| {
                                matches!(
                                    ev.lease_state,
                                    Some(
                                        LinearLeaseState::Live
                                            | LinearLeaseState::DeferredRelease { .. }
                                    )
                                )
                            });
                        if was_outer_live {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::LinearLeaseViolation,
                                    use_span,
                                    format!(
                                        "cannot move/consume outer live Lease<T> variable '{}' inside a loop",
                                        var.name
                                    ),
                                )
                                .with_note("release the outer Lease<T> after the loop exits (or allocate/release inside each loop iteration)"),
                            );
                            return;
                        }
                    }
                }

                if let Some(defer_owner) = self.active_defer_owner_depth {
                    if var.decl_depth <= defer_owner {
                        if let Some(v) = self.lookup_local_mut(&ident.name) {
                            v.lease_state = Some(LinearLeaseState::DeferredRelease {
                                span: use_span,
                                scope_depth: defer_owner,
                            });
                        }
                        return;
                    }
                }

                if let Some(v) = self.lookup_local_mut(&ident.name) {
                    v.lease_state = Some(LinearLeaseState::Moved {
                        moved_at: use_span,
                        reason,
                    });
                }
                if let Some(key) = var.lease_gen_key {
                    let _ = self.leases.invalidate(&key, use_span, reason);
                }
            }
            Some(LinearLeaseState::DeferredRelease { span, .. }) => {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        use_span,
                        format!(
                            "cannot move/consume Lease<T> variable '{}' via {}: `defer` release was already scheduled at {}",
                            var.name, reason, span
                        ),
                    )
                    .with_note("remove either the manual consume or `defer lease.release();` so it is released once"),
                );
            }
            Some(LinearLeaseState::Moved { .. }) | Some(LinearLeaseState::Uninit) | None => {
                // `check_ident_expr` already reported use-after-move / uninit if accessed as rvalue.
            }
        }
    }

    /// Push a nested lexical block (`depth + 1`), execute `f`, and pop the nested block.
    fn with_nested_scope<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.scopes.push(BTreeMap::new());
        let res = f(self);
        self.scopes.pop();
        res
    }

    fn snapshot_outer_vars(&self) -> Vec<BTreeMap<String, LocalVar>> {
        self.scopes.clone()
    }

    fn merge_branch_vars(
        &mut self,
        base: &[BTreeMap<String, LocalVar>],
        branches: &[Vec<BTreeMap<String, LocalVar>>],
        join_span: Span,
    ) {
        if branches.is_empty() {
            return;
        }
        for (depth_idx, scope) in self.scopes.iter_mut().enumerate() {
            if depth_idx >= base.len() {
                break;
            }
            for (name, var) in scope.iter_mut() {
                let mut merged_prov: Option<Provenance> = None;
                let mut merged_lease: Option<LinearLeaseState> = None;
                let mut lease_mismatch = false;
                for b in branches {
                    if let Some(b_var) = b.get(depth_idx).and_then(|s| s.get(name)) {
                        merged_prov = Some(match merged_prov {
                            None => b_var.prov.clone(),
                            Some(acc) => acc.merge(&b_var.prov),
                        });
                        if let Some(bs) = b_var.lease_state {
                            match merged_lease {
                                None => merged_lease = Some(bs),
                                Some(prev) => {
                                    let same = match (prev, bs) {
                                        (LinearLeaseState::Live, LinearLeaseState::Live)
                                        | (LinearLeaseState::Uninit, LinearLeaseState::Uninit)
                                        | (
                                            LinearLeaseState::Moved { .. },
                                            LinearLeaseState::Moved { .. },
                                        ) => true,
                                        (
                                            LinearLeaseState::DeferredRelease {
                                                scope_depth: d1,
                                                ..
                                            },
                                            LinearLeaseState::DeferredRelease {
                                                scope_depth: d2,
                                                ..
                                            },
                                        ) => d1 == d2,
                                        _ => false,
                                    };
                                    if !same {
                                        lease_mismatch = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(mp) = merged_prov {
                    var.prov = mp;
                }
                if lease_mismatch {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::LinearLeaseViolation,
                            join_span,
                            format!(
                                "conditional branches disagree on linear ownership state of Lease<T> variable '{}'",
                                name
                            ),
                        )
                        .with_note("every control-flow path reaching the branch merge must either leave the Lease<T> unconsumed or consume it along all branches"),
                    );
                    var.lease_state = Some(LinearLeaseState::Moved {
                        moved_at: join_span,
                        reason: "branch mismatch",
                    });
                } else if let Some(ml) = merged_lease {
                    var.lease_state = Some(ml);
                }
            }
        }
    }

    /// Analyze a slice of statements, returning `(diverged, warned_unreachable)`.
    fn check_stmt_slice(&mut self, stmts: &[Stmt]) -> (bool, bool) {
        let prev_after_diverge = self.after_diverge;
        let mut diverged = false;
        let mut warned_unreachable = false;
        for stmt in stmts {
            if diverged {
                if !warned_unreachable {
                    warned_unreachable = true;
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::UnreachableCode,
                        stmt.span,
                        "unreachable statement following diverging control flow",
                    ));
                }
                self.after_diverge = true;
            }
            if self.check_stmt(stmt) {
                diverged = true;
            }
        }
        self.after_diverge = prev_after_diverge;
        (diverged, warned_unreachable)
    }

    /// Analyze a list of statements in the current scope. Returns `true` if control flow diverges.
    fn check_block_stmts(&mut self, stmts: &[Stmt]) -> bool {
        self.check_stmt_slice(stmts).0
    }

    fn check_nested_block(&mut self, block: &Block) -> bool {
        self.with_nested_scope(|this| {
            let div = this.check_block_stmts(&block.stmts);
            this.finish_lexical_block(div, block.span);
            div
        })
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> bool {
        match &stmt.kind {
            StmtKind::VarDecl { ty, name, init } => {
                let resolved_ty = self.sema.resolve_type(ty);
                if resolved_ty.is_void() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        ty.span,
                        format!("local variable '{}' cannot have type 'void'", name.name),
                    ));
                }
                self.sema.check_type_no_byval_mmio(
                    &resolved_ty,
                    ty.span,
                    &format!("local variable '{}'", name.name),
                );
                self.sema.check_disallowed_lease_type(
                    &resolved_ty,
                    ty.span,
                    &format!("local variable '{}'", name.name),
                    true,
                );
                if matches!(
                    &resolved_ty,
                    Ty::OsPrimitive(
                        ResolvedOsPrimitive::Pool { .. } | ResolvedOsPrimitive::DmaSlot { .. }
                    )
                ) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        ty.span,
                        format!(
                            "static OS primitive '{}' cannot be declared as local variable '{}' (declare at module scope or in a subsystem struct)",
                            resolved_ty, name.name
                        ),
                    ));
                }
                if self.scopes.last().unwrap().contains_key(&name.name) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::DuplicateDefinition,
                        name.span,
                        format!("duplicate variable '{}' in the same scope", name.name),
                    ));
                }

                let decl_depth = self.current_depth();
                let mut prov = Provenance::pure_static();
                let mut field_prov = BTreeMap::new();
                let mut endpoint_alias = None;
                let mut field_endpoints = BTreeMap::new();
                let mut lease_state = if resolved_ty.is_linear_lease() {
                    Some(LinearLeaseState::Uninit)
                } else {
                    None
                };
                let mut lease_gen_key = None;

                if let Some(init_expr) = init {
                    let val = self.check_expr_infallible(init_expr);
                    if val.ty.is_linear_lease() {
                        self.consume_lease_value(
                            init_expr,
                            &val,
                            "variable initialization",
                            init_expr.span,
                        );
                    }
                    if matches!(init_expr.kind, ExprKind::StructLit { .. }) {
                        if let Some(fp) = self
                            .last_struct_lit_fields
                            .remove(&(init_expr.span.start, init_expr.span.end))
                        {
                            field_prov = fp;
                        }
                    }
                    field_endpoints = val.field_endpoints.clone();
                    if self.sema.check_assignable_or_emit(
                        &val.ty,
                        &resolved_ty,
                        init_expr.span,
                        DiagnosticKind::TypeMismatch,
                        || {
                            format!(
                                "variable '{}' declared as '{}', but initialized with '{}'",
                                name.name, resolved_ty, val.ty
                            )
                        },
                    ) {
                        self.sema.check_f32_coercion_expr(init_expr, &resolved_ty);
                    }
                    if resolved_ty.is_linear_lease() {
                        if val.ty.is_error() {
                            lease_state = Some(LinearLeaseState::Moved {
                                moved_at: init_expr.span,
                                reason: "error init",
                            });
                        } else {
                            lease_state = Some(LinearLeaseState::Live);
                            lease_gen_key = Some(self.fresh_lease_gen_key(&name.name));
                        }
                    }
                    if resolved_ty.contains_reference(&self.sema.structs)
                        || resolved_ty.contains_fn_ptr(&self.sema.structs)
                    {
                        if decl_depth < val.prov.origin_depth {
                            if !val.prov.isr_state_origins.is_empty() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::InvalidIsrState,
                                    init_expr.span,
                                    format!(
                                        "cannot escape `&` of ISR `state` variable into outer variable '{}'",
                                        name.name
                                    ),
                                ));
                            } else {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ScopeDepthEscape,
                                    init_expr.span,
                                    format!(
                                        "cannot store reference/slice of inner scope depth {} into variable '{}' at outer scope depth {}",
                                        val.prov.origin_depth, name.name, decl_depth
                                    ),
                                ));
                            }
                        }
                        prov = val.prov;
                    }
                    endpoint_alias = val.endpoint_name;
                }

                self.scopes.last_mut().unwrap().insert(
                    name.name.clone(),
                    LocalVar {
                        name: name.name.clone(),
                        ty: resolved_ty,
                        decl_depth,
                        prov,
                        field_prov,
                        endpoint_alias,
                        field_endpoints,
                        is_isr_state: false,
                        lease_state,
                        lease_gen_key,
                    },
                );
                false
            }

            StmtKind::StateVar { ty, name, init } => {
                if !matches!(self.body_kind, ActiveBodyKind::Isr) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        stmt.span,
                        format!(
                            "`state` variable '{}' is only allowed inside an `isr` handler",
                            name.name
                        ),
                    ));
                }

                let dup_in_isr = if matches!(self.body_kind, ActiveBodyKind::Isr) {
                    !self.isr_state_names.insert(name.name.clone())
                } else {
                    self.scopes.last().unwrap().contains_key(&name.name)
                };
                if dup_in_isr {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        name.span,
                        format!(
                            "duplicate `state` variable '{}' inside ISR '{}'",
                            name.name,
                            self.display_caller_name()
                        ),
                    ));
                } else if matches!(self.body_kind, ActiveBodyKind::Isr)
                    && self.param_names.iter().any(|p| p == &name.name)
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        name.span,
                        format!(
                            "`state` variable '{}' shadows parameter of ISR '{}'",
                            name.name,
                            self.display_caller_name()
                        ),
                    ));
                }

                let resolved_ty = self.sema.resolve_type(ty);
                if resolved_ty.is_void() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        ty.span,
                        format!("`state` variable '{}' cannot have type 'void'", name.name),
                    ));
                }
                self.sema.check_type_no_byval_mmio(
                    &resolved_ty,
                    ty.span,
                    &format!("`state` variable '{}'", name.name),
                );
                self.sema.check_disallowed_lease_type(
                    &resolved_ty,
                    ty.span,
                    &format!("ISR `state` variable '{}'", name.name),
                    false,
                );
                if resolved_ty.contains_reference(&self.sema.structs)
                    || resolved_ty.contains_fn_ptr(&self.sema.structs)
                    || matches!(
                        resolved_ty,
                        Ty::OsPrimitive(_) | Ty::MmioInstance { .. }
                    )
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        ty.span,
                        format!(
                            "`state` variable '{}' cannot have reference-bearing or handle type '{}'",
                            name.name, resolved_ty
                        ),
                    ));
                }

                match init {
                    None => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidIsrState,
                            stmt.span,
                            format!(
                                "`state` variable '{}' inside ISR requires a compile-time constant initializer",
                                name.name
                            ),
                        ));
                    }
                    Some(init_expr) => {
                        let val = self.check_expr_infallible(init_expr);
                        if !val.ty.is_assignable_to(&resolved_ty) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                init_expr.span,
                                format!(
                                    "`state` variable '{}' declared as '{}', but initialized with '{}'",
                                    name.name, resolved_ty, val.ty
                                ),
                            ));
                        } else {
                            self.sema.check_f32_coercion_expr(init_expr, &resolved_ty);
                        }
                        let shadowed = self.active_local_names();
                        if !self.sema.is_compile_time_const_expr_with_shadowed(
                            init_expr,
                            &shadowed,
                            self.prog_idx,
                        ) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidIsrState,
                                init_expr.span,
                                format!(
                                    "`state` variable '{}' initializer must be a compile-time constant expression",
                                    name.name
                                ),
                            ));
                        }
                    }
                }

                let decl_depth = self.current_depth();
                self.scopes.last_mut().unwrap().insert(
                    name.name.clone(),
                    LocalVar {
                        name: name.name.clone(),
                        ty: resolved_ty.clone(),
                        decl_depth,
                        prov: Provenance::pure_static(),
                        field_prov: BTreeMap::new(),
                        endpoint_alias: None,
                        field_endpoints: BTreeMap::new(),
                        is_isr_state: true,
                        lease_state: None,
                        lease_gen_key: None,
                    },
                );
                if matches!(self.body_kind, ActiveBodyKind::Isr) {
                    self.sema.isr_states.push(IsrStateInfo {
                        isr_name: self.display_caller_name().to_string(),
                        name: name.name.clone(),
                        ty: resolved_ty,
                        span: stmt.span,
                    });
                }
                false
            }
            StmtKind::Assign { target, op, value } => {
                let rhs = self.check_expr_infallible(value);
                self.check_assignment(target, *op, value, rhs, stmt.span);
                false
            }

            StmtKind::Return(opt_expr) => {
                if self.in_defer() {
                    if let Some(e) = opt_expr {
                        let _ = self.check_expr_infallible(e);
                    }
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'return' cannot be used inside a 'defer' cleanup statement",
                    ));
                    return true;
                }
                match &self.body_kind.clone() {
                    ActiveBodyKind::Isr => {
                        if let Some(e) = opt_expr {
                            let _ = self.check_expr_infallible(e);
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::ReturnMismatch,
                                stmt.span,
                                "ISR handlers cannot return a value",
                            ));
                        }
                        self.check_unconsumed_leases_on_exit(stmt.span, "return", 0);
                    }
                    ActiveBodyKind::Task => {
                        if let Some(e) = opt_expr {
                            let _ = self.check_expr_infallible(e);
                        }
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::NonDivergingTask,
                            stmt.span,
                            format!(
                                "cannot 'return' from task '{}'; tasks are permanent actors (keep running inside 'loop' / 'loop every', park on 'inbox.recv()' / 'signal.wait()', reset via 'restart;', or move finite workflows into yielding 'fn's)",
                                self.display_caller_name()
                            ),
                        ));
                    }
                    ActiveBodyKind::Fn { ret_ty, .. } => {
                        match (ret_ty.is_void(), opt_expr) {
                            (true, None) => {}
                            (true, Some(e)) => {
                                let val = self.check_expr_infallible(e);
                                if val.ty.is_linear_lease() {
                                    self.consume_lease_value(e, &val, "return", e.span);
                                }
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ReturnMismatch,
                                    stmt.span,
                                    format!(
                                        "void function '{}' cannot return value of type '{}'",
                                        self.caller_key, val.ty
                                    ),
                                ));
                            }
                            (false, None) => {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ReturnMismatch,
                                    stmt.span,
                                    format!(
                                        "function '{}' must return a value of type '{}'",
                                        self.caller_key, ret_ty
                                    ),
                                ));
                            }
                            (false, Some(e)) => {
                                let val = self.check_expr_infallible(e);
                                if val.ty.is_linear_lease() {
                                    self.consume_lease_value(e, &val, "return", e.span);
                                }
                                if self.sema.check_assignable_or_emit(
                                    &val.ty,
                                    ret_ty,
                                    e.span,
                                    DiagnosticKind::ReturnMismatch,
                                    || {
                                        format!(
                                            "return value has type '{}', expected '{}'",
                                            val.ty, ret_ty
                                        )
                                    },
                                ) {
                                    self.sema.check_f32_coercion_expr(e, ret_ty);
                                }
                                if ret_ty.contains_reference(&self.sema.structs) {
                                    if val.prov.has_non_returning_origin() {
                                        self.sema.emit(
                                            Diagnostic::new(
                                                DiagnosticKind::ReturnLocalReference,
                                                e.span,
                                                format!(
                                                    "cannot return pointer/slice/wrapper of type '{}' referencing local variable or temporary DMA lease from '{}'",
                                                    ret_ty, self.caller_key
                                                ),
                                            )
                                            .with_note(format!(
                                                "referenced origins: {:?}",
                                                val.prov.local_origins
                                            )),
                                        );
                                    } else {
                                        for &idx in &val.prov.param_origins {
                                            self.return_param_origins.insert(idx);
                                        }
                                    }
                                }
                            }
                        }
                        self.check_unconsumed_leases_on_exit(stmt.span, "return", 0);
                    }
                }
                true
            }

            StmtKind::Error { reason, partial } => {
                let r_val = reason.as_ref().map(|r| self.check_expr_infallible(r));
                let p_val = partial.as_ref().map(|p| self.check_expr_infallible(p));

                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'error' exit cannot be used inside a 'defer' cleanup statement",
                    ));
                    return true;
                }

                match &self.body_kind.clone() {
                    ActiveBodyKind::Task | ActiveBodyKind::Isr => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::ErrorSpecMismatch,
                            stmt.span,
                            format!(
                                "'error' exit statement is only valid inside a function declared with 'error', not inside '{}'",
                                self.display_caller_name()
                            ),
                        ));
                    }
                    ActiveBodyKind::Fn { ret_ty, error_spec } => match (error_spec, &r_val, &p_val)
                    {
                        (ResolvedErrorSpec::None, _, _) => {
                            let sig_example = if ret_ty.is_void() {
                                format!("fn {}(...) error ErrEnum", self.caller_key)
                            } else {
                                format!("{} {}(...) error ErrEnum", ret_ty, self.caller_key)
                            };
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    stmt.span,
                                    format!(
                                        "function '{}' is not declared with 'error' exit specification",
                                        self.caller_key
                                    ),
                                )
                                .with_note(format!(
                                    "add `error` (for `error;`) or `error ErrEnum` (for `error Reason;` / `error Reason, partial;`) to signature: `{}`",
                                    sig_example
                                )),
                            );
                        }
                        // 0-arg `error;`
                        (ResolvedErrorSpec::Untyped, None, None) => {}
                        (ResolvedErrorSpec::Untyped, Some(_), _) => {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    stmt.span,
                                    format!(
                                        "function '{}' declares bare 'error' without an error payload type; use 'error;'",
                                        self.caller_key
                                    ),
                                )
                                .with_note(format!(
                                    "either exit with `error;` (0 args), or declare typed `{}` to exit with `error Reason;`",
                                    if ret_ty.is_void() {
                                        format!("fn {}(...) error ErrEnum", self.caller_key)
                                    } else {
                                        format!("{} {}(...) error ErrEnum", ret_ty, self.caller_key)
                                    }
                                )),
                            );
                        }
                        (ResolvedErrorSpec::Typed(err_ty), None, _) => {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    stmt.span,
                                    format!(
                                        "function '{}' declares typed 'error {}' and requires an error reason ('error <{}>;')",
                                        self.caller_key, err_ty, err_ty
                                    ),
                                )
                                .with_note(format!(
                                    "use `error <{} variant>;` (or `error <{} variant>, partial_val;` to report partial progress)",
                                    err_ty, err_ty
                                )),
                            );
                        }
                        // 1-arg `error Reason;`
                        (ResolvedErrorSpec::Typed(err_ty), Some(rv), None) => {
                            if !rv.ty.is_assignable_to(err_ty) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    rv.span,
                                    format!(
                                        "error reason has type '{}', expected '{}'",
                                        rv.ty, err_ty
                                    ),
                                ));
                            }
                        }
                        // 2-arg `error Reason, partial_val;`
                        (ResolvedErrorSpec::Typed(err_ty), Some(rv), Some(pv)) => {
                            if pv.ty.is_linear_lease() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::LinearLeaseViolation,
                                    pv.span,
                                    "cannot return linear 'Lease<T>' as partial-progress payload on 'error' exit",
                                ));
                            }
                            if !rv.ty.is_assignable_to(err_ty) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    rv.span,
                                    format!(
                                        "error reason has type '{}', expected '{}'",
                                        rv.ty, err_ty
                                    ),
                                ));
                            }
                            if ret_ty.is_void() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    pv.span,
                                    format!(
                                        "void function '{}' cannot return a partial-progress value on 'error'",
                                        self.caller_key
                                    ),
                                ));
                            } else if !pv.ty.is_assignable_to(ret_ty) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    pv.span,
                                    format!(
                                        "error partial-progress value has type '{}', expected '{}'",
                                        pv.ty, ret_ty
                                    ),
                                ));
                            } else {
                                if let Some(p_expr) = partial {
                                    self.sema.check_f32_coercion_expr(p_expr, ret_ty);
                                }
                                if ret_ty.contains_reference(&self.sema.structs)
                                    && pv.prov.has_non_returning_origin()
                                {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::ReturnLocalReference,
                                        pv.span,
                                        format!(
                                            "cannot return partial-progress value of type '{}' referencing local variable or DMA lease",
                                            ret_ty
                                        ),
                                    ));
                                }
                            }
                        }
                        (ResolvedErrorSpec::Untyped, None, Some(_)) => {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::ErrorSpecMismatch,
                                stmt.span,
                                "partial-progress error requires a typed 'error ErrType' signature",
                            ));
                        }
                    },
                }
                self.check_unconsumed_leases_on_exit(stmt.span, "error", 0);
                true
            }

            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                let cv = self.check_expr_infallible(cond);
                if !cv.ty.is_bool() && !cv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        cond.span,
                        format!("'if' condition must be 'bool', found '{}'", cv.ty),
                    ));
                }

                let base_vars = self.snapshot_outer_vars();
                let base_leases = self.leases.clone();

                self.branch_depth += 1;
                self.boot_block_stack.push(Vec::new());
                let then_div = self.check_nested_block(then_block);
                let then_irq_nodes = self.boot_block_stack.pop().unwrap_or_default();
                self.branch_depth -= 1;
                let then_vars = self.snapshot_outer_vars();
                let then_leases = self.leases.clone();

                self.scopes = base_vars.clone();
                self.leases = base_leases.clone();

                if let Some(eb) = else_block {
                    self.branch_depth += 1;
                    self.boot_block_stack.push(Vec::new());
                    let else_div = self.check_nested_block(eb);
                    let else_irq_nodes = self.boot_block_stack.pop().unwrap_or_default();
                    self.branch_depth -= 1;
                    let else_vars = self.snapshot_outer_vars();
                    let else_leases = self.leases.clone();

                    if self.can_record_boot_step()
                        && (!then_irq_nodes.is_empty() || !else_irq_nodes.is_empty())
                    {
                        if let Some(top) = self.boot_block_stack.last_mut() {
                            top.push(IrqBootNode::BranchGroup {
                                arms: vec![
                                    (then_div, then_irq_nodes),
                                    (else_div, else_irq_nodes),
                                ],
                            });
                        }
                    }

                    self.scopes = base_vars.clone();
                    let mut live_var_branches = Vec::new();
                    let mut live_lease_branches = Vec::new();
                    if !then_div {
                        live_var_branches.push(then_vars);
                        live_lease_branches.push(then_leases);
                    }
                    if !else_div {
                        live_var_branches.push(else_vars);
                        live_lease_branches.push(else_leases);
                    }
                    self.merge_branch_vars(&base_vars, &live_var_branches, stmt.span);
                    self.leases = base_leases;
                    self.leases.join_branches(&live_lease_branches);
                    then_div && else_div
                } else {
                    if self.can_record_boot_step() && !then_irq_nodes.is_empty() {
                        if let Some(top) = self.boot_block_stack.last_mut() {
                            top.push(IrqBootNode::BranchGroup {
                                arms: vec![(then_div, then_irq_nodes), (false, Vec::new())],
                            });
                        }
                    }
                    let mut live_var_branches = vec![base_vars.clone()];
                    let mut live_lease_branches = vec![base_leases.clone()];
                    if !then_div {
                        live_var_branches.push(then_vars);
                        live_lease_branches.push(then_leases);
                    }
                    self.merge_branch_vars(&base_vars, &live_var_branches, stmt.span);
                    self.leases = base_leases;
                    self.leases.join_branches(&live_lease_branches);
                    false
                }
            }

            StmtKind::While { cond, body } => {
                let is_while_true = matches!(cond.kind, ExprKind::Literal(Literal::Bool(true)));
                let manual_yield =
                    has_attr_manual_yield(&stmt.attrs) || has_attr_noyield(&stmt.attrs);
                let has_break = block_has_break(body);

                let cv = self.check_expr_infallible(cond);
                if !cv.ty.is_bool() && !cv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        cond.span,
                        format!("'while' condition must be 'bool', found '{}'", cv.ty),
                    ));
                }

                if is_while_true {
                    if self.in_critical() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::UnboundedLoopInCritical,
                            stmt.span,
                            "unbounded loop ('while (true)') is forbidden inside 'critical' block while hardware IRQs are masked",
                        ));
                    } else if self.in_isr() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::YieldInIsr,
                            stmt.span,
                            format!(
                                "unbounded 'while (true)' loop is forbidden inside ISR '{}'",
                                self.display_caller_name()
                            ),
                        ));
                    } else if self.in_defer() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::YieldInNoYield,
                            stmt.span,
                            "unbounded 'while (true)' loop is forbidden inside 'defer' cleanup statement",
                        ));
                    } else if self.in_noyield() {
                        if !manual_yield {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::YieldInNoYield,
                                stmt.span,
                                "auto-yield 'while (true)' loop is forbidden inside 'noyield' block",
                            ));
                        } else if !has_break {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::YieldInNoYield,
                                stmt.span,
                                "infinite 'while (true)' without 'break' inside 'noyield' block would permanently starve the cooperative scheduler",
                            ));
                        }
                    } else if !has_break {
                        self.sema.call_graph.mark_direct_yield(
                            &self.caller_key,
                            FnEffect::HardBlocking,
                            "while (true) (infinite loop)",
                        );
                    } else if !manual_yield {
                        self.sema.call_graph.mark_direct_yield(
                            &self.caller_key,
                            FnEffect::SoftYielding,
                            "while (true) (should_share)",
                        );
                    }
                }

                if is_while_true && !has_break {
                    self.check_unconsumed_leases_on_exit(stmt.span, "enter infinite while loop", 0);
                }
                self.check_loop_body(body, is_while_true && !has_break);
                if is_while_true && self.caller_key == "main" {
                    self.main_supervisor_started = true;
                }
                is_while_true && !has_break
            }

            StmtKind::Loop { body } => {
                let manual_yield =
                    has_attr_manual_yield(&stmt.attrs) || has_attr_noyield(&stmt.attrs);
                let has_break = block_has_break(body);

                if self.in_critical() {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UnboundedLoopInCritical,
                            stmt.span,
                            "unbounded 'loop' is forbidden inside 'critical' block (IRQ-masked blocks must stay bounded)",
                        )
                        .with_note("move the loop outside 'critical' or use a bounded 'for..in' / 'while' loop"),
                    );
                } else if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::YieldInIsr,
                        stmt.span,
                        format!(
                            "unbounded / auto-yield 'loop' is forbidden inside ISR '{}'",
                            self.display_caller_name()
                        ),
                    ));
                } else if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::YieldInNoYield,
                        stmt.span,
                        "'loop' is forbidden inside 'defer' cleanup statement",
                    ));
                } else if self.in_noyield() {
                    if !manual_yield {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::YieldInNoYield,
                                stmt.span,
                                "auto-yield 'loop' is forbidden inside 'noyield' block",
                            )
                            .with_note(
                                "use a bounded 'for..in' / 'while' loop or '@manual_yield' with an explicit 'break'",
                            ),
                        );
                    } else if !has_break {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::YieldInNoYield,
                            stmt.span,
                            "infinite 'loop' without 'break' inside 'noyield' block would permanently starve the cooperative scheduler",
                        ));
                    }
                } else if !has_break {
                    self.sema.call_graph.mark_direct_yield(
                        &self.caller_key,
                        FnEffect::HardBlocking,
                        "loop (infinite)",
                    );
                } else if !manual_yield {
                    self.sema.call_graph.mark_direct_yield(
                        &self.caller_key,
                        FnEffect::SoftYielding,
                        "loop (should_share)",
                    );
                }

                if !has_break {
                    self.check_unconsumed_leases_on_exit(stmt.span, "enter infinite loop", 0);
                }
                self.check_loop_body(body, !has_break);
                if self.caller_key == "main" {
                    self.main_supervisor_started = true;
                }
                !has_break
            }

            StmtKind::LoopEvery { period, body } => {
                let pv = self.check_expr_infallible(period);
                if pv.ty.is_integer() {
                    let is_zero = match &period.kind {
                        ExprKind::Literal(Literal::Int(0)) => true,
                        ExprKind::Ident(id) => self
                            .sema
                            .consts
                            .get(&id.name)
                            .is_some_and(|c| c.int_value == Some(0)),
                        _ => false,
                    };
                    if is_zero {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidDurationResolution,
                            period.span,
                            "periodic 'loop every' interval must be greater than 0",
                        ));
                    }
                } else if pv.ty != Ty::Duration && !pv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        period.span,
                        format!(
                            "'loop every' period must be a duration (e.g. 1ms) or integer expression, found '{}'",
                            pv.ty
                        ),
                    ));
                }

                self.record_direct_yield(
                    stmt.span,
                    FnEffect::HardBlocking,
                    "'loop every' periodic deadline",
                );
                let has_break = block_has_break(body);
                if !has_break {
                    self.check_unconsumed_leases_on_exit(
                        stmt.span,
                        "enter infinite loop every",
                        0,
                    );
                }
                self.check_loop_body(body, !has_break);
                if self.caller_key == "main" {
                    self.main_supervisor_started = true;
                }
                !has_break
            }

            StmtKind::ForRange {
                var_ty,
                var_name,
                start,
                end,
                inclusive: _,
                body,
            } => {
                let resolved_var_ty = self.sema.resolve_type(var_ty);
                if !resolved_var_ty.is_integer() && !resolved_var_ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        var_ty.span,
                        format!(
                            "'for' range loop variable '{}' must have an integer type, found '{}'",
                            var_name.name, resolved_var_ty
                        ),
                    ));
                }
                let sv = self.check_expr_infallible(start);
                if !sv.ty.is_assignable_to(&resolved_var_ty)
                    && !resolved_var_ty.is_error()
                    && !sv.ty.is_error()
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        start.span,
                        format!(
                            "'for' range start has type '{}', expected '{}'",
                            sv.ty, resolved_var_ty
                        ),
                    ));
                }
                let ev = self.check_expr_infallible(end);
                if !ev.ty.is_assignable_to(&resolved_var_ty)
                    && !resolved_var_ty.is_error()
                    && !ev.ty.is_error()
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        end.span,
                        format!(
                            "'for' range end has type '{}', expected '{}'",
                            ev.ty, resolved_var_ty
                        ),
                    ));
                }

                self.check_for_loop_with_binding(
                    &var_name.name,
                    resolved_var_ty,
                    Provenance::pure_static(),
                    iter_span_none(),
                    body,
                );
                false
            }

            StmtKind::ForEach {
                elem_ty,
                elem_name,
                iter,
                body,
            } => {
                let resolved_elem_ty = self.sema.resolve_type(elem_ty);
                let iv = self.check_expr_infallible(iter);
                let deref_iter_ty = match &iv.ty {
                    Ty::Ptr(inner) => (**inner).clone(),
                    other => other.clone(),
                };
                match &deref_iter_ty {
                    Ty::Slice(seq_elem) | Ty::Array { elem: seq_elem, .. } => {
                        if !seq_elem.is_assignable_to(&resolved_elem_ty)
                            && !resolved_elem_ty.is_error()
                            && !seq_elem.is_error()
                        {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                elem_ty.span,
                                format!(
                                    "'for..in' element '{}' declared as '{}', but iterable yields '{}'",
                                    elem_name.name, resolved_elem_ty, seq_elem
                                ),
                            ));
                        }
                    }
                    _ => {
                        if !iv.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                iter.span,
                                format!(
                                    "cannot iterate over type '{}' in 'for..in' loop (expected slice '[]T' or array '[N]T')",
                                    iv.ty
                                ),
                            ));
                        }
                    }
                }

                self.check_for_loop_with_binding(
                    &elem_name.name,
                    resolved_elem_ty,
                    iv.prov,
                    Some(iter.span),
                    body,
                );
                false
            }

            StmtKind::Defer(inner) => {
                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "nested 'defer' inside a 'defer' statement is not allowed",
                    ));
                }
                if matches!(inner.kind, StmtKind::VarDecl { .. }) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        inner.span,
                        "cannot declare a standalone variable directly inside 'defer'; wrap cleanup statements in 'defer { ... }' if temporaries are needed",
                    ));
                }
                self.defer_depth += 1;
                self.noyield_depth += 1;
                let prev_defer_owner = self.active_defer_owner_depth;
                self.active_defer_owner_depth = Some(self.current_depth());
                let _ = self.with_nested_scope(|this| {
                    let div = this.check_stmt(inner);
                    this.finish_lexical_block(div, inner.span);
                    div
                });
                self.active_defer_owner_depth = prev_defer_owner;
                self.noyield_depth -= 1;
                self.defer_depth -= 1;
                false
            }

            StmtKind::Restart => {
                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'restart;' cannot be used inside a 'defer' cleanup statement",
                    ));
                } else if !matches!(self.body_kind, ActiveBodyKind::Task) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        format!(
                            "'restart;' is only valid inside a 'task' body, not inside '{}'",
                            self.display_caller_name()
                        ),
                    ));
                } else {
                    self.record_direct_yield(stmt.span, FnEffect::SoftYielding, "'restart;'");
                }
                self.check_unconsumed_leases_on_exit(stmt.span, "restart", 0);
                true
            }

            StmtKind::NoYield { body } => {
                self.noyield_depth += 1;
                let div = self.check_nested_block(body);
                self.noyield_depth -= 1;
                div
            }

            StmtKind::Critical { body } => {
                self.critical_depth += 1;
                self.noyield_depth += 1;
                let div = self.check_nested_block(body);
                self.noyield_depth -= 1;
                self.critical_depth -= 1;
                div
            }

            StmtKind::Switch { scrutinee, arms } => {
                let sv = self.check_expr_infallible(scrutinee);
                let base_vars = self.snapshot_outer_vars();
                let base_leases = self.leases.clone();
                let mut live_var_branches = Vec::new();
                let mut live_lease_branches = Vec::new();
                let mut has_wildcard = false;
                let mut covered_variants = BTreeSet::new();
                let mut all_arms_diverge = !arms.is_empty();
                let mut arm_irq_nodes: Vec<(bool, Vec<IrqBootNode>)> =
                    Vec::with_capacity(arms.len());

                for arm in arms {
                    match &arm.pat {
                        SwitchPat::Wildcard(_) => {
                            has_wildcard = true;
                        }
                        SwitchPat::Literal(lit, pat_span) => {
                            let lit_ty = match lit {
                                Literal::Int(_) => Ty::IntLit,
                                Literal::Float { suffix: Some(p), .. } => Ty::Prim(*p),
                                Literal::Float { suffix: None, .. } => Ty::FloatLit,
                                Literal::Bool(_) => Ty::bool(),
                                Literal::Duration(d) => {
                                    self.sema.validate_duration(d);
                                    Ty::Duration
                                }
                                Literal::String(_) => Ty::Str,
                            };
                            if !lit_ty.is_assignable_to(&sv.ty) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    *pat_span,
                                    format!(
                                        "switch pattern has type '{}', expected '{}'",
                                        lit_ty, sv.ty
                                    ),
                                ));
                            }
                        }
                        SwitchPat::Ident {
                            prefix,
                            name,
                            span: pat_span,
                        } => {
                            if let Ty::Enum(enum_name) = &sv.ty {
                                if let Some(pref) = prefix {
                                    if pref.name != *enum_name {
                                        self.sema.emit(Diagnostic::new(
                                            DiagnosticKind::TypeMismatch,
                                            pref.span,
                                            format!(
                                                "switch pattern enum '{}' does not match scrutinee enum '{}'",
                                                pref.name, enum_name
                                            ),
                                        ));
                                    }
                                }
                                if self
                                    .sema
                                    .enums
                                    .get(enum_name)
                                    .is_some_and(|e| e.has_variant(&name.name))
                                {
                                    covered_variants.insert(name.name.clone());
                                } else {
                                    let hint = self.suggest_enum_variant_hint(enum_name, &name.name);
                                    self.sema.emit(
                                        Diagnostic::new(
                                            DiagnosticKind::UndefinedSymbol,
                                            *pat_span,
                                            format!(
                                                "unknown variant '{}' for enum '{}'",
                                                name.name, enum_name
                                            ),
                                        )
                                        .with_note(hint),
                                    );
                                }
                            } else if let Some(pref) = prefix {
                                if !sv.ty.is_error() {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::UndefinedSymbol,
                                        *pat_span,
                                        format!("unknown switch pattern '{}.{}'", pref.name, name.name),
                                    ));
                                }
                            } else if let Some(c) = self.sema.consts.get(&name.name) {
                                if !c.ty.is_assignable_to(&sv.ty) {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::TypeMismatch,
                                        *pat_span,
                                        format!(
                                            "switch constant '{}' has type '{}', expected '{}'",
                                            name.name, c.ty, sv.ty
                                        ),
                                    ));
                                }
                            } else if !sv.ty.is_error() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::UndefinedSymbol,
                                    *pat_span,
                                    format!("unknown switch pattern '{}'", name.name),
                                ));
                            }
                        }
                        SwitchPat::Qualified { path, span: pat_span } => {
                            if !sv.ty.is_error() {
                                let rendered = path
                                    .iter()
                                    .map(|s| s.name.as_str())
                                    .collect::<Vec<_>>()
                                    .join(".");
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::UndefinedSymbol,
                                    *pat_span,
                                    format!("unknown qualified switch pattern '{rendered}'"),
                                ));
                            }
                        }
                    }

                    self.scopes = base_vars.clone();
                    self.leases = base_leases.clone();
                    self.branch_depth += 1;
                    self.boot_block_stack.push(Vec::new());
                    let arm_div = self.check_nested_block(&arm.body);
                    let this_arm_irq_nodes = self.boot_block_stack.pop().unwrap_or_default();
                    self.branch_depth -= 1;
                    arm_irq_nodes.push((arm_div, this_arm_irq_nodes));
                    if !arm_div {
                        all_arms_diverge = false;
                        live_var_branches.push(self.snapshot_outer_vars());
                        live_lease_branches.push(self.leases.clone());
                    }
                }

                let exhaustive = has_wildcard
                    || matches!(&sv.ty, Ty::Enum(en) if self.sema.enums.get(en).is_some_and(|e| e.variants.iter().all(|v| covered_variants.contains(v))));

                if !exhaustive {
                    all_arms_diverge = false;
                    live_var_branches.push(base_vars.clone());
                    live_lease_branches.push(base_leases.clone());
                    arm_irq_nodes.push((false, Vec::new()));
                }

                if self.can_record_boot_step()
                    && arm_irq_nodes.iter().any(|(_, n)| !n.is_empty())
                {
                    if let Some(top) = self.boot_block_stack.last_mut() {
                        top.push(IrqBootNode::BranchGroup {
                            arms: arm_irq_nodes,
                        });
                    }
                }

                self.scopes = base_vars.clone();
                self.merge_branch_vars(&base_vars, &live_var_branches, stmt.span);
                self.leases = base_leases;
                self.leases.join_branches(&live_lease_branches);
                all_arms_diverge
            }

            StmtKind::Spawn(sp) => {
                self.check_spawn(sp);
                false
            }

            StmtKind::Continue => {
                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'continue' cannot be used inside a 'defer' cleanup statement",
                    ));
                } else if self.loop_depth == 0 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'continue' may only be used inside a 'loop', 'loop every', 'while', or 'for' body",
                    ));
                } else {
                    let loop_entry = self.loop_lease_frames.last().map(|(d, _)| *d).unwrap_or(0);
                    self.check_unconsumed_leases_on_exit(stmt.span, "continue", loop_entry);
                    if let Some((entry_d, entry_vars)) = self.loop_lease_frames.last().cloned() {
                        self.check_loop_reassigned_outer_live(
                            &entry_vars,
                            entry_d,
                            stmt.span,
                            "continue",
                        );
                    }
                }
                true
            }

            StmtKind::Break => {
                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'break' cannot be used inside a 'defer' cleanup statement",
                    ));
                } else if self.loop_depth == 0 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'break' may only be used inside a 'loop', 'loop every', 'while', or 'for' body",
                    ));
                } else {
                    let loop_entry = self.loop_lease_frames.last().map(|(d, _)| *d).unwrap_or(0);
                    self.check_unconsumed_leases_on_exit(stmt.span, "break", loop_entry);
                    if let Some((entry_d, entry_vars)) = self.loop_lease_frames.last().cloned() {
                        self.check_loop_reassigned_outer_live(
                            &entry_vars,
                            entry_d,
                            stmt.span,
                            "break",
                        );
                    }
                }
                true
            }

            StmtKind::Checkpoint => {
                self.record_direct_yield(stmt.span, FnEffect::SoftYielding, "'checkpoint()'");
                false
            }

            StmtKind::Block(b) => self.check_nested_block(b),

            StmtKind::Expr(e) => {
                let ev = self.check_expr_infallible(e);
                if ev.ty.is_linear_lease() {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::LinearLeaseViolation,
                            e.span,
                            format!(
                                "unused expression of linear type '{}' drops ownership and leaks pool slot",
                                ev.ty
                            ),
                        )
                        .with_note("bind the `Lease<T>` to a local variable (`Lease<T> l = ...;`) or consume it via `.release()`, `dma_slot.park(...)`, or `inbox.post(...)`"),
                    );
                }
                let exits = self.lookup_local("sys").is_none() && is_sys_exit_expr(e);
                if exits {
                    self.check_unconsumed_leases_on_exit(e.span, "sys.exit", 0);
                }
                exits
            }
        }
    }

    fn check_loop_reassigned_outer_live(
        &mut self,
        entry_vars: &[BTreeMap<String, LocalVar>],
        entry_depth: u32,
        span: Span,
        ctx: &str,
    ) {
        let mut offenders = Vec::new();
        for (idx, scope) in self.scopes.iter().enumerate() {
            for (name, var) in scope {
                if var.decl_depth <= entry_depth
                    && matches!(var.lease_state, Some(LinearLeaseState::Live))
                {
                    let was_live = entry_vars
                        .get(idx)
                        .and_then(|s| s.get(name))
                        .is_some_and(|ev| {
                            matches!(
                                ev.lease_state,
                                Some(
                                    LinearLeaseState::Live
                                        | LinearLeaseState::DeferredRelease { .. }
                                )
                            )
                        });
                    if !was_live {
                        offenders.push(name.clone());
                    }
                }
            }
        }
        for name in offenders {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!(
                    "Lease<T> variable '{}' acquired inside loop must be consumed before '{}'",
                    name, ctx
                ),
            ));
            if let Some(v) = self.lookup_local_mut(&name) {
                v.lease_state = Some(LinearLeaseState::Moved {
                    moved_at: span,
                    reason: "loop leak",
                });
            }
        }
    }

    fn check_for_loop_with_binding(
        &mut self,
        var_name: &str,
        var_ty: Ty,
        var_prov: Provenance,
        iter_span: Option<Span>,
        body: &Block,
    ) {
        let leases_before = self.leases.clone();
        let entry_depth = self.current_depth();
        let loop_entry_vars = self.snapshot_outer_vars();
        self.loop_depth += 1;
        self.loop_entry_depths.push(entry_depth);
        self.loop_outer_reads.push(Vec::new());
        self.loop_lease_frames.push((entry_depth, loop_entry_vars.clone()));

        let body_div = self.with_nested_scope(|this| {
            let bind_depth = this.current_depth();
            this.scopes.last_mut().unwrap().insert(
                var_name.to_string(),
                LocalVar {
                    name: var_name.to_string(),
                    ty: var_ty,
                    decl_depth: bind_depth,
                    prov: var_prov.clone(),
                    field_prov: BTreeMap::new(),
                    endpoint_alias: None,
                    field_endpoints: BTreeMap::new(),
                    is_isr_state: false,
                    lease_state: None,
                    lease_gen_key: None,
                },
            );
            let div = this.check_block_stmts(&body.stmts);
            this.finish_lexical_block(div, body.span);
            div
        });
        if !body_div {
            self.check_loop_reassigned_outer_live(
                &loop_entry_vars,
                entry_depth,
                body.span,
                "end of loop iteration",
            );
        }

        let _ = self.loop_lease_frames.pop();
        let outer_reads = self.loop_outer_reads.pop().unwrap_or_default();
        self.loop_entry_depths.pop();
        self.loop_depth -= 1;

        let modified = self.leases.modified_sources_since(&leases_before);
        if !modified.is_empty() {
            if let Some(isp) = iter_span {
                for tok in &var_prov.leases {
                    if let Some(inv_span) = modified.get(&tok.source) {
                        if leases_before.check_valid(tok).is_ok() {
                            let inv_method = self.leases.last_invalidator(&tok.source);
                            let display_src = tok.source.strip_suffix("#write").unwrap_or(&tok.source);
                            let (msg, note_msg) = if tok.acq_method == "wait_chunk" {
                                (
                                    format!(
                                        "cannot iterate over slice derived from '{}.wait_chunk()' across a loop body that calls 'wait_chunk()'",
                                        tok.source
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "DMA lease invalidated inside 'for..in' body via 'wait_chunk()' at {}",
                                            sp
                                        )
                                    }),
                                )
                            } else {
                                (
                                    format!(
                                        "cannot iterate over slice derived from '{}.{}()' across a loop body that calls '{}.{}()'",
                                        display_src, tok.acq_method, display_src, inv_method
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "byte_ring lease invalidated inside 'for..in' body via '{}()' at {}",
                                            inv_method, sp
                                        )
                                    }),
                                )
                            };
                            let mut diag = Diagnostic::new(DiagnosticKind::StaleDmaLease, isp, msg);
                            if let Some(nm) = note_msg {
                                diag = diag.with_note(nm);
                            }
                            self.sema.emit(diag);
                        }
                    }
                }
            }

            for (r_name, read_env, r_prov, read_span) in outer_reads {
                for tok in &r_prov.leases {
                    if let Some(inv_span) = modified.get(&tok.source) {
                        if read_env.check_valid(tok).is_ok() {
                            let inv_method = self.leases.last_invalidator(&tok.source);
                            let display_src = tok.source.strip_suffix("#write").unwrap_or(&tok.source);
                            let (msg, note_msg) = if tok.acq_method == "wait_chunk" {
                                (
                                    format!(
                                        "cannot read outer variable '{}' derived from '{}.wait_chunk()' across loop iterations that call 'wait_chunk()'",
                                        r_name, tok.source
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "subsequent loop iteration invalidates lease via 'wait_chunk()' at {}",
                                            sp
                                        )
                                    }),
                                )
                            } else {
                                (
                                    format!(
                                        "cannot read outer variable '{}' derived from '{}.{}()' across loop iterations that call '{}.{}()'",
                                        r_name, display_src, tok.acq_method, display_src, inv_method
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "subsequent loop iteration invalidates lease via '{}()' at {}",
                                            inv_method, sp
                                        )
                                    }),
                                )
                            };
                            let mut diag =
                                Diagnostic::new(DiagnosticKind::StaleDmaLease, read_span, msg);
                            if let Some(nm) = note_msg {
                                diag = diag.with_note(nm);
                            }
                            self.sema.emit(diag);
                        }
                    }
                }
            }
        }
    }

    fn check_loop_body(&mut self, body: &Block, _infinite: bool) {
        let leases_before = self.leases.clone();
        let entry_depth = self.current_depth();
        let loop_entry_vars = self.snapshot_outer_vars();
        self.loop_depth += 1;
        self.loop_entry_depths.push(entry_depth);
        self.loop_outer_reads.push(Vec::new());
        self.loop_lease_frames.push((entry_depth, loop_entry_vars.clone()));

        let body_div = self.check_nested_block(body);
        if !body_div {
            self.check_loop_reassigned_outer_live(
                &loop_entry_vars,
                entry_depth,
                body.span,
                "end of loop iteration",
            );
        }

        let _ = self.loop_lease_frames.pop();
        let outer_reads = self.loop_outer_reads.pop().unwrap_or_default();
        self.loop_entry_depths.pop();
        self.loop_depth -= 1;

        let modified = self.leases.modified_sources_since(&leases_before);
        if !modified.is_empty() {
            // Check if any outer variable holding a pre-loop DMA/ring lease on a source mutated inside
            // the loop was read in the loop (which would read a stale chunk on iteration 2).
            for (var_name, read_env, var_prov, read_span) in outer_reads {
                for tok in &var_prov.leases {
                    if let Some(inv_span) = modified.get(&tok.source) {
                        // Only report if `read_env` hadn't already reported it stale on iteration 1.
                        if read_env.check_valid(tok).is_ok() {
                            let inv_method = self.leases.last_invalidator(&tok.source);
                            let display_src = tok.source.strip_suffix("#write").unwrap_or(&tok.source);
                            let (msg, note_msg) = if tok.acq_method == "wait_chunk" {
                                (
                                    format!(
                                        "cannot read outer variable '{}' derived from '{}.wait_chunk()' across loop iterations that call 'wait_chunk()'",
                                        var_name, tok.source
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "subsequent loop iteration invalidates lease via 'wait_chunk()' at {}",
                                            sp
                                        )
                                    }),
                                )
                            } else {
                                (
                                    format!(
                                        "cannot read outer variable '{}' derived from '{}.{}()' across loop iterations that call '{}.{}()'",
                                        var_name, display_src, tok.acq_method, display_src, inv_method
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "subsequent loop iteration invalidates lease via '{}()' at {}",
                                            inv_method, sp
                                        )
                                    }),
                                )
                            };
                            let mut diag =
                                Diagnostic::new(DiagnosticKind::StaleDmaLease, read_span, msg);
                            if let Some(nm) = note_msg {
                                diag = diag.with_note(nm);
                            }
                            self.sema.emit(diag);
                        }
                    }
                }
            }
            let _ = entry_depth;
        }
    }

    fn check_spawn(&mut self, sp: &SpawnStmt) {
        let mut invalid_ctx = false;
        if self.in_defer() {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidControlFlow,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' inside a 'defer' cleanup statement",
                    sp.task.name
                ),
            ));
        } else if self.in_isr() {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidSpawnContext,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' inside ISR '{}'; tasks must be statically spawned in 'main()' boot setup",
                    sp.task.name,
                    self.display_caller_name()
                ),
            ));
        } else if matches!(self.body_kind, ActiveBodyKind::Task) {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidSpawnContext,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' inside task '{}'; dynamic runtime task spawning is forbidden—wire all tasks in 'main()' boot setup",
                    sp.task.name,
                    self.display_caller_name()
                ),
            ));
        } else if self.loop_depth > 0 {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidSpawnContext,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' inside a loop; every static boot-tree path must map 1:1 to a task slot",
                    sp.task.name
                ),
            ));
        } else if self.after_diverge || self.main_supervisor_started {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidSpawnContext,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' after 'main()' supervisor loop or diverging control flow",
                    sp.task.name
                ),
            ));
        }

        let Some(task_info) = self.sema.tasks.get(&sp.task.name).cloned() else {
            if self.sema.functions.contains_key(&sp.task.name) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    sp.task.span,
                    format!(
                        "cannot 'spawn' function '{}'; only 'task' blueprints may be spawned",
                        sp.task.name
                    ),
                ));
            } else {
                let task_names: Vec<&str> = self.sema.tasks.keys().map(String::as_str).collect();
                let mut diag = Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    sp.task.span,
                    format!("undeclared task blueprint '{}'", sp.task.name),
                );
                if let Some(best) = closest_match(&sp.task.name, task_names, 2) {
                    diag = diag.with_note(format!("did you mean `spawn {}(...)`?", best));
                }
                self.sema.emit(diag);
            }
            for arg in &sp.args {
                let _ = self.check_expr_infallible(&arg.value);
            }
            return;
        };

        if sp.args.len() != task_info.params.len() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                sp.span,
                format!(
                    "task '{}' expects {} argument(s), found {}",
                    sp.task.name,
                    task_info.params.len(),
                    sp.args.len()
                ),
            ));
        }

        let mut arg_bindings = Vec::with_capacity(sp.args.len());
        for (i, arg) in sp.args.iter().enumerate() {
            let val = self.check_expr_infallible(&arg.value);
            let mut ref_param_origins = BTreeSet::new();
            if let Some(param) = task_info.params.get(i) {
                if self.sema.check_assignable_or_emit(
                    &val.ty,
                    &param.ty,
                    arg.span,
                    DiagnosticKind::TypeMismatch,
                    || {
                        format!(
                            "argument {} to 'spawn {}' has type '{}', expected '{}'",
                            i + 1,
                            sp.task.name,
                            val.ty,
                            param.ty
                        )
                    },
                ) {
                    self.sema.check_f32_coercion_expr(&arg.value, &param.ty);
                }
                if param.ty.contains_reference(&self.sema.structs) {
                    if val.prov.has_non_returning_origin() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::SpawnNonStaticReference,
                            arg.span,
                            format!(
                                "cannot pass non-static reference/slice into 'spawn {}' (argument {}): spawned tasks outlive local stack variables and DMA leases",
                                sp.task.name,
                                i + 1
                            ),
                        ));
                    } else {
                        ref_param_origins = val.prov.param_origins.clone();
                    }
                }
            }
            let field_endpoints = self.extract_field_endpoint_specs(&val);
            let endpoint_from_param = self.resolve_param_endpoint_index(&val.endpoint_name);
            let static_endpoint = val
                .endpoint_name
                .filter(|ep| !ep.starts_with("param:"));
            arg_bindings.push(BootArgBinding {
                static_endpoint,
                endpoint_from_param,
                field_endpoints,
                ref_param_origins,
            });
        }

        if self.caller_key == "main" {
            self.has_spawned = true;
        }

        if !invalid_ctx && self.can_record_boot_step() {
            self.sema
                .call_graph
                .mark_direct_spawn(&self.caller_key, &sp.task.name);
            self.boot_steps.push(BootStep::DirectSpawn {
                task_name: sp.task.name.clone(),
                priority: sp.priority.unwrap_or(Priority::Normal),
                core: sp.core,
                conditional: self.branch_depth > 0,
                arg_bindings,
                span: sp.span,
            });
        }
    }

    fn check_assignment(
        &mut self,
        target: &Expr,
        op: AssignOp,
        value_expr: &Expr,
        rhs: ExprVal,
        stmt_span: Span,
    ) {
        match &target.kind {
            ExprKind::Ident(ident) => {
                if self.sema.consts.contains_key(&ident.name) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!("cannot assign to immutable const '{}'", ident.name),
                    ));
                    return;
                }
                if self.sema.primitives.contains_key(&ident.name) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!(
                            "cannot overwrite OS primitive '{}' directly; use its methods (e.g. .publish(), .post())",
                            ident.name
                        ),
                    ));
                    return;
                }
                if self.sema.subsystems.contains_key(&ident.name) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!(
                            "cannot overwrite subsystem instance '{}' directly; operate on its fields via methods",
                            ident.name
                        ),
                    ));
                    return;
                }
                if self.sema.mmio_instances.contains_key(&ident.name)
                    || self.sema.irqs.contains_key(&ident.name)
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!(
                            "cannot reassign hardware handle '{}' directly",
                            ident.name
                        ),
                    ));
                    return;
                }
                if let Some(ig) = self.sema.illegal_globals.get(&ident.name).cloned() {
                    let ty_str = if ig.ty.is_error() {
                        "u32".to_string()
                    } else {
                        ig.ty.to_string()
                    };
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::GlobalIsolationViolation,
                            target.span,
                            format!(
                                "task/ISR isolation violation: '{}' cannot write to raw mutable global '{}'",
                                self.display_caller_name(),
                                ident.name
                            ),
                        )
                        .with_note(format!(
                            "declared at {}; replace raw global `{name}` with `watch<{ty}> {name} = 0;` (then `{name}.publish(val);`) or `inbox<{ty}, 8> {name};` (then `{name}.post(ev);`)",
                            ig.decl_span,
                            name = ident.name,
                            ty = ty_str
                        )),
                    );
                    return;
                }

                // If compound assignment (`+=`, etc.), first validate reading `ident`.
                if op != AssignOp::Assign {
                    let cur = self.check_expr_infallible(target);
                    self.check_compound_op_types(op, &cur.ty, value_expr, &rhs.ty, stmt_span);
                    return;
                }

                let Some(var_info) = self.lookup_local(&ident.name).cloned() else {
                    let mut diag = Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        ident.span,
                        format!("undeclared variable '{}'", ident.name),
                    );
                    if let Some(hint) = self.suggest_ident_hint(&ident.name) {
                        diag = diag.with_note(hint);
                    }
                    self.sema.emit(diag);
                    return;
                };

                if self.sema.check_assignable_or_emit(
                    &rhs.ty,
                    &var_info.ty,
                    rhs.span,
                    DiagnosticKind::TypeMismatch,
                    || {
                        format!(
                            "cannot assign '{}' to variable '{}' of type '{}'",
                            rhs.ty, ident.name, var_info.ty
                        )
                    },
                ) {
                    self.sema.check_f32_coercion_expr(value_expr, &var_info.ty);
                }

                if var_info.ty.is_linear_lease() {
                    match var_info.lease_state {
                        Some(LinearLeaseState::Live) => {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::LinearLeaseViolation,
                                    stmt_span,
                                    format!(
                                        "cannot overwrite live Lease<T> variable '{}' before consuming its current handle",
                                        ident.name
                                    ),
                                )
                                .with_note("call `.release()`, `dma_slot.park(...)`, or `inbox.post(...)` on it before reassigning"),
                            );
                        }
                        Some(LinearLeaseState::DeferredRelease { span, .. }) => {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::LinearLeaseViolation,
                                    stmt_span,
                                    format!(
                                        "cannot overwrite Lease<T> variable '{}' while `defer` release from {} is pending",
                                        ident.name, span
                                    ),
                                ),
                            );
                        }
                        Some(LinearLeaseState::Moved { .. })
                        | Some(LinearLeaseState::Uninit)
                        | None => {}
                    }
                    if rhs.ty.is_linear_lease() {
                        self.consume_lease_value(value_expr, &rhs, "assignment", stmt_span);
                    }
                    let new_key = self.fresh_lease_gen_key(&ident.name);
                    if let Some(v) = self.lookup_local_mut(&ident.name) {
                        v.lease_state = Some(if rhs.ty.is_error() {
                            LinearLeaseState::Moved {
                                moved_at: stmt_span,
                                reason: "error assign",
                            }
                        } else {
                            LinearLeaseState::Live
                        });
                        v.lease_gen_key = Some(new_key);
                    }
                }

                if var_info.ty.contains_reference(&self.sema.structs)
                    || var_info.ty.contains_fn_ptr(&self.sema.structs)
                {
                    if var_info.decl_depth < rhs.prov.origin_depth {
                        if !rhs.prov.isr_state_origins.is_empty() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidIsrState,
                                stmt_span,
                                format!(
                                    "cannot escape `&` of ISR `state` variable into outer variable '{}' at depth {}",
                                    ident.name, var_info.decl_depth
                                ),
                            ));
                        } else {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::ScopeDepthEscape,
                                    stmt_span,
                                    format!(
                                        "cannot escape temporary/leased value of scope depth {} into outer variable '{}' declared at scope depth {}",
                                        rhs.prov.origin_depth, ident.name, var_info.decl_depth
                                    ),
                                )
                                .with_note(format!(
                                    "origin depth is {} (nested block/DMA lease) while '{}' lives at outer depth {} (0=static, 1=fn/task body, 2+=nested block); pass the slice/struct down into a helper `fn`, decode into a value struct (`Type.from_bytes(slice)`), or copy bytes into a fixed `[N]u8` buffer",
                                    rhs.prov.origin_depth, ident.name, var_info.decl_depth
                                )),
                            );
                        }
                    }
                    if let Some(v) = self.lookup_local_mut(&ident.name) {
                        v.prov = rhs.prov;
                        v.field_prov.clear();
                        v.endpoint_alias = rhs.endpoint_name;
                        v.field_endpoints = rhs.field_endpoints;
                    }
                }
            }

            ExprKind::Field {
                target: base,
                field,
            } => {
                // Check field or MMIO register write (`r.buf = half;`, `USART1.icr = 1;`, `regs.cr1 |= 1;`).
                let base_val = self.check_projection_base(base);
                let (struct_name, via_ptr) = match &base_val.ty {
                    Ty::Struct(s) => (s.clone(), false),
                    Ty::MmioLayout(layout_name) | Ty::MmioInstance { layout: layout_name, .. } => {
                        let ln = layout_name.clone();
                        self.check_mmio_reg_write_access(
                            &ln,
                            field,
                            None,
                            op,
                            value_expr,
                            &rhs,
                            stmt_span,
                        );
                        return;
                    }
                    Ty::Ptr(inner) => match &**inner {
                        Ty::Struct(s) => (s.clone(), true),
                        Ty::MmioLayout(layout_name) => {
                            let ln = layout_name.clone();
                            self.check_mmio_reg_write_access(
                                &ln,
                                field,
                                None,
                                op,
                                value_expr,
                                &rhs,
                                stmt_span,
                            );
                            return;
                        }
                        _ => {
                        if !base_val.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                base.span,
                                format!("type '{}' has no field '{}'", base_val.ty, field.name),
                            ));
                        }
                        return;
                    }
                },
                _ => {
                        if !base_val.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                base.span,
                                format!("type '{}' has no assignable field '{}'", base_val.ty, field.name),
                            ));
                        }
                        return;
                    }
                };

                let Some(field_ty) = self
                    .sema
                    .structs
                    .get(&struct_name)
                    .and_then(|s| s.field(&field.name))
                    .map(|f| f.ty.clone())
                else {
                    let hint = self.suggest_struct_field_hint(&struct_name, &field.name);
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UndefinedSymbol,
                            field.span,
                            format!("struct '{}' has no field '{}'", struct_name, field.name),
                        )
                        .with_note(hint),
                    );
                    return;
                };

                if self
                    .sema
                    .structs
                    .get(&struct_name)
                    .is_some_and(|s| s.is_subsystem)
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!(
                            "cannot overwrite field '{}.{}' of subsystem struct directly; operate on it via its methods",
                            struct_name, field.name
                        ),
                    ));
                    return;
                }

                if op == AssignOp::Assign {
                    if self.sema.check_assignable_or_emit(
                        &rhs.ty,
                        &field_ty,
                        rhs.span,
                        DiagnosticKind::TypeMismatch,
                        || {
                            format!(
                                "cannot assign '{}' to field '{}.{}' of type '{}'",
                                rhs.ty, struct_name, field.name, field_ty
                            )
                        },
                    ) {
                        self.sema.check_f32_coercion_expr(value_expr, &field_ty);
                    }
                } else {
                    self.check_compound_op_types(op, &field_ty, value_expr, &rhs.ty, stmt_span);
                }

                if field_ty.contains_reference(&self.sema.structs)
                    || field_ty.contains_fn_ptr(&self.sema.structs)
                {
                    if via_ptr {
                        let dest_depth = base_val.prov.pointee_depth.unwrap_or(0);
                        if (base_val.prov.pointee_is_caller_or_static
                            && rhs.prov.has_non_returning_origin())
                            || dest_depth < rhs.prov.origin_depth
                        {
                            if !rhs.prov.isr_state_origins.is_empty() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::InvalidIsrState,
                                    stmt_span,
                                    format!(
                                        "cannot escape `&` of ISR `state` variable through pointer field '{}'",
                                        field.name
                                    ),
                                ));
                            } else {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ScopeDepthEscape,
                                    stmt_span,
                                    format!(
                                        "cannot store inner/leased reference of depth {} into struct field '{}' through outer pointer (depth {})",
                                        rhs.prov.origin_depth, field.name, dest_depth
                                    ),
                                ));
                            }
                        }
                    } else if let Some(root_name) = root_ident_of(base) {
                        if let Some(root_var) = self.lookup_local(&root_name).cloned() {
                            if root_var.decl_depth < rhs.prov.origin_depth {
                                if !rhs.prov.isr_state_origins.is_empty() {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::InvalidIsrState,
                                        stmt_span,
                                        format!(
                                            "cannot escape `&` of ISR `state` variable into outer field '{}.{}'",
                                            root_name, field.name
                                        ),
                                    ));
                                } else {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::ScopeDepthEscape,
                                        stmt_span,
                                        format!(
                                            "cannot escape value of scope depth {} into field '{}' of outer struct '{}' declared at scope depth {}",
                                            rhs.prov.origin_depth, field.name, root_name, root_var.decl_depth
                                        ),
                                    ));
                                }
                            }
                            if let Some(v) = self.lookup_local_mut(&root_name) {
                                v.field_prov.insert(field.name.clone(), rhs.prov.clone());
                                let mut combined = Provenance::pure_static();
                                for fp in v.field_prov.values() {
                                    combined = combined.merge(fp);
                                }
                                v.prov = combined;
                                if matches!(&base.kind, ExprKind::Ident(_)) {
                                    if let Some(ep) = &rhs.endpoint_name {
                                        v.field_endpoints.insert(field.name.clone(), ep.clone());
                                    } else {
                                        v.field_endpoints.remove(&field.name);
                                    }
                                    let prefix = format!("{}.", field.name);
                                    v.field_endpoints.retain(|k, _| !k.starts_with(&prefix));
                                    for (sub_k, sub_ep) in &rhs.field_endpoints {
                                        v.field_endpoints
                                            .insert(format!("{}.{}", field.name, sub_k), sub_ep.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }

            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: ptr_expr,
            } => {
                let pv = self.check_expr_infallible(ptr_expr);
                let Ty::Ptr(pointee_ty) = &pv.ty else {
                    if !pv.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            target.span,
                            format!("cannot dereference non-pointer type '{}'", pv.ty),
                        ));
                    }
                    return;
                };
                if let Some(ln) = ty_contains_byval_mmio(pointee_ty) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        target.span,
                        format!(
                            "cannot overwrite MMIO register block '{}' by value; assign individual registers instead",
                            ln
                        ),
                    ));
                    return;
                }
                if let Some(sub) = ty_contains_byval_subsys(pointee_ty, &self.sema.structs) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        target.span,
                        format!(
                            "cannot overwrite subsystem struct '{}' by value; operate on its fields via methods",
                            sub
                        ),
                    ));
                    return;
                }
                if ty_embeds_os_primitive(pointee_ty)
                    || ty_contains_bare_unsized_prim(pointee_ty).is_some()
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        target.span,
                        format!(
                            "cannot overwrite static OS primitive '{}' by value; use its methods instead",
                            pointee_ty
                        ),
                    ));
                    return;
                }
                if op == AssignOp::Assign {
                    if self.sema.check_assignable_or_emit(
                        &rhs.ty,
                        pointee_ty,
                        rhs.span,
                        DiagnosticKind::TypeMismatch,
                        || {
                            format!(
                                "cannot assign '{}' to dereferenced '*{}'",
                                rhs.ty, pointee_ty
                            )
                        },
                    ) {
                        self.sema.check_f32_coercion_expr(value_expr, pointee_ty);
                    }
                } else {
                    self.check_compound_op_types(op, pointee_ty, value_expr, &rhs.ty, stmt_span);
                }
                if pointee_ty.contains_reference(&self.sema.structs) {
                    let dest_depth = pv.prov.pointee_depth.unwrap_or(0);
                    if (pv.prov.pointee_is_caller_or_static && rhs.prov.has_non_returning_origin())
                        || dest_depth < rhs.prov.origin_depth
                    {
                        if !rhs.prov.isr_state_origins.is_empty() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidIsrState,
                                stmt_span,
                                "cannot escape `&` of ISR `state` variable through outer or caller pointer",
                            ));
                        } else {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::ScopeDepthEscape,
                                stmt_span,
                                format!(
                                    "cannot store inner/leased reference of depth {} through pointer targeting outer depth {}",
                                    rhs.prov.origin_depth, dest_depth
                                ),
                            ));
                        }
                    }
                }
            }

            ExprKind::Index {
                target: seq_expr,
                index,
            } => {
                // Handle indexed MMIO array register writes (`regs.brr[i] = val;`, `regs.brr[i] |= mask;`)
                // directly so `@wo` array registers are never misread before checking write qualifiers.
                if let ExprKind::Field {
                    target: mmio_base,
                    field,
                } = &seq_expr.kind
                {
                    let base_val = self.check_projection_base(mmio_base);
                    let mmio_layout = match &base_val.ty {
                        Ty::MmioLayout(ln) | Ty::MmioInstance { layout: ln, .. } => {
                            Some(ln.clone())
                        }
                        Ty::Ptr(inner) => match &**inner {
                            Ty::MmioLayout(ln) => Some(ln.clone()),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(ln) = mmio_layout {
                        self.check_mmio_reg_write_access(
                            &ln,
                            field,
                            Some(index),
                            op,
                            value_expr,
                            &rhs,
                            stmt_span,
                        );
                        return;
                    }
                }

                let sv = self.check_expr_infallible(seq_expr);
                let iv = self.check_expr_infallible(index);
                if !iv.ty.is_integer() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        index.span,
                        format!("array/slice index must be an integer, found '{}'", iv.ty),
                    ));
                }
                let elem_ty = match &sv.ty {
                    Ty::Slice(e) | Ty::Array { elem: e, .. } => (**e).clone(),
                    _ => {
                        if !sv.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                seq_expr.span,
                                format!("cannot index into type '{}'", sv.ty),
                            ));
                        }
                        return;
                    }
                };
                if op == AssignOp::Assign {
                    if self.sema.check_assignable_or_emit(
                        &rhs.ty,
                        &elem_ty,
                        rhs.span,
                        DiagnosticKind::TypeMismatch,
                        || {
                            format!(
                                "cannot assign '{}' to element of type '{}'",
                                rhs.ty, elem_ty
                            )
                        },
                    ) {
                        self.sema.check_f32_coercion_expr(value_expr, &elem_ty);
                    }
                } else {
                    self.check_compound_op_types(op, &elem_ty, value_expr, &rhs.ty, stmt_span);
                }
                if elem_ty.contains_reference(&self.sema.structs) {
                    if let Some(root_name) = root_ident_of(seq_expr) {
                        if let Some(root_var) = self.lookup_local(&root_name) {
                            if root_var.decl_depth < rhs.prov.origin_depth {
                                if !rhs.prov.isr_state_origins.is_empty() {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::InvalidIsrState,
                                        stmt_span,
                                        format!(
                                            "cannot escape `&` of ISR `state` variable into outer array/slice '{}'",
                                            root_name
                                        ),
                                    ));
                                } else {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::ScopeDepthEscape,
                                        stmt_span,
                                        format!(
                                            "cannot escape value of scope depth {} into array/slice '{}' at scope depth {}",
                                            rhs.prov.origin_depth, root_name, root_var.decl_depth
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            _ => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::ImmutableAssignment,
                    target.span,
                    "invalid assignment target",
                ));
            }
        }
    }

    fn check_compound_op_types(
        &mut self,
        op: AssignOp,
        lhs: &Ty,
        rhs_expr: &Expr,
        rhs: &Ty,
        span: Span,
    ) {
        if lhs.is_error() || rhs.is_error() {
            return;
        }
        let float_arith_ok = matches!(
            op,
            AssignOp::AddAssign
                | AssignOp::SubAssign
                | AssignOp::MulAssign
                | AssignOp::DivAssign
                | AssignOp::RemAssign
        ) && lhs.is_float()
            && rhs.is_float()
            && rhs.is_assignable_to(lhs);
        if !(lhs.is_integer() && rhs.is_integer()) && !float_arith_ok {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "compound assignment '{}' requires compatible numeric operands, found '{}' and '{}'",
                    op.as_str(),
                    lhs,
                    rhs
                ),
            ));
        } else if float_arith_ok {
            self.sema.check_f32_coercion_expr(rhs_expr, lhs);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Expression Checking
    // ─────────────────────────────────────────────────────────────────────────

    fn take_mmio_array_tag(&mut self, span: Span) -> Option<(String, String, usize)> {
        self.mmio_array_expr_spans.remove(&(span.start, span.end))
    }

    fn reject_unindexed_mmio_array(&mut self, span: Span) -> bool {
        if let Some((layout_name, reg_name, _)) = self.take_mmio_array_tag(span) {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                span,
                format!(
                    "cannot use unindexed MMIO register array '{}.{}'; index an element with '[i]' or read '.len'",
                    layout_name, reg_name
                ),
            ));
            true
        } else {
            false
        }
    }

    fn check_expr_infallible_allow_mmio_array(
        &mut self,
        expr: &Expr,
    ) -> (ExprVal, Option<(String, String, usize)>) {
        let mut val = self.check_expr_raw(expr);
        let mmio_arr = self.take_mmio_array_tag(expr.span);
        if val.error_spec.is_fallible() {
            if self
                .proven_infallible_spans
                .remove(&(expr.span.start, expr.span.end))
            {
                val.error_spec = ResolvedErrorSpec::None;
                return (val, mmio_arr);
            }
            let note = match (&val.error_spec, val.ty.is_void()) {
                (ResolvedErrorSpec::Typed(err_ty), false) => format!(
                    "unwrap with `expr or <fallback>` (e.g. `{} x = expr or default_val;`) or `expr catch ({} err) {{ ... }}` / `expr catch ({} err, {} partial) {{ ... }}`",
                    val.ty, err_ty, err_ty, val.ty
                ),
                (ResolvedErrorSpec::Typed(err_ty), true) => format!(
                    "handle failure with `expr catch ({} err) {{ ... }};`",
                    err_ty
                ),
                (_, false) => format!(
                    "unwrap with `expr or <fallback>` (e.g. `{} x = expr or default_val;`) or `expr catch {{ ... }}`",
                    val.ty
                ),
                (_, true) => "handle failure with `expr catch { ... };`".to_string(),
            };
            self.sema.emit(
                Diagnostic::new(
                    DiagnosticKind::UnhandledFallibleExpr,
                    expr.span,
                    "unhandled fallible expression (may exit via 'error'); handle failure explicitly with 'catch' or 'or'",
                )
                .with_note(note),
            );
            val.error_spec = ResolvedErrorSpec::None;
        }
        (val, mmio_arr)
    }

    fn check_expr_infallible(&mut self, expr: &Expr) -> ExprVal {
        let (val, mmio_arr) = self.check_expr_infallible_allow_mmio_array(expr);
        if let Some((layout_name, reg_name, _)) = mmio_arr {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                expr.span,
                format!(
                    "cannot use unindexed MMIO register array '{}.{}'; index an element with '[i]' or read '.len'",
                    layout_name, reg_name
                ),
            ));
            return ExprVal::error(expr.span);
        }
        val
    }

    fn check_expr_raw(&mut self, expr: &Expr) -> ExprVal {
        match &expr.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Int(_) => ExprVal::scalar(Ty::IntLit, expr.span),
                Literal::Float {
                    suffix: Some(p), ..
                } => ExprVal::scalar(Ty::Prim(*p), expr.span),
                Literal::Float { suffix: None, .. } => ExprVal::scalar(Ty::FloatLit, expr.span),
                Literal::Bool(_) => ExprVal::scalar(Ty::bool(), expr.span),
                Literal::String(_) => ExprVal::scalar(Ty::Str, expr.span),
                Literal::Duration(dur) => {
                    self.sema.validate_duration(dur);
                    ExprVal::scalar(Ty::Duration, expr.span)
                }
            },

            ExprKind::Priority(_) => ExprVal::scalar(Ty::Priority, expr.span),

            ExprKind::Ident(ident) => self.check_ident_expr(ident),

            ExprKind::Unary { op, expr: inner } => self.check_unary_expr(*op, inner, expr.span),

            ExprKind::Binary { op, lhs, rhs } => self.check_binary_expr(*op, lhs, rhs, expr.span),

            ExprKind::Cast { expr: inner, ty } => {
                let iv = self.check_expr_infallible(inner);
                let target_ty = self.sema.resolve_type(ty);
                self.sema
                    .check_type_no_byval_mmio(&target_ty, ty.span, "cast target type");
                if !iv.ty.is_error()
                    && !target_ty.is_error()
                    && ty_contains_byval_mmio(&target_ty).is_none()
                    && !is_valid_cast(&iv.ty, &target_ty)
                {
                    let kind = if iv.ty.contains_mmio() || target_ty.contains_mmio() {
                        DiagnosticKind::MmioViolation
                    } else {
                        DiagnosticKind::TypeMismatch
                    };
                    self.sema.emit(Diagnostic::new(
                        kind,
                        expr.span,
                        format!("cannot cast value of type '{}' to '{}'", iv.ty, target_ty),
                    ));
                }
                let mut prov = iv.prov;
                if !target_ty.contains_reference(&self.sema.structs)
                    && !target_ty.contains_fn_ptr(&self.sema.structs)
                {
                    prov = prov.into_scalar_value();
                }
                ExprVal {
                    ty: target_ty,
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name: None,
                    field_endpoints: BTreeMap::new(),
                    span: expr.span,
                }
            }

            ExprKind::Field { target, field } => self.check_field_expr(target, field, expr.span),

            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                if !generic_args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::GenericInstantiationError,
                        expr.span,
                        "unmonomorphized generic call arguments",
                    ));
                }
                self.check_call_expr(callee, args, expr.span)
            }

            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => self.check_method_call(receiver, method, args, expr.span),

            ExprKind::Index { target, index } => {
                let (tv, mmio_arr) = self.check_expr_infallible_allow_mmio_array(target);
                let iv = self.check_expr_infallible(index);
                if !iv.ty.is_integer() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        index.span,
                        format!("index must be an integer, found '{}'", iv.ty),
                    ));
                }
                if let Some((layout_name, reg_name, len)) = mmio_arr {
                    let shadowed = self.active_local_names();
                    if let Some(idx) = self.sema.eval_const_expr_i128_shadowed(
                        index,
                        &mut BTreeSet::new(),
                        &shadowed,
                    ) {
                        if idx < 0 || idx as u128 >= len as u128 {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::MmioViolation,
                                index.span,
                                format!(
                                    "MMIO register array '{}.{}' index {} out of bounds (length {})",
                                    layout_name, reg_name, idx, len
                                ),
                            ));
                        }
                    }
                }
                match &tv.ty {
                    Ty::Slice(elem) | Ty::Array { elem, .. } => {
                        let elem_ty = (**elem).clone();
                        let prov = if elem_ty.contains_reference(&self.sema.structs) {
                            tv.prov
                        } else {
                            Provenance::pure_static()
                        };
                        ExprVal {
                            ty: elem_ty,
                            error_spec: ResolvedErrorSpec::None,
                            prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span: expr.span,
                        }
                    }
                    _ => {
                        if !tv.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                target.span,
                                format!("cannot index into type '{}'", tv.ty),
                            ));
                        }
                        ExprVal::error(expr.span)
                    }
                }
            }

            ExprKind::Slice { target, start, end } => {
                if let Some(s) = start {
                    let sv = self.check_expr_infallible(s);
                    if !sv.ty.is_integer() && !sv.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            s.span,
                            format!("slice start index must be an integer, found '{}'", sv.ty),
                        ));
                    }
                }
                if let Some(e) = end {
                    let ev = self.check_expr_infallible(e);
                    if !ev.ty.is_integer() && !ev.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            e.span,
                            format!("slice end index must be an integer, found '{}'", ev.ty),
                        ));
                    }
                }
                let (tv, place_prov, _) = self.check_addr_of_place(target, expr.span);
                match &tv.ty {
                    // Sub-slice of an existing slice `buf[0..5]` inherits `buf`'s exact scope depth and DMA lease!
                    Ty::Slice(elem) => ExprVal {
                        ty: Ty::Slice(elem.clone()),
                        error_spec: ResolvedErrorSpec::None,
                        prov: tv.prov,
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span: expr.span,
                    },
                    // Slicing a fixed array `arr[0..5]` / `s.arr[0..5]` / `(*p).arr[0..5]` preserves backing provenance!
                    Ty::Array { elem, .. } => ExprVal {
                        ty: Ty::Slice(elem.clone()),
                        error_spec: ResolvedErrorSpec::None,
                        prov: place_prov,
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span: expr.span,
                    },
                    Ty::Ptr(inner) => match &**inner {
                        Ty::Array { elem, .. } => ExprVal {
                            ty: Ty::Slice(elem.clone()),
                            error_spec: ResolvedErrorSpec::None,
                            prov: tv.prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span: expr.span,
                        },
                        _ => {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                target.span,
                                format!("cannot slice pointer type '{}'", tv.ty),
                            ));
                            ExprVal::error(expr.span)
                        }
                    },
                    _ => {
                        if !tv.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                target.span,
                                format!("cannot take sub-slice of type '{}'", tv.ty),
                            ));
                        }
                        ExprVal::error(expr.span)
                    }
                }
            }

            ExprKind::StructLit {
                module: _,
                name,
                generic_args,
                fields,
            } => {
                if !generic_args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::GenericInstantiationError,
                        expr.span,
                        format!("unmonomorphized generic struct literal '{}'", name.name),
                    ));
                }
                let Some(st_info) = self.sema.structs.get(&name.name).cloned() else {
                    let struct_names: Vec<&str> = self.sema.structs.keys().map(String::as_str).collect();
                    let mut diag = Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        name.span,
                        format!("unknown struct '{}'", name.name),
                    );
                    if let Some(best) = closest_match(&name.name, struct_names, 2) {
                        diag = diag.with_note(format!("did you mean struct `{}`?", best));
                    }
                    self.sema.emit(diag);
                    for fi in fields {
                        let _ = self.check_expr_infallible(&fi.value);
                    }
                    return ExprVal::error(expr.span);
                };

                if st_info.is_subsystem {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        expr.span,
                        format!(
                            "cannot construct subsystem struct '{}' via struct literal; declare a top-level static instance instead",
                            name.name
                        ),
                    ));
                    for fi in fields {
                        let _ = self.check_expr_infallible(&fi.value);
                    }
                    return ExprVal::scalar(Ty::Struct(name.name.clone()), expr.span);
                }

                let mut seen = BTreeSet::new();
                let mut combined_prov = Provenance::pure_static();
                let mut per_field_prov = BTreeMap::new();
                let mut field_endpoints = BTreeMap::new();

                for fi in fields {
                    if !seen.insert(fi.name.name.clone()) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::DuplicateDefinition,
                            fi.name.span,
                            format!(
                                "duplicate field '{}' in '{}' struct literal",
                                fi.name.name, name.name
                            ),
                        ));
                    }
                    let fv = self.check_expr_infallible(&fi.value);
                    if let Some(ep) = &fv.endpoint_name {
                        field_endpoints.insert(fi.name.name.clone(), ep.clone());
                    }
                    for (sub_k, sub_ep) in &fv.field_endpoints {
                        field_endpoints
                            .insert(format!("{}.{}", fi.name.name, sub_k), sub_ep.clone());
                    }
                    if let Some(decl_field) = st_info.field(&fi.name.name) {
                        if self.sema.check_assignable_or_emit(
                            &fv.ty,
                            &decl_field.ty,
                            fi.value.span,
                            DiagnosticKind::TypeMismatch,
                            || {
                                format!(
                                    "field '{}.{}' expects '{}', found '{}'",
                                    name.name, fi.name.name, decl_field.ty, fv.ty
                                )
                            },
                        ) {
                            self.sema.check_f32_coercion_expr(&fi.value, &decl_field.ty);
                        }
                        if decl_field.ty.contains_reference(&self.sema.structs)
                            || decl_field.ty.contains_fn_ptr(&self.sema.structs)
                        {
                            combined_prov = combined_prov.merge(&fv.prov);
                            per_field_prov.insert(fi.name.name.clone(), fv.prov);
                        }
                    } else {
                        let hint = self.suggest_struct_field_hint(&name.name, &fi.name.name);
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::UndefinedSymbol,
                                fi.name.span,
                                format!("struct '{}' has no field '{}'", name.name, fi.name.name),
                            )
                            .with_note(hint),
                        );
                    }
                }

                self.last_struct_lit_fields
                    .insert((expr.span.start, expr.span.end), per_field_prov);

                for decl_field in &st_info.fields {
                    if !seen.contains(&decl_field.name) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            expr.span,
                            format!(
                                "missing field '{}' in '{}' struct literal",
                                decl_field.name, name.name
                            ),
                        ));
                    }
                }

                ExprVal {
                    ty: Ty::Struct(name.name.clone()),
                    error_spec: ResolvedErrorSpec::None,
                    prov: combined_prov,
                    endpoint_name: None,
                    field_endpoints,
                    span: expr.span,
                }
            }

            ExprKind::ArrayLit(elems) => {
                if elems.is_empty() {
                    return ExprVal::error(expr.span);
                }
                let mut elem_ty: Option<Ty> = None;
                let mut combined_prov = Provenance::pure_static();
                for e in elems {
                    let ev = self.check_expr_infallible(e);
                    if ev.ty.is_linear_lease() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::LinearLeaseViolation,
                            e.span,
                            "linear 'Lease<T>' handles cannot be stored inside an array literal",
                        ));
                    }
                    if ev.ty.contains_reference(&self.sema.structs) {
                        combined_prov = combined_prov.merge(&ev.prov);
                    }
                    match &elem_ty {
                        None => elem_ty = Some(ev.ty),
                        Some(cur) if matches!(cur, Ty::IntLit) && ev.ty.is_integer() => {
                            elem_ty = Some(ev.ty);
                        }
                        Some(cur) if matches!(cur, Ty::FloatLit) && ev.ty.is_float() => {
                            elem_ty = Some(ev.ty);
                        }
                        Some(cur) => {
                            let _ = self.sema.check_assignable_or_emit(
                                &ev.ty,
                                cur,
                                e.span,
                                DiagnosticKind::TypeMismatch,
                                || format!("array element has type '{}', expected '{}'", ev.ty, cur),
                            );
                        }
                    }
                }
                let final_elem = elem_ty.unwrap_or(Ty::Error);
                if final_elem == Ty::f32() {
                    for e in elems {
                        self.sema.check_f32_coercion_expr(e, &final_elem);
                    }
                }
                ExprVal {
                    ty: Ty::Array {
                        elem: Box::new(final_elem),
                        len: elems.len(),
                    },
                    error_spec: ResolvedErrorSpec::None,
                    prov: combined_prov,
                    endpoint_name: None,
                    field_endpoints: BTreeMap::new(),
                    span: expr.span,
                }
            }

            ExprKind::FormatString(parts) => {
                let _ = self.check_format_string(parts);
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        expr.span,
                        "zero-heap format string 'f\"...\"' cannot allocate a standalone string value",
                    )
                    .with_note(
                        "pass `f\"...\"` directly into a sink: `println(f\"...\");`, `print(f\"...\");`, or `fmt_buf(dst, f\"...\")`",
                    ),
                );
                ExprVal::error(expr.span)
            }

            ExprKind::Or {
                expr: inner,
                fallback,
            } => {
                let iv = self.check_expr_raw(inner);
                let _ = self.reject_unindexed_mmio_array(inner.span);
                let _ = self
                    .proven_infallible_spans
                    .remove(&(inner.span.start, inner.span.end));
                if !iv.error_spec.is_fallible() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOrFallback,
                        expr.span,
                        "cannot apply 'or' fallback to an infallible expression",
                    ));
                }
                let base_vars = self.snapshot_outer_vars();
                let base_leases = self.leases.clone();
                let fv = self.check_expr_raw(fallback);
                let _ = self.reject_unindexed_mmio_array(fallback.span);
                if fv.ty.is_linear_lease() {
                    self.consume_lease_value(fallback, &fv, "`or` fallback", fallback.span);
                }
                let fallback_vars = self.snapshot_outer_vars();
                let fallback_leases = self.leases.clone();
                self.scopes = base_vars.clone();
                self.merge_branch_vars(&base_vars, &[base_vars.clone(), fallback_vars], expr.span);
                self.leases = base_leases.clone();
                self.leases.join_branches(&[base_leases, fallback_leases]);
                if !fv.ty.is_assignable_to(&iv.ty) && !iv.ty.is_assignable_to(&fv.ty) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOrFallback,
                        fallback.span,
                        format!(
                            "'or' fallback has type '{}', expected '{}'",
                            fv.ty, iv.ty
                        ),
                    ));
                }
                let res_ty = if matches!(iv.ty, Ty::IntLit | Ty::FloatLit) {
                    fv.ty.clone()
                } else {
                    iv.ty.clone()
                };
                if res_ty == Ty::f32() {
                    self.sema.check_f32_coercion_expr(inner, &res_ty);
                    self.sema.check_f32_coercion_expr(fallback, &res_ty);
                }
                let prov = if res_ty.contains_reference(&self.sema.structs) {
                    iv.prov.merge(&fv.prov)
                } else {
                    Provenance::pure_static()
                };
                ExprVal {
                    ty: res_ty,
                    error_spec: fv.error_spec,
                    prov,
                    endpoint_name: None,
                    field_endpoints: BTreeMap::new(),
                    span: expr.span,
                }
            }

            ExprKind::Catch {
                expr: inner,
                clause,
            } => self.check_catch_expr(inner, clause, expr.span),
        }
    }

    fn check_ident_expr(&mut self, ident: &Ident) -> ExprVal {
        // 1. Local variable or parameter.
        if let Some(var) = self.lookup_local(&ident.name).cloned() {
            // Check for invalidated DMA / `byte_ring` lease on read!
            if let Some((stale_tok, inv_span)) = self.leases.find_stale_lease(&var.prov) {
                let inv_method = self.leases.last_invalidator(&stale_tok.source);
                let display_src = stale_tok
                    .source
                    .strip_suffix("#write")
                    .unwrap_or(&stale_tok.source);
                let mut diag = if stale_tok.acq_method == "wait_chunk" {
                    let mut d = Diagnostic::new(
                        DiagnosticKind::StaleDmaLease,
                        ident.span,
                        format!(
                            "cannot read '{}' derived from '{}.wait_chunk()' after a subsequent 'wait_chunk()' invalidated its DMA buffer lease",
                            ident.name, stale_tok.source
                        ),
                    );
                    if let Some(sp) = inv_span {
                        d = d.with_note(format!(
                            "DMA lease acquired at {} was invalidated by subsequent `{}.wait_chunk()` at {}; finish decoding/parsing `{}` (e.g. `Type.from_bytes(slice)`) or copy needed bytes into a local `[N]u8` buffer before calling `wait_chunk()` again",
                            stale_tok.acquired_at, stale_tok.source, sp, ident.name
                        ));
                    } else {
                        d = d.with_note(format!(
                            "finish reading/decoding `{}` or copy needed bytes out before calling `{}.wait_chunk()` again",
                            ident.name, stale_tok.source
                        ));
                    }
                    d
                } else if stale_tok.acq_method == "get" {
                    let lease_name = Self::display_lease_source(&stale_tok.source);
                    let mut d = Diagnostic::new(
                        DiagnosticKind::StaleDmaLease,
                        ident.span,
                        format!(
                            "cannot read '{}' derived from '{}.get()' after '{}' moved or released its Lease",
                            ident.name, lease_name, lease_name
                        ),
                    );
                    if let Some(sp) = inv_span {
                        d = d.with_note(format!(
                            "pointer borrowed from `{}.get()` at {} was invalidated by `{}` at {}; finish dereferencing `{}` before calling `.release()`, `.park()`, `.post()`, or moving `{}`",
                            lease_name, stale_tok.acquired_at, inv_method, sp, ident.name, lease_name
                        ));
                    } else {
                        d = d.with_note(format!(
                            "finish reading `{}` before `{}` is moved or released",
                            ident.name, lease_name
                        ));
                    }
                    d
                } else {
                    let mut d = Diagnostic::new(
                        DiagnosticKind::StaleDmaLease,
                        ident.span,
                        format!(
                            "cannot read '{}' derived from '{}.{}()' after '{}.{}()' invalidated its byte_ring lease",
                            ident.name, display_src, stale_tok.acq_method, display_src, inv_method
                        ),
                    );
                    if let Some(sp) = inv_span {
                        d = d.with_note(format!(
                            "byte_ring slice acquired at {} was invalidated by `{}.{}()` at {}; finish reading/copying bytes from `{}` before calling `consume()`, `pop()`, `commit_write()`, or `grant_write()`",
                            stale_tok.acquired_at, display_src, inv_method, sp, ident.name
                        ));
                    } else {
                        d = d.with_note(format!(
                            "finish reading/copying bytes from `{}` before calling `{}.{}()`",
                            ident.name, display_src, inv_method
                        ));
                    }
                    d
                };
                let _ = &mut diag;
                self.sema.emit(diag);
            } else if !var.prov.leases.is_empty() && !self.loop_outer_reads.is_empty() {
                // Record reads of variables declared OUTSIDE the enclosing loop so cross-iteration
                // DMA lease invalidations are caught too.
                let loop_entry = self.loop_entry_depths.last().copied().unwrap_or(0);
                if var.decl_depth <= loop_entry {
                    if let Some(top_loop) = self.loop_outer_reads.last_mut() {
                        top_loop.push((
                            var.name.clone(),
                            self.leases.clone(),
                            var.prov.clone(),
                            ident.span,
                        ));
                    }
                }
            }

            if var.ty.is_linear_lease() {
                match var.lease_state {
                    Some(LinearLeaseState::Moved { moved_at, reason }) => {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::LinearLeaseViolation,
                                ident.span,
                                format!("use of moved linear Lease<T> variable '{}'", ident.name),
                            )
                            .with_note(format!(
                                "`{}` was previously consumed via {} at {}",
                                ident.name, reason, moved_at
                            )),
                        );
                        return ExprVal::error(ident.span);
                    }
                    Some(LinearLeaseState::Uninit) => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::LinearLeaseViolation,
                            ident.span,
                            format!(
                                "use of uninitialized linear Lease<T> variable '{}'",
                                ident.name
                            ),
                        ));
                        return ExprVal::error(ident.span);
                    }
                    _ => {}
                }
            }

            return ExprVal {
                ty: var.ty,
                error_spec: ResolvedErrorSpec::None,
                prov: var.prov,
                endpoint_name: var.endpoint_alias,
                field_endpoints: var.field_endpoints,
                span: ident.span,
            };
        }

        // 2. Illegal mutable raw global -> report Task/ISR isolation violation at access site!
        if let Some(ig) = self.sema.illegal_globals.get(&ident.name).cloned() {
            let ty_str = if ig.ty.is_error() {
                "u32".to_string()
            } else {
                ig.ty.to_string()
            };
            self.sema.emit(
                Diagnostic::new(
                    DiagnosticKind::GlobalIsolationViolation,
                    ident.span,
                    format!(
                        "task/ISR isolation violation: '{}' cannot access raw mutable global '{}'",
                        self.display_caller_name(),
                        ident.name
                    ),
                )
                .with_note(format!(
                    "raw global declared at {}; replace with `watch<{ty}> {name} = 0;` (read via `{name}.read()`, update via `{name}.publish(val)`) or `inbox<{ty}, 8> {name};` (queued events via `{name}.post(ev)` / `{name}.recv()`)",
                    ig.decl_span,
                    ty = ty_str,
                    name = ident.name
                )),
            );
            return ExprVal::scalar(ig.ty, ident.span);
        }

        // 3. Static OS primitive endpoint.
        if let Some(p) = self.sema.primitives.get(&ident.name).cloned() {
            return ExprVal {
                ty: Ty::OsPrimitive(p.primitive),
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::pure_static(),
                endpoint_name: Some(ident.name.clone()),
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }

        // 3b. Static subsystem struct instance (`uart0`).
        if let Some(sub) = self.sema.subsystems.get(&ident.name).cloned() {
            if self.in_isr() {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::GlobalIsolationViolation,
                        ident.span,
                        format!(
                            "ISR isolation violation: '{}' cannot access global subsystem instance '{}' directly",
                            self.display_caller_name(),
                            ident.name
                        ),
                    )
                    .with_note(format!(
                        "pass `&{name}` through `irq.attach({isr}(&{name}))` as a `*{st}` ISR parameter instead",
                        name = ident.name,
                        isr = self.display_caller_name(),
                        st = sub.struct_name,
                    )),
                );
            }
            return ExprVal {
                ty: Ty::Struct(sub.struct_name),
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::addr_of_static(),
                endpoint_name: Some(ident.name.clone()),
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }

        // 4. Immutable `const` global.
        if let Some(c) = self.sema.consts.get(&ident.name).cloned() {
            return ExprVal::scalar(c.ty, ident.span);
        }

        // 5. Unqualified enum variant (`Overcurrent`, `BadPacket`, `TooShort`, `BadCrc`, ...),
        //    scoped to enums visible in the current module.
        let visible_enums = self.sema.visible_enums_for_variant(self.prog_idx, &ident.name);
        if !visible_enums.is_empty() {
            if visible_enums.len() == 1 {
                return ExprVal::scalar(Ty::Enum(visible_enums[0].clone()), ident.span);
            }
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                ident.span,
                format!(
                    "ambiguous enum variant '{}'; qualify with enum name (candidates: {})",
                    ident.name,
                    visible_enums.join(", ")
                ),
            ));
            return ExprVal::error(ident.span);
        }

        // 6. Declared function (e.g. callback reference `register_cb(my_fn)`),
        //    MMIO peripheral instance (`USART1`), or hardware IRQ token (`IRQ_USART1`).
        if let Some(inst) = self.sema.mmio_instances.get(&ident.name).cloned() {
            return ExprVal {
                ty: Ty::MmioInstance {
                    name: inst.name.clone(),
                    layout: inst.layout,
                },
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::addr_of_static(),
                endpoint_name: Some(inst.name),
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }
        if let Some(iq) = self.sema.irqs.get(&ident.name).cloned() {
            return ExprVal {
                ty: Ty::IrqToken,
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::pure_static(),
                endpoint_name: Some(iq.name),
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }
        if let Some(fn_info) = self.sema.functions.get(&ident.name).cloned() {
            self.sema.check_extern_c_access(self.prog_idx, ident);
            let fn_ty = Ty::FnPtr {
                params: fn_info.params.iter().map(|p| p.ty.clone()).collect(),
                ret: Box::new(fn_info.ret_ty),
            };
            return ExprVal {
                ty: fn_ty,
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::fn_callback(ident.name.clone()),
                endpoint_name: None,
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }

        // 7. Built-in or type namespaces (`sys`, `sim`, `CmdFrame`, `ParseErr`).
        if ident.name == "sys" {
            return ExprVal::scalar(Ty::Namespace(NamespaceKind::Sys), ident.span);
        }
        if ident.name == "sim" {
            return ExprVal::scalar(Ty::Namespace(NamespaceKind::Sim), ident.span);
        }
        if self.sema.structs.contains_key(&ident.name) {
            return ExprVal::scalar(
                Ty::Namespace(NamespaceKind::Type(ident.name.clone())),
                ident.span,
            );
        }
        if self.sema.enums.contains_key(&ident.name) {
            return ExprVal::scalar(
                Ty::Namespace(NamespaceKind::Enum(ident.name.clone())),
                ident.span,
            );
        }

        let mut diag = Diagnostic::new(
            DiagnosticKind::UndefinedSymbol,
            ident.span,
            format!("undeclared identifier '{}'", ident.name),
        );
        if let Some(hint) = self.suggest_ident_hint(&ident.name) {
            diag = diag.with_note(hint);
        }
        self.sema.emit(diag);
        ExprVal::error(ident.span)
    }

    fn check_unary_expr(&mut self, op: UnaryOp, inner: &Expr, span: Span) -> ExprVal {
        match op {
            UnaryOp::AddrOf => {
                // `&fn_name`, `&MMIO_PERIPH`, `&irq`, `&const`.
                if let ExprKind::Ident(id) = &inner.kind {
                    if let Some(inst) = self.sema.mmio_instances.get(&id.name).cloned() {
                        return ExprVal {
                            ty: Ty::Ptr(Box::new(Ty::MmioLayout(inst.layout))),
                            error_spec: ResolvedErrorSpec::None,
                            prov: Provenance::addr_of_static(),
                            endpoint_name: Some(id.name.clone()),
                            field_endpoints: BTreeMap::new(),
                            span,
                        };
                    }
                    if self.sema.irqs.contains_key(&id.name) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::ImmutableAssignment,
                            span,
                            format!("cannot take address '&{}' of hardware IRQ token", id.name),
                        ));
                        return ExprVal::error(span);
                    }
                    if let Some(fn_info) = self.sema.functions.get(&id.name).cloned() {
                        self.sema.check_extern_c_access(self.prog_idx, id);
                        let fn_ty = Ty::FnPtr {
                            params: fn_info.params.iter().map(|p| p.ty.clone()).collect(),
                            ret: Box::new(fn_info.ret_ty),
                        };
                        return ExprVal {
                            ty: fn_ty,
                            error_spec: ResolvedErrorSpec::None,
                            prov: Provenance::fn_callback(id.name.clone()),
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        };
                    }
                    if self.sema.consts.contains_key(&id.name) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::ImmutableAssignment,
                            span,
                            format!("cannot take mutable pointer '&{}' to immutable 'const'", id.name),
                        ));
                        return ExprVal::error(span);
                    }
                }

                let (iv, prov, endpoint_name) = self.check_addr_of_place(inner, span);
                if iv.ty.is_error() {
                    return ExprVal::error(span);
                }
                if iv.ty.is_linear_lease() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        span,
                        "cannot take address `&` of linear Lease<T> variable (use `.get()` to obtain a tracked `*T` buffer pointer)",
                    ));
                    return ExprVal::error(span);
                }

                ExprVal {
                    ty: Ty::Ptr(Box::new(iv.ty)),
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name,
                    field_endpoints: iv.field_endpoints,
                    span,
                }
            }

            UnaryOp::Deref => {
                let iv = self.check_expr_infallible(inner);
                self.finish_deref_expr(iv, span, true)
            }

            UnaryOp::Not => {
                let iv = self.check_expr_infallible(inner);
                if !iv.ty.is_bool() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("logical '!' requires 'bool', found '{}'", iv.ty),
                    ));
                }
                ExprVal::scalar(Ty::bool(), span)
            }

            UnaryOp::BitNot => {
                let iv = self.check_expr_infallible(inner);
                if !iv.ty.is_integer() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("bitwise '~' requires an integer, found '{}'", iv.ty),
                    ));
                }
                ExprVal::scalar(iv.ty, span)
            }

            UnaryOp::Neg => {
                let iv = self.check_expr_infallible(inner);
                if !iv.ty.is_numeric() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("unary '-' requires an integer or float, found '{}'", iv.ty),
                    ));
                }
                let out_ty = match iv.ty {
                    Ty::IntLit => Ty::Prim(PrimType::I32),
                    other => other,
                };
                ExprVal::scalar(out_ty, span)
            }
        }
    }

    fn check_binary_expr(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr, span: Span) -> ExprVal {
        // S-H2: Short-circuit `&&` and `||` conditionally evaluate `rhs` only when `lhs` does not
        // short-circuit, so snapshot/fork variable move states & lease generations around `rhs`!
        if matches!(op, BinaryOp::And | BinaryOp::Or) {
            let lv = self.check_expr_infallible(lhs);
            let after_lhs_vars = self.snapshot_outer_vars();
            let after_lhs_leases = self.leases.clone();
            self.branch_depth += 1;
            let rv = self.check_expr_infallible(rhs);
            self.branch_depth -= 1;
            let after_rhs_vars = self.snapshot_outer_vars();
            let after_rhs_leases = self.leases.clone();

            self.scopes = after_lhs_vars.clone();
            self.merge_branch_vars(&after_lhs_vars, &[after_lhs_vars.clone(), after_rhs_vars], span);
            self.leases = after_lhs_leases.clone();
            self.leases.join_branches(&[after_lhs_leases, after_rhs_leases]);

            if lv.ty.is_error() || rv.ty.is_error() {
                return ExprVal::error(span);
            }
            if !lv.ty.is_bool() || !rv.ty.is_bool() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    span,
                    format!(
                        "logical '{}' requires 'bool' operands, found '{}' and '{}'",
                        op.as_str(),
                        lv.ty,
                        rv.ty
                    ),
                ));
            }
            return ExprVal::scalar(Ty::bool(), span);
        }

        let lv = self.check_expr_infallible(lhs);
        let rv = self.check_expr_infallible(rhs);
        if lv.ty.is_error() || rv.ty.is_error() {
            return ExprVal::error(span);
        }

        match op {
            BinaryOp::And | BinaryOp::Or => unreachable!(),

            BinaryOp::Eq | BinaryOp::Ne => {
                if lv.ty.is_linear_lease() || rv.ty.is_linear_lease() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        span,
                        "linear Lease<T> values cannot be compared directly with `==` / `!=` (compare `.addr()` instead)",
                    ));
                    return ExprVal::error(span);
                }
                let unified_float = unify_float_arith(&lv.ty, &rv.ty);
                let compat = lv.ty.is_assignable_to(&rv.ty)
                    || rv.ty.is_assignable_to(&lv.ty)
                    || (lv.ty.is_integer() && rv.ty.is_integer())
                    || unified_float.is_some();
                if !compat {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "cannot compare incompatible types '{}' and '{}' with '{}'",
                            lv.ty,
                            rv.ty,
                            op.as_str()
                        ),
                    ));
                } else if unified_float == Some(Ty::f32()) {
                    self.sema.check_f32_coercion_expr(lhs, &Ty::f32());
                    self.sema.check_f32_coercion_expr(rhs, &Ty::f32());
                }
                ExprVal::scalar(Ty::bool(), span)
            }

            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                let unified_float = unify_float_arith(&lv.ty, &rv.ty);
                let valid_cmp = (lv.ty.is_integer() && rv.ty.is_integer())
                    || unified_float.is_some();
                if !valid_cmp {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "comparison '{}' requires matching numeric operands, found '{}' and '{}'",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                } else if unified_float == Some(Ty::f32()) {
                    self.sema.check_f32_coercion_expr(lhs, &Ty::f32());
                    self.sema.check_f32_coercion_expr(rhs, &Ty::f32());
                }
                ExprVal::scalar(Ty::bool(), span)
            }

            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                if lv.ty.is_integer() && rv.ty.is_integer() {
                    let res_ty = unify_integer_arith(&lv.ty, &rv.ty);
                    ExprVal::scalar(res_ty, span)
                } else if let Some(float_ty) = unify_float_arith(&lv.ty, &rv.ty) {
                    if float_ty == Ty::f32() {
                        self.sema.check_f32_coercion_expr(lhs, &Ty::f32());
                        self.sema.check_f32_coercion_expr(rhs, &Ty::f32());
                    }
                    ExprVal::scalar(float_ty, span)
                } else {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "operator '{}' requires matching integer or float operands, found '{}' and '{}'",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                    ExprVal::error(span)
                }
            }

            BinaryOp::BitOr | BinaryOp::BitAnd | BinaryOp::BitXor => {
                if !lv.ty.is_integer() || !rv.ty.is_integer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "operator '{}' requires integer operands, found '{}' and '{}'",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                    return ExprVal::error(span);
                }
                let res_ty = unify_integer_arith(&lv.ty, &rv.ty);
                ExprVal::scalar(res_ty, span)
            }

            BinaryOp::Shl | BinaryOp::Shr => {
                if !lv.ty.is_integer() || !rv.ty.is_integer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "operator '{}' requires integer operands, found '{}' and '{}'",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                    return ExprVal::error(span);
                }
                // L-H1: Shift result takes the host integer type of the shifted left operand
                // rather than promoting to a wider right-hand shift amount (`0x8000_0000u32 >> 4i32` -> `u32`).
                let res_ty = match &lv.ty {
                    Ty::Prim(p) => Ty::Prim(Ty::host_prim(*p)),
                    Ty::IntLit => Ty::IntLit,
                    _ => lv.ty.clone(),
                };
                ExprVal::scalar(res_ty, span)
            }
        }
    }

    fn finish_deref_expr(&mut self, iv: ExprVal, span: Span, reject_byval: bool) -> ExprVal {
        match iv.ty {
            Ty::Ptr(pointee) => {
                if reject_byval {
                    if let Some(ln) = ty_contains_byval_mmio(&pointee) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::MmioViolation,
                            span,
                            format!(
                                "cannot dereference MMIO layout pointer '*{}' by value (access its registers via '.reg' projection)",
                                ln
                            ),
                        ));
                        return ExprVal::error(span);
                    }
                    if let Some(sub) = ty_contains_byval_subsys(&pointee, &self.sema.structs) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!(
                                "cannot dereference subsystem struct '{}' by value (access its fields or methods via pointer)",
                                sub
                            ),
                        ));
                        return ExprVal::error(span);
                    }
                    if ty_embeds_os_primitive(&pointee)
                        || ty_contains_bare_unsized_prim(&pointee).is_some()
                    {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!(
                                "cannot dereference OS primitive pointer '*{}' by value (use its methods via pointer)",
                                pointee
                            ),
                        ));
                        return ExprVal::error(span);
                    }
                }
                let prov = if pointee.contains_reference(&self.sema.structs)
                    || pointee.contains_fn_ptr(&self.sema.structs)
                {
                    iv.prov
                } else {
                    Provenance::pure_static()
                };
                ExprVal {
                    ty: *pointee,
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name: iv.endpoint_name,
                    field_endpoints: iv.field_endpoints,
                    span,
                }
            }
            Ty::Error => ExprVal::error(span),
            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    span,
                    format!("cannot dereference non-pointer type '{}'", other),
                ));
                ExprVal::error(span)
            }
        }
    }

    fn check_projection_base_allow_mmio_array(
        &mut self,
        expr: &Expr,
    ) -> (ExprVal, Option<(String, String, usize)>) {
        if let ExprKind::Unary {
            op: UnaryOp::Deref,
            expr: inner,
        } = &expr.kind
        {
            let iv = self.check_expr_infallible(inner);
            (self.finish_deref_expr(iv, expr.span, false), None)
        } else {
            self.check_expr_infallible_allow_mmio_array(expr)
        }
    }

    fn check_projection_base(&mut self, expr: &Expr) -> ExprVal {
        if let ExprKind::Unary {
            op: UnaryOp::Deref,
            expr: inner,
        } = &expr.kind
        {
            let iv = self.check_expr_infallible(inner);
            self.finish_deref_expr(iv, expr.span, false)
        } else {
            self.check_expr_infallible(expr)
        }
    }

    fn check_addr_of_place(
        &mut self,
        place: &Expr,
        addr_span: Span,
    ) -> (ExprVal, Provenance, Option<String>) {
        let (val, prov, ep, _) = self.check_addr_of_place_impl(place, addr_span, false);
        (val, prov, ep)
    }

    fn check_addr_of_place_impl(
        &mut self,
        place: &Expr,
        addr_span: Span,
        allow_mmio_array: bool,
    ) -> (
        ExprVal,
        Provenance,
        Option<String>,
        Option<(String, String, usize)>,
    ) {
        match &place.kind {
            ExprKind::Ident(id) => {
                let iv = self.check_expr_infallible(place);
                if iv.ty.is_error() {
                    return (iv, Provenance::pure_static(), None, None);
                }
                let mut endpoint_name = iv.endpoint_name.clone();
                let base_place_prov = if let Some(local) = self.lookup_local(&id.name) {
                    if endpoint_name.is_none() {
                        endpoint_name = local.endpoint_alias.clone();
                    }
                    if local.is_isr_state {
                        Provenance::addr_of_isr_state(&local.name, local.decl_depth)
                    } else {
                        Provenance::addr_of_local(&local.name, local.decl_depth)
                    }
                } else if self.sema.primitives.contains_key(&id.name)
                    || self.sema.subsystems.contains_key(&id.name)
                    || self.sema.mmio_instances.contains_key(&id.name)
                    || self.sema.consts.contains_key(&id.name)
                {
                    if endpoint_name.is_none() {
                        endpoint_name = Some(id.name.clone());
                    }
                    Provenance::addr_of_static()
                } else {
                    Provenance::addr_of_local("<temp>", self.current_depth())
                };
                let prov = base_place_prov.merge_place_with_val_origins(&iv.prov);
                (iv, prov, endpoint_name, None)
            }

            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: inner,
            } => {
                // `&(*p)` yields `p` directly without triggering by-value deref rejection and
                // preserves `p`'s backing pointer provenance (`leases`, depth, origins, endpoint).
                let ptr_v = self.check_expr_infallible(inner);
                let val = self.finish_deref_expr(ptr_v.clone(), place.span, false);
                if val.ty.is_error() {
                    return (val, Provenance::pure_static(), None, None);
                }
                let prov = ptr_v.prov.as_backing_ptr();
                let ep = val.endpoint_name.clone();
                (val, prov, ep, None)
            }

            ExprKind::Field { target, field } => {
                let (bv, target_place_prov, target_ep, target_mmio_arr) =
                    self.check_addr_of_place_impl(target, addr_span, true);
                if let Some((layout_name, reg_name, _)) = target_mmio_arr {
                    if field.name != "len" {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::MmioViolation,
                            place.span,
                            format!(
                                "cannot access '.{}' on MMIO register array '{}.{}' (only element indexing '[i]' and '.len' are allowed)",
                                field.name, layout_name, reg_name
                            ),
                        ));
                        return (ExprVal::error(place.span), Provenance::pure_static(), None, None);
                    }
                }
                if bv.ty.is_error() {
                    return (ExprVal::error(place.span), Provenance::pure_static(), None, None);
                }
                if is_mmio_ty(&bv.ty) {
                    let layout_name = match &bv.ty {
                        Ty::MmioLayout(ln) | Ty::MmioInstance { layout: ln, .. } => {
                            Some(ln.as_str())
                        }
                        Ty::Ptr(inner) => match &**inner {
                            Ty::MmioLayout(ln) => Some(ln.as_str()),
                            _ => None,
                        },
                        _ => None,
                    };
                    if allow_mmio_array {
                        if let Some(ln) = layout_name {
                            if let Some(reg) = self
                                .sema
                                .mmio_layouts
                                .get(ln)
                                .and_then(|l| l.regs.iter().find(|r| r.name == field.name))
                                .cloned()
                            {
                                if let Some(len) = reg.array_len {
                                    let reg_ty = Ty::Array {
                                        elem: Box::new(Ty::Prim(reg.elem_ty)),
                                        len,
                                    };
                                    return (
                                        ExprVal::scalar(reg_ty, place.span),
                                        Provenance::pure_static(),
                                        None,
                                        Some((ln.to_string(), reg.name, len)),
                                    );
                                }
                            }
                        }
                    }
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        addr_span,
                        format!(
                            "cannot take raw address `&` of MMIO register '{}'; access MMIO registers through `*Layout` pointers instead",
                            field.name
                        ),
                    ));
                    return (ExprVal::error(place.span), Provenance::pure_static(), None, None);
                }
                let fv = self.check_resolved_field_val_with_place(
                    bv.clone(),
                    Some(target_place_prov.clone()),
                    target,
                    field,
                    place.span,
                );
                if fv.ty.is_error() {
                    return (fv, Provenance::pure_static(), None, None);
                }
                let base_place_prov = if matches!(bv.ty, Ty::Ptr(_)) {
                    bv.prov.as_backing_ptr()
                } else {
                    target_place_prov
                };
                let prov = base_place_prov.merge_place_with_val_origins(&fv.prov);
                let ep = fv.endpoint_name.clone().or(target_ep);
                (fv, prov, ep, None)
            }

            ExprKind::Index { target: seq, index } => {
                let (seq_v, seq_place_prov, seq_ep, mmio_arr) =
                    self.check_addr_of_place_impl(seq, addr_span, true);
                let idx_v = self.check_expr_infallible(index);
                if !idx_v.ty.is_integer() && !idx_v.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        index.span,
                        format!("index must be an integer, found '{}'", idx_v.ty),
                    ));
                }
                if let Some((_, reg_name, _)) = mmio_arr {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        addr_span,
                        format!(
                            "cannot take raw address `&` of MMIO register '{}'; access MMIO registers through `*Layout` pointers instead",
                            reg_name
                        ),
                    ));
                    return (ExprVal::error(place.span), Provenance::pure_static(), None, None);
                }
                if seq_v.ty.is_error() {
                    return (ExprVal::error(place.span), Provenance::pure_static(), None, None);
                }
                let elem_v =
                    self.check_resolved_index_val(seq_v.clone(), seq, index.span, place.span);
                if elem_v.ty.is_error() {
                    return (elem_v, Provenance::pure_static(), None, None);
                }
                let base_place_prov = match &seq_v.ty {
                    Ty::Slice(_) | Ty::Ptr(_) => seq_v.prov.as_backing_ptr(),
                    _ => seq_place_prov,
                };
                let prov = base_place_prov.merge_place_with_val_origins(&elem_v.prov);
                let ep = elem_v.endpoint_name.clone().or(seq_ep);
                (elem_v, prov, ep, None)
            }

            _ => {
                let iv = self.check_expr_infallible(place);
                if iv.ty.is_error() {
                    return (iv, Provenance::pure_static(), None, None);
                }
                let base = Provenance::addr_of_local("<temp>", self.current_depth());
                let prov = base.merge_place_with_val_origins(&iv.prov);
                let ep = iv.endpoint_name.clone();
                (iv, prov, ep, None)
            }
        }
    }

    fn check_field_expr(&mut self, target: &Expr, field: &Ident, span: Span) -> ExprVal {
        if field.name == "ptr" {
            let (tv, target_place_prov, _) = self.check_addr_of_place(target, span);
            if tv.ty.is_error() {
                return ExprVal::error(span);
            }
            return self.check_resolved_field_val_with_place(
                tv,
                Some(target_place_prov),
                target,
                field,
                span,
            );
        }
        let (tv, mmio_arr) = self.check_projection_base_allow_mmio_array(target);
        if let Some((layout_name, reg_name, _)) = mmio_arr {
            if field.name != "len" {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    span,
                    format!(
                        "cannot access '.{}' on MMIO register array '{}.{}' (only element indexing '[i]' and '.len' are allowed)",
                        field.name, layout_name, reg_name
                    ),
                ));
                return ExprVal::error(span);
            }
        }
        self.check_resolved_field_val_with_place(tv, None, target, field, span)
    }

    fn check_resolved_field_val_with_place(
        &mut self,
        tv: ExprVal,
        target_place_prov: Option<Provenance>,
        target: &Expr,
        field: &Ident,
        span: Span,
    ) -> ExprVal {
        if tv.ty.is_error() {
            return ExprVal::error(span);
        }

        // Auto-dereference `*Struct` / `*Slice` / `*MmioLayout` on field access.
        let deref_ty = match &tv.ty {
            Ty::Ptr(inner) => (**inner).clone(),
            other => other.clone(),
        };

        match deref_ty {
            Ty::Slice(_) => {
                if field.name == "len" {
                    ExprVal::scalar(Ty::usize(), span)
                } else if field.name == "ptr" {
                    if let Ty::Slice(elem) = deref_ty {
                        ExprVal {
                            ty: Ty::Ptr(elem),
                            error_spec: ResolvedErrorSpec::None,
                            prov: tv.prov.as_backing_ptr(),
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    } else {
                        unreachable!()
                    }
                } else {
                    let note = if matches!(field.name.as_str(), "length" | "size" | "count" | "cap") {
                        "did you mean `.len`? (slices provide `.len` of type `usize` and `.ptr` of type `*T`)".to_string()
                    } else if let Some(best) = closest_match(&field.name, ["len", "ptr"], 2) {
                        format!("did you mean `.{}`?", best)
                    } else {
                        "slices provide `.len` (`usize`) and `.ptr` (`*T`)".to_string()
                    };
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UndefinedSymbol,
                            field.span,
                            format!("slice has no property '{}' (supported: 'len', 'ptr')", field.name),
                        )
                        .with_note(note),
                    );
                    ExprVal::error(span)
                }
            }

            Ty::Array { len: _, elem } => {
                if field.name == "len" {
                    ExprVal::scalar(Ty::usize(), span)
                } else if field.name == "ptr" {
                    let prov = if matches!(tv.ty, Ty::Ptr(_)) {
                        tv.prov.as_backing_ptr()
                    } else {
                        target_place_prov.unwrap_or(tv.prov)
                    };
                    ExprVal {
                        ty: Ty::Ptr(elem),
                        error_spec: ResolvedErrorSpec::None,
                        prov,
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                } else {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        field.span,
                        format!("array has no property '{}'", field.name),
                    ));
                    ExprVal::error(span)
                }
            }

            Ty::Struct(struct_name) => {
                let Some(f_info) = self
                    .sema
                    .structs
                    .get(&struct_name)
                    .and_then(|s| s.field(&field.name))
                    .cloned()
                else {
                    let hint = self.suggest_struct_field_hint(&struct_name, &field.name);
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UndefinedSymbol,
                            field.span,
                            format!("struct '{}' has no field '{}'", struct_name, field.name),
                        )
                        .with_note(hint),
                    );
                    return ExprVal::error(span);
                };

                let is_subsys_field = matches!(&f_info.ty, Ty::OsPrimitive(_))
                    || matches!(&f_info.ty, Ty::Ptr(inner) if matches!(&**inner, Ty::OsPrimitive(_)))
                    || matches!(
                        &f_info.ty,
                        Ty::Struct(s) if self.sema.structs.get(s).is_some_and(|st| st.is_subsystem)
                    )
                    || matches!(
                        &f_info.ty,
                        Ty::Ptr(inner) if matches!(&**inner, Ty::Struct(s) if self.sema.structs.get(s).is_some_and(|st| st.is_subsystem))
                    );
                let is_view_field = is_subsys_field
                    || ty_contains_os_prim_view(&f_info.ty, &self.sema.structs, &mut BTreeSet::new());
                let prov = if is_view_field
                    || f_info.ty.contains_reference(&self.sema.structs)
                    || f_info.ty.contains_fn_ptr(&self.sema.structs)
                {
                    if let Some(root) = root_ident_of(target) {
                        if let Some(v) = self.lookup_local(&root) {
                            if !matches!(v.ty, Ty::Ptr(_)) {
                                v.field_prov
                                    .get(&field.name)
                                    .cloned()
                                    .unwrap_or_else(|| tv.prov.clone())
                            } else {
                                tv.prov.clone()
                            }
                        } else {
                            tv.prov.clone()
                        }
                    } else {
                        tv.prov.clone()
                    }
                } else {
                    Provenance::pure_static()
                };

                let mut out_field_endpoints = BTreeMap::new();
                let prefix = format!("{}.", field.name);
                for (k, v) in &tv.field_endpoints {
                    if let Some(rest) = k.strip_prefix(&prefix) {
                        out_field_endpoints.insert(rest.to_string(), v.clone());
                    }
                }

                let endpoint_name = if let Some(ep) = tv.field_endpoints.get(&field.name) {
                    Some(ep.clone())
                } else if is_view_field {
                    tv.endpoint_name
                        .as_ref()
                        .cloned()
                        .or_else(|| root_ident_of(target))
                        .map(|parent| format!("{parent}.{}", field.name))
                } else {
                    None
                };

                ExprVal {
                    ty: f_info.ty,
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name,
                    field_endpoints: out_field_endpoints,
                    span,
                }
            }

            Ty::MmioLayout(layout_name) | Ty::MmioInstance { layout: layout_name, .. } => {
                let Some(layout_info) = self.sema.mmio_layouts.get(&layout_name).cloned() else {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        field.span,
                        format!("unknown MMIO register layout '{}'", layout_name),
                    ));
                    return ExprVal::error(span);
                };
                let Some(reg) = layout_info.regs.iter().find(|r| r.name == field.name).cloned()
                else {
                    let candidates = layout_info.regs.iter().map(|r| r.name.as_str());
                    let mut diag = Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        field.span,
                        format!(
                            "MMIO layout '{}' has no register '{}'",
                            layout_name, field.name
                        ),
                    );
                    if let Some(best) = closest_match(&field.name, candidates, 2) {
                        diag = diag.with_note(format!("did you mean register `.{}`?", best));
                    }
                    self.sema.emit(diag);
                    return ExprVal::error(span);
                };
                if reg.access == MmioAccess::Wo {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        span,
                        format!(
                            "cannot read write-only `@wo` MMIO register '{}.{}'",
                            layout_name, reg.name
                        ),
                    ));
                }
                let reg_ty = if let Some(len) = reg.array_len {
                    self.mmio_array_expr_spans.insert(
                        (span.start, span.end),
                        (layout_name, reg.name, len),
                    );
                    Ty::Array {
                        elem: Box::new(Ty::Prim(reg.elem_ty)),
                        len,
                    }
                } else {
                    Ty::Prim(reg.elem_ty)
                };
                ExprVal::scalar(reg_ty, span)
            }

            Ty::Namespace(NamespaceKind::Enum(enum_name)) => {
                if self
                    .sema
                    .enums
                    .get(&enum_name)
                    .is_some_and(|e| e.has_variant(&field.name))
                {
                    ExprVal::scalar(Ty::Enum(enum_name), span)
                } else {
                    let hint = self.suggest_enum_variant_hint(&enum_name, &field.name);
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UndefinedSymbol,
                            field.span,
                            format!("enum '{}' has no variant '{}'", enum_name, field.name),
                        )
                        .with_note(hint),
                    );
                    ExprVal::error(span)
                }
            }

            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    field.span,
                    format!("type '{}' has no field '{}'", other, field.name),
                ));
                ExprVal::error(span)
            }
        }
    }

    fn check_call_expr(&mut self, callee: &Expr, args: &[CallArg], span: Span) -> ExprVal {
        // Normalize `a.b(args)` if parser represented it as `Call { callee: Field { target, field }, args }`.
        if let ExprKind::Field { target, field } = &callee.kind {
            // Unless `target` is a struct whose `field` is a `FnPtr`! Check if `target` has a `FnPtr` field first.
            let is_struct_fn_ptr = if let Some(root) = root_ident_of(target) {
                if let Some(v) = self.lookup_local(&root) {
                    let st_ty = match &v.ty {
                        Ty::Struct(s) => Some(s.as_str()),
                        Ty::Ptr(inner) => match &**inner {
                            Ty::Struct(s) => Some(s.as_str()),
                            _ => None,
                        },
                        _ => None,
                    };
                    st_ty
                        .and_then(|sn| self.sema.structs.get(sn))
                        .and_then(|s| s.field(&field.name))
                        .is_some_and(|f| matches!(f.ty, Ty::FnPtr { .. }))
                } else {
                    false
                }
            } else {
                false
            };
            if !is_struct_fn_ptr {
                return self.check_method_call(target, field, args, span);
            }
        }

        if let ExprKind::Ident(ident) = &callee.kind {
            // 1. User-declared or `extern "C"` function.
            if let Some(fn_info) = self.sema.functions.get(&ident.name).cloned() {
                self.sema.check_extern_c_access(self.prog_idx, ident);
                return self.check_direct_fn_call(&fn_info, args, span);
            }

            // 2. Calling a `task` blueprint like a regular function is an error (use `spawn`!).
            if self.sema.tasks.contains_key(&ident.name) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    ident.span,
                    format!(
                        "cannot call task blueprint '{}' directly; instantiate it with 'spawn {}(...)'",
                        ident.name, ident.name
                    ),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                return ExprVal::error(span);
            }

            // 3. Built-in runtime functions (`set_priority`, `checkpoint`, `delay`, `sleep`, `print`, `assert`, ...).
            if let Some(res) = self.try_check_builtin_fn(&ident.name, args, span) {
                return res;
            }
        }

        // 4. Indirect call through a local `FnPtr` variable or expression.
        let cv = self.check_expr_infallible(callee);
        match cv.ty {
            Ty::FnPtr { params, ret } => {
                if args.len() != params.len() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "function pointer expects {} argument(s), found {}",
                            params.len(),
                            args.len()
                        ),
                    ));
                }
                for (i, arg) in args.iter().enumerate() {
                    let av = self.check_expr_infallible(&arg.value);
                    if av.ty.is_linear_lease() {
                        self.consume_lease_value(&arg.value, &av, "function pointer argument", arg.span);
                    }
                    if let Some(expected) = params.get(i) {
                        if self.sema.check_assignable_or_emit(
                            &av.ty,
                            expected,
                            arg.span,
                            DiagnosticKind::TypeMismatch,
                            || format!("argument {} has type '{}', expected '{}'", i + 1, av.ty, expected),
                        ) {
                            self.sema.check_f32_coercion_expr(&arg.value, expected);
                        }
                    }
                }
                for cb_target in &cv.prov.fn_callbacks {
                    self.sema.call_graph.add_edge(CallEdge {
                        caller: self.caller_key.clone(),
                        caller_kind: self.caller_kind,
                        callee: cb_target.clone(),
                        span,
                        ctx: self.call_context(),
                    });
                }
                ExprVal::scalar(*ret, span)
            }
            Ty::Error => {
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                ExprVal::error(span)
            }
            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    callee.span,
                    format!("expression of type '{}' is not callable", other),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                ExprVal::error(span)
            }
        }
    }

    fn check_direct_fn_call(
        &mut self,
        fn_info: &FunctionInfo,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        // Record call-graph edge for whole-program yield inference and recursion cycle detection.
        self.sema.call_graph.add_edge(CallEdge {
            caller: self.caller_key.clone(),
            caller_kind: self.caller_kind,
            callee: fn_info.name.clone(),
            span,
            ctx: self.call_context(),
        });

        if (!fn_info.is_variadic && args.len() != fn_info.params.len())
            || (fn_info.is_variadic && args.len() < fn_info.params.len())
        {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "function '{}' expects {}{} argument(s), found {}",
                    fn_info.name,
                    if fn_info.is_variadic { "at least " } else { "" },
                    fn_info.params.len(),
                    args.len()
                ),
            ));
        }

        let mut arg_vals = Vec::with_capacity(args.len());
        let mut boot_call_args = Vec::with_capacity(args.len());
        let mut boot_attach_args = Vec::with_capacity(args.len());
        for (i, arg) in args.iter().enumerate() {
            let av = self.check_expr_infallible(&arg.value);
            if av.ty.is_linear_lease() {
                self.consume_lease_value(&arg.value, &av, "function argument", arg.span);
            }
            if let Some(param) = fn_info.params.get(i) {
                if self.sema.check_assignable_or_emit(
                    &av.ty,
                    &param.ty,
                    arg.span,
                    DiagnosticKind::TypeMismatch,
                    || {
                        format!(
                            "argument {} ('{}') to '{}' has type '{}', expected '{}'",
                            i + 1,
                            param.name,
                            fn_info.name,
                            av.ty,
                            param.ty
                        )
                    },
                ) {
                    self.sema.check_f32_coercion_expr(&arg.value, &param.ty);
                }
            }

            // If calling an `extern "C"` function, record any user `fn` callbacks passed in `arg`!
            if fn_info.is_extern_c {
                for cb_fn in &av.prov.fn_callbacks {
                    self.sema.call_graph.add_extern_callback(ExternCallbackSite {
                        extern_fn: fn_info.name.clone(),
                        callback_fn: cb_fn.clone(),
                        span: arg.span,
                    });
                }
            }

            let endpoint_from_param = self.resolve_param_endpoint_index(&av.endpoint_name);
            let static_endpoint = av
                .endpoint_name
                .clone()
                .filter(|ep| !ep.starts_with("param:"));
            let field_endpoints = self.extract_field_endpoint_specs(&av);
            let has_non_static_local_ref = av.prov.has_non_returning_origin();
            let has_isr_state_ref = !av.prov.isr_state_origins.is_empty();
            boot_call_args.push(BootCallArg {
                static_endpoint,
                endpoint_from_param,
                field_endpoints,
                has_non_static_local_ref,
                has_isr_state_ref,
                ref_param_origins: av.prov.param_origins.clone(),
                span: arg.span,
            });
            boot_attach_args.push(BootAttachCallArg {
                spec: self.classify_irq_attach_arg(&arg.value, &av),
            });
            arg_vals.push(av);
        }

        if !fn_info.is_extern_c {
            let in_main_post_boot = self.in_main_post_boot();
            self.sema.recorded_call_args.push(RecordedCallArgs {
                caller: self.caller_key.clone(),
                callee: fn_info.name.clone(),
                args: boot_call_args.clone(),
                in_main_post_boot,
            });
            if self.caller_key == "main" {
                let (spawns, attaches) = self.sema.fn_boot_flags(&fn_info.name);
                if spawns {
                    self.has_spawned = true;
                }
                if attaches {
                    self.has_attached_irq = true;
                }
            }
            if self.can_record_boot_step() {
                self.boot_steps.push(BootStep::HelperCall {
                    callee: fn_info.name.clone(),
                    conditional: self.branch_depth > 0,
                    args: boot_call_args,
                });
                if let Some(top) = self.boot_block_stack.last_mut() {
                    top.push(IrqBootNode::Call {
                        callee: fn_info.name.clone(),
                        args: boot_attach_args,
                    });
                }
            }
        }

        // Compute return value provenance if `fn_info.ret_ty` carries references/slices.
        let mut ret_prov = Provenance::pure_static();
        if fn_info.ret_ty.contains_reference(&self.sema.structs) {
            if !fn_info.return_param_origins.is_empty() {
                for &idx in &fn_info.return_param_origins {
                    if let Some(av) = arg_vals.get(idx) {
                        ret_prov = ret_prov.merge(&av.prov);
                    }
                }
            } else {
                // Conservative fallback for forward/mutual non-yielding helpers not yet analyzed:
                // propagate reference-bearing arguments into reference-bearing return value.
                for (i, av) in arg_vals.iter().enumerate() {
                    if fn_info
                        .params
                        .get(i)
                        .is_some_and(|p| p.ty.contains_reference(&self.sema.structs))
                    {
                        ret_prov = ret_prov.merge(&av.prov);
                    }
                }
            }
        }

        ExprVal {
            ty: fn_info.ret_ty.clone(),
            error_spec: fn_info.error_spec.clone(),
            prov: ret_prov,
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }

    fn try_check_builtin_fn(
        &mut self,
        name: &str,
        args: &[CallArg],
        span: Span,
    ) -> Option<ExprVal> {
        match name {
            "set_priority" => {
                if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        span,
                        format!(
                            "cannot call 'set_priority' inside ISR '{}'",
                            self.display_caller_name()
                        ),
                    ));
                }
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'set_priority' expects 1 priority argument, found {}", args.len()),
                    ));
                }
                for a in args {
                    let av = self.check_expr_infallible(&a.value);
                    if av.ty != Ty::Priority && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            a.span,
                            format!("'set_priority' expects a priority literal (@critical..@low), found '{}'", av.ty),
                        ));
                    }
                    if self.caller_key == "main" {
                        if let ExprKind::Priority(prio) = a.value.kind {
                            self.sema.main_supervisor_priority = Some(prio);
                        }
                    }
                }
                Some(ExprVal::scalar(Ty::void(), span))
            }

            "checkpoint" => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        "'checkpoint()' takes 0 arguments",
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                self.record_direct_yield(span, FnEffect::SoftYielding, "'checkpoint()'");
                Some(ExprVal::scalar(Ty::void(), span))
            }

            "delay" | "sleep" => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'{}' expects 1 duration argument, found {}", name, args.len()),
                    ));
                }
                for a in args {
                    let av = self.check_expr_infallible(&a.value);
                    if av.ty != Ty::Duration && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            a.span,
                            format!("'{}' expects a duration literal (e.g. 10ms), found '{}'", name, av.ty),
                        ));
                    }
                }
                self.record_direct_yield(span, FnEffect::HardBlocking, "'delay()'");
                Some(ExprVal::scalar(Ty::void(), span))
            }

            "now_ticks" | "now_ms" => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'{}()' takes 0 arguments, found {}", name, args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                Some(ExprVal::scalar(Ty::u32(), span))
            }

            "print" | "println" => {
                if args.len() != 1 {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "'{}' expects 1 argument (`*u8`, string literal, or `f\"...\"` format string), found {}",
                                name,
                                args.len()
                            ),
                        )
                        .with_note(format!(
                            "pass `{}(f\"val={{x}}\");` or `{}(\"boot ok\");`",
                            name, name
                        )),
                    );
                    for a in args {
                        if let ExprKind::FormatString(parts) = &a.value.kind {
                            let _ = self.check_format_string(parts);
                        } else {
                            let _ = self.check_expr_infallible(&a.value);
                        }
                    }
                } else if let ExprKind::FormatString(parts) = &args[0].value.kind {
                    let _ = self.check_format_string(parts);
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !is_print_arg_ty(&av.ty) && !av.ty.is_error() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                args[0].span,
                                format!(
                                    "'{}' expects a string or byte slice (`*u8`, `[]u8`, string literal, or `f\"...\"`), found '{}'",
                                    name, av.ty
                                ),
                            )
                            .with_note(format!(
                                "use `{}(f\"val={{expr}}\");`, `sys.log_kv(\"val=\", expr);`, or `sys.log_hex(\"val=\", expr);` to print values",
                                name
                            )),
                        );
                    }
                }
                Some(ExprVal::scalar(Ty::void(), span))
            }

            "fmt_buf" => Some(self.check_fmt_buf_call(args, span)),
            "fmt_ring" => Some(self.check_fmt_ring_call(args, span)),

            "assert" => {
                if args.len() != 1 {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!("'assert' expects 1 'bool' condition argument, found {}", args.len()),
                        )
                        .with_note("pass a boolean comparison such as `assert(count > 0);`"),
                    );
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !av.ty.is_bool() && !av.ty.is_error() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                args[0].span,
                                format!("'assert' expects a 'bool' condition, found '{}'", av.ty),
                            )
                            .with_note("pass a boolean comparison such as `assert(val == expected);`"),
                        );
                    }
                }
                Some(ExprVal::scalar(Ty::void(), span))
            }

            _ => None,
        }
    }

    fn check_method_call(
        &mut self,
        receiver: &Expr,
        method: &Ident,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        let rv = self.check_projection_base(receiver);
        if rv.ty.is_error() {
            for a in args {
                let _ = self.check_expr_infallible(&a.value);
            }
            return ExprVal::error(span);
        }

        // Auto-deref `*OsPrimitive` / `*Struct` on method calls (`rx_dma.wait_chunk()`, `target_rpm.read()`).
        let deref_ty = match &rv.ty {
            Ty::Ptr(inner) => (**inner).clone(),
            other => other.clone(),
        };

        match deref_ty {
            Ty::Namespace(NamespaceKind::Sys) => self.check_sys_method(method, args, span),

            Ty::Namespace(NamespaceKind::Sim) => {
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                let suggestion = match method.name.as_str() {
                    "log" => "did you mean `sys.log_kv(\"label=\", val)` or `sys.log(\"msg\")`?",
                    "now_ms" | "now_ticks" => "did you mean `sys.now_ms()` or `sys.now_ticks()`?",
                    "exit" | "stop" => "did you mean `sys.exit(0)`?",
                    "pend_irq" | "fire_isr" => "did you mean `sys.pend_irq(irq_num)`?",
                    _ => "`sim.*` is not a built-in Zev namespace; use portable `sys.*` runtime methods instead",
                };
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        method.span,
                        format!("unknown method 'sim.{}' — {}", method.name, suggestion),
                    )
                    .with_note(VALID_SYS_METHODS_NOTE),
                );
                ExprVal::error(span)
            }

            Ty::Namespace(NamespaceKind::Type(type_name)) => {
                if method.name == "from_bytes" {
                    if args.len() != 1 {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "'{}.from_bytes' expects 1 '[]u8' argument, found {}",
                                type_name,
                                args.len()
                            ),
                        ));
                    }
                    let expected_slice = Ty::Slice(Box::new(Ty::u8()));
                    for a in args {
                        let av = self.check_expr_infallible(&a.value);
                        if !av.ty.is_assignable_to(&expected_slice) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                a.span,
                                format!(
                                    "'{}.from_bytes' expects '[]u8', found '{}'",
                                    type_name, av.ty
                                ),
                            ));
                        }
                    }
                    let struct_ty = Ty::Struct(type_name.clone());
                    if self
                        .sema
                        .structs
                        .get(&type_name)
                        .is_some_and(|s| s.is_subsystem)
                    {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!(
                                "'{}.from_bytes' cannot construct subsystem struct '{}' (requires a wire-value struct)",
                                type_name, type_name
                            ),
                        ));
                    } else if struct_ty.contains_reference(&self.sema.structs) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!(
                                "'{}.from_bytes' requires a wire-value struct without pointer or slice fields",
                                type_name
                            ),
                        ));
                    }
                    ExprVal {
                        ty: struct_ty,
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                } else {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UndefinedSymbol,
                            method.span,
                            format!(
                                "unknown associated function '{}.{}'",
                                type_name, method.name
                            ),
                        )
                        .with_note(format!(
                            "did you mean `{}.from_bytes(slice)`? (decodes a `[]u8` wire slice into `{}`)",
                            type_name, type_name
                        )),
                    );
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                    ExprVal::error(span)
                }
            }

            Ty::OsPrimitive(os_prim) => {
                let source_key = rv
                    .endpoint_name
                    .clone()
                    .or_else(|| root_ident_of(receiver))
                    .unwrap_or_else(|| "<anon_endpoint>".to_string());
                self.check_os_primitive_method(&os_prim, &source_key, method, args, span)
            }

            Ty::Lease(elem) => {
                let mut pos_vals = Vec::new();
                let mut has_boost = false;
                for a in args {
                    let av = self.check_expr_infallible(&a.value);
                    if let Some(arg_name) = &a.name {
                        match arg_name.name.as_str() {
                            "timeout" => {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::InvalidOperation,
                                    a.span,
                                    format!("`timeout:` is not valid on `Lease.{}`", method.name),
                                ));
                            }
                            "boost" => {
                                has_boost = true;
                                if av.ty != Ty::Priority && !av.ty.is_error() {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::TypeMismatch,
                                        a.span,
                                        format!(
                                            "'boost:' must be a priority (@critical..@low), found '{}'",
                                            av.ty
                                        ),
                                    ));
                                }
                                if let ExprKind::Priority(prio) = a.value.kind {
                                    self.recorded_boosts.push(PriorityBoostSite {
                                        primitive_name: "<Lease>".to_string(),
                                        boost_priority: prio,
                                        span: a.span,
                                    });
                                }
                            }
                            other => {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    arg_name.span,
                                    format!(
                                        "unknown named argument '{}:' on `Lease.{}`",
                                        other, method.name
                                    ),
                                ));
                            }
                        }
                    } else {
                        pos_vals.push((av, a.value.clone(), a.span));
                    }
                }
                match method.name.as_str() {
                    "get" => {
                        self.expect_positional_arity("get", &pos_vals, 0, span);
                        if has_boost {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                span,
                                "`boost:` is only valid on `Lease.release(...)`",
                            ));
                        }
                        if let ExprKind::Ident(id) = &receiver.kind {
                            if let Some(var) = self.lookup_local(&id.name).cloned() {
                                if let Some(key) = var.lease_gen_key {
                                    let gen = self.leases.acquire_next(&key, span);
                                    let prov = Provenance::dma_lease_get(
                                        key,
                                        gen,
                                        var.decl_depth,
                                        span,
                                    );
                                    return ExprVal {
                                        ty: Ty::Ptr(elem),
                                        error_spec: ResolvedErrorSpec::None,
                                        prov,
                                        endpoint_name: None,
                                        field_endpoints: BTreeMap::new(),
                                        span,
                                    };
                                }
                            }
                            ExprVal::scalar(Ty::Ptr(elem), span)
                        } else {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::LinearLeaseViolation,
                                span,
                                "cannot borrow `.get()` from a temporary unnamed `Lease<T>` (bind it to a local variable and release it after use)",
                            ));
                            ExprVal::error(span)
                        }
                    }
                    "addr" => {
                        self.expect_positional_arity("addr", &pos_vals, 0, span);
                        if has_boost {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                span,
                                "`boost:` is only valid on `Lease.release(...)`",
                            ));
                        }
                        if !matches!(receiver.kind, ExprKind::Ident(_)) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::LinearLeaseViolation,
                                span,
                                "cannot call `.addr()` on a temporary unnamed `Lease<T>` (bind it to a local variable first)",
                            ));
                            return ExprVal::error(span);
                        }
                        ExprVal::scalar(Ty::usize(), span)
                    }
                    "release" => {
                        self.expect_positional_arity("release", &pos_vals, 0, span);
                        self.consume_lease_value(receiver, &rv, "release", span);
                        ExprVal::scalar(Ty::void(), span)
                    }
                    _ => {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::UndefinedSymbol,
                                method.span,
                                format!("linear `Lease<{}>` has no method '{}'", elem, method.name),
                            )
                            .with_note("valid methods on `Lease<T>`: `.get() -> *T`, `.addr() -> usize`, `.release([boost: @prio])`"),
                        );
                        ExprVal::error(span)
                    }
                }
            }

            Ty::Struct(struct_name) => {
                if let Some(FieldInfo {
                    ty: Ty::FnPtr { params, ret },
                    ..
                }) = self
                    .sema
                    .structs
                    .get(&struct_name)
                    .and_then(|s| s.field(&method.name))
                    .cloned()
                {
                    let field_prov = if let Some(root) = root_ident_of(receiver) {
                        if let Some(v) = self.lookup_local(&root) {
                            v.field_prov
                                .get(&method.name)
                                .cloned()
                                .unwrap_or_else(|| rv.prov.clone())
                        } else {
                            rv.prov.clone()
                        }
                    } else {
                        rv.prov.clone()
                    };
                    if args.len() != params.len() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "function pointer field '{}.{}' expects {} argument(s), found {}",
                                struct_name,
                                method.name,
                                params.len(),
                                args.len()
                            ),
                        ));
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let av = self.check_expr_infallible(&arg.value);
                        if av.ty.is_linear_lease() {
                            self.consume_lease_value(&arg.value, &av, "function pointer field argument", arg.span);
                        }
                        if let Some(expected) = params.get(i) {
                            if !av.ty.is_assignable_to(expected) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    arg.span,
                                    format!(
                                        "argument {} to '{}.{}' has type '{}', expected '{}'",
                                        i + 1,
                                        struct_name,
                                        method.name,
                                        av.ty,
                                        expected
                                    ),
                                ));
                            } else {
                                self.sema.check_f32_coercion_expr(&arg.value, expected);
                            }
                        }
                    }
                    for cb_target in &field_prov.fn_callbacks {
                        self.sema.call_graph.add_edge(CallEdge {
                            caller: self.caller_key.clone(),
                            caller_kind: self.caller_kind,
                            callee: cb_target.clone(),
                            span,
                            ctx: self.call_context(),
                        });
                    }
                    return ExprVal::scalar(*ret, span);
                }

                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    method.span,
                    format!("type '{}' has no method '{}'", struct_name, method.name),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                ExprVal::error(span)
            }

            Ty::IrqToken => self.check_irq_method(&rv, method, args, span),

            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    method.span,
                    format!("type '{}' has no method '{}'", other, method.name),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                ExprVal::error(span)
            }
        }
    }

    fn check_sys_method(&mut self, method: &Ident, args: &[CallArg], span: Span) -> ExprVal {
        let m = method.name.as_str();
        match m {
            "use_systick" => {
                if args.is_empty() || args.len() > 2 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.use_systick' expects 1 argument (u32 hz), found {}",
                            args.len()
                        ),
                    ));
                }
                if let Some(hz_arg) = args.first() {
                    let hv = self.check_expr_infallible(&hz_arg.value);
                    if !hv.ty.is_integer() && !hv.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            hz_arg.span,
                            format!(
                                "'sys.use_systick' clock frequency must be an integer (`u32`), found '{}'",
                                hv.ty
                            ),
                        ));
                    }
                }
                // Optional second argument configures custom tick period (`sys.use_systick(hz, 500us)`).
                if let Some(tick_arg) = args.get(1) {
                    if !matches!(&tick_arg.value.kind, ExprKind::Literal(Literal::Duration(d)) if d.as_micros() > 0)
                    {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidDurationResolution,
                            tick_arg.span,
                            "'sys.use_systick' tick period must be a non-zero duration literal",
                        ));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "set_tick" | "set_tick_period" => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.{}' expects 1 duration literal argument", m),
                    ));
                } else if !matches!(&args[0].value.kind, ExprKind::Literal(Literal::Duration(d)) if d.as_micros() > 0)
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidDurationResolution,
                        args[0].span,
                        format!("'sys.{}' requires a non-zero duration literal", m),
                    ));
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "set_tick_us" => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        "'sys.set_tick_us' expects 1 integer microseconds argument",
                    ));
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !av.ty.is_integer() && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!(
                                "'sys.set_tick_us' expects integer microseconds, found '{}'",
                                av.ty
                            ),
                        ));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "delay" => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.delay' expects 1 duration argument, found {}", args.len()),
                    ));
                }
                for a in args {
                    let av = self.check_expr_infallible(&a.value);
                    if av.ty != Ty::Duration && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            a.span,
                            format!("'sys.delay' expects duration literal, found '{}'", av.ty),
                        ));
                    }
                }
                self.record_direct_yield(span, FnEffect::HardBlocking, "sys.delay()");
                ExprVal::scalar(Ty::void(), span)
            }

            "tick" | "tick_ms" => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.{}()' takes 0 arguments, found {}", m, args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "tick_by" => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.tick_by(u32 ms)' expects 1 argument, found {}", args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !av.ty.is_assignable_to(&Ty::u32()) && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!("'sys.tick_by' expects `u32` milliseconds, found '{}'", av.ty),
                        ));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "set_putc" => {
                if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        span,
                        "cannot call 'sys.set_putc' inside an ISR",
                    ));
                }
                let expected_ty = Ty::FnPtr {
                    params: vec![Ty::u8()],
                    ret: Box::new(Ty::void()),
                };
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.set_putc(fn(u8))' expects 1 argument, found {}",
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    self.check_sys_callback_arg(
                        "set_putc",
                        "console byte-sink",
                        &args[0],
                        &expected_ty,
                    );
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "set_clock_ms" | "set_exit" | "set_fault_handler" | "set_idle" => {
                if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        span,
                        format!("cannot call 'sys.{}' inside an ISR", m),
                    ));
                }
                let (sig_desc, role_desc, expected_ty) = match m {
                    "set_clock_ms" => (
                        "fn() -> u32",
                        "clock-source",
                        Ty::FnPtr {
                            params: vec![],
                            ret: Box::new(Ty::u32()),
                        },
                    ),
                    "set_exit" => (
                        "fn(i32)",
                        "exit-handler",
                        Ty::FnPtr {
                            params: vec![Ty::Prim(PrimType::I32)],
                            ret: Box::new(Ty::void()),
                        },
                    ),
                    "set_fault_handler" => (
                        "fn()",
                        "fault-handler",
                        Ty::FnPtr {
                            params: vec![],
                            ret: Box::new(Ty::void()),
                        },
                    ),
                    "set_idle" => (
                        "fn(u32, u32)",
                        "idle-hook",
                        Ty::FnPtr {
                            params: vec![Ty::u32(), Ty::u32()],
                            ret: Box::new(Ty::void()),
                        },
                    ),
                    _ => unreachable!(),
                };
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.{}({})' expects 1 argument, found {}",
                            m,
                            sig_desc,
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    self.check_sys_callback_arg(m, role_desc, &args[0], &expected_ty);
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "set_irq_router" => {
                if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        span,
                        "cannot call 'sys.set_irq_router' inside an ISR",
                    ));
                }
                let route_ty = Ty::FnPtr {
                    params: vec![Ty::u32(), Ty::u32()],
                    ret: Box::new(Ty::void()),
                };
                let ack_ty = Ty::FnPtr {
                    params: vec![Ty::u32()],
                    ret: Box::new(Ty::void()),
                };
                if args.len() != 2 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.set_irq_router(fn(u32, u32), fn(u32))' expects 2 arguments, found {}",
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    self.check_sys_callback_arg("set_irq_router", "IRQ-route", &args[0], &route_ty);
                    self.check_sys_callback_arg("set_irq_router", "IRQ-ack", &args[1], &ack_ty);
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "rom_call_u32" => self.check_sys_rom_call(m, args, span, true, Ty::u32()),
            "rom_call_void" => self.check_sys_rom_call(m, args, span, true, Ty::void()),
            "rom_call_noreturn" => self.check_sys_rom_call(m, args, span, false, Ty::void()),

            "now_ticks" | "now_ms" | "current_prio" | "sleep_depth" | "wfi_count" => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.{}()' takes 0 arguments, found {}", m, args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                ExprVal::scalar(Ty::u32(), span)
            }

            "trace_sleep" => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.trace_sleep(bool on)' expects 1 argument, found {}", args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if av.ty != Ty::bool() && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!("'sys.trace_sleep' expects 'bool', found '{}'", av.ty),
                        ));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "nop" => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.{}()' takes 0 arguments, found {}", m, args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "pend_irq" => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.pend_irq(u32 irq)' expects 1 argument, found {}", args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if av.ty != Ty::IrqToken
                        && !av.ty.is_assignable_to(&Ty::u32())
                        && !av.ty.is_error()
                    {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!(
                                "'sys.pend_irq' expects `u32` or `irq` vector token, found '{}'",
                                av.ty
                            ),
                        ));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "exit" => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.exit(i32 code)' expects 1 argument, found {}", args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !av.ty.is_assignable_to(&Ty::Prim(PrimType::I32)) && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!("'sys.exit' expects `i32` exit code, found '{}'", av.ty),
                        ));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "log" => {
                if args.len() != 1 {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!("'sys.log(*u8 msg)' expects 1 string argument, found {}", args.len()),
                        )
                        .with_note("for labeled integer telemetry, use `sys.log_kv(\"label=\", val);` or `sys.log_hex(\"reg=\", val);`"),
                    );
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !is_string_arg_ty(&av.ty) && !av.ty.is_error() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                args[0].span,
                                format!("'sys.log' expects a string (`*u8`), found '{}'", av.ty),
                            )
                            .with_note("use `sys.log_kv(\"label=\", val);` or `sys.log_hex(\"label=\", val);` to log numeric values"),
                        );
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            "log_kv" | "log_hex" => {
                let val_ty = if m == "log_kv" {
                    Ty::Prim(PrimType::I32)
                } else {
                    Ty::u32()
                };
                if args.len() != 2 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.{}(*u8 label, {} val)' expects 2 arguments, found {}",
                            m,
                            val_ty,
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let label_val = self.check_expr_infallible(&args[0].value);
                    if !is_string_arg_ty(&label_val.ty) && !label_val.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!(
                                "argument 1 ('label') to 'sys.{}' must be a string (`*u8`), found '{}'",
                                m, label_val.ty
                            ),
                        ));
                    }
                    let num_val = self.check_expr_infallible(&args[1].value);
                    if !num_val.ty.is_assignable_to(&val_ty) && !num_val.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[1].span,
                            format!(
                                "argument 2 ('val') to 'sys.{}' must be '{}', found '{}'",
                                m, val_ty, num_val.ty
                            ),
                        ));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            other => {
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                let valid_sys = [
                    "use_systick",
                    "tick",
                    "tick_ms",
                    "tick_by",
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
                    "nop",
                    "exit",
                    "log",
                    "log_kv",
                    "log_hex",
                    "delay",
                ];
                let msg = if matches!(other, "print" | "println" | "puts") {
                    format!(
                        "unknown runtime method 'sys.{}' — did you mean top-level `println(\"...\")` or `sys.log(\"...\")`?",
                        other
                    )
                } else if let Some(best) = closest_match(other, valid_sys, 2) {
                    format!(
                        "unknown runtime method 'sys.{}' — did you mean `sys.{}(...)`?",
                        other, best
                    )
                } else {
                    format!("unknown runtime method 'sys.{}'", other)
                };
                self.sema.emit(
                    Diagnostic::new(DiagnosticKind::UndefinedSymbol, method.span, msg)
                        .with_note(VALID_SYS_METHODS_NOTE),
                );
                ExprVal::error(span)
            }
        }
    }

    fn check_sys_callback_arg(
        &mut self,
        method_name: &str,
        role_desc: &str,
        arg: &CallArg,
        expected_ty: &Ty,
    ) {
        let av = self.check_expr_infallible(&arg.value);
        if av.ty != *expected_ty && !av.ty.is_error() {
            let msg = if method_name == "set_putc" {
                format!(
                    "'sys.set_putc' expects callback of type '{}', found '{}'",
                    expected_ty, av.ty
                )
            } else {
                format!(
                    "'sys.{}' expects {} callback of type '{}', found '{}'",
                    method_name, role_desc, expected_ty, av.ty
                )
            };
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                arg.span,
                msg,
            ));
        }
        for cb_fn in &av.prov.fn_callbacks {
            if let Some(fn_info) = self.sema.functions.get(cb_fn) {
                if fn_info.error_spec.is_fallible() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        arg.span,
                        format!(
                            "cannot pass fallible function '{}' (`error`) to 'sys.{}': {} callbacks must not fail",
                            cb_fn, method_name, role_desc
                        ),
                    ));
                }
            }
            self.sema
                .call_graph
                .add_extern_callback(ExternCallbackSite {
                    extern_fn: format!("sys.{}", method_name),
                    callback_fn: cb_fn.clone(),
                    span: arg.span,
                });
        }
    }

    fn check_sys_rom_call(
        &mut self,
        method_name: &str,
        args: &[CallArg],
        span: Span,
        has_stack_bound: bool,
        ret_ty: Ty,
    ) -> ExprVal {
        let fixed = if has_stack_bound { 2 } else { 1 };
        let max_args = fixed + 4;
        if args.len() < fixed || args.len() > max_args {
            let sig = if has_stack_bound {
                format!("'sys.{}(usize fn_addr, usize stack_bound, ...args)'", method_name)
            } else {
                format!("'sys.{}(usize fn_addr, ...args)'", method_name)
            };
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "{} expects {}..={} arguments (up to 4 ABI register args), found {}",
                    sig,
                    fixed,
                    max_args,
                    args.len()
                ),
            ));
            for a in args {
                let _ = self.check_expr_infallible(&a.value);
            }
            return ExprVal::scalar(ret_ty, span);
        }

        let addr_val = self.check_expr_infallible(&args[0].value);
        if !addr_val.ty.is_assignable_to(&Ty::usize()) && !addr_val.ty.is_error() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                args[0].span,
                format!(
                    "'sys.{}' argument 1 (`fn_addr`) expects `usize` address, found '{}'",
                    method_name, addr_val.ty
                ),
            ));
        }

        if has_stack_bound {
            let bound_val = self.check_expr_infallible(&args[1].value);
            if !bound_val.ty.is_assignable_to(&Ty::usize()) && !bound_val.ty.is_error() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    args[1].span,
                    format!(
                        "'sys.{}' argument 2 (`stack_bound`) expects `usize` bytes, found '{}'",
                        method_name, bound_val.ty
                    ),
                ));
            }
            let const_bound = self
                .eval_const_usize_expr(&args[1].value)
                .and_then(|v| u32::try_from(v).ok());
            if const_bound.is_none() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    args[1].span,
                    format!(
                        "'sys.{}' argument 2 (`stack_bound`) must be a non-negative compile-time constant integer (`0..={}`)",
                        method_name,
                        u32::MAX
                    ),
                ));
            }
        }

        for (idx, a) in args[fixed..].iter().enumerate() {
            let av = self.check_expr_infallible(&a.value);
            if !av.ty.is_integer() && !matches!(av.ty, Ty::Ptr(_)) && !av.ty.is_error() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    a.span,
                    format!(
                        "'sys.{}' ABI argument {} must be an integer or pointer, found '{}'",
                        method_name,
                        fixed + idx + 1,
                        av.ty
                    ),
                ));
            }
        }

        ExprVal::scalar(ret_ty, span)
    }

    fn check_os_primitive_method(
        &mut self,
        prim: &ResolvedOsPrimitive,
        source_key: &str,
        method: &Ident,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        // Partition positional vs named arguments (`timeout: <dur>`, `boost: @prio`).
        let mut pos_vals = Vec::new();
        let mut has_timeout = false;
        let mut has_boost = false;

        for a in args {
            let av = self.check_expr_infallible(&a.value);
            if let Some(arg_name) = &a.name {
                match arg_name.name.as_str() {
                    "timeout" => {
                        has_timeout = true;
                        if av.ty != Ty::Duration && !av.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                a.span,
                                format!("'timeout:' must be a duration literal, found '{}'", av.ty),
                            ));
                        }
                    }
                    "boost" => {
                        has_boost = true;
                        if av.ty != Ty::Priority && !av.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                a.span,
                                format!(
                                    "'boost:' must be a priority (@critical..@low), found '{}'",
                                    av.ty
                                ),
                            ));
                        }
                        if let ExprKind::Priority(prio) = a.value.kind {
                            self.recorded_boosts.push(PriorityBoostSite {
                                primitive_name: source_key.to_string(),
                                boost_priority: prio,
                                span: a.span,
                            });
                        }
                    }
                    other => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            arg_name.span,
                            format!("unknown named argument '{}:' on '{}'", other, prim),
                        ));
                    }
                }
            } else {
                pos_vals.push((av, a.value.clone(), a.span));
            }
        }

        let m = method.name.as_str();
        match prim {
            ResolvedOsPrimitive::Signal => match m {
                "wait" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.wait()", source_key),
                    );
                    ExprVal {
                        ty: Ty::void(),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "notify" | "clear" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "try_wait" | "is_set" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::bool(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::Watch { elem } => match m {
                "read" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar((**elem).clone(), span)
                }
                "publish" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "wait_change" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.wait_change()", source_key),
                    );
                    ExprVal {
                        ty: (**elem).clone(),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::Inbox { elem, .. } | ResolvedOsPrimitive::InboxUnsized { elem } => match m {
                "post" | "try_post" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    if elem.is_linear_lease() {
                        if let Some((av, ae, asp)) = pos_vals.first() {
                            self.consume_lease_value(ae, av, if m == "post" { "inbox.post" } else { "inbox.try_post" }, *asp);
                        }
                    }
                    ExprVal::scalar(Ty::void(), span)
                }
                "try_send" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    if elem.is_linear_lease() {
                        if let Some((av, ae, asp)) = pos_vals.first() {
                            self.consume_lease_value(ae, av, "inbox.try_send", *asp);
                        }
                    }
                    ExprVal::scalar(Ty::bool(), span)
                }
                "send" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    if elem.is_linear_lease() {
                        if let Some((av, ae, asp)) = pos_vals.first() {
                            self.consume_lease_value(ae, av, "inbox.send", *asp);
                        }
                    }
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.send()", source_key),
                    );
                    ExprVal {
                        ty: Ty::void(),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "recv" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.recv()", source_key),
                    );
                    ExprVal {
                        ty: (**elem).clone(),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "try_recv" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal {
                        ty: (**elem).clone(),
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "len" | "cap" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::usize(), span)
                }
                "is_empty" | "is_full" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::bool(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::PingPong { elem, .. } => match m {
                "flip" | "start" | "activate" | "stop" | "deactivate" | "release" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "complete_half" => {
                    self.expect_single_elem(m, &pos_vals, &Ty::u8(), span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "wait_chunk" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.wait_chunk()", source_key),
                    );
                    let gen = self.leases.acquire_next(source_key, span);
                    let prov =
                        Provenance::dma_lease(source_key.to_string(), gen, self.current_depth(), span);
                    ExprVal {
                        ty: Ty::Slice(elem.clone()),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov,
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "active_buf" | "dma_buf" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::Slice(elem.clone()), span)
                }
                "half_len" | "cap" | "base_addr" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::usize(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::Pool { elem, .. } => match m {
                "try_alloc" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`pool.try_alloc()` is non-blocking and does not accept `timeout:` (use `pool.alloc(timeout: <dur>)`)",
                        ));
                    }
                    if has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`pool.try_alloc()` does not accept `boost:`",
                        ));
                    }
                    ExprVal {
                        ty: Ty::Lease(elem.clone()),
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "alloc" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`pool.alloc()` does not accept `boost:`",
                        ));
                    }
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.alloc()", source_key),
                    );
                    ExprVal {
                        ty: Ty::Lease(elem.clone()),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "avail" | "in_use" | "cap" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!("`pool.{}()` does not accept `timeout:` or `boost:`", m),
                        ));
                    }
                    ExprVal::scalar(Ty::usize(), span)
                }
                "is_exhausted" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`pool.is_exhausted()` does not accept `timeout:` or `boost:`",
                        ));
                    }
                    ExprVal::scalar(Ty::bool(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::DmaSlot { elem } => match m {
                "park" => {
                    self.expect_single_elem(m, &pos_vals, &Ty::Lease(elem.clone()), span);
                    if has_timeout {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`dma_slot.park()` does not accept `timeout:`",
                        ));
                    }
                    if let Some((av, ae, asp)) = pos_vals.first() {
                        self.consume_lease_value(ae, av, "dma_slot.park", *asp);
                    }
                    ExprVal {
                        ty: Ty::void(),
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "take" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`dma_slot.take()` does not accept `timeout:` or `boost:`",
                        ));
                    }
                    ExprVal {
                        ty: Ty::Lease(elem.clone()),
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "release" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`dma_slot.release()` does not accept `timeout:`",
                        ));
                    }
                    ExprVal::scalar(Ty::bool(), span)
                }
                "is_busy" | "is_empty" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::bool(), span)
                }
                "addr" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::usize(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::ByteRing { .. } | ResolvedOsPrimitive::ByteRingUnsized => {
                let spsc_role = match m {
                    "push" | "push_byte" | "grant_write" | "commit_write" => {
                        Some(ByteRingOpRole::Producer)
                    }
                    "wait_read" | "read_slice" | "pop" | "try_pop_byte" | "consume" => {
                        Some(ByteRingOpRole::Consumer)
                    }
                    "clear" => Some(ByteRingOpRole::Both),
                    _ => None,
                };
                if let Some(role) = spsc_role {
                    self.record_byte_ring_op(role, m, source_key, span);
                }
                let u8_slice = Ty::Slice(Box::new(Ty::u8()));
                match m {
                    "push" => {
                        self.expect_single_elem(m, &pos_vals, &u8_slice, span);
                        let wkey = format!("{source_key}#write");
                        let _ = self.leases.invalidate(&wkey, span, "push");
                        ExprVal::scalar(Ty::usize(), span)
                    }
                    "push_byte" => {
                        self.expect_single_elem(m, &pos_vals, &Ty::u8(), span);
                        let wkey = format!("{source_key}#write");
                        let _ = self.leases.invalidate(&wkey, span, "push_byte");
                        ExprVal::scalar(Ty::bool(), span)
                    }
                    "grant_write" => {
                        self.expect_single_elem(m, &pos_vals, &Ty::usize(), span);
                        let wkey = format!("{source_key}#write");
                        let gen = self.leases.invalidate(&wkey, span, "grant_write");
                        let prov = Provenance::ring_lease(
                            wkey,
                            gen,
                            self.current_depth(),
                            span,
                            "grant_write",
                        );
                        ExprVal {
                            ty: u8_slice,
                            error_spec: ResolvedErrorSpec::Untyped,
                            prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    }
                    "commit_write" => {
                        self.expect_single_elem(m, &pos_vals, &Ty::usize(), span);
                        let wkey = format!("{source_key}#write");
                        let _ = self.leases.invalidate(&wkey, span, "commit_write");
                        ExprVal::scalar(Ty::void(), span)
                    }
                    "read_slice" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        let gen = self.leases.acquire_current(source_key);
                        let prov = Provenance::ring_lease(
                            source_key.to_string(),
                            gen,
                            self.current_depth(),
                            span,
                            "read_slice",
                        );
                        ExprVal {
                            ty: u8_slice,
                            error_spec: ResolvedErrorSpec::None,
                            prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    }
                    "wait_read" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        self.record_direct_yield(
                            span,
                            FnEffect::HardBlocking,
                            &format!("{}.wait_read()", source_key),
                        );
                        let gen = self.leases.acquire_current(source_key);
                        let prov = Provenance::ring_lease(
                            source_key.to_string(),
                            gen,
                            self.current_depth(),
                            span,
                            "wait_read",
                        );
                        ExprVal {
                            ty: u8_slice,
                            error_spec: if has_timeout {
                                ResolvedErrorSpec::Untyped
                            } else {
                                ResolvedErrorSpec::None
                            },
                            prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    }
                    "consume" => {
                        self.expect_single_elem(m, &pos_vals, &Ty::usize(), span);
                        let _ = self.leases.invalidate(source_key, span, "consume");
                        ExprVal::scalar(Ty::void(), span)
                    }
                    "pop" => {
                        self.expect_single_elem(m, &pos_vals, &u8_slice, span);
                        if pos_vals
                            .first()
                            .is_some_and(|(v, _, _)| matches!(v.ty, Ty::Str))
                        {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                pos_vals[0].2,
                                "cannot call 'pop' into an immutable string literal; pass a mutable `[N]u8` or `[]u8` buffer",
                            ));
                        }
                        let _ = self.leases.invalidate(source_key, span, "pop");
                        ExprVal::scalar(Ty::usize(), span)
                    }
                    "try_pop_byte" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        let _ = self.leases.invalidate(source_key, span, "try_pop_byte");
                        ExprVal {
                            ty: Ty::u8(),
                            error_spec: ResolvedErrorSpec::Untyped,
                            prov: Provenance::pure_static(),
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    }
                    "clear" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        let _ = self.leases.invalidate(source_key, span, "clear");
                        let wkey = format!("{source_key}#write");
                        let _ = self.leases.invalidate(&wkey, span, "clear");
                        ExprVal::scalar(Ty::void(), span)
                    }
                    "len" | "cap" | "free_space" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        ExprVal::scalar(Ty::usize(), span)
                    }
                    "is_empty" | "is_full" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        ExprVal::scalar(Ty::bool(), span)
                    }
                    _ => self.unknown_prim_method(prim, method, span),
                }
            }
        }
    }

    fn expect_positional_arity(
        &mut self,
        method: &str,
        pos_vals: &[(ExprVal, Expr, Span)],
        expected: usize,
        span: Span,
    ) {
        if pos_vals.len() != expected {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "method '{}' expects {} positional argument(s), found {}",
                    method,
                    expected,
                    pos_vals.len()
                ),
            ));
        }
    }

    fn expect_single_elem(
        &mut self,
        method: &str,
        pos_vals: &[(ExprVal, Expr, Span)],
        elem: &Ty,
        span: Span,
    ) {
        if pos_vals.len() != 1 {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "method '{}' expects 1 value argument of type '{}', found {}",
                    method,
                    elem,
                    pos_vals.len()
                ),
            ));
            return;
        }
        let (val, arg_expr, arg_span) = &pos_vals[0];
        if !val.prov.isr_state_origins.is_empty() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidIsrState,
                *arg_span,
                format!(
                    "cannot pass `&` of ISR `state` variable to OS primitive method '{}'",
                    method
                ),
            ));
        }
        if !val.prov.param_origins.is_empty() {
            self.sema
                .prim_escaping_ref_params
                .entry(self.caller_key.clone())
                .or_default()
                .extend(val.prov.param_origins.iter().copied());
        }
        if self.sema.check_assignable_or_emit(
            &val.ty,
            elem,
            *arg_span,
            DiagnosticKind::TypeMismatch,
            || {
                format!(
                    "argument to '{}' has type '{}', expected '{}'",
                    method, val.ty, elem
                )
            },
        ) {
            self.sema.check_f32_coercion_expr(arg_expr, elem);
        }
    }

    fn unknown_prim_method(
        &mut self,
        prim: &ResolvedOsPrimitive,
        method: &Ident,
        span: Span,
    ) -> ExprVal {
        let (did_you_mean, valid_table) = advise_primitive_method(prim, &method.name);
        let msg = if let Some(suggestion) = &did_you_mean {
            format!(
                "OS primitive '{}' has no method '{}' — did you mean {}?",
                prim, method.name, suggestion
            )
        } else {
            format!("OS primitive '{}' has no method '{}'", prim, method.name)
        };
        self.sema.emit(
            Diagnostic::new(DiagnosticKind::UndefinedSymbol, method.span, msg)
                .with_note(valid_table),
        );
        ExprVal::error(span)
    }

    fn check_catch_expr(
        &mut self,
        inner: &Expr,
        clause: &CatchClause,
        span: Span,
    ) -> ExprVal {
        let iv = self.check_expr_raw(inner);
        let _ = self.reject_unindexed_mmio_array(inner.span);
        let _ = self
            .proven_infallible_spans
            .remove(&(inner.span.start, inner.span.end));

        if !iv.error_spec.is_fallible() && !iv.ty.is_error() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidCatchSignature,
                clause.span,
                "cannot attach 'catch' handler to an infallible expression",
            ));
        }

        // Validate catch parameter count (0, 1, or 2) and types against `(iv.error_spec, iv.ty)`.
        let mut bound_err: Option<(String, Ty, Span)> = None;
        let mut bound_partial: Option<(String, Ty, Span)> = None;

        if let Some(ep) = &clause.err_param {
            let decl_err_ty = self.sema.resolve_type(&ep.ty);
            self.sema
                .check_type_no_byval_mmio(&decl_err_ty, ep.ty.span, "'catch' error parameter");
            self.sema.check_disallowed_lease_type(
                &decl_err_ty,
                ep.ty.span,
                "'catch' error parameter",
                false,
            );
            match &iv.error_spec {
                ResolvedErrorSpec::None => {}
                ResolvedErrorSpec::Untyped => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidCatchSignature,
                        ep.span,
                        format!(
                            "'catch' binds error parameter '{}', but expression has untyped bare 'error' (use 'catch {{ ... }}')",
                            ep.name.name
                        ),
                    ));
                }
                ResolvedErrorSpec::Typed(actual_err_ty) => {
                    if !actual_err_ty.is_assignable_to(&decl_err_ty) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidCatchSignature,
                            ep.span,
                            format!(
                                "'catch' error parameter '{}' declared as '{}', but expression fails with '{}'",
                                ep.name.name, decl_err_ty, actual_err_ty
                            ),
                        ));
                    }
                }
            }
            bound_err = Some((ep.name.name.clone(), decl_err_ty, ep.span));
        }

        if let Some(pp) = &clause.partial_param {
            if clause.err_param.is_none() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidCatchSignature,
                    pp.span,
                    "'catch' partial-progress parameter requires an error parameter first",
                ));
            }
            let decl_part_ty = self.sema.resolve_type(&pp.ty);
            self.sema.check_type_no_byval_mmio(
                &decl_part_ty,
                pp.ty.span,
                "'catch' partial-progress parameter",
            );
            if decl_part_ty.is_linear_lease()
                || iv.ty.is_linear_lease()
                || ty_contains_linear_lease(&decl_part_ty)
            {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::LinearLeaseViolation,
                    pp.ty.span,
                    "cannot bind linear Lease<T> in 'catch' partial-progress parameter (linear leases cannot be partially produced on error)",
                ));
            } else {
                self.sema.check_disallowed_lease_type(
                    &decl_part_ty,
                    pp.ty.span,
                    "'catch' partial-progress parameter",
                    false,
                );
            }
            if matches!(iv.error_spec, ResolvedErrorSpec::Untyped) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidCatchSignature,
                    pp.span,
                    "'catch' cannot bind partial-progress parameter on untyped bare 'error'",
                ));
            } else if iv.ty.is_void() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidCatchSignature,
                    pp.span,
                    "'catch' cannot bind partial-progress parameter on a 'void' expression",
                ));
            } else if !iv.ty.is_assignable_to(&decl_part_ty) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidCatchSignature,
                    pp.span,
                    format!(
                        "'catch' partial parameter '{}' declared as '{}', expected '{}'",
                        pp.name.name, decl_part_ty, iv.ty
                    ),
                ));
            }
            bound_partial = Some((pp.name.name.clone(), decl_part_ty, pp.span));
        }

        let base_vars = self.snapshot_outer_vars();
        let base_leases = self.leases.clone();

        let (catch_div, trailing_val) = self.with_nested_scope(|this| {
            let depth = this.current_depth();
            if let Some((ename, ety, espan)) = bound_err {
                let init_lease_state = if ety.is_linear_lease() {
                    Some(LinearLeaseState::Moved {
                        moved_at: espan,
                        reason: "invalid catch error binding",
                    })
                } else {
                    None
                };
                this.scopes.last_mut().unwrap().insert(
                    ename.clone(),
                    LocalVar {
                        name: ename,
                        ty: ety,
                        decl_depth: depth,
                        prov: Provenance::pure_static(),
                        field_prov: BTreeMap::new(),
                        endpoint_alias: None,
                        field_endpoints: BTreeMap::new(),
                        is_isr_state: false,
                        lease_state: init_lease_state,
                        lease_gen_key: None,
                    },
                );
            }
            if let Some((pname, pty, pspan)) = bound_partial {
                if let Some(ep) = &clause.err_param {
                    if ep.name.name == pname {
                        this.sema.emit(Diagnostic::new(
                            DiagnosticKind::DuplicateDefinition,
                            pspan,
                            format!("duplicate parameter '{}' in 'catch' clause", pname),
                        ));
                    }
                }
                let init_lease_state = if pty.is_linear_lease() {
                    Some(LinearLeaseState::Moved {
                        moved_at: pspan,
                        reason: "invalid catch partial binding",
                    })
                } else {
                    None
                };
                this.scopes.last_mut().unwrap().insert(
                    pname.clone(),
                    LocalVar {
                        name: pname,
                        ty: pty,
                        decl_depth: depth,
                        prov: iv.prov.clone(),
                        field_prov: BTreeMap::new(),
                        endpoint_alias: None,
                        field_endpoints: BTreeMap::new(),
                        is_isr_state: false,
                        lease_state: init_lease_state,
                        lease_gen_key: None,
                    },
                );
            }

            if !iv.ty.is_void() && !clause.body.stmts.is_empty() {
                let (prefix, last) = clause.body.stmts.split_at(clause.body.stmts.len() - 1);
                let (mut div, warned) = this.check_stmt_slice(prefix);
                let prev_after = this.after_diverge;
                if div {
                    if !warned {
                        this.sema.emit(Diagnostic::new(
                            DiagnosticKind::UnreachableCode,
                            last[0].span,
                            "unreachable statement following diverging control flow",
                        ));
                    }
                    this.after_diverge = true;
                }
                if let StmtKind::Expr(last_expr) = &last[0].kind {
                    let lv = this.check_expr_infallible(last_expr);
                    if lv.ty.is_linear_lease() {
                        this.consume_lease_value(last_expr, &lv, "catch fallback", last_expr.span);
                    }
                    let sys_exit =
                        this.lookup_local("sys").is_none() && is_sys_exit_expr(last_expr);
                    let res_div = div || sys_exit;
                    this.finish_lexical_block(res_div, clause.span);
                    if sys_exit {
                        this.after_diverge = prev_after;
                        return (true, None);
                    }
                    this.after_diverge = prev_after;
                    return (div, Some(lv));
                }
                if this.check_stmt(&last[0]) {
                    div = true;
                }
                this.finish_lexical_block(div, clause.span);
                this.after_diverge = prev_after;
                (div, None)
            } else {
                let div = this.check_block_stmts(&clause.body.stmts);
                this.finish_lexical_block(div, clause.span);
                (div, None)
            }
        });

        let catch_vars = self.snapshot_outer_vars();
        let catch_leases = self.leases.clone();

        let mut merged_prov = iv.prov.clone();
        if !catch_div {
            self.scopes = base_vars.clone();
            self.merge_branch_vars(&base_vars, &[base_vars.clone(), catch_vars], clause.span);
            self.leases = base_leases.clone();
            self.leases.join_branches(&[base_leases, catch_leases]);

            if !iv.ty.is_void() && !iv.ty.is_error() {
                if let Some(tv) = trailing_val {
                    if !tv.ty.is_assignable_to(&iv.ty) && !tv.ty.is_error() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::InvalidCatchSignature,
                                tv.span,
                                format!(
                                    "'catch' fallback expression has type '{}', expected '{}'",
                                    tv.ty, iv.ty
                                ),
                            )
                            .with_note(format!(
                                "a non-diverging 'catch' block for a '{}' expression must end with an expression assignable to '{}'",
                                iv.ty, iv.ty
                            )),
                        );
                    } else if let Some(Stmt {
                        kind: StmtKind::Expr(last_expr),
                        ..
                    }) = clause.body.stmts.last()
                    {
                        self.sema.check_f32_coercion_expr(last_expr, &iv.ty);
                    }
                    merged_prov = merged_prov.merge(&tv.prov);
                } else {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidCatchSignature,
                            clause.span,
                            format!(
                                "'catch' block for non-void '{}' expression must either diverge ('return'/'error'/'continue'/'break') or end with a fallback expression of type '{}'",
                                iv.ty, iv.ty
                            ),
                        )
                        .with_note(format!(
                            "either diverge inside `catch` (`return;`, `continue;`, `break;`, `error ...;`) or end the block with a replacement `{}` value (e.g. `catch (...) {{ count += 1; fallback_val }}`)",
                            iv.ty
                        )),
                    );
                }
            }
        } else {
            self.scopes = base_vars;
            self.leases = base_leases;
        }

        ExprVal {
            ty: iv.ty,
            error_spec: ResolvedErrorSpec::None,
            prov: merged_prov,
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Zero-Heap `f"..."` Format Verification & `E0028` Static Fit Proof
    // ─────────────────────────────────────────────────────────────────────────

    fn check_fmt_buf_call(&mut self, args: &[CallArg], span: Span) -> ExprVal {
        let ret_slice_ty = Ty::Slice(Box::new(Ty::u8()));
        let fmt_err_spec = ResolvedErrorSpec::Typed(Ty::Enum("FmtErr".to_string()));

        if args.len() != 2 {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "'fmt_buf' expects 2 arguments (dst, f\"...\"), found {}",
                    args.len()
                ),
            ));
            for a in args {
                if let ExprKind::FormatString(parts) = &a.value.kind {
                    let _ = self.check_format_string(parts);
                } else {
                    let _ = self.check_expr_infallible(&a.value);
                }
            }
            self.proven_infallible_spans.insert((span.start, span.end));
            return ExprVal {
                ty: ret_slice_ty,
                error_spec: fmt_err_spec,
                prov: Provenance::pure_static(),
                endpoint_name: None,
                field_endpoints: BTreeMap::new(),
                span,
            };
        }

        let dst_expr = &args[0].value;
        let dv = self.check_expr_infallible(dst_expr);
        let is_valid_dst = match &dv.ty {
            Ty::Slice(elem) | Ty::Array { elem, .. } => **elem == Ty::u8(),
            Ty::Ptr(inner) => match &**inner {
                Ty::Slice(elem) | Ty::Array { elem, .. } => **elem == Ty::u8(),
                _ => false,
            },
            Ty::Error => true,
            _ => false,
        };
        if !is_valid_dst {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                args[0].span,
                format!(
                    "'fmt_buf' destination must be '[]u8', '[N]u8', or '*[N]u8', found '{}'",
                    dv.ty
                ),
            ));
        }

        let mut ret_prov = dv.prov.clone();
        if matches!(dv.ty, Ty::Array { .. }) {
            if let Some(root) = root_ident_of(dst_expr) {
                if let Some(var) = self.lookup_local(&root) {
                    ret_prov = ret_prov.merge(&Provenance::addr_of_local(&var.name, var.decl_depth));
                }
            }
        }
        let mut dst_alias_prov = ret_prov.clone();
        if let Some(root) = root_ident_of(dst_expr) {
            if let Some(var) = self.lookup_local(&root) {
                if var.ty.contains_reference(&self.sema.structs)
                    || matches!(var.ty, Ty::Array { .. })
                {
                    dst_alias_prov = dst_alias_prov
                        .merge(&Provenance::addr_of_local(&var.name, var.decl_depth));
                }
            }
        }

        let worst_len = if let ExprKind::FormatString(parts) = &args[1].value.kind {
            self.check_format_string_with_dst(parts, Some(&dst_alias_prov))
        } else {
            let _ = self.check_expr_infallible(&args[1].value);
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                args[1].span,
                "'fmt_buf' second argument must be an 'f\"...\"' format string",
            ));
            None
        };

        if let (Some(cap), Some(worst)) = (self.static_dst_capacity(dst_expr, &dv.ty), worst_len) {
            if worst > cap {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::FormatBufferTooSmall,
                        span,
                        format!(
                            "format string worst-case length ({} bytes) exceeds destination buffer capacity ({} bytes)",
                            worst, cap
                        ),
                    )
                    .with_note(format!(
                        "increase the destination buffer to at least `[{}]u8`, or format into a `[]u8` slice and handle truncation with `catch (FmtErr err, []u8 clamped) {{ ... }}`",
                        worst
                    )),
                );
                self.proven_infallible_spans.insert((span.start, span.end));
            } else {
                // Worst-case length is statically proven to fit inside `cap`!
                self.proven_infallible_spans.insert((span.start, span.end));
            }
        }

        ExprVal {
            ty: ret_slice_ty,
            error_spec: fmt_err_spec,
            prov: ret_prov,
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }

    fn check_fmt_ring_call(&mut self, args: &[CallArg], span: Span) -> ExprVal {
        let fmt_err_spec = ResolvedErrorSpec::Typed(Ty::Enum("FmtErr".to_string()));
        let mut pos_args = Vec::new();
        let mut boost_prio: Option<(Priority, Span)> = None;

        for a in args {
            if let Some(arg_name) = &a.name {
                if arg_name.name == "boost" {
                    let bv = self.check_expr_infallible(&a.value);
                    if bv.ty != Ty::Priority && !bv.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            a.span,
                            format!(
                                "'boost:' must be a priority (@critical..@low), found '{}'",
                                bv.ty
                            ),
                        ));
                    }
                    if let ExprKind::Priority(prio) = a.value.kind {
                        boost_prio = Some((prio, a.span));
                    }
                } else {
                    let _ = self.check_expr_infallible(&a.value);
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        arg_name.span,
                        format!(
                            "unknown named argument '{}:' in 'fmt_ring'",
                            arg_name.name
                        ),
                    ));
                }
            } else {
                pos_args.push(a);
            }
        }

        if pos_args.len() != 2 {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "'fmt_ring' expects 2 positional arguments (tx, f\"...\" [, boost: @prio]), found {}",
                    pos_args.len()
                ),
            ));
            for a in &pos_args {
                if let ExprKind::FormatString(parts) = &a.value.kind {
                    let _ = self.check_format_string(parts);
                } else {
                    let _ = self.check_expr_infallible(&a.value);
                }
            }
            return ExprVal {
                ty: Ty::usize(),
                error_spec: fmt_err_spec,
                prov: Provenance::pure_static(),
                endpoint_name: None,
                field_endpoints: BTreeMap::new(),
                span,
            };
        }

        let ring_arg = pos_args[0];
        let fmt_arg = pos_args[1];
        let rv = self.check_expr_infallible(&ring_arg.value);
        let (is_byte_ring, ring_cap) = match &rv.ty {
            Ty::Ptr(inner) => match &**inner {
                Ty::OsPrimitive(ResolvedOsPrimitive::ByteRing { cap }) => (true, Some(*cap)),
                Ty::OsPrimitive(ResolvedOsPrimitive::ByteRingUnsized) => (true, None),
                Ty::Error => (false, None),
                _ => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        ring_arg.span,
                        format!(
                            "'fmt_ring' first argument must be '*byte_ring<N>' or '*byte_ring', found '{}'",
                            rv.ty
                        ),
                    ));
                    (false, None)
                }
            },
            Ty::Error => (false, None),
            _ => {
                let mut diag = Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    ring_arg.span,
                    format!(
                        "'fmt_ring' first argument must be '*byte_ring<N>' or '*byte_ring', found '{}'",
                        rv.ty
                    ),
                );
                if matches!(
                    rv.ty,
                    Ty::OsPrimitive(ResolvedOsPrimitive::ByteRing { .. })
                ) {
                    diag = diag.with_note("pass `&ring` or a `*byte_ring` view pointer");
                }
                self.sema.emit(diag);
                (false, None)
            }
        };

        if is_byte_ring {
            let source_key = rv
                .endpoint_name
                .clone()
                .or_else(|| root_ident_of(&ring_arg.value))
                .unwrap_or_else(|| "<anon_endpoint>".to_string());
            if let Some((prio, boost_span)) = boost_prio {
                self.recorded_boosts.push(PriorityBoostSite {
                    primitive_name: source_key.clone(),
                    boost_priority: prio,
                    span: boost_span,
                });
            }
            self.record_byte_ring_op(ByteRingOpRole::Producer, "fmt_ring", &source_key, span);
            let wkey = format!("{source_key}#write");
            let _ = self.leases.invalidate(&wkey, span, "fmt_ring");
        }

        let min_len = if let ExprKind::FormatString(parts) = &fmt_arg.value.kind {
            let _ = self.check_format_string(parts);
            self.min_format_string_len(parts)
        } else {
            let fv = self.check_expr_infallible(&fmt_arg.value);
            if !fv.ty.is_error() {
                let _ = self.check_format_hole(
                    &fmt_arg.value,
                    &fv.ty,
                    FormatSpec::Default,
                    fmt_arg.span,
                );
            }
            self.min_format_value_len(&fv.ty, &FormatSpec::Default, &fmt_arg.value)
        };

        if let Some(cap) = ring_cap {
            if min_len > cap {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::FormatBufferTooSmall,
                        span,
                        format!(
                            "format output minimum length ({} bytes) exceeds byte_ring<{}> capacity ({} bytes)",
                            min_len, cap, cap
                        ),
                    )
                    .with_note(format!(
                        "increase the byte_ring capacity to at least `byte_ring<{}>`, or use an unsized `*byte_ring` view if capacity varies",
                        min_len
                    )),
                );
            }
        }

        ExprVal {
            ty: Ty::usize(),
            error_spec: fmt_err_spec,
            prov: Provenance::pure_static(),
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }

    fn min_format_string_len(&self, parts: &[FormatPart]) -> usize {
        let mut total = 0usize;
        for part in parts {
            let step = match part {
                FormatPart::Literal(s) => zev_ast::decoded_string_byte_len(s),
                FormatPart::Hole { expr, spec, .. } => {
                    let Some(ty) = self.structural_expr_ty(expr).or_else(|| match &expr.kind {
                        ExprKind::Literal(Literal::String(_)) => Some(Ty::Str),
                        ExprKind::Literal(Literal::Int(_)) => Some(Ty::IntLit),
                        ExprKind::Literal(Literal::Float { .. }) => Some(Ty::FloatLit),
                        ExprKind::Literal(Literal::Bool(_)) => Some(Ty::bool()),
                        _ => None,
                    }) else {
                        continue;
                    };
                    self.min_format_value_len(&ty, spec, expr)
                }
            };
            total = total.saturating_add(step);
        }
        total
    }

    fn min_format_value_len(&self, ty: &Ty, spec: &FormatSpec, expr: &Expr) -> usize {
        if let ExprKind::Literal(Literal::String(s)) = &expr.kind {
            return zev_ast::decoded_string_byte_len(s);
        }
        if ty.is_bool() {
            return if let ExprKind::Literal(Literal::Bool(b)) = &expr.kind {
                if *b { 4 } else { 5 }
            } else {
                4
            };
        }
        if ty.is_integer() {
            return match spec {
                FormatSpec::Hex { width, .. } | FormatSpec::Bin { width } => {
                    width.unwrap_or(1).max(1) as usize
                }
                _ => match &expr.kind {
                    ExprKind::Literal(Literal::Int(n)) => {
                        if *n == 0 {
                            1
                        } else {
                            n.ilog10() as usize + 1
                        }
                    }
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        expr: inner,
                    } => {
                        if let ExprKind::Literal(Literal::Int(n)) = &inner.kind {
                            let d = if *n == 0 { 1 } else { n.ilog10() as usize + 1 };
                            d + 1
                        } else {
                            1
                        }
                    }
                    _ => 1,
                },
            };
        }
        if ty.is_float() {
            return match spec {
                FormatSpec::Fixed { precision: p } => {
                    if *p == 0 {
                        1
                    } else {
                        2 + (*p as usize)
                    }
                }
                FormatSpec::Sci { .. } => 3,
                _ => 1,
            };
        }
        if let Ty::Enum(en_name) = ty {
            return self
                .sema
                .enums
                .get(en_name)
                .and_then(|en| en.variants.iter().map(|v| v.len()).min())
                .unwrap_or(1);
        }
        match (ty, spec) {
            (Ty::Array { elem, len }, FormatSpec::Default) if **elem == Ty::u8() => *len,
            (Ty::Struct(_) | Ty::Slice(_) | Ty::Array { .. }, FormatSpec::Debug) => {
                let mut visiting = BTreeSet::new();
                self.debug_min_case_ty(ty, &mut visiting)
            }
            (Ty::Ptr(_), FormatSpec::Debug) => 3,
            _ => 0,
        }
    }

    fn debug_min_case_ty(&self, ty: &Ty, visiting: &mut BTreeSet<String>) -> usize {
        match ty {
            Ty::Prim(PrimType::Bool) => 4,
            Ty::Prim(p) if p.is_integer() => 1,
            Ty::Prim(p) if p.is_float() => 1,
            Ty::IntLit | Ty::FloatLit => 1,
            Ty::Enum(en_name) => self
                .sema
                .enums
                .get(en_name)
                .and_then(|en| en.variants.iter().map(|v| v.len()).min())
                .unwrap_or(1),
            Ty::Struct(st_name) => {
                if !visiting.insert(st_name.clone()) {
                    return 0;
                }
                let Some(st_info) = self.sema.structs.get(st_name) else {
                    visiting.remove(st_name);
                    return 0;
                };
                if st_info.fields.is_empty() {
                    visiting.remove(st_name);
                    return st_name.len() + 4;
                }
                let mut acc = st_name.len() + 3;
                for (i, field) in st_info.fields.iter().enumerate() {
                    if i > 0 {
                        acc = acc.saturating_add(2);
                    }
                    acc = acc.saturating_add(field.name.len() + 2);
                    acc = acc.saturating_add(self.debug_min_case_ty(&field.ty, visiting));
                }
                visiting.remove(st_name);
                acc.saturating_add(2)
            }
            Ty::Array { elem, len } => {
                if *len == 0 {
                    2
                } else {
                    2usize
                        .saturating_add(2 * (*len - 1))
                        .saturating_add(len.saturating_mul(self.debug_min_case_ty(elem, visiting)))
                }
            }
            Ty::Slice(_) => 2,
            Ty::Ptr(_) => 3,
            _ => 0,
        }
    }

    fn static_dst_capacity(&self, dst_expr: &Expr, dst_ty: &Ty) -> Option<usize> {
        if matches!(&dst_expr.kind, ExprKind::Slice { .. }) {
            return self.structural_u8_capacity(dst_expr);
        }

        match dst_ty {
            Ty::Array { elem, len } if **elem == Ty::u8() => Some(*len),
            Ty::Ptr(inner) => match &**inner {
                Ty::Array { elem, len } if **elem == Ty::u8() => Some(*len),
                _ => None,
            },
            _ => None,
        }
    }

    fn structural_u8_capacity(&self, expr: &Expr) -> Option<usize> {
        if let ExprKind::Slice { target, start, end } = &expr.kind {
            let base_cap = self.structural_u8_capacity(target)?;
            let s_raw = match start {
                None => 0usize,
                Some(s) => self.eval_const_usize_expr(s)?,
            };
            let e_raw = match end {
                None => base_cap,
                Some(e) => self.eval_const_usize_expr(e)?,
            };
            let e = e_raw.min(base_cap);
            let s = s_raw.min(e);
            return Some(e - s);
        }

        match self.structural_expr_ty(expr)? {
            Ty::Array { elem, len } if *elem == Ty::u8() => Some(len),
            Ty::Ptr(inner) => match *inner {
                Ty::Array { elem, len } if *elem == Ty::u8() => Some(len),
                _ => None,
            },
            _ => None,
        }
    }

    fn structural_expr_ty(&self, expr: &Expr) -> Option<Ty> {
        match &expr.kind {
            ExprKind::Ident(id) => {
                if let Some(var) = self.lookup_local(&id.name) {
                    return Some(var.ty.clone());
                }
                if let Some(c) = self.sema.consts.get(&id.name) {
                    return Some(c.ty.clone());
                }
                None
            }
            ExprKind::Field { target, field } => {
                let base_ty = match self.structural_expr_ty(target)? {
                    Ty::Ptr(inner) => *inner,
                    other => other,
                };
                if let Ty::Struct(st_name) = base_ty {
                    return self
                        .sema
                        .structs
                        .get(&st_name)
                        .and_then(|st| st.field(&field.name))
                        .map(|f| f.ty.clone());
                }
                None
            }
            ExprKind::Index { target, .. } => match self.structural_expr_ty(target)? {
                Ty::Array { elem, .. } | Ty::Slice(elem) => Some(*elem),
                Ty::Ptr(inner) => match *inner {
                    Ty::Array { elem, .. } | Ty::Slice(elem) => Some(*elem),
                    _ => None,
                },
                _ => None,
            },
            ExprKind::Slice { target, .. } => match self.structural_expr_ty(target)? {
                Ty::Array { elem, .. } | Ty::Slice(elem) => Some(Ty::Slice(elem)),
                Ty::Ptr(inner) => match *inner {
                    Ty::Array { elem, .. } | Ty::Slice(elem) => Some(Ty::Slice(elem)),
                    _ => None,
                },
                _ => None,
            },
            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: inner,
            } => match self.structural_expr_ty(inner)? {
                Ty::Ptr(pointee) => Some(*pointee),
                _ => None,
            },
            ExprKind::Unary {
                op: UnaryOp::AddrOf,
                expr: inner,
            } => Some(Ty::Ptr(Box::new(self.structural_expr_ty(inner)?))),
            _ => None,
        }
    }

    fn eval_const_usize_expr(&self, expr: &Expr) -> Option<usize> {
        let shadowed = self.active_local_names();
        self.sema
            .eval_const_expr_i128_shadowed(expr, &mut BTreeSet::new(), &shadowed)
            .and_then(|v| usize::try_from(v).ok())
    }

    /// Typecheck all holes in an `f"..."` format string and compute its exact worst-case byte width
    /// (`Some(bytes)` when every hole is statically bounded, or `None` when any hole is a dynamic slice/string).
    fn check_format_string(&mut self, parts: &[FormatPart]) -> Option<usize> {
        self.check_format_string_with_dst(parts, None)
    }

    fn check_format_string_with_dst(
        &mut self,
        parts: &[FormatPart],
        alias_dst: Option<&Provenance>,
    ) -> Option<usize> {
        let mut worst_total: Option<usize> = Some(0);
        for part in parts {
            match part {
                FormatPart::Literal(s) => {
                    let byte_len = zev_ast::decoded_string_byte_len(s);
                    worst_total = worst_total.and_then(|acc| acc.checked_add(byte_len));
                }
                FormatPart::Hole {
                    expr,
                    spec,
                    span: hole_span,
                } => {
                    let hv = self.check_expr_infallible(expr);
                    if let Some(dst_prov) = alias_dst {
                        if !hv.ty.is_error()
                            && (hv.ty.contains_reference(&self.sema.structs)
                                || matches!(hv.ty, Ty::Array { .. }))
                        {
                            let mut hole_prov = hv.prov.clone();
                            if let Some(root) = root_ident_of(expr) {
                                if let Some(var) = self.lookup_local(&root) {
                                    if var.ty.contains_reference(&self.sema.structs)
                                        || matches!(var.ty, Ty::Array { .. })
                                    {
                                        hole_prov = hole_prov.merge(&Provenance::addr_of_local(
                                            &var.name,
                                            var.decl_depth,
                                        ));
                                    }
                                }
                            }
                            if !dst_prov.local_origins.is_disjoint(&hole_prov.local_origins)
                                || !dst_prov.param_origins.is_disjoint(&hole_prov.param_origins)
                            {
                                self.sema.emit(
                                    Diagnostic::new(
                                        DiagnosticKind::InvalidOperation,
                                        *hole_span,
                                        "cannot format slice/buffer into the same destination buffer in 'fmt_buf' (in-place aliasing would corrupt output)",
                                    )
                                    .with_note(
                                        "format into a separate `[N]u8` scratch buffer first, then copy or use the resulting slice",
                                    ),
                                );
                            }
                        }
                    }
                    let hw = self.check_format_hole(expr, &hv.ty, *spec, *hole_span);
                    worst_total = match (worst_total, hw) {
                        (Some(acc), Some(w)) => acc.checked_add(w),
                        _ => None,
                    };
                }
            }
        }
        worst_total
    }

    fn check_format_hole(
        &mut self,
        expr: &Expr,
        ty: &Ty,
        spec: FormatSpec,
        span: Span,
    ) -> Option<usize> {
        if ty.is_error() {
            return None;
        }

        // 1. String literal hole (`"ok"`).
        if let ExprKind::Literal(Literal::String(s)) = &expr.kind {
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => {
                    Some(zev_ast::decoded_string_byte_len(s))
                }
                _ => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        "numeric format specifier cannot be applied to a string literal",
                    ));
                    None
                }
            };
        }

        // 2. Boolean (`true` / `false` -> max 5 bytes).
        if ty.is_bool() {
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => Some(5),
                _ => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        "numeric format specifier cannot be applied to 'bool'",
                    ));
                    None
                }
            };
        }

        // 3. Integers (`u8`..`u64`, `i8`..`i64`, `usize`, `isize`, endian ints, `IntLit`).
        if ty.is_integer() {
            let int_prim = match ty {
                Ty::Prim(p) => *p,
                Ty::IntLit => match &expr.kind {
                    ExprKind::Literal(Literal::Int(v)) if *v > i32::MAX as u64 => PrimType::U64,
                    _ => PrimType::I32,
                },
                _ => PrimType::I32,
            };
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => Some(int_decimal_worst_case(int_prim)),
                FormatSpec::Hex { width, .. } => {
                    let w = width.unwrap_or(0) as usize;
                    Some(int_hex_worst_case(int_prim).max(w))
                }
                FormatSpec::Bin { width } => {
                    let w = width.unwrap_or(0) as usize;
                    Some(int_bin_worst_case(int_prim).max(w))
                }
                FormatSpec::Fixed { .. } | FormatSpec::Sci { .. } => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "floating-point format specifier requires 'f32' or 'f64', found '{}'",
                            ty
                        ),
                    ));
                    None
                }
            };
        }

        // 4. Floating-point (`f32`, `f64`, `FloatLit`).
        if ty.is_float() {
            let is_f32 = matches!(ty, Ty::Prim(PrimType::F32));
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => Some(14),
                FormatSpec::Fixed { precision: p } => {
                    if p > 9 {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "unsupported float format precision '.{}'; supported float precision is 0..=9",
                                p
                            ),
                        ));
                        None
                    } else {
                        Some(if p == 0 { 11 } else { 12 + p as usize })
                    }
                }
                FormatSpec::Sci { precision } => {
                    let p_u8 = precision.unwrap_or(2);
                    if p_u8 > 9 {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "unsupported scientific float format precision '.{}e'; supported float precision is 0..=9",
                                p_u8
                            ),
                        ));
                        None
                    } else {
                        let p = p_u8 as usize;
                        Some(if is_f32 { 9 + p } else { 10 + p })
                    }
                }
                FormatSpec::Hex { .. } | FormatSpec::Bin { .. } => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "hex/binary format specifier requires an integer type, found '{}'",
                            ty
                        ),
                    ));
                    None
                }
            };
        }

        // 5. Enum variant names (`{err}` or `{err:?}`), bounded by `<EnumName>(?)` fallback.
        if let Ty::Enum(enum_name) = ty {
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => {
                    self.sema.enums.get(enum_name).map(|en| {
                        let variants_max = en.variants.iter().map(|v| v.len()).max().unwrap_or(0);
                        variants_max.max(enum_name.len() + 3)
                    })
                }
                _ => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "numeric format specifier cannot be applied to enum '{}'",
                            enum_name
                        ),
                    ));
                    None
                }
            };
        }

        // 6. Byte strings/slices, structs under `:?`, and uniform pointer addresses under `:?`.
        match (ty, spec) {
            (Ty::Str, FormatSpec::Default | FormatSpec::Debug) => None,
            (Ty::Slice(elem), FormatSpec::Default) if **elem == Ty::u8() => None,
            (Ty::Ptr(inner), FormatSpec::Default)
                if matches!(**inner, Ty::Prim(PrimType::U8 | PrimType::I8)) =>
            {
                None
            }
            (Ty::Array { elem, len }, FormatSpec::Default) if **elem == Ty::u8() => Some(*len),

            (Ty::Struct(st_name), FormatSpec::Debug) => {
                let mut visiting = BTreeSet::new();
                self.debug_worst_case_ty(&Ty::Struct(st_name.clone()), span, &mut visiting)
            }
            (Ty::Ptr(_), FormatSpec::Debug) => Some(18),
            (Ty::Struct(st_name), FormatSpec::Default) => {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "struct '{}' requires debug specifier '{{expr:?}}' or explicit field formatting",
                            st_name
                        ),
                    )
                    .with_note(format!(
                        "use `{{val:?}}` for compile-time struct debug expansion or format fields explicitly (`f\"{{val.field}}\"`)"
                    )),
                );
                None
            }
            (Ty::Slice(_) | Ty::Array { .. }, FormatSpec::Debug) => {
                let mut visiting = BTreeSet::new();
                self.debug_worst_case_ty(ty, span, &mut visiting)
            }
            _ => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    span,
                    format!(
                        "type '{}' cannot be formatted with specifier '{:?}' in 'f\"...\"'",
                        ty, spec
                    ),
                ));
                None
            }
        }
    }

    fn debug_worst_case_ty(
        &mut self,
        ty: &Ty,
        span: Span,
        visiting: &mut BTreeSet<String>,
    ) -> Option<usize> {
        match ty {
            Ty::Error => None,
            Ty::Prim(PrimType::Bool) => Some(5),
            Ty::Prim(p) if p.is_integer() => Some(int_decimal_worst_case(*p)),
            Ty::Prim(p) if p.is_float() => Some(14),
            Ty::IntLit => Some(11),
            Ty::FloatLit => Some(14),
            Ty::Str => None,
            Ty::Enum(en_name) => self.sema.enums.get(en_name).map(|en| {
                let variants_max = en.variants.iter().map(|v| v.len()).max().unwrap_or(0);
                variants_max.max(en_name.len() + 3)
            }),
            Ty::Struct(st_name) => {
                if !visiting.insert(st_name.clone()) {
                    return None;
                }
                let Some(st_info) = self.sema.structs.get(st_name).cloned() else {
                    visiting.remove(st_name);
                    return None;
                };
                if st_info.fields.is_empty() {
                    visiting.remove(st_name);
                    // `"Name { }"` (matches `zev-mir` `expand_struct_debug_be`)
                    return Some(st_name.len() + 4);
                }
                // `"Name { f1: <v1>, f2: <v2> }"`
                let mut acc: Option<usize> = Some(st_name.len() + 3); // `"Name { "`
                for (i, field) in st_info.fields.iter().enumerate() {
                    if i > 0 {
                        acc = acc.and_then(|a| a.checked_add(2)); // `", "`
                    }
                    acc = acc.and_then(|a| a.checked_add(field.name.len() + 2)); // `"field: "`
                    let fw = self.debug_worst_case_ty(&field.ty, span, visiting);
                    acc = match (acc, fw) {
                        (Some(a), Some(w)) => a.checked_add(w),
                        _ => None,
                    };
                }
                visiting.remove(st_name);
                acc.and_then(|a| a.checked_add(2)) // `" }"`
            }
            Ty::Slice(elem) => {
                if matches!(&**elem, Ty::Struct(_) | Ty::Slice(_) | Ty::Array { .. }) {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "dynamic slice '{}' of composite elements cannot be formatted with ':?'",
                                ty
                            ),
                        )
                        .with_note(
                            "iterate over the slice explicitly (`for (Elem x in slice) { println(f\"{x:?}\"); }`) or format a fixed-size `[N]T` array",
                        ),
                    );
                    return None;
                }
                let _ = self.debug_worst_case_ty(elem, span, visiting);
                None
            }
            Ty::Array { elem, len } => {
                let ew = self.debug_worst_case_ty(elem, span, visiting)?;
                let items = ew.checked_mul(*len)?;
                let seps = len.saturating_sub(1).checked_mul(2)?;
                items.checked_add(seps)?.checked_add(2)
            }
            Ty::Ptr(_) => {
                // Pointer addresses formatted uniformly as hex (`0x` + up to 16 hex digits).
                Some(18)
            }
            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    span,
                    format!("field/element of type '{}' cannot be formatted with ':?'", other),
                ));
                None
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small Pure Helpers
// ─────────────────────────────────────────────────────────────────────────────

const VALID_SYS_METHODS_NOTE: &str = "valid portable `sys.*` methods: `sys.use_systick(u32 hz)`, `sys.tick()`, `sys.tick_ms()`, `sys.tick_by(u32 ms)`, `sys.set_putc(fn(u8))`, `sys.set_clock_ms(fn() -> u32)`, `sys.set_exit(fn(i32))`, `sys.set_fault_handler(fn())`, `sys.set_idle(fn(u32, u32))`, `sys.set_irq_router(fn(u32, u32), fn(u32))`, `sys.rom_call_u32(usize fn_addr, usize stack_bound, ...) -> u32`, `sys.rom_call_void(usize fn_addr, usize stack_bound, ...)`, `sys.rom_call_noreturn(usize fn_addr, ...)`, `sys.now_ms() -> u32`, `sys.now_ticks() -> u32`, `sys.current_prio() -> u32`, `sys.sleep_depth() -> u32`, `sys.wfi_count() -> u32`, `sys.trace_sleep(bool on)`, `sys.pend_irq(u32 irq)`, `sys.nop()`, `sys.exit(i32 code)`, `sys.log(*u8 msg)`, `sys.log_kv(*u8 label, i32 val)`, `sys.log_hex(*u8 label, u32 val)`, `sys.delay(dur)`";

fn is_string_arg_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::Str)
        || matches!(ty, Ty::Ptr(inner) if matches!(**inner, Ty::Prim(PrimType::U8 | PrimType::I8)))
}

fn is_print_arg_ty(ty: &Ty) -> bool {
    is_string_arg_ty(ty)
        || matches!(ty, Ty::Slice(inner) | Ty::Array { elem: inner, .. } if **inner == Ty::u8())
}

fn has_attr_dma_safe(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| matches!(a.kind, AttrKind::DmaSafe))
}

fn has_attr_manual_yield(attrs: &[Attr]) -> bool {
    attrs
        .iter()
        .any(|a| matches!(a.kind, AttrKind::ManualYield))
}

fn has_attr_noyield(attrs: &[Attr]) -> bool {
    attrs
        .iter()
        .any(|a| matches!(&a.kind, AttrKind::Custom { name, .. } if name == "noyield"))
}

fn root_ident_of(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Ident(id) => Some(id.name.clone()),
        ExprKind::Field { target, .. }
        | ExprKind::Index { target, .. }
        | ExprKind::Slice { target, .. } => root_ident_of(target),
        ExprKind::Unary {
            op: UnaryOp::Deref | UnaryOp::AddrOf,
            expr: inner,
        } => root_ident_of(inner),
        _ => None,
    }
}

fn extract_monomorphized_attr(attrs: &[Attr]) -> Option<String> {
    for a in attrs {
        if let AttrKind::Custom {
            name,
            arg: Some(sig),
        } = &a.kind
        {
            if name == "monomorphized" {
                return Some(sig.clone());
            }
        }
    }
    None
}

fn extract_instantiated_from_attr(attrs: &[Attr]) -> Option<String> {
    for a in attrs {
        if let AttrKind::Custom {
            name,
            arg: Some(note),
        } = &a.kind
        {
            if name == "instantiated_from" {
                return Some(note.clone());
            }
        }
    }
    None
}

fn ty_contains_os_prim_view(
    ty: &Ty,
    structs: &BTreeMap<String, StructInfo>,
    visiting: &mut BTreeSet<String>,
) -> bool {
    match ty {
        Ty::OsPrimitive(_) => true,
        Ty::Ptr(inner) | Ty::Slice(inner) | Ty::Array { elem: inner, .. } | Ty::Lease(inner) => {
            ty_contains_os_prim_view(inner, structs, visiting)
        }
        Ty::Struct(name) => {
            if !visiting.insert(name.clone()) {
                return false;
            }
            let has = structs
                .get(name)
                .is_some_and(|st| {
                    st.fields
                        .iter()
                        .any(|f| ty_contains_os_prim_view(&f.ty, structs, visiting))
                });
            visiting.remove(name);
            has
        }
        _ => false,
    }
}

fn is_valid_cast(src: &Ty, dst: &Ty) -> bool {
    if ty_contains_linear_lease(src) || ty_contains_linear_lease(dst) {
        return false;
    }
    if src.contains_mmio() || dst.contains_mmio() {
        return match (src, dst) {
            (Ty::Ptr(a), Ty::Ptr(b)) => {
                matches!((&**a, &**b), (Ty::MmioLayout(la), Ty::MmioLayout(lb)) if la == lb)
            }
            (Ty::MmioInstance { layout: la, .. }, Ty::Ptr(b)) => {
                matches!(&**b, Ty::MmioLayout(lb) if la == lb)
            }
            (Ty::MmioInstance { .. }, b) => b.is_integer(),
            _ => false,
        };
    }
    if src.contains_os_primitive() || dst.contains_os_primitive() {
        return match (src, dst) {
            (Ty::Ptr(a), Ty::Ptr(b)) => match (&**a, &**b) {
                (Ty::OsPrimitive(sa), Ty::OsPrimitive(sb)) => {
                    sa == sb
                        || matches!(
                            (sa, sb),
                            (
                                ResolvedOsPrimitive::ByteRing { .. },
                                ResolvedOsPrimitive::ByteRingUnsized
                            )
                        )
                        || matches!(
                            (sa, sb),
                            (
                                ResolvedOsPrimitive::Inbox { elem: ea, .. },
                                ResolvedOsPrimitive::InboxUnsized { elem: eb }
                            ) if ea == eb
                        )
                }
                _ => false,
            },
            _ => false,
        };
    }
    if src == dst {
        return true;
    }
    match (src, dst) {
        // Any numeric <-> numeric (`int <-> int`, `int <-> float`, `f32 <-> f64`).
        (a, b) if a.is_numeric() && b.is_numeric() => true,
        // `bool` -> integer or float.
        (Ty::Prim(PrimType::Bool), b) if b.is_numeric() => true,
        // `enum` -> integer.
        (Ty::Enum(_), b) if b.is_integer() => true,
        // Pointer <-> pointer or pointer <-> integer.
        (Ty::Ptr(_), Ty::Ptr(_)) => true,
        (Ty::Ptr(_), b) if b.is_integer() => true,
        (a, Ty::Ptr(_)) if a.is_integer() => true,
        (Ty::IrqToken, b) if b.is_integer() => true,
        (Ty::Str, Ty::Ptr(inner))
            if inner.is_void() || matches!(**inner, Ty::Prim(PrimType::U8 | PrimType::I8)) =>
        {
            true
        }
        (Ty::FnPtr { .. }, Ty::Ptr(inner)) if inner.is_void() => true,
        _ => false,
    }
}

fn unify_float_arith(lhs: &Ty, rhs: &Ty) -> Option<Ty> {
    match (lhs, rhs) {
        (Ty::FloatLit, Ty::FloatLit) => Some(Ty::FloatLit),
        (Ty::FloatLit, Ty::Prim(PrimType::F32)) | (Ty::Prim(PrimType::F32), Ty::FloatLit) => {
            Some(Ty::f32())
        }
        (Ty::FloatLit, Ty::Prim(PrimType::F64)) | (Ty::Prim(PrimType::F64), Ty::FloatLit) => {
            Some(Ty::f64())
        }
        (Ty::Prim(PrimType::F32), Ty::Prim(PrimType::F32)) => Some(Ty::f32()),
        (Ty::Prim(PrimType::F64), Ty::Prim(PrimType::F64)) => Some(Ty::f64()),
        _ => None,
    }
}

const fn int_decimal_worst_case(p: PrimType) -> usize {
    match p {
        PrimType::U8 => 3,
        PrimType::I8 => 4,
        PrimType::U16 | PrimType::U16Be | PrimType::U16Le => 5,
        PrimType::I16 => 6,
        PrimType::U32 | PrimType::U32Be | PrimType::U32Le => 10,
        PrimType::I32 => 11,
        PrimType::U64
        | PrimType::U64Be
        | PrimType::U64Le
        | PrimType::Usize
        | PrimType::I64
        | PrimType::Isize => 20,
        _ => 11,
    }
}

const fn int_hex_worst_case(p: PrimType) -> usize {
    match p {
        PrimType::U8 | PrimType::I8 => 2,
        PrimType::U16 | PrimType::U16Be | PrimType::U16Le | PrimType::I16 => 4,
        PrimType::U32 | PrimType::U32Be | PrimType::U32Le | PrimType::I32 => 8,
        _ => 16,
    }
}

const fn int_bin_worst_case(p: PrimType) -> usize {
    match p {
        PrimType::U8 | PrimType::I8 => 8,
        PrimType::U16 | PrimType::U16Be | PrimType::U16Le | PrimType::I16 => 16,
        PrimType::U32 | PrimType::U32Be | PrimType::U32Le | PrimType::I32 => 32,
        _ => 64,
    }
}

fn unify_integer_arith(lhs: &Ty, rhs: &Ty) -> Ty {
    match (lhs, rhs) {
        (Ty::IntLit, Ty::IntLit) => Ty::IntLit,
        (Ty::IntLit, Ty::Prim(p)) | (Ty::Prim(p), Ty::IntLit) => Ty::Prim(Ty::host_prim(*p)),
        (Ty::Prim(a), Ty::Prim(b)) => {
            let ha = Ty::host_prim(*a);
            let hb = Ty::host_prim(*b);
            if ha == hb {
                Ty::Prim(ha)
            } else if ha.byte_size(4) >= hb.byte_size(4) {
                Ty::Prim(ha)
            } else {
                Ty::Prim(hb)
            }
        }
        _ => Ty::u32(),
    }
}

const fn iter_span_none() -> Option<Span> {
    None
}

fn is_sys_exit_expr(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::MethodCall {
            receiver, method, ..
        } => {
            matches!(&receiver.kind, ExprKind::Ident(id) if id.name == "sys")
                && matches!(method.name.as_str(), "exit" | "rom_call_noreturn")
        }
        ExprKind::Call { callee, .. } => match &callee.kind {
            ExprKind::Field { target, field } => {
                matches!(&target.kind, ExprKind::Ident(id) if id.name == "sys")
                    && matches!(field.name.as_str(), "exit" | "rom_call_noreturn")
            }
            _ => false,
        },
        _ => false,
    }
}

/// Returns `true` if `block` contains a `break` targeting the enclosing loop (not nested inside an inner loop).
fn block_has_break(block: &Block) -> bool {
    stmts_have_break(&block.stmts)
}

fn stmts_have_break(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Break => return true,
            StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                if block_has_break(then_block)
                    || else_block.as_ref().is_some_and(block_has_break)
                {
                    return true;
                }
            }
            StmtKind::Switch { arms, .. } => {
                if arms.iter().any(|a| block_has_break(&a.body)) {
                    return true;
                }
            }
            StmtKind::NoYield { body } | StmtKind::Critical { body } | StmtKind::Block(body) => {
                if block_has_break(body) {
                    return true;
                }
            }
            StmtKind::Expr(expr) | StmtKind::VarDecl { init: Some(expr), .. } => {
                if expr_has_break(expr) {
                    return true;
                }
            }
            // Inner loops consume their own `break`.
            StmtKind::Loop { .. }
            | StmtKind::While { .. }
            | StmtKind::LoopEvery { .. }
            | StmtKind::ForRange { .. }
            | StmtKind::ForEach { .. } => {}
            _ => {}
        }
    }
    false
}

fn expr_has_break(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Catch { expr: inner, clause } => {
            expr_has_break(inner) || block_has_break(&clause.body)
        }
        ExprKind::Or { expr: a, fallback: b } => expr_has_break(a) || expr_has_break(b),
        _ => false,
    }
}

fn is_mmio_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::MmioLayout(_) | Ty::MmioInstance { .. })
        || matches!(ty, Ty::Ptr(inner) if matches!(&**inner, Ty::MmioLayout(_)))
}

fn ty_contains_linear_lease(ty: &Ty) -> bool {
    match ty {
        Ty::Lease(_) => true,
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Ptr(elem) => ty_contains_linear_lease(elem),
        Ty::OsPrimitive(os) => os.elem_ty().is_some_and(ty_contains_linear_lease),
        Ty::FnPtr { params, ret } => {
            params.iter().any(ty_contains_linear_lease) || ty_contains_linear_lease(ret)
        }
        _ => false,
    }
}

fn ty_contains_byval_mmio(ty: &Ty) -> Option<&str> {
    match ty {
        Ty::MmioLayout(name) => Some(name.as_str()),
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Lease(elem) => ty_contains_byval_mmio(elem),
        Ty::OsPrimitive(os) => os.elem_ty().and_then(ty_contains_byval_mmio),
        Ty::FnPtr { params, ret } => params
            .iter()
            .find_map(ty_contains_byval_mmio)
            .or_else(|| ty_contains_byval_mmio(ret)),
        _ => None,
    }
}

fn ty_contains_bare_unsized_prim(ty: &Ty) -> Option<String> {
    match ty {
        Ty::OsPrimitive(os) if os.is_unsized_view() => Some(os.to_string()),
        Ty::OsPrimitive(os) => os.elem_ty().and_then(ty_contains_bare_unsized_prim),
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Lease(elem) => {
            ty_contains_bare_unsized_prim(elem)
        }
        Ty::FnPtr { params, ret } => params
            .iter()
            .find_map(ty_contains_bare_unsized_prim)
            .or_else(|| ty_contains_bare_unsized_prim(ret)),
        _ => None,
    }
}

fn ty_embeds_os_primitive(ty: &Ty) -> bool {
    match ty {
        Ty::OsPrimitive(os) => !os.is_unsized_view(),
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Lease(elem) => ty_embeds_os_primitive(elem),
        _ => false,
    }
}

fn ty_contains_byval_subsys<'a>(
    ty: &'a Ty,
    structs: &'a BTreeMap<String, StructInfo>,
) -> Option<&'a str> {
    match ty {
        Ty::Struct(name) if structs.get(name).is_some_and(|s| s.is_subsystem) => {
            Some(name.as_str())
        }
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Lease(elem) => {
            ty_contains_byval_subsys(elem, structs)
        }
        Ty::OsPrimitive(os) => os
            .elem_ty()
            .and_then(|e| ty_contains_byval_subsys(e, structs)),
        Ty::FnPtr { params, ret } => params
            .iter()
            .find_map(|p| ty_contains_byval_subsys(p, structs))
            .or_else(|| ty_contains_byval_subsys(ret, structs)),
        _ => None,
    }
}

impl Analyzer {
    fn validate_pool_slot_elem(&mut self, elem: &Ty, span: Span, owner_desc: &str) {
        if elem.is_error() || elem.is_void() {
            return;
        }
        if ty_contains_linear_lease(elem) {
            self.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!("{} cannot wrap linear 'Lease<...>' inside its payload element type '{}'", owner_desc, elem),
            ));
            return;
        }
        if let Some(mmio) = ty_contains_byval_mmio(elem) {
            self.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!("{} cannot store MMIO register layout '{}' by value", owner_desc, mmio),
            ));
            return;
        }
        if ty_embeds_os_primitive(elem) || ty_contains_byval_subsys(elem, &self.structs).is_some() {
            self.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!("{} cannot embed OS primitives or subsystem structs inside element type '{}'", owner_desc, elem),
            ));
            return;
        }
        if elem.contains_reference(&self.structs) || elem.contains_fn_ptr(&self.structs) {
            self.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!("{} requires a plain non-reference value type (no '*T', '[]T', or 'fn(...)' fields in '{}')", owner_desc, elem),
            ));
        }
    }

    fn check_disallowed_lease_type(
        &mut self,
        ty: &Ty,
        span: Span,
        ctx: &str,
        allow_top_level_lease: bool,
    ) {
        match ty {
            Ty::Lease(elem) => {
                if !allow_top_level_lease {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        span,
                        format!(
                            "linear type '{}' is not allowed in {} (only permitted as locals, `fn` params/returns, `inbox<Lease<T>, N>`, `*inbox<Lease<T>>`, and managed via `dma_slot<T>`)",
                            ty, ctx
                        ),
                    ));
                } else {
                    self.validate_pool_slot_elem(elem, span, "Lease<T>");
                }
            }
            Ty::OsPrimitive(
                ResolvedOsPrimitive::Inbox { elem, .. }
                | ResolvedOsPrimitive::InboxUnsized { elem },
            ) => {
                if let Ty::Lease(inner) = &**elem {
                    self.validate_pool_slot_elem(inner, span, "Lease<T>");
                } else {
                    self.check_disallowed_lease_type(elem, span, ctx, false);
                }
            }
            Ty::OsPrimitive(os) => {
                if let Some(elem) = os.elem_ty() {
                    self.check_disallowed_lease_type(elem, span, ctx, false);
                }
            }
            Ty::Ptr(inner) => match &**inner {
                Ty::OsPrimitive(
                    ResolvedOsPrimitive::Inbox { .. }
                    | ResolvedOsPrimitive::InboxUnsized { .. }
                    | ResolvedOsPrimitive::Pool { .. }
                    | ResolvedOsPrimitive::DmaSlot { .. },
                ) => {
                    self.check_disallowed_lease_type(inner, span, ctx, false);
                }
                _ => self.check_disallowed_lease_type(inner, span, ctx, false),
            },
            Ty::Array { elem, .. } | Ty::Slice(elem) => {
                self.check_disallowed_lease_type(elem, span, ctx, false);
            }
            Ty::FnPtr { params, ret } => {
                for p in params {
                    self.check_disallowed_lease_type(p, span, ctx, false);
                }
                self.check_disallowed_lease_type(ret, span, ctx, false);
            }
            _ => {}
        }
    }
    fn check_extern_c_access(&mut self, prog_idx: usize, ident: &Ident) {
        if self.program_extern_c_scopes.is_empty()
            || !self.non_root_extern_c_names.contains(&ident.name)
        {
            return;
        }
        let Some(fn_info) = self.functions.get(&ident.name) else {
            return;
        };
        if !fn_info.is_extern_c {
            return;
        }
        let declared_locally = self
            .program_extern_c_scopes
            .get(prog_idx)
            .is_some_and(|s| s.contains(&ident.name));
        let imported_pub = self
            .rewritten_pub_extern_sites
            .contains(&(prog_idx, ident.span.start, ident.span.end));
        if !declared_locally && !imported_pub {
            self.emit(Diagnostic::new(
                DiagnosticKind::ModuleError,
                ident.span,
                format!(
                    "`extern \"C\"` function '{}' belongs to another module and is not declared here or imported as `pub`",
                    ident.name
                ),
            ));
        }
    }

    fn is_compile_time_const_expr_with_shadowed(
        &self,
        expr: &Expr,
        shadowed: &BTreeSet<String>,
        prog_idx: usize,
    ) -> bool {
        let mut visiting = BTreeSet::new();
        self.is_const_expr_inner(expr, &mut visiting, shadowed, prog_idx)
    }

    fn is_const_expr_inner(
        &self,
        expr: &Expr,
        visiting: &mut BTreeSet<String>,
        shadowed: &BTreeSet<String>,
        prog_idx: usize,
    ) -> bool {
        match &expr.kind {
            ExprKind::Literal(_) | ExprKind::Priority(_) => true,
            ExprKind::Ident(id) => {
                if shadowed.contains(&id.name) {
                    return false;
                }
                if self.consts.contains_key(&id.name) {
                    if let Some((raw, _, const_prog)) = self.raw_const_exprs.get(&id.name) {
                        if visiting.insert(id.name.clone()) {
                            let ok = self.is_const_expr_inner(
                                raw,
                                visiting,
                                &BTreeSet::new(),
                                *const_prog,
                            );
                            visiting.remove(&id.name);
                            return ok;
                        }
                    }
                    true
                } else {
                    self.visible_enums_for_variant(prog_idx, &id.name).len() == 1
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(en) = &target.kind {
                    if !shadowed.contains(&en.name)
                        && self
                            .enums
                            .get(&en.name)
                            .is_some_and(|e| e.has_variant(&field.name))
                    {
                        return true;
                    }
                }
                false
            }
            ExprKind::Unary { op, expr: inner } => match op {
                UnaryOp::Neg | UnaryOp::Not | UnaryOp::BitNot => {
                    self.is_const_expr_inner(inner, visiting, shadowed, prog_idx)
                }
                UnaryOp::AddrOf | UnaryOp::Deref => false,
            },
            ExprKind::Binary { lhs, rhs, .. } => {
                self.is_const_expr_inner(lhs, visiting, shadowed, prog_idx)
                    && self.is_const_expr_inner(rhs, visiting, shadowed, prog_idx)
            }
            ExprKind::Cast { expr: inner, .. } => {
                self.is_const_expr_inner(inner, visiting, shadowed, prog_idx)
            }
            ExprKind::ArrayLit(elems) => elems
                .iter()
                .all(|e| self.is_const_expr_inner(e, visiting, shadowed, prog_idx)),
            ExprKind::StructLit { fields, .. } => fields
                .iter()
                .all(|f| self.is_const_expr_inner(&f.value, visiting, shadowed, prog_idx)),
            _ => false,
        }
    }

    fn eval_const_bool_shadowed(
        &self,
        expr: &Expr,
        visiting: &mut BTreeSet<String>,
        shadowed: &BTreeSet<String>,
    ) -> Option<bool> {
        match &expr.kind {
            ExprKind::Literal(Literal::Bool(b)) => Some(*b),
            ExprKind::Ident(id) => {
                if shadowed.contains(&id.name) {
                    return None;
                }
                if !visiting.insert(id.name.clone()) {
                    return None;
                }
                let res = self
                    .raw_const_exprs
                    .get(&id.name)
                    .and_then(|(raw, _, _)| {
                        self.eval_const_bool_shadowed(raw, visiting, &BTreeSet::new())
                    });
                visiting.remove(&id.name);
                res
            }
            ExprKind::Unary {
                op: UnaryOp::Not,
                expr: inner,
            } => Some(!self.eval_const_bool_shadowed(inner, visiting, shadowed)?),
            ExprKind::Binary { op, lhs, rhs } => match op {
                BinaryOp::And => Some(
                    self.eval_const_bool_shadowed(lhs, visiting, shadowed)?
                        && self.eval_const_bool_shadowed(rhs, visiting, shadowed)?,
                ),
                BinaryOp::Or => Some(
                    self.eval_const_bool_shadowed(lhs, visiting, shadowed)?
                        || self.eval_const_bool_shadowed(rhs, visiting, shadowed)?,
                ),
                BinaryOp::Eq | BinaryOp::Ne => {
                    if let (Some(a), Some(b)) = (
                        self.eval_const_bool_shadowed(lhs, visiting, shadowed),
                        self.eval_const_bool_shadowed(rhs, visiting, shadowed),
                    ) {
                        return Some(if matches!(op, BinaryOp::Eq) { a == b } else { a != b });
                    }
                    let a = self.eval_const_expr_i128_shadowed(lhs, visiting, shadowed)?;
                    let b = self.eval_const_expr_i128_shadowed(rhs, visiting, shadowed)?;
                    Some(if matches!(op, BinaryOp::Eq) { a == b } else { a != b })
                }
                BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                    let a = self.eval_const_expr_i128_shadowed(lhs, visiting, shadowed)?;
                    let b = self.eval_const_expr_i128_shadowed(rhs, visiting, shadowed)?;
                    Some(match op {
                        BinaryOp::Lt => a < b,
                        BinaryOp::Le => a <= b,
                        BinaryOp::Gt => a > b,
                        BinaryOp::Ge => a >= b,
                        _ => unreachable!(),
                    })
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn eval_compile_time_irq_arg_with_shadowed(
        &self,
        expr: &Expr,
        ty: &Ty,
        shadowed: &BTreeSet<String>,
        prog_idx: usize,
    ) -> Option<IrqBindingArg> {
        if ty.is_bool() {
            let mut visiting = BTreeSet::new();
            if let Some(b) = self.eval_const_bool_shadowed(expr, &mut visiting, shadowed) {
                return Some(IrqBindingArg::ConstBool(b));
            }
        }
        if ty.is_integer() || matches!(ty, Ty::IntLit) {
            let mut visiting = BTreeSet::new();
            if let Some(value) = self.eval_const_expr_i128_shadowed(expr, &mut visiting, shadowed)
            {
                let final_ty = if matches!(ty, Ty::IntLit) {
                    Ty::u32()
                } else {
                    ty.clone()
                };
                return Some(IrqBindingArg::ConstInt {
                    ty: final_ty,
                    value,
                });
            }
        }
        if self.is_compile_time_const_expr_with_shadowed(expr, shadowed, prog_idx) {
            return Some(IrqBindingArg::ConstExpr {
                ty: ty.clone(),
                expr: expr.clone(),
            });
        }
        None
    }
}

impl<'a> BodyChecker<'a> {
    fn check_resolved_index_val(
        &mut self,
        tv: ExprVal,
        target: &Expr,
        _index_span: Span,
        span: Span,
    ) -> ExprVal {
        match &tv.ty {
            Ty::Slice(elem) | Ty::Array { elem, .. } => {
                let elem_ty = (**elem).clone();
                let prov = if elem_ty.contains_reference(&self.sema.structs) {
                    tv.prov
                } else {
                    Provenance::pure_static()
                };
                ExprVal {
                    ty: elem_ty,
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name: None,
                    field_endpoints: BTreeMap::new(),
                    span,
                }
            }
            _ => {
                if !tv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        target.span,
                        format!("cannot index into type '{}'", tv.ty),
                    ));
                }
                ExprVal::error(span)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_mmio_reg_write_access(
        &mut self,
        layout_name: &str,
        field: &Ident,
        elem_index: Option<&Expr>,
        op: AssignOp,
        value_expr: &Expr,
        rhs: &ExprVal,
        stmt_span: Span,
    ) {
        if let Some(idx_expr) = elem_index {
            let iv = self.check_expr_infallible(idx_expr);
            if !iv.ty.is_integer() && !iv.ty.is_error() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    idx_expr.span,
                    format!("array index must be an integer, found '{}'", iv.ty),
                ));
            }
        }

        let Some(layout_info) = self.sema.mmio_layouts.get(layout_name).cloned() else {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::UndefinedSymbol,
                field.span,
                format!("unknown MMIO register layout '{}'", layout_name),
            ));
            return;
        };
        let Some(reg) = layout_info.regs.iter().find(|r| r.name == field.name).cloned() else {
            let candidates = layout_info.regs.iter().map(|r| r.name.as_str());
            let mut diag = Diagnostic::new(
                DiagnosticKind::UndefinedSymbol,
                field.span,
                format!(
                    "MMIO layout '{}' has no register '{}'",
                    layout_name, field.name
                ),
            );
            if let Some(best) = closest_match(&field.name, candidates, 2) {
                diag = diag.with_note(format!("did you mean register `.{}`?", best));
            }
            self.sema.emit(diag);
            return;
        };

        let target_ty = if let Some(idx_expr) = elem_index {
            let Some(len) = reg.array_len else {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    field.span,
                    format!(
                        "cannot index scalar MMIO register '{}.{}' of type '{}'",
                        layout_name, reg.name, reg.elem_ty
                    ),
                ));
                return;
            };
            let shadowed = self.active_local_names();
            if let Some(idx) = self.sema.eval_const_expr_i128_shadowed(
                idx_expr,
                &mut BTreeSet::new(),
                &shadowed,
            ) {
                if idx < 0 || idx as u128 >= len as u128 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        idx_expr.span,
                        format!(
                            "MMIO register array '{}.{}' index {} out of bounds (length {})",
                            layout_name, reg.name, idx, len
                        ),
                    ));
                }
            }
            Ty::Prim(reg.elem_ty)
        } else if reg.array_len.is_some() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                stmt_span,
                format!(
                    "cannot assign whole MMIO register array '{}.{}'; assign an element with '{}.{}[i] = ...' instead",
                    layout_name, reg.name, layout_name, reg.name
                ),
            ));
            return;
        } else {
            Ty::Prim(reg.elem_ty)
        };

        match reg.access {
            MmioAccess::Rw => {}
            MmioAccess::Ro => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    stmt_span,
                    format!(
                        "cannot write to read-only `@ro` MMIO register '{}.{}'",
                        layout_name, reg.name
                    ),
                ));
                return;
            }
            MmioAccess::Wo => {
                if op != AssignOp::Assign {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        stmt_span,
                        format!(
                            "cannot perform compound read-modify-write `{}` on write-only `@wo` MMIO register '{}.{}'; use plain `=`",
                            op.as_str(),
                            layout_name,
                            reg.name
                        ),
                    ));
                    return;
                }
            }
            MmioAccess::W1c => {
                if op != AssignOp::Assign {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        stmt_span,
                        format!(
                            "cannot perform compound read-modify-write `{}` on write-1-to-clear `@w1c` MMIO register '{}.{}'; write masks directly using `=`",
                            op.as_str(),
                            layout_name,
                            reg.name
                        ),
                    ));
                    return;
                }
            }
        }

        if op == AssignOp::Assign {
            if !rhs.ty.is_assignable_to(&target_ty) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    rhs.span,
                    format!(
                        "cannot assign '{}' to MMIO register '{}.{}' of type '{}'",
                        rhs.ty, layout_name, reg.name, target_ty
                    ),
                ));
            } else {
                self.sema.check_f32_coercion_expr(value_expr, &target_ty);
            }
        } else {
            self.check_compound_op_types(op, &target_ty, value_expr, &rhs.ty, stmt_span);
        }
    }

    fn build_subsystem_field_chain(
        &self,
        mut base_ty: Ty,
        field_names: &[String],
    ) -> Option<Vec<(String, String)>> {
        let mut chain = Vec::with_capacity(field_names.len());
        for fname in field_names {
            let Ty::Struct(sname) = base_ty else {
                return None;
            };
            let f = self.sema.structs.get(&sname)?.field(fname)?;
            chain.push((fname.clone(), f.ty.to_string()));
            base_ty = f.ty.clone();
        }
        Some(chain)
    }

    fn classify_irq_attach_arg(&self, expr: &Expr, val: &ExprVal) -> Option<BootAttachArgSpec> {
        match &expr.kind {
            ExprKind::Unary {
                op: UnaryOp::AddrOf,
                expr: inner,
            } => {
                let mut rev_fields = Vec::new();
                let mut cur = &**inner;
                while let ExprKind::Field { target, field } = &cur.kind {
                    rev_fields.push(field.name.clone());
                    cur = target;
                }
                rev_fields.reverse();
                if let ExprKind::Ident(id) = &cur.kind {
                    if rev_fields.is_empty() {
                        if self.sema.primitives.contains_key(&id.name) {
                            return Some(BootAttachArgSpec::Static(IrqBindingArg::PrimAddr(
                                id.name.clone(),
                            )));
                        }
                        if self.sema.mmio_instances.contains_key(&id.name) {
                            return Some(BootAttachArgSpec::Static(IrqBindingArg::MmioAddr(
                                id.name.clone(),
                            )));
                        }
                    }
                    if let Some(sub) = self.sema.subsystems.get(&id.name) {
                        let base_ty = Ty::Struct(sub.struct_name.clone());
                        if let Some(field_chain) =
                            self.build_subsystem_field_chain(base_ty, &rev_fields)
                        {
                            return Some(BootAttachArgSpec::Static(
                                IrqBindingArg::SubsystemPath {
                                    root: id.name.clone(),
                                    root_struct: sub.struct_name.clone(),
                                    field_chain,
                                },
                            ));
                        }
                    }
                    if !rev_fields.is_empty() {
                        if let Some(idx) = self.param_names.iter().position(|p| p == &id.name) {
                            if let Some(local) = self.lookup_local(&id.name) {
                                let base_ty = match &local.ty {
                                    Ty::Ptr(inner_ty) => (**inner_ty).clone(),
                                    other => other.clone(),
                                };
                                if let Some(field_chain) =
                                    self.build_subsystem_field_chain(base_ty, &rev_fields)
                                {
                                    return Some(BootAttachArgSpec::Param {
                                        index: idx,
                                        field_chain,
                                    });
                                }
                            }
                        }
                    }
                }
                None
            }
            ExprKind::Ident(id) => {
                if let Some(idx) = self.param_names.iter().position(|p| p == &id.name) {
                    return Some(BootAttachArgSpec::Param {
                        index: idx,
                        field_chain: Vec::new(),
                    });
                }
                if self.sema.mmio_instances.contains_key(&id.name) {
                    return Some(BootAttachArgSpec::Static(IrqBindingArg::MmioAddr(
                        id.name.clone(),
                    )));
                }
                let shadowed = self.active_local_names();
                self.sema
                    .eval_compile_time_irq_arg_with_shadowed(
                        expr,
                        &val.ty,
                        &shadowed,
                        self.prog_idx,
                    )
                    .map(BootAttachArgSpec::Static)
            }
            _ => {
                let shadowed = self.active_local_names();
                self.sema
                    .eval_compile_time_irq_arg_with_shadowed(
                        expr,
                        &val.ty,
                        &shadowed,
                        self.prog_idx,
                    )
                    .map(BootAttachArgSpec::Static)
            }
        }
    }

    fn check_irq_method(
        &mut self,
        rv: &ExprVal,
        method: &Ident,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        let is_shared = match method.name.as_str() {
            "attach" => false,
            "attach_shared" => true,
            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    method.span,
                    format!(
                        "unknown method '{}' on hardware `irq` token; supported: `.attach(...)` and `.attach_shared(...)`",
                        other
                    ),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                return ExprVal::error(span);
            }
        };

        let Some(irq_name) = rv.endpoint_name.clone() else {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                "`.attach(...)` requires a named top-level `irq` declaration",
            ));
            return ExprVal::error(span);
        };
        let irq_num = self
            .sema
            .irqs
            .get(&irq_name)
            .map(|q| q.irq_num)
            .unwrap_or(0);

        let mut invalid_ctx = false;
        if self.in_defer() {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' inside a 'defer' cleanup statement",
                    irq_name
                ),
            ));
        } else if self.in_isr() {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' inside ISR '{}'; IRQ bindings must happen during sequential boot",
                    irq_name,
                    self.display_caller_name()
                ),
            ));
        } else if matches!(self.body_kind, ActiveBodyKind::Task) {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' inside task '{}'; IRQ bindings must happen during sequential boot",
                    irq_name,
                    self.display_caller_name()
                ),
            ));
        } else if self.loop_depth > 0 {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' inside a loop; IRQ bindings must be static in the boot tree",
                    irq_name
                ),
            ));
        } else if self.after_diverge || self.main_supervisor_started {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' after 'main()' supervisor loop or diverging control flow",
                    irq_name
                ),
            ));
        }

        if args.len() != 1 || args[0].name.is_some() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "`{}.{}(...)` expects 1 positional ISR handler argument (`handler(args...)` or `handler`)",
                    irq_name, method.name
                ),
            ));
            for a in args {
                let _ = self.check_expr_infallible(&a.value);
            }
            return ExprVal::scalar(Ty::void(), span);
        }

        let handler_arg = &args[0];
        let (isr_ident, isr_call_args): (&Ident, &[CallArg]) = match &handler_arg.value.kind {
            ExprKind::Ident(id) => (id, &[]),
            ExprKind::Call {
                callee,
                generic_args,
                args: call_args,
            } => {
                if !generic_args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::IrqAttachmentError,
                        handler_arg.span,
                        "`isr` handlers cannot accept generic type arguments",
                    ));
                }
                match &callee.kind {
                    ExprKind::Ident(id) => (id, call_args.as_slice()),
                    _ => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            callee.span,
                            "expected named `isr` handler in `.attach(...)`",
                        ));
                        for a in call_args {
                            let _ = self.check_expr_infallible(&a.value);
                        }
                        return ExprVal::scalar(Ty::void(), span);
                    }
                }
            }
            _ => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    handler_arg.span,
                    format!(
                        "expected `isr` handler invocation `isr_name(args...)` or `isr_name` in `{}.{}(...)`",
                        irq_name, method.name
                    ),
                ));
                let _ = self.check_expr_infallible(&handler_arg.value);
                return ExprVal::scalar(Ty::void(), span);
            }
        };

        let Some(isr_info) = self.sema.isrs.get(&isr_ident.name).cloned() else {
            if self.sema.functions.contains_key(&isr_ident.name) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    isr_ident.span,
                    format!(
                        "'{}' is a `fn`, not an `isr` handler; declare `isr {}(...) {{ ... }}` instead",
                        isr_ident.name, isr_ident.name
                    ),
                ));
            } else if self.sema.tasks.contains_key(&isr_ident.name) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    isr_ident.span,
                    format!(
                        "'{}' is a `task`, not an `isr` handler",
                        isr_ident.name
                    ),
                ));
            } else {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    isr_ident.span,
                    format!("undeclared `isr` handler '{}'", isr_ident.name),
                ));
            }
            for a in isr_call_args {
                let _ = self.check_expr_infallible(&a.value);
            }
            return ExprVal::scalar(Ty::void(), span);
        };

        if isr_call_args.len() != isr_info.params.len() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                handler_arg.span,
                format!(
                    "ISR handler '{}' expects {} argument(s), found {}",
                    isr_info.vector,
                    isr_info.params.len(),
                    isr_call_args.len()
                ),
            ));
        }

        let mut arg_specs = Vec::with_capacity(isr_call_args.len());
        for (idx, a) in isr_call_args.iter().enumerate() {
            let av = self.check_expr_infallible(&a.value);
            let param_name = isr_info
                .params
                .get(idx)
                .map(|p| p.name.clone())
                .unwrap_or_else(|| format!("arg{idx}"));
            if let Some(param) = isr_info.params.get(idx) {
                if !av.ty.is_assignable_to(&param.ty) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        a.span,
                        format!(
                            "argument {} ('{}') to ISR handler '{}' has type '{}', expected '{}'",
                            idx + 1,
                            param.name,
                            isr_info.vector,
                            av.ty,
                            param.ty
                        ),
                    ));
                } else {
                    self.sema.check_f32_coercion_expr(&a.value, &param.ty);
                }
            }
            let spec = self.classify_irq_attach_arg(&a.value, &av);
            if spec.is_none() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    a.span,
                    format!(
                        "argument {} ('{}') passed to `{}.{}` handler '{}' must be a static OS primitive `&prim`, `mmio` instance, or compile-time constant expression",
                        idx + 1,
                        param_name,
                        irq_name,
                        method.name,
                        isr_info.vector
                    ),
                ));
            }
            let forwarded_pname = match &spec {
                Some(BootAttachArgSpec::Param { index, .. }) => self
                    .param_names
                    .get(*index)
                    .cloned()
                    .unwrap_or(param_name),
                _ => param_name,
            };
            arg_specs.push((spec, a.span, forwarded_pname));
        }

        if !invalid_ctx && self.can_record_boot_step() {
            self.sema
                .call_graph
                .mark_direct_attach(&self.caller_key, &irq_name);
        }
        if self.can_record_boot_step() {
            let in_caller = self.display_caller_name().to_string();
            let file = self.sema.current_file.clone();
            if let Some(top) = self.boot_block_stack.last_mut() {
                top.push(IrqBootNode::Attach(PendingIrqAttach {
                    irq_name,
                    irq_num,
                    is_shared,
                    isr_name: isr_info.vector,
                    args: arg_specs,
                    span,
                    file,
                    in_caller,
                    invalid_ctx,
                }));
            }
        }

        if self.caller_key == "main" {
            self.has_attached_irq = true;
        }
        ExprVal::scalar(Ty::void(), span)
    }
}
