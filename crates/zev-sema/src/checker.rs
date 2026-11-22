//! Multi-pass semantic analyzer and static verifier for the Zev Language-OS.

use std::collections::{BTreeMap, BTreeSet};

use zev_ast::{
    AssignOp, Attr, AttrKind, BinaryOp, Block, CallArg, CatchClause, ConstDecl, DurationLit,
    EnumDecl, ErrorSpec, Expr, ExprKind, FnDecl, GlobalVarDecl, Ident, ImportDecl, IsrDecl, Item,
    ItemKind, Literal, OsPrimitiveDecl, OsPrimitiveType, Param, PrimType, Priority, Program,
    SpawnStmt, Span, Stmt, StmtKind, StructDecl, SwitchPat, TaskDecl, TypeExpr, TypeKind, UnaryOp,
};

use crate::call_graph::{CallContext, CallEdge, CallGraph, CallableKind, ExternCallbackSite};
use crate::diag::{Diagnostic, DiagnosticKind};
use crate::model::{
    ConstInfo, DEFAULT_TICK_PERIOD_US, FnEffect, FunctionInfo, IsrInfo, ParamInfo, PrimitiveInfo,
    PriorityBoostSite, SemanticModel, SpawnedTaskInstance, TaskInfo,
};
use crate::provenance::{LeaseEnv, Provenance};
use crate::suggest::{
    advise_primitive_method, closest_match, suggest_foreign_type_name,
};
use crate::types::{
    EnumInfo, FieldInfo, NamespaceKind, ResolvedErrorSpec, ResolvedOsPrimitive, StructInfo, Ty,
};

/// Configuration options for semantic analysis.
#[derive(Clone, Debug)]
pub struct AnalyzeOptions {
    /// Default scheduler tick resolution in microseconds (`1000` = `1ms`).
    pub default_tick_period_us: u64,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            default_tick_period_us: DEFAULT_TICK_PERIOD_US,
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

/// Built-in SVD bitmask/pin constants available when a `chip.*` hardware module is imported
/// (unless shadowed by an explicit user `const` declaration).
const BUILTIN_SVD_CONSTS: &[&str] = &[
    "MOTOR_EN",
    "FAULT_LED",
    "HALF_TRANSFER",
    "FULL_TRANSFER",
    "TRANSFER_ERROR",
    "TRANSFER_COMPLETE",
    "TXE",
    "RXNE",
    "TC",
    "IDLE",
    "PIN_0",
    "PIN_1",
    "PIN_2",
    "PIN_3",
    "PIN_4",
    "PIN_5",
    "PIN_6",
    "PIN_7",
];

#[derive(Clone, Debug)]
struct IllegalGlobalInfo {
    ty: Ty,
    decl_span: Span,
}

#[derive(Clone, Debug)]
struct LocalVar {
    name: String,
    ty: Ty,
    decl_depth: u32,
    prov: Provenance,
    field_prov: BTreeMap<String, Provenance>,
    endpoint_alias: Option<String>,
}

#[derive(Clone, Debug)]
struct ExprVal {
    ty: Ty,
    error_spec: ResolvedErrorSpec,
    prov: Provenance,
    endpoint_name: Option<String>,
    span: Span,
}

impl ExprVal {
    fn error(span: Span) -> Self {
        Self {
            ty: Ty::Error,
            error_spec: ResolvedErrorSpec::None,
            prov: Provenance::pure_static(),
            endpoint_name: None,
            span,
        }
    }

    fn scalar(ty: Ty, span: Span) -> Self {
        Self {
            ty,
            error_spec: ResolvedErrorSpec::None,
            prov: Provenance::pure_static(),
            endpoint_name: None,
            span,
        }
    }
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
    endpoint_from_param: Option<usize>,
    ref_param_origins: BTreeSet<usize>,
}

#[derive(Clone, Debug)]
struct BootCallArg {
    static_endpoint: Option<String>,
    endpoint_from_param: Option<usize>,
    has_non_static_local_ref: bool,
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
struct RecordedCallArgs {
    caller: String,
    callee: String,
    args: Vec<BootCallArg>,
}

pub(crate) struct Analyzer {
    tick_period_us: u64,
    tick_configured_explicitly: bool,
    systick_hz: Option<u64>,
    has_chip_import: bool,
    chip_imports: BTreeMap<String, String>,
    structs: BTreeMap<String, StructInfo>,
    enums: BTreeMap<String, EnumInfo>,
    /// Maps unqualified enum variant name -> list of enums defining that variant.
    variant_to_enums: BTreeMap<String, Vec<String>>,
    consts: BTreeMap<String, ConstInfo>,
    builtin_svd_consts: BTreeSet<String>,
    primitives: BTreeMap<String, PrimitiveInfo>,
    illegal_globals: BTreeMap<String, IllegalGlobalInfo>,
    functions: BTreeMap<String, FunctionInfo>,
    fn_decl_order: Vec<String>,
    fn_boot_steps: BTreeMap<String, Vec<BootStep>>,
    recorded_call_args: Vec<RecordedCallArgs>,
    tasks: BTreeMap<String, TaskInfo>,
    isrs: BTreeMap<String, IsrInfo>,
    spawned_tasks: Vec<SpawnedTaskInstance>,
    soft_yielding_in_coro: BTreeSet<String>,
    soft_yielding_in_noyield: BTreeSet<String>,
    dual_specialized_fns: BTreeSet<String>,
    main_supervisor_priority: Option<Priority>,
    call_graph: CallGraph,
    current_file: Option<String>,
    node_files: BTreeMap<String, String>,
    diagnostics: Vec<Diagnostic>,
}

impl Analyzer {
    pub fn new(opts: &AnalyzeOptions) -> Self {
        Self {
            tick_period_us: opts.default_tick_period_us.max(1),
            tick_configured_explicitly: false,
            systick_hz: None,
            has_chip_import: false,
            chip_imports: BTreeMap::new(),
            structs: BTreeMap::new(),
            enums: BTreeMap::new(),
            variant_to_enums: BTreeMap::new(),
            consts: BTreeMap::new(),
            builtin_svd_consts: BTreeSet::new(),
            primitives: BTreeMap::new(),
            illegal_globals: BTreeMap::new(),
            functions: BTreeMap::new(),
            fn_decl_order: Vec::new(),
            fn_boot_steps: BTreeMap::new(),
            recorded_call_args: Vec::new(),
            tasks: BTreeMap::new(),
            isrs: BTreeMap::new(),
            spawned_tasks: Vec::new(),
            soft_yielding_in_coro: BTreeSet::new(),
            soft_yielding_in_noyield: BTreeSet::new(),
            dual_specialized_fns: BTreeSet::new(),
            main_supervisor_priority: None,
            call_graph: CallGraph::new(),
            current_file: None,
            node_files: BTreeMap::new(),
            diagnostics: Vec::new(),
        }
    }

    fn emit(&mut self, mut diag: Diagnostic) {
        if diag.file.is_none() {
            diag.file = self.current_file.clone();
        }
        if diag.note.is_none() {
            diag.note = Some(diag.kind.doc_hint().to_string());
        }
        self.diagnostics.push(diag);
    }

    pub fn run(mut self, programs: &[(&Program, Option<&str>)]) -> AnalysisOutput {
        // Merge items while tracking optional file names per item if needed.
        let mut all_items: Vec<(&Item, Option<String>)> = Vec::new();
        for (prog, file) in programs {
            for item in &prog.items {
                all_items.push((item, file.map(str::to_string)));
            }
        }

        // Pass 1a: Imports, struct names, and enum declarations.
        for (item, file) in &all_items {
            self.current_file = file.clone();
            match &item.kind {
                ItemKind::Import(imp) => self.collect_import(imp),
                ItemKind::Struct(st) => self.register_struct_header(st),
                ItemKind::Enum(en) => self.collect_enum(en),
                _ => {}
            }
        }

        if self.has_chip_import {
            for &c in BUILTIN_SVD_CONSTS {
                self.builtin_svd_consts.insert(c.to_string());
            }
        }

        // Pass 1b: Resolve struct fields and verify packed struct & recursive value invariants.
        for (item, file) in &all_items {
            self.current_file = file.clone();
            if let ItemKind::Struct(st) = &item.kind {
                self.resolve_struct_fields(st);
            }
        }
        self.current_file = None;
        self.check_recursive_struct_values();

        // Pass 1c: Pre-scan for `sys.use_systick` / `sys.set_tick` so custom tick period is known
        // before validating duration literals anywhere in the program.
        for (item, _) in &all_items {
            if let ItemKind::Fn(f) = &item.kind {
                if let Some(body) = &f.body {
                    self.prescan_tick_config(body);
                }
            }
        }

        // Pass 1d: Collect consts, OS primitives, illegal mutable globals, and callable signatures.
        for (item, file) in &all_items {
            self.current_file = file.clone();
            match &item.kind {
                ItemKind::Const(c) => self.collect_const(c),
                ItemKind::OsPrimitive(p) => self.collect_os_primitive(&item.attrs, p),
                ItemKind::GlobalVar(g) => self.collect_global_var(&item.attrs, g),
                ItemKind::Fn(f) => self.collect_fn_sig(&item.attrs, f),
                ItemKind::Task(t) => self.collect_task_sig(&item.attrs, t),
                ItemKind::Isr(i) => self.collect_isr_sig(&item.attrs, i),
                _ => {}
            }
        }

        // Pass 2a: Analyze non-`main` helper functions first so interprocedural `return_param_origins`
        // summaries are ready before `main` and `task` bodies call them.
        for (item, file) in &all_items {
            self.current_file = file.clone();
            if let ItemKind::Fn(f) = &item.kind {
                if f.name.name != "main" {
                    self.check_fn_body(&item.attrs, f);
                }
            }
        }
        // Pass 2b: Analyze `main()`, `task` blueprints, and `isr` handlers.
        for (item, file) in &all_items {
            self.current_file = file.clone();
            match &item.kind {
                ItemKind::Fn(f) if f.name.name == "main" => self.check_fn_body(&item.attrs, f),
                ItemKind::Task(t) => self.check_task_body(&item.attrs, t),
                ItemKind::Isr(i) => self.check_isr_body(&item.attrs, i),
                _ => {}
            }
        }
        self.current_file = None;

        // Pass 3: Whole-program transitive yield inference, `noyield`/`critical`/`isr` call checks,
        // `extern "C"` callback checks, and yielding recursion cycle detection.
        self.run_whole_program_verification();

        let main_can_yield = self
            .functions
            .get("main")
            .is_some_and(|m| m.can_yield);

        let model = SemanticModel {
            tick_period_us: self.tick_period_us,
            systick_hz: self.systick_hz,
            chip_imports: self.chip_imports,
            structs: self.structs,
            enums: self.enums,
            consts: self.consts,
            primitives: self.primitives,
            functions: self.functions,
            tasks: self.tasks,
            isrs: self.isrs,
            spawned_tasks: self.spawned_tasks,
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

    fn collect_import(&mut self, imp: &ImportDecl) {
        let full_path = imp
            .path
            .iter()
            .map(|id| id.name.as_str())
            .collect::<Vec<_>>()
            .join(".");
        let alias = imp
            .alias
            .as_ref()
            .map(|a| a.name.clone())
            .or_else(|| imp.path.last().map(|l| l.name.clone()))
            .unwrap_or_else(|| full_path.clone());

        if imp.path.first().is_some_and(|p| p.name == "chip") {
            self.has_chip_import = true;
            self.chip_imports.insert(alias, full_path);
        } else {
            self.chip_imports.insert(alias, full_path);
        }
    }

    fn register_struct_header(&mut self, st: &StructDecl) {
        if self.structs.contains_key(&st.name.name) || self.enums.contains_key(&st.name.name) {
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
                fields: Vec::new(),
                span: st.span,
            },
        );
    }

    fn collect_enum(&mut self, en: &EnumDecl) {
        if self.enums.contains_key(&en.name.name) || self.structs.contains_key(&en.name.name) {
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
            fields.push(FieldInfo {
                name: f.name.name.clone(),
                ty,
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
                self.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    span,
                    format!(
                        "recursive struct '{}' has infinite size without indirection",
                        name
                    ),
                ));
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

    fn collect_const(&mut self, c: &ConstDecl) {
        if !self.check_top_level_name_unique(&c.name) {
            return;
        }
        // User const shadows any built-in SVD constant of the same name cleanly.
        self.builtin_svd_consts.remove(&c.name.name);

        let ty = self.resolve_type(&c.ty);
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

    fn collect_os_primitive(&mut self, item_attrs: &[Attr], p: &OsPrimitiveDecl) {
        if !self.check_top_level_name_unique(&p.name) {
            return;
        }
        let dma_safe = has_attr_dma_safe(item_attrs) || has_attr_dma_safe(&p.attrs);
        let resolved = self.resolve_os_primitive(&p.ty, p.span);

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
        let capacity = resolved.capacity();
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
                clamps_sleep_depth: kind.clamps_sleep_depth(),
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
                    "mutable raw global variable '{}' is forbidden; top-level declarations must be 'const' or an OS primitive (signal, inbox, watch, ping_pong, lease)",
                    g.name.name
                ),
            )
            .with_note(format!(
                "replace with `const {ty} {name} = ...;` (immutable), `watch<{ty}> {name} = 0;` (single-writer latest value via `.read()`/`.publish(val)`), or `inbox<{ty}, 8> {name};` (queued events via `.post(ev)`/`.recv()`)",
                ty = ty_str,
                name = g.name.name
            )),
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
        let error_spec = match &f.error_spec {
            None => ResolvedErrorSpec::None,
            Some(ErrorSpec::Bare(_)) => ResolvedErrorSpec::Untyped,
            Some(ErrorSpec::Typed(err_ty_expr)) => {
                let err_ty = self.resolve_type(err_ty_expr);
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

        let params = self.resolve_params(&f.params, &f.name.name);
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
        let params = self.resolve_params(&t.params, &t.name.name);
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
                callees: Vec::new(),
                priority_boosts: Vec::new(),
                span: i.span,
            },
        );
    }

    fn resolve_params(&mut self, params: &[Param], owner: &str) -> Vec<ParamInfo> {
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
            out.push(ParamInfo {
                name: p.name.name.clone(),
                ty,
                span: p.span,
            });
        }
        out
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
                if self.chip_imports.contains_key(&ident.name) {
                    return Ty::HwPeripheral(ident.name.clone());
                }
                let hint = if let Some(z_ty) = suggest_foreign_type_name(&ident.name) {
                    format!(
                        "did you mean Zev built-in type `{}`? (Zev uses explicit fixed-width types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `usize`, `bool`, `u16be`, `u32le`, `[]u8`)",
                        z_ty
                    )
                } else {
                    let prim_names = [
                        "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize",
                        "u16be", "u16le", "u32be", "u32le", "u64be", "u64le", "bool", "void",
                    ];
                    let candidates = prim_names
                        .into_iter()
                        .chain(self.structs.keys().map(String::as_str))
                        .chain(self.enums.keys().map(String::as_str));
                    if let Some(best) = closest_match(&ident.name, candidates, 2) {
                        format!("did you mean type `{}`?", best)
                    } else {
                        "built-in Zev types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `usize`, `bool`, `void`, `u16be`, `u32le`, `[]T`, `*T`".to_string()
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
                if let Some(first) = segments.first() {
                    if first.name == "chip" || self.chip_imports.contains_key(&first.name) {
                        let path = segments
                            .iter()
                            .map(|s| s.name.as_str())
                            .collect::<Vec<_>>()
                            .join(".");
                        return Ty::HwPeripheral(path);
                    }
                }
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
            TypeKind::Slice(elem) => Ty::Slice(Box::new(self.resolve_type(elem))),
            TypeKind::Array { elem, len } => {
                if *len == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        te.span,
                        "fixed array length must be greater than 0",
                    ));
                }
                Ty::Array {
                    elem: Box::new(self.resolve_type(elem)),
                    len: *len,
                }
            }
            TypeKind::Ptr(inner) => Ty::Ptr(Box::new(self.resolve_type(inner))),
            TypeKind::OsPrimitive(os) => Ty::OsPrimitive(self.resolve_os_primitive(os, te.span)),
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
            OsPrimitiveType::Lease { elem, cap } => {
                if *cap == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        span,
                        "lease buffer capacity must be greater than 0",
                    ));
                }
                ResolvedOsPrimitive::Lease {
                    elem: Box::new(self.resolve_non_void_elem(elem)),
                    cap: *cap,
                }
            }
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
        let force_noyield = has_attr_noyield(item_attrs) || has_attr_noyield(&f.attrs);
        let mut checker = BodyChecker::new(
            self,
            &f.name.name,
            CallableKind::Fn,
            ActiveBodyKind::Fn {
                ret_ty: fn_info.ret_ty.clone(),
                error_spec: fn_info.error_spec.clone(),
            },
        );
        if force_noyield {
            checker.noyield_depth += 1;
        }
        checker.bind_params(&fn_info.params);
        let diverges = checker.check_block_stmts(&body.stmts);
        let ret_origins: Vec<usize> = checker.return_param_origins.into_iter().collect();
        let boot_steps = checker.boot_steps;

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

        self.fn_boot_steps.insert(f.name.name.clone(), boot_steps);
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
        if !self.isrs.contains_key(&i.vector.name) {
            return;
        }
        let node_key = format!("isr::{}", i.vector.name);
        let mut checker =
            BodyChecker::new(self, &node_key, CallableKind::Isr, ActiveBodyKind::Isr);
        let _ = checker.check_block_stmts(&i.body.stmts);
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
                        .with_file(caller_file)
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
                        .with_file(caller_file)
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
                        .with_file(caller_file)
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
                        .with_file(caller_file)
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
                        .with_file(caller_file)
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                }
            }
        }

        // 2. Check all `extern "C"` callback arguments for yielding or boot-spawning functions.
        let cb_sites = self.call_graph.extern_callbacks.clone();
        for site in &cb_sites {
            if let Some(cb_node) = self.call_graph.nodes.get(&site.callback_fn).cloned() {
                if cb_node.any_yield_effect.can_yield() {
                    let cb_file = self.node_files.get(&site.callback_fn).cloned();
                    let sub_chain = self
                        .call_graph
                        .shortest_any_yield_chain(&site.callback_fn)
                        .join(" -> ");
                    let note = format!(
                        "yield chain: extern \"C\" {}(...) callback -> {}; stackless coroutines cannot suspend across C stack frames—pass a non-yielding `fn` callback",
                        site.extern_fn, sub_chain
                    );
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::YieldingCallbackToExternC,
                            site.span,
                            format!(
                                "cannot pass yielding function '{}' as callback to extern \"C\" function '{}': stackless coroutine yields cannot unwind through C stack frames",
                                site.callback_fn, site.extern_fn
                            ),
                        )
                        .with_file(cb_file)
                        .with_note(note),
                    );
                }
                if cb_node.spawns_in_boot {
                    let cb_file = self.node_files.get(&site.callback_fn).cloned();
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            site.span,
                            format!(
                                "cannot pass boot-spawning function '{}' as callback to extern \"C\" function '{}'",
                                site.callback_fn, site.extern_fn
                            ),
                        )
                        .with_file(cb_file),
                    );
                }
            }
        }

        // 3. Detect recursive call cycles across boot-spawning functions (`E0027`) and yielding functions (`E0012`).
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

        // 4. Verify interprocedural static-reference provenance for parameters forwarded into `spawn` (`E0010`),
        //    then unroll `main()`'s static boot tree into `self.spawned_tasks`.
        self.verify_boot_spawn_arg_provenance();
        self.unroll_boot_spawn_tree();

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

        loop {
            let mut changed = false;
            for call in &self.recorded_call_args {
                let Some(callee_params) = spawning_ref_params.get(&call.callee).cloned() else {
                    continue;
                };
                for p_idx in callee_params {
                    if let Some(arg) = call.args.get(p_idx) {
                        for &caller_p in &arg.ref_param_origins {
                            if spawning_ref_params
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

    fn unroll_boot_spawn_tree(&mut self) {
        self.spawned_tasks.clear();
        for t in self.tasks.values_mut() {
            t.instance_count = 0;
        }

        let roots: Vec<String> = if self.functions.contains_key("main") {
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
            self.unroll_boot_fn(&root, &mut path, false, &[], &mut visiting);
        }
    }

    fn unroll_boot_fn(
        &mut self,
        fn_name: &str,
        path: &mut Vec<String>,
        cond_active: bool,
        param_endpoints: &[Option<String>],
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
                    for b in &arg_bindings {
                        let resolved = b.static_endpoint.clone().or_else(|| {
                            b.endpoint_from_param
                                .and_then(|idx| param_endpoints.get(idx).cloned().flatten())
                        });
                        endpoint_bindings.push(resolved);
                    }
                    let mut provenance_path = path.clone();
                    provenance_path.push(task_name.clone());
                    let instance_id = self.spawned_tasks.len();
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
                    for a in &args {
                        let resolved = a.static_endpoint.clone().or_else(|| {
                            a.endpoint_from_param
                                .and_then(|idx| param_endpoints.get(idx).cloned().flatten())
                        });
                        child_endpoints.push(resolved);
                    }
                    path.push(callee.clone());
                    self.unroll_boot_fn(
                        &callee,
                        path,
                        cond_active || conditional,
                        &child_endpoints,
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
    param_names: Vec<String>,
    boot_steps: Vec<BootStep>,
    return_param_origins: BTreeSet<usize>,
    recorded_boosts: Vec<PriorityBoostSite>,
    last_struct_lit_fields: BTreeMap<(u32, u32), BTreeMap<String, Provenance>>,
    /// Tracks lexical scope depths at loop entry and outer variables carrying DMA leases read inside the loop.
    loop_entry_depths: Vec<u32>,
    loop_outer_reads: Vec<Vec<(String, LeaseEnv, Provenance, Span)>>,
}

impl<'a> BodyChecker<'a> {
    fn new(
        sema: &'a mut Analyzer,
        caller_key: &str,
        caller_kind: CallableKind,
        body_kind: ActiveBodyKind,
    ) -> Self {
        Self {
            sema,
            caller_key: caller_key.to_string(),
            caller_kind,
            body_kind,
            scopes: vec![BTreeMap::new()], // depth 1 = top-level fn/task/isr body
            leases: LeaseEnv::new(),
            loop_depth: 0,
            noyield_depth: 0,
            critical_depth: 0,
            defer_depth: 0,
            branch_depth: 0,
            after_diverge: false,
            main_supervisor_started: false,
            param_names: Vec::new(),
            boot_steps: Vec::new(),
            return_param_origins: BTreeSet::new(),
            recorded_boosts: Vec::new(),
            last_struct_lit_fields: BTreeMap::new(),
            loop_entry_depths: Vec::new(),
            loop_outer_reads: Vec::new(),
        }
    }

    fn current_depth(&self) -> u32 {
        self.scopes.len() as u32
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

    fn resolve_param_endpoint_index(&self, endpoint_name: &Option<String>) -> Option<usize> {
        let ep = endpoint_name.as_deref()?;
        let pname = ep.strip_prefix("param:")?;
        self.param_names.iter().position(|n| n == pname)
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
            let endpoint_alias = match &p.ty {
                Ty::OsPrimitive(_) => Some(format!("param:{}", p.name)),
                Ty::Ptr(inner) if matches!(**inner, Ty::OsPrimitive(_)) => {
                    Some(format!("param:{}", p.name))
                }
                _ => None,
            };
            let prov = if p.ty.contains_reference(&self.sema.structs) {
                Provenance::for_param(idx, is_ptr, endpoint_alias.clone())
            } else if matches!(p.ty, Ty::FnPtr { .. }) {
                Provenance::pure_static()
            } else {
                Provenance::pure_static()
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
        pool.extend(self.sema.builtin_svd_consts.iter().map(String::as_str));
        pool.extend(self.sema.variant_to_enums.keys().map(String::as_str));
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

    fn merge_branch_vars(&mut self, base: &[BTreeMap<String, LocalVar>], branches: &[Vec<BTreeMap<String, LocalVar>>]) {
        if branches.is_empty() {
            return;
        }
        for (depth_idx, scope) in self.scopes.iter_mut().enumerate() {
            if depth_idx >= base.len() {
                break;
            }
            for (name, var) in scope.iter_mut() {
                let mut merged_prov: Option<Provenance> = None;
                for b in branches {
                    if let Some(b_var) = b.get(depth_idx).and_then(|s| s.get(name)) {
                        merged_prov = Some(match merged_prov {
                            None => b_var.prov.clone(),
                            Some(acc) => acc.merge(&b_var.prov),
                        });
                    }
                }
                if let Some(mp) = merged_prov {
                    var.prov = mp;
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
        self.with_nested_scope(|this| this.check_block_stmts(&block.stmts))
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

                if let Some(init_expr) = init {
                    let val = self.check_expr_infallible(init_expr);
                    if matches!(init_expr.kind, ExprKind::StructLit { .. }) {
                        if let Some(fp) = self
                            .last_struct_lit_fields
                            .remove(&(init_expr.span.start, init_expr.span.end))
                        {
                            field_prov = fp;
                        }
                    }
                    if !val.ty.is_assignable_to(&resolved_ty) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            init_expr.span,
                            format!(
                                "variable '{}' declared as '{}', but initialized with '{}'",
                                name.name, resolved_ty, val.ty
                            ),
                        ));
                    }
                    if resolved_ty.contains_reference(&self.sema.structs)
                        || resolved_ty.contains_fn_ptr(&self.sema.structs)
                    {
                        if decl_depth < val.prov.origin_depth {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::ScopeDepthEscape,
                                init_expr.span,
                                format!(
                                    "cannot store reference/slice of inner scope depth {} into variable '{}' at outer scope depth {}",
                                    val.prov.origin_depth, name.name, decl_depth
                                ),
                            ));
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
                    },
                );
                false
            }

            StmtKind::Assign { target, op, value } => {
                let rhs = self.check_expr_infallible(value);
                self.check_assignment(target, *op, rhs, stmt.span);
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
                    ActiveBodyKind::Fn { ret_ty, .. } => match (ret_ty.is_void(), opt_expr) {
                        (true, None) => {}
                        (true, Some(e)) => {
                            let val = self.check_expr_infallible(e);
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
                            if !val.ty.is_assignable_to(ret_ty) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ReturnMismatch,
                                    e.span,
                                    format!(
                                        "return value has type '{}', expected '{}'",
                                        val.ty, ret_ty
                                    ),
                                ));
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
                    },
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
                            } else if ret_ty.contains_reference(&self.sema.structs)
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
                        (ResolvedErrorSpec::Untyped, None, Some(_)) => {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::ErrorSpecMismatch,
                                stmt.span,
                                "partial-progress error requires a typed 'error ErrType' signature",
                            ));
                        }
                    },
                }
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
                let then_div = self.check_nested_block(then_block);
                self.branch_depth -= 1;
                let then_vars = self.snapshot_outer_vars();
                let then_leases = self.leases.clone();

                self.scopes = base_vars.clone();
                self.leases = base_leases.clone();

                if let Some(eb) = else_block {
                    self.branch_depth += 1;
                    let else_div = self.check_nested_block(eb);
                    self.branch_depth -= 1;
                    let else_vars = self.snapshot_outer_vars();
                    let else_leases = self.leases.clone();

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
                    self.merge_branch_vars(&base_vars, &live_var_branches);
                    self.leases = base_leases;
                    self.leases.join_branches(&live_lease_branches);
                    then_div && else_div
                } else {
                    let mut live_var_branches = vec![base_vars.clone()];
                    let mut live_lease_branches = vec![base_leases.clone()];
                    if !then_div {
                        live_var_branches.push(then_vars);
                        live_lease_branches.push(then_leases);
                    }
                    self.merge_branch_vars(&base_vars, &live_var_branches);
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
                let _ = self.with_nested_scope(|this| this.check_stmt(inner));
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

                for arm in arms {
                    match &arm.pat {
                        SwitchPat::Wildcard(_) => {
                            has_wildcard = true;
                        }
                        SwitchPat::Literal(lit, pat_span) => {
                            let lit_ty = match lit {
                                Literal::Int(_) => Ty::IntLit,
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
                    }

                    self.scopes = base_vars.clone();
                    self.leases = base_leases.clone();
                    self.branch_depth += 1;
                    let arm_div = self.check_nested_block(&arm.body);
                    self.branch_depth -= 1;
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
                }

                self.scopes = base_vars.clone();
                self.merge_branch_vars(&base_vars, &live_var_branches);
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
                }
                true
            }

            StmtKind::Checkpoint => {
                self.record_direct_yield(stmt.span, FnEffect::SoftYielding, "'checkpoint()'");
                false
            }

            StmtKind::Block(b) => self.check_nested_block(b),

            StmtKind::Expr(e) => {
                let _ = self.check_expr_infallible(e);
                self.lookup_local("sys").is_none() && is_sys_exit_expr(e)
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
        self.loop_depth += 1;
        self.loop_entry_depths.push(entry_depth);
        self.loop_outer_reads.push(Vec::new());

        self.with_nested_scope(|this| {
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
                },
            );
            let _ = this.check_block_stmts(&body.stmts);
        });

        let outer_reads = self.loop_outer_reads.pop().unwrap_or_default();
        self.loop_entry_depths.pop();
        self.loop_depth -= 1;

        let modified = self.leases.modified_sources_since(&leases_before);
        if !modified.is_empty() {
            if let Some(isp) = iter_span {
                for tok in &var_prov.leases {
                    if let Some(inv_span) = modified.get(&tok.source) {
                        if leases_before.check_valid(tok).is_ok() {
                            let mut diag = Diagnostic::new(
                                DiagnosticKind::StaleDmaLease,
                                isp,
                                format!(
                                    "cannot iterate over slice derived from '{}.wait_chunk()' across a loop body that calls 'wait_chunk()'",
                                    tok.source
                                ),
                            );
                            if let Some(sp) = inv_span {
                                diag = diag.with_note(format!(
                                    "DMA lease invalidated inside 'for..in' body via 'wait_chunk()' at {}",
                                    sp
                                ));
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
                            let mut diag = Diagnostic::new(
                                DiagnosticKind::StaleDmaLease,
                                read_span,
                                format!(
                                    "cannot read outer variable '{}' derived from '{}.wait_chunk()' across loop iterations that call 'wait_chunk()'",
                                    r_name, tok.source
                                ),
                            );
                            if let Some(sp) = inv_span {
                                diag = diag.with_note(format!(
                                    "subsequent loop iteration invalidates lease via 'wait_chunk()' at {}",
                                    sp
                                ));
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
        self.loop_depth += 1;
        self.loop_entry_depths.push(entry_depth);
        self.loop_outer_reads.push(Vec::new());

        let _ = self.check_nested_block(body);

        let outer_reads = self.loop_outer_reads.pop().unwrap_or_default();
        self.loop_entry_depths.pop();
        self.loop_depth -= 1;

        let modified = self.leases.modified_sources_since(&leases_before);
        if !modified.is_empty() {
            // Check if any outer variable holding a pre-loop DMA lease on a source mutated inside
            // the loop was read in the loop (which would read a stale chunk on iteration 2).
            for (var_name, read_env, var_prov, read_span) in outer_reads {
                for tok in &var_prov.leases {
                    if let Some(inv_span) = modified.get(&tok.source) {
                        // Only report if `read_env` hadn't already reported it stale on iteration 1.
                        if read_env.check_valid(tok).is_ok() {
                            let mut diag = Diagnostic::new(
                                DiagnosticKind::StaleDmaLease,
                                read_span,
                                format!(
                                    "cannot read outer variable '{}' derived from '{}.wait_chunk()' across loop iterations that call 'wait_chunk()'",
                                    var_name, tok.source
                                ),
                            );
                            if let Some(sp) = inv_span {
                                diag = diag.with_note(format!(
                                    "subsequent loop iteration invalidates lease via 'wait_chunk()' at {}",
                                    sp
                                ));
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
                if !val.ty.is_assignable_to(&param.ty) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        arg.span,
                        format!(
                            "argument {} to 'spawn {}' has type '{}', expected '{}'",
                            i + 1,
                            sp.task.name,
                            val.ty,
                            param.ty
                        ),
                    ));
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
            let endpoint_from_param = self.resolve_param_endpoint_index(&val.endpoint_name);
            let static_endpoint = val
                .endpoint_name
                .filter(|ep| !ep.starts_with("param:"));
            arg_bindings.push(BootArgBinding {
                static_endpoint,
                endpoint_from_param,
                ref_param_origins,
            });
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

    fn check_assignment(&mut self, target: &Expr, op: AssignOp, rhs: ExprVal, stmt_span: Span) {
        match &target.kind {
            ExprKind::Ident(ident) => {
                if self.sema.consts.contains_key(&ident.name)
                    || self.sema.builtin_svd_consts.contains(&ident.name)
                {
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
                    self.check_compound_op_types(op, &cur.ty, &rhs.ty, stmt_span);
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

                if !rhs.ty.is_assignable_to(&var_info.ty) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        rhs.span,
                        format!(
                            "cannot assign '{}' to variable '{}' of type '{}'",
                            rhs.ty, ident.name, var_info.ty
                        ),
                    ));
                }

                if var_info.ty.contains_reference(&self.sema.structs)
                    || var_info.ty.contains_fn_ptr(&self.sema.structs)
                {
                    if var_info.decl_depth < rhs.prov.origin_depth {
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
                    if let Some(v) = self.lookup_local_mut(&ident.name) {
                        v.prov = rhs.prov;
                        v.field_prov.clear();
                        v.endpoint_alias = rhs.endpoint_name;
                    }
                }
            }

            ExprKind::Field {
                target: base,
                field,
            } => {
                // Check field write (`r.buf = half;` or `ptr_to_struct.field = val;`).
                let base_val = self.check_expr_infallible(base);
                let (struct_name, via_ptr) = match &base_val.ty {
                    Ty::Struct(s) => (s.clone(), false),
                    Ty::Ptr(inner) => match &**inner {
                        Ty::Struct(s) => (s.clone(), true),
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
                    Ty::HwPeripheral(_) => return,
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

                if op == AssignOp::Assign {
                    if !rhs.ty.is_assignable_to(&field_ty) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            rhs.span,
                            format!(
                                "cannot assign '{}' to field '{}.{}' of type '{}'",
                                rhs.ty, struct_name, field.name, field_ty
                            ),
                        ));
                    }
                } else {
                    self.check_compound_op_types(op, &field_ty, &rhs.ty, stmt_span);
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
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::ScopeDepthEscape,
                                stmt_span,
                                format!(
                                    "cannot store inner/leased reference of depth {} into struct field '{}' through outer pointer (depth {})",
                                    rhs.prov.origin_depth, field.name, dest_depth
                                ),
                            ));
                        }
                    } else if let Some(root_name) = root_ident_of(base) {
                        if let Some(root_var) = self.lookup_local(&root_name).cloned() {
                            if root_var.decl_depth < rhs.prov.origin_depth {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ScopeDepthEscape,
                                    stmt_span,
                                    format!(
                                        "cannot escape value of scope depth {} into field '{}' of outer struct '{}' declared at scope depth {}",
                                        rhs.prov.origin_depth, field.name, root_name, root_var.decl_depth
                                    ),
                                ));
                            }
                            if let Some(v) = self.lookup_local_mut(&root_name) {
                                v.field_prov.insert(field.name.clone(), rhs.prov.clone());
                                let mut combined = Provenance::pure_static();
                                for fp in v.field_prov.values() {
                                    combined = combined.merge(fp);
                                }
                                v.prov = combined;
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
                if op == AssignOp::Assign {
                    if !rhs.ty.is_assignable_to(pointee_ty) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            rhs.span,
                            format!(
                                "cannot assign '{}' to dereferenced '*{}'",
                                rhs.ty, pointee_ty
                            ),
                        ));
                    }
                } else {
                    self.check_compound_op_types(op, pointee_ty, &rhs.ty, stmt_span);
                }
                if pointee_ty.contains_reference(&self.sema.structs) {
                    let dest_depth = pv.prov.pointee_depth.unwrap_or(0);
                    if (pv.prov.pointee_is_caller_or_static && rhs.prov.has_non_returning_origin())
                        || dest_depth < rhs.prov.origin_depth
                    {
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

            ExprKind::Index {
                target: seq_expr,
                index,
            } => {
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
                    if !rhs.ty.is_assignable_to(&elem_ty) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            rhs.span,
                            format!(
                                "cannot assign '{}' to element of type '{}'",
                                rhs.ty, elem_ty
                            ),
                        ));
                    }
                } else {
                    self.check_compound_op_types(op, &elem_ty, &rhs.ty, stmt_span);
                }
                if elem_ty.contains_reference(&self.sema.structs) {
                    if let Some(root_name) = root_ident_of(seq_expr) {
                        if let Some(root_var) = self.lookup_local(&root_name) {
                            if root_var.decl_depth < rhs.prov.origin_depth {
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

            _ => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::ImmutableAssignment,
                    target.span,
                    "invalid assignment target",
                ));
            }
        }
    }

    fn check_compound_op_types(&mut self, op: AssignOp, lhs: &Ty, rhs: &Ty, span: Span) {
        if lhs.is_error() || rhs.is_error() {
            return;
        }
        if !lhs.is_integer() || !rhs.is_integer() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "compound assignment '{}' requires integer operands, found '{}' and '{}'",
                    op.as_str(),
                    lhs,
                    rhs
                ),
            ));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Expression Checking
    // ─────────────────────────────────────────────────────────────────────────

    fn check_expr_infallible(&mut self, expr: &Expr) -> ExprVal {
        let mut val = self.check_expr_raw(expr);
        if val.error_spec.is_fallible() {
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
        val
    }

    fn check_expr_raw(&mut self, expr: &Expr) -> ExprVal {
        match &expr.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Int(_) => ExprVal::scalar(Ty::IntLit, expr.span),
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
                    span: expr.span,
                }
            }

            ExprKind::Field { target, field } => self.check_field_expr(target, field, expr.span),

            ExprKind::Call { callee, args } => self.check_call_expr(callee, args, expr.span),

            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => self.check_method_call(receiver, method, args, expr.span),

            ExprKind::Index { target, index } => {
                let tv = self.check_expr_infallible(target);
                let iv = self.check_expr_infallible(index);
                if !iv.ty.is_integer() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        index.span,
                        format!("index must be an integer, found '{}'", iv.ty),
                    ));
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
                let tv = self.check_expr_infallible(target);
                match &tv.ty {
                    // Sub-slice of an existing slice `buf[0..5]` inherits `buf`'s exact scope depth and DMA lease!
                    Ty::Slice(elem) => ExprVal {
                        ty: Ty::Slice(elem.clone()),
                        error_spec: ResolvedErrorSpec::None,
                        prov: tv.prov,
                        endpoint_name: None,
                        span: expr.span,
                    },
                    // Slicing a fixed array `arr[0..5]` creates a slice bound to `arr`'s lexical scope depth!
                    Ty::Array { elem, .. } => {
                        let mut prov = tv.prov;
                        if let Some(root) = root_ident_of(target) {
                            if let Some(var) = self.lookup_local(&root) {
                                prov = prov.merge(&Provenance::addr_of_local(&var.name, var.decl_depth));
                            }
                        }
                        ExprVal {
                            ty: Ty::Slice(elem.clone()),
                            error_spec: ResolvedErrorSpec::None,
                            prov,
                            endpoint_name: None,
                            span: expr.span,
                        }
                    }
                    Ty::Ptr(inner) => match &**inner {
                        Ty::Array { elem, .. } => ExprVal {
                            ty: Ty::Slice(elem.clone()),
                            error_spec: ResolvedErrorSpec::None,
                            prov: tv.prov,
                            endpoint_name: None,
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

            ExprKind::StructLit { name, fields } => {
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

                let mut seen = BTreeSet::new();
                let mut combined_prov = Provenance::pure_static();
                let mut per_field_prov = BTreeMap::new();

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
                    if let Some(decl_field) = st_info.field(&fi.name.name) {
                        if !fv.ty.is_assignable_to(&decl_field.ty) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                fi.value.span,
                                format!(
                                    "field '{}.{}' expects '{}', found '{}'",
                                    name.name, fi.name.name, decl_field.ty, fv.ty
                                ),
                            ));
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
                    if ev.ty.contains_reference(&self.sema.structs) {
                        combined_prov = combined_prov.merge(&ev.prov);
                    }
                    match &elem_ty {
                        None => elem_ty = Some(ev.ty),
                        Some(cur) if matches!(cur, Ty::IntLit) && ev.ty.is_integer() => {
                            elem_ty = Some(ev.ty);
                        }
                        Some(cur) => {
                            if !ev.ty.is_assignable_to(cur) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    e.span,
                                    format!(
                                        "array element has type '{}', expected '{}'",
                                        ev.ty, cur
                                    ),
                                ));
                            }
                        }
                    }
                }
                let final_elem = match elem_ty.unwrap_or(Ty::Error) {
                    Ty::IntLit => Ty::u8(),
                    other => other,
                };
                ExprVal {
                    ty: Ty::Array {
                        elem: Box::new(final_elem),
                        len: elems.len(),
                    },
                    error_spec: ResolvedErrorSpec::None,
                    prov: combined_prov,
                    endpoint_name: None,
                    span: expr.span,
                }
            }

            ExprKind::Or {
                expr: inner,
                fallback,
            } => {
                let iv = self.check_expr_raw(inner);
                if !iv.error_spec.is_fallible() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOrFallback,
                        expr.span,
                        "cannot apply 'or' fallback to an infallible expression",
                    ));
                }
                let fv = self.check_expr_raw(fallback);
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
                let res_ty = if matches!(iv.ty, Ty::IntLit) {
                    fv.ty.clone()
                } else {
                    iv.ty.clone()
                };
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
            // Check for invalidated DMA lease on read!
            if let Some((stale_tok, inv_span)) = self.leases.find_stale_lease(&var.prov) {
                let mut diag = Diagnostic::new(
                    DiagnosticKind::StaleDmaLease,
                    ident.span,
                    format!(
                        "cannot read '{}' derived from '{}.wait_chunk()' after a subsequent 'wait_chunk()' invalidated its DMA buffer lease",
                        ident.name, stale_tok.source
                    ),
                );
                if let Some(sp) = inv_span {
                    diag = diag.with_note(format!(
                        "DMA lease acquired at {} was invalidated by subsequent `{}.wait_chunk()` at {}; finish decoding/parsing `{}` (e.g. `Type.from_bytes(slice)`) or copy needed bytes into a local `[N]u8` buffer before calling `wait_chunk()` again",
                        stale_tok.acquired_at, stale_tok.source, sp, ident.name
                    ));
                } else {
                    diag = diag.with_note(format!(
                        "finish reading/decoding `{}` or copy needed bytes out before calling `{}.wait_chunk()` again",
                        ident.name, stale_tok.source
                    ));
                }
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

            return ExprVal {
                ty: var.ty,
                error_spec: ResolvedErrorSpec::None,
                prov: var.prov,
                endpoint_name: var.endpoint_alias,
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
                span: ident.span,
            };
        }

        // 4. Immutable `const` global or built-in SVD constant.
        if let Some(c) = self.sema.consts.get(&ident.name).cloned() {
            return ExprVal::scalar(c.ty, ident.span);
        }
        if self.sema.builtin_svd_consts.contains(&ident.name) {
            return ExprVal::scalar(Ty::u32(), ident.span);
        }

        // 5. Unqualified enum variant (`Overcurrent`, `BadPacket`, `TooShort`, `BadCrc`, ...).
        if let Some(enum_names) = self.sema.variant_to_enums.get(&ident.name).cloned() {
            if enum_names.len() == 1 {
                return ExprVal::scalar(Ty::Enum(enum_names[0].clone()), ident.span);
            }
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                ident.span,
                format!(
                    "ambiguous enum variant '{}'; qualify with enum name (candidates: {})",
                    ident.name,
                    enum_names.join(", ")
                ),
            ));
            return ExprVal::error(ident.span);
        }

        // 6. Declared function (e.g. callback reference `register_cb(my_fn)`).
        if let Some(fn_info) = self.sema.functions.get(&ident.name).cloned() {
            let fn_ty = Ty::FnPtr {
                params: fn_info.params.iter().map(|p| p.ty.clone()).collect(),
                ret: Box::new(fn_info.ret_ty),
            };
            return ExprVal {
                ty: fn_ty,
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::fn_callback(ident.name.clone()),
                endpoint_name: None,
                span: ident.span,
            };
        }

        // 7. Built-in or imported namespaces (`sys`, `sim`, `hw`, `CmdFrame`, `ParseErr`).
        if ident.name == "sys" {
            return ExprVal::scalar(Ty::Namespace(NamespaceKind::Sys), ident.span);
        }
        if ident.name == "sim" {
            return ExprVal::scalar(Ty::Namespace(NamespaceKind::Sim), ident.span);
        }
        if let Some(chip_path) = self.sema.chip_imports.get(&ident.name).cloned() {
            return ExprVal::scalar(Ty::Namespace(NamespaceKind::Chip(chip_path)), ident.span);
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
                // `&fn_name` preserves callback provenance.
                if let ExprKind::Ident(id) = &inner.kind {
                    if let Some(fn_info) = self.sema.functions.get(&id.name).cloned() {
                        let fn_ty = Ty::FnPtr {
                            params: fn_info.params.iter().map(|p| p.ty.clone()).collect(),
                            ret: Box::new(fn_info.ret_ty),
                        };
                        return ExprVal {
                            ty: fn_ty,
                            error_spec: ResolvedErrorSpec::None,
                            prov: Provenance::fn_callback(id.name.clone()),
                            endpoint_name: None,
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

                let iv = self.check_expr_infallible(inner);
                if iv.ty.is_error() {
                    return ExprVal::error(span);
                }

                let mut prov = iv.prov.clone();
                let mut endpoint_name = iv.endpoint_name.clone();
                if let Some(root_name) = root_ident_of(inner) {
                    if let Some(local) = self.lookup_local(&root_name) {
                        prov = prov.merge(&Provenance::addr_of_local(&local.name, local.decl_depth));
                        if endpoint_name.is_none() {
                            endpoint_name = local.endpoint_alias.clone();
                        }
                    } else if self.sema.primitives.contains_key(&root_name) {
                        prov = Provenance::addr_of_static();
                        endpoint_name = Some(root_name);
                    }
                } else {
                    // Taking address of temporary expression at current lexical depth.
                    prov = prov.merge(&Provenance::addr_of_local("<temp>", self.current_depth()));
                }

                ExprVal {
                    ty: Ty::Ptr(Box::new(iv.ty)),
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name,
                    span,
                }
            }

            UnaryOp::Deref => {
                let iv = self.check_expr_infallible(inner);
                match iv.ty {
                    Ty::Ptr(pointee) => {
                        let prov = if pointee.contains_reference(&self.sema.structs) {
                            iv.prov
                        } else {
                            Provenance::pure_static()
                        };
                        ExprVal {
                            ty: *pointee,
                            error_spec: ResolvedErrorSpec::None,
                            prov,
                            endpoint_name: iv.endpoint_name,
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
                if !iv.ty.is_integer() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("unary '-' requires an integer, found '{}'", iv.ty),
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
        let lv = self.check_expr_infallible(lhs);
        let rv = self.check_expr_infallible(rhs);
        if lv.ty.is_error() || rv.ty.is_error() {
            return ExprVal::error(span);
        }

        match op {
            BinaryOp::And | BinaryOp::Or => {
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
                ExprVal::scalar(Ty::bool(), span)
            }

            BinaryOp::Eq | BinaryOp::Ne => {
                let compat = lv.ty.is_assignable_to(&rv.ty)
                    || rv.ty.is_assignable_to(&lv.ty)
                    || (lv.ty.is_integer() && rv.ty.is_integer());
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
                }
                ExprVal::scalar(Ty::bool(), span)
            }

            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                if !lv.ty.is_integer() || !rv.ty.is_integer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "comparison '{}' requires integer operands, found '{}' and '{}'",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                }
                ExprVal::scalar(Ty::bool(), span)
            }

            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Rem
            | BinaryOp::BitOr
            | BinaryOp::BitAnd
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr => {
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
        }
    }

    fn check_field_expr(&mut self, target: &Expr, field: &Ident, span: Span) -> ExprVal {
        let tv = self.check_expr_infallible(target);
        if tv.ty.is_error() {
            return ExprVal::error(span);
        }

        // Auto-dereference `*Struct` / `*Slice` on field access.
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
                            prov: tv.prov,
                            endpoint_name: None,
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
                    ExprVal {
                        ty: Ty::Ptr(elem),
                        error_spec: ResolvedErrorSpec::None,
                        prov: tv.prov,
                        endpoint_name: None,
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

                let prov = if f_info.ty.contains_reference(&self.sema.structs)
                    || f_info.ty.contains_fn_ptr(&self.sema.structs)
                {
                    if let Some(root) = root_ident_of(target) {
                        if let Some(v) = self.lookup_local(&root) {
                            v.field_prov
                                .get(&field.name)
                                .cloned()
                                .unwrap_or_else(|| tv.prov.clone())
                        } else {
                            tv.prov
                        }
                    } else {
                        tv.prov
                    }
                } else {
                    Provenance::pure_static()
                };

                ExprVal {
                    ty: f_info.ty,
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name: None,
                    span,
                }
            }

            Ty::Namespace(NamespaceKind::Chip(chip_name)) => {
                ExprVal::scalar(Ty::HwPeripheral(format!("{}.{}", chip_name, field.name)), span)
            }

            Ty::HwPeripheral(path) => {
                ExprVal::scalar(Ty::HwPeripheral(format!("{}.{}", path, field.name)), span)
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
                    if let Some(expected) = params.get(i) {
                        if !av.ty.is_assignable_to(expected) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                arg.span,
                                format!(
                                    "argument {} has type '{}', expected '{}'",
                                    i + 1,
                                    av.ty,
                                    expected
                                ),
                            ));
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
        for (i, arg) in args.iter().enumerate() {
            let av = self.check_expr_infallible(&arg.value);
            if let Some(param) = fn_info.params.get(i) {
                if !av.ty.is_assignable_to(&param.ty) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        arg.span,
                        format!(
                            "argument {} ('{}') to '{}' has type '{}', expected '{}'",
                            i + 1,
                            param.name,
                            fn_info.name,
                            av.ty,
                            param.ty
                        ),
                    ));
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
            let has_non_static_local_ref = av.prov.has_non_returning_origin();
            boot_call_args.push(BootCallArg {
                static_endpoint,
                endpoint_from_param,
                has_non_static_local_ref,
                ref_param_origins: av.prov.param_origins.clone(),
                span: arg.span,
            });
            arg_vals.push(av);
        }

        if !fn_info.is_extern_c {
            self.sema.recorded_call_args.push(RecordedCallArgs {
                caller: self.caller_key.clone(),
                callee: fn_info.name.clone(),
                args: boot_call_args.clone(),
            });
            if self.can_record_boot_step() {
                self.boot_steps.push(BootStep::HelperCall {
                    callee: fn_info.name.clone(),
                    conditional: self.branch_depth > 0,
                    args: boot_call_args,
                });
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
                                "'{}' expects 1 string argument (`*u8` or string literal), found {}",
                                name,
                                args.len()
                            ),
                        )
                        .with_note(format!(
                            "pass a string literal like `{}(\"boot ok\");` or use `sys.log_kv(\"val=\", x);` to print numeric values",
                            name
                        )),
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
                                format!(
                                    "'{}' expects a string (`*u8` or string literal), found '{}'",
                                    name, av.ty
                                ),
                            )
                            .with_note(format!(
                                "use `sys.log_kv(\"val=\", expr);` (signed) or `sys.log_hex(\"val=\", expr);` (hex) to print numbers, or pass a string literal to `{}(\"...\")`",
                                name
                            )),
                        );
                    }
                }
                Some(ExprVal::scalar(Ty::void(), span))
            }

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
        let rv = self.check_expr_infallible(receiver);
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
                    if struct_ty.contains_reference(&self.sema.structs) {
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

            Ty::HwPeripheral(_) | Ty::Namespace(NamespaceKind::Chip(_)) => {
                // Virtual CMSIS-SVD hardware register/peripheral operations are non-yielding (`noyield`).
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                let ret = if method.name.starts_with("read")
                    || method.name.starts_with("get")
                    || method.name.starts_with("is_")
                {
                    Ty::IntLit
                } else {
                    Ty::void()
                };
                ExprVal::scalar(ret, span)
            }

            Ty::OsPrimitive(os_prim) => {
                let source_key = rv
                    .endpoint_name
                    .clone()
                    .or_else(|| root_ident_of(receiver))
                    .unwrap_or_else(|| "<anon_endpoint>".to_string());
                self.check_os_primitive_method(&os_prim, &source_key, method, args, span)
            }

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

            "now_ticks" | "now_ms" => {
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
                    if !av.ty.is_assignable_to(&Ty::u32()) && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!("'sys.pend_irq' expects `u32` IRQ number, found '{}'", av.ty),
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
                    "now_ms",
                    "now_ticks",
                    "pend_irq",
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
                pos_vals.push((av, a.span));
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
                        span,
                    }
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::Inbox { elem, .. } => match m {
                "post" | "try_post" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "try_send" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    ExprVal::scalar(Ty::bool(), span)
                }
                "send" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
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
                        span,
                    }
                }
                "len" => {
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
                        span,
                    }
                }
                "active_buf" | "dma_buf" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::Slice(elem.clone()), span)
                }
                "half_len" | "cap" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::usize(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::Lease { elem, .. } => match m {
                "release" | "complete" | "flip" | "start" | "activate" | "stop" | "deactivate" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "wait_chunk" | "acquire" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.{}()", source_key, m),
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
                        span,
                    }
                }
                "try_acquire" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    let gen = self.leases.acquire_next(source_key, span);
                    let prov =
                        Provenance::dma_lease(source_key.to_string(), gen, self.current_depth(), span);
                    ExprVal {
                        ty: Ty::Slice(elem.clone()),
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov,
                        endpoint_name: None,
                        span,
                    }
                }
                _ => self.unknown_prim_method(prim, method, span),
            },
        }
    }

    fn expect_positional_arity(
        &mut self,
        method: &str,
        pos_vals: &[(ExprVal, Span)],
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
        pos_vals: &[(ExprVal, Span)],
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
        let (val, arg_span) = &pos_vals[0];
        if !val.ty.is_assignable_to(elem) {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                *arg_span,
                format!(
                    "argument to '{}' has type '{}', expected '{}'",
                    method, val.ty, elem
                ),
            ));
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
            if let Some((ename, ety, _)) = bound_err {
                this.scopes.last_mut().unwrap().insert(
                    ename.clone(),
                    LocalVar {
                        name: ename,
                        ty: ety,
                        decl_depth: depth,
                        prov: Provenance::pure_static(),
                        field_prov: BTreeMap::new(),
                        endpoint_alias: None,
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
                this.scopes.last_mut().unwrap().insert(
                    pname.clone(),
                    LocalVar {
                        name: pname,
                        ty: pty,
                        decl_depth: depth,
                        prov: iv.prov.clone(),
                        field_prov: BTreeMap::new(),
                        endpoint_alias: None,
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
                    if this.lookup_local("sys").is_none() && is_sys_exit_expr(last_expr) {
                        this.after_diverge = prev_after;
                        return (true, None);
                    }
                    this.after_diverge = prev_after;
                    return (div, Some(lv));
                }
                if this.check_stmt(&last[0]) {
                    div = true;
                }
                this.after_diverge = prev_after;
                (div, None)
            } else {
                (this.check_block_stmts(&clause.body.stmts), None)
            }
        });

        let catch_vars = self.snapshot_outer_vars();
        let catch_leases = self.leases.clone();

        let mut merged_prov = iv.prov.clone();
        if !catch_div {
            self.scopes = base_vars.clone();
            self.merge_branch_vars(&base_vars, &[base_vars.clone(), catch_vars]);
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
            span,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small Pure Helpers
// ─────────────────────────────────────────────────────────────────────────────

const VALID_SYS_METHODS_NOTE: &str = "valid portable `sys.*` methods: `sys.use_systick(u32 hz)`, `sys.now_ms() -> u32`, `sys.now_ticks() -> u32`, `sys.pend_irq(u32 irq)`, `sys.exit(i32 code)`, `sys.log(*u8 msg)`, `sys.log_kv(*u8 label, i32 val)`, `sys.log_hex(*u8 label, u32 val)`, `sys.delay(dur)`";

fn is_string_arg_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::Str)
        || matches!(ty, Ty::Ptr(inner) if matches!(**inner, Ty::Prim(PrimType::U8 | PrimType::I8)))
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
        } => matches!(&receiver.kind, ExprKind::Ident(id) if id.name == "sys") && method.name == "exit",
        ExprKind::Call { callee, .. } => match &callee.kind {
            ExprKind::Field { target, field } => {
                matches!(&target.kind, ExprKind::Ident(id) if id.name == "sys")
                    && field.name == "exit"
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
