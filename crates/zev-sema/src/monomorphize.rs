//! Compile-time `<T>` and `<T, usize N>` monomorphizer and constant `usize` evaluator.
//!
//! Lowers generic `struct` and `fn` templates into concrete top-level declarations before
//! whole-program semantic verification and MIR lowering:
//! - Deterministic collision-free mangled names: `<orig>__<arg1>__<arg2>` (`_P` ptr, `_S` slice,
//!   `_A<N>_` array, `_D` path, `_G..._L_..._R_` nested generic, `_F<arity>_..._T_...` fn ptr, `_0` escaped `_`).
//! - Attaches `@monomorphized("<human_signature>")` and `@instantiated_from("<origin>")` to synthesized `Item.attrs`.
//! - Evaluates all `ArrayLen::Expr(...)` bounds to concrete `ArrayLen::Fixed(n)`.
//! - Strips generic template declarations (`!generic_params.is_empty()`) from the output AST.
//! - Reports `E0029` (`GenericInstantiationError`) on wrong arity, type-vs-const mismatch,
//!   uninferable type/const parameters, non-constant `usize` arguments, recursive generic
//!   expansion (`depth > 32`), or total monomorphized instances exceeding `1024`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use zev_ast::{
    ArrayLen, Attr, AttrKind, BinaryOp, Block, CallArg, CatchClause, CatchParam, ConstDecl,
    ErrorSpec, Expr, ExprKind, FieldInit, FnDecl, FormatPart, GenericArg, GenericParam,
    GenericParamKind, GlobalVarDecl, Ident, IrqDecl, IsrDecl, Item, ItemKind, Literal,
    MethodDecl, MmioDecl, MmioDeclKind, MmioRegDecl, OsPrimitiveDecl, OsPrimitiveType, Param,
    PrimType, Program, SelfReceiver, SpawnStmt, Span, Stmt, StmtKind, StructDecl, StructField,
    SwitchArm, SwitchPat, TaskDecl, TypeExpr, TypeKind, UnaryOp, WaitArm, WaitTimerArm,
};

use crate::diag::{Diagnostic, DiagnosticKind};
use crate::model::{MonomorphizedInstance, MonomorphizedKind};

/// Maximum nested generic instantiation chain depth before rejecting infinite generic recursion (`E0029`).
pub const MAX_GENERIC_DEPTH: usize = 32;

/// Maximum total number of unique monomorphized `struct` + `fn` instantiations per compilation (`E0029`).
pub const MAX_MONOMORPHIZED_INSTANCES: usize = 1024;

/// Output of monomorphizing one or more parsed [`Program`]s.
#[derive(Clone, Debug)]
pub struct MonomorphizeOutput {
    pub programs: Vec<Program>,
    pub instances: Vec<MonomorphizedInstance>,
    pub diagnostics: Vec<Diagnostic>,
}

impl MonomorphizeOutput {
    pub fn is_ok(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn into_result(
        self,
    ) -> Result<(Vec<Program>, Vec<MonomorphizedInstance>), Vec<Diagnostic>> {
        if self.diagnostics.is_empty() {
            Ok((self.programs, self.instances))
        } else {
            Err(self.diagnostics)
        }
    }
}

/// Monomorphize a slice of [`Program`]s without filenames, returning concrete [`Program`]s or diagnostics.
pub fn monomorphize_programs(programs: &[Program]) -> Result<Vec<Program>, Vec<Diagnostic>> {
    let named: Vec<(&Program, Option<&str>)> = programs.iter().map(|p| (p, None)).collect();
    let out = monomorphize_named_programs(&named);
    if out.diagnostics.is_empty() {
        Ok(out.programs)
    } else {
        Err(out.diagnostics)
    }
}

/// Monomorphize named `(Program, Option<filename>)` pairs, preserving 1:1 `Program` slots and diagnostic filenames.
pub fn monomorphize_named_programs<'a>(
    programs: &[(&'a Program, Option<&'a str>)],
) -> MonomorphizeOutput {
    monomorphize_named_programs_with_non_root_count(programs, 0)
}

pub(crate) fn monomorphize_named_programs_with_non_root_count<'a>(
    programs: &[(&'a Program, Option<&'a str>)],
    non_root_prog_count: usize,
) -> MonomorphizeOutput {
    Monomorphizer::new(programs, non_root_prog_count).run()
}

#[derive(Clone, Debug)]
struct TemplateEntry<D> {
    attrs: Vec<Attr>,
    decl: D,
    prog_idx: usize,
    file: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ResolvedGenArg {
    Type(TypeExpr),
    Const(usize),
}

#[derive(Clone, Debug, Default)]
struct SubstEnv {
    types: BTreeMap<String, TypeExpr>,
    consts: BTreeMap<String, usize>,
    depth: usize,
    prog_idx: usize,
    file: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum InferTy {
    Concrete(TypeExpr),
    IntLit,
    FloatLit,
    Unknown,
}

#[derive(Clone, Debug)]
struct PendingFnBody {
    prog_idx: usize,
    item_attrs: Vec<Attr>,
    fn_attrs: Vec<Attr>,
    is_extern_c: bool,
    mangled: String,
    name_span: Span,
    human_sig: String,
    instantiated_origin: String,
    params: Vec<Param>,
    is_variadic: bool,
    ret_type: TypeExpr,
    error_spec: Option<ErrorSpec>,
    raw_body: Option<Block>,
    fn_span: Span,
    child_env: SubstEnv,
}

#[derive(Clone, Debug)]
struct ConcreteMethodInfo {
    fn_name: String,
    receiver: Option<SelfReceiver>,
    ret_ty: TypeExpr,
    is_pub: bool,
    owner_scope: Option<usize>,
}

#[derive(Clone, Debug)]
struct GenericImplMethodEntry {
    impl_generic_params: Vec<GenericParam>,
    target_generic_args: Vec<GenericArg>,
    method: MethodDecl,
    prog_idx: usize,
    owner_scope: Option<usize>,
    file: Option<String>,
    pattern_key: String,
}

struct Monomorphizer<'a> {
    inputs: &'a [(&'a Program, Option<&'a str>)],
    raw_consts: BTreeMap<String, (Expr, Option<String>)>,
    eval_consts: BTreeMap<String, usize>,
    const_types: BTreeMap<String, TypeExpr>,
    global_var_types: BTreeMap<String, TypeExpr>,
    known_structs: BTreeMap<String, StructDecl>,
    known_enums: BTreeMap<String, Vec<String>>,
    enum_prog_idx: BTreeMap<String, Option<usize>>,
    variant_to_enums: BTreeMap<String, Vec<String>>,
    non_root_prog_count: usize,
    prim_types: BTreeMap<String, OsPrimitiveType>,
    mmio_layouts: BTreeMap<String, BTreeMap<String, TypeExpr>>,
    mmio_instances: BTreeMap<String, TypeExpr>,
    non_generic_fn_rets: BTreeMap<String, TypeExpr>,
    struct_templates: BTreeMap<String, TemplateEntry<StructDecl>>,
    fn_templates: BTreeMap<String, TemplateEntry<FnDecl>>,
    concrete_methods: BTreeMap<(String, String), ConcreteMethodInfo>,
    generic_impl_methods: BTreeMap<String, Vec<GenericImplMethodEntry>>,
    /// Mangled struct name -> human signature (`"FixedVec__u32__16" -> "FixedVec<u32, 16>"`).
    human_type_names: BTreeMap<String, String>,
    /// Mangled struct name -> `(template_name, resolved_args)` for call-site generic inference (`*FixedVec<T, N>`).
    instantiated_struct_args: BTreeMap<String, (String, Vec<ResolvedGenArg>)>,
    instantiated_structs: BTreeSet<String>,
    /// Mangled fn name -> concrete return `TypeExpr`.
    instantiated_fns: BTreeMap<String, TypeExpr>,
    cap_exceeded_emitted: bool,
    synth_structs_by_prog: Vec<Vec<Item>>,
    synth_fns_by_prog: Vec<Vec<Item>>,
    pending_fn_bodies: VecDeque<PendingFnBody>,
    instances: BTreeMap<(MonomorphizedKind, String), MonomorphizedInstance>,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> Monomorphizer<'a> {
    fn new(inputs: &'a [(&'a Program, Option<&'a str>)], non_root_prog_count: usize) -> Self {
        let n = inputs.len().max(1);
        let mut known_enums = BTreeMap::new();
        known_enums.insert("FmtErr".to_string(), vec!["Truncated".to_string()]);
        let mut enum_prog_idx = BTreeMap::new();
        enum_prog_idx.insert("FmtErr".to_string(), None);
        Self {
            inputs,
            raw_consts: BTreeMap::new(),
            eval_consts: BTreeMap::new(),
            const_types: BTreeMap::new(),
            global_var_types: BTreeMap::new(),
            known_structs: BTreeMap::new(),
            known_enums,
            enum_prog_idx,
            variant_to_enums: BTreeMap::new(),
            non_root_prog_count,
            prim_types: BTreeMap::new(),
            mmio_layouts: BTreeMap::new(),
            mmio_instances: BTreeMap::new(),
            non_generic_fn_rets: BTreeMap::new(),
            struct_templates: BTreeMap::new(),
            fn_templates: BTreeMap::new(),
            concrete_methods: BTreeMap::new(),
            generic_impl_methods: BTreeMap::new(),
            human_type_names: BTreeMap::new(),
            instantiated_struct_args: BTreeMap::new(),
            instantiated_structs: BTreeSet::new(),
            instantiated_fns: BTreeMap::new(),
            cap_exceeded_emitted: false,
            synth_structs_by_prog: vec![Vec::new(); n],
            synth_fns_by_prog: vec![Vec::new(); n],
            pending_fn_bodies: VecDeque::new(),
            instances: BTreeMap::new(),
            diagnostics: Vec::new(),
        }
    }

    fn emit(&mut self, file: &Option<String>, mut diag: Diagnostic) {
        if diag.file.is_none() {
            diag.file = file.clone();
        }
        if diag.note.is_none() {
            diag.note = Some(diag.kind.doc_hint().to_string());
        }
        self.diagnostics.push(diag);
    }

    fn run(mut self) -> MonomorphizeOutput {
        // Pass 1: Collect module consts, enums, non-generic signatures, and generic templates.
        for (prog_idx, (prog, file_opt)) in self.inputs.iter().enumerate() {
            let file = file_opt.map(str::to_string);
            for item in &prog.items {
                match &item.kind {
                    ItemKind::Const(c) => {
                        self.raw_consts
                            .insert(c.name.name.clone(), (c.value.clone(), file.clone()));
                        self.const_types.insert(c.name.name.clone(), c.ty.clone());
                    }
                    ItemKind::Enum(en) => {
                        let scope_idx = if prog_idx < self.non_root_prog_count {
                            Some(prog_idx)
                        } else {
                            None
                        };
                        self.enum_prog_idx.insert(en.name.name.clone(), scope_idx);
                        let mut vars = Vec::new();
                        for v in &en.variants {
                            vars.push(v.name.name.clone());
                            self.variant_to_enums
                                .entry(v.name.name.clone())
                                .or_default()
                                .push(en.name.name.clone());
                        }
                        self.known_enums.insert(en.name.name.clone(), vars);
                    }
                    ItemKind::Struct(st) => {
                        if st.generic_params.is_empty() {
                            self.known_structs.insert(st.name.name.clone(), st.clone());
                        } else {
                            self.validate_generic_params(&st.name.name, &st.generic_params, &file);
                            if self.struct_templates.contains_key(&st.name.name)
                                || self.known_structs.contains_key(&st.name.name)
                            {
                                self.emit(
                                    &file,
                                    Diagnostic::new(
                                        DiagnosticKind::DuplicateDefinition,
                                        st.name.span,
                                        format!("duplicate type definition '{}'", st.name.name),
                                    ),
                                );
                            } else {
                                self.struct_templates.insert(
                                    st.name.name.clone(),
                                    TemplateEntry {
                                        attrs: item.attrs.clone(),
                                        decl: st.clone(),
                                        prog_idx,
                                        file: file.clone(),
                                    },
                                );
                            }
                        }
                    }
                    ItemKind::Fn(f) => {
                        if f.generic_params.is_empty() {
                            self.non_generic_fn_rets
                                .insert(f.name.name.clone(), f.ret_type.clone());
                        } else {
                            self.validate_generic_params(&f.name.name, &f.generic_params, &file);
                            if f.name.name == "main" {
                                self.emit(
                                    &file,
                                    Diagnostic::new(
                                        DiagnosticKind::GenericInstantiationError,
                                        f.name.span,
                                        "entry point 'main' cannot declare generic parameters",
                                    ),
                                );
                            }
                            if f.is_extern_c {
                                self.emit(
                                    &file,
                                    Diagnostic::new(
                                        DiagnosticKind::GenericInstantiationError,
                                        f.name.span,
                                        format!(
                                            "'extern \"C\"' function '{}' cannot declare generic parameters",
                                            f.name.name
                                        ),
                                    ),
                                );
                            }
                            if self.fn_templates.contains_key(&f.name.name)
                                || self.non_generic_fn_rets.contains_key(&f.name.name)
                            {
                                self.emit(
                                    &file,
                                    Diagnostic::new(
                                        DiagnosticKind::DuplicateDefinition,
                                        f.name.span,
                                        format!("duplicate top-level symbol '{}'", f.name.name),
                                    ),
                                );
                            } else {
                                self.fn_templates.insert(
                                    f.name.name.clone(),
                                    TemplateEntry {
                                        attrs: item.attrs.clone(),
                                        decl: f.clone(),
                                        prog_idx,
                                        file: file.clone(),
                                    },
                                );
                            }
                        }
                    }
                    ItemKind::OsPrimitive(p) => {
                        self.prim_types.insert(p.name.name.clone(), p.ty.clone());
                    }
                    ItemKind::Mmio(mm) => match &mm.kind {
                        MmioDeclKind::Layout { regs } => {
                            let fields = regs
                                .iter()
                                .map(|r| (r.name.name.clone(), r.ty.clone()))
                                .collect();
                            self.mmio_layouts.insert(mm.name.name.clone(), fields);
                        }
                        MmioDeclKind::Instance { layout, .. } => {
                            self.mmio_instances.insert(
                                mm.name.name.clone(),
                                TypeExpr::new(TypeKind::Ptr(Box::new(layout.clone())), mm.span),
                            );
                        }
                        MmioDeclKind::Inline { regs, .. } => {
                            let fields = regs
                                .iter()
                                .map(|r| (r.name.name.clone(), r.ty.clone()))
                                .collect();
                            self.mmio_layouts.insert(mm.name.name.clone(), fields);
                            self.mmio_instances.insert(
                                mm.name.name.clone(),
                                TypeExpr::new(
                                    TypeKind::Ptr(Box::new(TypeExpr::new(
                                        TypeKind::Named(mm.name.clone()),
                                        mm.span,
                                    ))),
                                    mm.span,
                                ),
                            );
                        }
                    },
                    _ => {}
                }
            }
        }

        if !self.variant_to_enums.contains_key("Truncated") {
            self.variant_to_enums
                .insert("Truncated".to_string(), vec!["FmtErr".to_string()]);
        }

        // Evaluate top-level integer constants for `usize N` & `[CAP]T` resolution.
        let const_names: Vec<String> = self.raw_consts.keys().cloned().collect();
        for cname in const_names {
            let mut visiting = BTreeSet::new();
            if let Some(val) = self.eval_named_const(&cname, &mut visiting) {
                if let Ok(u) = usize::try_from(val) {
                    self.eval_consts.insert(cname, u);
                }
            }
        }

        // Pass 1.5: Rewrite non-generic struct field types, `const` types, OS primitive types, and
        // non-generic `fn` return types ONCE with each item's exact `file` in `SubstEnv`, caching by
        // `(prog_idx, item_idx)` so Pass 2 never re-emits duplicate `E0029`s or `<unknown>` files.
        let mut rewritten_structs: BTreeMap<(usize, usize), StructDecl> = BTreeMap::new();
        let mut rewritten_const_tys: BTreeMap<(usize, usize), TypeExpr> = BTreeMap::new();
        let mut rewritten_global_tys: BTreeMap<(usize, usize), TypeExpr> = BTreeMap::new();
        let mut rewritten_prim_tys: BTreeMap<(usize, usize), OsPrimitiveType> = BTreeMap::new();
        let mut rewritten_fn_rets: BTreeMap<(usize, usize), TypeExpr> = BTreeMap::new();

        for (prog_idx, (prog, file_opt)) in self.inputs.iter().enumerate() {
            let env = SubstEnv {
                types: BTreeMap::new(),
                consts: BTreeMap::new(),
                depth: 0,
                prog_idx,
                file: file_opt.map(str::to_string),
            };
            for (item_idx, item) in prog.items.iter().enumerate() {
                match &item.kind {
                    ItemKind::Struct(st) if st.generic_params.is_empty() => {
                        let fields = st
                            .fields
                            .iter()
                            .map(|f| StructField {
                                attrs: f.attrs.clone(),
                                ty: self.rewrite_type_expr(&f.ty, &env),
                                name: f.name.clone(),
                                span: f.span,
                            })
                            .collect();
                        let r_st = StructDecl {
                            fields,
                            ..st.clone()
                        };
                        self.known_structs
                            .insert(st.name.name.clone(), r_st.clone());
                        rewritten_structs.insert((prog_idx, item_idx), r_st);
                    }
                    ItemKind::Const(c) => {
                        let rt = self.rewrite_type_expr(&c.ty, &env);
                        self.const_types.insert(c.name.name.clone(), rt.clone());
                        rewritten_const_tys.insert((prog_idx, item_idx), rt);
                    }
                    ItemKind::GlobalVar(g) => {
                        let rt = self.rewrite_type_expr(&g.ty, &env);
                        self.global_var_types
                            .insert(g.name.name.clone(), rt.clone());
                        rewritten_global_tys.insert((prog_idx, item_idx), rt);
                    }
                    ItemKind::OsPrimitive(p) => {
                        let ros = self.rewrite_os_prim_type(&p.ty, &env);
                        self.prim_types.insert(p.name.name.clone(), ros.clone());
                        rewritten_prim_tys.insert((prog_idx, item_idx), ros);
                    }
                    ItemKind::Fn(f) if f.generic_params.is_empty() => {
                        let rrt = self.rewrite_type_expr(&f.ret_type, &env);
                        self.non_generic_fn_rets
                            .insert(f.name.name.clone(), rrt.clone());
                        rewritten_fn_rets.insert((prog_idx, item_idx), rrt);
                    }
                    _ => {}
                }
            }
        }

        // Pass 1.6: Collect concrete and generic `impl` method signatures before Pass 2 body rewriting.
        let mut concrete_impl_fns_by_prog: Vec<Vec<(Span, FnDecl, TypeExpr)>> =
            vec![Vec::new(); self.inputs.len()];
        for (prog_idx, (prog, file_opt)) in self.inputs.iter().enumerate() {
            let file = file_opt.map(str::to_string);
            let owner_scope = if prog_idx < self.non_root_prog_count {
                Some(prog_idx)
            } else {
                None
            };
            let env = SubstEnv {
                types: BTreeMap::new(),
                consts: BTreeMap::new(),
                depth: 0,
                prog_idx,
                file: file.clone(),
            };
            for item in &prog.items {
                let ItemKind::Impl(imp) = &item.kind else {
                    continue;
                };
                if imp.generic_params.is_empty() && imp.target_generic_args.is_empty() {
                    if !self.known_structs.contains_key(&imp.target_name.name) {
                        let diag = if self.struct_templates.contains_key(&imp.target_name.name) {
                            Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                imp.target_name.span,
                                format!(
                                    "missing generic arguments `<...>` for `impl` on generic struct '{}'",
                                    imp.target_name.name
                                ),
                            )
                        } else {
                            Diagnostic::new(
                                DiagnosticKind::UndefinedSymbol,
                                imp.target_name.span,
                                format!(
                                    "cannot find struct '{}' for `impl` block",
                                    imp.target_name.name
                                ),
                            )
                        };
                        self.emit(&file, diag);
                        continue;
                    }
                    for m in &imp.methods {
                        let key = (imp.target_name.name.clone(), m.func.name.name.clone());
                        if self.concrete_methods.contains_key(&key) {
                            self.emit(
                                &file,
                                Diagnostic::new(
                                    DiagnosticKind::DuplicateDefinition,
                                    m.func.name.span,
                                    format!(
                                        "duplicate method '{}' in `impl {}`",
                                        m.func.name.name, imp.target_name.name
                                    ),
                                ),
                            );
                            continue;
                        }
                        let fn_name = format!("{}__{}", imp.target_name.name, m.func.name.name);
                        if self.non_generic_fn_rets.contains_key(&fn_name)
                            || self.fn_templates.contains_key(&fn_name)
                        {
                            self.emit(
                                &file,
                                Diagnostic::new(
                                    DiagnosticKind::DuplicateDefinition,
                                    m.func.name.span,
                                    format!(
                                        "method '{}' in `impl {}` collides with top-level function '{}'",
                                        m.func.name.name, imp.target_name.name, fn_name
                                    ),
                                ),
                            );
                            continue;
                        }
                        let ret_ty = self.rewrite_type_expr(&m.func.ret_type, &env);
                        let mut params = Vec::with_capacity(
                            m.func.params.len() + usize::from(m.receiver.is_some()),
                        );
                        if let Some(recv) = m.receiver {
                            let target_ty = TypeExpr::new(
                                TypeKind::Named(imp.target_name.clone()),
                                imp.target_name.span,
                            );
                            let self_ty = match recv {
                                SelfReceiver::Value => target_ty,
                                SelfReceiver::Ptr => {
                                    TypeExpr::new(TypeKind::Ptr(Box::new(target_ty)), m.span)
                                }
                            };
                            params.push(Param {
                                ty: self_ty,
                                name: Ident::new("self".to_string(), m.span),
                                span: m.span,
                            });
                        }
                        params.extend(m.func.params.iter().cloned());
                        let synth_fn = FnDecl {
                            attrs: m.func.attrs.clone(),
                            is_extern_c: false,
                            name: Ident::new(fn_name.clone(), m.func.name.span),
                            generic_params: Vec::new(),
                            params,
                            is_variadic: false,
                            ret_type: m.func.ret_type.clone(),
                            error_spec: m.func.error_spec.clone(),
                            body: m.func.body.clone(),
                            span: m.func.span,
                        };
                        self.concrete_methods.insert(
                            key,
                            ConcreteMethodInfo {
                                fn_name: fn_name.clone(),
                                receiver: m.receiver,
                                ret_ty: ret_ty.clone(),
                                is_pub: m.is_pub,
                                owner_scope,
                            },
                        );
                        self.non_generic_fn_rets.insert(fn_name, ret_ty.clone());
                        concrete_impl_fns_by_prog[prog_idx].push((m.span, synth_fn, ret_ty));
                    }
                } else {
                    self.validate_generic_params(&imp.target_name.name, &imp.generic_params, &file);
                    let Some(tmpl) = self.struct_templates.get(&imp.target_name.name).cloned() else {
                        let msg = if self.known_structs.contains_key(&imp.target_name.name) {
                            format!(
                                "struct '{}' is not generic (expected 0 target generic arguments)",
                                imp.target_name.name
                            )
                        } else {
                            format!("unknown generic struct '{}' in `impl`", imp.target_name.name)
                        };
                        self.emit(
                            &file,
                            Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                imp.target_name.span,
                                msg,
                            ),
                        );
                        continue;
                    };
                    if imp.target_generic_args.len() != tmpl.decl.generic_params.len() {
                        self.emit(
                            &file,
                            Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                imp.target_name.span,
                                format!(
                                    "`impl {}` expects {} target generic argument(s), found {}",
                                    imp.target_name.name,
                                    tmpl.decl.generic_params.len(),
                                    imp.target_generic_args.len()
                                ),
                            ),
                        );
                        continue;
                    }
                    let pattern_key = self.compute_impl_pattern_key(
                        &imp.generic_params,
                        &imp.target_generic_args,
                        &tmpl.decl.generic_params,
                        &env,
                    );
                    for m in &imp.methods {
                        let dup = self
                            .generic_impl_methods
                            .get(&imp.target_name.name)
                            .is_some_and(|entries| {
                                entries.iter().any(|e| {
                                    e.method.func.name.name == m.func.name.name
                                        && e.pattern_key == pattern_key
                                })
                            });
                        if dup {
                            self.emit(
                                &file,
                                Diagnostic::new(
                                    DiagnosticKind::DuplicateDefinition,
                                    m.func.name.span,
                                    format!(
                                        "duplicate method '{}' for identical specialization in `impl {}`",
                                        m.func.name.name, imp.target_name.name
                                    ),
                                ),
                            );
                            continue;
                        }
                        self.generic_impl_methods
                            .entry(imp.target_name.name.clone())
                            .or_default()
                            .push(GenericImplMethodEntry {
                                impl_generic_params: imp.generic_params.clone(),
                                target_generic_args: imp.target_generic_args.clone(),
                                method: m.clone(),
                                prog_idx,
                                owner_scope,
                                file: file.clone(),
                                pattern_key: pattern_key.clone(),
                            });
                    }
                }
            }
        }

        // Pass 2: Rewrite all concrete items in each input `Program` (stripping generic templates).
        let mut rewritten_items_by_prog: Vec<Vec<Item>> = Vec::with_capacity(self.inputs.len());
        for (prog_idx, (prog, file_opt)) in self.inputs.iter().enumerate() {
            let env = SubstEnv {
                types: BTreeMap::new(),
                consts: BTreeMap::new(),
                depth: 0,
                prog_idx,
                file: file_opt.map(str::to_string),
            };
            let mut out_items = Vec::new();
            for (item_idx, item) in prog.items.iter().enumerate() {
                match &item.kind {
                    ItemKind::Import(_) | ItemKind::Enum(_) => {
                        out_items.push(item.clone());
                    }
                    ItemKind::Impl(_) => {
                        // Desugared into concrete `FnDecl` items below or instantiated lazily on demand.
                        continue;
                    }
                    ItemKind::Mmio(mm) => {
                        let mut scopes = vec![BTreeMap::new()];
                        let new_kind = match &mm.kind {
                            MmioDeclKind::Layout { regs } => MmioDeclKind::Layout {
                                regs: regs
                                    .iter()
                                    .map(|r| MmioRegDecl {
                                        access: r.access,
                                        ty: self.rewrite_type_expr(&r.ty, &env),
                                        name: r.name.clone(),
                                        offset: self
                                            .rewrite_expr(&r.offset, &env, &mut scopes, None)
                                            .0,
                                        span: r.span,
                                    })
                                    .collect(),
                            },
                            MmioDeclKind::Instance { layout, base_addr } => {
                                MmioDeclKind::Instance {
                                    layout: self.rewrite_type_expr(layout, &env),
                                    base_addr: self
                                        .rewrite_expr(base_addr, &env, &mut scopes, None)
                                        .0,
                                }
                            }
                            MmioDeclKind::Inline { base_addr, regs } => MmioDeclKind::Inline {
                                base_addr: self.rewrite_expr(base_addr, &env, &mut scopes, None).0,
                                regs: regs
                                    .iter()
                                    .map(|r| MmioRegDecl {
                                        access: r.access,
                                        ty: self.rewrite_type_expr(&r.ty, &env),
                                        name: r.name.clone(),
                                        offset: self
                                            .rewrite_expr(&r.offset, &env, &mut scopes, None)
                                            .0,
                                        span: r.span,
                                    })
                                    .collect(),
                            },
                        };
                        out_items.push(Item {
                            attrs: item.attrs.clone(),
                            is_pub: item.is_pub,
                            kind: ItemKind::Mmio(MmioDecl {
                                name: mm.name.clone(),
                                kind: new_kind,
                                span: mm.span,
                            }),
                            span: item.span,
                        });
                    }
                    ItemKind::Irq(iq) => {
                        let mut scopes = vec![BTreeMap::new()];
                        let number = self.rewrite_expr(&iq.number, &env, &mut scopes, None).0;
                        out_items.push(Item {
                            attrs: item.attrs.clone(),
                            is_pub: item.is_pub,
                            kind: ItemKind::Irq(IrqDecl {
                                name: iq.name.clone(),
                                number,
                                span: iq.span,
                            }),
                            span: item.span,
                        });
                    }
                    ItemKind::Struct(st) => {
                        if !st.generic_params.is_empty() {
                            // Strip template struct declaration; concrete instantiations are emitted separately.
                            continue;
                        }
                        let rewritten = rewritten_structs
                            .remove(&(prog_idx, item_idx))
                            .unwrap_or_else(|| st.clone());
                        out_items.push(Item {
                            attrs: item.attrs.clone(),
                            is_pub: item.is_pub,
                            kind: ItemKind::Struct(rewritten),
                            span: item.span,
                        });
                    }
                    ItemKind::Const(c) => {
                        let ty = rewritten_const_tys
                            .remove(&(prog_idx, item_idx))
                            .unwrap_or_else(|| c.ty.clone());
                        let mut scopes = vec![BTreeMap::new()];
                        let value = self.rewrite_expr(&c.value, &env, &mut scopes, Some(&ty)).0;
                        out_items.push(Item {
                            attrs: item.attrs.clone(),
                            is_pub: item.is_pub,
                            kind: ItemKind::Const(ConstDecl {
                                ty,
                                name: c.name.clone(),
                                value,
                                span: c.span,
                            }),
                            span: item.span,
                        });
                    }
                    ItemKind::OsPrimitive(p) => {
                        let ty = rewritten_prim_tys
                            .remove(&(prog_idx, item_idx))
                            .unwrap_or_else(|| p.ty.clone());
                        let mut scopes = vec![BTreeMap::new()];
                        let init = p
                            .init
                            .as_ref()
                            .map(|e| self.rewrite_expr(e, &env, &mut scopes, None).0);
                        out_items.push(Item {
                            attrs: item.attrs.clone(),
                            is_pub: item.is_pub,
                            kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                                attrs: p.attrs.clone(),
                                ty,
                                name: p.name.clone(),
                                init,
                                span: p.span,
                            }),
                            span: item.span,
                        });
                    }
                    ItemKind::GlobalVar(g) => {
                        let ty = rewritten_global_tys
                            .remove(&(prog_idx, item_idx))
                            .unwrap_or_else(|| self.rewrite_type_expr(&g.ty, &env));
                        let mut scopes = vec![BTreeMap::new()];
                        let init = g
                            .init
                            .as_ref()
                            .map(|e| self.rewrite_expr(e, &env, &mut scopes, Some(&ty)).0);
                        out_items.push(Item {
                            attrs: item.attrs.clone(),
                            is_pub: item.is_pub,
                            kind: ItemKind::GlobalVar(GlobalVarDecl {
                                ty,
                                name: g.name.clone(),
                                init,
                                span: g.span,
                            }),
                            span: item.span,
                        });
                    }
                    ItemKind::Fn(f) => {
                        if !f.generic_params.is_empty() {
                            // Strip template fn declaration.
                            continue;
                        }
                        let pre_ret = rewritten_fn_rets.remove(&(prog_idx, item_idx));
                        let rewritten_fn = self.rewrite_fn_decl(f, pre_ret, &env);
                        out_items.push(Item {
                            attrs: item.attrs.clone(),
                            is_pub: item.is_pub,
                            kind: ItemKind::Fn(rewritten_fn),
                            span: item.span,
                        });
                    }
                    ItemKind::Task(t) => {
                        let params: Vec<Param> = t
                            .params
                            .iter()
                            .map(|p| Param {
                                ty: self.rewrite_type_expr(&p.ty, &env),
                                name: p.name.clone(),
                                span: p.span,
                            })
                            .collect();
                        let mut scopes = vec![BTreeMap::new()];
                        for p in &params {
                            scopes[0].insert(p.name.name.clone(), p.ty.clone());
                        }
                        let body = self.rewrite_block(&t.body, &env, &mut scopes);
                        out_items.push(Item {
                            attrs: item.attrs.clone(),
                            is_pub: item.is_pub,
                            kind: ItemKind::Task(TaskDecl {
                                attrs: t.attrs.clone(),
                                name: t.name.clone(),
                                params,
                                body,
                                span: t.span,
                            }),
                            span: item.span,
                        });
                    }
                    ItemKind::Isr(i) => {
                        let params: Vec<Param> = i
                            .params
                            .iter()
                            .map(|p| Param {
                                ty: self.rewrite_type_expr(&p.ty, &env),
                                name: p.name.clone(),
                                span: p.span,
                            })
                            .collect();
                        let mut scopes = vec![BTreeMap::new()];
                        for p in &params {
                            scopes[0].insert(p.name.name.clone(), p.ty.clone());
                        }
                        let body = self.rewrite_block(&i.body, &env, &mut scopes);
                        out_items.push(Item {
                            attrs: item.attrs.clone(),
                            is_pub: item.is_pub,
                            kind: ItemKind::Isr(IsrDecl {
                                attrs: i.attrs.clone(),
                                vector: i.vector.clone(),
                                params,
                                body,
                                span: i.span,
                            }),
                            span: item.span,
                        });
                    }
                }
            }
            for (m_span, m_fn, m_ret) in std::mem::take(&mut concrete_impl_fns_by_prog[prog_idx]) {
                let rewritten_fn = self.rewrite_fn_decl(&m_fn, Some(m_ret), &env);
                out_items.push(Item {
                    attrs: rewritten_fn.attrs.clone(),
                    is_pub: false,
                    kind: ItemKind::Fn(rewritten_fn),
                    span: m_span,
                });
            }
            rewritten_items_by_prog.push(out_items);
        }

        // Iteratively drain queued generic function bodies in O(1) stack depth.
        while let Some(job) = self.pending_fn_bodies.pop_front() {
            let body = job.raw_body.as_ref().map(|b| {
                let mut scopes = vec![BTreeMap::new()];
                for p in &job.params {
                    scopes[0].insert(p.name.name.clone(), p.ty.clone());
                }
                self.rewrite_block(b, &job.child_env, &mut scopes)
            });

            let synth_fn = FnDecl {
                attrs: job.fn_attrs,
                is_extern_c: job.is_extern_c,
                name: Ident::new(job.mangled, job.name_span),
                generic_params: Vec::new(),
                params: job.params,
                is_variadic: job.is_variadic,
                ret_type: job.ret_type,
                error_spec: job.error_spec,
                body,
                span: job.fn_span,
            };

            let mut attrs = job.item_attrs;
            attrs.push(Attr {
                kind: AttrKind::Custom {
                    name: "monomorphized".to_string(),
                    arg: Some(job.human_sig),
                },
                span: job.fn_span,
            });
            attrs.push(Attr {
                kind: AttrKind::Custom {
                    name: "instantiated_from".to_string(),
                    arg: Some(job.instantiated_origin),
                },
                span: job.fn_span,
            });
            self.synth_fns_by_prog[job.prog_idx].push(Item {
                attrs,
                is_pub: false,
                kind: ItemKind::Fn(synth_fn),
                span: job.fn_span,
            });
        }

        // Assemble final concrete `Program`s: place Imports/Enums/Structs + synthesized Structs first
        // so concrete struct layouts precede OS primitives and functions cleanly, followed by remaining items + synthesized Fns.
        let mut out_programs = Vec::with_capacity(self.inputs.len());
        for (idx, (orig_prog, _)) in self.inputs.iter().enumerate() {
            let mut combined = Vec::new();
            let base_items = std::mem::take(&mut rewritten_items_by_prog[idx]);
            let synth_structs = std::mem::take(&mut self.synth_structs_by_prog[idx]);
            let synth_fns = std::mem::take(&mut self.synth_fns_by_prog[idx]);

            let mut inserted_structs = false;
            for it in base_items {
                let is_early = matches!(
                    it.kind,
                    ItemKind::Import(_) | ItemKind::Struct(_) | ItemKind::Enum(_) | ItemKind::Const(_)
                );
                if !is_early && !inserted_structs {
                    inserted_structs = true;
                    combined.extend(synth_structs.clone());
                }
                combined.push(it);
            }
            if !inserted_structs {
                combined.extend(synth_structs);
            }
            combined.extend(synth_fns);
            out_programs.push(Program::new(combined, orig_prog.span));
        }

        let mut instances: Vec<MonomorphizedInstance> = self.instances.into_values().collect();
        instances.sort();
        instances.dedup();

        MonomorphizeOutput {
            programs: out_programs,
            instances,
            diagnostics: self.diagnostics,
        }
    }

    fn validate_generic_params(
        &mut self,
        owner: &str,
        params: &[GenericParam],
        file: &Option<String>,
    ) {
        let mut seen = BTreeSet::new();
        for gp in params {
            if !seen.insert(gp.name.name.clone()) {
                self.emit(
                    file,
                    Diagnostic::new(
                        DiagnosticKind::GenericInstantiationError,
                        gp.span,
                        format!(
                            "duplicate generic parameter '{}' on '{}'",
                            gp.name.name, owner
                        ),
                    ),
                );
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Constant `usize` Evaluation
    // ─────────────────────────────────────────────────────────────────────────

    fn eval_named_const(&self, name: &str, visiting: &mut BTreeSet<String>) -> Option<i128> {
        if let Some(&v) = self.eval_consts.get(name) {
            return Some(v as i128);
        }
        let (expr, _) = self.raw_consts.get(name)?;
        if !visiting.insert(name.to_string()) {
            return None;
        }
        let raw = self.eval_const_expr_i128(expr, &BTreeMap::new(), visiting);
        visiting.remove(name);
        let val = raw?;
        if let Some(TypeExpr {
            kind: TypeKind::Prim(p),
            ..
        }) = self.const_types.get(name)
        {
            if p.is_integer() {
                return Some(truncate_int_prim(val, *p));
            }
        }
        Some(val)
    }

    fn infer_const_prim_type(
        &self,
        expr: &Expr,
        local_consts: &BTreeMap<String, usize>,
    ) -> Option<PrimType> {
        match &expr.kind {
            ExprKind::Ident(id) => {
                if local_consts.contains_key(&id.name) {
                    Some(PrimType::Usize)
                } else if let Some(TypeExpr {
                    kind: TypeKind::Prim(p),
                    ..
                }) = self.const_types.get(&id.name)
                {
                    p.is_integer().then_some(*p)
                } else {
                    None
                }
            }
            ExprKind::Cast { ty, .. } => match &ty.kind {
                TypeKind::Prim(p) if p.is_integer() => Some(*p),
                _ => None,
            },
            ExprKind::Unary { expr: inner, .. } => self.infer_const_prim_type(inner, local_consts),
            ExprKind::Binary { lhs, rhs, .. } => self
                .infer_const_prim_type(lhs, local_consts)
                .or_else(|| self.infer_const_prim_type(rhs, local_consts)),
            _ => None,
        }
    }

    fn eval_const_expr_i128(
        &self,
        expr: &Expr,
        local_consts: &BTreeMap<String, usize>,
        visiting: &mut BTreeSet<String>,
    ) -> Option<i128> {
        match &expr.kind {
            ExprKind::Literal(Literal::Int { value, .. }) => Some(*value as i128),
            ExprKind::Ident(id) => {
                if let Some(&v) = local_consts.get(&id.name) {
                    Some(v as i128)
                } else {
                    self.eval_named_const(&id.name, visiting)
                }
            }
            ExprKind::Cast { expr: inner, ty } => {
                let v = self.eval_const_expr_i128(inner, local_consts, visiting)?;
                let target_prim = match &ty.kind {
                    TypeKind::Prim(p) if p.is_integer() => Some(*p),
                    TypeKind::Named(id) => PrimType::from_name(&id.name).filter(|p| p.is_integer()),
                    _ => None,
                }?;
                Some(truncate_int_prim(v, target_prim))
            }
            ExprKind::Unary { op, expr: inner } => {
                let v = self.eval_const_expr_i128(inner, local_consts, visiting)?;
                match op {
                    UnaryOp::Neg => {
                        let neg = v.checked_neg()?;
                        if let Some(p) = self.infer_const_prim_type(inner, local_consts) {
                            if p.is_signed() {
                                return Some(truncate_int_prim(neg, p));
                            }
                        }
                        Some(neg)
                    }
                    UnaryOp::BitNot => {
                        let inv = !v;
                        if let Some(p) = self.infer_const_prim_type(inner, local_consts) {
                            Some(truncate_int_prim(inv, p))
                        } else {
                            Some(inv)
                        }
                    }
                    _ => None,
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.eval_const_expr_i128(lhs, local_consts, visiting)?;
                let r = self.eval_const_expr_i128(rhs, local_consts, visiting)?;
                match op {
                    BinaryOp::Add => l.checked_add(r),
                    BinaryOp::Sub => l.checked_sub(r),
                    BinaryOp::Mul => l.checked_mul(r),
                    BinaryOp::Div => {
                        if r == 0 {
                            None
                        } else {
                            l.checked_div(r)
                        }
                    }
                    BinaryOp::Rem => {
                        if r == 0 {
                            None
                        } else {
                            l.checked_rem(r)
                        }
                    }
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

    fn eval_usize_in_env(&self, expr: &Expr, env: &SubstEnv) -> Option<usize> {
        let mut visiting = BTreeSet::new();
        let val = self.eval_const_expr_i128(expr, &env.consts, &mut visiting)?;
        usize::try_from(val).ok()
    }

    fn is_known_type_ident(&self, name: &str, env: &SubstEnv) -> bool {
        PrimType::from_name(name).is_some()
            || env.types.contains_key(name)
            || self.known_structs.contains_key(name)
            || self.struct_templates.contains_key(name)
            || self.known_enums.contains_key(name)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Type Substitution, ArrayLen Evaluation & Struct Monomorphization
    // ─────────────────────────────────────────────────────────────────────────

    fn resolve_explicit_generic_args(
        &mut self,
        target_name: &str,
        gparams: &[GenericParam],
        gargs: &[GenericArg],
        env: &SubstEnv,
        span: Span,
    ) -> Option<Vec<ResolvedGenArg>> {
        const PAREN_CMP_NOTE: &str = "if `<` and `>` were intended as comparisons rather than generic arguments, parenthesize them: e.g. `f((a < b), (c > (d)))`";
        if gargs.len() != gparams.len() {
            let mut diag = Diagnostic::new(
                DiagnosticKind::GenericInstantiationError,
                span,
                format!(
                    "'{}' expects {} generic argument(s), found {}",
                    target_name,
                    gparams.len(),
                    gargs.len()
                ),
            );
            if gargs.len() >= 2 {
                diag = diag.with_note(PAREN_CMP_NOTE);
            }
            self.emit(&env.file, diag);
            return None;
        }

        let mut out = Vec::with_capacity(gparams.len());
        for (gp, ga) in gparams.iter().zip(gargs.iter()) {
            match (gp.kind, ga) {
                (GenericParamKind::Type, GenericArg::Type(te)) => {
                    if let TypeKind::Named(id) = &te.kind {
                        if env.consts.contains_key(&id.name)
                            || (self.raw_consts.contains_key(&id.name)
                                && !self.is_known_type_ident(&id.name, env))
                        {
                            self.emit(
                                &env.file,
                                Diagnostic::new(
                                    DiagnosticKind::GenericInstantiationError,
                                    te.span,
                                    format!(
                                        "expected type argument for generic parameter '{}' of '{}', found constant '{}'",
                                        gp.name.name, target_name, id.name
                                    ),
                                )
                                .with_note(PAREN_CMP_NOTE),
                            );
                            return None;
                        }
                    }
                    let resolved_te = self.rewrite_type_expr(te, env);
                    if resolved_te.is_void() {
                        self.emit(
                            &env.file,
                            Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                te.span,
                                format!(
                                    "generic type argument '{}' for '{}' cannot be 'void'",
                                    gp.name.name, target_name
                                ),
                            ),
                        );
                        return None;
                    }
                    out.push(ResolvedGenArg::Type(resolved_te));
                }
                (GenericParamKind::Type, GenericArg::Const(expr)) => {
                    self.emit(
                        &env.file,
                        Diagnostic::new(
                            DiagnosticKind::GenericInstantiationError,
                            expr.span,
                            format!(
                                "expected type argument for generic parameter '{}' of '{}', found constant expression",
                                gp.name.name, target_name
                            ),
                        )
                        .with_note(PAREN_CMP_NOTE),
                    );
                    return None;
                }
                (GenericParamKind::ConstUsize, GenericArg::Const(expr)) => {
                    if let Some(n) = self.eval_usize_in_env(expr, env) {
                        out.push(ResolvedGenArg::Const(n));
                    } else {
                        self.emit(
                            &env.file,
                            Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                expr.span,
                                format!(
                                    "generic parameter 'usize {}' of '{}' requires a non-negative compile-time constant expression",
                                    gp.name.name, target_name
                                ),
                            )
                            .with_note(PAREN_CMP_NOTE),
                        );
                        return None;
                    }
                }
                // Per `#104`: bare identifiers like `N` or `CAP` inside `<T, N>` parse as `GenericArg::Type(Named(ident))`.
                (GenericParamKind::ConstUsize, GenericArg::Type(te)) => match &te.kind {
                    TypeKind::Named(id) => {
                        let synth_expr = Expr::new(ExprKind::Ident(id.clone()), id.span);
                        if let Some(n) = self.eval_usize_in_env(&synth_expr, env) {
                            out.push(ResolvedGenArg::Const(n));
                        } else if self.is_known_type_ident(&id.name, env) {
                            self.emit(
                                &env.file,
                                Diagnostic::new(
                                    DiagnosticKind::GenericInstantiationError,
                                    te.span,
                                    format!(
                                        "expected constant 'usize' argument for generic parameter '{}' of '{}', found type '{}'",
                                        gp.name.name, target_name, id.name
                                    ),
                                ),
                            );
                            return None;
                        } else {
                            self.emit(
                                &env.file,
                                Diagnostic::new(
                                    DiagnosticKind::GenericInstantiationError,
                                    te.span,
                                    format!(
                                        "non-constant or unknown 'usize' generic argument '{}' for parameter '{}' of '{}'",
                                        id.name, gp.name.name, target_name
                                    ),
                                )
                                .with_note(PAREN_CMP_NOTE),
                            );
                            return None;
                        }
                    }
                    _ => {
                        self.emit(
                            &env.file,
                            Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                te.span,
                                format!(
                                    "expected constant 'usize' argument for generic parameter '{}' of '{}', found type",
                                    gp.name.name, target_name
                                ),
                            ),
                        );
                        return None;
                    }
                },
            }
        }
        Some(out)
    }

    fn rewrite_type_expr(&mut self, te: &TypeExpr, env: &SubstEnv) -> TypeExpr {
        let kind = match &te.kind {
            TypeKind::Prim(p) => TypeKind::Prim(*p),
            TypeKind::Named(id) => {
                if let Some(sub) = env.types.get(&id.name) {
                    return TypeExpr::new(sub.kind.clone(), te.span);
                }
                if let Some(tmpl) = self.struct_templates.get(&id.name).cloned() {
                    self.emit(
                        &env.file,
                        Diagnostic::new(
                            DiagnosticKind::GenericInstantiationError,
                            te.span,
                            format!(
                                "generic struct '{}' requires {} generic argument(s)",
                                id.name,
                                tmpl.decl.generic_params.len()
                            ),
                        ),
                    );
                }
                TypeKind::Named(id.clone())
            }
            TypeKind::Path(segs) => TypeKind::Path(segs.clone()),
            TypeKind::Generic { base, args } => {
                if let Some(tmpl) = self.struct_templates.get(&base.name).cloned() {
                    if let Some(resolved_args) = self.resolve_explicit_generic_args(
                        &base.name,
                        &tmpl.decl.generic_params,
                        args,
                        env,
                        te.span,
                    ) {
                        if let Some(mangled) =
                            self.instantiate_struct(&base.name, &resolved_args, te.span, env)
                        {
                            return TypeExpr::new(
                                TypeKind::Named(Ident::new(mangled, base.span)),
                                te.span,
                            );
                        }
                    }
                    return TypeExpr::new(TypeKind::Prim(PrimType::U32), te.span);
                }
                self.emit(
                    &env.file,
                    Diagnostic::new(
                        DiagnosticKind::GenericInstantiationError,
                        te.span,
                        if self.known_structs.contains_key(&base.name)
                            || self.known_enums.contains_key(&base.name)
                            || PrimType::from_name(&base.name).is_some()
                        {
                            format!(
                                "type '{}' is not generic (expected 0 generic arguments, found {})",
                                base.name,
                                args.len()
                            )
                        } else {
                            format!("unknown generic type '{}'", base.name)
                        },
                    ),
                );
                TypeKind::Named(base.clone())
            }
            TypeKind::GenericPath { path, args } => {
                if let Some(last) = path.last() {
                    if let Some(tmpl) = self.struct_templates.get(&last.name).cloned() {
                        if let Some(resolved_args) = self.resolve_explicit_generic_args(
                            &last.name,
                            &tmpl.decl.generic_params,
                            args,
                            env,
                            te.span,
                        ) {
                            if let Some(mangled) =
                                self.instantiate_struct(&last.name, &resolved_args, te.span, env)
                            {
                                return TypeExpr::new(
                                    TypeKind::Named(Ident::new(mangled, last.span)),
                                    te.span,
                                );
                            }
                        }
                        return TypeExpr::new(TypeKind::Prim(PrimType::U32), te.span);
                    }
                }
                let joined = path
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                self.emit(
                    &env.file,
                    Diagnostic::new(
                        DiagnosticKind::GenericInstantiationError,
                        te.span,
                        format!("unknown generic type '{}'", joined),
                    ),
                );
                TypeKind::Path(path.clone())
            }
            TypeKind::Slice(inner) => TypeKind::Slice(Box::new(self.rewrite_type_expr(inner, env))),
            TypeKind::Array { elem, len } => {
                let new_elem = Box::new(self.rewrite_type_expr(elem, env));
                let fixed_len = match len {
                    ArrayLen::Fixed(n) => *n,
                    ArrayLen::Expr(len_expr) => {
                        if let Some(n) = self.eval_usize_in_env(len_expr, env) {
                            n
                        } else {
                            self.emit(
                                &env.file,
                                Diagnostic::new(
                                    DiagnosticKind::GenericInstantiationError,
                                    len_expr.span,
                                    "array length must be a compile-time constant 'usize' expression",
                                ),
                            );
                            1
                        }
                    }
                };
                TypeKind::Array {
                    elem: new_elem,
                    len: ArrayLen::Fixed(fixed_len),
                }
            }
            TypeKind::Ptr(inner) => TypeKind::Ptr(Box::new(self.rewrite_type_expr(inner, env))),
            TypeKind::OsPrimitive(os) => TypeKind::OsPrimitive(self.rewrite_os_prim_type(os, env)),
            TypeKind::Lease(inner) => TypeKind::Lease(Box::new(self.rewrite_type_expr(inner, env))),
            TypeKind::FnPtr { params, ret } => TypeKind::FnPtr {
                params: params.iter().map(|p| self.rewrite_type_expr(p, env)).collect(),
                ret: Box::new(self.rewrite_type_expr(ret, env)),
            },
        };
        TypeExpr::new(kind, te.span)
    }

    fn rewrite_os_prim_type(&mut self, os: &OsPrimitiveType, env: &SubstEnv) -> OsPrimitiveType {
        match os {
            OsPrimitiveType::Signal => OsPrimitiveType::Signal,
            OsPrimitiveType::WaitQueue => OsPrimitiveType::WaitQueue,
            OsPrimitiveType::Inbox { elem, cap } => OsPrimitiveType::Inbox {
                elem: Box::new(self.rewrite_type_expr(elem, env)),
                cap: *cap,
            },
            OsPrimitiveType::InboxUnsized { elem } => OsPrimitiveType::InboxUnsized {
                elem: Box::new(self.rewrite_type_expr(elem, env)),
            },
            OsPrimitiveType::Watch { elem } => OsPrimitiveType::Watch {
                elem: Box::new(self.rewrite_type_expr(elem, env)),
            },
            OsPrimitiveType::Guarded { elem } => OsPrimitiveType::Guarded {
                elem: Box::new(self.rewrite_type_expr(elem, env)),
            },
            OsPrimitiveType::PingPong { elem, cap } => OsPrimitiveType::PingPong {
                elem: Box::new(self.rewrite_type_expr(elem, env)),
                cap: *cap,
            },
            OsPrimitiveType::Pool { elem, cap } => OsPrimitiveType::Pool {
                elem: Box::new(self.rewrite_type_expr(elem, env)),
                cap: *cap,
            },
            OsPrimitiveType::DmaSlot { elem } => OsPrimitiveType::DmaSlot {
                elem: Box::new(self.rewrite_type_expr(elem, env)),
            },
            OsPrimitiveType::ByteRing { cap } => OsPrimitiveType::ByteRing { cap: *cap },
            OsPrimitiveType::ByteRingUnsized => OsPrimitiveType::ByteRingUnsized,
        }
    }

    fn format_mangled_arg(&self, arg: &ResolvedGenArg) -> String {
        match arg {
            ResolvedGenArg::Type(te) => self.format_mangled_type(te),
            ResolvedGenArg::Const(n) => n.to_string(),
        }
    }

    fn format_mangled_type(&self, te: &TypeExpr) -> String {
        match &te.kind {
            TypeKind::Named(id) => {
                if let Some((orig_base, r_args)) = self.instantiated_struct_args.get(&id.name) {
                    let base_esc = orig_base
                        .split('.')
                        .map(escape_mangle_ident)
                        .collect::<Vec<_>>()
                        .join("_D");
                    let arg_strs: Vec<String> =
                        r_args.iter().map(|a| self.format_mangled_arg(a)).collect();
                    return format!("_G{}_L_{}_R_", base_esc, arg_strs.join("__"));
                }
                id.name
                    .split('.')
                    .map(escape_mangle_ident)
                    .collect::<Vec<_>>()
                    .join("_D")
            }
            TypeKind::Ptr(inner) => format!("_P{}", self.format_mangled_type(inner)),
            TypeKind::Lease(inner) => {
                format!("_GLease_L_{}_R_", self.format_mangled_type(inner))
            }
            TypeKind::Slice(inner) => format!("_S{}", self.format_mangled_type(inner)),
            TypeKind::Array { elem, len } => {
                let n = len.as_fixed().unwrap_or(0);
                format!("_A{}_{}", n, self.format_mangled_type(elem))
            }
            TypeKind::OsPrimitive(os) => match os {
                OsPrimitiveType::Signal => "signal".to_string(),
                OsPrimitiveType::WaitQueue => "wait_queue".to_string(),
                OsPrimitiveType::Watch { elem } => {
                    format!("_Gwatch_L_{}_R_", self.format_mangled_type(elem))
                }
                OsPrimitiveType::Guarded { elem } => {
                    format!("_Gguarded_L_{}_R_", self.format_mangled_type(elem))
                }
                OsPrimitiveType::Inbox { elem, cap } => {
                    format!("_Ginbox_L_{}__{}_R_", self.format_mangled_type(elem), cap)
                }
                OsPrimitiveType::InboxUnsized { elem } => {
                    format!("_Ginbox_L_{}_R_", self.format_mangled_type(elem))
                }
                OsPrimitiveType::PingPong { elem, cap } => {
                    format!(
                        "_Gping_0pong_L_{}__{}_R_",
                        self.format_mangled_type(elem),
                        cap
                    )
                }
                OsPrimitiveType::Pool { elem, cap } => {
                    format!("_Gpool_L_{}__{}_R_", self.format_mangled_type(elem), cap)
                }
                OsPrimitiveType::DmaSlot { elem } => {
                    format!("_Gdma_0slot_L_{}_R_", self.format_mangled_type(elem))
                }
                OsPrimitiveType::ByteRing { cap } => {
                    format!("_Gbyte_0ring_L_{}_R_", cap)
                }
                OsPrimitiveType::ByteRingUnsized => "byte_ring".to_string(),
            },
            TypeKind::FnPtr { params, ret } => {
                let ps: Vec<String> = params.iter().map(|p| self.format_mangled_type(p)).collect();
                format!(
                    "_F{}_{}_T_{}",
                    params.len(),
                    ps.join("__"),
                    self.format_mangled_type(ret)
                )
            }
            _ => mangle_type_expr(te),
        }
    }

    fn format_origin(human_sig: &str, file: &Option<String>, span: Span) -> String {
        match file {
            Some(f) => format!("{} at {}:{}:{}", human_sig, f, span.line, span.col),
            None => format!("{} at {}:{}", human_sig, span.line, span.col),
        }
    }

    fn format_human_arg(&self, arg: &ResolvedGenArg) -> String {
        match arg {
            ResolvedGenArg::Type(te) => self.format_human_type(te),
            ResolvedGenArg::Const(n) => n.to_string(),
        }
    }

    fn format_human_type(&self, te: &TypeExpr) -> String {
        match &te.kind {
            TypeKind::Prim(p) => p.as_str().to_string(),
            TypeKind::Named(id) => self
                .human_type_names
                .get(&id.name)
                .cloned()
                .unwrap_or_else(|| id.name.clone()),
            TypeKind::Path(segs) => segs
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
                .join("."),
            TypeKind::Slice(inner) => format!("[]{}", self.format_human_type(inner)),
            TypeKind::Array { elem, len } => {
                let n = len.as_fixed().unwrap_or(0);
                format!("[{}]{}", n, self.format_human_type(elem))
            }
            TypeKind::Ptr(inner) => format!("*{}", self.format_human_type(inner)),
            TypeKind::Lease(inner) => format!("Lease<{}>", self.format_human_type(inner)),
            TypeKind::OsPrimitive(os) => match os {
                OsPrimitiveType::Signal => "signal".to_string(),
                OsPrimitiveType::WaitQueue => "wait_queue".to_string(),
                OsPrimitiveType::Watch { elem } => {
                    format!("watch<{}>", self.format_human_type(elem))
                }
                OsPrimitiveType::Guarded { elem } => {
                    format!("guarded<{}>", self.format_human_type(elem))
                }
                OsPrimitiveType::Inbox { elem, cap } => {
                    format!("inbox<{}, {}>", self.format_human_type(elem), cap)
                }
                OsPrimitiveType::InboxUnsized { elem } => {
                    format!("inbox<{}>", self.format_human_type(elem))
                }
                OsPrimitiveType::PingPong { elem, cap } => {
                    format!("ping_pong<{}, {}>", self.format_human_type(elem), cap)
                }
                OsPrimitiveType::Pool { elem, cap } => {
                    format!("pool<{}, {}>", self.format_human_type(elem), cap)
                }
                OsPrimitiveType::DmaSlot { elem } => {
                    format!("dma_slot<{}>", self.format_human_type(elem))
                }
                OsPrimitiveType::ByteRing { cap } => {
                    format!("byte_ring<{}>", cap)
                }
                OsPrimitiveType::ByteRingUnsized => "byte_ring".to_string(),
            },
            TypeKind::FnPtr { params, ret } => {
                let ps: Vec<String> = params.iter().map(|p| self.format_human_type(p)).collect();
                format!("fn({}) -> {}", ps.join(", "), self.format_human_type(ret))
            }
            TypeKind::Generic { base, .. } => base.name.clone(),
            TypeKind::GenericPath { path, .. } => path
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
                .join("."),
        }
    }

    fn build_names(
        &self,
        orig_name: &str,
        args: &[ResolvedGenArg],
    ) -> (String, String) {
        let mangled_parts: Vec<String> = args.iter().map(|a| self.format_mangled_arg(a)).collect();
        let human_parts: Vec<String> = args.iter().map(|a| self.format_human_arg(a)).collect();
        let mangled = format!("{}__{}", orig_name, mangled_parts.join("__"));
        let human = format!("{}<{}>", orig_name, human_parts.join(", "));
        (mangled, human)
    }

    fn check_total_instance_cap(
        &mut self,
        human_sig: &str,
        span: Span,
        caller_file: &Option<String>,
    ) -> bool {
        if self.instantiated_structs.len() + self.instantiated_fns.len()
            >= MAX_MONOMORPHIZED_INSTANCES
        {
            if !self.cap_exceeded_emitted {
                self.cap_exceeded_emitted = true;
                self.emit(
                    caller_file,
                    Diagnostic::new(
                        DiagnosticKind::GenericInstantiationError,
                        span,
                        format!(
                            "total monomorphized generic instantiations exceeded maximum limit of {} while instantiating '{}'",
                            MAX_MONOMORPHIZED_INSTANCES, human_sig
                        ),
                    ),
                );
            }
            return false;
        }
        true
    }

    fn instantiate_struct(
        &mut self,
        orig_name: &str,
        resolved_args: &[ResolvedGenArg],
        use_span: Span,
        caller_env: &SubstEnv,
    ) -> Option<String> {
        let (mangled, human_sig) = self.build_names(orig_name, resolved_args);
        if caller_env.depth > MAX_GENERIC_DEPTH {
            self.emit(
                &caller_env.file,
                Diagnostic::new(
                    DiagnosticKind::GenericInstantiationError,
                    use_span,
                    format!(
                        "recursive generic instantiation exceeded maximum depth of {} while instantiating '{}'",
                        MAX_GENERIC_DEPTH, human_sig
                    ),
                ),
            );
            return None;
        }

        if self.instantiated_structs.contains(&mangled) {
            return Some(mangled);
        }
        if !self.check_total_instance_cap(&human_sig, use_span, &caller_env.file) {
            return None;
        }

        self.human_type_names
            .insert(mangled.clone(), human_sig.clone());
        self.instantiated_struct_args.insert(
            mangled.clone(),
            (orig_name.to_string(), resolved_args.to_vec()),
        );
        self.instances.insert(
            (MonomorphizedKind::Struct, mangled.clone()),
            MonomorphizedInstance {
                kind: MonomorphizedKind::Struct,
                signature: human_sig.clone(),
                mangled_name: mangled.clone(),
            },
        );
        self.instantiated_structs.insert(mangled.clone());

        let tmpl = self.struct_templates.get(orig_name)?.clone();
        let instantiated_origin = Self::format_origin(&human_sig, &caller_env.file, use_span);
        let mut child_env = SubstEnv {
            types: BTreeMap::new(),
            consts: BTreeMap::new(),
            depth: caller_env.depth + 1,
            prog_idx: tmpl.prog_idx,
            file: tmpl.file.clone(),
        };
        for (gp, ga) in tmpl.decl.generic_params.iter().zip(resolved_args.iter()) {
            match ga {
                ResolvedGenArg::Type(te) => {
                    child_env.types.insert(gp.name.name.clone(), te.clone());
                }
                ResolvedGenArg::Const(n) => {
                    child_env.consts.insert(gp.name.name.clone(), *n);
                }
            }
        }

        let mut fields = Vec::with_capacity(tmpl.decl.fields.len());
        for f in &tmpl.decl.fields {
            fields.push(StructField {
                attrs: f.attrs.clone(),
                ty: self.rewrite_type_expr(&f.ty, &child_env),
                name: f.name.clone(),
                span: f.span,
            });
        }

        let synth_decl = StructDecl {
            is_packed: tmpl.decl.is_packed,
            name: Ident::new(mangled.clone(), tmpl.decl.name.span),
            generic_params: Vec::new(),
            fields,
            span: tmpl.decl.span,
        };
        self.known_structs
            .insert(mangled.clone(), synth_decl.clone());

        let mut attrs = tmpl.attrs.clone();
        attrs.push(Attr {
            kind: AttrKind::Custom {
                name: "monomorphized".to_string(),
                arg: Some(human_sig),
            },
            span: tmpl.decl.span,
        });
        attrs.push(Attr {
            kind: AttrKind::Custom {
                name: "instantiated_from".to_string(),
                arg: Some(instantiated_origin),
            },
            span: tmpl.decl.span,
        });
        self.synth_structs_by_prog[tmpl.prog_idx].push(Item {
            attrs,
            is_pub: false,
            kind: ItemKind::Struct(synth_decl),
            span: tmpl.decl.span,
        });

        Some(mangled)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Function Monomorphization & Body Rewriting
    // ─────────────────────────────────────────────────────────────────────────

    fn instantiate_fn(
        &mut self,
        orig_name: &str,
        resolved_args: &[ResolvedGenArg],
        call_span: Span,
        caller_env: &SubstEnv,
    ) -> Option<(String, TypeExpr)> {
        let (mangled, human_sig) = self.build_names(orig_name, resolved_args);
        if caller_env.depth > MAX_GENERIC_DEPTH {
            self.emit(
                &caller_env.file,
                Diagnostic::new(
                    DiagnosticKind::GenericInstantiationError,
                    call_span,
                    format!(
                        "recursive generic instantiation exceeded maximum depth of {} while instantiating '{}'",
                        MAX_GENERIC_DEPTH, human_sig
                    ),
                ),
            );
            return None;
        }

        if let Some(ret_ty) = self.instantiated_fns.get(&mangled).cloned() {
            return Some((mangled, ret_ty));
        }
        if !self.check_total_instance_cap(&human_sig, call_span, &caller_env.file) {
            return None;
        }

        self.instances.insert(
            (MonomorphizedKind::Fn, mangled.clone()),
            MonomorphizedInstance {
                kind: MonomorphizedKind::Fn,
                signature: human_sig.clone(),
                mangled_name: mangled.clone(),
            },
        );

        let tmpl = self.fn_templates.get(orig_name)?.clone();
        let instantiated_origin = Self::format_origin(&human_sig, &caller_env.file, call_span);
        let mut child_env = SubstEnv {
            types: BTreeMap::new(),
            consts: BTreeMap::new(),
            depth: caller_env.depth + 1,
            prog_idx: tmpl.prog_idx,
            file: tmpl.file.clone(),
        };
        for (gp, ga) in tmpl.decl.generic_params.iter().zip(resolved_args.iter()) {
            match ga {
                ResolvedGenArg::Type(te) => {
                    child_env.types.insert(gp.name.name.clone(), te.clone());
                }
                ResolvedGenArg::Const(n) => {
                    child_env.consts.insert(gp.name.name.clone(), *n);
                }
            }
        }

        let params: Vec<Param> = tmpl
            .decl
            .params
            .iter()
            .map(|p| Param {
                ty: self.rewrite_type_expr(&p.ty, &child_env),
                name: p.name.clone(),
                span: p.span,
            })
            .collect();
        let ret_type = self.rewrite_type_expr(&tmpl.decl.ret_type, &child_env);
        let error_spec = tmpl.decl.error_spec.as_ref().map(|es| match es {
            ErrorSpec::Bare(sp) => ErrorSpec::Bare(*sp),
            ErrorSpec::Typed(te) => ErrorSpec::Typed(self.rewrite_type_expr(te, &child_env)),
        });

        // Register signature immediately and enqueue body rewrite onto iterative worklist (`O(1)` stack depth).
        self.instantiated_fns
            .insert(mangled.clone(), ret_type.clone());
        self.non_generic_fn_rets
            .insert(mangled.clone(), ret_type.clone());

        self.pending_fn_bodies.push_back(PendingFnBody {
            prog_idx: tmpl.prog_idx,
            item_attrs: tmpl.attrs,
            fn_attrs: tmpl.decl.attrs,
            is_extern_c: tmpl.decl.is_extern_c,
            mangled: mangled.clone(),
            name_span: tmpl.decl.name.span,
            human_sig,
            instantiated_origin,
            params,
            is_variadic: tmpl.decl.is_variadic,
            ret_type: ret_type.clone(),
            error_spec,
            raw_body: tmpl.decl.body,
            fn_span: tmpl.decl.span,
            child_env,
        });

        Some((mangled, ret_type))
    }

    fn compute_impl_pattern_key(
        &self,
        impl_generics: &[GenericParam],
        target_args: &[GenericArg],
        struct_generics: &[GenericParam],
        env: &SubstEnv,
    ) -> String {
        let mut t_idx = BTreeMap::new();
        let mut c_idx = BTreeMap::new();
        for (idx, gp) in impl_generics.iter().enumerate() {
            match gp.kind {
                GenericParamKind::Type => {
                    t_idx.insert(gp.name.name.as_str(), idx);
                }
                GenericParamKind::ConstUsize => {
                    c_idx.insert(gp.name.name.as_str(), idx);
                }
            }
        }
        let mut parts = Vec::with_capacity(target_args.len());
        for (i, arg) in target_args.iter().enumerate() {
            let is_const_slot = struct_generics
                .get(i)
                .is_some_and(|gp| gp.kind == GenericParamKind::ConstUsize);
            if is_const_slot {
                let bound_name = match arg {
                    GenericArg::Type(TypeExpr {
                        kind: TypeKind::Named(id),
                        ..
                    })
                    | GenericArg::Const(Expr {
                        kind: ExprKind::Ident(id),
                        ..
                    }) => c_idx.get(id.name.as_str()).copied(),
                    _ => None,
                };
                if let Some(ci) = bound_name {
                    parts.push(format!("#C{ci}"));
                } else {
                    let expr = match arg {
                        GenericArg::Const(e) => Some(e.clone()),
                        GenericArg::Type(TypeExpr {
                            kind: TypeKind::Named(id),
                            span,
                        }) => Some(Expr::new(ExprKind::Ident(id.clone()), *span)),
                        _ => None,
                    };
                    if let Some(n) = expr.and_then(|e| self.eval_usize_in_env(&e, env)) {
                        parts.push(format!("K{n}"));
                    } else {
                        parts.push(format!("{arg:?}"));
                    }
                }
            } else {
                parts.push(Self::pattern_type_key(arg, &t_idx, &c_idx));
            }
        }
        parts.join(",")
    }

    fn pattern_type_key(
        arg: &GenericArg,
        t_idx: &BTreeMap<&str, usize>,
        c_idx: &BTreeMap<&str, usize>,
    ) -> String {
        fn norm_ty(
            te: &TypeExpr,
            t_idx: &BTreeMap<&str, usize>,
            c_idx: &BTreeMap<&str, usize>,
        ) -> String {
            match &te.kind {
                TypeKind::Prim(p) => p.as_str().to_string(),
                TypeKind::Named(id) => {
                    if let Some(ti) = t_idx.get(id.name.as_str()) {
                        format!("#T{ti}")
                    } else {
                        id.name.clone()
                    }
                }
                TypeKind::Path(segs) => segs
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join("."),
                TypeKind::Ptr(inner) => format!("*{}", norm_ty(inner, t_idx, c_idx)),
                TypeKind::Slice(inner) => format!("[]{}", norm_ty(inner, t_idx, c_idx)),
                TypeKind::Lease(inner) => format!("Lease<{}>", norm_ty(inner, t_idx, c_idx)),
                TypeKind::Array { elem, len } => {
                    let lstr = match len {
                        ArrayLen::Fixed(n) => n.to_string(),
                        ArrayLen::Expr(e) => {
                            if let ExprKind::Ident(id) = &e.kind {
                                if let Some(ci) = c_idx.get(id.name.as_str()) {
                                    format!("#C{ci}")
                                } else {
                                    format!("{e:?}")
                                }
                            } else {
                                format!("{e:?}")
                            }
                        }
                    };
                    format!("[{lstr}]{}", norm_ty(elem, t_idx, c_idx))
                }
                TypeKind::Generic { base, args } => {
                    let gs: Vec<String> = args
                        .iter()
                        .map(|a| match a {
                            GenericArg::Type(t) => norm_ty(t, t_idx, c_idx),
                            GenericArg::Const(e) => format!("{e:?}"),
                        })
                        .collect();
                    format!("{}<{}>", base.name, gs.join(","))
                }
                other => format!("{other:?}"),
            }
        }
        match arg {
            GenericArg::Type(te) => norm_ty(te, t_idx, c_idx),
            GenericArg::Const(e) => format!("{e:?}"),
        }
    }

    fn unify_const_arg_pattern(
        &self,
        pat_arg: &GenericArg,
        conc_n: usize,
        impl_consts: &BTreeSet<String>,
        bound_consts: &mut BTreeMap<String, usize>,
        score: &mut usize,
        tmpl_env: &SubstEnv,
    ) -> bool {
        let ident_name = match pat_arg {
            GenericArg::Type(TypeExpr {
                kind: TypeKind::Named(id),
                ..
            })
            | GenericArg::Const(Expr {
                kind: ExprKind::Ident(id),
                ..
            }) => Some(id.clone()),
            _ => None,
        };
        if let Some(id) = &ident_name {
            if impl_consts.contains(&id.name) {
                if let Some(&prev) = bound_consts.get(&id.name) {
                    return prev == conc_n;
                }
                bound_consts.insert(id.name.clone(), conc_n);
                return true;
            }
        }
        *score += 10;
        let eval_expr = match pat_arg {
            GenericArg::Const(e) => Some(e.clone()),
            GenericArg::Type(TypeExpr {
                kind: TypeKind::Named(id),
                span,
            }) => Some(Expr::new(ExprKind::Ident(id.clone()), *span)),
            _ => None,
        };
        eval_expr
            .and_then(|e| self.eval_usize_in_env(&e, tmpl_env))
            .is_some_and(|v| v == conc_n)
    }

    fn unify_type_pattern(
        &self,
        pat: &TypeExpr,
        conc: &TypeExpr,
        impl_types: &BTreeSet<String>,
        impl_consts: &BTreeSet<String>,
        bound_types: &mut BTreeMap<String, TypeExpr>,
        bound_consts: &mut BTreeMap<String, usize>,
        score: &mut usize,
        tmpl_env: &SubstEnv,
    ) -> bool {
        if let TypeKind::Named(id) = &pat.kind {
            if impl_types.contains(&id.name) {
                if let Some(prev) = bound_types.get(&id.name) {
                    return self.types_structurally_equal_mono(prev, conc);
                }
                bound_types.insert(id.name.clone(), conc.clone());
                return true;
            }
        }
        *score += 10;
        match (&pat.kind, &conc.kind) {
            (TypeKind::Prim(a), TypeKind::Prim(b)) => a == b,
            (TypeKind::Named(a), TypeKind::Named(b)) => a.name == b.name,
            (TypeKind::Ptr(a), TypeKind::Ptr(b))
            | (TypeKind::Slice(a), TypeKind::Slice(b))
            | (TypeKind::Lease(a), TypeKind::Lease(b)) => self.unify_type_pattern(
                a,
                b,
                impl_types,
                impl_consts,
                bound_types,
                bound_consts,
                score,
                tmpl_env,
            ),
            (
                TypeKind::Array { elem: ea, len: la },
                TypeKind::Array { elem: eb, len: lb },
            ) => {
                if !self.unify_type_pattern(
                    ea,
                    eb,
                    impl_types,
                    impl_consts,
                    bound_types,
                    bound_consts,
                    score,
                    tmpl_env,
                ) {
                    return false;
                }
                let Some(conc_n) = lb.as_fixed() else {
                    return false;
                };
                match la {
                    ArrayLen::Fixed(n) => {
                        *score += 10;
                        *n == conc_n
                    }
                    ArrayLen::Expr(e) => self.unify_const_arg_pattern(
                        &GenericArg::Const((**e).clone()),
                        conc_n,
                        impl_consts,
                        bound_consts,
                        score,
                        tmpl_env,
                    ),
                }
            }
            (TypeKind::Generic { base, args }, TypeKind::Named(conc_id)) => {
                let Some((conc_base, conc_args)) = self.instantiated_struct_args.get(&conc_id.name)
                else {
                    return false;
                };
                if base.name != *conc_base || args.len() != conc_args.len() {
                    return false;
                }
                let Some(st_tmpl) = self.struct_templates.get(conc_base) else {
                    return false;
                };
                for (i, (pa, ca)) in args.iter().zip(conc_args.iter()).enumerate() {
                    let is_c = st_tmpl
                        .decl
                        .generic_params
                        .get(i)
                        .is_some_and(|gp| gp.kind == GenericParamKind::ConstUsize);
                    match (is_c, pa, ca) {
                        (true, _, ResolvedGenArg::Const(cn)) => {
                            if !self.unify_const_arg_pattern(
                                pa,
                                *cn,
                                impl_consts,
                                bound_consts,
                                score,
                                tmpl_env,
                            ) {
                                return false;
                            }
                        }
                        (false, GenericArg::Type(pt), ResolvedGenArg::Type(ct)) => {
                            if !self.unify_type_pattern(
                                pt,
                                ct,
                                impl_types,
                                impl_consts,
                                bound_types,
                                bound_consts,
                                score,
                                tmpl_env,
                            ) {
                                return false;
                            }
                        }
                        _ => return false,
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn types_structurally_equal_mono(&self, a: &TypeExpr, b: &TypeExpr) -> bool {
        match (&a.kind, &b.kind) {
            (TypeKind::Prim(x), TypeKind::Prim(y)) => x == y,
            (TypeKind::Named(x), TypeKind::Named(y)) => x.name == y.name,
            (TypeKind::Path(x), TypeKind::Path(y)) => {
                x.len() == y.len() && x.iter().zip(y.iter()).all(|(l, r)| l.name == r.name)
            }
            (TypeKind::Ptr(x), TypeKind::Ptr(y))
            | (TypeKind::Slice(x), TypeKind::Slice(y))
            | (TypeKind::Lease(x), TypeKind::Lease(y)) => {
                self.types_structurally_equal_mono(x, y)
            }
            (
                TypeKind::Array { elem: ex, len: lx },
                TypeKind::Array { elem: ey, len: ly },
            ) => lx == ly && self.types_structurally_equal_mono(ex, ey),
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_generic_method_call(
        &mut self,
        base_struct: &str,
        concrete_struct_name: &str,
        concrete_args: &[ResolvedGenArg],
        method_name: &str,
        call_span: Span,
        caller_scope: Option<usize>,
        caller_env: &SubstEnv,
    ) -> Option<(String, Option<SelfReceiver>, TypeExpr, bool, Option<usize>)> {
        let candidates = self.generic_impl_methods.get(base_struct)?.clone();
        let st_tmpl = self.struct_templates.get(base_struct)?.clone();

        let mut matches: Vec<(
            usize,
            GenericImplMethodEntry,
            BTreeMap<String, TypeExpr>,
            BTreeMap<String, usize>,
        )> = Vec::new();

        for entry in candidates {
            if entry.method.func.name.name != method_name {
                continue;
            }
            if entry.target_generic_args.len() != concrete_args.len() {
                continue;
            }
            let impl_types: BTreeSet<String> = entry
                .impl_generic_params
                .iter()
                .filter(|g| g.kind == GenericParamKind::Type)
                .map(|g| g.name.name.clone())
                .collect();
            let impl_consts: BTreeSet<String> = entry
                .impl_generic_params
                .iter()
                .filter(|g| g.kind == GenericParamKind::ConstUsize)
                .map(|g| g.name.name.clone())
                .collect();

            let mut bound_types = BTreeMap::new();
            let mut bound_consts = BTreeMap::new();
            let mut score = 0usize;
            let mut ok = true;
            let tmpl_env = SubstEnv {
                types: BTreeMap::new(),
                consts: BTreeMap::new(),
                depth: 0,
                prog_idx: entry.prog_idx,
                file: entry.file.clone(),
            };

            for (i, (pat_arg, conc_arg)) in entry
                .target_generic_args
                .iter()
                .zip(concrete_args.iter())
                .enumerate()
            {
                let is_const = st_tmpl
                    .decl
                    .generic_params
                    .get(i)
                    .is_some_and(|gp| gp.kind == GenericParamKind::ConstUsize);
                match (is_const, pat_arg, conc_arg) {
                    (true, _, ResolvedGenArg::Const(cn)) => {
                        if !self.unify_const_arg_pattern(
                            pat_arg,
                            *cn,
                            &impl_consts,
                            &mut bound_consts,
                            &mut score,
                            &tmpl_env,
                        ) {
                            ok = false;
                            break;
                        }
                    }
                    (false, GenericArg::Type(pat_ty), ResolvedGenArg::Type(conc_ty)) => {
                        if !self.unify_type_pattern(
                            pat_ty,
                            conc_ty,
                            &impl_types,
                            &impl_consts,
                            &mut bound_types,
                            &mut bound_consts,
                            &mut score,
                            &tmpl_env,
                        ) {
                            ok = false;
                            break;
                        }
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok
                && bound_types.len() == impl_types.len()
                && bound_consts.len() == impl_consts.len()
            {
                matches.push((score, entry, bound_types, bound_consts));
            }
        }

        if matches.is_empty() {
            return None;
        }
        matches.sort_by(|a, b| b.0.cmp(&a.0));
        let top_score = matches[0].0;
        let tied = matches.iter().take_while(|m| m.0 == top_score).count();
        let Some((_, entry, bound_types, bound_consts)) = matches.into_iter().next() else {
            return None;
        };

        if tied > 1 {
            let human_st = self
                .human_type_names
                .get(concrete_struct_name)
                .cloned()
                .unwrap_or_else(|| concrete_struct_name.to_string());
            self.emit(
                &caller_env.file,
                Diagnostic::new(
                    DiagnosticKind::GenericInstantiationError,
                    call_span,
                    format!(
                        "ambiguous `impl` specialization for method '{method_name}' on '{human_st}'"
                    ),
                ),
            );
            return None;
        }

        if !entry.method.is_pub && entry.owner_scope != caller_scope {
            let human_st = self
                .human_type_names
                .get(concrete_struct_name)
                .cloned()
                .unwrap_or_else(|| concrete_struct_name.to_string());
            self.emit(
                &caller_env.file,
                Diagnostic::new(
                    DiagnosticKind::ModuleError,
                    call_span,
                    format!("method `{method_name}` on `{human_st}` is private; mark it `pub` to export it"),
                ),
            );
        }

        let mangled = format!("{concrete_struct_name}__{method_name}");
        let human_st = self
            .human_type_names
            .get(concrete_struct_name)
            .cloned()
            .unwrap_or_else(|| concrete_struct_name.to_string());
        let human_sig = format!("{human_st}::{method_name}");

        if caller_env.depth > MAX_GENERIC_DEPTH {
            self.emit(
                &caller_env.file,
                Diagnostic::new(
                    DiagnosticKind::GenericInstantiationError,
                    call_span,
                    format!(
                        "recursive generic instantiation exceeded maximum depth of {} while instantiating '{human_sig}'",
                        MAX_GENERIC_DEPTH
                    ),
                ),
            );
            return None;
        }

        if let Some(ret_ty) = self.instantiated_fns.get(&mangled).cloned() {
            return Some((
                mangled,
                entry.method.receiver,
                ret_ty,
                entry.method.is_pub,
                entry.owner_scope,
            ));
        }

        let child_env = SubstEnv {
            types: bound_types,
            consts: bound_consts,
            depth: caller_env.depth + 1,
            prog_idx: entry.prog_idx,
            file: entry.file.clone(),
        };
        let ret_type = self.rewrite_type_expr(&entry.method.func.ret_type, &child_env);

        if !self.check_total_instance_cap(&human_sig, call_span, &caller_env.file) {
            return None;
        }

        let instantiated_origin = Self::format_origin(&human_sig, &caller_env.file, call_span);
        self.instances.insert(
            (MonomorphizedKind::Fn, mangled.clone()),
            MonomorphizedInstance {
                kind: MonomorphizedKind::Fn,
                signature: human_sig.clone(),
                mangled_name: mangled.clone(),
            },
        );

        let mut params = Vec::with_capacity(
            entry.method.func.params.len() + usize::from(entry.method.receiver.is_some()),
        );
        if let Some(recv) = entry.method.receiver {
            let target_ty = TypeExpr::new(
                TypeKind::Named(Ident::new(
                    concrete_struct_name.to_string(),
                    entry.method.span,
                )),
                entry.method.span,
            );
            let self_ty = match recv {
                SelfReceiver::Value => target_ty,
                SelfReceiver::Ptr => {
                    TypeExpr::new(TypeKind::Ptr(Box::new(target_ty)), entry.method.span)
                }
            };
            params.push(Param {
                ty: self_ty,
                name: Ident::new("self".to_string(), entry.method.span),
                span: entry.method.span,
            });
        }
        for p in &entry.method.func.params {
            params.push(Param {
                ty: self.rewrite_type_expr(&p.ty, &child_env),
                name: p.name.clone(),
                span: p.span,
            });
        }
        let error_spec = entry.method.func.error_spec.as_ref().map(|es| match es {
            ErrorSpec::Bare(sp) => ErrorSpec::Bare(*sp),
            ErrorSpec::Typed(te) => ErrorSpec::Typed(self.rewrite_type_expr(te, &child_env)),
        });

        self.instantiated_fns
            .insert(mangled.clone(), ret_type.clone());
        self.non_generic_fn_rets
            .insert(mangled.clone(), ret_type.clone());

        self.pending_fn_bodies.push_back(PendingFnBody {
            prog_idx: entry.prog_idx,
            item_attrs: Vec::new(),
            fn_attrs: entry.method.func.attrs.clone(),
            is_extern_c: false,
            mangled: mangled.clone(),
            name_span: entry.method.func.name.span,
            human_sig,
            instantiated_origin,
            params,
            is_variadic: false,
            ret_type: ret_type.clone(),
            error_spec,
            raw_body: entry.method.func.body.clone(),
            fn_span: entry.method.func.span,
            child_env,
        });

        Some((
            mangled,
            entry.method.receiver,
            ret_type,
            entry.method.is_pub,
            entry.owner_scope,
        ))
    }

    fn rewrite_fn_decl(
        &mut self,
        f: &FnDecl,
        pre_ret: Option<TypeExpr>,
        env: &SubstEnv,
    ) -> FnDecl {
        let params: Vec<Param> = f
            .params
            .iter()
            .map(|p| Param {
                ty: self.rewrite_type_expr(&p.ty, env),
                name: p.name.clone(),
                span: p.span,
            })
            .collect();
        let ret_type = pre_ret.unwrap_or_else(|| self.rewrite_type_expr(&f.ret_type, env));
        let error_spec = f.error_spec.as_ref().map(|es| match es {
            ErrorSpec::Bare(sp) => ErrorSpec::Bare(*sp),
            ErrorSpec::Typed(te) => ErrorSpec::Typed(self.rewrite_type_expr(te, env)),
        });
        let body = f.body.as_ref().map(|b| {
            let mut scopes = vec![BTreeMap::new()];
            for p in &params {
                scopes[0].insert(p.name.name.clone(), p.ty.clone());
            }
            self.rewrite_block(b, env, &mut scopes)
        });
        FnDecl {
            attrs: f.attrs.clone(),
            is_extern_c: f.is_extern_c,
            name: f.name.clone(),
            generic_params: Vec::new(),
            params,
            is_variadic: f.is_variadic,
            ret_type,
            error_spec,
            body,
            span: f.span,
        }
    }

    fn rewrite_block(
        &mut self,
        block: &Block,
        env: &SubstEnv,
        scopes: &mut Vec<BTreeMap<String, TypeExpr>>,
    ) -> Block {
        scopes.push(BTreeMap::new());
        let mut stmts = Vec::with_capacity(block.stmts.len());
        for s in &block.stmts {
            stmts.push(self.rewrite_stmt(s, env, scopes));
        }
        scopes.pop();
        Block::new(stmts, block.span)
    }

    fn rewrite_stmt(
        &mut self,
        stmt: &Stmt,
        env: &SubstEnv,
        scopes: &mut Vec<BTreeMap<String, TypeExpr>>,
    ) -> Stmt {
        let kind = match &stmt.kind {
            StmtKind::VarDecl { ty, name, init } => {
                let r_ty = self.rewrite_type_expr(ty, env);
                let r_init = init
                    .as_ref()
                    .map(|e| self.rewrite_expr(e, env, scopes, Some(&r_ty)).0);
                if let Some(top) = scopes.last_mut() {
                    top.insert(name.name.clone(), r_ty.clone());
                }
                StmtKind::VarDecl {
                    ty: r_ty,
                    name: name.clone(),
                    init: r_init,
                }
            }
            StmtKind::StateVar { ty, name, init } => {
                let r_ty = self.rewrite_type_expr(ty, env);
                let r_init = init
                    .as_ref()
                    .map(|e| self.rewrite_expr(e, env, scopes, Some(&r_ty)).0);
                if let Some(top) = scopes.last_mut() {
                    top.insert(name.name.clone(), r_ty.clone());
                }
                StmtKind::StateVar {
                    ty: r_ty,
                    name: name.clone(),
                    init: r_init,
                }
            }
            StmtKind::Assign { target, op, value } => {
                let (rt, lhs_ty) = self.rewrite_expr(target, env, scopes, None);
                let hint = match lhs_ty {
                    InferTy::Concrete(ref t) => Some(t),
                    _ => None,
                };
                let rv = self.rewrite_expr(value, env, scopes, hint).0;
                StmtKind::Assign {
                    target: rt,
                    op: *op,
                    value: rv,
                }
            }
            StmtKind::Return(opt) => StmtKind::Return(
                opt.as_ref()
                    .map(|e| self.rewrite_expr(e, env, scopes, None).0),
            ),
            StmtKind::Error { reason, partial } => StmtKind::Error {
                reason: reason
                    .as_ref()
                    .map(|e| self.rewrite_expr(e, env, scopes, None).0),
                partial: partial
                    .as_ref()
                    .map(|e| self.rewrite_expr(e, env, scopes, None).0),
            },
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => StmtKind::If {
                cond: self.rewrite_expr(cond, env, scopes, None).0,
                then_block: self.rewrite_block(then_block, env, scopes),
                else_block: else_block
                    .as_ref()
                    .map(|eb| self.rewrite_block(eb, env, scopes)),
            },
            StmtKind::While { cond, body } => StmtKind::While {
                cond: self.rewrite_expr(cond, env, scopes, None).0,
                body: self.rewrite_block(body, env, scopes),
            },
            StmtKind::ForRange {
                var_ty,
                var_name,
                start,
                end,
                inclusive,
                body,
            } => {
                let r_vt = self.rewrite_type_expr(var_ty, env);
                let r_start = Box::new(self.rewrite_expr(start, env, scopes, Some(&r_vt)).0);
                let r_end = Box::new(self.rewrite_expr(end, env, scopes, Some(&r_vt)).0);
                scopes.push(BTreeMap::new());
                if let Some(top) = scopes.last_mut() {
                    top.insert(var_name.name.clone(), r_vt.clone());
                }
                let r_body = self.rewrite_block(body, env, scopes);
                scopes.pop();
                StmtKind::ForRange {
                    var_ty: r_vt,
                    var_name: var_name.clone(),
                    start: r_start,
                    end: r_end,
                    inclusive: *inclusive,
                    body: r_body,
                }
            }
            StmtKind::ForEach {
                elem_ty,
                elem_name,
                iter,
                body,
            } => {
                let r_et = self.rewrite_type_expr(elem_ty, env);
                let r_iter = Box::new(self.rewrite_expr(iter, env, scopes, None).0);
                scopes.push(BTreeMap::new());
                if let Some(top) = scopes.last_mut() {
                    top.insert(elem_name.name.clone(), r_et.clone());
                }
                let r_body = self.rewrite_block(body, env, scopes);
                scopes.pop();
                StmtKind::ForEach {
                    elem_ty: r_et,
                    elem_name: elem_name.clone(),
                    iter: r_iter,
                    body: r_body,
                }
            }
            StmtKind::Loop { body } => StmtKind::Loop {
                body: self.rewrite_block(body, env, scopes),
            },
            StmtKind::LoopEvery { period, body } => StmtKind::LoopEvery {
                period: Box::new(self.rewrite_expr(period, env, scopes, None).0),
                body: self.rewrite_block(body, env, scopes),
            },
            StmtKind::Defer(inner) => {
                scopes.push(BTreeMap::new());
                let r_inner = Box::new(self.rewrite_stmt(inner, env, scopes));
                scopes.pop();
                StmtKind::Defer(r_inner)
            }
            StmtKind::NoYield { body } => StmtKind::NoYield {
                body: self.rewrite_block(body, env, scopes),
            },
            StmtKind::Critical { body } => StmtKind::Critical {
                body: self.rewrite_block(body, env, scopes),
            },
            StmtKind::Switch { scrutinee, arms } => {
                let r_scrut = self.rewrite_expr(scrutinee, env, scopes, None).0;
                let mut r_arms = Vec::with_capacity(arms.len());
                for a in arms {
                    let pat = match &a.pat {
                        SwitchPat::Ident {
                            prefix: None,
                            name,
                            span,
                        } if !scopes_contain(scopes, &name.name)
                            && env.consts.contains_key(&name.name) =>
                        {
                            let n = env.consts[&name.name];
                            SwitchPat::Literal(
                                Literal::Int {
                                    value: n as u64,
                                    suffix: Some(PrimType::Usize),
                                },
                                *span,
                            )
                        }
                        SwitchPat::Ident { prefix, name, span } => SwitchPat::Ident {
                            prefix: prefix.clone(),
                            name: name.clone(),
                            span: *span,
                        },
                        SwitchPat::Qualified { path, span } => SwitchPat::Qualified {
                            path: path.clone(),
                            span: *span,
                        },
                        SwitchPat::Literal(l, sp) => SwitchPat::Literal(l.clone(), *sp),
                        SwitchPat::Wildcard(sp) => SwitchPat::Wildcard(*sp),
                    };
                    let body = self.rewrite_block(&a.body, env, scopes);
                    r_arms.push(SwitchArm {
                        pat,
                        body,
                        span: a.span,
                    });
                }
                StmtKind::Switch {
                    scrutinee: r_scrut,
                    arms: r_arms,
                }
            }
            StmtKind::Wait { arms, timer } => {
                let mut r_arms = Vec::with_capacity(arms.len());
                for a in arms {
                    let r_bind = a
                        .bind
                        .as_ref()
                        .map(|(ty, id)| (self.rewrite_type_expr(ty, env), id.clone()));
                    let r_call = self.rewrite_expr(&a.call, env, scopes, None).0;
                    scopes.push(BTreeMap::new());
                    if let (Some((rt, id)), Some(sc)) = (&r_bind, scopes.last_mut()) {
                        sc.insert(id.name.clone(), rt.clone());
                    }
                    let body = self.rewrite_block(&a.body, env, scopes);
                    scopes.pop();
                    r_arms.push(WaitArm {
                        bind: r_bind,
                        call: r_call,
                        body,
                        span: a.span,
                    });
                }
                let r_timer = timer.as_ref().map(|t| WaitTimerArm {
                    kind: t.kind,
                    expr: self.rewrite_expr(&t.expr, env, scopes, None).0,
                    body: self.rewrite_block(&t.body, env, scopes),
                    span: t.span,
                });
                StmtKind::Wait {
                    arms: r_arms,
                    timer: r_timer,
                }
            }
            StmtKind::Spawn(sp) => {
                let args = sp
                    .args
                    .iter()
                    .map(|a| CallArg {
                        name: a.name.clone(),
                        value: self.rewrite_expr(&a.value, env, scopes, None).0,
                        span: a.span,
                    })
                    .collect();
                StmtKind::Spawn(SpawnStmt {
                    module: sp.module.clone(),
                    task: sp.task.clone(),
                    args,
                    priority: sp.priority,
                    core: sp.core,
                    span: sp.span,
                })
            }
            StmtKind::Continue => StmtKind::Continue,
            StmtKind::Break => StmtKind::Break,
            StmtKind::Restart => StmtKind::Restart,
            StmtKind::Checkpoint => StmtKind::Checkpoint,
            StmtKind::Block(b) => StmtKind::Block(self.rewrite_block(b, env, scopes)),
            StmtKind::Expr(e) => StmtKind::Expr(self.rewrite_expr(e, env, scopes, None).0),
        };
        Stmt::with_attrs(stmt.attrs.clone(), kind, stmt.span)
    }

    fn lookup_scope_ty(
        &self,
        name: &str,
        env: &SubstEnv,
        scopes: &[BTreeMap<String, TypeExpr>],
    ) -> InferTy {
        for sc in scopes.iter().rev() {
            if let Some(t) = sc.get(name) {
                return InferTy::Concrete(t.clone());
            }
        }
        if env.consts.contains_key(name) {
            return InferTy::Concrete(TypeExpr::new(
                TypeKind::Prim(PrimType::Usize),
                Span::dummy(),
            ));
        }
        if let Some(ct) = self.const_types.get(name) {
            return InferTy::Concrete(ct.clone());
        }
        if let Some(gt) = self.global_var_types.get(name) {
            return InferTy::Concrete(gt.clone());
        }
        if let Some(os) = self.prim_types.get(name) {
            return InferTy::Concrete(TypeExpr::new(
                TypeKind::OsPrimitive(os.clone()),
                Span::dummy(),
            ));
        }
        if let Some(mmio_ptr_ty) = self.mmio_instances.get(name) {
            return InferTy::Concrete(mmio_ptr_ty.clone());
        }
        if let Some(enums) = self.variant_to_enums.get(name) {
            let visible: Vec<&String> = enums
                .iter()
                .filter(|en| match self.enum_prog_idx.get(*en).copied().flatten() {
                    Some(decl_idx) => decl_idx == env.prog_idx,
                    None => env.prog_idx >= self.non_root_prog_count,
                })
                .collect();
            if visible.len() == 1 {
                return InferTy::Concrete(TypeExpr::new(
                    TypeKind::Named(Ident::dummy(visible[0].clone())),
                    Span::dummy(),
                ));
            }
        }
        InferTy::Unknown
    }

    fn rewrite_expr(
        &mut self,
        expr: &Expr,
        env: &SubstEnv,
        scopes: &mut Vec<BTreeMap<String, TypeExpr>>,
        expected_ty: Option<&TypeExpr>,
    ) -> (Expr, InferTy) {
        match &expr.kind {
            ExprKind::Literal(lit) => {
                let ity = match lit {
                    Literal::Int { suffix: Some(p), .. } => {
                        InferTy::Concrete(TypeExpr::new(TypeKind::Prim(*p), expr.span))
                    }
                    Literal::Int { suffix: None, .. } => InferTy::IntLit,
                    Literal::Float { suffix: Some(p), .. } => {
                        InferTy::Concrete(TypeExpr::new(TypeKind::Prim(*p), expr.span))
                    }
                    Literal::Float { suffix: None, .. } => InferTy::FloatLit,
                    Literal::Bool(_) => {
                        InferTy::Concrete(TypeExpr::new(TypeKind::Prim(PrimType::Bool), expr.span))
                    }
                    Literal::String(_) => InferTy::Concrete(TypeExpr::new(
                        TypeKind::Ptr(Box::new(TypeExpr::new(
                            TypeKind::Prim(PrimType::U8),
                            expr.span,
                        ))),
                        expr.span,
                    )),
                    Literal::Duration(_) => InferTy::Unknown,
                };
                (expr.clone(), ity)
            }

            ExprKind::Ident(id) => {
                // Local variables/parameters shadow generic `ConstUsize` parameters of the same name (`M10`).
                if !scopes_contain(scopes, &id.name) {
                    if let Some(&n) = env.consts.get(&id.name) {
                        let lit_expr = Expr::new(
                            ExprKind::Literal(Literal::Int {
                                value: n as u64,
                                suffix: Some(PrimType::Usize),
                            }),
                            expr.span,
                        );
                        let ity = InferTy::Concrete(TypeExpr::new(
                            TypeKind::Prim(PrimType::Usize),
                            expr.span,
                        ));
                        return (lit_expr, ity);
                    }
                }
                // Likewise a local shadows a same-named generic `fn` template, so only a bare
                // reference that actually resolves to the template is an `E0029`.
                if !scopes_contain(scopes, &id.name) && self.fn_templates.contains_key(&id.name) {
                    self.emit(
                        &env.file,
                        Diagnostic::new(
                            DiagnosticKind::GenericInstantiationError,
                            id.span,
                            format!(
                                "generic function '{}' cannot be referenced without concrete call arguments",
                                id.name
                            ),
                        ),
                    );
                }
                let ity = self.lookup_scope_ty(&id.name, env, scopes);
                (expr.clone(), ity)
            }

            ExprKind::Priority(_) => (expr.clone(), InferTy::Unknown),

            ExprKind::Unary { op, expr: inner } => {
                let (ri, inner_ty) = self.rewrite_expr(inner, env, scopes, None);
                let out_ty = match op {
                    UnaryOp::AddrOf => match inner_ty {
                        InferTy::Concrete(ct) => InferTy::Concrete(TypeExpr::new(
                            TypeKind::Ptr(Box::new(ct)),
                            expr.span,
                        )),
                        _ => InferTy::Unknown,
                    },
                    UnaryOp::Deref => match inner_ty {
                        InferTy::Concrete(TypeExpr {
                            kind: TypeKind::Ptr(pointee),
                            ..
                        }) => InferTy::Concrete(*pointee),
                        _ => InferTy::Unknown,
                    },
                    UnaryOp::Not => {
                        InferTy::Concrete(TypeExpr::new(TypeKind::Prim(PrimType::Bool), expr.span))
                    }
                    UnaryOp::BitNot | UnaryOp::Neg => inner_ty,
                };
                (
                    Expr::new(
                        ExprKind::Unary {
                            op: *op,
                            expr: Box::new(ri),
                        },
                        expr.span,
                    ),
                    out_ty,
                )
            }

            ExprKind::Binary { op, lhs, rhs } => {
                let (rl, lt) = self.rewrite_expr(lhs, env, scopes, None);
                let (rr, rt) = self.rewrite_expr(rhs, env, scopes, None);
                let out_ty = match op {
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::And
                    | BinaryOp::Or => {
                        InferTy::Concrete(TypeExpr::new(TypeKind::Prim(PrimType::Bool), expr.span))
                    }
                    _ => match (lt, rt) {
                        (InferTy::Concrete(a), _) => InferTy::Concrete(a),
                        (_, InferTy::Concrete(b)) => InferTy::Concrete(b),
                        (InferTy::FloatLit, _) | (_, InferTy::FloatLit) => InferTy::FloatLit,
                        (InferTy::IntLit, InferTy::IntLit) => InferTy::IntLit,
                        _ => InferTy::Unknown,
                    },
                };
                (
                    Expr::new(
                        ExprKind::Binary {
                            op: *op,
                            lhs: Box::new(rl),
                            rhs: Box::new(rr),
                        },
                        expr.span,
                    ),
                    out_ty,
                )
            }

            ExprKind::Cast { expr: inner, ty } => {
                let (ri, _) = self.rewrite_expr(inner, env, scopes, None);
                let r_ty = self.rewrite_type_expr(ty, env);
                (
                    Expr::new(
                        ExprKind::Cast {
                            expr: Box::new(ri),
                            ty: r_ty.clone(),
                        },
                        expr.span,
                    ),
                    InferTy::Concrete(r_ty),
                )
            }

            ExprKind::Field { target, field } => {
                // Note: If `target` is a generic type name `TypeName` in `TypeName.from_bytes` or `Enum::Variant`:
                let (rt, base_ty) = self.rewrite_expr(target, env, scopes, None);
                let mut out_ty = InferTy::Unknown;
                if let InferTy::Concrete(ct) = &base_ty {
                    let deref = match &ct.kind {
                        TypeKind::Ptr(inner) => &**inner,
                        _ => ct,
                    };
                    match &deref.kind {
                        TypeKind::Named(sname) => {
                            if let Some(st) = self.known_structs.get(&sname.name) {
                                if let Some(sf) = st.fields.iter().find(|f| f.name.name == field.name)
                                {
                                    out_ty = InferTy::Concrete(sf.ty.clone());
                                }
                            } else if let Some(regs) = self.mmio_layouts.get(&sname.name) {
                                if let Some(r_ty) = regs.get(&field.name) {
                                    out_ty = InferTy::Concrete(r_ty.clone());
                                }
                            }
                        }
                        TypeKind::Slice(_) | TypeKind::Array { .. } if field.name == "len" => {
                            out_ty = InferTy::Concrete(TypeExpr::new(
                                TypeKind::Prim(PrimType::Usize),
                                expr.span,
                            ));
                        }
                        _ => {}
                    }
                } else if let ExprKind::Ident(id) = &rt.kind {
                    if self.known_enums.contains_key(&id.name) {
                        out_ty = InferTy::Concrete(TypeExpr::new(
                            TypeKind::Named(id.clone()),
                            expr.span,
                        ));
                    }
                }
                (
                    Expr::new(
                        ExprKind::Field {
                            target: Box::new(rt),
                            field: field.clone(),
                        },
                        expr.span,
                    ),
                    out_ty,
                )
            }

            ExprKind::Index { target, index } => {
                let (rt, bt) = self.rewrite_expr(target, env, scopes, None);
                let (ri, _) = self.rewrite_expr(index, env, scopes, None);
                let out_ty = match bt {
                    InferTy::Concrete(TypeExpr {
                        kind: TypeKind::Slice(elem) | TypeKind::Array { elem, .. },
                        ..
                    }) => InferTy::Concrete(*elem),
                    _ => InferTy::Unknown,
                };
                (
                    Expr::new(
                        ExprKind::Index {
                            target: Box::new(rt),
                            index: Box::new(ri),
                        },
                        expr.span,
                    ),
                    out_ty,
                )
            }

            ExprKind::Slice { target, start, end } => {
                let (rt, bt) = self.rewrite_expr(target, env, scopes, None);
                let rs = start
                    .as_ref()
                    .map(|s| Box::new(self.rewrite_expr(s, env, scopes, None).0));
                let re = end
                    .as_ref()
                    .map(|e| Box::new(self.rewrite_expr(e, env, scopes, None).0));
                let out_ty = match bt {
                    InferTy::Concrete(ct) => {
                        let inner_deref: &TypeExpr = match &ct.kind {
                            TypeKind::Ptr(p) => p,
                            _ => &ct,
                        };
                        match &inner_deref.kind {
                            TypeKind::Slice(elem) | TypeKind::Array { elem, .. } => {
                                InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Slice(elem.clone()),
                                    expr.span,
                                ))
                            }
                            _ => InferTy::Unknown,
                        }
                    }
                    _ => InferTy::Unknown,
                };
                (
                    Expr::new(
                        ExprKind::Slice {
                            target: Box::new(rt),
                            start: rs,
                            end: re,
                        },
                        expr.span,
                    ),
                    out_ty,
                )
            }

            ExprKind::StructLit {
                module,
                name,
                generic_args,
                fields,
            } => {
                // Handle type-alias substitution if `T` itself was used as `T { ... }`!
                let eff_name = if let Some(TypeExpr {
                    kind: TypeKind::Named(mapped),
                    ..
                }) = env.types.get(&name.name)
                {
                    mapped.clone()
                } else {
                    name.clone()
                };

                let mut resolved_struct_name = eff_name.name.clone();
                if let Some(tmpl) = self.struct_templates.get(&eff_name.name).cloned() {
                    let mut rewritten_fields = Vec::with_capacity(fields.len());
                    let mut field_tys = BTreeMap::new();
                    for fi in fields {
                        let (rv, fty) = self.rewrite_expr(&fi.value, env, scopes, None);
                        field_tys.insert(fi.name.name.clone(), fty);
                        rewritten_fields.push(FieldInit {
                            name: fi.name.clone(),
                            value: rv,
                            span: fi.span,
                        });
                    }

                    let resolved_gargs = if !generic_args.is_empty() {
                        self.resolve_explicit_generic_args(
                            &eff_name.name,
                            &tmpl.decl.generic_params,
                            generic_args,
                            env,
                            expr.span,
                        )
                    } else if let Some(TypeExpr {
                        kind: TypeKind::Named(exp_id),
                        ..
                    }) = expected_ty
                    {
                        if let Some((orig, inst_args)) =
                            self.instantiated_struct_args.get(&exp_id.name).cloned()
                        {
                            if orig == eff_name.name {
                                Some(inst_args)
                            } else {
                                self.infer_struct_lit_args(&tmpl.decl, &field_tys, env, expr.span)
                            }
                        } else {
                            self.infer_struct_lit_args(&tmpl.decl, &field_tys, env, expr.span)
                        }
                    } else {
                        self.infer_struct_lit_args(&tmpl.decl, &field_tys, env, expr.span)
                    };

                    if let Some(r_args) = resolved_gargs {
                        if let Some(mangled) =
                            self.instantiate_struct(&eff_name.name, &r_args, expr.span, env)
                        {
                            resolved_struct_name = mangled;
                        }
                    }

                    let out_ident = Ident::new(resolved_struct_name, eff_name.span);
                    let out_ty =
                        InferTy::Concrete(TypeExpr::new(TypeKind::Named(out_ident.clone()), expr.span));
                    return (
                        Expr::new(
                            ExprKind::StructLit {
                                module: module.clone(),
                                name: out_ident,
                                generic_args: Vec::new(),
                                fields: rewritten_fields,
                            },
                            expr.span,
                        ),
                        out_ty,
                    );
                }

                if !generic_args.is_empty() && module.is_none() {
                    self.emit(
                        &env.file,
                        Diagnostic::new(
                            DiagnosticKind::GenericInstantiationError,
                            expr.span,
                            format!(
                                "struct '{}' is not generic (expected 0 generic arguments, found {})",
                                eff_name.name,
                                generic_args.len()
                            ),
                        ),
                    );
                }

                let mut rewritten_fields = Vec::with_capacity(fields.len());
                let known_st = self.known_structs.get(&eff_name.name).cloned();
                for fi in fields {
                    let f_hint = known_st
                        .as_ref()
                        .and_then(|s| s.fields.iter().find(|f| f.name.name == fi.name.name))
                        .map(|f| &f.ty);
                    let (rv, _) = self.rewrite_expr(&fi.value, env, scopes, f_hint);
                    rewritten_fields.push(FieldInit {
                        name: fi.name.clone(),
                        value: rv,
                        span: fi.span,
                    });
                }

                let out_ty =
                    InferTy::Concrete(TypeExpr::new(TypeKind::Named(eff_name.clone()), expr.span));
                (
                    Expr::new(
                        ExprKind::StructLit {
                            module: module.clone(),
                            name: eff_name,
                            generic_args: Vec::new(),
                            fields: rewritten_fields,
                        },
                        expr.span,
                    ),
                    out_ty,
                )
            }

            ExprKind::ArrayLit(elems) => {
                let elem_hint = match expected_ty.map(|t| &t.kind) {
                    Some(TypeKind::Array { elem, .. } | TypeKind::Slice(elem)) => Some(&**elem),
                    _ => None,
                };
                let mut r_elems = Vec::with_capacity(elems.len());
                let mut first_ty: Option<InferTy> = None;
                for e in elems {
                    let (re, et) = self.rewrite_expr(e, env, scopes, elem_hint);
                    r_elems.push(re);
                    match (&first_ty, et) {
                        (None, t) => first_ty = Some(t),
                        (Some(InferTy::IntLit | InferTy::FloatLit), InferTy::Concrete(c)) => {
                            first_ty = Some(InferTy::Concrete(c));
                        }
                        _ => {}
                    }
                }
                let concrete_elem = match first_ty {
                    Some(InferTy::Concrete(c)) => Some(c),
                    Some(InferTy::FloatLit) => {
                        if let Some(hint) = elem_hint.filter(|h| is_float_type_expr(h)) {
                            Some(hint.clone())
                        } else {
                            Some(TypeExpr::new(TypeKind::Prim(PrimType::F64), expr.span))
                        }
                    }
                    Some(InferTy::IntLit) => {
                        if let Some(hint) = elem_hint.filter(|h| is_int_type_expr(h)) {
                            Some(hint.clone())
                        } else {
                            Some(TypeExpr::new(TypeKind::Prim(PrimType::I32), expr.span))
                        }
                    }
                    _ => None,
                };
                let out_ty = concrete_elem.map_or(InferTy::Unknown, |ce| {
                    InferTy::Concrete(TypeExpr::new(
                        TypeKind::Array {
                            elem: Box::new(ce),
                            len: ArrayLen::Fixed(r_elems.len()),
                        },
                        expr.span,
                    ))
                });
                (Expr::new(ExprKind::ArrayLit(r_elems), expr.span), out_ty)
            }

            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => self.rewrite_call_expr(callee, generic_args, args, expr.span, env, scopes),

            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Support `Struct<T, N>.from_bytes(...)` if `receiver` is a generic struct name or type alias!
                let (rr, r_ty) = if let ExprKind::Ident(id) = &receiver.kind {
                    if let Some(TypeExpr {
                        kind: TypeKind::Named(mapped),
                        ..
                    }) = env.types.get(&id.name)
                    {
                        (
                            Expr::new(ExprKind::Ident(mapped.clone()), receiver.span),
                            InferTy::Unknown,
                        )
                    } else {
                        self.rewrite_expr(receiver, env, scopes, None)
                    }
                } else {
                    self.rewrite_expr(receiver, env, scopes, None)
                };

                let mut r_args = Vec::with_capacity(args.len());
                for a in args {
                    r_args.push(CallArg {
                        name: a.name.clone(),
                        value: self.rewrite_expr(&a.value, env, scopes, None).0,
                        span: a.span,
                    });
                }

                let caller_scope = if env.prog_idx < self.non_root_prog_count {
                    Some(env.prog_idx)
                } else {
                    None
                };

                // 1. Associated struct method (`Type.method(...)` or `<mod>.Type.method(...)`).
                if let ExprKind::Ident(id) = &rr.kind {
                    if !scopes_contain(scopes, &id.name) {
                        if let Some(info) = self
                            .concrete_methods
                            .get(&(id.name.clone(), method.name.clone()))
                            .cloned()
                        {
                            if info.receiver.is_some() {
                                self.emit(
                                    &env.file,
                                    Diagnostic::new(
                                        DiagnosticKind::InvalidOperation,
                                        expr.span,
                                        format!(
                                            "cannot call instance method `{}.{}` without a receiver value",
                                            id.name, method.name
                                        ),
                                    ),
                                );
                            } else {
                                if !info.is_pub && info.owner_scope != caller_scope {
                                    self.emit(
                                        &env.file,
                                        Diagnostic::new(
                                            DiagnosticKind::ModuleError,
                                            expr.span,
                                            format!(
                                                "method `{}.{}` is private; mark it `pub` to export it",
                                                id.name, method.name
                                            ),
                                        ),
                                    );
                                }
                                let c_span = receiver.span.merge(method.span);
                                return (
                                    Expr::new(
                                        ExprKind::Call {
                                            callee: Box::new(Expr::new(
                                                ExprKind::Ident(Ident::new(info.fn_name, c_span)),
                                                c_span,
                                            )),
                                            generic_args: Vec::new(),
                                            args: r_args,
                                        },
                                        expr.span,
                                    ),
                                    InferTy::Concrete(info.ret_ty),
                                );
                            }
                        }
                    }
                }

                // 2. Instance method on concrete or monomorphized generic struct (`val.method(...)` / `ptr.method(...)`).
                if let InferTy::Concrete(ct) = &r_ty {
                    let (recv_is_ptr, struct_name_opt) = match &ct.kind {
                        TypeKind::Ptr(inner) => match &inner.kind {
                            TypeKind::Named(s) => (true, Some(s.name.clone())),
                            _ => (true, None),
                        },
                        TypeKind::Named(s) => (false, Some(s.name.clone())),
                        _ => (false, None),
                    };
                    if let Some(struct_name) = struct_name_opt {
                        let resolved = if let Some(info) = self
                            .concrete_methods
                            .get(&(struct_name.clone(), method.name.clone()))
                            .cloned()
                        {
                            if !info.is_pub && info.owner_scope != caller_scope {
                                self.emit(
                                    &env.file,
                                    Diagnostic::new(
                                        DiagnosticKind::ModuleError,
                                        expr.span,
                                        format!(
                                            "method `{}` on `{}` is private; mark it `pub` to export it",
                                            method.name, struct_name
                                        ),
                                    ),
                                );
                            }
                            Some((info.fn_name, info.receiver, info.ret_ty))
                        } else if let Some((base_struct, conc_args)) =
                            self.instantiated_struct_args.get(&struct_name).cloned()
                        {
                            self.resolve_generic_method_call(
                                &base_struct,
                                &struct_name,
                                &conc_args,
                                &method.name,
                                expr.span,
                                caller_scope,
                                env,
                            )
                            .map(|(fn_name, rcv, rty, _, _)| (fn_name, rcv, rty))
                        } else {
                            None
                        };

                        if let Some((fn_name, recv_kind_opt, ret_ty)) = resolved {
                            match recv_kind_opt {
                                None => {
                                    self.emit(
                                        &env.file,
                                        Diagnostic::new(
                                            DiagnosticKind::InvalidOperation,
                                            expr.span,
                                            format!(
                                                "cannot call associated function `{}.{}` on an instance value",
                                                struct_name, method.name
                                            ),
                                        ),
                                    );
                                }
                                Some(recv_kind) => {
                                    let recv_arg = match (recv_kind, recv_is_ptr) {
                                        (SelfReceiver::Value, false) | (SelfReceiver::Ptr, true) => {
                                            rr.clone()
                                        }
                                        (SelfReceiver::Value, true) => Expr::new(
                                            ExprKind::Unary {
                                                op: UnaryOp::Deref,
                                                expr: Box::new(rr.clone()),
                                            },
                                            receiver.span,
                                        ),
                                        (SelfReceiver::Ptr, false) => Expr::new(
                                            ExprKind::Unary {
                                                op: UnaryOp::AddrOf,
                                                expr: Box::new(rr.clone()),
                                            },
                                            receiver.span,
                                        ),
                                    };
                                    let mut full_args = Vec::with_capacity(r_args.len() + 1);
                                    full_args.push(CallArg {
                                        name: None,
                                        value: recv_arg,
                                        span: receiver.span,
                                    });
                                    full_args.extend(r_args);
                                    let c_span = receiver.span.merge(method.span);
                                    return (
                                        Expr::new(
                                            ExprKind::Call {
                                                callee: Box::new(Expr::new(
                                                    ExprKind::Ident(Ident::new(fn_name, c_span)),
                                                    c_span,
                                                )),
                                                generic_args: Vec::new(),
                                                args: full_args,
                                            },
                                            expr.span,
                                        ),
                                        InferTy::Concrete(ret_ty),
                                    );
                                }
                            }
                        }
                    }
                }

                let mut out_ty = InferTy::Unknown;
                if let ExprKind::Ident(id) = &rr.kind {
                    if method.name == "from_bytes" && self.known_structs.contains_key(&id.name) {
                        out_ty =
                            InferTy::Concrete(TypeExpr::new(TypeKind::Named(id.clone()), expr.span));
                    }
                }
                if let InferTy::Concrete(ct) = &r_ty {
                    let deref = match &ct.kind {
                        TypeKind::Ptr(inner) => &**inner,
                        _ => ct,
                    };
                    if let TypeKind::OsPrimitive(os) = &deref.kind {
                        match (os, method.name.as_str()) {
                            (
                                OsPrimitiveType::Watch { elem },
                                "read" | "wait_change",
                            )
                            | (OsPrimitiveType::Guarded { elem }, "get")
                            | (
                                OsPrimitiveType::Inbox { elem, .. }
                                | OsPrimitiveType::InboxUnsized { elem },
                                "recv" | "try_recv",
                            ) => {
                                out_ty = InferTy::Concrete((**elem).clone());
                            }
                            (
                                OsPrimitiveType::Inbox { .. }
                                | OsPrimitiveType::InboxUnsized { .. },
                                "len" | "cap",
                            ) => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Prim(PrimType::Usize),
                                    expr.span,
                                ));
                            }
                            (
                                OsPrimitiveType::Inbox { .. }
                                | OsPrimitiveType::InboxUnsized { .. },
                                "try_send" | "is_empty" | "is_full",
                            ) => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Prim(PrimType::Bool),
                                    expr.span,
                                ));
                            }
                            (
                                OsPrimitiveType::PingPong { elem, .. },
                                "wait_chunk" | "acquire" | "try_acquire" | "active_buf" | "dma_buf",
                            ) => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Slice(elem.clone()),
                                    expr.span,
                                ));
                            }
                            (
                                OsPrimitiveType::Pool { elem, .. },
                                "try_alloc" | "alloc",
                            )
                            | (OsPrimitiveType::DmaSlot { elem }, "take") => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Lease(elem.clone()),
                                    expr.span,
                                ));
                            }
                            (
                                OsPrimitiveType::Pool { .. },
                                "avail" | "in_use" | "cap",
                            )
                            | (OsPrimitiveType::DmaSlot { .. }, "addr") => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Prim(PrimType::Usize),
                                    expr.span,
                                ));
                            }
                            (OsPrimitiveType::Pool { .. }, "is_exhausted")
                            | (
                                OsPrimitiveType::DmaSlot { .. },
                                "release" | "is_busy" | "is_empty",
                            ) => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Prim(PrimType::Bool),
                                    expr.span,
                                ));
                            }
                            (
                                OsPrimitiveType::ByteRing { .. }
                                | OsPrimitiveType::ByteRingUnsized,
                                "read_slice" | "wait_read" | "grant_write",
                            ) => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Slice(Box::new(TypeExpr::new(
                                        TypeKind::Prim(PrimType::U8),
                                        expr.span,
                                    ))),
                                    expr.span,
                                ));
                            }
                            (
                                OsPrimitiveType::ByteRing { .. }
                                | OsPrimitiveType::ByteRingUnsized,
                                "try_pop_byte",
                            ) => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Prim(PrimType::U8),
                                    expr.span,
                                ));
                            }
                            (
                                OsPrimitiveType::ByteRing { .. }
                                | OsPrimitiveType::ByteRingUnsized,
                                "push" | "pop" | "len" | "cap" | "free_space",
                            ) => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Prim(PrimType::Usize),
                                    expr.span,
                                ));
                            }
                            (
                                OsPrimitiveType::ByteRing { .. }
                                | OsPrimitiveType::ByteRingUnsized,
                                "push_byte" | "is_empty" | "is_full",
                            ) => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Prim(PrimType::Bool),
                                    expr.span,
                                ));
                            }
                            _ => {}
                        }
                    } else if let TypeKind::Lease(elem) = &deref.kind {
                        match method.name.as_str() {
                            "get" => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Ptr(elem.clone()),
                                    expr.span,
                                ));
                            }
                            "addr" => {
                                out_ty = InferTy::Concrete(TypeExpr::new(
                                    TypeKind::Prim(PrimType::Usize),
                                    expr.span,
                                ));
                            }
                            _ => {}
                        }
                    }
                }

                (
                    Expr::new(
                        ExprKind::MethodCall {
                            receiver: Box::new(rr),
                            method: method.clone(),
                            args: r_args,
                        },
                        expr.span,
                    ),
                    out_ty,
                )
            }

            ExprKind::FormatString(parts) => {
                let mut r_parts = Vec::with_capacity(parts.len());
                for p in parts {
                    match p {
                        FormatPart::Literal(s) => r_parts.push(FormatPart::Literal(s.clone())),
                        FormatPart::Hole {
                            expr: h_expr,
                            spec,
                            span: h_span,
                        } => {
                            let (rh, _) = self.rewrite_expr(h_expr, env, scopes, None);
                            r_parts.push(FormatPart::Hole {
                                expr: Box::new(rh),
                                spec: *spec,
                                span: *h_span,
                            });
                        }
                    }
                }
                (Expr::new(ExprKind::FormatString(r_parts), expr.span), InferTy::Unknown)
            }

            ExprKind::Or {
                expr: inner,
                fallback,
            } => {
                let (ri, it) = self.rewrite_expr(inner, env, scopes, expected_ty);
                let (rf, ft) = self.rewrite_expr(fallback, env, scopes, expected_ty);
                let out_ty = match (it, ft) {
                    (InferTy::Concrete(c), _) => InferTy::Concrete(c),
                    (_, InferTy::Concrete(c)) => InferTy::Concrete(c),
                    _ => InferTy::Unknown,
                };
                (
                    Expr::new(
                        ExprKind::Or {
                            expr: Box::new(ri),
                            fallback: Box::new(rf),
                        },
                        expr.span,
                    ),
                    out_ty,
                )
            }

            ExprKind::Catch {
                expr: inner,
                clause,
            } => {
                let (ri, it) = self.rewrite_expr(inner, env, scopes, expected_ty);
                scopes.push(BTreeMap::new());
                let err_param = clause.err_param.as_ref().map(|ep| {
                    let rty = self.rewrite_type_expr(&ep.ty, env);
                    if let Some(top) = scopes.last_mut() {
                        top.insert(ep.name.name.clone(), rty.clone());
                    }
                    CatchParam {
                        ty: rty,
                        name: ep.name.clone(),
                        span: ep.span,
                    }
                });
                let partial_param = clause.partial_param.as_ref().map(|pp| {
                    let rty = self.rewrite_type_expr(&pp.ty, env);
                    if let Some(top) = scopes.last_mut() {
                        top.insert(pp.name.name.clone(), rty.clone());
                    }
                    CatchParam {
                        ty: rty,
                        name: pp.name.clone(),
                        span: pp.span,
                    }
                });
                let body = self.rewrite_block(&clause.body, env, scopes);
                scopes.pop();
                (
                    Expr::new(
                        ExprKind::Catch {
                            expr: Box::new(ri),
                            clause: CatchClause {
                                err_param,
                                partial_param,
                                body,
                                span: clause.span,
                            },
                        },
                        expr.span,
                    ),
                    it,
                )
            }
        }
    }

    fn rewrite_call_expr(
        &mut self,
        callee: &Expr,
        generic_args: &[GenericArg],
        args: &[CallArg],
        span: Span,
        env: &SubstEnv,
        scopes: &mut Vec<BTreeMap<String, TypeExpr>>,
    ) -> (Expr, InferTy) {
        let mut r_args = Vec::with_capacity(args.len());
        let mut pos_arg_tys = Vec::new();
        for a in args {
            let (rv, aty) = self.rewrite_expr(&a.value, env, scopes, None);
            if a.name.is_none() {
                pos_arg_tys.push(aty);
            }
            r_args.push(CallArg {
                name: a.name.clone(),
                value: rv,
                span: a.span,
            });
        }

        const PAREN_CMP_NOTE: &str = "if `<` and `>` were intended as comparisons rather than generic arguments, parenthesize them: e.g. `f((a < b), (c > (d)))`";

        // Extract target generic function name (`clamp(...)` or `mod.clamp<f32>(...)`).
        // A local (e.g. a `fn(...)` pointer) shadowing a generic template is an indirect call.
        let direct_fn_ident: Option<Ident> = match &callee.kind {
            ExprKind::Ident(id) if !scopes_contain(scopes, &id.name) => Some(id.clone()),
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(base_id) = &target.kind {
                    if matches!(self.lookup_scope_ty(&base_id.name, env, scopes), InferTy::Unknown)
                        && self.fn_templates.contains_key(&field.name)
                    {
                        Some(field.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(fn_id) = direct_fn_ident {
            if let Some(tmpl) = self.fn_templates.get(&fn_id.name).cloned() {
                let resolved_gargs = if !generic_args.is_empty() {
                    self.resolve_explicit_generic_args(
                        &fn_id.name,
                        &tmpl.decl.generic_params,
                        generic_args,
                        env,
                        span,
                    )
                } else {
                    self.infer_fn_generic_args(&tmpl.decl, &pos_arg_tys, env, span)
                };

                if let Some(r_gargs) = resolved_gargs {
                    if let Some((mangled, ret_ty)) =
                        self.instantiate_fn(&fn_id.name, &r_gargs, span, env)
                    {
                        let new_callee =
                            Expr::new(ExprKind::Ident(Ident::new(mangled, fn_id.span)), callee.span);
                        return (
                            Expr::new(
                                ExprKind::Call {
                                    callee: Box::new(new_callee),
                                    generic_args: Vec::new(),
                                    args: r_args,
                                },
                                span,
                            ),
                            InferTy::Concrete(ret_ty),
                        );
                    }
                }

                // Error recovery: emit dummy concrete call so sema doesn't cascade an undefined symbol.
                let fallback_callee =
                    Expr::new(ExprKind::Ident(fn_id), callee.span);
                return (
                    Expr::new(
                        ExprKind::Call {
                            callee: Box::new(fallback_callee),
                            generic_args: Vec::new(),
                            args: r_args,
                        },
                        span,
                    ),
                    InferTy::Unknown,
                );
            }

            if !generic_args.is_empty() {
                self.emit(
                    &env.file,
                    Diagnostic::new(
                        DiagnosticKind::GenericInstantiationError,
                        span,
                        if self.non_generic_fn_rets.contains_key(&fn_id.name)
                            || matches!(
                                fn_id.name.as_str(),
                                "print"
                                    | "println"
                                    | "fmt_buf"
                                    | "fmt_ring"
                                    | "assert"
                                    | "delay"
                                    | "checkpoint"
                            )
                        {
                            format!(
                                "function '{}' is not generic (expected 0 generic arguments, found {})",
                                fn_id.name,
                                generic_args.len()
                            )
                        } else {
                            format!("unknown generic function '{}'", fn_id.name)
                        },
                    )
                    .with_note(PAREN_CMP_NOTE),
                );
            }

            if fn_id.name == "fmt_buf" {
                let slice_u8 = TypeExpr::new(
                    TypeKind::Slice(Box::new(TypeExpr::new(
                        TypeKind::Prim(PrimType::U8),
                        span,
                    ))),
                    span,
                );
                return (
                    Expr::new(
                        ExprKind::Call {
                            callee: Box::new(callee.clone()),
                            generic_args: Vec::new(),
                            args: r_args,
                        },
                        span,
                    ),
                    InferTy::Concrete(slice_u8),
                );
            }

            if fn_id.name == "fmt_ring" {
                let usize_ty = TypeExpr::new(TypeKind::Prim(PrimType::Usize), span);
                return (
                    Expr::new(
                        ExprKind::Call {
                            callee: Box::new(callee.clone()),
                            generic_args: Vec::new(),
                            args: r_args,
                        },
                        span,
                    ),
                    InferTy::Concrete(usize_ty),
                );
            }

            if let Some(ret_te) = self.non_generic_fn_rets.get(&fn_id.name).cloned() {
                return (
                    Expr::new(
                        ExprKind::Call {
                            callee: Box::new(callee.clone()),
                            generic_args: Vec::new(),
                            args: r_args,
                        },
                        span,
                    ),
                    InferTy::Concrete(ret_te),
                );
            }
        } else if !generic_args.is_empty() {
            self.emit(
                &env.file,
                Diagnostic::new(
                    DiagnosticKind::GenericInstantiationError,
                    span,
                    format!(
                        "cannot apply generic arguments `<...>` ({} argument(s)) to non-generic expression",
                        generic_args.len()
                    ),
                )
                .with_note(PAREN_CMP_NOTE),
            );
        }

        let (r_callee, _) = self.rewrite_expr(callee, env, scopes, None);
        (
            Expr::new(
                ExprKind::Call {
                    callee: Box::new(r_callee),
                    generic_args: Vec::new(),
                    args: r_args,
                },
                span,
            ),
            InferTy::Unknown,
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Generic Argument Unification / Inference (`clamp(x, 0.0, 1.0)`, `push(&vec, 42)`)
    // ─────────────────────────────────────────────────────────────────────────

    fn infer_fn_generic_args(
        &mut self,
        f: &FnDecl,
        arg_tys: &[InferTy],
        env: &SubstEnv,
        call_span: Span,
    ) -> Option<Vec<ResolvedGenArg>> {
        let mut param_kinds = BTreeMap::new();
        for gp in &f.generic_params {
            param_kinds.insert(gp.name.name.clone(), gp.kind);
        }

        let mut inferred_types: BTreeMap<String, InferTy> = BTreeMap::new();
        let mut inferred_consts: BTreeMap<String, usize> = BTreeMap::new();
        let mut conflict: Option<String> = None;

        for (p, actual) in f.params.iter().zip(arg_tys.iter()) {
            self.unify_pattern(
                &p.ty,
                actual,
                &param_kinds,
                &mut inferred_types,
                &mut inferred_consts,
                &mut conflict,
            );
        }

        if let Some(err_msg) = conflict {
            self.emit(
                &env.file,
                Diagnostic::new(DiagnosticKind::GenericInstantiationError, call_span, err_msg),
            );
            return None;
        }

        self.materialize_inferred_args(&f.name.name, &f.generic_params, &inferred_types, &inferred_consts, env, call_span)
    }

    fn infer_struct_lit_args(
        &mut self,
        st: &StructDecl,
        field_tys: &BTreeMap<String, InferTy>,
        env: &SubstEnv,
        lit_span: Span,
    ) -> Option<Vec<ResolvedGenArg>> {
        let mut param_kinds = BTreeMap::new();
        for gp in &st.generic_params {
            param_kinds.insert(gp.name.name.clone(), gp.kind);
        }

        let mut inferred_types: BTreeMap<String, InferTy> = BTreeMap::new();
        let mut inferred_consts: BTreeMap<String, usize> = BTreeMap::new();
        let mut conflict: Option<String> = None;

        for sf in &st.fields {
            if let Some(actual) = field_tys.get(&sf.name.name) {
                self.unify_pattern(
                    &sf.ty,
                    actual,
                    &param_kinds,
                    &mut inferred_types,
                    &mut inferred_consts,
                    &mut conflict,
                );
            }
        }

        if let Some(err_msg) = conflict {
            self.emit(
                &env.file,
                Diagnostic::new(DiagnosticKind::GenericInstantiationError, lit_span, err_msg),
            );
            return None;
        }

        self.materialize_inferred_args(&st.name.name, &st.generic_params, &inferred_types, &inferred_consts, env, lit_span)
    }

    fn materialize_inferred_args(
        &mut self,
        owner: &str,
        gparams: &[GenericParam],
        inferred_types: &BTreeMap<String, InferTy>,
        inferred_consts: &BTreeMap<String, usize>,
        env: &SubstEnv,
        span: Span,
    ) -> Option<Vec<ResolvedGenArg>> {
        let mut out = Vec::with_capacity(gparams.len());
        for gp in gparams {
            match gp.kind {
                GenericParamKind::Type => match inferred_types.get(&gp.name.name) {
                    Some(InferTy::Concrete(ct)) => {
                        if ct.is_void() {
                            self.emit(
                                &env.file,
                                Diagnostic::new(
                                    DiagnosticKind::GenericInstantiationError,
                                    span,
                                    format!(
                                        "inferred generic type argument '{}' for '{}' cannot be 'void'",
                                        gp.name.name, owner
                                    ),
                                ),
                            );
                            return None;
                        }
                        out.push(ResolvedGenArg::Type(ct.clone()));
                    }
                    Some(InferTy::FloatLit) => {
                        out.push(ResolvedGenArg::Type(TypeExpr::new(
                            TypeKind::Prim(PrimType::F64),
                            span,
                        )));
                    }
                    Some(InferTy::IntLit) => {
                        out.push(ResolvedGenArg::Type(TypeExpr::new(
                            TypeKind::Prim(PrimType::I32),
                            span,
                        )));
                    }
                    None | Some(InferTy::Unknown) => {
                        self.emit(
                            &env.file,
                            Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                span,
                                format!(
                                    "cannot infer generic type parameter '{}' for '{}'; provide explicit generic arguments '{}<...>'" ,
                                    gp.name.name, owner, owner
                                ),
                            ),
                        );
                        return None;
                    }
                },
                GenericParamKind::ConstUsize => match inferred_consts.get(&gp.name.name) {
                    Some(&n) => out.push(ResolvedGenArg::Const(n)),
                    None => {
                        self.emit(
                            &env.file,
                            Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                span,
                                format!(
                                    "cannot infer constant 'usize {}' parameter for '{}'; provide explicit generic arguments '{}<...>'" ,
                                    gp.name.name, owner, owner
                                ),
                            ),
                        );
                        return None;
                    }
                },
            }
        }
        Some(out)
    }

    fn unify_pattern(
        &self,
        pat: &TypeExpr,
        actual: &InferTy,
        param_kinds: &BTreeMap<String, GenericParamKind>,
        inferred_types: &mut BTreeMap<String, InferTy>,
        inferred_consts: &mut BTreeMap<String, usize>,
        conflict: &mut Option<String>,
    ) {
        if conflict.is_some() || matches!(actual, InferTy::Unknown) {
            return;
        }

        match &pat.kind {
            TypeKind::Named(id) => {
                if param_kinds.get(&id.name) == Some(&GenericParamKind::Type) {
                    match (inferred_types.get(&id.name).cloned(), actual) {
                        (None, act) => {
                            inferred_types.insert(id.name.clone(), act.clone());
                        }
                        (Some(InferTy::IntLit), InferTy::Concrete(ct)) if is_int_type_expr(ct) => {
                            inferred_types.insert(id.name.clone(), InferTy::Concrete(ct.clone()));
                        }
                        (Some(InferTy::Concrete(ct)), InferTy::IntLit) if is_int_type_expr(&ct) => {}
                        (Some(InferTy::IntLit), InferTy::IntLit) => {}
                        (Some(InferTy::FloatLit), InferTy::Concrete(ct))
                            if is_float_type_expr(ct) =>
                        {
                            inferred_types.insert(id.name.clone(), InferTy::Concrete(ct.clone()));
                        }
                        (Some(InferTy::Concrete(ct)), InferTy::FloatLit)
                            if is_float_type_expr(&ct) => {}
                        (Some(InferTy::FloatLit), InferTy::FloatLit) => {}
                        (Some(InferTy::Concrete(prev)), InferTy::Concrete(curr)) => {
                            if self.format_mangled_type(&prev) == self.format_mangled_type(curr) {
                                // Exact match
                            } else if is_int_type_expr(&prev) && is_int_type_expr(curr) {
                                // Keep the wider concrete integer type
                                if int_type_rank(curr) > int_type_rank(&prev) {
                                    inferred_types
                                        .insert(id.name.clone(), InferTy::Concrete(curr.clone()));
                                }
                            } else {
                                *conflict = Some(format!(
                                    "conflicting types '{}' and '{}' inferred for generic parameter '{}'",
                                    self.format_human_type(&prev),
                                    self.format_human_type(curr),
                                    id.name
                                ));
                            }
                        }
                        (Some(prev), act) => {
                            *conflict = Some(format!(
                                "conflicting argument types ({:?} vs {:?}) inferred for generic parameter '{}'",
                                prev, act, id.name
                            ));
                        }
                    }
                }
            }

            TypeKind::Ptr(p_inner) => {
                if let InferTy::Concrete(TypeExpr {
                    kind: TypeKind::Ptr(a_inner),
                    ..
                }) = actual
                {
                    self.unify_pattern(
                        p_inner,
                        &InferTy::Concrete((**a_inner).clone()),
                        param_kinds,
                        inferred_types,
                        inferred_consts,
                        conflict,
                    );
                }
            }

            TypeKind::Lease(p_inner) => {
                if let InferTy::Concrete(TypeExpr {
                    kind: TypeKind::Lease(a_inner),
                    ..
                }) = actual
                {
                    self.unify_pattern(
                        p_inner,
                        &InferTy::Concrete((**a_inner).clone()),
                        param_kinds,
                        inferred_types,
                        inferred_consts,
                        conflict,
                    );
                }
            }

            TypeKind::OsPrimitive(p_os) => {
                if let InferTy::Concrete(TypeExpr {
                    kind: TypeKind::OsPrimitive(a_os),
                    ..
                }) = actual
                {
                    match (p_os, a_os) {
                        (
                            OsPrimitiveType::Watch { elem: pe },
                            OsPrimitiveType::Watch { elem: ae },
                        )
                        | (
                            OsPrimitiveType::Guarded { elem: pe },
                            OsPrimitiveType::Guarded { elem: ae },
                        )
                        | (
                            OsPrimitiveType::DmaSlot { elem: pe },
                            OsPrimitiveType::DmaSlot { elem: ae },
                        )
                        | (
                            OsPrimitiveType::InboxUnsized { elem: pe },
                            OsPrimitiveType::InboxUnsized { elem: ae },
                        )
                        | (
                            OsPrimitiveType::InboxUnsized { elem: pe },
                            OsPrimitiveType::Inbox { elem: ae, .. },
                        ) => {
                            self.unify_pattern(
                                pe,
                                &InferTy::Concrete((**ae).clone()),
                                param_kinds,
                                inferred_types,
                                inferred_consts,
                                conflict,
                            );
                        }
                        (
                            OsPrimitiveType::Inbox { elem: pe, cap: pc },
                            OsPrimitiveType::Inbox { elem: ae, cap: ac },
                        )
                        | (
                            OsPrimitiveType::PingPong { elem: pe, cap: pc },
                            OsPrimitiveType::PingPong { elem: ae, cap: ac },
                        )
                        | (
                            OsPrimitiveType::Pool { elem: pe, cap: pc },
                            OsPrimitiveType::Pool { elem: ae, cap: ac },
                        ) if pc == ac => {
                            self.unify_pattern(
                                pe,
                                &InferTy::Concrete((**ae).clone()),
                                param_kinds,
                                inferred_types,
                                inferred_consts,
                                conflict,
                            );
                        }
                        _ => {}
                    }
                }
            }

            TypeKind::Slice(p_elem) => {
                if let InferTy::Concrete(TypeExpr { kind, .. }) = actual {
                    match kind {
                        TypeKind::Slice(a_elem) | TypeKind::Array { elem: a_elem, .. } => {
                            self.unify_pattern(
                                p_elem,
                                &InferTy::Concrete((**a_elem).clone()),
                                param_kinds,
                                inferred_types,
                                inferred_consts,
                                conflict,
                            );
                        }
                        _ => {}
                    }
                }
            }

            TypeKind::Array {
                elem: p_elem,
                len: p_len,
            } => {
                if let InferTy::Concrete(TypeExpr {
                    kind:
                        TypeKind::Array {
                            elem: a_elem,
                            len: ArrayLen::Fixed(a_n),
                        },
                    ..
                }) = actual
                {
                    self.unify_pattern(
                        p_elem,
                        &InferTy::Concrete((**a_elem).clone()),
                        param_kinds,
                        inferred_types,
                        inferred_consts,
                        conflict,
                    );
                    if let ArrayLen::Expr(len_expr) = p_len {
                        if let ExprKind::Ident(n_id) = &len_expr.kind {
                            if param_kinds.get(&n_id.name) == Some(&GenericParamKind::ConstUsize) {
                                if let Some(&prev_n) = inferred_consts.get(&n_id.name) {
                                    if prev_n != *a_n {
                                        *conflict = Some(format!(
                                            "conflicting array lengths {} and {} inferred for generic parameter 'usize {}'",
                                            prev_n, a_n, n_id.name
                                        ));
                                    }
                                } else {
                                    inferred_consts.insert(n_id.name.clone(), *a_n);
                                }
                            }
                        }
                    }
                }
            }

            TypeKind::Generic {
                base,
                args: pat_args,
            } => {
                if let InferTy::Concrete(TypeExpr {
                    kind: TypeKind::Named(mangled_id),
                    ..
                }) = actual
                {
                    if let Some((orig_base, inst_args)) =
                        self.instantiated_struct_args.get(&mangled_id.name)
                    {
                        if orig_base == &base.name && pat_args.len() == inst_args.len() {
                            for (pa, ia) in pat_args.iter().zip(inst_args.iter()) {
                                match (pa, ia) {
                                    (GenericArg::Type(p_te), ResolvedGenArg::Type(i_te)) => {
                                        self.unify_pattern(
                                            p_te,
                                            &InferTy::Concrete(i_te.clone()),
                                            param_kinds,
                                            inferred_types,
                                            inferred_consts,
                                            conflict,
                                        );
                                    }
                                    (GenericArg::Type(p_te), ResolvedGenArg::Const(i_n)) => {
                                        if let TypeKind::Named(cid) = &p_te.kind {
                                            if param_kinds.get(&cid.name)
                                                == Some(&GenericParamKind::ConstUsize)
                                            {
                                                if let Some(&prev_n) =
                                                    inferred_consts.get(&cid.name)
                                                {
                                                    if prev_n != *i_n {
                                                        *conflict = Some(format!(
                                                            "conflicting bounds {} and {} inferred for generic parameter 'usize {}'",
                                                            prev_n, i_n, cid.name
                                                        ));
                                                    }
                                                } else {
                                                    inferred_consts.insert(cid.name.clone(), *i_n);
                                                }
                                            }
                                        }
                                    }
                                    (GenericArg::Const(Expr {
                                        kind: ExprKind::Ident(cid),
                                        ..
                                    }), ResolvedGenArg::Const(i_n)) => {
                                        if param_kinds.get(&cid.name)
                                            == Some(&GenericParamKind::ConstUsize)
                                        {
                                            if let Some(&prev_n) = inferred_consts.get(&cid.name) {
                                                if prev_n != *i_n {
                                                    *conflict = Some(format!(
                                                        "conflicting bounds {} and {} inferred for generic parameter 'usize {}'",
                                                        prev_n, i_n, cid.name
                                                    ));
                                                }
                                            } else {
                                                inferred_consts.insert(cid.name.clone(), *i_n);
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }

            _ => {}
        }
    }
}

/// Escape every `_` inside a user or primitive identifier segment as `_0` for collision-free mangling.
fn escape_mangle_ident(s: &str) -> String {
    s.replace('_', "_0")
}

fn mangle_generic_arg_ast(arg: &GenericArg) -> String {
    match arg {
        GenericArg::Type(te) => mangle_type_expr(te),
        GenericArg::Const(Expr {
            kind: ExprKind::Literal(Literal::Int { value: v, .. }),
            ..
        }) => v.to_string(),
        GenericArg::Const(Expr {
            kind: ExprKind::Ident(id),
            ..
        }) => escape_mangle_ident(&id.name),
        GenericArg::Const(_) => "expr".to_string(),
    }
}

/// Collision-free injective type mangler (`H4`):
/// - Identifier/Named: `_` escaped as `_0` (`u32`, `my_0type`, `_0leading`)
/// - `*T`: `_P<mangle(T)>`
/// - `[]T`: `_S<mangle(T)>`
/// - `[N]T`: `_A<N>_<mangle(T)>`
/// - Path `a.b`: `<esc(a)>_D<esc(b)>`
/// - Nested generic `Base<A, B>`: `_G<esc(Base)>_L_<mangle(A)>__<mangle(B)>_R_`
/// - `fn(P1, P2) -> R`: `_F<arity>_<mangle(P1)>__<mangle(P2)>_T_<mangle(R)>`
pub fn mangle_type_expr(te: &TypeExpr) -> String {
    match &te.kind {
        TypeKind::Prim(p) => escape_mangle_ident(p.as_str()),
        TypeKind::Named(id) => id
            .name
            .split('.')
            .map(escape_mangle_ident)
            .collect::<Vec<_>>()
            .join("_D"),
        TypeKind::Path(segs) => segs
            .iter()
            .map(|s| escape_mangle_ident(&s.name))
            .collect::<Vec<_>>()
            .join("_D"),
        TypeKind::Ptr(inner) => format!("_P{}", mangle_type_expr(inner)),
        TypeKind::Lease(inner) => format!("_GLease_L_{}_R_", mangle_type_expr(inner)),
        TypeKind::Slice(inner) => format!("_S{}", mangle_type_expr(inner)),
        TypeKind::Array { elem, len } => {
            let n = len.as_fixed().unwrap_or(0);
            format!("_A{}_{}", n, mangle_type_expr(elem))
        }
        TypeKind::OsPrimitive(os) => match os {
            OsPrimitiveType::Signal => "signal".to_string(),
            OsPrimitiveType::WaitQueue => "wait_queue".to_string(),
            OsPrimitiveType::Watch { elem } => format!("_Gwatch_L_{}_R_", mangle_type_expr(elem)),
            OsPrimitiveType::Guarded { elem } => {
                format!("_Gguarded_L_{}_R_", mangle_type_expr(elem))
            }
            OsPrimitiveType::Inbox { elem, cap } => {
                format!("_Ginbox_L_{}__{}_R_", mangle_type_expr(elem), cap)
            }
            OsPrimitiveType::InboxUnsized { elem } => {
                format!("_Ginbox_L_{}_R_", mangle_type_expr(elem))
            }
            OsPrimitiveType::PingPong { elem, cap } => {
                format!("_Gping_0pong_L_{}__{}_R_", mangle_type_expr(elem), cap)
            }
            OsPrimitiveType::Pool { elem, cap } => {
                format!("_Gpool_L_{}__{}_R_", mangle_type_expr(elem), cap)
            }
            OsPrimitiveType::DmaSlot { elem } => {
                format!("_Gdma_0slot_L_{}_R_", mangle_type_expr(elem))
            }
            OsPrimitiveType::ByteRing { cap } => {
                format!("_Gbyte_0ring_L_{}_R_", cap)
            }
            OsPrimitiveType::ByteRingUnsized => "byte_ring".to_string(),
        },
        TypeKind::FnPtr { params, ret } => {
            let ps: Vec<String> = params.iter().map(mangle_type_expr).collect();
            format!(
                "_F{}_{}_T_{}",
                params.len(),
                ps.join("__"),
                mangle_type_expr(ret)
            )
        }
        TypeKind::Generic { base, args } => {
            let base_esc = base
                .name
                .split('.')
                .map(escape_mangle_ident)
                .collect::<Vec<_>>()
                .join("_D");
            let arg_strs: Vec<String> = args.iter().map(mangle_generic_arg_ast).collect();
            format!("_G{}_L_{}_R_", base_esc, arg_strs.join("__"))
        }
        TypeKind::GenericPath { path, args } => {
            let path_esc = path
                .iter()
                .map(|s| escape_mangle_ident(&s.name))
                .collect::<Vec<_>>()
                .join("_D");
            let arg_strs: Vec<String> = args.iter().map(mangle_generic_arg_ast).collect();
            format!("_G{}_L_{}_R_", path_esc, arg_strs.join("__"))
        }
    }
}

fn scopes_contain(scopes: &[BTreeMap<String, TypeExpr>], name: &str) -> bool {
    scopes.iter().any(|sc| sc.contains_key(name))
}

pub(crate) const fn truncate_int_prim(v: i128, p: PrimType) -> i128 {
    match p {
        PrimType::U8 => (v as u8) as i128,
        PrimType::I8 => (v as i8) as i128,
        PrimType::U16 | PrimType::U16Be | PrimType::U16Le => (v as u16) as i128,
        PrimType::I16 => (v as i16) as i128,
        PrimType::U32 | PrimType::U32Be | PrimType::U32Le => (v as u32) as i128,
        PrimType::I32 => (v as i32) as i128,
        PrimType::U64 | PrimType::U64Be | PrimType::U64Le | PrimType::Usize => (v as u64) as i128,
        PrimType::I64 | PrimType::Isize => (v as i64) as i128,
        PrimType::Bool => (v != 0) as i128,
        PrimType::F32 | PrimType::F64 | PrimType::Void => v,
    }
}

fn is_int_type_expr(te: &TypeExpr) -> bool {
    matches!(te.kind, TypeKind::Prim(p) if p.is_integer())
}

fn is_float_type_expr(te: &TypeExpr) -> bool {
    matches!(te.kind, TypeKind::Prim(p) if p.is_float())
}

fn int_type_rank(te: &TypeExpr) -> usize {
    match te.kind {
        TypeKind::Prim(p) => p.byte_size(4),
        _ => 0,
    }
}
