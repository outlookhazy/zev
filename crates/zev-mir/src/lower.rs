//! Lowering pass from `zev_ast::Program` into Coroutine State-Machine & Kernel MIR (`MirModule`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use zev_ast::{
    AssignOp, Attr, AttrKind, BinaryOp, Block, CallArg, DurationLit, ErrorSpec, Expr, ExprKind,
    FnDecl, IsrDecl, ItemKind, Literal, OsPrimitiveType, PrimType, Priority, Program, Span, Stmt,
    StmtKind, SwitchPat, TaskDecl, TypeExpr, TypeKind, UnaryOp,
};

use crate::ir::{
    BlockId, MirAssignOp, MirBinOp, MirBlock, MirBootSequence, MirBootStep, MirCallDst,
    MirChildCallSlot, MirConst, MirEnum, MirErrorSpec, MirExpr, MirExprKind, MirExternFn, MirField,
    MirFnBody, MirFunction, MirImport, MirIntrinsicOp, MirIsr, MirModule, MirPlace, MirPrimKind,
    MirPrimNonBlockingOp, MirPrimReadOp, MirPrimTarget, MirPrimitive, MirPriority,
    MirSpawnInstance, MirStmt, MirStruct, MirSwitchPattern, MirTaskBlueprint, MirTerminator,
    MirType, MirUnaryOp, MirVar, MirYieldKind, VarId, VarRole,
};
use crate::liveness::{analyze_fn_frame, analyze_task_frame};
use crate::ram::{populate_module_ram_report, RamReport};

/// Error produced during AST -> MIR lowering if an invalid construct reaches MIR without `zev-sema`.
#[derive(Debug, Clone)]
pub struct MirLowerError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for MirLowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: MIR lowering error: {}", self.span, self.message)
    }
}

impl std::error::Error for MirLowerError {}

#[derive(Clone)]
struct FnSig {
    params: Vec<MirType>,
    ret_ty: MirType,
    error_spec: MirErrorSpec,
    is_extern_c: bool,
    is_yielding: bool,
}

#[derive(Clone)]
struct PrimInfo {
    kind: MirPrimKind,
    dma_safe: bool,
    sleep_lock_bit: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstVal {
    Int(i128),
    Bool(bool),
}

/// Global module symbol table used during lowering.
struct ModuleSymbols {
    tick_period_us: u64,
    systick_hz: Option<u64>,
    import_aliases: BTreeMap<String, Vec<String>>,
    enums: BTreeMap<String, Vec<(String, i64)>>,
    variant_to_enum: BTreeMap<String, (String, i64)>,
    structs: BTreeMap<String, MirStruct>,
    consts: BTreeMap<String, MirType>,
    const_vals: BTreeMap<String, ConstVal>,
    primitives: BTreeMap<String, PrimInfo>,
    fns: BTreeMap<String, FnSig>,
    tasks: BTreeMap<String, Vec<MirType>>,
}

impl ModuleSymbols {
    fn resolve_type(&self, te: &TypeExpr) -> MirType {
        match &te.kind {
            TypeKind::Prim(pt) => convert_prim_type(*pt),
            TypeKind::Named(id) => {
                let n = id.name.as_str();
                if let Some(pt) = PrimType::from_name(n) {
                    return convert_prim_type(pt);
                }
                if n == "str" {
                    return MirType::Str;
                }
                if self.enums.contains_key(n) {
                    MirType::Enum(id.name.clone())
                } else if self.structs.contains_key(n) {
                    MirType::Struct(id.name.clone())
                } else {
                    // Could be hardware handle typedef or external opaque type
                    MirType::HwPeripheral(id.name.clone())
                }
            }
            TypeKind::Path(segs) => {
                let joined: Vec<String> = segs.iter().map(|s| s.name.clone()).collect();
                if joined.len() == 1 {
                    let n = &joined[0];
                    if self.enums.contains_key(n) {
                        return MirType::Enum(n.clone());
                    }
                    if self.structs.contains_key(n) {
                        return MirType::Struct(n.clone());
                    }
                }
                MirType::HwPeripheral(joined.join("."))
            }
            TypeKind::Slice(inner) => MirType::Slice(Box::new(self.resolve_type(inner))),
            TypeKind::Array { elem, len } => MirType::Array {
                elem: Box::new(self.resolve_type(elem)),
                len: *len,
            },
            TypeKind::Ptr(inner) => MirType::Ptr(Box::new(self.resolve_type(inner))),
            TypeKind::OsPrimitive(op) => MirType::Primitive(self.resolve_os_prim_type(op)),
            TypeKind::FnPtr { params, ret } => MirType::FnPtr {
                params: params.iter().map(|p| self.resolve_type(p)).collect(),
                ret: Box::new(self.resolve_type(ret)),
            },
        }
    }

    fn resolve_os_prim_type(&self, op: &OsPrimitiveType) -> MirPrimKind {
        match op {
            OsPrimitiveType::Signal => MirPrimKind::Signal,
            OsPrimitiveType::Watch { elem } => MirPrimKind::Watch {
                elem: Box::new(self.resolve_type(elem)),
            },
            OsPrimitiveType::Inbox { elem, cap } => MirPrimKind::Inbox {
                elem: Box::new(self.resolve_type(elem)),
                cap: *cap,
            },
            OsPrimitiveType::Chan { elem, cap } => MirPrimKind::Chan {
                elem: Box::new(self.resolve_type(elem)),
                cap: *cap,
            },
            OsPrimitiveType::PingPong { elem, cap } => MirPrimKind::PingPong {
                elem: Box::new(self.resolve_type(elem)),
                cap: *cap,
            },
            OsPrimitiveType::Lease { elem, cap } => MirPrimKind::Lease {
                elem: Box::new(self.resolve_type(elem)),
                cap: *cap,
            },
        }
    }

    fn resolve_error_spec(&self, spec: &Option<ErrorSpec>) -> MirErrorSpec {
        match spec {
            None => MirErrorSpec::None,
            Some(ErrorSpec::Bare(_)) => MirErrorSpec::Untyped,
            Some(ErrorSpec::Typed(te)) => MirErrorSpec::Typed(self.resolve_type(te)),
        }
    }

    fn duration_to_ticks(&self, dur: &DurationLit) -> Result<u64, MirLowerError> {
        let us = dur.as_micros();
        if us == 0 {
            return Ok(0);
        }
        if us < self.tick_period_us || us % self.tick_period_us != 0 {
            return Err(MirLowerError {
                message: format!(
                    "duration `{dur}` ({us}us) is not an exact multiple of tick resolution ({}us)",
                    self.tick_period_us
                ),
                span: dur.span,
            });
        }
        Ok(us / self.tick_period_us)
    }

    fn is_hw_root(&self, ident: &str) -> bool {
        ident == "hw" || self.import_aliases.contains_key(ident)
    }
}

fn convert_prim_type(pt: PrimType) -> MirType {
    match pt {
        PrimType::Void => MirType::Void,
        PrimType::Bool => MirType::Bool,
        PrimType::U8 => MirType::U8,
        PrimType::U16 => MirType::U16,
        PrimType::U32 => MirType::U32,
        PrimType::U64 => MirType::U64,
        PrimType::Usize => MirType::Usize,
        PrimType::I8 => MirType::I8,
        PrimType::I16 => MirType::I16,
        PrimType::I32 => MirType::I32,
        PrimType::I64 => MirType::I64,
        PrimType::Isize => MirType::Isize,
        PrimType::U16Be => MirType::EndianInt {
            bits: 16,
            signed: false,
            big_endian: true,
        },
        PrimType::U16Le => MirType::EndianInt {
            bits: 16,
            signed: false,
            big_endian: false,
        },
        PrimType::U32Be => MirType::EndianInt {
            bits: 32,
            signed: false,
            big_endian: true,
        },
        PrimType::U32Le => MirType::EndianInt {
            bits: 32,
            signed: false,
            big_endian: false,
        },
        PrimType::U64Be => MirType::EndianInt {
            bits: 64,
            signed: false,
            big_endian: true,
        },
        PrimType::U64Le => MirType::EndianInt {
            bits: 64,
            signed: false,
            big_endian: false,
        },
    }
}

fn convert_priority(p: Priority) -> MirPriority {
    match p {
        Priority::Critical => MirPriority::Critical,
        Priority::High => MirPriority::High,
        Priority::Normal => MirPriority::Normal,
        Priority::Low => MirPriority::Low,
    }
}

fn has_attr_dma_safe(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| matches!(a.kind, AttrKind::DmaSafe))
}

fn has_attr_manual_yield(attrs: &[Attr]) -> bool {
    attrs
        .iter()
        .any(|a| matches!(a.kind, AttrKind::ManualYield))
}

/// Lower multiple parsed `zev_ast::Program`s into a single optimized `MirModule`.
pub fn lower_programs(programs: &[Program]) -> Result<MirModule, MirLowerError> {
    let merged = Program::merge_all(programs.iter().cloned());
    lower_program(&merged)
}

/// Lower a parsed `zev_ast::Program` into an optimized `MirModule` with static `RamReport`.
pub fn lower_program(program: &Program) -> Result<MirModule, MirLowerError> {
    let mut syms = ModuleSymbols {
        tick_period_us: 1_000,
        systick_hz: None,
        import_aliases: BTreeMap::new(),
        enums: BTreeMap::new(),
        variant_to_enum: BTreeMap::new(),
        structs: BTreeMap::new(),
        consts: BTreeMap::new(),
        const_vals: BTreeMap::new(),
        primitives: BTreeMap::new(),
        fns: BTreeMap::new(),
        tasks: BTreeMap::new(),
    };

    let mut mir_imports = Vec::new();
    let mut mir_enums = Vec::new();

    // Pass 1a: Imports & Enums (so Structs/Primitives/Fns can reference Enum types).
    for item in &program.items {
        match &item.kind {
            ItemKind::Import(imp) => {
                let path: Vec<String> = imp.path.iter().map(|i| i.name.clone()).collect();
                let alias = imp.alias.as_ref().map(|a| a.name.clone());
                if let Some(al) = &alias {
                    syms.import_aliases.insert(al.clone(), path.clone());
                } else if let Some(last) = path.last() {
                    syms.import_aliases.insert(last.clone(), path.clone());
                }
                mir_imports.push(MirImport { path, alias });
            }
            ItemKind::Enum(ed) => {
                let mut next_disc = 0i64;
                let mut variants = Vec::new();
                for v in &ed.variants {
                    let d = v.discriminant.unwrap_or(next_disc);
                    next_disc = d + 1;
                    variants.push((v.name.name.clone(), d));
                    syms.variant_to_enum
                        .insert(v.name.name.clone(), (ed.name.name.clone(), d));
                }
                syms.enums.insert(ed.name.name.clone(), variants.clone());
                mir_enums.push(MirEnum {
                    name: ed.name.name.clone(),
                    variants,
                });
            }
            _ => {}
        }
    }

    // Pass 1b: Struct names first (so pointers/fields can reference any struct), then fields.
    for item in &program.items {
        if let ItemKind::Struct(sd) = &item.kind {
            syms.structs.insert(
                sd.name.name.clone(),
                MirStruct {
                    name: sd.name.name.clone(),
                    packed: sd.is_packed,
                    fields: Vec::new(),
                },
            );
        }
    }

    let mut mir_structs = Vec::new();
    for item in &program.items {
        if let ItemKind::Struct(sd) = &item.kind {
            let fields: Vec<MirField> = sd
                .fields
                .iter()
                .map(|f| MirField {
                    name: f.name.name.clone(),
                    ty: syms.resolve_type(&f.ty),
                })
                .collect();
            let ms = MirStruct {
                name: sd.name.name.clone(),
                packed: sd.is_packed,
                fields,
            };
            syms.structs.insert(sd.name.name.clone(), ms.clone());
            mir_structs.push(ms);
        }
    }

    // Pass 1c: Constants, OS Primitives, Function & Task Signatures.
    let mut next_sleep_lock_bit = 0u8;
    let mut raw_fn_decls: Vec<&FnDecl> = Vec::new();
    let mut main_decl: Option<&FnDecl> = None;
    let mut raw_task_decls: Vec<&TaskDecl> = Vec::new();
    let mut raw_isr_decls: Vec<&IsrDecl> = Vec::new();
    let mut mir_extern_fns: Vec<MirExternFn> = Vec::new();

    for item in &program.items {
        match &item.kind {
            ItemKind::Const(cd) => {
                syms.consts
                    .insert(cd.name.name.clone(), syms.resolve_type(&cd.ty));
                if let Some(cv) = eval_const_expr(&cd.value, &syms) {
                    syms.const_vals.insert(cd.name.name.clone(), cv);
                }
            }
            ItemKind::OsPrimitive(pd) => {
                let kind = syms.resolve_os_prim_type(&pd.ty);
                let dma_safe = has_attr_dma_safe(&item.attrs) || has_attr_dma_safe(&pd.attrs);
                let sleep_lock_bit = if kind.holds_dma_sleep_lock() {
                    let bit = next_sleep_lock_bit;
                    next_sleep_lock_bit = next_sleep_lock_bit.saturating_add(1).min(31);
                    Some(bit)
                } else {
                    None
                };
                syms.primitives.insert(
                    pd.name.name.clone(),
                    PrimInfo {
                        kind,
                        dma_safe,
                        sleep_lock_bit,
                    },
                );
            }
            ItemKind::Fn(fd) => {
                let params: Vec<MirType> =
                    fd.params.iter().map(|p| syms.resolve_type(&p.ty)).collect();
                let ret_ty = syms.resolve_type(&fd.ret_type);
                let error_spec = syms.resolve_error_spec(&fd.error_spec);
                let is_extern = fd.is_extern_c || fd.body.is_none();

                if is_extern {
                    mir_extern_fns.push(MirExternFn {
                        name: fd.name.name.clone(),
                        params: fd
                            .params
                            .iter()
                            .map(|p| (p.name.name.clone(), syms.resolve_type(&p.ty)))
                            .collect(),
                        ret_ty: ret_ty.clone(),
                        variadic: fd.is_variadic,
                    });
                }

                syms.fns.insert(
                    fd.name.name.clone(),
                    FnSig {
                        params,
                        ret_ty,
                        error_spec,
                        is_extern_c: is_extern,
                        is_yielding: false,
                    },
                );

                if !is_extern {
                    if fd.name.name == "main" {
                        main_decl = Some(fd);
                    } else {
                        raw_fn_decls.push(fd);
                    }
                }
            }
            ItemKind::Task(td) => {
                let params: Vec<MirType> =
                    td.params.iter().map(|p| syms.resolve_type(&p.ty)).collect();
                syms.tasks.insert(td.name.name.clone(), params);
                raw_task_decls.push(td);
            }
            ItemKind::Isr(id) => {
                raw_isr_decls.push(id);
            }
            ItemKind::GlobalVar(gv) => {
                return Err(MirLowerError {
                    message: format!(
                        "mutable raw global `{}` is forbidden; use an OS primitive (`watch`, `inbox`, `chan`, `signal`, `ping_pong`)",
                        gv.name.name
                    ),
                    span: gv.span,
                });
            }
            ItemKind::Import(_) | ItemKind::Enum(_) | ItemKind::Struct(_) => {}
        }
    }

    // Discover `sys.use_systick(hz)` in `main` if present.
    if let Some(md) = main_decl {
        if let Some(body) = &md.body {
            discover_systick_config(&body.stmts, &mut syms);
        }
    }

    // Pass 2: Whole-program call-graph yield inference across `raw_fn_decls`.
    infer_yielding_functions(&raw_fn_decls, &mut syms);

    // Pass 3: Lower constants & top-level primitive initializers.
    let mut mir_consts = Vec::new();
    let mut mir_primitives = Vec::new();
    for item in &program.items {
        match &item.kind {
            ItemKind::Const(cd) => {
                let ty = syms.resolve_type(&cd.ty);
                let mut dummy_ctx = DirectLowerCtx::new(&syms, MirType::Void, MirErrorSpec::None);
                let init = dummy_ctx.lower_pure_expr(&cd.value, Some(&ty))?;
                mir_consts.push(MirConst {
                    name: cd.name.name.clone(),
                    ty,
                    init,
                });
            }
            ItemKind::OsPrimitive(pd) => {
                let pinfo = syms.primitives.get(&pd.name.name).unwrap().clone();
                let init = if let Some(ie) = &pd.init {
                    let elem_ty = match &pinfo.kind {
                        MirPrimKind::Watch { elem } => Some(elem.as_ref()),
                        _ => None,
                    };
                    let mut dummy_ctx =
                        DirectLowerCtx::new(&syms, MirType::Void, MirErrorSpec::None);
                    Some(dummy_ctx.lower_pure_expr(ie, elem_ty)?)
                } else {
                    None
                };
                mir_primitives.push(MirPrimitive {
                    name: pd.name.name.clone(),
                    kind: pinfo.kind,
                    dma_safe: pinfo.dma_safe,
                    init,
                    sleep_lock_bit: pinfo.sleep_lock_bit,
                });
            }
            _ => {}
        }
    }

    // Pass 4: Lower ISRs (`noyield` direct statements).
    let mut mir_isrs = Vec::new();
    for id in raw_isr_decls {
        let mut dctx = DirectLowerCtx::new(&syms, MirType::Void, MirErrorSpec::None);
        dctx.in_noyield = true;
        let body = dctx.lower_block_stmts(&id.body.stmts)?;
        mir_isrs.push(MirIsr {
            name: id.vector.name.clone(),
            vars: dctx.vars,
            body,
        });
    }

    // Pass 5: Lower Functions (both Direct non-yielding and Coroutine yielding functions).
    let mut mir_functions = Vec::new();
    for fd in raw_fn_decls {
        let sig = syms.fns.get(&fd.name.name).unwrap().clone();
        let manual_yield = has_attr_manual_yield(&fd.attrs);
        let body_ast = fd.body.as_ref().unwrap();

        let mut func = if sig.is_yielding {
            let mut cctx = CoroutineLowerCtx::new(
                &syms,
                sig.ret_ty.clone(),
                sig.error_spec.clone(),
                manual_yield,
            );
            let mut param_ids = Vec::new();
            for p in &fd.params {
                let pty = syms.resolve_type(&p.ty);
                let vid = cctx.declare_var(&p.name.name, pty, VarRole::Param);
                param_ids.push(vid);
            }
            cctx.lower_stmts(&body_ast.stmts)?;
            cctx.finish_with_implicit_return();

            MirFunction {
                name: fd.name.name.clone(),
                params: param_ids,
                ret_ty: sig.ret_ty,
                error_spec: sig.error_spec,
                manual_yield,
                is_yielding: true,
                vars: cctx.vars,
                body: MirFnBody::Coroutine {
                    blocks: cctx.blocks,
                    child_slots: cctx.child_slots,
                },
            }
        } else {
            let mut dctx = DirectLowerCtx::new(&syms, sig.ret_ty.clone(), sig.error_spec.clone());
            let mut param_ids = Vec::new();
            for p in &fd.params {
                let pty = syms.resolve_type(&p.ty);
                let vid = dctx.declare_var(&p.name.name, pty, VarRole::Param);
                param_ids.push(vid);
            }
            let stmts = dctx.lower_block_stmts(&body_ast.stmts)?;
            MirFunction {
                name: fd.name.name.clone(),
                params: param_ids,
                ret_ty: sig.ret_ty,
                error_spec: sig.error_spec,
                manual_yield,
                is_yielding: false,
                vars: dctx.vars,
                body: MirFnBody::Direct(stmts),
            }
        };

        analyze_fn_frame(&mut func);
        mir_functions.push(func);
    }

    // Pass 6: Lower Reusable `task` Blueprints.
    let mut mir_blueprints = Vec::new();
    for td in raw_task_decls {
        let manual_yield = has_attr_manual_yield(&td.attrs);
        let mut cctx =
            CoroutineLowerCtx::new(&syms, MirType::Void, MirErrorSpec::None, manual_yield);
        let mut param_ids = Vec::new();
        for p in &td.params {
            let pty = syms.resolve_type(&p.ty);
            let vid = cctx.declare_var(&p.name.name, pty, VarRole::Param);
            param_ids.push(vid);
        }
        cctx.lower_stmts(&td.body.stmts)?;
        cctx.finish_with_implicit_return();

        let mut bp = MirTaskBlueprint {
            name: td.name.name.clone(),
            is_main_supervisor: false,
            manual_yield,
            params: param_ids,
            vars: cctx.vars,
            blocks: cctx.blocks,
            child_slots: cctx.child_slots,
        };
        analyze_task_frame(&mut bp);
        mir_blueprints.push(bp);
    }

    // Pass 7: Lower `fn main()` into Sequential Boot + Static Spawn Slots + Optional Supervisor Task.
    let mut spawned_tasks = Vec::new();
    let boot = if let Some(md) = main_decl {
        lower_main_entry(
            md,
            &syms,
            &mut mir_blueprints,
            &mut spawned_tasks,
        )?
    } else {
        MirBootSequence {
            present: false,
            vars: Vec::new(),
            steps: Vec::new(),
            boot_yielding_callees: Vec::new(),
        }
    };

    let mut module = MirModule {
        tick_period_us: syms.tick_period_us,
        systick_hz: syms.systick_hz,
        imports: mir_imports,
        consts: mir_consts,
        enums: mir_enums,
        structs: mir_structs,
        primitives: mir_primitives,
        extern_fns: mir_extern_fns,
        isrs: mir_isrs,
        functions: mir_functions,
        task_blueprints: mir_blueprints,
        spawned_tasks,
        boot,
        ram_report: RamReport::default(),
    };

    populate_module_ram_report(&mut module);
    Ok(module)
}

// ─────────────────────────────────────────────────────────────────────────────
// Call-Graph Yield Inference & `main()` Splitting Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn discover_systick_config(stmts: &[Stmt], syms: &mut ModuleSymbols) {
    for stmt in stmts {
        if let StmtKind::Expr(Expr {
            kind: ExprKind::MethodCall {
                receiver,
                method,
                args,
            },
            ..
        }) = &stmt.kind
        {
            if let ExprKind::Ident(id) = &receiver.kind {
                if id.name == "sys" && method.name == "use_systick" {
                    if let Some(first) = args.first() {
                        if let ExprKind::Literal(Literal::Int(hz)) = &first.value.kind {
                            syms.systick_hz = Some(*hz);
                        }
                    }
                } else if id.name == "sys" && method.name == "set_tick_us" {
                    if let Some(first) = args.first() {
                        if let ExprKind::Literal(Literal::Int(us)) = &first.value.kind {
                            if *us > 0 {
                                syms.tick_period_us = *us;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn infer_yielding_functions(fn_decls: &[&FnDecl], syms: &mut ModuleSymbols) {
    let mut direct_yield: BTreeSet<String> = BTreeSet::new();
    let mut callees_of: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for fd in fn_decls {
        let manual = has_attr_manual_yield(&fd.attrs);
        let mut direct = false;
        let mut callees = BTreeSet::new();
        if let Some(body) = &fd.body {
            scan_stmts_yield(&body.stmts, manual, false, syms, &mut direct, &mut callees);
        }
        if direct {
            direct_yield.insert(fd.name.name.clone());
        }
        callees_of.insert(fd.name.name.clone(), callees);
    }

    let mut yielding = direct_yield;
    let mut changed = true;
    while changed {
        changed = false;
        for (fname, callees) in &callees_of {
            if !yielding.contains(fname) && callees.iter().any(|c| yielding.contains(c)) {
                yielding.insert(fname.clone());
                changed = true;
            }
        }
    }

    for yname in yielding {
        if let Some(sig) = syms.fns.get_mut(&yname) {
            if !sig.is_extern_c {
                sig.is_yielding = true;
            }
        }
    }
}

fn scan_stmts_yield(
    stmts: &[Stmt],
    manual_yield: bool,
    in_noyield: bool,
    syms: &ModuleSymbols,
    direct: &mut bool,
    callees: &mut BTreeSet<String>,
) {
    for stmt in stmts {
        let stmt_manual = manual_yield || has_attr_manual_yield(&stmt.attrs);
        match &stmt.kind {
            StmtKind::Every { body, .. } => {
                *direct = true;
                scan_stmts_yield(&body.stmts, stmt_manual, in_noyield, syms, direct, callees);
            }
            StmtKind::Checkpoint => {
                *direct = true;
            }
            StmtKind::Loop { body } => {
                if !stmt_manual && !in_noyield {
                    *direct = true;
                }
                scan_stmts_yield(&body.stmts, stmt_manual, in_noyield, syms, direct, callees);
            }
            StmtKind::While { cond, body } => {
                scan_expr_yield(cond, syms, direct, callees);
                scan_stmts_yield(&body.stmts, stmt_manual, in_noyield, syms, direct, callees);
            }
            StmtKind::NoYield { body } | StmtKind::Critical { body } => {
                scan_stmts_yield(&body.stmts, stmt_manual, true, syms, direct, callees);
            }
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                scan_expr_yield(cond, syms, direct, callees);
                scan_stmts_yield(
                    &then_block.stmts,
                    stmt_manual,
                    in_noyield,
                    syms,
                    direct,
                    callees,
                );
                if let Some(eb) = else_block {
                    scan_stmts_yield(&eb.stmts, stmt_manual, in_noyield, syms, direct, callees);
                }
            }
            StmtKind::Switch { scrutinee, arms } => {
                scan_expr_yield(scrutinee, syms, direct, callees);
                for arm in arms {
                    scan_stmts_yield(
                        &arm.body.stmts,
                        stmt_manual,
                        in_noyield,
                        syms,
                        direct,
                        callees,
                    );
                }
            }
            StmtKind::Block(b) => {
                scan_stmts_yield(&b.stmts, stmt_manual, in_noyield, syms, direct, callees);
            }
            StmtKind::VarDecl { init: Some(e), .. }
            | StmtKind::Return(Some(e))
            | StmtKind::Expr(e) => {
                scan_expr_yield(e, syms, direct, callees);
            }
            StmtKind::Assign { target, value, .. } => {
                scan_expr_yield(target, syms, direct, callees);
                scan_expr_yield(value, syms, direct, callees);
            }
            StmtKind::Error { reason, partial } => {
                if let Some(r) = reason {
                    scan_expr_yield(r, syms, direct, callees);
                }
                if let Some(p) = partial {
                    scan_expr_yield(p, syms, direct, callees);
                }
            }
            StmtKind::Spawn(sp) => {
                for a in &sp.args {
                    scan_expr_yield(&a.value, syms, direct, callees);
                }
            }
            StmtKind::VarDecl { init: None, .. }
            | StmtKind::Return(None)
            | StmtKind::Continue
            | StmtKind::Break => {}
        }
    }
}

fn scan_expr_yield(
    expr: &Expr,
    syms: &ModuleSymbols,
    direct: &mut bool,
    callees: &mut BTreeSet<String>,
) {
    match &expr.kind {
        ExprKind::Call { callee, args } => {
            if let ExprKind::Ident(id) = &callee.kind {
                if matches!(id.name.as_str(), "delay" | "delay_ms" | "sleep" | "checkpoint") {
                    *direct = true;
                } else {
                    callees.insert(id.name.clone());
                }
            }
            scan_expr_yield(callee, syms, direct, callees);
            for a in args {
                scan_expr_yield(&a.value, syms, direct, callees);
            }
        }
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => {
            let m = method.name.as_str();
            if matches!(m, "wait_chunk" | "recv" | "wait" | "send" | "delay") {
                *direct = true;
            }
            scan_expr_yield(receiver, syms, direct, callees);
            for a in args {
                scan_expr_yield(&a.value, syms, direct, callees);
            }
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::Field { target: inner, .. } => {
            scan_expr_yield(inner, syms, direct, callees);
        }
        ExprKind::Binary { lhs, rhs, .. }
        | ExprKind::Index {
            target: lhs,
            index: rhs,
        }
        | ExprKind::Or {
            expr: lhs,
            fallback: rhs,
        } => {
            scan_expr_yield(lhs, syms, direct, callees);
            scan_expr_yield(rhs, syms, direct, callees);
        }
        ExprKind::Slice { target, start, end } => {
            scan_expr_yield(target, syms, direct, callees);
            if let Some(s) = start {
                scan_expr_yield(s, syms, direct, callees);
            }
            if let Some(e) = end {
                scan_expr_yield(e, syms, direct, callees);
            }
        }
        ExprKind::StructLit { fields, .. } => {
            for f in fields {
                scan_expr_yield(&f.value, syms, direct, callees);
            }
        }
        ExprKind::ArrayLit(elems) => {
            for e in elems {
                scan_expr_yield(e, syms, direct, callees);
            }
        }
        ExprKind::Catch { expr: inner, clause } => {
            scan_expr_yield(inner, syms, direct, callees);
            scan_stmts_yield(&clause.body.stmts, false, false, syms, direct, callees);
        }
        ExprKind::Literal(_) | ExprKind::Ident(_) | ExprKind::Priority(_) => {}
    }
}

fn stmt_slice_can_yield(stmts: &[Stmt], manual_yield: bool, syms: &ModuleSymbols) -> bool {
    let mut direct = false;
    let mut callees = BTreeSet::new();
    scan_stmts_yield(stmts, manual_yield, false, syms, &mut direct, &mut callees);
    if direct {
        return true;
    }
    callees
        .iter()
        .any(|c| syms.fns.get(c).map_or(false, |f| f.is_yielding))
}

fn detect_static_set_priority(stmts: &[Stmt]) -> Option<MirPriority> {
    for stmt in stmts {
        if let StmtKind::Expr(Expr {
            kind: ExprKind::Call { callee, args },
            ..
        }) = &stmt.kind
        {
            if let ExprKind::Ident(id) = &callee.kind {
                if id.name == "set_priority" {
                    if let Some(arg) = args.first() {
                        if let ExprKind::Priority(p) = &arg.value.kind {
                            return Some(convert_priority(*p));
                        }
                    }
                }
            }
        }
    }
    None
}

fn lower_main_entry(
    md: &FnDecl,
    syms: &ModuleSymbols,
    blueprints: &mut Vec<MirTaskBlueprint>,
    spawned_tasks: &mut Vec<MirSpawnInstance>,
) -> Result<MirBootSequence, MirLowerError> {
    let body = md.body.as_ref().unwrap();
    let stmts = &body.stmts;
    let manual_yield = has_attr_manual_yield(&md.attrs);

    // Locate the last top-level `spawn` statement (if any).
    let last_spawn_idx = stmts
        .iter()
        .rposition(|s| matches!(s.kind, StmtKind::Spawn(_)));

    let (boot_slice, tail_slice) = match last_spawn_idx {
        Some(idx) => (&stmts[..=idx], &stmts[idx + 1..]),
        None => {
            if stmt_slice_can_yield(stmts, manual_yield, syms) {
                // No `spawn` statements, `main` itself is a yielding task; keep leading
                // `sys.use_systick` in boot if it is the first statement, or run all in `main` task.
                let split = stmts
                    .iter()
                    .take_while(|s| is_pure_boot_setup_stmt(s))
                    .count();
                (&stmts[..split], &stmts[split..])
            } else {
                (&stmts[..], &[][..])
            }
        }
    };

    let tail_becomes_task =
        !tail_slice.is_empty() && stmt_slice_can_yield(tail_slice, manual_yield, syms);

    let mut dctx = DirectLowerCtx::new(syms, MirType::Void, MirErrorSpec::None);
    let mut boot_steps: Vec<MirBootStep> = Vec::new();
    let mut boot_yielding_callees: BTreeSet<String> = BTreeSet::new();
    let mut bp_instance_counts: BTreeMap<String, usize> = BTreeMap::new();

    let combined_boot_iter = if tail_becomes_task {
        boot_slice.iter().chain([].iter())
    } else {
        boot_slice.iter().chain(tail_slice.iter())
    };

    for stmt in combined_boot_iter {
        match &stmt.kind {
            StmtKind::Spawn(sp) => {
                let bp_name = sp.task.name.clone();
                let param_tys = syms.tasks.get(&bp_name).cloned().unwrap_or_default();
                let mut lowered_args = Vec::new();
                for (idx, arg) in sp.args.iter().enumerate() {
                    let expected = param_tys.get(idx);
                    lowered_args.push(dctx.lower_pure_expr(&arg.value, expected)?);
                }
                let count_entry = bp_instance_counts.entry(bp_name.clone()).or_insert(0);
                let inst_num = *count_entry;
                *count_entry += 1;

                let slot_id = spawned_tasks.len();
                let instance_name = format!("{bp_name}_{inst_num}");
                let prio = sp.priority.map_or(MirPriority::Normal, convert_priority);
                let core_pin = sp.core.map(|c| c as u8);

                spawned_tasks.push(MirSpawnInstance {
                    slot_id,
                    instance_name,
                    blueprint_name: bp_name,
                    priority: prio,
                    core_pin,
                    init_args: lowered_args.clone(),
                    is_main_supervisor: false,
                });

                boot_steps.push(MirBootStep::SpawnActivate {
                    slot_id,
                    args: lowered_args,
                });
            }
            // Support sequential boot calls to yielding functions (e.g. `run_power_on_self_test();`)!
            StmtKind::Expr(Expr {
                kind: ExprKind::Call { callee, args },
                ..
            }) if is_yielding_fn_callee(callee, syms) => {
                let cname = match &callee.kind {
                    ExprKind::Ident(id) => id.name.clone(),
                    _ => unreachable!(),
                };
                let fn_sig = syms.fns.get(&cname).unwrap().clone();
                let mut largs = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    largs.push(dctx.lower_pure_expr(&a.value, fn_sig.params.get(i))?);
                }
                boot_yielding_callees.insert(cname.clone());
                boot_steps.push(MirBootStep::RunYieldingFn {
                    callee: cname,
                    args: largs,
                    dst: None,
                });
            }
            StmtKind::VarDecl {
                ty,
                name,
                init:
                    Some(Expr {
                        kind: ExprKind::Call { callee, args },
                        ..
                    }),
            } if is_yielding_fn_callee(callee, syms) => {
                let cname = match &callee.kind {
                    ExprKind::Ident(id) => id.name.clone(),
                    _ => unreachable!(),
                };
                let fn_sig = syms.fns.get(&cname).unwrap().clone();
                let vty = syms.resolve_type(ty);
                let vid = dctx.declare_var(&name.name, vty, VarRole::UserLocal);
                let mut largs = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    largs.push(dctx.lower_pure_expr(&a.value, fn_sig.params.get(i))?);
                }
                boot_yielding_callees.insert(cname.clone());
                let dst = if fn_sig.error_spec.has_error() {
                    Some(MirCallDst::Result(vid))
                } else {
                    Some(MirCallDst::Value(vid))
                };
                boot_steps.push(MirBootStep::RunYieldingFn {
                    callee: cname,
                    args: largs,
                    dst,
                });
            }
            _ => {
                let lowered = dctx.lower_stmt(stmt)?;
                for ls in lowered {
                    boot_steps.push(MirBootStep::Stmt(ls));
                }
            }
        }
    }

    if tail_becomes_task {
        // Check if any boot variables are referenced inside `tail_slice`; if so, pass them
        // into `main` supervisor task's static frame parameters so they persist cleanly!
        let mut tail_ident_refs = BTreeSet::new();
        collect_ast_idents_stmts(tail_slice, &mut tail_ident_refs);

        let mut forwarded_params: Vec<(String, MirType, MirExpr)> = Vec::new();
        for bvar in &dctx.vars {
            if bvar.role == VarRole::UserLocal && tail_ident_refs.contains(&bvar.name) {
                forwarded_params.push((
                    bvar.name.clone(),
                    bvar.ty.clone(),
                    MirExpr::new(MirExprKind::Var(bvar.id), bvar.ty.clone()),
                ));
            }
        }

        let mut cctx =
            CoroutineLowerCtx::new(syms, MirType::Void, MirErrorSpec::None, manual_yield);
        let mut param_ids = Vec::new();
        let mut init_args = Vec::new();
        for (pname, pty, init_expr) in forwarded_params {
            let vid = cctx.declare_var(&pname, pty, VarRole::Param);
            param_ids.push(vid);
            init_args.push(init_expr);
        }

        cctx.lower_stmts(tail_slice)?;
        cctx.finish_with_implicit_return();

        let mut main_bp = MirTaskBlueprint {
            name: "main".to_string(),
            is_main_supervisor: true,
            manual_yield,
            params: param_ids,
            vars: cctx.vars,
            blocks: cctx.blocks,
            child_slots: cctx.child_slots,
        };
        analyze_task_frame(&mut main_bp);
        blueprints.push(main_bp);

        let prio = detect_static_set_priority(tail_slice).unwrap_or(MirPriority::Normal);
        let slot_id = spawned_tasks.len();
        spawned_tasks.push(MirSpawnInstance {
            slot_id,
            instance_name: "main".to_string(),
            blueprint_name: "main".to_string(),
            priority: prio,
            core_pin: None,
            init_args: init_args.clone(),
            is_main_supervisor: true,
        });
        boot_steps.push(MirBootStep::SpawnActivate {
            slot_id,
            args: init_args,
        });
    }

    Ok(MirBootSequence {
        present: true,
        vars: dctx.vars,
        steps: boot_steps,
        boot_yielding_callees: boot_yielding_callees.into_iter().collect(),
    })
}

fn is_yielding_fn_callee(callee: &Expr, syms: &ModuleSymbols) -> bool {
    if let ExprKind::Ident(id) = &callee.kind {
        syms.fns.get(&id.name).map_or(false, |f| f.is_yielding)
    } else {
        false
    }
}

fn is_pure_boot_setup_stmt(stmt: &Stmt) -> bool {
    match &stmt.kind {
        StmtKind::Expr(Expr {
            kind: ExprKind::MethodCall { receiver, .. },
            ..
        }) => {
            if let ExprKind::Ident(id) = &receiver.kind {
                id.name == "sys" || id.name == "hw"
            } else {
                false
            }
        }
        _ => false,
    }
}

fn collect_ast_idents_stmts(stmts: &[Stmt], out: &mut BTreeSet<String>) {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::VarDecl { init, .. } => {
                if let Some(e) = init {
                    collect_ast_idents_expr(e, out);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                collect_ast_idents_expr(target, out);
                collect_ast_idents_expr(value, out);
            }
            StmtKind::Return(Some(e)) | StmtKind::Expr(e) => collect_ast_idents_expr(e, out),
            StmtKind::Error { reason, partial } => {
                if let Some(r) = reason {
                    collect_ast_idents_expr(r, out);
                }
                if let Some(p) = partial {
                    collect_ast_idents_expr(p, out);
                }
            }
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                collect_ast_idents_expr(cond, out);
                collect_ast_idents_stmts(&then_block.stmts, out);
                if let Some(eb) = else_block {
                    collect_ast_idents_stmts(&eb.stmts, out);
                }
            }
            StmtKind::While { cond, body } => {
                collect_ast_idents_expr(cond, out);
                collect_ast_idents_stmts(&body.stmts, out);
            }
            StmtKind::Loop { body }
            | StmtKind::Every { body, .. }
            | StmtKind::NoYield { body }
            | StmtKind::Critical { body }
            | StmtKind::Block(body) => {
                collect_ast_idents_stmts(&body.stmts, out);
            }
            StmtKind::Switch { scrutinee, arms } => {
                collect_ast_idents_expr(scrutinee, out);
                for arm in arms {
                    collect_ast_idents_stmts(&arm.body.stmts, out);
                }
            }
            StmtKind::Spawn(sp) => {
                for a in &sp.args {
                    collect_ast_idents_expr(&a.value, out);
                }
            }
            _ => {}
        }
    }
}

fn collect_ast_idents_expr(expr: &Expr, out: &mut BTreeSet<String>) {
    match &expr.kind {
        ExprKind::Ident(id) => {
            out.insert(id.name.clone());
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::Field { target: inner, .. } => collect_ast_idents_expr(inner, out),
        ExprKind::Binary { lhs, rhs, .. }
        | ExprKind::Index {
            target: lhs,
            index: rhs,
        }
        | ExprKind::Or {
            expr: lhs,
            fallback: rhs,
        } => {
            collect_ast_idents_expr(lhs, out);
            collect_ast_idents_expr(rhs, out);
        }
        ExprKind::Slice { target, start, end } => {
            collect_ast_idents_expr(target, out);
            if let Some(s) = start {
                collect_ast_idents_expr(s, out);
            }
            if let Some(e) = end {
                collect_ast_idents_expr(e, out);
            }
        }
        ExprKind::Call { callee, args } => {
            collect_ast_idents_expr(callee, out);
            for a in args {
                collect_ast_idents_expr(&a.value, out);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            collect_ast_idents_expr(receiver, out);
            for a in args {
                collect_ast_idents_expr(&a.value, out);
            }
        }
        ExprKind::StructLit { fields, .. } => {
            for f in fields {
                collect_ast_idents_expr(&f.value, out);
            }
        }
        ExprKind::ArrayLit(elems) => {
            for e in elems {
                collect_ast_idents_expr(e, out);
            }
        }
        ExprKind::Catch { expr: inner, clause } => {
            collect_ast_idents_expr(inner, out);
            collect_ast_idents_stmts(&clause.body.stmts, out);
        }
        ExprKind::Literal(_) | ExprKind::Priority(_) => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared Scope & Expression Helpers
// ─────────────────────────────────────────────────────────────────────────────

struct VarScopes {
    frames: Vec<BTreeMap<String, VarId>>,
}

impl VarScopes {
    fn new() -> Self {
        Self {
            frames: vec![BTreeMap::new()],
        }
    }

    fn push(&mut self) {
        self.frames.push(BTreeMap::new());
    }

    fn pop(&mut self) {
        self.frames.pop();
    }

    fn insert(&mut self, name: String, vid: VarId) {
        if let Some(top) = self.frames.last_mut() {
            top.insert(name, vid);
        }
    }

    fn lookup(&self, name: &str) -> Option<VarId> {
        for frame in self.frames.iter().rev() {
            if let Some(&vid) = frame.get(name) {
                return Some(vid);
            }
        }
        None
    }
}

fn convert_assign_op(op: AssignOp) -> MirAssignOp {
    match op {
        AssignOp::Assign => MirAssignOp::Assign,
        AssignOp::AddAssign => MirAssignOp::AddAssign,
        AssignOp::SubAssign => MirAssignOp::SubAssign,
        AssignOp::MulAssign => MirAssignOp::MulAssign,
        AssignOp::DivAssign => MirAssignOp::DivAssign,
        AssignOp::RemAssign => MirAssignOp::RemAssign,
        AssignOp::BitAndAssign => MirAssignOp::BitAndAssign,
        AssignOp::BitOrAssign => MirAssignOp::BitOrAssign,
        AssignOp::BitXorAssign => MirAssignOp::BitXorAssign,
        AssignOp::ShlAssign => MirAssignOp::ShlAssign,
        AssignOp::ShrAssign => MirAssignOp::ShrAssign,
    }
}

fn convert_bin_op(op: BinaryOp) -> MirBinOp {
    match op {
        BinaryOp::Add => MirBinOp::Add,
        BinaryOp::Sub => MirBinOp::Sub,
        BinaryOp::Mul => MirBinOp::Mul,
        BinaryOp::Div => MirBinOp::Div,
        BinaryOp::Rem => MirBinOp::Rem,
        BinaryOp::BitOr => MirBinOp::BitOr,
        BinaryOp::BitAnd => MirBinOp::BitAnd,
        BinaryOp::BitXor => MirBinOp::BitXor,
        BinaryOp::Shl => MirBinOp::Shl,
        BinaryOp::Shr => MirBinOp::Shr,
        BinaryOp::Eq => MirBinOp::Eq,
        BinaryOp::Ne => MirBinOp::Ne,
        BinaryOp::Lt => MirBinOp::Lt,
        BinaryOp::Le => MirBinOp::Le,
        BinaryOp::Gt => MirBinOp::Gt,
        BinaryOp::Ge => MirBinOp::Ge,
        BinaryOp::And => MirBinOp::LogicalAnd,
        BinaryOp::Or => MirBinOp::LogicalOr,
    }
}

fn try_extract_hw_path(expr: &Expr, syms: &ModuleSymbols) -> Option<Vec<String>> {
    match &expr.kind {
        ExprKind::Ident(id) if syms.is_hw_root(&id.name) => Some(vec![id.name.clone()]),
        ExprKind::Field { target, field } => {
            let mut base = try_extract_hw_path(target, syms)?;
            base.push(field.name.clone());
            Some(base)
        }
        _ => None,
    }
}

fn find_named_arg<'a>(args: &'a [CallArg], name: &str) -> Option<&'a Expr> {
    args.iter()
        .find(|a| a.name.as_ref().map_or(false, |n| n.name == name))
        .map(|a| &a.value)
}

fn first_positional_arg(args: &[CallArg]) -> Option<&Expr> {
    args.iter().find(|a| a.name.is_none()).map(|a| &a.value)
}

fn extract_boost_arg(args: &[CallArg]) -> Option<MirPriority> {
    if let Some(be) = find_named_arg(args, "boost") {
        if let ExprKind::Priority(p) = &be.kind {
            return Some(convert_priority(*p));
        }
    }
    None
}

fn built_in_hw_const(name: &str) -> Option<i128> {
    match name {
        "MOTOR_EN" => Some(1 << 0),
        "HALF_TRANSFER" => Some(1 << 1),
        "FULL_TRANSFER" => Some(1 << 2),
        _ => None,
    }
}

fn eval_const_expr(expr: &Expr, syms: &ModuleSymbols) -> Option<ConstVal> {
    match &expr.kind {
        ExprKind::Literal(Literal::Int(v)) => Some(ConstVal::Int(*v as i128)),
        ExprKind::Literal(Literal::Bool(b)) => Some(ConstVal::Bool(*b)),
        ExprKind::Ident(id) => {
            if let Some(&cv) = syms.const_vals.get(&id.name) {
                return Some(cv);
            }
            if let Some(v) = built_in_hw_const(&id.name) {
                return Some(ConstVal::Int(v));
            }
            if let Some((_, disc)) = syms.variant_to_enum.get(&id.name) {
                return Some(ConstVal::Int(*disc as i128));
            }
            None
        }
        ExprKind::Cast { expr: inner, .. } => eval_const_expr(inner, syms),
        ExprKind::Unary { op, expr: inner } => match (op, eval_const_expr(inner, syms)?) {
            (UnaryOp::Neg, ConstVal::Int(v)) => Some(ConstVal::Int(-v)),
            (UnaryOp::BitNot, ConstVal::Int(v)) => Some(ConstVal::Int(!v)),
            (UnaryOp::Not, ConstVal::Bool(b)) => Some(ConstVal::Bool(!b)),
            _ => None,
        },
        ExprKind::Binary { op, lhs, rhs } => {
            let l = eval_const_expr(lhs, syms)?;
            let r = eval_const_expr(rhs, syms)?;
            match (op, l, r) {
                (BinaryOp::Add, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a.wrapping_add(b))),
                (BinaryOp::Sub, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a.wrapping_sub(b))),
                (BinaryOp::Mul, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a.wrapping_mul(b))),
                (BinaryOp::Div, ConstVal::Int(a), ConstVal::Int(b)) if b != 0 => Some(ConstVal::Int(a / b)),
                (BinaryOp::Rem, ConstVal::Int(a), ConstVal::Int(b)) if b != 0 => Some(ConstVal::Int(a % b)),
                (BinaryOp::BitOr, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a | b)),
                (BinaryOp::BitAnd, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a & b)),
                (BinaryOp::BitXor, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a ^ b)),
                (BinaryOp::Shl, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a.wrapping_shl(b as u32))),
                (BinaryOp::Shr, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a.wrapping_shr(b as u32))),
                (BinaryOp::Eq, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a == b)),
                (BinaryOp::Ne, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a != b)),
                (BinaryOp::Lt, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a < b)),
                (BinaryOp::Le, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a <= b)),
                (BinaryOp::Gt, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a > b)),
                (BinaryOp::Ge, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a >= b)),
                (BinaryOp::And, ConstVal::Bool(a), ConstVal::Bool(b)) => Some(ConstVal::Bool(a && b)),
                (BinaryOp::Or, ConstVal::Bool(a), ConstVal::Bool(b)) => Some(ConstVal::Bool(a || b)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn convert_switch_pat(
    pat: &SwitchPat,
    scrutinee_ty: &MirType,
    syms: &ModuleSymbols,
) -> Option<MirSwitchPattern> {
    match pat {
        SwitchPat::Wildcard(_) => None,
        SwitchPat::Literal(lit, _) => match lit {
            Literal::Int(v) => Some(MirSwitchPattern::Int(*v as i128)),
            Literal::Bool(b) => Some(MirSwitchPattern::Bool(*b)),
            _ => None,
        },
        SwitchPat::Ident { prefix, name, .. } => {
            if name.name == "_" || name.name == "default" {
                return None;
            }
            if let Some(pref) = prefix {
                return Some(MirSwitchPattern::EnumVariant {
                    enum_name: pref.name.clone(),
                    variant: name.name.clone(),
                });
            }
            if let MirType::Enum(en) = scrutinee_ty {
                return Some(MirSwitchPattern::EnumVariant {
                    enum_name: en.clone(),
                    variant: name.name.clone(),
                });
            }
            if let Some(&cv) = syms.const_vals.get(&name.name) {
                return match cv {
                    ConstVal::Int(v) => Some(MirSwitchPattern::Int(v)),
                    ConstVal::Bool(b) => Some(MirSwitchPattern::Bool(b)),
                };
            }
            if let Some(v) = built_in_hw_const(&name.name) {
                return Some(MirSwitchPattern::Int(v));
            }
            if let Some((en, _)) = syms.variant_to_enum.get(&name.name) {
                return Some(MirSwitchPattern::EnumVariant {
                    enum_name: en.clone(),
                    variant: name.name.clone(),
                });
            }
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Direct (Non-Yielding) Function / ISR / Boot Statement Lowerer
// ─────────────────────────────────────────────────────────────────────────────

struct DirectLowerCtx<'a> {
    syms: &'a ModuleSymbols,
    ret_ty: MirType,
    error_spec: MirErrorSpec,
    vars: Vec<MirVar>,
    scopes: VarScopes,
    in_noyield: bool,
    pending_stmts: Vec<MirStmt>,
}

impl<'a> DirectLowerCtx<'a> {
    fn new(syms: &'a ModuleSymbols, ret_ty: MirType, error_spec: MirErrorSpec) -> Self {
        Self {
            syms,
            ret_ty,
            error_spec,
            vars: Vec::new(),
            scopes: VarScopes::new(),
            in_noyield: false,
            pending_stmts: Vec::new(),
        }
    }

    fn declare_var(&mut self, name: &str, ty: MirType, role: VarRole) -> VarId {
        let vid = VarId(self.vars.len());
        self.vars.push(MirVar {
            id: vid,
            name: name.to_string(),
            ty,
            role,
            in_noyield_scope: self.in_noyield,
            addr_taken: false,
            hoisted: false,
        });
        if role != VarRole::CompilerTemp {
            self.scopes.insert(name.to_string(), vid);
        }
        vid
    }

    fn alloc_temp(&mut self, prefix: &str, ty: MirType) -> VarId {
        let name = format!("_tmp_{}_{}", prefix, self.vars.len());
        self.declare_var(&name, ty, VarRole::CompilerTemp)
    }

    fn lower_block_stmts(&mut self, stmts: &[Stmt]) -> Result<Vec<MirStmt>, MirLowerError> {
        self.scopes.push();
        let mut out = Vec::new();
        for s in stmts {
            let chunk = self.lower_stmt(s)?;
            out.extend(chunk);
        }
        self.scopes.pop();
        Ok(out)
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<Vec<MirStmt>, MirLowerError> {
        let mut out = Vec::new();
        match &stmt.kind {
            StmtKind::VarDecl { ty, name, init } => {
                let vty = self.syms.resolve_type(ty);
                let vid = self.declare_var(&name.name, vty.clone(), VarRole::UserLocal);
                if let Some(ie) = init {
                    let val = self.lower_expr_with_stmts(ie, Some(&vty), &mut out)?;
                    out.push(MirStmt::Assign {
                        target: MirPlace::Var(vid),
                        op: MirAssignOp::Assign,
                        value: coerce_expr_if_needed(val, &vty),
                    });
                }
            }
            StmtKind::Assign { target, op, value } => {
                let (place, pty) = self.lower_place(target, &mut out)?;
                let rhs = self.lower_expr_with_stmts(value, Some(&pty.host_value_type()), &mut out)?;
                out.push(MirStmt::Assign {
                    target: place,
                    op: convert_assign_op(*op),
                    value: coerce_expr_if_needed(rhs, &pty.host_value_type()),
                });
            }
            StmtKind::Return(expr_opt) => {
                let ret_ty = self.ret_ty.clone();
                let val = if let Some(e) = expr_opt {
                    Some(self.lower_expr_with_stmts(e, Some(&ret_ty), &mut out)?)
                } else {
                    None
                };
                out.push(MirStmt::Return(val));
            }
            StmtKind::Error { reason, partial } => {
                let ret_ty = self.ret_ty.clone();
                let err_ty = self.error_spec.err_type();
                let r = if let Some(re) = reason {
                    Some(self.lower_expr_with_stmts(re, err_ty.as_ref(), &mut out)?)
                } else {
                    None
                };
                let p = if let Some(pe) = partial {
                    Some(self.lower_expr_with_stmts(pe, Some(&ret_ty), &mut out)?)
                } else {
                    None
                };
                out.push(MirStmt::Error {
                    reason: r,
                    partial: p,
                });
            }
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                let c = self.lower_expr_with_stmts(cond, Some(&MirType::Bool), &mut out)?;
                let tb = self.lower_block_stmts(&then_block.stmts)?;
                let eb = if let Some(elb) = else_block {
                    self.lower_block_stmts(&elb.stmts)?
                } else {
                    Vec::new()
                };
                out.push(MirStmt::If {
                    cond: c,
                    then_body: tb,
                    else_body: eb,
                });
            }
            StmtKind::While { cond, body } => {
                let c = self.lower_expr_with_stmts(cond, Some(&MirType::Bool), &mut out)?;
                let b = self.lower_block_stmts(&body.stmts)?;
                out.push(MirStmt::While { cond: c, body: b });
            }
            StmtKind::Loop { body } => {
                let b = self.lower_block_stmts(&body.stmts)?;
                out.push(MirStmt::While {
                    cond: MirExpr::new(MirExprKind::BoolLit(true), MirType::Bool),
                    body: b,
                });
            }
            StmtKind::NoYield { body } => {
                let prev = self.in_noyield;
                self.in_noyield = true;
                let b = self.lower_block_stmts(&body.stmts)?;
                self.in_noyield = prev;
                out.extend(b);
            }
            StmtKind::Critical { body } => {
                let prev = self.in_noyield;
                self.in_noyield = true;
                out.push(MirStmt::EnterCritical);
                let b = self.lower_block_stmts(&body.stmts)?;
                out.extend(b);
                out.push(MirStmt::ExitCritical);
                self.in_noyield = prev;
            }
            StmtKind::Switch { scrutinee, arms } => {
                let d = self.lower_expr_with_stmts(scrutinee, None, &mut out)?;
                let mut mir_arms = Vec::new();
                let mut default_body = Vec::new();
                for arm in arms {
                    let body = self.lower_block_stmts(&arm.body.stmts)?;
                    if let Some(pat) = convert_switch_pat(&arm.pat, &d.ty, self.syms) {
                        mir_arms.push((pat, body));
                    } else {
                        default_body = body;
                    }
                }
                out.push(MirStmt::Switch {
                    discr: d,
                    arms: mir_arms,
                    default: default_body,
                });
            }
            StmtKind::Continue => out.push(MirStmt::Continue),
            StmtKind::Break => out.push(MirStmt::Break),
            StmtKind::Block(b) => {
                out.extend(self.lower_block_stmts(&b.stmts)?);
            }
            StmtKind::Expr(e) => {
                if let Some(stmt_op) = self.try_lower_stmt_call(e, &mut out)? {
                    out.push(stmt_op);
                } else {
                    let me = self.lower_expr_with_stmts(e, None, &mut out)?;
                    out.push(MirStmt::Expr(me));
                }
            }
            StmtKind::Every { .. } | StmtKind::Checkpoint | StmtKind::Spawn(_) => {
                return Err(MirLowerError {
                    message: "yielding or spawn statement encountered inside non-yielding context"
                        .to_string(),
                    span: stmt.span,
                });
            }
        }
        Ok(out)
    }

    fn lower_pure_expr(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError> {
        let mut sink = Vec::new();
        self.lower_expr_with_stmts(expr, expected, &mut sink)
    }

    fn lower_expr_with_stmts(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<MirExpr, MirLowerError> {
        match &expr.kind {
            ExprKind::Or {
                expr: inner,
                fallback,
            } => {
                let res_expr = self.lower_failable_expr(inner, stmts)?;
                let (ok_ty, res_ty) = extract_result_types(&res_expr.ty);
                let res_vid = self.alloc_temp("or_res", res_ty);
                stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(res_vid),
                    op: MirAssignOp::Assign,
                    value: res_expr,
                });
                let out_vid = self.alloc_temp("or_val", ok_ty.clone());
                let mut fb_stmts = Vec::new();
                let fb_val = self.lower_expr_with_stmts(fallback, Some(&ok_ty), &mut fb_stmts)?;
                fb_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(out_vid),
                    op: MirAssignOp::Assign,
                    value: fb_val,
                });
                let ok_stmts = vec![MirStmt::Assign {
                    target: MirPlace::Var(out_vid),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(MirExprKind::ResultVal(res_vid), ok_ty.clone()),
                }];
                stmts.push(MirStmt::If {
                    cond: MirExpr::new(MirExprKind::ResultIsErr(res_vid), MirType::Bool),
                    then_body: fb_stmts,
                    else_body: ok_stmts,
                });
                Ok(MirExpr::new(MirExprKind::Var(out_vid), ok_ty))
            }
            ExprKind::Catch {
                expr: inner,
                clause,
            } => {
                let res_expr = self.lower_failable_expr(inner, stmts)?;
                let (ok_ty, res_ty) = extract_result_types(&res_expr.ty);
                let err_ty_opt = match &res_ty {
                    MirType::Result { err, .. } => err.as_deref().cloned(),
                    _ => None,
                };
                let res_vid = self.alloc_temp("catch_res", res_ty);
                stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(res_vid),
                    op: MirAssignOp::Assign,
                    value: res_expr,
                });

                let out_vid_opt = if ok_ty.is_void() {
                    None
                } else {
                    Some(self.alloc_temp("catch_val", ok_ty.clone()))
                };

                self.scopes.push();
                let mut err_stmts = Vec::new();
                if let Some(ep) = &clause.err_param {
                    let ety = err_ty_opt.unwrap_or_else(|| self.syms.resolve_type(&ep.ty));
                    let evid = self.declare_var(&ep.name.name, ety.clone(), VarRole::UserLocal);
                    err_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(evid),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::ResultErr(res_vid), ety),
                    });
                }
                if let Some(pp) = &clause.partial_param {
                    let pty = self.syms.resolve_type(&pp.ty);
                    let pvid = self.declare_var(&pp.name.name, pty.clone(), VarRole::UserLocal);
                    err_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(pvid),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::ResultVal(res_vid), pty),
                    });
                }
                if let (Some(out_vid), Some((last_stmt, prefix_stmts))) =
                    (out_vid_opt, clause.body.stmts.split_last())
                {
                    if let StmtKind::Expr(tail_expr) = &last_stmt.kind {
                        for s in prefix_stmts {
                            err_stmts.extend(self.lower_stmt(s)?);
                        }
                        if let Some(stmt_op) = self.try_lower_stmt_call(tail_expr, &mut err_stmts)?
                        {
                            err_stmts.push(stmt_op);
                        } else {
                            let tail_val =
                                self.lower_expr_with_stmts(tail_expr, Some(&ok_ty), &mut err_stmts)?;
                            if tail_val.ty.is_void() {
                                err_stmts.push(MirStmt::Expr(tail_val));
                            } else {
                                err_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(out_vid),
                                    op: MirAssignOp::Assign,
                                    value: coerce_expr_if_needed(tail_val, &ok_ty),
                                });
                            }
                        }
                    } else {
                        for s in &clause.body.stmts {
                            err_stmts.extend(self.lower_stmt(s)?);
                        }
                    }
                } else {
                    for s in &clause.body.stmts {
                        err_stmts.extend(self.lower_stmt(s)?);
                    }
                }
                self.scopes.pop();

                if let Some(out_vid) = out_vid_opt {
                    let ok_stmts = vec![MirStmt::Assign {
                        target: MirPlace::Var(out_vid),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::ResultVal(res_vid), ok_ty.clone()),
                    }];
                    stmts.push(MirStmt::If {
                        cond: MirExpr::new(MirExprKind::ResultIsErr(res_vid), MirType::Bool),
                        then_body: err_stmts,
                        else_body: ok_stmts,
                    });
                    Ok(MirExpr::new(MirExprKind::Var(out_vid), ok_ty))
                } else {
                    stmts.push(MirStmt::If {
                        cond: MirExpr::new(MirExprKind::ResultIsErr(res_vid), MirType::Bool),
                        then_body: err_stmts,
                        else_body: Vec::new(),
                    });
                    Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void))
                }
            }
            _ => {
                let _ = &self.pending_stmts;
                let mut be = DirectBackend {
                    ctx: self,
                    stmts,
                };
                lower_common_expr(expr, expected, &mut be)
            }
        }
    }

    fn lower_failable_expr(
        &mut self,
        expr: &Expr,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<MirExpr, MirLowerError> {
        let me = self.lower_expr_with_stmts(expr, None, stmts)?;
        Ok(me)
    }

    fn lower_place(
        &mut self,
        expr: &Expr,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<(MirPlace, MirType), MirLowerError> {
        let mut be = DirectBackend {
            ctx: self,
            stmts,
        };
        lower_place_common(expr, &mut be)
    }

    fn try_lower_stmt_call(
        &mut self,
        expr: &Expr,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<Option<MirStmt>, MirLowerError> {
        let mut be = DirectBackend {
            ctx: self,
            stmts,
        };
        try_lower_nonblocking_call_or_method(expr, &mut be)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Coroutine CFG State-Machine Lowerer (for `task`s and yielding `fn`s)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct LoopContext {
    continue_bb: BlockId,
    break_bb: BlockId,
    has_inbox_recv: bool,
}

struct CoroutineLowerCtx<'a> {
    syms: &'a ModuleSymbols,
    ret_ty: MirType,
    error_spec: MirErrorSpec,
    manual_yield: bool,
    in_noyield: bool,
    vars: Vec<MirVar>,
    scopes: VarScopes,
    blocks: Vec<MirBlock>,
    cur_bb: BlockId,
    cur_is_resume: bool,
    cur_stmts: Vec<MirStmt>,
    next_bb_id: u16,
    child_slots: Vec<MirChildCallSlot>,
    loop_stack: Vec<LoopContext>,
}

impl<'a> CoroutineLowerCtx<'a> {
    fn new(
        syms: &'a ModuleSymbols,
        ret_ty: MirType,
        error_spec: MirErrorSpec,
        manual_yield: bool,
    ) -> Self {
        Self {
            syms,
            ret_ty,
            error_spec,
            manual_yield,
            in_noyield: false,
            vars: Vec::new(),
            scopes: VarScopes::new(),
            blocks: Vec::new(),
            cur_bb: BlockId(0),
            cur_is_resume: true,
            cur_stmts: Vec::new(),
            next_bb_id: 1,
            child_slots: Vec::new(),
            loop_stack: Vec::new(),
        }
    }

    fn alloc_bb(&mut self) -> BlockId {
        let id = BlockId(self.next_bb_id);
        self.next_bb_id += 1;
        id
    }

    fn seal_bb(&mut self, term: MirTerminator, next_bb: BlockId, next_is_resume: bool) {
        let stmts = std::mem::take(&mut self.cur_stmts);
        self.blocks.push(MirBlock {
            id: self.cur_bb,
            is_resume_entry: self.cur_is_resume,
            stmts,
            terminator: term,
        });
        self.cur_bb = next_bb;
        self.cur_is_resume = next_is_resume;
    }

    fn declare_var(&mut self, name: &str, ty: MirType, role: VarRole) -> VarId {
        let vid = VarId(self.vars.len());
        self.vars.push(MirVar {
            id: vid,
            name: name.to_string(),
            ty,
            role,
            in_noyield_scope: self.in_noyield,
            addr_taken: false,
            hoisted: false,
        });
        if role != VarRole::CompilerTemp {
            self.scopes.insert(name.to_string(), vid);
        }
        vid
    }

    fn alloc_temp(&mut self, prefix: &str, ty: MirType) -> VarId {
        let name = format!("_tmp_{}_{}", prefix, self.vars.len());
        self.declare_var(&name, ty, VarRole::CompilerTemp)
    }

    fn finish_with_implicit_return(&mut self) {
        let stmts = std::mem::take(&mut self.cur_stmts);
        let default_ret = if self.ret_ty.is_void() {
            None
        } else {
            Some(MirExpr::new(MirExprKind::IntLit(0), self.ret_ty.clone()))
        };
        self.blocks.push(MirBlock {
            id: self.cur_bb,
            is_resume_entry: self.cur_is_resume,
            stmts,
            terminator: MirTerminator::Return(default_ret),
        });
    }

    fn lower_scoped_block(&mut self, block: &Block) -> Result<(), MirLowerError> {
        self.scopes.push();
        self.lower_stmts(&block.stmts)?;
        self.scopes.pop();
        Ok(())
    }

    fn lower_stmts(&mut self, stmts: &[Stmt]) -> Result<(), MirLowerError> {
        for stmt in stmts {
            self.lower_stmt(stmt)?;
        }
        Ok(())
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), MirLowerError> {
        let stmt_manual = self.manual_yield || has_attr_manual_yield(&stmt.attrs);
        match &stmt.kind {
            StmtKind::VarDecl { ty, name, init } => {
                let vty = self.syms.resolve_type(ty);
                let vid = self.declare_var(&name.name, vty.clone(), VarRole::UserLocal);
                if let Some(ie) = init {
                    if !self.try_lower_yielding_into_var(ie, vid, &vty)? {
                        let val = self.lower_expr(ie, Some(&vty))?;
                        self.cur_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(vid),
                            op: MirAssignOp::Assign,
                            value: coerce_expr_if_needed(val, &vty),
                        });
                    }
                }
            }
            StmtKind::Assign { target, op, value } => {
                if *op == MirAssignOp::Assign.into_ast_check() {
                    // handled uniformly below
                }
                if matches!(op, AssignOp::Assign) {
                    if let ExprKind::Ident(id) = &target.kind {
                        if let Some(vid) = self.scopes.lookup(&id.name) {
                            let vty = self.vars[vid.0].ty.clone();
                            if self.try_lower_yielding_into_var(value, vid, &vty)? {
                                return Ok(());
                            }
                        }
                    }
                }
                let (place, pty) = self.lower_place(target)?;
                let rhs = self.lower_expr(value, Some(&pty.host_value_type()))?;
                self.cur_stmts.push(MirStmt::Assign {
                    target: place,
                    op: convert_assign_op(*op),
                    value: coerce_expr_if_needed(rhs, &pty.host_value_type()),
                });
            }
            StmtKind::Return(expr_opt) => {
                let ret_ty = self.ret_ty.clone();
                let val = if let Some(e) = expr_opt {
                    Some(self.lower_expr(e, Some(&ret_ty))?)
                } else {
                    None
                };
                let dead_bb = self.alloc_bb();
                self.seal_bb(MirTerminator::Return(val), dead_bb, false);
            }
            StmtKind::Error { reason, partial } => {
                let ret_ty = self.ret_ty.clone();
                let err_ty = self.error_spec.err_type();
                let r = if let Some(re) = reason {
                    Some(self.lower_expr(re, err_ty.as_ref())?)
                } else {
                    None
                };
                let p = if let Some(pe) = partial {
                    Some(self.lower_expr(pe, Some(&ret_ty))?)
                } else {
                    None
                };
                let dead_bb = self.alloc_bb();
                self.seal_bb(MirTerminator::Error { reason: r, partial: p }, dead_bb, false);
            }
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                let c = self.lower_expr(cond, Some(&MirType::Bool))?;
                let then_bb = self.alloc_bb();
                let else_bb = self.alloc_bb();
                let join_bb = self.alloc_bb();
                self.seal_bb(
                    MirTerminator::Branch {
                        cond: c,
                        then_bb,
                        else_bb,
                    },
                    then_bb,
                    false,
                );
                self.lower_scoped_block(then_block)?;
                self.seal_bb(MirTerminator::Goto(join_bb), else_bb, false);
                if let Some(eb) = else_block {
                    self.lower_scoped_block(eb)?;
                }
                self.seal_bb(MirTerminator::Goto(join_bb), join_bb, false);
            }
            StmtKind::While { cond, body } => {
                let cond_bb = self.alloc_bb();
                let body_bb = self.alloc_bb();
                let latch_bb = self.alloc_bb();
                let exit_bb = self.alloc_bb();

                self.seal_bb(MirTerminator::Goto(cond_bb), cond_bb, false);
                let c = self.lower_expr(cond, Some(&MirType::Bool))?;
                self.seal_bb(
                    MirTerminator::Branch {
                        cond: c,
                        then_bb: body_bb,
                        else_bb: exit_bb,
                    },
                    body_bb,
                    false,
                );

                let has_inbox = ast_stmts_have_inbox_recv(&body.stmts);
                self.loop_stack.push(LoopContext {
                    continue_bb: latch_bb,
                    break_bb: exit_bb,
                    has_inbox_recv: has_inbox,
                });
                self.lower_scoped_block(body)?;
                self.loop_stack.pop();

                self.seal_bb(MirTerminator::Goto(latch_bb), latch_bb, false);
                if has_inbox {
                    self.cur_stmts.push(MirStmt::EndInboxTurn);
                }
                self.seal_bb(MirTerminator::Goto(cond_bb), exit_bb, false);
            }
            StmtKind::Loop { body } => {
                let top_bb = self.alloc_bb();
                let latch_bb = self.alloc_bb();
                let exit_bb = self.alloc_bb();
                let auto_share = !stmt_manual && !self.in_noyield;
                let has_inbox = ast_stmts_have_inbox_recv(&body.stmts);

                self.seal_bb(MirTerminator::Goto(top_bb), top_bb, auto_share);
                self.loop_stack.push(LoopContext {
                    continue_bb: latch_bb,
                    break_bb: exit_bb,
                    has_inbox_recv: has_inbox,
                });
                self.lower_scoped_block(body)?;
                self.loop_stack.pop();

                self.seal_bb(MirTerminator::Goto(latch_bb), latch_bb, false);
                if has_inbox {
                    self.cur_stmts.push(MirStmt::EndInboxTurn);
                }
                if auto_share {
                    self.seal_bb(
                        MirTerminator::Yield(MirYieldKind::ShareCheckpoint { resume_bb: top_bb }),
                        exit_bb,
                        false,
                    );
                } else {
                    self.seal_bb(MirTerminator::Goto(top_bb), exit_bb, false);
                }
            }
            StmtKind::Every { interval, body } => {
                let period_ticks = self.syms.duration_to_ticks(interval)?;
                let deadline_var = self.alloc_temp("every_next", MirType::U32);
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(deadline_var),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(MirExprKind::NowTicks, MirType::U32),
                });

                let body_bb = self.alloc_bb();
                let latch_bb = self.alloc_bb();
                let exit_bb = self.alloc_bb();

                self.seal_bb(MirTerminator::Goto(body_bb), body_bb, true);
                let has_inbox = ast_stmts_have_inbox_recv(&body.stmts);
                self.loop_stack.push(LoopContext {
                    continue_bb: latch_bb,
                    break_bb: exit_bb,
                    has_inbox_recv: has_inbox,
                });
                self.lower_scoped_block(body)?;
                self.loop_stack.pop();

                self.seal_bb(MirTerminator::Goto(latch_bb), latch_bb, false);
                if has_inbox {
                    self.cur_stmts.push(MirStmt::EndInboxTurn);
                }
                self.seal_bb(
                    MirTerminator::Yield(MirYieldKind::EveryLatch {
                        next_deadline_var: deadline_var,
                        period_ticks,
                        resume_bb: body_bb,
                    }),
                    exit_bb,
                    false,
                );
            }
            StmtKind::NoYield { body } => {
                let prev = self.in_noyield;
                self.in_noyield = true;
                self.lower_scoped_block(body)?;
                self.in_noyield = prev;
            }
            StmtKind::Critical { body } => {
                let prev = self.in_noyield;
                self.in_noyield = true;
                self.cur_stmts.push(MirStmt::EnterCritical);
                self.lower_scoped_block(body)?;
                self.cur_stmts.push(MirStmt::ExitCritical);
                self.in_noyield = prev;
            }
            StmtKind::Switch { scrutinee, arms } => {
                let d = self.lower_expr(scrutinee, None)?;
                let join_bb = self.alloc_bb();
                let mut arm_bbs: Vec<(Option<MirSwitchPattern>, BlockId, &Block)> = Vec::new();
                let mut default_bb_opt: Option<BlockId> = None;

                for arm in arms {
                    let abb = self.alloc_bb();
                    let pat_opt = convert_switch_pat(&arm.pat, &d.ty, self.syms);
                    if pat_opt.is_none() {
                        default_bb_opt = Some(abb);
                    }
                    arm_bbs.push((pat_opt, abb, &arm.body));
                }

                let default_bb = default_bb_opt.unwrap_or(join_bb);
                let switch_arms: Vec<(MirSwitchPattern, BlockId)> = arm_bbs
                    .iter()
                    .filter_map(|(p, b, _)| p.clone().map(|pat| (pat, *b)))
                    .collect();

                let first_target = arm_bbs.first().map(|(_, b, _)| *b).unwrap_or(join_bb);
                self.seal_bb(
                    MirTerminator::Switch {
                        discr: d,
                        arms: switch_arms,
                        default_bb,
                    },
                    first_target,
                    false,
                );

                for (idx, (_, abb, abody)) in arm_bbs.iter().enumerate() {
                    if self.cur_bb != *abb {
                        self.cur_bb = *abb;
                        self.cur_is_resume = false;
                    }
                    self.lower_scoped_block(abody)?;
                    let next_after = arm_bbs
                        .get(idx + 1)
                        .map(|(_, b, _)| *b)
                        .unwrap_or(join_bb);
                    self.seal_bb(MirTerminator::Goto(join_bb), next_after, false);
                }

                if self.cur_bb != join_bb {
                    self.cur_bb = join_bb;
                    self.cur_is_resume = false;
                }
            }
            StmtKind::Continue => {
                if let Some(&lc) = self.loop_stack.last() {
                    let dead_bb = self.alloc_bb();
                    self.seal_bb(MirTerminator::Goto(lc.continue_bb), dead_bb, false);
                }
            }
            StmtKind::Break => {
                if let Some(&lc) = self.loop_stack.last() {
                    if lc.has_inbox_recv {
                        self.cur_stmts.push(MirStmt::EndInboxTurn);
                    }
                    let dead_bb = self.alloc_bb();
                    self.seal_bb(MirTerminator::Goto(lc.break_bb), dead_bb, false);
                }
            }
            StmtKind::Checkpoint => {
                let resume_bb = self.alloc_bb();
                self.seal_bb(
                    MirTerminator::Yield(MirYieldKind::ShareCheckpoint { resume_bb }),
                    resume_bb,
                    true,
                );
            }
            StmtKind::Block(b) => {
                self.lower_scoped_block(b)?;
            }
            StmtKind::Expr(e) => {
                if self.try_lower_yielding_stmt_expr(e)? {
                    return Ok(());
                }
                if let Some(op_stmt) = try_lower_nonblocking_call_or_method(e, self)? {
                    self.cur_stmts.push(op_stmt);
                } else {
                    let me = self.lower_expr(e, None)?;
                    if !me.ty.is_void() || matches!(me.kind, MirExprKind::CallDirect { .. }) {
                        self.cur_stmts.push(MirStmt::Expr(me));
                    }
                }
            }
            StmtKind::Spawn(_) => {
                return Err(MirLowerError {
                    message: "`spawn` is only valid inside `fn main()`".to_string(),
                    span: stmt.span,
                });
            }
        }
        Ok(())
    }

    /// Directly lower `VarType dst = <yielding_op>;` without allocating an extra copy temp!
    fn try_lower_yielding_into_var(
        &mut self,
        expr: &Expr,
        dst_vid: VarId,
        dst_ty: &MirType,
    ) -> Result<bool, MirLowerError> {
        match &expr.kind {
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(id) = &callee.kind {
                    if let Some(sig) = self.syms.fns.get(&id.name).cloned() {
                        if sig.is_yielding && !sig.error_spec.has_error() {
                            self.emit_yielding_fn_call(
                                &id.name,
                                &sig,
                                args,
                                Some(MirCallDst::Value(dst_vid)),
                            )?;
                            return Ok(true);
                        }
                    }
                }
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let m = method.name.as_str();
                if let Some((target, pkind)) = self.resolve_prim_receiver(receiver)? {
                    match (m, pkind) {
                        ("wait_chunk", MirPrimKind::PingPong { elem, cap }) => {
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::PingPongWaitChunk {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    dst_slice: dst_vid,
                                    poll_bb,
                                    resume_bb,
                                }),
                                resume_bb,
                                false,
                            );
                            return Ok(true);
                        }
                        ("wait_chunk" | "wait", MirPrimKind::Lease { elem, cap }) => {
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::LeaseWaitChunk {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    dst_slice: dst_vid,
                                    poll_bb,
                                    resume_bb,
                                }),
                                resume_bb,
                                false,
                            );
                            return Ok(true);
                        }
                        ("recv", MirPrimKind::Inbox { elem, cap })
                            if find_named_arg(args, "timeout").is_none() =>
                        {
                            self.cur_stmts.push(MirStmt::EndInboxTurn);
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::InboxRecv {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    timeout_deadline_var: None,
                                    dst_val: Some(dst_vid),
                                    dst_res: None,
                                    poll_bb,
                                    resume_bb,
                                }),
                                resume_bb,
                                false,
                            );
                            return Ok(true);
                        }
                        ("recv", MirPrimKind::Chan { elem, cap })
                            if find_named_arg(args, "timeout").is_none() =>
                        {
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::ChanRecv {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    timeout_deadline_var: None,
                                    dst_val: Some(dst_vid),
                                    dst_res: None,
                                    poll_bb,
                                    resume_bb,
                                }),
                                resume_bb,
                                false,
                            );
                            return Ok(true);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        let _ = dst_ty;
        Ok(false)
    }

    fn is_void_yielding_stmt_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Call { callee, .. } => {
                if let ExprKind::Ident(id) = &callee.kind {
                    if matches!(id.name.as_str(), "delay" | "delay_ms" | "sleep" | "checkpoint") {
                        return true;
                    }
                    if let Some(sig) = self.syms.fns.get(&id.name) {
                        return sig.is_yielding && sig.ret_ty.is_void() && !sig.error_spec.has_error();
                    }
                }
                false
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                if let ExprKind::Ident(id) = &receiver.kind {
                    if id.name == "sys" && method.name == "delay" {
                        return true;
                    }
                }
                let m = method.name.as_str();
                m == "send" || (m == "wait" && find_named_arg(args, "timeout").is_none())
            }
            _ => false,
        }
    }

    fn try_lower_yielding_stmt_expr(&mut self, expr: &Expr) -> Result<bool, MirLowerError> {
        match &expr.kind {
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(id) = &callee.kind {
                    if matches!(id.name.as_str(), "delay" | "delay_ms" | "sleep") {
                        if let Some(first) = args.first() {
                            self.emit_delay(&first.value)?;
                            return Ok(true);
                        }
                    }
                    if id.name == "checkpoint" {
                        let resume_bb = self.alloc_bb();
                        self.seal_bb(
                            MirTerminator::Yield(MirYieldKind::ShareCheckpoint { resume_bb }),
                            resume_bb,
                            true,
                        );
                        return Ok(true);
                    }
                    if let Some(sig) = self.syms.fns.get(&id.name).cloned() {
                        if sig.is_yielding {
                            self.emit_yielding_fn_call(&id.name, &sig, args, None)?;
                            return Ok(true);
                        }
                    }
                }
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                if let ExprKind::Ident(id) = &receiver.kind {
                    if id.name == "sys" && method.name == "delay" {
                        if let Some(first) = args.first() {
                            self.emit_delay(&first.value)?;
                            return Ok(true);
                        }
                    }
                }
                if let Some((target, pkind)) = self.resolve_prim_receiver(receiver)? {
                    let m = method.name.as_str();
                    match (m, pkind) {
                        ("send", MirPrimKind::Chan { elem, cap }) => {
                            let vexpr = if let Some(first) = first_positional_arg(args) {
                                self.lower_expr(first, Some(&elem))?
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                            };
                            let val_var = self.alloc_temp("chan_tx", *elem.clone());
                            self.cur_stmts.push(MirStmt::Assign {
                                target: MirPlace::Var(val_var),
                                op: MirAssignOp::Assign,
                                value: vexpr,
                            });
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::ChanSend {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    val_var,
                                    poll_bb,
                                    resume_bb,
                                }),
                                resume_bb,
                                false,
                            );
                            return Ok(true);
                        }
                        ("wait", MirPrimKind::Signal)
                            if find_named_arg(args, "timeout").is_none() =>
                        {
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::SignalWait {
                                    target,
                                    timeout_deadline_var: None,
                                    dst_res: None,
                                    poll_bb,
                                    resume_bb,
                                }),
                                resume_bb,
                                false,
                            );
                            return Ok(true);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn emit_delay(&mut self, dur_expr: &Expr) -> Result<(), MirLowerError> {
        let ticks = self.lower_expr(dur_expr, Some(&MirType::U32))?;
        let deadline_var = self.alloc_temp("delay_until", MirType::U32);
        self.cur_stmts.push(MirStmt::Assign {
            target: MirPlace::Var(deadline_var),
            op: MirAssignOp::Assign,
            value: MirExpr::new(
                MirExprKind::Binary {
                    op: MirBinOp::Add,
                    lhs: Box::new(MirExpr::new(MirExprKind::NowTicks, MirType::U32)),
                    rhs: Box::new(ticks),
                },
                MirType::U32,
            ),
        });
        let poll_bb = self.alloc_bb();
        let resume_bb = self.alloc_bb();
        self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
        self.seal_bb(
            MirTerminator::Yield(MirYieldKind::DelayPoll {
                deadline_var,
                poll_bb,
                resume_bb,
            }),
            resume_bb,
            false,
        );
        Ok(())
    }

    fn emit_yielding_fn_call(
        &mut self,
        callee: &str,
        sig: &FnSig,
        args: &[CallArg],
        dst: Option<MirCallDst>,
    ) -> Result<(), MirLowerError> {
        let mut largs = Vec::new();
        for (idx, a) in args.iter().enumerate() {
            largs.push(self.lower_expr(&a.value, sig.params.get(idx))?);
        }
        let slot_idx = self.child_slots.len();
        let slot_name = format!("c{slot_idx}_{callee}");
        self.child_slots.push(MirChildCallSlot {
            slot_name: slot_name.clone(),
            callee: callee.to_string(),
            union_group: 0,
        });

        let poll_bb = self.alloc_bb();
        let resume_bb = self.alloc_bb();
        self.seal_bb(
            MirTerminator::Yield(MirYieldKind::CallYielding {
                callee: callee.to_string(),
                slot_name: slot_name.clone(),
                args: largs,
                dst: dst.clone(),
                poll_bb,
                resume_bb,
            }),
            poll_bb,
            true,
        );
        self.seal_bb(
            MirTerminator::Yield(MirYieldKind::PollChild {
                callee: callee.to_string(),
                slot_name,
                dst,
                poll_bb,
                resume_bb,
            }),
            resume_bb,
            false,
        );
        Ok(())
    }

    fn resolve_prim_receiver(
        &mut self,
        receiver: &Expr,
    ) -> Result<Option<(MirPrimTarget, MirPrimKind)>, MirLowerError> {
        if let ExprKind::Ident(id) = &receiver.kind {
            if let Some(pinfo) = self.syms.primitives.get(&id.name) {
                return Ok(Some((
                    MirPrimTarget::Global(id.name.clone()),
                    pinfo.kind.clone(),
                )));
            }
            if let Some(vid) = self.scopes.lookup(&id.name) {
                let vty = self.vars[vid.0].ty.clone();
                if let MirType::Ptr(inner) = &vty {
                    if let MirType::Primitive(pk) = inner.as_ref() {
                        return Ok(Some((
                            MirPrimTarget::PtrExpr(MirExpr::new(MirExprKind::Var(vid), vty.clone())),
                            pk.clone(),
                        )));
                    }
                }
            }
        }
        Ok(None)
    }

    fn lower_expr(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError> {
        match &expr.kind {
            ExprKind::Or {
                expr: inner,
                fallback,
            } => {
                let (res_vid, ok_ty, _) = self.lower_into_result_var(inner)?;
                let out_vid = self.alloc_temp("or_val", ok_ty.clone());
                let err_bb = self.alloc_bb();
                let ok_bb = self.alloc_bb();
                let join_bb = self.alloc_bb();

                self.seal_bb(
                    MirTerminator::Branch {
                        cond: MirExpr::new(MirExprKind::ResultIsErr(res_vid), MirType::Bool),
                        then_bb: err_bb,
                        else_bb: ok_bb,
                    },
                    err_bb,
                    false,
                );
                let fb_val = self.lower_expr(fallback, Some(&ok_ty))?;
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(out_vid),
                    op: MirAssignOp::Assign,
                    value: coerce_expr_if_needed(fb_val, &ok_ty),
                });
                self.seal_bb(MirTerminator::Goto(join_bb), ok_bb, false);
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(out_vid),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(MirExprKind::ResultVal(res_vid), ok_ty.clone()),
                });
                self.seal_bb(MirTerminator::Goto(join_bb), join_bb, false);
                Ok(MirExpr::new(MirExprKind::Var(out_vid), ok_ty))
            }
            ExprKind::Catch {
                expr: inner,
                clause,
            } => {
                let (res_vid, ok_ty, err_ty_opt) = self.lower_into_result_var(inner)?;
                let out_vid_opt = if ok_ty.is_void() {
                    None
                } else {
                    Some(self.alloc_temp("catch_val", ok_ty.clone()))
                };
                let err_bb = self.alloc_bb();
                let ok_bb = self.alloc_bb();
                let join_bb = self.alloc_bb();

                self.seal_bb(
                    MirTerminator::Branch {
                        cond: MirExpr::new(MirExprKind::ResultIsErr(res_vid), MirType::Bool),
                        then_bb: err_bb,
                        else_bb: ok_bb,
                    },
                    err_bb,
                    false,
                );

                self.scopes.push();
                if let Some(ep) = &clause.err_param {
                    let ety = err_ty_opt.unwrap_or_else(|| self.syms.resolve_type(&ep.ty));
                    let evid = self.declare_var(&ep.name.name, ety.clone(), VarRole::UserLocal);
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(evid),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::ResultErr(res_vid), ety),
                    });
                }
                if let Some(pp) = &clause.partial_param {
                    let pty = self.syms.resolve_type(&pp.ty);
                    let pvid = self.declare_var(&pp.name.name, pty.clone(), VarRole::UserLocal);
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(pvid),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::ResultVal(res_vid), pty),
                    });
                }
                if let (Some(out_vid), Some((last_stmt, prefix_stmts))) =
                    (out_vid_opt, clause.body.stmts.split_last())
                {
                    if let StmtKind::Expr(tail_expr) = &last_stmt.kind {
                        self.lower_stmts(prefix_stmts)?;
                        if self.is_void_yielding_stmt_expr(tail_expr)
                            && self.try_lower_yielding_stmt_expr(tail_expr)?
                        {
                            // void yielding statement executed in err_bb
                        } else if let Some(op_stmt) =
                            try_lower_nonblocking_call_or_method(tail_expr, self)?
                        {
                            self.cur_stmts.push(op_stmt);
                        } else {
                            let tail_val = self.lower_expr(tail_expr, Some(&ok_ty))?;
                            if tail_val.ty.is_void() {
                                if matches!(tail_val.kind, MirExprKind::CallDirect { .. }) {
                                    self.cur_stmts.push(MirStmt::Expr(tail_val));
                                }
                            } else {
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(out_vid),
                                    op: MirAssignOp::Assign,
                                    value: coerce_expr_if_needed(tail_val, &ok_ty),
                                });
                            }
                        }
                    } else {
                        self.lower_stmts(&clause.body.stmts)?;
                    }
                } else {
                    self.lower_stmts(&clause.body.stmts)?;
                }
                self.scopes.pop();
                self.seal_bb(MirTerminator::Goto(join_bb), ok_bb, false);

                if let Some(out_vid) = out_vid_opt {
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(out_vid),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::ResultVal(res_vid), ok_ty.clone()),
                    });
                    self.seal_bb(MirTerminator::Goto(join_bb), join_bb, false);
                    Ok(MirExpr::new(MirExprKind::Var(out_vid), ok_ty))
                } else {
                    self.seal_bb(MirTerminator::Goto(join_bb), join_bb, false);
                    Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void))
                }
            }
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(id) = &callee.kind {
                    if let Some(sig) = self.syms.fns.get(&id.name).cloned() {
                        if sig.is_yielding {
                            let env_ty = if sig.error_spec.has_error() {
                                MirType::Result {
                                    ok: Box::new(sig.ret_ty.clone()),
                                    err: sig.error_spec.err_type().map(Box::new),
                                }
                            } else {
                                sig.ret_ty.clone()
                            };
                            let dst_vid = self.alloc_temp("ycall", env_ty.clone());
                            let cdst = if sig.error_spec.has_error() {
                                MirCallDst::Result(dst_vid)
                            } else {
                                MirCallDst::Value(dst_vid)
                            };
                            self.emit_yielding_fn_call(&id.name, &sig, args, Some(cdst))?;
                            return Ok(MirExpr::new(MirExprKind::Var(dst_vid), env_ty));
                        }
                    }
                }
                self.lower_non_yielding_expr(expr, expected)
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                if let Some((target, pkind)) = self.resolve_prim_receiver(receiver)? {
                    let m = method.name.as_str();
                    match (m, pkind) {
                        ("wait_chunk", MirPrimKind::PingPong { elem, cap }) => {
                            let sty = MirType::Slice(elem.clone());
                            let dst_vid = self.alloc_temp("pp_chunk", sty.clone());
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::PingPongWaitChunk {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    dst_slice: dst_vid,
                                    poll_bb,
                                    resume_bb,
                                }),
                                resume_bb,
                                false,
                            );
                            return Ok(MirExpr::new(MirExprKind::Var(dst_vid), sty));
                        }
                        ("wait_chunk" | "wait", MirPrimKind::Lease { elem, cap }) => {
                            let sty = MirType::Slice(elem.clone());
                            let dst_vid = self.alloc_temp("lease_chunk", sty.clone());
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::LeaseWaitChunk {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    dst_slice: dst_vid,
                                    poll_bb,
                                    resume_bb,
                                }),
                                resume_bb,
                                false,
                            );
                            return Ok(MirExpr::new(MirExprKind::Var(dst_vid), sty));
                        }
                        ("recv", MirPrimKind::Inbox { elem, cap }) => {
                            self.cur_stmts.push(MirStmt::EndInboxTurn);
                            if let Some(texpr) = find_named_arg(args, "timeout") {
                                let ticks = self.lower_expr(texpr, Some(&MirType::U32))?;
                                let dl_var = self.alloc_temp("inbox_deadline", MirType::U32);
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(dl_var),
                                    op: MirAssignOp::Assign,
                                    value: MirExpr::new(
                                        MirExprKind::Binary {
                                            op: MirBinOp::Add,
                                            lhs: Box::new(MirExpr::new(
                                                MirExprKind::NowTicks,
                                                MirType::U32,
                                            )),
                                            rhs: Box::new(ticks),
                                        },
                                        MirType::U32,
                                    ),
                                });
                                let rty = MirType::Result {
                                    ok: elem.clone(),
                                    err: None,
                                };
                                let dst_res = self.alloc_temp("inbox_res", rty.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                                self.seal_bb(
                                    MirTerminator::Yield(MirYieldKind::InboxRecv {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        timeout_deadline_var: Some(dl_var),
                                        dst_val: None,
                                        dst_res: Some(dst_res),
                                        poll_bb,
                                        resume_bb,
                                    }),
                                    resume_bb,
                                    false,
                                );
                                return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                            } else {
                                let dst_val = self.alloc_temp("inbox_val", *elem.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                                self.seal_bb(
                                    MirTerminator::Yield(MirYieldKind::InboxRecv {
                                        target,
                                        elem_ty: *elem.clone(),
                                        cap,
                                        timeout_deadline_var: None,
                                        dst_val: Some(dst_val),
                                        dst_res: None,
                                        poll_bb,
                                        resume_bb,
                                    }),
                                    resume_bb,
                                    false,
                                );
                                return Ok(MirExpr::new(MirExprKind::Var(dst_val), *elem));
                            }
                        }
                        ("recv", MirPrimKind::Chan { elem, cap }) => {
                            if let Some(texpr) = find_named_arg(args, "timeout") {
                                let ticks = self.lower_expr(texpr, Some(&MirType::U32))?;
                                let dl_var = self.alloc_temp("chan_deadline", MirType::U32);
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(dl_var),
                                    op: MirAssignOp::Assign,
                                    value: MirExpr::new(
                                        MirExprKind::Binary {
                                            op: MirBinOp::Add,
                                            lhs: Box::new(MirExpr::new(
                                                MirExprKind::NowTicks,
                                                MirType::U32,
                                            )),
                                            rhs: Box::new(ticks),
                                        },
                                        MirType::U32,
                                    ),
                                });
                                let rty = MirType::Result {
                                    ok: elem.clone(),
                                    err: None,
                                };
                                let dst_res = self.alloc_temp("chan_res", rty.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                                self.seal_bb(
                                    MirTerminator::Yield(MirYieldKind::ChanRecv {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        timeout_deadline_var: Some(dl_var),
                                        dst_val: None,
                                        dst_res: Some(dst_res),
                                        poll_bb,
                                        resume_bb,
                                    }),
                                    resume_bb,
                                    false,
                                );
                                return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                            } else {
                                let dst_val = self.alloc_temp("chan_val", *elem.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                                self.seal_bb(
                                    MirTerminator::Yield(MirYieldKind::ChanRecv {
                                        target,
                                        elem_ty: *elem.clone(),
                                        cap,
                                        timeout_deadline_var: None,
                                        dst_val: Some(dst_val),
                                        dst_res: None,
                                        poll_bb,
                                        resume_bb,
                                    }),
                                    resume_bb,
                                    false,
                                );
                                return Ok(MirExpr::new(MirExprKind::Var(dst_val), *elem));
                            }
                        }
                        ("wait", MirPrimKind::Signal) => {
                            let timeout_var = if let Some(texpr) = find_named_arg(args, "timeout") {
                                let ticks = self.lower_expr(texpr, Some(&MirType::U32))?;
                                let dl_var = self.alloc_temp("sig_deadline", MirType::U32);
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(dl_var),
                                    op: MirAssignOp::Assign,
                                    value: MirExpr::new(
                                        MirExprKind::Binary {
                                            op: MirBinOp::Add,
                                            lhs: Box::new(MirExpr::new(
                                                MirExprKind::NowTicks,
                                                MirType::U32,
                                            )),
                                            rhs: Box::new(ticks),
                                        },
                                        MirType::U32,
                                    ),
                                });
                                Some(dl_var)
                            } else {
                                None
                            };
                            let rty = MirType::Result {
                                ok: Box::new(MirType::Void),
                                err: None,
                            };
                            let dst_res = self.alloc_temp("sig_res", rty.clone());
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::SignalWait {
                                    target,
                                    timeout_deadline_var: timeout_var,
                                    dst_res: Some(dst_res),
                                    poll_bb,
                                    resume_bb,
                                }),
                                resume_bb,
                                false,
                            );
                            return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                        }
                        _ => {}
                    }
                }
                self.lower_non_yielding_expr(expr, expected)
            }
            _ => self.lower_non_yielding_expr(expr, expected),
        }
    }

    fn lower_into_result_var(
        &mut self,
        expr: &Expr,
    ) -> Result<(VarId, MirType, Option<MirType>), MirLowerError> {
        let me = self.lower_expr(expr, None)?;
        let (ok_ty, res_ty) = extract_result_types(&me.ty);
        let err_ty_opt = match &res_ty {
            MirType::Result { err, .. } => err.as_deref().cloned(),
            _ => None,
        };
        if let MirExprKind::Var(vid) = me.kind {
            if matches!(self.vars[vid.0].ty, MirType::Result { .. }) {
                return Ok((vid, ok_ty, err_ty_opt));
            }
        }
        let vid = self.alloc_temp("failable_res", res_ty);
        self.cur_stmts.push(MirStmt::Assign {
            target: MirPlace::Var(vid),
            op: MirAssignOp::Assign,
            value: me,
        });
        Ok((vid, ok_ty, err_ty_opt))
    }

    fn lower_non_yielding_expr(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError> {
        // Split borrows cleanly by using temporary helper closures via raw self calls:
        match &expr.kind {
            ExprKind::Unary {
                op: UnaryOp::AddrOf,
                expr: inner,
            } => {
                if let ExprKind::Ident(id) = &inner.kind {
                    if let Some(pinfo) = self.syms.primitives.get(&id.name) {
                        return Ok(MirExpr::new(
                            MirExprKind::GlobalPrimAddr(id.name.clone()),
                            MirType::Ptr(Box::new(MirType::Primitive(pinfo.kind.clone()))),
                        ));
                    }
                    if self.syms.fns.contains_key(&id.name) {
                        return Ok(MirExpr::new(
                            MirExprKind::FnAddr(id.name.clone()),
                            expected.cloned().unwrap_or(MirType::Usize),
                        ));
                    }
                }
                let (place, pty) = self.lower_place(inner)?;
                Ok(MirExpr::new(
                    MirExprKind::AddrOfPlace(Box::new(place)),
                    MirType::Ptr(Box::new(pty)),
                ))
            }
            ExprKind::Unary { op, expr: inner } => {
                let sub = self.lower_expr(inner, expected)?;
                finish_unary(*op, sub)
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let is_cmp_or_logic = matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::Ne
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge
                        | BinaryOp::And
                        | BinaryOp::Or
                );
                let op_expected = if is_cmp_or_logic { None } else { expected };
                let (l, r) = if ast_is_untyped_int_lit(lhs) && !ast_is_untyped_int_lit(rhs) {
                    let r = self.lower_expr(rhs, op_expected)?;
                    let l = self.lower_expr(lhs, Some(&r.ty.host_value_type()))?;
                    (l, r)
                } else {
                    let l = self.lower_expr(lhs, op_expected)?;
                    let r = self.lower_expr(rhs, Some(&l.ty.host_value_type()))?;
                    (l, r)
                };
                Ok(finish_binary(*op, l, r))
            }
            ExprKind::Cast { expr: inner, ty } => {
                let target_ty = self.syms.resolve_type(ty);
                let sub = self.lower_expr(inner, None)?;
                Ok(MirExpr::new(
                    MirExprKind::Cast {
                        expr: Box::new(sub),
                        target_ty: target_ty.clone(),
                    },
                    target_ty,
                ))
            }
            ExprKind::Field { target, field } => {
                if let Some(hw_path) = try_extract_hw_path(expr, self.syms) {
                    return Ok(MirExpr::new(
                        MirExprKind::HwPeriphVal(hw_path.clone()),
                        MirType::HwPeripheral(hw_path.join(".")),
                    ));
                }
                if let ExprKind::Ident(id) = &target.kind {
                    if self.syms.enums.contains_key(&id.name) {
                        return Ok(MirExpr::new(
                            MirExprKind::EnumVariant {
                                enum_name: id.name.clone(),
                                variant: field.name.clone(),
                            },
                            MirType::Enum(id.name.clone()),
                        ));
                    }
                }
                let base = self.lower_expr(target, None)?;
                finish_field_access(base, &field.name, self.syms, expr.span)
            }
            ExprKind::Index { target, index } => {
                let base = self.lower_expr(target, None)?;
                let idx = self.lower_expr(index, Some(&MirType::Usize))?;
                finish_index(base, idx, expr.span)
            }
            ExprKind::Slice { target, start, end } => {
                let s = if let Some(se) = start {
                    self.lower_expr(se, Some(&MirType::Usize))?
                } else {
                    MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)
                };
                if let Ok((place, MirType::Array { elem, len })) = self.lower_place(target) {
                    let e = if let Some(ee) = end {
                        self.lower_expr(ee, Some(&MirType::Usize))?
                    } else {
                        MirExpr::new(MirExprKind::IntLit(len as i128), MirType::Usize)
                    };
                    let slice_expr = MirExpr::new(
                        MirExprKind::ArrayToSlice {
                            array_place: Box::new(place),
                            len,
                            elem_ty: *elem.clone(),
                        },
                        MirType::Slice(elem.clone()),
                    );
                    return Ok(MirExpr::new(
                        MirExprKind::SubSlice {
                            base: Box::new(slice_expr),
                            start: Box::new(s),
                            end: Box::new(e),
                            elem_ty: *elem.clone(),
                            base_is_array: false,
                        },
                        MirType::Slice(elem),
                    ));
                }
                let base = self.lower_expr(target, None)?;
                let elem_ty = match &base.ty {
                    MirType::Slice(inner) => *inner.clone(),
                    MirType::Array { elem, .. } => *elem.clone(),
                    _ => MirType::U8,
                };
                let e = if let Some(ee) = end {
                    self.lower_expr(ee, Some(&MirType::Usize))?
                } else {
                    MirExpr::new(MirExprKind::SliceLen(Box::new(base.clone())), MirType::Usize)
                };
                Ok(MirExpr::new(
                    MirExprKind::SubSlice {
                        base: Box::new(base),
                        start: Box::new(s),
                        end: Box::new(e),
                        elem_ty: elem_ty.clone(),
                        base_is_array: false,
                    },
                    MirType::Slice(Box::new(elem_ty)),
                ))
            }
            ExprKind::StructLit { name, fields } => {
                let st = self.syms.structs.get(&name.name).cloned();
                let packed = st.as_ref().map_or(false, |s| s.packed);
                let mut mfields = Vec::new();
                for fi in fields {
                    let fty = st
                        .as_ref()
                        .and_then(|s| s.fields.iter().find(|f| f.name == fi.name.name))
                        .map(|f| f.ty.clone())
                        .unwrap_or(MirType::U32);
                    let fv = self.lower_expr(&fi.value, Some(&fty.host_value_type()))?;
                    mfields.push((
                        fi.name.name.clone(),
                        coerce_expr_if_needed(fv, &fty.host_value_type()),
                        fty,
                    ));
                }
                Ok(MirExpr::new(
                    MirExprKind::StructLit {
                        struct_name: name.name.clone(),
                        packed,
                        fields: mfields,
                    },
                    MirType::Struct(name.name.clone()),
                ))
            }
            ExprKind::ArrayLit(elems) => {
                let elem_ty = match expected {
                    Some(MirType::Array { elem, .. }) | Some(MirType::Slice(elem)) => {
                        *elem.clone()
                    }
                    _ => MirType::U8,
                };
                let mut melems = Vec::new();
                for el in elems {
                    let me = self.lower_expr(el, Some(&elem_ty))?;
                    melems.push(coerce_expr_if_needed(me, &elem_ty));
                }
                let len = melems.len();
                Ok(MirExpr::new(
                    MirExprKind::ArrayLit {
                        elems: melems,
                        elem_ty: elem_ty.clone(),
                    },
                    MirType::Array {
                        elem: Box::new(elem_ty),
                        len,
                    },
                ))
            }
            ExprKind::Call { callee, args } => {
                let mut largs = Vec::new();
                if let ExprKind::Ident(id) = &callee.kind {
                    if matches!(id.name.as_str(), "now_ticks" | "now_ms") {
                        return Ok(MirExpr::new(MirExprKind::NowTicks, MirType::U32));
                    }
                    if let Some(sig) = self.syms.fns.get(&id.name).cloned() {
                        for (idx, a) in args.iter().enumerate() {
                            let me = self.lower_expr(&a.value, sig.params.get(idx))?;
                            let coerced = if let Some(pty) = sig.params.get(idx) {
                                self.coerce_arg_for_param(me, pty)
                            } else {
                                me
                            };
                            largs.push(coerced);
                        }
                        let ret = if sig.error_spec.has_error() {
                            MirType::Result {
                                ok: Box::new(sig.ret_ty),
                                err: sig.error_spec.err_type().map(Box::new),
                            }
                        } else {
                            sig.ret_ty
                        };
                        return Ok(MirExpr::new(
                            MirExprKind::CallDirect {
                                callee: id.name.clone(),
                                is_extern_c: sig.is_extern_c,
                                args: largs,
                            },
                            ret,
                        ));
                    }
                    for a in args {
                        largs.push(self.lower_expr(&a.value, None)?);
                    }
                    return Ok(MirExpr::new(
                        MirExprKind::CallDirect {
                            callee: id.name.clone(),
                            is_extern_c: true,
                            args: largs,
                        },
                        expected.cloned().unwrap_or(MirType::Void),
                    ));
                }
                Err(MirLowerError {
                    message: "unsupported indirect function call target".to_string(),
                    span: expr.span,
                })
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                // `Type.from_bytes(slice)`
                if let ExprKind::Ident(id) = &receiver.kind {
                    if self.syms.structs.contains_key(&id.name) && method.name == "from_bytes" {
                        let sarg = first_positional_arg(args).ok_or_else(|| MirLowerError {
                            message: "from_bytes expects a slice argument".to_string(),
                            span: expr.span,
                        })?;
                        let slice_ty = MirType::Slice(Box::new(MirType::U8));
                        let s = self.lower_expr(sarg, Some(&slice_ty))?;
                        let s = self.coerce_arg_for_param(s, &slice_ty);
                        return Ok(MirExpr::new(
                            MirExprKind::FromBytes {
                                struct_name: id.name.clone(),
                                slice: Box::new(s),
                            },
                            MirType::Result {
                                ok: Box::new(MirType::Struct(id.name.clone())),
                                err: None,
                            },
                        ));
                    }
                    if id.name == "sys" && matches!(method.name.as_str(), "now_ms" | "now_ticks") {
                        return Ok(MirExpr::new(MirExprKind::NowTicks, MirType::U32));
                    }
                }
                if let Some((target, pkind)) = self.resolve_prim_receiver(receiver)? {
                    return finish_prim_read_method(target, pkind, &method.name, expr.span);
                }
                Err(MirLowerError {
                    message: format!("unsupported method expression `.{}()`", method.name),
                    span: expr.span,
                })
            }
            _ => lower_leaf_expr(expr, expected, self.syms, &self.vars, &self.scopes),
        }
    }

    fn coerce_arg_for_param(&mut self, arg: MirExpr, param_ty: &MirType) -> MirExpr {
        if let (
            MirType::Array { elem, len },
            MirType::Slice(_),
            MirExprKind::Var(vid),
        ) = (&arg.ty, param_ty, &arg.kind)
        {
            return MirExpr::new(
                MirExprKind::ArrayToSlice {
                    array_place: Box::new(MirPlace::Var(*vid)),
                    len: *len,
                    elem_ty: *elem.clone(),
                },
                param_ty.clone(),
            );
        }
        coerce_expr_if_needed(arg, param_ty)
    }

    fn lower_place(&mut self, expr: &Expr) -> Result<(MirPlace, MirType), MirLowerError> {
        lower_place_common(expr, self)
    }
}

trait LowerBackend {
    fn syms(&self) -> &ModuleSymbols;
    fn vars(&self) -> &[MirVar];
    fn scopes(&self) -> &VarScopes;
    fn lower_sub(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError>;
    fn lower_place_be(&mut self, expr: &Expr) -> Result<(MirPlace, MirType), MirLowerError>;
}

struct DirectBackend<'b, 'a> {
    ctx: &'b mut DirectLowerCtx<'a>,
    stmts: &'b mut Vec<MirStmt>,
}

impl<'b, 'a> LowerBackend for DirectBackend<'b, 'a> {
    fn syms(&self) -> &ModuleSymbols {
        self.ctx.syms
    }
    fn vars(&self) -> &[MirVar] {
        &self.ctx.vars
    }
    fn scopes(&self) -> &VarScopes {
        &self.ctx.scopes
    }
    fn lower_sub(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError> {
        self.ctx.lower_expr_with_stmts(expr, expected, self.stmts)
    }
    fn lower_place_be(&mut self, expr: &Expr) -> Result<(MirPlace, MirType), MirLowerError> {
        lower_place_common(expr, self)
    }
}

impl<'a> LowerBackend for CoroutineLowerCtx<'a> {
    fn syms(&self) -> &ModuleSymbols {
        self.syms
    }
    fn vars(&self) -> &[MirVar] {
        &self.vars
    }
    fn scopes(&self) -> &VarScopes {
        &self.scopes
    }
    fn lower_sub(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError> {
        self.lower_expr(expr, expected)
    }
    fn lower_place_be(&mut self, expr: &Expr) -> Result<(MirPlace, MirType), MirLowerError> {
        lower_place_common(expr, self)
    }
}

trait AssignOpExt {
    fn into_ast_check(self) -> AssignOp;
}

impl AssignOpExt for MirAssignOp {
    fn into_ast_check(self) -> AssignOp {
        AssignOp::Assign
    }
}

fn ast_stmts_have_inbox_recv(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::VarDecl { init: Some(e), .. }
            | StmtKind::Assign { value: e, .. }
            | StmtKind::Expr(e) => {
                if ast_expr_has_inbox_recv(e) {
                    return true;
                }
            }
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                if ast_expr_has_inbox_recv(cond)
                    || ast_stmts_have_inbox_recv(&then_block.stmts)
                    || else_block
                        .as_ref()
                        .map_or(false, |eb| ast_stmts_have_inbox_recv(&eb.stmts))
                {
                    return true;
                }
            }
            StmtKind::Block(b) => {
                if ast_stmts_have_inbox_recv(&b.stmts) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn ast_expr_has_inbox_recv(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::MethodCall { method, .. } if method.name == "recv" => true,
        ExprKind::Or { expr: l, fallback: r } => {
            ast_expr_has_inbox_recv(l) || ast_expr_has_inbox_recv(r)
        }
        ExprKind::Catch { expr: inner, .. } => ast_expr_has_inbox_recv(inner),
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure / Leaf / Non-blocking Shared Lowering Routines
// ─────────────────────────────────────────────────────────────────────────────

fn extract_result_types(ty: &MirType) -> (MirType, MirType) {
    match ty {
        MirType::Result { ok, .. } => (*ok.clone(), ty.clone()),
        other => (
            other.clone(),
            MirType::Result {
                ok: Box::new(other.clone()),
                err: None,
            },
        ),
    }
}

fn coerce_expr_if_needed(expr: MirExpr, target_ty: &MirType) -> MirExpr {
    if &expr.ty == target_ty || target_ty.is_void() {
        return expr;
    }
    if let MirExprKind::IntLit(v) = expr.kind {
        if matches!(
            target_ty.host_value_type(),
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
        ) {
            return MirExpr::new(MirExprKind::IntLit(v), target_ty.host_value_type());
        }
    }
    expr
}

fn lower_leaf_expr(
    expr: &Expr,
    expected: Option<&MirType>,
    syms: &ModuleSymbols,
    vars: &[MirVar],
    scopes: &VarScopes,
) -> Result<MirExpr, MirLowerError> {
    match &expr.kind {
        ExprKind::Literal(lit) => match lit {
            Literal::Int(v) => {
                let ty = expected
                    .map(|t| t.host_value_type())
                    .filter(|t| {
                        matches!(
                            t,
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
                        )
                    })
                    .unwrap_or(MirType::U32);
                Ok(MirExpr::new(MirExprKind::IntLit(*v as i128), ty))
            }
            Literal::Duration(d) => {
                let ticks = syms.duration_to_ticks(d)?;
                Ok(MirExpr::new(MirExprKind::TicksLit(ticks), MirType::U32))
            }
            Literal::Bool(b) => Ok(MirExpr::new(MirExprKind::BoolLit(*b), MirType::Bool)),
            Literal::String(s) => Ok(MirExpr::new(MirExprKind::StringLit(s.clone()), MirType::Str)),
        },
        ExprKind::Priority(p) => Ok(MirExpr::new(
            MirExprKind::IntLit(convert_priority(*p).as_u8() as i128),
            MirType::U8,
        )),
        ExprKind::Ident(id) => {
            if let Some(vid) = scopes.lookup(&id.name) {
                let vty = vars[vid.0].ty.clone();
                return Ok(MirExpr::new(MirExprKind::Var(vid), vty));
            }
            if let Some((ename, _)) = syms.variant_to_enum.get(&id.name) {
                return Ok(MirExpr::new(
                    MirExprKind::EnumVariant {
                        enum_name: ename.clone(),
                        variant: id.name.clone(),
                    },
                    MirType::Enum(ename.clone()),
                ));
            }
            if let Some(cty) = syms.consts.get(&id.name) {
                return Ok(MirExpr::new(
                    MirExprKind::ConstRef(id.name.clone()),
                    cty.clone(),
                ));
            }
            if let Some(val) = built_in_hw_const(&id.name) {
                return Ok(MirExpr::new(MirExprKind::IntLit(val), MirType::U32));
            }
            if let Some(pinfo) = syms.primitives.get(&id.name) {
                return Ok(MirExpr::new(
                    MirExprKind::GlobalPrimAddr(id.name.clone()),
                    MirType::Ptr(Box::new(MirType::Primitive(pinfo.kind.clone()))),
                ));
            }
            if syms.fns.contains_key(&id.name) {
                return Ok(MirExpr::new(
                    MirExprKind::FnAddr(id.name.clone()),
                    expected.cloned().unwrap_or(MirType::Usize),
                ));
            }
            Err(MirLowerError {
                message: format!("unresolved identifier `{}`", id.name),
                span: id.span,
            })
        }
        _ => Err(MirLowerError {
            message: "unexpected expression form".to_string(),
            span: expr.span,
        }),
    }
}

fn lower_common_expr<B: LowerBackend>(
    expr: &Expr,
    expected: Option<&MirType>,
    be: &mut B,
) -> Result<MirExpr, MirLowerError> {
    match &expr.kind {
        ExprKind::Unary {
            op: UnaryOp::AddrOf,
            expr: inner,
        } => {
            if let ExprKind::Ident(id) = &inner.kind {
                if let Some(pinfo) = be.syms().primitives.get(&id.name) {
                    return Ok(MirExpr::new(
                        MirExprKind::GlobalPrimAddr(id.name.clone()),
                        MirType::Ptr(Box::new(MirType::Primitive(pinfo.kind.clone()))),
                    ));
                }
                if be.syms().fns.contains_key(&id.name) {
                    return Ok(MirExpr::new(
                        MirExprKind::FnAddr(id.name.clone()),
                        expected.cloned().unwrap_or(MirType::Usize),
                    ));
                }
            }
            let (place, pty) = be.lower_place_be(inner)?;
            Ok(MirExpr::new(
                MirExprKind::AddrOfPlace(Box::new(place)),
                MirType::Ptr(Box::new(pty)),
            ))
        }
        ExprKind::Unary { op, expr: inner } => {
            let sub = be.lower_sub(inner, expected)?;
            finish_unary(*op, sub)
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let is_cmp_or_logic = matches!(
                op,
                BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::And
                    | BinaryOp::Or
            );
            let op_expected = if is_cmp_or_logic { None } else { expected };
            let (l, r) = if ast_is_untyped_int_lit(lhs) && !ast_is_untyped_int_lit(rhs) {
                let r = be.lower_sub(rhs, op_expected)?;
                let l = be.lower_sub(lhs, Some(&r.ty.host_value_type()))?;
                (l, r)
            } else {
                let l = be.lower_sub(lhs, op_expected)?;
                let r = be.lower_sub(rhs, Some(&l.ty.host_value_type()))?;
                (l, r)
            };
            Ok(finish_binary(*op, l, r))
        }
        ExprKind::Cast { expr: inner, ty } => {
            let target_ty = be.syms().resolve_type(ty);
            let sub = be.lower_sub(inner, None)?;
            Ok(MirExpr::new(
                MirExprKind::Cast {
                    expr: Box::new(sub),
                    target_ty: target_ty.clone(),
                },
                target_ty,
            ))
        }
        ExprKind::Field { target, field } => {
            if let Some(hw_path) = try_extract_hw_path(expr, be.syms()) {
                return Ok(MirExpr::new(
                    MirExprKind::HwPeriphVal(hw_path.clone()),
                    MirType::HwPeripheral(hw_path.join(".")),
                ));
            }
            if let ExprKind::Ident(id) = &target.kind {
                if be.syms().enums.contains_key(&id.name) {
                    return Ok(MirExpr::new(
                        MirExprKind::EnumVariant {
                            enum_name: id.name.clone(),
                            variant: field.name.clone(),
                        },
                        MirType::Enum(id.name.clone()),
                    ));
                }
            }
            let base = be.lower_sub(target, None)?;
            finish_field_access(base, &field.name, be.syms(), expr.span)
        }
        ExprKind::Index { target, index } => {
            let base = be.lower_sub(target, None)?;
            let idx = be.lower_sub(index, Some(&MirType::Usize))?;
            finish_index(base, idx, expr.span)
        }
        ExprKind::Slice { target, start, end } => {
            let s = if let Some(se) = start {
                be.lower_sub(se, Some(&MirType::Usize))?
            } else {
                MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)
            };
            if let Ok((place, MirType::Array { elem, len })) = be.lower_place_be(target) {
                let e = if let Some(ee) = end {
                    be.lower_sub(ee, Some(&MirType::Usize))?
                } else {
                    MirExpr::new(MirExprKind::IntLit(len as i128), MirType::Usize)
                };
                let slice_expr = MirExpr::new(
                    MirExprKind::ArrayToSlice {
                        array_place: Box::new(place),
                        len,
                        elem_ty: *elem.clone(),
                    },
                    MirType::Slice(elem.clone()),
                );
                return Ok(MirExpr::new(
                    MirExprKind::SubSlice {
                        base: Box::new(slice_expr),
                        start: Box::new(s),
                        end: Box::new(e),
                        elem_ty: *elem.clone(),
                        base_is_array: false,
                    },
                    MirType::Slice(elem),
                ));
            }
            let base = be.lower_sub(target, None)?;
            let elem_ty = match &base.ty {
                MirType::Slice(inner) => *inner.clone(),
                MirType::Array { elem, .. } => *elem.clone(),
                _ => MirType::U8,
            };
            let e = if let Some(ee) = end {
                be.lower_sub(ee, Some(&MirType::Usize))?
            } else {
                MirExpr::new(MirExprKind::SliceLen(Box::new(base.clone())), MirType::Usize)
            };
            Ok(MirExpr::new(
                MirExprKind::SubSlice {
                    base: Box::new(base),
                    start: Box::new(s),
                    end: Box::new(e),
                    elem_ty: elem_ty.clone(),
                    base_is_array: false,
                },
                MirType::Slice(Box::new(elem_ty)),
            ))
        }
        ExprKind::StructLit { name, fields } => {
            let st = be.syms().structs.get(&name.name).cloned();
            let packed = st.as_ref().map_or(false, |s| s.packed);
            let mut mfields = Vec::new();
            for fi in fields {
                let fty = st
                    .as_ref()
                    .and_then(|s| s.fields.iter().find(|f| f.name == fi.name.name))
                    .map(|f| f.ty.clone())
                    .unwrap_or(MirType::U32);
                let fv = be.lower_sub(&fi.value, Some(&fty.host_value_type()))?;
                mfields.push((
                    fi.name.name.clone(),
                    coerce_expr_if_needed(fv, &fty.host_value_type()),
                    fty,
                ));
            }
            Ok(MirExpr::new(
                MirExprKind::StructLit {
                    struct_name: name.name.clone(),
                    packed,
                    fields: mfields,
                },
                MirType::Struct(name.name.clone()),
            ))
        }
        ExprKind::ArrayLit(elems) => {
            let elem_ty = match expected {
                Some(MirType::Array { elem, .. }) | Some(MirType::Slice(elem)) => *elem.clone(),
                _ => MirType::U8,
            };
            let mut melems = Vec::new();
            for el in elems {
                let me = be.lower_sub(el, Some(&elem_ty))?;
                melems.push(coerce_expr_if_needed(me, &elem_ty));
            }
            let len = melems.len();
            Ok(MirExpr::new(
                MirExprKind::ArrayLit {
                    elems: melems,
                    elem_ty: elem_ty.clone(),
                },
                MirType::Array {
                    elem: Box::new(elem_ty),
                    len,
                },
            ))
        }
        ExprKind::Call { callee, args } => {
            let mut largs = Vec::new();
            if let ExprKind::Ident(id) = &callee.kind {
                if matches!(id.name.as_str(), "now_ticks" | "now_ms") {
                    return Ok(MirExpr::new(MirExprKind::NowTicks, MirType::U32));
                }
                if let Some(sig) = be.syms().fns.get(&id.name).cloned() {
                    for (idx, a) in args.iter().enumerate() {
                        let me = be.lower_sub(&a.value, sig.params.get(idx))?;
                        let coerced = match (&me.ty, sig.params.get(idx), &me.kind) {
                            (
                                MirType::Array { elem, len },
                                Some(MirType::Slice(_)),
                                MirExprKind::Var(vid),
                            ) => MirExpr::new(
                                MirExprKind::ArrayToSlice {
                                    array_place: Box::new(MirPlace::Var(*vid)),
                                    len: *len,
                                    elem_ty: *elem.clone(),
                                },
                                sig.params[idx].clone(),
                            ),
                            (_, Some(pty), _) => coerce_expr_if_needed(me, pty),
                            _ => me,
                        };
                        largs.push(coerced);
                    }
                    let ret = if sig.error_spec.has_error() {
                        MirType::Result {
                            ok: Box::new(sig.ret_ty),
                            err: sig.error_spec.err_type().map(Box::new),
                        }
                    } else {
                        sig.ret_ty
                    };
                    return Ok(MirExpr::new(
                        MirExprKind::CallDirect {
                            callee: id.name.clone(),
                            is_extern_c: sig.is_extern_c,
                            args: largs,
                        },
                        ret,
                    ));
                }
                for a in args {
                    largs.push(be.lower_sub(&a.value, None)?);
                }
                return Ok(MirExpr::new(
                    MirExprKind::CallDirect {
                        callee: id.name.clone(),
                        is_extern_c: true,
                        args: largs,
                    },
                    expected.cloned().unwrap_or(MirType::Void),
                ));
            }
            Err(MirLowerError {
                message: "unsupported function call target".to_string(),
                span: expr.span,
            })
        }
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => {
            if let ExprKind::Ident(id) = &receiver.kind {
                if be.syms().structs.contains_key(&id.name) && method.name == "from_bytes" {
                    let sarg = first_positional_arg(args).ok_or_else(|| MirLowerError {
                        message: "from_bytes expects a slice argument".to_string(),
                        span: expr.span,
                    })?;
                    let slice_ty = MirType::Slice(Box::new(MirType::U8));
                    let s = be.lower_sub(sarg, Some(&slice_ty))?;
                    let s = match (&s.ty, &s.kind) {
                        (MirType::Array { elem, len }, MirExprKind::Var(vid)) => MirExpr::new(
                            MirExprKind::ArrayToSlice {
                                array_place: Box::new(MirPlace::Var(*vid)),
                                len: *len,
                                elem_ty: *elem.clone(),
                            },
                            slice_ty,
                        ),
                        _ => s,
                    };
                    return Ok(MirExpr::new(
                        MirExprKind::FromBytes {
                            struct_name: id.name.clone(),
                            slice: Box::new(s),
                        },
                        MirType::Result {
                            ok: Box::new(MirType::Struct(id.name.clone())),
                            err: None,
                        },
                    ));
                }
                if id.name == "sys" && matches!(method.name.as_str(), "now_ms" | "now_ticks") {
                    return Ok(MirExpr::new(MirExprKind::NowTicks, MirType::U32));
                }
                if let Some(pinfo) = be.syms().primitives.get(&id.name) {
                    return finish_prim_read_method(
                        MirPrimTarget::Global(id.name.clone()),
                        pinfo.kind.clone(),
                        &method.name,
                        expr.span,
                    );
                }
                if let Some(vid) = be.scopes().lookup(&id.name) {
                    let vty = be.vars()[vid.0].ty.clone();
                    let pk_opt = match &vty {
                        MirType::Ptr(inner) => match inner.as_ref() {
                            MirType::Primitive(pk) => Some(pk.clone()),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(pk) = pk_opt {
                        return finish_prim_read_method(
                            MirPrimTarget::PtrExpr(MirExpr::new(MirExprKind::Var(vid), vty)),
                            pk,
                            &method.name,
                            expr.span,
                        );
                    }
                }
            }
            Err(MirLowerError {
                message: format!("unsupported method call `.{}()`", method.name),
                span: expr.span,
            })
        }
        _ => lower_leaf_expr(expr, expected, be.syms(), be.vars(), be.scopes()),
    }
}

fn finish_unary(op: UnaryOp, sub: MirExpr) -> Result<MirExpr, MirLowerError> {
    match op {
        UnaryOp::Deref => {
            let elem = match &sub.ty {
                MirType::Ptr(inner) => *inner.clone(),
                _ => sub.ty.clone(),
            };
            Ok(MirExpr::new(
                MirExprKind::Unary {
                    op: MirUnaryOp::Deref,
                    expr: Box::new(sub),
                },
                elem,
            ))
        }
        UnaryOp::Not => Ok(MirExpr::new(
            MirExprKind::Unary {
                op: MirUnaryOp::Not,
                expr: Box::new(sub),
            },
            MirType::Bool,
        )),
        UnaryOp::BitNot => {
            let ty = sub.ty.clone();
            Ok(MirExpr::new(
                MirExprKind::Unary {
                    op: MirUnaryOp::BitNot,
                    expr: Box::new(sub),
                },
                ty,
            ))
        }
        UnaryOp::Neg => {
            let signed_ty = match sub.ty.host_value_type() {
                MirType::U64 => MirType::I64,
                MirType::Usize => MirType::Isize,
                MirType::U8 | MirType::U16 | MirType::U32 => MirType::I32,
                other => other,
            };
            let coerced_sub = coerce_or_cast_int(sub, &signed_ty);
            Ok(MirExpr::new(
                MirExprKind::Unary {
                    op: MirUnaryOp::Neg,
                    expr: Box::new(coerced_sub),
                },
                signed_ty,
            ))
        }
        UnaryOp::AddrOf => unreachable!(),
    }
}

fn ast_is_untyped_int_lit(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Literal(Literal::Int(_)) => true,
        ExprKind::Unary {
            op: UnaryOp::Neg,
            expr: inner,
        } => matches!(inner.kind, ExprKind::Literal(Literal::Int(_))),
        _ => false,
    }
}

fn is_scalar_int_mir_type(ty: &MirType) -> bool {
    matches!(
        ty.host_value_type(),
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
    )
}

fn is_signed_int_mir_type(ty: &MirType) -> bool {
    matches!(
        ty.host_value_type(),
        MirType::I8 | MirType::I16 | MirType::I32 | MirType::I64 | MirType::Isize
    )
}

fn coerce_or_cast_int(expr: MirExpr, target_ty: &MirType) -> MirExpr {
    let target = target_ty.host_value_type();
    if expr.ty.host_value_type() == target || target.is_void() {
        return expr;
    }
    if let MirExprKind::IntLit(v) = expr.kind {
        if is_scalar_int_mir_type(&target) {
            return MirExpr::new(MirExprKind::IntLit(v), target);
        }
        return MirExpr::new(MirExprKind::IntLit(v), expr.ty);
    }
    MirExpr::new(
        MirExprKind::Cast {
            expr: Box::new(expr),
            target_ty: target.clone(),
        },
        target,
    )
}

fn finish_binary(op: BinaryOp, mut lhs: MirExpr, mut rhs: MirExpr) -> MirExpr {
    // If one operand has a concrete signed integer type and the other is an unsigned integer
    // literal (e.g. `300 - (cur as i32)`), unify both operands to the signed integer type!
    if is_signed_int_mir_type(&rhs.ty) && lhs.ty.is_unsigned() && is_scalar_int_mir_type(&lhs.ty) {
        let target = rhs.ty.host_value_type();
        lhs = coerce_or_cast_int(lhs, &target);
    } else if is_signed_int_mir_type(&lhs.ty)
        && rhs.ty.is_unsigned()
        && is_scalar_int_mir_type(&rhs.ty)
    {
        let target = lhs.ty.host_value_type();
        rhs = coerce_or_cast_int(rhs, &target);
    }

    let mop = convert_bin_op(op);
    let out_ty = match mop {
        MirBinOp::Eq
        | MirBinOp::Ne
        | MirBinOp::Lt
        | MirBinOp::Le
        | MirBinOp::Gt
        | MirBinOp::Ge
        | MirBinOp::LogicalAnd
        | MirBinOp::LogicalOr => MirType::Bool,
        _ => lhs.ty.host_value_type(),
    };
    let rhs_coerced = coerce_expr_if_needed(rhs, &lhs.ty.host_value_type());
    MirExpr::new(
        MirExprKind::Binary {
            op: mop,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs_coerced),
        },
        out_ty,
    )
}

fn finish_field_access(
    base: MirExpr,
    field: &str,
    syms: &ModuleSymbols,
    span: Span,
) -> Result<MirExpr, MirLowerError> {
    if field == "len" {
        match &base.ty {
            MirType::Slice(_) => {
                return Ok(MirExpr::new(MirExprKind::SliceLen(Box::new(base)), MirType::Usize));
            }
            MirType::Array { len, .. } => {
                return Ok(MirExpr::new(MirExprKind::ArrayLen(*len), MirType::Usize));
            }
            _ => {}
        }
    }
    let (sname, is_ptr) = match &base.ty {
        MirType::Struct(n) => (n.clone(), false),
        MirType::Ptr(inner) => match inner.as_ref() {
            MirType::Struct(n) => (n.clone(), true),
            _ => {
                return Err(MirLowerError {
                    message: format!("cannot access field `{field}` on pointer `{}`", base.ty.display_name()),
                    span,
                });
            }
        },
        _ => {
            return Err(MirLowerError {
                message: format!("cannot access field `{field}` on `{}`", base.ty.display_name()),
                span,
            });
        }
    };
    let st = syms.structs.get(&sname).ok_or_else(|| MirLowerError {
        message: format!("unknown struct `{sname}`"),
        span,
    })?;
    let sf = st
        .fields
        .iter()
        .find(|f| f.name == field)
        .ok_or_else(|| MirLowerError {
            message: format!("struct `{sname}` has no field `{field}`"),
            span,
        })?;
    let host_ty = sf.ty.host_value_type();
    Ok(MirExpr::new(
        MirExprKind::FieldAccess {
            base: Box::new(base),
            struct_name: sname,
            packed: st.packed,
            is_ptr,
            field: field.to_string(),
            raw_field_ty: sf.ty.clone(),
        },
        host_ty,
    ))
}

fn finish_index(base: MirExpr, idx: MirExpr, span: Span) -> Result<MirExpr, MirLowerError> {
    match &base.ty {
        MirType::Slice(elem) => {
            let et = elem.host_value_type();
            Ok(MirExpr::new(
                MirExprKind::Index {
                    base: Box::new(base),
                    index: Box::new(idx),
                    is_slice: true,
                },
                et,
            ))
        }
        MirType::Array { elem, .. } => {
            let et = elem.host_value_type();
            Ok(MirExpr::new(
                MirExprKind::Index {
                    base: Box::new(base),
                    index: Box::new(idx),
                    is_slice: false,
                },
                et,
            ))
        }
        _ => Err(MirLowerError {
            message: format!("cannot index into `{}`", base.ty.display_name()),
            span,
        }),
    }
}

fn finish_prim_read_method(
    target: MirPrimTarget,
    pkind: MirPrimKind,
    method: &str,
    span: Span,
) -> Result<MirExpr, MirLowerError> {
    match (method, pkind) {
        ("read" | "get", MirPrimKind::Watch { elem }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                target,
                elem_ty: *elem.clone(),
            })),
            *elem,
        )),
        ("try_recv", MirPrimKind::Inbox { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxTryRecv {
                target,
                elem_ty: *elem.clone(),
                cap,
            })),
            MirType::Result {
                ok: elem,
                err: None,
            },
        )),
        ("try_recv", MirPrimKind::Chan { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ChanTryRecv {
                target,
                elem_ty: *elem.clone(),
                cap,
            })),
            MirType::Result {
                ok: elem,
                err: None,
            },
        )),
        ("is_set", MirPrimKind::Signal) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::SignalIsSet { target })),
            MirType::Bool,
        )),
        _ => Err(MirLowerError {
            message: format!("unknown primitive read method `.{method}()`"),
            span,
        }),
    }
}

fn lower_place_common<B: LowerBackend>(
    expr: &Expr,
    be: &mut B,
) -> Result<(MirPlace, MirType), MirLowerError> {
    match &expr.kind {
        ExprKind::Ident(id) => {
            if let Some(vid) = be.scopes().lookup(&id.name) {
                let ty = be.vars()[vid.0].ty.clone();
                return Ok((MirPlace::Var(vid), ty));
            }
            Err(MirLowerError {
                message: format!("cannot assign/borrow unknown local `{}`", id.name),
                span: id.span,
            })
        }
        ExprKind::Unary {
            op: UnaryOp::Deref,
            expr: inner,
        } => {
            let ptr = be.lower_sub(inner, None)?;
            let pointee = match &ptr.ty {
                MirType::Ptr(t) => *t.clone(),
                other => other.clone(),
            };
            Ok((MirPlace::Deref(Box::new(ptr), pointee.clone()), pointee))
        }
        ExprKind::Field { target, field } => {
            let (base_place, base_ty) = match lower_place_common(target, be) {
                Ok((_p, MirType::Ptr(inner))) => {
                    let ptr_expr = be.lower_sub(target, None)?;
                    (MirPlace::Deref(Box::new(ptr_expr), *inner.clone()), *inner)
                }
                Ok(pair) => pair,
                Err(_) => {
                    let ptr_expr = be.lower_sub(target, None)?;
                    let pointee = match &ptr_expr.ty {
                        MirType::Ptr(inner) => *inner.clone(),
                        other => other.clone(),
                    };
                    (MirPlace::Deref(Box::new(ptr_expr), pointee.clone()), pointee)
                }
            };
            let sname = match &base_ty {
                MirType::Struct(n) => n.clone(),
                _ => {
                    return Err(MirLowerError {
                        message: format!("field assignment on non-struct `{}`", base_ty.display_name()),
                        span: expr.span,
                    });
                }
            };
            let st = be.syms().structs.get(&sname).ok_or_else(|| MirLowerError {
                message: format!("unknown struct `{sname}`"),
                span: expr.span,
            })?;
            let sf = st
                .fields
                .iter()
                .find(|f| f.name == field.name)
                .ok_or_else(|| MirLowerError {
                    message: format!("struct `{sname}` has no field `{}`", field.name),
                    span: expr.span,
                })?;
            Ok((
                MirPlace::Field {
                    base: Box::new(base_place),
                    struct_name: sname,
                    packed: st.packed,
                    field: field.name.clone(),
                    field_ty: sf.ty.clone(),
                },
                sf.ty.clone(),
            ))
        }
        ExprKind::Index { target, index } => {
            let (base_place, base_ty) = lower_place_common(target, be)?;
            let idx = be.lower_sub(index, Some(&MirType::Usize))?;
            match base_ty {
                MirType::Slice(elem) => Ok((
                    MirPlace::Index {
                        base: Box::new(base_place),
                        index: Box::new(idx),
                        elem_ty: *elem.clone(),
                        is_slice: true,
                    },
                    *elem,
                )),
                MirType::Array { elem, .. } => Ok((
                    MirPlace::Index {
                        base: Box::new(base_place),
                        index: Box::new(idx),
                        elem_ty: *elem.clone(),
                        is_slice: false,
                    },
                    *elem,
                )),
                _ => Err(MirLowerError {
                    message: format!("cannot index-assign `{}`", base_ty.display_name()),
                    span: expr.span,
                }),
            }
        }
        _ => Err(MirLowerError {
            message: "invalid assignment target".to_string(),
            span: expr.span,
        }),
    }
}

fn try_lower_nonblocking_call_or_method<B: LowerBackend>(
    expr: &Expr,
    be: &mut B,
) -> Result<Option<MirStmt>, MirLowerError> {
    match &expr.kind {
        ExprKind::Call { callee, args } => {
            if let ExprKind::Ident(id) = &callee.kind {
                if id.name == "set_priority" {
                    if let Some(first) = args.first() {
                        if let ExprKind::Priority(p) = &first.value.kind {
                            return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::SetPriority {
                                prio: convert_priority(*p),
                            })));
                        }
                    }
                }
            }
            Ok(None)
        }
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => {
            // 1. `sys.*` intrinsics (excluding pure value queries `now_ms` / `now_ticks`)
            if let ExprKind::Ident(id) = &receiver.kind {
                if id.name == "sys" && !matches!(method.name.as_str(), "now_ms" | "now_ticks") {
                    if method.name == "use_systick" {
                        let hz = if let Some(first) = first_positional_arg(args) {
                            be.lower_sub(first, Some(&MirType::U32))?
                        } else {
                            MirExpr::new(MirExprKind::IntLit(1_000_000), MirType::U32)
                        };
                        return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::UseSystick { hz })));
                    }
                    let mut largs = Vec::new();
                    for (idx, a) in args.iter().enumerate() {
                        let exp = match (method.name.as_str(), idx) {
                            ("log", 0) | ("log_kv", 0) | ("log_hex", 0) => Some(&MirType::Str),
                            ("log_kv", 1) | ("exit", 0) => Some(&MirType::I32),
                            ("log_hex", 1) | ("pend_irq", 0) => Some(&MirType::U32),
                            _ => None,
                        };
                        let lowered = be.lower_sub(&a.value, exp)?;
                        let coerced = if let Some(ety) = exp {
                            coerce_expr_if_needed(lowered, ety)
                        } else {
                            lowered
                        };
                        largs.push(coerced);
                    }
                    return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                        method: method.name.clone(),
                        args: largs,
                    })));
                }
            }

            // 2. `hw.<PERIPH>.<method>(...)` SVD operations
            if let Some(periph_path) = try_extract_hw_path(receiver, be.syms()) {
                let mut largs = Vec::new();
                for a in args {
                    largs.push(be.lower_sub(&a.value, None)?);
                }
                return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::HwOp {
                    periph_path,
                    method: method.name.clone(),
                    args: largs,
                })));
            }

            // 3. Non-blocking OS & DMA primitive operations
            let prim_target_opt: Option<(MirPrimTarget, MirPrimKind)> =
                if let ExprKind::Ident(id) = &receiver.kind {
                    if let Some(pinfo) = be.syms().primitives.get(&id.name) {
                        Some((
                            MirPrimTarget::Global(id.name.clone()),
                            pinfo.kind.clone(),
                        ))
                    } else if let Some(vid) = be.scopes().lookup(&id.name) {
                        let vty = be.vars()[vid.0].ty.clone();
                        let pk_opt = match &vty {
                            MirType::Ptr(inner) => match inner.as_ref() {
                                MirType::Primitive(pk) => Some(pk.clone()),
                                _ => None,
                            },
                            _ => None,
                        };
                        pk_opt.map(|pk| {
                            (
                                MirPrimTarget::PtrExpr(MirExpr::new(MirExprKind::Var(vid), vty)),
                                pk,
                            )
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

            if let Some((target, pkind)) = prim_target_opt {
                let m = method.name.as_str();
                match (m, pkind) {
                    ("publish" | "write" | "set", MirPrimKind::Watch { elem }) => {
                        let v = if let Some(first) = first_positional_arg(args) {
                            be.lower_sub(first, Some(&elem))?
                        } else {
                            MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                        };
                        return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                            target,
                            elem_ty: *elem.clone(),
                            value: coerce_expr_if_needed(v, &elem),
                        })));
                    }
                    ("post" | "try_post", MirPrimKind::Inbox { elem, cap }) => {
                        let v = if let Some(first) = first_positional_arg(args) {
                            be.lower_sub(first, Some(&elem))?
                        } else {
                            MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                        };
                        let boost = extract_boost_arg(args);
                        return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::InboxPost {
                            target,
                            elem_ty: *elem.clone(),
                            cap,
                            value: coerce_expr_if_needed(v, &elem),
                            boost,
                        })));
                    }
                    ("try_send" | "post", MirPrimKind::Chan { elem, cap }) => {
                        let v = if let Some(first) = first_positional_arg(args) {
                            be.lower_sub(first, Some(&elem))?
                        } else {
                            MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                        };
                        return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::ChanTrySend {
                            target,
                            elem_ty: *elem.clone(),
                            cap,
                            value: coerce_expr_if_needed(v, &elem),
                        })));
                    }
                    ("notify" | "set" | "post" | "trigger", MirPrimKind::Signal) => {
                        let boost = extract_boost_arg(args);
                        return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::SignalNotify {
                            target,
                            boost,
                        })));
                    }
                    ("clear", MirPrimKind::Signal) => {
                        return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::SignalClear {
                            target,
                        })));
                    }
                    ("flip", MirPrimKind::PingPong { elem, cap }) => {
                        return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::PingPongFlip {
                            target,
                            elem_ty: *elem,
                            cap,
                        })));
                    }
                    ("start" | "activate", MirPrimKind::PingPong { elem, cap }) => {
                        return Ok(Some(MirStmt::PrimOp(
                            MirPrimNonBlockingOp::PingPongSetActive {
                                target,
                                elem_ty: *elem,
                                cap,
                                active: true,
                            },
                        )));
                    }
                    ("stop" | "deactivate" | "release", MirPrimKind::PingPong { elem, cap }) => {
                        return Ok(Some(MirStmt::PrimOp(
                            MirPrimNonBlockingOp::PingPongSetActive {
                                target,
                                elem_ty: *elem,
                                cap,
                                active: false,
                            },
                        )));
                    }
                    ("complete" | "flip", MirPrimKind::Lease { elem, cap }) => {
                        return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::LeaseComplete {
                            target,
                            elem_ty: *elem,
                            cap,
                        })));
                    }
                    ("acquire" | "start" | "activate", MirPrimKind::Lease { elem, cap }) => {
                        return Ok(Some(MirStmt::PrimOp(
                            MirPrimNonBlockingOp::LeaseSetActive {
                                target,
                                elem_ty: *elem,
                                cap,
                                active: true,
                            },
                        )));
                    }
                    ("release" | "stop" | "deactivate", MirPrimKind::Lease { elem, cap }) => {
                        return Ok(Some(MirStmt::PrimOp(
                            MirPrimNonBlockingOp::LeaseSetActive {
                                target,
                                elem_ty: *elem,
                                cap,
                                active: false,
                            },
                        )));
                    }
                    _ => {}
                }
            }

            Ok(None)
        }
        _ => Ok(None),
    }
}
