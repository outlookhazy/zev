//! Lowering pass from `zev_ast::Program` into Coroutine State-Machine & Kernel MIR (`MirModule`).

pub(super) use std::collections::{BTreeMap, BTreeSet, VecDeque};
pub(super) use std::fmt;

pub(super) use zev_ast::{
    ArrayLen, AssignOp, Attr, AttrKind, BinaryOp, Block, BuiltinFn, CallArg, CatchClause,
    DurationLit, ErrorSpec, Expr, ExprKind, FnDecl, FormatPart, FormatSpec, IsrDecl, ItemKind,
    Literal, MmioAccess, MmioDeclKind, MmioRegDecl, OsPrimitiveType, PrimType, Priority, Program,
    Span, SpawnStmt, Stmt, StmtKind, SwitchPat, SysMethod, TaskDecl, TypeExpr, TypeKind, UnaryOp,
    SYS_NAMESPACE,
};

pub(super) use crate::ir::{
    BlockId, MirAssignOp, MirBakedArg, MirBinOp, MirBlock, MirBootSequence, MirBootStep,
    MirCallDst, MirChildCallSlot, MirConst, MirEnum, MirErrorSpec, MirExpr, MirExprKind,
    MirExternFn, MirField, MirFloatFmt, MirFmtChunk, MirFnBody, MirFunction, MirImport, MirIntFmt,
    MirIntrinsicOp, MirIrqBinding, MirIrqVector, MirIsr, MirIsrStateDecl, MirIsrStateVar,
    MirMmioAccess, MirMmioInstance, MirMmioLayout, MirMmioReg, MirModule, MirPlace, MirPrimKind,
    MirPrimNonBlockingOp, MirPrimReadOp, MirPrimTarget, MirPrimViewKind, MirPrimitive, MirPriority,
    MirSpawnInstance, MirStmt, MirStruct, MirSubsystemGlobal, MirSwitchPattern, MirSysQuery,
    MirTaskBlueprint, MirTerminator, MirType, MirUnaryOp, MirVar, MirWaitArm, MirWaitArmOp,
    MirWaitTimer, MirYieldKind, MonomorphizedItemKind, MonomorphizedMirItem, VarId, VarRole,
};
pub(super) use crate::liveness::{analyze_fn_frame, analyze_isr_frame, analyze_task_frame};
pub(super) use crate::ram::{populate_module_ram_report, RamReport};

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

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum MirFnEffect {
    #[default]
    NonYielding,
    SoftYielding,
    HardBlocking,
}

impl MirFnEffect {
    pub(super) const fn can_yield(self) -> bool {
        !matches!(self, Self::NonYielding)
    }
}

#[derive(Clone)]
pub(super) struct FnSig {
    pub(super) params: Vec<MirType>,
    pub(super) ret_ty: MirType,
    pub(super) error_spec: MirErrorSpec,
    pub(super) is_extern_c: bool,
    pub(super) is_yielding: bool,
    pub(super) effect: MirFnEffect,
    pub(super) spawns_in_boot: bool,
}

#[derive(Clone)]
pub(super) struct PrimInfo {
    pub(super) kind: MirPrimKind,
    pub(super) dma_safe: bool,
    pub(super) sleep_lock_bit: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConstVal {
    Int(i128),
    Bool(bool),
}

/// Global module symbol table used during lowering.
pub(super) struct ModuleSymbols {
    pub(super) tick_period_us: u64,
    pub(super) systick_hz: Option<u64>,
    pub(super) import_aliases: BTreeMap<String, Vec<String>>,
    pub(super) enums: BTreeMap<String, Vec<(String, i64)>>,
    pub(super) variant_to_enum: BTreeMap<String, (String, i64)>,
    pub(super) structs: BTreeMap<String, MirStruct>,
    pub(super) mmio_layouts: BTreeMap<String, MirMmioLayout>,
    pub(super) mmio_instances: BTreeMap<String, MirMmioInstance>,
    pub(super) irqs: BTreeMap<String, u32>,
    pub(super) isrs: BTreeMap<String, Vec<MirType>>,
    pub(super) consts: BTreeMap<String, MirType>,
    pub(super) const_vals: BTreeMap<String, ConstVal>,
    pub(super) primitives: BTreeMap<String, PrimInfo>,
    pub(super) subsystems: BTreeMap<String, (String, bool)>,
    pub(super) fns: BTreeMap<String, FnSig>,
    pub(super) tasks: BTreeMap<String, Vec<MirType>>,
    pub(super) soft_yielding_in_coro: BTreeSet<String>,
    pub(super) soft_yielding_in_noyield: BTreeSet<String>,
    pub(super) dual_specialized_fns: BTreeSet<String>,
    pub(super) boost_capable_fns: BTreeSet<String>,
}

impl ModuleSymbols {
    pub(super) fn candidate_module_names(&self, alias_or_mod: &str, item: &str) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(path) = self.import_aliases.get(alias_or_mod) {
            if let Some(last) = path.last() {
                out.push(format!("{last}__{item}"));
            }
            out.push(format!("{}__{item}", path.join("_")));
            out.push(format!("{}__{item}", path.join("__")));
        }
        out.push(format!("{alias_or_mod}__{item}"));
        out.push(item.to_string());
        out
    }

    pub(super) fn resolve_mmio_layout_name(&self, te: &TypeExpr) -> Option<String> {
        match &te.kind {
            TypeKind::Named(id) => {
                if self.mmio_layouts.contains_key(&id.name) {
                    Some(id.name.clone())
                } else {
                    self.mmio_instances
                        .get(&id.name)
                        .map(|inst| inst.layout.clone())
                }
            }
            TypeKind::Path(segs) if !segs.is_empty() => {
                let joined_dunder = segs
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join("__");
                if self.mmio_layouts.contains_key(&joined_dunder) {
                    return Some(joined_dunder);
                }
                if segs.len() == 2 {
                    for cand in self.candidate_module_names(&segs[0].name, &segs[1].name) {
                        if self.mmio_layouts.contains_key(&cand) {
                            return Some(cand);
                        }
                        if let Some(inst) = self.mmio_instances.get(&cand) {
                            return Some(inst.layout.clone());
                        }
                    }
                }
                let last = &segs.last()?.name;
                if self.mmio_layouts.contains_key(last) {
                    Some(last.clone())
                } else {
                    self.mmio_instances.get(last).map(|i| i.layout.clone())
                }
            }
            _ => None,
        }
    }

    pub(in crate::lower) fn resolve_mmio_instance_field_be(
        &self,
        expr: &Expr,
        scopes: &VarScopes,
    ) -> Option<String> {
        if let ExprKind::Field { target, field } = &expr.kind {
            if let ExprKind::Ident(alias) = &target.kind {
                if scopes.lookup(&alias.name).is_none()
                    && self.import_aliases.contains_key(&alias.name)
                {
                    for cand in self.candidate_module_names(&alias.name, &field.name) {
                        if self.mmio_instances.contains_key(&cand) {
                            return Some(cand);
                        }
                    }
                }
            }
        }
        None
    }

    pub(in crate::lower) fn resolve_irq_field_be(
        &self,
        expr: &Expr,
        scopes: &VarScopes,
    ) -> Option<(String, u32)> {
        if let ExprKind::Field { target, field } = &expr.kind {
            if let ExprKind::Ident(alias) = &target.kind {
                if scopes.lookup(&alias.name).is_none()
                    && self.import_aliases.contains_key(&alias.name)
                {
                    for cand in self.candidate_module_names(&alias.name, &field.name) {
                        if let Some(&num) = self.irqs.get(&cand) {
                            return Some((cand, num));
                        }
                    }
                }
            }
        }
        None
    }

    pub(super) fn resolve_type(&self, te: &TypeExpr) -> MirType {
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
                if n == "irq" {
                    return MirType::U32;
                }
                if self.enums.contains_key(n) {
                    MirType::Enum(id.name.clone())
                } else if self.structs.contains_key(n) {
                    MirType::Struct(id.name.clone())
                } else {
                    MirType::Usize
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
                } else if joined.len() == 2 && self.import_aliases.contains_key(&joined[0]) {
                    for cand in self.candidate_module_names(&joined[0], &joined[1]) {
                        if self.enums.contains_key(&cand) {
                            return MirType::Enum(cand);
                        }
                        if self.structs.contains_key(&cand) {
                            return MirType::Struct(cand);
                        }
                    }
                }
                MirType::Usize
            }
            TypeKind::Slice(inner) => MirType::Slice(Box::new(self.resolve_type(inner))),
            TypeKind::Array { elem, len } => {
                let resolved_len = match len {
                    ArrayLen::Fixed(n) => *n,
                    ArrayLen::Expr(e) => match eval_const_expr(e, self) {
                        Some(ConstVal::Int(v)) if v >= 0 => v as usize,
                        _ => 0,
                    },
                };
                MirType::Array {
                    elem: Box::new(self.resolve_type(elem)),
                    len: resolved_len,
                }
            }
            TypeKind::Generic { base, .. } => MirType::Struct(base.name.clone()),
            TypeKind::GenericPath { path, .. } => MirType::Struct(
                path.iter()
                    .map(|s| s.name.clone())
                    .collect::<Vec<_>>()
                    .join("__"),
            ),
            TypeKind::Ptr(inner) => {
                if let Some(layout) = self.resolve_mmio_layout_name(inner) {
                    MirType::MmioPtr(layout)
                } else if let TypeKind::OsPrimitive(os) = &inner.kind {
                    match os {
                        OsPrimitiveType::ByteRingUnsized => {
                            MirType::PrimView(MirPrimViewKind::ByteRing)
                        }
                        OsPrimitiveType::InboxUnsized { elem } => {
                            MirType::PrimView(MirPrimViewKind::Inbox {
                                elem: Box::new(self.resolve_type(elem)),
                            })
                        }
                        _ => MirType::Ptr(Box::new(self.resolve_type(inner))),
                    }
                } else {
                    MirType::Ptr(Box::new(self.resolve_type(inner)))
                }
            }
            TypeKind::OsPrimitive(op) => MirType::Primitive(self.resolve_os_prim_type(op)),
            TypeKind::FnPtr {
                params,
                ret,
                error_spec,
            } => MirType::FnPtr {
                params: params.iter().map(|p| self.resolve_type(p)).collect(),
                ret: Box::new(self.resolve_type(ret)),
                error_spec: Box::new(match error_spec.as_deref() {
                    None => MirErrorSpec::None,
                    Some(ErrorSpec::Bare(_)) => MirErrorSpec::Untyped,
                    Some(ErrorSpec::Typed(et)) => MirErrorSpec::Typed(self.resolve_type(et)),
                }),
            },
            TypeKind::Lease(elem) => MirType::Lease(Box::new(self.resolve_type(elem))),
        }
    }

    pub(super) fn resolve_os_prim_type(&self, op: &OsPrimitiveType) -> MirPrimKind {
        match op {
            OsPrimitiveType::Signal => MirPrimKind::Signal,
            OsPrimitiveType::WaitQueue => MirPrimKind::WaitQueue,
            OsPrimitiveType::Watch { elem } => MirPrimKind::Watch {
                elem: Box::new(self.resolve_type(elem)),
            },
            OsPrimitiveType::Guarded { elem } => MirPrimKind::Guarded {
                elem: Box::new(self.resolve_type(elem)),
            },
            OsPrimitiveType::Inbox { elem, cap } => MirPrimKind::Inbox {
                elem: Box::new(self.resolve_type(elem)),
                cap: *cap,
            },
            OsPrimitiveType::InboxUnsized { elem } => MirPrimKind::Inbox {
                elem: Box::new(self.resolve_type(elem)),
                cap: 0,
            },
            OsPrimitiveType::PingPong { elem, cap } => MirPrimKind::PingPong {
                elem: Box::new(self.resolve_type(elem)),
                cap: *cap,
            },
            OsPrimitiveType::Pool { elem, cap } => MirPrimKind::Pool {
                elem: Box::new(self.resolve_type(elem)),
                cap: *cap,
            },
            OsPrimitiveType::DmaSlot { elem } => MirPrimKind::DmaSlot {
                elem: Box::new(self.resolve_type(elem)),
            },
            OsPrimitiveType::ByteRing { cap } => MirPrimKind::ByteRing { cap: *cap },
            OsPrimitiveType::ByteRingUnsized => MirPrimKind::ByteRing { cap: 0 },
        }
    }

    pub(super) fn resolve_error_spec(&self, spec: &Option<ErrorSpec>) -> MirErrorSpec {
        match spec {
            None => MirErrorSpec::None,
            Some(ErrorSpec::Bare(_)) => MirErrorSpec::Untyped,
            Some(ErrorSpec::Typed(te)) => MirErrorSpec::Typed(self.resolve_type(te)),
        }
    }

    pub(super) fn duration_to_ticks(&self, dur: &DurationLit) -> Result<u64, MirLowerError> {
        let us = dur.as_micros();
        if us == 0 {
            return Ok(0);
        }
        if us < self.tick_period_us || !us.is_multiple_of(self.tick_period_us) {
            return Err(MirLowerError {
                message: format!(
                    "duration `{dur}` ({us}us) is not an exact multiple of tick resolution ({}us)",
                    self.tick_period_us
                ),
                span: dur.span,
            });
        }
        let ticks = us / self.tick_period_us;
        if ticks > zev_ast::MAX_DURATION_TICKS {
            return Err(MirLowerError {
                message: format!(
                    "duration `{dur}` ({ticks} ticks) exceeds the scheduler deadline horizon of {} ticks",
                    zev_ast::MAX_DURATION_TICKS
                ),
                span: dur.span,
            });
        }
        Ok(ticks)
    }
}

pub(super) fn convert_prim_type(pt: PrimType) -> MirType {
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
        PrimType::F32 => MirType::F32,
        PrimType::F64 => MirType::F64,
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

pub(super) fn convert_priority(p: Priority) -> MirPriority {
    match p {
        Priority::Critical => MirPriority::Critical,
        Priority::High => MirPriority::High,
        Priority::Normal => MirPriority::Normal,
        Priority::Low => MirPriority::Low,
    }
}

pub(super) fn has_attr_dma_safe(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| matches!(a.kind, AttrKind::DmaSafe))
}

pub(super) fn has_attr_manual_yield(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| match &a.kind {
        AttrKind::ManualYield => true,
        AttrKind::Custom { name, .. } => name == "noyield",
        _ => false,
    })
}

pub(super) fn extract_monomorphized_sig(attrs: &[Attr]) -> Option<String> {
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

pub(super) fn normalize_f32_bits(bits: u64) -> u64 {
    ((f64::from_bits(bits) as f32) as f64).to_bits()
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
        mmio_layouts: BTreeMap::new(),
        mmio_instances: BTreeMap::new(),
        irqs: BTreeMap::new(),
        isrs: BTreeMap::new(),
        consts: BTreeMap::new(),
        const_vals: BTreeMap::new(),
        primitives: BTreeMap::new(),
        subsystems: BTreeMap::new(),
        fns: BTreeMap::new(),
        tasks: BTreeMap::new(),
        soft_yielding_in_coro: BTreeSet::new(),
        soft_yielding_in_noyield: BTreeSet::new(),
        dual_specialized_fns: BTreeSet::new(),
        boost_capable_fns: BTreeSet::new(),
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

    // Pre-register built-in `FmtErr { Truncated = 0 }` if not declared by user code.
    if !syms.enums.contains_key("FmtErr") {
        let variants = vec![("Truncated".to_string(), 0i64)];
        syms.variant_to_enum
            .entry("Truncated".to_string())
            .or_insert_with(|| ("FmtErr".to_string(), 0));
        syms.enums.insert("FmtErr".to_string(), variants.clone());
        mir_enums.push(MirEnum {
            name: "FmtErr".to_string(),
            variants,
        });
    }

    // Pre-seed constant values (fixpoint loop so forward-referenced consts evaluate too)
    // before `Mmio`, `Irq`, and `[CONST_N]T` inside Struct fields.
    for _ in 0..program.items.len().max(1) {
        let mut progress = false;
        for item in &program.items {
            if let ItemKind::Const(cd) = &item.kind {
                if !syms.const_vals.contains_key(&cd.name.name) {
                    if let Some(cv) = eval_const_expr(&cd.value, &syms) {
                        syms.const_vals.insert(cd.name.name.clone(), cv);
                        progress = true;
                    }
                }
            }
        }
        if !progress {
            break;
        }
    }

    // Pass 1a.5: Lower `ItemKind::Irq` and `ItemKind::Mmio` (layouts & inline first, then instances).
    for item in &program.items {
        if let ItemKind::Irq(iq) = &item.kind {
            let num = eval_const_u32(&iq.number, &syms, "IRQ number", iq.span)?;
            syms.irqs.insert(iq.name.name.clone(), num);
        }
    }

    let mut mir_mmio_layouts = Vec::new();
    for item in &program.items {
        if let ItemKind::Mmio(md) = &item.kind {
            match &md.kind {
                MmioDeclKind::Layout { regs } => {
                    let (mir_regs, byte_size) = lower_mmio_regs(regs, &syms)?;
                    let layout = MirMmioLayout {
                        name: md.name.name.clone(),
                        byte_size,
                        regs: mir_regs,
                    };
                    syms.mmio_layouts
                        .insert(md.name.name.clone(), layout.clone());
                    mir_mmio_layouts.push(layout);
                }
                MmioDeclKind::Inline { regs, .. } => {
                    let (mir_regs, byte_size) = lower_mmio_regs(regs, &syms)?;
                    let layout = MirMmioLayout {
                        name: md.name.name.clone(),
                        byte_size,
                        regs: mir_regs,
                    };
                    syms.mmio_layouts
                        .insert(md.name.name.clone(), layout.clone());
                    mir_mmio_layouts.push(layout);
                }
                MmioDeclKind::Instance { .. } => {}
            }
        }
    }

    let mut mir_mmio_instances = Vec::new();
    for item in &program.items {
        if let ItemKind::Mmio(md) = &item.kind {
            match &md.kind {
                MmioDeclKind::Instance { layout, base_addr } => {
                    let layout_name =
                        syms.resolve_mmio_layout_name(layout)
                            .ok_or_else(|| MirLowerError {
                                message: format!(
                                    "unknown MMIO layout for instance `{}`",
                                    md.name.name
                                ),
                                span: md.span,
                            })?;
                    let byte_size = syms.mmio_layouts[&layout_name].byte_size;
                    let addr = eval_const_u64(base_addr, &syms, "MMIO base address", md.span)?;
                    let inst = MirMmioInstance {
                        name: md.name.name.clone(),
                        layout: layout_name,
                        base_addr: addr,
                        byte_size,
                    };
                    syms.mmio_instances
                        .insert(md.name.name.clone(), inst.clone());
                    mir_mmio_instances.push(inst);
                }
                MmioDeclKind::Inline { base_addr, .. } => {
                    let byte_size = syms.mmio_layouts[&md.name.name].byte_size;
                    let addr = eval_const_u64(base_addr, &syms, "MMIO base address", md.span)?;
                    let inst = MirMmioInstance {
                        name: md.name.name.clone(),
                        layout: md.name.name.clone(),
                        base_addr: addr,
                        byte_size,
                    };
                    syms.mmio_instances
                        .insert(md.name.name.clone(), inst.clone());
                    mir_mmio_instances.push(inst);
                }
                MmioDeclKind::Layout { .. } => {}
            }
        }
    }

    let mut mono_map: BTreeMap<(MonomorphizedItemKind, String, String), MonomorphizedMirItem> =
        BTreeMap::new();

    // Pass 1b: Struct names first (so pointers/fields can reference any struct), then fields.
    for item in &program.items {
        if let ItemKind::Struct(sd) = &item.kind {
            if !sd.generic_params.is_empty() {
                // Defensively skip unmonomorphized generic struct templates.
                continue;
            }
            if let Some(sig) = extract_monomorphized_sig(&item.attrs) {
                let key = (
                    MonomorphizedItemKind::Struct,
                    sig.clone(),
                    sd.name.name.clone(),
                );
                mono_map.insert(
                    key,
                    MonomorphizedMirItem {
                        kind: MonomorphizedItemKind::Struct,
                        signature: sig,
                        mangled_name: sd.name.name.clone(),
                        size_32: 0,
                        size_64: 0,
                    },
                );
            }
            syms.structs.insert(
                sd.name.name.clone(),
                MirStruct {
                    name: sd.name.name.clone(),
                    packed: sd.is_packed,
                    is_subsystem: false,
                    fields: Vec::new(),
                },
            );
        }
    }

    let mut mir_structs = Vec::new();
    for item in &program.items {
        if let ItemKind::Struct(sd) = &item.kind {
            if !sd.generic_params.is_empty() {
                continue;
            }
            let fields: Vec<MirField> = sd
                .fields
                .iter()
                .map(|f| MirField {
                    name: f.name.name.clone(),
                    ty: syms.resolve_type(&f.ty),
                    dma_safe: has_attr_dma_safe(&f.attrs),
                })
                .collect();
            let ms = MirStruct {
                name: sd.name.name.clone(),
                packed: sd.is_packed,
                is_subsystem: false,
                fields,
            };
            syms.structs.insert(sd.name.name.clone(), ms.clone());
            mir_structs.push(ms);
        }
    }
    loop {
        let mut changed = false;
        for st in &mut mir_structs {
            if st.is_subsystem {
                continue;
            }
            let has_subsys = st.fields.iter().any(|f| match &f.ty {
                MirType::Primitive(_) => true,
                MirType::Struct(sub) => syms.structs.get(sub).is_some_and(|s| s.is_subsystem),
                _ => false,
            });
            if has_subsys {
                st.is_subsystem = true;
                if let Some(entry) = syms.structs.get_mut(&st.name) {
                    entry.is_subsystem = true;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Pass 1c: Constants, OS Primitives, Function & Task Signatures.
    let mut next_sleep_lock_bit = 0u8;
    let mut raw_fn_decls: Vec<&FnDecl> = Vec::new();
    let mut main_decl: Option<&FnDecl> = None;
    let mut raw_task_decls: Vec<&TaskDecl> = Vec::new();
    let mut raw_isr_decls: Vec<&IsrDecl> = Vec::new();
    let mut mir_extern_fns: Vec<MirExternFn> = Vec::new();
    let mut mir_subsystems: Vec<MirSubsystemGlobal> = Vec::new();

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
                let sleep_lock_bit = if matches!(kind, MirPrimKind::PingPong { .. }) {
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
                if !fd.generic_params.is_empty() {
                    // Defensively skip unmonomorphized generic fn templates.
                    continue;
                }
                if let Some(sig) = extract_monomorphized_sig(&item.attrs)
                    .or_else(|| extract_monomorphized_sig(&fd.attrs))
                {
                    let key = (MonomorphizedItemKind::Fn, sig.clone(), fd.name.name.clone());
                    mono_map.insert(
                        key,
                        MonomorphizedMirItem {
                            kind: MonomorphizedItemKind::Fn,
                            signature: sig,
                            mangled_name: fd.name.name.clone(),
                            size_32: 0,
                            size_64: 0,
                        },
                    );
                }
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
                        effect: MirFnEffect::NonYielding,
                        spawns_in_boot: false,
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
                let params: Vec<MirType> =
                    id.params.iter().map(|p| syms.resolve_type(&p.ty)).collect();
                syms.isrs.insert(id.vector.name.clone(), params);
                raw_isr_decls.push(id);
            }
            ItemKind::GlobalVar(gv) => {
                let gty = syms.resolve_type(&gv.ty);
                if let MirType::Struct(sname) = &gty {
                    if syms.structs.get(sname).is_some_and(|s| s.is_subsystem) {
                        let dma_safe = has_attr_dma_safe(&item.attrs);
                        syms.subsystems
                            .insert(gv.name.name.clone(), (sname.clone(), dma_safe));
                        mir_subsystems.push(MirSubsystemGlobal {
                            name: gv.name.name.clone(),
                            struct_name: sname.clone(),
                            dma_safe,
                        });
                        continue;
                    }
                }
                return Err(MirLowerError {
                    message: format!(
                        "mutable raw global `{}` is forbidden; use an OS primitive (`watch`, `inbox`, `signal`, `ping_pong`)",
                        gv.name.name
                    ),
                    span: gv.span,
                });
            }
            ItemKind::Import(_)
            | ItemKind::Impl(_)
            | ItemKind::Enum(_)
            | ItemKind::Struct(_)
            | ItemKind::Mmio(_)
            | ItemKind::Irq(_) => {}
        }
    }

    let fn_decl_map: BTreeMap<String, &FnDecl> = raw_fn_decls
        .iter()
        .map(|fd| (fd.name.name.clone(), *fd))
        .collect();

    // Discover `sys.use_systick(hz)` in `main` (and boot helpers) if present.
    if let Some(md) = main_decl {
        if let Some(body) = &md.body {
            let mut visited_boot = BTreeSet::new();
            discover_systick_config(&body.stmts, &fn_decl_map, &mut visited_boot, &mut syms);
        }
    }

    // Pass 2: Whole-program 3-tier `FnEffect`, `spawns_in_boot`, and `SoftYielding` specialization inference.
    infer_effects_and_specializations(
        &raw_fn_decls,
        &raw_task_decls,
        main_decl,
        &program.items,
        &mut syms,
    );

    // Pass 3: Lower constants & top-level primitive initializers.
    let mut mir_consts = Vec::new();
    let mut mir_primitives = Vec::new();
    for item in &program.items {
        match &item.kind {
            ItemKind::Const(cd) => {
                let ty = syms.resolve_type(&cd.ty);
                let mut dummy_ctx = DirectLowerCtx::new(&syms, MirType::Void, MirErrorSpec::None);
                let init =
                    coerce_expr_if_needed(dummy_ctx.lower_pure_expr(&cd.value, Some(&ty))?, &ty);
                mir_consts.push(MirConst {
                    name: cd.name.name.clone(),
                    ty,
                    init,
                });
            }
            ItemKind::OsPrimitive(pd) => {
                let pinfo =
                    syms.primitives
                        .get(&pd.name.name)
                        .cloned()
                        .ok_or_else(|| MirLowerError {
                            message: format!(
                                "internal error: missing primitive '{}'",
                                pd.name.name
                            ),
                            span: pd.span,
                        })?;
                let init = if let Some(ie) = &pd.init {
                    let elem_ty = match &pinfo.kind {
                        MirPrimKind::Watch { elem } | MirPrimKind::Guarded { elem } => {
                            Some(elem.as_ref())
                        }
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
                    dead_stripped: false,
                    irq_mask_elided: false,
                });
            }
            _ => {}
        }
    }

    // Pass 4: Lower ISRs (`noyield` direct statements, optional params & `state` vars).
    let mut mir_isrs = Vec::new();
    for id in raw_isr_decls {
        let mut dctx = DirectLowerCtx::new(&syms, MirType::Void, MirErrorSpec::None);
        dctx.in_noyield = true;
        dctx.in_isr = true;
        let mut param_ids = Vec::new();
        for p in &id.params {
            let pty = syms.resolve_type(&p.ty);
            let vid = dctx.declare_var(&p.name.name, pty, VarRole::Param);
            param_ids.push(vid);
        }
        let body = dctx.lower_top_level_stmts(&id.body.stmts)?;
        let mut mir_isr = MirIsr {
            name: id.vector.name.clone(),
            params: param_ids,
            state_vars: dctx.isr_state_vars,
            vars: dctx.vars,
            body,
        };
        analyze_isr_frame(&mut mir_isr);
        mir_isrs.push(mir_isr);
    }

    // Pass 5: Lower Functions (Direct non-yielding, Coroutine yielding, and `__noyield` SoftYielding clones).
    let mut mir_functions = Vec::new();
    for fd in &raw_fn_decls {
        let sig = syms
            .fns
            .get(&fd.name.name)
            .cloned()
            .ok_or_else(|| MirLowerError {
                message: format!("internal error: missing fn signature '{}'", fd.name.name),
                span: fd.span,
            })?;
        if sig.spawns_in_boot {
            // Boot-tree setup helpers containing `spawn` are unrolled inline in `main()`'s sequential boot phase.
            continue;
        }
        let manual_yield = has_attr_manual_yield(&fd.attrs);
        let body_ast = fd.body.as_ref().ok_or_else(|| MirLowerError {
            message: format!("function '{}' has no body for MIR lowering", fd.name.name),
            span: fd.span,
        })?;

        match sig.effect {
            MirFnEffect::NonYielding => {
                let mut dctx =
                    DirectLowerCtx::new(&syms, sig.ret_ty.clone(), sig.error_spec.clone());
                let mut param_ids = Vec::new();
                for p in &fd.params {
                    let pty = syms.resolve_type(&p.ty);
                    let vid = dctx.declare_var(&p.name.name, pty, VarRole::Param);
                    param_ids.push(vid);
                }
                let stmts = dctx.lower_top_level_stmts(&body_ast.stmts)?;
                let mut func = MirFunction {
                    name: fd.name.name.clone(),
                    params: param_ids,
                    ret_ty: sig.ret_ty,
                    error_spec: sig.error_spec,
                    manual_yield,
                    is_yielding: false,
                    vars: dctx.vars,
                    body: MirFnBody::Direct(stmts),
                };
                analyze_fn_frame(&mut func);
                mir_functions.push(func);
            }
            MirFnEffect::HardBlocking => {
                let mut func = lower_coroutine_fn_decl(fd, &sig, manual_yield, &syms)?;
                analyze_fn_frame(&mut func);
                mir_functions.push(func);
            }
            MirFnEffect::SoftYielding => {
                let in_coro = syms.soft_yielding_in_coro.contains(&fd.name.name);
                let in_noyield = syms.soft_yielding_in_noyield.contains(&fd.name.name);
                let emit_coro = in_coro || !in_noyield;
                let emit_noyield = in_noyield;

                if emit_coro {
                    let mut func = lower_coroutine_fn_decl(fd, &sig, manual_yield, &syms)?;
                    analyze_fn_frame(&mut func);
                    mir_functions.push(func);
                }
                if emit_noyield {
                    let mut dctx =
                        DirectLowerCtx::new(&syms, sig.ret_ty.clone(), sig.error_spec.clone());
                    dctx.in_noyield = true;
                    let mut param_ids = Vec::new();
                    for p in &fd.params {
                        let pty = syms.resolve_type(&p.ty);
                        let vid = dctx.declare_var(&p.name.name, pty, VarRole::Param);
                        param_ids.push(vid);
                    }
                    let stmts = dctx.lower_top_level_stmts(&body_ast.stmts)?;
                    let mut ny_func = MirFunction {
                        name: format!("{}__noyield", fd.name.name),
                        params: param_ids,
                        ret_ty: sig.ret_ty.clone(),
                        error_spec: sig.error_spec.clone(),
                        manual_yield,
                        is_yielding: false,
                        vars: dctx.vars,
                        body: MirFnBody::Direct(stmts),
                    };
                    analyze_fn_frame(&mut ny_func);
                    mir_functions.push(ny_func);
                }
            }
        }
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
        cctx.lower_top_level_stmts(&td.body.stmts)?;
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

    // Pass 7: Lower `fn main()` into Sequential Boot + Static Spawn Slots + IRQ Vectors + Optional Supervisor Task.
    let mut spawned_tasks = Vec::new();
    let mut mir_irq_vectors = Vec::new();
    let boot = if let Some(md) = main_decl {
        lower_main_entry(
            md,
            &syms,
            &fn_decl_map,
            &mir_isrs,
            &mut mir_blueprints,
            &mut spawned_tasks,
            &mut mir_irq_vectors,
        )?
    } else {
        MirBootSequence {
            present: false,
            from_main_entry: false,
            vars: Vec::new(),
            steps: Vec::new(),
            boot_yielding_callees: Vec::new(),
        }
    };

    let dual_specialized_fns: Vec<String> = syms.dual_specialized_fns.iter().cloned().collect();
    let monomorphized_instances: Vec<MonomorphizedMirItem> = mono_map.into_values().collect();
    let formatted_enums =
        collect_formatted_enums(&mir_isrs, &mir_functions, &mir_blueprints, &boot);

    let mut module = MirModule {
        tick_period_us: syms.tick_period_us,
        systick_hz: syms.systick_hz,
        imports: mir_imports,
        consts: mir_consts,
        enums: mir_enums,
        structs: mir_structs,
        mmio_layouts: mir_mmio_layouts,
        mmio_instances: mir_mmio_instances,
        primitives: mir_primitives,
        subsystems: mir_subsystems,
        extern_fns: mir_extern_fns,
        isrs: mir_isrs,
        irq_vectors: mir_irq_vectors,
        functions: mir_functions,
        task_blueprints: mir_blueprints,
        spawned_tasks,
        dual_specialized_fns,
        monomorphized_instances,
        formatted_enums,
        boot,
        visibility: crate::opt::VisibilityReport::default(),
        ram_report: RamReport::default(),
    };

    crate::opt::prune_unreachable_functions(&mut module);
    module.formatted_enums = collect_formatted_enums(
        &module.isrs,
        &module.functions,
        &module.task_blueprints,
        &module.boot,
    );
    crate::opt::fold_singletons(&mut module);
    let vis = crate::opt::analyze_visibility(&module);
    for p in &mut module.primitives {
        p.dead_stripped = vis.can_strip_bss(&p.name);
        p.irq_mask_elided = !p.dead_stripped
            && !vis.isr_contended_globals.contains(&p.name)
            && !vis.is_prim_family_contended(&p.kind);
    }
    module.visibility = vis;

    populate_module_ram_report(&mut module);
    Ok(module)
}

pub(super) fn lower_coroutine_fn_decl(
    fd: &FnDecl,
    sig: &FnSig,
    manual_yield: bool,
    syms: &ModuleSymbols,
) -> Result<MirFunction, MirLowerError> {
    let body_ast = fd.body.as_ref().ok_or_else(|| MirLowerError {
        message: format!(
            "function '{}' has no body for coroutine lowering",
            fd.name.name
        ),
        span: fd.span,
    })?;
    let mut cctx = CoroutineLowerCtx::new(
        syms,
        sig.ret_ty.clone(),
        sig.error_spec.clone(),
        manual_yield,
    );
    cctx.is_yielding_fn = true;
    let mut param_ids = Vec::new();
    for p in &fd.params {
        let pty = syms.resolve_type(&p.ty);
        let vid = cctx.declare_var(&p.name.name, pty, VarRole::Param);
        param_ids.push(vid);
    }
    cctx.lower_top_level_stmts(&body_ast.stmts)?;
    cctx.finish_with_implicit_return();

    Ok(MirFunction {
        name: fd.name.name.clone(),
        params: param_ids,
        ret_ty: sig.ret_ty.clone(),
        error_spec: sig.error_spec.clone(),
        manual_yield,
        is_yielding: true,
        vars: cctx.vars,
        body: MirFnBody::Coroutine {
            blocks: cctx.blocks,
            child_slots: cctx.child_slots,
        },
    })
}

mod coroutine;
mod direct;
mod effects;
mod exprs;
mod scopes;

pub(in crate::lower) use coroutine::*;
pub(in crate::lower) use direct::*;
pub(in crate::lower) use effects::*;
pub(in crate::lower) use exprs::*;
pub(in crate::lower) use scopes::*;
