//! Lowering pass from `zev_ast::Program` into Coroutine State-Machine & Kernel MIR (`MirModule`).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use zev_ast::{
    ArrayLen, AssignOp, Attr, AttrKind, BinaryOp, Block, CallArg, CatchClause, DurationLit,
    ErrorSpec, Expr, ExprKind, FnDecl, FormatPart, FormatSpec, IsrDecl, ItemKind, Literal,
    MmioAccess, MmioDeclKind, MmioRegDecl, OsPrimitiveType, PrimType, Priority, Program, Span,
    SpawnStmt, Stmt, StmtKind, SwitchPat, TaskDecl, TypeExpr, TypeKind, UnaryOp,
};

use crate::ir::{
    BlockId, MirAssignOp, MirBakedArg, MirBinOp, MirBlock, MirBootSequence, MirBootStep,
    MirCallDst, MirChildCallSlot, MirConst, MirEnum, MirErrorSpec, MirExpr, MirExprKind,
    MirExternFn, MirField, MirFloatFmt, MirFmtChunk, MirFnBody, MirFunction, MirImport, MirIntFmt,
    MirIntrinsicOp, MirIrqBinding, MirIrqVector, MirIsr, MirIsrStateDecl, MirIsrStateVar,
    MirMmioAccess, MirMmioInstance, MirMmioLayout, MirMmioReg, MirModule, MirPlace, MirPrimKind,
    MirPrimNonBlockingOp, MirPrimReadOp, MirPrimTarget, MirPrimitive, MirPriority,
    MirSpawnInstance, MirStmt, MirStruct, MirSwitchPattern, MirSysQuery, MirTaskBlueprint,
    MirTerminator, MirType, MirUnaryOp, MirVar, MirYieldKind, MonomorphizedItemKind,
    MonomorphizedMirItem, VarId, VarRole,
};
use crate::liveness::{analyze_fn_frame, analyze_isr_frame, analyze_task_frame};
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

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum MirFnEffect {
    #[default]
    NonYielding,
    SoftYielding,
    HardBlocking,
}

impl MirFnEffect {
    const fn can_yield(self) -> bool {
        !matches!(self, Self::NonYielding)
    }
}

#[derive(Clone)]
struct FnSig {
    params: Vec<MirType>,
    ret_ty: MirType,
    error_spec: MirErrorSpec,
    is_extern_c: bool,
    is_yielding: bool,
    effect: MirFnEffect,
    spawns_in_boot: bool,
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
    mmio_layouts: BTreeMap<String, MirMmioLayout>,
    mmio_instances: BTreeMap<String, MirMmioInstance>,
    irqs: BTreeMap<String, u32>,
    isrs: BTreeMap<String, Vec<MirType>>,
    consts: BTreeMap<String, MirType>,
    const_vals: BTreeMap<String, ConstVal>,
    primitives: BTreeMap<String, PrimInfo>,
    fns: BTreeMap<String, FnSig>,
    tasks: BTreeMap<String, Vec<MirType>>,
    soft_yielding_in_coro: BTreeSet<String>,
    soft_yielding_in_noyield: BTreeSet<String>,
    dual_specialized_fns: BTreeSet<String>,
}

impl ModuleSymbols {
    fn candidate_module_names(&self, alias_or_mod: &str, item: &str) -> Vec<String> {
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

    fn resolve_mmio_layout_name(&self, te: &TypeExpr) -> Option<String> {
        match &te.kind {
            TypeKind::Named(id) => {
                if self.mmio_layouts.contains_key(&id.name) {
                    Some(id.name.clone())
                } else if let Some(inst) = self.mmio_instances.get(&id.name) {
                    Some(inst.layout.clone())
                } else {
                    None
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
                let last = &segs.last().unwrap().name;
                if self.mmio_layouts.contains_key(last) {
                    Some(last.clone())
                } else {
                    self.mmio_instances.get(last).map(|i| i.layout.clone())
                }
            }
            _ => None,
        }
    }

    fn resolve_mmio_instance_field_be(
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

    fn resolve_irq_field_be(&self, expr: &Expr, scopes: &VarScopes) -> Option<(String, u32)> {
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
            TypeKind::GenericPath { path, .. } => {
                MirType::Struct(path.iter().map(|s| s.name.clone()).collect::<Vec<_>>().join("__"))
            }
            TypeKind::Ptr(inner) => {
                if let Some(layout) = self.resolve_mmio_layout_name(inner) {
                    MirType::MmioPtr(layout)
                } else {
                    MirType::Ptr(Box::new(self.resolve_type(inner)))
                }
            }
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
    attrs.iter().any(|a| match &a.kind {
        AttrKind::ManualYield => true,
        AttrKind::Custom { name, .. } => name == "noyield",
        _ => false,
    })
}

fn extract_monomorphized_sig(attrs: &[Attr]) -> Option<String> {
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

fn normalize_f32_bits(bits: u64) -> u64 {
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
        fns: BTreeMap::new(),
        tasks: BTreeMap::new(),
        soft_yielding_in_coro: BTreeSet::new(),
        soft_yielding_in_noyield: BTreeSet::new(),
        dual_specialized_fns: BTreeSet::new(),
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
                    syms.mmio_layouts.insert(md.name.name.clone(), layout.clone());
                    mir_mmio_layouts.push(layout);
                }
                MmioDeclKind::Inline { regs, .. } => {
                    let (mir_regs, byte_size) = lower_mmio_regs(regs, &syms)?;
                    let layout = MirMmioLayout {
                        name: md.name.name.clone(),
                        byte_size,
                        regs: mir_regs,
                    };
                    syms.mmio_layouts.insert(md.name.name.clone(), layout.clone());
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
                    let layout_name = syms.resolve_mmio_layout_name(layout).ok_or_else(|| MirLowerError {
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
                    syms.mmio_instances.insert(md.name.name.clone(), inst.clone());
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
                    syms.mmio_instances.insert(md.name.name.clone(), inst.clone());
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
                if !fd.generic_params.is_empty() {
                    // Defensively skip unmonomorphized generic fn templates.
                    continue;
                }
                if let Some(sig) = extract_monomorphized_sig(&item.attrs)
                    .or_else(|| extract_monomorphized_sig(&fd.attrs))
                {
                    let key = (
                        MonomorphizedItemKind::Fn,
                        sig.clone(),
                        fd.name.name.clone(),
                    );
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
                return Err(MirLowerError {
                    message: format!(
                        "mutable raw global `{}` is forbidden; use an OS primitive (`watch`, `inbox`, `signal`, `ping_pong`)",
                        gv.name.name
                    ),
                    span: gv.span,
                });
            }
            ItemKind::Import(_)
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
    infer_effects_and_specializations(&raw_fn_decls, &raw_task_decls, main_decl, &mut syms);

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
        let sig = syms.fns.get(&fd.name.name).unwrap().clone();
        if sig.spawns_in_boot {
            // Boot-tree setup helpers containing `spawn` are unrolled inline in `main()`'s sequential boot phase.
            continue;
        }
        let manual_yield = has_attr_manual_yield(&fd.attrs);
        let body_ast = fd.body.as_ref().unwrap();

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
        ram_report: RamReport::default(),
    };

    populate_module_ram_report(&mut module);
    Ok(module)
}

fn lower_coroutine_fn_decl(
    fd: &FnDecl,
    sig: &FnSig,
    manual_yield: bool,
    syms: &ModuleSymbols,
) -> Result<MirFunction, MirLowerError> {
    let body_ast = fd.body.as_ref().unwrap();
    let mut cctx = CoroutineLowerCtx::new(
        syms,
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

// ─────────────────────────────────────────────────────────────────────────────
// Call-Graph Yield Inference & `main()` Splitting Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn discover_systick_config(
    stmts: &[Stmt],
    fn_decls: &BTreeMap<String, &FnDecl>,
    visited: &mut BTreeSet<String>,
    syms: &mut ModuleSymbols,
) {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Expr(Expr {
                kind:
                    ExprKind::MethodCall {
                        receiver,
                        method,
                        args,
                    },
                ..
            }) => {
                if let ExprKind::Ident(id) = &receiver.kind {
                    if id.name == "sys" && method.name == "use_systick" {
                        if let Some(first) = args.first() {
                            if let ExprKind::Literal(Literal::Int(hz)) = &first.value.kind {
                                syms.systick_hz = Some(*hz);
                            }
                        }
                        if let Some(second) = args.get(1) {
                            if let ExprKind::Literal(Literal::Duration(d)) = &second.value.kind {
                                let us = d.as_micros();
                                if us > 0 {
                                    syms.tick_period_us = us;
                                }
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
                    } else if id.name == "sys"
                        && matches!(method.name.as_str(), "set_tick" | "set_tick_period")
                    {
                        if let Some(first) = args.first() {
                            if let ExprKind::Literal(Literal::Duration(d)) = &first.value.kind {
                                let us = d.as_micros();
                                if us > 0 {
                                    syms.tick_period_us = us;
                                }
                            }
                        }
                    }
                }
            }
            StmtKind::Expr(Expr {
                kind: ExprKind::Call { callee, .. },
                ..
            }) => {
                if let ExprKind::Ident(id) = &callee.kind {
                    if visited.insert(id.name.clone()) {
                        if let Some(fd) = fn_decls.get(&id.name) {
                            if let Some(b) = &fd.body {
                                discover_systick_config(&b.stmts, fn_decls, visited, syms);
                            }
                        }
                    }
                }
            }
            StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                discover_systick_config(&then_block.stmts, fn_decls, visited, syms);
                if let Some(eb) = else_block {
                    discover_systick_config(&eb.stmts, fn_decls, visited, syms);
                }
            }
            StmtKind::Block(b) | StmtKind::NoYield { body: b } | StmtKind::Critical { body: b } => {
                discover_systick_config(&b.stmts, fn_decls, visited, syms);
            }
            _ => {}
        }
    }
}

#[derive(Clone)]
struct RawCallEdge {
    caller: String,
    callee: String,
    in_noyield: bool,
    in_restricted: bool, // critical, defer, or isr
}

fn infer_effects_and_specializations(
    fn_decls: &[&FnDecl],
    task_decls: &[&TaskDecl],
    main_decl: Option<&FnDecl>,
    syms: &mut ModuleSymbols,
) {
    let mut direct_eff: BTreeMap<String, MirFnEffect> = BTreeMap::new();
    let mut direct_spawns: BTreeSet<String> = BTreeSet::new();
    let mut edges: Vec<RawCallEdge> = Vec::new();
    let mut fn_names: BTreeSet<String> = BTreeSet::new();

    for fd in fn_decls {
        let fname = fd.name.name.clone();
        fn_names.insert(fname.clone());
        let manual = has_attr_manual_yield(&fd.attrs);
        let mut eff = MirFnEffect::NonYielding;
        let mut spawns = false;
        if let Some(body) = &fd.body {
            scan_stmts_effect(
                &fname,
                &body.stmts,
                manual,
                false,
                false,
                &mut eff,
                &mut spawns,
                &mut edges,
            );
        }
        direct_eff.insert(fname.clone(), eff);
        if spawns {
            direct_spawns.insert(fname);
        }
    }

    if let Some(md) = main_decl {
        let manual = has_attr_manual_yield(&md.attrs);
        let mut eff = MirFnEffect::NonYielding;
        let mut spawns = false;
        if let Some(body) = &md.body {
            scan_stmts_effect(
                "main",
                &body.stmts,
                manual,
                false,
                false,
                &mut eff,
                &mut spawns,
                &mut edges,
            );
        }
    }

    for td in task_decls {
        let tkey = format!("task::{}", td.name.name);
        let manual = has_attr_manual_yield(&td.attrs);
        let mut eff = MirFnEffect::NonYielding;
        let mut spawns = false;
        scan_stmts_effect(
            &tkey,
            &td.body.stmts,
            manual,
            false,
            false,
            &mut eff,
            &mut spawns,
            &mut edges,
        );
    }

    let mut cur_eff = direct_eff;
    let mut spawns_in_boot = direct_spawns;
    loop {
        let mut changed = false;
        for edge in &edges {
            if !fn_names.contains(&edge.caller) || !fn_names.contains(&edge.callee) {
                continue;
            }
            let callee_eff = cur_eff
                .get(&edge.callee)
                .copied()
                .unwrap_or(MirFnEffect::NonYielding);
            let edge_eff = match callee_eff {
                MirFnEffect::HardBlocking => MirFnEffect::HardBlocking,
                MirFnEffect::SoftYielding if !edge.in_noyield => MirFnEffect::SoftYielding,
                _ => MirFnEffect::NonYielding,
            };
            let prev = cur_eff
                .get(&edge.caller)
                .copied()
                .unwrap_or(MirFnEffect::NonYielding);
            let next = prev.max(edge_eff);
            if next != prev {
                cur_eff.insert(edge.caller.clone(), next);
                changed = true;
            }
            if spawns_in_boot.contains(&edge.callee) && spawns_in_boot.insert(edge.caller.clone()) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for fname in &fn_names {
        let eff = cur_eff
            .get(fname)
            .copied()
            .unwrap_or(MirFnEffect::NonYielding);
        let sp = spawns_in_boot.contains(fname);
        if let Some(sig) = syms.fns.get_mut(fname) {
            if !sig.is_extern_c {
                sig.effect = eff;
                sig.is_yielding = eff.can_yield();
                sig.spawns_in_boot = sp;
            }
        }
    }

    // Compute 2-state `(Coro | NoYield)` reachability for `SoftYielding` functions.
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    enum ExecMode {
        Coro,
        NoYield,
    }

    let mut has_incoming: BTreeSet<&str> = BTreeSet::new();
    let mut adj: BTreeMap<&str, Vec<&RawCallEdge>> = BTreeMap::new();
    for e in &edges {
        if fn_names.contains(&e.callee) {
            has_incoming.insert(e.callee.as_str());
            adj.entry(e.caller.as_str()).or_default().push(e);
        }
    }

    let mut queue: VecDeque<(String, ExecMode)> = VecDeque::new();
    let mut visited: BTreeSet<(String, ExecMode)> = BTreeSet::new();

    if main_decl.is_some() {
        visited.insert(("main".to_string(), ExecMode::Coro));
        queue.push_back(("main".to_string(), ExecMode::Coro));
    }
    for td in task_decls {
        let tkey = format!("task::{}", td.name.name);
        if visited.insert((tkey.clone(), ExecMode::Coro)) {
            queue.push_back((tkey, ExecMode::Coro));
        }
    }
    for fname in &fn_names {
        if !has_incoming.contains(fname.as_str())
            && visited.insert((fname.clone(), ExecMode::Coro))
        {
            queue.push_back((fname.clone(), ExecMode::Coro));
        }
    }

    let mut soft_in_coro = BTreeSet::new();
    let mut soft_in_noyield = BTreeSet::new();

    while let Some((curr, mode)) = queue.pop_front() {
        let Some(out_edges) = adj.get(curr.as_str()) else {
            continue;
        };
        for edge in out_edges {
            if edge.in_restricted {
                continue;
            }
            let Some(callee_sig) = syms.fns.get(&edge.callee) else {
                continue;
            };
            if callee_sig.is_extern_c {
                continue;
            }
            let next_mode = if mode == ExecMode::NoYield || edge.in_noyield {
                ExecMode::NoYield
            } else {
                ExecMode::Coro
            };
            if callee_sig.effect == MirFnEffect::SoftYielding {
                match next_mode {
                    ExecMode::Coro => {
                        soft_in_coro.insert(edge.callee.clone());
                    }
                    ExecMode::NoYield => {
                        soft_in_noyield.insert(edge.callee.clone());
                    }
                }
            }
            if visited.insert((edge.callee.clone(), next_mode)) {
                queue.push_back((edge.callee.clone(), next_mode));
            }
        }
    }

    // Register `{f}__noyield` signatures in `syms.fns` for all `soft_in_noyield` functions.
    for ny_fn in &soft_in_noyield {
        if let Some(orig_sig) = syms.fns.get(ny_fn).cloned() {
            syms.fns.insert(
                format!("{ny_fn}__noyield"),
                FnSig {
                    params: orig_sig.params,
                    ret_ty: orig_sig.ret_ty,
                    error_spec: orig_sig.error_spec,
                    is_extern_c: false,
                    is_yielding: false,
                    effect: MirFnEffect::NonYielding,
                    spawns_in_boot: false,
                },
            );
        }
    }

    syms.dual_specialized_fns = soft_in_coro
        .intersection(&soft_in_noyield)
        .cloned()
        .collect();
    syms.soft_yielding_in_coro = soft_in_coro;
    syms.soft_yielding_in_noyield = soft_in_noyield;
}

fn ast_block_has_break(block: &Block) -> bool {
    ast_stmts_have_break(&block.stmts)
}

fn ast_stmts_have_break(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Break => return true,
            StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                if ast_block_has_break(then_block)
                    || else_block.as_ref().is_some_and(ast_block_has_break)
                {
                    return true;
                }
            }
            StmtKind::Switch { arms, .. } => {
                if arms.iter().any(|a| ast_block_has_break(&a.body)) {
                    return true;
                }
            }
            StmtKind::NoYield { body } | StmtKind::Critical { body } | StmtKind::Block(body) => {
                if ast_block_has_break(body) {
                    return true;
                }
            }
            StmtKind::Expr(e) | StmtKind::VarDecl { init: Some(e), .. } => {
                if ast_expr_has_break(e) {
                    return true;
                }
            }
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

fn ast_expr_has_break(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Catch { expr: inner, clause } => {
            ast_expr_has_break(inner) || ast_block_has_break(&clause.body)
        }
        ExprKind::Or { expr: a, fallback: b } => ast_expr_has_break(a) || ast_expr_has_break(b),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_stmts_effect(
    caller: &str,
    stmts: &[Stmt],
    manual_yield: bool,
    in_noyield: bool,
    in_restricted: bool,
    eff: &mut MirFnEffect,
    spawns: &mut bool,
    edges: &mut Vec<RawCallEdge>,
) {
    for stmt in stmts {
        let stmt_manual = manual_yield || has_attr_manual_yield(&stmt.attrs);
        match &stmt.kind {
            StmtKind::StateVar { .. } => {}
            StmtKind::LoopEvery { period, body } => {
                if !in_noyield && !in_restricted {
                    *eff = (*eff).max(MirFnEffect::HardBlocking);
                }
                scan_expr_effect(caller, period, in_noyield, in_restricted, eff, spawns, edges);
                scan_stmts_effect(
                    caller,
                    &body.stmts,
                    stmt_manual,
                    in_noyield,
                    in_restricted,
                    eff,
                    spawns,
                    edges,
                );
            }
            StmtKind::Checkpoint | StmtKind::Restart => {
                if !in_noyield && !in_restricted {
                    *eff = (*eff).max(MirFnEffect::SoftYielding);
                }
            }
            StmtKind::Loop { body } => {
                let has_break = ast_block_has_break(body);
                if !in_noyield && !in_restricted {
                    if !has_break {
                        *eff = (*eff).max(MirFnEffect::HardBlocking);
                    } else if !stmt_manual {
                        *eff = (*eff).max(MirFnEffect::SoftYielding);
                    }
                }
                scan_stmts_effect(
                    caller,
                    &body.stmts,
                    stmt_manual,
                    in_noyield,
                    in_restricted,
                    eff,
                    spawns,
                    edges,
                );
            }
            StmtKind::While { cond, body } => {
                scan_expr_effect(caller, cond, in_noyield, in_restricted, eff, spawns, edges);
                let is_while_true = matches!(cond.kind, ExprKind::Literal(Literal::Bool(true)));
                if is_while_true && !in_noyield && !in_restricted {
                    let has_break = ast_block_has_break(body);
                    if !has_break {
                        *eff = (*eff).max(MirFnEffect::HardBlocking);
                    } else if !stmt_manual {
                        *eff = (*eff).max(MirFnEffect::SoftYielding);
                    }
                }
                scan_stmts_effect(
                    caller,
                    &body.stmts,
                    stmt_manual,
                    in_noyield,
                    in_restricted,
                    eff,
                    spawns,
                    edges,
                );
            }
            StmtKind::ForRange {
                start, end, body, ..
            } => {
                scan_expr_effect(caller, start, in_noyield, in_restricted, eff, spawns, edges);
                scan_expr_effect(caller, end, in_noyield, in_restricted, eff, spawns, edges);
                scan_stmts_effect(
                    caller,
                    &body.stmts,
                    stmt_manual,
                    in_noyield,
                    in_restricted,
                    eff,
                    spawns,
                    edges,
                );
            }
            StmtKind::ForEach { iter, body, .. } => {
                scan_expr_effect(caller, iter, in_noyield, in_restricted, eff, spawns, edges);
                scan_stmts_effect(
                    caller,
                    &body.stmts,
                    stmt_manual,
                    in_noyield,
                    in_restricted,
                    eff,
                    spawns,
                    edges,
                );
            }
            StmtKind::Defer(inner) => {
                scan_stmts_effect(
                    caller,
                    std::slice::from_ref(inner.as_ref()),
                    stmt_manual,
                    true,
                    true,
                    eff,
                    spawns,
                    edges,
                );
            }
            StmtKind::NoYield { body } => {
                scan_stmts_effect(
                    caller,
                    &body.stmts,
                    stmt_manual,
                    true,
                    in_restricted,
                    eff,
                    spawns,
                    edges,
                );
            }
            StmtKind::Critical { body } => {
                scan_stmts_effect(
                    caller,
                    &body.stmts,
                    stmt_manual,
                    true,
                    true,
                    eff,
                    spawns,
                    edges,
                );
            }
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                scan_expr_effect(caller, cond, in_noyield, in_restricted, eff, spawns, edges);
                scan_stmts_effect(
                    caller,
                    &then_block.stmts,
                    stmt_manual,
                    in_noyield,
                    in_restricted,
                    eff,
                    spawns,
                    edges,
                );
                if let Some(eb) = else_block {
                    scan_stmts_effect(
                        caller,
                        &eb.stmts,
                        stmt_manual,
                        in_noyield,
                        in_restricted,
                        eff,
                        spawns,
                        edges,
                    );
                }
            }
            StmtKind::Switch { scrutinee, arms } => {
                scan_expr_effect(caller, scrutinee, in_noyield, in_restricted, eff, spawns, edges);
                for arm in arms {
                    scan_stmts_effect(
                        caller,
                        &arm.body.stmts,
                        stmt_manual,
                        in_noyield,
                        in_restricted,
                        eff,
                        spawns,
                        edges,
                    );
                }
            }
            StmtKind::Block(b) => {
                scan_stmts_effect(
                    caller,
                    &b.stmts,
                    stmt_manual,
                    in_noyield,
                    in_restricted,
                    eff,
                    spawns,
                    edges,
                );
            }
            StmtKind::VarDecl { init: Some(e), .. }
            | StmtKind::Return(Some(e))
            | StmtKind::Expr(e) => {
                scan_expr_effect(caller, e, in_noyield, in_restricted, eff, spawns, edges);
            }
            StmtKind::Assign { target, value, .. } => {
                scan_expr_effect(caller, target, in_noyield, in_restricted, eff, spawns, edges);
                scan_expr_effect(caller, value, in_noyield, in_restricted, eff, spawns, edges);
            }
            StmtKind::Error { reason, partial } => {
                if let Some(r) = reason {
                    scan_expr_effect(caller, r, in_noyield, in_restricted, eff, spawns, edges);
                }
                if let Some(p) = partial {
                    scan_expr_effect(caller, p, in_noyield, in_restricted, eff, spawns, edges);
                }
            }
            StmtKind::Spawn(sp) => {
                *spawns = true;
                for a in &sp.args {
                    scan_expr_effect(caller, &a.value, in_noyield, in_restricted, eff, spawns, edges);
                }
            }
            StmtKind::VarDecl { init: None, .. }
            | StmtKind::Return(None)
            | StmtKind::Continue
            | StmtKind::Break => {}
        }
    }
}

fn scan_expr_effect(
    caller: &str,
    expr: &Expr,
    in_noyield: bool,
    in_restricted: bool,
    eff: &mut MirFnEffect,
    spawns: &mut bool,
    edges: &mut Vec<RawCallEdge>,
) {
    match &expr.kind {
        ExprKind::Call { callee, args, .. } => {
            if let ExprKind::Ident(id) = &callee.kind {
                match id.name.as_str() {
                    "delay" | "delay_ms" | "sleep" => {
                        if !in_noyield && !in_restricted {
                            *eff = (*eff).max(MirFnEffect::HardBlocking);
                        }
                    }
                    "checkpoint" => {
                        if !in_noyield && !in_restricted {
                            *eff = (*eff).max(MirFnEffect::SoftYielding);
                        }
                    }
                    _ => {
                        edges.push(RawCallEdge {
                            caller: caller.to_string(),
                            callee: id.name.clone(),
                            in_noyield,
                            in_restricted,
                        });
                    }
                }
            }
            scan_expr_effect(caller, callee, in_noyield, in_restricted, eff, spawns, edges);
            for a in args {
                scan_expr_effect(caller, &a.value, in_noyield, in_restricted, eff, spawns, edges);
            }
        }
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => {
            let m = method.name.as_str();
            if matches!(m, "attach" | "attach_shared") {
                *spawns = true;
            }
            if matches!(
                m,
                "wait_chunk" | "recv" | "wait" | "send" | "delay" | "wait_change"
            ) && !in_noyield
                && !in_restricted
            {
                *eff = (*eff).max(MirFnEffect::HardBlocking);
            }
            scan_expr_effect(caller, receiver, in_noyield, in_restricted, eff, spawns, edges);
            for a in args {
                scan_expr_effect(caller, &a.value, in_noyield, in_restricted, eff, spawns, edges);
            }
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::Field { target: inner, .. } => {
            scan_expr_effect(caller, inner, in_noyield, in_restricted, eff, spawns, edges);
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
            scan_expr_effect(caller, lhs, in_noyield, in_restricted, eff, spawns, edges);
            scan_expr_effect(caller, rhs, in_noyield, in_restricted, eff, spawns, edges);
        }
        ExprKind::Slice { target, start, end } => {
            scan_expr_effect(caller, target, in_noyield, in_restricted, eff, spawns, edges);
            if let Some(s) = start {
                scan_expr_effect(caller, s, in_noyield, in_restricted, eff, spawns, edges);
            }
            if let Some(e) = end {
                scan_expr_effect(caller, e, in_noyield, in_restricted, eff, spawns, edges);
            }
        }
        ExprKind::StructLit { fields, .. } => {
            for f in fields {
                scan_expr_effect(caller, &f.value, in_noyield, in_restricted, eff, spawns, edges);
            }
        }
        ExprKind::ArrayLit(elems) => {
            for e in elems {
                scan_expr_effect(caller, e, in_noyield, in_restricted, eff, spawns, edges);
            }
        }
        ExprKind::FormatString(parts) => {
            for part in parts {
                if let FormatPart::Hole { expr: he, .. } = part {
                    scan_expr_effect(caller, he, in_noyield, in_restricted, eff, spawns, edges);
                }
            }
        }
        ExprKind::Catch { expr: inner, clause } => {
            scan_expr_effect(caller, inner, in_noyield, in_restricted, eff, spawns, edges);
            scan_stmts_effect(
                caller,
                &clause.body.stmts,
                false,
                in_noyield,
                in_restricted,
                eff,
                spawns,
                edges,
            );
        }
        ExprKind::Literal(_) | ExprKind::Ident(_) | ExprKind::Priority(_) => {}
    }
}

fn stmt_slice_can_yield(stmts: &[Stmt], manual_yield: bool, syms: &ModuleSymbols) -> bool {
    let mut eff = MirFnEffect::NonYielding;
    let mut dummy_sp = false;
    let mut edges = Vec::new();
    scan_stmts_effect(
        "__slice",
        stmts,
        manual_yield,
        false,
        false,
        &mut eff,
        &mut dummy_sp,
        &mut edges,
    );
    if eff.can_yield() {
        return true;
    }
    edges.iter().any(|e| {
        syms.fns.get(&e.callee).map_or(false, |f| {
            f.effect == MirFnEffect::HardBlocking
                || (f.effect == MirFnEffect::SoftYielding && !e.in_noyield)
        })
    })
}

fn stmt_contains_boot_spawn(stmt: &Stmt, syms: &ModuleSymbols) -> bool {
    match &stmt.kind {
        StmtKind::Spawn(_) => true,
        StmtKind::If {
            cond,
            then_block,
            else_block,
        } => {
            expr_contains_boot_spawn(cond, syms)
                || then_block
                    .stmts
                    .iter()
                    .any(|s| stmt_contains_boot_spawn(s, syms))
                || else_block.as_ref().is_some_and(|eb| {
                    eb.stmts
                        .iter()
                        .any(|s| stmt_contains_boot_spawn(s, syms))
                })
        }
        StmtKind::Switch { scrutinee, arms } => {
            expr_contains_boot_spawn(scrutinee, syms)
                || arms.iter().any(|a| {
                    a.body
                        .stmts
                        .iter()
                        .any(|s| stmt_contains_boot_spawn(s, syms))
                })
        }
        StmtKind::Block(b) | StmtKind::NoYield { body: b } | StmtKind::Critical { body: b } => {
            b.stmts.iter().any(|s| stmt_contains_boot_spawn(s, syms))
        }
        StmtKind::VarDecl { init: Some(e), .. }
        | StmtKind::Assign { value: e, .. }
        | StmtKind::Expr(e)
        | StmtKind::Return(Some(e)) => expr_contains_boot_spawn(e, syms),
        _ => false,
    }
}

fn expr_contains_boot_spawn(expr: &Expr, syms: &ModuleSymbols) -> bool {
    match &expr.kind {
        ExprKind::Call { callee, args, .. } => {
            if let ExprKind::Ident(id) = &callee.kind {
                if syms
                    .fns
                    .get(&id.name)
                    .is_some_and(|sig| sig.spawns_in_boot)
                {
                    return true;
                }
            }
            expr_contains_boot_spawn(callee, syms)
                || args.iter().any(|a| expr_contains_boot_spawn(&a.value, syms))
        }
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => {
            if matches!(method.name.as_str(), "attach" | "attach_shared") {
                return true;
            }
            expr_contains_boot_spawn(receiver, syms)
                || args.iter().any(|a| expr_contains_boot_spawn(&a.value, syms))
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::Field { target: inner, .. } => expr_contains_boot_spawn(inner, syms),
        ExprKind::Binary { lhs, rhs, .. }
        | ExprKind::Index {
            target: lhs,
            index: rhs,
        }
        | ExprKind::Or {
            expr: lhs,
            fallback: rhs,
        } => expr_contains_boot_spawn(lhs, syms) || expr_contains_boot_spawn(rhs, syms),
        ExprKind::Catch { expr: inner, clause } => {
            expr_contains_boot_spawn(inner, syms)
                || clause
                    .body
                    .stmts
                    .iter()
                    .any(|s| stmt_contains_boot_spawn(s, syms))
        }
        ExprKind::FormatString(parts) => parts.iter().any(|p| match p {
            FormatPart::Literal(_) => false,
            FormatPart::Hole { expr: he, .. } => expr_contains_boot_spawn(he, syms),
        }),
        _ => false,
    }
}

fn detect_static_set_priority(stmts: &[Stmt]) -> Option<MirPriority> {
    for stmt in stmts {
        if let StmtKind::Expr(Expr {
            kind: ExprKind::Call { callee, args, .. },
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

struct BootIrqBindingRec {
    branch_path: Vec<(usize, usize)>,
    binding: MirIrqBinding,
}

struct BootIrqVectorAcc {
    irq_name: String,
    irq_num: u32,
    is_shared: bool,
    selector_symbol: Option<String>,
    next_selector_val: u8,
    branch_selectors: Vec<(Vec<(usize, usize)>, u8)>,
    records: Vec<BootIrqBindingRec>,
}

struct BootSpawnState<'a> {
    fn_decls: &'a BTreeMap<String, &'a FnDecl>,
    mir_isrs: &'a [MirIsr],
    spawned_tasks: &'a mut Vec<MirSpawnInstance>,
    irq_vectors: &'a mut BTreeMap<u32, BootIrqVectorAcc>,
    bp_instance_counts: BTreeMap<String, usize>,
    provenance_stack: Vec<String>,
    branch_depth: usize,
    branch_stack: Vec<(usize, usize)>,
    next_branch_id: usize,
    loop_depth: usize,
    next_binding_id: usize,
    irq_param_bindings: BTreeMap<String, (String, u32)>,
}

fn is_const_mir_expr(expr: &MirExpr) -> bool {
    match &expr.kind {
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::StringLit(_)
        | MirExprKind::TicksLit(_)
        | MirExprKind::EnumVariant { .. }
        | MirExprKind::GlobalPrimAddr(_)
        | MirExprKind::FnAddr(_)
        | MirExprKind::MmioAddr(_)
        | MirExprKind::ConstRef(_) => true,
        MirExprKind::Unary { expr, .. } | MirExprKind::Cast { expr, .. } => {
            is_const_mir_expr(expr)
        }
        MirExprKind::Binary { lhs, rhs, .. } => {
            is_const_mir_expr(lhs) && is_const_mir_expr(rhs)
        }
        MirExprKind::StructLit { fields, .. } => {
            fields.iter().all(|(_, e, _)| is_const_mir_expr(e))
        }
        MirExprKind::ArrayLit { elems, .. } => elems.iter().all(is_const_mir_expr),
        _ => false,
    }
}

fn resolve_boot_const_expr(expr: &MirExpr, env: &BTreeMap<VarId, MirExpr>) -> MirExpr {
    match &expr.kind {
        MirExprKind::Var(vid) => env.get(vid).cloned().unwrap_or_else(|| expr.clone()),
        MirExprKind::Unary { op, expr: inner } => MirExpr::new(
            MirExprKind::Unary {
                op: *op,
                expr: Box::new(resolve_boot_const_expr(inner, env)),
            },
            expr.ty.clone(),
        ),
        MirExprKind::Binary { op, lhs, rhs } => MirExpr::new(
            MirExprKind::Binary {
                op: *op,
                lhs: Box::new(resolve_boot_const_expr(lhs, env)),
                rhs: Box::new(resolve_boot_const_expr(rhs, env)),
            },
            expr.ty.clone(),
        ),
        MirExprKind::Cast {
            expr: inner,
            target_ty,
        } => MirExpr::new(
            MirExprKind::Cast {
                expr: Box::new(resolve_boot_const_expr(inner, env)),
                target_ty: target_ty.clone(),
            },
            expr.ty.clone(),
        ),
        MirExprKind::StructLit {
            struct_name,
            packed,
            fields,
        } => MirExpr::new(
            MirExprKind::StructLit {
                struct_name: struct_name.clone(),
                packed: *packed,
                fields: fields
                    .iter()
                    .map(|(k, v, t)| (k.clone(), resolve_boot_const_expr(v, env), t.clone()))
                    .collect(),
            },
            expr.ty.clone(),
        ),
        MirExprKind::ArrayLit { elems, elem_ty } => MirExpr::new(
            MirExprKind::ArrayLit {
                elems: elems
                    .iter()
                    .map(|e| resolve_boot_const_expr(e, env))
                    .collect(),
                elem_ty: elem_ty.clone(),
            },
            expr.ty.clone(),
        ),
        _ => expr.clone(),
    }
}

fn zero_init_expr(ty: &MirType, syms: &ModuleSymbols) -> MirExpr {
    match ty {
        MirType::Bool => MirExpr::new(MirExprKind::BoolLit(false), MirType::Bool),
        MirType::F32 | MirType::F64 => MirExpr::new(MirExprKind::FloatLit(0), ty.clone()),
        MirType::U8
        | MirType::U16
        | MirType::U32
        | MirType::U64
        | MirType::Usize
        | MirType::I8
        | MirType::I16
        | MirType::I32
        | MirType::I64
        | MirType::Isize => MirExpr::new(MirExprKind::IntLit(0), ty.clone()),
        MirType::EndianInt { .. } => {
            MirExpr::new(MirExprKind::IntLit(0), ty.host_value_type())
        }
        MirType::Enum(name) => {
            if let Some(en) = syms.enums.get(name) {
                if let Some((first_name, _)) = en.first() {
                    return MirExpr::new(
                        MirExprKind::EnumVariant {
                            enum_name: name.clone(),
                            variant: first_name.clone(),
                        },
                        ty.clone(),
                    );
                }
            }
            MirExpr::new(MirExprKind::IntLit(0), ty.clone())
        }
        MirType::Array { elem, len } => {
            let z = zero_init_expr(elem, syms);
            MirExpr::new(
                MirExprKind::ArrayLit {
                    elems: std::iter::repeat(z).take(*len).collect(),
                    elem_ty: (**elem).clone(),
                },
                ty.clone(),
            )
        }
        MirType::Struct(name) => {
            let (packed, fields) = syms
                .structs
                .get(name)
                .map(|st| {
                    (
                        st.packed,
                        st.fields
                            .iter()
                            .map(|f| (f.name.clone(), zero_init_expr(&f.ty, syms), f.ty.clone()))
                            .collect(),
                    )
                })
                .unwrap_or((false, Vec::new()));
            MirExpr::new(
                MirExprKind::StructLit {
                    struct_name: name.clone(),
                    packed,
                    fields,
                },
                ty.clone(),
            )
        }
        _ => MirExpr::new(MirExprKind::IntLit(0), ty.clone()),
    }
}

fn place_root_var(place: &MirPlace) -> Option<VarId> {
    match place {
        MirPlace::Var(vid) => Some(*vid),
        MirPlace::Field { base, .. } | MirPlace::Index { base, .. } => place_root_var(base),
        MirPlace::Deref(_, _) => None,
    }
}

fn paths_mutually_exclusive(a: &[(usize, usize)], b: &[(usize, usize)]) -> bool {
    a.iter()
        .zip(b.iter())
        .any(|((ba, aa), (bb, ab))| ba == bb && aa != ab)
}

impl<'a> BootSpawnState<'a> {
    fn attach_irq(
        &mut self,
        irq_name: &str,
        irq_num: u32,
        is_shared: bool,
        handler_name: &str,
        baked_args: Vec<MirBakedArg>,
        span: Span,
    ) -> Result<Option<MirStmt>, MirLowerError> {
        if self.loop_depth > 0 {
            return Err(MirLowerError {
                message: format!(
                    "IRQ `.{}` cannot be called inside a loop",
                    if is_shared { "attach_shared" } else { "attach" }
                ),
                span,
            });
        }
        let isr_decl = self
            .mir_isrs
            .iter()
            .find(|isr| isr.name == handler_name)
            .ok_or_else(|| MirLowerError {
                message: format!("unknown ISR handler `{handler_name}` in IRQ attachment"),
                span,
            })?;
        if baked_args.len() != isr_decl.params.len() {
            return Err(MirLowerError {
                message: format!(
                    "ISR `{handler_name}` expects {} argument(s), got {}",
                    isr_decl.params.len(),
                    baked_args.len()
                ),
                span,
            });
        }

        let binding_id = self.next_binding_id;
        self.next_binding_id += 1;
        let state_vars: Vec<MirIsrStateVar> = isr_decl
            .state_vars
            .iter()
            .map(|sv| MirIsrStateVar {
                storage_symbol: format!(
                    "__zev_isr_state_{}_{}",
                    binding_id,
                    sanitize_ident(&sv.name)
                ),
                var_name: sv.name.clone(),
                ty: sv.ty.clone(),
                init: sv.init.clone(),
            })
            .collect();

        let acc = self
            .irq_vectors
            .entry(irq_num)
            .or_insert_with(|| BootIrqVectorAcc {
                irq_name: irq_name.to_string(),
                irq_num,
                is_shared,
                selector_symbol: None,
                next_selector_val: 1,
                branch_selectors: Vec::new(),
                records: Vec::new(),
            });
        if acc.is_shared != is_shared {
            return Err(MirLowerError {
                message: format!(
                    "cannot mix `.attach` and `.attach_shared` on IRQ `{}` ({})",
                    acc.irq_name, irq_num
                ),
                span,
            });
        }

        let mut emit_sel_stmt = None;
        let cond_selector = if !is_shared {
            for rec in &acc.records {
                if !paths_mutually_exclusive(&self.branch_stack, &rec.branch_path) {
                    return Err(MirLowerError {
                        message: format!(
                            "conflicting `.attach` calls on IRQ `{}` ({}); use `.attach_shared` or mutually exclusive boot branches",
                            acc.irq_name, irq_num
                        ),
                        span,
                    });
                }
            }
            if self.branch_depth > 0 {
                let sel_sym = acc
                    .selector_symbol
                    .get_or_insert_with(|| format!("__zev_irq_sel_{irq_num}"))
                    .clone();
                let sel_val = acc.next_selector_val;
                acc.next_selector_val =
                    acc.next_selector_val
                        .checked_add(1)
                        .ok_or_else(|| MirLowerError {
                            message: format!(
                                "too many conditional `.attach` branches on IRQ `{irq_num}`"
                            ),
                            span,
                        })?;
                acc.branch_selectors
                    .push((self.branch_stack.clone(), sel_val));
                emit_sel_stmt = Some(MirStmt::SetIrqSelector {
                    symbol: sel_sym.clone(),
                    value: sel_val,
                });
                Some((sel_sym, sel_val))
            } else {
                None
            }
        } else if self.branch_depth > 0 {
            if let Some((_, existing_val)) = acc
                .branch_selectors
                .iter()
                .find(|(bp, _)| *bp == self.branch_stack)
            {
                let sel_sym = acc.selector_symbol.as_ref().unwrap().clone();
                Some((sel_sym, *existing_val))
            } else {
                for (bp, _) in &acc.branch_selectors {
                    if !paths_mutually_exclusive(&self.branch_stack, bp) {
                        return Err(MirLowerError {
                            message: format!(
                                "conditional `.attach_shared` calls on IRQ `{}` ({}) must share the same branch or be mutually exclusive",
                                acc.irq_name, irq_num
                            ),
                            span,
                        });
                    }
                }
                let sel_sym = acc
                    .selector_symbol
                    .get_or_insert_with(|| format!("__zev_irq_sel_{irq_num}"))
                    .clone();
                let sel_val = acc.next_selector_val;
                acc.next_selector_val =
                    acc.next_selector_val
                        .checked_add(1)
                        .ok_or_else(|| MirLowerError {
                            message: format!(
                                "too many conditional `.attach_shared` branches on IRQ `{irq_num}`"
                            ),
                            span,
                        })?;
                acc.branch_selectors
                    .push((self.branch_stack.clone(), sel_val));
                emit_sel_stmt = Some(MirStmt::SetIrqSelector {
                    symbol: sel_sym.clone(),
                    value: sel_val,
                });
                Some((sel_sym, sel_val))
            }
        } else {
            None
        };

        acc.records.push(BootIrqBindingRec {
            branch_path: self.branch_stack.clone(),
            binding: MirIrqBinding {
                binding_id,
                isr_name: handler_name.to_string(),
                baked_args,
                state_vars,
                cond_selector,
            },
        });
        Ok(emit_sel_stmt)
    }
}

fn lower_main_entry<'a>(
    md: &FnDecl,
    syms: &'a ModuleSymbols,
    fn_decls: &'a BTreeMap<String, &'a FnDecl>,
    mir_isrs: &'a [MirIsr],
    blueprints: &mut Vec<MirTaskBlueprint>,
    spawned_tasks: &mut Vec<MirSpawnInstance>,
    mir_irq_vectors: &mut Vec<MirIrqVector>,
) -> Result<MirBootSequence, MirLowerError> {
    let body = md.body.as_ref().unwrap();
    let stmts = &body.stmts;
    let manual_yield = has_attr_manual_yield(&md.attrs);

    // Locate the last top-level boot statement containing a `spawn` or IRQ `.attach` (directly, conditionally, or via boot helper).
    let last_spawn_idx = stmts
        .iter()
        .rposition(|s| stmt_contains_boot_spawn(s, syms));

    let (boot_slice, tail_slice) = match last_spawn_idx {
        Some(idx) => (&stmts[..=idx], &stmts[idx + 1..]),
        None => {
            if stmt_slice_can_yield(stmts, manual_yield, syms) {
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

    let mut boot_steps: Vec<MirBootStep> = Vec::new();
    let mut boot_yielding_callees: BTreeSet<String> = BTreeSet::new();
    let mut irq_vectors_acc: BTreeMap<u32, BootIrqVectorAcc> = BTreeMap::new();

    let combined_boot_iter: Vec<&Stmt> = if tail_becomes_task {
        boot_slice.iter().collect()
    } else {
        boot_slice.iter().chain(tail_slice.iter()).collect()
    };

    let boot_vars = {
        let boot_state = BootSpawnState {
            fn_decls,
            mir_isrs,
            spawned_tasks,
            irq_vectors: &mut irq_vectors_acc,
            bp_instance_counts: BTreeMap::new(),
            provenance_stack: vec!["main".to_string()],
            branch_depth: 0,
            branch_stack: Vec::new(),
            next_branch_id: 0,
            loop_depth: 0,
            next_binding_id: 0,
            irq_param_bindings: BTreeMap::new(),
        };
        let mut dctx = DirectLowerCtx::new_with_boot(
            syms,
            MirType::Void,
            MirErrorSpec::None,
            boot_state,
        );

        for stmt in &combined_boot_iter {
            match &stmt.kind {
                // Support sequential boot calls to yielding functions (e.g. `run_power_on_self_test();`)!
                StmtKind::Expr(Expr {
                    kind: ExprKind::Call { callee, args, .. },
                    ..
                }) if is_yielding_fn_callee(callee, syms) => {
                    let cname = match &callee.kind {
                        ExprKind::Ident(id) => id.name.clone(),
                        _ => unreachable!(),
                    };
                    let fn_sig = syms.fns.get(&cname).unwrap().clone();
                    let mut pre_stmts = Vec::new();
                    let mut largs = Vec::new();
                    for (i, a) in args.iter().enumerate() {
                        largs.push(dctx.lower_expr_with_stmts(
                            &a.value,
                            fn_sig.params.get(i),
                            &mut pre_stmts,
                        )?);
                    }
                    for ps in pre_stmts {
                        boot_steps.push(MirBootStep::Stmt(ps));
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
                            kind: ExprKind::Call { callee, args, .. },
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
                    let mut pre_stmts = Vec::new();
                    let mut largs = Vec::new();
                    for (i, a) in args.iter().enumerate() {
                        largs.push(dctx.lower_expr_with_stmts(
                            &a.value,
                            fn_sig.params.get(i),
                            &mut pre_stmts,
                        )?);
                    }
                    for ps in pre_stmts {
                        boot_steps.push(MirBootStep::Stmt(ps));
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
                        match ls {
                            MirStmt::SpawnActivate { slot_id, args } => {
                                boot_steps.push(MirBootStep::SpawnActivate { slot_id, args });
                            }
                            other => boot_steps.push(MirBootStep::Stmt(other)),
                        }
                    }
                }
            }
        }

        if !tail_becomes_task {
            let root_defers = dctx.pop_and_lower_top_defer_scope()?;
            for ls in root_defers {
                boot_steps.push(MirBootStep::Stmt(ls));
            }
        }

        // Capture `main`'s root lexical scope vars before dropping `dctx` so `tail_slice` can forward them.
        let root_scope_map = dctx.scopes.frames.first().cloned().unwrap_or_default();
        (dctx.vars, root_scope_map)
    };

    let (boot_vars_vec, root_scope_map) = boot_vars;

    for (_, acc) in irq_vectors_acc {
        mir_irq_vectors.push(MirIrqVector {
            irq_name: acc.irq_name,
            irq_num: acc.irq_num,
            is_shared: acc.is_shared,
            selector_symbol: acc.selector_symbol,
            bindings: acc.records.into_iter().map(|r| r.binding).collect(),
        });
    }

    if tail_becomes_task {
        // Forward any root `main` boot variables referenced inside `tail_slice` into `main` supervisor's frame!
        let mut tail_ident_refs = BTreeSet::new();
        collect_ast_idents_stmts(tail_slice, &mut tail_ident_refs);

        let mut forwarded_params: Vec<(String, MirType, MirExpr)> = Vec::new();
        for (vname, &vid) in &root_scope_map {
            if tail_ident_refs.contains(vname) {
                let bvar = &boot_vars_vec[vid.0];
                forwarded_params.push((
                    vname.clone(),
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

        cctx.lower_top_level_stmts(tail_slice)?;
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
            provenance_path: vec!["main".to_string()],
            conditional: false,
        });
        boot_steps.push(MirBootStep::SpawnActivate {
            slot_id,
            args: init_args,
        });
    }

    Ok(MirBootSequence {
        present: true,
        vars: boot_vars_vec,
        steps: boot_steps,
        boot_yielding_callees: boot_yielding_callees.into_iter().collect(),
    })
}

fn is_yielding_fn_callee(callee: &Expr, syms: &ModuleSymbols) -> bool {
    if let ExprKind::Ident(id) = &callee.kind {
        syms.fns
            .get(&id.name)
            .map_or(false, |f| f.is_yielding && !f.spawns_in_boot)
    } else {
        false
    }
}

fn is_pure_boot_setup_stmt(stmt: &Stmt) -> bool {
    match &stmt.kind {
        StmtKind::Expr(Expr {
            kind: ExprKind::MethodCall {
                receiver, method, ..
            },
            ..
        }) => {
            if matches!(method.name.as_str(), "attach" | "attach_shared") {
                return true;
            }
            if let ExprKind::Ident(id) = &receiver.kind {
                id.name == "sys"
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
            StmtKind::LoopEvery { period, body } => {
                collect_ast_idents_expr(period, out);
                collect_ast_idents_stmts(&body.stmts, out);
            }
            StmtKind::ForRange {
                start, end, body, ..
            } => {
                collect_ast_idents_expr(start, out);
                collect_ast_idents_expr(end, out);
                collect_ast_idents_stmts(&body.stmts, out);
            }
            StmtKind::ForEach { iter, body, .. } => {
                collect_ast_idents_expr(iter, out);
                collect_ast_idents_stmts(&body.stmts, out);
            }
            StmtKind::Defer(inner) => {
                collect_ast_idents_stmts(std::slice::from_ref(inner.as_ref()), out);
            }
            StmtKind::Loop { body }
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
        ExprKind::Call { callee, args, .. } => {
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
        ExprKind::FormatString(parts) => {
            for part in parts {
                if let FormatPart::Hole { expr: he, .. } = part {
                    collect_ast_idents_expr(he, out);
                }
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

fn sanitize_ident(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

fn eval_const_u32(
    expr: &Expr,
    syms: &ModuleSymbols,
    what: &str,
    span: Span,
) -> Result<u32, MirLowerError> {
    match eval_const_expr(expr, syms) {
        Some(ConstVal::Int(v)) => u32::try_from(v).map_err(|_| MirLowerError {
            message: format!("{what} `{v}` is out of `u32` range"),
            span,
        }),
        _ => Err(MirLowerError {
            message: format!("{what} must be a compile-time integer constant"),
            span,
        }),
    }
}

fn eval_const_u64(
    expr: &Expr,
    syms: &ModuleSymbols,
    what: &str,
    span: Span,
) -> Result<u64, MirLowerError> {
    match eval_const_expr(expr, syms) {
        Some(ConstVal::Int(v)) => u64::try_from(v).map_err(|_| MirLowerError {
            message: format!("{what} `{v}` is out of `u64` range"),
            span,
        }),
        _ => Err(MirLowerError {
            message: format!("{what} must be a compile-time integer constant"),
            span,
        }),
    }
}

fn mmio_elem_bits_from_type(ty: &TypeExpr, span: Span) -> Result<u8, MirLowerError> {
    match &ty.kind {
        TypeKind::Prim(PrimType::U8) => Ok(8),
        TypeKind::Prim(PrimType::U16) => Ok(16),
        TypeKind::Prim(PrimType::U32) => Ok(32),
        TypeKind::Named(id) => match id.name.as_str() {
            "u8" => Ok(8),
            "u16" => Ok(16),
            "u32" => Ok(32),
            _ => Err(MirLowerError {
                message: format!(
                    "MMIO register element type `{}` must be `u8`, `u16`, or `u32`",
                    id.name
                ),
                span,
            }),
        },
        _ => Err(MirLowerError {
            message: "MMIO register element type must be `u8`, `u16`, or `u32`".to_string(),
            span,
        }),
    }
}

fn lower_mmio_regs(
    regs: &[MmioRegDecl],
    syms: &ModuleSymbols,
) -> Result<(Vec<MirMmioReg>, usize), MirLowerError> {
    let mut out = Vec::with_capacity(regs.len());
    let mut byte_size = 0usize;
    for r in regs {
        let access = match r.access {
            MmioAccess::Rw => MirMmioAccess::Rw,
            MmioAccess::Ro => MirMmioAccess::Ro,
            MmioAccess::Wo => MirMmioAccess::Wo,
            MmioAccess::W1c => MirMmioAccess::W1c,
        };
        let (elem_bits, array_len) = match &r.ty.kind {
            TypeKind::Array { elem, len } => {
                let bits = mmio_elem_bits_from_type(elem, r.span)?;
                let n = match len {
                    ArrayLen::Fixed(n) => *n,
                    ArrayLen::Expr(e) => match eval_const_expr(e, syms) {
                        Some(ConstVal::Int(v)) if v > 0 => v as usize,
                        _ => {
                            return Err(MirLowerError {
                                message: format!(
                                    "MMIO register array `{}` length must be a positive constant",
                                    r.name.name
                                ),
                                span: r.span,
                            });
                        }
                    },
                };
                if n == 0 {
                    return Err(MirLowerError {
                        message: format!(
                            "MMIO register array `{}` length must be greater than zero",
                            r.name.name
                        ),
                        span: r.span,
                    });
                }
                (bits, Some(n))
            }
            _ => (mmio_elem_bits_from_type(&r.ty, r.span)?, None),
        };
        let offset = eval_const_u32(&r.offset, syms, "MMIO register offset", r.span)?;
        let reg_bytes = (elem_bits as usize / 8) * array_len.unwrap_or(1);
        let end = (offset as usize).checked_add(reg_bytes).ok_or_else(|| MirLowerError {
            message: format!("MMIO register `{}` offset overflow", r.name.name),
            span: r.span,
        })?;
        if end > byte_size {
            byte_size = end;
        }
        out.push(MirMmioReg {
            name: r.name.name.clone(),
            access,
            elem_bits,
            array_len,
            offset,
        });
    }
    Ok((out, byte_size))
}

fn mmio_elem_mir_type(elem_bits: u8) -> MirType {
    match elem_bits {
        8 => MirType::U8,
        16 => MirType::U16,
        _ => MirType::U32,
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

fn eval_const_expr(expr: &Expr, syms: &ModuleSymbols) -> Option<ConstVal> {
    match &expr.kind {
        ExprKind::Literal(Literal::Int(v)) => Some(ConstVal::Int(*v as i128)),
        ExprKind::Literal(Literal::Bool(b)) => Some(ConstVal::Bool(*b)),
        ExprKind::Ident(id) => {
            if let Some(&cv) = syms.const_vals.get(&id.name) {
                return Some(cv);
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
        ExprKind::Field { target, field } => {
            if let ExprKind::Ident(id) = &target.kind {
                if let Some(variants) = syms.enums.get(&id.name) {
                    if let Some((_, disc)) = variants.iter().find(|(vn, _)| vn == &field.name) {
                        return Some(ConstVal::Int(*disc as i128));
                    }
                }
                if syms.import_aliases.contains_key(&id.name) {
                    for cand in syms.candidate_module_names(&id.name, &field.name) {
                        if let Some(&cv) = syms.const_vals.get(&cand) {
                            return Some(cv);
                        }
                    }
                }
            }
            None
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
        SwitchPat::Wildcard(_) | SwitchPat::Qualified { .. } => None,
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

struct InlinedHelperFrame {
    ret_ty: MirType,
    ret_val_var: Option<VarId>,
    ret_flag_var: Option<VarId>,
    entry_defer_depth: usize,
}

struct DirectLowerCtx<'a> {
    syms: &'a ModuleSymbols,
    ret_ty: MirType,
    error_spec: MirErrorSpec,
    vars: Vec<MirVar>,
    scopes: VarScopes,
    defer_stack: Vec<Vec<Stmt>>,
    loop_defer_depths: Vec<usize>,
    boot_state: Option<BootSpawnState<'a>>,
    boot_const_env: BTreeMap<VarId, MirExpr>,
    inlined_helper_stack: Vec<InlinedHelperFrame>,
    in_noyield: bool,
    in_isr: bool,
    isr_state_vars: Vec<MirIsrStateDecl>,
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
            defer_stack: vec![Vec::new()],
            loop_defer_depths: Vec::new(),
            boot_state: None,
            boot_const_env: BTreeMap::new(),
            inlined_helper_stack: Vec::new(),
            in_noyield: false,
            in_isr: false,
            isr_state_vars: Vec::new(),
            pending_stmts: Vec::new(),
        }
    }

    fn new_with_boot(
        syms: &'a ModuleSymbols,
        ret_ty: MirType,
        error_spec: MirErrorSpec,
        boot_state: BootSpawnState<'a>,
    ) -> Self {
        Self {
            syms,
            ret_ty,
            error_spec,
            vars: Vec::new(),
            scopes: VarScopes::new(),
            defer_stack: vec![Vec::new()],
            loop_defer_depths: Vec::new(),
            boot_state: Some(boot_state),
            boot_const_env: BTreeMap::new(),
            inlined_helper_stack: Vec::new(),
            in_noyield: false,
            in_isr: false,
            isr_state_vars: Vec::new(),
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

    fn has_active_defers_from(&self, min_depth: usize) -> bool {
        self.defer_stack
            .iter()
            .skip(min_depth)
            .any(|frame| !frame.is_empty())
    }

    fn lower_single_deferred_stmt(&mut self, deferred: &Stmt) -> Result<Vec<MirStmt>, MirLowerError> {
        let prev_ny = self.in_noyield;
        self.in_noyield = true;
        self.scopes.push();
        self.defer_stack.push(Vec::new());
        let res = self.lower_stmt(deferred);
        self.defer_stack.pop();
        self.scopes.pop();
        self.in_noyield = prev_ny;
        res
    }

    fn pop_and_lower_top_defer_scope(&mut self) -> Result<Vec<MirStmt>, MirLowerError> {
        let Some(defers) = self.defer_stack.pop() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for d in defers.iter().rev() {
            out.extend(self.lower_single_deferred_stmt(d)?);
        }
        Ok(out)
    }

    /// Emit active `defer` statements from innermost scope down to `min_depth` (inclusive) WITHOUT
    /// popping `self.defer_stack` (used for `return`, `error`, `break`, `continue`, `sys.exit`).
    fn emit_defers_down_to(&mut self, min_depth: usize) -> Result<Vec<MirStmt>, MirLowerError> {
        let top = self.defer_stack.len();
        if min_depth >= top {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let mut popped_var_frames: Vec<BTreeMap<String, VarId>> = Vec::new();

        for depth in (min_depth..top).rev() {
            while self.scopes.frames.len() > depth + 1 {
                if let Some(vf) = self.scopes.frames.pop() {
                    popped_var_frames.push(vf);
                } else {
                    break;
                }
            }
            let cloned_defers = self.defer_stack[depth].clone();
            for d in cloned_defers.iter().rev() {
                out.extend(self.lower_single_deferred_stmt(d)?);
            }
        }

        while let Some(vf) = popped_var_frames.pop() {
            self.scopes.frames.push(vf);
        }
        Ok(out)
    }

    fn lower_top_level_stmts(&mut self, stmts: &[Stmt]) -> Result<Vec<MirStmt>, MirLowerError> {
        let mut out = self.lower_stmt_seq(stmts)?;
        out.extend(self.pop_and_lower_top_defer_scope()?);
        Ok(out)
    }

    fn lower_block_stmts(&mut self, stmts: &[Stmt]) -> Result<Vec<MirStmt>, MirLowerError> {
        self.scopes.push();
        self.defer_stack.push(Vec::new());
        let mut out = self.lower_stmt_seq(stmts)?;
        out.extend(self.pop_and_lower_top_defer_scope()?);
        self.scopes.pop();
        Ok(out)
    }

    fn lower_stmt_seq(&mut self, stmts: &[Stmt]) -> Result<Vec<MirStmt>, MirLowerError> {
        let ret_flag_opt = self
            .inlined_helper_stack
            .last()
            .and_then(|f| f.ret_flag_var);
        let mut out = Vec::new();
        for s in stmts {
            let chunk = self.lower_stmt(s)?;
            if let Some(rf) = ret_flag_opt {
                if !chunk.is_empty() {
                    out.push(MirStmt::If {
                        cond: MirExpr::new(
                            MirExprKind::Unary {
                                op: MirUnaryOp::Not,
                                expr: Box::new(MirExpr::new(MirExprKind::Var(rf), MirType::Bool)),
                            },
                            MirType::Bool,
                        ),
                        then_body: chunk,
                        else_body: Vec::new(),
                    });
                }
            } else {
                out.extend(chunk);
            }
        }
        Ok(out)
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<Vec<MirStmt>, MirLowerError> {
        let mut out = Vec::new();
        match &stmt.kind {
            StmtKind::VarDecl { ty, name, init } => {
                let vty = self.syms.resolve_type(ty);
                let role = if self.inlined_helper_stack.is_empty() {
                    VarRole::UserLocal
                } else {
                    VarRole::CompilerTemp
                };
                let vid = self.declare_var(&name.name, vty.clone(), role);
                if role == VarRole::CompilerTemp {
                    self.scopes.insert(name.name.clone(), vid);
                }
                if let Some(ie) = init {
                    let val = self.lower_expr_with_stmts(ie, Some(&vty), &mut out)?;
                    let coerced = coerce_expr_if_needed(val, &vty);
                    if self.boot_state.is_some() {
                        let resolved = resolve_boot_const_expr(&coerced, &self.boot_const_env);
                        if is_const_mir_expr(&resolved) {
                            self.boot_const_env.insert(vid, resolved);
                        }
                    }
                    out.push(MirStmt::Assign {
                        target: MirPlace::Var(vid),
                        op: MirAssignOp::Assign,
                        value: coerced,
                    });
                }
            }
            StmtKind::StateVar { ty, name, init } => {
                if !self.in_isr || self.scopes.frames.len() != 1 {
                    return Err(MirLowerError {
                        message: format!(
                            "`state` variable `{}` may only be declared at the top level of an `isr` handler",
                            name.name
                        ),
                        span: stmt.span,
                    });
                }
                let vty = self.syms.resolve_type(ty);
                let coerced_init = if let Some(ie) = init {
                    let mut init_stmts = Vec::new();
                    let val = self.lower_expr_with_stmts(ie, Some(&vty), &mut init_stmts)?;
                    let coerced = coerce_expr_if_needed(val, &vty);
                    if !init_stmts.is_empty() || !is_const_mir_expr(&coerced) {
                        return Err(MirLowerError {
                            message: format!(
                                "`state` variable `{}` initializer must be a compile-time constant expression",
                                name.name
                            ),
                            span: stmt.span,
                        });
                    }
                    coerced
                } else {
                    zero_init_expr(&vty, self.syms)
                };
                let vid = self.declare_var(&name.name, vty.clone(), VarRole::IsrState);
                self.isr_state_vars.push(MirIsrStateDecl {
                    var_id: vid,
                    name: name.name.clone(),
                    ty: vty,
                    init: coerced_init,
                });
            }
            StmtKind::Assign { target, op, value } => {
                let handled = {
                    let mut be = DirectBackend {
                        ctx: self,
                        stmts: &mut out,
                    };
                    try_lower_mmio_assign_be(target, *op, value, &mut be)?
                };
                if !handled {
                    let (place, pty) = self.lower_place(target, &mut out)?;
                    if self.boot_state.is_some() {
                        if let Some(vid) = place_root_var(&place) {
                            self.boot_const_env.remove(&vid);
                        }
                    }
                    let rhs =
                        self.lower_expr_with_stmts(value, Some(&pty.host_value_type()), &mut out)?;
                    out.push(MirStmt::Assign {
                        target: place,
                        op: convert_assign_op(*op),
                        value: coerce_expr_if_needed(rhs, &pty.host_value_type()),
                    });
                }
            }
            StmtKind::Defer(inner) => {
                if let Some(top) = self.defer_stack.last_mut() {
                    top.push((**inner).clone());
                }
            }
            StmtKind::Return(expr_opt) => {
                if let Some(hframe) = self.inlined_helper_stack.last() {
                    let h_ret_ty = hframe.ret_ty.clone();
                    let h_ret_val = hframe.ret_val_var;
                    let h_ret_flag = hframe.ret_flag_var;
                    let h_entry_depth = hframe.entry_defer_depth;
                    if let Some(e) = expr_opt {
                        let v = self.lower_expr_with_stmts(e, Some(&h_ret_ty), &mut out)?;
                        if let Some(rv) = h_ret_val {
                            out.push(MirStmt::Assign {
                                target: MirPlace::Var(rv),
                                op: MirAssignOp::Assign,
                                value: coerce_expr_if_needed(v, &h_ret_ty),
                            });
                        }
                    }
                    out.extend(self.emit_defers_down_to(h_entry_depth)?);
                    if let Some(rf) = h_ret_flag {
                        out.push(MirStmt::Assign {
                            target: MirPlace::Var(rf),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(MirExprKind::BoolLit(true), MirType::Bool),
                        });
                    }
                    return Ok(out);
                }

                let ret_ty = self.ret_ty.clone();
                let has_defers = self.has_active_defers_from(0);
                let val = if let Some(e) = expr_opt {
                    let raw = self.lower_expr_with_stmts(e, Some(&ret_ty), &mut out)?;
                    let coerced = coerce_expr_if_needed(raw, &ret_ty);
                    if has_defers && !ret_ty.is_void() {
                        // Evaluate return expression into a temp BEFORE running `defer` statements!
                        let tmp = self.alloc_temp("ret_val", ret_ty.clone());
                        out.push(MirStmt::Assign {
                            target: MirPlace::Var(tmp),
                            op: MirAssignOp::Assign,
                            value: coerced,
                        });
                        Some(MirExpr::new(MirExprKind::Var(tmp), ret_ty))
                    } else {
                        Some(coerced)
                    }
                } else {
                    None
                };
                out.extend(self.emit_defers_down_to(0)?);
                out.push(MirStmt::Return(val));
            }
            StmtKind::Error { reason, partial } => {
                let ret_ty = self.ret_ty.clone();
                let err_ty = self.error_spec.err_type();
                let has_defers = self.has_active_defers_from(0);
                let r = if let Some(re) = reason {
                    let raw = self.lower_expr_with_stmts(re, err_ty.as_ref(), &mut out)?;
                    if let (true, Some(ety)) = (has_defers, err_ty.clone()) {
                        let tmp = self.alloc_temp("err_reason", ety.clone());
                        out.push(MirStmt::Assign {
                            target: MirPlace::Var(tmp),
                            op: MirAssignOp::Assign,
                            value: coerce_expr_if_needed(raw, &ety),
                        });
                        Some(MirExpr::new(MirExprKind::Var(tmp), ety))
                    } else {
                        Some(raw)
                    }
                } else {
                    None
                };
                let p = if let Some(pe) = partial {
                    let raw = self.lower_expr_with_stmts(pe, Some(&ret_ty), &mut out)?;
                    if has_defers && !ret_ty.is_void() {
                        let tmp = self.alloc_temp("err_partial", ret_ty.clone());
                        out.push(MirStmt::Assign {
                            target: MirPlace::Var(tmp),
                            op: MirAssignOp::Assign,
                            value: coerce_expr_if_needed(raw, &ret_ty),
                        });
                        Some(MirExpr::new(MirExprKind::Var(tmp), ret_ty))
                    } else {
                        Some(raw)
                    }
                } else {
                    None
                };
                out.extend(self.emit_defers_down_to(0)?);
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
                let bid = self.boot_state.as_mut().map(|bs| {
                    let id = bs.next_branch_id;
                    bs.next_branch_id += 1;
                    bs.branch_depth += 1;
                    bs.branch_stack.push((id, 0));
                    id
                });
                let tb = self.lower_block_stmts(&then_block.stmts)?;
                if let (Some(bs), Some(id)) = (&mut self.boot_state, bid) {
                    bs.branch_stack.pop();
                    bs.branch_stack.push((id, 1));
                }
                let eb = if let Some(elb) = else_block {
                    self.lower_block_stmts(&elb.stmts)?
                } else {
                    Vec::new()
                };
                if let Some(bs) = &mut self.boot_state {
                    if bid.is_some() {
                        bs.branch_stack.pop();
                    }
                    bs.branch_depth -= 1;
                }
                out.push(MirStmt::If {
                    cond: c,
                    then_body: tb,
                    else_body: eb,
                });
            }
            StmtKind::While { cond, body } => {
                let c = self.lower_expr_with_stmts(cond, Some(&MirType::Bool), &mut out)?;
                let loop_depth = self.defer_stack.len();
                self.loop_defer_depths.push(loop_depth);
                if let Some(bs) = &mut self.boot_state {
                    bs.loop_depth += 1;
                }
                let b = self.lower_block_stmts(&body.stmts)?;
                if let Some(bs) = &mut self.boot_state {
                    bs.loop_depth -= 1;
                }
                self.loop_defer_depths.pop();
                out.push(MirStmt::While { cond: c, body: b });
            }
            StmtKind::Loop { body } => {
                let loop_depth = self.defer_stack.len();
                self.loop_defer_depths.push(loop_depth);
                if let Some(bs) = &mut self.boot_state {
                    bs.loop_depth += 1;
                }
                let b = self.lower_block_stmts(&body.stmts)?;
                if let Some(bs) = &mut self.boot_state {
                    bs.loop_depth -= 1;
                }
                self.loop_defer_depths.pop();
                out.push(MirStmt::While {
                    cond: MirExpr::new(MirExprKind::BoolLit(true), MirType::Bool),
                    body: b,
                });
            }
            StmtKind::ForRange {
                var_ty,
                var_name,
                start,
                end,
                inclusive,
                body,
            } => {
                let vty = self.syms.resolve_type(var_ty);
                let s_raw = self.lower_expr_with_stmts(start, Some(&vty), &mut out)?;
                let e_raw = self.lower_expr_with_stmts(end, Some(&vty), &mut out)?;
                let s_tmp = self.alloc_temp("for_start", vty.clone());
                let e_tmp = self.alloc_temp("for_end", vty.clone());
                out.push(MirStmt::Assign {
                    target: MirPlace::Var(s_tmp),
                    op: MirAssignOp::Assign,
                    value: coerce_expr_if_needed(s_raw, &vty),
                });
                out.push(MirStmt::Assign {
                    target: MirPlace::Var(e_tmp),
                    op: MirAssignOp::Assign,
                    value: coerce_expr_if_needed(e_raw, &vty),
                });

                self.scopes.push();
                let ivar = self.declare_var(&var_name.name, vty.clone(), VarRole::UserLocal);
                let first_tmp = self.alloc_temp("for_first", MirType::Bool);

                let mut guarded = Vec::new();
                guarded.push(MirStmt::Assign {
                    target: MirPlace::Var(ivar),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(MirExprKind::Var(s_tmp), vty.clone()),
                });
                guarded.push(MirStmt::Assign {
                    target: MirPlace::Var(first_tmp),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(MirExprKind::BoolLit(true), MirType::Bool),
                });

                let step_else = if *inclusive {
                    // Check `i >= end` BEFORE incrementing so `end == T::MAX` never overflows!
                    vec![
                        MirStmt::If {
                            cond: MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Ge,
                                    lhs: Box::new(MirExpr::new(MirExprKind::Var(ivar), vty.clone())),
                                    rhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(e_tmp),
                                        vty.clone(),
                                    )),
                                },
                                MirType::Bool,
                            ),
                            then_body: vec![MirStmt::Break],
                            else_body: Vec::new(),
                        },
                        MirStmt::Assign {
                            target: MirPlace::Var(ivar),
                            op: MirAssignOp::AddAssign,
                            value: MirExpr::new(MirExprKind::IntLit(1), vty.clone()),
                        },
                    ]
                } else {
                    vec![
                        MirStmt::Assign {
                            target: MirPlace::Var(ivar),
                            op: MirAssignOp::AddAssign,
                            value: MirExpr::new(MirExprKind::IntLit(1), vty.clone()),
                        },
                        MirStmt::If {
                            cond: MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Ge,
                                    lhs: Box::new(MirExpr::new(MirExprKind::Var(ivar), vty.clone())),
                                    rhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(e_tmp),
                                        vty.clone(),
                                    )),
                                },
                                MirType::Bool,
                            ),
                            then_body: vec![MirStmt::Break],
                            else_body: Vec::new(),
                        },
                    ]
                };

                let mut while_body = vec![MirStmt::If {
                    cond: MirExpr::new(MirExprKind::Var(first_tmp), MirType::Bool),
                    then_body: vec![MirStmt::Assign {
                        target: MirPlace::Var(first_tmp),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::BoolLit(false), MirType::Bool),
                    }],
                    else_body: step_else,
                }];

                let loop_depth = self.defer_stack.len();
                self.loop_defer_depths.push(loop_depth);
                if let Some(bs) = &mut self.boot_state {
                    bs.loop_depth += 1;
                }
                let user_body = self.lower_block_stmts(&body.stmts)?;
                if let Some(bs) = &mut self.boot_state {
                    bs.loop_depth -= 1;
                }
                self.loop_defer_depths.pop();
                while_body.extend(user_body);

                self.scopes.pop();

                guarded.push(MirStmt::While {
                    cond: MirExpr::new(MirExprKind::BoolLit(true), MirType::Bool),
                    body: while_body,
                });

                let enter_op = if *inclusive {
                    MirBinOp::Le
                } else {
                    MirBinOp::Lt
                };
                out.push(MirStmt::If {
                    cond: MirExpr::new(
                        MirExprKind::Binary {
                            op: enter_op,
                            lhs: Box::new(MirExpr::new(MirExprKind::Var(s_tmp), vty.clone())),
                            rhs: Box::new(MirExpr::new(MirExprKind::Var(e_tmp), vty)),
                        },
                        MirType::Bool,
                    ),
                    then_body: guarded,
                    else_body: Vec::new(),
                });
            }
            StmtKind::ForEach {
                elem_ty,
                elem_name,
                iter,
                body,
            } => {
                let ety = self.syms.resolve_type(elem_ty);
                let (slice_tmp, seq_elem_ty) = self.lower_iter_into_slice_temp(iter, &ety, &mut out)?;
                let len_tmp = self.alloc_temp("for_len", MirType::Usize);
                let slice_ty = MirType::Slice(Box::new(seq_elem_ty.clone()));
                out.push(MirStmt::Assign {
                    target: MirPlace::Var(len_tmp),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(
                        MirExprKind::SliceLen(Box::new(MirExpr::new(
                            MirExprKind::Var(slice_tmp),
                            slice_ty.clone(),
                        ))),
                        MirType::Usize,
                    ),
                });

                self.scopes.push();
                let elem_var = self.declare_var(&elem_name.name, ety.clone(), VarRole::UserLocal);
                let idx_tmp = self.alloc_temp("for_idx", MirType::Usize);
                let first_tmp = self.alloc_temp("for_first", MirType::Bool);

                let mut guarded = vec![
                    MirStmt::Assign {
                        target: MirPlace::Var(idx_tmp),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::IntLit(0), MirType::Usize),
                    },
                    MirStmt::Assign {
                        target: MirPlace::Var(first_tmp),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::BoolLit(true), MirType::Bool),
                    },
                ];

                let raw_elem_expr = MirExpr::new(
                    MirExprKind::Index {
                        base: Box::new(MirExpr::new(MirExprKind::Var(slice_tmp), slice_ty)),
                        index: Box::new(MirExpr::new(MirExprKind::Var(idx_tmp), MirType::Usize)),
                        is_slice: true,
                    },
                    seq_elem_ty.host_value_type(),
                );

                let mut while_body = vec![
                    MirStmt::If {
                        cond: MirExpr::new(MirExprKind::Var(first_tmp), MirType::Bool),
                        then_body: vec![MirStmt::Assign {
                            target: MirPlace::Var(first_tmp),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(MirExprKind::BoolLit(false), MirType::Bool),
                        }],
                        else_body: vec![
                            MirStmt::Assign {
                                target: MirPlace::Var(idx_tmp),
                                op: MirAssignOp::AddAssign,
                                value: MirExpr::new(MirExprKind::IntLit(1), MirType::Usize),
                            },
                            MirStmt::If {
                                cond: MirExpr::new(
                                    MirExprKind::Binary {
                                        op: MirBinOp::Ge,
                                        lhs: Box::new(MirExpr::new(
                                            MirExprKind::Var(idx_tmp),
                                            MirType::Usize,
                                        )),
                                        rhs: Box::new(MirExpr::new(
                                            MirExprKind::Var(len_tmp),
                                            MirType::Usize,
                                        )),
                                    },
                                    MirType::Bool,
                                ),
                                then_body: vec![MirStmt::Break],
                                else_body: Vec::new(),
                            },
                        ],
                    },
                    MirStmt::Assign {
                        target: MirPlace::Var(elem_var),
                        op: MirAssignOp::Assign,
                        value: coerce_or_cast_int(raw_elem_expr, &ety),
                    },
                ];

                let loop_depth = self.defer_stack.len();
                self.loop_defer_depths.push(loop_depth);
                if let Some(bs) = &mut self.boot_state {
                    bs.loop_depth += 1;
                }
                let user_body = self.lower_block_stmts(&body.stmts)?;
                if let Some(bs) = &mut self.boot_state {
                    bs.loop_depth -= 1;
                }
                self.loop_defer_depths.pop();
                while_body.extend(user_body);

                self.scopes.pop();

                guarded.push(MirStmt::While {
                    cond: MirExpr::new(MirExprKind::BoolLit(true), MirType::Bool),
                    body: while_body,
                });

                out.push(MirStmt::If {
                    cond: MirExpr::new(
                        MirExprKind::Binary {
                            op: MirBinOp::Gt,
                            lhs: Box::new(MirExpr::new(MirExprKind::Var(len_tmp), MirType::Usize)),
                            rhs: Box::new(MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)),
                        },
                        MirType::Bool,
                    ),
                    then_body: guarded,
                    else_body: Vec::new(),
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
                let bid = self.boot_state.as_mut().map(|bs| {
                    let id = bs.next_branch_id;
                    bs.next_branch_id += 1;
                    bs.branch_depth += 1;
                    id
                });
                let mut mir_arms = Vec::new();
                let mut default_body = Vec::new();
                for (arm_idx, arm) in arms.iter().enumerate() {
                    if let (Some(bs), Some(id)) = (&mut self.boot_state, bid) {
                        bs.branch_stack.push((id, arm_idx));
                    }
                    let body = self.lower_block_stmts(&arm.body.stmts)?;
                    if let (Some(bs), Some(_)) = (&mut self.boot_state, bid) {
                        bs.branch_stack.pop();
                    }
                    if let Some(pat) = convert_switch_pat(&arm.pat, &d.ty, self.syms) {
                        mir_arms.push((pat, body));
                    } else {
                        default_body = body;
                    }
                }
                if let Some(bs) = &mut self.boot_state {
                    bs.branch_depth -= 1;
                }
                out.push(MirStmt::Switch {
                    discr: d,
                    arms: mir_arms,
                    default: default_body,
                });
            }
            StmtKind::Continue => {
                if let Some(&loop_depth) = self.loop_defer_depths.last() {
                    out.extend(self.emit_defers_down_to(loop_depth)?);
                }
                out.push(MirStmt::Continue);
            }
            StmtKind::Break => {
                if let Some(&loop_depth) = self.loop_defer_depths.last() {
                    out.extend(self.emit_defers_down_to(loop_depth)?);
                }
                out.push(MirStmt::Break);
            }
            StmtKind::Block(b) => {
                out.extend(self.lower_block_stmts(&b.stmts)?);
            }
            StmtKind::Checkpoint => {
                if !self.in_noyield {
                    return Err(MirLowerError {
                        message: "`checkpoint()` encountered inside non-yielding context"
                            .to_string(),
                        span: stmt.span,
                    });
                }
                // Inside a `__noyield` SoftYielding clone, `checkpoint()` strips to a zero-cost no-op!
            }
            StmtKind::Spawn(sp) => {
                let activate = self.lower_boot_spawn_stmt(sp, &mut out, stmt.span)?;
                out.push(activate);
            }
            StmtKind::Expr(e) => {
                if self.in_noyield && is_checkpoint_call_expr(e) {
                    return Ok(out);
                }
                if let Some(code_arg_opt) = extract_sys_exit_arg(e) {
                    let code_expr = self.lower_sys_exit_code_before_defers(code_arg_opt, &mut out)?;
                    out.extend(self.emit_defers_down_to(0)?);
                    out.push(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                        method: "exit".to_string(),
                        args: vec![code_expr],
                    }));
                    return Ok(out);
                }
                if self.try_lower_irq_attach_stmt(e, &mut out)? {
                    return Ok(out);
                }
                if let Some(stmt_op) = self.try_lower_stmt_call(e, &mut out)? {
                    out.push(stmt_op);
                } else {
                    let me = self.lower_expr_with_stmts(e, None, &mut out)?;
                    if !me.ty.is_void() || matches!(me.kind, MirExprKind::CallDirect { .. }) {
                        out.push(MirStmt::Expr(me));
                    }
                }
            }
            StmtKind::LoopEvery { .. } | StmtKind::Restart => {
                return Err(MirLowerError {
                    message: "yielding `loop every` or `restart` encountered inside non-yielding context"
                        .to_string(),
                    span: stmt.span,
                });
            }
        }
        Ok(out)
    }

    fn lower_sys_exit_code_before_defers(
        &mut self,
        code_arg_opt: Option<&Expr>,
        out: &mut Vec<MirStmt>,
    ) -> Result<MirExpr, MirLowerError> {
        if let Some(ce) = code_arg_opt {
            let raw = self.lower_expr_with_stmts(ce, Some(&MirType::I32), out)?;
            let coerced = coerce_or_cast_int(raw, &MirType::I32);
            if self.has_active_defers_from(0) {
                let tmp = self.alloc_temp("exit_code", MirType::I32);
                out.push(MirStmt::Assign {
                    target: MirPlace::Var(tmp),
                    op: MirAssignOp::Assign,
                    value: coerced,
                });
                Ok(MirExpr::new(MirExprKind::Var(tmp), MirType::I32))
            } else {
                Ok(coerced)
            }
        } else {
            Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::I32))
        }
    }

    fn lower_iter_into_slice_temp(
        &mut self,
        iter: &Expr,
        fallback_elem_ty: &MirType,
        out: &mut Vec<MirStmt>,
    ) -> Result<(VarId, MirType), MirLowerError> {
        if let Ok((place, MirType::Array { elem, len })) = self.lower_place(iter, out) {
            let slice_ty = MirType::Slice(elem.clone());
            let slice_tmp = self.alloc_temp("for_slice", slice_ty.clone());
            out.push(MirStmt::Assign {
                target: MirPlace::Var(slice_tmp),
                op: MirAssignOp::Assign,
                value: MirExpr::new(
                    MirExprKind::ArrayToSlice {
                        array_place: Box::new(place),
                        len,
                        elem_ty: *elem.clone(),
                    },
                    slice_ty,
                ),
            });
            return Ok((slice_tmp, *elem));
        }

        let iv = unwrap_ok_result_expr(self.lower_expr_with_stmts(iter, None, out)?);
        match iv.ty.clone() {
            MirType::Slice(elem) => {
                let et = *elem.clone();
                let slice_tmp = self.alloc_temp("for_slice", iv.ty.clone());
                out.push(MirStmt::Assign {
                    target: MirPlace::Var(slice_tmp),
                    op: MirAssignOp::Assign,
                    value: iv,
                });
                Ok((slice_tmp, et))
            }
            MirType::Ptr(inner) => match inner.as_ref() {
                MirType::Slice(elem) => {
                    let et = *elem.clone();
                    let slice_ty = MirType::Slice(elem.clone());
                    let slice_tmp = self.alloc_temp("for_slice", slice_ty.clone());
                    out.push(MirStmt::Assign {
                        target: MirPlace::Var(slice_tmp),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(
                            MirExprKind::Unary {
                                op: MirUnaryOp::Deref,
                                expr: Box::new(iv),
                            },
                            slice_ty,
                        ),
                    });
                    Ok((slice_tmp, et))
                }
                MirType::Array { elem, len } => {
                    let et = *elem.clone();
                    let len_val = *len;
                    let ptr_tmp = self.alloc_temp("for_arr_ptr", iv.ty.clone());
                    out.push(MirStmt::Assign {
                        target: MirPlace::Var(ptr_tmp),
                        op: MirAssignOp::Assign,
                        value: iv.clone(),
                    });
                    let arr_ty = MirType::Array {
                        elem: elem.clone(),
                        len: len_val,
                    };
                    let slice_ty = MirType::Slice(elem.clone());
                    let slice_tmp = self.alloc_temp("for_slice", slice_ty.clone());
                    out.push(MirStmt::Assign {
                        target: MirPlace::Var(slice_tmp),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(
                            MirExprKind::ArrayToSlice {
                                array_place: Box::new(MirPlace::Deref(
                                    Box::new(MirExpr::new(MirExprKind::Var(ptr_tmp), iv.ty)),
                                    arr_ty,
                                )),
                                len: len_val,
                                elem_ty: et.clone(),
                            },
                            slice_ty,
                        ),
                    });
                    Ok((slice_tmp, et))
                }
                _ => {
                    let slice_ty = MirType::Slice(Box::new(fallback_elem_ty.clone()));
                    let slice_tmp = self.alloc_temp("for_slice", slice_ty);
                    Ok((slice_tmp, fallback_elem_ty.clone()))
                }
            },
            MirType::Array { elem, len } => {
                let et = *elem.clone();
                let len_val = len;
                let arr_tmp = self.alloc_temp("for_arr", iv.ty.clone());
                out.push(MirStmt::Assign {
                    target: MirPlace::Var(arr_tmp),
                    op: MirAssignOp::Assign,
                    value: iv,
                });
                let slice_ty = MirType::Slice(elem.clone());
                let slice_tmp = self.alloc_temp("for_slice", slice_ty.clone());
                out.push(MirStmt::Assign {
                    target: MirPlace::Var(slice_tmp),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(
                        MirExprKind::ArrayToSlice {
                            array_place: Box::new(MirPlace::Var(arr_tmp)),
                            len: len_val,
                            elem_ty: et.clone(),
                        },
                        slice_ty,
                    ),
                });
                Ok((slice_tmp, et))
            }
            _ => {
                let slice_ty = MirType::Slice(Box::new(fallback_elem_ty.clone()));
                let slice_tmp = self.alloc_temp("for_slice", slice_ty);
                Ok((slice_tmp, fallback_elem_ty.clone()))
            }
        }
    }

    fn lower_boot_spawn_stmt(
        &mut self,
        sp: &SpawnStmt,
        out: &mut Vec<MirStmt>,
        span: Span,
    ) -> Result<MirStmt, MirLowerError> {
        if self.boot_state.is_none() {
            return Err(MirLowerError {
                message: "`spawn` encountered outside `fn main()` boot sequence".to_string(),
                span,
            });
        }
        let bp_name = if let Some(mod_id) = &sp.module {
            self.syms
                .candidate_module_names(&mod_id.name, &sp.task.name)
                .into_iter()
                .find(|c| self.syms.tasks.contains_key(c))
                .unwrap_or_else(|| format!("{}.{}", mod_id.name, sp.task.name))
        } else {
            sp.task.name.clone()
        };
        let param_tys = self.syms.tasks.get(&bp_name).cloned().unwrap_or_default();
        let mut lowered_args = Vec::new();
        for (idx, arg) in sp.args.iter().enumerate() {
            let expected = param_tys.get(idx);
            let val = self.lower_expr_with_stmts(&arg.value, expected, out)?;
            let coerced = if let Some(ety) = expected {
                coerce_expr_if_needed(val, ety)
            } else {
                val
            };
            lowered_args.push(coerced);
        }

        let bs = self.boot_state.as_mut().unwrap();
        let count_entry = bs.bp_instance_counts.entry(bp_name.clone()).or_insert(0);
        let inst_num = *count_entry;
        *count_entry += 1;

        let slot_id = bs.spawned_tasks.len();
        let instance_name = format!("{bp_name}_{inst_num}");
        let prio = sp.priority.map_or(MirPriority::Normal, convert_priority);
        let core_pin = sp.core.map(|c| c as u8);
        let mut provenance_path = bs.provenance_stack.clone();
        provenance_path.push(bp_name.clone());
        let conditional = bs.branch_depth > 0;

        bs.spawned_tasks.push(MirSpawnInstance {
            slot_id,
            instance_name,
            blueprint_name: bp_name,
            priority: prio,
            core_pin,
            init_args: lowered_args.clone(),
            is_main_supervisor: false,
            provenance_path,
            conditional,
        });

        Ok(MirStmt::SpawnActivate {
            slot_id,
            args: lowered_args,
        })
    }

    fn try_unroll_boot_helper_call(
        &mut self,
        callee_name: &str,
        args: &[CallArg],
        out: &mut Vec<MirStmt>,
    ) -> Result<Option<MirExpr>, MirLowerError> {
        let Some(sig) = self.syms.fns.get(callee_name).cloned() else {
            return Ok(None);
        };
        if !sig.spawns_in_boot {
            return Ok(None);
        }
        let Some(fd) = self
            .boot_state
            .as_ref()
            .and_then(|bs| bs.fn_decls.get(callee_name).copied())
        else {
            return Ok(None);
        };
        let Some(body_ast) = &fd.body else {
            return Ok(None);
        };

        // 1. Evaluate call arguments in the CALLER's lexical scope first!
        let mut lowered_args = Vec::new();
        let mut irq_arg_bindings: Vec<(String, (String, u32))> = Vec::new();
        for (idx, a) in args.iter().enumerate() {
            if let Some(p) = fd.params.get(idx) {
                if let Some((iname, inum)) = self.try_resolve_irq_token(&a.value)? {
                    irq_arg_bindings.push((p.name.name.clone(), (iname, inum)));
                    lowered_args.push(MirExpr::new(MirExprKind::IntLit(inum as i128), MirType::U32));
                    continue;
                }
            }
            let expected = sig.params.get(idx);
            let me = self.lower_expr_with_stmts(&a.value, expected, out)?;
            let coerced = match (&me.ty, expected, &me.kind) {
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
            lowered_args.push(coerced);
        }

        // 2. Prepare optional structured return flag & value slot if helper contains `return`.
        let has_return = ast_stmts_have_return(&body_ast.stmts);
        let ret_flag_var = if has_return {
            let rf = self.alloc_temp("boot_ret_flag", MirType::Bool);
            out.push(MirStmt::Assign {
                target: MirPlace::Var(rf),
                op: MirAssignOp::Assign,
                value: MirExpr::new(MirExprKind::BoolLit(false), MirType::Bool),
            });
            Some(rf)
        } else {
            None
        };
        let ret_val_var = if !sig.ret_ty.is_void() {
            Some(self.alloc_temp("boot_ret_val", sig.ret_ty.clone()))
        } else {
            None
        };

        // 3. Push helper scope, bind parameters, push provenance, and unroll helper body!
        self.scopes.push();
        self.defer_stack.push(Vec::new());
        let entry_defer_depth = self.defer_stack.len() - 1;

        for (idx, p) in fd.params.iter().enumerate() {
            let pty = self.syms.resolve_type(&p.ty);
            let pvid = self.declare_var(&p.name.name, pty.clone(), VarRole::CompilerTemp);
            self.scopes.insert(p.name.name.clone(), pvid);
            if let Some(arg_val) = lowered_args.get(idx).cloned() {
                let resolved_arg = resolve_boot_const_expr(&arg_val, &self.boot_const_env);
                if is_const_mir_expr(&resolved_arg) {
                    self.boot_const_env.insert(pvid, resolved_arg);
                }
                out.push(MirStmt::Assign {
                    target: MirPlace::Var(pvid),
                    op: MirAssignOp::Assign,
                    value: arg_val,
                });
            }
        }

        let prev_irq_bindings = if let Some(bs) = &mut self.boot_state {
            bs.provenance_stack.push(callee_name.to_string());
            Some(std::mem::replace(
                &mut bs.irq_param_bindings,
                irq_arg_bindings.into_iter().collect(),
            ))
        } else {
            None
        };
        self.inlined_helper_stack.push(InlinedHelperFrame {
            ret_ty: sig.ret_ty.clone(),
            ret_val_var,
            ret_flag_var,
            entry_defer_depth,
        });

        let helper_res = self.lower_stmt_seq(&body_ast.stmts);

        self.inlined_helper_stack.pop();
        if let Some(bs) = &mut self.boot_state {
            bs.provenance_stack.pop();
            if let Some(prev) = prev_irq_bindings {
                bs.irq_param_bindings = prev;
            }
        }
        out.extend(helper_res?);

        let exit_defers = self.pop_and_lower_top_defer_scope()?;
        if !exit_defers.is_empty() {
            if let Some(rf) = ret_flag_var {
                out.push(MirStmt::If {
                    cond: MirExpr::new(
                        MirExprKind::Unary {
                            op: MirUnaryOp::Not,
                            expr: Box::new(MirExpr::new(MirExprKind::Var(rf), MirType::Bool)),
                        },
                        MirType::Bool,
                    ),
                    then_body: exit_defers,
                    else_body: Vec::new(),
                });
            } else {
                out.extend(exit_defers);
            }
        }
        self.scopes.pop();

        if let Some(rv) = ret_val_var {
            Ok(Some(MirExpr::new(MirExprKind::Var(rv), sig.ret_ty)))
        } else {
            Ok(Some(MirExpr::new(MirExprKind::IntLit(0), MirType::Void)))
        }
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
            ExprKind::Or { .. } => {
                let mut or_chain = Vec::new();
                flatten_or_operands(expr, &mut or_chain);
                let first_res = self.lower_failable_expr(or_chain[0], stmts)?;
                let (ok_ty, first_res_ty) = extract_result_types(&first_res.ty);
                let first_vid = self.alloc_temp("or_res", first_res_ty);
                stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(first_vid),
                    op: MirAssignOp::Assign,
                    value: first_res,
                });
                let out_vid_opt = if ok_ty.is_void() {
                    None
                } else {
                    Some(self.alloc_temp("or_val", ok_ty.clone()))
                };
                self.lower_or_tail_direct(&or_chain[1..], first_vid, out_vid_opt, &ok_ty, stmts)?;
                if let Some(out_vid) = out_vid_opt {
                    Ok(MirExpr::new(MirExprKind::Var(out_vid), ok_ty))
                } else {
                    Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void))
                }
            }
            ExprKind::Catch {
                expr: inner,
                clause,
            } => {
                let mut or_chain = Vec::new();
                flatten_or_operands(inner, &mut or_chain);
                self.lower_catch_or_chain_direct(&or_chain, None, clause, stmts)
            }
            ExprKind::Call { callee, args, .. } if self.boot_state.is_some() => {
                if let ExprKind::Ident(id) = &callee.kind {
                    if let Some(unrolled) = self.try_unroll_boot_helper_call(&id.name, args, stmts)?
                    {
                        return Ok(unrolled);
                    }
                }
                let _ = &self.pending_stmts;
                let mut be = DirectBackend {
                    ctx: self,
                    stmts,
                };
                lower_common_expr(expr, expected, &mut be)
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

    fn lower_or_tail_direct(
        &mut self,
        rem: &[&Expr],
        prev_res_vid: VarId,
        out_vid_opt: Option<VarId>,
        ok_ty: &MirType,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<(), MirLowerError> {
        let mut fb_stmts = Vec::new();
        if rem.len() == 1 {
            let fb_val = self.lower_expr_with_stmts(rem[0], Some(ok_ty), &mut fb_stmts)?;
            if let Some(out_vid) = out_vid_opt {
                fb_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(out_vid),
                    op: MirAssignOp::Assign,
                    value: coerce_expr_if_needed(fb_val, ok_ty),
                });
            } else if !fb_val.ty.is_void() || matches!(fb_val.kind, MirExprKind::CallDirect { .. }) {
                fb_stmts.push(MirStmt::Expr(fb_val));
            }
        } else {
            let next_res = self.lower_failable_expr(rem[0], &mut fb_stmts)?;
            let (_, next_res_ty) = extract_result_types(&next_res.ty);
            let next_vid = self.alloc_temp("or_res", next_res_ty);
            fb_stmts.push(MirStmt::Assign {
                target: MirPlace::Var(next_vid),
                op: MirAssignOp::Assign,
                value: next_res,
            });
            self.lower_or_tail_direct(&rem[1..], next_vid, out_vid_opt, ok_ty, &mut fb_stmts)?;
        }

        let ok_stmts = if let Some(out_vid) = out_vid_opt {
            vec![MirStmt::Assign {
                target: MirPlace::Var(out_vid),
                op: MirAssignOp::Assign,
                value: coerce_expr_if_needed(
                    MirExpr::new(MirExprKind::ResultVal(prev_res_vid), ok_ty.clone()),
                    ok_ty,
                ),
            }]
        } else {
            Vec::new()
        };

        stmts.push(MirStmt::If {
            cond: MirExpr::new(MirExprKind::ResultIsErr(prev_res_vid), MirType::Bool),
            then_body: fb_stmts,
            else_body: ok_stmts,
        });
        Ok(())
    }

    fn lower_catch_or_chain_direct(
        &mut self,
        chain: &[&Expr],
        shared_out_vid: Option<(Option<VarId>, MirType)>,
        clause: &CatchClause,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<MirExpr, MirLowerError> {
        if chain.len() > 1 {
            let head_res = self.lower_failable_expr(chain[0], stmts)?;
            let (head_ok_ty, head_res_ty) = extract_result_types(&head_res.ty);
            let head_vid = self.alloc_temp("or_catch_res", head_res_ty);
            stmts.push(MirStmt::Assign {
                target: MirPlace::Var(head_vid),
                op: MirAssignOp::Assign,
                value: head_res,
            });
            let (out_vid_opt, ok_ty) = match shared_out_vid {
                Some(pair) => pair,
                None => {
                    let ov = if head_ok_ty.is_void() {
                        None
                    } else {
                        Some(self.alloc_temp("catch_val", head_ok_ty.clone()))
                    };
                    (ov, head_ok_ty)
                }
            };
            let mut next_stmts = Vec::new();
            let _ = self.lower_catch_or_chain_direct(
                &chain[1..],
                Some((out_vid_opt, ok_ty.clone())),
                clause,
                &mut next_stmts,
            )?;
            let ok_stmts = if let Some(out_vid) = out_vid_opt {
                vec![MirStmt::Assign {
                    target: MirPlace::Var(out_vid),
                    op: MirAssignOp::Assign,
                    value: coerce_expr_if_needed(
                        MirExpr::new(MirExprKind::ResultVal(head_vid), ok_ty.clone()),
                        &ok_ty,
                    ),
                }]
            } else {
                Vec::new()
            };
            stmts.push(MirStmt::If {
                cond: MirExpr::new(MirExprKind::ResultIsErr(head_vid), MirType::Bool),
                then_body: next_stmts,
                else_body: ok_stmts,
            });
            return if let Some(out_vid) = out_vid_opt {
                Ok(MirExpr::new(MirExprKind::Var(out_vid), ok_ty))
            } else {
                Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void))
            };
        }

        let res_expr = self.lower_failable_expr(chain[0], stmts)?;
        let (tail_ok_ty, res_ty) = extract_result_types(&res_expr.ty);
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

        let (out_vid_opt, ok_ty) = match shared_out_vid {
            Some(pair) => pair,
            None => {
                let ov = if tail_ok_ty.is_void() {
                    None
                } else {
                    Some(self.alloc_temp("catch_val", tail_ok_ty.clone()))
                };
                (ov, tail_ok_ty)
            }
        };

        self.scopes.push();
        self.defer_stack.push(Vec::new());
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
                if extract_sys_exit_arg(tail_expr).is_some() {
                    err_stmts.extend(self.lower_stmt(last_stmt)?);
                } else if let Some(stmt_op) =
                    self.try_lower_stmt_call(tail_expr, &mut err_stmts)?
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
        err_stmts.extend(self.pop_and_lower_top_defer_scope()?);
        self.scopes.pop();

        if let Some(out_vid) = out_vid_opt {
            let ok_stmts = vec![MirStmt::Assign {
                target: MirPlace::Var(out_vid),
                op: MirAssignOp::Assign,
                value: coerce_expr_if_needed(
                    MirExpr::new(MirExprKind::ResultVal(res_vid), ok_ty.clone()),
                    &ok_ty,
                ),
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

    fn try_resolve_irq_token(&self, expr: &Expr) -> Result<Option<(String, u32)>, MirLowerError> {
        match &expr.kind {
            ExprKind::Ident(id) => {
                if let Some(bound) = self
                    .boot_state
                    .as_ref()
                    .and_then(|bs| bs.irq_param_bindings.get(&id.name))
                {
                    return Ok(Some(bound.clone()));
                }
                if self.scopes.lookup(&id.name).is_none() {
                    if let Some(&num) = self.syms.irqs.get(&id.name) {
                        return Ok(Some((id.name.clone(), num)));
                    }
                }
                Ok(None)
            }
            ExprKind::Field { .. } => Ok(self.syms.resolve_irq_field_be(expr, &self.scopes)),
            _ => Ok(None),
        }
    }

    fn lower_baked_isr_arg(
        &mut self,
        arg_expr: &Expr,
        param_ty: &MirType,
    ) -> Result<MirBakedArg, MirLowerError> {
        if let Some((_, num)) = self.try_resolve_irq_token(arg_expr)? {
            return Ok(MirBakedArg::Const(MirExpr::new(
                MirExprKind::IntLit(num as i128),
                MirType::U32,
            )));
        }
        let mut arg_stmts = Vec::new();
        let raw = self.lower_expr_with_stmts(arg_expr, Some(param_ty), &mut arg_stmts)?;
        if !arg_stmts.is_empty() {
            return Err(MirLowerError {
                message: "ISR attachment argument must be statically known (no runtime side effects)"
                    .to_string(),
                span: arg_expr.span,
            });
        }
        let resolved = resolve_boot_const_expr(&raw, &self.boot_const_env);
        let coerced = coerce_expr_if_needed(resolved, param_ty);
        match &coerced.kind {
            MirExprKind::GlobalPrimAddr(name) => Ok(MirBakedArg::PrimAddr(name.clone())),
            MirExprKind::MmioAddr(name) => Ok(MirBakedArg::MmioAddr(name.clone())),
            _ if is_const_mir_expr(&coerced) => Ok(MirBakedArg::Const(coerced)),
            _ => Err(MirLowerError {
                message: "ISR attachment argument must be a compile-time constant, static `&primitive`, or `mmio` peripheral pointer"
                    .to_string(),
                span: arg_expr.span,
            }),
        }
    }

    fn try_lower_irq_attach_stmt(
        &mut self,
        expr: &Expr,
        out: &mut Vec<MirStmt>,
    ) -> Result<bool, MirLowerError> {
        let ExprKind::MethodCall {
            receiver,
            method,
            args,
        } = &expr.kind
        else {
            return Ok(false);
        };
        if method.name != "attach" && method.name != "attach_shared" {
            return Ok(false);
        }
        let is_shared = method.name == "attach_shared";

        let Some((irq_name, irq_num)) = self.try_resolve_irq_token(receiver)? else {
            return Ok(false);
        };

        if self.boot_state.is_none() {
            return Err(MirLowerError {
                message: format!(
                    "IRQ `.{}` must be called inside `fn main()` boot sequence (or a boot helper called from `main()`)",
                    method.name
                ),
                span: expr.span,
            });
        }

        if args.len() != 1 {
            return Err(MirLowerError {
                message: format!(
                    "IRQ `.{}` expects 1 ISR attachment argument, found {}",
                    method.name,
                    args.len()
                ),
                span: expr.span,
            });
        }

        let (handler_name, call_args): (String, &[CallArg]) = match &args[0].value.kind {
            ExprKind::Call {
                callee,
                args: cargs,
                ..
            } => {
                let hname = match &callee.kind {
                    ExprKind::Ident(id) => id.name.clone(),
                    ExprKind::Field { target, field } => {
                        if let ExprKind::Ident(alias) = &target.kind {
                            self.syms
                                .candidate_module_names(&alias.name, &field.name)
                                .into_iter()
                                .find(|c| self.syms.isrs.contains_key(c))
                                .unwrap_or_else(|| format!("{}.{}", alias.name, field.name))
                        } else {
                            return Err(MirLowerError {
                                message: format!("invalid ISR callee in IRQ `.{}`", method.name),
                                span: callee.span,
                            });
                        }
                    }
                    _ => {
                        return Err(MirLowerError {
                            message: format!("invalid ISR callee in IRQ `.{}`", method.name),
                            span: callee.span,
                        });
                    }
                };
                (hname, cargs.as_slice())
            }
            ExprKind::Ident(id) if self.syms.isrs.contains_key(&id.name) => {
                (id.name.clone(), &[][..])
            }
            _ => {
                return Err(MirLowerError {
                    message: format!(
                        "IRQ `.{}` argument must be an `isr` handler invocation like `my_isr(...)`",
                        method.name
                    ),
                    span: args[0].value.span,
                });
            }
        };

        let isr_param_tys = self
            .syms
            .isrs
            .get(&handler_name)
            .cloned()
            .ok_or_else(|| MirLowerError {
                message: format!(
                    "unknown ISR handler `{handler_name}` in IRQ `.{}`",
                    method.name
                ),
                span: args[0].value.span,
            })?;

        if call_args.len() != isr_param_tys.len() {
            return Err(MirLowerError {
                message: format!(
                    "ISR `{handler_name}` expects {} argument(s), got {}",
                    isr_param_tys.len(),
                    call_args.len()
                ),
                span: args[0].value.span,
            });
        }

        let mut baked_args = Vec::with_capacity(call_args.len());
        for (idx, carg) in call_args.iter().enumerate() {
            let pty = &isr_param_tys[idx];
            baked_args.push(self.lower_baked_isr_arg(&carg.value, pty)?);
        }

        let bs = self.boot_state.as_mut().unwrap();
        if let Some(sel_stmt) =
            bs.attach_irq(&irq_name, irq_num, is_shared, &handler_name, baked_args, expr.span)?
        {
            out.push(sel_stmt);
        }
        Ok(true)
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
    defer_depth: usize,
}

struct CoroutineLowerCtx<'a> {
    syms: &'a ModuleSymbols,
    ret_ty: MirType,
    error_spec: MirErrorSpec,
    manual_yield: bool,
    in_noyield: bool,
    has_restart: bool,
    vars: Vec<MirVar>,
    scopes: VarScopes,
    defer_stack: Vec<Vec<Stmt>>,
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
            has_restart: false,
            vars: Vec::new(),
            scopes: VarScopes::new(),
            defer_stack: vec![Vec::new()],
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

    fn has_active_defers_from(&self, min_depth: usize) -> bool {
        self.defer_stack
            .iter()
            .skip(min_depth)
            .any(|frame| !frame.is_empty())
    }

    fn lower_single_deferred_stmt(&mut self, deferred: &Stmt) -> Result<(), MirLowerError> {
        let prev_ny = self.in_noyield;
        self.in_noyield = true;
        self.scopes.push();
        self.defer_stack.push(Vec::new());
        let res = self.lower_stmt(deferred);
        self.defer_stack.pop();
        self.scopes.pop();
        self.in_noyield = prev_ny;
        res
    }

    fn pop_and_lower_top_defer_scope(&mut self) -> Result<(), MirLowerError> {
        let Some(defers) = self.defer_stack.pop() else {
            return Ok(());
        };
        for d in defers.iter().rev() {
            self.lower_single_deferred_stmt(d)?;
        }
        Ok(())
    }

    /// Emit active `defer` statements from innermost scope down to `min_depth` (inclusive) in LIFO order
    /// without mutating `self.defer_stack`.
    fn emit_defers_down_to(&mut self, min_depth: usize) -> Result<(), MirLowerError> {
        let top = self.defer_stack.len();
        if min_depth >= top {
            return Ok(());
        }
        let mut popped_var_frames: Vec<BTreeMap<String, VarId>> = Vec::new();
        for depth in (min_depth..top).rev() {
            while self.scopes.frames.len() > depth + 1 {
                if let Some(vf) = self.scopes.frames.pop() {
                    popped_var_frames.push(vf);
                } else {
                    break;
                }
            }
            let cloned_defers = self.defer_stack[depth].clone();
            for d in cloned_defers.iter().rev() {
                self.lower_single_deferred_stmt(d)?;
            }
        }
        while let Some(vf) = popped_var_frames.pop() {
            self.scopes.frames.push(vf);
        }
        Ok(())
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
        if self.has_restart {
            if let Some(b0) = self.blocks.first_mut() {
                b0.is_resume_entry = true;
            }
        }
    }

    fn lower_top_level_stmts(&mut self, stmts: &[Stmt]) -> Result<(), MirLowerError> {
        self.lower_stmts(stmts)?;
        self.pop_and_lower_top_defer_scope()?;
        Ok(())
    }

    fn lower_scoped_block(&mut self, block: &Block) -> Result<(), MirLowerError> {
        self.scopes.push();
        self.defer_stack.push(Vec::new());
        self.lower_stmts(&block.stmts)?;
        self.pop_and_lower_top_defer_scope()?;
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
            StmtKind::StateVar { name, .. } => {
                return Err(MirLowerError {
                    message: format!(
                        "`state` variable `{}` may only be declared at the top level of an `isr` handler",
                        name.name
                    ),
                    span: stmt.span,
                });
            }
            StmtKind::Assign { target, op, value } => {
                if try_lower_mmio_assign_be(target, *op, value, self)? {
                    return Ok(());
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
            StmtKind::Defer(inner) => {
                if let Some(top) = self.defer_stack.last_mut() {
                    top.push((**inner).clone());
                }
            }
            StmtKind::Return(expr_opt) => {
                let ret_ty = self.ret_ty.clone();
                let has_defers = self.has_active_defers_from(0);
                let val = if let Some(e) = expr_opt {
                    let raw = self.lower_expr(e, Some(&ret_ty))?;
                    let coerced = coerce_expr_if_needed(raw, &ret_ty);
                    if has_defers && !ret_ty.is_void() {
                        let tmp = self.alloc_temp("ret_val", ret_ty.clone());
                        self.cur_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(tmp),
                            op: MirAssignOp::Assign,
                            value: coerced,
                        });
                        Some(MirExpr::new(MirExprKind::Var(tmp), ret_ty))
                    } else {
                        Some(coerced)
                    }
                } else {
                    None
                };
                self.emit_defers_down_to(0)?;
                let dead_bb = self.alloc_bb();
                self.seal_bb(MirTerminator::Return(val), dead_bb, false);
            }
            StmtKind::Error { reason, partial } => {
                let ret_ty = self.ret_ty.clone();
                let err_ty = self.error_spec.err_type();
                let has_defers = self.has_active_defers_from(0);
                let r = if let Some(re) = reason {
                    let raw = self.lower_expr(re, err_ty.as_ref())?;
                    if let (true, Some(ety)) = (has_defers, err_ty.clone()) {
                        let tmp = self.alloc_temp("err_reason", ety.clone());
                        self.cur_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(tmp),
                            op: MirAssignOp::Assign,
                            value: coerce_expr_if_needed(raw, &ety),
                        });
                        Some(MirExpr::new(MirExprKind::Var(tmp), ety))
                    } else {
                        Some(raw)
                    }
                } else {
                    None
                };
                let p = if let Some(pe) = partial {
                    let raw = self.lower_expr(pe, Some(&ret_ty))?;
                    if has_defers && !ret_ty.is_void() {
                        let tmp = self.alloc_temp("err_partial", ret_ty.clone());
                        self.cur_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(tmp),
                            op: MirAssignOp::Assign,
                            value: coerce_expr_if_needed(raw, &ret_ty),
                        });
                        Some(MirExpr::new(MirExprKind::Var(tmp), ret_ty))
                    } else {
                        Some(raw)
                    }
                } else {
                    None
                };
                self.emit_defers_down_to(0)?;
                let dead_bb = self.alloc_bb();
                self.seal_bb(
                    MirTerminator::Error {
                        reason: r,
                        partial: p,
                    },
                    dead_bb,
                    false,
                );
            }
            StmtKind::Restart => {
                self.has_restart = true;
                self.emit_defers_down_to(0)?;
                self.cur_stmts.push(MirStmt::EndInboxTurn);
                let dead_bb = self.alloc_bb();
                self.seal_bb(
                    MirTerminator::Yield(MirYieldKind::ShareCheckpoint {
                        resume_bb: BlockId(0),
                    }),
                    dead_bb,
                    false,
                );
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
                let is_while_true = matches!(cond.kind, ExprKind::Literal(Literal::Bool(true)));
                let auto_share = is_while_true && !stmt_manual && !self.in_noyield;

                self.seal_bb(MirTerminator::Goto(cond_bb), cond_bb, auto_share);
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
                let defer_depth = self.defer_stack.len();
                self.loop_stack.push(LoopContext {
                    continue_bb: latch_bb,
                    break_bb: exit_bb,
                    has_inbox_recv: has_inbox,
                    defer_depth,
                });
                self.lower_scoped_block(body)?;
                self.loop_stack.pop();

                self.seal_bb(MirTerminator::Goto(latch_bb), latch_bb, false);
                if has_inbox {
                    self.cur_stmts.push(MirStmt::EndInboxTurn);
                }
                if auto_share {
                    self.seal_bb(
                        MirTerminator::Yield(MirYieldKind::ShareCheckpoint { resume_bb: cond_bb }),
                        exit_bb,
                        false,
                    );
                } else {
                    self.seal_bb(MirTerminator::Goto(cond_bb), exit_bb, false);
                }
            }
            StmtKind::Loop { body } => {
                let top_bb = self.alloc_bb();
                let latch_bb = self.alloc_bb();
                let exit_bb = self.alloc_bb();
                let auto_share = !stmt_manual && !self.in_noyield;
                let has_inbox = ast_stmts_have_inbox_recv(&body.stmts);

                self.seal_bb(MirTerminator::Goto(top_bb), top_bb, auto_share);
                let defer_depth = self.defer_stack.len();
                self.loop_stack.push(LoopContext {
                    continue_bb: latch_bb,
                    break_bb: exit_bb,
                    has_inbox_recv: has_inbox,
                    defer_depth,
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
            StmtKind::LoopEvery { period, body } => {
                let period_expr = if let ExprKind::Literal(Literal::Duration(dur)) = &period.kind {
                    let ticks = self.syms.duration_to_ticks(dur)?;
                    if ticks == 0 {
                        return Err(MirLowerError {
                            message: "periodic `loop every` interval must be greater than 0"
                                .to_string(),
                            span: period.span,
                        });
                    }
                    MirExpr::new(MirExprKind::TicksLit(ticks), MirType::U32)
                } else {
                    let raw_p = self.lower_expr(period, Some(&MirType::U32))?;
                    let p_u32 = coerce_or_cast_int(raw_p, &MirType::U32);
                    if self.syms.tick_period_us != 1_000 && self.syms.tick_period_us > 0 {
                        let p_u64 = coerce_or_cast_int(p_u32, &MirType::U64);
                        let scaled = MirExpr::new(
                            MirExprKind::Binary {
                                op: MirBinOp::Div,
                                lhs: Box::new(MirExpr::new(
                                    MirExprKind::Binary {
                                        op: MirBinOp::Mul,
                                        lhs: Box::new(p_u64),
                                        rhs: Box::new(MirExpr::new(
                                            MirExprKind::IntLit(1_000),
                                            MirType::U64,
                                        )),
                                    },
                                    MirType::U64,
                                )),
                                rhs: Box::new(MirExpr::new(
                                    MirExprKind::IntLit(self.syms.tick_period_us as i128),
                                    MirType::U64,
                                )),
                            },
                            MirType::U64,
                        );
                        coerce_or_cast_int(scaled, &MirType::U32)
                    } else {
                        p_u32
                    }
                };

                let period_var = self.alloc_temp("every_period", MirType::U32);
                let deadline_var = self.alloc_temp("every_next", MirType::U32);
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(period_var),
                    op: MirAssignOp::Assign,
                    value: period_expr,
                });
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
                let defer_depth = self.defer_stack.len();
                self.loop_stack.push(LoopContext {
                    continue_bb: latch_bb,
                    break_bb: exit_bb,
                    has_inbox_recv: has_inbox,
                    defer_depth,
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
                        period_var,
                        resume_bb: body_bb,
                    }),
                    exit_bb,
                    false,
                );
            }
            StmtKind::ForRange {
                var_ty,
                var_name,
                start,
                end,
                inclusive,
                body,
            } => {
                let vty = self.syms.resolve_type(var_ty);
                let s_raw = self.lower_expr(start, Some(&vty))?;
                let e_raw = self.lower_expr(end, Some(&vty))?;
                let s_tmp = self.alloc_temp("for_start", vty.clone());
                let e_tmp = self.alloc_temp("for_end", vty.clone());
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(s_tmp),
                    op: MirAssignOp::Assign,
                    value: coerce_expr_if_needed(s_raw, &vty),
                });
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(e_tmp),
                    op: MirAssignOp::Assign,
                    value: coerce_expr_if_needed(e_raw, &vty),
                });

                self.scopes.push();
                let ivar = self.declare_var(&var_name.name, vty.clone(), VarRole::UserLocal);
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(ivar),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(MirExprKind::Var(s_tmp), vty.clone()),
                });

                let done_tmp_opt = if *inclusive {
                    let dt = self.alloc_temp("for_done", MirType::Bool);
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(dt),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(
                            MirExprKind::Binary {
                                op: MirBinOp::Gt,
                                lhs: Box::new(MirExpr::new(MirExprKind::Var(s_tmp), vty.clone())),
                                rhs: Box::new(MirExpr::new(MirExprKind::Var(e_tmp), vty.clone())),
                            },
                            MirType::Bool,
                        ),
                    });
                    Some(dt)
                } else {
                    None
                };

                let cond_bb = self.alloc_bb();
                let body_bb = self.alloc_bb();
                let step_bb = self.alloc_bb();
                let exit_bb = self.alloc_bb();

                self.seal_bb(MirTerminator::Goto(cond_bb), cond_bb, false);
                let cond_expr = if let Some(dt) = done_tmp_opt {
                    MirExpr::new(
                        MirExprKind::Unary {
                            op: MirUnaryOp::Not,
                            expr: Box::new(MirExpr::new(MirExprKind::Var(dt), MirType::Bool)),
                        },
                        MirType::Bool,
                    )
                } else {
                    MirExpr::new(
                        MirExprKind::Binary {
                            op: MirBinOp::Lt,
                            lhs: Box::new(MirExpr::new(MirExprKind::Var(ivar), vty.clone())),
                            rhs: Box::new(MirExpr::new(MirExprKind::Var(e_tmp), vty.clone())),
                        },
                        MirType::Bool,
                    )
                };
                self.seal_bb(
                    MirTerminator::Branch {
                        cond: cond_expr,
                        then_bb: body_bb,
                        else_bb: exit_bb,
                    },
                    body_bb,
                    false,
                );

                let has_inbox = ast_stmts_have_inbox_recv(&body.stmts);
                let defer_depth = self.defer_stack.len();
                self.loop_stack.push(LoopContext {
                    continue_bb: step_bb,
                    break_bb: exit_bb,
                    has_inbox_recv: has_inbox,
                    defer_depth,
                });
                self.lower_scoped_block(body)?;
                self.loop_stack.pop();
                self.scopes.pop();

                self.seal_bb(MirTerminator::Goto(step_bb), step_bb, false);
                if has_inbox {
                    self.cur_stmts.push(MirStmt::EndInboxTurn);
                }
                if let Some(dt) = done_tmp_opt {
                    // Inclusive `..=`: check `ivar >= e_tmp` BEFORE incrementing so `e_tmp == T::MAX` is overflow-free!
                    self.cur_stmts.push(MirStmt::If {
                        cond: MirExpr::new(
                            MirExprKind::Binary {
                                op: MirBinOp::Ge,
                                lhs: Box::new(MirExpr::new(MirExprKind::Var(ivar), vty.clone())),
                                rhs: Box::new(MirExpr::new(MirExprKind::Var(e_tmp), vty.clone())),
                            },
                            MirType::Bool,
                        ),
                        then_body: vec![MirStmt::Assign {
                            target: MirPlace::Var(dt),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(MirExprKind::BoolLit(true), MirType::Bool),
                        }],
                        else_body: vec![MirStmt::Assign {
                            target: MirPlace::Var(ivar),
                            op: MirAssignOp::AddAssign,
                            value: MirExpr::new(MirExprKind::IntLit(1), vty),
                        }],
                    });
                } else {
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(ivar),
                        op: MirAssignOp::AddAssign,
                        value: MirExpr::new(MirExprKind::IntLit(1), vty),
                    });
                }
                self.seal_bb(MirTerminator::Goto(cond_bb), exit_bb, false);
            }
            StmtKind::ForEach {
                elem_ty,
                elem_name,
                iter,
                body,
            } => {
                let ety = self.syms.resolve_type(elem_ty);
                let (slice_tmp, seq_elem_ty) = self.lower_iter_into_slice_temp(iter, &ety)?;
                let slice_ty = MirType::Slice(Box::new(seq_elem_ty.clone()));
                let len_tmp = self.alloc_temp("for_len", MirType::Usize);
                let idx_tmp = self.alloc_temp("for_idx", MirType::Usize);
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(len_tmp),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(
                        MirExprKind::SliceLen(Box::new(MirExpr::new(
                            MirExprKind::Var(slice_tmp),
                            slice_ty.clone(),
                        ))),
                        MirType::Usize,
                    ),
                });
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(idx_tmp),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(MirExprKind::IntLit(0), MirType::Usize),
                });

                self.scopes.push();
                let elem_var = self.declare_var(&elem_name.name, ety.clone(), VarRole::UserLocal);

                let cond_bb = self.alloc_bb();
                let body_bb = self.alloc_bb();
                let step_bb = self.alloc_bb();
                let exit_bb = self.alloc_bb();

                self.seal_bb(MirTerminator::Goto(cond_bb), cond_bb, false);
                self.seal_bb(
                    MirTerminator::Branch {
                        cond: MirExpr::new(
                            MirExprKind::Binary {
                                op: MirBinOp::Lt,
                                lhs: Box::new(MirExpr::new(MirExprKind::Var(idx_tmp), MirType::Usize)),
                                rhs: Box::new(MirExpr::new(MirExprKind::Var(len_tmp), MirType::Usize)),
                            },
                            MirType::Bool,
                        ),
                        then_bb: body_bb,
                        else_bb: exit_bb,
                    },
                    body_bb,
                    false,
                );

                let raw_elem_expr = MirExpr::new(
                    MirExprKind::Index {
                        base: Box::new(MirExpr::new(MirExprKind::Var(slice_tmp), slice_ty)),
                        index: Box::new(MirExpr::new(MirExprKind::Var(idx_tmp), MirType::Usize)),
                        is_slice: true,
                    },
                    seq_elem_ty.host_value_type(),
                );
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(elem_var),
                    op: MirAssignOp::Assign,
                    value: coerce_or_cast_int(raw_elem_expr, &ety),
                });

                let has_inbox = ast_stmts_have_inbox_recv(&body.stmts);
                let defer_depth = self.defer_stack.len();
                self.loop_stack.push(LoopContext {
                    continue_bb: step_bb,
                    break_bb: exit_bb,
                    has_inbox_recv: has_inbox,
                    defer_depth,
                });
                self.lower_scoped_block(body)?;
                self.loop_stack.pop();
                self.scopes.pop();

                self.seal_bb(MirTerminator::Goto(step_bb), step_bb, false);
                if has_inbox {
                    self.cur_stmts.push(MirStmt::EndInboxTurn);
                }
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(idx_tmp),
                    op: MirAssignOp::AddAssign,
                    value: MirExpr::new(MirExprKind::IntLit(1), MirType::Usize),
                });
                self.seal_bb(MirTerminator::Goto(cond_bb), exit_bb, false);
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
                    self.emit_defers_down_to(lc.defer_depth)?;
                    let dead_bb = self.alloc_bb();
                    self.seal_bb(MirTerminator::Goto(lc.continue_bb), dead_bb, false);
                }
            }
            StmtKind::Break => {
                if let Some(&lc) = self.loop_stack.last() {
                    self.emit_defers_down_to(lc.defer_depth)?;
                    if lc.has_inbox_recv {
                        self.cur_stmts.push(MirStmt::EndInboxTurn);
                    }
                    let dead_bb = self.alloc_bb();
                    self.seal_bb(MirTerminator::Goto(lc.break_bb), dead_bb, false);
                }
            }
            StmtKind::Checkpoint => {
                if !self.in_noyield {
                    let resume_bb = self.alloc_bb();
                    self.seal_bb(
                        MirTerminator::Yield(MirYieldKind::ShareCheckpoint { resume_bb }),
                        resume_bb,
                        true,
                    );
                }
            }
            StmtKind::Block(b) => {
                self.lower_scoped_block(b)?;
            }
            StmtKind::Expr(e) => {
                if let ExprKind::MethodCall {
                    receiver, method, ..
                } = &e.kind
                {
                    if (method.name == "attach" || method.name == "attach_shared")
                        && resolve_irq_token_be(receiver, self).is_some()
                    {
                        return Err(MirLowerError {
                            message: format!(
                                "IRQ `.{}` may only be called during `fn main()` boot sequence",
                                method.name
                            ),
                            span: stmt.span,
                        });
                    }
                }
                if let Some(code_arg_opt) = extract_sys_exit_arg(e) {
                    self.emit_coroutine_sys_exit(code_arg_opt)?;
                    return Ok(());
                }
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
                    message: "`spawn` is only valid during `fn main()` boot".to_string(),
                    span: stmt.span,
                });
            }
        }
        Ok(())
    }

    fn emit_coroutine_sys_exit(&mut self, code_arg_opt: Option<&Expr>) -> Result<(), MirLowerError> {
        let code_expr = if let Some(ce) = code_arg_opt {
            let raw = self.lower_expr(ce, Some(&MirType::I32))?;
            let coerced = coerce_or_cast_int(raw, &MirType::I32);
            if self.has_active_defers_from(0) {
                let tmp = self.alloc_temp("exit_code", MirType::I32);
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(tmp),
                    op: MirAssignOp::Assign,
                    value: coerced,
                });
                MirExpr::new(MirExprKind::Var(tmp), MirType::I32)
            } else {
                coerced
            }
        } else {
            MirExpr::new(MirExprKind::IntLit(0), MirType::I32)
        };
        self.emit_defers_down_to(0)?;
        self.cur_stmts.push(MirStmt::EndInboxTurn);
        self.cur_stmts.push(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
            method: "exit".to_string(),
            args: vec![code_expr],
        }));
        let dead_bb = self.alloc_bb();
        self.seal_bb(MirTerminator::Return(None), dead_bb, false);
        Ok(())
    }

    fn lower_iter_into_slice_temp(
        &mut self,
        iter: &Expr,
        fallback_elem_ty: &MirType,
    ) -> Result<(VarId, MirType), MirLowerError> {
        if let Ok((place, MirType::Array { elem, len })) = self.lower_place(iter) {
            let slice_ty = MirType::Slice(elem.clone());
            let slice_tmp = self.alloc_temp("for_slice", slice_ty.clone());
            self.cur_stmts.push(MirStmt::Assign {
                target: MirPlace::Var(slice_tmp),
                op: MirAssignOp::Assign,
                value: MirExpr::new(
                    MirExprKind::ArrayToSlice {
                        array_place: Box::new(place),
                        len,
                        elem_ty: *elem.clone(),
                    },
                    slice_ty,
                ),
            });
            return Ok((slice_tmp, *elem));
        }

        let iv = unwrap_ok_result_expr(self.lower_expr(iter, None)?);
        match iv.ty.clone() {
            MirType::Slice(elem) => {
                let et = *elem.clone();
                let slice_tmp = self.alloc_temp("for_slice", iv.ty.clone());
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(slice_tmp),
                    op: MirAssignOp::Assign,
                    value: iv,
                });
                Ok((slice_tmp, et))
            }
            MirType::Ptr(inner) => match inner.as_ref() {
                MirType::Slice(elem) => {
                    let et = *elem.clone();
                    let slice_ty = MirType::Slice(elem.clone());
                    let slice_tmp = self.alloc_temp("for_slice", slice_ty.clone());
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(slice_tmp),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(
                            MirExprKind::Unary {
                                op: MirUnaryOp::Deref,
                                expr: Box::new(iv),
                            },
                            slice_ty,
                        ),
                    });
                    Ok((slice_tmp, et))
                }
                MirType::Array { elem, len } => {
                    let et = *elem.clone();
                    let len_val = *len;
                    let ptr_tmp = self.alloc_temp("for_arr_ptr", iv.ty.clone());
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(ptr_tmp),
                        op: MirAssignOp::Assign,
                        value: iv.clone(),
                    });
                    let arr_ty = MirType::Array {
                        elem: elem.clone(),
                        len: len_val,
                    };
                    let slice_ty = MirType::Slice(elem.clone());
                    let slice_tmp = self.alloc_temp("for_slice", slice_ty.clone());
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(slice_tmp),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(
                            MirExprKind::ArrayToSlice {
                                array_place: Box::new(MirPlace::Deref(
                                    Box::new(MirExpr::new(MirExprKind::Var(ptr_tmp), iv.ty)),
                                    arr_ty,
                                )),
                                len: len_val,
                                elem_ty: et.clone(),
                            },
                            slice_ty,
                        ),
                    });
                    Ok((slice_tmp, et))
                }
                _ => {
                    let slice_ty = MirType::Slice(Box::new(fallback_elem_ty.clone()));
                    let slice_tmp = self.alloc_temp("for_slice", slice_ty);
                    Ok((slice_tmp, fallback_elem_ty.clone()))
                }
            },
            MirType::Array { elem, len } => {
                let et = *elem.clone();
                let len_val = len;
                let arr_tmp = self.alloc_temp("for_arr", iv.ty.clone());
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(arr_tmp),
                    op: MirAssignOp::Assign,
                    value: iv,
                });
                let slice_ty = MirType::Slice(elem.clone());
                let slice_tmp = self.alloc_temp("for_slice", slice_ty.clone());
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(slice_tmp),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(
                        MirExprKind::ArrayToSlice {
                            array_place: Box::new(MirPlace::Var(arr_tmp)),
                            len: len_val,
                            elem_ty: et.clone(),
                        },
                        slice_ty,
                    ),
                });
                Ok((slice_tmp, et))
            }
            _ => {
                let slice_ty = MirType::Slice(Box::new(fallback_elem_ty.clone()));
                let slice_tmp = self.alloc_temp("for_slice", slice_ty);
                Ok((slice_tmp, fallback_elem_ty.clone()))
            }
        }
    }

    /// Directly lower `VarType dst = <yielding_op>;` without allocating an extra copy temp!
    fn try_lower_yielding_into_var(
        &mut self,
        expr: &Expr,
        dst_vid: VarId,
        dst_ty: &MirType,
    ) -> Result<bool, MirLowerError> {
        match &expr.kind {
            ExprKind::Call { callee, args, .. } => {
                if let ExprKind::Ident(id) = &callee.kind {
                    if let Some(sig) = self.syms.fns.get(&id.name).cloned() {
                        if sig.spawns_in_boot {
                            return Err(MirLowerError {
                                message: format!(
                                    "boot helper `{}` (contains `spawn` or IRQ `.attach`) may only be called during `fn main()` boot",
                                    id.name
                                ),
                                span: expr.span,
                            });
                        }
                        let yields_here = sig.is_yielding
                            && !(self.in_noyield && sig.effect == MirFnEffect::SoftYielding);
                        if yields_here && !sig.error_spec.has_error() {
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
                        let yields_here = sig.is_yielding
                            && !(self.in_noyield && sig.effect == MirFnEffect::SoftYielding);
                        return yields_here && sig.ret_ty.is_void() && !sig.error_spec.has_error();
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
                (m == "send" || m == "wait") && find_named_arg(args, "timeout").is_none()
            }
            _ => false,
        }
    }

    fn try_lower_yielding_stmt_expr(&mut self, expr: &Expr) -> Result<bool, MirLowerError> {
        match &expr.kind {
            ExprKind::Call { callee, args, .. } => {
                if let ExprKind::Ident(id) = &callee.kind {
                    if matches!(id.name.as_str(), "delay" | "delay_ms" | "sleep") {
                        if let Some(first) = args.first() {
                            self.emit_delay(&first.value)?;
                            return Ok(true);
                        }
                    }
                    if id.name == "checkpoint" {
                        if !self.in_noyield {
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::ShareCheckpoint { resume_bb }),
                                resume_bb,
                                true,
                            );
                        }
                        return Ok(true);
                    }
                    if let Some(sig) = self.syms.fns.get(&id.name).cloned() {
                        if sig.spawns_in_boot {
                            return Err(MirLowerError {
                                message: format!(
                                    "boot helper `{}` (contains `spawn` or IRQ `.attach`) may only be called during `fn main()` boot",
                                    id.name
                                ),
                                span: expr.span,
                            });
                        }
                        let yields_here = sig.is_yielding
                            && !(self.in_noyield && sig.effect == MirFnEffect::SoftYielding);
                        if yields_here {
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
                        ("send", MirPrimKind::Inbox { elem, cap })
                            if find_named_arg(args, "timeout").is_none() =>
                        {
                            let vexpr = if let Some(first) = first_positional_arg(args) {
                                let raw = self.lower_expr(first, Some(&elem))?;
                                coerce_expr_if_needed(raw, &elem)
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                            };
                            let val_var = self.alloc_temp("inbox_tx", *elem.clone());
                            self.cur_stmts.push(MirStmt::Assign {
                                target: MirPlace::Var(val_var),
                                op: MirAssignOp::Assign,
                                value: vexpr,
                            });
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                            self.seal_bb(
                                MirTerminator::Yield(MirYieldKind::InboxSend {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    val_var,
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
            let me = self.lower_expr(&a.value, sig.params.get(idx))?;
            let coerced = if let Some(pty) = sig.params.get(idx) {
                self.coerce_arg_for_param(me, pty)
            } else {
                me
            };
            largs.push(coerced);
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
        resolve_prim_receiver_be(receiver, self)
    }

    fn lower_expr(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError> {
        match &expr.kind {
            ExprKind::Or { .. } => {
                let mut or_chain = Vec::new();
                flatten_or_operands(expr, &mut or_chain);
                let heads = &or_chain[..or_chain.len() - 1];
                let tail = or_chain[or_chain.len() - 1];

                let (first_res_vid, ok_ty, _) = self.lower_into_result_var(heads[0])?;
                let out_vid_opt = if ok_ty.is_void() {
                    None
                } else {
                    Some(self.alloc_temp("or_val", ok_ty.clone()))
                };
                let join_bb = self.alloc_bb();

                let mut cur_res_vid = first_res_vid;
                for (i, _) in heads.iter().enumerate() {
                    if i > 0 {
                        let (next_vid, _, _) = self.lower_into_result_var(heads[i])?;
                        cur_res_vid = next_vid;
                    }
                    let err_bb = self.alloc_bb();
                    let ok_bb = self.alloc_bb();
                    self.seal_bb(
                        MirTerminator::Branch {
                            cond: MirExpr::new(
                                MirExprKind::ResultIsErr(cur_res_vid),
                                MirType::Bool,
                            ),
                            then_bb: err_bb,
                            else_bb: ok_bb,
                        },
                        ok_bb,
                        false,
                    );
                    if let Some(out_vid) = out_vid_opt {
                        self.cur_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(out_vid),
                            op: MirAssignOp::Assign,
                            value: coerce_expr_if_needed(
                                MirExpr::new(MirExprKind::ResultVal(cur_res_vid), ok_ty.clone()),
                                &ok_ty,
                            ),
                        });
                    }
                    self.seal_bb(MirTerminator::Goto(join_bb), err_bb, false);
                }

                let fb_val = self.lower_expr(tail, Some(&ok_ty))?;
                if let Some(out_vid) = out_vid_opt {
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(out_vid),
                        op: MirAssignOp::Assign,
                        value: coerce_expr_if_needed(fb_val, &ok_ty),
                    });
                    self.seal_bb(MirTerminator::Goto(join_bb), join_bb, false);
                    Ok(MirExpr::new(MirExprKind::Var(out_vid), ok_ty))
                } else {
                    if !fb_val.ty.is_void()
                        || matches!(fb_val.kind, MirExprKind::CallDirect { .. })
                    {
                        self.cur_stmts.push(MirStmt::Expr(fb_val));
                    }
                    self.seal_bb(MirTerminator::Goto(join_bb), join_bb, false);
                    Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void))
                }
            }
            ExprKind::Catch {
                expr: inner,
                clause,
            } => {
                let mut or_chain = Vec::new();
                flatten_or_operands(inner, &mut or_chain);
                let heads = &or_chain[..or_chain.len() - 1];
                let tail = or_chain[or_chain.len() - 1];

                let join_bb = self.alloc_bb();
                let mut pre_out_opt: Option<(Option<VarId>, MirType)> = None;

                for &h in heads {
                    let (h_vid, h_ok_ty, _) = self.lower_into_result_var(h)?;
                    let (out_vid_opt, ok_ty_ref) = match &pre_out_opt {
                        Some((ov, ot)) => (*ov, ot.clone()),
                        None => {
                            let ov = if h_ok_ty.is_void() {
                                None
                            } else {
                                Some(self.alloc_temp("catch_val", h_ok_ty.clone()))
                            };
                            pre_out_opt = Some((ov, h_ok_ty.clone()));
                            (ov, h_ok_ty)
                        }
                    };
                    let next_try_bb = self.alloc_bb();
                    let ok_bb = self.alloc_bb();
                    self.seal_bb(
                        MirTerminator::Branch {
                            cond: MirExpr::new(MirExprKind::ResultIsErr(h_vid), MirType::Bool),
                            then_bb: next_try_bb,
                            else_bb: ok_bb,
                        },
                        ok_bb,
                        false,
                    );
                    if let Some(out_vid) = out_vid_opt {
                        self.cur_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(out_vid),
                            op: MirAssignOp::Assign,
                            value: coerce_expr_if_needed(
                                MirExpr::new(MirExprKind::ResultVal(h_vid), ok_ty_ref.clone()),
                                &ok_ty_ref,
                            ),
                        });
                    }
                    self.seal_bb(MirTerminator::Goto(join_bb), next_try_bb, false);
                }

                let (res_vid, tail_ok_ty, err_ty_opt) = self.lower_into_result_var(tail)?;
                let (out_vid_opt, ok_ty) = match pre_out_opt {
                    Some(pair) => pair,
                    None => {
                        let ov = if tail_ok_ty.is_void() {
                            None
                        } else {
                            Some(self.alloc_temp("catch_val", tail_ok_ty.clone()))
                        };
                        (ov, tail_ok_ty)
                    }
                };
                let err_bb = self.alloc_bb();
                let ok_bb = self.alloc_bb();

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
                self.defer_stack.push(Vec::new());
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
                        if let Some(code_arg_opt) = extract_sys_exit_arg(tail_expr) {
                            self.emit_coroutine_sys_exit(code_arg_opt)?;
                        } else if self.is_void_yielding_stmt_expr(tail_expr)
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
                self.pop_and_lower_top_defer_scope()?;
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
            ExprKind::Call { callee, args, .. } => {
                if let ExprKind::Ident(id) = &callee.kind {
                    if let Some(sig) = self.syms.fns.get(&id.name).cloned() {
                        if sig.spawns_in_boot {
                            return Err(MirLowerError {
                                message: format!(
                                    "boot helper `{}` (contains `spawn` or IRQ `.attach`) may only be called during `fn main()` boot",
                                    id.name
                                ),
                                span: expr.span,
                            });
                        }
                        let yields_here = sig.is_yielding
                            && !(self.in_noyield && sig.effect == MirFnEffect::SoftYielding);
                        if yields_here {
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
                        ("send", MirPrimKind::Inbox { elem, cap }) => {
                            let vexpr = if let Some(first) = first_positional_arg(args) {
                                let raw = self.lower_expr(first, Some(&elem))?;
                                coerce_expr_if_needed(raw, &elem)
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                            };
                            let val_var = self.alloc_temp("inbox_tx", *elem.clone());
                            self.cur_stmts.push(MirStmt::Assign {
                                target: MirPlace::Var(val_var),
                                op: MirAssignOp::Assign,
                                value: vexpr,
                            });
                            if let Some(texpr) = find_named_arg(args, "timeout") {
                                let ticks = self.lower_expr(texpr, Some(&MirType::U32))?;
                                let dl_var = self.alloc_temp("inbox_tx_deadline", MirType::U32);
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
                                    ok: Box::new(MirType::Void),
                                    err: None,
                                };
                                let dst_res = self.alloc_temp("inbox_tx_res", rty.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                                self.seal_bb(
                                    MirTerminator::Yield(MirYieldKind::InboxSend {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        val_var,
                                        timeout_deadline_var: Some(dl_var),
                                        dst_res: Some(dst_res),
                                        poll_bb,
                                        resume_bb,
                                    }),
                                    resume_bb,
                                    false,
                                );
                                return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                            } else {
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                                self.seal_bb(
                                    MirTerminator::Yield(MirYieldKind::InboxSend {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        val_var,
                                        timeout_deadline_var: None,
                                        dst_res: None,
                                        poll_bb,
                                        resume_bb,
                                    }),
                                    resume_bb,
                                    false,
                                );
                                return Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void));
                            }
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
        lower_common_expr(expr, expected, self)
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
    fn in_noyield(&self) -> bool;
    fn alloc_temp_be(&mut self, prefix: &str, ty: MirType) -> VarId;
    fn emit_stmt_be(&mut self, stmt: MirStmt);
    fn invalidate_boot_const(&mut self, _vid: VarId) {}
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
    fn in_noyield(&self) -> bool {
        self.ctx.in_noyield
    }
    fn alloc_temp_be(&mut self, prefix: &str, ty: MirType) -> VarId {
        self.ctx.alloc_temp(prefix, ty)
    }
    fn emit_stmt_be(&mut self, stmt: MirStmt) {
        self.stmts.push(stmt);
    }
    fn invalidate_boot_const(&mut self, vid: VarId) {
        if self.ctx.boot_state.is_some() {
            self.ctx.boot_const_env.remove(&vid);
        }
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
    fn in_noyield(&self) -> bool {
        self.in_noyield
    }
    fn alloc_temp_be(&mut self, prefix: &str, ty: MirType) -> VarId {
        self.alloc_temp(prefix, ty)
    }
    fn emit_stmt_be(&mut self, stmt: MirStmt) {
        self.cur_stmts.push(stmt);
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

fn resolve_irq_token_be<B: LowerBackend>(expr: &Expr, be: &B) -> Option<(String, u32)> {
    match &expr.kind {
        ExprKind::Ident(id) if be.scopes().lookup(&id.name).is_none() => be
            .syms()
            .irqs
            .get(&id.name)
            .map(|&n| (id.name.clone(), n)),
        ExprKind::Field { .. } => be.syms().resolve_irq_field_be(expr, be.scopes()),
        _ => None,
    }
}

fn lookup_mmio_reg<'s>(
    syms: &'s ModuleSymbols,
    layout_name: &str,
    reg_name: &str,
    span: Span,
) -> Result<&'s MirMmioReg, MirLowerError> {
    let layout = syms
        .mmio_layouts
        .get(layout_name)
        .ok_or_else(|| MirLowerError {
            message: format!("unknown MMIO layout `{layout_name}`"),
            span,
        })?;
    layout
        .regs
        .iter()
        .find(|r| r.name == reg_name)
        .ok_or_else(|| MirLowerError {
            message: format!("MMIO layout `{layout_name}` has no register `{reg_name}`"),
            span,
        })
}

fn peek_expr_struct_type<B: LowerBackend>(expr: &Expr, be: &B) -> Option<String> {
    match &expr.kind {
        ExprKind::Ident(id) => be.scopes().lookup(&id.name).and_then(|vid| {
            match &be.vars()[vid.0].ty {
                MirType::Struct(n) => Some(n.clone()),
                MirType::Ptr(inner) => match inner.as_ref() {
                    MirType::Struct(n) => Some(n.clone()),
                    _ => None,
                },
                _ => None,
            }
        }),
        ExprKind::Field { target, field } => {
            let sname = peek_expr_struct_type(target, be)?;
            let st = be.syms().structs.get(&sname)?;
            let sf = st.fields.iter().find(|f| f.name == field.name)?;
            match &sf.ty {
                MirType::Struct(n) => Some(n.clone()),
                MirType::Ptr(inner) => match inner.as_ref() {
                    MirType::Struct(n) => Some(n.clone()),
                    _ => None,
                },
                _ => None,
            }
        }
        _ => None,
    }
}

fn peek_expr_mmio_layout<B: LowerBackend>(expr: &Expr, be: &B) -> Option<String> {
    match &expr.kind {
        ExprKind::Unary {
            op: UnaryOp::Deref,
            expr: inner,
        } => peek_expr_mmio_layout(inner, be),
        ExprKind::Ident(id) => {
            if let Some(vid) = be.scopes().lookup(&id.name) {
                if let MirType::MmioPtr(layout) = &be.vars()[vid.0].ty {
                    return Some(layout.clone());
                }
                return None;
            }
            be.syms()
                .mmio_instances
                .get(&id.name)
                .map(|inst| inst.layout.clone())
        }
        ExprKind::Unary {
            op: UnaryOp::AddrOf,
            expr: inner,
        } => match &inner.kind {
            ExprKind::Ident(id) if be.scopes().lookup(&id.name).is_none() => be
                .syms()
                .mmio_instances
                .get(&id.name)
                .map(|inst| inst.layout.clone()),
            ExprKind::Field { .. } => be
                .syms()
                .resolve_mmio_instance_field_be(inner, be.scopes())
                .and_then(|inst| {
                    be.syms()
                        .mmio_instances
                        .get(&inst)
                        .map(|i| i.layout.clone())
                }),
            _ => None,
        },
        ExprKind::Field { target, field } => {
            if let Some(inst) = be.syms().resolve_mmio_instance_field_be(expr, be.scopes()) {
                return be
                    .syms()
                    .mmio_instances
                    .get(&inst)
                    .map(|i| i.layout.clone());
            }
            let sname = peek_expr_struct_type(target, be)?;
            let st = be.syms().structs.get(&sname)?;
            let sf = st.fields.iter().find(|f| f.name == field.name)?;
            if let MirType::MmioPtr(layout) = &sf.ty {
                Some(layout.clone())
            } else {
                None
            }
        }
        ExprKind::Call { callee, .. } => {
            if let ExprKind::Ident(id) = &callee.kind {
                if let Some(sig) = be.syms().fns.get(&id.name) {
                    if let MirType::MmioPtr(layout) = &sig.ret_ty {
                        return Some(layout.clone());
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn lower_mmio_base_be<B: LowerBackend>(
    mut base_ast: &Expr,
    be: &mut B,
) -> Result<MirExpr, MirLowerError> {
    while let ExprKind::Unary {
        op: UnaryOp::Deref,
        expr: inner,
    } = &base_ast.kind
    {
        if peek_expr_mmio_layout(inner, be).is_some() {
            base_ast = inner;
        } else {
            break;
        }
    }
    be.lower_sub(base_ast, None)
}

fn resolve_mmio_lvalue_be<B: LowerBackend>(
    target: &Expr,
    be: &mut B,
) -> Result<Option<(MirExpr, MirMmioReg, Option<MirExpr>)>, MirLowerError> {
    match &target.kind {
        ExprKind::Field {
            target: base_ast,
            field: reg_ident,
        } => {
            if peek_expr_mmio_layout(base_ast, be).is_none() {
                return Ok(None);
            }
            let base = lower_mmio_base_be(base_ast, be)?;
            let MirType::MmioPtr(layout_name) = &base.ty else {
                return Err(MirLowerError {
                    message: format!(
                        "expected MMIO peripheral pointer for register `.{}`, found `{}`",
                        reg_ident.name,
                        base.ty.display_name()
                    ),
                    span: target.span,
                });
            };
            let reg = lookup_mmio_reg(be.syms(), layout_name, &reg_ident.name, target.span)?.clone();
            if reg.array_len.is_some() {
                return Err(MirLowerError {
                    message: format!(
                        "MMIO register array `{}` on `{layout_name}` must be indexed (`[i]`)",
                        reg.name
                    ),
                    span: target.span,
                });
            }
            Ok(Some((base, reg, None)))
        }
        ExprKind::Index {
            target: inner_field,
            index: idx_ast,
        } => {
            let ExprKind::Field {
                target: base_ast,
                field: reg_ident,
            } = &inner_field.kind
            else {
                return Ok(None);
            };
            if peek_expr_mmio_layout(base_ast, be).is_none() {
                return Ok(None);
            }
            let base = lower_mmio_base_be(base_ast, be)?;
            let MirType::MmioPtr(layout_name) = &base.ty else {
                return Err(MirLowerError {
                    message: format!(
                        "expected MMIO peripheral pointer for register `.{}`, found `{}`",
                        reg_ident.name,
                        base.ty.display_name()
                    ),
                    span: target.span,
                });
            };
            let reg = lookup_mmio_reg(be.syms(), layout_name, &reg_ident.name, inner_field.span)?
                .clone();
            if reg.array_len.is_none() {
                return Err(MirLowerError {
                    message: format!(
                        "MMIO scalar register `{}` on `{layout_name}` cannot be indexed",
                        reg.name
                    ),
                    span: target.span,
                });
            }
            let idx = be.lower_sub(idx_ast, Some(&MirType::Usize))?;
            Ok(Some((base, reg, Some(idx))))
        }
        _ => Ok(None),
    }
}

fn compound_assign_to_binop(op: AssignOp) -> Option<MirBinOp> {
    match op {
        AssignOp::Assign => None,
        AssignOp::AddAssign => Some(MirBinOp::Add),
        AssignOp::SubAssign => Some(MirBinOp::Sub),
        AssignOp::MulAssign => Some(MirBinOp::Mul),
        AssignOp::DivAssign => Some(MirBinOp::Div),
        AssignOp::RemAssign => Some(MirBinOp::Rem),
        AssignOp::BitAndAssign => Some(MirBinOp::BitAnd),
        AssignOp::BitOrAssign => Some(MirBinOp::BitOr),
        AssignOp::BitXorAssign => Some(MirBinOp::BitXor),
        AssignOp::ShlAssign => Some(MirBinOp::Shl),
        AssignOp::ShrAssign => Some(MirBinOp::Shr),
    }
}

fn try_lower_mmio_assign_be<B: LowerBackend>(
    target: &Expr,
    op: AssignOp,
    value: &Expr,
    be: &mut B,
) -> Result<bool, MirLowerError> {
    let Some((base, reg, index_opt)) = resolve_mmio_lvalue_be(target, be)? else {
        return Ok(false);
    };
    let reg_ty = mmio_elem_mir_type(reg.elem_bits);
    if op == AssignOp::Assign {
        if reg.access == MirMmioAccess::Ro {
            return Err(MirLowerError {
                message: format!("cannot write to read-only `@ro` MMIO register `{}`", reg.name),
                span: target.span,
            });
        }
        let rhs_raw = be.lower_sub(value, Some(&reg_ty))?;
        let rhs = coerce_expr_if_needed(rhs_raw, &reg_ty);
        be.emit_stmt_be(MirStmt::MmioWrite {
            base: Box::new(base),
            offset: reg.offset,
            elem_bits: reg.elem_bits,
            index: index_opt.map(Box::new),
            array_len: reg.array_len,
            value: rhs,
        });
        return Ok(true);
    }

    if reg.access != MirMmioAccess::Rw {
        return Err(MirLowerError {
            message: format!(
                "compound read-modify-write on MMIO register `{}` is only permitted on `@rw` registers (register is `{:?}`)",
                reg.name, reg.access
            ),
            span: target.span,
        });
    }

    // Evaluate base pointer and array index strictly once into CompilerTemp vars.
    let base_ty = base.ty.clone();
    let base_tmp = be.alloc_temp_be("mmio_base", base_ty.clone());
    be.emit_stmt_be(MirStmt::Assign {
        target: MirPlace::Var(base_tmp),
        op: MirAssignOp::Assign,
        value: base,
    });
    let base_ref = MirExpr::new(MirExprKind::Var(base_tmp), base_ty);

    let idx_ref = if let Some(idx_expr) = index_opt {
        let idx_tmp = be.alloc_temp_be("mmio_idx", MirType::Usize);
        be.emit_stmt_be(MirStmt::Assign {
            target: MirPlace::Var(idx_tmp),
            op: MirAssignOp::Assign,
            value: coerce_expr_if_needed(idx_expr, &MirType::Usize),
        });
        Some(MirExpr::new(MirExprKind::Var(idx_tmp), MirType::Usize))
    } else {
        None
    };

    let read_expr = MirExpr::new(
        MirExprKind::MmioRead {
            base: Box::new(base_ref.clone()),
            offset: reg.offset,
            elem_bits: reg.elem_bits,
            index: idx_ref.clone().map(Box::new),
            array_len: reg.array_len,
        },
        reg_ty.clone(),
    );

    let rhs_raw = be.lower_sub(value, Some(&reg_ty))?;
    let rhs = coerce_expr_if_needed(rhs_raw, &reg_ty);
    let bin_op = compound_assign_to_binop(op).unwrap();
    let updated = MirExpr::new(
        MirExprKind::Binary {
            op: bin_op,
            lhs: Box::new(read_expr),
            rhs: Box::new(rhs),
        },
        reg_ty,
    );

    be.emit_stmt_be(MirStmt::MmioWrite {
        base: Box::new(base_ref),
        offset: reg.offset,
        elem_bits: reg.elem_bits,
        index: idx_ref.map(Box::new),
        array_len: reg.array_len,
        value: updated,
    });
    Ok(true)
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

fn flatten_or_operands<'e>(expr: &'e Expr, out: &mut Vec<&'e Expr>) {
    if let ExprKind::Or {
        expr: l,
        fallback: r,
    } = &expr.kind
    {
        flatten_or_operands(l, out);
        flatten_or_operands(r, out);
    } else {
        out.push(expr);
    }
}

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

fn unwrap_ok_result_expr(expr: MirExpr) -> MirExpr {
    if let (MirType::Result { ok, .. }, MirExprKind::Var(vid)) = (&expr.ty, &expr.kind) {
        return MirExpr::new(MirExprKind::ResultVal(*vid), *ok.clone());
    }
    expr
}

fn coerce_expr_if_needed(expr: MirExpr, target_ty: &MirType) -> MirExpr {
    if &expr.ty == target_ty || target_ty.is_void() {
        return expr;
    }
    let host_target = target_ty.host_value_type();
    if let (MirType::Result { ok, .. }, MirExprKind::Var(vid)) = (&expr.ty, &expr.kind) {
        if **ok == *target_ty || ok.host_value_type() == host_target {
            return MirExpr::new(MirExprKind::ResultVal(*vid), ok.as_ref().clone());
        }
    }
    match expr.kind {
        MirExprKind::IntLit(v) => {
            if matches!(
                host_target,
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
                return MirExpr::new(MirExprKind::IntLit(v), host_target);
            }
            MirExpr::new(MirExprKind::IntLit(v), expr.ty)
        }
        MirExprKind::FloatLit(bits) => match host_target {
            MirType::F32 => MirExpr::new(
                MirExprKind::FloatLit(normalize_f32_bits(bits)),
                MirType::F32,
            ),
            MirType::F64 => MirExpr::new(MirExprKind::FloatLit(bits), MirType::F64),
            _ => MirExpr::new(MirExprKind::FloatLit(bits), expr.ty),
        },
        other_kind => MirExpr::new(other_kind, expr.ty),
    }
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
            Literal::Float { bits, suffix } => {
                let ty = match suffix {
                    Some(PrimType::F32) => MirType::F32,
                    Some(PrimType::F64) => MirType::F64,
                    _ => match expected.map(|t| t.host_value_type()) {
                        Some(MirType::F32) => MirType::F32,
                        _ => MirType::F64,
                    },
                };
                let norm_bits = if ty == MirType::F32 {
                    normalize_f32_bits(*bits)
                } else {
                    *bits
                };
                Ok(MirExpr::new(MirExprKind::FloatLit(norm_bits), ty))
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
            if let Some(pinfo) = syms.primitives.get(&id.name) {
                return Ok(MirExpr::new(
                    MirExprKind::GlobalPrimAddr(id.name.clone()),
                    MirType::Ptr(Box::new(MirType::Primitive(pinfo.kind.clone()))),
                ));
            }
            if let Some(inst) = syms.mmio_instances.get(&id.name) {
                return Ok(MirExpr::new(
                    MirExprKind::MmioAddr(inst.name.clone()),
                    MirType::MmioPtr(inst.layout.clone()),
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
                if be.scopes().lookup(&id.name).is_none() {
                    if let Some(pinfo) = be.syms().primitives.get(&id.name) {
                        return Ok(MirExpr::new(
                            MirExprKind::GlobalPrimAddr(id.name.clone()),
                            MirType::Ptr(Box::new(MirType::Primitive(pinfo.kind.clone()))),
                        ));
                    }
                    if let Some(inst) = be.syms().mmio_instances.get(&id.name) {
                        return Ok(MirExpr::new(
                            MirExprKind::MmioAddr(inst.name.clone()),
                            MirType::MmioPtr(inst.layout.clone()),
                        ));
                    }
                    if be.syms().fns.contains_key(&id.name) {
                        return Ok(MirExpr::new(
                            MirExprKind::FnAddr(id.name.clone()),
                            expected.cloned().unwrap_or(MirType::Usize),
                        ));
                    }
                }
            } else if let Some(inst_name) = be
                .syms()
                .resolve_mmio_instance_field_be(inner, be.scopes())
            {
                let layout = be.syms().mmio_instances[&inst_name].layout.clone();
                return Ok(MirExpr::new(
                    MirExprKind::MmioAddr(inst_name),
                    MirType::MmioPtr(layout),
                ));
            }
            let (place, pty) = be.lower_place_be(inner)?;
            if let Some(vid) = place_root_var(&place) {
                be.invalidate_boot_const(vid);
            }
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
            if let Some(inst_name) = be
                .syms()
                .resolve_mmio_instance_field_be(expr, be.scopes())
            {
                let layout = be.syms().mmio_instances[&inst_name].layout.clone();
                return Ok(MirExpr::new(
                    MirExprKind::MmioAddr(inst_name),
                    MirType::MmioPtr(layout),
                ));
            }
            if let ExprKind::Ident(id) = &target.kind {
                if be.scopes().lookup(&id.name).is_none()
                    && be.syms().enums.contains_key(&id.name)
                {
                    return Ok(MirExpr::new(
                        MirExprKind::EnumVariant {
                            enum_name: id.name.clone(),
                            variant: field.name.clone(),
                        },
                        MirType::Enum(id.name.clone()),
                    ));
                }
            }
            if field.name == "len" {
                if let ExprKind::Field {
                    target: mmio_base_ast,
                    field: reg_ident,
                } = &target.kind
                {
                    if let Some(layout_name) = peek_expr_mmio_layout(mmio_base_ast, be) {
                        let _base = lower_mmio_base_be(mmio_base_ast, be)?;
                        let reg =
                            lookup_mmio_reg(be.syms(), &layout_name, &reg_ident.name, target.span)?;
                        if let Some(len) = reg.array_len {
                            return Ok(MirExpr::new(MirExprKind::ArrayLen(len), MirType::Usize));
                        }
                    }
                }
            }
            if field.name == "ptr" {
                if let Ok((place, MirType::Array { elem, .. })) = be.lower_place_be(target) {
                    if let Some(vid) = place_root_var(&place) {
                        be.invalidate_boot_const(vid);
                    }
                    let idx0 = MirPlace::Index {
                        base: Box::new(place),
                        index: Box::new(MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)),
                        elem_ty: *elem.clone(),
                        is_slice: false,
                    };
                    return Ok(MirExpr::new(
                        MirExprKind::AddrOfPlace(Box::new(idx0)),
                        MirType::Ptr(elem),
                    ));
                }
            }
            let base = if peek_expr_mmio_layout(target, be).is_some() {
                lower_mmio_base_be(target, be)?
            } else {
                be.lower_sub(target, None)?
            };
            finish_field_access(base, &field.name, be.syms(), expr.span)
        }
        ExprKind::Index { target, index } => {
            if let Some((base, reg, Some(idx))) = resolve_mmio_lvalue_be(expr, be)? {
                if reg.access == MirMmioAccess::Wo {
                    return Err(MirLowerError {
                        message: format!(
                            "cannot read write-only `@wo` MMIO register `{}`",
                            reg.name
                        ),
                        span: expr.span,
                    });
                }
                let reg_ty = mmio_elem_mir_type(reg.elem_bits);
                return Ok(MirExpr::new(
                    MirExprKind::MmioRead {
                        base: Box::new(base),
                        offset: reg.offset,
                        elem_bits: reg.elem_bits,
                        index: Some(Box::new(idx)),
                        array_len: reg.array_len,
                    },
                    reg_ty,
                ));
            }
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
            let base = unwrap_ok_result_expr(be.lower_sub(target, None)?);
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
        ExprKind::StructLit {
            module,
            name,
            fields,
            ..
        } => {
            let resolved_name = if let Some(mod_id) = module {
                be.syms()
                    .candidate_module_names(&mod_id.name, &name.name)
                    .into_iter()
                    .find(|c| be.syms().structs.contains_key(c))
                    .unwrap_or_else(|| format!("{}.{}", mod_id.name, name.name))
            } else {
                name.name.clone()
            };
            let st = be.syms().structs.get(&resolved_name).cloned();
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
                    struct_name: resolved_name.clone(),
                    packed,
                    fields: mfields,
                },
                MirType::Struct(resolved_name),
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
        ExprKind::Call { callee, args, .. } => {
            let mut largs = Vec::new();
            if let ExprKind::Ident(id) = &callee.kind {
                if id.name == "fmt_buf" {
                    return lower_fmt_buf_call_be(args, expr.span, be);
                }
                if matches!(id.name.as_str(), "now_ticks" | "now_ms") {
                    return Ok(MirExpr::new(MirExprKind::NowTicks, MirType::U32));
                }
                if id.name == "checkpoint" && be.in_noyield() {
                    return Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void));
                }
                if let Some(sig) = be.syms().fns.get(&id.name).cloned() {
                    if sig.spawns_in_boot {
                        return Err(MirLowerError {
                            message: format!(
                                "boot helper `{}` (contains `spawn` or IRQ `.attach`) may only be called during `fn main()` boot",
                                id.name
                            ),
                            span: expr.span,
                        });
                    }
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
                    let target_callee =
                        if be.in_noyield() && sig.effect == MirFnEffect::SoftYielding {
                            format!("{}__noyield", id.name)
                        } else {
                            id.name.clone()
                        };
                    return Ok(MirExpr::new(
                        MirExprKind::CallDirect {
                            callee: target_callee,
                            is_extern_c: sig.is_extern_c,
                            args: largs,
                        },
                        ret,
                    ));
                }
                if let Some(vid) = be.scopes().lookup(&id.name) {
                    if let MirType::FnPtr { params, ret } = be.vars()[vid.0].ty.clone() {
                        let fp_expr = MirExpr::new(
                            MirExprKind::Var(vid),
                            MirType::FnPtr {
                                params: params.clone(),
                                ret: ret.clone(),
                            },
                        );
                        return lower_indirect_fnptr_call_be(fp_expr, &params, *ret, args, be);
                    }
                }
                for a in args {
                    largs.push(unwrap_ok_result_expr(be.lower_sub(&a.value, None)?));
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
            if let Ok(ce) = be.lower_sub(callee, None) {
                if let MirType::FnPtr { params, ret } = ce.ty.clone() {
                    return lower_indirect_fnptr_call_be(ce, &params, *ret, args, be);
                }
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
                    let s = coerce_expr_if_needed(be.lower_sub(sarg, Some(&slice_ty))?, &slice_ty);
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
                if id.name == "sys" {
                    match method.name.as_str() {
                        "now_ms" | "now_ticks" => {
                            return Ok(MirExpr::new(MirExprKind::NowTicks, MirType::U32));
                        }
                        "current_prio" => {
                            return Ok(MirExpr::new(
                                MirExprKind::SysQuery(MirSysQuery::CurrentPrio),
                                MirType::U32,
                            ));
                        }
                        "sleep_depth" => {
                            return Ok(MirExpr::new(
                                MirExprKind::SysQuery(MirSysQuery::SleepDepth),
                                MirType::U32,
                            ));
                        }
                        "wfi_count" => {
                            return Ok(MirExpr::new(
                                MirExprKind::SysQuery(MirSysQuery::WfiCount),
                                MirType::U32,
                            ));
                        }
                        _ => {}
                    }
                }
            }
            if let Some((target, pkind)) = resolve_prim_receiver_be(receiver, be)? {
                return finish_prim_read_method(target, pkind, &method.name, args, expr.span, be);
            }
            if let Ok(recv_me) = be.lower_sub(receiver, None) {
                let (st_name_opt, is_ptr) = match &recv_me.ty {
                    MirType::Struct(s) => (Some(s.clone()), false),
                    MirType::Ptr(inner) => match inner.as_ref() {
                        MirType::Struct(s) => (Some(s.clone()), true),
                        _ => (None, false),
                    },
                    _ => (None, false),
                };
                if let Some(sname) = st_name_opt {
                    if let Some((fty, is_packed)) = be
                        .syms()
                        .structs
                        .get(&sname)
                        .and_then(|st| {
                            st.fields
                                .iter()
                                .find(|f| f.name == method.name)
                                .map(|f| (f.ty.clone(), st.packed))
                        })
                    {
                        if let MirType::FnPtr { params, ret } = fty.clone() {
                            let fp_expr = MirExpr::new(
                                MirExprKind::FieldAccess {
                                    base: Box::new(recv_me),
                                    struct_name: sname,
                                    is_ptr,
                                    packed: is_packed,
                                    field: method.name.clone(),
                                    raw_field_ty: fty.clone(),
                                },
                                fty,
                            );
                            return lower_indirect_fnptr_call_be(fp_expr, &params, *ret, args, be);
                        }
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

fn lower_indirect_fnptr_call_be<B: LowerBackend>(
    fp_expr: MirExpr,
    params: &[MirType],
    ret: MirType,
    args: &[CallArg],
    be: &mut B,
) -> Result<MirExpr, MirLowerError> {
    let fp_tmp = be.alloc_temp_be("fnptr_call", fp_expr.ty.clone());
    be.emit_stmt_be(MirStmt::Assign {
        target: MirPlace::Var(fp_tmp),
        op: MirAssignOp::Assign,
        value: fp_expr,
    });
    let callee_name = be.vars()[fp_tmp.0].name.clone();
    let mut largs = Vec::with_capacity(args.len());
    for (idx, a) in args.iter().enumerate() {
        let me = be.lower_sub(&a.value, params.get(idx))?;
        let coerced = match (&me.ty, params.get(idx), &me.kind) {
            (MirType::Array { elem, len }, Some(MirType::Slice(_)), MirExprKind::Var(avid)) => {
                MirExpr::new(
                    MirExprKind::ArrayToSlice {
                        array_place: Box::new(MirPlace::Var(*avid)),
                        len: *len,
                        elem_ty: *elem.clone(),
                    },
                    params[idx].clone(),
                )
            }
            (_, Some(pty), _) => coerce_expr_if_needed(me, pty),
            _ => unwrap_ok_result_expr(me),
        };
        largs.push(coerced);
    }
    Ok(MirExpr::new(
        MirExprKind::CallDirect {
            callee: callee_name,
            is_extern_c: true,
            args: largs,
        },
        ret,
    ))
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
            if sub.ty.is_float() {
                let fty = sub.ty.host_value_type();
                if let MirExprKind::FloatLit(bits) = sub.kind {
                    let neg_val = -f64::from_bits(bits);
                    let neg_bits = if fty == MirType::F32 {
                        normalize_f32_bits(neg_val.to_bits())
                    } else {
                        neg_val.to_bits()
                    };
                    return Ok(MirExpr::new(MirExprKind::FloatLit(neg_bits), fty));
                }
                return Ok(MirExpr::new(
                    MirExprKind::Unary {
                        op: MirUnaryOp::Neg,
                        expr: Box::new(sub),
                    },
                    fty,
                ));
            }
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
        ExprKind::Literal(Literal::Int(_))
        | ExprKind::Literal(Literal::Float { suffix: None, .. }) => true,
        ExprKind::Unary {
            op: UnaryOp::Neg,
            expr: inner,
        } => matches!(
            inner.kind,
            ExprKind::Literal(Literal::Int(_)) | ExprKind::Literal(Literal::Float { suffix: None, .. })
        ),
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
    // Unify float literal precision if one side is `f32` and the other is a `FloatLit`
    if lhs.ty == MirType::F32 && matches!(rhs.kind, MirExprKind::FloatLit(_)) {
        rhs = coerce_expr_if_needed(rhs, &MirType::F32);
    } else if rhs.ty == MirType::F32 && matches!(lhs.kind, MirExprKind::FloatLit(_)) {
        lhs = coerce_expr_if_needed(lhs, &MirType::F32);
    } else if is_signed_int_mir_type(&rhs.ty) && lhs.ty.is_unsigned() && is_scalar_int_mir_type(&lhs.ty) {
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
    let base = unwrap_ok_result_expr(base);
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
    if let MirType::MmioPtr(layout_name) = &base.ty {
        let reg = lookup_mmio_reg(syms, layout_name, field, span)?;
        if reg.array_len.is_some() {
            return Err(MirLowerError {
                message: format!(
                    "MMIO register array `{}` on `{layout_name}` must be indexed (`[i]`)",
                    reg.name
                ),
                span,
            });
        }
        if reg.access == MirMmioAccess::Wo {
            return Err(MirLowerError {
                message: format!(
                    "cannot read write-only `@wo` MMIO register `{}` on `{layout_name}`",
                    reg.name
                ),
                span,
            });
        }
        let elem_ty = mmio_elem_mir_type(reg.elem_bits);
        return Ok(MirExpr::new(
            MirExprKind::MmioRead {
                base: Box::new(base),
                offset: reg.offset,
                elem_bits: reg.elem_bits,
                index: None,
                array_len: None,
            },
            elem_ty,
        ));
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
    let base = unwrap_ok_result_expr(base);
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

fn resolve_prim_receiver_be<B: LowerBackend>(
    receiver: &Expr,
    be: &mut B,
) -> Result<Option<(MirPrimTarget, MirPrimKind)>, MirLowerError> {
    if let ExprKind::Ident(id) = &receiver.kind {
        if let Some(pinfo) = be.syms().primitives.get(&id.name) {
            return Ok(Some((
                MirPrimTarget::Global(id.name.clone()),
                pinfo.kind.clone(),
            )));
        }
        if let Some(vid) = be.scopes().lookup(&id.name) {
            let vty = be.vars()[vid.0].ty.clone();
            if let MirType::Ptr(inner) = vty.clone() {
                if let MirType::Primitive(pk) = *inner {
                    return Ok(Some((
                        MirPrimTarget::PtrExpr(MirExpr::new(MirExprKind::Var(vid), vty)),
                        pk,
                    )));
                }
            }
        }
        return Ok(None);
    }
    if let ExprKind::Field { .. } | ExprKind::Unary { op: UnaryOp::Deref, .. } = &receiver.kind {
        if let Ok(rv) = be.lower_sub(receiver, None) {
            match &rv.ty {
                MirType::Ptr(inner) => {
                    if let MirType::Primitive(pk) = inner.as_ref() {
                        let pk_c = pk.clone();
                        return Ok(Some((MirPrimTarget::PtrExpr(rv), pk_c)));
                    }
                }
                MirType::Primitive(pk) => {
                    let pk_c = pk.clone();
                    if let Ok((place, pty)) = be.lower_place_be(receiver) {
                        let ptr = MirExpr::new(
                            MirExprKind::AddrOfPlace(Box::new(place)),
                            MirType::Ptr(Box::new(pty)),
                        );
                        return Ok(Some((MirPrimTarget::PtrExpr(ptr), pk_c)));
                    }
                }
                _ => {}
            }
        }
    }
    Ok(None)
}

fn is_checkpoint_call_expr(expr: &Expr) -> bool {
    matches!(
        &expr.kind,
        ExprKind::Call { callee, args, .. }
            if args.is_empty() && matches!(&callee.kind, ExprKind::Ident(id) if id.name == "checkpoint")
    )
}

fn extract_sys_exit_arg(expr: &Expr) -> Option<Option<&Expr>> {
    match &expr.kind {
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } if matches!(&receiver.kind, ExprKind::Ident(id) if id.name == "sys")
            && method.name == "exit" =>
        {
            Some(first_positional_arg(args))
        }
        ExprKind::Call { callee, args, .. } => match &callee.kind {
            ExprKind::Field { target, field }
                if matches!(&target.kind, ExprKind::Ident(id) if id.name == "sys")
                    && field.name == "exit" =>
            {
                Some(first_positional_arg(args))
            }
            _ => None,
        },
        _ => None,
    }
}

fn ast_stmts_have_return(stmts: &[Stmt]) -> bool {
    for s in stmts {
        match &s.kind {
            StmtKind::Return(_) => return true,
            StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                if ast_stmts_have_return(&then_block.stmts)
                    || else_block
                        .as_ref()
                        .is_some_and(|eb| ast_stmts_have_return(&eb.stmts))
                {
                    return true;
                }
            }
            StmtKind::Switch { arms, .. } => {
                if arms.iter().any(|a| ast_stmts_have_return(&a.body.stmts)) {
                    return true;
                }
            }
            StmtKind::While { body, .. }
            | StmtKind::Loop { body }
            | StmtKind::LoopEvery { body, .. }
            | StmtKind::ForRange { body, .. }
            | StmtKind::ForEach { body, .. }
            | StmtKind::NoYield { body }
            | StmtKind::Critical { body }
            | StmtKind::Block(body) => {
                if ast_stmts_have_return(&body.stmts) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn finish_prim_read_method<B: LowerBackend>(
    target: MirPrimTarget,
    pkind: MirPrimKind,
    method: &str,
    args: &[CallArg],
    span: Span,
    be: &mut B,
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
        ("try_send", MirPrimKind::Inbox { elem, cap }) => {
            let v = if let Some(first) = first_positional_arg(args) {
                let raw = be.lower_sub(first, Some(&elem))?;
                coerce_expr_if_needed(raw, &elem)
            } else {
                MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
            };
            let boost = extract_boost_arg(args);
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxTrySend {
                    target,
                    elem_ty: *elem,
                    cap,
                    value: Box::new(v),
                    boost,
                })),
                MirType::Bool,
            ))
        }
        ("len", MirPrimKind::Inbox { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxLen {
                target,
                elem_ty: *elem,
                cap,
            })),
            MirType::Usize,
        )),
        ("is_empty", MirPrimKind::Inbox { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxIsEmpty {
                target,
                elem_ty: *elem,
                cap,
            })),
            MirType::Bool,
        )),
        ("is_full", MirPrimKind::Inbox { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxIsFull {
                target,
                elem_ty: *elem,
                cap,
            })),
            MirType::Bool,
        )),
        ("is_set", MirPrimKind::Signal) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::SignalIsSet { target })),
            MirType::Bool,
        )),
        ("try_wait", MirPrimKind::Signal) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::SignalTryWait { target })),
            MirType::Bool,
        )),
        ("half_len" | "cap", MirPrimKind::PingPong { cap, .. }) => {
            Ok(MirExpr::new(MirExprKind::IntLit(cap as i128), MirType::Usize))
        }
        ("dma_buf" | "active_buf", MirPrimKind::PingPong { elem, cap }) => {
            let slice_ty = MirType::Slice(elem.clone());
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::PingPongDmaBuf {
                    target,
                    elem_ty: *elem,
                    cap,
                })),
                slice_ty,
            ))
        }
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
        ExprKind::Call { callee, args, .. } => {
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
                if matches!(id.name.as_str(), "print" | "println") {
                    let newline = id.name == "println";
                    if let Some(first) = first_positional_arg(args) {
                        if let Some(pfmt) = try_lower_print_arg_be(first, newline, be)? {
                            return Ok(Some(pfmt));
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
            // 1. `sys.*` intrinsics (excluding pure value queries `now_ms`, `now_ticks`, `current_prio`, `sleep_depth`, `wfi_count`)
            if let ExprKind::Ident(id) = &receiver.kind {
                if id.name == "sys"
                    && !matches!(
                        method.name.as_str(),
                        "now_ms" | "now_ticks" | "current_prio" | "sleep_depth" | "wfi_count"
                    )
                {
                    if matches!(method.name.as_str(), "log" | "println" | "print" | "puts") {
                        let newline = matches!(method.name.as_str(), "log" | "println");
                        if let Some(first) = first_positional_arg(args) {
                            if let Some(pfmt) = try_lower_print_arg_be(first, newline, be)? {
                                return Ok(Some(pfmt));
                            }
                        }
                    }
                    if method.name == "use_systick" {
                        let hz = if let Some(first) = first_positional_arg(args) {
                            be.lower_sub(first, Some(&MirType::U32))?
                        } else {
                            MirExpr::new(MirExprKind::IntLit(1_000_000), MirType::U32)
                        };
                        return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::UseSystick { hz })));
                    }
                    if method.name == "pend_irq" && args.len() == 1 {
                        if let Some((_, num)) = resolve_irq_token_be(&args[0].value, be) {
                            return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                                method: "pend_irq".to_string(),
                                args: vec![MirExpr::new(
                                    MirExprKind::IntLit(num as i128),
                                    MirType::U32,
                                )],
                            })));
                        }
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

            // 2. Non-blocking OS & DMA primitive operations
            if let Some((target, pkind)) = resolve_prim_receiver_be(receiver, be)? {
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

// ─────────────────────────────────────────────────────────────────────────────
// Zero-Heap `f"..."` Format Unpacking, `{pkt:?}` Struct Expansion & `fmt_buf`
// ─────────────────────────────────────────────────────────────────────────────

fn push_fmt_lit(chunks: &mut Vec<MirFmtChunk>, s: &str) {
    if s.is_empty() {
        return;
    }
    if let Some(MirFmtChunk::Literal(prev)) = chunks.last_mut() {
        prev.push_str(s);
    } else {
        chunks.push(MirFmtChunk::Literal(s.to_string()));
    }
}

fn try_lower_print_arg_be<B: LowerBackend>(
    arg: &Expr,
    newline: bool,
    be: &mut B,
) -> Result<Option<MirStmt>, MirLowerError> {
    if let ExprKind::FormatString(parts) = &arg.kind {
        let chunks = lower_format_parts_be(parts, be)?;
        return Ok(Some(MirStmt::PrintFmt { chunks, newline }));
    }
    // Leave standard string literals and `*u8` C strings on the existing `print` / `sys.log` path,
    // while routing `[]u8` / `[N]u8` byte slices (e.g. returned from `fmt_buf`) through `PrintFmt`!
    if matches!(&arg.kind, ExprKind::Literal(Literal::String(_))) {
        return Ok(None);
    }
    if let Ok(lowered_raw) = be.lower_sub(arg, None) {
        let lowered = unwrap_ok_result_expr(lowered_raw);
        match &lowered.ty {
            MirType::Slice(inner) if **inner == MirType::U8 => {
                return Ok(Some(MirStmt::PrintFmt {
                    chunks: vec![MirFmtChunk::Bytes(lowered)],
                    newline,
                }));
            }
            MirType::Array { elem, len } if **elem == MirType::U8 => {
                let arr_len = *len;
                let slice_expr = array_expr_to_slice_be(lowered, MirType::U8, arr_len, be);
                return Ok(Some(MirStmt::PrintFmt {
                    chunks: vec![MirFmtChunk::Bytes(slice_expr)],
                    newline,
                }));
            }
            _ => {
                // Note: if `lowered` is e.g. a `*u8` variable/call, do NOT re-evaluate side effects:
                // emit `PrintFmt` with `MirFmtChunk::Str(lowered)`!
                if matches!(
                    &lowered.ty,
                    MirType::Str | MirType::Ptr(_)
                ) {
                    return Ok(Some(MirStmt::PrintFmt {
                        chunks: vec![MirFmtChunk::Str(lowered)],
                        newline,
                    }));
                }
            }
        }
    }
    Ok(None)
}

fn array_expr_to_slice_be<B: LowerBackend>(
    arr_expr: MirExpr,
    elem_ty: MirType,
    len: usize,
    be: &mut B,
) -> MirExpr {
    let slice_ty = MirType::Slice(Box::new(elem_ty.clone()));
    if let MirExprKind::Var(vid) = arr_expr.kind {
        return MirExpr::new(
            MirExprKind::ArrayToSlice {
                array_place: Box::new(MirPlace::Var(vid)),
                len,
                elem_ty,
            },
            slice_ty,
        );
    }
    let tmp = be.alloc_temp_be("fmt_arr", arr_expr.ty.clone());
    be.emit_stmt_be(MirStmt::Assign {
        target: MirPlace::Var(tmp),
        op: MirAssignOp::Assign,
        value: arr_expr,
    });
    MirExpr::new(
        MirExprKind::ArrayToSlice {
            array_place: Box::new(MirPlace::Var(tmp)),
            len,
            elem_ty,
        },
        slice_ty,
    )
}

fn lower_fmt_buf_dst_be<B: LowerBackend>(
    dst_ast: &Expr,
    be: &mut B,
) -> Result<MirExpr, MirLowerError> {
    let u8_slice_ty = MirType::Slice(Box::new(MirType::U8));
    if let Ok((place, MirType::Array { elem, len })) = be.lower_place_be(dst_ast) {
        if *elem == MirType::U8 {
            return Ok(MirExpr::new(
                MirExprKind::ArrayToSlice {
                    array_place: Box::new(place),
                    len,
                    elem_ty: MirType::U8,
                },
                u8_slice_ty,
            ));
        }
    }
    let dv = be.lower_sub(dst_ast, Some(&u8_slice_ty))?;
    match dv.ty.clone() {
        MirType::Slice(_) => Ok(dv),
        MirType::Array { elem, len } => Ok(array_expr_to_slice_be(dv, *elem, len, be)),
        MirType::Ptr(inner) => match *inner {
            MirType::Slice(_) => Ok(MirExpr::new(
                MirExprKind::Unary {
                    op: MirUnaryOp::Deref,
                    expr: Box::new(dv),
                },
                u8_slice_ty,
            )),
            MirType::Array { elem, len } => {
                let arr_ty = MirType::Array {
                    elem: elem.clone(),
                    len,
                };
                Ok(MirExpr::new(
                    MirExprKind::ArrayToSlice {
                        array_place: Box::new(MirPlace::Deref(Box::new(dv), arr_ty)),
                        len,
                        elem_ty: *elem,
                    },
                    u8_slice_ty,
                ))
            }
            _ => Ok(dv),
        },
        _ => Ok(dv),
    }
}

fn lower_fmt_buf_call_be<B: LowerBackend>(
    args: &[CallArg],
    span: Span,
    be: &mut B,
) -> Result<MirExpr, MirLowerError> {
    let dst_arg = args.first().ok_or_else(|| MirLowerError {
        message: "`fmt_buf` expects 2 arguments `(dst, f\"...\")`".to_string(),
        span,
    })?;
    let fmt_arg = args.get(1).ok_or_else(|| MirLowerError {
        message: "`fmt_buf` expects a second `f\"...\"` format string argument".to_string(),
        span,
    })?;
    let buf_slice = lower_fmt_buf_dst_be(&dst_arg.value, be)?;
    let chunks = match &fmt_arg.value.kind {
        ExprKind::FormatString(parts) => lower_format_parts_be(parts, be)?,
        ExprKind::Literal(Literal::String(s)) => {
            if s.is_empty() {
                Vec::new()
            } else {
                vec![MirFmtChunk::Literal(s.clone())]
            }
        }
        _ => {
            let v = be.lower_sub(&fmt_arg.value, None)?;
            let mut out = Vec::new();
            emit_fmt_value_be(v, FormatSpec::Default, fmt_arg.span, &mut out, be)?;
            out
        }
    };
    let res_ty = MirType::Result {
        ok: Box::new(MirType::Slice(Box::new(MirType::U8))),
        err: Some(Box::new(MirType::Enum("FmtErr".to_string()))),
    };
    let dst_res = be.alloc_temp_be("fmt_res", res_ty.clone());
    be.emit_stmt_be(MirStmt::FmtBuf {
        dst_res,
        buf: buf_slice,
        chunks,
    });
    Ok(MirExpr::new(MirExprKind::Var(dst_res), res_ty))
}

fn lower_format_parts_be<B: LowerBackend>(
    parts: &[FormatPart],
    be: &mut B,
) -> Result<Vec<MirFmtChunk>, MirLowerError> {
    let mut chunks = Vec::new();
    for part in parts {
        match part {
            FormatPart::Literal(s) => push_fmt_lit(&mut chunks, s),
            FormatPart::Hole { expr, spec, span } => {
                let expected = match spec {
                    FormatSpec::Fixed { .. } | FormatSpec::Sci { .. } => Some(&MirType::F64),
                    _ => None,
                };
                // Only pass `F64` hint when `expr` is an unsuffixed float literal; otherwise keep `expr`'s native type (`f32` vs `f64`).
                let hint = if matches!(
                    &expr.kind,
                    ExprKind::Literal(Literal::Float { suffix: None, .. })
                ) {
                    expected
                } else {
                    None
                };
                let val = be.lower_sub(expr, hint)?;
                emit_fmt_value_be(val, *spec, *span, &mut chunks, be)?;
            }
        }
    }
    Ok(chunks)
}

fn emit_ptr_hex_debug(val: MirExpr, chunks: &mut Vec<MirFmtChunk>) {
    push_fmt_lit(chunks, "0x");
    let cast_usize = MirExpr::new(
        MirExprKind::Cast {
            expr: Box::new(val),
            target_ty: MirType::Usize,
        },
        MirType::Usize,
    );
    chunks.push(MirFmtChunk::Int {
        value: cast_usize,
        signed: false,
        bits: 0,
        fmt: MirIntFmt::Hex {
            upper: false,
            zero_pad_width: 0,
        },
    });
}

fn emit_fmt_value_be<B: LowerBackend>(
    val: MirExpr,
    spec: FormatSpec,
    span: Span,
    chunks: &mut Vec<MirFmtChunk>,
    be: &mut B,
) -> Result<(), MirLowerError> {
    let val = unwrap_ok_result_expr(val);
    let host_ty = val.ty.host_value_type();
    if spec == FormatSpec::Debug {
        match &host_ty {
            MirType::Struct(sname) => {
                return expand_struct_debug_be(val, sname, span, chunks, be);
            }
            MirType::Ptr(_) => {
                emit_ptr_hex_debug(val, chunks);
                return Ok(());
            }
            MirType::Slice(elem) => {
                chunks.push(MirFmtChunk::DebugSlice {
                    elem_ty: elem.host_value_type(),
                    slice: val,
                });
                return Ok(());
            }
            MirType::Array { elem, len } => {
                let el = *elem.clone();
                if matches!(
                    el.host_value_type(),
                    MirType::Struct(_) | MirType::Array { .. } | MirType::Slice(_)
                ) {
                    return expand_array_debug_be(val, &el, *len, span, chunks, be);
                }
                let slice_expr = array_expr_to_slice_be(val, el.clone(), *len, be);
                chunks.push(MirFmtChunk::DebugSlice {
                    elem_ty: el.host_value_type(),
                    slice: slice_expr,
                });
                return Ok(());
            }
            _ => {}
        }
    }

    match host_ty {
        MirType::Bool => {
            chunks.push(MirFmtChunk::Bool(val));
        }
        MirType::Enum(ename) => {
            chunks.push(MirFmtChunk::EnumName {
                enum_name: ename,
                value: val,
            });
        }
        MirType::Str => {
            chunks.push(MirFmtChunk::Str(val));
        }
        MirType::Ptr(inner) if matches!(*inner, MirType::U8 | MirType::I8) => {
            chunks.push(MirFmtChunk::Str(val));
        }
        MirType::Slice(inner) if *inner == MirType::U8 => {
            chunks.push(MirFmtChunk::Bytes(val));
        }
        MirType::Array { elem, len } if *elem == MirType::U8 => {
            let slice_expr = array_expr_to_slice_be(val, MirType::U8, len, be);
            chunks.push(MirFmtChunk::Bytes(slice_expr));
        }
        MirType::F32 | MirType::F64 => {
            let is_f64 = host_ty == MirType::F64;
            let fmt = match spec {
                FormatSpec::Fixed { precision } => MirFloatFmt::Fixed { precision },
                FormatSpec::Sci { precision } => MirFloatFmt::Sci {
                    precision: precision.unwrap_or(2),
                },
                _ => MirFloatFmt::Fixed { precision: 2 },
            };
            chunks.push(MirFmtChunk::Float {
                value: val,
                is_f64,
                fmt,
            });
        }
        MirType::U8
        | MirType::U16
        | MirType::U32
        | MirType::U64
        | MirType::Usize
        | MirType::I8
        | MirType::I16
        | MirType::I32
        | MirType::I64
        | MirType::Isize => {
            let (signed, bits) = match host_ty {
                MirType::U8 => (false, 8),
                MirType::I8 => (true, 8),
                MirType::U16 => (false, 16),
                MirType::I16 => (true, 16),
                MirType::U32 => (false, 32),
                MirType::I32 => (true, 32),
                MirType::U64 => (false, 64),
                MirType::I64 => (true, 64),
                MirType::Usize => (false, 0),
                MirType::Isize => (true, 0),
                _ => (false, 32),
            };
            let fmt = match spec {
                FormatSpec::Hex { upper, width } => MirIntFmt::Hex {
                    upper,
                    zero_pad_width: width.unwrap_or(0),
                },
                FormatSpec::Bin { width } => MirIntFmt::Bin {
                    zero_pad_width: width.unwrap_or(0),
                },
                _ => MirIntFmt::Dec,
            };
            let coerced_val = if val.ty != host_ty {
                MirExpr::new(
                    MirExprKind::Cast {
                        expr: Box::new(val),
                        target_ty: host_ty.clone(),
                    },
                    host_ty,
                )
            } else {
                val
            };
            chunks.push(MirFmtChunk::Int {
                value: coerced_val,
                signed,
                bits,
                fmt,
            });
        }
        MirType::Struct(sname) => {
            expand_struct_debug_be(val, &sname, span, chunks, be)?;
        }
        _ => {
            emit_ptr_hex_debug(val, chunks);
        }
    }
    Ok(())
}

fn expand_array_debug_be<B: LowerBackend>(
    val: MirExpr,
    elem_ty: &MirType,
    len: usize,
    span: Span,
    chunks: &mut Vec<MirFmtChunk>,
    be: &mut B,
) -> Result<(), MirLowerError> {
    if len == 0 {
        push_fmt_lit(chunks, "[]");
        return Ok(());
    }
    let base_var_expr = match &val.kind {
        MirExprKind::Var(_) => val,
        _ => {
            let tmp = be.alloc_temp_be("fmt_arr_dbg", val.ty.clone());
            let vty = val.ty.clone();
            be.emit_stmt_be(MirStmt::Assign {
                target: MirPlace::Var(tmp),
                op: MirAssignOp::Assign,
                value: val,
            });
            MirExpr::new(MirExprKind::Var(tmp), vty)
        }
    };
    push_fmt_lit(chunks, "[");
    for idx in 0..len {
        if idx > 0 {
            push_fmt_lit(chunks, ", ");
        }
        let elem_expr = MirExpr::new(
            MirExprKind::Index {
                base: Box::new(base_var_expr.clone()),
                index: Box::new(MirExpr::new(MirExprKind::IntLit(idx as i128), MirType::Usize)),
                is_slice: false,
            },
            elem_ty.host_value_type(),
        );
        emit_fmt_value_be(elem_expr, FormatSpec::Debug, span, chunks, be)?;
    }
    push_fmt_lit(chunks, "]");
    Ok(())
}

fn expand_struct_debug_be<B: LowerBackend>(
    val: MirExpr,
    sname: &str,
    span: Span,
    chunks: &mut Vec<MirFmtChunk>,
    be: &mut B,
) -> Result<(), MirLowerError> {
    let st = be.syms().structs.get(sname).cloned().ok_or_else(|| MirLowerError {
        message: format!("unknown struct `{sname}` in `{{:?}}` format hole"),
        span,
    })?;
    if st.fields.is_empty() {
        push_fmt_lit(chunks, &format!("{sname} {{ }}"));
        return Ok(());
    }
    let base_var_expr = match &val.kind {
        MirExprKind::Var(_) => val,
        _ => {
            let tmp = be.alloc_temp_be("fmt_st", val.ty.clone());
            let vty = val.ty.clone();
            be.emit_stmt_be(MirStmt::Assign {
                target: MirPlace::Var(tmp),
                op: MirAssignOp::Assign,
                value: val,
            });
            MirExpr::new(MirExprKind::Var(tmp), vty)
        }
    };
    push_fmt_lit(chunks, &format!("{sname} {{ "));
    for (idx, sf) in st.fields.iter().enumerate() {
        if idx > 0 {
            push_fmt_lit(chunks, ", ");
        }
        push_fmt_lit(chunks, &format!("{}: ", sf.name));
        let fexpr = MirExpr::new(
            MirExprKind::FieldAccess {
                base: Box::new(base_var_expr.clone()),
                struct_name: sname.to_string(),
                packed: st.packed,
                is_ptr: false,
                field: sf.name.clone(),
                raw_field_ty: sf.ty.clone(),
            },
            sf.ty.host_value_type(),
        );
        emit_fmt_value_be(fexpr, FormatSpec::Debug, span, chunks, be)?;
    }
    push_fmt_lit(chunks, " }");
    Ok(())
}

fn collect_formatted_enums(
    isrs: &[MirIsr],
    functions: &[MirFunction],
    blueprints: &[MirTaskBlueprint],
    boot: &MirBootSequence,
) -> Vec<String> {
    let mut set = BTreeSet::new();
    for isr in isrs {
        for s in &isr.body {
            scan_stmt_formatted_enums(s, &mut set);
        }
    }
    for f in functions {
        match &f.body {
            MirFnBody::Direct(stmts) => {
                for s in stmts {
                    scan_stmt_formatted_enums(s, &mut set);
                }
            }
            MirFnBody::Coroutine { blocks, .. } => {
                for b in blocks {
                    for s in &b.stmts {
                        scan_stmt_formatted_enums(s, &mut set);
                    }
                }
            }
        }
    }
    for bp in blueprints {
        for b in &bp.blocks {
            for s in &b.stmts {
                scan_stmt_formatted_enums(s, &mut set);
            }
        }
    }
    for step in &boot.steps {
        if let MirBootStep::Stmt(s) = step {
            scan_stmt_formatted_enums(s, &mut set);
        }
    }
    set.into_iter().collect()
}

fn scan_stmt_formatted_enums(stmt: &MirStmt, out: &mut BTreeSet<String>) {
    match stmt {
        MirStmt::PrintFmt { chunks, .. } | MirStmt::FmtBuf { chunks, .. } => {
            for ch in chunks {
                match ch {
                    MirFmtChunk::EnumName { enum_name, .. }
                    | MirFmtChunk::DebugSlice {
                        elem_ty: MirType::Enum(enum_name),
                        ..
                    } => {
                        out.insert(enum_name.clone());
                    }
                    _ => {}
                }
            }
        }
        MirStmt::If {
            then_body,
            else_body,
            ..
        } => {
            for s in then_body {
                scan_stmt_formatted_enums(s, out);
            }
            for s in else_body {
                scan_stmt_formatted_enums(s, out);
            }
        }
        MirStmt::While { body, .. } => {
            for s in body {
                scan_stmt_formatted_enums(s, out);
            }
        }
        MirStmt::Switch { arms, default, .. } => {
            for (_, b) in arms {
                for s in b {
                    scan_stmt_formatted_enums(s, out);
                }
            }
            for s in default {
                scan_stmt_formatted_enums(s, out);
            }
        }
        _ => {}
    }
}
