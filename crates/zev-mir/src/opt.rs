//! Whole-program visibility analysis, dead-watch optimizer, and singleton constant-folding
//! for `zev-mir`.
//!
//! Computes:
//! - Whole-program singleton (`instance_count == 1`) subsystem, single-spawned task parameter,
//!   uniform-constant function parameter, and single-assignment local constant folding
//!   (`fold_singletons`).
//! - Which top-level `watch<T>` primitives are provably dead (never read/waited directly,
//!   and either never address-taken or no pointer-based `WatchRead` exists for `T`).
//! - Which global OS primitives (`isr_contended_globals`) and mangled primitive families
//!   (`isr_contended_prim_mangles`) can be touched by hardware ISRs (including `MirIrqVector`
//!   `baked_args`, ISR bodies, and transitive ISR callees), versus pure task-to-task
//!   globals (`elided_globals`).
//! - Dead-watch `WatchPublish` stripping (`strip_dead_watch_publishes`), rewriting dead
//!   direct `WatchPublish` statements to `MirStmt::Expr(value)` to preserve any side effects.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use crate::ir::{
    MirAssignOp, MirBakedArg, MirBinOp, MirBlock, MirBootStep, MirCallDst, MirExpr, MirExprKind,
    MirFmtChunk, MirFnBody, MirFunction, MirIntrinsicOp, MirModule, MirPlace, MirPrimKind,
    MirPrimNonBlockingOp, MirPrimReadOp, MirPrimTarget, MirPrimViewKind, MirStmt, MirStruct,
    MirTerminator, MirType, MirUnaryOp, MirVar, MirWaitArmOp, MirYieldKind, VarId, VarRole,
};

/// Whole-program primitive contention, dead-watch, and singleton folding analysis report.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VisibilityReport {
    /// Top-level `watch<T>` primitive names that are never read/waited anywhere.
    pub dead_watches: BTreeSet<String>,
    /// Top-level primitive names whose address was taken via `GlobalPrimAddr` or `MirBakedArg::PrimAddr`.
    pub addr_taken_prims: BTreeSet<String>,
    /// Static SRAM bytes saved by stripping dead watches on 32-bit MCU targets.
    pub dead_watch_bytes_saved_32: usize,
    /// Static SRAM bytes saved by stripping dead watches on 64-bit host targets.
    pub dead_watch_bytes_saved_64: usize,
    /// Top-level primitive names reachable from any ISR, `MirIrqVector` `baked_args`, or transitive ISR callee.
    pub isr_contended_globals: BTreeSet<String>,
    /// `MirPrimKind::mangle_name()` values reachable from any ISR or transitive ISR callee.
    pub isr_contended_prim_mangles: BTreeSet<String>,
    /// Non-dead top-level primitives that are pure task-to-task (never touched by an ISR).
    pub elided_globals: BTreeSet<String>,
    /// Static subsystem global names and unique nested paths whose struct type has `instance_count == 1`.
    pub singleton_subsystems: BTreeSet<String>,
    /// Subsystem struct names (`S`) whose whole-program static instance count is `== 1` -> unique path.
    pub singleton_subsystem_structs: BTreeMap<String, String>,
    /// Folded parameter/local names per task blueprint (`blueprint_name -> [var_name, ...]`).
    pub folded_task_vars: BTreeMap<String, Vec<String>>,
    /// Folded parameter/local `VarId.0` set per task blueprint (`blueprint_name -> {var_id, ...}`).
    pub folded_task_var_ids: BTreeMap<String, BTreeSet<usize>>,
    /// Folded parameter/local names per function (`fn_name -> [var_name, ...]`).
    pub folded_fn_vars: BTreeMap<String, Vec<String>>,
    /// Static SRAM bytes saved per spawned task instance `(saved_32, saved_64)`.
    pub singleton_task_bytes_saved: BTreeMap<String, (usize, usize)>,
    /// Total static SRAM bytes saved by singleton folding on 32-bit MCU targets.
    pub singleton_bytes_saved_32: usize,
    /// Total static SRAM bytes saved by singleton folding on 64-bit host targets.
    pub singleton_bytes_saved_64: usize,
}

impl VisibilityReport {
    /// True iff `name` is an unread dead `watch<T>` whose address is never taken,
    /// so its `.bss` global storage can be stripped to `0 B`.
    pub fn can_strip_bss(&self, name: &str) -> bool {
        self.dead_watches.contains(name) && !self.addr_taken_prims.contains(name)
    }

    /// Return `module.visibility` when populated, or compute it on demand for
    /// manually constructed `MirModule` values (e.g. unit-test fixtures).
    pub fn effective_for(module: &MirModule) -> Self {
        if module.visibility.dead_watches.is_empty()
            && module.visibility.isr_contended_globals.is_empty()
            && module.visibility.isr_contended_prim_mangles.is_empty()
            && module.visibility.elided_globals.is_empty()
            && module.visibility.addr_taken_prims.is_empty()
            && (!module.primitives.is_empty()
                || !module.subsystems.is_empty()
                || !module.isrs.is_empty()
                || !module.irq_vectors.is_empty())
        {
            analyze_visibility(module)
        } else {
            module.visibility.clone()
        }
    }

    /// True if any ISR can access primitives of concrete family `pk` (directly or via unsized views).
    pub fn is_prim_family_contended(&self, pk: &MirPrimKind) -> bool {
        if self.isr_contended_prim_mangles.contains(&pk.mangle_name()) {
            return true;
        }
        match pk {
            MirPrimKind::ByteRing { .. } => {
                self.isr_contended_prim_mangles.contains("view_byte_ring")
            }
            MirPrimKind::Inbox { elem, .. } => self
                .isr_contended_prim_mangles
                .contains(&MirPrimViewKind::Inbox { elem: elem.clone() }.mangle_name()),
            MirPrimKind::Pool { elem, .. } => self
                .isr_contended_prim_mangles
                .contains(&format!("lease_release_{}", elem.mangle_name())),
            _ => false,
        }
    }

    /// True if any ISR can touch a `pool<T, K>` or `Lease<T>` (used for shared `__syn_pool_release` IRQ elision).
    pub fn any_pool_or_lease_contended(&self) -> bool {
        self.isr_contended_prim_mangles
            .iter()
            .any(|m| m.starts_with("pool_") || m.starts_with("lease_release_"))
    }

    /// True if `*inbox<elem>` views may alias an ISR-contended `inbox<elem, N>`.
    pub fn is_inbox_view_contended(&self, elem: &MirType) -> bool {
        let vm = MirPrimViewKind::Inbox {
            elem: Box::new(elem.clone()),
        }
        .mangle_name();
        let prefix = format!("inbox_{}_", elem.mangle_name());
        self.isr_contended_prim_mangles
            .iter()
            .any(|m| m == &vm || m.starts_with(&prefix))
    }

    /// True if `*byte_ring` views may alias any ISR-contended `byte_ring<N>`.
    pub fn is_byte_ring_view_contended(&self) -> bool {
        self.isr_contended_prim_mangles
            .iter()
            .any(|m| m == "view_byte_ring" || m.starts_with("byte_ring_"))
    }
}

const fn align_up(offset: usize, align: usize) -> usize {
    if align <= 1 {
        offset
    } else {
        (offset + align - 1) & !(align - 1)
    }
}

fn struct_size_align(fields: &[(usize, usize)], packed: bool) -> (usize, usize) {
    if fields.is_empty() {
        return (1, 1);
    }
    if packed {
        let total: usize = fields.iter().map(|(s, _)| *s).sum();
        return (total.max(1), 1);
    }
    let mut offset = 0usize;
    let mut max_align = 1usize;
    for &(sz, al) in fields {
        let a = al.max(1);
        max_align = max_align.max(a);
        offset = align_up(offset, a);
        offset += sz;
    }
    (align_up(offset, max_align).max(1), max_align)
}

fn type_size_align(
    ty: &MirType,
    parent_dma_safe: bool,
    ptr_bytes: usize,
    structs: &BTreeMap<&str, &MirStruct>,
    dma_layout_structs: &BTreeSet<String>,
    seen_structs: &mut BTreeSet<String>,
) -> (usize, usize) {
    match ty {
        MirType::Void => (0, 1),
        MirType::Bool | MirType::U8 | MirType::I8 => (1, 1),
        MirType::U16 | MirType::I16 => (2, 2),
        MirType::U32 | MirType::I32 | MirType::F32 | MirType::Enum(_) => (4, 4),
        MirType::U64 | MirType::I64 | MirType::F64 => (8, 8),
        MirType::Usize
        | MirType::Isize
        | MirType::Str
        | MirType::Ptr(_)
        | MirType::MmioPtr(_)
        | MirType::FnPtr { .. } => (ptr_bytes, ptr_bytes),
        MirType::EndianInt { bits, .. } => ((*bits as usize) / 8, 1),
        MirType::Struct(name) => {
            if let Some(st) = structs.get(name.as_str()) {
                if !seen_structs.insert(name.clone()) {
                    return (4, 4);
                }
                let struct_dma = parent_dma_safe || dma_layout_structs.contains(name.as_str());
                let fl: Vec<(usize, usize)> = st
                    .fields
                    .iter()
                    .map(|f| {
                        let eff_dma = struct_dma || f.dma_safe;
                        match &f.ty {
                            MirType::Primitive(pk) => watch_or_prim_size_align(
                                pk,
                                eff_dma,
                                ptr_bytes,
                                structs,
                                dma_layout_structs,
                                seen_structs,
                            ),
                            other => type_size_align(
                                other,
                                eff_dma,
                                ptr_bytes,
                                structs,
                                dma_layout_structs,
                                seen_structs,
                            ),
                        }
                    })
                    .collect();
                seen_structs.remove(name.as_str());
                struct_size_align(&fl, st.packed)
            } else {
                (4, 4)
            }
        }
        MirType::Slice(_) => (ptr_bytes * 2, ptr_bytes),
        MirType::Lease(_) => (ptr_bytes * 3, ptr_bytes),
        MirType::Array { elem, len } => {
            let (sz, al) = type_size_align(
                elem,
                parent_dma_safe,
                ptr_bytes,
                structs,
                dma_layout_structs,
                seen_structs,
            );
            (sz * len, al)
        }
        MirType::Primitive(pk) => watch_or_prim_size_align(
            pk,
            parent_dma_safe,
            ptr_bytes,
            structs,
            dma_layout_structs,
            seen_structs,
        ),
        MirType::PrimView(MirPrimViewKind::ByteRing) => (ptr_bytes * 2, ptr_bytes),
        MirType::PrimView(MirPrimViewKind::Inbox { .. }) => (ptr_bytes * 3, ptr_bytes),
        MirType::Result { ok, err } => {
            let mut fields = vec![(1, 1)];
            if let Some(et) = err {
                if !et.is_void() {
                    fields.push(type_size_align(
                        et,
                        false,
                        ptr_bytes,
                        structs,
                        dma_layout_structs,
                        seen_structs,
                    ));
                }
            }
            if !ok.is_void() {
                fields.push(type_size_align(
                    ok,
                    false,
                    ptr_bytes,
                    structs,
                    dma_layout_structs,
                    seen_structs,
                ));
            }
            struct_size_align(&fields, false)
        }
    }
}

fn watch_or_prim_size_align(
    kind: &MirPrimKind,
    dma_safe: bool,
    ptr_bytes: usize,
    structs: &BTreeMap<&str, &MirStruct>,
    dma_layout_structs: &BTreeSet<String>,
    seen_structs: &mut BTreeSet<String>,
) -> (usize, usize) {
    let raw = match kind {
        MirPrimKind::Signal => struct_size_align(&[(1, 1), (1, 1), (4, 4)], false),
        MirPrimKind::WaitQueue => struct_size_align(&[(4, 4)], false),
        MirPrimKind::Watch { elem } => {
            let el = type_size_align(
                elem,
                false,
                ptr_bytes,
                structs,
                dma_layout_structs,
                seen_structs,
            );
            struct_size_align(&[el, (4, 4), (4, 4)], false)
        }
        MirPrimKind::Guarded { elem } => {
            let el = type_size_align(
                elem,
                false,
                ptr_bytes,
                structs,
                dma_layout_structs,
                seen_structs,
            );
            struct_size_align(&[el], false)
        }
        MirPrimKind::Inbox { elem, cap } => {
            let el = type_size_align(
                elem,
                false,
                ptr_bytes,
                structs,
                dma_layout_structs,
                seen_structs,
            );
            struct_size_align(
                &[
                    (4, 4),
                    (4, 4),
                    (4, 4),
                    (4, 4),
                    (4, 4),
                    (*cap, 1),
                    (el.0 * cap, el.1),
                ],
                false,
            )
        }
        MirPrimKind::PingPong { elem, cap } => {
            let el = type_size_align(
                elem,
                false,
                ptr_bytes,
                structs,
                dma_layout_structs,
                seen_structs,
            );
            struct_size_align(
                &[
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (4, 4),
                    (el.0 * cap * 2, el.1),
                ],
                false,
            )
        }
        MirPrimKind::Pool { elem, cap } => {
            let hdr = struct_size_align(
                &[(4, 4), (4, 4), (ptr_bytes, ptr_bytes), (1, 1), (1, 1)],
                false,
            );
            let el = type_size_align(
                elem,
                false,
                ptr_bytes,
                structs,
                dma_layout_structs,
                seen_structs,
            );
            let elem_sz = el.0.max(1);
            let elem_al = el.1.max(1);
            if dma_safe {
                let slot_al = elem_al.max(32);
                let slot_stride = align_up(elem_sz, slot_al);
                let hdr_aligned = align_up(hdr.0, slot_al);
                return (hdr_aligned + slot_stride * *cap, slot_al);
            }
            struct_size_align(&[hdr, (elem_sz * *cap, elem_al)], false)
        }
        MirPrimKind::DmaSlot { .. } => struct_size_align(
            &[
                (ptr_bytes, ptr_bytes),
                (ptr_bytes, ptr_bytes),
                (1, 1),
                (1, 1),
            ],
            false,
        ),
        MirPrimKind::ByteRing { cap } => struct_size_align(
            &[
                (4, 4),
                (4, 4),
                (4, 4),
                (4, 4),
                (4, 4),
                (4, 4),
                (4, 4),
                (1, 1),
                (*cap, 1),
            ],
            false,
        ),
    };
    if dma_safe {
        let al = raw.1.max(4);
        (align_up(raw.0, al), al)
    } else {
        raw
    }
}

fn watch_byte_size(
    elem: &MirType,
    dma_safe: bool,
    ptr_bytes: usize,
    structs: &BTreeMap<&str, &MirStruct>,
    dma_layout_structs: &BTreeSet<String>,
) -> usize {
    let mut seen = BTreeSet::new();
    let kind = MirPrimKind::Watch {
        elem: Box::new(elem.clone()),
    };
    watch_or_prim_size_align(
        &kind,
        dma_safe,
        ptr_bytes,
        structs,
        dma_layout_structs,
        &mut seen,
    )
    .0
}

fn is_benign_c_builtin(name: &str) -> bool {
    matches!(
        name,
        "assert" | "printf" | "puts" | "putchar" | "print" | "println"
    ) || name.starts_with("syn_sim_")
}

#[derive(Default)]
struct ScanAcc {
    direct_watch_reads: BTreeSet<String>,
    ptr_watch_read_elems: HashSet<MirType>,
    addr_taken_prims: BTreeSet<String>,
    referenced_prims: BTreeSet<String>,
    referenced_subsystems: BTreeSet<String>,
    isr_contended_globals: BTreeSet<String>,
    isr_contended_prim_mangles: BTreeSet<String>,
    isr_fnptr_sigs: Vec<(Vec<MirType>, MirType)>,
    isr_calls_real_extern_c: bool,
    sys_isr_callbacks: BTreeSet<String>,
}

struct ScanCtx<'a> {
    prim_kinds: BTreeMap<&'a str, &'a MirPrimKind>,
    structs: BTreeMap<&'a str, &'a MirStruct>,
    cur_vars: Option<&'a [MirVar]>,
    acc: ScanAcc,
}

impl<'a> ScanCtx<'a> {
    fn new(module: &'a MirModule) -> Self {
        let prim_kinds = module
            .primitives
            .iter()
            .map(|p| (p.name.as_str(), &p.kind))
            .collect();
        let structs = module
            .structs
            .iter()
            .map(|s| (s.name.as_str(), s))
            .collect();
        Self {
            prim_kinds,
            structs,
            cur_vars: None,
            acc: ScanAcc::default(),
        }
    }

    fn record_isr_prim_global(&mut self, name: &str, explicit_kind: Option<&MirPrimKind>) {
        self.acc.isr_contended_globals.insert(name.to_string());
        if let Some(pk) = explicit_kind {
            self.acc.isr_contended_prim_mangles.insert(pk.mangle_name());
        } else if let Some(pk) = self.prim_kinds.get(name) {
            self.acc.isr_contended_prim_mangles.insert(pk.mangle_name());
        }
    }

    fn record_sys_isr_callback(
        &mut self,
        arg: Option<&MirExpr>,
        params: &[MirType],
        ret: &MirType,
    ) {
        let Some(mut expr) = arg else {
            return;
        };
        while let MirExprKind::Cast { expr: inner, .. } = &expr.kind {
            expr = inner;
        }
        if let MirExprKind::FnAddr(name) = &expr.kind {
            self.acc.sys_isr_callbacks.insert(name.clone());
        } else {
            let sig = (params.to_vec(), ret.clone());
            if !self.acc.isr_fnptr_sigs.contains(&sig) {
                self.acc.isr_fnptr_sigs.push(sig);
            }
        }
    }

    fn record_prim_types_in_ty(&mut self, ty: &MirType, seen_structs: &mut BTreeSet<String>) {
        match ty {
            MirType::Primitive(pk) => {
                self.acc.isr_contended_prim_mangles.insert(pk.mangle_name());
            }
            MirType::PrimView(vk) => {
                self.acc.isr_contended_prim_mangles.insert(vk.mangle_name());
            }
            MirType::Ptr(inner) | MirType::Slice(inner) | MirType::Array { elem: inner, .. } => {
                self.record_prim_types_in_ty(inner, seen_structs);
            }
            MirType::Lease(inner) => {
                self.acc
                    .isr_contended_prim_mangles
                    .insert(format!("lease_release_{}", inner.mangle_name()));
                self.record_prim_types_in_ty(inner, seen_structs);
            }
            MirType::FnPtr { params, ret } => {
                for p in params {
                    self.record_prim_types_in_ty(p, seen_structs);
                }
                self.record_prim_types_in_ty(ret, seen_structs);
            }
            MirType::Result { ok, err } => {
                self.record_prim_types_in_ty(ok, seen_structs);
                if let Some(e) = err {
                    self.record_prim_types_in_ty(e, seen_structs);
                }
            }
            MirType::Struct(name) => {
                if seen_structs.insert(name.clone()) {
                    if let Some(st) = self.structs.get(name.as_str()).copied() {
                        for f in &st.fields {
                            self.record_prim_types_in_ty(&f.ty, seen_structs);
                        }
                    }
                    seen_structs.remove(name);
                }
            }
            MirType::Void
            | MirType::Bool
            | MirType::U8
            | MirType::U16
            | MirType::U32
            | MirType::U64
            | MirType::I8
            | MirType::I16
            | MirType::I32
            | MirType::I64
            | MirType::Usize
            | MirType::Isize
            | MirType::F32
            | MirType::F64
            | MirType::Str
            | MirType::EndianInt { .. }
            | MirType::Enum(_)
            | MirType::MmioPtr(_) => {}
        }
    }

    fn visit_prim_target(
        &mut self,
        target: &MirPrimTarget,
        kind: &MirPrimKind,
        in_isr: bool,
        callees: &mut BTreeSet<String>,
    ) {
        if in_isr {
            match kind {
                MirPrimKind::DmaSlot { elem } => {
                    self.acc
                        .isr_contended_prim_mangles
                        .insert(format!("lease_release_{}", elem.mangle_name()));
                }
                MirPrimKind::Inbox { elem, .. } => {
                    if let MirType::Lease(inner) = elem.as_ref() {
                        self.acc
                            .isr_contended_prim_mangles
                            .insert(format!("lease_release_{}", inner.mangle_name()));
                    }
                }
                _ => {}
            }
        }
        match target {
            MirPrimTarget::Global(name) => {
                self.acc.referenced_prims.insert(name.clone());
                if in_isr {
                    self.record_isr_prim_global(name, Some(kind));
                }
            }
            MirPrimTarget::PtrExpr(expr) => {
                if in_isr {
                    self.acc
                        .isr_contended_prim_mangles
                        .insert(kind.mangle_name());
                }
                self.visit_expr(expr, in_isr, callees);
            }
        }
    }

    fn visit_place(&mut self, place: &MirPlace, in_isr: bool, callees: &mut BTreeSet<String>) {
        match place {
            MirPlace::Var(_) => {}
            MirPlace::Deref(ptr_expr, _) => self.visit_expr(ptr_expr, in_isr, callees),
            MirPlace::Field { base, .. } => self.visit_place(base, in_isr, callees),
            MirPlace::Index { base, index, .. } => {
                self.visit_place(base, in_isr, callees);
                self.visit_expr(index, in_isr, callees);
            }
        }
    }

    fn visit_fmt_chunk(
        &mut self,
        chunk: &MirFmtChunk,
        in_isr: bool,
        callees: &mut BTreeSet<String>,
    ) {
        match chunk {
            MirFmtChunk::Literal(_) => {}
            MirFmtChunk::Int { value, .. }
            | MirFmtChunk::Float { value, .. }
            | MirFmtChunk::Bool(value)
            | MirFmtChunk::Str(value)
            | MirFmtChunk::Bytes(value)
            | MirFmtChunk::EnumName { value, .. }
            | MirFmtChunk::DebugSlice { slice: value, .. } => {
                self.visit_expr(value, in_isr, callees);
            }
        }
    }

    fn visit_expr(&mut self, expr: &MirExpr, in_isr: bool, callees: &mut BTreeSet<String>) {
        match &expr.kind {
            MirExprKind::IntLit(_)
            | MirExprKind::FloatLit(_)
            | MirExprKind::BoolLit(_)
            | MirExprKind::StringLit(_)
            | MirExprKind::TicksLit(_)
            | MirExprKind::EnumVariant { .. }
            | MirExprKind::ConstRef(_)
            | MirExprKind::MmioAddr(_)
            | MirExprKind::Var(_)
            | MirExprKind::ArrayLen(_)
            | MirExprKind::ResultIsErr(_)
            | MirExprKind::ResultErr(_)
            | MirExprKind::ResultVal(_)
            | MirExprKind::NowTicks
            | MirExprKind::SysQuery(_) => {}
            MirExprKind::GlobalPrimAddr(name) => {
                self.acc.addr_taken_prims.insert(name.clone());
                self.acc.referenced_prims.insert(name.clone());
                if in_isr {
                    self.record_isr_prim_global(name, None);
                }
            }
            MirExprKind::SubsystemAddr(name) => {
                self.acc.referenced_subsystems.insert(name.clone());
                if in_isr {
                    self.acc.isr_contended_globals.insert(name.clone());
                }
            }
            MirExprKind::MakePrimView { ptr, cap, kind } => {
                if in_isr {
                    self.acc
                        .isr_contended_prim_mangles
                        .insert(kind.mangle_name());
                    let concrete = match kind {
                        MirPrimViewKind::ByteRing => MirPrimKind::ByteRing { cap: *cap },
                        MirPrimViewKind::Inbox { elem } => MirPrimKind::Inbox {
                            elem: elem.clone(),
                            cap: *cap,
                        },
                    };
                    self.acc
                        .isr_contended_prim_mangles
                        .insert(concrete.mangle_name());
                }
                self.visit_expr(ptr, in_isr, callees);
            }
            MirExprKind::FnAddr(name) => {
                callees.insert(name.clone());
            }
            MirExprKind::Unary { expr: inner, .. }
            | MirExprKind::Cast { expr: inner, .. }
            | MirExprKind::FieldAccess { base: inner, .. }
            | MirExprKind::SliceLen(inner)
            | MirExprKind::FromBytes { slice: inner, .. }
            | MirExprKind::LeaseGet { lease: inner, .. }
            | MirExprKind::LeaseAddr { lease: inner, .. } => {
                self.visit_expr(inner, in_isr, callees);
            }
            MirExprKind::AddrOfPlace(place)
            | MirExprKind::ArrayToSlice {
                array_place: place, ..
            } => {
                self.visit_place(place, in_isr, callees);
            }
            MirExprKind::Binary { lhs, rhs, .. }
            | MirExprKind::Index {
                base: lhs,
                index: rhs,
                ..
            } => {
                self.visit_expr(lhs, in_isr, callees);
                self.visit_expr(rhs, in_isr, callees);
            }
            MirExprKind::SubSlice {
                base, start, end, ..
            } => {
                self.visit_expr(base, in_isr, callees);
                self.visit_expr(start, in_isr, callees);
                self.visit_expr(end, in_isr, callees);
            }
            MirExprKind::StructLit { fields, .. } => {
                for (_, fe, _) in fields {
                    self.visit_expr(fe, in_isr, callees);
                }
            }
            MirExprKind::ArrayLit { elems, .. } => {
                for el in elems {
                    self.visit_expr(el, in_isr, callees);
                }
            }
            MirExprKind::MmioRead { base, index, .. } => {
                self.visit_expr(base, in_isr, callees);
                if let Some(idx) = index {
                    self.visit_expr(idx, in_isr, callees);
                }
            }
            MirExprKind::CallDirect {
                callee,
                is_extern_c,
                args,
            } => {
                if !*is_extern_c {
                    callees.insert(callee.clone());
                } else if in_isr && callee != "assert" {
                    let fnptr_sig = self.cur_vars.and_then(|vars| {
                        vars.iter()
                            .rev()
                            .find(|v| v.name == *callee && matches!(v.ty, MirType::FnPtr { .. }))
                            .and_then(|v| match &v.ty {
                                MirType::FnPtr { params, ret } => {
                                    Some((params.clone(), (**ret).clone()))
                                }
                                _ => None,
                            })
                    });
                    if let Some(sig) = fnptr_sig {
                        if !self.acc.isr_fnptr_sigs.contains(&sig) {
                            self.acc.isr_fnptr_sigs.push(sig);
                        }
                    } else if !is_benign_c_builtin(callee) {
                        self.acc.isr_calls_real_extern_c = true;
                    }
                }
                for a in args {
                    self.visit_expr(a, in_isr, callees);
                }
            }
            MirExprKind::RomCallU32 { addr, args, .. } => {
                self.visit_expr(addr, in_isr, callees);
                for a in args {
                    self.visit_expr(a, in_isr, callees);
                }
            }
            MirExprKind::PrimRead(op) => match op.as_ref() {
                MirPrimReadOp::WatchRead { target, elem_ty }
                | MirPrimReadOp::WatchGen { target, elem_ty } => {
                    let kind = MirPrimKind::Watch {
                        elem: Box::new(elem_ty.clone()),
                    };
                    match target {
                        MirPrimTarget::Global(name) => {
                            self.acc.direct_watch_reads.insert(name.clone());
                        }
                        MirPrimTarget::PtrExpr(_) => {
                            self.acc.ptr_watch_read_elems.insert(elem_ty.clone());
                        }
                    }
                    self.visit_prim_target(target, &kind, in_isr, callees);
                }
                MirPrimReadOp::GuardedGet { target, elem_ty } => {
                    let kind = MirPrimKind::Guarded {
                        elem: Box::new(elem_ty.clone()),
                    };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                }
                MirPrimReadOp::InboxTryRecv {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::InboxLen {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::InboxIsEmpty {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::InboxIsFull {
                    target,
                    elem_ty,
                    cap,
                } => {
                    let kind = MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                }
                MirPrimReadOp::InboxTrySend {
                    target,
                    elem_ty,
                    cap,
                    value,
                    ..
                } => {
                    let kind = MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                    self.visit_expr(value, in_isr, callees);
                }
                MirPrimReadOp::SignalIsSet { target } | MirPrimReadOp::SignalTryWait { target } => {
                    self.visit_prim_target(target, &MirPrimKind::Signal, in_isr, callees);
                }
                MirPrimReadOp::WaitQueueWakeOne { target, .. } => {
                    self.visit_prim_target(target, &MirPrimKind::WaitQueue, in_isr, callees);
                }
                MirPrimReadOp::PingPongDmaBuf {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::PingPongBaseAddr {
                    target,
                    elem_ty,
                    cap,
                } => {
                    let kind = MirPrimKind::PingPong {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                }
                MirPrimReadOp::ByteRingPush {
                    target, cap, src, ..
                }
                | MirPrimReadOp::ByteRingPop {
                    target,
                    cap,
                    dst: src,
                } => {
                    let kind = MirPrimKind::ByteRing { cap: *cap };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                    self.visit_expr(src, in_isr, callees);
                }
                MirPrimReadOp::ByteRingPushByte {
                    target, cap, byte, ..
                } => {
                    let kind = MirPrimKind::ByteRing { cap: *cap };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                    self.visit_expr(byte, in_isr, callees);
                }
                MirPrimReadOp::ByteRingGrantWrite { target, cap, len } => {
                    let kind = MirPrimKind::ByteRing { cap: *cap };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                    self.visit_expr(len, in_isr, callees);
                }
                MirPrimReadOp::ByteRingReadSlice { target, cap }
                | MirPrimReadOp::ByteRingTryPopByte { target, cap }
                | MirPrimReadOp::ByteRingLen { target, cap }
                | MirPrimReadOp::ByteRingFreeSpace { target, cap }
                | MirPrimReadOp::ByteRingIsEmpty { target, cap }
                | MirPrimReadOp::ByteRingIsFull { target, cap } => {
                    let kind = MirPrimKind::ByteRing { cap: *cap };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                }
                MirPrimReadOp::InboxViewTryRecv { view, elem_ty }
                | MirPrimReadOp::InboxViewLen { view, elem_ty }
                | MirPrimReadOp::InboxViewCap { view, elem_ty }
                | MirPrimReadOp::InboxViewIsEmpty { view, elem_ty }
                | MirPrimReadOp::InboxViewIsFull { view, elem_ty } => {
                    if in_isr {
                        let vk = MirPrimViewKind::Inbox {
                            elem: Box::new(elem_ty.clone()),
                        };
                        self.acc.isr_contended_prim_mangles.insert(vk.mangle_name());
                        if let MirType::Lease(inner) = elem_ty {
                            self.acc
                                .isr_contended_prim_mangles
                                .insert(format!("lease_release_{}", inner.mangle_name()));
                        }
                    }
                    self.visit_expr(view, in_isr, callees);
                }
                MirPrimReadOp::InboxViewTrySend {
                    view,
                    elem_ty,
                    value,
                    ..
                } => {
                    if in_isr {
                        let vk = MirPrimViewKind::Inbox {
                            elem: Box::new(elem_ty.clone()),
                        };
                        self.acc.isr_contended_prim_mangles.insert(vk.mangle_name());
                        if let MirType::Lease(inner) = elem_ty {
                            self.acc
                                .isr_contended_prim_mangles
                                .insert(format!("lease_release_{}", inner.mangle_name()));
                        }
                    }
                    self.visit_expr(view, in_isr, callees);
                    self.visit_expr(value, in_isr, callees);
                }
                MirPrimReadOp::ByteRingViewPush { view, src, .. }
                | MirPrimReadOp::ByteRingViewPop { view, dst: src } => {
                    if in_isr {
                        self.acc
                            .isr_contended_prim_mangles
                            .insert(MirPrimViewKind::ByteRing.mangle_name());
                    }
                    self.visit_expr(view, in_isr, callees);
                    self.visit_expr(src, in_isr, callees);
                }
                MirPrimReadOp::ByteRingViewPushByte { view, byte, .. } => {
                    if in_isr {
                        self.acc
                            .isr_contended_prim_mangles
                            .insert(MirPrimViewKind::ByteRing.mangle_name());
                    }
                    self.visit_expr(view, in_isr, callees);
                    self.visit_expr(byte, in_isr, callees);
                }
                MirPrimReadOp::ByteRingViewGrantWrite { view, len } => {
                    if in_isr {
                        self.acc
                            .isr_contended_prim_mangles
                            .insert(MirPrimViewKind::ByteRing.mangle_name());
                    }
                    self.visit_expr(view, in_isr, callees);
                    self.visit_expr(len, in_isr, callees);
                }
                MirPrimReadOp::ByteRingViewReadSlice { view }
                | MirPrimReadOp::ByteRingViewTryPopByte { view }
                | MirPrimReadOp::ByteRingViewLen { view }
                | MirPrimReadOp::ByteRingViewCap { view }
                | MirPrimReadOp::ByteRingViewFreeSpace { view }
                | MirPrimReadOp::ByteRingViewIsEmpty { view }
                | MirPrimReadOp::ByteRingViewIsFull { view } => {
                    if in_isr {
                        self.acc
                            .isr_contended_prim_mangles
                            .insert(MirPrimViewKind::ByteRing.mangle_name());
                    }
                    self.visit_expr(view, in_isr, callees);
                }
                MirPrimReadOp::PoolTryAlloc {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::PoolAvail {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::PoolInUse {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::PoolIsExhausted {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::PoolCap {
                    target,
                    elem_ty,
                    cap,
                } => {
                    let kind = MirPrimKind::Pool {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                }
                MirPrimReadOp::DmaSlotPark {
                    target,
                    elem_ty,
                    lease,
                } => {
                    let kind = MirPrimKind::DmaSlot {
                        elem: Box::new(elem_ty.clone()),
                    };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                    self.visit_expr(lease, in_isr, callees);
                }
                MirPrimReadOp::DmaSlotTake { target, elem_ty }
                | MirPrimReadOp::DmaSlotRelease {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::DmaSlotIsBusy { target, elem_ty }
                | MirPrimReadOp::DmaSlotIsEmpty { target, elem_ty }
                | MirPrimReadOp::DmaSlotAddr { target, elem_ty } => {
                    let kind = MirPrimKind::DmaSlot {
                        elem: Box::new(elem_ty.clone()),
                    };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                }
            },
        }
    }

    fn visit_prim_nonblocking(
        &mut self,
        op: &MirPrimNonBlockingOp,
        in_isr: bool,
        callees: &mut BTreeSet<String>,
    ) {
        match op {
            MirPrimNonBlockingOp::WatchPublish {
                target,
                elem_ty,
                value,
            } => {
                let kind = MirPrimKind::Watch {
                    elem: Box::new(elem_ty.clone()),
                };
                self.visit_prim_target(target, &kind, in_isr, callees);
                self.visit_expr(value, in_isr, callees);
            }
            MirPrimNonBlockingOp::GuardedSet {
                target,
                elem_ty,
                value,
            } => {
                let kind = MirPrimKind::Guarded {
                    elem: Box::new(elem_ty.clone()),
                };
                self.visit_prim_target(target, &kind, in_isr, callees);
                self.visit_expr(value, in_isr, callees);
            }
            MirPrimNonBlockingOp::InboxPost {
                target,
                elem_ty,
                cap,
                value,
                ..
            } => {
                let kind = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                self.visit_prim_target(target, &kind, in_isr, callees);
                self.visit_expr(value, in_isr, callees);
            }
            MirPrimNonBlockingOp::SignalNotify { target, .. }
            | MirPrimNonBlockingOp::SignalClear { target } => {
                self.visit_prim_target(target, &MirPrimKind::Signal, in_isr, callees);
            }
            MirPrimNonBlockingOp::WaitQueuePrepare { target }
            | MirPrimNonBlockingOp::WaitQueueCancel { target }
            | MirPrimNonBlockingOp::WaitQueueWakeAll { target, .. } => {
                self.visit_prim_target(target, &MirPrimKind::WaitQueue, in_isr, callees);
            }
            MirPrimNonBlockingOp::PingPongFlip {
                target,
                elem_ty,
                cap,
            }
            | MirPrimNonBlockingOp::PingPongSetActive {
                target,
                elem_ty,
                cap,
                ..
            } => {
                let kind = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                self.visit_prim_target(target, &kind, in_isr, callees);
            }
            MirPrimNonBlockingOp::PingPongCompleteHalf {
                target,
                elem_ty,
                cap,
                half,
            } => {
                let kind = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                self.visit_prim_target(target, &kind, in_isr, callees);
                self.visit_expr(half, in_isr, callees);
            }
            MirPrimNonBlockingOp::LeaseRelease { value, elem_ty, .. } => {
                if in_isr {
                    self.acc
                        .isr_contended_prim_mangles
                        .insert(format!("lease_release_{}", elem_ty.mangle_name()));
                }
                self.visit_expr(value, in_isr, callees);
            }
            MirPrimNonBlockingOp::LeaseNullify { .. } => {}
            MirPrimNonBlockingOp::ByteRingConsume { target, cap, len }
            | MirPrimNonBlockingOp::ByteRingCommitWrite {
                target, cap, len, ..
            } => {
                let kind = MirPrimKind::ByteRing { cap: *cap };
                self.visit_prim_target(target, &kind, in_isr, callees);
                self.visit_expr(len, in_isr, callees);
            }
            MirPrimNonBlockingOp::ByteRingClear { target, cap } => {
                let kind = MirPrimKind::ByteRing { cap: *cap };
                self.visit_prim_target(target, &kind, in_isr, callees);
            }
            MirPrimNonBlockingOp::InboxViewPost {
                view,
                elem_ty,
                value,
                ..
            } => {
                if in_isr {
                    let vk = MirPrimViewKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                    };
                    self.acc.isr_contended_prim_mangles.insert(vk.mangle_name());
                    if let MirType::Lease(inner) = elem_ty {
                        self.acc
                            .isr_contended_prim_mangles
                            .insert(format!("lease_release_{}", inner.mangle_name()));
                    }
                }
                self.visit_expr(view, in_isr, callees);
                self.visit_expr(value, in_isr, callees);
            }
            MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
            | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                if in_isr {
                    self.acc
                        .isr_contended_prim_mangles
                        .insert(MirPrimViewKind::ByteRing.mangle_name());
                }
                self.visit_expr(view, in_isr, callees);
                self.visit_expr(len, in_isr, callees);
            }
            MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                if in_isr {
                    self.acc
                        .isr_contended_prim_mangles
                        .insert(MirPrimViewKind::ByteRing.mangle_name());
                }
                self.visit_expr(view, in_isr, callees);
            }
        }
    }

    fn visit_stmt(&mut self, stmt: &MirStmt, in_isr: bool, callees: &mut BTreeSet<String>) {
        match stmt {
            MirStmt::Assign { target, value, .. } => {
                self.visit_place(target, in_isr, callees);
                self.visit_expr(value, in_isr, callees);
            }
            MirStmt::Expr(expr) => self.visit_expr(expr, in_isr, callees),
            MirStmt::PrimOp(op) => self.visit_prim_nonblocking(op, in_isr, callees),
            MirStmt::Intrinsic(intr) => match intr {
                MirIntrinsicOp::UseSystick { hz } => self.visit_expr(hz, in_isr, callees),
                MirIntrinsicOp::SetPriority { .. } => {}
                MirIntrinsicOp::SysOp { method, args } => {
                    for a in args {
                        self.visit_expr(a, in_isr, callees);
                    }
                    if method == "set_fault_handler" {
                        self.record_sys_isr_callback(args.first(), &[], &MirType::Void);
                    } else if method == "set_irq_router" {
                        self.record_sys_isr_callback(args.get(1), &[MirType::U32], &MirType::Void);
                    }
                }
            },
            MirStmt::EnterCritical
            | MirStmt::ExitCritical
            | MirStmt::EndInboxTurn
            | MirStmt::Break
            | MirStmt::Continue
            | MirStmt::SetIrqSelector { .. }
            | MirStmt::BoundedForLatch
            | MirStmt::ZeroTaskLocals => {}
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.visit_expr(cond, in_isr, callees);
                for s in then_body {
                    self.visit_stmt(s, in_isr, callees);
                }
                for s in else_body {
                    self.visit_stmt(s, in_isr, callees);
                }
            }
            MirStmt::While { cond, body } => {
                self.visit_expr(cond, in_isr, callees);
                for s in body {
                    self.visit_stmt(s, in_isr, callees);
                }
            }
            MirStmt::Switch {
                discr,
                arms,
                default,
            } => {
                self.visit_expr(discr, in_isr, callees);
                for (_, body) in arms {
                    for s in body {
                        self.visit_stmt(s, in_isr, callees);
                    }
                }
                for s in default {
                    self.visit_stmt(s, in_isr, callees);
                }
            }
            MirStmt::Return(expr_opt) => {
                if let Some(e) = expr_opt {
                    self.visit_expr(e, in_isr, callees);
                }
            }
            MirStmt::Error { reason, partial } => {
                if let Some(r) = reason {
                    self.visit_expr(r, in_isr, callees);
                }
                if let Some(p) = partial {
                    self.visit_expr(p, in_isr, callees);
                }
            }
            MirStmt::SpawnActivate { args, .. } => {
                for a in args {
                    self.visit_expr(a, in_isr, callees);
                }
            }
            MirStmt::PrintFmt { chunks, .. } => {
                for ch in chunks {
                    self.visit_fmt_chunk(ch, in_isr, callees);
                }
            }
            MirStmt::FmtBuf { buf, chunks, .. } => {
                self.visit_expr(buf, in_isr, callees);
                for ch in chunks {
                    self.visit_fmt_chunk(ch, in_isr, callees);
                }
            }
            MirStmt::FmtRing { ring, chunks, .. } => {
                self.visit_expr(ring, in_isr, callees);
                for ch in chunks {
                    self.visit_fmt_chunk(ch, in_isr, callees);
                }
            }
            MirStmt::MmioWrite {
                base, index, value, ..
            } => {
                self.visit_expr(base, in_isr, callees);
                if let Some(idx) = index {
                    self.visit_expr(idx, in_isr, callees);
                }
                self.visit_expr(value, in_isr, callees);
            }
        }
    }

    fn visit_terminator(
        &mut self,
        term: &MirTerminator,
        in_isr: bool,
        callees: &mut BTreeSet<String>,
    ) {
        match term {
            MirTerminator::Goto(_) => {}
            MirTerminator::Branch { cond, .. } => self.visit_expr(cond, in_isr, callees),
            MirTerminator::Switch { discr, .. } => self.visit_expr(discr, in_isr, callees),
            MirTerminator::Return(expr_opt) => {
                if let Some(e) = expr_opt {
                    self.visit_expr(e, in_isr, callees);
                }
            }
            MirTerminator::Error { reason, partial } => {
                if let Some(r) = reason {
                    self.visit_expr(r, in_isr, callees);
                }
                if let Some(p) = partial {
                    self.visit_expr(p, in_isr, callees);
                }
            }
            MirTerminator::Yield(yk) => match yk {
                MirYieldKind::EveryLatch { .. }
                | MirYieldKind::DelayPoll { .. }
                | MirYieldKind::ShareCheckpoint { .. }
                | MirYieldKind::PollChild { .. } => {}
                MirYieldKind::Wait { arms, .. } => {
                    for arm in arms {
                        match &arm.op {
                            MirWaitArmOp::InboxRecv {
                                target,
                                elem_ty,
                                cap,
                                ..
                            }
                            | MirWaitArmOp::InboxSend {
                                target,
                                elem_ty,
                                cap,
                                ..
                            } => {
                                let kind = MirPrimKind::Inbox {
                                    elem: Box::new(elem_ty.clone()),
                                    cap: *cap,
                                };
                                self.visit_prim_target(target, &kind, in_isr, callees);
                            }
                            MirWaitArmOp::SignalWait { target, .. } => {
                                self.visit_prim_target(
                                    target,
                                    &MirPrimKind::Signal,
                                    in_isr,
                                    callees,
                                );
                            }
                            MirWaitArmOp::WaitQueueCommit { target, .. } => {
                                self.visit_prim_target(
                                    target,
                                    &MirPrimKind::WaitQueue,
                                    in_isr,
                                    callees,
                                );
                            }
                            MirWaitArmOp::WatchWaitChange {
                                target, elem_ty, ..
                            } => {
                                let kind = MirPrimKind::Watch {
                                    elem: Box::new(elem_ty.clone()),
                                };
                                match target {
                                    MirPrimTarget::Global(name) => {
                                        self.acc.direct_watch_reads.insert(name.clone());
                                    }
                                    MirPrimTarget::PtrExpr(_) => {
                                        self.acc.ptr_watch_read_elems.insert(elem_ty.clone());
                                    }
                                }
                                self.visit_prim_target(target, &kind, in_isr, callees);
                            }
                            MirWaitArmOp::PingPongWaitChunk {
                                target,
                                elem_ty,
                                cap,
                                ..
                            } => {
                                let kind = MirPrimKind::PingPong {
                                    elem: Box::new(elem_ty.clone()),
                                    cap: *cap,
                                };
                                self.visit_prim_target(target, &kind, in_isr, callees);
                            }
                            MirWaitArmOp::PoolAlloc {
                                target,
                                elem_ty,
                                cap,
                                ..
                            } => {
                                let kind = MirPrimKind::Pool {
                                    elem: Box::new(elem_ty.clone()),
                                    cap: *cap,
                                };
                                self.visit_prim_target(target, &kind, in_isr, callees);
                            }
                            MirWaitArmOp::ByteRingWaitRead { target, cap, .. }
                            | MirWaitArmOp::ByteRingWaitWrite { target, cap, .. } => {
                                let kind = MirPrimKind::ByteRing { cap: *cap };
                                self.visit_prim_target(target, &kind, in_isr, callees);
                            }
                            MirWaitArmOp::InboxViewRecv { view, elem_ty, .. }
                            | MirWaitArmOp::InboxViewSend { view, elem_ty, .. } => {
                                if in_isr {
                                    let vk = MirPrimViewKind::Inbox {
                                        elem: Box::new(elem_ty.clone()),
                                    };
                                    self.acc.isr_contended_prim_mangles.insert(vk.mangle_name());
                                }
                                self.visit_expr(view, in_isr, callees);
                            }
                            MirWaitArmOp::ByteRingViewWaitRead { view, .. }
                            | MirWaitArmOp::ByteRingViewWaitWrite { view, .. } => {
                                if in_isr {
                                    self.acc
                                        .isr_contended_prim_mangles
                                        .insert(MirPrimViewKind::ByteRing.mangle_name());
                                }
                                self.visit_expr(view, in_isr, callees);
                            }
                        }
                    }
                }
                MirYieldKind::CallYielding { callee, args, .. } => {
                    callees.insert(callee.clone());
                    for a in args {
                        self.visit_expr(a, in_isr, callees);
                    }
                }
            },
        }
    }

    fn visit_blocks(&mut self, blocks: &[MirBlock], in_isr: bool, callees: &mut BTreeSet<String>) {
        for blk in blocks {
            for s in &blk.stmts {
                self.visit_stmt(s, in_isr, callees);
            }
            self.visit_terminator(&blk.terminator, in_isr, callees);
        }
    }

    fn visit_function(
        &mut self,
        func: &'a MirFunction,
        in_isr: bool,
        callees: &mut BTreeSet<String>,
    ) {
        if in_isr {
            let mut seen_structs = BTreeSet::new();
            for v in &func.vars {
                self.record_prim_types_in_ty(&v.ty, &mut seen_structs);
            }
        }
        let prev_vars = self.cur_vars;
        self.cur_vars = Some(&func.vars);
        match &func.body {
            MirFnBody::Direct(stmts) => {
                for s in stmts {
                    self.visit_stmt(s, in_isr, callees);
                }
            }
            MirFnBody::Coroutine { blocks, .. } => {
                self.visit_blocks(blocks, in_isr, callees);
            }
        }
        self.cur_vars = prev_vars;
    }

    fn visit_non_prim_roots(&mut self, module: &'a MirModule, callees: &mut BTreeSet<String>) {
        for c in &module.consts {
            self.visit_expr(&c.init, false, callees);
        }
        for bp in &module.task_blueprints {
            let prev_vars = self.cur_vars;
            self.cur_vars = Some(&bp.vars);
            self.visit_blocks(&bp.blocks, false, callees);
            self.cur_vars = prev_vars;
        }
        for step in &module.boot.steps {
            match step {
                MirBootStep::Stmt(s) => self.visit_stmt(s, false, callees),
                MirBootStep::RunYieldingFn { args, .. }
                | MirBootStep::SpawnActivate { args, .. } => {
                    for a in args {
                        self.visit_expr(a, false, callees);
                    }
                }
            }
        }
        for sp in &module.spawned_tasks {
            for a in &sp.init_args {
                self.visit_expr(a, false, callees);
            }
        }
    }

    fn visit_irq_vectors_and_isrs(
        &mut self,
        module: &'a MirModule,
        isr_queue: &mut VecDeque<String>,
    ) {
        for vec in &module.irq_vectors {
            for b in &vec.bindings {
                for barg in &b.baked_args {
                    match barg {
                        MirBakedArg::PrimAddr(name) => {
                            self.acc.addr_taken_prims.insert(name.clone());
                            self.acc.referenced_prims.insert(name.clone());
                            self.record_isr_prim_global(name, None);
                        }
                        MirBakedArg::SubsystemPath {
                            root,
                            root_struct,
                            field_chain,
                        } => {
                            self.acc.referenced_subsystems.insert(root.clone());
                            self.acc.isr_contended_globals.insert(root.clone());
                            let path_key = if field_chain.is_empty() {
                                root.clone()
                            } else {
                                let chain: Vec<&str> =
                                    field_chain.iter().map(|(f, _)| f.as_str()).collect();
                                format!("{root}.{}", chain.join("."))
                            };
                            self.acc.isr_contended_globals.insert(path_key);
                            let mut cur_ty = MirType::Struct(root_struct.clone());
                            for (fname, fstruct) in field_chain {
                                let sname = match &cur_ty {
                                    MirType::Struct(s) => s.as_str(),
                                    _ => fstruct.as_str(),
                                };
                                if let Some(st) = self.structs.get(sname) {
                                    if let Some(fld) = st.fields.iter().find(|f| &f.name == fname) {
                                        cur_ty = fld.ty.clone();
                                        continue;
                                    }
                                }
                                cur_ty = MirType::Struct(fstruct.clone());
                            }
                            let mut seen_structs = BTreeSet::new();
                            self.record_prim_types_in_ty(&cur_ty, &mut seen_structs);
                        }
                        MirBakedArg::MmioAddr(_) => {}
                        MirBakedArg::Const(expr) => {
                            let mut c = BTreeSet::new();
                            self.visit_expr(expr, true, &mut c);
                            for callee in c {
                                isr_queue.push_back(callee);
                            }
                        }
                    }
                }
                for sv in &b.state_vars {
                    let mut c = BTreeSet::new();
                    self.visit_expr(&sv.init, true, &mut c);
                    for callee in c {
                        isr_queue.push_back(callee);
                    }
                }
            }
        }

        for isr in &module.isrs {
            let mut seen_structs = BTreeSet::new();
            for v in &isr.vars {
                self.record_prim_types_in_ty(&v.ty, &mut seen_structs);
            }
            let prev_vars = self.cur_vars;
            self.cur_vars = Some(&isr.vars);
            let mut isr_callees = BTreeSet::new();
            for sv in &isr.state_vars {
                self.visit_expr(&sv.init, true, &mut isr_callees);
            }
            for stmt in &isr.body {
                self.visit_stmt(stmt, true, &mut isr_callees);
            }
            self.cur_vars = prev_vars;
            for callee in isr_callees {
                isr_queue.push_back(callee);
            }
        }
    }
}

fn fnptr_sig_matches(func: &MirFunction, params: &[MirType], ret: &MirType) -> bool {
    if func.is_yielding || func.params.len() != params.len() {
        return false;
    }
    let env_ret = func.envelope_return_type();
    if env_ret != *ret && env_ret.host_value_type() != ret.host_value_type() {
        return false;
    }
    func.params.iter().zip(params.iter()).all(|(&vid, pty)| {
        func.vars
            .get(vid.0)
            .is_some_and(|v| v.ty == *pty || v.ty.host_value_type() == pty.host_value_type())
    })
}

/// Analyze whole-program ISR primitive contention and dead `watch<T>` primitives across
/// `consts`, `primitives`, `functions`, `task_blueprints`, `isrs`, `boot`, and `irq_vectors`.
pub fn analyze_visibility(module: &MirModule) -> VisibilityReport {
    let mut ctx = ScanCtx::new(module);
    let dma_layout_structs = crate::ram::collect_dma_layout_structs(module);
    let (addr_taken_fns, _) = collect_all_fn_addrs_and_calls(module);
    let fn_map: BTreeMap<&str, &MirFunction> = module
        .functions
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();

    // 1. Scan all functions (non-ISR pass) to record watch reads, addr escapes, and call graph edges.
    let mut fn_callees: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for func in &module.functions {
        let mut callees = BTreeSet::new();
        ctx.visit_function(func, false, &mut callees);
        fn_callees.insert(func.name.clone(), callees);
    }

    // 2. Scan constants, primitive initializers, task blueprints, spawned init args, and boot sequence (non-ISR).
    let mut dummy_callees = BTreeSet::new();
    ctx.visit_non_prim_roots(module, &mut dummy_callees);
    for p in &module.primitives {
        if let Some(init) = &p.init {
            ctx.visit_expr(init, false, &mut dummy_callees);
        }
    }

    // 3. Scan `MirIrqVector` `baked_args` & state initializers, plus all ISR bodies + transitive callees.
    let mut isr_queue: VecDeque<String> = VecDeque::new();
    ctx.visit_irq_vectors_and_isrs(module, &mut isr_queue);

    for callee in ctx.acc.sys_isr_callbacks.clone() {
        isr_queue.push_back(callee);
    }

    let mut visited_isr_fns: BTreeSet<String> = BTreeSet::new();
    let mut processed_fnptr_sigs = 0usize;
    let mut seeded_real_extern_c = false;
    let seed_indirect_isr_callees =
        |ctx: &ScanCtx<'_>,
         visited_isr_fns: &BTreeSet<String>,
         isr_queue: &mut VecDeque<String>,
         processed_fnptr_sigs: &mut usize,
         seeded_real_extern_c: &mut bool| {
            while *processed_fnptr_sigs < ctx.acc.isr_fnptr_sigs.len() {
                let (params, ret) = &ctx.acc.isr_fnptr_sigs[*processed_fnptr_sigs];
                *processed_fnptr_sigs += 1;
                for fname in &addr_taken_fns {
                    if !visited_isr_fns.contains(fname)
                        && fn_map
                            .get(fname.as_str())
                            .is_some_and(|f| fnptr_sig_matches(f, params, ret))
                    {
                        isr_queue.push_back(fname.clone());
                    }
                }
            }
            if ctx.acc.isr_calls_real_extern_c && !*seeded_real_extern_c {
                *seeded_real_extern_c = true;
                for fname in &addr_taken_fns {
                    if !visited_isr_fns.contains(fname)
                        && fn_map.get(fname.as_str()).is_some_and(|f| !f.is_yielding)
                    {
                        isr_queue.push_back(fname.clone());
                    }
                }
            }
        };

    seed_indirect_isr_callees(
        &ctx,
        &visited_isr_fns,
        &mut isr_queue,
        &mut processed_fnptr_sigs,
        &mut seeded_real_extern_c,
    );

    while let Some(fname) = isr_queue.pop_front() {
        if !visited_isr_fns.insert(fname.clone()) {
            continue;
        }
        if let Some(&func) = fn_map.get(fname.as_str()) {
            let mut sub_callees = BTreeSet::new();
            ctx.visit_function(func, true, &mut sub_callees);
            for callee in sub_callees {
                if !visited_isr_fns.contains(&callee) {
                    isr_queue.push_back(callee);
                }
            }
            seed_indirect_isr_callees(
                &ctx,
                &visited_isr_fns,
                &mut isr_queue,
                &mut processed_fnptr_sigs,
                &mut seeded_real_extern_c,
            );
        } else if let Some(sub_callees) = fn_callees.get(&fname) {
            for callee in sub_callees {
                if !visited_isr_fns.contains(callee) {
                    isr_queue.push_back(callee.clone());
                }
            }
        }
    }

    // 4. Identify dead watches and compute 32-bit / 64-bit static SRAM savings.
    let mut dead_watches = BTreeSet::new();
    let mut dead_watch_bytes_saved_32 = 0usize;
    let mut dead_watch_bytes_saved_64 = 0usize;

    for p in &module.primitives {
        if let MirPrimKind::Watch { elem } = &p.kind {
            let has_direct_read = ctx.acc.direct_watch_reads.contains(&p.name);
            let addr_taken = ctx.acc.addr_taken_prims.contains(&p.name);
            let has_ptr_read = ctx.acc.ptr_watch_read_elems.contains(elem.as_ref());
            if !has_direct_read && (!addr_taken || !has_ptr_read) {
                dead_watches.insert(p.name.clone());
                if !addr_taken {
                    dead_watch_bytes_saved_32 +=
                        watch_byte_size(elem, p.dma_safe, 4, &ctx.structs, &dma_layout_structs);
                    dead_watch_bytes_saved_64 +=
                        watch_byte_size(elem, p.dma_safe, 8, &ctx.structs, &dma_layout_structs);
                }
            }
        }
    }

    // 5. Remaining non-dead top-level primitives are pure task-to-task (`elided_globals`).
    let mut elided_globals = BTreeSet::new();
    for p in &module.primitives {
        if !dead_watches.contains(&p.name) && !ctx.acc.isr_contended_globals.contains(&p.name) {
            elided_globals.insert(p.name.clone());
        }
    }

    VisibilityReport {
        dead_watches,
        addr_taken_prims: ctx.acc.addr_taken_prims,
        dead_watch_bytes_saved_32,
        dead_watch_bytes_saved_64,
        isr_contended_globals: ctx.acc.isr_contended_globals,
        isr_contended_prim_mangles: ctx.acc.isr_contended_prim_mangles,
        elided_globals,
        singleton_subsystems: module.visibility.singleton_subsystems.clone(),
        singleton_subsystem_structs: module.visibility.singleton_subsystem_structs.clone(),
        folded_task_vars: module.visibility.folded_task_vars.clone(),
        folded_task_var_ids: module.visibility.folded_task_var_ids.clone(),
        folded_fn_vars: module.visibility.folded_fn_vars.clone(),
        singleton_task_bytes_saved: module.visibility.singleton_task_bytes_saved.clone(),
        singleton_bytes_saved_32: module.visibility.singleton_bytes_saved_32,
        singleton_bytes_saved_64: module.visibility.singleton_bytes_saved_64,
    }
}

fn strip_stmt(stmt: &mut MirStmt, dead_watches: &BTreeSet<String>) {
    match stmt {
        MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
            target: MirPrimTarget::Global(name),
            value,
            ..
        }) if dead_watches.contains(name) => {
            let placeholder = MirExpr::new(MirExprKind::BoolLit(false), MirType::Void);
            let val = std::mem::replace(value, placeholder);
            *stmt = MirStmt::Expr(val);
        }
        MirStmt::If {
            then_body,
            else_body,
            ..
        } => {
            for s in then_body {
                strip_stmt(s, dead_watches);
            }
            for s in else_body {
                strip_stmt(s, dead_watches);
            }
        }
        MirStmt::While { body, .. } => {
            for s in body {
                strip_stmt(s, dead_watches);
            }
        }
        MirStmt::Switch { arms, default, .. } => {
            for (_, body) in arms {
                for s in body {
                    strip_stmt(s, dead_watches);
                }
            }
            for s in default {
                strip_stmt(s, dead_watches);
            }
        }
        _ => {}
    }
}

/// Rewrite direct `WatchPublish` statements targeting dead watches into `MirStmt::Expr(value)`
/// across `functions`, `task_blueprints`, `isrs`, and `boot`.
pub fn strip_dead_watch_publishes(module: &mut MirModule, report: &VisibilityReport) {
    if report.dead_watches.is_empty() {
        return;
    }
    for func in &mut module.functions {
        match &mut func.body {
            MirFnBody::Direct(stmts) => {
                for s in stmts {
                    strip_stmt(s, &report.dead_watches);
                }
            }
            MirFnBody::Coroutine { blocks, .. } => {
                for blk in blocks {
                    for s in &mut blk.stmts {
                        strip_stmt(s, &report.dead_watches);
                    }
                }
            }
        }
    }
    for bp in &mut module.task_blueprints {
        for blk in &mut bp.blocks {
            for s in &mut blk.stmts {
                strip_stmt(s, &report.dead_watches);
            }
        }
    }
    for isr in &mut module.isrs {
        for s in &mut isr.body {
            strip_stmt(s, &report.dead_watches);
        }
    }
    for step in &mut module.boot.steps {
        if let MirBootStep::Stmt(s) = step {
            strip_stmt(s, &report.dead_watches);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Whole-Program Singleton Subsystem & Task Constant-Folding (`fold_singletons`)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SubsysInstancePath {
    display_path: String,
}

type SubsystemInstanceCounts = (
    BTreeMap<String, usize>,
    BTreeSet<String>,
    BTreeMap<String, String>,
    BTreeSet<String>,
);

fn collect_subsystem_instance_counts(module: &MirModule) -> SubsystemInstanceCounts {
    let structs: BTreeMap<&str, &MirStruct> = module
        .structs
        .iter()
        .map(|s| (s.name.as_str(), s))
        .collect();

    let mut subsys_struct_names: BTreeSet<String> = module
        .structs
        .iter()
        .filter(|s| s.is_subsystem)
        .map(|s| s.name.clone())
        .collect();
    for sub in &module.subsystems {
        subsys_struct_names.insert(sub.struct_name.clone());
    }
    loop {
        let mut changed = false;
        for st in &module.structs {
            if subsys_struct_names.contains(&st.name) {
                continue;
            }
            if st
                .fields
                .iter()
                .any(|f| ty_mentions_any_struct(&f.ty, &subsys_struct_names))
            {
                subsys_struct_names.insert(st.name.clone());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut instances_by_struct: BTreeMap<String, Vec<SubsysInstancePath>> = BTreeMap::new();
    let mut visited = BTreeSet::new();

    #[allow(clippy::too_many_arguments)]
    fn walk_struct_instances(
        struct_name: &str,
        cur_path: &str,
        mult: usize,
        structs: &BTreeMap<&str, &MirStruct>,
        subsys_struct_names: &BTreeSet<String>,
        instances_by_struct: &mut BTreeMap<String, Vec<SubsysInstancePath>>,
        visited: &mut BTreeSet<String>,
    ) {
        if mult == 0 {
            return;
        }
        for k in 0..mult {
            let p = if mult == 1 {
                cur_path.to_string()
            } else {
                format!("{cur_path}[{k}]")
            };
            instances_by_struct
                .entry(struct_name.to_string())
                .or_default()
                .push(SubsysInstancePath { display_path: p });
        }
        if !visited.insert(struct_name.to_string()) {
            return;
        }
        if let Some(st) = structs.get(struct_name) {
            for f in &st.fields {
                let mut inner_ty = &f.ty;
                let mut arr_mult = 1usize;
                while let MirType::Array { elem, len } = inner_ty {
                    arr_mult = arr_mult.saturating_mul(*len);
                    inner_ty = elem;
                }
                if let MirType::Struct(sub_s) = inner_ty {
                    if subsys_struct_names.contains(sub_s) {
                        let next_path = format!("{cur_path}.{}", f.name);
                        walk_struct_instances(
                            sub_s,
                            &next_path,
                            mult.saturating_mul(arr_mult),
                            structs,
                            subsys_struct_names,
                            instances_by_struct,
                            visited,
                        );
                    }
                }
            }
        }
        visited.remove(struct_name);
    }

    for sub in &module.subsystems {
        walk_struct_instances(
            &sub.struct_name,
            &sub.name,
            1,
            &structs,
            &subsys_struct_names,
            &mut instances_by_struct,
            &mut visited,
        );
    }

    let mut counts = BTreeMap::new();
    let mut singleton_subsystems = BTreeSet::new();
    let mut singleton_subsystem_structs = BTreeMap::new();
    for (sname, paths) in instances_by_struct {
        let cnt = paths.len();
        counts.insert(sname.clone(), cnt);
        if cnt == 1 && subsys_struct_names.contains(&sname) {
            let path = paths[0].display_path.clone();
            singleton_subsystems.insert(path.clone());
            singleton_subsystem_structs.insert(sname, path);
        }
    }

    (
        counts,
        singleton_subsystems,
        singleton_subsystem_structs,
        subsys_struct_names,
    )
}

fn ty_mentions_any_struct(ty: &MirType, set: &BTreeSet<String>) -> bool {
    match ty {
        MirType::Struct(s) => set.contains(s),
        MirType::Array { elem, .. } => ty_mentions_any_struct(elem, set),
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

fn int_lit_fits_mir_type(v: i128, target: &MirType) -> bool {
    match target.host_value_type() {
        MirType::U8 | MirType::I8 => (i128::from(i8::MIN)..=i128::from(u8::MAX)).contains(&v),
        MirType::U16 | MirType::I16 => (i128::from(i16::MIN)..=i128::from(u16::MAX)).contains(&v),
        MirType::U32 | MirType::I32 | MirType::Enum(_) => {
            (i128::from(i32::MIN)..=i128::from(u32::MAX)).contains(&v)
        }
        MirType::U64 | MirType::I64 | MirType::Usize | MirType::Isize => {
            (i128::from(i64::MIN)..=i128::from(u64::MAX)).contains(&v)
        }
        _ => false,
    }
}

fn coerce_immediate_to_var_ty(expr: &MirExpr, target_ty: &MirType) -> Option<MirExpr> {
    if &expr.ty == target_ty {
        return Some(expr.clone());
    }
    if let MirExprKind::IntLit(v) = expr.kind {
        if is_scalar_int_mir_type(target_ty) && int_lit_fits_mir_type(v, target_ty) {
            return Some(MirExpr::new(
                MirExprKind::IntLit(v),
                target_ty.host_value_type(),
            ));
        }
    }
    if expr.ty.host_value_type() == target_ty.host_value_type() {
        return Some(expr.clone());
    }
    None
}

fn is_pure_static_immediate(
    expr: &MirExpr,
    module: &MirModule,
    subsys_counts: &BTreeMap<String, usize>,
    subsys_structs: &BTreeSet<String>,
) -> bool {
    match &expr.kind {
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::StringLit(_)
        | MirExprKind::TicksLit(_)
        | MirExprKind::EnumVariant { .. }
        | MirExprKind::ConstRef(_)
        | MirExprKind::GlobalPrimAddr(_)
        | MirExprKind::MmioAddr(_)
        | MirExprKind::FnAddr(_)
        | MirExprKind::ArrayLen(_) => true,
        MirExprKind::SubsystemAddr(global) => module
            .subsystems
            .iter()
            .find(|s| &s.name == global)
            .is_some_and(|s| subsys_counts.get(&s.struct_name).copied() == Some(1)),
        MirExprKind::MakePrimView { ptr, .. } => {
            is_pure_static_immediate(ptr, module, subsys_counts, subsys_structs)
        }
        MirExprKind::AddrOfPlace(place) => {
            if !is_pure_static_place(place, module, subsys_counts, subsys_structs) {
                return false;
            }
            if let MirType::Ptr(inner) = &expr.ty {
                if let MirType::Struct(sname) = inner.as_ref() {
                    if subsys_structs.contains(sname) {
                        return subsys_counts.get(sname).copied() == Some(1);
                    }
                }
            }
            true
        }
        MirExprKind::Unary { op, expr: inner } => {
            !matches!(op, MirUnaryOp::Deref)
                && is_pure_static_immediate(inner, module, subsys_counts, subsys_structs)
        }
        MirExprKind::Cast { expr: inner, .. } => {
            is_pure_static_immediate(inner, module, subsys_counts, subsys_structs)
        }
        MirExprKind::Binary { op, lhs, rhs } => {
            if matches!(op, MirBinOp::Div | MirBinOp::Rem)
                && matches!(rhs.kind, MirExprKind::IntLit(0))
            {
                return false;
            }
            is_pure_static_immediate(lhs, module, subsys_counts, subsys_structs)
                && is_pure_static_immediate(rhs, module, subsys_counts, subsys_structs)
        }
        MirExprKind::FieldAccess {
            base,
            is_ptr: false,
            ..
        } => is_pure_static_immediate(base, module, subsys_counts, subsys_structs),
        MirExprKind::FieldAccess { is_ptr: true, .. }
        | MirExprKind::StructLit { .. }
        | MirExprKind::ArrayLit { .. }
        | MirExprKind::MmioRead { .. }
        | MirExprKind::Var(_)
        | MirExprKind::SliceLen(_)
        | MirExprKind::Index { .. }
        | MirExprKind::SubSlice { .. }
        | MirExprKind::ArrayToSlice { .. }
        | MirExprKind::FromBytes { .. }
        | MirExprKind::PrimRead(_)
        | MirExprKind::CallDirect { .. }
        | MirExprKind::RomCallU32 { .. }
        | MirExprKind::ResultIsErr(_)
        | MirExprKind::ResultErr(_)
        | MirExprKind::ResultVal(_)
        | MirExprKind::NowTicks
        | MirExprKind::SysQuery(_)
        | MirExprKind::LeaseGet { .. }
        | MirExprKind::LeaseAddr { .. } => false,
    }
}

fn is_pure_static_place(
    place: &MirPlace,
    module: &MirModule,
    subsys_counts: &BTreeMap<String, usize>,
    subsys_structs: &BTreeSet<String>,
) -> bool {
    match place {
        MirPlace::Var(_) => false,
        MirPlace::Deref(ptr_expr, _) => {
            is_pure_static_immediate(ptr_expr, module, subsys_counts, subsys_structs)
        }
        MirPlace::Field { base, .. } => {
            is_pure_static_place(base, module, subsys_counts, subsys_structs)
        }
        MirPlace::Index {
            base,
            index,
            is_slice: false,
            ..
        } => {
            is_pure_static_place(base, module, subsys_counts, subsys_structs)
                && is_pure_static_immediate(index, module, subsys_counts, subsys_structs)
        }
        MirPlace::Index { is_slice: true, .. } => false,
    }
}

fn mir_place_eq(a: &MirPlace, b: &MirPlace) -> bool {
    match (a, b) {
        (MirPlace::Var(va), MirPlace::Var(vb)) => va == vb,
        (MirPlace::Deref(ea, ta), MirPlace::Deref(eb, tb)) => ta == tb && mir_expr_eq(ea, eb),
        (
            MirPlace::Field {
                base: ba,
                struct_name: sa,
                packed: pa,
                field: fa,
                field_ty: fta,
            },
            MirPlace::Field {
                base: bb,
                struct_name: sb,
                packed: pb,
                field: fb,
                field_ty: ftb,
            },
        ) => sa == sb && pa == pb && fa == fb && fta == ftb && mir_place_eq(ba, bb),
        (
            MirPlace::Index {
                base: ba,
                index: ia,
                elem_ty: ea,
                is_slice: isa,
            },
            MirPlace::Index {
                base: bb,
                index: ib,
                elem_ty: eb,
                is_slice: isb,
            },
        ) => ea == eb && isa == isb && mir_place_eq(ba, bb) && mir_expr_eq(ia, ib),
        _ => false,
    }
}

fn mir_prim_target_eq(a: &MirPrimTarget, b: &MirPrimTarget) -> bool {
    match (a, b) {
        (MirPrimTarget::Global(na), MirPrimTarget::Global(nb)) => na == nb,
        (MirPrimTarget::PtrExpr(ea), MirPrimTarget::PtrExpr(eb)) => mir_expr_eq(ea, eb),
        _ => false,
    }
}

fn mir_expr_eq(a: &MirExpr, b: &MirExpr) -> bool {
    if a.ty != b.ty {
        return false;
    }
    match (&a.kind, &b.kind) {
        (MirExprKind::IntLit(x), MirExprKind::IntLit(y)) => x == y,
        (MirExprKind::FloatLit(x), MirExprKind::FloatLit(y)) => x == y,
        (MirExprKind::BoolLit(x), MirExprKind::BoolLit(y)) => x == y,
        (MirExprKind::StringLit(x), MirExprKind::StringLit(y)) => x == y,
        (MirExprKind::TicksLit(x), MirExprKind::TicksLit(y)) => x == y,
        (
            MirExprKind::EnumVariant {
                enum_name: ea,
                variant: va,
            },
            MirExprKind::EnumVariant {
                enum_name: eb,
                variant: vb,
            },
        ) => ea == eb && va == vb,
        (MirExprKind::ConstRef(x), MirExprKind::ConstRef(y))
        | (MirExprKind::GlobalPrimAddr(x), MirExprKind::GlobalPrimAddr(y))
        | (MirExprKind::SubsystemAddr(x), MirExprKind::SubsystemAddr(y))
        | (MirExprKind::MmioAddr(x), MirExprKind::MmioAddr(y))
        | (MirExprKind::FnAddr(x), MirExprKind::FnAddr(y)) => x == y,
        (
            MirExprKind::MakePrimView {
                ptr: pa,
                cap: ca,
                kind: ka,
            },
            MirExprKind::MakePrimView {
                ptr: pb,
                cap: cb,
                kind: kb,
            },
        ) => ca == cb && ka == kb && mir_expr_eq(pa, pb),
        (MirExprKind::Var(x), MirExprKind::Var(y)) => x == y,
        (MirExprKind::ArrayLen(x), MirExprKind::ArrayLen(y)) => x == y,
        (MirExprKind::NowTicks, MirExprKind::NowTicks) => true,
        (MirExprKind::SysQuery(x), MirExprKind::SysQuery(y)) => x == y,
        (MirExprKind::ResultIsErr(x), MirExprKind::ResultIsErr(y))
        | (MirExprKind::ResultErr(x), MirExprKind::ResultErr(y))
        | (MirExprKind::ResultVal(x), MirExprKind::ResultVal(y)) => x == y,
        (MirExprKind::Unary { op: oa, expr: ea }, MirExprKind::Unary { op: ob, expr: eb }) => {
            oa == ob && mir_expr_eq(ea, eb)
        }
        (MirExprKind::AddrOfPlace(pa), MirExprKind::AddrOfPlace(pb)) => mir_place_eq(pa, pb),
        (
            MirExprKind::Binary {
                op: oa,
                lhs: la,
                rhs: ra,
            },
            MirExprKind::Binary {
                op: ob,
                lhs: lb,
                rhs: rb,
            },
        ) => oa == ob && mir_expr_eq(la, lb) && mir_expr_eq(ra, rb),
        (
            MirExprKind::Cast {
                expr: ea,
                target_ty: ta,
            },
            MirExprKind::Cast {
                expr: eb,
                target_ty: tb,
            },
        ) => ta == tb && mir_expr_eq(ea, eb),
        (
            MirExprKind::FieldAccess {
                base: ba,
                struct_name: sa,
                packed: pa,
                is_ptr: ipa,
                field: fa,
                raw_field_ty: rta,
            },
            MirExprKind::FieldAccess {
                base: bb,
                struct_name: sb,
                packed: pb,
                is_ptr: ipb,
                field: fb,
                raw_field_ty: rtb,
            },
        ) => sa == sb && pa == pb && ipa == ipb && fa == fb && rta == rtb && mir_expr_eq(ba, bb),
        (
            MirExprKind::StructLit {
                struct_name: sa,
                packed: pa,
                fields: fa,
            },
            MirExprKind::StructLit {
                struct_name: sb,
                packed: pb,
                fields: fb,
            },
        ) => {
            sa == sb
                && pa == pb
                && fa.len() == fb.len()
                && fa
                    .iter()
                    .zip(fb.iter())
                    .all(|((na, ea, ta), (nb, eb, tb))| na == nb && ta == tb && mir_expr_eq(ea, eb))
        }
        (
            MirExprKind::ArrayLit {
                elems: ea,
                elem_ty: ta,
            },
            MirExprKind::ArrayLit {
                elems: eb,
                elem_ty: tb,
            },
        ) => {
            ta == tb
                && ea.len() == eb.len()
                && ea.iter().zip(eb.iter()).all(|(x, y)| mir_expr_eq(x, y))
        }
        (MirExprKind::SliceLen(ea), MirExprKind::SliceLen(eb)) => mir_expr_eq(ea, eb),
        (
            MirExprKind::Index {
                base: ba,
                index: ia,
                is_slice: isa,
            },
            MirExprKind::Index {
                base: bb,
                index: ib,
                is_slice: isb,
            },
        ) => isa == isb && mir_expr_eq(ba, bb) && mir_expr_eq(ia, ib),
        (
            MirExprKind::SubSlice {
                base: ba,
                start: sa,
                end: ea,
                elem_ty: ta,
                base_is_array: aa,
            },
            MirExprKind::SubSlice {
                base: bb,
                start: sb,
                end: eb,
                elem_ty: tb,
                base_is_array: ab,
            },
        ) => {
            ta == tb
                && aa == ab
                && mir_expr_eq(ba, bb)
                && mir_expr_eq(sa, sb)
                && mir_expr_eq(ea, eb)
        }
        (
            MirExprKind::ArrayToSlice {
                array_place: pa,
                len: la,
                elem_ty: ta,
            },
            MirExprKind::ArrayToSlice {
                array_place: pb,
                len: lb,
                elem_ty: tb,
            },
        ) => la == lb && ta == tb && mir_place_eq(pa, pb),
        (
            MirExprKind::FromBytes {
                struct_name: sa,
                slice: la,
            },
            MirExprKind::FromBytes {
                struct_name: sb,
                slice: lb,
            },
        ) => sa == sb && mir_expr_eq(la, lb),
        (
            MirExprKind::MmioRead {
                base: ba,
                offset: oa,
                elem_bits: eba,
                index: ia,
                array_len: ala,
            },
            MirExprKind::MmioRead {
                base: bb,
                offset: ob,
                elem_bits: ebb,
                index: ib,
                array_len: alb,
            },
        ) => {
            oa == ob
                && eba == ebb
                && ala == alb
                && mir_expr_eq(ba, bb)
                && match (ia, ib) {
                    (Some(x), Some(y)) => mir_expr_eq(x, y),
                    (None, None) => true,
                    _ => false,
                }
        }
        (
            MirExprKind::CallDirect {
                callee: ca,
                is_extern_c: ea,
                args: aa,
            },
            MirExprKind::CallDirect {
                callee: cb,
                is_extern_c: eb,
                args: ab,
            },
        ) => {
            ca == cb
                && ea == eb
                && aa.len() == ab.len()
                && aa.iter().zip(ab.iter()).all(|(x, y)| mir_expr_eq(x, y))
        }
        (
            MirExprKind::RomCallU32 {
                addr: aa,
                stack_bound: sa,
                args: xa,
            },
            MirExprKind::RomCallU32 {
                addr: ab,
                stack_bound: sb,
                args: xb,
            },
        ) => {
            sa == sb
                && mir_expr_eq(aa, ab)
                && xa.len() == xb.len()
                && xa.iter().zip(xb.iter()).all(|(x, y)| mir_expr_eq(x, y))
        }
        (MirExprKind::PrimRead(pa), MirExprKind::PrimRead(pb)) => {
            match (pa.as_ref(), pb.as_ref()) {
                (
                    MirPrimReadOp::WatchRead {
                        target: ta,
                        elem_ty: ea,
                    },
                    MirPrimReadOp::WatchRead {
                        target: tb,
                        elem_ty: eb,
                    },
                )
                | (
                    MirPrimReadOp::GuardedGet {
                        target: ta,
                        elem_ty: ea,
                    },
                    MirPrimReadOp::GuardedGet {
                        target: tb,
                        elem_ty: eb,
                    },
                ) => ea == eb && mir_prim_target_eq(ta, tb),
                _ => false,
            }
        }
        _ => false,
    }
}

fn place_root_var(place: &MirPlace) -> Option<VarId> {
    match place {
        MirPlace::Var(vid) => Some(*vid),
        MirPlace::Field { base, .. } | MirPlace::Index { base, .. } => place_root_var(base),
        MirPlace::Deref(_, _) => None,
    }
}

#[derive(Default)]
struct VarUsageInfo {
    write_counts: BTreeMap<VarId, usize>,
    addr_taken_vars: BTreeSet<VarId>,
    yield_slot_vars: BTreeSet<VarId>,
    result_vars: BTreeSet<VarId>,
    indirect_fnptr_callees: BTreeSet<String>,
}

impl VarUsageInfo {
    fn record_write(&mut self, vid: VarId) {
        *self.write_counts.entry(vid).or_insert(0) += 1;
    }

    fn visit_place_exprs(&mut self, place: &MirPlace) {
        match place {
            MirPlace::Var(_) => {}
            MirPlace::Deref(ptr_expr, _) => self.visit_expr(ptr_expr),
            MirPlace::Field { base, .. } => self.visit_place_exprs(base),
            MirPlace::Index { base, index, .. } => {
                self.visit_place_exprs(base);
                self.visit_expr(index);
            }
        }
    }

    fn visit_prim_target(&mut self, target: &MirPrimTarget) {
        if let MirPrimTarget::PtrExpr(e) = target {
            self.visit_expr(e);
        }
    }

    fn visit_fmt_chunk(&mut self, ch: &MirFmtChunk) {
        match ch {
            MirFmtChunk::Literal(_) => {}
            MirFmtChunk::Int { value, .. }
            | MirFmtChunk::Float { value, .. }
            | MirFmtChunk::Bool(value)
            | MirFmtChunk::Str(value)
            | MirFmtChunk::Bytes(value)
            | MirFmtChunk::EnumName { value, .. }
            | MirFmtChunk::DebugSlice { slice: value, .. } => self.visit_expr(value),
        }
    }

    fn visit_expr(&mut self, expr: &MirExpr) {
        match &expr.kind {
            MirExprKind::IntLit(_)
            | MirExprKind::FloatLit(_)
            | MirExprKind::BoolLit(_)
            | MirExprKind::StringLit(_)
            | MirExprKind::TicksLit(_)
            | MirExprKind::EnumVariant { .. }
            | MirExprKind::ConstRef(_)
            | MirExprKind::GlobalPrimAddr(_)
            | MirExprKind::SubsystemAddr(_)
            | MirExprKind::MmioAddr(_)
            | MirExprKind::FnAddr(_)
            | MirExprKind::Var(_)
            | MirExprKind::ArrayLen(_)
            | MirExprKind::NowTicks
            | MirExprKind::SysQuery(_) => {}
            MirExprKind::ResultIsErr(vid)
            | MirExprKind::ResultErr(vid)
            | MirExprKind::ResultVal(vid) => {
                self.result_vars.insert(*vid);
            }
            MirExprKind::MakePrimView { ptr, .. } => self.visit_expr(ptr),
            MirExprKind::MmioRead { base, index, .. } => {
                self.visit_expr(base);
                if let Some(idx) = index {
                    self.visit_expr(idx);
                }
            }
            MirExprKind::Unary { expr: inner, .. }
            | MirExprKind::Cast { expr: inner, .. }
            | MirExprKind::FieldAccess { base: inner, .. }
            | MirExprKind::SliceLen(inner)
            | MirExprKind::FromBytes { slice: inner, .. }
            | MirExprKind::LeaseGet { lease: inner, .. }
            | MirExprKind::LeaseAddr { lease: inner, .. } => self.visit_expr(inner),
            MirExprKind::AddrOfPlace(place)
            | MirExprKind::ArrayToSlice {
                array_place: place, ..
            } => {
                if let Some(vid) = place_root_var(place) {
                    self.addr_taken_vars.insert(vid);
                }
                self.visit_place_exprs(place);
            }
            MirExprKind::Binary { lhs, rhs, .. }
            | MirExprKind::Index {
                base: lhs,
                index: rhs,
                ..
            } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            MirExprKind::SubSlice {
                base, start, end, ..
            } => {
                self.visit_expr(base);
                self.visit_expr(start);
                self.visit_expr(end);
            }
            MirExprKind::StructLit { fields, .. } => {
                for (_, fe, _) in fields {
                    self.visit_expr(fe);
                }
            }
            MirExprKind::ArrayLit { elems, .. } => {
                for el in elems {
                    self.visit_expr(el);
                }
            }
            MirExprKind::CallDirect { callee, args, .. } => {
                self.indirect_fnptr_callees.insert(callee.clone());
                for a in args {
                    self.visit_expr(a);
                }
            }
            MirExprKind::RomCallU32 { addr, args, .. } => {
                self.visit_expr(addr);
                for a in args {
                    self.visit_expr(a);
                }
            }
            MirExprKind::PrimRead(op) => match op.as_ref() {
                MirPrimReadOp::WatchRead { target, .. }
                | MirPrimReadOp::WatchGen { target, .. }
                | MirPrimReadOp::GuardedGet { target, .. }
                | MirPrimReadOp::InboxTryRecv { target, .. }
                | MirPrimReadOp::InboxLen { target, .. }
                | MirPrimReadOp::InboxIsEmpty { target, .. }
                | MirPrimReadOp::InboxIsFull { target, .. }
                | MirPrimReadOp::SignalIsSet { target }
                | MirPrimReadOp::SignalTryWait { target }
                | MirPrimReadOp::WaitQueueWakeOne { target, .. }
                | MirPrimReadOp::PingPongDmaBuf { target, .. }
                | MirPrimReadOp::PingPongBaseAddr { target, .. }
                | MirPrimReadOp::PoolTryAlloc { target, .. }
                | MirPrimReadOp::PoolAvail { target, .. }
                | MirPrimReadOp::PoolInUse { target, .. }
                | MirPrimReadOp::PoolIsExhausted { target, .. }
                | MirPrimReadOp::PoolCap { target, .. }
                | MirPrimReadOp::DmaSlotTake { target, .. }
                | MirPrimReadOp::DmaSlotRelease { target, .. }
                | MirPrimReadOp::DmaSlotIsBusy { target, .. }
                | MirPrimReadOp::DmaSlotIsEmpty { target, .. }
                | MirPrimReadOp::DmaSlotAddr { target, .. }
                | MirPrimReadOp::ByteRingReadSlice { target, .. }
                | MirPrimReadOp::ByteRingTryPopByte { target, .. }
                | MirPrimReadOp::ByteRingLen { target, .. }
                | MirPrimReadOp::ByteRingFreeSpace { target, .. }
                | MirPrimReadOp::ByteRingIsEmpty { target, .. }
                | MirPrimReadOp::ByteRingIsFull { target, .. } => self.visit_prim_target(target),
                MirPrimReadOp::DmaSlotPark { target, lease, .. } => {
                    self.visit_prim_target(target);
                    self.visit_expr(lease);
                }
                MirPrimReadOp::InboxTrySend { target, value, .. } => {
                    self.visit_prim_target(target);
                    self.visit_expr(value);
                }
                MirPrimReadOp::ByteRingPush { target, src, .. }
                | MirPrimReadOp::ByteRingPop {
                    target, dst: src, ..
                } => {
                    self.visit_prim_target(target);
                    self.visit_expr(src);
                }
                MirPrimReadOp::ByteRingPushByte { target, byte, .. } => {
                    self.visit_prim_target(target);
                    self.visit_expr(byte);
                }
                MirPrimReadOp::ByteRingGrantWrite { target, len, .. } => {
                    self.visit_prim_target(target);
                    self.visit_expr(len);
                }
                MirPrimReadOp::InboxViewTryRecv { view, .. }
                | MirPrimReadOp::InboxViewLen { view, .. }
                | MirPrimReadOp::InboxViewCap { view, .. }
                | MirPrimReadOp::InboxViewIsEmpty { view, .. }
                | MirPrimReadOp::InboxViewIsFull { view, .. }
                | MirPrimReadOp::ByteRingViewReadSlice { view }
                | MirPrimReadOp::ByteRingViewTryPopByte { view }
                | MirPrimReadOp::ByteRingViewLen { view }
                | MirPrimReadOp::ByteRingViewCap { view }
                | MirPrimReadOp::ByteRingViewFreeSpace { view }
                | MirPrimReadOp::ByteRingViewIsEmpty { view }
                | MirPrimReadOp::ByteRingViewIsFull { view } => self.visit_expr(view),
                MirPrimReadOp::InboxViewTrySend { view, value, .. } => {
                    self.visit_expr(view);
                    self.visit_expr(value);
                }
                MirPrimReadOp::ByteRingViewPush { view, src, .. }
                | MirPrimReadOp::ByteRingViewPop { view, dst: src } => {
                    self.visit_expr(view);
                    self.visit_expr(src);
                }
                MirPrimReadOp::ByteRingViewPushByte { view, byte, .. } => {
                    self.visit_expr(view);
                    self.visit_expr(byte);
                }
                MirPrimReadOp::ByteRingViewGrantWrite { view, len } => {
                    self.visit_expr(view);
                    self.visit_expr(len);
                }
            },
        }
    }

    fn visit_stmt(&mut self, stmt: &MirStmt) {
        match stmt {
            MirStmt::Assign { target, value, .. } => {
                if let Some(vid) = place_root_var(target) {
                    self.record_write(vid);
                }
                self.visit_place_exprs(target);
                self.visit_expr(value);
            }
            MirStmt::Expr(e) => self.visit_expr(e),
            MirStmt::PrimOp(op) => match op {
                MirPrimNonBlockingOp::WatchPublish { target, value, .. }
                | MirPrimNonBlockingOp::GuardedSet { target, value, .. }
                | MirPrimNonBlockingOp::InboxPost { target, value, .. } => {
                    self.visit_prim_target(target);
                    self.visit_expr(value);
                }
                MirPrimNonBlockingOp::SignalNotify { target, .. }
                | MirPrimNonBlockingOp::SignalClear { target }
                | MirPrimNonBlockingOp::WaitQueuePrepare { target }
                | MirPrimNonBlockingOp::WaitQueueCancel { target }
                | MirPrimNonBlockingOp::WaitQueueWakeAll { target, .. }
                | MirPrimNonBlockingOp::PingPongFlip { target, .. }
                | MirPrimNonBlockingOp::PingPongSetActive { target, .. }
                | MirPrimNonBlockingOp::ByteRingClear { target, .. } => {
                    self.visit_prim_target(target);
                }
                MirPrimNonBlockingOp::LeaseRelease { value, .. } => {
                    self.visit_expr(value);
                }
                MirPrimNonBlockingOp::LeaseNullify { target, .. } => {
                    if let Some(vid) = place_root_var(target) {
                        self.record_write(vid);
                    }
                    self.visit_place_exprs(target);
                }
                MirPrimNonBlockingOp::ByteRingConsume { target, len, .. }
                | MirPrimNonBlockingOp::ByteRingCommitWrite { target, len, .. }
                | MirPrimNonBlockingOp::PingPongCompleteHalf {
                    target, half: len, ..
                } => {
                    self.visit_prim_target(target);
                    self.visit_expr(len);
                }
                MirPrimNonBlockingOp::InboxViewPost { view, value, .. } => {
                    self.visit_expr(view);
                    self.visit_expr(value);
                }
                MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
                | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                    self.visit_expr(view);
                    self.visit_expr(len);
                }
                MirPrimNonBlockingOp::ByteRingViewClear { view } => self.visit_expr(view),
            },
            MirStmt::Intrinsic(intr) => match intr {
                MirIntrinsicOp::UseSystick { hz } => self.visit_expr(hz),
                MirIntrinsicOp::SetPriority { .. } => {}
                MirIntrinsicOp::SysOp { args, .. } => {
                    for a in args {
                        self.visit_expr(a);
                    }
                }
            },
            MirStmt::EnterCritical
            | MirStmt::ExitCritical
            | MirStmt::EndInboxTurn
            | MirStmt::Break
            | MirStmt::Continue
            | MirStmt::SetIrqSelector { .. }
            | MirStmt::BoundedForLatch
            | MirStmt::ZeroTaskLocals => {}
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.visit_expr(cond);
                for s in then_body {
                    self.visit_stmt(s);
                }
                for s in else_body {
                    self.visit_stmt(s);
                }
            }
            MirStmt::While { cond, body } => {
                self.visit_expr(cond);
                for s in body {
                    self.visit_stmt(s);
                }
            }
            MirStmt::Switch {
                discr,
                arms,
                default,
            } => {
                self.visit_expr(discr);
                for (_, body) in arms {
                    for s in body {
                        self.visit_stmt(s);
                    }
                }
                for s in default {
                    self.visit_stmt(s);
                }
            }
            MirStmt::Return(opt) => {
                if let Some(e) = opt {
                    self.visit_expr(e);
                }
            }
            MirStmt::Error { reason, partial } => {
                if let Some(r) = reason {
                    self.visit_expr(r);
                }
                if let Some(p) = partial {
                    self.visit_expr(p);
                }
            }
            MirStmt::SpawnActivate { args, .. } => {
                for a in args {
                    self.visit_expr(a);
                }
            }
            MirStmt::PrintFmt { chunks, .. } => {
                for ch in chunks {
                    self.visit_fmt_chunk(ch);
                }
            }
            MirStmt::FmtBuf {
                dst_res,
                buf,
                chunks,
            } => {
                self.record_write(*dst_res);
                self.result_vars.insert(*dst_res);
                self.visit_expr(buf);
                for ch in chunks {
                    self.visit_fmt_chunk(ch);
                }
            }
            MirStmt::FmtRing {
                dst_res,
                ring,
                chunks,
                ..
            } => {
                self.record_write(*dst_res);
                self.result_vars.insert(*dst_res);
                self.visit_expr(ring);
                for ch in chunks {
                    self.visit_fmt_chunk(ch);
                }
            }
            MirStmt::MmioWrite {
                base, index, value, ..
            } => {
                self.visit_expr(base);
                if let Some(idx) = index {
                    self.visit_expr(idx);
                }
                self.visit_expr(value);
            }
        }
    }

    fn visit_terminator(&mut self, term: &MirTerminator) {
        match term {
            MirTerminator::Goto(_) => {}
            MirTerminator::Branch { cond, .. } => self.visit_expr(cond),
            MirTerminator::Switch { discr, .. } => self.visit_expr(discr),
            MirTerminator::Return(opt) => {
                if let Some(e) = opt {
                    self.visit_expr(e);
                }
            }
            MirTerminator::Error { reason, partial } => {
                if let Some(r) = reason {
                    self.visit_expr(r);
                }
                if let Some(p) = partial {
                    self.visit_expr(p);
                }
            }
            MirTerminator::Yield(yk) => match yk {
                MirYieldKind::EveryLatch {
                    next_deadline_var,
                    period_var,
                    ..
                } => {
                    self.yield_slot_vars.insert(*next_deadline_var);
                    self.yield_slot_vars.insert(*period_var);
                }
                MirYieldKind::DelayPoll { deadline_var, .. } => {
                    self.yield_slot_vars.insert(*deadline_var);
                }
                MirYieldKind::ShareCheckpoint { .. } => {}
                MirYieldKind::Wait { arms, timer, .. } => {
                    if let Some(t) = timer {
                        self.yield_slot_vars.insert(t.deadline_var);
                        if let Some(r) = t.timeout_err_res {
                            self.yield_slot_vars.insert(r);
                            self.record_write(r);
                            self.result_vars.insert(r);
                        }
                        if let Some((lv, _)) = &t.release_lease_on_timeout {
                            self.yield_slot_vars.insert(*lv);
                        }
                    }
                    for arm in arms {
                        match &arm.op {
                            MirWaitArmOp::InboxRecv {
                                target,
                                dst_val,
                                dst_res,
                                ..
                            }
                            | MirWaitArmOp::ByteRingWaitRead {
                                target,
                                dst_val,
                                dst_res,
                                ..
                            }
                            | MirWaitArmOp::ByteRingWaitWrite {
                                target,
                                dst_val,
                                dst_res,
                                ..
                            }
                            | MirWaitArmOp::PoolAlloc {
                                target,
                                dst_val,
                                dst_res,
                                ..
                            } => {
                                self.visit_prim_target(target);
                                for v in [dst_val, dst_res].into_iter().flatten() {
                                    self.yield_slot_vars.insert(*v);
                                    self.record_write(*v);
                                }
                                if let Some(r) = dst_res {
                                    self.result_vars.insert(*r);
                                }
                            }
                            MirWaitArmOp::WatchWaitChange {
                                target,
                                last_gen_var,
                                seeded_var,
                                dst_val,
                                dst_res,
                                ..
                            } => {
                                self.visit_prim_target(target);
                                self.yield_slot_vars.insert(*last_gen_var);
                                self.record_write(*last_gen_var);
                                if let Some(sv) = seeded_var {
                                    self.yield_slot_vars.insert(*sv);
                                    self.record_write(*sv);
                                }
                                for v in [dst_val, dst_res].into_iter().flatten() {
                                    self.yield_slot_vars.insert(*v);
                                    self.record_write(*v);
                                }
                                if let Some(r) = dst_res {
                                    self.result_vars.insert(*r);
                                }
                            }
                            MirWaitArmOp::InboxSend {
                                target,
                                val_var,
                                dst_res,
                                ..
                            } => {
                                self.visit_prim_target(target);
                                self.yield_slot_vars.insert(*val_var);
                                if let Some(r) = dst_res {
                                    self.yield_slot_vars.insert(*r);
                                    self.record_write(*r);
                                    self.result_vars.insert(*r);
                                }
                            }
                            MirWaitArmOp::SignalWait {
                                target, dst_res, ..
                            }
                            | MirWaitArmOp::WaitQueueCommit {
                                target, dst_res, ..
                            } => {
                                self.visit_prim_target(target);
                                if let Some(r) = dst_res {
                                    self.yield_slot_vars.insert(*r);
                                    self.record_write(*r);
                                    self.result_vars.insert(*r);
                                }
                            }
                            MirWaitArmOp::PingPongWaitChunk {
                                target, dst_slice, ..
                            } => {
                                self.visit_prim_target(target);
                                self.yield_slot_vars.insert(*dst_slice);
                                self.record_write(*dst_slice);
                            }
                            MirWaitArmOp::InboxViewRecv {
                                view,
                                dst_val,
                                dst_res,
                                ..
                            }
                            | MirWaitArmOp::ByteRingViewWaitRead {
                                view,
                                dst_val,
                                dst_res,
                                ..
                            }
                            | MirWaitArmOp::ByteRingViewWaitWrite {
                                view,
                                dst_val,
                                dst_res,
                                ..
                            } => {
                                self.visit_expr(view);
                                for v in [dst_val, dst_res].into_iter().flatten() {
                                    self.yield_slot_vars.insert(*v);
                                    self.record_write(*v);
                                }
                                if let Some(r) = dst_res {
                                    self.result_vars.insert(*r);
                                }
                            }
                            MirWaitArmOp::InboxViewSend {
                                view,
                                val_var,
                                dst_res,
                                ..
                            } => {
                                self.visit_expr(view);
                                self.yield_slot_vars.insert(*val_var);
                                if let Some(r) = dst_res {
                                    self.yield_slot_vars.insert(*r);
                                    self.record_write(*r);
                                    self.result_vars.insert(*r);
                                }
                            }
                        }
                    }
                }
                MirYieldKind::CallYielding { args, dst, .. } => {
                    for a in args {
                        self.visit_expr(a);
                    }
                    if let Some(d) = dst {
                        match d {
                            MirCallDst::Value(v) => {
                                self.yield_slot_vars.insert(*v);
                                self.record_write(*v);
                            }
                            MirCallDst::Result(r) => {
                                self.yield_slot_vars.insert(*r);
                                self.record_write(*r);
                                self.result_vars.insert(*r);
                            }
                        }
                    }
                }
                MirYieldKind::PollChild { dst, .. } => {
                    if let Some(d) = dst {
                        match d {
                            MirCallDst::Value(v) => {
                                self.yield_slot_vars.insert(*v);
                                self.record_write(*v);
                            }
                            MirCallDst::Result(r) => {
                                self.yield_slot_vars.insert(*r);
                                self.record_write(*r);
                                self.result_vars.insert(*r);
                            }
                        }
                    }
                }
            },
        }
    }

    fn collect_blocks(blocks: &[MirBlock]) -> Self {
        let mut info = Self::default();
        for blk in blocks {
            for s in &blk.stmts {
                info.visit_stmt(s);
            }
            info.visit_terminator(&blk.terminator);
        }
        info
    }

    fn collect_fn(func: &MirFunction) -> Self {
        match &func.body {
            MirFnBody::Direct(stmts) => {
                let mut info = Self::default();
                for s in stmts {
                    info.visit_stmt(s);
                }
                info
            }
            MirFnBody::Coroutine { blocks, .. } => Self::collect_blocks(blocks),
        }
    }
}

fn expr_reads_var(expr: &MirExpr, target: VarId) -> bool {
    match &expr.kind {
        MirExprKind::Var(vid)
        | MirExprKind::ResultIsErr(vid)
        | MirExprKind::ResultErr(vid)
        | MirExprKind::ResultVal(vid) => *vid == target,
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::StringLit(_)
        | MirExprKind::TicksLit(_)
        | MirExprKind::EnumVariant { .. }
        | MirExprKind::ConstRef(_)
        | MirExprKind::GlobalPrimAddr(_)
        | MirExprKind::SubsystemAddr(_)
        | MirExprKind::MmioAddr(_)
        | MirExprKind::FnAddr(_)
        | MirExprKind::ArrayLen(_)
        | MirExprKind::NowTicks
        | MirExprKind::SysQuery(_) => false,
        MirExprKind::MakePrimView { ptr, .. } => expr_reads_var(ptr, target),
        MirExprKind::MmioRead { base, index, .. } => {
            expr_reads_var(base, target)
                || index.as_ref().is_some_and(|i| expr_reads_var(i, target))
        }
        MirExprKind::Unary { expr: inner, .. }
        | MirExprKind::Cast { expr: inner, .. }
        | MirExprKind::FieldAccess { base: inner, .. }
        | MirExprKind::SliceLen(inner)
        | MirExprKind::FromBytes { slice: inner, .. }
        | MirExprKind::LeaseGet { lease: inner, .. }
        | MirExprKind::LeaseAddr { lease: inner, .. } => expr_reads_var(inner, target),
        MirExprKind::AddrOfPlace(place)
        | MirExprKind::ArrayToSlice {
            array_place: place, ..
        } => place_reads_var(place, target, true),
        MirExprKind::Binary { lhs, rhs, .. }
        | MirExprKind::Index {
            base: lhs,
            index: rhs,
            ..
        } => expr_reads_var(lhs, target) || expr_reads_var(rhs, target),
        MirExprKind::SubSlice {
            base, start, end, ..
        } => {
            expr_reads_var(base, target)
                || expr_reads_var(start, target)
                || expr_reads_var(end, target)
        }
        MirExprKind::StructLit { fields, .. } => {
            fields.iter().any(|(_, fe, _)| expr_reads_var(fe, target))
        }
        MirExprKind::ArrayLit { elems, .. } | MirExprKind::CallDirect { args: elems, .. } => {
            elems.iter().any(|e| expr_reads_var(e, target))
        }
        MirExprKind::RomCallU32 { addr, args, .. } => {
            expr_reads_var(addr, target) || args.iter().any(|e| expr_reads_var(e, target))
        }
        MirExprKind::PrimRead(op) => match op.as_ref() {
            MirPrimReadOp::WatchRead { target: t, .. }
            | MirPrimReadOp::WatchGen { target: t, .. }
            | MirPrimReadOp::GuardedGet { target: t, .. }
            | MirPrimReadOp::InboxTryRecv { target: t, .. }
            | MirPrimReadOp::InboxLen { target: t, .. }
            | MirPrimReadOp::InboxIsEmpty { target: t, .. }
            | MirPrimReadOp::InboxIsFull { target: t, .. }
            | MirPrimReadOp::SignalIsSet { target: t }
            | MirPrimReadOp::SignalTryWait { target: t }
            | MirPrimReadOp::WaitQueueWakeOne { target: t, .. }
            | MirPrimReadOp::PingPongDmaBuf { target: t, .. }
            | MirPrimReadOp::PingPongBaseAddr { target: t, .. }
            | MirPrimReadOp::PoolTryAlloc { target: t, .. }
            | MirPrimReadOp::PoolAvail { target: t, .. }
            | MirPrimReadOp::PoolInUse { target: t, .. }
            | MirPrimReadOp::PoolIsExhausted { target: t, .. }
            | MirPrimReadOp::PoolCap { target: t, .. }
            | MirPrimReadOp::DmaSlotTake { target: t, .. }
            | MirPrimReadOp::DmaSlotRelease { target: t, .. }
            | MirPrimReadOp::DmaSlotIsBusy { target: t, .. }
            | MirPrimReadOp::DmaSlotIsEmpty { target: t, .. }
            | MirPrimReadOp::DmaSlotAddr { target: t, .. }
            | MirPrimReadOp::ByteRingReadSlice { target: t, .. }
            | MirPrimReadOp::ByteRingTryPopByte { target: t, .. }
            | MirPrimReadOp::ByteRingLen { target: t, .. }
            | MirPrimReadOp::ByteRingFreeSpace { target: t, .. }
            | MirPrimReadOp::ByteRingIsEmpty { target: t, .. }
            | MirPrimReadOp::ByteRingIsFull { target: t, .. } => prim_target_reads_var(t, target),
            MirPrimReadOp::DmaSlotPark {
                target: t, lease, ..
            } => prim_target_reads_var(t, target) || expr_reads_var(lease, target),
            MirPrimReadOp::InboxTrySend {
                target: t, value, ..
            } => prim_target_reads_var(t, target) || expr_reads_var(value, target),
            MirPrimReadOp::ByteRingPush { target: t, src, .. }
            | MirPrimReadOp::ByteRingPop {
                target: t,
                dst: src,
                ..
            } => prim_target_reads_var(t, target) || expr_reads_var(src, target),
            MirPrimReadOp::ByteRingPushByte {
                target: t, byte, ..
            } => prim_target_reads_var(t, target) || expr_reads_var(byte, target),
            MirPrimReadOp::ByteRingGrantWrite { target: t, len, .. } => {
                prim_target_reads_var(t, target) || expr_reads_var(len, target)
            }
            MirPrimReadOp::InboxViewTryRecv { view, .. }
            | MirPrimReadOp::InboxViewLen { view, .. }
            | MirPrimReadOp::InboxViewCap { view, .. }
            | MirPrimReadOp::InboxViewIsEmpty { view, .. }
            | MirPrimReadOp::InboxViewIsFull { view, .. }
            | MirPrimReadOp::ByteRingViewReadSlice { view }
            | MirPrimReadOp::ByteRingViewTryPopByte { view }
            | MirPrimReadOp::ByteRingViewLen { view }
            | MirPrimReadOp::ByteRingViewCap { view }
            | MirPrimReadOp::ByteRingViewFreeSpace { view }
            | MirPrimReadOp::ByteRingViewIsEmpty { view }
            | MirPrimReadOp::ByteRingViewIsFull { view } => expr_reads_var(view, target),
            MirPrimReadOp::InboxViewTrySend { view, value, .. } => {
                expr_reads_var(view, target) || expr_reads_var(value, target)
            }
            MirPrimReadOp::ByteRingViewPush { view, src, .. }
            | MirPrimReadOp::ByteRingViewPop { view, dst: src } => {
                expr_reads_var(view, target) || expr_reads_var(src, target)
            }
            MirPrimReadOp::ByteRingViewPushByte { view, byte, .. } => {
                expr_reads_var(view, target) || expr_reads_var(byte, target)
            }
            MirPrimReadOp::ByteRingViewGrantWrite { view, len } => {
                expr_reads_var(view, target) || expr_reads_var(len, target)
            }
        },
    }
}

fn prim_target_reads_var(t: &MirPrimTarget, target: VarId) -> bool {
    match t {
        MirPrimTarget::Global(_) => false,
        MirPrimTarget::PtrExpr(e) => expr_reads_var(e, target),
    }
}

fn place_reads_var(place: &MirPlace, target: VarId, count_root: bool) -> bool {
    match place {
        MirPlace::Var(vid) => count_root && *vid == target,
        MirPlace::Deref(ptr_expr, _) => expr_reads_var(ptr_expr, target),
        MirPlace::Field { base, .. } => place_reads_var(base, target, count_root),
        MirPlace::Index { base, index, .. } => {
            place_reads_var(base, target, count_root) || expr_reads_var(index, target)
        }
    }
}

fn fmt_chunk_reads_var(ch: &MirFmtChunk, target: VarId) -> bool {
    match ch {
        MirFmtChunk::Literal(_) => false,
        MirFmtChunk::Int { value, .. }
        | MirFmtChunk::Float { value, .. }
        | MirFmtChunk::Bool(value)
        | MirFmtChunk::Str(value)
        | MirFmtChunk::Bytes(value)
        | MirFmtChunk::EnumName { value, .. }
        | MirFmtChunk::DebugSlice { slice: value, .. } => expr_reads_var(value, target),
    }
}

fn stmt_reads_var(stmt: &MirStmt, target: VarId) -> bool {
    match stmt {
        MirStmt::Assign {
            target: place,
            value,
            ..
        } => place_reads_var(place, target, false) || expr_reads_var(value, target),
        MirStmt::Expr(e) => expr_reads_var(e, target),
        MirStmt::PrimOp(op) => match op {
            MirPrimNonBlockingOp::WatchPublish {
                target: t, value, ..
            }
            | MirPrimNonBlockingOp::GuardedSet {
                target: t, value, ..
            }
            | MirPrimNonBlockingOp::InboxPost {
                target: t, value, ..
            } => prim_target_reads_var(t, target) || expr_reads_var(value, target),
            MirPrimNonBlockingOp::SignalNotify { target: t, .. }
            | MirPrimNonBlockingOp::SignalClear { target: t }
            | MirPrimNonBlockingOp::WaitQueuePrepare { target: t }
            | MirPrimNonBlockingOp::WaitQueueCancel { target: t }
            | MirPrimNonBlockingOp::WaitQueueWakeAll { target: t, .. }
            | MirPrimNonBlockingOp::PingPongFlip { target: t, .. }
            | MirPrimNonBlockingOp::PingPongSetActive { target: t, .. }
            | MirPrimNonBlockingOp::ByteRingClear { target: t, .. } => {
                prim_target_reads_var(t, target)
            }
            MirPrimNonBlockingOp::LeaseRelease { value, .. } => expr_reads_var(value, target),
            MirPrimNonBlockingOp::LeaseNullify { target: place, .. } => {
                place_reads_var(place, target, false)
            }
            MirPrimNonBlockingOp::ByteRingConsume { target: t, len, .. }
            | MirPrimNonBlockingOp::ByteRingCommitWrite { target: t, len, .. }
            | MirPrimNonBlockingOp::PingPongCompleteHalf {
                target: t,
                half: len,
                ..
            } => prim_target_reads_var(t, target) || expr_reads_var(len, target),
            MirPrimNonBlockingOp::InboxViewPost { view, value, .. } => {
                expr_reads_var(view, target) || expr_reads_var(value, target)
            }
            MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
            | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                expr_reads_var(view, target) || expr_reads_var(len, target)
            }
            MirPrimNonBlockingOp::ByteRingViewClear { view } => expr_reads_var(view, target),
        },
        MirStmt::Intrinsic(intr) => match intr {
            MirIntrinsicOp::UseSystick { hz } => expr_reads_var(hz, target),
            MirIntrinsicOp::SetPriority { .. } => false,
            MirIntrinsicOp::SysOp { args, .. } => args.iter().any(|a| expr_reads_var(a, target)),
        },
        MirStmt::EnterCritical
        | MirStmt::ExitCritical
        | MirStmt::EndInboxTurn
        | MirStmt::Break
        | MirStmt::Continue
        | MirStmt::SetIrqSelector { .. }
        | MirStmt::BoundedForLatch
        | MirStmt::ZeroTaskLocals => false,
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            expr_reads_var(cond, target)
                || then_body.iter().any(|s| stmt_reads_var(s, target))
                || else_body.iter().any(|s| stmt_reads_var(s, target))
        }
        MirStmt::While { cond, body } => {
            expr_reads_var(cond, target) || body.iter().any(|s| stmt_reads_var(s, target))
        }
        MirStmt::Switch {
            discr,
            arms,
            default,
        } => {
            expr_reads_var(discr, target)
                || arms
                    .iter()
                    .any(|(_, b)| b.iter().any(|s| stmt_reads_var(s, target)))
                || default.iter().any(|s| stmt_reads_var(s, target))
        }
        MirStmt::Return(opt) => opt.as_ref().is_some_and(|e| expr_reads_var(e, target)),
        MirStmt::Error { reason, partial } => {
            reason.as_ref().is_some_and(|e| expr_reads_var(e, target))
                || partial.as_ref().is_some_and(|e| expr_reads_var(e, target))
        }
        MirStmt::SpawnActivate { args, .. } => args.iter().any(|a| expr_reads_var(a, target)),
        MirStmt::PrintFmt { chunks, .. } => chunks.iter().any(|c| fmt_chunk_reads_var(c, target)),
        MirStmt::FmtBuf { buf, chunks, .. } => {
            expr_reads_var(buf, target) || chunks.iter().any(|c| fmt_chunk_reads_var(c, target))
        }
        MirStmt::FmtRing { ring, chunks, .. } => {
            expr_reads_var(ring, target) || chunks.iter().any(|c| fmt_chunk_reads_var(c, target))
        }
        MirStmt::MmioWrite {
            base, index, value, ..
        } => {
            expr_reads_var(base, target)
                || index.as_ref().is_some_and(|i| expr_reads_var(i, target))
                || expr_reads_var(value, target)
        }
    }
}

fn boot_step_reads_var(step: &MirBootStep, target: VarId) -> bool {
    match step {
        MirBootStep::Stmt(s) => stmt_reads_var(s, target),
        MirBootStep::RunYieldingFn { args, .. } | MirBootStep::SpawnActivate { args, .. } => {
            args.iter().any(|a| expr_reads_var(a, target))
        }
    }
}

fn subst_prim_target(t: &mut MirPrimTarget, target: VarId, repl: &MirExpr) {
    if let MirPrimTarget::PtrExpr(pe) = t {
        subst_expr(pe, target, repl);
        if let MirExprKind::GlobalPrimAddr(g) = &pe.kind {
            *t = MirPrimTarget::Global(g.clone());
        }
    }
}

fn subst_place(place: &mut MirPlace, target: VarId, repl: &MirExpr, subst_root: bool) {
    match place {
        MirPlace::Var(vid) => {
            let _ = (vid, subst_root);
        }
        MirPlace::Deref(ptr_expr, _) => {
            subst_expr(ptr_expr, target, repl);
            if let MirExprKind::AddrOfPlace(inner) = &mut ptr_expr.kind {
                let inner_place = inner.as_ref().clone();
                *place = inner_place;
            }
        }
        MirPlace::Field { base, .. } => subst_place(base, target, repl, subst_root),
        MirPlace::Index { base, index, .. } => {
            subst_place(base, target, repl, subst_root);
            subst_expr(index, target, repl);
        }
    }
}

fn subst_fmt_chunk(ch: &mut MirFmtChunk, target: VarId, repl: &MirExpr) {
    match ch {
        MirFmtChunk::Literal(_) => {}
        MirFmtChunk::Int { value, .. }
        | MirFmtChunk::Float { value, .. }
        | MirFmtChunk::Bool(value)
        | MirFmtChunk::Str(value)
        | MirFmtChunk::Bytes(value)
        | MirFmtChunk::EnumName { value, .. }
        | MirFmtChunk::DebugSlice { slice: value, .. } => subst_expr(value, target, repl),
    }
}

fn subst_expr(expr: &mut MirExpr, target: VarId, repl: &MirExpr) {
    match &mut expr.kind {
        MirExprKind::Var(vid) if *vid == target => {
            *expr = repl.clone();
        }
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::StringLit(_)
        | MirExprKind::TicksLit(_)
        | MirExprKind::EnumVariant { .. }
        | MirExprKind::ConstRef(_)
        | MirExprKind::GlobalPrimAddr(_)
        | MirExprKind::SubsystemAddr(_)
        | MirExprKind::MmioAddr(_)
        | MirExprKind::FnAddr(_)
        | MirExprKind::Var(_)
        | MirExprKind::ArrayLen(_)
        | MirExprKind::ResultIsErr(_)
        | MirExprKind::ResultErr(_)
        | MirExprKind::ResultVal(_)
        | MirExprKind::NowTicks
        | MirExprKind::SysQuery(_) => {}
        MirExprKind::MakePrimView { ptr, .. } => subst_expr(ptr, target, repl),
        MirExprKind::MmioRead { base, index, .. } => {
            subst_expr(base, target, repl);
            if let Some(idx) = index {
                subst_expr(idx, target, repl);
            }
        }
        MirExprKind::Unary { expr: inner, .. }
        | MirExprKind::Cast { expr: inner, .. }
        | MirExprKind::FieldAccess { base: inner, .. }
        | MirExprKind::SliceLen(inner)
        | MirExprKind::FromBytes { slice: inner, .. }
        | MirExprKind::LeaseGet { lease: inner, .. }
        | MirExprKind::LeaseAddr { lease: inner, .. } => subst_expr(inner, target, repl),
        MirExprKind::AddrOfPlace(place) => {
            subst_place(place, target, repl, false);
            if let MirPlace::Deref(ptr_expr, _) = place.as_ref() {
                if ptr_expr.ty == expr.ty {
                    *expr = ptr_expr.as_ref().clone();
                }
            }
        }
        MirExprKind::ArrayToSlice { array_place, .. } => {
            subst_place(array_place, target, repl, false);
        }
        MirExprKind::Binary { lhs, rhs, .. }
        | MirExprKind::Index {
            base: lhs,
            index: rhs,
            ..
        } => {
            subst_expr(lhs, target, repl);
            subst_expr(rhs, target, repl);
        }
        MirExprKind::SubSlice {
            base, start, end, ..
        } => {
            subst_expr(base, target, repl);
            subst_expr(start, target, repl);
            subst_expr(end, target, repl);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, fe, _) in fields {
                subst_expr(fe, target, repl);
            }
        }
        MirExprKind::ArrayLit { elems, .. } | MirExprKind::CallDirect { args: elems, .. } => {
            for e in elems {
                subst_expr(e, target, repl);
            }
        }
        MirExprKind::RomCallU32 { addr, args, .. } => {
            subst_expr(addr, target, repl);
            for e in args {
                subst_expr(e, target, repl);
            }
        }
        MirExprKind::PrimRead(op) => match op.as_mut() {
            MirPrimReadOp::WatchRead { target: t, .. }
            | MirPrimReadOp::WatchGen { target: t, .. }
            | MirPrimReadOp::GuardedGet { target: t, .. }
            | MirPrimReadOp::InboxTryRecv { target: t, .. }
            | MirPrimReadOp::InboxLen { target: t, .. }
            | MirPrimReadOp::InboxIsEmpty { target: t, .. }
            | MirPrimReadOp::InboxIsFull { target: t, .. }
            | MirPrimReadOp::SignalIsSet { target: t }
            | MirPrimReadOp::SignalTryWait { target: t }
            | MirPrimReadOp::WaitQueueWakeOne { target: t, .. }
            | MirPrimReadOp::PingPongDmaBuf { target: t, .. }
            | MirPrimReadOp::PingPongBaseAddr { target: t, .. }
            | MirPrimReadOp::PoolTryAlloc { target: t, .. }
            | MirPrimReadOp::PoolAvail { target: t, .. }
            | MirPrimReadOp::PoolInUse { target: t, .. }
            | MirPrimReadOp::PoolIsExhausted { target: t, .. }
            | MirPrimReadOp::PoolCap { target: t, .. }
            | MirPrimReadOp::DmaSlotTake { target: t, .. }
            | MirPrimReadOp::DmaSlotRelease { target: t, .. }
            | MirPrimReadOp::DmaSlotIsBusy { target: t, .. }
            | MirPrimReadOp::DmaSlotIsEmpty { target: t, .. }
            | MirPrimReadOp::DmaSlotAddr { target: t, .. }
            | MirPrimReadOp::ByteRingReadSlice { target: t, .. }
            | MirPrimReadOp::ByteRingTryPopByte { target: t, .. }
            | MirPrimReadOp::ByteRingLen { target: t, .. }
            | MirPrimReadOp::ByteRingFreeSpace { target: t, .. }
            | MirPrimReadOp::ByteRingIsEmpty { target: t, .. }
            | MirPrimReadOp::ByteRingIsFull { target: t, .. } => {
                subst_prim_target(t, target, repl);
            }
            MirPrimReadOp::DmaSlotPark {
                target: t, lease, ..
            } => {
                subst_prim_target(t, target, repl);
                subst_expr(lease, target, repl);
            }
            MirPrimReadOp::InboxTrySend {
                target: t, value, ..
            } => {
                subst_prim_target(t, target, repl);
                subst_expr(value, target, repl);
            }
            MirPrimReadOp::ByteRingPush { target: t, src, .. }
            | MirPrimReadOp::ByteRingPop {
                target: t,
                dst: src,
                ..
            } => {
                subst_prim_target(t, target, repl);
                subst_expr(src, target, repl);
            }
            MirPrimReadOp::ByteRingPushByte {
                target: t, byte, ..
            } => {
                subst_prim_target(t, target, repl);
                subst_expr(byte, target, repl);
            }
            MirPrimReadOp::ByteRingGrantWrite { target: t, len, .. } => {
                subst_prim_target(t, target, repl);
                subst_expr(len, target, repl);
            }
            MirPrimReadOp::InboxViewTryRecv { view, .. }
            | MirPrimReadOp::InboxViewLen { view, .. }
            | MirPrimReadOp::InboxViewCap { view, .. }
            | MirPrimReadOp::InboxViewIsEmpty { view, .. }
            | MirPrimReadOp::InboxViewIsFull { view, .. }
            | MirPrimReadOp::ByteRingViewReadSlice { view }
            | MirPrimReadOp::ByteRingViewTryPopByte { view }
            | MirPrimReadOp::ByteRingViewLen { view }
            | MirPrimReadOp::ByteRingViewCap { view }
            | MirPrimReadOp::ByteRingViewFreeSpace { view }
            | MirPrimReadOp::ByteRingViewIsEmpty { view }
            | MirPrimReadOp::ByteRingViewIsFull { view } => subst_expr(view, target, repl),
            MirPrimReadOp::InboxViewTrySend { view, value, .. } => {
                subst_expr(view, target, repl);
                subst_expr(value, target, repl);
            }
            MirPrimReadOp::ByteRingViewPush { view, src, .. }
            | MirPrimReadOp::ByteRingViewPop { view, dst: src } => {
                subst_expr(view, target, repl);
                subst_expr(src, target, repl);
            }
            MirPrimReadOp::ByteRingViewPushByte { view, byte, .. } => {
                subst_expr(view, target, repl);
                subst_expr(byte, target, repl);
            }
            MirPrimReadOp::ByteRingViewGrantWrite { view, len } => {
                subst_expr(view, target, repl);
                subst_expr(len, target, repl);
            }
        },
    }
}

fn subst_stmt(stmt: &mut MirStmt, target: VarId, repl: &MirExpr) {
    match stmt {
        MirStmt::Assign {
            target: place,
            value,
            ..
        } => {
            subst_place(place, target, repl, false);
            subst_expr(value, target, repl);
        }
        MirStmt::Expr(e) => subst_expr(e, target, repl),
        MirStmt::PrimOp(op) => match op {
            MirPrimNonBlockingOp::WatchPublish {
                target: t, value, ..
            }
            | MirPrimNonBlockingOp::GuardedSet {
                target: t, value, ..
            }
            | MirPrimNonBlockingOp::InboxPost {
                target: t, value, ..
            } => {
                subst_prim_target(t, target, repl);
                subst_expr(value, target, repl);
            }
            MirPrimNonBlockingOp::SignalNotify { target: t, .. }
            | MirPrimNonBlockingOp::SignalClear { target: t }
            | MirPrimNonBlockingOp::WaitQueuePrepare { target: t }
            | MirPrimNonBlockingOp::WaitQueueCancel { target: t }
            | MirPrimNonBlockingOp::WaitQueueWakeAll { target: t, .. }
            | MirPrimNonBlockingOp::PingPongFlip { target: t, .. }
            | MirPrimNonBlockingOp::PingPongSetActive { target: t, .. }
            | MirPrimNonBlockingOp::ByteRingClear { target: t, .. } => {
                subst_prim_target(t, target, repl);
            }
            MirPrimNonBlockingOp::LeaseRelease { value, .. } => subst_expr(value, target, repl),
            MirPrimNonBlockingOp::LeaseNullify { target: place, .. } => {
                subst_place(place, target, repl, false);
            }
            MirPrimNonBlockingOp::ByteRingConsume { target: t, len, .. }
            | MirPrimNonBlockingOp::ByteRingCommitWrite { target: t, len, .. }
            | MirPrimNonBlockingOp::PingPongCompleteHalf {
                target: t,
                half: len,
                ..
            } => {
                subst_prim_target(t, target, repl);
                subst_expr(len, target, repl);
            }
            MirPrimNonBlockingOp::InboxViewPost { view, value, .. } => {
                subst_expr(view, target, repl);
                subst_expr(value, target, repl);
            }
            MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
            | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                subst_expr(view, target, repl);
                subst_expr(len, target, repl);
            }
            MirPrimNonBlockingOp::ByteRingViewClear { view } => subst_expr(view, target, repl),
        },
        MirStmt::Intrinsic(intr) => match intr {
            MirIntrinsicOp::UseSystick { hz } => subst_expr(hz, target, repl),
            MirIntrinsicOp::SetPriority { .. } => {}
            MirIntrinsicOp::SysOp { args, .. } => {
                for a in args {
                    subst_expr(a, target, repl);
                }
            }
        },
        MirStmt::EnterCritical
        | MirStmt::ExitCritical
        | MirStmt::EndInboxTurn
        | MirStmt::Break
        | MirStmt::Continue
        | MirStmt::SetIrqSelector { .. }
        | MirStmt::BoundedForLatch
        | MirStmt::ZeroTaskLocals => {}
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            subst_expr(cond, target, repl);
            for s in then_body {
                subst_stmt(s, target, repl);
            }
            for s in else_body {
                subst_stmt(s, target, repl);
            }
        }
        MirStmt::While { cond, body } => {
            subst_expr(cond, target, repl);
            for s in body {
                subst_stmt(s, target, repl);
            }
        }
        MirStmt::Switch {
            discr,
            arms,
            default,
        } => {
            subst_expr(discr, target, repl);
            for (_, body) in arms {
                for s in body {
                    subst_stmt(s, target, repl);
                }
            }
            for s in default {
                subst_stmt(s, target, repl);
            }
        }
        MirStmt::Return(opt) => {
            if let Some(e) = opt {
                subst_expr(e, target, repl);
            }
        }
        MirStmt::Error { reason, partial } => {
            if let Some(r) = reason {
                subst_expr(r, target, repl);
            }
            if let Some(p) = partial {
                subst_expr(p, target, repl);
            }
        }
        MirStmt::SpawnActivate { args, .. } => {
            for a in args {
                subst_expr(a, target, repl);
            }
        }
        MirStmt::PrintFmt { chunks, .. } => {
            for ch in chunks {
                subst_fmt_chunk(ch, target, repl);
            }
        }
        MirStmt::FmtBuf { buf, chunks, .. } => {
            subst_expr(buf, target, repl);
            for ch in chunks {
                subst_fmt_chunk(ch, target, repl);
            }
        }
        MirStmt::FmtRing { ring, chunks, .. } => {
            subst_expr(ring, target, repl);
            for ch in chunks {
                subst_fmt_chunk(ch, target, repl);
            }
        }
        MirStmt::MmioWrite {
            base, index, value, ..
        } => {
            subst_expr(base, target, repl);
            if let Some(idx) = index {
                subst_expr(idx, target, repl);
            }
            subst_expr(value, target, repl);
        }
    }
}

fn subst_terminator(term: &mut MirTerminator, target: VarId, repl: &MirExpr) {
    match term {
        MirTerminator::Goto(_) => {}
        MirTerminator::Branch { cond, .. } => subst_expr(cond, target, repl),
        MirTerminator::Switch { discr, .. } => subst_expr(discr, target, repl),
        MirTerminator::Return(opt) => {
            if let Some(e) = opt {
                subst_expr(e, target, repl);
            }
        }
        MirTerminator::Error { reason, partial } => {
            if let Some(r) = reason {
                subst_expr(r, target, repl);
            }
            if let Some(p) = partial {
                subst_expr(p, target, repl);
            }
        }
        MirTerminator::Yield(yk) => match yk {
            MirYieldKind::EveryLatch { .. }
            | MirYieldKind::DelayPoll { .. }
            | MirYieldKind::ShareCheckpoint { .. }
            | MirYieldKind::PollChild { .. } => {}
            MirYieldKind::Wait { arms, .. } => {
                for arm in arms {
                    match &mut arm.op {
                        MirWaitArmOp::InboxRecv { target: t, .. }
                        | MirWaitArmOp::InboxSend { target: t, .. }
                        | MirWaitArmOp::SignalWait { target: t, .. }
                        | MirWaitArmOp::WaitQueueCommit { target: t, .. }
                        | MirWaitArmOp::WatchWaitChange { target: t, .. }
                        | MirWaitArmOp::PingPongWaitChunk { target: t, .. }
                        | MirWaitArmOp::PoolAlloc { target: t, .. }
                        | MirWaitArmOp::ByteRingWaitRead { target: t, .. }
                        | MirWaitArmOp::ByteRingWaitWrite { target: t, .. } => {
                            subst_prim_target(t, target, repl);
                        }
                        MirWaitArmOp::InboxViewRecv { view, .. }
                        | MirWaitArmOp::InboxViewSend { view, .. }
                        | MirWaitArmOp::ByteRingViewWaitRead { view, .. }
                        | MirWaitArmOp::ByteRingViewWaitWrite { view, .. } => {
                            subst_expr(view, target, repl);
                        }
                    }
                }
            }
            MirYieldKind::CallYielding { args, .. } => {
                for a in args {
                    subst_expr(a, target, repl);
                }
            }
        },
    }
}

fn subst_blocks(blocks: &mut [MirBlock], target: VarId, repl: &MirExpr) {
    for blk in blocks {
        for s in &mut blk.stmts {
            subst_stmt(s, target, repl);
        }
        subst_terminator(&mut blk.terminator, target, repl);
    }
}

fn collect_fn_addrs_and_calls_expr(
    expr: &MirExpr,
    addr_taken_fns: &mut BTreeSet<String>,
    call_sites: &mut BTreeMap<String, Vec<Vec<MirExpr>>>,
) {
    match &expr.kind {
        MirExprKind::FnAddr(name) => {
            addr_taken_fns.insert(name.clone());
        }
        MirExprKind::CallDirect {
            callee,
            is_extern_c,
            args,
        } => {
            if !*is_extern_c {
                call_sites
                    .entry(callee.clone())
                    .or_default()
                    .push(args.clone());
            }
            for a in args {
                collect_fn_addrs_and_calls_expr(a, addr_taken_fns, call_sites);
            }
        }
        MirExprKind::RomCallU32 { addr, args, .. } => {
            collect_fn_addrs_and_calls_expr(addr, addr_taken_fns, call_sites);
            for a in args {
                collect_fn_addrs_and_calls_expr(a, addr_taken_fns, call_sites);
            }
        }
        MirExprKind::MakePrimView { ptr, .. }
        | MirExprKind::Unary { expr: ptr, .. }
        | MirExprKind::Cast { expr: ptr, .. }
        | MirExprKind::FieldAccess { base: ptr, .. }
        | MirExprKind::SliceLen(ptr)
        | MirExprKind::FromBytes { slice: ptr, .. }
        | MirExprKind::LeaseGet { lease: ptr, .. }
        | MirExprKind::LeaseAddr { lease: ptr, .. } => {
            collect_fn_addrs_and_calls_expr(ptr, addr_taken_fns, call_sites);
        }
        MirExprKind::MmioRead { base, index, .. } => {
            collect_fn_addrs_and_calls_expr(base, addr_taken_fns, call_sites);
            if let Some(i) = index {
                collect_fn_addrs_and_calls_expr(i, addr_taken_fns, call_sites);
            }
        }
        MirExprKind::AddrOfPlace(p) | MirExprKind::ArrayToSlice { array_place: p, .. } => {
            collect_fn_addrs_and_calls_place(p, addr_taken_fns, call_sites);
        }
        MirExprKind::Binary { lhs, rhs, .. }
        | MirExprKind::Index {
            base: lhs,
            index: rhs,
            ..
        } => {
            collect_fn_addrs_and_calls_expr(lhs, addr_taken_fns, call_sites);
            collect_fn_addrs_and_calls_expr(rhs, addr_taken_fns, call_sites);
        }
        MirExprKind::SubSlice {
            base, start, end, ..
        } => {
            collect_fn_addrs_and_calls_expr(base, addr_taken_fns, call_sites);
            collect_fn_addrs_and_calls_expr(start, addr_taken_fns, call_sites);
            collect_fn_addrs_and_calls_expr(end, addr_taken_fns, call_sites);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, fe, _) in fields {
                collect_fn_addrs_and_calls_expr(fe, addr_taken_fns, call_sites);
            }
        }
        MirExprKind::ArrayLit { elems, .. } => {
            for e in elems {
                collect_fn_addrs_and_calls_expr(e, addr_taken_fns, call_sites);
            }
        }
        MirExprKind::PrimRead(op) => match op.as_ref() {
            MirPrimReadOp::WatchRead { target, .. }
            | MirPrimReadOp::WatchGen { target, .. }
            | MirPrimReadOp::GuardedGet { target, .. }
            | MirPrimReadOp::InboxTryRecv { target, .. }
            | MirPrimReadOp::InboxLen { target, .. }
            | MirPrimReadOp::InboxIsEmpty { target, .. }
            | MirPrimReadOp::InboxIsFull { target, .. }
            | MirPrimReadOp::SignalIsSet { target }
            | MirPrimReadOp::SignalTryWait { target }
            | MirPrimReadOp::WaitQueueWakeOne { target, .. }
            | MirPrimReadOp::PingPongDmaBuf { target, .. }
            | MirPrimReadOp::PingPongBaseAddr { target, .. }
            | MirPrimReadOp::PoolTryAlloc { target, .. }
            | MirPrimReadOp::PoolAvail { target, .. }
            | MirPrimReadOp::PoolInUse { target, .. }
            | MirPrimReadOp::PoolIsExhausted { target, .. }
            | MirPrimReadOp::PoolCap { target, .. }
            | MirPrimReadOp::DmaSlotTake { target, .. }
            | MirPrimReadOp::DmaSlotRelease { target, .. }
            | MirPrimReadOp::DmaSlotIsBusy { target, .. }
            | MirPrimReadOp::DmaSlotIsEmpty { target, .. }
            | MirPrimReadOp::DmaSlotAddr { target, .. }
            | MirPrimReadOp::ByteRingReadSlice { target, .. }
            | MirPrimReadOp::ByteRingTryPopByte { target, .. }
            | MirPrimReadOp::ByteRingLen { target, .. }
            | MirPrimReadOp::ByteRingFreeSpace { target, .. }
            | MirPrimReadOp::ByteRingIsEmpty { target, .. }
            | MirPrimReadOp::ByteRingIsFull { target, .. } => {
                if let MirPrimTarget::PtrExpr(pe) = target {
                    collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                }
            }
            MirPrimReadOp::DmaSlotPark { target, lease, .. } => {
                if let MirPrimTarget::PtrExpr(pe) = target {
                    collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                }
                collect_fn_addrs_and_calls_expr(lease, addr_taken_fns, call_sites);
            }
            MirPrimReadOp::InboxTrySend { target, value, .. } => {
                if let MirPrimTarget::PtrExpr(pe) = target {
                    collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                }
                collect_fn_addrs_and_calls_expr(value, addr_taken_fns, call_sites);
            }
            MirPrimReadOp::ByteRingPush { target, src, .. }
            | MirPrimReadOp::ByteRingPop {
                target, dst: src, ..
            } => {
                if let MirPrimTarget::PtrExpr(pe) = target {
                    collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                }
                collect_fn_addrs_and_calls_expr(src, addr_taken_fns, call_sites);
            }
            MirPrimReadOp::ByteRingPushByte { target, byte, .. } => {
                if let MirPrimTarget::PtrExpr(pe) = target {
                    collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                }
                collect_fn_addrs_and_calls_expr(byte, addr_taken_fns, call_sites);
            }
            MirPrimReadOp::ByteRingGrantWrite { target, len, .. } => {
                if let MirPrimTarget::PtrExpr(pe) = target {
                    collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                }
                collect_fn_addrs_and_calls_expr(len, addr_taken_fns, call_sites);
            }
            MirPrimReadOp::InboxViewTryRecv { view, .. }
            | MirPrimReadOp::InboxViewLen { view, .. }
            | MirPrimReadOp::InboxViewCap { view, .. }
            | MirPrimReadOp::InboxViewIsEmpty { view, .. }
            | MirPrimReadOp::InboxViewIsFull { view, .. }
            | MirPrimReadOp::ByteRingViewReadSlice { view }
            | MirPrimReadOp::ByteRingViewTryPopByte { view }
            | MirPrimReadOp::ByteRingViewLen { view }
            | MirPrimReadOp::ByteRingViewCap { view }
            | MirPrimReadOp::ByteRingViewFreeSpace { view }
            | MirPrimReadOp::ByteRingViewIsEmpty { view }
            | MirPrimReadOp::ByteRingViewIsFull { view } => {
                collect_fn_addrs_and_calls_expr(view, addr_taken_fns, call_sites);
            }
            MirPrimReadOp::InboxViewTrySend { view, value, .. } => {
                collect_fn_addrs_and_calls_expr(view, addr_taken_fns, call_sites);
                collect_fn_addrs_and_calls_expr(value, addr_taken_fns, call_sites);
            }
            MirPrimReadOp::ByteRingViewPush { view, src, .. }
            | MirPrimReadOp::ByteRingViewPop { view, dst: src } => {
                collect_fn_addrs_and_calls_expr(view, addr_taken_fns, call_sites);
                collect_fn_addrs_and_calls_expr(src, addr_taken_fns, call_sites);
            }
            MirPrimReadOp::ByteRingViewPushByte { view, byte, .. } => {
                collect_fn_addrs_and_calls_expr(view, addr_taken_fns, call_sites);
                collect_fn_addrs_and_calls_expr(byte, addr_taken_fns, call_sites);
            }
            MirPrimReadOp::ByteRingViewGrantWrite { view, len } => {
                collect_fn_addrs_and_calls_expr(view, addr_taken_fns, call_sites);
                collect_fn_addrs_and_calls_expr(len, addr_taken_fns, call_sites);
            }
        },
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::StringLit(_)
        | MirExprKind::TicksLit(_)
        | MirExprKind::EnumVariant { .. }
        | MirExprKind::ConstRef(_)
        | MirExprKind::GlobalPrimAddr(_)
        | MirExprKind::SubsystemAddr(_)
        | MirExprKind::MmioAddr(_)
        | MirExprKind::Var(_)
        | MirExprKind::ArrayLen(_)
        | MirExprKind::ResultIsErr(_)
        | MirExprKind::ResultErr(_)
        | MirExprKind::ResultVal(_)
        | MirExprKind::NowTicks
        | MirExprKind::SysQuery(_) => {}
    }
}

fn collect_fn_addrs_and_calls_place(
    place: &MirPlace,
    addr_taken_fns: &mut BTreeSet<String>,
    call_sites: &mut BTreeMap<String, Vec<Vec<MirExpr>>>,
) {
    match place {
        MirPlace::Var(_) => {}
        MirPlace::Deref(ptr_expr, _) => {
            collect_fn_addrs_and_calls_expr(ptr_expr, addr_taken_fns, call_sites);
        }
        MirPlace::Field { base, .. } => {
            collect_fn_addrs_and_calls_place(base, addr_taken_fns, call_sites);
        }
        MirPlace::Index { base, index, .. } => {
            collect_fn_addrs_and_calls_place(base, addr_taken_fns, call_sites);
            collect_fn_addrs_and_calls_expr(index, addr_taken_fns, call_sites);
        }
    }
}

fn collect_fn_addrs_and_calls_fmt(
    ch: &MirFmtChunk,
    addr_taken_fns: &mut BTreeSet<String>,
    call_sites: &mut BTreeMap<String, Vec<Vec<MirExpr>>>,
) {
    match ch {
        MirFmtChunk::Literal(_) => {}
        MirFmtChunk::Int { value, .. }
        | MirFmtChunk::Float { value, .. }
        | MirFmtChunk::Bool(value)
        | MirFmtChunk::Str(value)
        | MirFmtChunk::Bytes(value)
        | MirFmtChunk::EnumName { value, .. }
        | MirFmtChunk::DebugSlice { slice: value, .. } => {
            collect_fn_addrs_and_calls_expr(value, addr_taken_fns, call_sites);
        }
    }
}

fn collect_fn_addrs_and_calls_stmt(
    stmt: &MirStmt,
    addr_taken_fns: &mut BTreeSet<String>,
    call_sites: &mut BTreeMap<String, Vec<Vec<MirExpr>>>,
) {
    match stmt {
        MirStmt::Assign { target, value, .. } => {
            collect_fn_addrs_and_calls_place(target, addr_taken_fns, call_sites);
            collect_fn_addrs_and_calls_expr(value, addr_taken_fns, call_sites);
        }
        MirStmt::Expr(e) => collect_fn_addrs_and_calls_expr(e, addr_taken_fns, call_sites),
        MirStmt::PrimOp(op) => match op {
            MirPrimNonBlockingOp::WatchPublish { target, value, .. }
            | MirPrimNonBlockingOp::GuardedSet { target, value, .. }
            | MirPrimNonBlockingOp::InboxPost { target, value, .. } => {
                if let MirPrimTarget::PtrExpr(pe) = target {
                    collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                }
                collect_fn_addrs_and_calls_expr(value, addr_taken_fns, call_sites);
            }
            MirPrimNonBlockingOp::SignalNotify { target, .. }
            | MirPrimNonBlockingOp::SignalClear { target }
            | MirPrimNonBlockingOp::WaitQueuePrepare { target }
            | MirPrimNonBlockingOp::WaitQueueCancel { target }
            | MirPrimNonBlockingOp::WaitQueueWakeAll { target, .. }
            | MirPrimNonBlockingOp::PingPongFlip { target, .. }
            | MirPrimNonBlockingOp::PingPongSetActive { target, .. }
            | MirPrimNonBlockingOp::ByteRingClear { target, .. } => {
                if let MirPrimTarget::PtrExpr(pe) = target {
                    collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                }
            }
            MirPrimNonBlockingOp::LeaseRelease { value, .. } => {
                collect_fn_addrs_and_calls_expr(value, addr_taken_fns, call_sites);
            }
            MirPrimNonBlockingOp::LeaseNullify { target, .. } => {
                collect_fn_addrs_and_calls_place(target, addr_taken_fns, call_sites);
            }
            MirPrimNonBlockingOp::ByteRingConsume { target, len, .. }
            | MirPrimNonBlockingOp::ByteRingCommitWrite { target, len, .. }
            | MirPrimNonBlockingOp::PingPongCompleteHalf {
                target, half: len, ..
            } => {
                if let MirPrimTarget::PtrExpr(pe) = target {
                    collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                }
                collect_fn_addrs_and_calls_expr(len, addr_taken_fns, call_sites);
            }
            MirPrimNonBlockingOp::InboxViewPost { view, value, .. } => {
                collect_fn_addrs_and_calls_expr(view, addr_taken_fns, call_sites);
                collect_fn_addrs_and_calls_expr(value, addr_taken_fns, call_sites);
            }
            MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
            | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                collect_fn_addrs_and_calls_expr(view, addr_taken_fns, call_sites);
                collect_fn_addrs_and_calls_expr(len, addr_taken_fns, call_sites);
            }
            MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                collect_fn_addrs_and_calls_expr(view, addr_taken_fns, call_sites);
            }
        },
        MirStmt::Intrinsic(intr) => match intr {
            MirIntrinsicOp::UseSystick { hz } => {
                collect_fn_addrs_and_calls_expr(hz, addr_taken_fns, call_sites);
            }
            MirIntrinsicOp::SetPriority { .. } => {}
            MirIntrinsicOp::SysOp { args, .. } => {
                for a in args {
                    collect_fn_addrs_and_calls_expr(a, addr_taken_fns, call_sites);
                }
            }
        },
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            collect_fn_addrs_and_calls_expr(cond, addr_taken_fns, call_sites);
            for s in then_body {
                collect_fn_addrs_and_calls_stmt(s, addr_taken_fns, call_sites);
            }
            for s in else_body {
                collect_fn_addrs_and_calls_stmt(s, addr_taken_fns, call_sites);
            }
        }
        MirStmt::While { cond, body } => {
            collect_fn_addrs_and_calls_expr(cond, addr_taken_fns, call_sites);
            for s in body {
                collect_fn_addrs_and_calls_stmt(s, addr_taken_fns, call_sites);
            }
        }
        MirStmt::Switch {
            discr,
            arms,
            default,
        } => {
            collect_fn_addrs_and_calls_expr(discr, addr_taken_fns, call_sites);
            for (_, body) in arms {
                for s in body {
                    collect_fn_addrs_and_calls_stmt(s, addr_taken_fns, call_sites);
                }
            }
            for s in default {
                collect_fn_addrs_and_calls_stmt(s, addr_taken_fns, call_sites);
            }
        }
        MirStmt::Return(opt) => {
            if let Some(e) = opt {
                collect_fn_addrs_and_calls_expr(e, addr_taken_fns, call_sites);
            }
        }
        MirStmt::Error { reason, partial } => {
            if let Some(r) = reason {
                collect_fn_addrs_and_calls_expr(r, addr_taken_fns, call_sites);
            }
            if let Some(p) = partial {
                collect_fn_addrs_and_calls_expr(p, addr_taken_fns, call_sites);
            }
        }
        MirStmt::SpawnActivate { args, .. } => {
            for a in args {
                collect_fn_addrs_and_calls_expr(a, addr_taken_fns, call_sites);
            }
        }
        MirStmt::PrintFmt { chunks, .. } => {
            for ch in chunks {
                collect_fn_addrs_and_calls_fmt(ch, addr_taken_fns, call_sites);
            }
        }
        MirStmt::FmtBuf { buf, chunks, .. } => {
            collect_fn_addrs_and_calls_expr(buf, addr_taken_fns, call_sites);
            for ch in chunks {
                collect_fn_addrs_and_calls_fmt(ch, addr_taken_fns, call_sites);
            }
        }
        MirStmt::FmtRing { ring, chunks, .. } => {
            collect_fn_addrs_and_calls_expr(ring, addr_taken_fns, call_sites);
            for ch in chunks {
                collect_fn_addrs_and_calls_fmt(ch, addr_taken_fns, call_sites);
            }
        }
        MirStmt::MmioWrite {
            base, index, value, ..
        } => {
            collect_fn_addrs_and_calls_expr(base, addr_taken_fns, call_sites);
            if let Some(idx) = index {
                collect_fn_addrs_and_calls_expr(idx, addr_taken_fns, call_sites);
            }
            collect_fn_addrs_and_calls_expr(value, addr_taken_fns, call_sites);
        }
        MirStmt::EnterCritical
        | MirStmt::ExitCritical
        | MirStmt::EndInboxTurn
        | MirStmt::Break
        | MirStmt::Continue
        | MirStmt::SetIrqSelector { .. }
        | MirStmt::BoundedForLatch
        | MirStmt::ZeroTaskLocals => {}
    }
}

fn collect_fn_addrs_and_calls_term(
    term: &MirTerminator,
    addr_taken_fns: &mut BTreeSet<String>,
    call_sites: &mut BTreeMap<String, Vec<Vec<MirExpr>>>,
) {
    match term {
        MirTerminator::Goto(_) => {}
        MirTerminator::Branch { cond, .. } => {
            collect_fn_addrs_and_calls_expr(cond, addr_taken_fns, call_sites);
        }
        MirTerminator::Switch { discr, .. } => {
            collect_fn_addrs_and_calls_expr(discr, addr_taken_fns, call_sites);
        }
        MirTerminator::Return(opt) => {
            if let Some(e) = opt {
                collect_fn_addrs_and_calls_expr(e, addr_taken_fns, call_sites);
            }
        }
        MirTerminator::Error { reason, partial } => {
            if let Some(r) = reason {
                collect_fn_addrs_and_calls_expr(r, addr_taken_fns, call_sites);
            }
            if let Some(p) = partial {
                collect_fn_addrs_and_calls_expr(p, addr_taken_fns, call_sites);
            }
        }
        MirTerminator::Yield(yk) => match yk {
            MirYieldKind::EveryLatch { .. }
            | MirYieldKind::DelayPoll { .. }
            | MirYieldKind::ShareCheckpoint { .. }
            | MirYieldKind::PollChild { .. } => {}
            MirYieldKind::Wait { arms, .. } => {
                for arm in arms {
                    match &arm.op {
                        MirWaitArmOp::InboxRecv { target, .. }
                        | MirWaitArmOp::InboxSend { target, .. }
                        | MirWaitArmOp::SignalWait { target, .. }
                        | MirWaitArmOp::WaitQueueCommit { target, .. }
                        | MirWaitArmOp::WatchWaitChange { target, .. }
                        | MirWaitArmOp::PingPongWaitChunk { target, .. }
                        | MirWaitArmOp::PoolAlloc { target, .. }
                        | MirWaitArmOp::ByteRingWaitRead { target, .. }
                        | MirWaitArmOp::ByteRingWaitWrite { target, .. } => {
                            if let MirPrimTarget::PtrExpr(pe) = target {
                                collect_fn_addrs_and_calls_expr(pe, addr_taken_fns, call_sites);
                            }
                        }
                        MirWaitArmOp::InboxViewRecv { view, .. }
                        | MirWaitArmOp::InboxViewSend { view, .. }
                        | MirWaitArmOp::ByteRingViewWaitRead { view, .. }
                        | MirWaitArmOp::ByteRingViewWaitWrite { view, .. } => {
                            collect_fn_addrs_and_calls_expr(view, addr_taken_fns, call_sites);
                        }
                    }
                }
            }
            MirYieldKind::CallYielding { callee, args, .. } => {
                call_sites
                    .entry(callee.clone())
                    .or_default()
                    .push(args.clone());
                for a in args {
                    collect_fn_addrs_and_calls_expr(a, addr_taken_fns, call_sites);
                }
            }
        },
    }
}

fn collect_all_fn_addrs_and_calls(
    module: &MirModule,
) -> (BTreeSet<String>, BTreeMap<String, Vec<Vec<MirExpr>>>) {
    let mut addr_taken_fns = BTreeSet::new();
    let mut call_sites: BTreeMap<String, Vec<Vec<MirExpr>>> = BTreeMap::new();

    for c in &module.consts {
        collect_fn_addrs_and_calls_expr(&c.init, &mut addr_taken_fns, &mut call_sites);
    }
    for p in &module.primitives {
        if let Some(init) = &p.init {
            collect_fn_addrs_and_calls_expr(init, &mut addr_taken_fns, &mut call_sites);
        }
    }
    for step in &module.boot.steps {
        match step {
            MirBootStep::Stmt(s) => {
                collect_fn_addrs_and_calls_stmt(s, &mut addr_taken_fns, &mut call_sites);
            }
            MirBootStep::RunYieldingFn { callee, args, .. } => {
                call_sites
                    .entry(callee.clone())
                    .or_default()
                    .push(args.clone());
                for a in args {
                    collect_fn_addrs_and_calls_expr(a, &mut addr_taken_fns, &mut call_sites);
                }
            }
            MirBootStep::SpawnActivate { args, .. } => {
                for a in args {
                    collect_fn_addrs_and_calls_expr(a, &mut addr_taken_fns, &mut call_sites);
                }
            }
        }
    }
    for f in &module.functions {
        match &f.body {
            MirFnBody::Direct(stmts) => {
                for s in stmts {
                    collect_fn_addrs_and_calls_stmt(s, &mut addr_taken_fns, &mut call_sites);
                }
            }
            MirFnBody::Coroutine { blocks, .. } => {
                for blk in blocks {
                    for s in &blk.stmts {
                        collect_fn_addrs_and_calls_stmt(s, &mut addr_taken_fns, &mut call_sites);
                    }
                    collect_fn_addrs_and_calls_term(
                        &blk.terminator,
                        &mut addr_taken_fns,
                        &mut call_sites,
                    );
                }
            }
        }
    }
    for bp in &module.task_blueprints {
        for blk in &bp.blocks {
            for s in &blk.stmts {
                collect_fn_addrs_and_calls_stmt(s, &mut addr_taken_fns, &mut call_sites);
            }
            collect_fn_addrs_and_calls_term(&blk.terminator, &mut addr_taken_fns, &mut call_sites);
        }
    }
    for isr in &module.isrs {
        for sv in &isr.state_vars {
            collect_fn_addrs_and_calls_expr(&sv.init, &mut addr_taken_fns, &mut call_sites);
        }
        for s in &isr.body {
            collect_fn_addrs_and_calls_stmt(s, &mut addr_taken_fns, &mut call_sites);
        }
    }
    for vec in &module.irq_vectors {
        for b in &vec.bindings {
            for barg in &b.baked_args {
                if let MirBakedArg::Const(e) = barg {
                    collect_fn_addrs_and_calls_expr(e, &mut addr_taken_fns, &mut call_sites);
                }
            }
            for sv in &b.state_vars {
                collect_fn_addrs_and_calls_expr(&sv.init, &mut addr_taken_fns, &mut call_sites);
            }
        }
    }

    for sp in &module.spawned_tasks {
        for a in &sp.init_args {
            collect_fn_addrs_and_calls_expr(a, &mut addr_taken_fns, &mut call_sites);
        }
    }

    (addr_taken_fns, call_sites)
}

#[derive(Clone)]
enum SpawnActivateLoc {
    Boot,
    Fn(String),
}

struct SpawnActivateSite {
    loc: SpawnActivateLoc,
    in_loop: bool,
    args: Vec<MirExpr>,
}

fn collect_spawn_activates_stmt(
    stmt: &MirStmt,
    loc: &SpawnActivateLoc,
    in_loop: bool,
    out: &mut BTreeMap<usize, Vec<SpawnActivateSite>>,
) {
    match stmt {
        MirStmt::SpawnActivate { slot_id, args } => {
            out.entry(*slot_id).or_default().push(SpawnActivateSite {
                loc: loc.clone(),
                in_loop,
                args: args.clone(),
            });
        }
        MirStmt::If {
            then_body,
            else_body,
            ..
        } => {
            for s in then_body {
                collect_spawn_activates_stmt(s, loc, in_loop, out);
            }
            for s in else_body {
                collect_spawn_activates_stmt(s, loc, in_loop, out);
            }
        }
        MirStmt::While { body, .. } => {
            for s in body {
                collect_spawn_activates_stmt(s, loc, true, out);
            }
        }
        MirStmt::Switch { arms, default, .. } => {
            for (_, body) in arms {
                for s in body {
                    collect_spawn_activates_stmt(s, loc, in_loop, out);
                }
            }
            for s in default {
                collect_spawn_activates_stmt(s, loc, in_loop, out);
            }
        }
        _ => {}
    }
}

fn block_in_cfg_cycle(blocks: &[MirBlock], target_idx: usize) -> bool {
    let target_id = blocks[target_idx].id;
    let block_index: BTreeMap<_, usize> =
        blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();
    for succ in blocks[target_idx].terminator.successors() {
        queue.push_back(succ);
    }
    while let Some(bid) = queue.pop_front() {
        if bid == target_id {
            return true;
        }
        if visited.insert(bid) {
            if let Some(&idx) = block_index.get(&bid) {
                for succ in blocks[idx].terminator.successors() {
                    queue.push_back(succ);
                }
            }
        }
    }
    false
}

fn collect_all_spawn_activates(module: &MirModule) -> BTreeMap<usize, Vec<SpawnActivateSite>> {
    let mut out: BTreeMap<usize, Vec<SpawnActivateSite>> = BTreeMap::new();
    for step in &module.boot.steps {
        match step {
            MirBootStep::SpawnActivate { slot_id, args } => {
                out.entry(*slot_id).or_default().push(SpawnActivateSite {
                    loc: SpawnActivateLoc::Boot,
                    in_loop: false,
                    args: args.clone(),
                });
            }
            MirBootStep::Stmt(s) => {
                collect_spawn_activates_stmt(s, &SpawnActivateLoc::Boot, false, &mut out);
            }
            MirBootStep::RunYieldingFn { .. } => {}
        }
    }
    for f in &module.functions {
        let loc = SpawnActivateLoc::Fn(f.name.clone());
        match &f.body {
            MirFnBody::Direct(stmts) => {
                for s in stmts {
                    collect_spawn_activates_stmt(s, &loc, false, &mut out);
                }
            }
            MirFnBody::Coroutine { blocks, .. } => {
                for (bidx, blk) in blocks.iter().enumerate() {
                    let blk_in_cycle = block_in_cfg_cycle(blocks, bidx);
                    for s in &blk.stmts {
                        collect_spawn_activates_stmt(s, &loc, blk_in_cycle, &mut out);
                    }
                }
            }
        }
    }
    out
}

fn is_single_boot_called_fn(
    fn_name: &str,
    module: &MirModule,
    addr_taken_fns: &BTreeSet<String>,
    call_sites: &BTreeMap<String, Vec<Vec<MirExpr>>>,
) -> bool {
    if addr_taken_fns.contains(fn_name) {
        return false;
    }
    if call_sites.get(fn_name).map_or(0, Vec::len) != 1 {
        return false;
    }
    fn stmt_has_direct_call_outside_loop(
        stmt: &MirStmt,
        target: &str,
        in_loop: bool,
    ) -> Option<bool> {
        match stmt {
            MirStmt::Expr(MirExpr {
                kind:
                    MirExprKind::CallDirect {
                        callee,
                        is_extern_c: false,
                        ..
                    },
                ..
            }) if callee == target => Some(!in_loop),
            MirStmt::If {
                then_body,
                else_body,
                ..
            } => {
                for s in then_body.iter().chain(else_body.iter()) {
                    if let Some(ok) = stmt_has_direct_call_outside_loop(s, target, in_loop) {
                        return Some(ok);
                    }
                }
                None
            }
            MirStmt::While { body, .. } => {
                for s in body {
                    if let Some(ok) = stmt_has_direct_call_outside_loop(s, target, true) {
                        return Some(ok);
                    }
                }
                None
            }
            MirStmt::Switch { arms, default, .. } => {
                for s in arms
                    .iter()
                    .flat_map(|(_, b)| b.iter())
                    .chain(default.iter())
                {
                    if let Some(ok) = stmt_has_direct_call_outside_loop(s, target, in_loop) {
                        return Some(ok);
                    }
                }
                None
            }
            _ => None,
        }
    }
    for step in &module.boot.steps {
        match step {
            MirBootStep::RunYieldingFn { callee, .. } if callee == fn_name => return true,
            MirBootStep::Stmt(s) => {
                if let Some(ok) = stmt_has_direct_call_outside_loop(s, fn_name, false) {
                    return ok;
                }
            }
            _ => {}
        }
    }
    false
}

fn var_by_id(vars: &[MirVar], vid: VarId) -> Option<&MirVar> {
    vars.get(vid.0)
        .filter(|v| v.id == vid)
        .or_else(|| vars.iter().find(|v| v.id == vid))
}

fn var_by_id_mut(vars: &mut [MirVar], vid: VarId) -> Option<&mut MirVar> {
    if vars.get(vid.0).is_some_and(|v| v.id == vid) {
        return vars.get_mut(vid.0);
    }
    vars.iter_mut().find(|v| v.id == vid)
}

fn propagate_boot_consts(
    module: &mut MirModule,
    subsys_counts: &BTreeMap<String, usize>,
    subsys_structs: &BTreeSet<String>,
) -> bool {
    let mut changed = false;
    loop {
        let mut usage = VarUsageInfo::default();
        for step in &module.boot.steps {
            match step {
                MirBootStep::Stmt(s) => usage.visit_stmt(s),
                MirBootStep::RunYieldingFn { args, dst, .. } => {
                    for a in args {
                        usage.visit_expr(a);
                    }
                    if let Some(d) = dst {
                        match d {
                            MirCallDst::Value(v) => usage.record_write(*v),
                            MirCallDst::Result(r) => {
                                usage.record_write(*r);
                                usage.result_vars.insert(*r);
                            }
                        }
                    }
                }
                MirBootStep::SpawnActivate { args, .. } => {
                    for a in args {
                        usage.visit_expr(a);
                    }
                }
            }
        }
        for sp in &module.spawned_tasks {
            for a in &sp.init_args {
                usage.visit_expr(a);
            }
        }

        let mut candidate: Option<(usize, VarId, MirExpr)> = None;
        for (idx, step) in module.boot.steps.iter().enumerate() {
            let MirBootStep::Stmt(stmt) = step else {
                continue;
            };
            if let MirStmt::Assign {
                target: MirPlace::Var(vid),
                op: MirAssignOp::Assign,
                value,
            } = stmt.as_ref()
            {
                let Some(v) = var_by_id(&module.boot.vars, *vid) else {
                    continue;
                };
                if v.role == VarRole::Param
                    || v.role == VarRole::IsrState
                    || v.addr_taken
                    || usage.addr_taken_vars.contains(vid)
                    || usage.result_vars.contains(vid)
                    || usage.write_counts.get(vid).copied().unwrap_or(0) != 1
                    || (matches!(v.ty, MirType::FnPtr { .. })
                        && usage.indirect_fnptr_callees.contains(&v.name))
                {
                    continue;
                }
                if expr_reads_var(value, *vid)
                    || !is_pure_static_immediate(value, module, subsys_counts, subsys_structs)
                {
                    continue;
                }
                let read_before = module.boot.steps[..idx]
                    .iter()
                    .any(|s| boot_step_reads_var(s, *vid));
                if read_before {
                    continue;
                }
                let Some(repl) = coerce_immediate_to_var_ty(value, &v.ty) else {
                    continue;
                };
                candidate = Some((idx, *vid, repl));
                break;
            }
        }

        let Some((assign_idx, vid, repl)) = candidate else {
            break;
        };

        let mut post_slots = Vec::new();
        for step in &mut module.boot.steps[assign_idx + 1..] {
            match step {
                MirBootStep::Stmt(s) => {
                    collect_spawn_slots_stmt(s, &mut post_slots);
                    subst_stmt(s, vid, &repl);
                }
                MirBootStep::RunYieldingFn { args, .. } => {
                    for a in args {
                        subst_expr(a, vid, &repl);
                    }
                }
                MirBootStep::SpawnActivate { slot_id, args } => {
                    post_slots.push(*slot_id);
                    for a in args {
                        subst_expr(a, vid, &repl);
                    }
                }
            }
        }
        for sp in &mut module.spawned_tasks {
            if post_slots.contains(&sp.slot_id) {
                for a in &mut sp.init_args {
                    subst_expr(a, vid, &repl);
                }
            }
        }
        let still_read = module
            .boot
            .steps
            .iter()
            .enumerate()
            .any(|(i, s)| i != assign_idx && boot_step_reads_var(s, vid))
            || module
                .spawned_tasks
                .iter()
                .any(|sp| sp.init_args.iter().any(|a| expr_reads_var(a, vid)));
        if !still_read {
            module.boot.steps.remove(assign_idx);
        }
        changed = true;
    }
    changed
}

fn collect_spawn_slots_stmt(stmt: &MirStmt, out: &mut Vec<usize>) {
    match stmt {
        MirStmt::SpawnActivate { slot_id, .. } => out.push(*slot_id),
        MirStmt::If {
            then_body,
            else_body,
            ..
        } => {
            for s in then_body.iter().chain(else_body.iter()) {
                collect_spawn_slots_stmt(s, out);
            }
        }
        MirStmt::While { body, .. } => {
            for s in body {
                collect_spawn_slots_stmt(s, out);
            }
        }
        MirStmt::Switch { arms, default, .. } => {
            for s in arms
                .iter()
                .flat_map(|(_, b)| b.iter())
                .chain(default.iter())
            {
                collect_spawn_slots_stmt(s, out);
            }
        }
        _ => {}
    }
}

fn is_static_ref_or_view_ty(ty: &MirType) -> bool {
    matches!(
        ty,
        MirType::Ptr(_) | MirType::MmioPtr(_) | MirType::PrimView(_) | MirType::FnPtr { .. }
    )
}

fn propagate_local_consts_in_stmts(
    stmts: &mut Vec<MirStmt>,
    vars: &mut [MirVar],
    usage: &VarUsageInfo,
    module_consts_ctx: (&MirModule, &BTreeMap<String, usize>, &BTreeSet<String>),
) -> Option<(VarId, String, MirExpr)> {
    let (module, subsys_counts, subsys_structs) = module_consts_ctx;
    for (idx, stmt) in stmts.iter().enumerate() {
        let MirStmt::Assign {
            target: MirPlace::Var(vid),
            op: MirAssignOp::Assign,
            value,
        } = stmt
        else {
            continue;
        };
        let Some(v) = var_by_id(vars, *vid) else {
            continue;
        };
        if v.role == VarRole::Param
            || v.role == VarRole::IsrState
            || !is_static_ref_or_view_ty(&v.ty)
            || v.addr_taken
            || usage.addr_taken_vars.contains(vid)
            || usage.yield_slot_vars.contains(vid)
            || usage.result_vars.contains(vid)
            || usage.write_counts.get(vid).copied().unwrap_or(0) != 1
            || (matches!(v.ty, MirType::FnPtr { .. })
                && usage.indirect_fnptr_callees.contains(&v.name))
        {
            continue;
        }
        if expr_reads_var(value, *vid)
            || !is_pure_static_immediate(value, module, subsys_counts, subsys_structs)
        {
            continue;
        }
        let read_before = stmts[..idx].iter().any(|s| stmt_reads_var(s, *vid));
        if read_before {
            continue;
        }
        let Some(repl) = coerce_immediate_to_var_ty(value, &v.ty) else {
            continue;
        };
        let target_vid = *vid;
        let vname = v.name.clone();
        stmts.remove(idx);
        if let Some(vm) = var_by_id_mut(vars, target_vid) {
            vm.hoisted = false;
        }
        return Some((target_vid, vname, repl));
    }
    None
}

fn record_folded_var_name(map: &mut BTreeMap<String, Vec<String>>, owner: &str, vname: &str) {
    let entry = map.entry(owner.to_string()).or_default();
    if !entry.iter().any(|n| n == vname) {
        entry.push(vname.to_string());
    }
}

/// Fold whole-program singletons (`instance_count == 1`):
/// - Singleton subsystems (`module.subsystems` + nested subsystem fields)
/// - Parameters of non-`main` `MirTaskBlueprint`s spawned once (`instance_count == 1`)
/// - Uniform-constant parameters of non-address-taken `MirFunction`s
/// - Single-assignment entry-block locals in `MirTaskBlueprint`s and `MirFunction`s (fixed-point)
pub fn fold_singletons(module: &mut MirModule) {
    let before_ram = crate::ram::compute_ram_report_with_subsystems(
        &module.structs,
        &module.functions,
        &module.task_blueprints,
        &module.spawned_tasks,
        &module.primitives,
        &module.subsystems,
        &module.irq_vectors,
        &module.boot.boot_yielding_callees,
    );

    let (subsys_counts, singleton_subsystems, singleton_subsystem_structs, subsys_structs) =
        collect_subsystem_instance_counts(module);

    let mut folded_task_vars: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut folded_task_var_ids: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    let mut folded_fn_vars: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut folded_fn_param_ids: BTreeSet<(String, usize)> = BTreeSet::new();

    for _iter in 0..64 {
        let mut changed = false;

        if propagate_boot_consts(module, &subsys_counts, &subsys_structs) {
            changed = true;
        }

        let (addr_taken_fns, call_sites) = collect_all_fn_addrs_and_calls(module);

        // 1. Fold single-spawned non-main `MirTaskBlueprint` parameters
        let mut bp_spawn_map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for sp in &module.spawned_tasks {
            bp_spawn_map
                .entry(sp.blueprint_name.clone())
                .or_default()
                .push(sp.slot_id);
        }
        let spawn_sites = collect_all_spawn_activates(module);

        let bp_count = module.task_blueprints.len();
        for bp_idx in 0..bp_count {
            let bp_name = module.task_blueprints[bp_idx].name.clone();
            let is_main = module.task_blueprints[bp_idx].is_main_supervisor;
            if is_main {
                continue;
            }
            let Some(slots) = bp_spawn_map.get(&bp_name) else {
                continue;
            };
            if slots.len() != 1 {
                continue;
            }
            let slot_id = slots[0];
            let Some(sites) = spawn_sites.get(&slot_id) else {
                continue;
            };
            if sites.len() != 1 || sites[0].in_loop {
                continue;
            }
            let site = &sites[0];
            if let SpawnActivateLoc::Fn(ref fname) = site.loc {
                if !is_single_boot_called_fn(fname, module, &addr_taken_fns, &call_sites) {
                    continue;
                }
            }

            let usage = VarUsageInfo::collect_blocks(&module.task_blueprints[bp_idx].blocks);
            let params = module.task_blueprints[bp_idx].params.clone();
            for (pidx, &pvid) in params.iter().enumerate() {
                if folded_task_var_ids
                    .get(&bp_name)
                    .is_some_and(|s| s.contains(&pvid.0))
                {
                    continue;
                }
                let Some(arg_expr) = site.args.get(pidx) else {
                    continue;
                };
                if usage.write_counts.contains_key(&pvid)
                    || usage.addr_taken_vars.contains(&pvid)
                    || usage.yield_slot_vars.contains(&pvid)
                    || usage.result_vars.contains(&pvid)
                {
                    continue;
                }
                let Some(pvar) = var_by_id(&module.task_blueprints[bp_idx].vars, pvid) else {
                    continue;
                };
                if pvar.addr_taken
                    || (matches!(pvar.ty, MirType::FnPtr { .. })
                        && usage.indirect_fnptr_callees.contains(&pvar.name))
                    || !is_pure_static_immediate(arg_expr, module, &subsys_counts, &subsys_structs)
                {
                    continue;
                }
                let Some(repl) = coerce_immediate_to_var_ty(arg_expr, &pvar.ty) else {
                    continue;
                };
                let pname = pvar.name.clone();

                let bp = &mut module.task_blueprints[bp_idx];
                subst_blocks(&mut bp.blocks, pvid, &repl);
                if let Some(pv) = var_by_id_mut(&mut bp.vars, pvid) {
                    pv.hoisted = false;
                }
                record_folded_var_name(&mut folded_task_vars, &bp_name, &pname);
                folded_task_var_ids
                    .entry(bp_name.clone())
                    .or_default()
                    .insert(pvid.0);
                changed = true;
            }
        }

        // 2. Fold uniform-constant parameters of non-address-taken `MirFunction`s
        let fn_count = module.functions.len();
        for fn_idx in 0..fn_count {
            let fn_name = module.functions[fn_idx].name.clone();
            if addr_taken_fns.contains(&fn_name) {
                continue;
            }
            let Some(calls) = call_sites.get(&fn_name) else {
                continue;
            };
            if calls.is_empty() {
                continue;
            }
            let usage = VarUsageInfo::collect_fn(&module.functions[fn_idx]);
            let params = module.functions[fn_idx].params.clone();
            for (pidx, &pvid) in params.iter().enumerate() {
                if folded_fn_param_ids.contains(&(fn_name.clone(), pvid.0)) {
                    continue;
                }
                if usage.write_counts.contains_key(&pvid)
                    || usage.addr_taken_vars.contains(&pvid)
                    || usage.yield_slot_vars.contains(&pvid)
                    || usage.result_vars.contains(&pvid)
                {
                    continue;
                }
                if !calls.iter().all(|a| a.get(pidx).is_some()) {
                    continue;
                }
                let first = &calls[0][pidx];
                if !is_pure_static_immediate(first, module, &subsys_counts, &subsys_structs) {
                    continue;
                }
                if !calls[1..].iter().all(|a| mir_expr_eq(first, &a[pidx])) {
                    continue;
                }
                let Some(pvar) = var_by_id(&module.functions[fn_idx].vars, pvid) else {
                    continue;
                };
                if pvar.addr_taken
                    || (matches!(pvar.ty, MirType::FnPtr { .. })
                        && usage.indirect_fnptr_callees.contains(&pvar.name))
                {
                    continue;
                }
                let Some(repl) = coerce_immediate_to_var_ty(first, &pvar.ty) else {
                    continue;
                };
                let pname = pvar.name.clone();

                let func = &mut module.functions[fn_idx];
                match &mut func.body {
                    MirFnBody::Direct(stmts) => {
                        for s in stmts {
                            subst_stmt(s, pvid, &repl);
                        }
                    }
                    MirFnBody::Coroutine { blocks, .. } => {
                        subst_blocks(blocks, pvid, &repl);
                        if let Some(pv) = var_by_id_mut(&mut func.vars, pvid) {
                            pv.hoisted = false;
                        }
                    }
                }
                record_folded_var_name(&mut folded_fn_vars, &fn_name, &pname);
                folded_fn_param_ids.insert((fn_name.clone(), pvid.0));
                changed = true;
            }
        }

        // 3. Single-assignment entry-block local constant propagation in tasks & functions
        for bp_idx in 0..bp_count {
            loop {
                if module.task_blueprints[bp_idx].blocks.is_empty() {
                    break;
                }
                let usage = VarUsageInfo::collect_blocks(&module.task_blueprints[bp_idx].blocks);
                let res = {
                    let module_ref: &MirModule = module;
                    let bp_blocks = &module_ref.task_blueprints[bp_idx].blocks;
                    let bp_vars = &module_ref.task_blueprints[bp_idx].vars;
                    let mut stmts_copy = bp_blocks[0].stmts.clone();
                    let mut vars_copy = bp_vars.clone();
                    let found = propagate_local_consts_in_stmts(
                        &mut stmts_copy,
                        &mut vars_copy,
                        &usage,
                        (module_ref, &subsys_counts, &subsys_structs),
                    );
                    found.map(|f| (f, stmts_copy, vars_copy))
                };
                let Some(((vid, vname, repl), new_entry_stmts, new_vars)) = res else {
                    break;
                };
                let bp = &mut module.task_blueprints[bp_idx];
                let bp_name = bp.name.clone();
                bp.blocks[0].stmts = new_entry_stmts;
                bp.vars = new_vars;
                subst_blocks(&mut bp.blocks, vid, &repl);
                record_folded_var_name(&mut folded_task_vars, &bp_name, &vname);
                folded_task_var_ids
                    .entry(bp_name)
                    .or_default()
                    .insert(vid.0);
                changed = true;
            }
        }

        for fn_idx in 0..fn_count {
            loop {
                let usage = VarUsageInfo::collect_fn(&module.functions[fn_idx]);
                let res = {
                    let module_ref: &MirModule = module;
                    let func = &module_ref.functions[fn_idx];
                    let mut vars_copy = func.vars.clone();
                    match &func.body {
                        MirFnBody::Direct(stmts) => {
                            let mut stmts_copy = stmts.clone();
                            propagate_local_consts_in_stmts(
                                &mut stmts_copy,
                                &mut vars_copy,
                                &usage,
                                (module_ref, &subsys_counts, &subsys_structs),
                            )
                            .map(|f| (f, stmts_copy, vars_copy))
                        }
                        MirFnBody::Coroutine { blocks, .. } => {
                            if blocks.is_empty() {
                                None
                            } else {
                                let mut stmts_copy = blocks[0].stmts.clone();
                                propagate_local_consts_in_stmts(
                                    &mut stmts_copy,
                                    &mut vars_copy,
                                    &usage,
                                    (module_ref, &subsys_counts, &subsys_structs),
                                )
                                .map(|f| (f, stmts_copy, vars_copy))
                            }
                        }
                    }
                };
                let Some(((vid, vname, repl), new_stmts, new_vars)) = res else {
                    break;
                };
                let func = &mut module.functions[fn_idx];
                let fn_name = func.name.clone();
                func.vars = new_vars;
                match &mut func.body {
                    MirFnBody::Direct(stmts) => {
                        *stmts = new_stmts;
                        for s in stmts {
                            subst_stmt(s, vid, &repl);
                        }
                    }
                    MirFnBody::Coroutine { blocks, .. } => {
                        blocks[0].stmts = new_stmts;
                        subst_blocks(blocks, vid, &repl);
                    }
                }
                record_folded_var_name(&mut folded_fn_vars, &fn_name, &vname);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    let after_ram = crate::ram::compute_ram_report_with_subsystems(
        &module.structs,
        &module.functions,
        &module.task_blueprints,
        &module.spawned_tasks,
        &module.primitives,
        &module.subsystems,
        &module.irq_vectors,
        &module.boot.boot_yielding_callees,
    );

    let mut singleton_task_bytes_saved = BTreeMap::new();
    for (bt, at) in before_ram
        .spawned_tasks
        .iter()
        .zip(after_ram.spawned_tasks.iter())
    {
        let s32 = bt.frame_bytes_32.saturating_sub(at.frame_bytes_32);
        let s64 = bt.frame_bytes_64.saturating_sub(at.frame_bytes_64);
        if s32 > 0 || s64 > 0 {
            singleton_task_bytes_saved.insert(at.instance_name.clone(), (s32, s64));
        }
    }
    let singleton_bytes_saved_32 = (before_ram.mcu32.task_frames_bytes
        + before_ram.mcu32.boot_frames_bytes)
        .saturating_sub(after_ram.mcu32.task_frames_bytes + after_ram.mcu32.boot_frames_bytes);
    let singleton_bytes_saved_64 = (before_ram.host64.task_frames_bytes
        + before_ram.host64.boot_frames_bytes)
        .saturating_sub(after_ram.host64.task_frames_bytes + after_ram.host64.boot_frames_bytes);

    module.visibility.singleton_subsystems = singleton_subsystems;
    module.visibility.singleton_subsystem_structs = singleton_subsystem_structs;
    module.visibility.folded_task_vars = folded_task_vars;
    module.visibility.folded_task_var_ids = folded_task_var_ids;
    module.visibility.folded_fn_vars = folded_fn_vars;
    module.visibility.singleton_task_bytes_saved = singleton_task_bytes_saved;
    module.visibility.singleton_bytes_saved_32 = singleton_bytes_saved_32;
    module.visibility.singleton_bytes_saved_64 = singleton_bytes_saved_64;
}

/// Prune unreferenced imported-module OS primitives and subsystems when lowering from a real `main` entry.
fn prune_unused_imported_globals(module: &mut MirModule) {
    let (live_prims, live_subsystems) = {
        let mut ctx = ScanCtx::new(module);
        let mut dummy_callees = BTreeSet::new();
        for f in &module.functions {
            ctx.visit_function(f, false, &mut dummy_callees);
        }
        ctx.visit_non_prim_roots(module, &mut dummy_callees);
        let mut isr_queue = VecDeque::new();
        ctx.visit_irq_vectors_and_isrs(module, &mut isr_queue);

        let mut scanned_prim_inits = BTreeSet::new();
        for p in &module.primitives {
            if !p.name.contains("__") {
                ctx.acc.referenced_prims.insert(p.name.clone());
            }
        }
        loop {
            let mut progressed = false;
            let live_now = ctx.acc.referenced_prims.clone();
            for p in &module.primitives {
                if live_now.contains(&p.name) && scanned_prim_inits.insert(p.name.clone()) {
                    if let Some(init) = &p.init {
                        ctx.visit_expr(init, false, &mut dummy_callees);
                        progressed = true;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        (ctx.acc.referenced_prims, ctx.acc.referenced_subsystems)
    };
    module
        .primitives
        .retain(|p| !p.name.contains("__") || live_prims.contains(&p.name));
    module
        .subsystems
        .retain(|s| !s.name.contains("__") || live_subsystems.contains(&s.name));
}

/// Prune `MirFunction`s, `dual_specialized_fns`, unattached ISRs, unused imported globals, and
/// monomorphized `Fn` entries not reachable from boot, tasks, ISRs, IRQ vectors, consts, or
/// primitives (no-op on fragment-only test modules).
pub fn prune_unreachable_functions(module: &mut MirModule) {
    if module.boot.from_main_entry {
        let attached_isrs: BTreeSet<String> = module
            .irq_vectors
            .iter()
            .flat_map(|v| v.bindings.iter().map(|b| b.isr_name.clone()))
            .collect();
        module.isrs.retain(|isr| attached_isrs.contains(&isr.name));
    }

    if module.boot.steps.is_empty()
        && module.boot.boot_yielding_callees.is_empty()
        && module.spawned_tasks.is_empty()
        && module.task_blueprints.is_empty()
        && module.isrs.is_empty()
        && module.irq_vectors.is_empty()
    {
        if module.boot.from_main_entry {
            prune_unused_imported_globals(module);
        }
        return;
    }

    let mut addr_taken_fns = BTreeSet::new();
    let mut call_sites: BTreeMap<String, Vec<Vec<MirExpr>>> = BTreeMap::new();
    let mut reachable = BTreeSet::new();
    let mut worklist = VecDeque::new();

    let mut seed = |name: String| {
        if reachable.insert(name.clone()) {
            worklist.push_back(name);
        }
    };

    for c in &module.consts {
        collect_fn_addrs_and_calls_expr(&c.init, &mut addr_taken_fns, &mut call_sites);
    }
    for p in &module.primitives {
        if let Some(init) = &p.init {
            collect_fn_addrs_and_calls_expr(init, &mut addr_taken_fns, &mut call_sites);
        }
    }
    for callee in &module.boot.boot_yielding_callees {
        seed(callee.clone());
    }
    for step in &module.boot.steps {
        match step {
            MirBootStep::Stmt(s) => {
                collect_fn_addrs_and_calls_stmt(s, &mut addr_taken_fns, &mut call_sites);
            }
            MirBootStep::RunYieldingFn { callee, args, .. } => {
                seed(callee.clone());
                for a in args {
                    collect_fn_addrs_and_calls_expr(a, &mut addr_taken_fns, &mut call_sites);
                }
            }
            MirBootStep::SpawnActivate { args, .. } => {
                for a in args {
                    collect_fn_addrs_and_calls_expr(a, &mut addr_taken_fns, &mut call_sites);
                }
            }
        }
    }
    for bp in &module.task_blueprints {
        for slot in &bp.child_slots {
            seed(slot.callee.clone());
        }
        for blk in &bp.blocks {
            for s in &blk.stmts {
                collect_fn_addrs_and_calls_stmt(s, &mut addr_taken_fns, &mut call_sites);
            }
            collect_fn_addrs_and_calls_term(&blk.terminator, &mut addr_taken_fns, &mut call_sites);
        }
    }
    for sp in &module.spawned_tasks {
        for a in &sp.init_args {
            collect_fn_addrs_and_calls_expr(a, &mut addr_taken_fns, &mut call_sites);
        }
    }
    for isr in &module.isrs {
        for sv in &isr.state_vars {
            collect_fn_addrs_and_calls_expr(&sv.init, &mut addr_taken_fns, &mut call_sites);
        }
        for s in &isr.body {
            collect_fn_addrs_and_calls_stmt(s, &mut addr_taken_fns, &mut call_sites);
        }
    }
    for vec in &module.irq_vectors {
        for b in &vec.bindings {
            for barg in &b.baked_args {
                if let MirBakedArg::Const(e) = barg {
                    collect_fn_addrs_and_calls_expr(e, &mut addr_taken_fns, &mut call_sites);
                }
            }
            for sv in &b.state_vars {
                collect_fn_addrs_and_calls_expr(&sv.init, &mut addr_taken_fns, &mut call_sites);
            }
        }
    }

    for name in addr_taken_fns.into_iter().chain(call_sites.into_keys()) {
        seed(name);
    }

    let fn_index: BTreeMap<String, usize> = module
        .functions
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.clone(), i))
        .collect();

    while let Some(name) = worklist.pop_front() {
        let Some(&idx) = fn_index.get(&name) else {
            continue;
        };
        let mut sub_addrs = BTreeSet::new();
        let mut sub_calls: BTreeMap<String, Vec<Vec<MirExpr>>> = BTreeMap::new();
        let mut direct_callees = Vec::new();
        match &module.functions[idx].body {
            MirFnBody::Direct(stmts) => {
                for s in stmts {
                    collect_fn_addrs_and_calls_stmt(s, &mut sub_addrs, &mut sub_calls);
                }
            }
            MirFnBody::Coroutine {
                blocks,
                child_slots,
            } => {
                for slot in child_slots {
                    direct_callees.push(slot.callee.clone());
                }
                for blk in blocks {
                    for s in &blk.stmts {
                        collect_fn_addrs_and_calls_stmt(s, &mut sub_addrs, &mut sub_calls);
                    }
                    collect_fn_addrs_and_calls_term(
                        &blk.terminator,
                        &mut sub_addrs,
                        &mut sub_calls,
                    );
                }
            }
        }
        for next in direct_callees
            .into_iter()
            .chain(sub_addrs)
            .chain(sub_calls.into_keys())
        {
            if reachable.insert(next.clone()) {
                worklist.push_back(next);
            }
        }
    }

    module.functions.retain(|f| reachable.contains(&f.name));
    module
        .dual_specialized_fns
        .retain(|name| reachable.contains(name) || reachable.contains(&format!("{name}__noyield")));
    module.monomorphized_instances.retain(|item| {
        item.kind == crate::ir::MonomorphizedItemKind::Struct
            || reachable.contains(&item.mangled_name)
    });
    if module.boot.from_main_entry {
        prune_unused_imported_globals(module);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;
    use crate::ram::{populate_module_ram_report, RamReport};

    fn empty_module() -> MirModule {
        MirModule {
            tick_period_us: 1000,
            systick_hz: None,
            imports: vec![],
            consts: vec![],
            enums: vec![],
            structs: vec![],
            subsystems: vec![],
            mmio_layouts: vec![],
            mmio_instances: vec![],
            primitives: vec![],
            extern_fns: vec![],
            isrs: vec![],
            irq_vectors: vec![],
            functions: vec![],
            task_blueprints: vec![],
            spawned_tasks: vec![],
            dual_specialized_fns: vec![],
            monomorphized_instances: vec![],
            formatted_enums: vec![],
            boot: MirBootSequence {
                present: true,
                from_main_entry: false,
                vars: vec![],
                steps: vec![],
                boot_yielding_callees: vec![],
            },
            visibility: VisibilityReport::default(),
            ram_report: RamReport::default(),
        }
    }

    fn lower_ast(items: Vec<zev_ast::Item>) -> MirModule {
        crate::lower_program(&zev_ast::Program::new(items, zev_ast::Span::dummy()))
            .expect("lower AST program to MIR")
    }

    #[test]
    fn dead_watch_stripping_and_bytes_saved_work_on_lowered_program() {
        use zev_ast::*;
        let sp = Span::dummy();
        let id = |s: &str| Ident::dummy(s);
        let ty_prim = |p| TypeExpr::new(TypeKind::Prim(p), sp);
        let ty_watch = |p| {
            TypeExpr::new(
                TypeKind::OsPrimitive(OsPrimitiveType::Watch {
                    elem: Box::new(ty_prim(p)),
                }),
                sp,
            )
        };
        let ty_ptr_watch = |p| TypeExpr::new(TypeKind::Ptr(Box::new(ty_watch(p))), sp);
        let expr_int = |v| {
            Expr::new(
                ExprKind::Literal(Literal::Int {
                    value: v,
                    suffix: None,
                }),
                sp,
            )
        };
        let expr_id = |n: &str| Expr::new(ExprKind::Ident(Ident::dummy(n)), sp);
        let expr_addr = |n: &str| {
            Expr::new(
                ExprKind::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(expr_id(n)),
                },
                sp,
            )
        };
        let expr_call = |callee: &str, args: Vec<Expr>| {
            Expr::new(
                ExprKind::Call {
                    callee: Box::new(expr_id(callee)),
                    generic_args: vec![],
                    args: args.into_iter().map(CallArg::positional).collect(),
                },
                sp,
            )
        };
        let expr_method = |recv: Expr, m: &str, args: Vec<Expr>| {
            Expr::new(
                ExprKind::MethodCall {
                    receiver: Box::new(recv),
                    method: Ident::dummy(m),
                    args: args.into_iter().map(CallArg::positional).collect(),
                },
                sp,
            )
        };

        let mut mir = lower_ast(vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                    attrs: vec![],
                    ty: OsPrimitiveType::Watch {
                        elem: Box::new(ty_prim(PrimType::U32)),
                    },
                    name: id("dead_pub_only"),
                    init: Some(expr_int(0)),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                    attrs: vec![],
                    ty: OsPrimitiveType::Watch {
                        elem: Box::new(ty_prim(PrimType::U16)),
                    },
                    name: id("live_read_task"),
                    init: Some(expr_int(0)),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                    attrs: vec![],
                    ty: OsPrimitiveType::Watch {
                        elem: Box::new(ty_prim(PrimType::U64)),
                    },
                    name: id("dead_ptr_pub_only"),
                    init: Some(expr_int(0)),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                    attrs: vec![],
                    ty: OsPrimitiveType::Watch {
                        elem: Box::new(ty_prim(PrimType::U8)),
                    },
                    name: id("live_ptr_read"),
                    init: Some(expr_int(0)),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("side_effect_val"),
                    generic_params: vec![],
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("x"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::U32),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Return(Some(Expr::new(
                                ExprKind::Binary {
                                    op: BinaryOp::Add,
                                    lhs: Box::new(expr_id("x")),
                                    rhs: Box::new(expr_int(7)),
                                },
                                sp,
                            ))),
                            sp,
                        )],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("publish_via_ptr"),
                    generic_params: vec![],
                    params: vec![
                        Param {
                            ty: ty_ptr_watch(PrimType::U64),
                            name: id("w"),
                            span: sp,
                        },
                        Param {
                            ty: ty_prim(PrimType::U64),
                            name: id("v"),
                            span: sp,
                        },
                    ],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Expr(expr_method(
                                expr_id("w"),
                                "publish",
                                vec![expr_id("v")],
                            )),
                            sp,
                        )],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("read_via_ptr"),
                    generic_params: vec![],
                    params: vec![Param {
                        ty: ty_ptr_watch(PrimType::U8),
                        name: id("w"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::U8),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Return(Some(expr_method(expr_id("w"), "read", vec![]))),
                            sp,
                        )],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Task(TaskDecl {
                    attrs: vec![],
                    name: id("worker"),
                    params: vec![],
                    body: Block::new(
                        vec![Stmt::new(
                            StmtKind::LoopEvery {
                                period: Box::new(Expr::new(
                                    ExprKind::Literal(Literal::Duration(DurationLit {
                                        value: 5,
                                        unit: DurationUnit::Ms,
                                        span: sp,
                                    })),
                                    sp,
                                )),
                                body: Block::new(
                                    vec![
                                        Stmt::new(
                                            StmtKind::Expr(expr_method(
                                                expr_id("dead_pub_only"),
                                                "publish",
                                                vec![expr_call(
                                                    "side_effect_val",
                                                    vec![expr_int(1)],
                                                )],
                                            )),
                                            sp,
                                        ),
                                        Stmt::new(
                                            StmtKind::Expr(expr_method(
                                                expr_id("live_read_task"),
                                                "publish",
                                                vec![expr_int(42)],
                                            )),
                                            sp,
                                        ),
                                        Stmt::new(
                                            StmtKind::Expr(expr_call(
                                                "publish_via_ptr",
                                                vec![expr_addr("dead_ptr_pub_only"), expr_int(99)],
                                            )),
                                            sp,
                                        ),
                                        Stmt::new(
                                            StmtKind::Expr(expr_method(
                                                expr_id("live_ptr_read"),
                                                "publish",
                                                vec![expr_int(5)],
                                            )),
                                            sp,
                                        ),
                                    ],
                                    sp,
                                ),
                            },
                            sp,
                        )],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("main"),
                    generic_params: vec![],
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::Expr(expr_method(
                                    expr_id("dead_pub_only"),
                                    "publish",
                                    vec![expr_int(10)],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::If {
                                    cond: Expr::new(ExprKind::Literal(Literal::Bool(true)), sp),
                                    then_block: Block::new(
                                        vec![Stmt::new(
                                            StmtKind::Expr(expr_method(
                                                expr_id("dead_pub_only"),
                                                "publish",
                                                vec![expr_call(
                                                    "side_effect_val",
                                                    vec![expr_int(2)],
                                                )],
                                            )),
                                            sp,
                                        )],
                                        sp,
                                    ),
                                    else_block: None,
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Spawn(SpawnStmt {
                                    module: None,
                                    task: id("worker"),
                                    args: vec![],
                                    priority: Some(Priority::Normal),
                                    core: None,
                                    span: sp,
                                }),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U16),
                                    name: id("a"),
                                    init: Some(expr_method(
                                        expr_id("live_read_task"),
                                        "read",
                                        vec![],
                                    )),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U8),
                                    name: id("b"),
                                    init: Some(expr_call(
                                        "read_via_ptr",
                                        vec![expr_addr("live_ptr_read")],
                                    )),
                                },
                                sp,
                            ),
                        ],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
        ]);
        let report = analyze_visibility(&mir);

        assert_eq!(
            report.dead_watches,
            BTreeSet::from(["dead_pub_only".to_string(), "dead_ptr_pub_only".to_string()])
        );
        // `dead_pub_only` (`watch<u32>`, 12 B) is stripped; `dead_ptr_pub_only` has `&` taken so its `.bss` stays.
        assert_eq!(report.dead_watch_bytes_saved_32, 12);
        assert_eq!(report.dead_watch_bytes_saved_64, 12);
        assert!(report.isr_contended_globals.is_empty());
        assert!(report.isr_contended_prim_mangles.is_empty());
        assert_eq!(
            report.elided_globals,
            BTreeSet::from(["live_read_task".to_string(), "live_ptr_read".to_string()])
        );

        strip_dead_watch_publishes(&mut mir, &report);

        let mut remaining_dead_publishes = 0usize;
        let mut preserved_side_effect_calls = 0usize;
        let mut live_publishes = 0usize;

        fn inspect_stmts(
            stmts: &[MirStmt],
            dead: &BTreeSet<String>,
            rem_dead: &mut usize,
            kept_calls: &mut usize,
            live_pubs: &mut usize,
        ) {
            for s in stmts {
                match s {
                    MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                        target: MirPrimTarget::Global(name),
                        ..
                    }) => {
                        if dead.contains(name) {
                            *rem_dead += 1;
                        } else {
                            *live_pubs += 1;
                        }
                    }
                    MirStmt::Expr(MirExpr {
                        kind: MirExprKind::CallDirect { callee, .. },
                        ..
                    }) if callee == "side_effect_val" => {
                        *kept_calls += 1;
                    }
                    MirStmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        inspect_stmts(then_body, dead, rem_dead, kept_calls, live_pubs);
                        inspect_stmts(else_body, dead, rem_dead, kept_calls, live_pubs);
                    }
                    MirStmt::While { body, .. } => {
                        inspect_stmts(body, dead, rem_dead, kept_calls, live_pubs);
                    }
                    MirStmt::Switch { arms, default, .. } => {
                        for (_, b) in arms {
                            inspect_stmts(b, dead, rem_dead, kept_calls, live_pubs);
                        }
                        inspect_stmts(default, dead, rem_dead, kept_calls, live_pubs);
                    }
                    _ => {}
                }
            }
        }

        for step in &mir.boot.steps {
            if let MirBootStep::Stmt(s) = step {
                inspect_stmts(
                    std::slice::from_ref(s),
                    &report.dead_watches,
                    &mut remaining_dead_publishes,
                    &mut preserved_side_effect_calls,
                    &mut live_publishes,
                );
            }
        }
        for bp in &mir.task_blueprints {
            for blk in &bp.blocks {
                inspect_stmts(
                    &blk.stmts,
                    &report.dead_watches,
                    &mut remaining_dead_publishes,
                    &mut preserved_side_effect_calls,
                    &mut live_publishes,
                );
            }
        }

        assert_eq!(remaining_dead_publishes, 0);
        assert_eq!(preserved_side_effect_calls, 2);
        assert_eq!(live_publishes, 2);
    }

    #[test]
    fn addr_taken_with_ptr_watch_read_keeps_matching_watches_alive() {
        let mut mir = empty_module();
        mir.primitives = vec![
            MirPrimitive {
                name: "w_a".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U16),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "w_b".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U16),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "w_other_ty".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
        ];
        mir.functions.push(MirFunction {
            name: "read_any".to_string(),
            params: vec![VarId(0)],
            ret_ty: MirType::U16,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![MirVar {
                id: VarId(0),
                name: "w".to_string(),
                ty: MirType::Ptr(Box::new(MirType::Primitive(MirPrimKind::Watch {
                    elem: Box::new(MirType::U16),
                }))),
                role: VarRole::Param,
                in_noyield_scope: false,
                addr_taken: false,
                hoisted: false,
            }],
            body: MirFnBody::Direct(vec![MirStmt::Return(Some(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                    target: MirPrimTarget::PtrExpr(MirExpr::new(
                        MirExprKind::Var(VarId(0)),
                        MirType::Ptr(Box::new(MirType::Primitive(MirPrimKind::Watch {
                            elem: Box::new(MirType::U16),
                        }))),
                    )),
                    elem_ty: MirType::U16,
                })),
                MirType::U16,
            )))]),
        });
        let ptr_u16_watch = MirType::Ptr(Box::new(MirType::Primitive(MirPrimKind::Watch {
            elem: Box::new(MirType::U16),
        })));
        mir.boot.steps = vec![
            MirBootStep::Stmt(Box::new(MirStmt::Expr(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "read_any".to_string(),
                    is_extern_c: false,
                    args: vec![MirExpr::new(
                        MirExprKind::GlobalPrimAddr("w_a".to_string()),
                        ptr_u16_watch.clone(),
                    )],
                },
                MirType::U16,
            )))),
            MirBootStep::Stmt(Box::new(MirStmt::PrimOp(
                MirPrimNonBlockingOp::WatchPublish {
                    target: MirPrimTarget::PtrExpr(MirExpr::new(
                        MirExprKind::GlobalPrimAddr("w_b".to_string()),
                        ptr_u16_watch,
                    )),
                    elem_ty: MirType::U16,
                    value: MirExpr::new(MirExprKind::IntLit(20), MirType::U16),
                },
            ))),
            MirBootStep::Stmt(Box::new(MirStmt::PrimOp(
                MirPrimNonBlockingOp::WatchPublish {
                    target: MirPrimTarget::Global("w_other_ty".to_string()),
                    elem_ty: MirType::U32,
                    value: MirExpr::new(MirExprKind::IntLit(30), MirType::U32),
                },
            ))),
        ];

        let report = analyze_visibility(&mir);
        assert!(
            !report.dead_watches.contains("w_a"),
            "w_a is addr-taken and WatchRead(*watch<u16>) exists"
        );
        assert!(
            !report.dead_watches.contains("w_b"),
            "w_b is conservatively kept alive when addr-taken and WatchRead(*watch<u16>) exists"
        );
        assert!(
            report.dead_watches.contains("w_other_ty"),
            "w_other_ty has different elem type (u32 vs u16) and no reads"
        );
    }

    #[test]
    fn isr_contention_tracks_baked_args_direct_ops_and_transitive_callees() {
        let mut mir = empty_module();
        mir.primitives = vec![
            MirPrimitive {
                name: "isr_baked_w".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "task_only_w".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "dead_isr_w".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "isr_callee_q".to_string(),
                kind: MirPrimKind::Inbox {
                    elem: Box::new(MirType::U16),
                    cap: 4,
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "task_only_sig".to_string(),
                kind: MirPrimKind::Signal,
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "isr_ptr_ring".to_string(),
                kind: MirPrimKind::ByteRing { cap: 16 },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
        ];

        let q_ptr_ty = MirType::Ptr(Box::new(MirType::Primitive(MirPrimKind::Inbox {
            elem: Box::new(MirType::U16),
            cap: 4,
        })));
        let w_ptr_ty = MirType::Ptr(Box::new(MirType::Primitive(MirPrimKind::Watch {
            elem: Box::new(MirType::U32),
        })));
        let r_ptr_ty = MirType::Ptr(Box::new(MirType::Primitive(MirPrimKind::ByteRing {
            cap: 16,
        })));

        mir.functions.push(MirFunction {
            name: "leaf_notify".to_string(),
            params: vec![VarId(0)],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![MirVar {
                id: VarId(0),
                name: "q".to_string(),
                ty: q_ptr_ty.clone(),
                role: VarRole::Param,
                in_noyield_scope: false,
                addr_taken: false,
                hoisted: false,
            }],
            body: MirFnBody::Direct(vec![MirStmt::PrimOp(MirPrimNonBlockingOp::InboxPost {
                target: MirPrimTarget::PtrExpr(MirExpr::new(
                    MirExprKind::Var(VarId(0)),
                    q_ptr_ty.clone(),
                )),
                elem_ty: MirType::U16,
                cap: 4,
                value: MirExpr::new(MirExprKind::IntLit(55), MirType::U16),
                boost: None,
            })]),
        });
        mir.functions.push(MirFunction {
            name: "mid_helper".to_string(),
            params: vec![],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![MirStmt::Expr(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "leaf_notify".to_string(),
                    is_extern_c: false,
                    args: vec![MirExpr::new(
                        MirExprKind::GlobalPrimAddr("isr_callee_q".to_string()),
                        q_ptr_ty,
                    )],
                },
                MirType::Void,
            ))]),
        });
        mir.isrs.push(MirIsr {
            name: "handler_a".to_string(),
            params: vec![VarId(0), VarId(1)],
            state_vars: vec![],
            vars: vec![
                MirVar {
                    id: VarId(0),
                    name: "w".to_string(),
                    ty: w_ptr_ty.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: false,
                },
                MirVar {
                    id: VarId(1),
                    name: "rx".to_string(),
                    ty: r_ptr_ty.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: false,
                },
            ],
            body: vec![
                MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                    target: MirPrimTarget::PtrExpr(MirExpr::new(
                        MirExprKind::Var(VarId(0)),
                        w_ptr_ty,
                    )),
                    elem_ty: MirType::U32,
                    value: MirExpr::new(MirExprKind::IntLit(1), MirType::U32),
                }),
                MirStmt::Expr(MirExpr::new(
                    MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingPushByte {
                        target: MirPrimTarget::PtrExpr(MirExpr::new(
                            MirExprKind::Var(VarId(1)),
                            r_ptr_ty,
                        )),
                        cap: 16,
                        byte: Box::new(MirExpr::new(MirExprKind::IntLit(65), MirType::U8)),
                        boost: None,
                    })),
                    MirType::Bool,
                )),
                MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                    target: MirPrimTarget::Global("dead_isr_w".to_string()),
                    elem_ty: MirType::U32,
                    value: MirExpr::new(MirExprKind::IntLit(99), MirType::U32),
                }),
            ],
        });
        mir.isrs.push(MirIsr {
            name: "handler_b".to_string(),
            params: vec![],
            state_vars: vec![],
            vars: vec![],
            body: vec![MirStmt::Expr(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "mid_helper".to_string(),
                    is_extern_c: false,
                    args: vec![],
                },
                MirType::Void,
            ))],
        });
        mir.irq_vectors = vec![
            MirIrqVector {
                irq_name: "IRQ_EXTI0".to_string(),
                irq_num: 6,
                is_shared: false,
                selector_symbol: None,
                bindings: vec![MirIrqBinding {
                    binding_id: 0,
                    isr_name: "handler_a".to_string(),
                    baked_args: vec![
                        MirBakedArg::PrimAddr("isr_baked_w".to_string()),
                        MirBakedArg::PrimAddr("isr_ptr_ring".to_string()),
                    ],
                    state_vars: vec![],
                    cond_selector: None,
                }],
            },
            MirIrqVector {
                irq_name: "IRQ_EXTI1".to_string(),
                irq_num: 7,
                is_shared: false,
                selector_symbol: None,
                bindings: vec![MirIrqBinding {
                    binding_id: 1,
                    isr_name: "handler_b".to_string(),
                    baked_args: vec![],
                    state_vars: vec![],
                    cond_selector: None,
                }],
            },
        ];
        // Task reads `isr_baked_w`, `task_only_w`, and uses `task_only_sig`
        mir.task_blueprints.push(MirTaskBlueprint {
            name: "consumer".to_string(),
            is_main_supervisor: false,
            manual_yield: false,
            params: vec![],
            vars: vec![],
            child_slots: vec![],
            blocks: vec![MirBlock {
                id: BlockId(0),
                is_resume_entry: false,
                stmts: vec![
                    MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                        target: MirPrimTarget::Global("task_only_w".to_string()),
                        elem_ty: MirType::U32,
                        value: MirExpr::new(MirExprKind::IntLit(7), MirType::U32),
                    }),
                    MirStmt::Expr(MirExpr::new(
                        MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                            target: MirPrimTarget::Global("isr_baked_w".to_string()),
                            elem_ty: MirType::U32,
                        })),
                        MirType::U32,
                    )),
                    MirStmt::Expr(MirExpr::new(
                        MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                            target: MirPrimTarget::Global("task_only_w".to_string()),
                            elem_ty: MirType::U32,
                        })),
                        MirType::U32,
                    )),
                    MirStmt::PrimOp(MirPrimNonBlockingOp::SignalNotify {
                        target: MirPrimTarget::Global("task_only_sig".to_string()),
                        boost: None,
                    }),
                ],
                terminator: MirTerminator::Return(None),
            }],
        });

        let report = analyze_visibility(&mir);

        assert!(report.dead_watches.contains("dead_isr_w"));
        assert!(!report.dead_watches.contains("isr_baked_w"));
        assert!(!report.dead_watches.contains("task_only_w"));

        assert_eq!(
            report.isr_contended_globals,
            BTreeSet::from([
                "dead_isr_w".to_string(),
                "isr_baked_w".to_string(),
                "isr_callee_q".to_string(),
                "isr_ptr_ring".to_string(),
            ])
        );
        assert_eq!(
            report.isr_contended_prim_mangles,
            BTreeSet::from([
                "watch_u32".to_string(),
                "inbox_u16_4".to_string(),
                "byte_ring_16".to_string(),
            ])
        );
        assert_eq!(
            report.elided_globals,
            BTreeSet::from(["task_only_w".to_string(), "task_only_sig".to_string()])
        );

        strip_dead_watch_publishes(&mut mir, &report);
        let handler_a = mir.isrs.iter().find(|i| i.name == "handler_a").unwrap();
        assert!(!handler_a.body.iter().any(|s| matches!(
            s,
            MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                target: MirPrimTarget::Global(name),
                ..
            }) if name == "dead_isr_w"
        )));
    }

    #[test]
    fn constructed_mir_with_cyclic_callees_and_pointer_watch_size_check() {
        let mut mir = empty_module();
        mir.structs.push(MirStruct {
            name: "Pair".to_string(),
            packed: false,
            is_subsystem: false,
            fields: vec![
                MirField {
                    name: "a".to_string(),
                    ty: MirType::Usize,
                    dma_safe: false,
                },
                MirField {
                    name: "b".to_string(),
                    ty: MirType::U8,
                    dma_safe: false,
                },
            ],
        });
        mir.primitives.push(MirPrimitive {
            name: "dead_pair_w".to_string(),
            kind: MirPrimKind::Watch {
                elem: Box::new(MirType::Struct("Pair".to_string())),
            },
            dma_safe: false,
            init: None,
            sleep_lock_bit: None,
            dead_stripped: false,
            irq_mask_elided: false,
        });
        mir.primitives.push(MirPrimitive {
            name: "task_sig".to_string(),
            kind: MirPrimKind::Signal,
            dma_safe: false,
            init: None,
            sleep_lock_bit: None,
            dead_stripped: false,
            irq_mask_elided: false,
        });
        // Cyclic helper functions reachable from ISR (must terminate cleanly without stack overflow)
        mir.functions.push(MirFunction {
            name: "cyc_a".to_string(),
            params: vec![],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![MirStmt::Expr(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "cyc_b".to_string(),
                    is_extern_c: false,
                    args: vec![],
                },
                MirType::Void,
            ))]),
        });
        mir.functions.push(MirFunction {
            name: "cyc_b".to_string(),
            params: vec![],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![
                MirStmt::Expr(MirExpr::new(
                    MirExprKind::CallDirect {
                        callee: "cyc_a".to_string(),
                        is_extern_c: false,
                        args: vec![],
                    },
                    MirType::Void,
                )),
                MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                    target: MirPrimTarget::Global("dead_pair_w".to_string()),
                    elem_ty: MirType::Struct("Pair".to_string()),
                    value: MirExpr::new(
                        MirExprKind::IntLit(0),
                        MirType::Struct("Pair".to_string()),
                    ),
                }),
            ]),
        });
        mir.isrs.push(MirIsr {
            name: "my_isr".to_string(),
            params: vec![],
            state_vars: vec![],
            vars: vec![],
            body: vec![MirStmt::Expr(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "cyc_a".to_string(),
                    is_extern_c: false,
                    args: vec![],
                },
                MirType::Void,
            ))],
        });
        populate_module_ram_report(&mut mir);

        let report = analyze_visibility(&mir);
        assert_eq!(
            report.dead_watches,
            BTreeSet::from(["dead_pair_w".to_string()])
        );
        // Match ram.rs primitive size calculations identically:
        let prim_ram = mir
            .ram_report
            .primitives
            .iter()
            .find(|p| p.name == "dead_pair_w")
            .unwrap();
        assert_eq!(report.dead_watch_bytes_saved_32, prim_ram.bytes_32);
        assert_eq!(report.dead_watch_bytes_saved_64, prim_ram.bytes_64);
        assert_eq!(
            report.isr_contended_globals,
            BTreeSet::from(["dead_pair_w".to_string()])
        );
        assert_eq!(
            report.elided_globals,
            BTreeSet::from(["task_sig".to_string()])
        );

        strip_dead_watch_publishes(&mut mir, &report);
        assert!(matches!(
            &mir.functions[1].body,
            MirFnBody::Direct(stmts) if matches!(&stmts[1], MirStmt::Expr(_))
        ));
    }

    #[test]
    fn coroutine_fn_strips_dead_watch_while_print_fmt_watch_read_stays_live() {
        let mut mir = empty_module();
        mir.primitives = vec![
            MirPrimitive {
                name: "coro_dead_w".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: true,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "fmt_live_w".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U16),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
        ];
        mir.functions.push(MirFunction {
            name: "coro_worker".to_string(),
            params: vec![],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: true,
            vars: vec![],
            body: MirFnBody::Coroutine {
                blocks: vec![MirBlock {
                    id: BlockId(0),
                    is_resume_entry: false,
                    stmts: vec![
                        MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                            target: MirPrimTarget::Global("coro_dead_w".to_string()),
                            elem_ty: MirType::U32,
                            value: MirExpr::new(MirExprKind::IntLit(123), MirType::U32),
                        }),
                        MirStmt::PrintFmt {
                            chunks: vec![MirFmtChunk::Int {
                                value: MirExpr::new(
                                    MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                                        target: MirPrimTarget::Global("fmt_live_w".to_string()),
                                        elem_ty: MirType::U16,
                                    })),
                                    MirType::U16,
                                ),
                                signed: false,
                                bits: 16,
                                fmt: MirIntFmt::Dec,
                            }],
                            newline: true,
                        },
                    ],
                    terminator: MirTerminator::Return(None),
                }],
                child_slots: vec![],
            },
        });

        let report = analyze_visibility(&mir);
        assert_eq!(
            report.dead_watches,
            BTreeSet::from(["coro_dead_w".to_string()])
        );
        assert_eq!(
            report.elided_globals,
            BTreeSet::from(["fmt_live_w".to_string()])
        );
        assert_eq!(report.dead_watch_bytes_saved_32, 12);
        assert_eq!(report.dead_watch_bytes_saved_64, 12);

        strip_dead_watch_publishes(&mut mir, &report);
        match &mir.functions[0].body {
            MirFnBody::Coroutine { blocks, .. } => {
                assert!(matches!(blocks[0].stmts[0], MirStmt::Expr(_)));
                assert!(matches!(blocks[0].stmts[1], MirStmt::PrintFmt { .. }));
            }
            _ => panic!("expected Coroutine body"),
        }
    }

    #[test]
    fn fold_singletons_folds_single_subsystem_task_params_locals_and_uniform_fn() {
        let mut mir = empty_module();
        let ring_ty = MirType::Primitive(MirPrimKind::ByteRing { cap: 64 });
        let watch_ty = MirType::Primitive(MirPrimKind::Watch {
            elem: Box::new(MirType::U8),
        });
        let port_ptr_ty = MirType::Ptr(Box::new(MirType::Struct("SerialPort".to_string())));
        let view_ty = MirType::PrimView(MirPrimViewKind::ByteRing);
        let standalone_w_ptr = MirType::Ptr(Box::new(MirType::Primitive(MirPrimKind::Watch {
            elem: Box::new(MirType::U32),
        })));

        mir.structs.push(MirStruct {
            name: "SerialPort".to_string(),
            packed: false,
            is_subsystem: true,
            fields: vec![
                MirField {
                    name: "rx".to_string(),
                    ty: ring_ty.clone(),
                    dma_safe: false,
                },
                MirField {
                    name: "status".to_string(),
                    ty: watch_ty.clone(),
                    dma_safe: false,
                },
            ],
        });
        mir.subsystems.push(MirSubsystemGlobal {
            name: "usb_serial".to_string(),
            struct_name: "SerialPort".to_string(),
            dma_safe: false,
        });
        mir.primitives.push(MirPrimitive {
            name: "standalone_w".to_string(),
            kind: MirPrimKind::Watch {
                elem: Box::new(MirType::U32),
            },
            dma_safe: false,
            init: None,
            sleep_lock_bit: None,
            dead_stripped: false,
            irq_mask_elided: false,
        });

        let subsys_addr = MirExpr::new(
            MirExprKind::SubsystemAddr("usb_serial".to_string()),
            port_ptr_ty.clone(),
        );
        let w_addr = MirExpr::new(
            MirExprKind::GlobalPrimAddr("standalone_w".to_string()),
            standalone_w_ptr.clone(),
        );

        // Non-address-taken Direct fn called twice with uniform constant arguments.
        mir.functions.push(MirFunction {
            name: "flush_tx".to_string(),
            params: vec![VarId(0), VarId(1)],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![
                MirVar {
                    id: VarId(0),
                    name: "port".to_string(),
                    ty: port_ptr_ty.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: false,
                },
                MirVar {
                    id: VarId(1),
                    name: "baud".to_string(),
                    ty: MirType::U32,
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: false,
                },
            ],
            body: MirFnBody::Direct(vec![MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                target: MirPrimTarget::PtrExpr(MirExpr::new(
                    MirExprKind::AddrOfPlace(Box::new(MirPlace::Field {
                        base: Box::new(MirPlace::Deref(
                            Box::new(MirExpr::new(
                                MirExprKind::Var(VarId(0)),
                                port_ptr_ty.clone(),
                            )),
                            MirType::Struct("SerialPort".to_string()),
                        )),
                        struct_name: "SerialPort".to_string(),
                        packed: false,
                        field: "status".to_string(),
                        field_ty: watch_ty.clone(),
                    })),
                    MirType::Ptr(Box::new(watch_ty.clone())),
                )),
                elem_ty: MirType::U8,
                value: MirExpr::new(
                    MirExprKind::Cast {
                        expr: Box::new(MirExpr::new(MirExprKind::Var(VarId(1)), MirType::U32)),
                        target_ty: MirType::U8,
                    },
                    MirType::U8,
                ),
            })]),
        });

        mir.boot.steps.push(MirBootStep::SpawnActivate {
            slot_id: 0,
            args: vec![
                subsys_addr.clone(),
                MirExpr::new(MirExprKind::IntLit(9600), MirType::U32),
                w_addr.clone(),
            ],
        });
        mir.spawned_tasks.push(MirSpawnInstance {
            slot_id: 0,
            instance_name: "usb_cmd_worker_0".to_string(),
            blueprint_name: "usb_cmd_worker".to_string(),
            priority: MirPriority::Normal,
            core_pin: None,
            init_args: vec![
                subsys_addr,
                MirExpr::new(MirExprKind::IntLit(9600), MirType::U32),
                w_addr,
            ],
            is_main_supervisor: false,
            provenance_path: vec!["main".to_string(), "usb_cmd_worker".to_string()],
            conditional: false,
        });

        mir.task_blueprints.push(MirTaskBlueprint {
            name: "usb_cmd_worker".to_string(),
            is_main_supervisor: false,
            manual_yield: false,
            params: vec![VarId(0), VarId(1), VarId(2)],
            vars: vec![
                MirVar {
                    id: VarId(0),
                    name: "port".to_string(),
                    ty: port_ptr_ty.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(1),
                    name: "rate".to_string(),
                    ty: MirType::U32,
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(2),
                    name: "w".to_string(),
                    ty: standalone_w_ptr.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(3),
                    name: "rx".to_string(),
                    ty: view_ty.clone(),
                    role: VarRole::UserLocal,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(4),
                    name: "scalar_keep".to_string(),
                    ty: MirType::F64,
                    role: VarRole::UserLocal,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
            ],
            child_slots: vec![],
            blocks: vec![
                MirBlock {
                    id: BlockId(0),
                    is_resume_entry: false,
                    stmts: vec![
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(3)),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(
                                MirExprKind::MakePrimView {
                                    ptr: Box::new(MirExpr::new(
                                        MirExprKind::AddrOfPlace(Box::new(MirPlace::Field {
                                            base: Box::new(MirPlace::Deref(
                                                Box::new(MirExpr::new(
                                                    MirExprKind::Var(VarId(0)),
                                                    port_ptr_ty.clone(),
                                                )),
                                                MirType::Struct("SerialPort".to_string()),
                                            )),
                                            struct_name: "SerialPort".to_string(),
                                            packed: false,
                                            field: "rx".to_string(),
                                            field_ty: ring_ty.clone(),
                                        })),
                                        MirType::Ptr(Box::new(ring_ty.clone())),
                                    )),
                                    cap: 64,
                                    kind: MirPrimViewKind::ByteRing,
                                },
                                view_ty.clone(),
                            ),
                        },
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(4)),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(
                                MirExprKind::FloatLit(2.5_f64.to_bits()),
                                MirType::F64,
                            ),
                        },
                    ],
                    terminator: MirTerminator::Yield(MirYieldKind::ShareCheckpoint {
                        resume_bb: BlockId(1),
                    }),
                },
                MirBlock {
                    id: BlockId(1),
                    is_resume_entry: true,
                    stmts: vec![
                        MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                            target: MirPrimTarget::PtrExpr(MirExpr::new(
                                MirExprKind::Var(VarId(2)),
                                standalone_w_ptr.clone(),
                            )),
                            elem_ty: MirType::U32,
                            value: MirExpr::new(
                                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewLen {
                                    view: MirExpr::new(MirExprKind::Var(VarId(3)), view_ty.clone()),
                                })),
                                MirType::U32,
                            ),
                        }),
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::CallDirect {
                                callee: "flush_tx".to_string(),
                                is_extern_c: false,
                                args: vec![
                                    MirExpr::new(MirExprKind::Var(VarId(0)), port_ptr_ty.clone()),
                                    MirExpr::new(MirExprKind::IntLit(115200), MirType::U32),
                                ],
                            },
                            MirType::Void,
                        )),
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::CallDirect {
                                callee: "flush_tx".to_string(),
                                is_extern_c: false,
                                args: vec![
                                    MirExpr::new(MirExprKind::Var(VarId(0)), port_ptr_ty.clone()),
                                    MirExpr::new(MirExprKind::IntLit(115200), MirType::U32),
                                ],
                            },
                            MirType::Void,
                        )),
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::Binary {
                                op: MirBinOp::Add,
                                lhs: Box::new(MirExpr::new(
                                    MirExprKind::Cast {
                                        expr: Box::new(MirExpr::new(
                                            MirExprKind::Var(VarId(1)),
                                            MirType::U32,
                                        )),
                                        target_ty: MirType::F64,
                                    },
                                    MirType::F64,
                                )),
                                rhs: Box::new(MirExpr::new(
                                    MirExprKind::Var(VarId(4)),
                                    MirType::F64,
                                )),
                            },
                            MirType::F64,
                        )),
                    ],
                    terminator: MirTerminator::Return(None),
                },
            ],
        });

        fold_singletons(&mut mir);
        mir.visibility = analyze_visibility(&mir);
        populate_module_ram_report(&mut mir);

        assert!(mir.visibility.singleton_subsystems.contains("usb_serial"));
        assert!(mir
            .visibility
            .singleton_subsystem_structs
            .contains_key("SerialPort"));
        assert!(mir.visibility.singleton_bytes_saved_32 > 0);
        assert!(mir.visibility.singleton_bytes_saved_64 > 0);

        let bp = &mir.task_blueprints[0];
        assert!(!bp.vars[0].hoisted, "port param should fold");
        assert!(!bp.vars[1].hoisted, "rate param should fold");
        assert!(!bp.vars[2].hoisted, "w param should fold");
        assert!(!bp.vars[3].hoisted, "*byte_ring rx local should fold");
        assert!(
            bp.vars[4].hoisted,
            "non-ref scalar local scalar_keep must remain hoisted"
        );

        // Dead assign for `rx` removed, `scalar_keep` assign retained.
        assert_eq!(bp.blocks[0].stmts.len(), 1);
        // `MirPrimTarget::PtrExpr(GlobalPrimAddr("standalone_w"))` rewritten to `Global("standalone_w")`.
        match &bp.blocks[1].stmts[0] {
            MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish { target, value, .. }) => {
                assert!(matches!(target, MirPrimTarget::Global(name) if name == "standalone_w"));
                assert!(matches!(
                    value.kind,
                    MirExprKind::PrimRead(ref r)
                        if matches!(**r, MirPrimReadOp::ByteRingViewLen { ref view }
                            if matches!(view.kind, MirExprKind::MakePrimView { .. }))
                ));
            }
            other => panic!("expected WatchPublish, got {other:?}"),
        }

        // `flush_tx` params intact (ABI preserved), but body reads folded to constants.
        assert_eq!(mir.functions[0].params.len(), 2);
        match &mir.functions[0].body {
            MirFnBody::Direct(stmts) => {
                assert!(matches!(
                    &stmts[0],
                    MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                        value: MirExpr { kind: MirExprKind::Cast { expr, .. }, .. },
                        ..
                    }) if matches!(expr.kind, MirExprKind::IntLit(115200))
                ));
            }
            _ => panic!("expected Direct body"),
        }

        let table = mir.ram_report.format_table();
        assert!(table.contains("[singleton -> folded]"));
        assert!(table.contains("Singleton Folding Bytes Saved"));
    }

    #[test]
    fn fold_singletons_nested_subsystem_unique_vs_multi_instance() {
        let mut mir = empty_module();
        let watch_u8 = MirType::Primitive(MirPrimKind::Watch {
            elem: Box::new(MirType::U8),
        });
        let inner_one_ptr = MirType::Ptr(Box::new(MirType::Struct("InnerOne".to_string())));
        let inner_two_ptr = MirType::Ptr(Box::new(MirType::Struct("InnerTwo".to_string())));

        for name in ["InnerOne", "InnerTwo"] {
            mir.structs.push(MirStruct {
                name: name.to_string(),
                packed: false,
                is_subsystem: true,
                fields: vec![MirField {
                    name: "flag".to_string(),
                    ty: watch_u8.clone(),
                    dma_safe: false,
                }],
            });
        }
        mir.structs.push(MirStruct {
            name: "OuterSingle".to_string(),
            packed: false,
            is_subsystem: true,
            fields: vec![MirField {
                name: "ch0".to_string(),
                ty: MirType::Struct("InnerOne".to_string()),
                dma_safe: false,
            }],
        });
        mir.structs.push(MirStruct {
            name: "OuterMulti".to_string(),
            packed: false,
            is_subsystem: true,
            fields: vec![
                MirField {
                    name: "ch0".to_string(),
                    ty: MirType::Struct("InnerTwo".to_string()),
                    dma_safe: false,
                },
                MirField {
                    name: "ch1".to_string(),
                    ty: MirType::Struct("InnerTwo".to_string()),
                    dma_safe: false,
                },
            ],
        });
        mir.subsystems.push(MirSubsystemGlobal {
            name: "outer_single".to_string(),
            struct_name: "OuterSingle".to_string(),
            dma_safe: false,
        });
        mir.subsystems.push(MirSubsystemGlobal {
            name: "outer_multi".to_string(),
            struct_name: "OuterMulti".to_string(),
            dma_safe: false,
        });

        let arg_single_nested = MirExpr::new(
            MirExprKind::AddrOfPlace(Box::new(MirPlace::Field {
                base: Box::new(MirPlace::Deref(
                    Box::new(MirExpr::new(
                        MirExprKind::SubsystemAddr("outer_single".to_string()),
                        MirType::Ptr(Box::new(MirType::Struct("OuterSingle".to_string()))),
                    )),
                    MirType::Struct("OuterSingle".to_string()),
                )),
                struct_name: "OuterSingle".to_string(),
                packed: false,
                field: "ch0".to_string(),
                field_ty: MirType::Struct("InnerOne".to_string()),
            })),
            inner_one_ptr.clone(),
        );
        let arg_multi_nested = MirExpr::new(
            MirExprKind::AddrOfPlace(Box::new(MirPlace::Field {
                base: Box::new(MirPlace::Deref(
                    Box::new(MirExpr::new(
                        MirExprKind::SubsystemAddr("outer_multi".to_string()),
                        MirType::Ptr(Box::new(MirType::Struct("OuterMulti".to_string()))),
                    )),
                    MirType::Struct("OuterMulti".to_string()),
                )),
                struct_name: "OuterMulti".to_string(),
                packed: false,
                field: "ch0".to_string(),
                field_ty: MirType::Struct("InnerTwo".to_string()),
            })),
            inner_two_ptr.clone(),
        );

        mir.boot.steps.push(MirBootStep::SpawnActivate {
            slot_id: 0,
            args: vec![arg_single_nested.clone(), arg_multi_nested.clone()],
        });
        mir.spawned_tasks.push(MirSpawnInstance {
            slot_id: 0,
            instance_name: "nested_worker_0".to_string(),
            blueprint_name: "nested_worker".to_string(),
            priority: MirPriority::Normal,
            core_pin: None,
            init_args: vec![arg_single_nested, arg_multi_nested],
            is_main_supervisor: false,
            provenance_path: vec!["main".to_string(), "nested_worker".to_string()],
            conditional: false,
        });

        mir.task_blueprints.push(MirTaskBlueprint {
            name: "nested_worker".to_string(),
            is_main_supervisor: false,
            manual_yield: false,
            params: vec![VarId(0), VarId(1)],
            vars: vec![
                MirVar {
                    id: VarId(0),
                    name: "s_ch".to_string(),
                    ty: inner_one_ptr.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(1),
                    name: "m_ch".to_string(),
                    ty: inner_two_ptr.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
            ],
            child_slots: vec![],
            blocks: vec![MirBlock {
                id: BlockId(0),
                is_resume_entry: false,
                stmts: vec![
                    MirStmt::Expr(MirExpr::new(MirExprKind::Var(VarId(0)), inner_one_ptr)),
                    MirStmt::Expr(MirExpr::new(MirExprKind::Var(VarId(1)), inner_two_ptr)),
                ],
                terminator: MirTerminator::Return(None),
            }],
        });

        fold_singletons(&mut mir);

        let bp = &mir.task_blueprints[0];
        assert!(
            !bp.vars[0].hoisted,
            "unique nested InnerOne (count == 1) must fold"
        );
        assert!(
            bp.vars[1].hoisted,
            "multi-instance nested InnerTwo (count == 2) must NOT fold"
        );
    }

    #[test]
    fn fold_singletons_negative_multi_instance_subsystem_and_multi_spawn_task() {
        let mut mir = empty_module();
        let bus_ptr = MirType::Ptr(Box::new(MirType::Struct("FieldBus".to_string())));
        mir.structs.push(MirStruct {
            name: "FieldBus".to_string(),
            packed: false,
            is_subsystem: true,
            fields: vec![MirField {
                name: "rx_count".to_string(),
                ty: MirType::Primitive(MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                }),
                dma_safe: false,
            }],
        });
        mir.subsystems.push(MirSubsystemGlobal {
            name: "bus1".to_string(),
            struct_name: "FieldBus".to_string(),
            dma_safe: false,
        });
        mir.subsystems.push(MirSubsystemGlobal {
            name: "bus2".to_string(),
            struct_name: "FieldBus".to_string(),
            dma_safe: false,
        });

        let a1 = vec![
            MirExpr::new(
                MirExprKind::SubsystemAddr("bus1".to_string()),
                bus_ptr.clone(),
            ),
            MirExpr::new(MirExprKind::IntLit(1), MirType::U32),
        ];
        let a2 = vec![
            MirExpr::new(
                MirExprKind::SubsystemAddr("bus2".to_string()),
                bus_ptr.clone(),
            ),
            MirExpr::new(MirExprKind::IntLit(2), MirType::U32),
        ];
        mir.boot.steps.push(MirBootStep::SpawnActivate {
            slot_id: 0,
            args: a1.clone(),
        });
        mir.boot.steps.push(MirBootStep::SpawnActivate {
            slot_id: 1,
            args: a2.clone(),
        });
        mir.spawned_tasks.push(MirSpawnInstance {
            slot_id: 0,
            instance_name: "fieldbus_listener_0".to_string(),
            blueprint_name: "fieldbus_listener".to_string(),
            priority: MirPriority::Normal,
            core_pin: None,
            init_args: a1,
            is_main_supervisor: false,
            provenance_path: vec!["main".to_string(), "fieldbus_listener".to_string()],
            conditional: false,
        });
        mir.spawned_tasks.push(MirSpawnInstance {
            slot_id: 1,
            instance_name: "fieldbus_listener_1".to_string(),
            blueprint_name: "fieldbus_listener".to_string(),
            priority: MirPriority::Normal,
            core_pin: None,
            init_args: a2,
            is_main_supervisor: false,
            provenance_path: vec!["main".to_string(), "fieldbus_listener".to_string()],
            conditional: false,
        });
        mir.task_blueprints.push(MirTaskBlueprint {
            name: "fieldbus_listener".to_string(),
            is_main_supervisor: false,
            manual_yield: false,
            params: vec![VarId(0), VarId(1)],
            vars: vec![
                MirVar {
                    id: VarId(0),
                    name: "bus".to_string(),
                    ty: bus_ptr.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(1),
                    name: "node_id".to_string(),
                    ty: MirType::U32,
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
            ],
            child_slots: vec![],
            blocks: vec![MirBlock {
                id: BlockId(0),
                is_resume_entry: false,
                stmts: vec![
                    MirStmt::Expr(MirExpr::new(MirExprKind::Var(VarId(0)), bus_ptr)),
                    MirStmt::Expr(MirExpr::new(MirExprKind::Var(VarId(1)), MirType::U32)),
                ],
                terminator: MirTerminator::Return(None),
            }],
        });

        fold_singletons(&mut mir);
        mir.visibility = analyze_visibility(&mir);

        assert!(mir.visibility.singleton_subsystems.is_empty());
        assert!(mir.visibility.singleton_subsystem_structs.is_empty());
        assert!(mir.visibility.folded_task_vars.is_empty());
        assert_eq!(mir.visibility.singleton_bytes_saved_32, 0);
        assert!(mir.task_blueprints[0].vars[0].hoisted);
        assert!(mir.task_blueprints[0].vars[1].hoisted);
    }

    #[test]
    fn fold_singletons_negative_conservatism_mutated_addr_cond_indirect_fnptr_and_fnaddr() {
        let mut mir = empty_module();
        let port_ptr = MirType::Ptr(Box::new(MirType::Struct("SerialPort".to_string())));
        let fn_ptr_ty = MirType::FnPtr {
            params: vec![],
            ret: Box::new(MirType::Void),
        };
        mir.structs.push(MirStruct {
            name: "SerialPort".to_string(),
            packed: false,
            is_subsystem: true,
            fields: vec![MirField {
                name: "flag".to_string(),
                ty: MirType::U8,
                dma_safe: false,
            }],
        });
        mir.subsystems.push(MirSubsystemGlobal {
            name: "usb_serial".to_string(),
            struct_name: "SerialPort".to_string(),
            dma_safe: false,
        });
        let subsys_expr = MirExpr::new(
            MirExprKind::SubsystemAddr("usb_serial".to_string()),
            port_ptr.clone(),
        );

        // Callback whose FnAddr is taken: must NOT fold its param.
        mir.functions.push(MirFunction {
            name: "cb_taken".to_string(),
            params: vec![VarId(0)],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![MirVar {
                id: VarId(0),
                name: "p".to_string(),
                ty: port_ptr.clone(),
                role: VarRole::Param,
                in_noyield_scope: false,
                addr_taken: false,
                hoisted: false,
            }],
            body: MirFnBody::Direct(vec![MirStmt::Expr(MirExpr::new(
                MirExprKind::Var(VarId(0)),
                port_ptr.clone(),
            ))]),
        });

        // Varying args: param 0 (`p`) identical across calls -> folds; param 1 (`k`) differs -> does NOT fold.
        mir.functions.push(MirFunction {
            name: "varying_fn".to_string(),
            params: vec![VarId(0), VarId(1)],
            ret_ty: MirType::U32,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![
                MirVar {
                    id: VarId(0),
                    name: "p".to_string(),
                    ty: port_ptr.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: false,
                },
                MirVar {
                    id: VarId(1),
                    name: "k".to_string(),
                    ty: MirType::U32,
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: false,
                },
            ],
            body: MirFnBody::Direct(vec![
                MirStmt::Expr(MirExpr::new(MirExprKind::Var(VarId(0)), port_ptr.clone())),
                MirStmt::Return(Some(MirExpr::new(MirExprKind::Var(VarId(1)), MirType::U32))),
            ]),
        });

        // Target for indirect fnptr dispatch via CallDirect { callee: "fnptr_call_0", is_extern_c: true }.
        mir.functions.push(MirFunction {
            name: "nop_cb".to_string(),
            params: vec![],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![MirStmt::Return(None)]),
        });

        mir.boot.steps.push(MirBootStep::SpawnActivate {
            slot_id: 0,
            args: vec![subsys_expr.clone(), subsys_expr.clone()],
        });
        mir.spawned_tasks.push(MirSpawnInstance {
            slot_id: 0,
            instance_name: "conservative_0".to_string(),
            blueprint_name: "conservative".to_string(),
            priority: MirPriority::Normal,
            core_pin: None,
            init_args: vec![subsys_expr.clone(), subsys_expr.clone()],
            is_main_supervisor: false,
            provenance_path: vec!["main".to_string(), "conservative".to_string()],
            conditional: false,
        });

        mir.task_blueprints.push(MirTaskBlueprint {
            name: "conservative".to_string(),
            is_main_supervisor: false,
            manual_yield: false,
            params: vec![VarId(0), VarId(1)],
            vars: vec![
                MirVar {
                    id: VarId(0),
                    name: "mut_p".to_string(),
                    ty: port_ptr.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(1),
                    name: "addr_p".to_string(),
                    ty: port_ptr.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(2),
                    name: "p_reassigned".to_string(),
                    ty: port_ptr.clone(),
                    role: VarRole::UserLocal,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(3),
                    name: "p_cond".to_string(),
                    ty: port_ptr.clone(),
                    role: VarRole::UserLocal,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(4),
                    name: "fnptr_call_0".to_string(),
                    ty: fn_ptr_ty.clone(),
                    role: VarRole::UserLocal,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
            ],
            child_slots: vec![],
            blocks: vec![
                MirBlock {
                    id: BlockId(0),
                    is_resume_entry: false,
                    stmts: vec![
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(2)),
                            op: MirAssignOp::Assign,
                            value: subsys_expr.clone(),
                        },
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(2)),
                            op: MirAssignOp::Assign,
                            value: subsys_expr.clone(),
                        },
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(4)),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(
                                MirExprKind::FnAddr("nop_cb".to_string()),
                                fn_ptr_ty,
                            ),
                        },
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::CallDirect {
                                callee: "fnptr_call_0".to_string(),
                                is_extern_c: true,
                                args: vec![],
                            },
                            MirType::Void,
                        )),
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::AddrOfPlace(Box::new(MirPlace::Var(VarId(1)))),
                            MirType::Ptr(Box::new(port_ptr.clone())),
                        )),
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::FnAddr("cb_taken".to_string()),
                            MirType::FnPtr {
                                params: vec![port_ptr.clone()],
                                ret: Box::new(MirType::Void),
                            },
                        )),
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::CallDirect {
                                callee: "cb_taken".to_string(),
                                is_extern_c: false,
                                args: vec![subsys_expr.clone()],
                            },
                            MirType::Void,
                        )),
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::CallDirect {
                                callee: "varying_fn".to_string(),
                                is_extern_c: false,
                                args: vec![
                                    subsys_expr.clone(),
                                    MirExpr::new(MirExprKind::IntLit(1), MirType::U32),
                                ],
                            },
                            MirType::U32,
                        )),
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::CallDirect {
                                callee: "varying_fn".to_string(),
                                is_extern_c: false,
                                args: vec![
                                    subsys_expr.clone(),
                                    MirExpr::new(MirExprKind::IntLit(2), MirType::U32),
                                ],
                            },
                            MirType::U32,
                        )),
                    ],
                    terminator: MirTerminator::Branch {
                        cond: MirExpr::new(MirExprKind::BoolLit(true), MirType::Bool),
                        then_bb: BlockId(1),
                        else_bb: BlockId(2),
                    },
                },
                MirBlock {
                    id: BlockId(1),
                    is_resume_entry: false,
                    stmts: vec![
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(0)),
                            op: MirAssignOp::Assign,
                            value: subsys_expr.clone(),
                        },
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(3)),
                            op: MirAssignOp::Assign,
                            value: subsys_expr,
                        },
                    ],
                    terminator: MirTerminator::Return(None),
                },
                MirBlock {
                    id: BlockId(2),
                    is_resume_entry: false,
                    stmts: vec![],
                    terminator: MirTerminator::Return(None),
                },
            ],
        });

        fold_singletons(&mut mir);
        mir.visibility = analyze_visibility(&mir);

        let bp = &mir.task_blueprints[0];
        assert!(bp.vars[0].hoisted, "mut_p must stay hoisted");
        assert!(bp.vars[1].hoisted, "addr_p must stay hoisted");
        assert!(bp.vars[2].hoisted, "p_reassigned must stay hoisted");
        assert!(
            bp.vars[3].hoisted,
            "p_cond in non-entry bb must stay hoisted"
        );
        assert!(
            bp.vars[4].hoisted,
            "indirect fnptr_call_0 slot must NOT be eliminated"
        );
        assert!(matches!(
            bp.blocks[0].stmts[2],
            MirStmt::Assign {
                target: MirPlace::Var(VarId(4)),
                ..
            }
        ));

        // cb_taken still reads Var(0) because its FnAddr was taken.
        match &mir.functions[0].body {
            MirFnBody::Direct(stmts) => assert!(matches!(
                &stmts[0],
                MirStmt::Expr(MirExpr {
                    kind: MirExprKind::Var(VarId(0)),
                    ..
                })
            )),
            _ => panic!("expected Direct"),
        }

        // varying_fn folds p (VarId(0) -> SubsystemAddr) while preserving k (VarId(1)).
        match &mir.functions[1].body {
            MirFnBody::Direct(stmts) => {
                assert!(matches!(
                    &stmts[0],
                    MirStmt::Expr(MirExpr {
                        kind: MirExprKind::SubsystemAddr(name),
                        ..
                    }) if name == "usb_serial"
                ));
                assert!(matches!(
                    &stmts[1],
                    MirStmt::Return(Some(MirExpr {
                        kind: MirExprKind::Var(VarId(1)),
                        ..
                    }))
                ));
            }
            _ => panic!("expected Direct"),
        }
    }

    #[test]
    fn fold_singletons_single_boot_helper_spawn_and_restart_task() {
        let mut mir = empty_module();
        let ring_ty = MirType::Primitive(MirPrimKind::ByteRing { cap: 16 });
        let ctrl_ptr = MirType::Ptr(Box::new(MirType::Struct("Ctrl".to_string())));
        let view_ty = MirType::PrimView(MirPrimViewKind::ByteRing);

        mir.structs.push(MirStruct {
            name: "Ctrl".to_string(),
            packed: false,
            is_subsystem: true,
            fields: vec![MirField {
                name: "rx".to_string(),
                ty: ring_ty.clone(),
                dma_safe: false,
            }],
        });
        mir.subsystems.push(MirSubsystemGlobal {
            name: "ctrl".to_string(),
            struct_name: "Ctrl".to_string(),
            dma_safe: false,
        });

        let subsys_expr = MirExpr::new(
            MirExprKind::SubsystemAddr("ctrl".to_string()),
            ctrl_ptr.clone(),
        );
        // Boot sequence has local `boot_ptr = &ctrl` passed into single-boot-called `init_subsystems`.
        mir.boot.vars.push(MirVar {
            id: VarId(0),
            name: "boot_ptr".to_string(),
            ty: ctrl_ptr.clone(),
            role: VarRole::UserLocal,
            in_noyield_scope: false,
            addr_taken: false,
            hoisted: false,
        });
        mir.boot
            .steps
            .push(MirBootStep::Stmt(Box::new(MirStmt::Assign {
                target: MirPlace::Var(VarId(0)),
                op: MirAssignOp::Assign,
                value: subsys_expr.clone(),
            })));
        mir.boot
            .steps
            .push(MirBootStep::Stmt(Box::new(MirStmt::Expr(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "init_subsystems".to_string(),
                    is_extern_c: false,
                    args: vec![],
                },
                MirType::Void,
            )))));

        mir.functions.push(MirFunction {
            name: "init_subsystems".to_string(),
            params: vec![],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![MirStmt::SpawnActivate {
                slot_id: 0,
                args: vec![
                    subsys_expr.clone(),
                    MirExpr::new(MirExprKind::IntLit(42), MirType::U32),
                ],
            }]),
        });

        mir.spawned_tasks.push(MirSpawnInstance {
            slot_id: 0,
            instance_name: "restart_worker_0".to_string(),
            blueprint_name: "restart_worker".to_string(),
            priority: MirPriority::Normal,
            core_pin: None,
            init_args: vec![
                subsys_expr,
                MirExpr::new(MirExprKind::IntLit(42), MirType::U32),
            ],
            is_main_supervisor: false,
            provenance_path: vec![
                "main".to_string(),
                "init_subsystems".to_string(),
                "restart_worker".to_string(),
            ],
            conditional: false,
        });

        mir.task_blueprints.push(MirTaskBlueprint {
            name: "restart_worker".to_string(),
            is_main_supervisor: false,
            manual_yield: false,
            params: vec![VarId(0), VarId(1)],
            vars: vec![
                MirVar {
                    id: VarId(0),
                    name: "c".to_string(),
                    ty: ctrl_ptr.clone(),
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(1),
                    name: "rate".to_string(),
                    ty: MirType::U32,
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(2),
                    name: "rx".to_string(),
                    ty: view_ty.clone(),
                    role: VarRole::UserLocal,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
                MirVar {
                    id: VarId(3),
                    name: "retries".to_string(),
                    ty: MirType::U32,
                    role: VarRole::UserLocal,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: true,
                },
            ],
            child_slots: vec![],
            blocks: vec![
                MirBlock {
                    id: BlockId(0),
                    is_resume_entry: false,
                    stmts: vec![
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(2)),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(
                                MirExprKind::MakePrimView {
                                    ptr: Box::new(MirExpr::new(
                                        MirExprKind::AddrOfPlace(Box::new(MirPlace::Field {
                                            base: Box::new(MirPlace::Deref(
                                                Box::new(MirExpr::new(
                                                    MirExprKind::Var(VarId(0)),
                                                    ctrl_ptr,
                                                )),
                                                MirType::Struct("Ctrl".to_string()),
                                            )),
                                            struct_name: "Ctrl".to_string(),
                                            packed: false,
                                            field: "rx".to_string(),
                                            field_ty: ring_ty.clone(),
                                        })),
                                        MirType::Ptr(Box::new(ring_ty)),
                                    )),
                                    cap: 16,
                                    kind: MirPrimViewKind::ByteRing,
                                },
                                view_ty.clone(),
                            ),
                        },
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(3)),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(MirExprKind::IntLit(0), MirType::U32),
                        },
                    ],
                    terminator: MirTerminator::Yield(MirYieldKind::ShareCheckpoint {
                        resume_bb: BlockId(1),
                    }),
                },
                MirBlock {
                    id: BlockId(1),
                    is_resume_entry: true,
                    stmts: vec![
                        MirStmt::Assign {
                            target: MirPlace::Var(VarId(3)),
                            op: MirAssignOp::AddAssign,
                            value: MirExpr::new(MirExprKind::Var(VarId(1)), MirType::U32),
                        },
                        MirStmt::Expr(MirExpr::new(
                            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewLen {
                                view: MirExpr::new(MirExprKind::Var(VarId(2)), view_ty),
                            })),
                            MirType::U32,
                        )),
                    ],
                    terminator: MirTerminator::Yield(MirYieldKind::ShareCheckpoint {
                        resume_bb: BlockId(0),
                    }),
                },
            ],
        });

        fold_singletons(&mut mir);
        mir.visibility = analyze_visibility(&mir);

        let bp = &mir.task_blueprints[0];
        assert!(!bp.vars[0].hoisted, "param `c` must fold via boot helper");
        assert!(
            !bp.vars[1].hoisted,
            "param `rate` must fold via boot helper"
        );
        assert!(!bp.vars[2].hoisted, "local `*byte_ring rx` must fold");
        assert!(
            bp.vars[3].hoisted,
            "mutated local `retries` must remain hoisted"
        );
        assert!(matches!(
            bp.blocks[1].terminator,
            MirTerminator::Yield(MirYieldKind::ShareCheckpoint {
                resume_bb: BlockId(0)
            })
        ));
        assert!(matches!(
            &bp.blocks[1].stmts[0],
            MirStmt::Assign {
                value: MirExpr {
                    kind: MirExprKind::IntLit(42),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn b_h1_indirect_fnptr_and_const_prim_init_visibility() {
        let mut mir = empty_module();
        let fn_void_ty = MirType::FnPtr {
            params: vec![MirType::U32],
            ret: Box::new(MirType::Void),
        };
        let fn_other_ty = MirType::FnPtr {
            params: vec![MirType::U8, MirType::U8],
            ret: Box::new(MirType::Void),
        };

        mir.primitives = vec![
            MirPrimitive {
                name: "w_const_init".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: Some(MirExpr::new(MirExprKind::IntLit(1), MirType::U32)),
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "w_cb_contended".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "w_cb_level2".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U16),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "w_task_only".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
            MirPrimitive {
                name: "w_prim_init_ptr".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::Usize),
                },
                dma_safe: false,
                init: Some(MirExpr::new(
                    MirExprKind::Cast {
                        expr: Box::new(MirExpr::new(
                            MirExprKind::GlobalPrimAddr("w_const_init".to_string()),
                            MirType::Ptr(Box::new(MirType::Primitive(MirPrimKind::Watch {
                                elem: Box::new(MirType::U32),
                            }))),
                        )),
                        target_ty: MirType::Usize,
                    },
                    MirType::Usize,
                )),
                sleep_lock_bit: None,
                dead_stripped: false,
                irq_mask_elided: false,
            },
        ];

        // Const takes `&cb_from_const`, boot takes `&cb_level2`, task takes `&cb_other_sig`
        mir.consts.push(MirConst {
            name: "CB_CONST".to_string(),
            ty: fn_void_ty.clone(),
            init: MirExpr::new(
                MirExprKind::FnAddr("cb_from_const".to_string()),
                fn_void_ty.clone(),
            ),
        });
        mir.boot
            .steps
            .push(MirBootStep::Stmt(Box::new(MirStmt::Expr(MirExpr::new(
                MirExprKind::FnAddr("cb_level2".to_string()),
                fn_other_ty.clone(),
            )))));

        mir.functions = vec![
            // `cb_from_const(u32 v)`: touches `w_cb_contended` and invokes `fn(u8, u8) -> void` fnptr!
            MirFunction {
                name: "cb_from_const".to_string(),
                params: vec![VarId(0)],
                ret_ty: MirType::Void,
                error_spec: MirErrorSpec::None,
                manual_yield: false,
                is_yielding: false,
                vars: vec![
                    MirVar {
                        id: VarId(0),
                        name: "v".to_string(),
                        ty: MirType::U32,
                        role: VarRole::Param,
                        in_noyield_scope: false,
                        addr_taken: false,
                        hoisted: false,
                    },
                    MirVar {
                        id: VarId(1),
                        name: "_tmp_fnptr_call_1".to_string(),
                        ty: fn_other_ty.clone(),
                        role: VarRole::CompilerTemp,
                        in_noyield_scope: false,
                        addr_taken: false,
                        hoisted: false,
                    },
                ],
                body: MirFnBody::Direct(vec![
                    MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                        target: MirPrimTarget::Global("w_cb_contended".to_string()),
                        elem_ty: MirType::U32,
                        value: MirExpr::new(MirExprKind::Var(VarId(0)), MirType::U32),
                    }),
                    MirStmt::Expr(MirExpr::new(
                        MirExprKind::CallDirect {
                            callee: "_tmp_fnptr_call_1".to_string(),
                            is_extern_c: true,
                            args: vec![
                                MirExpr::new(MirExprKind::IntLit(1), MirType::U8),
                                MirExpr::new(MirExprKind::IntLit(2), MirType::U8),
                            ],
                        },
                        MirType::Void,
                    )),
                ]),
            },
            // `cb_level2(u8 a, u8 b)`: touches `w_cb_level2`
            MirFunction {
                name: "cb_level2".to_string(),
                params: vec![VarId(0), VarId(1)],
                ret_ty: MirType::Void,
                error_spec: MirErrorSpec::None,
                manual_yield: false,
                is_yielding: false,
                vars: vec![
                    MirVar {
                        id: VarId(0),
                        name: "a".to_string(),
                        ty: MirType::U8,
                        role: VarRole::Param,
                        in_noyield_scope: false,
                        addr_taken: false,
                        hoisted: false,
                    },
                    MirVar {
                        id: VarId(1),
                        name: "b".to_string(),
                        ty: MirType::U8,
                        role: VarRole::Param,
                        in_noyield_scope: false,
                        addr_taken: false,
                        hoisted: false,
                    },
                ],
                body: MirFnBody::Direct(vec![MirStmt::PrimOp(
                    MirPrimNonBlockingOp::WatchPublish {
                        target: MirPrimTarget::Global("w_cb_level2".to_string()),
                        elem_ty: MirType::U16,
                        value: MirExpr::new(MirExprKind::IntLit(9), MirType::U16),
                    },
                )]),
            },
            // `cb_unmatched(u64 x)`: signature does NOT match ISR's fn(u32)->void or fn(u8, u8)->void
            MirFunction {
                name: "cb_unmatched".to_string(),
                params: vec![VarId(0)],
                ret_ty: MirType::Void,
                error_spec: MirErrorSpec::None,
                manual_yield: false,
                is_yielding: false,
                vars: vec![MirVar {
                    id: VarId(0),
                    name: "x".to_string(),
                    ty: MirType::U64,
                    role: VarRole::Param,
                    in_noyield_scope: false,
                    addr_taken: false,
                    hoisted: false,
                }],
                body: MirFnBody::Direct(vec![MirStmt::PrimOp(
                    MirPrimNonBlockingOp::WatchPublish {
                        target: MirPrimTarget::Global("w_task_only".to_string()),
                        elem_ty: MirType::U32,
                        value: MirExpr::new(MirExprKind::IntLit(5), MirType::U32),
                    },
                )]),
            },
        ];

        // Task reads all watches and takes FnAddr of `cb_unmatched`
        mir.task_blueprints.push(MirTaskBlueprint {
            name: "worker".to_string(),
            is_main_supervisor: false,
            manual_yield: false,
            params: vec![],
            vars: vec![],
            child_slots: vec![],
            blocks: vec![MirBlock {
                id: BlockId(0),
                is_resume_entry: false,
                stmts: vec![
                    MirStmt::Expr(MirExpr::new(
                        MirExprKind::FnAddr("cb_unmatched".to_string()),
                        MirType::Usize,
                    )),
                    MirStmt::Expr(MirExpr::new(
                        MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                            target: MirPrimTarget::Global("w_cb_contended".to_string()),
                            elem_ty: MirType::U32,
                        })),
                        MirType::U32,
                    )),
                    MirStmt::Expr(MirExpr::new(
                        MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                            target: MirPrimTarget::Global("w_cb_level2".to_string()),
                            elem_ty: MirType::U16,
                        })),
                        MirType::U16,
                    )),
                    MirStmt::Expr(MirExpr::new(
                        MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                            target: MirPrimTarget::Global("w_task_only".to_string()),
                            elem_ty: MirType::U32,
                        })),
                        MirType::U32,
                    )),
                ],
                terminator: MirTerminator::Return(None),
            }],
        });

        // ISR invokes `fn(u32) -> void` via `_tmp_fnptr_call_0`
        mir.isrs.push(MirIsr {
            name: "irq_cb".to_string(),
            params: vec![],
            state_vars: vec![],
            vars: vec![MirVar {
                id: VarId(0),
                name: "_tmp_fnptr_call_0".to_string(),
                ty: fn_void_ty,
                role: VarRole::CompilerTemp,
                in_noyield_scope: false,
                addr_taken: false,
                hoisted: false,
            }],
            body: vec![MirStmt::Expr(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "_tmp_fnptr_call_0".to_string(),
                    is_extern_c: true,
                    args: vec![MirExpr::new(MirExprKind::IntLit(42), MirType::U32)],
                },
                MirType::Void,
            ))],
        });

        let report = analyze_visibility(&mir);
        assert!(
            report.addr_taken_prims.contains("w_const_init"),
            "primitive init GlobalPrimAddr must be tracked"
        );
        assert!(
            report.isr_contended_globals.contains("w_cb_contended"),
            "ISR FnPtr call must seed matching addr_taken fn `cb_from_const`"
        );
        assert!(
            report.isr_contended_globals.contains("w_cb_level2"),
            "transitive ISR FnPtr call inside `cb_from_const` must seed `cb_level2`"
        );
        assert!(
            !report.isr_contended_globals.contains("w_task_only"),
            "non-matching FnPtr signature `cb_unmatched` must stay task-only"
        );
        assert!(report.elided_globals.contains("w_task_only"));

        // Also test real `extern "C"` call in ISR seeding all non-yielding `addr_taken_fns`
        mir.isrs[0].body = vec![MirStmt::Expr(MirExpr::new(
            MirExprKind::CallDirect {
                callee: "c_irq_dispatch".to_string(),
                is_extern_c: true,
                args: vec![],
            },
            MirType::Void,
        ))];
        let report_ext = analyze_visibility(&mir);
        assert!(report_ext.isr_contended_globals.contains("w_task_only"));
    }

    #[test]
    fn b_l2_uniform_coro_fn_param_clears_hoisted_flag() {
        let mut mir = empty_module();
        mir.functions.push(MirFunction {
            name: "coro_helper".to_string(),
            params: vec![VarId(0)],
            ret_ty: MirType::U32,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: true,
            vars: vec![MirVar {
                id: VarId(0),
                name: "k".to_string(),
                ty: MirType::U32,
                role: VarRole::Param,
                in_noyield_scope: false,
                addr_taken: false,
                hoisted: true,
            }],
            body: MirFnBody::Coroutine {
                blocks: vec![MirBlock {
                    id: BlockId(0),
                    is_resume_entry: true,
                    stmts: vec![],
                    terminator: MirTerminator::Return(Some(MirExpr::new(
                        MirExprKind::Var(VarId(0)),
                        MirType::U32,
                    ))),
                }],
                child_slots: vec![],
            },
        });
        mir.boot
            .boot_yielding_callees
            .push("coro_helper".to_string());
        mir.boot.steps.push(MirBootStep::RunYieldingFn {
            callee: "coro_helper".to_string(),
            args: vec![MirExpr::new(MirExprKind::IntLit(7), MirType::U32)],
            dst: None,
        });

        fold_singletons(&mut mir);
        assert!(
            !mir.functions[0].vars[0].hoisted,
            "uniform folded coroutine fn param must clear hoisted"
        );
    }

    #[test]
    fn prune_unreachable_functions_drops_dead_helpers_and_keeps_transitive_and_fnaddr() {
        let mut mir = empty_module();
        let ret_u32 =
            |v: i128| MirStmt::Return(Some(MirExpr::new(MirExprKind::IntLit(v), MirType::U32)));

        mir.functions.push(MirFunction {
            name: "used_direct".to_string(),
            params: vec![],
            ret_ty: MirType::U32,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![MirStmt::Return(Some(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "used_leaf".to_string(),
                    is_extern_c: false,
                    args: vec![],
                },
                MirType::U32,
            )))]),
        });
        mir.functions.push(MirFunction {
            name: "used_leaf".to_string(),
            params: vec![],
            ret_ty: MirType::U32,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![ret_u32(1)]),
        });
        mir.functions.push(MirFunction {
            name: "putc_hook".to_string(),
            params: vec![],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![MirStmt::Return(None)]),
        });
        mir.functions.push(MirFunction {
            name: "dead_pure".to_string(),
            params: vec![],
            ret_ty: MirType::U32,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![ret_u32(99)]),
        });
        mir.functions.push(MirFunction {
            name: "dead_yielding".to_string(),
            params: vec![],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: true,
            vars: vec![],
            body: MirFnBody::Coroutine {
                blocks: vec![MirBlock {
                    id: BlockId(0),
                    is_resume_entry: true,
                    stmts: vec![],
                    terminator: MirTerminator::Return(None),
                }],
                child_slots: vec![],
            },
        });
        mir.dual_specialized_fns.push("dead_yielding".to_string());
        mir.monomorphized_instances
            .push(crate::ir::MonomorphizedMirItem {
                kind: crate::ir::MonomorphizedItemKind::Fn,
                signature: "dead_pure".to_string(),
                mangled_name: "dead_pure".to_string(),
                size_32: 0,
                size_64: 0,
            });
        mir.boot
            .steps
            .push(MirBootStep::Stmt(Box::new(MirStmt::Intrinsic(
                MirIntrinsicOp::SysOp {
                    method: "set_putc".to_string(),
                    args: vec![MirExpr::new(
                        MirExprKind::FnAddr("putc_hook".to_string()),
                        MirType::Usize,
                    )],
                },
            ))));
        mir.boot
            .steps
            .push(MirBootStep::Stmt(Box::new(MirStmt::Expr(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "used_direct".to_string(),
                    is_extern_c: false,
                    args: vec![],
                },
                MirType::U32,
            )))));

        prune_unreachable_functions(&mut mir);
        crate::ram::populate_module_ram_report(&mut mir);

        let kept: BTreeSet<&str> = mir.functions.iter().map(|f| f.name.as_str()).collect();
        assert!(kept.contains("used_direct"));
        assert!(kept.contains("used_leaf"));
        assert!(kept.contains("putc_hook"));
        assert!(!kept.contains("dead_pure"));
        assert!(!kept.contains("dead_yielding"));
        assert!(
            mir.ram_report.fn_frames.is_empty(),
            "fn_frames={:?}",
            mir.ram_report.fn_frames
        );
        assert!(mir.dual_specialized_fns.is_empty());
        assert!(mir.monomorphized_instances.is_empty());
    }

    #[test]
    fn prune_unreachable_functions_drops_unused_imported_console_globals_on_main_entry() {
        let mut mir = empty_module();
        mir.boot.from_main_entry = true;

        let make_ring = |name: &str| MirPrimitive {
            name: name.to_string(),
            kind: MirPrimKind::ByteRing { cap: 64 },
            dma_safe: false,
            init: None,
            sleep_lock_bit: None,
            dead_stripped: false,
            irq_mask_elided: false,
        };
        mir.primitives.push(make_ring("board__console_rx"));
        mir.primitives.push(make_ring("board__console_tx"));
        mir.primitives.push(MirPrimitive {
            name: "board__used_sig".to_string(),
            kind: MirPrimKind::Signal,
            dma_safe: false,
            init: None,
            sleep_lock_bit: None,
            dead_stripped: false,
            irq_mask_elided: false,
        });
        mir.primitives.push(MirPrimitive {
            name: "app_watch".to_string(),
            kind: MirPrimKind::Watch {
                elem: Box::new(MirType::U32),
            },
            dma_safe: false,
            init: None,
            sleep_lock_bit: None,
            dead_stripped: false,
            irq_mask_elided: false,
        });
        mir.subsystems.push(MirSubsystemGlobal {
            name: "board__unused_sub".to_string(),
            struct_name: "UnusedSub".to_string(),
            dma_safe: false,
        });
        mir.isrs.push(MirIsr {
            name: "board__console_rx_isr".to_string(),
            params: vec![],
            state_vars: vec![],
            vars: vec![],
            body: vec![MirStmt::PrimOp(MirPrimNonBlockingOp::ByteRingClear {
                target: MirPrimTarget::Global("board__console_rx".to_string()),
                cap: 64,
            })],
        });
        mir.functions.push(MirFunction {
            name: "board__console_open".to_string(),
            params: vec![],
            ret_ty: MirType::Void,
            error_spec: MirErrorSpec::None,
            manual_yield: false,
            is_yielding: false,
            vars: vec![],
            body: MirFnBody::Direct(vec![
                MirStmt::Expr(MirExpr::new(
                    MirExprKind::GlobalPrimAddr("board__console_rx".to_string()),
                    MirType::Usize,
                )),
                MirStmt::Expr(MirExpr::new(
                    MirExprKind::GlobalPrimAddr("board__console_tx".to_string()),
                    MirType::Usize,
                )),
                MirStmt::Return(None),
            ]),
        });
        mir.boot
            .steps
            .push(MirBootStep::Stmt(Box::new(MirStmt::PrimOp(
                MirPrimNonBlockingOp::SignalNotify {
                    target: MirPrimTarget::Global("board__used_sig".to_string()),
                    boost: None,
                },
            ))));

        prune_unreachable_functions(&mut mir);

        assert!(mir.isrs.is_empty(), "unattached console ISR must be pruned");
        assert!(
            mir.functions
                .iter()
                .all(|f| f.name != "board__console_open"),
            "unused board__console_open must be pruned"
        );
        let kept_prims: BTreeSet<&str> = mir.primitives.iter().map(|p| p.name.as_str()).collect();
        assert!(!kept_prims.contains("board__console_rx"));
        assert!(!kept_prims.contains("board__console_tx"));
        assert!(kept_prims.contains("board__used_sig"));
        assert!(kept_prims.contains("app_watch"));
        assert!(mir.subsystems.is_empty());
    }
}
