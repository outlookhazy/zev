//! Whole-program visibility analysis and dead-watch optimizer for `zev-mir`.
//!
//! Computes:
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
    MirBakedArg, MirBlock, MirBootStep, MirExpr, MirExprKind, MirFmtChunk, MirFnBody,
    MirFunction, MirIntrinsicOp, MirModule, MirPlace, MirPrimKind, MirPrimNonBlockingOp,
    MirPrimReadOp, MirPrimTarget, MirStmt, MirStruct, MirTerminator, MirType, MirYieldKind,
};

/// Whole-program primitive contention and dead-watch analysis report.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VisibilityReport {
    /// Top-level `watch<T>` primitive names that are never read/waited anywhere.
    pub dead_watches: BTreeSet<String>,
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
    ptr_bytes: usize,
    structs: &BTreeMap<&str, &MirStruct>,
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
                let fl: Vec<(usize, usize)> = st
                    .fields
                    .iter()
                    .map(|f| type_size_align(&f.ty, ptr_bytes, structs, seen_structs))
                    .collect();
                seen_structs.remove(name.as_str());
                struct_size_align(&fl, st.packed)
            } else {
                (4, 4)
            }
        }
        MirType::Slice(_) => (ptr_bytes * 2, ptr_bytes),
        MirType::Array { elem, len } => {
            let (sz, al) = type_size_align(elem, ptr_bytes, structs, seen_structs);
            (sz * len, al)
        }
        MirType::Primitive(pk) => watch_or_prim_size_align(pk, false, ptr_bytes, structs, seen_structs),
        MirType::Result { ok, err } => {
            let mut fields = vec![(1, 1)];
            if let Some(et) = err {
                if !et.is_void() {
                    fields.push(type_size_align(et, ptr_bytes, structs, seen_structs));
                }
            }
            if !ok.is_void() {
                fields.push(type_size_align(ok, ptr_bytes, structs, seen_structs));
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
    seen_structs: &mut BTreeSet<String>,
) -> (usize, usize) {
    let raw = match kind {
        MirPrimKind::Signal => struct_size_align(&[(ptr_bytes, ptr_bytes), (1, 1), (1, 1)], false),
        MirPrimKind::Watch { elem } => {
            let el = type_size_align(elem, ptr_bytes, structs, seen_structs);
            struct_size_align(&[el, (4, 4)], false)
        }
        MirPrimKind::Inbox { elem, cap } => {
            let el = type_size_align(elem, ptr_bytes, structs, seen_structs);
            struct_size_align(
                &[
                    (el.0 * cap, el.1),
                    (*cap, 1),
                    (ptr_bytes, ptr_bytes),
                    (ptr_bytes, ptr_bytes),
                    (2, 2),
                    (2, 2),
                    (2, 2),
                ],
                false,
            )
        }
        MirPrimKind::PingPong { elem, cap } => {
            let el = type_size_align(elem, ptr_bytes, structs, seen_structs);
            struct_size_align(
                &[
                    (el.0 * cap * 2, el.1),
                    (ptr_bytes, ptr_bytes),
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                ],
                false,
            )
        }
        MirPrimKind::Lease { elem, cap } => {
            let el = type_size_align(elem, ptr_bytes, structs, seen_structs);
            struct_size_align(
                &[
                    (el.0 * cap, el.1),
                    (ptr_bytes, ptr_bytes),
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                ],
                false,
            )
        }
        MirPrimKind::ByteRing { cap } => struct_size_align(
            &[
                (*cap, 1),
                (ptr_bytes, ptr_bytes),
                (4, 4),
                (4, 4),
                (4, 4),
                (4, 4),
                (4, 4),
                (4, 4),
                (1, 1),
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
) -> usize {
    let mut seen = BTreeSet::new();
    let kind = MirPrimKind::Watch {
        elem: Box::new(elem.clone()),
    };
    watch_or_prim_size_align(&kind, dma_safe, ptr_bytes, structs, &mut seen).0
}

#[derive(Default)]
struct ScanAcc {
    direct_watch_reads: BTreeSet<String>,
    ptr_watch_read_elems: HashSet<MirType>,
    addr_taken_prims: BTreeSet<String>,
    isr_contended_globals: BTreeSet<String>,
    isr_contended_prim_mangles: BTreeSet<String>,
}

struct ScanCtx<'a> {
    prim_kinds: BTreeMap<&'a str, &'a MirPrimKind>,
    structs: BTreeMap<&'a str, &'a MirStruct>,
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

    fn record_prim_types_in_ty(&mut self, ty: &MirType, seen_structs: &mut BTreeSet<String>) {
        match ty {
            MirType::Primitive(pk) => {
                self.acc.isr_contended_prim_mangles.insert(pk.mangle_name());
            }
            MirType::Ptr(inner) | MirType::Slice(inner) | MirType::Array { elem: inner, .. } => {
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
        match target {
            MirPrimTarget::Global(name) => {
                if in_isr {
                    self.record_isr_prim_global(name, Some(kind));
                }
            }
            MirPrimTarget::PtrExpr(expr) => {
                if in_isr {
                    self.acc.isr_contended_prim_mangles.insert(kind.mangle_name());
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
                if in_isr {
                    self.record_isr_prim_global(name, None);
                }
            }
            MirExprKind::FnAddr(name) => {
                callees.insert(name.clone());
            }
            MirExprKind::Unary { expr: inner, .. }
            | MirExprKind::Cast { expr: inner, .. }
            | MirExprKind::FieldAccess { base: inner, .. }
            | MirExprKind::SliceLen(inner)
            | MirExprKind::FromBytes { slice: inner, .. } => {
                self.visit_expr(inner, in_isr, callees);
            }
            MirExprKind::AddrOfPlace(place) | MirExprKind::ArrayToSlice { array_place: place, .. } => {
                self.visit_place(place, in_isr, callees);
            }
            MirExprKind::Binary { lhs, rhs, .. } | MirExprKind::Index { base: lhs, index: rhs, .. } => {
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
                }
                for a in args {
                    self.visit_expr(a, in_isr, callees);
                }
            }
            MirExprKind::PrimRead(op) => match op.as_ref() {
                MirPrimReadOp::WatchRead { target, elem_ty } => {
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
                MirPrimReadOp::PingPongDmaBuf {
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
                MirPrimReadOp::ByteRingPush { target, cap, src, .. }
                | MirPrimReadOp::ByteRingPop {
                    target, cap, dst: src,
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
            MirPrimNonBlockingOp::LeaseComplete {
                target,
                elem_ty,
                cap,
            }
            | MirPrimNonBlockingOp::LeaseSetActive {
                target,
                elem_ty,
                cap,
                ..
            } => {
                let kind = MirPrimKind::Lease {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                self.visit_prim_target(target, &kind, in_isr, callees);
            }
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
                MirIntrinsicOp::SysOp { args, .. } => {
                    for a in args {
                        self.visit_expr(a, in_isr, callees);
                    }
                }
            },
            MirStmt::EnterCritical
            | MirStmt::ExitCritical
            | MirStmt::EndInboxTurn
            | MirStmt::Break
            | MirStmt::Continue
            | MirStmt::SetIrqSelector { .. } => {}
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
                MirYieldKind::InboxRecv {
                    target,
                    elem_ty,
                    cap,
                    ..
                }
                | MirYieldKind::InboxSend {
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
                MirYieldKind::SignalWait { target, .. } => {
                    self.visit_prim_target(target, &MirPrimKind::Signal, in_isr, callees);
                }
                MirYieldKind::PingPongWaitChunk {
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
                MirYieldKind::LeaseWaitChunk {
                    target,
                    elem_ty,
                    cap,
                    ..
                } => {
                    let kind = MirPrimKind::Lease {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    };
                    self.visit_prim_target(target, &kind, in_isr, callees);
                }
                MirYieldKind::ByteRingWaitRead { target, cap, .. } => {
                    let kind = MirPrimKind::ByteRing { cap: *cap };
                    self.visit_prim_target(target, &kind, in_isr, callees);
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

    fn visit_blocks(
        &mut self,
        blocks: &[MirBlock],
        in_isr: bool,
        callees: &mut BTreeSet<String>,
    ) {
        for blk in blocks {
            for s in &blk.stmts {
                self.visit_stmt(s, in_isr, callees);
            }
            self.visit_terminator(&blk.terminator, in_isr, callees);
        }
    }

    fn visit_function(
        &mut self,
        func: &MirFunction,
        in_isr: bool,
        callees: &mut BTreeSet<String>,
    ) {
        if in_isr {
            let mut seen_structs = BTreeSet::new();
            for v in &func.vars {
                self.record_prim_types_in_ty(&v.ty, &mut seen_structs);
            }
        }
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
    }
}

/// Analyze whole-program ISR primitive contention and dead `watch<T>` primitives across
/// `functions`, `task_blueprints`, `isrs`, `boot`, and `irq_vectors`.
pub fn analyze_visibility(module: &MirModule) -> VisibilityReport {
    let mut ctx = ScanCtx::new(module);
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

    // 2. Scan all task blueprints and boot sequence (non-ISR).
    let mut dummy_callees = BTreeSet::new();
    for bp in &module.task_blueprints {
        ctx.visit_blocks(&bp.blocks, false, &mut dummy_callees);
    }
    for step in &module.boot.steps {
        match step {
            MirBootStep::Stmt(s) => ctx.visit_stmt(s, false, &mut dummy_callees),
            MirBootStep::RunYieldingFn { args, .. } | MirBootStep::SpawnActivate { args, .. } => {
                for a in args {
                    ctx.visit_expr(a, false, &mut dummy_callees);
                }
            }
        }
    }
    for sp in &module.spawned_tasks {
        for a in &sp.init_args {
            ctx.visit_expr(a, false, &mut dummy_callees);
        }
    }

    // 3. Scan `MirIrqVector` `baked_args` & state initializers, plus all ISR bodies + transitive callees.
    let mut isr_queue: VecDeque<String> = VecDeque::new();
    for vec in &module.irq_vectors {
        for b in &vec.bindings {
            for barg in &b.baked_args {
                match barg {
                    MirBakedArg::PrimAddr(name) => {
                        ctx.acc.addr_taken_prims.insert(name.clone());
                        ctx.record_isr_prim_global(name, None);
                    }
                    MirBakedArg::MmioAddr(_) => {}
                    MirBakedArg::Const(expr) => {
                        let mut c = BTreeSet::new();
                        ctx.visit_expr(expr, true, &mut c);
                        for callee in c {
                            isr_queue.push_back(callee);
                        }
                    }
                }
            }
            for sv in &b.state_vars {
                let mut c = BTreeSet::new();
                ctx.visit_expr(&sv.init, true, &mut c);
                for callee in c {
                    isr_queue.push_back(callee);
                }
            }
        }
    }

    for isr in &module.isrs {
        let mut seen_structs = BTreeSet::new();
        for v in &isr.vars {
            ctx.record_prim_types_in_ty(&v.ty, &mut seen_structs);
        }
        let mut isr_callees = BTreeSet::new();
        for sv in &isr.state_vars {
            ctx.visit_expr(&sv.init, true, &mut isr_callees);
        }
        for stmt in &isr.body {
            ctx.visit_stmt(stmt, true, &mut isr_callees);
        }
        for callee in isr_callees {
            isr_queue.push_back(callee);
        }
    }

    let mut visited_isr_fns: BTreeSet<String> = BTreeSet::new();
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
                dead_watch_bytes_saved_32 += watch_byte_size(elem, p.dma_safe, 4, &ctx.structs);
                dead_watch_bytes_saved_64 += watch_byte_size(elem, p.dma_safe, 8, &ctx.structs);
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
        dead_watch_bytes_saved_32,
        dead_watch_bytes_saved_64,
        isr_contended_globals: ctx.acc.isr_contended_globals,
        isr_contended_prim_mangles: ctx.acc.isr_contended_prim_mangles,
        elided_globals,
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
                vars: vec![],
                steps: vec![],
                boot_yielding_callees: vec![],
            },
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
        let expr_int = |v| Expr::new(ExprKind::Literal(Literal::Int(v)), sp);
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
                            StmtKind::Expr(expr_method(expr_id("w"), "publish", vec![expr_id("v")])),
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
                                                vec![expr_call("side_effect_val", vec![expr_int(1)])],
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
                                                vec![expr_call("side_effect_val", vec![expr_int(2)])],
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
                                    init: Some(expr_method(expr_id("live_read_task"), "read", vec![])),
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
            BTreeSet::from([
                "dead_pub_only".to_string(),
                "dead_ptr_pub_only".to_string()
            ])
        );
        // `watch<u32>` = 8 B (32b & 64b); `watch<u64>` = { u64(8,8), u32(4,4) } -> 16 B (32b & 64b)
        assert_eq!(report.dead_watch_bytes_saved_32, 24);
        assert_eq!(report.dead_watch_bytes_saved_64, 24);
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
            },
            MirPrimitive {
                name: "w_b".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U16),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
            },
            MirPrimitive {
                name: "w_other_ty".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
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
            MirBootStep::Stmt(MirStmt::Expr(MirExpr::new(
                MirExprKind::CallDirect {
                    callee: "read_any".to_string(),
                    is_extern_c: false,
                    args: vec![MirExpr::new(
                        MirExprKind::GlobalPrimAddr("w_a".to_string()),
                        ptr_u16_watch.clone(),
                    )],
                },
                MirType::U16,
            ))),
            MirBootStep::Stmt(MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                target: MirPrimTarget::PtrExpr(MirExpr::new(
                    MirExprKind::GlobalPrimAddr("w_b".to_string()),
                    ptr_u16_watch,
                )),
                elem_ty: MirType::U16,
                value: MirExpr::new(MirExprKind::IntLit(20), MirType::U16),
            })),
            MirBootStep::Stmt(MirStmt::PrimOp(MirPrimNonBlockingOp::WatchPublish {
                target: MirPrimTarget::Global("w_other_ty".to_string()),
                elem_ty: MirType::U32,
                value: MirExpr::new(MirExprKind::IntLit(30), MirType::U32),
            })),
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
            },
            MirPrimitive {
                name: "task_only_w".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
            },
            MirPrimitive {
                name: "dead_isr_w".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U32),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
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
            },
            MirPrimitive {
                name: "task_only_sig".to_string(),
                kind: MirPrimKind::Signal,
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
            },
            MirPrimitive {
                name: "isr_ptr_ring".to_string(),
                kind: MirPrimKind::ByteRing { cap: 16 },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
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
            fields: vec![
                MirField {
                    name: "a".to_string(),
                    ty: MirType::Usize,
                },
                MirField {
                    name: "b".to_string(),
                    ty: MirType::U8,
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
        });
        mir.primitives.push(MirPrimitive {
            name: "task_sig".to_string(),
            kind: MirPrimKind::Signal,
            dma_safe: false,
            init: None,
            sleep_lock_bit: None,
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
                    value: MirExpr::new(MirExprKind::IntLit(0), MirType::Struct("Pair".to_string())),
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
            },
            MirPrimitive {
                name: "fmt_live_w".to_string(),
                kind: MirPrimKind::Watch {
                    elem: Box::new(MirType::U16),
                },
                dma_safe: false,
                init: None,
                sleep_lock_bit: None,
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
        assert_eq!(report.dead_watch_bytes_saved_32, 8);
        assert_eq!(report.dead_watch_bytes_saved_64, 8);

        strip_dead_watch_publishes(&mut mir, &report);
        match &mir.functions[0].body {
            MirFnBody::Coroutine { blocks, .. } => {
                assert!(matches!(blocks[0].stmts[0], MirStmt::Expr(_)));
                assert!(matches!(blocks[0].stmts[1], MirStmt::PrintFmt { .. }));
            }
            _ => panic!("expected Coroutine body"),
        }
    }
}
