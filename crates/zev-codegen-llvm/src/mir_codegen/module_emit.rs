use super::*;

pub(super) fn emit_sw_irq_dispatch_ir(
    out: &mut String,
    fn_name: &str,
    comment: &str,
    ctx: &ModuleCtx<'_>,
) {
    let fn_as = ctx.fn_as();
    let call_kw = ctx.call_kw();
    let _ = writeln!(out, "{comment}");
    let _ = writeln!(out, "define void {fn_name}(i32 %irq){fn_as} nounwind {{");
    let _ = writeln!(out, "entry:");
    let mut irqs = BTreeSet::new();
    for vec in &ctx.module.irq_vectors {
        irqs.insert(vec.irq_num);
    }
    let _ = writeln!(out, "  switch i32 %irq, label %done [");
    for &irq in &irqs {
        let _ = writeln!(out, "    i32 {irq}, label %irq_{irq}");
    }
    let _ = writeln!(out, "  ]");
    for &irq in &irqs {
        let _ = writeln!(out, "irq_{irq}:");
        let _ = writeln!(out, "  {call_kw} void @__syn_vec_irq_{irq}()");
        let _ = writeln!(out, "  br label %done");
    }
    let _ = writeln!(out, "done:");
    let _ = writeln!(out, "  ret void");
    let _ = writeln!(out, "}}\n");
}

pub(super) fn emit_cm_irq_dispatch_ir(out: &mut String, ctx: &ModuleCtx<'_>) {
    emit_sw_irq_dispatch_ir(
        out,
        "@__syn_cm_dispatch_irq",
        "; ── ARM Cortex-M Software/Boot IRQ Dispatcher ──",
        ctx,
    );
}

pub(super) fn emit_rv_irq_dispatch_ir(out: &mut String, ctx: &ModuleCtx<'_>) {
    emit_sw_irq_dispatch_ir(
        out,
        "@__syn_rv_dispatch_irq",
        "; ── RISC-V Software/Boot IRQ Dispatcher ──",
        ctx,
    );
}

pub(super) fn emit_avr_irq_dispatch_ir(out: &mut String, ctx: &ModuleCtx<'_>) {
    emit_sw_irq_dispatch_ir(
        out,
        "@__syn_avr_dispatch_irq",
        "; ── AVR Software/Vector IRQ Dispatcher ──",
        ctx,
    );
}

pub(super) fn emit_host_entry_ir(out: &mut String, ctx: &ModuleCtx<'_>) {
    emit_sw_irq_dispatch_ir(
        out,
        "@__syn_host_dispatch_irq",
        "; ── Host Simulation IRQ Dispatcher & Entry Point ──",
        ctx,
    );

    let _ = writeln!(
        out,
        r#"define i32 @main() nounwind {{
entry:
  call void @__syn_boot()
  store i8 0, ptr @__syn_host_in_boot, align 1
  call void @__syn_sched_loop()
  ret i32 0
}}
"#
    );
}

pub(super) fn scan_module_runtime_features(module: &MirModule) -> RuntimeIrFeatures {
    let mut feat = RuntimeIrFeatures {
        uses_i64: false,
        uses_floats: false,
        uses_fmt: false,
        uses_fmt_float_sci: false,
        uses_rv_irqs: false,
        uses_sys_tick: false,
        uses_clock_hook: false,
    };

    fn scan_ty(ty: &MirType, feat: &mut RuntimeIrFeatures) {
        match ty {
            MirType::F32 | MirType::F64 => feat.uses_floats = true,
            MirType::U64 | MirType::I64 | MirType::EndianInt { bits: 64, .. } => {
                feat.uses_i64 = true;
            }
            MirType::Ptr(inner) | MirType::Slice(inner) => scan_ty(inner, feat),
            MirType::FnPtr { params, ret } => {
                for p in params {
                    scan_ty(p, feat);
                }
                scan_ty(ret, feat);
            }
            MirType::Array { elem, .. } => scan_ty(elem, feat),
            MirType::Result { ok, err } => {
                scan_ty(ok, feat);
                if let Some(e) = err {
                    scan_ty(e, feat);
                }
            }
            MirType::Primitive(pk) => match pk {
                MirPrimKind::Signal | MirPrimKind::WaitQueue | MirPrimKind::ByteRing { .. } => {}
                MirPrimKind::Watch { elem }
                | MirPrimKind::Guarded { elem }
                | MirPrimKind::Inbox { elem, .. }
                | MirPrimKind::PingPong { elem, .. }
                | MirPrimKind::Pool { elem, .. }
                | MirPrimKind::DmaSlot { elem } => scan_ty(elem, feat),
            },
            MirType::Lease(elem) => scan_ty(elem, feat),
            MirType::PrimView(MirPrimViewKind::Inbox { elem }) => scan_ty(elem, feat),
            MirType::PrimView(MirPrimViewKind::ByteRing) => {}
            _ => {}
        }
    }

    fn scan_prim_target(t: &MirPrimTarget, feat: &mut RuntimeIrFeatures) {
        match t {
            MirPrimTarget::Global(_) => {}
            MirPrimTarget::PtrExpr(expr) => scan_expr(expr, feat),
        }
    }

    fn scan_place(p: &MirPlace, feat: &mut RuntimeIrFeatures) {
        match p {
            MirPlace::Var(_) => {}
            MirPlace::Deref(expr, ty) => {
                scan_ty(ty, feat);
                scan_expr(expr, feat);
            }
            MirPlace::Field { base, field_ty, .. } => {
                scan_ty(field_ty, feat);
                scan_place(base, feat);
            }
            MirPlace::Index {
                base,
                index,
                elem_ty,
                ..
            } => {
                scan_ty(elem_ty, feat);
                scan_place(base, feat);
                scan_expr(index, feat);
            }
        }
    }

    fn scan_expr(e: &MirExpr, feat: &mut RuntimeIrFeatures) {
        scan_ty(&e.ty, feat);
        match &e.kind {
            MirExprKind::FloatLit(_) => feat.uses_floats = true,
            MirExprKind::Unary { expr: sub, .. } => scan_expr(sub, feat),
            MirExprKind::AddrOfPlace(place) => scan_place(place, feat),
            MirExprKind::Binary { lhs, rhs, .. } => {
                scan_expr(lhs, feat);
                scan_expr(rhs, feat);
            }
            MirExprKind::Cast {
                expr: sub,
                target_ty,
            } => {
                scan_ty(target_ty, feat);
                scan_expr(sub, feat);
            }
            MirExprKind::StructLit { fields, .. } => {
                for (_, fe, ft) in fields {
                    scan_ty(ft, feat);
                    scan_expr(fe, feat);
                }
            }
            MirExprKind::ArrayLit { elems, elem_ty } => {
                scan_ty(elem_ty, feat);
                for el in elems {
                    scan_expr(el, feat);
                }
            }
            MirExprKind::FieldAccess {
                base, raw_field_ty, ..
            } => {
                scan_ty(raw_field_ty, feat);
                scan_expr(base, feat);
            }
            MirExprKind::SliceLen(sub) | MirExprKind::FromBytes { slice: sub, .. } => {
                scan_expr(sub, feat);
            }
            MirExprKind::Index { base, index, .. } => {
                scan_expr(base, feat);
                scan_expr(index, feat);
            }
            MirExprKind::SubSlice {
                base,
                start,
                end,
                elem_ty,
                ..
            } => {
                scan_ty(elem_ty, feat);
                scan_expr(base, feat);
                scan_expr(start, feat);
                scan_expr(end, feat);
            }
            MirExprKind::ArrayToSlice {
                array_place,
                elem_ty,
                ..
            } => {
                scan_ty(elem_ty, feat);
                scan_place(array_place, feat);
            }
            MirExprKind::CallDirect { args, .. } => {
                for a in args {
                    scan_expr(a, feat);
                }
            }
            MirExprKind::RomCallU32 { addr, args, .. } => {
                scan_expr(addr, feat);
                for a in args {
                    scan_expr(a, feat);
                }
            }
            MirExprKind::PrimRead(prop) => match &**prop {
                MirPrimReadOp::WatchRead { target, elem_ty }
                | MirPrimReadOp::WatchGen { target, elem_ty }
                | MirPrimReadOp::GuardedGet { target, elem_ty }
                | MirPrimReadOp::InboxTryRecv { target, elem_ty, .. }
                | MirPrimReadOp::InboxLen { target, elem_ty, .. }
                | MirPrimReadOp::InboxIsEmpty { target, elem_ty, .. }
                | MirPrimReadOp::InboxIsFull { target, elem_ty, .. } => {
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                }
                MirPrimReadOp::InboxTrySend {
                    target,
                    elem_ty,
                    value,
                    ..
                } => {
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                    scan_expr(value, feat);
                }
                MirPrimReadOp::SignalIsSet { target }
                | MirPrimReadOp::SignalTryWait { target }
                | MirPrimReadOp::WaitQueueWakeOne { target, .. } => {
                    scan_prim_target(target, feat);
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
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                    let _ = cap;
                }
                MirPrimReadOp::PoolTryAlloc {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::PoolAvail {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::PoolInUse {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::PoolIsExhausted {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::PoolCap {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::DmaSlotTake { target, elem_ty }
                | MirPrimReadOp::DmaSlotRelease {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::DmaSlotIsBusy { target, elem_ty }
                | MirPrimReadOp::DmaSlotIsEmpty { target, elem_ty }
                | MirPrimReadOp::DmaSlotAddr { target, elem_ty } => {
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                }
                MirPrimReadOp::DmaSlotPark {
                    target,
                    elem_ty,
                    lease,
                } => {
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                    scan_expr(lease, feat);
                }
                MirPrimReadOp::ByteRingPush {
                    target, src: val, ..
                }
                | MirPrimReadOp::ByteRingPop {
                    target, dst: val, ..
                }
                | MirPrimReadOp::ByteRingPushByte {
                    target, byte: val, ..
                }
                | MirPrimReadOp::ByteRingGrantWrite {
                    target, len: val, ..
                } => {
                    scan_prim_target(target, feat);
                    scan_expr(val, feat);
                }
                MirPrimReadOp::ByteRingReadSlice { target, .. }
                | MirPrimReadOp::ByteRingTryPopByte { target, .. }
                | MirPrimReadOp::ByteRingLen { target, .. }
                | MirPrimReadOp::ByteRingFreeSpace { target, .. }
                | MirPrimReadOp::ByteRingIsEmpty { target, .. }
                | MirPrimReadOp::ByteRingIsFull { target, .. } => {
                    scan_prim_target(target, feat);
                }
                MirPrimReadOp::InboxViewTryRecv { view, elem_ty }
                | MirPrimReadOp::InboxViewLen { view, elem_ty }
                | MirPrimReadOp::InboxViewCap { view, elem_ty }
                | MirPrimReadOp::InboxViewIsEmpty { view, elem_ty }
                | MirPrimReadOp::InboxViewIsFull { view, elem_ty } => {
                    scan_expr(view, feat);
                    scan_ty(elem_ty, feat);
                }
                MirPrimReadOp::InboxViewTrySend {
                    view,
                    elem_ty,
                    value,
                    ..
                } => {
                    scan_expr(view, feat);
                    scan_ty(elem_ty, feat);
                    scan_expr(value, feat);
                }
                MirPrimReadOp::ByteRingViewPush { view, src: val, .. }
                | MirPrimReadOp::ByteRingViewPop { view, dst: val }
                | MirPrimReadOp::ByteRingViewPushByte { view, byte: val, .. }
                | MirPrimReadOp::ByteRingViewGrantWrite { view, len: val } => {
                    scan_expr(view, feat);
                    scan_expr(val, feat);
                }
                MirPrimReadOp::ByteRingViewReadSlice { view }
                | MirPrimReadOp::ByteRingViewTryPopByte { view }
                | MirPrimReadOp::ByteRingViewLen { view }
                | MirPrimReadOp::ByteRingViewCap { view }
                | MirPrimReadOp::ByteRingViewFreeSpace { view }
                | MirPrimReadOp::ByteRingViewIsEmpty { view }
                | MirPrimReadOp::ByteRingViewIsFull { view } => {
                    scan_expr(view, feat);
                }
            },
            MirExprKind::MakePrimView { ptr, kind, .. } => {
                scan_expr(ptr, feat);
                if let MirPrimViewKind::Inbox { elem } = kind {
                    scan_ty(elem, feat);
                }
            }
            MirExprKind::MmioRead { base, index, .. } => {
                scan_expr(base, feat);
                if let Some(idx) = index {
                    scan_expr(idx, feat);
                }
            }
            MirExprKind::LeaseGet { lease, elem_ty } | MirExprKind::LeaseAddr { lease, elem_ty } => {
                scan_ty(elem_ty, feat);
                scan_expr(lease, feat);
            }
            _ => {}
        }
    }

    fn scan_fmt_chunks(chunks: &[MirFmtChunk], feat: &mut RuntimeIrFeatures) {
        feat.uses_fmt = true;
        for ch in chunks {
            match ch {
                MirFmtChunk::Literal(_) => {}
                MirFmtChunk::Int { value, bits, .. } => {
                    if *bits == 64 {
                        feat.uses_i64 = true;
                    }
                    scan_expr(value, feat);
                }
                MirFmtChunk::Float { value, fmt, .. } => {
                    feat.uses_floats = true;
                    if matches!(fmt, MirFloatFmt::Sci { .. }) {
                        feat.uses_fmt_float_sci = true;
                    }
                    scan_expr(value, feat);
                }
                MirFmtChunk::Bool(v) | MirFmtChunk::Str(v) | MirFmtChunk::Bytes(v) => {
                    scan_expr(v, feat);
                }
                MirFmtChunk::EnumName { value, .. } => scan_expr(value, feat),
                MirFmtChunk::DebugSlice { elem_ty, slice } => {
                    scan_ty(elem_ty, feat);
                    scan_expr(slice, feat);
                }
            }
        }
    }

    fn scan_stmts(stmts: &[MirStmt], feat: &mut RuntimeIrFeatures) {
        for s in stmts {
            match s {
                MirStmt::Assign { target, value, .. } => {
                    scan_place(target, feat);
                    scan_expr(value, feat);
                }
                MirStmt::Expr(value) => scan_expr(value, feat),
                MirStmt::PrimOp(pop) => match pop {
                    MirPrimNonBlockingOp::WatchPublish {
                        target,
                        elem_ty,
                        value,
                    }
                    | MirPrimNonBlockingOp::GuardedSet {
                        target,
                        elem_ty,
                        value,
                    }
                    | MirPrimNonBlockingOp::InboxPost {
                        target,
                        elem_ty,
                        value,
                        ..
                    } => {
                        scan_prim_target(target, feat);
                        scan_ty(elem_ty, feat);
                        scan_expr(value, feat);
                    }
                    MirPrimNonBlockingOp::SignalNotify { target, .. }
                    | MirPrimNonBlockingOp::SignalClear { target }
                    | MirPrimNonBlockingOp::WaitQueuePrepare { target }
                    | MirPrimNonBlockingOp::WaitQueueCancel { target }
                    | MirPrimNonBlockingOp::WaitQueueWakeAll { target, .. }
                    | MirPrimNonBlockingOp::ByteRingClear { target, .. } => {
                        scan_prim_target(target, feat);
                    }
                    MirPrimNonBlockingOp::ByteRingConsume { target, len, .. }
                    | MirPrimNonBlockingOp::ByteRingCommitWrite { target, len, .. } => {
                        scan_prim_target(target, feat);
                        scan_expr(len, feat);
                    }
                    MirPrimNonBlockingOp::PingPongFlip {
                        target, elem_ty, ..
                    }
                    | MirPrimNonBlockingOp::PingPongSetActive {
                        target, elem_ty, ..
                    } => {
                        scan_prim_target(target, feat);
                        scan_ty(elem_ty, feat);
                    }
                    MirPrimNonBlockingOp::PingPongCompleteHalf {
                        target,
                        elem_ty,
                        half,
                        ..
                    } => {
                        scan_prim_target(target, feat);
                        scan_ty(elem_ty, feat);
                        scan_expr(half, feat);
                    }
                    MirPrimNonBlockingOp::LeaseRelease {
                        value, elem_ty, ..
                    } => {
                        scan_ty(elem_ty, feat);
                        scan_expr(value, feat);
                    }
                    MirPrimNonBlockingOp::LeaseNullify { target, elem_ty } => {
                        scan_ty(elem_ty, feat);
                        scan_place(target, feat);
                    }
                    MirPrimNonBlockingOp::InboxViewPost {
                        view,
                        elem_ty,
                        value,
                        ..
                    } => {
                        scan_expr(view, feat);
                        scan_ty(elem_ty, feat);
                        scan_expr(value, feat);
                    }
                    MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
                    | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                        scan_expr(view, feat);
                        scan_expr(len, feat);
                    }
                    MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                        scan_expr(view, feat);
                    }
                },
                MirStmt::Intrinsic(iop) => match iop {
                    MirIntrinsicOp::UseSystick { hz } => scan_expr(hz, feat),
                    MirIntrinsicOp::SysOp { method, args } => {
                        if matches!(method.as_str(), "tick" | "tick_ms" | "tick_by") {
                            feat.uses_sys_tick = true;
                        }
                        for a in args {
                            scan_expr(a, feat);
                        }
                    }
                    MirIntrinsicOp::SetPriority { .. } => {}
                },
                MirStmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    scan_expr(cond, feat);
                    scan_stmts(then_body, feat);
                    scan_stmts(else_body, feat);
                }
                MirStmt::While { cond, body } => {
                    scan_expr(cond, feat);
                    scan_stmts(body, feat);
                }
                MirStmt::Switch {
                    discr,
                    arms,
                    default,
                } => {
                    scan_expr(discr, feat);
                    for (_, ab) in arms {
                        scan_stmts(ab, feat);
                    }
                    scan_stmts(default, feat);
                }
                MirStmt::Return(Some(e)) => scan_expr(e, feat),
                MirStmt::Error { reason, partial } => {
                    if let Some(r) = reason {
                        scan_expr(r, feat);
                    }
                    if let Some(p) = partial {
                        scan_expr(p, feat);
                    }
                }
                MirStmt::SpawnActivate { args, .. } => {
                    for a in args {
                        scan_expr(a, feat);
                    }
                }
                MirStmt::PrintFmt { chunks, .. } => scan_fmt_chunks(chunks, feat),
                MirStmt::FmtBuf { buf, chunks, .. } => {
                    scan_expr(buf, feat);
                    scan_fmt_chunks(chunks, feat);
                }
                MirStmt::FmtRing { ring, chunks, .. } => {
                    scan_expr(ring, feat);
                    scan_fmt_chunks(chunks, feat);
                }
                MirStmt::MmioWrite {
                    base, index, value, ..
                } => {
                    scan_expr(base, feat);
                    if let Some(idx) = index {
                        scan_expr(idx, feat);
                    }
                    scan_expr(value, feat);
                }
                _ => {}
            }
        }
    }

    fn scan_blocks(blocks: &[MirBlock], feat: &mut RuntimeIrFeatures) {
        for b in blocks {
            scan_stmts(&b.stmts, feat);
            match &b.terminator {
                MirTerminator::Branch { cond, .. } => scan_expr(cond, feat),
                MirTerminator::Switch { discr, .. } => scan_expr(discr, feat),
                MirTerminator::Return(Some(e)) => scan_expr(e, feat),
                MirTerminator::Error { reason, partial } => {
                    if let Some(r) = reason {
                        scan_expr(r, feat);
                    }
                    if let Some(p) = partial {
                        scan_expr(p, feat);
                    }
                }
                MirTerminator::Yield(yk) => match yk {
                    MirYieldKind::CallYielding { args, .. } => {
                        for a in args {
                            scan_expr(a, feat);
                        }
                    }
                    MirYieldKind::Wait { arms, .. } => {
                        for arm in arms {
                            match &arm.op {
                                MirWaitArmOp::InboxRecv {
                                    target, elem_ty, ..
                                }
                                | MirWaitArmOp::InboxSend {
                                    target, elem_ty, ..
                                }
                                | MirWaitArmOp::WatchWaitChange {
                                    target, elem_ty, ..
                                }
                                | MirWaitArmOp::PingPongWaitChunk {
                                    target, elem_ty, ..
                                }
                                | MirWaitArmOp::PoolAlloc {
                                    target, elem_ty, ..
                                } => {
                                    scan_prim_target(target, feat);
                                    scan_ty(elem_ty, feat);
                                }
                                MirWaitArmOp::SignalWait { target, .. }
                                | MirWaitArmOp::WaitQueueCommit { target, .. }
                                | MirWaitArmOp::ByteRingWaitRead { target, .. } => {
                                    scan_prim_target(target, feat);
                                }
                                MirWaitArmOp::InboxViewRecv { view, elem_ty, .. }
                                | MirWaitArmOp::InboxViewSend { view, elem_ty, .. } => {
                                    scan_expr(view, feat);
                                    scan_ty(elem_ty, feat);
                                }
                                MirWaitArmOp::ByteRingViewWaitRead { view, .. } => {
                                    scan_expr(view, feat);
                                }
                            }
                        }
                    }
                    MirYieldKind::EveryLatch { .. }
                    | MirYieldKind::DelayPoll { .. }
                    | MirYieldKind::ShareCheckpoint { .. }
                    | MirYieldKind::PollChild { .. } => {}
                },
                MirTerminator::Goto(_) | MirTerminator::Return(None) => {}
            }
        }
    }

    for c in &module.consts {
        scan_expr(&c.init, &mut feat);
    }
    for s in &module.structs {
        for f in &s.fields {
            scan_ty(&f.ty, &mut feat);
        }
    }
    for p in &module.primitives {
        scan_ty(&MirType::Primitive(p.kind.clone()), &mut feat);
        if let Some(init) = &p.init {
            scan_expr(init, &mut feat);
        }
    }
    for ext in &module.extern_fns {
        scan_ty(&ext.ret_ty, &mut feat);
        for (_, pt) in &ext.params {
            scan_ty(pt, &mut feat);
        }
    }
    for isr in &module.isrs {
        for v in &isr.vars {
            scan_ty(&v.ty, &mut feat);
        }
        for sv in &isr.state_vars {
            scan_ty(&sv.ty, &mut feat);
            scan_expr(&sv.init, &mut feat);
        }
        scan_stmts(&isr.body, &mut feat);
    }
    for vec in &module.irq_vectors {
        for b in &vec.bindings {
            for barg in &b.baked_args {
                if let MirBakedArg::Const(e) = barg {
                    scan_expr(e, &mut feat);
                }
            }
            for sv in &b.state_vars {
                scan_ty(&sv.ty, &mut feat);
                scan_expr(&sv.init, &mut feat);
            }
        }
    }
    for func in &module.functions {
        scan_ty(&func.ret_ty, &mut feat);
        for v in &func.vars {
            scan_ty(&v.ty, &mut feat);
        }
        match &func.body {
            MirFnBody::Direct(stmts) => scan_stmts(stmts, &mut feat),
            MirFnBody::Coroutine { blocks, .. } => scan_blocks(blocks, &mut feat),
        }
    }
    for bp in &module.task_blueprints {
        for v in &bp.vars {
            scan_ty(&v.ty, &mut feat);
        }
        scan_blocks(&bp.blocks, &mut feat);
    }
    for v in &module.boot.vars {
        scan_ty(&v.ty, &mut feat);
    }
    for step in &module.boot.steps {
        match step {
            MirBootStep::Stmt(s) => scan_stmts(std::slice::from_ref(s), &mut feat),
            MirBootStep::RunYieldingFn { args, .. } | MirBootStep::SpawnActivate { args, .. } => {
                for a in args {
                    scan_expr(a, &mut feat);
                }
            }
        }
    }
    feat
}

pub(super) fn is_rodata_const_ty(ty: &MirType) -> bool {
    matches!(
        ty,
        MirType::Struct(_) | MirType::Array { .. } | MirType::Slice(_) | MirType::Str
    )
}

impl<'a> ModuleCtx<'a> {
    pub(super) fn prepare(&mut self) -> Result<(), CodegenError> {
        // Enums
        for en in &self.module.enums {
            for (vname, disc) in &en.variants {
                self.enum_variants
                    .insert((en.name.clone(), vname.clone()), *disc);
                self.bare_variants.entry(vname.clone()).or_insert(*disc);
            }
        }

        // Consts (scalars/floats inlined; aggregate/slice/str constants live in .rodata)
        for c in &self.module.consts {
            if is_rodata_const_ty(&c.ty) {
                continue;
            }
            if c.init.ty.is_float() {
                let bits = self.eval_const_float_bits(&c.init);
                self.float_consts.insert(c.name.clone(), bits);
            } else {
                let v = self.eval_const_int(&c.init);
                self.consts.insert(c.name.clone(), v);
            }
        }

        // Structs in topological order
        let mut visited = BTreeSet::new();
        let by_name: HashMap<&str, &MirStruct> = self
            .module
            .structs
            .iter()
            .map(|s| (s.name.as_str(), s))
            .collect();
        for s in &self.module.structs {
            self.visit_struct(s.name.as_str(), &by_name, &mut visited);
        }

        for c in &self.module.consts {
            if is_rodata_const_ty(&c.ty) {
                self.rodata_consts.insert(
                    c.name.clone(),
                    (format!("@__zev_const_{}", sanitize_ident(&c.name)), c.ty.clone()),
                );
            }
        }

        // MMIO layouts & peripheral instances
        for layout in &self.module.mmio_layouts {
            self.mmio_layouts.insert(layout.name.clone(), layout.clone());
        }
        for inst in &self.module.mmio_instances {
            self.mmio_instances.insert(inst.name.clone(), inst.clone());
        }

        // Collect all distinct OS primitive types across structs, globals, ISRs, functions, tasks, and boot
        let structs = self.module.structs.clone();
        for st in &structs {
            for f in &st.fields {
                self.scan_mir_type_for_prims(&f.ty);
            }
        }
        for p in &self.module.primitives {
            if p.dead_stripped {
                continue;
            }
            self.prim_globals.insert(p.name.clone(), p.kind.clone());
            self.register_prim_kind(&p.kind);
        }
        for c in &self.module.consts {
            self.scan_expr_for_prims(&c.init);
        }
        for isr in &self.module.isrs {
            for v in &isr.vars {
                self.scan_mir_type_for_prims(&v.ty);
            }
            for sv in &isr.state_vars {
                self.scan_mir_type_for_prims(&sv.ty);
            }
            for s in &isr.body {
                self.scan_stmt_for_prims(s);
            }
        }
        for vec in &self.module.irq_vectors {
            for b in &vec.bindings {
                for ba in &b.baked_args {
                    if let MirBakedArg::Const(e) = ba {
                        self.scan_expr_for_prims(e);
                    }
                }
                for sv in &b.state_vars {
                    self.scan_mir_type_for_prims(&sv.ty);
                    self.scan_expr_for_prims(&sv.init);
                }
            }
        }
        for f in &self.module.functions {
            for v in &f.vars {
                self.scan_mir_type_for_prims(&v.ty);
            }
            self.scan_mir_type_for_prims(&f.ret_ty);
            match &f.body {
                MirFnBody::Direct(stmts) => {
                    for s in stmts {
                        self.scan_stmt_for_prims(s);
                    }
                }
                MirFnBody::Coroutine { blocks, .. } => self.scan_blocks_for_prims(blocks),
            }
        }
        for bp in &self.module.task_blueprints {
            for v in &bp.vars {
                self.scan_mir_type_for_prims(&v.ty);
            }
            self.scan_blocks_for_prims(&bp.blocks);
        }
        for ext in &self.module.extern_fns {
            for (_, pty) in &ext.params {
                self.scan_mir_type_for_prims(pty);
            }
            self.scan_mir_type_for_prims(&ext.ret_ty);
        }
        for v in &self.module.boot.vars {
            self.scan_mir_type_for_prims(&v.ty);
        }
        for step in &self.module.boot.steps {
            match step {
                MirBootStep::Stmt(s) => self.scan_stmt_for_prims(s),
                MirBootStep::RunYieldingFn { args, .. } | MirBootStep::SpawnActivate { args, .. } => {
                    for a in args {
                        self.scan_expr_for_prims(a);
                    }
                }
            }
        }

        // Function signatures
        for f in &self.module.functions {
            let params: Vec<MirType> = f.params.iter().map(|vid| f.vars[vid.0].ty.clone()).collect();
            let env_mir_ty = f.envelope_return_type();
            let ret_llvm_ty = if f.is_yielding {
                "i32".to_string()
            } else if env_mir_ty.is_void() {
                "void".to_string()
            } else {
                self.llvm_ty(&env_mir_ty)
            };
            let symbol = if f.is_yielding {
                format!("@__syn_fn_step_{}", sanitize_ident(&f.name))
            } else {
                format!("@__syn_fn_{}", sanitize_ident(&f.name))
            };
            self.fn_sigs.insert(
                f.name.clone(),
                FnLlvmSig {
                    symbol,
                    params,
                    ret_llvm_ty,
                },
            );
        }

        // Extern function signatures
        for ext in &self.module.extern_fns {
            self.extern_sigs.insert(
                ext.name.clone(),
                ExternLlvmSig {
                    name: ext.name.clone(),
                    params: ext.params.iter().map(|(_, t)| t.clone()).collect(),
                    ret_ty: ext.ret_ty.clone(),
                    variadic: ext.variadic,
                },
            );
        }

        // Yielding function frames in post-order (children before parents!)
        let fn_map: HashMap<&str, &MirFunction> = self
            .module
            .functions
            .iter()
            .filter(|f| f.is_yielding)
            .map(|f| (f.name.as_str(), f))
            .collect();
        let mut done_fns = BTreeSet::new();
        for f in &self.module.functions {
            if f.is_yielding {
                self.compute_fn_frame_layout(f.name.as_str(), &fn_map, &mut done_fns);
            }
        }

        // Task blueprint frames
        for bp in &self.module.task_blueprints {
            let layout = self.build_coroutine_layout(
                &format!("%TaskFrame_{}", sanitize_ident(&bp.name)),
                false,
                &MirType::Void,
                &bp.vars,
                &bp.child_slots,
            );
            self.bp_frames.insert(bp.name.clone(), layout);
        }

        Ok(())
    }

    pub(super) fn resolve_sys_hooks(&self) -> ResolvedSysHooks {
        fn unwrap_fn_addr(mut e: &MirExpr) -> Option<&str> {
            while let MirExprKind::Cast { expr, .. } = &e.kind {
                e = expr;
            }
            if let MirExprKind::FnAddr(name) = &e.kind {
                Some(name.as_str())
            } else {
                None
            }
        }

        fn record_hook_call<'a>(
            method: &str,
            args: &'a [MirExpr],
            sites: &mut BTreeMap<HookKind, Vec<Option<&'a str>>>,
        ) {
            match method {
                "set_putc" => sites
                    .entry(HookKind::Putc)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_clock_ms" => sites
                    .entry(HookKind::ClockMs)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_exit" => sites
                    .entry(HookKind::Exit)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_fault_handler" => sites
                    .entry(HookKind::FaultHandler)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_idle" => sites
                    .entry(HookKind::Idle)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_irq_router" => {
                    sites
                        .entry(HookKind::IrqRoute)
                        .or_default()
                        .push(args.first().and_then(unwrap_fn_addr));
                    sites
                        .entry(HookKind::IrqAck)
                        .or_default()
                        .push(args.get(1).and_then(unwrap_fn_addr));
                }
                _ => {}
            }
        }

        fn collect_stmts<'a>(
            stmts: &'a [MirStmt],
            sites: &mut BTreeMap<HookKind, Vec<Option<&'a str>>>,
        ) {
            for s in stmts {
                match s {
                    MirStmt::Intrinsic(MirIntrinsicOp::SysOp { method, args }) => {
                        record_hook_call(method, args, sites);
                    }
                    MirStmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        collect_stmts(then_body, sites);
                        collect_stmts(else_body, sites);
                    }
                    MirStmt::While { body, .. } => collect_stmts(body, sites),
                    MirStmt::Switch { arms, default, .. } => {
                        for (_, b) in arms {
                            collect_stmts(b, sites);
                        }
                        collect_stmts(default, sites);
                    }
                    _ => {}
                }
            }
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum WalkStop {
            Continue,
            FnReturns,
            ProgramExits,
        }

        fn stmts_may_return_or_exit(stmts: &[MirStmt]) -> bool {
            for s in stmts {
                match s {
                    MirStmt::Return(_) | MirStmt::Error { .. } => return true,
                    MirStmt::Intrinsic(MirIntrinsicOp::SysOp { method, .. })
                        if matches!(method.as_str(), "exit" | "stop" | "rom_call_noreturn") =>
                    {
                        return true;
                    }
                    MirStmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        if stmts_may_return_or_exit(then_body)
                            || stmts_may_return_or_exit(else_body)
                        {
                            return true;
                        }
                    }
                    MirStmt::While { body, .. } => {
                        if stmts_may_return_or_exit(body) {
                            return true;
                        }
                    }
                    MirStmt::Switch { arms, default, .. } => {
                        if arms.iter().any(|(_, b)| stmts_may_return_or_exit(b))
                            || stmts_may_return_or_exit(default)
                        {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        }

        fn walk_expr_uncond(
            expr: &MirExpr,
            fn_map: &HashMap<&str, &MirFnBody>,
            active: &mut BTreeSet<String>,
            uncond: &mut BTreeSet<HookKind>,
        ) -> WalkStop {
            match &expr.kind {
                MirExprKind::CallDirect {
                    callee,
                    is_extern_c: false,
                    args,
                } => {
                    for a in args {
                        if walk_expr_uncond(a, fn_map, active, uncond) == WalkStop::ProgramExits {
                            return WalkStop::ProgramExits;
                        }
                    }
                    if !active.contains(callee.as_str()) {
                        if let Some(MirFnBody::Direct(body)) = fn_map.get(callee.as_str()) {
                            active.insert(callee.clone());
                            let st = walk_stmts_uncond(body, fn_map, active, uncond);
                            active.remove(callee.as_str());
                            if st == WalkStop::ProgramExits {
                                return WalkStop::ProgramExits;
                            }
                        }
                    }
                    WalkStop::Continue
                }
                MirExprKind::Unary { expr: sub, .. } | MirExprKind::Cast { expr: sub, .. } => {
                    walk_expr_uncond(sub, fn_map, active, uncond)
                }
                _ => WalkStop::Continue,
            }
        }

        fn walk_stmt_uncond(
            stmt: &MirStmt,
            fn_map: &HashMap<&str, &MirFnBody>,
            active: &mut BTreeSet<String>,
            uncond: &mut BTreeSet<HookKind>,
        ) -> WalkStop {
            match stmt {
                MirStmt::Intrinsic(MirIntrinsicOp::SysOp { method, .. }) => {
                    match method.as_str() {
                        "set_putc" => {
                            uncond.insert(HookKind::Putc);
                        }
                        "set_clock_ms" => {
                            uncond.insert(HookKind::ClockMs);
                        }
                        "set_exit" => {
                            uncond.insert(HookKind::Exit);
                        }
                        "set_fault_handler" => {
                            uncond.insert(HookKind::FaultHandler);
                        }
                        "set_idle" => {
                            uncond.insert(HookKind::Idle);
                        }
                        "set_irq_router" => {
                            uncond.insert(HookKind::IrqRoute);
                            uncond.insert(HookKind::IrqAck);
                        }
                        "exit" | "stop" | "rom_call_noreturn" => return WalkStop::ProgramExits,
                        _ => {}
                    }
                    WalkStop::Continue
                }
                MirStmt::Return(_) | MirStmt::Error { .. } => WalkStop::FnReturns,
                MirStmt::Expr(e) | MirStmt::Assign { value: e, .. } => {
                    walk_expr_uncond(e, fn_map, active, uncond)
                }
                MirStmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    if stmts_may_return_or_exit(then_body) || stmts_may_return_or_exit(else_body) {
                        WalkStop::FnReturns
                    } else {
                        WalkStop::Continue
                    }
                }
                MirStmt::While { body, .. } => {
                    if stmts_may_return_or_exit(body) {
                        WalkStop::FnReturns
                    } else {
                        WalkStop::Continue
                    }
                }
                MirStmt::Switch { arms, default, .. } => {
                    if arms.iter().any(|(_, b)| stmts_may_return_or_exit(b))
                        || stmts_may_return_or_exit(default)
                    {
                        WalkStop::FnReturns
                    } else {
                        WalkStop::Continue
                    }
                }
                _ => WalkStop::Continue,
            }
        }

        fn walk_stmts_uncond(
            stmts: &[MirStmt],
            fn_map: &HashMap<&str, &MirFnBody>,
            active: &mut BTreeSet<String>,
            uncond: &mut BTreeSet<HookKind>,
        ) -> WalkStop {
            for s in stmts {
                let st = walk_stmt_uncond(s, fn_map, active, uncond);
                if st != WalkStop::Continue {
                    return st;
                }
            }
            WalkStop::Continue
        }

        let mut sites: BTreeMap<HookKind, Vec<Option<&str>>> = BTreeMap::new();
        for step in &self.module.boot.steps {
            if let MirBootStep::Stmt(s) = step {
                collect_stmts(std::slice::from_ref(s), &mut sites);
            }
        }
        let mut fn_map: HashMap<&str, &MirFnBody> = HashMap::new();
        for f in &self.module.functions {
            fn_map.insert(f.name.as_str(), &f.body);
            match &f.body {
                MirFnBody::Direct(stmts) => collect_stmts(stmts, &mut sites),
                MirFnBody::Coroutine { blocks, .. } => {
                    for b in blocks {
                        collect_stmts(&b.stmts, &mut sites);
                    }
                }
            }
        }
        for bp in &self.module.task_blueprints {
            for b in &bp.blocks {
                collect_stmts(&b.stmts, &mut sites);
            }
        }
        for isr in &self.module.isrs {
            collect_stmts(&isr.body, &mut sites);
        }

        let mut uncond_hooks = BTreeSet::new();
        if self.module.boot.present {
            let mut active = BTreeSet::new();
            for step in &self.module.boot.steps {
                if let MirBootStep::Stmt(s) = step {
                    if walk_stmt_uncond(s, &fn_map, &mut active, &mut uncond_hooks)
                        == WalkStop::ProgramExits
                    {
                        break;
                    }
                }
            }
        }

        let resolve_one = |kind: HookKind| -> StaticHookMode {
            let Some(entries) = sites.get(&kind) else {
                return StaticHookMode::Unset;
            };
            if entries.is_empty() {
                return StaticHookMode::Unset;
            }
            if entries.iter().all(Option::is_some) {
                let mut arms: Vec<(String, String)> = Vec::new();
                let mut seen = BTreeSet::new();
                let mut all_resolved = true;
                for name_opt in entries {
                    let Some(fname) = name_opt else {
                        continue;
                    };
                    if seen.insert(fname.to_string()) {
                        if let Some(sig) = self.fn_sigs.get(*fname) {
                            arms.push((fname.to_string(), sig.symbol.clone()));
                        } else {
                            all_resolved = false;
                            break;
                        }
                    }
                }
                if all_resolved {
                    if arms.len() == 1 && uncond_hooks.contains(&kind) {
                        if let Some((fn_name, symbol)) = arms.pop() {
                            return StaticHookMode::DirectUnconditional { fn_name, symbol };
                        }
                    }
                    return StaticHookMode::TagSwitch {
                        arms,
                        global_sym: format!("@__syn_hook_tag_{}", kind.suffix()),
                    };
                }
            }
            StaticHookMode::Indirect {
                global_sym: format!("@__syn_hook_fn_{}", kind.suffix()),
            }
        };

        ResolvedSysHooks {
            putc: resolve_one(HookKind::Putc),
            clock_ms: resolve_one(HookKind::ClockMs),
            exit: resolve_one(HookKind::Exit),
            fault_handler: resolve_one(HookKind::FaultHandler),
            idle: resolve_one(HookKind::Idle),
            irq_route: resolve_one(HookKind::IrqRoute),
            irq_ack: resolve_one(HookKind::IrqAck),
        }
    }

    pub(super) fn emit_sys_hook_globals_ir(&self, out: &mut String) {
        let fn_ptr_ty = self.spec.isa.fn_ptr_llvm_ty();
        let ptr_al = self.ptr_align();
        let all_kinds = [
            HookKind::Putc,
            HookKind::ClockMs,
            HookKind::Exit,
            HookKind::FaultHandler,
            HookKind::Idle,
            HookKind::IrqRoute,
            HookKind::IrqAck,
        ];
        let mut any = false;
        for kind in all_kinds {
            match self.hooks.mode(kind) {
                StaticHookMode::TagSwitch { ref global_sym, .. } => {
                    if !any {
                        let _ = writeln!(out, "; ── Explicit `sys.*` Hook Static Dispatch State ──");
                        any = true;
                    }
                    let _ = writeln!(out, "{global_sym} = internal global i8 0, align 1");
                }
                StaticHookMode::Indirect { global_sym } => {
                    if !any {
                        let _ = writeln!(out, "; ── Explicit `sys.*` Hook Static Dispatch State ──");
                        any = true;
                    }
                    let _ = writeln!(
                        out,
                        "{global_sym} = internal global {fn_ptr_ty} null, align {ptr_al}"
                    );
                }
                StaticHookMode::Unset | StaticHookMode::DirectUnconditional { .. } => {}
            }
        }
        if any {
            let _ = writeln!(out);
        }
    }

    pub(super) fn emit_hook_wrapper(
        &self,
        out: &mut String,
        kind: HookKind,
        params_sig: &str,
        ret_ty: &str,
        section_attr: &str,
        fallback_body: &str,
        pre_call_ir: &str,
        pass_args: &str,
    ) {
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let call_kw = self.call_kw();
        let fn_ptr_ty = self.spec.isa.fn_ptr_llvm_ty();
        let ptr_al = self.ptr_align();
        let sym = kind.wrapper_sym();
        let emit_call_and_ret = |callee: &str| -> String {
            if ret_ty == "void" {
                format!("{pre_call_ir}  {call_kw} void {callee}({pass_args})\n  ret void")
            } else {
                format!("{pre_call_ir}  %rv = {call_kw} {ret_ty} {callee}({pass_args})\n  ret {ret_ty} %rv")
            }
        };

        match self.hooks.mode(kind) {
            StaticHookMode::Unset => {
                let _ = writeln!(
                    out,
                    "define {ret_ty} {sym}({params_sig}){fn_as} nounwind{fn_sz}{section_attr} {{\nentry:\n{fallback_body}\n}}\n"
                );
            }
            StaticHookMode::DirectUnconditional { symbol, .. } => {
                let body = emit_call_and_ret(symbol);
                let _ = writeln!(
                    out,
                    "define {ret_ty} {sym}({params_sig}){fn_as} nounwind{fn_sz}{section_attr} {{\nentry:\n{body}\n}}\n"
                );
            }
            StaticHookMode::TagSwitch { arms, global_sym } => {
                let _ = writeln!(
                    out,
                    "define {ret_ty} {sym}({params_sig}){fn_as} nounwind{fn_sz}{section_attr} {{\nentry:"
                );
                let _ = writeln!(out, "  %tag = load i8, ptr {global_sym}, align 1");
                let _ = writeln!(out, "  switch i8 %tag, label %fallback [");
                for (i, _) in arms.iter().enumerate() {
                    let tag = i + 1;
                    let _ = writeln!(out, "    i8 {tag}, label %arm_{tag}");
                }
                let _ = writeln!(out, "  ]");
                for (i, (_, fn_sym)) in arms.iter().enumerate() {
                    let tag = i + 1;
                    let _ = writeln!(out, "arm_{tag}:");
                    let _ = writeln!(out, "{}", emit_call_and_ret(fn_sym));
                }
                let _ = writeln!(out, "fallback:\n{fallback_body}\n}}\n");
            }
            StaticHookMode::Indirect { global_sym } => {
                let call_body = emit_call_and_ret("%fp");
                let _ = writeln!(
                    out,
                    "define {ret_ty} {sym}({params_sig}){fn_as} nounwind{fn_sz}{section_attr} {{\nentry:\n  %fp = load {fn_ptr_ty}, ptr {global_sym}, align {ptr_al}\n  %has = icmp ne {fn_ptr_ty} %fp, null\n  br i1 %has, label %do_call, label %fallback\ndo_call:\n{call_body}\nfallback:\n{fallback_body}\n}}\n"
                );
            }
        }
    }

    pub(super) fn emit_sys_hooks_ir(&self, out: &mut String) {
        let _ = writeln!(out, "; ── Explicit `sys.*` Hook Wrappers (Zero-RAM when Unconditional/Unset) ──");
        let putc_fallback = if self.spec.isa == CpuIsa::HostX86_64Linux {
            "  %z = zext i8 %c to i32\n  %r = call i32 @putchar(i32 %z)\n  ret void"
        } else if self.spec.isa == CpuIsa::ArmCortexM && self.spec.semihosting && self.spec.has_thumb2 {
            "  %ch = alloca i8, align 1\n  store i8 %c, ptr %ch, align 1\n  call void asm sideeffect \"mov r0, #0x03\\0Abkpt #0xab\", \"{r1},~{r0},~{memory}\"(ptr %ch)\n  ret void"
        } else {
            "  ret void"
        };
        let iram_sec = if self.spec.isa.is_riscv() && self.spec.iram_in_data {
            " section \".iram1\""
        } else {
            ""
        };
        self.emit_hook_wrapper(
            out,
            HookKind::Putc,
            "i8 %c",
            "void",
            "",
            putc_fallback,
            "",
            "i8 %c",
        );
        let clock_fallback = if self.spec.isa == CpuIsa::HostX86_64Linux {
            "  %t = call i32 @__syn_host_now_ms()\n  ret i32 %t"
        } else {
            "  %t = load volatile i32, ptr @__syn_tick_ms, align 4\n  ret i32 %t"
        };
        self.emit_hook_wrapper(
            out,
            HookKind::ClockMs,
            "",
            "i32",
            "",
            clock_fallback,
            "",
            "",
        );
        self.emit_hook_wrapper(
            out,
            HookKind::Exit,
            "i32 %code",
            "void",
            "",
            "  ret void",
            "",
            "i32 %code",
        );
        self.emit_hook_wrapper(
            out,
            HookKind::FaultHandler,
            "",
            "void",
            iram_sec,
            "  ret void",
            "",
            "",
        );
        self.emit_hook_wrapper(
            out,
            HookKind::Idle,
            "i32 %in_sim, i32 %depth",
            "void",
            "",
            "  ret void",
            "",
            "i32 %in_sim, i32 %depth",
        );
        if self.spec.isa.is_riscv() {
            self.emit_hook_wrapper(
                out,
                HookKind::IrqRoute,
                "i32 %src, i32 %slot",
                "void",
                "",
                "  ret void",
                "",
                "i32 %src, i32 %slot",
            );
            self.emit_hook_wrapper(
                out,
                HookKind::IrqAck,
                "i32 %slot",
                "void",
                iram_sec,
                "  ret void",
                "",
                "i32 %slot",
            );
        }
    }

    pub(super) fn emit_rom_wrappers_ir(&self, out: &mut String) {
        if self.rom_wrappers.is_empty() {
            return;
        }
        let usize_ty = self.spec.isa.usize_llvm_ty();
        let fn_ptr_ty = self.spec.isa.fn_ptr_llvm_ty();
        let fn_as = self.fn_as();
        let call_kw = self.call_kw();
        let fn_sz = self.opts.fn_size_attrs();
        let _ = writeln!(out, "; ── Stack-Bounded ROM / Firmware ABI Wrappers ──");
        for w in &self.rom_wrappers {
            let sym = w.symbol();
            let mut def_params = vec![format!("{usize_ty} %addr")];
            let mut call_args = Vec::with_capacity(w.param_llvm_tys.len());
            for (i, pty) in w.param_llvm_tys.iter().enumerate() {
                def_params.push(format!("{pty} %a{i}"));
                call_args.push(format!("{pty} %a{i}"));
            }
            let def_sig = def_params.join(", ");
            let pass_args = call_args.join(", ");
            match w.kind {
                RomWrapperKind::U32 => {
                    let _ = writeln!(
                        out,
                        "define i32 {sym}({def_sig}){fn_as} nounwind noinline{fn_sz} {{\nentry:\n  %fp = inttoptr {usize_ty} %addr to {fn_ptr_ty}\n  %rv = {call_kw} i32 %fp({pass_args})\n  ret i32 %rv\n}}\n"
                    );
                }
                RomWrapperKind::Void => {
                    let _ = writeln!(
                        out,
                        "define void {sym}({def_sig}){fn_as} nounwind noinline{fn_sz} {{\nentry:\n  %fp = inttoptr {usize_ty} %addr to {fn_ptr_ty}\n  {call_kw} void %fp({pass_args})\n  ret void\n}}\n"
                    );
                }
                RomWrapperKind::Noreturn => {
                    let _ = writeln!(
                        out,
                        "define void {sym}({def_sig}){fn_as} noreturn nounwind noinline{fn_sz} {{\nentry:\n  %fp = inttoptr {usize_ty} %addr to {fn_ptr_ty}\n  {call_kw} void %fp({pass_args})\n  unreachable\n}}\n"
                    );
                }
            }
        }
    }

    pub(super) fn visit_struct<'b>(
        &mut self,
        name: &str,
        by_name: &HashMap<&'b str, &'b MirStruct>,
        visited: &mut BTreeSet<String>,
    ) {
        if visited.contains(name) {
            return;
        }
        visited.insert(name.to_string());
        let Some(&decl) = by_name.get(name) else {
            return;
        };
        for fld in &decl.fields {
            if let MirType::Struct(sub) = &fld.ty {
                self.visit_struct(sub, by_name, visited);
            }
        }
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let mut offset = 0usize;
        let mut max_align = 1usize;
        let mut llvm_off = 0usize;
        let mut llvm_max_nat_al = 1usize;
        let mut field_indices = HashMap::new();
        let mut llvm_fields = Vec::new();
        let struct_dma = self.dma_layout_structs.contains(&decl.name);
        for (idx, fld) in decl.fields.iter().enumerate() {
            let eff_fld_dma = fld.dma_safe || struct_dma;
            let is_dma_pool = eff_fld_dma
                && matches!(fld.ty, MirType::Primitive(MirPrimKind::Pool { .. }));
            let (fsize, mut falign) = self.size_align(&fld.ty, ptr_bytes);
            let fsize = if is_dma_pool {
                if let MirType::Primitive(MirPrimKind::Pool { elem, cap }) = &fld.ty {
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let stride = align_up(esz.max(1), eal.max(32));
                    32 + (*cap).max(1) * stride
                } else {
                    fsize
                }
            } else {
                fsize
            };
            if is_dma_pool {
                falign = falign.max(32);
            }
            if !decl.packed {
                offset = align_up(offset, falign);
                max_align = max_align.max(falign);
            }
            offset += fsize;

            let (nat_sz, nat_al) = self.llvm_abi_size_align(&fld.ty, eff_fld_dma);
            let req_al = if decl.is_subsystem {
                falign
            } else if decl.packed {
                1
            } else {
                nat_al
            };
            if decl.is_subsystem && req_al > nat_al {
                let aligned = align_up(llvm_off, req_al);
                if aligned > llvm_off {
                    llvm_fields.push((None, format!("[{} x i8]", aligned - llvm_off)));
                    llvm_off = aligned;
                }
            }
            if !decl.packed {
                llvm_off = align_up(llvm_off, nat_al);
                llvm_max_nat_al = llvm_max_nat_al.max(nat_al);
            }
            let llvm_idx = llvm_fields.len();
            llvm_fields.push((Some(idx), self.llvm_field_ty(&decl.name, fld)));
            llvm_off += nat_sz;
            field_indices.insert(fld.name.clone(), (llvm_idx, fld.ty.clone()));
        }
        if !decl.packed {
            offset = align_up(offset, max_align);
            llvm_off = align_up(llvm_off, llvm_max_nat_al);
        }
        let llvm_name = format!("%struct.{}", sanitize_ident(&decl.name));
        self.structs.insert(
            decl.name.clone(),
            StructInfo {
                decl: decl.clone(),
                llvm_name,
                byte_size: offset.max(1),
                align: if decl.packed { 1 } else { max_align.max(1) },
                llvm_abi_size: llvm_off.max(1),
                llvm_nat_align: if decl.packed {
                    1
                } else {
                    llvm_max_nat_al.max(1)
                },
                field_indices,
                llvm_fields,
            },
        );
        self.struct_order.push(decl.name.clone());
    }

    pub(super) fn scan_mir_type_for_prims(&mut self, ty: &MirType) {
        match ty {
            MirType::Primitive(pk) => self.register_prim_kind(pk),
            MirType::Lease(elem) => {
                self.uses_pool_lease = true;
                self.scan_mir_type_for_prims(elem);
            }
            MirType::PrimView(MirPrimViewKind::ByteRing) => self.uses_byte_ring_view = true,
            MirType::PrimView(MirPrimViewKind::Inbox { elem }) => self.register_inbox_view(elem),
            MirType::Ptr(inner) | MirType::Slice(inner) => self.scan_mir_type_for_prims(inner),
            MirType::Array { elem, .. } => self.scan_mir_type_for_prims(elem),
            MirType::Result { ok, err } => {
                self.scan_mir_type_for_prims(ok);
                if let Some(e) = err {
                    self.scan_mir_type_for_prims(e);
                }
            }
            _ => {}
        }
    }

    pub(super) fn register_inbox_view(&mut self, elem: &MirType) {
        self.scan_mir_type_for_prims(elem);
        let key = sanitize_ident(&elem.mangle_name());
        if !self.distinct_inbox_views.contains_key(&key) {
            self.distinct_inbox_views.insert(key, elem.clone());
        }
    }

    pub(super) fn register_prim_kind(&mut self, pk: &MirPrimKind) {
        if matches!(pk, MirPrimKind::Pool { .. } | MirPrimKind::DmaSlot { .. }) {
            self.uses_pool_lease = true;
        }
        match pk {
            MirPrimKind::Watch { elem }
            | MirPrimKind::Guarded { elem }
            | MirPrimKind::Inbox { elem, .. }
            | MirPrimKind::PingPong { elem, .. }
            | MirPrimKind::Pool { elem, .. }
            | MirPrimKind::DmaSlot { elem } => self.scan_mir_type_for_prims(elem),
            MirPrimKind::Signal | MirPrimKind::WaitQueue | MirPrimKind::ByteRing { .. } => {}
        }
        let key = pk.mangle_name();
        if !self.distinct_prims.contains_key(&key) {
            self.distinct_prims.insert(key, pk.clone());
        }
    }

    pub(super) fn scan_prim_target_for_prims(&mut self, target: &MirPrimTarget) {
        if let MirPrimTarget::PtrExpr(e) = target {
            self.scan_expr_for_prims(e);
        }
    }

    pub(super) fn scan_blocks_for_prims(&mut self, blocks: &[MirBlock]) {
        for b in blocks {
            for s in &b.stmts {
                self.scan_stmt_for_prims(s);
            }
            match &b.terminator {
                MirTerminator::Branch { cond, .. } | MirTerminator::Switch { discr: cond, .. } => {
                    self.scan_expr_for_prims(cond);
                }
                MirTerminator::Return(Some(e)) => self.scan_expr_for_prims(e),
                MirTerminator::Error { reason, partial } => {
                    if let Some(r) = reason {
                        self.scan_expr_for_prims(r);
                    }
                    if let Some(p) = partial {
                        self.scan_expr_for_prims(p);
                    }
                }
                MirTerminator::Yield(yk) => match yk {
                    MirYieldKind::CallYielding { args, .. } => {
                        for a in args {
                            self.scan_expr_for_prims(a);
                        }
                    }
                    MirYieldKind::Wait { arms, .. } => {
                        for arm in arms {
                            match &arm.op {
                                MirWaitArmOp::InboxViewRecv { view, elem_ty, .. }
                                | MirWaitArmOp::InboxViewSend { view, elem_ty, .. } => {
                                    self.scan_expr_for_prims(view);
                                    self.register_inbox_view(elem_ty);
                                }
                                MirWaitArmOp::ByteRingViewWaitRead { view, .. } => {
                                    self.scan_expr_for_prims(view);
                                    self.uses_byte_ring_view = true;
                                }
                                MirWaitArmOp::PoolAlloc {
                                    target,
                                    elem_ty,
                                    cap,
                                    ..
                                } => {
                                    self.scan_prim_target_for_prims(target);
                                    self.register_prim_kind(&MirPrimKind::Pool {
                                        elem: Box::new(elem_ty.clone()),
                                        cap: *cap,
                                    });
                                }
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
                                    self.scan_prim_target_for_prims(target);
                                    self.register_prim_kind(&MirPrimKind::Inbox {
                                        elem: Box::new(elem_ty.clone()),
                                        cap: *cap,
                                    });
                                }
                                MirWaitArmOp::WatchWaitChange {
                                    target, elem_ty, ..
                                } => {
                                    self.scan_prim_target_for_prims(target);
                                    self.register_prim_kind(&MirPrimKind::Watch {
                                        elem: Box::new(elem_ty.clone()),
                                    });
                                }
                                MirWaitArmOp::PingPongWaitChunk {
                                    target,
                                    elem_ty,
                                    cap,
                                    ..
                                } => {
                                    self.scan_prim_target_for_prims(target);
                                    self.register_prim_kind(&MirPrimKind::PingPong {
                                        elem: Box::new(elem_ty.clone()),
                                        cap: *cap,
                                    });
                                }
                                MirWaitArmOp::SignalWait { target, .. } => {
                                    self.scan_prim_target_for_prims(target);
                                    self.register_prim_kind(&MirPrimKind::Signal);
                                }
                                MirWaitArmOp::WaitQueueCommit { target, .. } => {
                                    self.scan_prim_target_for_prims(target);
                                    self.register_prim_kind(&MirPrimKind::WaitQueue);
                                }
                                MirWaitArmOp::ByteRingWaitRead { target, cap, .. } => {
                                    self.scan_prim_target_for_prims(target);
                                    self.register_prim_kind(&MirPrimKind::ByteRing { cap: *cap });
                                }
                            }
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    pub(super) fn scan_stmt_for_prims(&mut self, stmt: &MirStmt) {
        match stmt {
            MirStmt::Assign { value, .. } | MirStmt::Expr(value) | MirStmt::Return(Some(value)) => {
                self.scan_expr_for_prims(value);
            }
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.scan_expr_for_prims(cond);
                for s in then_body {
                    self.scan_stmt_for_prims(s);
                }
                for s in else_body {
                    self.scan_stmt_for_prims(s);
                }
            }
            MirStmt::While { cond, body } => {
                self.scan_expr_for_prims(cond);
                for s in body {
                    self.scan_stmt_for_prims(s);
                }
            }
            MirStmt::Switch {
                discr,
                arms,
                default,
            } => {
                self.scan_expr_for_prims(discr);
                for (_, ab) in arms {
                    for s in ab {
                        self.scan_stmt_for_prims(s);
                    }
                }
                for s in default {
                    self.scan_stmt_for_prims(s);
                }
            }
            MirStmt::FmtRing { ring, chunks, .. } => {
                self.scan_expr_for_prims(ring);
                self.uses_byte_ring_view = true;
                for ch in chunks {
                    self.scan_fmt_chunk_for_prims(ch);
                }
            }
            MirStmt::PrintFmt { chunks, .. } | MirStmt::FmtBuf { chunks, .. } => {
                if let MirStmt::FmtBuf { buf, .. } = stmt {
                    self.scan_expr_for_prims(buf);
                }
                for ch in chunks {
                    self.scan_fmt_chunk_for_prims(ch);
                }
            }
            MirStmt::PrimOp(pop) => match pop {
                MirPrimNonBlockingOp::InboxViewPost {
                    view,
                    elem_ty,
                    value,
                    ..
                } => {
                    self.scan_expr_for_prims(view);
                    self.register_inbox_view(elem_ty);
                    self.scan_expr_for_prims(value);
                }
                MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
                | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                    self.scan_expr_for_prims(view);
                    self.uses_byte_ring_view = true;
                    self.scan_expr_for_prims(len);
                }
                MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                    self.scan_expr_for_prims(view);
                    self.uses_byte_ring_view = true;
                }
                MirPrimNonBlockingOp::WatchPublish { target, value, .. }
                | MirPrimNonBlockingOp::InboxPost { target, value, .. } => {
                    self.scan_prim_target_for_prims(target);
                    self.scan_expr_for_prims(value);
                }
                MirPrimNonBlockingOp::GuardedSet {
                    target,
                    elem_ty,
                    value,
                } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::Guarded {
                        elem: Box::new(elem_ty.clone()),
                    });
                    self.scan_expr_for_prims(value);
                }
                MirPrimNonBlockingOp::LeaseRelease { value, elem_ty, .. } => {
                    self.uses_pool_lease = true;
                    self.scan_mir_type_for_prims(elem_ty);
                    self.scan_expr_for_prims(value);
                }
                MirPrimNonBlockingOp::LeaseNullify { elem_ty, .. } => {
                    self.uses_pool_lease = true;
                    self.scan_mir_type_for_prims(elem_ty);
                }
                MirPrimNonBlockingOp::ByteRingConsume { len, .. }
                | MirPrimNonBlockingOp::ByteRingCommitWrite { len, .. } => {
                    self.scan_expr_for_prims(len);
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
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::PingPong {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
                MirPrimNonBlockingOp::PingPongCompleteHalf {
                    target,
                    elem_ty,
                    cap,
                    half,
                } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::PingPong {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                    self.scan_expr_for_prims(half);
                }
                MirPrimNonBlockingOp::WaitQueuePrepare { target }
                | MirPrimNonBlockingOp::WaitQueueCancel { target }
                | MirPrimNonBlockingOp::WaitQueueWakeAll { target, .. } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::WaitQueue);
                }
                _ => {}
            },
            MirStmt::SpawnActivate { args, .. } => {
                for a in args {
                    self.scan_expr_for_prims(a);
                }
            }
            MirStmt::Intrinsic(iop) => match iop {
                MirIntrinsicOp::UseSystick { hz } => self.scan_expr_for_prims(hz),
                MirIntrinsicOp::SysOp { args, .. } => {
                    for a in args {
                        self.scan_expr_for_prims(a);
                    }
                }
                MirIntrinsicOp::SetPriority { .. } => {}
            },
            MirStmt::MmioWrite {
                base, index, value, ..
            } => {
                self.scan_expr_for_prims(base);
                if let Some(i) = index {
                    self.scan_expr_for_prims(i);
                }
                self.scan_expr_for_prims(value);
            }
            MirStmt::Error { reason, partial } => {
                if let Some(r) = reason {
                    self.scan_expr_for_prims(r);
                }
                if let Some(p) = partial {
                    self.scan_expr_for_prims(p);
                }
            }
            _ => {}
        }
    }

    pub(super) fn scan_fmt_chunk_for_prims(&mut self, ch: &MirFmtChunk) {
        match ch {
            MirFmtChunk::Int { value, .. }
            | MirFmtChunk::Float { value, .. }
            | MirFmtChunk::Bool(value)
            | MirFmtChunk::Str(value)
            | MirFmtChunk::Bytes(value)
            | MirFmtChunk::EnumName { value, .. } => self.scan_expr_for_prims(value),
            MirFmtChunk::DebugSlice { slice, .. } => self.scan_expr_for_prims(slice),
            MirFmtChunk::Literal(_) => {}
        }
    }

    pub(super) fn scan_expr_for_prims(&mut self, expr: &MirExpr) {
        self.scan_mir_type_for_prims(&expr.ty);
        match &expr.kind {
            MirExprKind::Unary { expr: e, .. }
            | MirExprKind::Cast { expr: e, .. }
            | MirExprKind::FieldAccess { base: e, .. }
            | MirExprKind::SliceLen(e)
            | MirExprKind::FromBytes { slice: e, .. } => self.scan_expr_for_prims(e),
            MirExprKind::Binary { lhs, rhs, .. }
            | MirExprKind::Index {
                base: lhs,
                index: rhs,
                ..
            } => {
                self.scan_expr_for_prims(lhs);
                self.scan_expr_for_prims(rhs);
            }
            MirExprKind::SubSlice {
                base, start, end, ..
            } => {
                self.scan_expr_for_prims(base);
                self.scan_expr_for_prims(start);
                self.scan_expr_for_prims(end);
            }
            MirExprKind::StructLit { fields, .. } => {
                for (_, fe, _) in fields {
                    self.scan_expr_for_prims(fe);
                }
            }
            MirExprKind::ArrayLit { elems, .. } | MirExprKind::CallDirect { args: elems, .. } => {
                for a in elems {
                    self.scan_expr_for_prims(a);
                }
            }
            MirExprKind::RomCallU32 { addr, args, .. } => {
                self.scan_expr_for_prims(addr);
                for a in args {
                    self.scan_expr_for_prims(a);
                }
            }
            MirExprKind::MakePrimView { ptr, cap, kind } => {
                self.scan_expr_for_prims(ptr);
                let n = (*cap).max(1);
                match kind {
                    MirPrimViewKind::ByteRing => {
                        self.uses_byte_ring_view = true;
                        self.register_prim_kind(&MirPrimKind::ByteRing { cap: n });
                    }
                    MirPrimViewKind::Inbox { elem } => {
                        self.register_inbox_view(elem);
                        self.register_prim_kind(&MirPrimKind::Inbox {
                            elem: elem.clone(),
                            cap: n,
                        });
                    }
                }
            }
            MirExprKind::PrimRead(ro) => match ro.as_ref() {
                MirPrimReadOp::InboxViewTryRecv { view, elem_ty }
                | MirPrimReadOp::InboxViewLen { view, elem_ty }
                | MirPrimReadOp::InboxViewCap { view, elem_ty }
                | MirPrimReadOp::InboxViewIsEmpty { view, elem_ty }
                | MirPrimReadOp::InboxViewIsFull { view, elem_ty } => {
                    self.scan_expr_for_prims(view);
                    self.register_inbox_view(elem_ty);
                }
                MirPrimReadOp::InboxViewTrySend {
                    view,
                    elem_ty,
                    value,
                    ..
                } => {
                    self.scan_expr_for_prims(view);
                    self.register_inbox_view(elem_ty);
                    self.scan_expr_for_prims(value);
                }
                MirPrimReadOp::ByteRingViewPush { view, src: val, .. }
                | MirPrimReadOp::ByteRingViewPop { view, dst: val }
                | MirPrimReadOp::ByteRingViewPushByte {
                    view, byte: val, ..
                }
                | MirPrimReadOp::ByteRingViewGrantWrite { view, len: val } => {
                    self.scan_expr_for_prims(view);
                    self.uses_byte_ring_view = true;
                    self.scan_expr_for_prims(val);
                }
                MirPrimReadOp::ByteRingViewReadSlice { view }
                | MirPrimReadOp::ByteRingViewTryPopByte { view }
                | MirPrimReadOp::ByteRingViewLen { view }
                | MirPrimReadOp::ByteRingViewCap { view }
                | MirPrimReadOp::ByteRingViewFreeSpace { view }
                | MirPrimReadOp::ByteRingViewIsEmpty { view }
                | MirPrimReadOp::ByteRingViewIsFull { view } => {
                    self.scan_expr_for_prims(view);
                    self.uses_byte_ring_view = true;
                }
                MirPrimReadOp::ByteRingPush {
                    target, src: val, ..
                }
                | MirPrimReadOp::ByteRingPop {
                    target, dst: val, ..
                }
                | MirPrimReadOp::ByteRingPushByte {
                    target, byte: val, ..
                }
                | MirPrimReadOp::ByteRingGrantWrite {
                    target, len: val, ..
                } => {
                    self.scan_prim_target_for_prims(target);
                    self.scan_expr_for_prims(val);
                }
                MirPrimReadOp::InboxTrySend {
                    target,
                    elem_ty,
                    cap,
                    value: val,
                    ..
                } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                    self.scan_expr_for_prims(val);
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
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::Pool {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
                MirPrimReadOp::DmaSlotPark {
                    target,
                    elem_ty,
                    lease,
                } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::DmaSlot {
                        elem: Box::new(elem_ty.clone()),
                    });
                    self.scan_expr_for_prims(lease);
                }
                MirPrimReadOp::DmaSlotTake { target, elem_ty }
                | MirPrimReadOp::DmaSlotRelease {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::DmaSlotIsBusy { target, elem_ty }
                | MirPrimReadOp::DmaSlotIsEmpty { target, elem_ty }
                | MirPrimReadOp::DmaSlotAddr { target, elem_ty } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::DmaSlot {
                        elem: Box::new(elem_ty.clone()),
                    });
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
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::PingPong {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
                MirPrimReadOp::WaitQueueWakeOne { target, .. } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::WaitQueue);
                }
                MirPrimReadOp::GuardedGet { target, elem_ty } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::Guarded {
                        elem: Box::new(elem_ty.clone()),
                    });
                }
                _ => {}
            },
            MirExprKind::LeaseGet { lease, elem_ty }
            | MirExprKind::LeaseAddr { lease, elem_ty } => {
                self.uses_pool_lease = true;
                self.scan_mir_type_for_prims(elem_ty);
                self.scan_expr_for_prims(lease);
            }
            MirExprKind::MmioRead { base, index, .. } => {
                self.scan_expr_for_prims(base);
                if let Some(i) = index {
                    self.scan_expr_for_prims(i);
                }
            }
            _ => {}
        }
    }

    pub(super) fn compute_fn_frame_layout<'b>(
        &mut self,
        name: &str,
        fn_map: &HashMap<&'b str, &'b MirFunction>,
        done: &mut BTreeSet<String>,
    ) {
        if done.contains(name) {
            return;
        }
        done.insert(name.to_string());
        let Some(&func) = fn_map.get(name) else {
            return;
        };
        if let MirFnBody::Coroutine { child_slots, .. } = &func.body {
            for slot in child_slots {
                self.compute_fn_frame_layout(&slot.callee, fn_map, done);
            }
            let env_ty = func.envelope_return_type();
            let has_ret = !env_ty.is_void();
            let layout = self.build_coroutine_layout(
                &format!("%FnFrame_{}", sanitize_ident(&func.name)),
                has_ret,
                &env_ty,
                &func.vars,
                child_slots,
            );
            self.fn_frames.insert(func.name.clone(), layout);
        }
    }

    pub(super) fn build_coroutine_layout(
        &self,
        llvm_name: &str,
        has_ret_slot: bool,
        ret_env_ty: &MirType,
        vars: &[MirVar],
        child_slots: &[MirChildCallSlot],
    ) -> CoroutineFrameLayout {
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let mut field_llvm_tys = Vec::new();
        // Field 0 is always `i32` (`_pc`)
        field_llvm_tys.push("i32".to_string());
        let mut offset = 4usize;
        let mut max_align = 4usize;

        let ret_llvm_ty = if has_ret_slot {
            let (sz, al) = self.size_align(ret_env_ty, ptr_bytes);
            offset = align_up(offset, al);
            offset += sz;
            max_align = max_align.max(al);
            let rty = self.llvm_ty(ret_env_ty);
            field_llvm_tys.push(rty.clone());
            rty
        } else {
            "void".to_string()
        };

        let mut hoisted_var_gep = HashMap::new();
        for v in vars {
            if v.hoisted && !v.ty.is_void() {
                let gep_idx = field_llvm_tys.len();
                let (sz, al) = self.size_align(&v.ty, ptr_bytes);
                offset = align_up(offset, al);
                offset += sz;
                max_align = max_align.max(al);
                field_llvm_tys.push(self.llvm_ty(&v.ty));
                hoisted_var_gep.insert(v.id, gep_idx);
            }
        }

        // Group disjoint child slots by `union_group`
        let mut group_max_bytes: BTreeMap<usize, usize> = BTreeMap::new();
        let mut slot_to_group = HashMap::new();
        for slot in child_slots {
            let child_bytes = self
                .fn_frames
                .get(&slot.callee)
                .map_or(8, |cf| cf.byte_size_bound);
            let entry = group_max_bytes.entry(slot.union_group).or_insert(8);
            *entry = (*entry).max(child_bytes);
            slot_to_group.insert(slot.slot_name.clone(), slot.union_group);
        }

        let mut union_group_gep = BTreeMap::new();
        for (ug, max_b) in group_max_bytes {
            let words_i64 = ((max_b + 7) / 8).max(1);
            let gep_idx = field_llvm_tys.len();
            offset = align_up(offset, 8);
            offset += words_i64 * 8;
            max_align = max_align.max(8);
            field_llvm_tys.push(format!("[{words_i64} x i64]"));
            union_group_gep.insert(ug, (gep_idx, words_i64));
        }

        let byte_size_bound = align_up(offset, max_align).max(8);
        CoroutineFrameLayout {
            llvm_name: llvm_name.to_string(),
            has_ret_slot,
            ret_llvm_ty,
            hoisted_var_gep,
            union_group_gep,
            slot_to_group,
            byte_size_bound,
            field_llvm_tys,
        }
    }

    pub(super) fn size_align(&self, ty: &MirType, ptr_bytes: usize) -> (usize, usize) {
        match ty {
            MirType::Void => (0, 1),
            MirType::Bool | MirType::U8 | MirType::I8 => (1, 1),
            MirType::U16 | MirType::I16 | MirType::EndianInt { bits: 16, .. } => (2, 2),
            MirType::U32
            | MirType::I32
            | MirType::F32
            | MirType::EndianInt { bits: 32, .. }
            | MirType::Enum(_) => (4, 4),
            MirType::U64 | MirType::I64 | MirType::F64 | MirType::EndianInt { bits: 64, .. } => {
                (8, 8)
            }
            MirType::EndianInt { .. } => (4, 4),
            MirType::Usize
            | MirType::Isize
            | MirType::Ptr(_)
            | MirType::MmioPtr(_)
            | MirType::FnPtr { .. }
            | MirType::Str => (ptr_bytes, ptr_bytes),
            MirType::Slice(_) => (ptr_bytes * 2, ptr_bytes),
            MirType::Array { elem, len } => {
                let (esz, eal) = self.size_align(elem, ptr_bytes);
                (esz * (*len).max(1), eal)
            }
            MirType::Struct(name) => {
                if let Some(si) = self.structs.get(name) {
                    (si.byte_size, si.align)
                } else {
                    (4, 4)
                }
            }
            MirType::Primitive(pk) => match pk {
                MirPrimKind::Signal => (8, 4),
                MirPrimKind::WaitQueue => (4, 4),
                MirPrimKind::Watch { elem } => {
                    let (esz, eal) = self.size_align(elem, ptr_bytes);
                    let al = eal.max(4);
                    (align_up(esz, 4) + 8, al)
                }
                MirPrimKind::Guarded { elem } => self.size_align(elem, ptr_bytes),
                MirPrimKind::Inbox { elem, cap } => {
                    let (esz, eal) = self.size_align(elem, ptr_bytes);
                    let al = eal.max(4);
                    (align_up(20 + (*cap).max(1), eal) + esz * (*cap).max(1), al)
                }
                MirPrimKind::PingPong { elem, cap } => {
                    let (esz, eal) = self.size_align(elem, ptr_bytes);
                    (align_up(8, eal) + 2 * esz * (*cap).max(1), eal.max(4))
                }
                MirPrimKind::Pool { elem, cap } => {
                    let (hdr_sz, hdr_al) = self.pool_hdr_llvm_size();
                    let (esz, eal) = self.size_align(elem, ptr_bytes);
                    let al = eal.max(hdr_al);
                    (align_up(hdr_sz, eal) + esz * (*cap).max(1), al)
                }
                MirPrimKind::DmaSlot { .. } => {
                    let al = self.ptr_align();
                    (align_up(2 * ptr_bytes + 2, al.max(1)), al.max(1))
                }
                MirPrimKind::ByteRing { cap } => (align_up(29 + (*cap).max(1), 4), 4),
            },
            MirType::Lease(_) => {
                let al = self.ptr_align();
                (align_up(2 * ptr_bytes + 1, al.max(1)), al.max(1))
            }
            MirType::PrimView(MirPrimViewKind::ByteRing) => (2 * ptr_bytes, ptr_bytes),
            MirType::PrimView(MirPrimViewKind::Inbox { .. }) => (3 * ptr_bytes, ptr_bytes),
            MirType::Result { ok, .. } => {
                let (osz, oal) = if ok.is_void() {
                    (4, 4)
                } else {
                    self.size_align(ok, ptr_bytes)
                };
                let al = oal.max(4);
                (align_up(osz, 4) + 4, al)
            }
        }
    }

    pub(super) fn ptr_align(&self) -> usize {
        if self.spec.isa.is_avr() {
            1
        } else {
            (self.spec.isa.ptr_bits() / 8) as usize
        }
    }

    pub(super) fn llvm_scalar_abi_align(&self, natural_align: usize) -> usize {
        if self.spec.isa.is_avr() {
            1
        } else {
            natural_align
        }
    }

    pub(super) fn pool_hdr_llvm_size(&self) -> (usize, usize) {
        match self.spec.isa {
            CpuIsa::HostX86_64Linux => (24, 8),
            CpuIsa::ArmCortexM | CpuIsa::Riscv32 => (16, 4),
            CpuIsa::Avr => (12, 1),
        }
    }

    pub(super) fn llvm_abi_size_align(&self, ty: &MirType, dma_safe: bool) -> (usize, usize) {
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let ptr_al = self.ptr_align();
        match ty {
            MirType::Void => (0, 1),
            MirType::Bool | MirType::U8 | MirType::I8 => (1, 1),
            MirType::U16 | MirType::I16 | MirType::EndianInt { bits: 16, .. } => {
                (2, self.llvm_scalar_abi_align(2))
            }
            MirType::U32
            | MirType::I32
            | MirType::F32
            | MirType::EndianInt { bits: 32, .. }
            | MirType::Enum(_) => (4, self.llvm_scalar_abi_align(4)),
            MirType::U64 | MirType::I64 | MirType::F64 | MirType::EndianInt { bits: 64, .. } => {
                (8, self.llvm_scalar_abi_align(8))
            }
            MirType::EndianInt { .. } => (4, self.llvm_scalar_abi_align(4)),
            MirType::Usize
            | MirType::Isize
            | MirType::Ptr(_)
            | MirType::MmioPtr(_)
            | MirType::FnPtr { .. }
            | MirType::Str => (ptr_bytes, ptr_al),
            MirType::Slice(_) | MirType::PrimView(MirPrimViewKind::ByteRing) => {
                (2 * ptr_bytes, ptr_al)
            }
            MirType::PrimView(MirPrimViewKind::Inbox { .. }) => (3 * ptr_bytes, ptr_al),
            MirType::Lease(_) => (align_up(2 * ptr_bytes + 1, ptr_al), ptr_al),
            MirType::Array { elem, len } => {
                let (esz, eal) = self.llvm_abi_size_align(elem, false);
                (esz * (*len).max(1), eal)
            }
            MirType::Struct(name) => {
                if let Some(si) = self.structs.get(name) {
                    (si.llvm_abi_size, si.llvm_nat_align)
                } else {
                    (4, self.llvm_scalar_abi_align(4))
                }
            }
            MirType::Primitive(pk) => match pk {
                MirPrimKind::Signal => {
                    let i32_al = self.llvm_scalar_abi_align(4);
                    (align_up(2, i32_al) + 4, i32_al)
                }
                MirPrimKind::WaitQueue => (4, self.llvm_scalar_abi_align(4)),
                MirPrimKind::Watch { elem } => {
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let i32_al = self.llvm_scalar_abi_align(4);
                    let al = eal.max(i32_al);
                    (align_up(align_up(esz, i32_al) + 8, al), al)
                }
                MirPrimKind::Guarded { elem } => self.llvm_abi_size_align(elem, false),
                MirPrimKind::Inbox { elem, cap } => {
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let i32_al = self.llvm_scalar_abi_align(4);
                    let c = (*cap).max(1);
                    let al = eal.max(i32_al);
                    (align_up(align_up(20 + c, eal) + c * esz, al), al)
                }
                MirPrimKind::PingPong { elem, cap } => {
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let i32_al = self.llvm_scalar_abi_align(4);
                    let c = (*cap).max(1);
                    let al = eal.max(i32_al);
                    (align_up(align_up(8, eal) + 2 * c * esz, al), al)
                }
                MirPrimKind::Pool { elem, cap } => {
                    let (hdr_sz, hdr_al) = self.pool_hdr_llvm_size();
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let c = (*cap).max(1);
                    if dma_safe {
                        let slot_stride = align_up(esz.max(1), eal.max(32));
                        let al = hdr_al.max(eal);
                        (align_up(32 + c * slot_stride, al), al)
                    } else {
                        let al = hdr_al.max(eal);
                        (align_up(align_up(hdr_sz, eal) + c * esz, al), al)
                    }
                }
                MirPrimKind::DmaSlot { .. } => (align_up(2 * ptr_bytes + 2, ptr_al), ptr_al),
                MirPrimKind::ByteRing { cap } => {
                    let i32_al = self.llvm_scalar_abi_align(4);
                    (align_up(29 + (*cap).max(1), i32_al), i32_al)
                }
            },
            MirType::Result { ok, .. } => {
                let i32_al = self.llvm_scalar_abi_align(4);
                let (osz, oal) = if ok.is_void() {
                    (4, i32_al)
                } else {
                    self.llvm_abi_size_align(ok, false)
                };
                let al = oal.max(i32_al);
                (align_up(align_up(osz, i32_al) + 4, al), al)
            }
        }
    }

    pub(super) fn llvm_field_ty(&self, struct_name: &str, fld: &zev_mir::MirField) -> String {
        let eff_dma = fld.dma_safe || self.dma_layout_structs.contains(struct_name);
        if eff_dma {
            if let MirType::Primitive(pk @ MirPrimKind::Pool { .. }) = &fld.ty {
                return format!("%prim.{}_dma", sanitize_ident(&pk.mangle_name()));
            }
        }
        self.llvm_ty(&fld.ty)
    }

    pub(super) fn llvm_ty(&self, ty: &MirType) -> String {
        match ty {
            MirType::Void => "void".to_string(),
            MirType::Bool | MirType::U8 | MirType::I8 => "i8".to_string(),
            MirType::U16 | MirType::I16 | MirType::EndianInt { bits: 16, .. } => "i16".to_string(),
            MirType::U32
            | MirType::I32
            | MirType::EndianInt { bits: 32, .. }
            | MirType::Enum(_) => "i32".to_string(),
            MirType::U64 | MirType::I64 | MirType::EndianInt { bits: 64, .. } => "i64".to_string(),
            MirType::F32 => "float".to_string(),
            MirType::F64 => "double".to_string(),
            MirType::EndianInt { .. } => "i32".to_string(),
            MirType::Usize | MirType::Isize => self.spec.isa.usize_llvm_ty().to_string(),
            MirType::Ptr(_) | MirType::MmioPtr(_) | MirType::Str => "ptr".to_string(),
            MirType::FnPtr { .. } => self.spec.isa.fn_ptr_llvm_ty().to_string(),
            MirType::Slice(_) => format!("{{ ptr, {} }}", self.spec.isa.usize_llvm_ty()),
            MirType::Array { elem, len } => {
                let n = (*len).max(1);
                format!("[{n} x {}]", self.llvm_ty(elem))
            }
            MirType::Struct(name) => format!("%struct.{}", sanitize_ident(name)),
            MirType::Primitive(pk) => format!("%prim.{}", sanitize_ident(&pk.mangle_name())),
            MirType::Lease(_) => "{ ptr, ptr, i8 }".to_string(),
            MirType::PrimView(MirPrimViewKind::ByteRing) => {
                format!("{{ ptr, {} }}", self.spec.isa.usize_llvm_ty())
            }
            MirType::PrimView(MirPrimViewKind::Inbox { .. }) => {
                format!("{{ ptr, ptr, {} }}", self.spec.isa.usize_llvm_ty())
            }
            MirType::Result { ok, .. } => {
                let ok_ty = if ok.is_void() {
                    "i32".to_string()
                } else {
                    self.llvm_ty(ok)
                };
                format!("{{ {ok_ty}, i32 }}")
            }
        }
    }

    pub(super) fn intern_string(&mut self, s: &str) -> usize {
        if let Some(&idx) = self.string_dedup.get(s) {
            return idx;
        }
        let idx = self.strings.len();
        self.strings.push(s.to_string());
        self.string_dedup.insert(s.to_string(), idx);
        idx
    }

    pub(super) fn intern_const_blob(&mut self, decl_body: &str) -> usize {
        if let Some(&idx) = self.const_blob_dedup.get(decl_body) {
            return idx;
        }
        let idx = self.const_blobs.len();
        self.const_blobs.push(decl_body.to_string());
        self.const_blob_dedup.insert(decl_body.to_string(), idx);
        idx
    }

    pub(super) fn eval_const_int(&self, expr: &MirExpr) -> i128 {
        match &expr.kind {
            MirExprKind::IntLit(v) => *v,
            MirExprKind::BoolLit(b) => i128::from(*b),
            MirExprKind::TicksLit(t) => *t as i128,
            MirExprKind::EnumVariant {
                enum_name,
                variant,
            } => self
                .enum_variants
                .get(&(enum_name.clone(), variant.clone()))
                .or_else(|| self.bare_variants.get(variant))
                .copied()
                .unwrap_or(0) as i128,
            MirExprKind::ConstRef(name) => self.consts.get(name).copied().unwrap_or(1),
            MirExprKind::Unary { op, expr: sub } => {
                let v = self.eval_const_int(sub);
                match op {
                    MirUnaryOp::Neg => -v,
                    MirUnaryOp::Not => i128::from(v == 0),
                    MirUnaryOp::BitNot => !v,
                    MirUnaryOp::Deref => v,
                }
            }
            MirExprKind::Binary { op, lhs, rhs } => {
                if lhs.ty.is_float() || rhs.ty.is_float() {
                    let lf = f64::from_bits(self.eval_const_float_bits(lhs));
                    let rf = f64::from_bits(self.eval_const_float_bits(rhs));
                    return match op {
                        MirBinOp::Eq => i128::from(lf == rf),
                        MirBinOp::Ne => i128::from(lf != rf),
                        MirBinOp::Lt => i128::from(lf < rf),
                        MirBinOp::Le => i128::from(lf <= rf),
                        MirBinOp::Gt => i128::from(lf > rf),
                        MirBinOp::Ge => i128::from(lf >= rf),
                        _ => 0,
                    };
                }
                let l = self.eval_const_int(lhs);
                let r = self.eval_const_int(rhs);
                match op {
                    MirBinOp::Add => l.wrapping_add(r),
                    MirBinOp::Sub => l.wrapping_sub(r),
                    MirBinOp::Mul => l.wrapping_mul(r),
                    MirBinOp::Div => {
                        if r != 0 {
                            l / r
                        } else {
                            0
                        }
                    }
                    MirBinOp::Rem => {
                        if r != 0 {
                            l % r
                        } else {
                            0
                        }
                    }
                    MirBinOp::BitOr => l | r,
                    MirBinOp::BitAnd => l & r,
                    MirBinOp::BitXor => l ^ r,
                    MirBinOp::Shl => {
                        let ptr_bits = self.spec.isa.ptr_bits();
                        let width = target_int_signed_and_bits(&lhs.ty, ptr_bits)
                            .map_or(64, |(_, b)| b);
                        let sh = (r as u32) & width.saturating_sub(1);
                        l << sh
                    }
                    MirBinOp::Shr => {
                        let ptr_bits = self.spec.isa.ptr_bits();
                        let width = target_int_signed_and_bits(&lhs.ty, ptr_bits)
                            .map_or(64, |(_, b)| b);
                        let sh = (r as u32) & width.saturating_sub(1);
                        if lhs.ty.is_unsigned() {
                            let masked = if lhs.ty == MirType::Bool {
                                (l as u128) & 1
                            } else {
                                match width {
                                    8 => (l as u8) as u128,
                                    16 => (l as u16) as u128,
                                    32 => (l as u32) as u128,
                                    64 => (l as u64) as u128,
                                    _ => l as u128,
                                }
                            };
                            (masked >> sh) as i128
                        } else {
                            let signed_l = match width {
                                8 => (l as i8) as i128,
                                16 => (l as i16) as i128,
                                32 => (l as i32) as i128,
                                64 => (l as i64) as i128,
                                _ => l,
                            };
                            signed_l >> sh
                        }
                    }
                    MirBinOp::Eq => i128::from(l == r),
                    MirBinOp::Ne => i128::from(l != r),
                    MirBinOp::Lt => i128::from(l < r),
                    MirBinOp::Le => i128::from(l <= r),
                    MirBinOp::Gt => i128::from(l > r),
                    MirBinOp::Ge => i128::from(l >= r),
                    MirBinOp::LogicalAnd => i128::from(l != 0 && r != 0),
                    MirBinOp::LogicalOr => i128::from(l != 0 || r != 0),
                }
            }
            MirExprKind::Cast {
                expr: sub,
                target_ty,
            } => {
                let ptr_bits = self.spec.isa.ptr_bits();
                if *target_ty == MirType::Bool {
                    if sub.ty.is_float() {
                        let f = f64::from_bits(self.eval_const_float_bits(sub));
                        return i128::from(f != 0.0);
                    }
                    return i128::from(self.eval_const_int(sub) != 0);
                }
                if sub.ty.is_float() {
                    let raw = f64::from_bits(self.eval_const_float_bits(sub));
                    let f = if sub.ty == MirType::F32 {
                        (raw as f32) as f64
                    } else {
                        raw
                    };
                    return saturating_float_to_int_i128(f, target_ty, ptr_bits);
                }
                mask_int_to_target_i128(self.eval_const_int(sub), target_ty, ptr_bits)
            }
            _ => 0,
        }
    }

    pub(super) fn eval_const_float_bits(&self, expr: &MirExpr) -> u64 {
        let raw_f = match &expr.kind {
            MirExprKind::FloatLit(bits) => f64::from_bits(*bits),
            MirExprKind::IntLit(v) => *v as f64,
            MirExprKind::ConstRef(name) => self
                .float_consts
                .get(name)
                .copied()
                .map(f64::from_bits)
                .or_else(|| self.consts.get(name).copied().map(|v| v as f64))
                .unwrap_or(0.0),
            MirExprKind::Unary {
                op: MirUnaryOp::Neg,
                expr: sub,
            } => -f64::from_bits(self.eval_const_float_bits(sub)),
            MirExprKind::Binary { op, lhs, rhs } => {
                let l = f64::from_bits(self.eval_const_float_bits(lhs));
                let r = f64::from_bits(self.eval_const_float_bits(rhs));
                match op {
                    MirBinOp::Add => l + r,
                    MirBinOp::Sub => l - r,
                    MirBinOp::Mul => l * r,
                    MirBinOp::Div => l / r,
                    MirBinOp::Rem => l % r,
                    _ => 0.0,
                }
            }
            MirExprKind::Cast { expr: sub, .. } => {
                if sub.ty.is_float() {
                    f64::from_bits(self.eval_const_float_bits(sub))
                } else {
                    self.eval_const_int(sub) as f64
                }
            }
            _ => 0.0,
        };
        if expr.ty == MirType::F32 {
            ((raw_f as f32) as f64).to_bits()
        } else {
            raw_f.to_bits()
        }
    }

    // ── Section Emitters ─────────────────────────────────────────────────────

    pub(super) fn emit_struct_types_ir(&self, out: &mut String) {
        let has_val_structs = self
            .struct_order
            .iter()
            .any(|name| !self.structs[name].decl.is_subsystem);
        if !has_val_structs {
            return;
        }
        let _ = writeln!(out, "; ── User Struct & Wire-Exact Packed Struct Types ──");
        for name in &self.struct_order {
            let si = &self.structs[name];
            if si.decl.is_subsystem {
                continue;
            }
            let fields_str: Vec<String> = if si.decl.fields.is_empty() {
                vec!["i8".to_string()]
            } else {
                si.decl.fields.iter().map(|f| self.llvm_ty(&f.ty)).collect()
            };
            let joined = fields_str.join(", ");
            if si.decl.packed {
                let _ = writeln!(out, "{} = type <{{ {joined} }}>", si.llvm_name);
            } else {
                let _ = writeln!(out, "{} = type {{ {joined} }}", si.llvm_name);
            }
        }
        let _ = writeln!(out);
    }

    pub(super) fn emit_mmio_host_buffers_ir(&self, out: &mut String) {
        if self.spec.isa != CpuIsa::HostX86_64Linux || self.module.mmio_instances.is_empty() {
            return;
        }
        let _ = writeln!(out, "; ── Host Simulation MMIO Backing Buffers ──");
        for inst in &self.module.mmio_instances {
            let layout_sz = self
                .mmio_layouts
                .get(&inst.layout)
                .map_or(0, |l| l.byte_size);
            let buf_len = inst.byte_size.max(layout_sz).max(4);
            let sym = format!("@__zev_mmio_buf_{}", sanitize_ident(&inst.name));
            let _ = writeln!(
                out,
                "{sym} = internal global [{buf_len} x i8] zeroinitializer, align 4"
            );
        }
        let _ = writeln!(out);
    }

    pub(super) fn emit_isr_state_and_selector_globals_ir(
        &mut self,
        out: &mut String,
    ) -> Result<(), CodegenError> {
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let mut emitted_state = BTreeSet::new();
        let mut state_lines = Vec::new();
        let mut selector_syms = BTreeSet::new();

        for vec in &self.module.irq_vectors.clone() {
            if let Some(sel) = &vec.selector_symbol {
                selector_syms.insert(sel.trim_start_matches('@').to_string());
            }
            for b in &vec.bindings {
                if let Some((sel, _)) = &b.cond_selector {
                    selector_syms.insert(sel.trim_start_matches('@').to_string());
                }
                for sv in &b.state_vars {
                    let sym = sanitize_ident(sv.storage_symbol.trim_start_matches('@'));
                    if emitted_state.insert(sym.clone()) {
                        let lty = self.llvm_ty(&sv.ty);
                        let (_, align) = self.size_align(&sv.ty, ptr_bytes);
                        let init_val = self.const_expr_to_llvm(&sv.init, &sv.ty)?;
                        state_lines.push(format!(
                            "@{sym} = internal global {lty} {init_val}, align {}",
                            align.max(1)
                        ));
                    }
                }
            }
        }

        fn scan_stmt_selectors(stmts: &[MirStmt], out: &mut BTreeSet<String>) {
            for s in stmts {
                match s {
                    MirStmt::SetIrqSelector { symbol, .. } => {
                        out.insert(symbol.trim_start_matches('@').to_string());
                    }
                    MirStmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        scan_stmt_selectors(then_body, out);
                        scan_stmt_selectors(else_body, out);
                    }
                    MirStmt::While { body, .. } => scan_stmt_selectors(body, out),
                    MirStmt::Switch { arms, default, .. } => {
                        for (_, ab) in arms {
                            scan_stmt_selectors(ab, out);
                        }
                        scan_stmt_selectors(default, out);
                    }
                    _ => {}
                }
            }
        }
        for step in &self.module.boot.steps {
            if let MirBootStep::Stmt(s) = step {
                scan_stmt_selectors(std::slice::from_ref(s), &mut selector_syms);
            }
        }
        for f in &self.module.functions {
            if let MirFnBody::Direct(stmts) = &f.body {
                scan_stmt_selectors(stmts, &mut selector_syms);
            }
        }

        if !state_lines.is_empty() || !selector_syms.is_empty() {
            let _ = writeln!(out, "; ── Per-Binding ISR Static State & Conditional IRQ Selectors ──");
            for line in state_lines {
                let _ = writeln!(out, "{line}");
            }
            for raw_sel in selector_syms {
                let sym = sanitize_ident(&raw_sel);
                let _ = writeln!(out, "@{sym} = internal global i8 0, align 1");
            }
            let _ = writeln!(out);
        }
        Ok(())
    }

    pub(super) fn const_expr_to_llvm(
        &mut self,
        expr: &MirExpr,
        target_ty: &MirType,
    ) -> Result<String, CodegenError> {
        if let MirExprKind::ConstRef(cname) = &expr.kind {
            if let Some(init) = self
                .module
                .consts
                .iter()
                .find(|c| c.name == *cname)
                .map(|c| c.init.clone())
            {
                return self.const_expr_to_llvm(&init, target_ty);
            }
        }
        let raw_lit = match &expr.kind {
            MirExprKind::IntLit(v) => Some(*v),
            MirExprKind::TicksLit(t) => Some(i128::from(*t)),
            _ => None,
        };
        if let Some(lit) = raw_lit {
            if matches!(
                target_ty,
                MirType::U8
                    | MirType::I8
                    | MirType::U16
                    | MirType::I16
                    | MirType::U32
                    | MirType::I32
                    | MirType::Enum(_)
                    | MirType::U64
                    | MirType::I64
                    | MirType::Usize
                    | MirType::Isize
                    | MirType::EndianInt { .. }
            ) {
                let target_lty = self.llvm_ty(target_ty);
                super::format_checked_llvm_int_const(&target_lty, lit)?;
            }
        }
        let ptr_bits = self.spec.isa.ptr_bits();
        Ok(match target_ty {
            MirType::Bool => {
                if self.eval_const_int(expr) != 0 {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
            MirType::U8 => {
                super::format_checked_llvm_int_const("i8", i128::from(self.eval_const_int(expr) as u8))?
            }
            MirType::I8 => {
                super::format_checked_llvm_int_const("i8", i128::from(self.eval_const_int(expr) as i8))?
            }
            MirType::U16
            | MirType::EndianInt {
                bits: 16,
                signed: false,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as u16;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                super::format_checked_llvm_int_const("i16", i128::from(v))?
            }
            MirType::I16
            | MirType::EndianInt {
                bits: 16,
                signed: true,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as i16;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                super::format_checked_llvm_int_const("i16", i128::from(v))?
            }
            MirType::U32
            | MirType::EndianInt {
                bits: 32,
                signed: false,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as u32;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                super::format_checked_llvm_int_const("i32", i128::from(v))?
            }
            MirType::I32
            | MirType::Enum(_)
            | MirType::EndianInt {
                bits: 32,
                signed: true,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as i32;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                super::format_checked_llvm_int_const("i32", i128::from(v))?
            }
            MirType::U64
            | MirType::EndianInt {
                bits: 64,
                signed: false,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as u64;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                super::format_checked_llvm_int_const("i64", i128::from(v))?
            }
            MirType::I64
            | MirType::EndianInt {
                bits: 64,
                signed: true,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as i64;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                super::format_checked_llvm_int_const("i64", i128::from(v))?
            }
            MirType::Usize => {
                if ptr_bits == 64 {
                    super::format_checked_llvm_int_const("i64", i128::from(self.eval_const_int(expr) as u64))?
                } else if ptr_bits == 16 {
                    super::format_checked_llvm_int_const("i16", i128::from(self.eval_const_int(expr) as u16))?
                } else {
                    super::format_checked_llvm_int_const("i32", i128::from(self.eval_const_int(expr) as u32))?
                }
            }
            MirType::Isize => {
                if ptr_bits == 64 {
                    super::format_checked_llvm_int_const("i64", i128::from(self.eval_const_int(expr) as i64))?
                } else if ptr_bits == 16 {
                    super::format_checked_llvm_int_const("i16", i128::from(self.eval_const_int(expr) as i16))?
                } else {
                    super::format_checked_llvm_int_const("i32", i128::from(self.eval_const_int(expr) as i32))?
                }
            }
            MirType::F32 => {
                let f = f64::from_bits(self.eval_const_float_bits(expr)) as f32;
                let norm = if f.is_nan() {
                    0x7FF8_0000_0000_0000u64
                } else {
                    (f as f64).to_bits()
                };
                format!("0x{norm:016X}")
            }
            MirType::F64 => {
                let bits = self.eval_const_float_bits(expr);
                format!("0x{bits:016X}")
            }
            MirType::Ptr(_) | MirType::MmioPtr(_) | MirType::FnPtr { .. } | MirType::Str => {
                let inner = match &expr.kind {
                    MirExprKind::Cast { expr: sub, .. } => sub.as_ref(),
                    _ => expr,
                };
                if let MirExprKind::ConstRef(cname) = &inner.kind {
                    if let Some(init) = self
                        .module
                        .consts
                        .iter()
                        .find(|c| c.name == *cname)
                        .map(|c| c.init.clone())
                    {
                        return self.const_expr_to_llvm(&init, target_ty);
                    }
                }
                match &inner.kind {
                    MirExprKind::GlobalPrimAddr(name) => {
                        format!("@__syn_prim_{}", sanitize_ident(name))
                    }
                    MirExprKind::SubsystemAddr(name) => {
                        format!("@__zev_subsys_{}", sanitize_ident(name))
                    }
                    MirExprKind::AddrOfPlace(place) => {
                        self.const_place_to_llvm_ptr(place).unwrap_or_else(|| "null".to_string())
                    }
                    MirExprKind::MmioAddr(inst) => {
                        if self.spec.isa == CpuIsa::HostX86_64Linux {
                            format!("@__zev_mmio_buf_{}", sanitize_ident(inst))
                        } else {
                            let addr = self
                                .mmio_instances
                                .get(inst)
                                .map_or(0, |m| m.base_addr as i32);
                            format!("inttoptr (i32 {addr} to ptr)")
                        }
                    }
                    MirExprKind::StringLit(s) => {
                        let idx = self.intern_string(s);
                        format!("@.str.{idx}")
                    }
                    MirExprKind::FnAddr(fname) => {
                        if let Some(sig) = self.fn_sigs.get(fname) {
                            sig.symbol.clone()
                        } else {
                            format!("@{}", sanitize_ident(fname))
                        }
                    }
                    MirExprKind::IntLit(0) => "null".to_string(),
                    MirExprKind::IntLit(v) => {
                        let dst_ptr = if matches!(target_ty, MirType::FnPtr { .. }) {
                            self.spec.isa.fn_ptr_llvm_ty()
                        } else {
                            "ptr"
                        };
                        if ptr_bits == 64 {
                            let s = super::format_checked_llvm_int_const("i64", i128::from(*v as i64))?;
                            format!("inttoptr (i64 {s} to {dst_ptr})")
                        } else if ptr_bits == 16 {
                            let s = super::format_checked_llvm_int_const("i16", i128::from(*v as i16))?;
                            format!("inttoptr (i16 {s} to {dst_ptr})")
                        } else {
                            let s = super::format_checked_llvm_int_const("i32", i128::from(*v as i32))?;
                            format!("inttoptr (i32 {s} to {dst_ptr})")
                        }
                    }
                    _ => "null".to_string(),
                }
            }
            MirType::Slice(elem) => {
                let usize_ty = self.spec.isa.usize_llvm_ty().to_string();
                match &expr.kind {
                    MirExprKind::Cast { expr: sub, .. } => {
                        return self.const_expr_to_llvm(sub, target_ty);
                    }
                    MirExprKind::SubSlice {
                        base,
                        start,
                        end,
                        elem_ty,
                        base_is_array,
                    } => {
                        let st = self.eval_const_int(start).max(0) as usize;
                        let en = self.eval_const_int(end).max(0) as usize;
                        if let MirExprKind::ConstRef(cname) = &base.kind {
                            if let Some((sym, MirType::Array { len, .. })) =
                                self.rodata_consts.get(cname).cloned()
                            {
                                let cl_st = st.min(len);
                                let cl_en = en.min(len).max(cl_st);
                                let ety = self.llvm_ty(elem_ty);
                                return Ok(format!(
                                    "{{ ptr getelementptr inbounds ([{len} x {ety}], ptr {sym}, {usize_ty} 0, {usize_ty} {cl_st}), {usize_ty} {} }}",
                                    cl_en - cl_st
                                ));
                            }
                        }
                        if *base_is_array {
                            let resolved_base = if let MirExprKind::ConstRef(cname) = &base.kind {
                                self.module
                                    .consts
                                    .iter()
                                    .find(|c| c.name == *cname)
                                    .map(|c| c.init.clone())
                                    .unwrap_or_else(|| base.as_ref().clone())
                            } else {
                                base.as_ref().clone()
                            };
                            if let MirExprKind::ArrayLit { elems, .. } = resolved_base.kind {
                                let len = elems.len();
                                let cl_st = st.min(len);
                                let cl_en = en.min(len).max(cl_st);
                                if cl_en == cl_st {
                                    return Ok(format!("{{ ptr null, {usize_ty} 0 }}"));
                                }
                                let ety = self.llvm_ty(elem);
                                let (_, eal) = self.size_align(elem, (ptr_bits / 8) as usize);
                                let mut parts = Vec::with_capacity(len);
                                for el in &elems {
                                    let cv = self.const_expr_to_llvm(el, elem)?;
                                    parts.push(format!("{ety} {cv}"));
                                }
                                let decl_body = format!(
                                    "private unnamed_addr constant [{len} x {ety}] [{}], align {eal}",
                                    parts.join(", ")
                                );
                                let idx = self.intern_const_blob(&decl_body);
                                return Ok(format!(
                                    "{{ ptr getelementptr inbounds ([{len} x {ety}], ptr @.cblob.{idx}, {usize_ty} 0, {usize_ty} {cl_st}), {usize_ty} {} }}",
                                    cl_en - cl_st
                                ));
                            }
                        }
                        "zeroinitializer".to_string()
                    }
                    MirExprKind::StringLit(s) => {
                        let idx = self.intern_string(s);
                        let slen = zev_ast::decoded_string_byte_len(s);
                        format!("{{ ptr @.str.{idx}, {usize_ty} {slen} }}")
                    }
                    MirExprKind::ArrayLit { elems, .. } => {
                        let n = elems.len();
                        if n == 0 {
                            format!("{{ ptr null, {usize_ty} 0 }}")
                        } else {
                            let ety = self.llvm_ty(elem);
                            let (_, eal) = self.size_align(elem, (ptr_bits / 8) as usize);
                            let mut parts = Vec::with_capacity(n);
                            for el in elems {
                                let cv = self.const_expr_to_llvm(el, elem)?;
                                parts.push(format!("{ety} {cv}"));
                            }
                            let decl_body = format!(
                                "private unnamed_addr constant [{n} x {ety}] [{}], align {eal}",
                                parts.join(", ")
                            );
                            let idx = self.intern_const_blob(&decl_body);
                            format!("{{ ptr @.cblob.{idx}, {usize_ty} {n} }}")
                        }
                    }
                    MirExprKind::ArrayToSlice {
                        array_place, len, ..
                    } => {
                        if let Some(aptr) = self.const_place_to_llvm_ptr(array_place) {
                            format!("{{ ptr {aptr}, {usize_ty} {len} }}")
                        } else {
                            "zeroinitializer".to_string()
                        }
                    }
                    _ => "zeroinitializer".to_string(),
                }
            }
            MirType::Array { elem, len } => {
                let n = (*len).max(1);
                if let MirExprKind::ArrayLit { elems, .. } = &expr.kind {
                    let ety = self.llvm_ty(elem);
                    let mut parts = Vec::with_capacity(n);
                    for i in 0..n {
                        if let Some(el) = elems.get(i) {
                            let cv = self.const_expr_to_llvm(el, elem)?;
                            parts.push(format!("{ety} {cv}"));
                        } else {
                            parts.push(format!("{ety} zeroinitializer"));
                        }
                    }
                    format!("[{}]", parts.join(", "))
                } else {
                    "zeroinitializer".to_string()
                }
            }
            MirType::Struct(sname) => {
                if let MirExprKind::StructLit { fields, .. } = &expr.kind {
                    if let Some(si) = self.structs.get(sname).cloned() {
                        if si.decl.fields.is_empty() {
                            return Ok("zeroinitializer".to_string());
                        }
                        let by_field: HashMap<&str, &MirExpr> = fields
                            .iter()
                            .map(|(fname, fexpr, _)| (fname.as_str(), fexpr))
                            .collect();
                        let mut parts = Vec::with_capacity(si.decl.fields.len());
                        for fld in &si.decl.fields {
                            let fty = self.llvm_ty(&fld.ty);
                            if let Some(&fexpr) = by_field.get(fld.name.as_str()) {
                                let cv = self.const_expr_to_llvm(fexpr, &fld.ty)?;
                                parts.push(format!("{fty} {cv}"));
                            } else {
                                parts.push(format!("{fty} zeroinitializer"));
                            }
                        }
                        if si.decl.packed {
                            format!("<{{ {} }}>", parts.join(", "))
                        } else {
                            format!("{{ {} }}", parts.join(", "))
                        }
                    } else {
                        "zeroinitializer".to_string()
                    }
                } else {
                    "zeroinitializer".to_string()
                }
            }
            MirType::PrimView(vk) => {
                if let MirExprKind::MakePrimView { ptr, cap, .. } = &expr.kind {
                    let usize_ty = self.spec.isa.usize_llvm_ty();
                    let pconst = self.const_expr_to_llvm(ptr, &ptr.ty)?;
                    let n = (*cap).max(1);
                    match vk {
                        MirPrimViewKind::ByteRing => format!("{{ ptr {pconst}, {usize_ty} {n} }}"),
                        MirPrimViewKind::Inbox { elem } => {
                            let pty = format!(
                                "%prim.{}",
                                sanitize_ident(
                                    &MirPrimKind::Inbox {
                                        elem: elem.clone(),
                                        cap: n,
                                    }
                                    .mangle_name()
                                )
                            );
                            format!(
                                "{{ ptr {pconst}, ptr getelementptr inbounds ({pty}, ptr {pconst}, i32 0, i32 6, i32 0), {usize_ty} {n} }}"
                            )
                        }
                    }
                } else {
                    "zeroinitializer".to_string()
                }
            }
            _ => "zeroinitializer".to_string(),
        })
    }

    pub(super) fn const_place_to_llvm_ptr(&self, place: &MirPlace) -> Option<String> {
        match place {
            MirPlace::Deref(ptr_expr, _) => match &ptr_expr.kind {
                MirExprKind::SubsystemAddr(name) => {
                    Some(format!("@__zev_subsys_{}", sanitize_ident(name)))
                }
                MirExprKind::GlobalPrimAddr(name) => {
                    Some(format!("@__syn_prim_{}", sanitize_ident(name)))
                }
                MirExprKind::AddrOfPlace(inner) => self.const_place_to_llvm_ptr(inner),
                _ => None,
            },
            MirPlace::Field {
                base,
                struct_name,
                field,
                ..
            } => {
                let bptr = self.const_place_to_llvm_ptr(base)?;
                let si = self.structs.get(struct_name)?;
                let (fidx, _) = si.field_indices.get(field)?;
                Some(format!(
                    "getelementptr inbounds ({}, ptr {bptr}, i32 0, i32 {fidx})",
                    si.llvm_name
                ))
            }
            MirPlace::Index {
                base,
                index,
                elem_ty,
                is_slice: false,
            } => {
                let bptr = self.const_place_to_llvm_ptr(base)?;
                let idx = self.eval_const_int(index);
                let ety = self.llvm_ty(elem_ty);
                Some(format!("getelementptr ({ety}, ptr {bptr}, i32 {idx})"))
            }
            _ => None,
        }
    }

    pub(super) fn struct_needs_const_init(&self, struct_name: &str) -> bool {
        let Some(si) = self.structs.get(struct_name) else {
            return false;
        };
        si.decl.fields.iter().any(|f| match &f.ty {
            MirType::Primitive(MirPrimKind::Signal | MirPrimKind::PingPong { .. }) => true,
            MirType::Struct(sub) => self.struct_needs_const_init(sub),
            _ => false,
        })
    }

    pub(super) fn subsystem_zero_init_const(&self, struct_name: &str, next_pp_bit: &mut u8) -> String {
        if !self.struct_needs_const_init(struct_name) {
            return "zeroinitializer".to_string();
        }
        let Some(si) = self.structs.get(struct_name) else {
            return "zeroinitializer".to_string();
        };
        if si.llvm_fields.is_empty() {
            return "zeroinitializer".to_string();
        }
        let parts: Vec<String> = si
            .llvm_fields
            .iter()
            .map(|(decl_idx, lty)| match decl_idx {
                Some(idx) => {
                    let f = &si.decl.fields[*idx];
                    match &f.ty {
                        MirType::Primitive(MirPrimKind::Signal) => {
                            format!("{lty} {{ i8 0, i8 127, i32 0 }}")
                        }
                        MirType::Primitive(MirPrimKind::PingPong { elem, cap }) => {
                            let ety = self.llvm_ty(elem);
                            let c = (*cap).max(1);
                            let lbit = *next_pp_bit;
                            *next_pp_bit = next_pp_bit.saturating_add(1).min(31);
                            format!(
                                "{lty} {{ i8 0, i8 0, i8 0, i8 {lbit}, i32 0, [{c} x {ety}] zeroinitializer, [{c} x {ety}] zeroinitializer }}"
                            )
                        }
                        MirType::Struct(sub) => {
                            format!("{lty} {}", self.subsystem_zero_init_const(sub, next_pp_bit))
                        }
                        _ => format!("{lty} zeroinitializer"),
                    }
                }
                None => format!("{lty} zeroinitializer"),
            })
            .collect();
        format!("{{ {} }}", parts.join(", "))
    }

    pub(super) fn wrap_const_ptr_as_view(&self, raw_ptr: &str, pk: &MirPrimKind) -> Option<String> {
        let usize_ty = self.spec.isa.usize_llvm_ty();
        match pk {
            MirPrimKind::ByteRing { cap } => {
                let n = (*cap).max(1);
                Some(format!("{{ ptr, {usize_ty} }} {{ ptr {raw_ptr}, {usize_ty} {n} }}"))
            }
            MirPrimKind::Inbox { elem, cap } => {
                let n = (*cap).max(1);
                let pty = format!(
                    "%prim.{}",
                    sanitize_ident(
                        &MirPrimKind::Inbox {
                            elem: elem.clone(),
                            cap: n,
                        }
                        .mangle_name()
                    )
                );
                Some(format!(
                    "{{ ptr, ptr, {usize_ty} }} {{ ptr {raw_ptr}, ptr getelementptr inbounds ({pty}, ptr {raw_ptr}, i32 0, i32 6, i32 0), {usize_ty} {n} }}"
                ))
            }
            _ => None,
        }
    }

    pub(super) fn emit_from_bytes_helpers_ir(&self, out: &mut String) {
        if self.struct_order.is_empty() {
            return;
        }
        let usize_ty = self.spec.isa.usize_llvm_ty();
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let tmp_al = if self.spec.isa.is_avr() {
            1
        } else {
            4
        };
        for name in &self.struct_order {
            let si = &self.structs[name];
            if si.decl.is_subsystem {
                continue;
            }
            let sname = sanitize_ident(name);
            let sty = &si.llvm_name;
            let sz = si.byte_size;
            let _ = writeln!(
                out,
                r#"define {{ {sty}, i32 }} @__syn_from_bytes_{sname}({{ ptr, {usize_ty} }} %slice){fn_as} nounwind{fn_sz} {{
entry:
  %tmp = alloca {sty}, align {tmp_al}
  %ptr = extractvalue {{ ptr, {usize_ty} }} %slice, 0
  %len = extractvalue {{ ptr, {usize_ty} }} %slice, 1
  %short = icmp ult {usize_ty} %len, {sz}
  br i1 %short, label %err_bb, label %ok_bb
err_bb:
  ret {{ {sty}, i32 }} {{ {sty} zeroinitializer, i32 1 }}
ok_bb:
  call void @llvm.memcpy.p0.p0.i32(ptr %tmp, ptr %ptr, i32 {sz}, i1 false)
  %val = load {sty}, ptr %tmp, align 1
  %r0 = insertvalue {{ {sty}, i32 }} poison, {sty} %val, 0
  %r1 = insertvalue {{ {sty}, i32 }} %r0, i32 0, 1
  ret {{ {sty}, i32 }} %r1
}}
"#
            );
        }
    }

    pub(super) fn emit_formatted_enum_names_ir(&mut self, out: &mut String) {
        if self.module.formatted_enums.is_empty() {
            return;
        }
        let fn_sz = self.opts.fn_size_attrs().to_string();
        let fn_as = self.fn_as();
        let formatted: Vec<String> = self.module.formatted_enums.clone();
        let _ = writeln!(out, "; ── On-Demand Enum Variant-Name `.rodata` Lookup Helpers ──");
        for ename in &formatted {
            let Some(en) = self.module.enums.iter().find(|e| &e.name == ename).cloned() else {
                continue;
            };
            let eid = sanitize_ident(&en.name);
            let unk_idx = self.intern_string(&format!("{}(?)", en.name));
            let mut seen_discs = BTreeSet::new();
            let mut arms: Vec<(i64, usize)> = Vec::new();
            for (vname, disc) in &en.variants {
                if seen_discs.insert(*disc) {
                    let sidx = self.intern_string(vname);
                    arms.push((*disc, sidx));
                }
            }
            let _ = writeln!(
                out,
                "define ptr @__syn_enum_name_{eid}(i32 %v){fn_as} nounwind{fn_sz} {{"
            );
            let _ = writeln!(out, "entry:");
            let _ = writeln!(out, "  switch i32 %v, label %def [");
            for (idx, (disc, _)) in arms.iter().enumerate() {
                let _ = writeln!(out, "    i32 {disc}, label %arm_{idx}");
            }
            let _ = writeln!(out, "  ]");
            for (idx, (_, sidx)) in arms.iter().enumerate() {
                let _ = writeln!(out, "arm_{idx}:");
                let _ = writeln!(out, "  ret ptr @.str.{sidx}");
            }
            let _ = writeln!(out, "def:");
            let _ = writeln!(out, "  ret ptr @.str.{unk_idx}");
            let _ = writeln!(out, "}}\n");
        }
    }

    pub(super) fn emit_primitives_ir(&mut self, out: &mut String) -> Result<(), CodegenError> {
        if self.distinct_prims.is_empty()
            && self.module.primitives.is_empty()
            && !self.uses_pool_lease
            && self.module.subsystems.is_empty()
            && self.distinct_inbox_views.is_empty()
        {
            return Ok(());
        }
        let fn_as = self.fn_as();
        let call_kw = self.call_kw();
        let vis = self.vis.clone();
        let _ = writeln!(out, "; ── Static OS & Zero-Copy DMA Primitive Endpoints ──");
        if self.uses_pool_lease {
            let _ = writeln!(out, "%prim.pool_hdr = type {{ i32, i32, ptr, i8, i8 }}");
        }
        for (mangled, pk) in &self.distinct_prims {
            let id = sanitize_ident(mangled);
            match pk {
                MirPrimKind::Signal => {
                    let _ = writeln!(out, "%prim.{id} = type {{ i8, i8, i32 }}");
                }
                MirPrimKind::WaitQueue => {
                    let _ = writeln!(out, "%prim.{id} = type {{ i32 }}");
                }
                MirPrimKind::Watch { elem } => {
                    let ety = self.llvm_ty(elem);
                    let _ = writeln!(out, "%prim.{id} = type {{ {ety}, i32, i32 }}");
                }
                MirPrimKind::Guarded { elem } => {
                    let ety = self.llvm_ty(elem);
                    let _ = writeln!(out, "%prim.{id} = type {{ {ety} }}");
                }
                MirPrimKind::Inbox { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1);
                    let _ = writeln!(
                        out,
                        "%prim.{id} = type {{ i32, i32, i32, i32, i32, [{c} x i8], [{c} x {ety}] }}"
                    );
                }
                MirPrimKind::PingPong { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1);
                    let _ = writeln!(
                        out,
                        "%prim.{id} = type {{ i8, i8, i8, i8, i32, [{c} x {ety}], [{c} x {ety}] }}"
                    );
                }
                MirPrimKind::Pool { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1);
                    let _ = writeln!(out, "%prim.{id} = type {{ %prim.pool_hdr, [{c} x {ety}] }}");
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let (hdr_sz, _) = self.pool_hdr_llvm_size();
                    let stride = align_up(esz.max(1), eal.max(32));
                    let pad = stride.saturating_sub(esz);
                    if pad > 0 {
                        let _ = writeln!(
                            out,
                            "%prim.{id}_dma_slot = type {{ {ety}, [{pad} x i8] }}"
                        );
                    } else {
                        let _ = writeln!(out, "%prim.{id}_dma_slot = type {{ {ety} }}");
                    }
                    let hdr_pad = 32usize.saturating_sub(hdr_sz);
                    if hdr_pad > 0 {
                        let _ = writeln!(
                            out,
                            "%prim.{id}_dma = type {{ %prim.pool_hdr, [{hdr_pad} x i8], [{c} x %prim.{id}_dma_slot] }}"
                        );
                    } else {
                        let _ = writeln!(
                            out,
                            "%prim.{id}_dma = type {{ %prim.pool_hdr, [{c} x %prim.{id}_dma_slot] }}"
                        );
                    }
                }
                MirPrimKind::DmaSlot { .. } => {
                    let _ = writeln!(out, "%prim.{id} = type {{ ptr, ptr, i8, i8 }}");
                }
                MirPrimKind::ByteRing { cap } => {
                    let c = (*cap).max(1);
                    let _ = writeln!(
                        out,
                        "%prim.{id} = type {{ i32, i32, i32, i32, i32, i32, i32, i8, [{c} x i8] }}"
                    );
                }
            }
        }
        if self.uses_byte_ring_view {
            let _ = writeln!(
                out,
                "%prim.byte_ring_hdr = type {{ i32, i32, i32, i32, i32, i32, i32, i8, [0 x i8] }}"
            );
        }
        if !self.distinct_inbox_views.is_empty() {
            let _ = writeln!(out, "%prim.inbox_hdr = type {{ i32, i32, i32, i32, i32, [0 x i8] }}");
        }
        for name in &self.struct_order {
            let si = &self.structs[name];
            if !si.decl.is_subsystem {
                continue;
            }
            let fields_str: Vec<String> = if si.llvm_fields.is_empty() {
                vec!["i8".to_string()]
            } else {
                si.llvm_fields.iter().map(|(_, lty)| lty.clone()).collect()
            };
            let joined = fields_str.join(", ");
            let _ = writeln!(out, "{} = type {{ {joined} }}", si.llvm_name);
        }
        let _ = writeln!(out);

        // Emit static subsystem instances
        let mut next_pp_bit: u8 = self
            .module
            .primitives
            .iter()
            .filter_map(|p| p.sleep_lock_bit)
            .map(|b| b.saturating_add(1))
            .max()
            .unwrap_or(0)
            .min(31);
        for sub in &self.module.subsystems {
            let gsym = format!("@__zev_subsys_{}", sanitize_ident(&sub.name));
            let si = &self.structs[&sub.struct_name];
            let init = self.subsystem_zero_init_const(&sub.struct_name, &mut next_pp_bit);
            let align = if sub.dma_safe {
                si.align.max(16)
            } else {
                si.align.max(4)
            };
            let _ = writeln!(
                out,
                "{gsym} = internal global {} {init}, align {align}",
                si.llvm_name
            );
        }

        // Emit static global primitive instances
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let prims = self.module.primitives.clone();
        for p in &prims {
            if p.dead_stripped {
                let _ = writeln!(out, "; dead watch stripped: @__syn_prim_{}", sanitize_ident(&p.name));
                continue;
            }
            let gsym = format!("@__syn_prim_{}", sanitize_ident(&p.name));
            let pty = format!("%prim.{}", sanitize_ident(&p.kind.mangle_name()));
            let align = if p.dma_safe {
                if matches!(p.kind, MirPrimKind::Pool { .. }) {
                    32
                } else {
                    16
                }
            } else {
                4
            };
            match &p.kind {
                MirPrimKind::Signal => {
                    let _ = writeln!(out, "{gsym} = global {pty} {{ i8 0, i8 127, i32 0 }}, align {align}");
                }
                MirPrimKind::Watch { elem } => {
                    let ety = self.llvm_ty(elem);
                    let (_, elem_align) = self.size_align(elem, ptr_bytes);
                    let watch_align = align.max(elem_align);
                    if let Some(init_expr) = p.init.clone() {
                        let iv = self.const_expr_to_llvm(&init_expr, elem)?;
                        let _ = writeln!(
                            out,
                            "{gsym} = global {pty} {{ {ety} {iv}, i32 0, i32 0 }}, align {watch_align}"
                        );
                    } else {
                        let _ = writeln!(
                            out,
                            "{gsym} = global {pty} zeroinitializer, align {watch_align}"
                        );
                    }
                }
                MirPrimKind::Guarded { elem } => {
                    let ety = self.llvm_ty(elem);
                    let (_, elem_align) = self.size_align(elem, ptr_bytes);
                    let guarded_align = align.max(elem_align);
                    if let Some(init_expr) = p.init.clone() {
                        let iv = self.const_expr_to_llvm(&init_expr, elem)?;
                        let _ = writeln!(
                            out,
                            "{gsym} = global {pty} {{ {ety} {iv} }}, align {guarded_align}"
                        );
                    } else {
                        let _ = writeln!(
                            out,
                            "{gsym} = global {pty} zeroinitializer, align {guarded_align}"
                        );
                    }
                }
                MirPrimKind::Inbox { .. }
                | MirPrimKind::ByteRing { .. }
                | MirPrimKind::WaitQueue => {
                    let _ = writeln!(out, "{gsym} = global {pty} zeroinitializer, align {align}");
                }
                MirPrimKind::PingPong { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1);
                    let lbit = p.sleep_lock_bit.unwrap_or(0);
                    let _ = writeln!(
                        out,
                        "{gsym} = global {pty} {{ i8 0, i8 0, i8 0, i8 {lbit}, i32 0, [{c} x {ety}] zeroinitializer, [{c} x {ety}] zeroinitializer }}, align {align}"
                    );
                }
                MirPrimKind::Pool { .. } => {
                    let gty = if p.dma_safe { format!("{pty}_dma") } else { pty };
                    let _ = writeln!(out, "{gsym} = global {gty} zeroinitializer, align {align}");
                }
                MirPrimKind::DmaSlot { .. } => {
                    let _ = writeln!(out, "{gsym} = global {pty} zeroinitializer, align {align}");
                }
            }
        }
        let _ = writeln!(out);

        // Emit shared pool lease release helper if pools, dma_slots, or Lease<T> values exist.
        let fn_sz = self.opts.fn_size_attrs();
        if self.uses_pool_lease {
            let (crit_enter, crit_exit) = crit_pair(vis.any_pool_or_lease_contended(), call_kw);
            let _ = write!(
                out,
                r#"define void @__syn_pool_release(ptr %pool_hdr, i8 %slot_idx, i8 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %is_null = icmp eq ptr %pool_hdr, null
  br i1 %is_null, label %done_null, label %chk_bounds
done_null:
  ret void
chk_bounds:
{crit_enter}  %cap_p = getelementptr inbounds %prim.pool_hdr, ptr %pool_hdr, i32 0, i32 4
  %cap = load i8, ptr %cap_p, align 1
  %valid = icmp ult i8 %slot_idx, %cap
  br i1 %valid, label %chk_bit, label %unlock_exit
chk_bit:
  %idx32 = zext i8 %slot_idx to i32
  %bit = shl i32 1, %idx32
  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %pool_hdr, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
  %was = and i32 %fm, %bit
  %is_used = icmp eq i32 %was, 0
  br i1 %is_used, label %free_slot, label %unlock_exit
free_slot:
  %new_fm = or i32 %fm, %bit
  store i32 %new_fm, ptr %fm_p, align 4
  %dma_p = getelementptr inbounds %prim.pool_hdr, ptr %pool_hdr, i32 0, i32 3
  %dma = load i8, ptr %dma_p, align 1
  %dma_lock = and i8 %dma, 1
  %is_dma = icmp ne i8 %dma_lock, 0
  br i1 %is_dma, label %dec_dma, label %wake
dec_dma:
  %dl = load volatile i32, ptr @__syn_dma_locks, align 4
  %has_dl = icmp ugt i32 %dl, 0
  br i1 %has_dl, label %do_dec, label %wake
do_dec:
  %dl1 = sub i32 %dl, 1
  store volatile i32 %dl1, ptr @__syn_dma_locks, align 4
  br label %wake
wake:
  %b32 = sext i8 %boost to i32
  %wm_p = getelementptr inbounds %prim.pool_hdr, ptr %pool_hdr, i32 0, i32 1
  {call_kw} void @__syn_wake_all_mask(ptr %wm_p, i32 %b32)
  br label %unlock_exit
unlock_exit:
{crit_exit}  ret void
}}

"#
            );
        }
        // Emit typed primitive runtime helpers for each distinct primitive kind
        for (mangled, pk) in &self.distinct_prims {
            let id = sanitize_ident(mangled);
            let pty = format!("%prim.{id}");
            let (crit_enter, crit_exit) = crit_pair(vis.is_prim_family_contended(pk), call_kw);
            let ptr_al = self.ptr_align();
            let i32_al = self.llvm_scalar_abi_align(4);
            match pk {
                MirPrimKind::WaitQueue => {
                    let _ = writeln!(
                        out,
                        r#"define void @__syn_wq_prepare(ptr %wq){fn_as} nounwind{fn_sz} {{
entry:
  %pm = {call_kw} i32 @__syn_crit_enter()
  %t = load ptr, ptr @__syn_current_task, align {ptr_al}
  %isnull = icmp eq ptr %t, null
  br i1 %isnull, label %done, label %set_bit
set_bit:
  %sl_p = getelementptr inbounds %SynTask, ptr %t, i32 0, i32 9
  %sl8 = load i8, ptr %sl_p, align 1
  %sl32 = zext i8 %sl8 to i32
  %sh = and i32 %sl32, 31
  %bit = shl i32 1, %sh
  %m = load volatile i32, ptr %wq, align {i32_al}
  %nm = or i32 %m, %bit
  store volatile i32 %nm, ptr %wq, align {i32_al}
  br label %done
done:
  {call_kw} void @__syn_crit_exit(i32 %pm)
  ret void
}}

define void @__syn_wq_cancel(ptr %wq){fn_as} nounwind{fn_sz} {{
entry:
  %pm = {call_kw} i32 @__syn_crit_enter()
  %t = load ptr, ptr @__syn_current_task, align {ptr_al}
  %isnull = icmp eq ptr %t, null
  br i1 %isnull, label %done, label %clr_bit
clr_bit:
  %sl_p = getelementptr inbounds %SynTask, ptr %t, i32 0, i32 9
  %sl8 = load i8, ptr %sl_p, align 1
  %sl32 = zext i8 %sl8 to i32
  %sh = and i32 %sl32, 31
  %bit = shl i32 1, %sh
  %mask = xor i32 %bit, -1
  %m = load volatile i32, ptr %wq, align {i32_al}
  %nm = and i32 %m, %mask
  store volatile i32 %nm, ptr %wq, align {i32_al}
  br label %done
done:
  {call_kw} void @__syn_crit_exit(i32 %pm)
  ret void
}}
"#
                    );
                }
                MirPrimKind::Signal => {
                    let _ = writeln!(
                        out,
                        r#"define void @__syn_sig_notify_{id}(ptr %s, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fp = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  store i8 1, ptr %fp, align 1
  %has_b = icmp sge i32 %boost, 0
  br i1 %has_b, label %upd_b, label %wake
upd_b:
  %pb_ptr = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %cur_pb = load i8, ptr %pb_ptr, align 1
  %b8 = trunc i32 %boost to i8
  %lt = icmp slt i8 %b8, %cur_pb
  %min_b = select i1 %lt, i8 %b8, i8 %cur_pb
  store i8 %min_b, ptr %pb_ptr, align 1
  br label %wake
wake:
  %wm_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 2
  {call_kw} void @__syn_wake_all_mask(ptr %wm_p, i32 %boost)
{crit_exit}  ret void
}}

define i8 @__syn_sig_poll_{id}(ptr %s, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fp = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %f = load i8, ptr %fp, align 1
  %set = icmp ne i8 %f, 0
  br i1 %set, label %take, label %empty
take:
  store i8 0, ptr %fp, align 1
  %pb_ptr = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %pb = load i8, ptr %pb_ptr, align 1
  %has_pb = icmp slt i8 %pb, 127
  br i1 %has_pb, label %apply_pb, label %done_ok
apply_pb:
  store i8 127, ptr %pb_ptr, align 1
  %now = {call_kw} i32 @__syn_now_ms()
  %pb32 = zext i8 %pb to i32
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %pb32, i32 %now)
  br label %done_ok
done_ok:
{crit_exit}  ret i8 1
empty:
{crit_exit}  ret i8 0
}}"#
                    );
                }
                MirPrimKind::Guarded { .. } => {
                    // Accessed exclusively inside caller-provided `critical { ... }` via inline volatile GEP load/store.
                }
                MirPrimKind::Watch { elem } => {
                    let ety = self.llvm_ty(elem);
                    let i32_al = self.llvm_scalar_abi_align(4);
                    let _ = writeln!(
                        out,
                        r#"define void @__syn_watch_pub_{id}(ptr %w, {ety} %val){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %vp = getelementptr inbounds {pty}, ptr %w, i32 0, i32 0
  store {ety} %val, ptr %vp, align 1
  %gp = getelementptr inbounds {pty}, ptr %w, i32 0, i32 1
  %g = load i32, ptr %gp, align {i32_al}
  %g1 = add i32 %g, 1
  store i32 %g1, ptr %gp, align {i32_al}
  %wm_p = getelementptr inbounds {pty}, ptr %w, i32 0, i32 2
  {call_kw} void @__syn_wake_all_mask(ptr %wm_p, i32 -1)
{crit_exit}  ret void
}}

define {ety} @__syn_watch_read_{id}(ptr %w){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %vp = getelementptr inbounds {pty}, ptr %w, i32 0, i32 0
  %v = load {ety}, ptr %vp, align 1
{crit_exit}  ret {ety} %v
}}

define i32 @__syn_watch_gen_{id}(ptr %w){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gp = getelementptr inbounds {pty}, ptr %w, i32 0, i32 1
  %g = load i32, ptr %gp, align {i32_al}
{crit_exit}  ret i32 %g
}}"#
                    );
                }
                MirPrimKind::Inbox { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1) as i32;
                    let is_lease = matches!(elem.as_ref(), MirType::Lease(_));
                    let push_fn = if is_lease {
                        format!("@__syn_q_push_keep_{id}")
                    } else {
                        format!("@__syn_q_push_{id}")
                    };
                    let clear_slot = if is_lease {
                        format!("\n  store {ety} zeroinitializer, ptr %slot_p, align 1")
                    } else {
                        String::new()
                    };
                    let _ = writeln!(
                        out,
                        r#"define i8 {push_fn}(ptr %q, {ety} %val, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cnt_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 2
  %cnt = load i32, ptr %cnt_p, align 4
  %space = icmp slt i32 %cnt, {c}
  br i1 %space, label %do_push, label %full
do_push:
  %tail_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 1
  %tail = load i32, ptr %tail_p, align 4
  %slot_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 6, i32 %tail
  store {ety} %val, ptr %slot_p, align 1
  %bp = getelementptr inbounds {pty}, ptr %q, i32 0, i32 5, i32 %tail
  %has_b = icmp sge i32 %boost, 0
  %b_p1 = add i32 %boost, 1
  %b8 = trunc i32 %b_p1 to i8
  %sb = select i1 %has_b, i8 %b8, i8 0
  store i8 %sb, ptr %bp, align 1
  %t1 = add i32 %tail, 1
  %twrap = icmp uge i32 %t1, {c}
  %tsub = sub i32 %t1, {c}
  %tnext = select i1 %twrap, i32 %tsub, i32 %t1
  store i32 %tnext, ptr %tail_p, align 4
  %c1 = add i32 %cnt, 1
  store i32 %c1, ptr %cnt_p, align 4
  %rx_wm = getelementptr inbounds {pty}, ptr %q, i32 0, i32 3
  {call_kw} void @__syn_wake_all_mask(ptr %rx_wm, i32 %boost)
{crit_exit}  ret i8 1
full:
{crit_exit}  ret i8 0
}}

define i8 @__syn_q_pop_{id}(ptr %q, ptr %out_val, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cnt_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 2
  %cnt = load i32, ptr %cnt_p, align 4
  %avail = icmp sgt i32 %cnt, 0
  br i1 %avail, label %do_pop, label %empty
do_pop:
  %head_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 0
  %head = load i32, ptr %head_p, align 4
  %slot_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 6, i32 %head
  %val = load {ety}, ptr %slot_p, align 1
  store {ety} %val, ptr %out_val, align 1{clear_slot}
  %bp = getelementptr inbounds {pty}, ptr %q, i32 0, i32 5, i32 %head
  %sb = load i8, ptr %bp, align 1
  store i8 0, ptr %bp, align 1
  %h1 = add i32 %head, 1
  %hwrap = icmp uge i32 %h1, {c}
  %hsub = sub i32 %h1, {c}
  %hnext = select i1 %hwrap, i32 %hsub, i32 %h1
  store i32 %hnext, ptr %head_p, align 4
  %c1 = sub i32 %cnt, 1
  store i32 %c1, ptr %cnt_p, align 4
  %has_pb = icmp ne i8 %sb, 0
  br i1 %has_pb, label %apply_pb, label %wake_senders
apply_pb:
  %now = {call_kw} i32 @__syn_now_ms()
  %sb32 = zext i8 %sb to i32
  %pb32 = sub i32 %sb32, 1
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %pb32, i32 %now)
  br label %wake_senders
wake_senders:
  %tx_wm = getelementptr inbounds {pty}, ptr %q, i32 0, i32 4
  {call_kw} void @__syn_wake_all_mask(ptr %tx_wm, i32 -1)
{crit_exit}  ret i8 1
empty:
{crit_exit}  ret i8 0
}}"#
                    );
                    if is_lease {
                        let _ = writeln!(
                            out,
                            r#"define i8 @__syn_q_push_{id}(ptr %q, {ety} %val, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %ok = {call_kw} i8 @__syn_q_push_keep_{id}(ptr %q, {ety} %val, i32 %boost)
  %ok_b = icmp ne i8 %ok, 0
  br i1 %ok_b, label %done, label %drop_lease
drop_lease:
  %ph = extractvalue {ety} %val, 1
  %idx = extractvalue {ety} %val, 2
  {call_kw} void @__syn_pool_release(ptr %ph, i8 %idx, i8 -1)
  ret i8 0
done:
  ret i8 1
}}"#
                        );
                    }
                }
                MirPrimKind::PingPong { .. } => {
                    let _ = writeln!(
                        out,
                        r#"define void @__syn_pp_set_active_{id}(ptr %pp, i8 %active){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %lb_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 3
  %lb = load i8, ptr %lb_p, align 1
  %lb32 = zext i8 %lb to i32
  %sh = and i32 %lb32, 31
  %bit = shl i32 1, %sh
  %old = load volatile i32, ptr @__syn_sleep_lock_mask, align 4
  %is_on = icmp ne i8 %active, 0
  %inv = xor i32 %bit, -1
  %cleared = and i32 %old, %inv
  %setv = or i32 %old, %bit
  %newv = select i1 %is_on, i32 %setv, i32 %cleared
  store volatile i32 %newv, ptr @__syn_sleep_lock_mask, align 4
{crit_exit}  ret void
}}

define void @__syn_pp_flip_{id}(ptr %pp){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %wh_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 0
  %rh_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 1
  %cr_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 2
  %wh = load i8, ptr %wh_p, align 1
  store i8 %wh, ptr %rh_p, align 1
  %next = xor i8 %wh, 1
  store i8 %next, ptr %wh_p, align 1
  store i8 1, ptr %cr_p, align 1
  %wm_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 4
  {call_kw} void @__syn_wake_all_mask(ptr %wm_p, i32 -1)
{crit_exit}  ret void
}}

define void @__syn_pp_complete_half_{id}(ptr %pp, i8 %half){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %wh_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 0
  %rh_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 1
  %cr_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 2
  %c = and i8 %half, 1
  store i8 %c, ptr %rh_p, align 1
  %next = xor i8 %c, 1
  store i8 %next, ptr %wh_p, align 1
  store i8 1, ptr %cr_p, align 1
  %wm_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 4
  {call_kw} void @__syn_wake_all_mask(ptr %wm_p, i32 -1)
{crit_exit}  ret void
}}

define ptr @__syn_pp_poll_{id}(ptr %pp){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cr_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 2
  %cr = load i8, ptr %cr_p, align 1
  %rdy = icmp ne i8 %cr, 0
  br i1 %rdy, label %take, label %none
take:
  store i8 0, ptr %cr_p, align 1
  %rh_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 1
  %rh = load i8, ptr %rh_p, align 1
  %is_b0 = icmp eq i8 %rh, 0
  %b0 = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 5, i32 0
  %b1 = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 6, i32 0
  %buf = select i1 %is_b0, ptr %b0, ptr %b1
{crit_exit}  ret ptr %buf
none:
{crit_exit}  ret ptr null
}}"#
                    );
                }
                MirPrimKind::Pool { .. } => {
                    let ptr_al = self.ptr_align();
                    let usize_ty = self.spec.isa.usize_llvm_ty();
                    let (hdr_sz, _) = self.pool_hdr_llvm_size();
                    let slots_fidx = if 32usize.saturating_sub(hdr_sz) > 0 {
                        2
                    } else {
                        1
                    };
                    let conv_usize = |src: &str, dst: &str| -> String {
                        match usize_ty {
                            "i16" => format!("  {dst} = trunc i32 {src} to i16"),
                            "i64" => format!("  {dst} = zext i32 {src} to i64"),
                            _ => format!("  {dst} = add i32 {src}, 0"),
                        }
                    };
                    let conv_av = conv_usize("%cnt", "%cnt_u");
                    let conv_used = conv_usize("%used", "%used_u");
                    let conv_cap = conv_usize("%cap32", "%cap_u");
                    let _ = writeln!(
                        out,
                        r#"define ptr @__syn_pool_slot_ptr_{id}(ptr %p, i8 %idx){fn_as} nounwind{fn_sz} {{
entry:
  %dma_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 3
  %dma = load i8, ptr %dma_p, align 1
  %is_dma = icmp ne i8 %dma, 0
  %idx32 = zext i8 %idx to i32
  %p_dma = getelementptr inbounds {pty}_dma, ptr %p, i32 0, i32 {slots_fidx}, i32 %idx32, i32 0
  %p_plain = getelementptr inbounds {pty}, ptr %p, i32 0, i32 1, i32 %idx32
  %res = select i1 %is_dma, ptr %p_dma, ptr %p_plain
  ret ptr %res
}}

define i8 @__syn_pool_try_alloc_raw_{id}(ptr %p, ptr %out_lease){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
  %has = icmp ne i32 %fm, 0
  br i1 %has, label %take, label %empty
take:
  %idx32 = call i32 @llvm.cttz.i32(i32 %fm, i1 true)
  %bit = shl i32 1, %idx32
  %inv = xor i32 %bit, -1
  %new_fm = and i32 %fm, %inv
  store i32 %new_fm, ptr %fm_p, align 4
  %dma_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 3
  %dma = load i8, ptr %dma_p, align 1
  %dma_lock = and i8 %dma, 1
  %is_dma = icmp ne i8 %dma_lock, 0
  br i1 %is_dma, label %inc_dma, label %finish
inc_dma:
  %dl = load volatile i32, ptr @__syn_dma_locks, align 4
  %dl1 = add i32 %dl, 1
  store volatile i32 %dl1, ptr @__syn_dma_locks, align 4
  br label %finish
finish:
  %idx8 = trunc i32 %idx32 to i8
{crit_exit}  %dp = {call_kw} ptr @__syn_pool_slot_ptr_{id}(ptr %p, i8 %idx8)
  %f0 = getelementptr inbounds {{ ptr, ptr, i8 }}, ptr %out_lease, i32 0, i32 0
  store ptr %dp, ptr %f0, align {ptr_al}
  %f1 = getelementptr inbounds {{ ptr, ptr, i8 }}, ptr %out_lease, i32 0, i32 1
  store ptr %p, ptr %f1, align {ptr_al}
  %f2 = getelementptr inbounds {{ ptr, ptr, i8 }}, ptr %out_lease, i32 0, i32 2
  store i8 %idx8, ptr %f2, align 1
  ret i8 1
empty:
{crit_exit}  ret i8 0
}}

define {{ {{ ptr, ptr, i8 }}, i32 }} @__syn_pool_try_alloc_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
  %tmp = alloca {{ ptr, ptr, i8 }}, align {ptr_al}
  store {{ ptr, ptr, i8 }} zeroinitializer, ptr %tmp, align {ptr_al}
  %ok = {call_kw} i8 @__syn_pool_try_alloc_raw_{id}(ptr %p, ptr %tmp)
  %got = icmp ne i8 %ok, 0
  %ls = load {{ ptr, ptr, i8 }}, ptr %tmp, align {ptr_al}
  %tag = select i1 %got, i32 0, i32 1
  %r0 = insertvalue {{ {{ ptr, ptr, i8 }}, i32 }} poison, {{ ptr, ptr, i8 }} %ls, 0
  %r1 = insertvalue {{ {{ ptr, ptr, i8 }}, i32 }} %r0, i32 %tag, 1
  ret {{ {{ ptr, ptr, i8 }}, i32 }} %r1
}}

define {usize_ty} @__syn_pool_avail_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
{crit_exit}  %cnt = call i32 @llvm.ctpop.i32(i32 %fm)
{conv_av}
  ret {usize_ty} %cnt_u
}}

define {usize_ty} @__syn_pool_in_use_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cap_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 4
  %cap = load i8, ptr %cap_p, align 1
  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
{crit_exit}  %avail = call i32 @llvm.ctpop.i32(i32 %fm)
  %cap32 = zext i8 %cap to i32
  %used = sub i32 %cap32, %avail
{conv_used}
  ret {usize_ty} %used_u
}}

define {usize_ty} @__syn_pool_cap_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
  %cap_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 4
  %cap = load i8, ptr %cap_p, align 1
  %cap32 = zext i8 %cap to i32
{conv_cap}
  ret {usize_ty} %cap_u
}}

define i8 @__syn_pool_is_exhausted_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
{crit_exit}  %empty = icmp eq i32 %fm, 0
  %b8 = zext i1 %empty to i8
  ret i8 %b8
}}"#
                    );
                }
                MirPrimKind::DmaSlot { .. } => {
                    let usize_ty = self.spec.isa.usize_llvm_ty();
                    let _ = writeln!(
                        out,
                        r#"define {{ i32, i32 }} @__syn_slot_park_{id}(ptr %s, {{ ptr, ptr, i8 }} %lease){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
  %busy = icmp ne i8 %occ, 0
  br i1 %busy, label %is_busy, label %do_park
is_busy:
{crit_exit}  %ph_b = extractvalue {{ ptr, ptr, i8 }} %lease, 1
  %idx_b = extractvalue {{ ptr, ptr, i8 }} %lease, 2
  {call_kw} void @__syn_pool_release(ptr %ph_b, i8 %idx_b, i8 -1)
  ret {{ i32, i32 }} {{ i32 0, i32 1 }}
do_park:
  %dp = extractvalue {{ ptr, ptr, i8 }} %lease, 0
  %ph = extractvalue {{ ptr, ptr, i8 }} %lease, 1
  %idx = extractvalue {{ ptr, ptr, i8 }} %lease, 2
  %f0 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %f1 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %f2 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 2
  store ptr %dp, ptr %f0, align 1
  store ptr %ph, ptr %f1, align 1
  store i8 %idx, ptr %f2, align 1
  store i8 1, ptr %occ_p, align 1
{crit_exit}  ret {{ i32, i32 }} zeroinitializer
}}

define {{ {{ ptr, ptr, i8 }}, i32 }} @__syn_slot_take_{id}(ptr %s){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
  %empty = icmp eq i8 %occ, 0
  br i1 %empty, label %is_empty, label %do_take
is_empty:
{crit_exit}  ret {{ {{ ptr, ptr, i8 }}, i32 }} {{ {{ ptr, ptr, i8 }} zeroinitializer, i32 1 }}
do_take:
  %f0 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %f1 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %f2 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 2
  %dp = load ptr, ptr %f0, align 1
  %ph = load ptr, ptr %f1, align 1
  %idx = load i8, ptr %f2, align 1
  store ptr null, ptr %f0, align 1
  store ptr null, ptr %f1, align 1
  store i8 0, ptr %f2, align 1
  store i8 0, ptr %occ_p, align 1
{crit_exit}  %l0 = insertvalue {{ ptr, ptr, i8 }} poison, ptr %dp, 0
  %l1 = insertvalue {{ ptr, ptr, i8 }} %l0, ptr %ph, 1
  %l2 = insertvalue {{ ptr, ptr, i8 }} %l1, i8 %idx, 2
  %r0 = insertvalue {{ {{ ptr, ptr, i8 }}, i32 }} poison, {{ ptr, ptr, i8 }} %l2, 0
  %r1 = insertvalue {{ {{ ptr, ptr, i8 }}, i32 }} %r0, i32 0, 1
  ret {{ {{ ptr, ptr, i8 }}, i32 }} %r1
}}

define i8 @__syn_slot_release_{id}(ptr %s, i8 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
  %empty = icmp eq i8 %occ, 0
  br i1 %empty, label %is_empty, label %do_rel
is_empty:
{crit_exit}  ret i8 0
do_rel:
  %f0 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %f1 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %f2 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 2
  %ph = load ptr, ptr %f1, align 1
  %idx = load i8, ptr %f2, align 1
  store ptr null, ptr %f0, align 1
  store ptr null, ptr %f1, align 1
  store i8 0, ptr %f2, align 1
  store i8 0, ptr %occ_p, align 1
{crit_exit}  {call_kw} void @__syn_pool_release(ptr %ph, i8 %idx, i8 %boost)
  ret i8 1
}}

define i8 @__syn_slot_is_busy_{id}(ptr %s){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
{crit_exit}  %b = icmp ne i8 %occ, 0
  %b8 = zext i1 %b to i8
  ret i8 %b8
}}

define i8 @__syn_slot_is_empty_{id}(ptr %s){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
{crit_exit}  %e = icmp eq i8 %occ, 0
  %b8 = zext i1 %e to i8
  ret i8 %b8
}}

define {usize_ty} @__syn_slot_addr_{id}(ptr %s){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
  %f0 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %dp = load ptr, ptr %f0, align 1
{crit_exit}  %busy = icmp ne i8 %occ, 0
  %p_sel = select i1 %busy, ptr %dp, ptr null
  %addr = ptrtoint ptr %p_sel to {usize_ty}
  ret {usize_ty} %addr
}}"#
                    );
                }
                MirPrimKind::ByteRing { cap } => {
                    let c = (*cap).max(1) as i32;
                    let usize_ty = self.spec.isa.usize_llvm_ty();
                    let slice_ty = format!("{{ ptr, {usize_ty} }}");
                    let res_slice_ty = format!("{{ {slice_ty}, i32 }}");
                    let to_usize = |src: &str, dst: &str| -> String {
                        match usize_ty {
                            "i16" => format!("  {dst} = trunc i32 {src} to i16"),
                            "i64" => format!("  {dst} = zext i32 {src} to i64"),
                            _ => format!("  {dst} = add i32 {src}, 0"),
                        }
                    };
                    let conv_req = to_usize("%req", "%req_u");
                    let conv_s1_r = to_usize("%s1", "%s1_u");
                    let conv_s1_p = to_usize("%s1", "%s1_u");
                    let mk_slice_ok = |tag: &str, ptr_v: &str, len_u: &str| -> String {
                        format!(
                            "  %sl0_{tag} = insertvalue {slice_ty} poison, ptr {ptr_v}, 0\n  %sl1_{tag} = insertvalue {slice_ty} %sl0_{tag}, {usize_ty} {len_u}, 1\n  %ok0_{tag} = insertvalue {res_slice_ty} zeroinitializer, {slice_ty} %sl1_{tag}, 0\n  ret {res_slice_ty} %ok0_{tag}"
                        )
                    };
                    let ok_z = mk_slice_ok("z", "%pz", "0");
                    let ok_back = mk_slice_ok("b", "%pb", "%req_u");
                    let ok_wrap = mk_slice_ok("w", "%pw", "%req_u");
                    let ok_gap = mk_slice_ok("g", "%pg", "%req_u");
                    let _ = writeln!(
                        out,
                        r#"define i32 @__syn_ring_seg1_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_z = icmp eq i32 %cnt, 0
  br i1 %is_z, label %z, label %nz
z:
  ret i32 0
nz:
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %has_se = icmp ugt i32 %se, 0
  br i1 %has_se, label %use_se, label %no_se
use_se:
  %d0 = sub i32 %se, %h
  ret i32 %d0
no_se:
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %h_lt_t = icmp ult i32 %h, %t
  %d1 = sub i32 %t, %h
  %d2 = sub i32 {c}, %h
  %res = select i1 %h_lt_t, i32 %d1, i32 %d2
  ret i32 %res
}}

define i32 @__syn_ring_free_nolock_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  br i1 %has_se, label %se_p, label %no_se
se_p:
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %h = load i32, ptr %hp, align 4
  %t = load i32, ptr %tp, align 4
  %fs0 = sub i32 %h, %t
  ret i32 %fs0
no_se:
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %fs1 = sub i32 {c}, %cnt
  ret i32 %fs1
}}

define i32 @__syn_ring_free_space_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fs = {call_kw} i32 @__syn_ring_free_nolock_{id}(ptr %r)
{crit_exit}  ret i32 %fs
}}

define void @__syn_ring_advance_{id}(ptr %r, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %lt = icmp ult i32 %req, %cnt
  %rem = select i1 %lt, i32 %req, i32 %cnt
  %is_z = icmp eq i32 %rem, 0
  br i1 %is_z, label %done, label %chk_all
chk_all:
  %s1 = {call_kw} i32 @__syn_ring_seg1_{id}(ptr %r)
  %is_all = icmp eq i32 %rem, %cnt
  br i1 %is_all, label %all_empty, label %partial
all_empty:
  %hp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %sp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %gp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  %t0 = load i32, ptr %tp0, align 4
  %gc = load i32, ptr %gp0, align 4
  %has_g = icmp ugt i32 %gc, 0
  %nh0 = select i1 %has_g, i32 %t0, i32 0
  store i32 %nh0, ptr %hp0, align 4
  br i1 %has_g, label %skip_tail_rst, label %do_tail_rst
do_tail_rst:
  store i32 0, ptr %tp0, align 4
  br label %skip_tail_rst
skip_tail_rst:
  store i32 0, ptr %cp, align 4
  store i32 0, ptr %sp0, align 4
  ret void
partial:
  %hp1 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h1 = load i32, ptr %hp1, align 4
  %in_s1 = icmp ult i32 %rem, %s1
  br i1 %in_s1, label %adv_h, label %wrap_h
adv_h:
  %nh1 = add i32 %h1, %rem
  %nc1 = sub i32 %cnt, %rem
  store i32 %nh1, ptr %hp1, align 4
  store i32 %nc1, ptr %cp, align 4
  ret void
wrap_h:
  %nh2 = sub i32 %rem, %s1
  %nc2 = sub i32 %cnt, %rem
  %sp2 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  store i32 %nh2, ptr %hp1, align 4
  store i32 %nc2, ptr %cp, align 4
  store i32 0, ptr %sp2, align 4
  ret void
done:
  ret void
}}

define void @__syn_ring_wake_{id}(ptr %r, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %has_b = icmp sge i32 %boost, 0
  br i1 %has_b, label %chk_b, label %do_wake
chk_b:
  %b_p1 = add i32 %boost, 1
  %enc = trunc i32 %b_p1 to i8
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7
  %cur = load i8, ptr %bp, align 1
  %is_z = icmp eq i8 %cur, 0
  %better = icmp ult i8 %enc, %cur
  %upd = or i1 %is_z, %better
  br i1 %upd, label %set_b, label %do_wake
set_b:
  store i8 %enc, ptr %bp, align 1
  br label %do_wake
do_wake:
  %wm_p = getelementptr inbounds {pty}, ptr %r, i32 0, i32 6
  {call_kw} void @__syn_wake_all_mask(ptr %wm_p, i32 %boost)
  ret void
}}

define void @__syn_ring_apply_boost_{id}(ptr %r, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7
  %enc = load i8, ptr %bp, align 1
  %has_b = icmp ne i8 %enc, 0
  %has_t = icmp ne ptr %task, null
  %go = and i1 %has_b, %has_t
  br i1 %go, label %apply, label %done
apply:
  store i8 0, ptr %bp, align 1
  %enc32 = zext i8 %enc to i32
  %prio = sub i32 %enc32, 1
  %now = {call_kw} i32 @__syn_now_ms()
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %prio, i32 %now)
  br label %done
done:
  ret void
}}

define i8 @__syn_ring_push_byte_{id}(ptr %r, i8 %b, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 0, ptr %gp, align 4
  %fs = {call_kw} i32 @__syn_ring_free_nolock_{id}(ptr %r)
  %has = icmp ugt i32 %fs, 0
  br i1 %has, label %do_wr, label %full
do_wr:
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %dp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 %t
  store i8 %b, ptr %dp, align 1
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  %t1 = add i32 %t, 1
  %wrap = icmp uge i32 %t1, {c}
  %nt_circ = select i1 %wrap, i32 0, i32 %t1
  %nt = select i1 %has_se, i32 %t1, i32 %nt_circ
  store i32 %nt, ptr %tp, align 4
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %nc = add i32 %cnt, 1
  store i32 %nc, ptr %cp, align 4
  {call_kw} void @__syn_ring_wake_{id}(ptr %r, i32 %boost)
{crit_exit}  ret i8 1
full:
{crit_exit}  ret i8 0
}}

define i32 @__syn_ring_push_{id}(ptr %r, ptr %src_ptr, i32 %src_len, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 0, ptr %gp, align 4
  %fs = {call_kw} i32 @__syn_ring_free_nolock_{id}(ptr %r)
  %is_null = icmp eq ptr %src_ptr, null
  %lt = icmp ult i32 %src_len, %fs
  %nw_raw = select i1 %lt, i32 %src_len, i32 %fs
  %nw = select i1 %is_null, i32 0, i32 %nw_raw
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %t_init = load i32, ptr %tp, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %body ]
  %t_cur = phi i32 [ %t_init, %entry ], [ %t_next, %body ]
  %done = icmp uge i32 %i, %nw
  br i1 %done, label %finish, label %body
body:
  %sp_in = getelementptr i8, ptr %src_ptr, i32 %i
  %bv = load i8, ptr %sp_in, align 1
  %dp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 %t_cur
  store i8 %bv, ptr %dp, align 1
  %t1 = add i32 %t_cur, 1
  %wrap = icmp uge i32 %t1, {c}
  %t_circ = select i1 %wrap, i32 0, i32 %t1
  %t_next = select i1 %has_se, i32 %t1, i32 %t_circ
  %inext = add i32 %i, 1
  br label %loop
finish:
  store i32 %t_cur, ptr %tp, align 4
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %nc = add i32 %cnt, %nw
  store i32 %nc, ptr %cp, align 4
  %any = icmp ugt i32 %nw, 0
  br i1 %any, label %wake, label %exit
wake:
  {call_kw} void @__syn_ring_wake_{id}(ptr %r, i32 %boost)
  br label %exit
exit:
{crit_exit}  ret i32 %nw
}}

define {res_slice_ty} @__syn_ring_grant_write_{id}(ptr %r, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}{conv_req}
  %is_z = icmp eq i32 %req, 0
  br i1 %is_z, label %ret_z, label %chk_cap
ret_z:
  %tp_z = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %tz = load i32, ptr %tp_z, align 4
  %gop_z = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp_z = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 %tz, ptr %gop_z, align 4
  store i32 0, ptr %gcp_z, align 4
  %pz = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 0
{crit_exit}{ok_z}
chk_cap:
  %too_big = icmp ugt i32 %req, {c}
  br i1 %too_big, label %fail, label %chk_empty
chk_empty:
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_emp = icmp eq i32 %cnt, 0
  br i1 %is_emp, label %empty_ok, label %non_empty
empty_ok:
  %hp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %sp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %gop0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 0, ptr %hp0, align 4
  store i32 0, ptr %tp0, align 4
  store i32 0, ptr %sp0, align 4
  store i32 0, ptr %gop0, align 4
  store i32 %req, ptr %gcp0, align 4
  %pz_ok = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 0
{crit_exit}  %sl0_e = insertvalue {slice_ty} poison, ptr %pz_ok, 0
  %sl1_e = insertvalue {slice_ty} %sl0_e, {usize_ty} %req_u, 1
  %ok_e = insertvalue {res_slice_ty} zeroinitializer, {slice_ty} %sl1_e, 0
  ret {res_slice_ty} %ok_e
non_empty:
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %h = load i32, ptr %hp, align 4
  %t = load i32, ptr %tp, align 4
  %se = load i32, ptr %sp, align 4
  %no_se = icmp eq i32 %se, 0
  %h_lt_t = icmp ult i32 %h, %t
  %unwrapped = and i1 %no_se, %h_lt_t
  br i1 %unwrapped, label %back_or_wrap, label %mid_gap
back_or_wrap:
  %rem_back = sub i32 {c}, %t
  %fit_back = icmp ule i32 %req, %rem_back
  br i1 %fit_back, label %ok_b, label %try_wrap
ok_b:
  %gop_b = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp_b = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 %t, ptr %gop_b, align 4
  store i32 %req, ptr %gcp_b, align 4
  %pb = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 %t
{crit_exit}{ok_back}
try_wrap:
  %fit_wrap = icmp ule i32 %req, %h
  br i1 %fit_wrap, label %ok_w, label %fail
ok_w:
  %gop_w = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp_w = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 0, ptr %gop_w, align 4
  store i32 %req, ptr %gcp_w, align 4
  %pw = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 0
{crit_exit}{ok_wrap}
mid_gap:
  %has_se = icmp ugt i32 %se, 0
  %h_gt_t = icmp ugt i32 %h, %t
  %has_gap = or i1 %has_se, %h_gt_t
  %raw_gap = sub i32 %h, %t
  %gap = select i1 %has_gap, i32 %raw_gap, i32 0
  %fit_gap = icmp ule i32 %req, %gap
  br i1 %fit_gap, label %ok_g, label %fail
ok_g:
  %gop_g = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp_g = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 %t, ptr %gop_g, align 4
  store i32 %req, ptr %gcp_g, align 4
  %pg = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 %t
{crit_exit}{ok_gap}
fail:
{crit_exit}  %err_r = insertvalue {res_slice_ty} zeroinitializer, i32 1, 1
  ret {res_slice_ty} %err_r
}}

define void @__syn_ring_commit_write_{id}(ptr %r, i32 %req, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gcp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  %gc = load i32, ptr %gcp, align 4
  store i32 0, ptr %gcp, align 4
  %lt = icmp ult i32 %req, %gc
  %nw = select i1 %lt, i32 %req, i32 %gc
  %is_z = icmp eq i32 %nw, 0
  br i1 %is_z, label %done, label %apply
apply:
  %gop = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %goff = load i32, ptr %gop, align 4
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_w0 = icmp eq i32 %goff, 0
  %t_gt0 = icmp ugt i32 %t, 0
  %se_eq0 = icmp eq i32 %se, 0
  %c0 = and i1 %is_w0, %t_gt0
  %is_early = and i1 %c0, %se_eq0
  br i1 %is_early, label %early_wrap, label %linear
early_wrap:
  %cnt_z = icmp eq i32 %cnt, 0
  br i1 %cnt_z, label %ew_empty, label %ew_split
ew_empty:
  %hp_e = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  store i32 0, ptr %hp_e, align 4
  store i32 %nw, ptr %tp, align 4
  store i32 %nw, ptr %cp, align 4
  br label %wake
ew_split:
  store i32 %t, ptr %sp, align 4
  store i32 %nw, ptr %tp, align 4
  %nc_ew = add i32 %cnt, %nw
  store i32 %nc_ew, ptr %cp, align 4
  br label %wake
linear:
  %nt = add i32 %t, %nw
  %wrap = icmp uge i32 %nt, {c}
  %do_wrap = and i1 %se_eq0, %wrap
  %nt_final = select i1 %do_wrap, i32 0, i32 %nt
  store i32 %nt_final, ptr %tp, align 4
  %nc_l = add i32 %cnt, %nw
  store i32 %nc_l, ptr %cp, align 4
  br label %wake
wake:
  {call_kw} void @__syn_ring_wake_{id}(ptr %r, i32 %boost)
  br label %done
done:
{crit_exit}  ret void
}}

define {slice_ty} @__syn_ring_read_slice_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %s1 = {call_kw} i32 @__syn_ring_seg1_{id}(ptr %r)
  %has = icmp ugt i32 %s1, 0
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %idx = select i1 %has, i32 %h, i32 0
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 %idx
{crit_exit}{conv_s1_r}
  %sl0 = insertvalue {slice_ty} poison, ptr %bp, 0
  %sl1 = insertvalue {slice_ty} %sl0, {usize_ty} %s1_u, 1
  ret {slice_ty} %sl1
}}

define {slice_ty} @__syn_ring_poll_read_{id}(ptr %r, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %s1 = {call_kw} i32 @__syn_ring_seg1_{id}(ptr %r)
  %has = icmp ugt i32 %s1, 0
  br i1 %has, label %ready, label %empty
ready:
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 %h
  {call_kw} void @__syn_ring_apply_boost_{id}(ptr %r, ptr %task)
{crit_exit}{conv_s1_p}
  %sl0 = insertvalue {slice_ty} poison, ptr %bp, 0
  %sl1 = insertvalue {slice_ty} %sl0, {usize_ty} %s1_u, 1
  ret {slice_ty} %sl1
empty:
{crit_exit}  ret {slice_ty} zeroinitializer
}}

define void @__syn_ring_consume_{id}(ptr %r, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  {call_kw} void @__syn_ring_advance_{id}(ptr %r, i32 %req)
{crit_exit}  ret void
}}

define i32 @__syn_ring_pop_{id}(ptr %r, ptr %dst_ptr, i32 %dst_len, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %lt = icmp ult i32 %dst_len, %cnt
  %rem_raw = select i1 %lt, i32 %dst_len, i32 %cnt
  %is_null = icmp eq ptr %dst_ptr, null
  %rem = select i1 %is_null, i32 0, i32 %rem_raw
  %has = icmp ugt i32 %rem, 0
  br i1 %has, label %do_pop, label %empty
do_pop:
  %s1 = {call_kw} i32 @__syn_ring_seg1_{id}(ptr %r)
  %lt_s1 = icmp ult i32 %rem, %s1
  %c1 = select i1 %lt_s1, i32 %rem, i32 %s1
  %c2 = sub i32 %rem, %c1
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  br label %l1
l1:
  %i1 = phi i32 [ 0, %do_pop ], [ %i1n, %b1 ]
  %d1 = icmp uge i32 %i1, %c1
  br i1 %d1, label %l2_init, label %b1
b1:
  %src1_i = add i32 %h, %i1
  %sp1 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 %src1_i
  %v1 = load i8, ptr %sp1, align 1
  %dp1 = getelementptr i8, ptr %dst_ptr, i32 %i1
  store i8 %v1, ptr %dp1, align 1
  %i1n = add i32 %i1, 1
  br label %l1
l2_init:
  br label %l2
l2:
  %i2 = phi i32 [ 0, %l2_init ], [ %i2n, %b2 ]
  %d2 = icmp uge i32 %i2, %c2
  br i1 %d2, label %adv, label %b2
b2:
  %sp2 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 %i2
  %v2 = load i8, ptr %sp2, align 1
  %dst2_i = add i32 %c1, %i2
  %dp2 = getelementptr i8, ptr %dst_ptr, i32 %dst2_i
  store i8 %v2, ptr %dp2, align 1
  %i2n = add i32 %i2, 1
  br label %l2
adv:
  {call_kw} void @__syn_ring_advance_{id}(ptr %r, i32 %rem)
  {call_kw} void @__syn_ring_apply_boost_{id}(ptr %r, ptr %task)
{crit_exit}  ret i32 %rem
empty:
{crit_exit}  ret i32 0
}}

define {{ i8, i32 }} @__syn_ring_try_pop_byte_{id}(ptr %r, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %has = icmp ugt i32 %cnt, 0
  br i1 %has, label %ok, label %empty
ok:
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 8, i32 %h
  %b = load i8, ptr %bp, align 1
  {call_kw} void @__syn_ring_advance_{id}(ptr %r, i32 1)
  {call_kw} void @__syn_ring_apply_boost_{id}(ptr %r, ptr %task)
{crit_exit}  %r0 = insertvalue {{ i8, i32 }} zeroinitializer, i8 %b, 0
  ret {{ i8, i32 }} %r0
empty:
{crit_exit}  %re = insertvalue {{ i8, i32 }} zeroinitializer, i32 1, 1
  ret {{ i8, i32 }} %re
}}

define void @__syn_ring_clear_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %gop = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7
  store i32 0, ptr %hp, align 4
  store i32 0, ptr %tp, align 4
  store i32 0, ptr %cp, align 4
  store i32 0, ptr %sp, align 4
  store i32 0, ptr %gop, align 4
  store i32 0, ptr %gcp, align 4
  store i8 0, ptr %bp, align 1
{crit_exit}  ret void
}}"#
                    );
                }
            }
        }

        // Emit capacity-erased unsized `*inbox<T>` view helpers
        for (key, elem) in &self.distinct_inbox_views {
            let ety = self.llvm_ty(elem);
            let is_lease = matches!(elem, MirType::Lease(_));
            let push_fn = if is_lease {
                format!("@__syn_q_view_push_keep_{key}")
            } else {
                format!("@__syn_q_view_push_{key}")
            };
            let clear_slot = if is_lease {
                format!("\n  store {ety} zeroinitializer, ptr %slot_p, align 1")
            } else {
                String::new()
            };
            let (crit_enter, crit_exit) = crit_pair(vis.is_inbox_view_contended(elem), call_kw);
            let _ = writeln!(
                out,
                r#"define i8 {push_fn}(ptr %hdr, ptr %items_base, i32 %cap, {ety} %val, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cnt_p = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 2
  %cnt = load i32, ptr %cnt_p, align 4
  %has_c = icmp sgt i32 %cap, 0
  %space = icmp slt i32 %cnt, %cap
  %ok = and i1 %has_c, %space
  br i1 %ok, label %do_push, label %full
do_push:
  %tail_p = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 1
  %tail = load i32, ptr %tail_p, align 4
  %slot_p = getelementptr inbounds {ety}, ptr %items_base, i32 %tail
  store {ety} %val, ptr %slot_p, align 1
  %bp = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 5, i32 %tail
  %has_b = icmp sge i32 %boost, 0
  %b_p1 = add i32 %boost, 1
  %b8 = trunc i32 %b_p1 to i8
  %sb = select i1 %has_b, i8 %b8, i8 0
  store i8 %sb, ptr %bp, align 1
  %t1 = add i32 %tail, 1
  %twrap = icmp uge i32 %t1, %cap
  %tsub = sub i32 %t1, %cap
  %tnext = select i1 %twrap, i32 %tsub, i32 %t1
  store i32 %tnext, ptr %tail_p, align 4
  %c1 = add i32 %cnt, 1
  store i32 %c1, ptr %cnt_p, align 4
  %rx_wm = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 3
  {call_kw} void @__syn_wake_all_mask(ptr %rx_wm, i32 %boost)
{crit_exit}  ret i8 1
full:
{crit_exit}  ret i8 0
}}

define i8 @__syn_q_view_pop_{key}(ptr %hdr, ptr %items_base, i32 %cap, ptr %out_val, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cnt_p = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 2
  %cnt = load i32, ptr %cnt_p, align 4
  %has_c = icmp sgt i32 %cap, 0
  %avail = icmp sgt i32 %cnt, 0
  %ok = and i1 %has_c, %avail
  br i1 %ok, label %do_pop, label %empty
do_pop:
  %head_p = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 0
  %head = load i32, ptr %head_p, align 4
  %slot_p = getelementptr inbounds {ety}, ptr %items_base, i32 %head
  %val = load {ety}, ptr %slot_p, align 1
  store {ety} %val, ptr %out_val, align 1{clear_slot}
  %bp = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 5, i32 %head
  %sb = load i8, ptr %bp, align 1
  store i8 0, ptr %bp, align 1
  %h1 = add i32 %head, 1
  %hwrap = icmp uge i32 %h1, %cap
  %hsub = sub i32 %h1, %cap
  %hnext = select i1 %hwrap, i32 %hsub, i32 %h1
  store i32 %hnext, ptr %head_p, align 4
  %c1 = sub i32 %cnt, 1
  store i32 %c1, ptr %cnt_p, align 4
  %has_pb = icmp ne i8 %sb, 0
  br i1 %has_pb, label %apply_pb, label %wake_senders
apply_pb:
  %now = {call_kw} i32 @__syn_now_ms()
  %sb32 = zext i8 %sb to i32
  %pb32 = sub i32 %sb32, 1
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %pb32, i32 %now)
  br label %wake_senders
wake_senders:
  %tx_wm = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 4
  {call_kw} void @__syn_wake_all_mask(ptr %tx_wm, i32 -1)
{crit_exit}  ret i8 1
empty:
{crit_exit}  ret i8 0
}}"#
            );
            if is_lease {
                let _ = writeln!(
                    out,
                    r#"define i8 @__syn_q_view_push_{key}(ptr %hdr, ptr %items_base, i32 %cap, {ety} %val, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %ok = {call_kw} i8 @__syn_q_view_push_keep_{key}(ptr %hdr, ptr %items_base, i32 %cap, {ety} %val, i32 %boost)
  %ok_b = icmp ne i8 %ok, 0
  br i1 %ok_b, label %done, label %drop_lease
drop_lease:
  %ph = extractvalue {ety} %val, 1
  %idx = extractvalue {ety} %val, 2
  {call_kw} void @__syn_pool_release(ptr %ph, i8 %idx, i8 -1)
  ret i8 0
done:
  ret i8 1
}}"#
                );
            }
        }

        // Emit capacity-erased unsized `*byte_ring` view helpers (pure SSA, zero non-entry allocas)
        if self.uses_byte_ring_view {
            let (crit_enter, crit_exit) = crit_pair(vis.is_byte_ring_view_contended(), call_kw);
            let usize_ty = self.spec.isa.usize_llvm_ty();
            let slice_ty = format!("{{ ptr, {usize_ty} }}");
            let res_slice_ty = format!("{{ {slice_ty}, i32 }}");
            let to_usize = |src: &str, dst: &str| -> String {
                match usize_ty {
                    "i16" => format!("  {dst} = trunc i32 {src} to i16"),
                    "i64" => format!("  {dst} = zext i32 {src} to i64"),
                    _ => format!("  {dst} = add i32 {src}, 0"),
                }
            };
            let conv_req = to_usize("%req", "%req_u");
            let conv_s1_r = to_usize("%s1", "%s1_u");
            let conv_s1_p = to_usize("%s1", "%s1_u");
            let mk_slice_ok = |tag: &str, ptr_v: &str, len_u: &str| -> String {
                format!(
                    "  %sl0_{tag} = insertvalue {slice_ty} poison, ptr {ptr_v}, 0\n  %sl1_{tag} = insertvalue {slice_ty} %sl0_{tag}, {usize_ty} {len_u}, 1\n  %ok0_{tag} = insertvalue {res_slice_ty} zeroinitializer, {slice_ty} %sl1_{tag}, 0\n  ret {res_slice_ty} %ok0_{tag}"
                )
            };
            let ok_z = mk_slice_ok("z", "%pz", "0");
            let ok_back = mk_slice_ok("b", "%pb", "%req_u");
            let ok_wrap = mk_slice_ok("w", "%pw", "%req_u");
            let ok_gap = mk_slice_ok("g", "%pg", "%req_u");
            let _ = writeln!(
                out,
                r#"define i32 @__syn_ring_view_seg1(ptr %r, i32 %cap){fn_as} nounwind{fn_sz} {{
entry:
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_z = icmp eq i32 %cnt, 0
  br i1 %is_z, label %z, label %nz
z:
  ret i32 0
nz:
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %has_se = icmp ugt i32 %se, 0
  br i1 %has_se, label %use_se, label %no_se
use_se:
  %d0 = sub i32 %se, %h
  ret i32 %d0
no_se:
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %h_lt_t = icmp ult i32 %h, %t
  %d1 = sub i32 %t, %h
  %d2 = sub i32 %cap, %h
  %res = select i1 %h_lt_t, i32 %d1, i32 %d2
  ret i32 %res
}}

define i32 @__syn_ring_view_free_nolock(ptr %r, i32 %cap){fn_as} nounwind{fn_sz} {{
entry:
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  br i1 %has_se, label %se_p, label %no_se
se_p:
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %h = load i32, ptr %hp, align 4
  %t = load i32, ptr %tp, align 4
  %fs0 = sub i32 %h, %t
  ret i32 %fs0
no_se:
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %fs1 = sub i32 %cap, %cnt
  ret i32 %fs1
}}

define i32 @__syn_ring_view_free_space(ptr %r, i32 %cap){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fs = {call_kw} i32 @__syn_ring_view_free_nolock(ptr %r, i32 %cap)
{crit_exit}  ret i32 %fs
}}

define void @__syn_ring_view_advance(ptr %r, i32 %cap, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %lt = icmp ult i32 %req, %cnt
  %rem = select i1 %lt, i32 %req, i32 %cnt
  %is_z = icmp eq i32 %rem, 0
  br i1 %is_z, label %done, label %chk_all
chk_all:
  %s1 = {call_kw} i32 @__syn_ring_view_seg1(ptr %r, i32 %cap)
  %is_all = icmp eq i32 %rem, %cnt
  br i1 %is_all, label %all_empty, label %partial
all_empty:
  %hp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %sp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %gp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  %t0 = load i32, ptr %tp0, align 4
  %gc = load i32, ptr %gp0, align 4
  %has_g = icmp ugt i32 %gc, 0
  %nh0 = select i1 %has_g, i32 %t0, i32 0
  store i32 %nh0, ptr %hp0, align 4
  br i1 %has_g, label %skip_tail_rst, label %do_tail_rst
do_tail_rst:
  store i32 0, ptr %tp0, align 4
  br label %skip_tail_rst
skip_tail_rst:
  store i32 0, ptr %cp, align 4
  store i32 0, ptr %sp0, align 4
  ret void
partial:
  %hp1 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h1 = load i32, ptr %hp1, align 4
  %in_s1 = icmp ult i32 %rem, %s1
  br i1 %in_s1, label %adv_h, label %wrap_h
adv_h:
  %nh1 = add i32 %h1, %rem
  %nc1 = sub i32 %cnt, %rem
  store i32 %nh1, ptr %hp1, align 4
  store i32 %nc1, ptr %cp, align 4
  ret void
wrap_h:
  %nh2 = sub i32 %rem, %s1
  %nc2 = sub i32 %cnt, %rem
  %sp2 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  store i32 %nh2, ptr %hp1, align 4
  store i32 %nc2, ptr %cp, align 4
  store i32 0, ptr %sp2, align 4
  ret void
done:
  ret void
}}

define void @__syn_ring_view_wake(ptr %r, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %has_b = icmp sge i32 %boost, 0
  br i1 %has_b, label %chk_b, label %do_wake
chk_b:
  %b_p1 = add i32 %boost, 1
  %enc = trunc i32 %b_p1 to i8
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7
  %cur = load i8, ptr %bp, align 1
  %is_z = icmp eq i8 %cur, 0
  %better = icmp ult i8 %enc, %cur
  %upd = or i1 %is_z, %better
  br i1 %upd, label %set_b, label %do_wake
set_b:
  store i8 %enc, ptr %bp, align 1
  br label %do_wake
do_wake:
  %wm_p = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 6
  {call_kw} void @__syn_wake_all_mask(ptr %wm_p, i32 %boost)
  ret void
}}

define void @__syn_ring_hdr_wake(ptr %r, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  {call_kw} void @__syn_ring_view_wake(ptr %r, i32 %boost)
{crit_exit}  ret void
}}

define void @__syn_ring_view_apply_boost(ptr %r, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7
  %enc = load i8, ptr %bp, align 1
  %has_b = icmp ne i8 %enc, 0
  %has_t = icmp ne ptr %task, null
  %go = and i1 %has_b, %has_t
  br i1 %go, label %apply, label %done
apply:
  store i8 0, ptr %bp, align 1
  %enc32 = zext i8 %enc to i32
  %prio = sub i32 %enc32, 1
  %now = {call_kw} i32 @__syn_now_ms()
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %prio, i32 %now)
  br label %done
done:
  ret void
}}

define i8 @__syn_ring_view_push_byte(ptr %r, i32 %cap, i8 %b, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 0, ptr %gp, align 4
  %fs = {call_kw} i32 @__syn_ring_view_free_nolock(ptr %r, i32 %cap)
  %has = icmp ugt i32 %fs, 0
  br i1 %has, label %do_wr, label %full
do_wr:
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %dp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 %t
  store i8 %b, ptr %dp, align 1
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  %t1 = add i32 %t, 1
  %wrap = icmp uge i32 %t1, %cap
  %nt_circ = select i1 %wrap, i32 0, i32 %t1
  %nt = select i1 %has_se, i32 %t1, i32 %nt_circ
  store i32 %nt, ptr %tp, align 4
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %nc = add i32 %cnt, 1
  store i32 %nc, ptr %cp, align 4
  {call_kw} void @__syn_ring_view_wake(ptr %r, i32 %boost)
{crit_exit}  ret i8 1
full:
{crit_exit}  ret i8 0
}}

define i32 @__syn_ring_view_push(ptr %r, i32 %cap, ptr %src_ptr, i32 %src_len, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 0, ptr %gp, align 4
  %fs = {call_kw} i32 @__syn_ring_view_free_nolock(ptr %r, i32 %cap)
  %is_null = icmp eq ptr %src_ptr, null
  %lt = icmp ult i32 %src_len, %fs
  %nw_raw = select i1 %lt, i32 %src_len, i32 %fs
  %nw = select i1 %is_null, i32 0, i32 %nw_raw
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %t_init = load i32, ptr %tp, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %body ]
  %t_cur = phi i32 [ %t_init, %entry ], [ %t_next, %body ]
  %done = icmp uge i32 %i, %nw
  br i1 %done, label %finish, label %body
body:
  %sp_in = getelementptr i8, ptr %src_ptr, i32 %i
  %bv = load i8, ptr %sp_in, align 1
  %dp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 %t_cur
  store i8 %bv, ptr %dp, align 1
  %t1 = add i32 %t_cur, 1
  %wrap = icmp uge i32 %t1, %cap
  %t_circ = select i1 %wrap, i32 0, i32 %t1
  %t_next = select i1 %has_se, i32 %t1, i32 %t_circ
  %inext = add i32 %i, 1
  br label %loop
finish:
  store i32 %t_cur, ptr %tp, align 4
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %nc = add i32 %cnt, %nw
  store i32 %nc, ptr %cp, align 4
  %any = icmp ugt i32 %nw, 0
  br i1 %any, label %wake, label %exit
wake:
  {call_kw} void @__syn_ring_view_wake(ptr %r, i32 %boost)
  br label %exit
exit:
{crit_exit}  ret i32 %nw
}}

define {res_slice_ty} @__syn_ring_view_grant_write(ptr %r, i32 %cap, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}{conv_req}
  %is_z = icmp eq i32 %req, 0
  br i1 %is_z, label %ret_z, label %chk_cap
ret_z:
  %tp_z = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %tz = load i32, ptr %tp_z, align 4
  %gop_z = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp_z = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 %tz, ptr %gop_z, align 4
  store i32 0, ptr %gcp_z, align 4
  %pz = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 0
{crit_exit}{ok_z}
chk_cap:
  %too_big = icmp ugt i32 %req, %cap
  br i1 %too_big, label %fail, label %chk_empty
chk_empty:
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_emp = icmp eq i32 %cnt, 0
  br i1 %is_emp, label %empty_ok, label %non_empty
empty_ok:
  %hp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %sp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %gop0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 0, ptr %hp0, align 4
  store i32 0, ptr %tp0, align 4
  store i32 0, ptr %sp0, align 4
  store i32 0, ptr %gop0, align 4
  store i32 %req, ptr %gcp0, align 4
  %pz_ok = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 0
{crit_exit}  %sl0_e = insertvalue {slice_ty} poison, ptr %pz_ok, 0
  %sl1_e = insertvalue {slice_ty} %sl0_e, {usize_ty} %req_u, 1
  %ok_e = insertvalue {res_slice_ty} zeroinitializer, {slice_ty} %sl1_e, 0
  ret {res_slice_ty} %ok_e
non_empty:
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %h = load i32, ptr %hp, align 4
  %t = load i32, ptr %tp, align 4
  %se = load i32, ptr %sp, align 4
  %no_se = icmp eq i32 %se, 0
  %h_lt_t = icmp ult i32 %h, %t
  %unwrapped = and i1 %no_se, %h_lt_t
  br i1 %unwrapped, label %back_or_wrap, label %mid_gap
back_or_wrap:
  %rem_back = sub i32 %cap, %t
  %fit_back = icmp ule i32 %req, %rem_back
  br i1 %fit_back, label %ok_b, label %try_wrap
ok_b:
  %gop_b = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp_b = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 %t, ptr %gop_b, align 4
  store i32 %req, ptr %gcp_b, align 4
  %pb = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 %t
{crit_exit}{ok_back}
try_wrap:
  %fit_wrap = icmp ule i32 %req, %h
  br i1 %fit_wrap, label %ok_w, label %fail
ok_w:
  %gop_w = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp_w = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 0, ptr %gop_w, align 4
  store i32 %req, ptr %gcp_w, align 4
  %pw = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 0
{crit_exit}{ok_wrap}
mid_gap:
  %has_se = icmp ugt i32 %se, 0
  %h_gt_t = icmp ugt i32 %h, %t
  %has_gap = or i1 %has_se, %h_gt_t
  %raw_gap = sub i32 %h, %t
  %gap = select i1 %has_gap, i32 %raw_gap, i32 0
  %fit_gap = icmp ule i32 %req, %gap
  br i1 %fit_gap, label %ok_g, label %fail
ok_g:
  %gop_g = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp_g = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 %t, ptr %gop_g, align 4
  store i32 %req, ptr %gcp_g, align 4
  %pg = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 %t
{crit_exit}{ok_gap}
fail:
{crit_exit}  %err_r = insertvalue {res_slice_ty} zeroinitializer, i32 1, 1
  ret {res_slice_ty} %err_r
}}

define void @__syn_ring_view_commit_write(ptr %r, i32 %cap, i32 %req, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gcp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  %gc = load i32, ptr %gcp, align 4
  store i32 0, ptr %gcp, align 4
  %lt = icmp ult i32 %req, %gc
  %nw = select i1 %lt, i32 %req, i32 %gc
  %is_z = icmp eq i32 %nw, 0
  br i1 %is_z, label %done, label %apply
apply:
  %gop = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %goff = load i32, ptr %gop, align 4
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_w0 = icmp eq i32 %goff, 0
  %t_gt0 = icmp ugt i32 %t, 0
  %se_eq0 = icmp eq i32 %se, 0
  %c0 = and i1 %is_w0, %t_gt0
  %is_early = and i1 %c0, %se_eq0
  br i1 %is_early, label %early_wrap, label %linear
early_wrap:
  %cnt_z = icmp eq i32 %cnt, 0
  br i1 %cnt_z, label %ew_empty, label %ew_split
ew_empty:
  %hp_e = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  store i32 0, ptr %hp_e, align 4
  store i32 %nw, ptr %tp, align 4
  store i32 %nw, ptr %cp, align 4
  br label %wake
ew_split:
  store i32 %t, ptr %sp, align 4
  store i32 %nw, ptr %tp, align 4
  %nc_ew = add i32 %cnt, %nw
  store i32 %nc_ew, ptr %cp, align 4
  br label %wake
linear:
  %nt = add i32 %t, %nw
  %wrap = icmp uge i32 %nt, %cap
  %do_wrap = and i1 %se_eq0, %wrap
  %nt_final = select i1 %do_wrap, i32 0, i32 %nt
  store i32 %nt_final, ptr %tp, align 4
  %nc_l = add i32 %cnt, %nw
  store i32 %nc_l, ptr %cp, align 4
  br label %wake
wake:
  {call_kw} void @__syn_ring_view_wake(ptr %r, i32 %boost)
  br label %done
done:
{crit_exit}  ret void
}}

define {slice_ty} @__syn_ring_view_read_slice(ptr %r, i32 %cap){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %s1 = {call_kw} i32 @__syn_ring_view_seg1(ptr %r, i32 %cap)
  %has = icmp ugt i32 %s1, 0
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %idx = select i1 %has, i32 %h, i32 0
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 %idx
{crit_exit}{conv_s1_r}
  %sl0 = insertvalue {slice_ty} poison, ptr %bp, 0
  %sl1 = insertvalue {slice_ty} %sl0, {usize_ty} %s1_u, 1
  ret {slice_ty} %sl1
}}

define {slice_ty} @__syn_ring_view_poll_read(ptr %r, i32 %cap, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %s1 = {call_kw} i32 @__syn_ring_view_seg1(ptr %r, i32 %cap)
  %has = icmp ugt i32 %s1, 0
  br i1 %has, label %ready, label %empty
ready:
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 %h
  {call_kw} void @__syn_ring_view_apply_boost(ptr %r, ptr %task)
{crit_exit}{conv_s1_p}
  %sl0 = insertvalue {slice_ty} poison, ptr %bp, 0
  %sl1 = insertvalue {slice_ty} %sl0, {usize_ty} %s1_u, 1
  ret {slice_ty} %sl1
empty:
{crit_exit}  ret {slice_ty} zeroinitializer
}}

define void @__syn_ring_view_consume(ptr %r, i32 %cap, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  {call_kw} void @__syn_ring_view_advance(ptr %r, i32 %cap, i32 %req)
{crit_exit}  ret void
}}

define i32 @__syn_ring_view_pop(ptr %r, i32 %cap, ptr %dst_ptr, i32 %dst_len, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %lt = icmp ult i32 %dst_len, %cnt
  %rem_raw = select i1 %lt, i32 %dst_len, i32 %cnt
  %is_null = icmp eq ptr %dst_ptr, null
  %rem = select i1 %is_null, i32 0, i32 %rem_raw
  %has = icmp ugt i32 %rem, 0
  br i1 %has, label %do_pop, label %empty
do_pop:
  %s1 = {call_kw} i32 @__syn_ring_view_seg1(ptr %r, i32 %cap)
  %lt_s1 = icmp ult i32 %rem, %s1
  %c1 = select i1 %lt_s1, i32 %rem, i32 %s1
  %c2 = sub i32 %rem, %c1
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  br label %l1
l1:
  %i1 = phi i32 [ 0, %do_pop ], [ %i1n, %b1 ]
  %d1 = icmp uge i32 %i1, %c1
  br i1 %d1, label %l2_init, label %b1
b1:
  %src1_i = add i32 %h, %i1
  %sp1 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 %src1_i
  %v1 = load i8, ptr %sp1, align 1
  %dp1 = getelementptr i8, ptr %dst_ptr, i32 %i1
  store i8 %v1, ptr %dp1, align 1
  %i1n = add i32 %i1, 1
  br label %l1
l2_init:
  br label %l2
l2:
  %i2 = phi i32 [ 0, %l2_init ], [ %i2n, %b2 ]
  %d2 = icmp uge i32 %i2, %c2
  br i1 %d2, label %adv, label %b2
b2:
  %sp2 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 %i2
  %v2 = load i8, ptr %sp2, align 1
  %dst2_i = add i32 %c1, %i2
  %dp2 = getelementptr i8, ptr %dst_ptr, i32 %dst2_i
  store i8 %v2, ptr %dp2, align 1
  %i2n = add i32 %i2, 1
  br label %l2
adv:
  {call_kw} void @__syn_ring_view_advance(ptr %r, i32 %cap, i32 %rem)
  {call_kw} void @__syn_ring_view_apply_boost(ptr %r, ptr %task)
{crit_exit}  ret i32 %rem
empty:
{crit_exit}  ret i32 0
}}

define {{ i8, i32 }} @__syn_ring_view_try_pop_byte(ptr %r, i32 %cap, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %has = icmp ugt i32 %cnt, 0
  br i1 %has, label %ok, label %empty
ok:
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 8, i32 %h
  %b = load i8, ptr %bp, align 1
  {call_kw} void @__syn_ring_view_advance(ptr %r, i32 %cap, i32 1)
  {call_kw} void @__syn_ring_view_apply_boost(ptr %r, ptr %task)
{crit_exit}  %r0 = insertvalue {{ i8, i32 }} zeroinitializer, i8 %b, 0
  ret {{ i8, i32 }} %r0
empty:
{crit_exit}  %re = insertvalue {{ i8, i32 }} zeroinitializer, i32 1, 1
  ret {{ i8, i32 }} %re
}}

define void @__syn_ring_view_clear(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %gop = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7
  store i32 0, ptr %hp, align 4
  store i32 0, ptr %tp, align 4
  store i32 0, ptr %cp, align 4
  store i32 0, ptr %sp, align 4
  store i32 0, ptr %gop, align 4
  store i32 0, ptr %gcp, align 4
  store i8 0, ptr %bp, align 1
{crit_exit}  ret void
}}"#
            );
        }
        Ok(())
    }

    pub(super) fn emit_coroutine_frame_globals_ir(&self, out: &mut String) {
        let _ = writeln!(
            out,
            "; ── Packed Static Coroutine Frames (with Union-Aliased Child Storage) ──"
        );
        for f in &self.module.functions {
            if let Some(layout) = self.fn_frames.get(&f.name) {
                let joined = layout.field_llvm_tys.join(", ");
                let _ = writeln!(out, "{} = type {{ {joined} }}", layout.llvm_name);
            }
        }
        for bp in &self.module.task_blueprints {
            if let Some(layout) = self.bp_frames.get(&bp.name) {
                let joined = layout.field_llvm_tys.join(", ");
                let _ = writeln!(out, "{} = type {{ {joined} }}", layout.llvm_name);
            }
        }
        for callee in &self.module.boot.boot_yielding_callees {
            if let Some(layout) = self.fn_frames.get(callee) {
                let sym = format!("@__syn_boot_frame_{}", sanitize_ident(callee));
                let _ = writeln!(out, "{sym} = global {} zeroinitializer, align 8", layout.llvm_name);
            }
        }
        for sp in &self.module.spawned_tasks {
            if let Some(layout) = self.bp_frames.get(&sp.blueprint_name) {
                let sym = format!("@__syn_task_frame_{}", sp.slot_id);
                let _ = writeln!(
                    out,
                    "{sym} = global {} zeroinitializer, align 8 ; instance `{}`",
                    layout.llvm_name, sp.instance_name
                );
            }
        }
        let _ = writeln!(out);
    }

    pub(super) fn emit_extern_fns_ir(&self, out: &mut String) {
        if self.module.extern_fns.is_empty() {
            return;
        }
        let fn_as = self.fn_as();
        let _ = writeln!(out, "; ── Genuine External \"C\" FFI Declarations ──");
        for ext in &self.module.extern_fns {
            let ret = self.llvm_ty(&ext.ret_ty);
            let mut ps: Vec<String> =
                ext.params.iter().map(|(_, t)| self.llvm_ty(t)).collect();
            if ext.variadic {
                ps.push("...".to_string());
            }
            let _ = writeln!(
                out,
                "declare {ret} @{}({}){fn_as} nounwind",
                sanitize_ident(&ext.name),
                ps.join(", ")
            );
        }
        let _ = writeln!(out);
    }

    pub(super) fn emit_isr_ir(&mut self, out: &mut String, isr: &MirIsr) -> Result<(), CodegenError> {
        let sym = format!("@__syn_isr_{}", sanitize_ident(&isr.name));
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let mut param_sig = Vec::new();
        for (idx, &vid) in isr.params.iter().enumerate() {
            let pty = self.llvm_ty(&isr.vars[vid.0].ty);
            param_sig.push(format!("{pty} %arg{idx}"));
        }
        for (sidx, _) in isr.state_vars.iter().enumerate() {
            param_sig.push(format!("ptr %__state_{sidx}"));
        }

        let mut fb = FnIrBuilder::new(self, &isr.vars, None, &MirType::Void, &MirErrorSpec::None);
        for (sidx, sd) in isr.state_vars.iter().enumerate() {
            fb.isr_state_ptrs
                .insert(sd.var_id, format!("%__state_{sidx}"));
        }
        fb.emit_entry_allocas(&isr.params);
        for stmt in &isr.body {
            fb.emit_stmt(stmt)?;
        }
        fb.ensure_terminated_void();
        let body = fb.finish();
        let _ = writeln!(out, "; ── Hardware ISR `{}` ──", isr.name);
        let _ = writeln!(
            out,
            "define void {sym}({}){fn_as} nounwind{fn_sz} {{\n{body}}}\n",
            param_sig.join(", ")
        );
        Ok(())
    }

    pub(super) fn format_irq_binding_call(&mut self, binding: &MirIrqBinding) -> Result<String, CodegenError> {
        let isr_opt = self
            .module
            .isrs
            .iter()
            .find(|i| i.name == binding.isr_name)
            .cloned();
        let isr_sym = format!("@__syn_isr_{}", sanitize_ident(&binding.isr_name));
        let mut args = Vec::new();

        for (idx, barg) in binding.baked_args.iter().enumerate() {
            let expected_ty = isr_opt
                .as_ref()
                .and_then(|isr| isr.params.get(idx).map(|vid| isr.vars[vid.0].ty.clone()));
            match barg {
                MirBakedArg::PrimAddr(prim_name) => {
                    let raw = format!("@__syn_prim_{}", sanitize_ident(prim_name));
                    if matches!(expected_ty, Some(MirType::PrimView(_))) {
                        if let Some(pk) = self.prim_globals.get(prim_name) {
                            if let Some(wrapped) = self.wrap_const_ptr_as_view(&raw, pk) {
                                args.push(wrapped);
                                continue;
                            }
                        }
                    }
                    args.push(format!("ptr {raw}"));
                }
                MirBakedArg::SubsystemPath {
                    root,
                    root_struct,
                    field_chain,
                } => {
                    let mut cur_ptr = format!("@__zev_subsys_{}", sanitize_ident(root));
                    let mut cur_struct = root_struct.as_str();
                    let mut leaf_ty: Option<MirType> = Some(MirType::Struct(root_struct.clone()));
                    for (fname, _) in field_chain {
                        let si = &self.structs[cur_struct];
                        let (fidx, fty) = &si.field_indices[fname];
                        cur_ptr = format!(
                            "getelementptr inbounds ({}, ptr {cur_ptr}, i32 0, i32 {fidx})",
                            si.llvm_name
                        );
                        leaf_ty = Some(fty.clone());
                        if let MirType::Struct(next_s) = fty {
                            cur_struct = next_s.as_str();
                        }
                    }
                    if matches!(expected_ty, Some(MirType::PrimView(_))) {
                        if let Some(MirType::Primitive(ref pk)) = leaf_ty {
                            if let Some(wrapped) = self.wrap_const_ptr_as_view(&cur_ptr, pk) {
                                args.push(wrapped);
                                continue;
                            }
                        }
                    }
                    args.push(format!("ptr {cur_ptr}"));
                }
                MirBakedArg::MmioAddr(inst_name) => {
                    if self.spec.isa == CpuIsa::HostX86_64Linux {
                        args.push(format!("ptr @__zev_mmio_buf_{}", sanitize_ident(inst_name)));
                    } else {
                        let addr = self
                            .mmio_instances
                            .get(inst_name)
                            .map_or(0, |m| m.base_addr as i32);
                        args.push(format!("ptr inttoptr (i32 {addr} to ptr)"));
                    }
                }
                MirBakedArg::Const(expr) => {
                    let target_mir_ty = expected_ty.unwrap_or_else(|| expr.ty.clone());
                    let lty = self.llvm_ty(&target_mir_ty);
                    let cval = self.const_expr_to_llvm(expr, &target_mir_ty)?;
                    args.push(format!("{lty} {cval}"));
                }
            }
        }

        for sv in &binding.state_vars {
            let sym = sanitize_ident(sv.storage_symbol.trim_start_matches('@'));
            args.push(format!("ptr @{sym}"));
        }

        let call_kw = self.call_kw();
        Ok(format!("{call_kw} void {isr_sym}({})", args.join(", ")))
    }

    pub(super) fn emit_irq_vector_dispatchers_ir(&mut self, out: &mut String) -> Result<(), CodegenError> {
        if self.module.irq_vectors.is_empty() {
            return Ok(());
        }
        let fn_sz = self.opts.fn_size_attrs().to_string();
        let fn_as = self.fn_as();
        let vectors: Vec<MirIrqVector> = self.module.irq_vectors.clone();
        let _ = writeln!(out, "; ── Zero-RAM Static Flash IRQ Vector Dispatchers ──");
        for vec in &vectors {
            let irq_num = vec.irq_num;
            let _ = writeln!(
                out,
                "define void @__syn_vec_irq_{irq_num}(){fn_as} nounwind{fn_sz} {{"
            );
            let _ = writeln!(out, "entry:");

            let eff_selector: Option<String> = vec.selector_symbol.clone().or_else(|| {
                vec.bindings
                    .iter()
                    .find_map(|b| b.cond_selector.as_ref().map(|(s, _)| s.clone()))
            });

            if let Some(sel_raw) = eff_selector {
                let sel_sym = sanitize_ident(sel_raw.trim_start_matches('@'));
                let _ = writeln!(out, "  %sel = load i8, ptr @{sel_sym}, align 1");
                if !vec.is_shared {
                    let mut uncond_calls = Vec::new();
                    let mut grouped: BTreeMap<u8, Vec<String>> = BTreeMap::new();
                    for b in &vec.bindings {
                        let call_ir = self.format_irq_binding_call(b)?;
                        if let Some((_, val)) = &b.cond_selector {
                            grouped.entry(*val).or_default().push(call_ir);
                        } else {
                            uncond_calls.push(call_ir);
                        }
                    }
                    for call_ir in uncond_calls {
                        let _ = writeln!(out, "  {call_ir}");
                    }
                    let _ = writeln!(out, "  switch i8 %sel, label %done [");
                    for &val in grouped.keys() {
                        let _ = writeln!(out, "    i8 {val}, label %arm_{val}");
                    }
                    let _ = writeln!(out, "  ]");
                    for (val, calls) in grouped {
                        let _ = writeln!(out, "arm_{val}:");
                        for call_ir in calls {
                            let _ = writeln!(out, "  {call_ir}");
                        }
                        let _ = writeln!(out, "  br label %done");
                    }
                    let _ = writeln!(out, "done:");
                    let _ = writeln!(out, "  ret void");
                } else {
                    for (idx, b) in vec.bindings.iter().enumerate() {
                        let call_ir = self.format_irq_binding_call(b)?;
                        if let Some((_, val)) = &b.cond_selector {
                            let _ = writeln!(out, "  %cmp_{idx} = icmp eq i8 %sel, {val}");
                            let _ = writeln!(
                                out,
                                "  br i1 %cmp_{idx}, label %shared_exec_{idx}, label %shared_next_{idx}"
                            );
                            let _ = writeln!(out, "shared_exec_{idx}:");
                            let _ = writeln!(out, "  {call_ir}");
                            let _ = writeln!(out, "  br label %shared_next_{idx}");
                            let _ = writeln!(out, "shared_next_{idx}:");
                        } else {
                            let _ = writeln!(out, "  {call_ir}");
                        }
                    }
                    let _ = writeln!(out, "  ret void");
                }
            } else {
                for b in &vec.bindings {
                    let call_ir = self.format_irq_binding_call(b)?;
                    let _ = writeln!(out, "  {call_ir}");
                }
                let _ = writeln!(out, "  ret void");
            }

            let _ = writeln!(out, "}}\n");
        }
        Ok(())
    }

    pub(super) fn emit_function_ir(
        &mut self,
        out: &mut String,
        func: &MirFunction,
    ) -> Result<(), CodegenError> {
        let sig = self.fn_sigs[&func.name].clone();
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        match &func.body {
            MirFnBody::Direct(stmts) => {
                let mut param_sig = Vec::new();
                for (idx, &vid) in func.params.iter().enumerate() {
                    let pty = self.llvm_ty(&func.vars[vid.0].ty);
                    param_sig.push(format!("{pty} %arg{idx}"));
                }
                let mut fb = FnIrBuilder::new(
                    self,
                    &func.vars,
                    None,
                    &func.ret_ty,
                    &func.error_spec,
                );
                fb.emit_entry_allocas(&func.params);
                for stmt in stmts {
                    fb.emit_stmt(stmt)?;
                }
                fb.ensure_terminated_default();
                let body = fb.finish();
                let _ = writeln!(
                    out,
                    "define {} {}({}){fn_as} nounwind{fn_sz} {{\n{body}}}\n",
                    sig.ret_llvm_ty,
                    sig.symbol,
                    param_sig.join(", ")
                );
            }
            MirFnBody::Coroutine { blocks, .. } => {
                let layout = self.fn_frames[&func.name].clone();
                let mut fb = FnIrBuilder::new(
                    self,
                    &func.vars,
                    Some(&layout),
                    &func.ret_ty,
                    &func.error_spec,
                );
                fb.emit_coroutine_cfg(blocks)?;
                let body = fb.finish();
                let _ = writeln!(
                    out,
                    "define i32 {}(ptr %__frame, ptr %__task){fn_as} nounwind{fn_sz} {{\n{body}}}\n",
                    sig.symbol
                );
            }
        }
        Ok(())
    }

    pub(super) fn emit_task_blueprint_ir(
        &mut self,
        out: &mut String,
        bp: &MirTaskBlueprint,
    ) -> Result<(), CodegenError> {
        let layout = self.bp_frames[&bp.name].clone();
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let sym = format!("@__syn_task_step_{}", sanitize_ident(&bp.name));
        let mut fb = FnIrBuilder::new(
            self,
            &bp.vars,
            Some(&layout),
            &MirType::Void,
            &MirErrorSpec::None,
        );
        fb.emit_coroutine_cfg(&bp.blocks)?;
        let body = fb.finish();
        let _ = writeln!(
            out,
            "define i32 {sym}(ptr %__frame, ptr %__task){fn_as} nounwind{fn_sz} {{\n{body}}}\n"
        );
        Ok(())
    }

    pub(super) fn emit_boot_ir(&mut self, out: &mut String) -> Result<(), CodegenError> {
        let boot_vars = self.module.boot.vars.clone();
        let boot_present = self.module.boot.present;
        let steps = self.module.boot.steps.clone();
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let mut fb = FnIrBuilder::new(
            self,
            &boot_vars,
            None,
            &MirType::Void,
            &MirErrorSpec::None,
        );
        fb.emit_entry_allocas(&[]);
        fb.emit_pool_boot_inits();

        if boot_present {
            for step in &steps {
                match step {
                    MirBootStep::Stmt(s) => fb.emit_stmt(s)?,
                    MirBootStep::RunYieldingFn { callee, args, dst } => {
                        fb.emit_boot_run_yielding(callee, args, dst.as_ref())?;
                    }
                    MirBootStep::SpawnActivate { slot_id, args } => {
                        fb.emit_boot_spawn_activate(*slot_id, args)?;
                    }
                }
            }
        }
        fb.ensure_terminated_void();
        let body = fb.finish();
        let _ = writeln!(out, "; ── Sequential `fn main()` Boot Phase ──");
        let _ = writeln!(
            out,
            "define void @__syn_boot(){fn_as} nounwind{fn_sz} {{\n{body}}}\n"
        );
        Ok(())
    }

    pub(super) fn emit_rodata_consts_ir(&mut self, out: &mut String) -> Result<(), CodegenError> {
        if self.rodata_consts.is_empty() {
            return Ok(());
        }
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let _ = writeln!(out, "; ── Immutable Aggregate / Slice / String Constants (Flash .rodata) ──");
        for i in 0..self.module.consts.len() {
            let cname = self.module.consts[i].name.clone();
            let Some((sym, _)) = self.rodata_consts.get(&cname).cloned() else {
                continue;
            };
            let cty = self.module.consts[i].ty.clone();
            let cinit = self.module.consts[i].init.clone();
            let lty = self.llvm_ty(&cty);
            let (_, al) = self.size_align(&cty, ptr_bytes);
            let al = al.max(1);
            let init = self.const_expr_to_llvm(&cinit, &cty)?;
            let _ = writeln!(
                out,
                "{sym} = private unnamed_addr constant {lty} {init}, align {al}"
            );
        }
        let _ = writeln!(out);
        Ok(())
    }

    pub(super) fn emit_string_constants_ir(&self, out: &mut String) {
        if self.strings.is_empty() && self.const_blobs.is_empty() {
            return;
        }
        let _ = writeln!(out, "; ── Pooled String & Slice Constants (Flash .rodata) ──");
        for (idx, s) in self.strings.iter().enumerate() {
            let bytes = zev_ast::decoded_string_bytes(s);
            let len = bytes.len() + 1;
            let mut esc = String::new();
            for &b in &bytes {
                if (0x20..=0x7E).contains(&b) && b != b'"' && b != b'\\' {
                    esc.push(b as char);
                } else {
                    let _ = write!(esc, "\\{b:02X}");
                }
            }
            esc.push_str("\\00");
            let _ = writeln!(
                out,
                "@.str.{idx} = private unnamed_addr constant [{len} x i8] c\"{esc}\", align 1"
            );
        }
        for (idx, decl_body) in self.const_blobs.iter().enumerate() {
            let _ = writeln!(out, "@.cblob.{idx} = {decl_body}");
        }
    }
}
