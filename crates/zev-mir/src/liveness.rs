//! Exact backward CFG liveness analysis across coroutine yield points,
//! automatic variable hoisting (static frame vs transient stack), and
//! disjoint child coroutine frame union-aliasing.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{
    BlockId, MirAssignOp, MirBlock, MirCallDst, MirChildCallSlot, MirExpr, MirExprKind,
    MirFmtChunk, MirFnBody, MirFunction, MirIsr, MirPlace, MirPrimNonBlockingOp, MirPrimReadOp,
    MirPrimTarget, MirStmt, MirTaskBlueprint, MirTerminator, MirType, MirVar, MirWaitArmOp,
    MirYieldKind, VarId, VarRole,
};

/// Record `addr_taken` flags on variables inside an ISR handler.
pub fn analyze_isr_frame(isr: &mut MirIsr) {
    for v in &mut isr.vars {
        v.hoisted = false;
    }
    for s in &isr.body {
        collect_addr_taken_stmt(s, &mut isr.vars);
    }
}

/// Run variable hoisting and child-frame union packing on a task blueprint.
pub fn analyze_task_frame(task: &mut MirTaskBlueprint) {
    let param_set: BTreeSet<VarId> = task.params.iter().copied().collect();
    compute_hoisting(&param_set, &mut task.vars, &task.blocks);
    assign_child_union_groups(&task.blocks, &mut task.child_slots);
}

/// Run variable hoisting and child-frame union packing on a yielding function.
pub fn analyze_fn_frame(func: &mut MirFunction) {
    let param_set: BTreeSet<VarId> = func.params.iter().copied().collect();
    match &mut func.body {
        MirFnBody::Direct(stmts) => {
            for v in &mut func.vars {
                v.hoisted = false;
            }
            for s in stmts {
                collect_addr_taken_stmt(s, &mut func.vars);
            }
        }
        MirFnBody::Coroutine {
            blocks,
            child_slots,
        } => {
            compute_hoisting(&param_set, &mut func.vars, blocks);
            assign_child_union_groups(blocks, child_slots);
        }
    }
}

fn compute_hoisting(param_set: &BTreeSet<VarId>, vars: &mut [MirVar], blocks: &[MirBlock]) {
    let block_index: BTreeMap<BlockId, usize> =
        blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    let n = blocks.len();
    let mut live_in: Vec<BTreeSet<VarId>> = vec![BTreeSet::new(); n];
    let mut live_out: Vec<BTreeSet<VarId>> = vec![BTreeSet::new(); n];
    let mut force_hoist: BTreeSet<VarId> = param_set.clone();

    // Pre-scan for variables whose address is passed into a yielding call or stored,
    // plus explicit coroutine timer/channel state variables referenced across yields.
    for blk in blocks {
        for stmt in &blk.stmts {
            collect_addr_taken_stmt(stmt, vars);
            // If `&var` is stored into another variable/struct, conservatively hoist `var`
            // if that pointer could cross a yield; we also mark `addr_taken` on `MirVar`.
        }
        collect_terminator_forced_hoists(&blk.terminator, &mut force_hoist, vars);
    }

    // Standard iterative backward dataflow to fixed point.
    let mut changed = true;
    while changed {
        changed = false;
        for i in (0..n).rev() {
            let blk = &blocks[i];
            let new_out = compute_terminator_live_before(&blk.terminator, &block_index, &live_in);

            let mut cur = new_out.clone();
            for stmt in blk.stmts.iter().rev() {
                transfer_stmt(stmt, &mut cur);
            }

            if new_out != live_out[i] || cur != live_in[i] {
                live_out[i] = new_out;
                live_in[i] = cur;
                changed = true;
            }
        }
    }

    // Collect all variables live at any resume entry block (other than initial entry bb0).
    let mut live_across_yield = force_hoist;
    for (idx, blk) in blocks.iter().enumerate() {
        if idx > 0 && blk.is_resume_entry {
            for &vid in &live_in[idx] {
                live_across_yield.insert(vid);
            }
        }
    }
    // Also inspect every Yield terminator's resume targets directly.
    for blk in blocks {
        for resume_id in blk.terminator.yield_resume_blocks() {
            if let Some(&ridx) = block_index.get(&resume_id) {
                for &vid in &live_in[ridx] {
                    live_across_yield.insert(vid);
                }
            }
        }
    }

    // If a pointer/slice/struct variable `p` is hoisted and was assigned `&x` or derived from `x`
    // (including via `fmt_buf(x, ...)`), ensure `x` is hoisted too so `_f->p` never holds a
    // dangling pointer to a C stack variable!
    let mut alias_edges: Vec<(VarId, VarId)> = Vec::new(); // (holder, pointee)
    for blk in blocks {
        for stmt in &blk.stmts {
            collect_alias_edges_stmt(stmt, &mut alias_edges);
        }
    }
    let mut alias_changed = true;
    while alias_changed {
        alias_changed = false;
        for &(holder, pointee) in &alias_edges {
            if live_across_yield.contains(&holder) && live_across_yield.insert(pointee) {
                alias_changed = true;
            }
        }
    }

    for v in vars.iter_mut() {
        v.hoisted = match v.role {
            VarRole::Param | VarRole::IsrState => true,
            VarRole::UserLocal | VarRole::CompilerTemp => live_across_yield.contains(&v.id),
        };
    }
}

/// Exact backward transfer function across `MirTerminator`.
fn compute_terminator_live_before(
    term: &MirTerminator,
    block_index: &BTreeMap<BlockId, usize>,
    live_in: &[BTreeSet<VarId>],
) -> BTreeSet<VarId> {
    let block_live = |bid: BlockId| -> BTreeSet<VarId> {
        block_index
            .get(&bid)
            .map(|&idx| live_in[idx].clone())
            .unwrap_or_default()
    };

    match term {
        MirTerminator::Goto(target) => block_live(*target),
        MirTerminator::Branch {
            cond,
            then_bb,
            else_bb,
        } => {
            let mut out = block_live(*then_bb);
            out.extend(block_live(*else_bb));
            collect_expr_uses(cond, &mut out);
            out
        }
        MirTerminator::Switch {
            discr,
            arms,
            default_bb,
        } => {
            let mut out = block_live(*default_bb);
            for (_, b) in arms {
                out.extend(block_live(*b));
            }
            collect_expr_uses(discr, &mut out);
            out
        }
        MirTerminator::Return(expr_opt) => {
            let mut out = BTreeSet::new();
            if let Some(e) = expr_opt {
                collect_expr_uses(e, &mut out);
            }
            out
        }
        MirTerminator::Error { reason, partial } => {
            let mut out = BTreeSet::new();
            if let Some(r) = reason {
                collect_expr_uses(r, &mut out);
            }
            if let Some(p) = partial {
                collect_expr_uses(p, &mut out);
            }
            out
        }
        MirTerminator::Yield(yk) => match yk {
            MirYieldKind::EveryLatch {
                next_deadline_var,
                period_var,
                resume_bb,
            } => {
                let mut out = block_live(*resume_bb);
                out.insert(*next_deadline_var);
                out.insert(*period_var);
                out
            }
            MirYieldKind::ShareCheckpoint { resume_bb } => block_live(*resume_bb),
            MirYieldKind::DelayPoll {
                deadline_var,
                resume_bb,
                ..
            } => {
                let mut out = block_live(*resume_bb);
                out.insert(*deadline_var);
                out
            }
            MirYieldKind::Wait { arms, timer, .. } => {
                let mut out = BTreeSet::new();
                for arm in arms {
                    let mut arm_live = block_live(arm.target_bb);
                    match &arm.op {
                        MirWaitArmOp::InboxRecv {
                            dst_val, dst_res, ..
                        }
                        | MirWaitArmOp::InboxViewRecv {
                            dst_val, dst_res, ..
                        }
                        | MirWaitArmOp::WatchWaitChange {
                            dst_val, dst_res, ..
                        }
                        | MirWaitArmOp::PoolAlloc {
                            dst_val, dst_res, ..
                        }
                        | MirWaitArmOp::ByteRingWaitRead {
                            dst_val, dst_res, ..
                        }
                        | MirWaitArmOp::ByteRingViewWaitRead {
                            dst_val, dst_res, ..
                        }
                        | MirWaitArmOp::ByteRingWaitWrite {
                            dst_val, dst_res, ..
                        }
                        | MirWaitArmOp::ByteRingViewWaitWrite {
                            dst_val, dst_res, ..
                        } => {
                            if let Some(dv) = dst_val {
                                arm_live.remove(dv);
                            }
                            if let Some(dr) = dst_res {
                                arm_live.remove(dr);
                            }
                        }
                        MirWaitArmOp::InboxSend { dst_res, .. }
                        | MirWaitArmOp::InboxViewSend { dst_res, .. }
                        | MirWaitArmOp::SignalWait { dst_res, .. }
                        | MirWaitArmOp::WaitQueueCommit { dst_res, .. } => {
                            if let Some(dr) = dst_res {
                                arm_live.remove(dr);
                            }
                        }
                        MirWaitArmOp::PingPongWaitChunk { dst_slice, .. } => {
                            arm_live.remove(dst_slice);
                        }
                    }
                    out.extend(arm_live);
                }
                if let Some(t) = timer {
                    let mut t_live = block_live(t.timeout_bb);
                    if let Some(dr) = t.timeout_err_res {
                        t_live.remove(&dr);
                    }
                    out.extend(t_live);
                    out.insert(t.deadline_var);
                    if let Some((lv, _)) = &t.release_lease_on_timeout {
                        out.insert(*lv);
                    }
                }
                for arm in arms {
                    match &arm.op {
                        MirWaitArmOp::InboxRecv { target, .. }
                        | MirWaitArmOp::PoolAlloc { target, .. }
                        | MirWaitArmOp::ByteRingWaitRead { target, .. }
                        | MirWaitArmOp::ByteRingWaitWrite { target, .. }
                        | MirWaitArmOp::SignalWait { target, .. }
                        | MirWaitArmOp::WaitQueueCommit { target, .. }
                        | MirWaitArmOp::PingPongWaitChunk { target, .. } => {
                            collect_prim_target_uses(target, &mut out);
                        }
                        MirWaitArmOp::WatchWaitChange {
                            target,
                            last_gen_var,
                            seeded_var,
                            ..
                        } => {
                            collect_prim_target_uses(target, &mut out);
                            out.insert(*last_gen_var);
                            if let Some(sv) = seeded_var {
                                out.insert(*sv);
                            }
                        }
                        MirWaitArmOp::InboxSend {
                            target, val_var, ..
                        } => {
                            collect_prim_target_uses(target, &mut out);
                            out.insert(*val_var);
                        }
                        MirWaitArmOp::InboxViewRecv { view, .. }
                        | MirWaitArmOp::ByteRingViewWaitRead { view, .. }
                        | MirWaitArmOp::ByteRingViewWaitWrite { view, .. } => {
                            collect_expr_uses(view, &mut out);
                        }
                        MirWaitArmOp::InboxViewSend { view, val_var, .. } => {
                            collect_expr_uses(view, &mut out);
                            out.insert(*val_var);
                        }
                    }
                }
                out
            }
            MirYieldKind::CallYielding { args, poll_bb, .. } => {
                let mut out = block_live(*poll_bb);
                for arg in args {
                    collect_expr_uses(arg, &mut out);
                }
                out
            }
            MirYieldKind::PollChild { dst, resume_bb, .. } => {
                let mut out = block_live(*resume_bb);
                if let Some(d) = dst {
                    match d {
                        MirCallDst::Value(v) | MirCallDst::Result(v) => {
                            out.remove(v);
                        }
                    }
                }
                out
            }
        },
    }
}

fn transfer_stmt(stmt: &MirStmt, live: &mut BTreeSet<VarId>) {
    match stmt {
        MirStmt::Assign { target, op, value } => {
            match target {
                MirPlace::Var(vid) if *op == MirAssignOp::Assign => {
                    // Full overwrite of `vid` kills `vid`, then adds `uses(value)`.
                    live.remove(vid);
                    collect_expr_uses(value, live);
                }
                MirPlace::Var(vid) => {
                    // Compound assignment (`x += v`) both reads and writes `x`.
                    live.insert(*vid);
                    collect_expr_uses(value, live);
                }
                other => {
                    // Partial update (`s.field = v`, `a[i] = v`) or pointer deref (`*p = v`):
                    // reads the base/index/pointer as well as `value`.
                    collect_place_uses_for_partial_write(other, live);
                    collect_expr_uses(value, live);
                }
            }
        }
        MirStmt::Expr(e) => collect_expr_uses(e, live),
        MirStmt::PrimOp(MirPrimNonBlockingOp::LeaseNullify { target, .. }) => match target {
            MirPlace::Var(vid) => {
                live.remove(vid);
            }
            other => collect_place_uses_for_partial_write(other, live),
        },
        MirStmt::PrimOp(op) => collect_prim_op_uses(op, live),
        MirStmt::Intrinsic(intr) => match intr {
            crate::ir::MirIntrinsicOp::UseSystick { hz } => collect_expr_uses(hz, live),
            crate::ir::MirIntrinsicOp::SetPriority { .. } => {}
            crate::ir::MirIntrinsicOp::SysOp { args, .. } => {
                for a in args {
                    collect_expr_uses(a, live);
                }
            }
        },
        MirStmt::EnterCritical
        | MirStmt::ExitCritical
        | MirStmt::EndInboxTurn
        | MirStmt::Break
        | MirStmt::Continue
        | MirStmt::BoundedForLatch
        | MirStmt::ZeroTaskLocals => {}
        MirStmt::SpawnActivate { args, .. } => {
            for a in args {
                collect_expr_uses(a, live);
            }
        }
        MirStmt::PrintFmt { chunks, .. } => {
            for ch in chunks {
                collect_fmt_chunk_uses(ch, live);
            }
        }
        MirStmt::FmtBuf {
            dst_res,
            buf,
            chunks,
        }
        | MirStmt::FmtRing {
            dst_res,
            ring: buf,
            chunks,
            ..
        } => {
            live.remove(dst_res);
            collect_expr_uses(buf, live);
            for ch in chunks {
                collect_fmt_chunk_uses(ch, live);
            }
        }
        MirStmt::MmioWrite {
            base, index, value, ..
        } => {
            collect_expr_uses(base, live);
            if let Some(idx) = index {
                collect_expr_uses(idx, live);
            }
            collect_expr_uses(value, live);
        }
        MirStmt::SetIrqSelector { .. } => {}
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            let mut then_live = live.clone();
            for s in then_body.iter().rev() {
                transfer_stmt(s, &mut then_live);
            }
            let mut else_live = live.clone();
            for s in else_body.iter().rev() {
                transfer_stmt(s, &mut else_live);
            }
            then_live.extend(else_live);
            collect_expr_uses(cond, &mut then_live);
            *live = then_live;
        }
        MirStmt::While { cond, body } => {
            let mut loop_live = live.clone();
            for _ in 0..2 {
                collect_expr_uses(cond, &mut loop_live);
                let mut body_live = loop_live.clone();
                for s in body.iter().rev() {
                    transfer_stmt(s, &mut body_live);
                }
                loop_live.extend(body_live);
            }
            *live = loop_live;
        }
        MirStmt::Switch {
            discr,
            arms,
            default,
        } => {
            let mut merged = BTreeSet::new();
            let mut def_live = live.clone();
            for s in default.iter().rev() {
                transfer_stmt(s, &mut def_live);
            }
            merged.extend(def_live);
            for (_, arm_stmts) in arms {
                let mut arm_live = live.clone();
                for s in arm_stmts.iter().rev() {
                    transfer_stmt(s, &mut arm_live);
                }
                merged.extend(arm_live);
            }
            collect_expr_uses(discr, &mut merged);
            *live = merged;
        }
        MirStmt::Return(expr_opt) => {
            live.clear();
            if let Some(e) = expr_opt {
                collect_expr_uses(e, live);
            }
        }
        MirStmt::Error { reason, partial } => {
            live.clear();
            if let Some(r) = reason {
                collect_expr_uses(r, live);
            }
            if let Some(p) = partial {
                collect_expr_uses(p, live);
            }
        }
    }
}

fn collect_place_uses_for_partial_write(place: &MirPlace, uses: &mut BTreeSet<VarId>) {
    match place {
        MirPlace::Var(vid) => {
            uses.insert(*vid);
        }
        MirPlace::Deref(ptr_expr, _) => {
            collect_expr_uses(ptr_expr, uses);
        }
        MirPlace::Field { base, .. } => {
            collect_place_uses_for_partial_write(base, uses);
        }
        MirPlace::Index { base, index, .. } => {
            collect_place_uses_for_partial_write(base, uses);
            collect_expr_uses(index, uses);
        }
    }
}

fn collect_prim_target_uses(target: &MirPrimTarget, uses: &mut BTreeSet<VarId>) {
    if let MirPrimTarget::PtrExpr(e) = target {
        collect_expr_uses(e, uses);
    }
}

fn collect_prim_op_uses(op: &MirPrimNonBlockingOp, uses: &mut BTreeSet<VarId>) {
    match op {
        MirPrimNonBlockingOp::WatchPublish { target, value, .. }
        | MirPrimNonBlockingOp::GuardedSet { target, value, .. }
        | MirPrimNonBlockingOp::InboxPost { target, value, .. } => {
            collect_prim_target_uses(target, uses);
            collect_expr_uses(value, uses);
        }
        MirPrimNonBlockingOp::ByteRingConsume { target, len, .. }
        | MirPrimNonBlockingOp::ByteRingCommitWrite { target, len, .. }
        | MirPrimNonBlockingOp::PingPongCompleteHalf {
            target, half: len, ..
        } => {
            collect_prim_target_uses(target, uses);
            collect_expr_uses(len, uses);
        }
        MirPrimNonBlockingOp::SignalNotify { target, .. }
        | MirPrimNonBlockingOp::SignalClear { target }
        | MirPrimNonBlockingOp::WaitQueuePrepare { target }
        | MirPrimNonBlockingOp::WaitQueueCancel { target }
        | MirPrimNonBlockingOp::WaitQueueWakeAll { target, .. }
        | MirPrimNonBlockingOp::PingPongFlip { target, .. }
        | MirPrimNonBlockingOp::PingPongSetActive { target, .. }
        | MirPrimNonBlockingOp::ByteRingClear { target, .. } => {
            collect_prim_target_uses(target, uses);
        }
        MirPrimNonBlockingOp::LeaseRelease { value, .. } => {
            collect_expr_uses(value, uses);
        }
        MirPrimNonBlockingOp::LeaseNullify { target, .. } => {
            collect_place_uses_for_partial_write(target, uses);
        }
        MirPrimNonBlockingOp::InboxViewPost { view, value, .. } => {
            collect_expr_uses(view, uses);
            collect_expr_uses(value, uses);
        }
        MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
        | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
            collect_expr_uses(view, uses);
            collect_expr_uses(len, uses);
        }
        MirPrimNonBlockingOp::ByteRingViewClear { view } => {
            collect_expr_uses(view, uses);
        }
    }
}

fn collect_fmt_chunk_uses(chunk: &MirFmtChunk, uses: &mut BTreeSet<VarId>) {
    match chunk {
        MirFmtChunk::Literal(_) => {}
        MirFmtChunk::Int { value, .. }
        | MirFmtChunk::Float { value, .. }
        | MirFmtChunk::Bool(value)
        | MirFmtChunk::PaddedBool { value, .. }
        | MirFmtChunk::Str(value)
        | MirFmtChunk::PaddedStr { cstr: value, .. }
        | MirFmtChunk::Bytes(value)
        | MirFmtChunk::PaddedBytes { slice: value, .. }
        | MirFmtChunk::EnumName { value, .. }
        | MirFmtChunk::PaddedEnumName { value, .. }
        | MirFmtChunk::DebugSlice { slice: value, .. } => {
            collect_expr_uses(value, uses);
        }
    }
}

fn collect_expr_uses(expr: &MirExpr, uses: &mut BTreeSet<VarId>) {
    match &expr.kind {
        MirExprKind::ZeroValue
        | MirExprKind::IntLit(_)
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
        | MirExprKind::SysQuery(_) => {}
        MirExprKind::Var(vid)
        | MirExprKind::ResultIsErr(vid)
        | MirExprKind::ResultErr(vid)
        | MirExprKind::ResultVal(vid) => {
            uses.insert(*vid);
        }
        MirExprKind::Unary { expr: inner, .. }
        | MirExprKind::Cast { expr: inner, .. }
        | MirExprKind::CStrToSlice(inner)
        | MirExprKind::SliceLen(inner)
        | MirExprKind::MakePrimView { ptr: inner, .. } => {
            collect_expr_uses(inner, uses);
        }
        MirExprKind::MmioRead { base, index, .. } => {
            collect_expr_uses(base, uses);
            if let Some(idx) = index {
                collect_expr_uses(idx, uses);
            }
        }
        MirExprKind::AddrOfPlace(place)
        | MirExprKind::ArrayToSlice {
            array_place: place, ..
        } => {
            collect_place_uses_for_partial_write(place, uses);
        }
        MirExprKind::Binary { lhs, rhs, .. }
        | MirExprKind::SeqEq { lhs, rhs, .. }
        | MirExprKind::Index {
            base: lhs,
            index: rhs,
            ..
        } => {
            collect_expr_uses(lhs, uses);
            collect_expr_uses(rhs, uses);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, fexpr, _) in fields {
                collect_expr_uses(fexpr, uses);
            }
        }
        MirExprKind::ArrayLit { elems, .. } => {
            for el in elems {
                collect_expr_uses(el, uses);
            }
        }
        MirExprKind::FieldAccess { base, .. } => {
            collect_expr_uses(base, uses);
        }
        MirExprKind::SubSlice {
            base, start, end, ..
        } => {
            collect_expr_uses(base, uses);
            collect_expr_uses(start, uses);
            collect_expr_uses(end, uses);
        }
        MirExprKind::FromBytes { slice, .. } => {
            collect_expr_uses(slice, uses);
        }
        MirExprKind::PrimRead(ro) => match ro.as_ref() {
            MirPrimReadOp::InboxTrySend { target, value, .. } => {
                collect_prim_target_uses(target, uses);
                collect_expr_uses(value, uses);
            }
            MirPrimReadOp::ByteRingPush { target, src, .. }
            | MirPrimReadOp::ByteRingPop {
                target, dst: src, ..
            } => {
                collect_prim_target_uses(target, uses);
                collect_expr_uses(src, uses);
            }
            MirPrimReadOp::ByteRingPushByte { target, byte, .. } => {
                collect_prim_target_uses(target, uses);
                collect_expr_uses(byte, uses);
            }
            MirPrimReadOp::ByteRingGrantWrite { target, len, .. } => {
                collect_prim_target_uses(target, uses);
                collect_expr_uses(len, uses);
            }
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
            | MirPrimReadOp::ByteRingReadSlice { target, .. }
            | MirPrimReadOp::ByteRingTryPopByte { target, .. }
            | MirPrimReadOp::ByteRingLen { target, .. }
            | MirPrimReadOp::ByteRingFreeSpace { target, .. }
            | MirPrimReadOp::ByteRingIsEmpty { target, .. }
            | MirPrimReadOp::ByteRingIsFull { target, .. } => {
                collect_prim_target_uses(target, uses);
            }
            MirPrimReadOp::InboxViewTrySend { view, value, .. } => {
                collect_expr_uses(view, uses);
                collect_expr_uses(value, uses);
            }
            MirPrimReadOp::ByteRingViewPush { view, src, .. }
            | MirPrimReadOp::ByteRingViewPop { view, dst: src } => {
                collect_expr_uses(view, uses);
                collect_expr_uses(src, uses);
            }
            MirPrimReadOp::ByteRingViewPushByte { view, byte, .. } => {
                collect_expr_uses(view, uses);
                collect_expr_uses(byte, uses);
            }
            MirPrimReadOp::ByteRingViewGrantWrite { view, len } => {
                collect_expr_uses(view, uses);
                collect_expr_uses(len, uses);
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
                collect_expr_uses(view, uses);
            }
            MirPrimReadOp::PoolTryAlloc { target, .. }
            | MirPrimReadOp::PoolAvail { target, .. }
            | MirPrimReadOp::PoolInUse { target, .. }
            | MirPrimReadOp::PoolIsExhausted { target, .. }
            | MirPrimReadOp::PoolCap { target, .. }
            | MirPrimReadOp::DmaSlotTake { target, .. }
            | MirPrimReadOp::DmaSlotRelease { target, .. }
            | MirPrimReadOp::DmaSlotIsBusy { target, .. }
            | MirPrimReadOp::DmaSlotIsEmpty { target, .. }
            | MirPrimReadOp::DmaSlotAddr { target, .. } => {
                collect_prim_target_uses(target, uses);
            }
            MirPrimReadOp::DmaSlotPark { target, lease, .. } => {
                collect_prim_target_uses(target, uses);
                collect_expr_uses(lease, uses);
            }
        },
        MirExprKind::CallDirect { args, .. } => {
            for a in args {
                collect_expr_uses(a, uses);
            }
        }
        MirExprKind::RomCallU32 { addr, args, .. } => {
            collect_expr_uses(addr, uses);
            for a in args {
                collect_expr_uses(a, uses);
            }
        }
        MirExprKind::LeaseGet { lease, .. } | MirExprKind::LeaseAddr { lease, .. } => {
            collect_expr_uses(lease, uses);
        }
    }
}

fn root_place_var(place: &MirPlace) -> Option<VarId> {
    match place {
        MirPlace::Var(vid) => Some(*vid),
        MirPlace::Field { base, .. } | MirPlace::Index { base, .. } => root_place_var(base),
        MirPlace::Deref(_, _) => None,
    }
}

fn collect_place_addr_of_vars(place: &MirPlace, out: &mut BTreeSet<VarId>) {
    match place {
        MirPlace::Var(vid) => {
            out.insert(*vid);
        }
        MirPlace::Field { base, .. } => collect_place_addr_of_vars(base, out),
        MirPlace::Index { base, index, .. } => {
            collect_place_addr_of_vars(base, out);
            collect_addr_of_vars(index, out);
        }
        MirPlace::Deref(ptr_expr, _) => {
            collect_addr_of_vars(ptr_expr, out);
        }
    }
}

fn collect_addr_of_vars(expr: &MirExpr, out: &mut BTreeSet<VarId>) {
    match &expr.kind {
        MirExprKind::AddrOfPlace(p) | MirExprKind::ArrayToSlice { array_place: p, .. } => {
            collect_place_addr_of_vars(p, out);
        }
        MirExprKind::Unary { expr: inner, .. }
        | MirExprKind::Cast { expr: inner, .. }
        | MirExprKind::CStrToSlice(inner)
        | MirExprKind::FieldAccess { base: inner, .. }
        | MirExprKind::FromBytes { slice: inner, .. }
        | MirExprKind::SliceLen(inner) => collect_addr_of_vars(inner, out),
        MirExprKind::MmioRead { base, index, .. } => {
            collect_addr_of_vars(base, out);
            if let Some(idx) = index {
                collect_addr_of_vars(idx, out);
            }
        }
        MirExprKind::Binary { lhs, rhs, .. }
        | MirExprKind::SeqEq { lhs, rhs, .. }
        | MirExprKind::Index {
            base: lhs,
            index: rhs,
            ..
        } => {
            collect_addr_of_vars(lhs, out);
            collect_addr_of_vars(rhs, out);
        }
        MirExprKind::SubSlice {
            base, start, end, ..
        } => {
            collect_addr_of_vars(base, out);
            collect_addr_of_vars(start, out);
            collect_addr_of_vars(end, out);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, fe, _) in fields {
                collect_addr_of_vars(fe, out);
            }
        }
        MirExprKind::ArrayLit { elems, .. } | MirExprKind::CallDirect { args: elems, .. } => {
            for e in elems {
                collect_addr_of_vars(e, out);
            }
        }
        MirExprKind::RomCallU32 { addr, args, .. } => {
            collect_addr_of_vars(addr, out);
            for a in args {
                collect_addr_of_vars(a, out);
            }
        }
        MirExprKind::PrimRead(ro) => match ro.as_ref() {
            MirPrimReadOp::InboxTrySend { value, .. }
            | MirPrimReadOp::DmaSlotPark { lease: value, .. } => {
                collect_addr_of_vars(value, out);
            }
            MirPrimReadOp::ByteRingPush { src: value, .. }
            | MirPrimReadOp::ByteRingPop { dst: value, .. }
            | MirPrimReadOp::ByteRingPushByte { byte: value, .. }
            | MirPrimReadOp::ByteRingGrantWrite { len: value, .. } => {
                collect_addr_of_vars(value, out);
            }
            MirPrimReadOp::InboxViewTrySend { view, value, .. } => {
                collect_addr_of_vars(view, out);
                collect_addr_of_vars(value, out);
            }
            MirPrimReadOp::ByteRingViewPush {
                view, src: value, ..
            }
            | MirPrimReadOp::ByteRingViewPop { view, dst: value }
            | MirPrimReadOp::ByteRingViewPushByte {
                view, byte: value, ..
            }
            | MirPrimReadOp::ByteRingViewGrantWrite { view, len: value } => {
                collect_addr_of_vars(view, out);
                collect_addr_of_vars(value, out);
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
                collect_addr_of_vars(view, out);
            }
            _ => {}
        },
        MirExprKind::MakePrimView { ptr, .. } => {
            collect_addr_of_vars(ptr, out);
        }
        MirExprKind::LeaseGet { lease, .. } | MirExprKind::LeaseAddr { lease, .. } => {
            collect_addr_of_vars(lease, out);
        }
        _ => {}
    }
}

fn type_may_carry_alias(ty: &MirType) -> bool {
    matches!(
        ty,
        MirType::Slice(_)
            | MirType::Ptr(_)
            | MirType::PrimView(_)
            | MirType::Struct(_)
            | MirType::Array { .. }
            | MirType::Result { .. }
    )
}

fn collect_place_alias_sources(place: &MirPlace, out: &mut BTreeSet<VarId>) {
    match place {
        MirPlace::Var(vid) => {
            out.insert(*vid);
        }
        MirPlace::Field { base, .. } => collect_place_alias_sources(base, out),
        MirPlace::Index { base, index, .. } => {
            collect_place_alias_sources(base, out);
            collect_addr_of_vars(index, out);
        }
        MirPlace::Deref(ptr_expr, _) => {
            collect_alias_sources_expr(ptr_expr, out);
        }
    }
}

fn collect_alias_sources_expr(expr: &MirExpr, out: &mut BTreeSet<VarId>) {
    collect_addr_of_vars(expr, out);
    match &expr.kind {
        MirExprKind::AddrOfPlace(p) | MirExprKind::ArrayToSlice { array_place: p, .. } => {
            collect_place_alias_sources(p, out);
        }
        MirExprKind::Var(vid) | MirExprKind::ResultVal(vid) => {
            if type_may_carry_alias(&expr.ty) {
                out.insert(*vid);
            }
        }
        MirExprKind::SubSlice { base, .. }
        | MirExprKind::FieldAccess { base, .. }
        | MirExprKind::Index { base, .. }
        | MirExprKind::Cast { expr: base, .. }
        | MirExprKind::CStrToSlice(base)
        | MirExprKind::Unary { expr: base, .. } => {
            collect_alias_sources_expr(base, out);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, fe, _) in fields {
                collect_alias_sources_expr(fe, out);
            }
        }
        MirExprKind::ArrayLit { elems, .. } => {
            for e in elems {
                collect_alias_sources_expr(e, out);
            }
        }
        MirExprKind::CallDirect { args, .. } => {
            if type_may_carry_alias(&expr.ty) {
                for a in args {
                    collect_alias_sources_expr(a, out);
                }
            }
        }
        MirExprKind::RomCallU32 { addr, args, .. } if type_may_carry_alias(&expr.ty) => {
            collect_alias_sources_expr(addr, out);
            for a in args {
                collect_alias_sources_expr(a, out);
            }
        }
        _ => {}
    }
}

fn collect_alias_edges_stmt(stmt: &MirStmt, alias_edges: &mut Vec<(VarId, VarId)>) {
    match stmt {
        MirStmt::Assign { target, value, .. } => {
            let mut holders = BTreeSet::new();
            if let Some(h) = root_place_var(target) {
                holders.insert(h);
            } else {
                collect_place_alias_sources(target, &mut holders);
            }
            if !holders.is_empty() {
                let mut pointees = BTreeSet::new();
                collect_alias_sources_expr(value, &mut pointees);
                for h in holders {
                    for &pt in &pointees {
                        if pt != h {
                            alias_edges.push((h, pt));
                        }
                    }
                }
            }
        }
        MirStmt::FmtBuf { dst_res, buf, .. } => {
            let mut pointees = BTreeSet::new();
            collect_alias_sources_expr(buf, &mut pointees);
            for pt in pointees {
                if pt != *dst_res {
                    alias_edges.push((*dst_res, pt));
                }
            }
        }
        MirStmt::If {
            then_body,
            else_body,
            ..
        } => {
            for s in then_body {
                collect_alias_edges_stmt(s, alias_edges);
            }
            for s in else_body {
                collect_alias_edges_stmt(s, alias_edges);
            }
        }
        MirStmt::While { body, .. } => {
            for s in body {
                collect_alias_edges_stmt(s, alias_edges);
            }
        }
        MirStmt::Switch { arms, default, .. } => {
            for (_, b) in arms {
                for s in b {
                    collect_alias_edges_stmt(s, alias_edges);
                }
            }
            for s in default {
                collect_alias_edges_stmt(s, alias_edges);
            }
        }
        _ => {}
    }
}

fn collect_fmt_chunk_addr_of(chunk: &MirFmtChunk, out: &mut BTreeSet<VarId>) {
    match chunk {
        MirFmtChunk::Literal(_) => {}
        MirFmtChunk::Int { value, .. }
        | MirFmtChunk::Float { value, .. }
        | MirFmtChunk::Bool(value)
        | MirFmtChunk::PaddedBool { value, .. }
        | MirFmtChunk::Str(value)
        | MirFmtChunk::PaddedStr { cstr: value, .. }
        | MirFmtChunk::Bytes(value)
        | MirFmtChunk::PaddedBytes { slice: value, .. }
        | MirFmtChunk::EnumName { value, .. }
        | MirFmtChunk::PaddedEnumName { value, .. }
        | MirFmtChunk::DebugSlice { slice: value, .. } => {
            collect_addr_of_vars(value, out);
        }
    }
}

fn collect_place_subexpr_addr_of_vars(place: &MirPlace, out: &mut BTreeSet<VarId>) {
    match place {
        MirPlace::Var(_) => {}
        MirPlace::Field { base, .. } => collect_place_subexpr_addr_of_vars(base, out),
        MirPlace::Index { base, index, .. } => {
            collect_place_subexpr_addr_of_vars(base, out);
            collect_addr_of_vars(index, out);
        }
        MirPlace::Deref(ptr_expr, _) => {
            collect_addr_of_vars(ptr_expr, out);
        }
    }
}

fn collect_addr_taken_stmt(stmt: &MirStmt, vars: &mut [MirVar]) {
    let mut taken = BTreeSet::new();
    match stmt {
        MirStmt::Assign { target, value, .. } => {
            collect_place_subexpr_addr_of_vars(target, &mut taken);
            collect_addr_of_vars(value, &mut taken);
        }
        MirStmt::Expr(value) => {
            collect_addr_of_vars(value, &mut taken);
        }
        MirStmt::MmioWrite {
            base, index, value, ..
        } => {
            collect_addr_of_vars(base, &mut taken);
            if let Some(idx) = index {
                collect_addr_of_vars(idx, &mut taken);
            }
            collect_addr_of_vars(value, &mut taken);
        }
        MirStmt::Intrinsic(intr) => match intr {
            crate::ir::MirIntrinsicOp::UseSystick { hz } => {
                collect_addr_of_vars(hz, &mut taken);
            }
            crate::ir::MirIntrinsicOp::SetPriority { .. } => {}
            crate::ir::MirIntrinsicOp::SysOp { args, .. } => {
                for a in args {
                    collect_addr_of_vars(a, &mut taken);
                }
            }
        },
        MirStmt::SpawnActivate { args, .. } => {
            for a in args {
                collect_addr_of_vars(a, &mut taken);
            }
        }
        MirStmt::PrintFmt { chunks, .. } => {
            for ch in chunks {
                collect_fmt_chunk_addr_of(ch, &mut taken);
            }
        }
        MirStmt::FmtBuf { buf, chunks, .. }
        | MirStmt::FmtRing {
            ring: buf, chunks, ..
        } => {
            collect_addr_of_vars(buf, &mut taken);
            for ch in chunks {
                collect_fmt_chunk_addr_of(ch, &mut taken);
            }
        }
        MirStmt::PrimOp(op) => match op {
            MirPrimNonBlockingOp::WatchPublish { value, .. }
            | MirPrimNonBlockingOp::GuardedSet { value, .. }
            | MirPrimNonBlockingOp::InboxPost { value, .. }
            | MirPrimNonBlockingOp::LeaseRelease { value, .. } => {
                collect_addr_of_vars(value, &mut taken);
            }
            MirPrimNonBlockingOp::LeaseNullify { target, .. } => {
                collect_place_subexpr_addr_of_vars(target, &mut taken);
            }
            MirPrimNonBlockingOp::InboxViewPost { view, value, .. } => {
                collect_addr_of_vars(view, &mut taken);
                collect_addr_of_vars(value, &mut taken);
            }
            MirPrimNonBlockingOp::ByteRingConsume { len, .. }
            | MirPrimNonBlockingOp::ByteRingCommitWrite { len, .. }
            | MirPrimNonBlockingOp::PingPongCompleteHalf { half: len, .. } => {
                collect_addr_of_vars(len, &mut taken);
            }
            MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
            | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                collect_addr_of_vars(view, &mut taken);
                collect_addr_of_vars(len, &mut taken);
            }
            MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                collect_addr_of_vars(view, &mut taken);
            }
            _ => {}
        },
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            collect_addr_of_vars(cond, &mut taken);
            for s in then_body {
                collect_addr_taken_stmt(s, vars);
            }
            for s in else_body {
                collect_addr_taken_stmt(s, vars);
            }
        }
        MirStmt::While { cond, body } => {
            collect_addr_of_vars(cond, &mut taken);
            for s in body {
                collect_addr_taken_stmt(s, vars);
            }
        }
        MirStmt::Switch {
            discr,
            arms,
            default,
        } => {
            collect_addr_of_vars(discr, &mut taken);
            for (_, body) in arms {
                for s in body {
                    collect_addr_taken_stmt(s, vars);
                }
            }
            for s in default {
                collect_addr_taken_stmt(s, vars);
            }
        }
        _ => {}
    }
    for vid in taken {
        if let Some(v) = vars.get_mut(vid.0) {
            v.addr_taken = true;
        }
    }
}

fn collect_terminator_forced_hoists(
    term: &MirTerminator,
    force_hoist: &mut BTreeSet<VarId>,
    vars: &mut [MirVar],
) {
    if let MirTerminator::Yield(yk) = term {
        match yk {
            MirYieldKind::EveryLatch {
                next_deadline_var,
                period_var,
                ..
            } => {
                force_hoist.insert(*next_deadline_var);
                force_hoist.insert(*period_var);
            }
            MirYieldKind::DelayPoll { deadline_var, .. } => {
                force_hoist.insert(*deadline_var);
            }
            MirYieldKind::Wait { arms, timer, .. } => {
                if let Some(t) = timer {
                    force_hoist.insert(t.deadline_var);
                    if let Some((lv, _)) = &t.release_lease_on_timeout {
                        force_hoist.insert(*lv);
                    }
                }
                for arm in arms {
                    match &arm.op {
                        MirWaitArmOp::InboxSend { val_var, .. }
                        | MirWaitArmOp::InboxViewSend { val_var, .. } => {
                            force_hoist.insert(*val_var);
                        }
                        MirWaitArmOp::WatchWaitChange {
                            last_gen_var,
                            seeded_var,
                            ..
                        } => {
                            force_hoist.insert(*last_gen_var);
                            if let Some(sv) = seeded_var {
                                force_hoist.insert(*sv);
                            }
                        }
                        _ => {}
                    }
                }
            }
            MirYieldKind::CallYielding { args, .. } => {
                // Any variable whose address `&v` or slice/pointer alias (`s = fmt_buf(buf, ...)`)
                // is passed into a yielding function MUST live in the static frame so the child
                // coroutine never holds a dangling C stack pointer across yields!
                let mut taken = BTreeSet::new();
                for arg in args {
                    collect_addr_of_vars(arg, &mut taken);
                    collect_alias_sources_expr(arg, force_hoist);
                }
                for vid in taken {
                    if let Some(v) = vars.get_mut(vid.0) {
                        v.addr_taken = true;
                    }
                    force_hoist.insert(vid);
                }
            }
            _ => {}
        }
    }
}

/// Compute active-interval interference across yielding child call sites and color
/// non-overlapping calls into shared `union` groups.
///
/// In a sequential coroutine without `select`, each `CallYielding(slot_i)` transitions
/// immediately to its dedicated `PollChild(slot_i)` block and exits to `resume_bb` upon
/// completion, so distinct call sites have strictly disjoint active block sets (`{poll_bb_i}`)
/// and all pack into `union_group = 0`! We compute the interference graph explicitly over
/// active `PollChild` blocks so any hypothetical overlapping extension is also handled soundly.
fn assign_child_union_groups(blocks: &[MirBlock], child_slots: &mut [MirChildCallSlot]) {
    if child_slots.is_empty() {
        return;
    }

    let mut active_blocks_per_slot: BTreeMap<String, BTreeSet<BlockId>> = BTreeMap::new();
    for blk in blocks {
        if let MirTerminator::Yield(MirYieldKind::PollChild { slot_name, .. }) = &blk.terminator {
            active_blocks_per_slot
                .entry(slot_name.clone())
                .or_default()
                .insert(blk.id);
        }
    }

    let k = child_slots.len();
    let mut colors: Vec<Option<usize>> = vec![None; k];

    for i in 0..k {
        let mut forbidden = BTreeSet::new();
        let set_i = active_blocks_per_slot
            .get(&child_slots[i].slot_name)
            .cloned()
            .unwrap_or_default();
        for j in 0..i {
            let set_j = active_blocks_per_slot
                .get(&child_slots[j].slot_name)
                .cloned()
                .unwrap_or_default();
            if !set_i.is_disjoint(&set_j) {
                if let Some(c) = colors[j] {
                    forbidden.insert(c);
                }
            }
        }
        let mut color = 0;
        while forbidden.contains(&color) {
            color += 1;
        }
        colors[i] = Some(color);
        child_slots[i].union_group = color;
    }
}
