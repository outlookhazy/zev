//! Exact backward CFG liveness analysis across coroutine yield points,
//! automatic variable hoisting (static frame vs transient stack), and
//! disjoint child coroutine frame union-aliasing.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{
    BlockId, MirAssignOp, MirBlock, MirCallDst, MirChildCallSlot, MirExpr, MirExprKind, MirFnBody,
    MirFunction, MirPlace, MirPrimNonBlockingOp, MirPrimReadOp, MirPrimTarget, MirStmt,
    MirTaskBlueprint, MirTerminator, MirVar, MirYieldKind, VarId, VarRole,
};

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
        MirFnBody::Direct(_) => {
            for v in &mut func.vars {
                v.hoisted = false;
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

fn compute_hoisting(
    param_set: &BTreeSet<VarId>,
    vars: &mut [MirVar],
    blocks: &[MirBlock],
) {
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

    // If a pointer/struct variable `p` is hoisted and was assigned `&x`, ensure `x` is hoisted too
    // so `_f->p` never holds a dangling pointer to a C stack variable!
    let mut alias_edges: Vec<(VarId, VarId)> = Vec::new(); // (holder, pointee)
    for blk in blocks {
        for stmt in &blk.stmts {
            if let MirStmt::Assign { target, value, .. } = stmt {
                let holder = root_place_var(target);
                let mut pointees = BTreeSet::new();
                collect_addr_of_vars(value, &mut pointees);
                if let Some(h) = holder {
                    for pt in pointees {
                        alias_edges.push((h, pt));
                    }
                }
            }
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
            VarRole::Param => true,
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
            MirYieldKind::InboxRecv {
                target,
                timeout_deadline_var,
                dst_val,
                dst_res,
                resume_bb,
                ..
            } => {
                let mut out = block_live(*resume_bb);
                // `dst_val` / `dst_res` are written ONLY when the receive completes before `resume_bb`!
                if let Some(dv) = dst_val {
                    out.remove(dv);
                }
                if let Some(dr) = dst_res {
                    out.remove(dr);
                }
                collect_prim_target_uses(target, &mut out);
                if let Some(tv) = timeout_deadline_var {
                    out.insert(*tv);
                }
                out
            }
            MirYieldKind::InboxSend {
                target,
                val_var,
                timeout_deadline_var,
                dst_res,
                resume_bb,
                ..
            } => {
                let mut out = block_live(*resume_bb);
                if let Some(dr) = dst_res {
                    out.remove(dr);
                }
                collect_prim_target_uses(target, &mut out);
                out.insert(*val_var);
                if let Some(tv) = timeout_deadline_var {
                    out.insert(*tv);
                }
                out
            }
            MirYieldKind::SignalWait {
                target,
                timeout_deadline_var,
                dst_res,
                resume_bb,
                ..
            } => {
                let mut out = block_live(*resume_bb);
                if let Some(dr) = dst_res {
                    out.remove(dr);
                }
                collect_prim_target_uses(target, &mut out);
                if let Some(tv) = timeout_deadline_var {
                    out.insert(*tv);
                }
                out
            }
            MirYieldKind::PingPongWaitChunk {
                target,
                dst_slice,
                resume_bb,
                ..
            }
            | MirYieldKind::LeaseWaitChunk {
                target,
                dst_slice,
                resume_bb,
                ..
            } => {
                let mut out = block_live(*resume_bb);
                out.remove(dst_slice);
                collect_prim_target_uses(target, &mut out);
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
        MirStmt::PrimOp(op) => collect_prim_op_uses(op, live),
        MirStmt::Intrinsic(intr) => match intr {
            crate::ir::MirIntrinsicOp::UseSystick { hz } => collect_expr_uses(hz, live),
            crate::ir::MirIntrinsicOp::SetPriority { .. } => {}
            crate::ir::MirIntrinsicOp::SysOp { args, .. }
            | crate::ir::MirIntrinsicOp::HwOp { args, .. } => {
                for a in args {
                    collect_expr_uses(a, live);
                }
            }
        },
        MirStmt::EnterCritical
        | MirStmt::ExitCritical
        | MirStmt::EndInboxTurn
        | MirStmt::Break
        | MirStmt::Continue => {}
        MirStmt::SpawnActivate { args, .. } => {
            for a in args {
                collect_expr_uses(a, live);
            }
        }
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
        | MirPrimNonBlockingOp::InboxPost { target, value, .. } => {
            collect_prim_target_uses(target, uses);
            collect_expr_uses(value, uses);
        }
        MirPrimNonBlockingOp::SignalNotify { target, .. }
        | MirPrimNonBlockingOp::SignalClear { target }
        | MirPrimNonBlockingOp::PingPongFlip { target, .. }
        | MirPrimNonBlockingOp::PingPongSetActive { target, .. }
        | MirPrimNonBlockingOp::LeaseComplete { target, .. }
        | MirPrimNonBlockingOp::LeaseSetActive { target, .. } => {
            collect_prim_target_uses(target, uses);
        }
    }
}

fn collect_expr_uses(expr: &MirExpr, uses: &mut BTreeSet<VarId>) {
    match &expr.kind {
        MirExprKind::IntLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::StringLit(_)
        | MirExprKind::TicksLit(_)
        | MirExprKind::EnumVariant { .. }
        | MirExprKind::ConstRef(_)
        | MirExprKind::GlobalPrimAddr(_)
        | MirExprKind::HwPeriphVal(_)
        | MirExprKind::FnAddr(_)
        | MirExprKind::ArrayLen(_)
        | MirExprKind::NowTicks => {}
        MirExprKind::Var(vid)
        | MirExprKind::ResultIsErr(vid)
        | MirExprKind::ResultErr(vid)
        | MirExprKind::ResultVal(vid) => {
            uses.insert(*vid);
        }
        MirExprKind::Unary { expr: inner, .. }
        | MirExprKind::Cast { expr: inner, .. }
        | MirExprKind::SliceLen(inner) => {
            collect_expr_uses(inner, uses);
        }
        MirExprKind::AddrOfPlace(place) | MirExprKind::ArrayToSlice { array_place: place, .. } => {
            collect_place_uses_for_partial_write(place, uses);
        }
        MirExprKind::Binary { lhs, rhs, .. } | MirExprKind::Index { base: lhs, index: rhs, .. } => {
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
            MirPrimReadOp::WatchRead { target, .. }
            | MirPrimReadOp::InboxTryRecv { target, .. }
            | MirPrimReadOp::InboxLen { target, .. }
            | MirPrimReadOp::InboxIsEmpty { target, .. }
            | MirPrimReadOp::InboxIsFull { target, .. }
            | MirPrimReadOp::SignalIsSet { target }
            | MirPrimReadOp::SignalTryWait { target } => {
                collect_prim_target_uses(target, uses);
            }
        },
        MirExprKind::CallDirect { args, .. } => {
            for a in args {
                collect_expr_uses(a, uses);
            }
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

fn collect_addr_of_vars(expr: &MirExpr, out: &mut BTreeSet<VarId>) {
    match &expr.kind {
        MirExprKind::AddrOfPlace(p) | MirExprKind::ArrayToSlice { array_place: p, .. } => {
            if let Some(vid) = root_place_var(p) {
                out.insert(vid);
            }
        }
        MirExprKind::Unary { expr: inner, .. }
        | MirExprKind::Cast { expr: inner, .. }
        | MirExprKind::FieldAccess { base: inner, .. }
        | MirExprKind::SliceLen(inner) => collect_addr_of_vars(inner, out),
        MirExprKind::Binary { lhs, rhs, .. } | MirExprKind::Index { base: lhs, index: rhs, .. } => {
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
        MirExprKind::PrimRead(ro) => {
            if let MirPrimReadOp::InboxTrySend { value, .. } = ro.as_ref() {
                collect_addr_of_vars(value, out);
            }
        }
        _ => {}
    }
}

fn collect_addr_taken_stmt(stmt: &MirStmt, vars: &mut [MirVar]) {
    let mut taken = BTreeSet::new();
    match stmt {
        MirStmt::Assign { value, .. } | MirStmt::Expr(value) => {
            collect_addr_of_vars(value, &mut taken);
        }
        MirStmt::SpawnActivate { args, .. } => {
            for a in args {
                collect_addr_of_vars(a, &mut taken);
            }
        }
        MirStmt::PrimOp(op) => match op {
            MirPrimNonBlockingOp::WatchPublish { value, .. }
            | MirPrimNonBlockingOp::InboxPost { value, .. } => {
                collect_addr_of_vars(value, &mut taken);
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
            MirYieldKind::InboxRecv {
                timeout_deadline_var: Some(tv),
                ..
            }
            | MirYieldKind::SignalWait {
                timeout_deadline_var: Some(tv),
                ..
            } => {
                force_hoist.insert(*tv);
            }
            MirYieldKind::InboxSend {
                val_var,
                timeout_deadline_var,
                ..
            } => {
                force_hoist.insert(*val_var);
                if let Some(tv) = timeout_deadline_var {
                    force_hoist.insert(*tv);
                }
            }
            MirYieldKind::CallYielding { args, .. } => {
                // Any variable whose address `&v` is passed into a yielding function MUST
                // live in the static frame so the child coroutine never holds a stack pointer!
                let mut taken = BTreeSet::new();
                for arg in args {
                    collect_addr_of_vars(arg, &mut taken);
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
