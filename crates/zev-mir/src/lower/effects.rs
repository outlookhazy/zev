use super::*;

pub(super) fn discover_systick_config(
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
                    if id.name == SYS_NAMESPACE {
                        match SysMethod::from_name(&method.name) {
                                Some(SysMethod::UseSystick) => {
                                    if let Some(first) = args.first() {
                                        if let ExprKind::Literal(Literal::Int { value: hz, .. }) =
                                            &first.value.kind
                                        {
                                            syms.systick_hz = Some(*hz);
                                        }
                                    }
                                    if let Some(second) = args.get(1) {
                                        if let ExprKind::Literal(Literal::Duration(d)) =
                                            &second.value.kind
                                        {
                                            let us = d.as_micros();
                                            if us > 0 {
                                                syms.tick_period_us = us;
                                            }
                                        }
                                    }
                                }
                                Some(SysMethod::SetTickUs) => {
                                    if let Some(first) = args.first() {
                                        if let ExprKind::Literal(Literal::Int { value: us, .. }) =
                                            &first.value.kind
                                        {
                                            if *us > 0 {
                                                syms.tick_period_us = *us;
                                            }
                                        }
                                    }
                                }
                            Some(SysMethod::SetTick | SysMethod::SetTickPeriod) => {
                                if let Some(first) = args.first() {
                                    if let ExprKind::Literal(Literal::Duration(d)) = &first.value.kind
                                    {
                                        let us = d.as_micros();
                                        if us > 0 {
                                            syms.tick_period_us = us;
                                        }
                                    }
                                }
                            }
                            _ => {}
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
pub(super) struct RawCallEdge {
    pub(super) caller: String,
    pub(super) callee: String,
    pub(super) in_noyield: bool,
    pub(super) in_restricted: bool, // critical, defer, or isr
}

pub(super) fn infer_effects_and_specializations(
    fn_decls: &[&FnDecl],
    task_decls: &[&TaskDecl],
    main_decl: Option<&FnDecl>,
    syms: &mut ModuleSymbols,
) {
    let mut direct_eff: BTreeMap<String, MirFnEffect> = BTreeMap::new();
    let mut direct_spawns: BTreeSet<String> = BTreeSet::new();
    let mut boost_capable: BTreeSet<String> = BTreeSet::new();
    let mut edges: Vec<RawCallEdge> = Vec::new();
    let mut fn_names: BTreeSet<String> = BTreeSet::new();

    for fd in fn_decls {
        let fname = fd.name.name.clone();
        fn_names.insert(fname.clone());
        let manual = has_attr_manual_yield(&fd.attrs);
        let mut eff = MirFnEffect::NonYielding;
        let mut spawns = false;
        if let Some(body) = &fd.body {
            if ast_stmts_have_boost_acquire(&body.stmts) {
                boost_capable.insert(fname.clone());
            }
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
            if boost_capable.contains(&edge.callee) && boost_capable.insert(edge.caller.clone()) {
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
    syms.boost_capable_fns = boost_capable;
}

pub(super) fn ast_block_has_break(block: &Block) -> bool {
    ast_stmts_have_break(&block.stmts)
}

pub(super) fn ast_stmts_have_break(stmts: &[Stmt]) -> bool {
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
            StmtKind::Wait { arms, timer } => {
                if arms.iter().any(|a| ast_block_has_break(&a.body))
                    || timer.as_ref().is_some_and(|t| ast_block_has_break(&t.body))
                {
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

pub(super) fn ast_expr_has_break(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Catch { expr: inner, clause } => {
            ast_expr_has_break(inner) || ast_block_has_break(&clause.body)
        }
        ExprKind::Or { expr: a, fallback: b } => ast_expr_has_break(a) || ast_expr_has_break(b),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scan_stmts_effect(
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
                } else if !is_while_true
                    && !in_noyield
                    && !in_restricted
                    && !stmt_manual
                    && caller != "main"
                {
                    *eff = (*eff).max(MirFnEffect::SoftYielding);
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
            StmtKind::Wait { arms, timer } => {
                if !in_noyield && !in_restricted {
                    *eff = (*eff).max(MirFnEffect::HardBlocking);
                }
                for arm in arms {
                    scan_expr_effect(caller, &arm.call, in_noyield, in_restricted, eff, spawns, edges);
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
                if let Some(t) = timer {
                    scan_expr_effect(caller, &t.expr, in_noyield, in_restricted, eff, spawns, edges);
                    scan_stmts_effect(
                        caller,
                        &t.body.stmts,
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

pub(super) fn scan_expr_effect(
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
                "wait_chunk"
                    | "wait_read"
                    | "recv"
                    | "wait"
                    | "send"
                    | "delay"
                    | "wait_change"
                    | "alloc"
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

pub(super) fn stmt_slice_can_yield(stmts: &[Stmt], manual_yield: bool, syms: &ModuleSymbols) -> bool {
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

pub(super) fn stmt_contains_boot_spawn(stmt: &Stmt, syms: &ModuleSymbols) -> bool {
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
        StmtKind::Wait { arms, timer } => {
            arms.iter().any(|a| {
                a.body
                    .stmts
                    .iter()
                    .any(|s| stmt_contains_boot_spawn(s, syms))
            }) || timer.as_ref().is_some_and(|t| {
                t.body
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

pub(super) fn expr_contains_boot_spawn(expr: &Expr, syms: &ModuleSymbols) -> bool {
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

pub(super) fn detect_static_set_priority(stmts: &[Stmt]) -> Option<MirPriority> {
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

pub(super) struct BootIrqBindingRec {
    pub(super) branch_path: Vec<(usize, usize)>,
    pub(super) binding: MirIrqBinding,
}

pub(super) struct BootIrqVectorAcc {
    pub(super) irq_name: String,
    pub(super) irq_num: u32,
    pub(super) is_shared: bool,
    pub(super) selector_symbol: Option<String>,
    pub(super) next_selector_val: u8,
    pub(super) branch_selectors: Vec<(Vec<(usize, usize)>, u8)>,
    pub(super) records: Vec<BootIrqBindingRec>,
}

pub(super) struct BootSpawnState<'a> {
    pub(super) fn_decls: &'a BTreeMap<String, &'a FnDecl>,
    pub(super) mir_isrs: &'a [MirIsr],
    pub(super) spawned_tasks: &'a mut Vec<MirSpawnInstance>,
    pub(super) irq_vectors: &'a mut BTreeMap<u32, BootIrqVectorAcc>,
    pub(super) bp_instance_counts: BTreeMap<String, usize>,
    pub(super) provenance_stack: Vec<String>,
    pub(super) branch_depth: usize,
    pub(super) branch_stack: Vec<(usize, usize)>,
    pub(super) next_branch_id: usize,
    pub(super) loop_depth: usize,
    pub(super) next_binding_id: usize,
    pub(super) irq_param_bindings: BTreeMap<String, (String, u32)>,
}

pub(super) fn is_const_mir_place(place: &MirPlace) -> bool {
    match place {
        MirPlace::Deref(inner, _) => is_const_mir_expr(inner),
        MirPlace::Field { base, .. } => is_const_mir_place(base),
        _ => false,
    }
}

pub(super) fn is_const_mir_expr(expr: &MirExpr) -> bool {
    match &expr.kind {
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::StringLit(_)
        | MirExprKind::TicksLit(_)
        | MirExprKind::EnumVariant { .. }
        | MirExprKind::GlobalPrimAddr(_)
        | MirExprKind::SubsystemAddr(_)
        | MirExprKind::FnAddr(_)
        | MirExprKind::MmioAddr(_)
        | MirExprKind::ConstRef(_) => true,
        MirExprKind::MakePrimView { ptr, .. } => is_const_mir_expr(ptr),
        MirExprKind::AddrOfPlace(place) => is_const_mir_place(place),
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

pub(super) fn resolve_boot_const_place(place: &MirPlace, env: &BTreeMap<VarId, MirExpr>) -> MirPlace {
    match place {
        MirPlace::Var(_) => place.clone(),
        MirPlace::Deref(inner, ty) => {
            MirPlace::Deref(Box::new(resolve_boot_const_expr(inner, env)), ty.clone())
        }
        MirPlace::Field {
            base,
            struct_name,
            packed,
            field,
            field_ty,
        } => MirPlace::Field {
            base: Box::new(resolve_boot_const_place(base, env)),
            struct_name: struct_name.clone(),
            packed: *packed,
            field: field.clone(),
            field_ty: field_ty.clone(),
        },
        MirPlace::Index {
            base,
            index,
            elem_ty,
            is_slice,
        } => MirPlace::Index {
            base: Box::new(resolve_boot_const_place(base, env)),
            index: Box::new(resolve_boot_const_expr(index, env)),
            elem_ty: elem_ty.clone(),
            is_slice: *is_slice,
        },
    }
}

pub(super) fn resolve_boot_const_expr(expr: &MirExpr, env: &BTreeMap<VarId, MirExpr>) -> MirExpr {
    match &expr.kind {
        MirExprKind::Var(vid) => env.get(vid).cloned().unwrap_or_else(|| expr.clone()),
        MirExprKind::MakePrimView { ptr, cap, kind } => MirExpr::new(
            MirExprKind::MakePrimView {
                ptr: Box::new(resolve_boot_const_expr(ptr, env)),
                cap: *cap,
                kind: kind.clone(),
            },
            expr.ty.clone(),
        ),
        MirExprKind::AddrOfPlace(place) => MirExpr::new(
            MirExprKind::AddrOfPlace(Box::new(resolve_boot_const_place(place, env))),
            expr.ty.clone(),
        ),
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

pub(super) fn zero_init_expr(ty: &MirType, syms: &ModuleSymbols) -> MirExpr {
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

pub(super) fn place_root_var(place: &MirPlace) -> Option<VarId> {
    match place {
        MirPlace::Var(vid) => Some(*vid),
        MirPlace::Field { base, .. } | MirPlace::Index { base, .. } => place_root_var(base),
        MirPlace::Deref(_, _) => None,
    }
}

pub(super) fn paths_mutually_exclusive(a: &[(usize, usize)], b: &[(usize, usize)]) -> bool {
    a.iter()
        .zip(b.iter())
        .any(|((ba, aa), (bb, ab))| ba == bb && aa != ab)
}

impl<'a> BootSpawnState<'a> {
    pub(super) fn attach_irq(
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
                let sel_sym = acc
                    .selector_symbol
                    .get_or_insert_with(|| format!("__zev_irq_sel_{irq_num}"))
                    .clone();
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

pub(super) fn lower_main_entry<'a>(
    md: &FnDecl,
    syms: &'a ModuleSymbols,
    fn_decls: &'a BTreeMap<String, &'a FnDecl>,
    mir_isrs: &'a [MirIsr],
    blueprints: &mut Vec<MirTaskBlueprint>,
    spawned_tasks: &mut Vec<MirSpawnInstance>,
    mir_irq_vectors: &mut Vec<MirIrqVector>,
) -> Result<MirBootSequence, MirLowerError> {
    let body = md.body.as_ref().ok_or_else(|| MirLowerError {
        message: "entry function `main` has no body".to_string(),
        span: md.span,
    })?;
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
                    let fn_sig = syms
                        .fns
                        .get(&cname)
                        .cloned()
                        .ok_or_else(|| MirLowerError {
                            message: format!("internal error: missing fn signature '{cname}'"),
                            span: stmt.span,
                        })?;
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
                    let fn_sig = syms
                        .fns
                        .get(&cname)
                        .cloned()
                        .ok_or_else(|| MirLowerError {
                            message: format!("internal error: missing fn signature '{cname}'"),
                            span: stmt.span,
                        })?;
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
        if slot_id >= 32 {
            return Err(MirLowerError {
                message: "program exceeds the 32-task static scheduler limit (`MAX_TASKS = 32` including `main` supervisor)"
                    .to_string(),
                span: md.span,
            });
        }
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

pub(super) fn is_yielding_fn_callee(callee: &Expr, syms: &ModuleSymbols) -> bool {
    if let ExprKind::Ident(id) = &callee.kind {
        syms.fns
            .get(&id.name)
            .map_or(false, |f| f.is_yielding && !f.spawns_in_boot)
    } else {
        false
    }
}

pub(super) fn is_pure_boot_setup_stmt(stmt: &Stmt) -> bool {
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
                id.name == SYS_NAMESPACE
            } else {
                false
            }
        }
        _ => false,
    }
}

pub(super) fn collect_ast_idents_stmts(stmts: &[Stmt], out: &mut BTreeSet<String>) {
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
            StmtKind::Wait { arms, timer } => {
                for arm in arms {
                    collect_ast_idents_expr(&arm.call, out);
                    collect_ast_idents_stmts(&arm.body.stmts, out);
                }
                if let Some(t) = timer {
                    collect_ast_idents_expr(&t.expr, out);
                    collect_ast_idents_stmts(&t.body.stmts, out);
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

pub(super) fn collect_ast_idents_expr(expr: &Expr, out: &mut BTreeSet<String>) {
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

