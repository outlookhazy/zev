use super::*;

pub(super) struct InlinedHelperFrame {
    pub(super) ret_ty: MirType,
    pub(super) ret_val_var: Option<VarId>,
    pub(super) ret_flag_var: Option<VarId>,
    pub(super) entry_defer_depth: usize,
}

pub(super) struct DirectLowerCtx<'a> {
    pub(super) syms: &'a ModuleSymbols,
    pub(super) ret_ty: MirType,
    pub(super) error_spec: MirErrorSpec,
    pub(super) vars: Vec<MirVar>,
    pub(super) scopes: VarScopes,
    pub(super) defer_stack: Vec<Vec<Stmt>>,
    pub(super) loop_defer_depths: Vec<usize>,
    pub(super) boot_state: Option<BootSpawnState<'a>>,
    pub(super) boot_const_env: BTreeMap<VarId, MirExpr>,
    pub(super) inlined_helper_stack: Vec<InlinedHelperFrame>,
    pub(super) in_noyield: bool,
    pub(super) in_isr: bool,
    pub(super) in_const_eval: bool,
    pub(super) isr_state_vars: Vec<MirIsrStateDecl>,
    pub(super) pending_stmts: Vec<MirStmt>,
    /// Shared sink-print scratch: (buffer var, its capacity, `fmt_buf` result var).
    pub(super) sink_fmt_scratch: Option<(VarId, usize, VarId)>,
}

impl<'a> DirectLowerCtx<'a> {
    pub(super) fn new(syms: &'a ModuleSymbols, ret_ty: MirType, error_spec: MirErrorSpec) -> Self {
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
            in_const_eval: false,
            isr_state_vars: Vec::new(),
            pending_stmts: Vec::new(),
            sink_fmt_scratch: None,
        }
    }

    pub(super) fn acquire_sink_fmt_scratch(&mut self, cap: usize) -> (VarId, VarId) {
        let cap = cap.max(1);
        if let Some((buf, cur_cap, res)) = self.sink_fmt_scratch {
            if cap > cur_cap {
                self.vars[buf.0].ty = MirType::Array {
                    elem: Box::new(MirType::U8),
                    len: cap,
                };
                self.sink_fmt_scratch = Some((buf, cap, res));
            }
            return (buf, res);
        }
        let buf = self.alloc_temp(
            "sink_fmt_buf",
            MirType::Array {
                elem: Box::new(MirType::U8),
                len: cap,
            },
        );
        let res = self.alloc_temp(
            "sink_fmt_res",
            MirType::Result {
                ok: Box::new(MirType::Slice(Box::new(MirType::U8))),
                err: Some(Box::new(MirType::Enum("FmtErr".to_string()))),
            },
        );
        self.sink_fmt_scratch = Some((buf, cap, res));
        (buf, res)
    }

    pub(super) fn new_with_boot(
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
            in_const_eval: false,
            isr_state_vars: Vec::new(),
            pending_stmts: Vec::new(),
            sink_fmt_scratch: None,
        }
    }

    pub(super) fn declare_var(&mut self, name: &str, ty: MirType, role: VarRole) -> VarId {
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

    pub(super) fn alloc_temp(&mut self, prefix: &str, ty: MirType) -> VarId {
        let name = format!("_tmp_{}_{}", prefix, self.vars.len());
        self.declare_var(&name, ty, VarRole::CompilerTemp)
    }

    pub(super) fn has_active_defers_from(&self, min_depth: usize) -> bool {
        self.defer_stack
            .iter()
            .skip(min_depth)
            .any(|frame| !frame.is_empty())
    }

    pub(super) fn lower_single_deferred_stmt(
        &mut self,
        deferred: &Stmt,
    ) -> Result<Vec<MirStmt>, MirLowerError> {
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

    pub(super) fn pop_and_lower_top_defer_scope(&mut self) -> Result<Vec<MirStmt>, MirLowerError> {
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
    pub(super) fn emit_defers_down_to(
        &mut self,
        min_depth: usize,
    ) -> Result<Vec<MirStmt>, MirLowerError> {
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

    pub(super) fn lower_top_level_stmts(
        &mut self,
        stmts: &[Stmt],
    ) -> Result<Vec<MirStmt>, MirLowerError> {
        let mut out = self.lower_stmt_seq(stmts)?;
        out.extend(self.pop_and_lower_top_defer_scope()?);
        Ok(out)
    }

    pub(super) fn lower_block_stmts(
        &mut self,
        stmts: &[Stmt],
    ) -> Result<Vec<MirStmt>, MirLowerError> {
        self.scopes.push();
        self.defer_stack.push(Vec::new());
        let mut out = self.lower_stmt_seq(stmts)?;
        out.extend(self.pop_and_lower_top_defer_scope()?);
        self.scopes.pop();
        Ok(out)
    }

    pub(super) fn lower_stmt_seq(&mut self, stmts: &[Stmt]) -> Result<Vec<MirStmt>, MirLowerError> {
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

    pub(super) fn lower_stmt(&mut self, stmt: &Stmt) -> Result<Vec<MirStmt>, MirLowerError> {
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
                    let mut be = DirectBackend {
                        ctx: self,
                        stmts: &mut out,
                    };
                    let coerced =
                        consume_moved_lease_be(coerce_expr_be(val, &vty, &mut be), &mut be);
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
                } else if !self.loop_defer_depths.is_empty() {
                    // Function entry zeroes every local once. A declaration inside a loop body runs
                    // again on each iteration and must start from zero each time.
                    out.push(MirStmt::Assign {
                        target: MirPlace::Var(vid),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::ZeroValue, vty),
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
                    let mut be = DirectBackend {
                        ctx: self,
                        stmts: &mut out,
                    };
                    let rhs_moved = consume_moved_lease_be(
                        coerce_expr_be(rhs, &pty.host_value_type(), &mut be),
                        &mut be,
                    );
                    out.push(MirStmt::Assign {
                        target: place,
                        op: convert_assign_op(*op),
                        value: rhs_moved,
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
                        let mut be = DirectBackend {
                            ctx: self,
                            stmts: &mut out,
                        };
                        let v_moved =
                            consume_moved_lease_be(coerce_expr_be(v, &h_ret_ty, &mut be), &mut be);
                        if let Some(rv) = h_ret_val {
                            out.push(MirStmt::Assign {
                                target: MirPlace::Var(rv),
                                op: MirAssignOp::Assign,
                                value: v_moved,
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
                    let mut be = DirectBackend {
                        ctx: self,
                        stmts: &mut out,
                    };
                    let coerced =
                        consume_moved_lease_be(coerce_expr_be(raw, &ret_ty, &mut be), &mut be);
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
                    let coerced = if let Some(ety) = &err_ty {
                        let mut be = DirectBackend {
                            ctx: self,
                            stmts: &mut out,
                        };
                        coerce_expr_be(raw, ety, &mut be)
                    } else {
                        raw
                    };
                    if let (true, Some(ety)) = (has_defers, err_ty.clone()) {
                        let tmp = self.alloc_temp("err_reason", ety.clone());
                        out.push(MirStmt::Assign {
                            target: MirPlace::Var(tmp),
                            op: MirAssignOp::Assign,
                            value: coerced,
                        });
                        Some(MirExpr::new(MirExprKind::Var(tmp), ety))
                    } else {
                        Some(coerced)
                    }
                } else {
                    None
                };
                let p = if let Some(pe) = partial {
                    let raw = self.lower_expr_with_stmts(pe, Some(&ret_ty), &mut out)?;
                    let mut be = DirectBackend {
                        ctx: self,
                        stmts: &mut out,
                    };
                    let coerced =
                        consume_moved_lease_be(coerce_expr_be(raw, &ret_ty, &mut be), &mut be);
                    if has_defers && !ret_ty.is_void() {
                        let tmp = self.alloc_temp("err_partial", ret_ty.clone());
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
                                    lhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(ivar),
                                        vty.clone(),
                                    )),
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
                                    lhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(ivar),
                                        vty.clone(),
                                    )),
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
                while_body.push(MirStmt::BoundedForLatch);

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
                let (slice_tmp, seq_elem_ty) =
                    self.lower_iter_into_slice_temp(iter, &ety, &mut out)?;
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
                        raw_elem_ty: seq_elem_ty.clone(),
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
                while_body.push(MirStmt::BoundedForLatch);

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
            StmtKind::Wait { .. } => {
                return Err(MirLowerError {
                    message: "`wait { ... }` encountered inside non-yielding context".to_string(),
                    span: stmt.span,
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
                    let code_expr =
                        self.lower_sys_exit_code_before_defers(code_arg_opt, &mut out)?;
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
                    if !me.ty.is_void()
                        || matches!(
                            me.kind,
                            MirExprKind::CallDirect { .. } | MirExprKind::RomCallU32 { .. }
                        )
                    {
                        out.push(MirStmt::Expr(me));
                    }
                }
            }
            StmtKind::LoopEvery { .. } | StmtKind::Restart => {
                return Err(MirLowerError {
                    message:
                        "yielding `loop every` or `restart` encountered inside non-yielding context"
                            .to_string(),
                    span: stmt.span,
                });
            }
        }
        Ok(out)
    }

    pub(super) fn lower_sys_exit_code_before_defers(
        &mut self,
        code_arg_opt: Option<&Expr>,
        out: &mut Vec<MirStmt>,
    ) -> Result<MirExpr, MirLowerError> {
        if let Some(ce) = code_arg_opt {
            let raw = self.lower_expr_with_stmts(ce, Some(&MirType::I32), out)?;
            let coerced = coerce_expr_if_needed(raw, &MirType::I32);
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

    pub(super) fn lower_iter_into_slice_temp(
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

    pub(super) fn lower_boot_spawn_stmt(
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
                let mut be = DirectBackend {
                    ctx: self,
                    stmts: out,
                };
                coerce_expr_be(val, ety, &mut be)
            } else {
                val
            };
            lowered_args.push(coerced);
        }

        let bs = self.boot_state.as_mut().ok_or_else(|| MirLowerError {
            message: "`spawn` encountered outside sequential `main()` boot context".to_string(),
            span: sp.span,
        })?;
        let count_entry = bs.bp_instance_counts.entry(bp_name.clone()).or_insert(0);
        let inst_num = *count_entry;
        *count_entry += 1;

        let slot_id = bs.spawned_tasks.len();
        if slot_id >= 32 {
            return Err(MirLowerError {
                message: "program exceeds the 32-task static scheduler limit (`MAX_TASKS = 32`)"
                    .to_string(),
                span: sp.span,
            });
        }
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

    pub(super) fn try_unroll_boot_helper_call(
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
                    lowered_args.push(MirExpr::new(
                        MirExprKind::IntLit(inum as i128),
                        MirType::U32,
                    ));
                    continue;
                }
            }
            let expected = sig.params.get(idx);
            let me = self.lower_expr_with_stmts(&a.value, expected, out)?;
            let mut be = DirectBackend {
                ctx: self,
                stmts: out,
            };
            let coerced = if let Some(pty) = expected {
                coerce_expr_be(me, pty, &mut be)
            } else {
                me
            };
            lowered_args.push(consume_moved_lease_be(coerced, &mut be));
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

    pub(super) fn lower_pure_expr(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError> {
        let prev = self.in_const_eval;
        self.in_const_eval = true;
        let mut sink = Vec::new();
        let res = self.lower_expr_with_stmts(expr, expected, &mut sink);
        self.in_const_eval = prev;
        res
    }

    pub(super) fn lower_expr_with_stmts(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<MirExpr, MirLowerError> {
        match &expr.kind {
            ExprKind::Or { .. } => {
                let mut or_chain = Vec::new();
                flatten_or_operands(expr, &mut or_chain);
                let tail = or_chain[or_chain.len() - 1];
                let first_res = self.lower_failable_expr(or_chain[0], stmts)?;
                let (src_ok_ty, first_res_ty) = extract_result_types(&first_res.ty);
                let ok_ty = resolve_or_result_ty(
                    src_ok_ty,
                    expected,
                    tail,
                    self.syms,
                    &self.scopes,
                    &self.vars,
                );
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
                    if let Some(unrolled) =
                        self.try_unroll_boot_helper_call(&id.name, args, stmts)?
                    {
                        return Ok(unrolled);
                    }
                }
                let _ = &self.pending_stmts;
                let mut be = DirectBackend { ctx: self, stmts };
                lower_common_expr(expr, expected, &mut be)
            }
            _ => {
                let _ = &self.pending_stmts;
                let mut be = DirectBackend { ctx: self, stmts };
                lower_common_expr(expr, expected, &mut be)
            }
        }
    }

    pub(super) fn lower_or_tail_direct(
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
                let mut fb_be = DirectBackend {
                    ctx: self,
                    stmts: &mut fb_stmts,
                };
                let fb_val =
                    consume_moved_lease_be(coerce_expr_be(fb_val, ok_ty, &mut fb_be), &mut fb_be);
                fb_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(out_vid),
                    op: MirAssignOp::Assign,
                    value: fb_val,
                });
            } else if !fb_val.ty.is_void()
                || matches!(
                    fb_val.kind,
                    MirExprKind::CallDirect { .. } | MirExprKind::RomCallU32 { .. }
                )
            {
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

        let mut ok_stmts = Vec::new();
        if let Some(out_vid) = out_vid_opt {
            let prev_ok_ty = extract_result_types(&self.vars[prev_res_vid.0].ty).0;
            let mut ok_be = DirectBackend {
                ctx: self,
                stmts: &mut ok_stmts,
            };
            let val = coerce_expr_be(
                MirExpr::new(MirExprKind::ResultVal(prev_res_vid), prev_ok_ty),
                ok_ty,
                &mut ok_be,
            );
            ok_stmts.push(MirStmt::Assign {
                target: MirPlace::Var(out_vid),
                op: MirAssignOp::Assign,
                value: val,
            });
        }

        stmts.push(MirStmt::If {
            cond: MirExpr::new(MirExprKind::ResultIsErr(prev_res_vid), MirType::Bool),
            then_body: fb_stmts,
            else_body: ok_stmts,
        });
        Ok(())
    }

    pub(super) fn lower_catch_or_chain_direct(
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
            let mut ok_stmts = Vec::new();
            if let Some(out_vid) = out_vid_opt {
                let mut ok_be = DirectBackend {
                    ctx: self,
                    stmts: &mut ok_stmts,
                };
                let val = coerce_expr_be(
                    MirExpr::new(MirExprKind::ResultVal(head_vid), ok_ty.clone()),
                    &ok_ty,
                    &mut ok_be,
                );
                ok_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(out_vid),
                    op: MirAssignOp::Assign,
                    value: val,
                });
            }
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
                } else if let Some(stmt_op) = self.try_lower_stmt_call(tail_expr, &mut err_stmts)? {
                    err_stmts.push(stmt_op);
                } else {
                    let tail_val =
                        self.lower_expr_with_stmts(tail_expr, Some(&ok_ty), &mut err_stmts)?;
                    if tail_val.ty.is_void() {
                        err_stmts.push(MirStmt::Expr(tail_val));
                    } else {
                        let mut err_be = DirectBackend {
                            ctx: self,
                            stmts: &mut err_stmts,
                        };
                        let tail_val = consume_moved_lease_be(
                            coerce_expr_be(tail_val, &ok_ty, &mut err_be),
                            &mut err_be,
                        );
                        err_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(out_vid),
                            op: MirAssignOp::Assign,
                            value: tail_val,
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
            let mut ok_stmts = Vec::new();
            let mut ok_be = DirectBackend {
                ctx: self,
                stmts: &mut ok_stmts,
            };
            let val = coerce_expr_be(
                MirExpr::new(MirExprKind::ResultVal(res_vid), ok_ty.clone()),
                &ok_ty,
                &mut ok_be,
            );
            ok_stmts.push(MirStmt::Assign {
                target: MirPlace::Var(out_vid),
                op: MirAssignOp::Assign,
                value: val,
            });
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

    pub(super) fn lower_failable_expr(
        &mut self,
        expr: &Expr,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<MirExpr, MirLowerError> {
        let me = self.lower_expr_with_stmts(expr, None, stmts)?;
        Ok(me)
    }

    pub(super) fn lower_place(
        &mut self,
        expr: &Expr,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<(MirPlace, MirType), MirLowerError> {
        let mut be = DirectBackend { ctx: self, stmts };
        lower_place_common(expr, &mut be)
    }

    pub(super) fn try_lower_stmt_call(
        &mut self,
        expr: &Expr,
        stmts: &mut Vec<MirStmt>,
    ) -> Result<Option<MirStmt>, MirLowerError> {
        let mut be = DirectBackend { ctx: self, stmts };
        try_lower_nonblocking_call_or_method(expr, &mut be)
    }

    pub(super) fn try_resolve_irq_token(
        &self,
        expr: &Expr,
    ) -> Result<Option<(String, u32)>, MirLowerError> {
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

    pub(super) fn lower_baked_isr_arg(
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
                message:
                    "ISR attachment argument must be statically known (no runtime side effects)"
                        .to_string(),
                span: arg_expr.span,
            });
        }
        let resolved = resolve_boot_const_expr(&raw, &self.boot_const_env);
        let coerced = coerce_expr_if_needed(resolved, param_ty);
        let mut core = &coerced;
        while let MirExprKind::MakePrimView { ptr, .. } = &core.kind {
            core = ptr;
        }
        match &core.kind {
            MirExprKind::GlobalPrimAddr(name) => Ok(MirBakedArg::PrimAddr(name.clone())),
            MirExprKind::MmioAddr(name) => Ok(MirBakedArg::MmioAddr(name.clone())),
            MirExprKind::SubsystemAddr(name) => {
                let root_struct = self
                    .syms
                    .subsystems
                    .get(name)
                    .map(|(s, _)| s.clone())
                    .unwrap_or_default();
                Ok(MirBakedArg::SubsystemPath {
                    root: name.clone(),
                    root_struct,
                    field_chain: Vec::new(),
                })
            }
            MirExprKind::AddrOfPlace(place) => {
                if let Some((root, chain)) = extract_subsystem_place_chain(place) {
                    let root_struct = self
                        .syms
                        .subsystems
                        .get(&root)
                        .map(|(s, _)| s.clone())
                        .unwrap_or_default();
                    return Ok(MirBakedArg::SubsystemPath {
                        root,
                        root_struct,
                        field_chain: chain,
                    });
                }
                Err(MirLowerError {
                    message: "ISR attachment argument must be a compile-time constant, static `&primitive`, `&subsystem`, or `mmio` peripheral pointer"
                        .to_string(),
                    span: arg_expr.span,
                })
            }
            _ if is_const_mir_expr(&coerced) => Ok(MirBakedArg::Const(coerced)),
            _ => Err(MirLowerError {
                message: "ISR attachment argument must be a compile-time constant, static `&primitive`, `&subsystem`, or `mmio` peripheral pointer"
                    .to_string(),
                span: arg_expr.span,
            }),
        }
    }

    pub(super) fn try_lower_irq_attach_stmt(
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

        let isr_param_tys =
            self.syms
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

        let bs = self.boot_state.as_mut().ok_or_else(|| MirLowerError {
            message: "`irq.attach` encountered outside sequential `main()` boot context"
                .to_string(),
            span: expr.span,
        })?;
        if let Some(sel_stmt) = bs.attach_irq(
            &irq_name,
            irq_num,
            is_shared,
            &handler_name,
            baked_args,
            expr.span,
        )? {
            out.push(sel_stmt);
        }
        Ok(true)
    }
}
