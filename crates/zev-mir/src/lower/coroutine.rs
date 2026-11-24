use super::*;

#[derive(Clone, Copy)]
pub(super) struct LoopContext {
    pub(super) continue_bb: BlockId,
    pub(super) break_bb: BlockId,
    pub(super) has_inbox_recv: bool,
    pub(super) defer_depth: usize,
}

pub(super) struct CoroutineLowerCtx<'a> {
    pub(super) syms: &'a ModuleSymbols,
    pub(super) ret_ty: MirType,
    pub(super) error_spec: MirErrorSpec,
    pub(super) manual_yield: bool,
    pub(super) in_noyield: bool,
    pub(super) is_yielding_fn: bool,
    pub(super) has_restart: bool,
    pub(super) may_hold_boost: bool,
    pub(super) vars: Vec<MirVar>,
    pub(super) scopes: VarScopes,
    pub(super) defer_stack: Vec<Vec<Stmt>>,
    pub(super) blocks: Vec<MirBlock>,
    pub(super) cur_bb: BlockId,
    pub(super) cur_is_resume: bool,
    pub(super) cur_stmts: Vec<MirStmt>,
    pub(super) next_bb_id: u16,
    pub(super) child_slots: Vec<MirChildCallSlot>,
    pub(super) loop_stack: Vec<LoopContext>,
}

impl<'a> CoroutineLowerCtx<'a> {
    pub(super) fn new(
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
            is_yielding_fn: false,
            has_restart: false,
            may_hold_boost: false,
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

    pub(super) fn emit_end_inbox_turn(&mut self) {
        self.cur_stmts.push(MirStmt::EndInboxTurn);
        self.may_hold_boost = false;
    }

    pub(super) fn alloc_bb(&mut self) -> BlockId {
        let id = BlockId(self.next_bb_id);
        self.next_bb_id += 1;
        id
    }

    pub(super) fn seal_bb(&mut self, term: MirTerminator, next_bb: BlockId, next_is_resume: bool) {
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
    ) -> Result<(), MirLowerError> {
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

    pub(super) fn pop_and_lower_top_defer_scope(&mut self) -> Result<(), MirLowerError> {
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
    pub(super) fn emit_defers_down_to(&mut self, min_depth: usize) -> Result<(), MirLowerError> {
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

    pub(super) fn finish_with_implicit_return(&mut self) {
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

    pub(super) fn lower_top_level_stmts(&mut self, stmts: &[Stmt]) -> Result<(), MirLowerError> {
        self.lower_stmts(stmts)?;
        self.pop_and_lower_top_defer_scope()?;
        Ok(())
    }

    pub(super) fn lower_scoped_block(&mut self, block: &Block) -> Result<(), MirLowerError> {
        self.scopes.push();
        self.defer_stack.push(Vec::new());
        self.lower_stmts(&block.stmts)?;
        self.pop_and_lower_top_defer_scope()?;
        self.scopes.pop();
        Ok(())
    }

    pub(super) fn lower_stmts(&mut self, stmts: &[Stmt]) -> Result<(), MirLowerError> {
        for stmt in stmts {
            self.lower_stmt(stmt)?;
        }
        Ok(())
    }

    pub(super) fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), MirLowerError> {
        let stmt_manual = self.manual_yield || has_attr_manual_yield(&stmt.attrs);
        match &stmt.kind {
            StmtKind::VarDecl { ty, name, init } => {
                let vty = self.syms.resolve_type(ty);
                let vid = self.declare_var(&name.name, vty.clone(), VarRole::UserLocal);
                if let Some(ie) = init {
                    if !self.try_lower_yielding_into_var(ie, vid, &vty)? {
                        let val = self.lower_expr(ie, Some(&vty))?;
                        let moved = consume_moved_lease_be(coerce_expr_if_needed(val, &vty), self);
                        self.cur_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(vid),
                            op: MirAssignOp::Assign,
                            value: moved,
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
                let moved = consume_moved_lease_be(
                    coerce_expr_if_needed(rhs, &pty.host_value_type()),
                    self,
                );
                self.cur_stmts.push(MirStmt::Assign {
                    target: place,
                    op: convert_assign_op(*op),
                    value: moved,
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
                    let coerced = consume_moved_lease_be(coerce_expr_if_needed(raw, &ret_ty), self);
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
                self.emit_end_inbox_turn();
                self.cur_stmts.push(MirStmt::ZeroTaskLocals);
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
                let entry_boost = self.may_hold_boost;
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
                let then_boost = self.may_hold_boost;
                self.seal_bb(MirTerminator::Goto(join_bb), else_bb, false);
                self.may_hold_boost = entry_boost;
                let else_boost = if let Some(eb) = else_block {
                    self.lower_scoped_block(eb)?;
                    self.may_hold_boost
                } else {
                    entry_boost
                };
                self.may_hold_boost = then_boost || else_boost;
                self.seal_bb(MirTerminator::Goto(join_bb), join_bb, false);
            }
            StmtKind::While { cond, body } => {
                let cond_bb = self.alloc_bb();
                let body_bb = self.alloc_bb();
                let latch_bb = self.alloc_bb();
                let exit_bb = self.alloc_bb();
                let is_while_true = matches!(cond.kind, ExprKind::Literal(Literal::Bool(true)));
                let auto_share = !stmt_manual && !self.in_noyield;
                let has_inbox = ast_expr_has_inbox_recv(cond, &self.syms.boost_capable_fns)
                    || ast_stmts_have_inbox_recv(&body.stmts, &self.syms.boost_capable_fns);
                if self.loop_stack.is_empty() && !has_inbox && self.may_hold_boost {
                    self.emit_end_inbox_turn();
                }
                let entry_boost = self.may_hold_boost;

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
                    self.emit_end_inbox_turn();
                    self.may_hold_boost = entry_boost && !is_while_true;
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
                let has_inbox =
                    ast_stmts_have_inbox_recv(&body.stmts, &self.syms.boost_capable_fns);
                if self.loop_stack.is_empty() && !has_inbox && self.may_hold_boost {
                    self.emit_end_inbox_turn();
                }

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
                    self.emit_end_inbox_turn();
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
                let (period_var, init_period_expr) =
                    if let ExprKind::Literal(Literal::Duration(dur)) = &period.kind {
                        let ticks = self.syms.duration_to_ticks(dur)?;
                        if ticks == 0 {
                            return Err(MirLowerError {
                                message: "periodic `loop every` interval must be greater than 0"
                                    .to_string(),
                                span: period.span,
                            });
                        }
                        let pv = self.alloc_temp("every_period", MirType::U32);
                        (
                            pv,
                            Some(MirExpr::new(MirExprKind::TicksLit(ticks), MirType::U32)),
                        )
                    } else {
                        let raw_p = self.lower_expr(period, None)?;
                        let host_ty = raw_p.ty.host_value_type();
                        let norm_ty = match host_ty {
                            MirType::I8 | MirType::I16 | MirType::I32 => MirType::I32,
                            MirType::I64 => MirType::I64,
                            MirType::Isize => MirType::Isize,
                            MirType::U64 => MirType::U64,
                            MirType::Usize => MirType::Usize,
                            _ => MirType::U32,
                        };
                        let raw_p = widen_arith_mir_expr(raw_p, &norm_ty);
                        let is_signed = matches!(
                            norm_ty,
                            MirType::I8
                                | MirType::I16
                                | MirType::I32
                                | MirType::I64
                                | MirType::Isize
                        );
                        if self.syms.tick_period_us != 1_000 && self.syms.tick_period_us > 0 {
                            let t_us = self.syms.tick_period_us;
                            let max_pre_u64 = (((zev_ast::MAX_DURATION_TICKS as u128)
                                .saturating_mul(t_us as u128))
                            .div_ceil(1_000))
                            .min((u64::MAX / 1_000) as u128)
                                as u64;
                            let pv = self.alloc_temp("every_period", MirType::U32);
                            let u_tmp = self.alloc_temp("every_u64", MirType::U64);
                            let scaled_var = MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Div,
                                    lhs: Box::new(MirExpr::new(
                                        MirExprKind::Binary {
                                            op: MirBinOp::Mul,
                                            lhs: Box::new(MirExpr::new(
                                                MirExprKind::Var(u_tmp),
                                                MirType::U64,
                                            )),
                                            rhs: Box::new(MirExpr::new(
                                                MirExprKind::IntLit(1_000),
                                                MirType::U64,
                                            )),
                                        },
                                        MirType::U64,
                                    )),
                                    rhs: Box::new(MirExpr::new(
                                        MirExprKind::IntLit(t_us as i128),
                                        MirType::U64,
                                    )),
                                },
                                MirType::U64,
                            );
                            let capped_stmt = MirStmt::If {
                                cond: MirExpr::new(
                                    MirExprKind::Binary {
                                        op: MirBinOp::Gt,
                                        lhs: Box::new(MirExpr::new(
                                            MirExprKind::Var(u_tmp),
                                            MirType::U64,
                                        )),
                                        rhs: Box::new(MirExpr::new(
                                            MirExprKind::IntLit(max_pre_u64 as i128),
                                            MirType::U64,
                                        )),
                                    },
                                    MirType::Bool,
                                ),
                                then_body: vec![MirStmt::Assign {
                                    target: MirPlace::Var(pv),
                                    op: MirAssignOp::Assign,
                                    value: MirExpr::new(
                                        MirExprKind::IntLit(zev_ast::MAX_DURATION_TICKS as i128),
                                        MirType::U32,
                                    ),
                                }],
                                else_body: vec![MirStmt::Assign {
                                    target: MirPlace::Var(pv),
                                    op: MirAssignOp::Assign,
                                    value: coerce_or_cast_int(scaled_var, &MirType::U32),
                                }],
                            };
                            if is_signed {
                                let s64 = coerce_or_cast_int(raw_p, &MirType::I64);
                                let s_tmp = self.alloc_temp("every_s64", MirType::I64);
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(s_tmp),
                                    op: MirAssignOp::Assign,
                                    value: s64,
                                });
                                self.cur_stmts.push(MirStmt::If {
                                    cond: MirExpr::new(
                                        MirExprKind::Binary {
                                            op: MirBinOp::Le,
                                            lhs: Box::new(MirExpr::new(
                                                MirExprKind::Var(s_tmp),
                                                MirType::I64,
                                            )),
                                            rhs: Box::new(MirExpr::new(
                                                MirExprKind::IntLit(0),
                                                MirType::I64,
                                            )),
                                        },
                                        MirType::Bool,
                                    ),
                                    then_body: vec![MirStmt::Assign {
                                        target: MirPlace::Var(pv),
                                        op: MirAssignOp::Assign,
                                        value: MirExpr::new(MirExprKind::IntLit(1), MirType::U32),
                                    }],
                                    else_body: vec![
                                        MirStmt::Assign {
                                            target: MirPlace::Var(u_tmp),
                                            op: MirAssignOp::Assign,
                                            value: MirExpr::new(
                                                MirExprKind::Cast {
                                                    expr: Box::new(MirExpr::new(
                                                        MirExprKind::Var(s_tmp),
                                                        MirType::I64,
                                                    )),
                                                    target_ty: MirType::U64,
                                                },
                                                MirType::U64,
                                            ),
                                        },
                                        capped_stmt,
                                    ],
                                });
                            } else {
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(u_tmp),
                                    op: MirAssignOp::Assign,
                                    value: coerce_or_cast_int(raw_p, &MirType::U64),
                                });
                                self.cur_stmts.push(capped_stmt);
                            }
                            (pv, None)
                        } else {
                            let pv = self.alloc_temp("every_period", norm_ty.clone());
                            (pv, Some(coerce_or_cast_int(raw_p, &norm_ty)))
                        }
                    };

                let deadline_var = self.alloc_temp("every_next", MirType::U32);
                if let Some(period_expr) = init_period_expr {
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(period_var),
                        op: MirAssignOp::Assign,
                        value: period_expr,
                    });
                }
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(deadline_var),
                    op: MirAssignOp::Assign,
                    value: MirExpr::new(MirExprKind::NowTicks, MirType::U32),
                });

                let body_bb = self.alloc_bb();
                let latch_bb = self.alloc_bb();
                let exit_bb = self.alloc_bb();
                let has_inbox = ast_expr_has_inbox_recv(period, &self.syms.boost_capable_fns)
                    || ast_stmts_have_inbox_recv(&body.stmts, &self.syms.boost_capable_fns);
                if self.loop_stack.is_empty() && !has_inbox && self.may_hold_boost {
                    self.emit_end_inbox_turn();
                }

                self.seal_bb(MirTerminator::Goto(body_bb), body_bb, true);
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
                self.emit_end_inbox_turn();
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

                let has_inbox =
                    ast_stmts_have_inbox_recv(&body.stmts, &self.syms.boost_capable_fns);
                let entry_boost = self.may_hold_boost;
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
                    self.emit_end_inbox_turn();
                    self.may_hold_boost = entry_boost;
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
                self.cur_stmts.push(MirStmt::BoundedForLatch);
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

                let has_inbox =
                    ast_stmts_have_inbox_recv(&body.stmts, &self.syms.boost_capable_fns);
                let entry_boost = self.may_hold_boost;
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
                    self.emit_end_inbox_turn();
                    self.may_hold_boost = entry_boost;
                }
                self.cur_stmts.push(MirStmt::Assign {
                    target: MirPlace::Var(idx_tmp),
                    op: MirAssignOp::AddAssign,
                    value: MirExpr::new(MirExprKind::IntLit(1), MirType::Usize),
                });
                self.cur_stmts.push(MirStmt::BoundedForLatch);
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

                let entry_boost = self.may_hold_boost;
                let mut exit_boost = if default_bb_opt.is_none() {
                    entry_boost
                } else {
                    false
                };
                for (idx, (_, abb, abody)) in arm_bbs.iter().enumerate() {
                    if self.cur_bb != *abb {
                        self.cur_bb = *abb;
                        self.cur_is_resume = false;
                    }
                    self.may_hold_boost = entry_boost;
                    self.lower_scoped_block(abody)?;
                    exit_boost |= self.may_hold_boost;
                    let next_after = arm_bbs.get(idx + 1).map(|(_, b, _)| *b).unwrap_or(join_bb);
                    self.seal_bb(MirTerminator::Goto(join_bb), next_after, false);
                }
                self.may_hold_boost = exit_boost;

                if self.cur_bb != join_bb {
                    self.cur_bb = join_bb;
                    self.cur_is_resume = false;
                }
            }
            StmtKind::Wait { arms, timer } => {
                self.emit_end_inbox_turn();

                struct LoweredArmMeta<'b> {
                    op: MirWaitArmOp,
                    target_bb: BlockId,
                    bound_var: Option<(String, VarId)>,
                    sets_boost: bool,
                    body: &'b Block,
                }

                let mut arm_metas: Vec<LoweredArmMeta<'_>> = Vec::with_capacity(arms.len());
                for arm in arms {
                    let (receiver, method) = match &arm.call.kind {
                        ExprKind::MethodCall {
                            receiver, method, ..
                        } => (receiver, method.name.as_str()),
                        _ => {
                            return Err(MirLowerError {
                                message:
                                    "`wait { ... }` arm must be a direct primitive method call"
                                        .to_string(),
                                span: arm.call.span,
                            });
                        }
                    };
                    let recv_kind =
                        resolve_prim_receiver_kind_be(receiver, self)?.ok_or_else(|| {
                            MirLowerError {
                                message: "`wait { ... }` arm receiver is not an OS primitive"
                                    .to_string(),
                                span: receiver.span,
                            }
                        })?;

                    let alloc_arm_dst = |this: &mut Self,
                                         bind: &Option<(TypeExpr, zev_ast::Ident)>,
                                         prefix: &str,
                                         ty: MirType|
                     -> (VarId, Option<(String, VarId)>) {
                        if let Some((_, bind_id)) = bind {
                            let vid = VarId(this.vars.len());
                            this.vars.push(MirVar {
                                id: vid,
                                name: bind_id.name.clone(),
                                ty,
                                role: VarRole::UserLocal,
                                in_noyield_scope: this.in_noyield,
                                addr_taken: false,
                                hoisted: false,
                            });
                            (vid, Some((bind_id.name.clone(), vid)))
                        } else {
                            let vid = this.alloc_temp(prefix, ty);
                            (vid, None)
                        }
                    };

                    let (op, bound_var, sets_boost) = match recv_kind {
                        LoweredPrimReceiver::InboxView(view, elem) if method == "recv" => {
                            let (vid, b) =
                                alloc_arm_dst(self, &arm.bind, "wait_inbox", elem.clone());
                            (
                                MirWaitArmOp::InboxViewRecv {
                                    view,
                                    elem_ty: elem,
                                    dst_val: Some(vid),
                                    dst_res: None,
                                },
                                b,
                                true,
                            )
                        }
                        LoweredPrimReceiver::ByteRingView(view) if method == "wait_read" => {
                            let sty = MirType::Slice(Box::new(MirType::U8));
                            let (vid, b) = alloc_arm_dst(self, &arm.bind, "wait_ring", sty);
                            (
                                MirWaitArmOp::ByteRingViewWaitRead {
                                    view,
                                    dst_val: Some(vid),
                                    dst_res: None,
                                },
                                b,
                                true,
                            )
                        }
                        LoweredPrimReceiver::ByteRingView(view) if method == "wait_write" => {
                            let (vid, b) =
                                alloc_arm_dst(self, &arm.bind, "wait_ring_w", MirType::Usize);
                            (
                                MirWaitArmOp::ByteRingViewWaitWrite {
                                    view,
                                    dst_val: Some(vid),
                                    dst_res: None,
                                },
                                b,
                                false,
                            )
                        }
                        LoweredPrimReceiver::Sized(target, pkind) => match (method, pkind) {
                            ("recv", MirPrimKind::Inbox { elem, cap }) => {
                                let (vid, b) =
                                    alloc_arm_dst(self, &arm.bind, "wait_inbox", *elem.clone());
                                (
                                    MirWaitArmOp::InboxRecv {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        dst_val: Some(vid),
                                        dst_res: None,
                                    },
                                    b,
                                    true,
                                )
                            }
                            ("wait", MirPrimKind::Signal) => (
                                MirWaitArmOp::SignalWait {
                                    target,
                                    dst_res: None,
                                },
                                None,
                                true,
                            ),
                            ("wait_read", MirPrimKind::ByteRing { cap }) => {
                                let sty = MirType::Slice(Box::new(MirType::U8));
                                let (vid, b) = alloc_arm_dst(self, &arm.bind, "wait_ring", sty);
                                (
                                    MirWaitArmOp::ByteRingWaitRead {
                                        target,
                                        cap,
                                        dst_val: Some(vid),
                                        dst_res: None,
                                    },
                                    b,
                                    true,
                                )
                            }
                            ("wait_write", MirPrimKind::ByteRing { cap }) => {
                                let (vid, b) =
                                    alloc_arm_dst(self, &arm.bind, "wait_ring_w", MirType::Usize);
                                (
                                    MirWaitArmOp::ByteRingWaitWrite {
                                        target,
                                        cap,
                                        dst_val: Some(vid),
                                        dst_res: None,
                                    },
                                    b,
                                    false,
                                )
                            }
                            ("wait_chunk", MirPrimKind::PingPong { elem, cap }) => {
                                let sty = MirType::Slice(elem.clone());
                                let (vid, b) = alloc_arm_dst(self, &arm.bind, "wait_pp", sty);
                                (
                                    MirWaitArmOp::PingPongWaitChunk {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        dst_slice: vid,
                                    },
                                    b,
                                    false,
                                )
                            }
                            ("wait_change", MirPrimKind::Watch { elem }) => {
                                let (last_gen_var, seeded_var) =
                                    self.prepare_watch_wait_change(&target, &elem);
                                let (vid, b) =
                                    alloc_arm_dst(self, &arm.bind, "wait_watch", *elem.clone());
                                (
                                    MirWaitArmOp::WatchWaitChange {
                                        target,
                                        elem_ty: *elem,
                                        last_gen_var,
                                        seeded_var,
                                        dst_val: Some(vid),
                                        dst_res: None,
                                    },
                                    b,
                                    false,
                                )
                            }
                            ("alloc", MirPrimKind::Pool { elem, cap }) => {
                                let lty = MirType::Lease(elem.clone());
                                let (vid, b) = alloc_arm_dst(self, &arm.bind, "wait_pool", lty);
                                (
                                    MirWaitArmOp::PoolAlloc {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        dst_val: Some(vid),
                                        dst_res: None,
                                    },
                                    b,
                                    false,
                                )
                            }
                            _ => {
                                return Err(MirLowerError {
                                    message: format!(
                                        "unsupported `wait {{ ... }}` arm method `{method}`"
                                    ),
                                    span: arm.call.span,
                                });
                            }
                        },
                        _ => {
                            return Err(MirLowerError {
                                message: format!(
                                    "unsupported `wait {{ ... }}` arm method `{method}`"
                                ),
                                span: arm.call.span,
                            });
                        }
                    };

                    let target_bb = self.alloc_bb();
                    arm_metas.push(LoweredArmMeta {
                        op,
                        target_bb,
                        bound_var,
                        sets_boost,
                        body: &arm.body,
                    });
                }

                let timer_meta = if let Some(t) = timer {
                    let dl_var = match t.kind {
                        zev_ast::WaitTimerKind::Timeout => {
                            self.emit_wait_deadline_var("wait_deadline", &t.expr)?
                        }
                        zev_ast::WaitTimerKind::Deadline => {
                            let dv = self.lower_expr(&t.expr, Some(&MirType::U32))?;
                            let dv = coerce_expr_if_needed(dv, &MirType::U32);
                            let v = self.alloc_temp("wait_deadline", MirType::U32);
                            self.cur_stmts.push(MirStmt::Assign {
                                target: MirPlace::Var(v),
                                op: MirAssignOp::Assign,
                                value: dv,
                            });
                            v
                        }
                    };
                    let t_bb = self.alloc_bb();
                    Some((dl_var, t_bb, &t.body))
                } else {
                    None
                };

                let poll_bb = self.alloc_bb();
                let join_bb = self.alloc_bb();
                let mir_arms: Vec<MirWaitArm> = arm_metas
                    .iter()
                    .map(|m| MirWaitArm {
                        op: m.op.clone(),
                        target_bb: m.target_bb,
                    })
                    .collect();
                let mir_timer = timer_meta.as_ref().map(|(dl_var, t_bb, _)| MirWaitTimer {
                    deadline_var: *dl_var,
                    timeout_bb: *t_bb,
                    timeout_err_res: None,
                    release_lease_on_timeout: None,
                });
                let first_target = arm_metas
                    .first()
                    .map(|m| m.target_bb)
                    .or_else(|| timer_meta.as_ref().map(|(_, tb, _)| *tb))
                    .unwrap_or(join_bb);

                self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
                self.seal_bb(
                    MirTerminator::Yield(MirYieldKind::Wait {
                        arms: mir_arms,
                        timer: mir_timer,
                        poll_bb,
                    }),
                    first_target,
                    false,
                );

                let mut exit_boost = false;
                for (idx, meta) in arm_metas.iter().enumerate() {
                    if self.cur_bb != meta.target_bb {
                        self.cur_bb = meta.target_bb;
                        self.cur_is_resume = false;
                    }
                    self.may_hold_boost = meta.sets_boost;
                    self.scopes.push();
                    self.defer_stack.push(Vec::new());
                    if let Some((name, vid)) = &meta.bound_var {
                        self.scopes.insert(name.clone(), *vid);
                    }
                    self.lower_stmts(&meta.body.stmts)?;
                    self.pop_and_lower_top_defer_scope()?;
                    self.scopes.pop();
                    exit_boost |= self.may_hold_boost;
                    let next_bb = arm_metas
                        .get(idx + 1)
                        .map(|n| n.target_bb)
                        .or_else(|| timer_meta.as_ref().map(|(_, tb, _)| *tb))
                        .unwrap_or(join_bb);
                    self.seal_bb(MirTerminator::Goto(join_bb), next_bb, false);
                }

                if let Some((_, t_bb, t_body)) = timer_meta {
                    if self.cur_bb != t_bb {
                        self.cur_bb = t_bb;
                        self.cur_is_resume = false;
                    }
                    self.may_hold_boost = false;
                    self.lower_scoped_block(t_body)?;
                    exit_boost |= self.may_hold_boost;
                    self.seal_bb(MirTerminator::Goto(join_bb), join_bb, false);
                }

                self.may_hold_boost = exit_boost;
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
                        self.emit_end_inbox_turn();
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
                    if !me.ty.is_void()
                        || matches!(
                            me.kind,
                            MirExprKind::CallDirect { .. } | MirExprKind::RomCallU32 { .. }
                        )
                    {
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

    pub(super) fn emit_coroutine_sys_exit(
        &mut self,
        code_arg_opt: Option<&Expr>,
    ) -> Result<(), MirLowerError> {
        let code_expr = if let Some(ce) = code_arg_opt {
            let raw = self.lower_expr(ce, Some(&MirType::I32))?;
            let coerced = coerce_expr_if_needed(raw, &MirType::I32);
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
        self.emit_end_inbox_turn();
        self.cur_stmts
            .push(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                method: "exit".to_string(),
                args: vec![code_expr],
            }));
        let dead_bb = self.alloc_bb();
        self.seal_bb(MirTerminator::Return(None), dead_bb, false);
        Ok(())
    }

    pub(super) fn lower_iter_into_slice_temp(
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
    pub(super) fn emit_wait_deadline_var(
        &mut self,
        prefix: &str,
        dur_expr: &Expr,
    ) -> Result<VarId, MirLowerError> {
        let ticks = self.lower_expr(dur_expr, Some(&MirType::U32))?;
        let dl_var = self.alloc_temp(prefix, MirType::U32);
        self.cur_stmts.push(MirStmt::Assign {
            target: MirPlace::Var(dl_var),
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
        Ok(dl_var)
    }

    pub(super) fn emit_single_arm_wait(
        &mut self,
        op: MirWaitArmOp,
        timer: Option<MirWaitTimer>,
        poll_bb: BlockId,
        resume_bb: BlockId,
    ) {
        self.seal_bb(MirTerminator::Goto(poll_bb), poll_bb, true);
        self.seal_bb(
            MirTerminator::Yield(MirYieldKind::Wait {
                arms: vec![MirWaitArm {
                    op,
                    target_bb: resume_bb,
                }],
                timer,
                poll_bb,
            }),
            resume_bb,
            false,
        );
    }

    pub(super) fn prepare_watch_wait_change(
        &mut self,
        target: &MirPrimTarget,
        elem_ty: &MirType,
    ) -> (VarId, Option<VarId>) {
        let last_gen_var = self.alloc_temp("watch_gen", MirType::U32);
        self.cur_stmts.push(MirStmt::Assign {
            target: MirPlace::Var(last_gen_var),
            op: MirAssignOp::Assign,
            value: MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchGen {
                    target: target.clone(),
                    elem_ty: elem_ty.clone(),
                })),
                MirType::U32,
            ),
        });
        (last_gen_var, None)
    }

    pub(super) fn try_lower_yielding_into_var(
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
                if let Some(recv_kind) = resolve_prim_receiver_kind_be(receiver, self)? {
                    match recv_kind {
                        LoweredPrimReceiver::ByteRingView(view)
                            if m == "wait_read" && find_named_arg(args, "timeout").is_none() =>
                        {
                            self.emit_end_inbox_turn();
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.emit_single_arm_wait(
                                MirWaitArmOp::ByteRingViewWaitRead {
                                    view,
                                    dst_val: Some(dst_vid),
                                    dst_res: None,
                                },
                                None,
                                poll_bb,
                                resume_bb,
                            );
                            self.may_hold_boost = true;
                            return Ok(true);
                        }
                        LoweredPrimReceiver::ByteRingView(view)
                            if m == "wait_write" && find_named_arg(args, "timeout").is_none() =>
                        {
                            self.emit_end_inbox_turn();
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.emit_single_arm_wait(
                                MirWaitArmOp::ByteRingViewWaitWrite {
                                    view,
                                    dst_val: Some(dst_vid),
                                    dst_res: None,
                                },
                                None,
                                poll_bb,
                                resume_bb,
                            );
                            return Ok(true);
                        }
                        LoweredPrimReceiver::InboxView(view, elem)
                            if m == "recv" && find_named_arg(args, "timeout").is_none() =>
                        {
                            self.emit_end_inbox_turn();
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.emit_single_arm_wait(
                                MirWaitArmOp::InboxViewRecv {
                                    view,
                                    elem_ty: elem,
                                    dst_val: Some(dst_vid),
                                    dst_res: None,
                                },
                                None,
                                poll_bb,
                                resume_bb,
                            );
                            self.may_hold_boost = true;
                            return Ok(true);
                        }
                        LoweredPrimReceiver::Sized(target, pkind) => match (m, pkind) {
                            ("wait_chunk", MirPrimKind::PingPong { elem, cap }) => {
                                self.emit_end_inbox_turn();
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::PingPongWaitChunk {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        dst_slice: dst_vid,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                return Ok(true);
                            }
                            ("alloc", MirPrimKind::Pool { elem, cap })
                                if find_named_arg(args, "timeout").is_none() =>
                            {
                                self.emit_end_inbox_turn();
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::PoolAlloc {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        dst_val: Some(dst_vid),
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                self.may_hold_boost = true;
                                return Ok(true);
                            }
                            ("recv", MirPrimKind::Inbox { elem, cap })
                                if find_named_arg(args, "timeout").is_none() =>
                            {
                                self.emit_end_inbox_turn();
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::InboxRecv {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        dst_val: Some(dst_vid),
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                self.may_hold_boost = true;
                                return Ok(true);
                            }
                            ("wait_read", MirPrimKind::ByteRing { cap })
                                if find_named_arg(args, "timeout").is_none() =>
                            {
                                self.emit_end_inbox_turn();
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::ByteRingWaitRead {
                                        target,
                                        cap,
                                        dst_val: Some(dst_vid),
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                self.may_hold_boost = true;
                                return Ok(true);
                            }
                            ("wait_write", MirPrimKind::ByteRing { cap })
                                if find_named_arg(args, "timeout").is_none() =>
                            {
                                self.emit_end_inbox_turn();
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::ByteRingWaitWrite {
                                        target,
                                        cap,
                                        dst_val: Some(dst_vid),
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                return Ok(true);
                            }
                            ("wait_change", MirPrimKind::Watch { elem })
                                if find_named_arg(args, "timeout").is_none() =>
                            {
                                self.emit_end_inbox_turn();
                                let (last_gen_var, seeded_var) =
                                    self.prepare_watch_wait_change(&target, &elem);
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::WatchWaitChange {
                                        target,
                                        elem_ty: *elem,
                                        last_gen_var,
                                        seeded_var,
                                        dst_val: Some(dst_vid),
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                return Ok(true);
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        let _ = dst_ty;
        Ok(false)
    }

    pub(super) fn is_void_yielding_stmt_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                if let Some((wfn, _)) = resolve_user_sink_write_callee(callee, generic_args, args) {
                    if let Some(sig) = self.syms.fns.get(&wfn) {
                        let yields_here = sig.is_yielding
                            && !(self.in_noyield && sig.effect == MirFnEffect::SoftYielding);
                        return yields_here && !sig.error_spec.has_error();
                    }
                }
                if let ExprKind::Ident(id) = &callee.kind {
                    if matches!(
                        id.name.as_str(),
                        "delay" | "delay_ms" | "sleep" | "checkpoint"
                    ) {
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
                    if id.name == SYS_NAMESPACE
                        && SysMethod::from_name(&method.name) == Some(SysMethod::Delay)
                    {
                        return true;
                    }
                }
                let m = method.name.as_str();
                (m == "send" || m == "wait" || m == "wait_change")
                    && find_named_arg(args, "timeout").is_none()
            }
            _ => false,
        }
    }

    pub(super) fn try_lower_yielding_stmt_expr(
        &mut self,
        expr: &Expr,
    ) -> Result<bool, MirLowerError> {
        match &expr.kind {
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                if let Some((wfn, newline)) =
                    resolve_user_sink_write_callee(callee, generic_args, args)
                {
                    if let Some(sig) = self.syms.fns.get(&wfn).cloned() {
                        let yields_here = sig.is_yielding
                            && !(self.in_noyield && sig.effect == MirFnEffect::SoftYielding);
                        if yields_here && !sig.error_spec.has_error() {
                            let _ = self.emit_yielding_user_sink_print(
                                &wfn, &sig, newline, args, expr.span,
                            )?;
                            return Ok(true);
                        }
                    }
                }
                if let ExprKind::Ident(id) = &callee.kind {
                    match BuiltinFn::from_name(&id.name) {
                        Some(b) if b.is_blocking_delay() => {
                            if let Some(first) = args.first() {
                                self.emit_delay(&first.value)?;
                                return Ok(true);
                            }
                        }
                        Some(BuiltinFn::Checkpoint) => {
                            if !self.in_noyield {
                                let resume_bb = self.alloc_bb();
                                self.seal_bb(
                                    MirTerminator::Yield(MirYieldKind::ShareCheckpoint {
                                        resume_bb,
                                    }),
                                    resume_bb,
                                    true,
                                );
                            }
                            return Ok(true);
                        }
                        _ => {}
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
                    if id.name == SYS_NAMESPACE
                        && SysMethod::from_name(&method.name) == Some(SysMethod::Delay)
                    {
                        if let Some(first) = args.first() {
                            self.emit_delay(&first.value)?;
                            return Ok(true);
                        }
                    }
                }
                let m = method.name.as_str();
                if let Some(recv_kind) = resolve_prim_receiver_kind_be(receiver, self)? {
                    match recv_kind {
                        LoweredPrimReceiver::InboxView(view, elem)
                            if m == "send" && find_named_arg(args, "timeout").is_none() =>
                        {
                            let vexpr = if let Some(first) = first_positional_arg(args) {
                                let raw = self.lower_expr(first, Some(&elem))?;
                                coerce_expr_if_needed(raw, &elem)
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), elem.clone())
                            };
                            let vexpr = consume_moved_lease_be(vexpr, self);
                            let val_var = self.alloc_temp("inbox_tx", elem.clone());
                            self.cur_stmts.push(MirStmt::Assign {
                                target: MirPlace::Var(val_var),
                                op: MirAssignOp::Assign,
                                value: vexpr,
                            });
                            let poll_bb = self.alloc_bb();
                            let resume_bb = self.alloc_bb();
                            self.emit_single_arm_wait(
                                MirWaitArmOp::InboxViewSend {
                                    view,
                                    elem_ty: elem,
                                    val_var,
                                    dst_res: None,
                                },
                                None,
                                poll_bb,
                                resume_bb,
                            );
                            return Ok(true);
                        }
                        LoweredPrimReceiver::Sized(target, pkind) => match (m, pkind) {
                            ("send", MirPrimKind::Inbox { elem, cap })
                                if find_named_arg(args, "timeout").is_none() =>
                            {
                                let vexpr = if let Some(first) = first_positional_arg(args) {
                                    let raw = self.lower_expr(first, Some(&elem))?;
                                    coerce_expr_if_needed(raw, &elem)
                                } else {
                                    MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                                };
                                let vexpr = consume_moved_lease_be(vexpr, self);
                                let val_var = self.alloc_temp("inbox_tx", *elem.clone());
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(val_var),
                                    op: MirAssignOp::Assign,
                                    value: vexpr,
                                });
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::InboxSend {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        val_var,
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                return Ok(true);
                            }
                            ("wait", MirPrimKind::Signal)
                                if find_named_arg(args, "timeout").is_none() =>
                            {
                                self.emit_end_inbox_turn();
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::SignalWait {
                                        target,
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                self.may_hold_boost = true;
                                return Ok(true);
                            }
                            ("commit_wait", MirPrimKind::WaitQueue)
                                if find_named_arg(args, "timeout").is_none() =>
                            {
                                self.emit_end_inbox_turn();
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::WaitQueueCommit {
                                        target,
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                self.may_hold_boost = true;
                                return Ok(true);
                            }
                            ("wait_change", MirPrimKind::Watch { elem })
                                if find_named_arg(args, "timeout").is_none() =>
                            {
                                self.emit_end_inbox_turn();
                                let (last_gen_var, seeded_var) =
                                    self.prepare_watch_wait_change(&target, &elem);
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::WatchWaitChange {
                                        target,
                                        elem_ty: *elem,
                                        last_gen_var,
                                        seeded_var,
                                        dst_val: None,
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                return Ok(true);
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        Ok(false)
    }

    pub(super) fn emit_delay(&mut self, dur_expr: &Expr) -> Result<(), MirLowerError> {
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

    pub(super) fn emit_yielding_fn_call_exprs(
        &mut self,
        callee: &str,
        largs: Vec<MirExpr>,
        dst: Option<MirCallDst>,
    ) {
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
        if self.syms.boost_capable_fns.contains(callee) {
            self.may_hold_boost = true;
        }
    }

    pub(super) fn emit_yielding_user_sink_print(
        &mut self,
        wfn: &str,
        sig: &FnSig,
        newline: bool,
        args: &[CallArg],
        span: Span,
    ) -> Result<MirExpr, MirLowerError> {
        if sig.spawns_in_boot {
            return Err(MirLowerError {
                message: format!(
                    "boot helper `{wfn}` (contains `spawn` or IRQ `.attach`) may only be called during `fn main()` boot"
                ),
                span,
            });
        }
        let sink_ptr_var = lower_user_sink_ptr_var_be(&args[0].value, self)?;
        let chunks = lower_user_sink_chunks_be(&args[1].value, newline, self)?;
        let scratch_cap = user_sink_scratch_cap(&chunks);
        let scratch = if scratch_cap > 0 {
            let sv = self.alloc_temp(
                "sink_fmt_buf",
                MirType::Array {
                    elem: Box::new(MirType::U8),
                    len: scratch_cap,
                },
            );
            Some((sv, scratch_cap))
        } else {
            None
        };

        let res_ty_opt = if sig.error_spec.has_error() {
            Some(MirType::Result {
                ok: Box::new(MirType::Void),
                err: sig.error_spec.err_type().map(Box::new),
            })
        } else {
            None
        };
        let res_vid_opt = res_ty_opt.as_ref().map(|rt| {
            let rv = self.alloc_temp("sink_write_res", rt.clone());
            self.cur_stmts.push(MirStmt::Assign {
                target: MirPlace::Var(rv),
                op: MirAssignOp::Assign,
                value: MirExpr::new(MirExprKind::IntLit(0), rt.clone()),
            });
            rv
        });
        let err_join_bb = res_vid_opt.map(|_| self.alloc_bb());
        let sink_ptr_ty = self.vars[sink_ptr_var.0].ty.clone();
        let slice_u8_ty = MirType::Slice(Box::new(MirType::U8));

        let emit_chunk_write = |this: &mut Self, slice_expr: MirExpr| {
            let call_args = vec![
                MirExpr::new(MirExprKind::Var(sink_ptr_var), sink_ptr_ty.clone()),
                slice_expr,
            ];
            this.emit_yielding_fn_call_exprs(wfn, call_args, res_vid_opt.map(MirCallDst::Result));
            if let (Some(rv), Some(ej_bb)) = (res_vid_opt, err_join_bb) {
                let ok_bb = this.alloc_bb();
                this.seal_bb(
                    MirTerminator::Branch {
                        cond: MirExpr::new(MirExprKind::ResultIsErr(rv), MirType::Bool),
                        then_bb: ej_bb,
                        else_bb: ok_bb,
                    },
                    ok_bb,
                    false,
                );
            }
        };

        for ch in chunks {
            match ch {
                MirFmtChunk::DebugSlice { slice, .. } => {
                    let ds_ty = slice.ty.clone();
                    let raw_el_ty = match &ds_ty {
                        MirType::Slice(inner) => (**inner).clone(),
                        _ => MirType::U8,
                    };
                    let ds_var = self.alloc_temp("sink_ds", ds_ty.clone());
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(ds_var),
                        op: MirAssignOp::Assign,
                        value: slice,
                    });
                    emit_chunk_write(
                        self,
                        MirExpr::new(MirExprKind::StringLit("[".to_string()), slice_u8_ty.clone()),
                    );
                    let idx_var = self.alloc_temp("sink_ds_i", MirType::Usize);
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(idx_var),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(MirExprKind::IntLit(0), MirType::Usize),
                    });
                    let hdr_bb = self.alloc_bb();
                    let body_bb = self.alloc_bb();
                    let sep_bb = self.alloc_bb();
                    let el_bb = self.alloc_bb();
                    let done_bb = self.alloc_bb();
                    self.seal_bb(MirTerminator::Goto(hdr_bb), hdr_bb, false);
                    let len_expr = MirExpr::new(
                        MirExprKind::SliceLen(Box::new(MirExpr::new(
                            MirExprKind::Var(ds_var),
                            ds_ty.clone(),
                        ))),
                        MirType::Usize,
                    );
                    self.seal_bb(
                        MirTerminator::Branch {
                            cond: MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Lt,
                                    lhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(idx_var),
                                        MirType::Usize,
                                    )),
                                    rhs: Box::new(len_expr),
                                },
                                MirType::Bool,
                            ),
                            then_bb: body_bb,
                            else_bb: done_bb,
                        },
                        body_bb,
                        false,
                    );
                    self.seal_bb(
                        MirTerminator::Branch {
                            cond: MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Gt,
                                    lhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(idx_var),
                                        MirType::Usize,
                                    )),
                                    rhs: Box::new(MirExpr::new(
                                        MirExprKind::IntLit(0),
                                        MirType::Usize,
                                    )),
                                },
                                MirType::Bool,
                            ),
                            then_bb: sep_bb,
                            else_bb: el_bb,
                        },
                        sep_bb,
                        false,
                    );
                    emit_chunk_write(
                        self,
                        MirExpr::new(
                            MirExprKind::StringLit(", ".to_string()),
                            slice_u8_ty.clone(),
                        ),
                    );
                    self.seal_bb(MirTerminator::Goto(el_bb), el_bb, false);
                    let el_expr = MirExpr::new(
                        MirExprKind::Index {
                            base: Box::new(MirExpr::new(MirExprKind::Var(ds_var), ds_ty.clone())),
                            index: Box::new(MirExpr::new(
                                MirExprKind::Var(idx_var),
                                MirType::Usize,
                            )),
                            is_slice: true,
                        },
                        raw_el_ty,
                    );
                    let mut el_chunks = Vec::new();
                    emit_fmt_value_be(el_expr, FormatSpec::Debug, span, &mut el_chunks, self)?;
                    for el_ch in el_chunks {
                        let sl = chunk_to_slice_or_scalar_be(el_ch, scratch, self);
                        emit_chunk_write(self, sl);
                    }
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(idx_var),
                        op: MirAssignOp::AddAssign,
                        value: MirExpr::new(MirExprKind::IntLit(1), MirType::Usize),
                    });
                    self.seal_bb(MirTerminator::Goto(hdr_bb), done_bb, false);
                    emit_chunk_write(
                        self,
                        MirExpr::new(MirExprKind::StringLit("]".to_string()), slice_u8_ty.clone()),
                    );
                }
                MirFmtChunk::PaddedBytes { slice, pad } => {
                    let raw_slice = slice;
                    let s_var = self.alloc_temp("sink_pad_s", slice_u8_ty.clone());
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(s_var),
                        op: MirAssignOp::Assign,
                        value: raw_slice,
                    });
                    let slen_var = self.alloc_temp("sink_pad_len", MirType::Usize);
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(slen_var),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(
                            MirExprKind::SliceLen(Box::new(MirExpr::new(
                                MirExprKind::Var(s_var),
                                slice_u8_ty.clone(),
                            ))),
                            MirType::Usize,
                        ),
                    });
                    let rem_var = self.alloc_temp("sink_pad_rem", MirType::Usize);
                    let w_lit =
                        MirExpr::new(MirExprKind::IntLit(i128::from(pad.width)), MirType::Usize);
                    self.cur_stmts.push(MirStmt::If {
                        cond: MirExpr::new(
                            MirExprKind::Binary {
                                op: MirBinOp::Lt,
                                lhs: Box::new(MirExpr::new(
                                    MirExprKind::Var(slen_var),
                                    MirType::Usize,
                                )),
                                rhs: Box::new(w_lit.clone()),
                            },
                            MirType::Bool,
                        ),
                        then_body: vec![MirStmt::Assign {
                            target: MirPlace::Var(rem_var),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Sub,
                                    lhs: Box::new(w_lit),
                                    rhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(slen_var),
                                        MirType::Usize,
                                    )),
                                },
                                MirType::Usize,
                            ),
                        }],
                        else_body: vec![MirStmt::Assign {
                            target: MirPlace::Var(rem_var),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(MirExprKind::IntLit(0), MirType::Usize),
                        }],
                    });
                    let emit_pad = |this: &mut Self| {
                        let fill_str = if pad.zero_pad {
                            "0000000000000000"
                        } else {
                            "                "
                        };
                        let hdr_bb = this.alloc_bb();
                        let body_bb = this.alloc_bb();
                        let done_bb = this.alloc_bb();
                        this.seal_bb(MirTerminator::Goto(hdr_bb), hdr_bb, false);
                        this.seal_bb(
                            MirTerminator::Branch {
                                cond: MirExpr::new(
                                    MirExprKind::Binary {
                                        op: MirBinOp::Gt,
                                        lhs: Box::new(MirExpr::new(
                                            MirExprKind::Var(rem_var),
                                            MirType::Usize,
                                        )),
                                        rhs: Box::new(MirExpr::new(
                                            MirExprKind::IntLit(0),
                                            MirType::Usize,
                                        )),
                                    },
                                    MirType::Bool,
                                ),
                                then_bb: body_bb,
                                else_bb: done_bb,
                            },
                            body_bb,
                            false,
                        );
                        let step_var = this.alloc_temp("sink_pad_step", MirType::Usize);
                        this.cur_stmts.push(MirStmt::If {
                            cond: MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Gt,
                                    lhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(rem_var),
                                        MirType::Usize,
                                    )),
                                    rhs: Box::new(MirExpr::new(
                                        MirExprKind::IntLit(16),
                                        MirType::Usize,
                                    )),
                                },
                                MirType::Bool,
                            ),
                            then_body: vec![MirStmt::Assign {
                                target: MirPlace::Var(step_var),
                                op: MirAssignOp::Assign,
                                value: MirExpr::new(MirExprKind::IntLit(16), MirType::Usize),
                            }],
                            else_body: vec![MirStmt::Assign {
                                target: MirPlace::Var(step_var),
                                op: MirAssignOp::Assign,
                                value: MirExpr::new(MirExprKind::Var(rem_var), MirType::Usize),
                            }],
                        });
                        let pad_slice = MirExpr::new(
                            MirExprKind::SubSlice {
                                base: Box::new(MirExpr::new(
                                    MirExprKind::StringLit(fill_str.to_string()),
                                    slice_u8_ty.clone(),
                                )),
                                start: Box::new(MirExpr::new(
                                    MirExprKind::IntLit(0),
                                    MirType::Usize,
                                )),
                                end: Box::new(MirExpr::new(
                                    MirExprKind::Var(step_var),
                                    MirType::Usize,
                                )),
                                elem_ty: MirType::U8,
                                base_is_array: false,
                            },
                            slice_u8_ty.clone(),
                        );
                        emit_chunk_write(this, pad_slice);
                        this.cur_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(rem_var),
                            op: MirAssignOp::SubAssign,
                            value: MirExpr::new(MirExprKind::Var(step_var), MirType::Usize),
                        });
                        this.seal_bb(MirTerminator::Goto(hdr_bb), done_bb, false);
                    };
                    if pad.left_align {
                        emit_chunk_write(
                            self,
                            MirExpr::new(MirExprKind::Var(s_var), slice_u8_ty.clone()),
                        );
                        emit_pad(self);
                    } else {
                        emit_pad(self);
                        emit_chunk_write(
                            self,
                            MirExpr::new(MirExprKind::Var(s_var), slice_u8_ty.clone()),
                        );
                    }
                }
                MirFmtChunk::PaddedStr { cstr, pad } => {
                    let raw_slice = if let MirExprKind::StringLit(s) = cstr.kind {
                        MirExpr::new(MirExprKind::StringLit(s), slice_u8_ty.clone())
                    } else {
                        MirExpr::new(
                            MirExprKind::CStrToSlice(Box::new(cstr)),
                            slice_u8_ty.clone(),
                        )
                    };
                    let s_var = self.alloc_temp("sink_pad_s", slice_u8_ty.clone());
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(s_var),
                        op: MirAssignOp::Assign,
                        value: raw_slice,
                    });
                    let slen_var = self.alloc_temp("sink_pad_len", MirType::Usize);
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(slen_var),
                        op: MirAssignOp::Assign,
                        value: MirExpr::new(
                            MirExprKind::SliceLen(Box::new(MirExpr::new(
                                MirExprKind::Var(s_var),
                                slice_u8_ty.clone(),
                            ))),
                            MirType::Usize,
                        ),
                    });
                    let rem_var = self.alloc_temp("sink_pad_rem", MirType::Usize);
                    let w_lit =
                        MirExpr::new(MirExprKind::IntLit(i128::from(pad.width)), MirType::Usize);
                    self.cur_stmts.push(MirStmt::If {
                        cond: MirExpr::new(
                            MirExprKind::Binary {
                                op: MirBinOp::Lt,
                                lhs: Box::new(MirExpr::new(
                                    MirExprKind::Var(slen_var),
                                    MirType::Usize,
                                )),
                                rhs: Box::new(w_lit.clone()),
                            },
                            MirType::Bool,
                        ),
                        then_body: vec![MirStmt::Assign {
                            target: MirPlace::Var(rem_var),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Sub,
                                    lhs: Box::new(w_lit),
                                    rhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(slen_var),
                                        MirType::Usize,
                                    )),
                                },
                                MirType::Usize,
                            ),
                        }],
                        else_body: vec![MirStmt::Assign {
                            target: MirPlace::Var(rem_var),
                            op: MirAssignOp::Assign,
                            value: MirExpr::new(MirExprKind::IntLit(0), MirType::Usize),
                        }],
                    });
                    let emit_pad = |this: &mut Self| {
                        let fill_str = if pad.zero_pad {
                            "0000000000000000"
                        } else {
                            "                "
                        };
                        let hdr_bb = this.alloc_bb();
                        let body_bb = this.alloc_bb();
                        let done_bb = this.alloc_bb();
                        this.seal_bb(MirTerminator::Goto(hdr_bb), hdr_bb, false);
                        this.seal_bb(
                            MirTerminator::Branch {
                                cond: MirExpr::new(
                                    MirExprKind::Binary {
                                        op: MirBinOp::Gt,
                                        lhs: Box::new(MirExpr::new(
                                            MirExprKind::Var(rem_var),
                                            MirType::Usize,
                                        )),
                                        rhs: Box::new(MirExpr::new(
                                            MirExprKind::IntLit(0),
                                            MirType::Usize,
                                        )),
                                    },
                                    MirType::Bool,
                                ),
                                then_bb: body_bb,
                                else_bb: done_bb,
                            },
                            body_bb,
                            false,
                        );
                        let step_var = this.alloc_temp("sink_pad_step", MirType::Usize);
                        this.cur_stmts.push(MirStmt::If {
                            cond: MirExpr::new(
                                MirExprKind::Binary {
                                    op: MirBinOp::Gt,
                                    lhs: Box::new(MirExpr::new(
                                        MirExprKind::Var(rem_var),
                                        MirType::Usize,
                                    )),
                                    rhs: Box::new(MirExpr::new(
                                        MirExprKind::IntLit(16),
                                        MirType::Usize,
                                    )),
                                },
                                MirType::Bool,
                            ),
                            then_body: vec![MirStmt::Assign {
                                target: MirPlace::Var(step_var),
                                op: MirAssignOp::Assign,
                                value: MirExpr::new(MirExprKind::IntLit(16), MirType::Usize),
                            }],
                            else_body: vec![MirStmt::Assign {
                                target: MirPlace::Var(step_var),
                                op: MirAssignOp::Assign,
                                value: MirExpr::new(MirExprKind::Var(rem_var), MirType::Usize),
                            }],
                        });
                        let pad_slice = MirExpr::new(
                            MirExprKind::SubSlice {
                                base: Box::new(MirExpr::new(
                                    MirExprKind::StringLit(fill_str.to_string()),
                                    slice_u8_ty.clone(),
                                )),
                                start: Box::new(MirExpr::new(
                                    MirExprKind::IntLit(0),
                                    MirType::Usize,
                                )),
                                end: Box::new(MirExpr::new(
                                    MirExprKind::Var(step_var),
                                    MirType::Usize,
                                )),
                                elem_ty: MirType::U8,
                                base_is_array: false,
                            },
                            slice_u8_ty.clone(),
                        );
                        emit_chunk_write(this, pad_slice);
                        this.cur_stmts.push(MirStmt::Assign {
                            target: MirPlace::Var(rem_var),
                            op: MirAssignOp::SubAssign,
                            value: MirExpr::new(MirExprKind::Var(step_var), MirType::Usize),
                        });
                        this.seal_bb(MirTerminator::Goto(hdr_bb), done_bb, false);
                    };
                    if pad.left_align {
                        emit_chunk_write(
                            self,
                            MirExpr::new(MirExprKind::Var(s_var), slice_u8_ty.clone()),
                        );
                        emit_pad(self);
                    } else {
                        emit_pad(self);
                        emit_chunk_write(
                            self,
                            MirExpr::new(MirExprKind::Var(s_var), slice_u8_ty.clone()),
                        );
                    }
                }
                other_ch => {
                    let sl = chunk_to_slice_or_scalar_be(other_ch, scratch, self);
                    emit_chunk_write(self, sl);
                }
            }
        }

        if let (Some(ej_bb), Some(rv), Some(rty)) = (err_join_bb, res_vid_opt, res_ty_opt) {
            self.seal_bb(MirTerminator::Goto(ej_bb), ej_bb, false);
            Ok(MirExpr::new(MirExprKind::Var(rv), rty))
        } else {
            Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void))
        }
    }

    pub(super) fn emit_yielding_fn_call(
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
            largs.push(consume_moved_lease_be(coerced, self));
        }
        self.emit_yielding_fn_call_exprs(callee, largs, dst);
        Ok(())
    }

    pub(super) fn lower_expr(
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
                    let fb_val =
                        consume_moved_lease_be(coerce_expr_if_needed(fb_val, &ok_ty), self);
                    self.cur_stmts.push(MirStmt::Assign {
                        target: MirPlace::Var(out_vid),
                        op: MirAssignOp::Assign,
                        value: fb_val,
                    });
                    self.seal_bb(MirTerminator::Goto(join_bb), join_bb, false);
                    Ok(MirExpr::new(MirExprKind::Var(out_vid), ok_ty))
                } else {
                    if !fb_val.ty.is_void()
                        || matches!(
                            fb_val.kind,
                            MirExprKind::CallDirect { .. } | MirExprKind::RomCallU32 { .. }
                        )
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
                                if matches!(
                                    tail_val.kind,
                                    MirExprKind::CallDirect { .. } | MirExprKind::RomCallU32 { .. }
                                ) {
                                    self.cur_stmts.push(MirStmt::Expr(tail_val));
                                }
                            } else {
                                let tail_val = consume_moved_lease_be(
                                    coerce_expr_if_needed(tail_val, &ok_ty),
                                    self,
                                );
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(out_vid),
                                    op: MirAssignOp::Assign,
                                    value: tail_val,
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
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                if let Some((wfn, newline)) =
                    resolve_user_sink_write_callee(callee, generic_args, args)
                {
                    if let Some(sig) = self.syms.fns.get(&wfn).cloned() {
                        let yields_here = sig.is_yielding
                            && !(self.in_noyield && sig.effect == MirFnEffect::SoftYielding);
                        if yields_here {
                            return self.emit_yielding_user_sink_print(
                                &wfn, &sig, newline, args, expr.span,
                            );
                        }
                    }
                    return self.lower_non_yielding_expr(expr, expected);
                }
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
                let m = method.name.as_str();
                if let Some(recv_kind) = resolve_prim_receiver_kind_be(receiver, self)? {
                    match recv_kind {
                        LoweredPrimReceiver::InboxView(view, elem) => match m {
                            "send" => {
                                let vexpr = if let Some(first) = first_positional_arg(args) {
                                    let raw = self.lower_expr(first, Some(&elem))?;
                                    coerce_expr_if_needed(raw, &elem)
                                } else {
                                    MirExpr::new(MirExprKind::IntLit(0), elem.clone())
                                };
                                let vexpr = consume_moved_lease_be(vexpr, self);
                                let val_var = self.alloc_temp("inbox_tx", elem.clone());
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(val_var),
                                    op: MirAssignOp::Assign,
                                    value: vexpr,
                                });
                                if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("inbox_tx_deadline", texpr)?;
                                    let rty = MirType::Result {
                                        ok: Box::new(MirType::Void),
                                        err: None,
                                    };
                                    let dst_res = self.alloc_temp("inbox_tx_res", rty.clone());
                                    let release_lease_on_timeout =
                                        matches!(&elem, MirType::Lease(_))
                                            .then(|| (val_var, elem.clone()));
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::InboxViewSend {
                                            view,
                                            elem_ty: elem,
                                            val_var,
                                            dst_res: Some(dst_res),
                                        },
                                        Some(MirWaitTimer {
                                            deadline_var: dl_var,
                                            timeout_bb: resume_bb,
                                            timeout_err_res: Some(dst_res),
                                            release_lease_on_timeout,
                                        }),
                                        poll_bb,
                                        resume_bb,
                                    );
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                                } else {
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::InboxViewSend {
                                            view,
                                            elem_ty: elem,
                                            val_var,
                                            dst_res: None,
                                        },
                                        None,
                                        poll_bb,
                                        resume_bb,
                                    );
                                    return Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void));
                                }
                            }
                            "recv" => {
                                self.emit_end_inbox_turn();
                                if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("inbox_deadline", texpr)?;
                                    let rty = MirType::Result {
                                        ok: Box::new(elem.clone()),
                                        err: None,
                                    };
                                    let dst_res = self.alloc_temp("inbox_res", rty.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::InboxViewRecv {
                                            view,
                                            elem_ty: elem,
                                            dst_val: None,
                                            dst_res: Some(dst_res),
                                        },
                                        Some(MirWaitTimer {
                                            deadline_var: dl_var,
                                            timeout_bb: resume_bb,
                                            timeout_err_res: Some(dst_res),
                                            release_lease_on_timeout: None,
                                        }),
                                        poll_bb,
                                        resume_bb,
                                    );
                                    self.may_hold_boost = true;
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                                } else {
                                    let dst_val = self.alloc_temp("inbox_val", elem.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::InboxViewRecv {
                                            view,
                                            elem_ty: elem.clone(),
                                            dst_val: Some(dst_val),
                                            dst_res: None,
                                        },
                                        None,
                                        poll_bb,
                                        resume_bb,
                                    );
                                    self.may_hold_boost = true;
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_val), elem));
                                }
                            }
                            _ => {}
                        },
                        LoweredPrimReceiver::ByteRingView(view) if m == "wait_read" => {
                            self.emit_end_inbox_turn();
                            let slice_ty = MirType::Slice(Box::new(MirType::U8));
                            if let Some(texpr) = find_named_arg(args, "timeout") {
                                let dl_var = self.emit_wait_deadline_var("ring_deadline", texpr)?;
                                let rty = MirType::Result {
                                    ok: Box::new(slice_ty),
                                    err: None,
                                };
                                let dst_res = self.alloc_temp("ring_res", rty.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::ByteRingViewWaitRead {
                                        view,
                                        dst_val: None,
                                        dst_res: Some(dst_res),
                                    },
                                    Some(MirWaitTimer {
                                        deadline_var: dl_var,
                                        timeout_bb: resume_bb,
                                        timeout_err_res: Some(dst_res),
                                        release_lease_on_timeout: None,
                                    }),
                                    poll_bb,
                                    resume_bb,
                                );
                                self.may_hold_boost = true;
                                return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                            } else {
                                let dst_val = self.alloc_temp("ring_chunk", slice_ty.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::ByteRingViewWaitRead {
                                        view,
                                        dst_val: Some(dst_val),
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                self.may_hold_boost = true;
                                return Ok(MirExpr::new(MirExprKind::Var(dst_val), slice_ty));
                            }
                        }
                        LoweredPrimReceiver::ByteRingView(view) if m == "wait_write" => {
                            self.emit_end_inbox_turn();
                            if let Some(texpr) = find_named_arg(args, "timeout") {
                                let dl_var = self.emit_wait_deadline_var("ring_deadline", texpr)?;
                                let rty = MirType::Result {
                                    ok: Box::new(MirType::Usize),
                                    err: None,
                                };
                                let dst_res = self.alloc_temp("ring_w_res", rty.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::ByteRingViewWaitWrite {
                                        view,
                                        dst_val: None,
                                        dst_res: Some(dst_res),
                                    },
                                    Some(MirWaitTimer {
                                        deadline_var: dl_var,
                                        timeout_bb: resume_bb,
                                        timeout_err_res: Some(dst_res),
                                        release_lease_on_timeout: None,
                                    }),
                                    poll_bb,
                                    resume_bb,
                                );
                                return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                            } else {
                                let dst_val = self.alloc_temp("ring_space", MirType::Usize);
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::ByteRingViewWaitWrite {
                                        view,
                                        dst_val: Some(dst_val),
                                        dst_res: None,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                return Ok(MirExpr::new(MirExprKind::Var(dst_val), MirType::Usize));
                            }
                        }
                        LoweredPrimReceiver::ByteRingView(_) => {}
                        LoweredPrimReceiver::Sized(target, pkind) => match (m, pkind) {
                            ("wait_chunk", MirPrimKind::PingPong { elem, cap }) => {
                                self.emit_end_inbox_turn();
                                let sty = MirType::Slice(elem.clone());
                                let dst_vid = self.alloc_temp("pp_chunk", sty.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::PingPongWaitChunk {
                                        target,
                                        elem_ty: *elem,
                                        cap,
                                        dst_slice: dst_vid,
                                    },
                                    None,
                                    poll_bb,
                                    resume_bb,
                                );
                                return Ok(MirExpr::new(MirExprKind::Var(dst_vid), sty));
                            }
                            ("alloc", MirPrimKind::Pool { elem, cap }) => {
                                self.emit_end_inbox_turn();
                                let lease_ty = MirType::Lease(elem.clone());
                                if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("pool_deadline", texpr)?;
                                    let rty = MirType::Result {
                                        ok: Box::new(lease_ty),
                                        err: None,
                                    };
                                    let dst_res = self.alloc_temp("pool_res", rty.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::PoolAlloc {
                                            target,
                                            elem_ty: *elem,
                                            cap,
                                            dst_val: None,
                                            dst_res: Some(dst_res),
                                        },
                                        Some(MirWaitTimer {
                                            deadline_var: dl_var,
                                            timeout_bb: resume_bb,
                                            timeout_err_res: Some(dst_res),
                                            release_lease_on_timeout: None,
                                        }),
                                        poll_bb,
                                        resume_bb,
                                    );
                                    self.may_hold_boost = true;
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                                } else {
                                    let dst_val = self.alloc_temp("pool_lease", lease_ty.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::PoolAlloc {
                                            target,
                                            elem_ty: *elem,
                                            cap,
                                            dst_val: Some(dst_val),
                                            dst_res: None,
                                        },
                                        None,
                                        poll_bb,
                                        resume_bb,
                                    );
                                    self.may_hold_boost = true;
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_val), lease_ty));
                                }
                            }
                            ("send", MirPrimKind::Inbox { elem, cap }) => {
                                let vexpr = if let Some(first) = first_positional_arg(args) {
                                    let raw = self.lower_expr(first, Some(&elem))?;
                                    coerce_expr_if_needed(raw, &elem)
                                } else {
                                    MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                                };
                                let vexpr = consume_moved_lease_be(vexpr, self);
                                let val_var = self.alloc_temp("inbox_tx", *elem.clone());
                                self.cur_stmts.push(MirStmt::Assign {
                                    target: MirPlace::Var(val_var),
                                    op: MirAssignOp::Assign,
                                    value: vexpr,
                                });
                                if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("inbox_tx_deadline", texpr)?;
                                    let rty = MirType::Result {
                                        ok: Box::new(MirType::Void),
                                        err: None,
                                    };
                                    let dst_res = self.alloc_temp("inbox_tx_res", rty.clone());
                                    let release_lease_on_timeout =
                                        matches!(elem.as_ref(), MirType::Lease(_))
                                            .then(|| (val_var, *elem.clone()));
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::InboxSend {
                                            target,
                                            elem_ty: *elem,
                                            cap,
                                            val_var,
                                            dst_res: Some(dst_res),
                                        },
                                        Some(MirWaitTimer {
                                            deadline_var: dl_var,
                                            timeout_bb: resume_bb,
                                            timeout_err_res: Some(dst_res),
                                            release_lease_on_timeout,
                                        }),
                                        poll_bb,
                                        resume_bb,
                                    );
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                                } else {
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::InboxSend {
                                            target,
                                            elem_ty: *elem,
                                            cap,
                                            val_var,
                                            dst_res: None,
                                        },
                                        None,
                                        poll_bb,
                                        resume_bb,
                                    );
                                    return Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void));
                                }
                            }
                            ("recv", MirPrimKind::Inbox { elem, cap }) => {
                                self.emit_end_inbox_turn();
                                if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("inbox_deadline", texpr)?;
                                    let rty = MirType::Result {
                                        ok: elem.clone(),
                                        err: None,
                                    };
                                    let dst_res = self.alloc_temp("inbox_res", rty.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::InboxRecv {
                                            target,
                                            elem_ty: *elem,
                                            cap,
                                            dst_val: None,
                                            dst_res: Some(dst_res),
                                        },
                                        Some(MirWaitTimer {
                                            deadline_var: dl_var,
                                            timeout_bb: resume_bb,
                                            timeout_err_res: Some(dst_res),
                                            release_lease_on_timeout: None,
                                        }),
                                        poll_bb,
                                        resume_bb,
                                    );
                                    self.may_hold_boost = true;
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                                } else {
                                    let dst_val = self.alloc_temp("inbox_val", *elem.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::InboxRecv {
                                            target,
                                            elem_ty: *elem.clone(),
                                            cap,
                                            dst_val: Some(dst_val),
                                            dst_res: None,
                                        },
                                        None,
                                        poll_bb,
                                        resume_bb,
                                    );
                                    self.may_hold_boost = true;
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_val), *elem));
                                }
                            }
                            ("wait_read", MirPrimKind::ByteRing { cap }) => {
                                self.emit_end_inbox_turn();
                                let slice_ty = MirType::Slice(Box::new(MirType::U8));
                                if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("ring_deadline", texpr)?;
                                    let rty = MirType::Result {
                                        ok: Box::new(slice_ty),
                                        err: None,
                                    };
                                    let dst_res = self.alloc_temp("ring_res", rty.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::ByteRingWaitRead {
                                            target,
                                            cap,
                                            dst_val: None,
                                            dst_res: Some(dst_res),
                                        },
                                        Some(MirWaitTimer {
                                            deadline_var: dl_var,
                                            timeout_bb: resume_bb,
                                            timeout_err_res: Some(dst_res),
                                            release_lease_on_timeout: None,
                                        }),
                                        poll_bb,
                                        resume_bb,
                                    );
                                    self.may_hold_boost = true;
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                                } else {
                                    let dst_val = self.alloc_temp("ring_chunk", slice_ty.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::ByteRingWaitRead {
                                            target,
                                            cap,
                                            dst_val: Some(dst_val),
                                            dst_res: None,
                                        },
                                        None,
                                        poll_bb,
                                        resume_bb,
                                    );
                                    self.may_hold_boost = true;
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_val), slice_ty));
                                }
                            }
                            ("wait_write", MirPrimKind::ByteRing { cap }) => {
                                self.emit_end_inbox_turn();
                                if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("ring_deadline", texpr)?;
                                    let rty = MirType::Result {
                                        ok: Box::new(MirType::Usize),
                                        err: None,
                                    };
                                    let dst_res = self.alloc_temp("ring_w_res", rty.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::ByteRingWaitWrite {
                                            target,
                                            cap,
                                            dst_val: None,
                                            dst_res: Some(dst_res),
                                        },
                                        Some(MirWaitTimer {
                                            deadline_var: dl_var,
                                            timeout_bb: resume_bb,
                                            timeout_err_res: Some(dst_res),
                                            release_lease_on_timeout: None,
                                        }),
                                        poll_bb,
                                        resume_bb,
                                    );
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                                } else {
                                    let dst_val = self.alloc_temp("ring_space", MirType::Usize);
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::ByteRingWaitWrite {
                                            target,
                                            cap,
                                            dst_val: Some(dst_val),
                                            dst_res: None,
                                        },
                                        None,
                                        poll_bb,
                                        resume_bb,
                                    );
                                    return Ok(MirExpr::new(
                                        MirExprKind::Var(dst_val),
                                        MirType::Usize,
                                    ));
                                }
                            }
                            ("wait", MirPrimKind::Signal) => {
                                self.emit_end_inbox_turn();
                                let timer = if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("sig_deadline", texpr)?;
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
                                let wait_timer = timer.map(|dl_var| MirWaitTimer {
                                    deadline_var: dl_var,
                                    timeout_bb: resume_bb,
                                    timeout_err_res: Some(dst_res),
                                    release_lease_on_timeout: None,
                                });
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::SignalWait {
                                        target,
                                        dst_res: Some(dst_res),
                                    },
                                    wait_timer,
                                    poll_bb,
                                    resume_bb,
                                );
                                self.may_hold_boost = true;
                                return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                            }
                            ("commit_wait", MirPrimKind::WaitQueue) => {
                                self.emit_end_inbox_turn();
                                let timer = if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("wq_deadline", texpr)?;
                                    Some(dl_var)
                                } else {
                                    None
                                };
                                let rty = MirType::Result {
                                    ok: Box::new(MirType::Void),
                                    err: None,
                                };
                                let dst_res = self.alloc_temp("wq_res", rty.clone());
                                let poll_bb = self.alloc_bb();
                                let resume_bb = self.alloc_bb();
                                let wait_timer = timer.map(|dl_var| MirWaitTimer {
                                    deadline_var: dl_var,
                                    timeout_bb: resume_bb,
                                    timeout_err_res: Some(dst_res),
                                    release_lease_on_timeout: None,
                                });
                                self.emit_single_arm_wait(
                                    MirWaitArmOp::WaitQueueCommit {
                                        target,
                                        dst_res: Some(dst_res),
                                    },
                                    wait_timer,
                                    poll_bb,
                                    resume_bb,
                                );
                                self.may_hold_boost = true;
                                return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                            }
                            ("wait_change", MirPrimKind::Watch { elem }) => {
                                self.emit_end_inbox_turn();
                                let (last_gen_var, seeded_var) =
                                    self.prepare_watch_wait_change(&target, &elem);
                                if let Some(texpr) = find_named_arg(args, "timeout") {
                                    let dl_var =
                                        self.emit_wait_deadline_var("watch_deadline", texpr)?;
                                    let rty = MirType::Result {
                                        ok: elem.clone(),
                                        err: None,
                                    };
                                    let dst_res = self.alloc_temp("watch_res", rty.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::WatchWaitChange {
                                            target,
                                            elem_ty: *elem,
                                            last_gen_var,
                                            seeded_var,
                                            dst_val: None,
                                            dst_res: Some(dst_res),
                                        },
                                        Some(MirWaitTimer {
                                            deadline_var: dl_var,
                                            timeout_bb: resume_bb,
                                            timeout_err_res: Some(dst_res),
                                            release_lease_on_timeout: None,
                                        }),
                                        poll_bb,
                                        resume_bb,
                                    );
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_res), rty));
                                } else {
                                    let dst_val = self.alloc_temp("watch_val", *elem.clone());
                                    let poll_bb = self.alloc_bb();
                                    let resume_bb = self.alloc_bb();
                                    self.emit_single_arm_wait(
                                        MirWaitArmOp::WatchWaitChange {
                                            target,
                                            elem_ty: *elem.clone(),
                                            last_gen_var,
                                            seeded_var,
                                            dst_val: Some(dst_val),
                                            dst_res: None,
                                        },
                                        None,
                                        poll_bb,
                                        resume_bb,
                                    );
                                    return Ok(MirExpr::new(MirExprKind::Var(dst_val), *elem));
                                }
                            }
                            _ => {}
                        },
                    }
                }
                self.lower_non_yielding_expr(expr, expected)
            }
            _ => self.lower_non_yielding_expr(expr, expected),
        }
    }

    pub(super) fn lower_into_result_var(
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

    pub(super) fn lower_non_yielding_expr(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError> {
        lower_common_expr(expr, expected, self)
    }

    pub(super) fn coerce_arg_for_param(&mut self, arg: MirExpr, param_ty: &MirType) -> MirExpr {
        coerce_call_arg_expr(arg, param_ty)
    }

    pub(super) fn lower_place(
        &mut self,
        expr: &Expr,
    ) -> Result<(MirPlace, MirType), MirLowerError> {
        lower_place_common(expr, self)
    }
}
