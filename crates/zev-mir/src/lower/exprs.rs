use super::*;

pub(super) trait LowerBackend {
    fn syms(&self) -> &ModuleSymbols;
    fn vars(&self) -> &[MirVar];
    fn scopes(&self) -> &VarScopes;
    fn in_noyield(&self) -> bool;
    fn alloc_temp_be(&mut self, prefix: &str, ty: MirType) -> VarId;
    fn emit_stmt_be(&mut self, stmt: MirStmt);
    fn invalidate_boot_const(&mut self, _vid: VarId) {}
    fn mark_may_hold_boost_be(&mut self) {}
    fn lower_sub(
        &mut self,
        expr: &Expr,
        expected: Option<&MirType>,
    ) -> Result<MirExpr, MirLowerError>;
    fn lower_place_be(&mut self, expr: &Expr) -> Result<(MirPlace, MirType), MirLowerError>;
}

pub(super) struct DirectBackend<'b, 'a> {
    pub(super) ctx: &'b mut DirectLowerCtx<'a>,
    pub(super) stmts: &'b mut Vec<MirStmt>,
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
    fn mark_may_hold_boost_be(&mut self) {
        self.may_hold_boost = true;
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

pub(super) fn resolve_irq_token_be<B: LowerBackend>(expr: &Expr, be: &B) -> Option<(String, u32)> {
    match &expr.kind {
        ExprKind::Ident(id) if be.scopes().lookup(&id.name).is_none() => {
            be.syms().irqs.get(&id.name).map(|&n| (id.name.clone(), n))
        }
        ExprKind::Field { .. } => be.syms().resolve_irq_field_be(expr, be.scopes()),
        _ => None,
    }
}

pub(super) fn lookup_mmio_reg<'s>(
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

pub(super) fn peek_expr_struct_type<B: LowerBackend>(expr: &Expr, be: &B) -> Option<String> {
    match &expr.kind {
        ExprKind::Ident(id) => {
            be.scopes()
                .lookup(&id.name)
                .and_then(|vid| match &be.vars()[vid.0].ty {
                    MirType::Struct(n) => Some(n.clone()),
                    MirType::Ptr(inner) => match inner.as_ref() {
                        MirType::Struct(n) => Some(n.clone()),
                        _ => None,
                    },
                    _ => None,
                })
        }
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

pub(super) fn peek_expr_mmio_layout<B: LowerBackend>(expr: &Expr, be: &B) -> Option<String> {
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

pub(super) fn lower_mmio_base_be<B: LowerBackend>(
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

pub(super) fn resolve_mmio_lvalue_be<B: LowerBackend>(
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
            let reg =
                lookup_mmio_reg(be.syms(), layout_name, &reg_ident.name, target.span)?.clone();
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
            let reg =
                lookup_mmio_reg(be.syms(), layout_name, &reg_ident.name, inner_field.span)?.clone();
            if reg.array_len.is_none() {
                return Err(MirLowerError {
                    message: format!(
                        "MMIO scalar register `{}` on `{layout_name}` cannot be indexed",
                        reg.name
                    ),
                    span: target.span,
                });
            }
            let raw_idx = be.lower_sub(idx_ast, Some(&MirType::Usize))?;
            let idx = coerce_expr_if_needed(raw_idx, &MirType::Usize);
            Ok(Some((base, reg, Some(idx))))
        }
        _ => Ok(None),
    }
}

pub(super) fn compound_assign_to_binop(op: AssignOp) -> Option<MirBinOp> {
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

pub(super) fn try_lower_mmio_assign_be<B: LowerBackend>(
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
                message: format!(
                    "cannot write to read-only `@ro` MMIO register `{}`",
                    reg.name
                ),
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
    let bin_op = compound_assign_to_binop(op).ok_or_else(|| MirLowerError {
        message: format!("unsupported compound assignment operator in MMIO write: {op:?}"),
        span: value.span,
    })?;
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

pub(super) fn ast_stmts_have_boost_acquire(stmts: &[Stmt]) -> bool {
    ast_stmts_check_boost_ops(stmts, None, false)
}

pub(super) fn ast_stmts_have_inbox_recv(stmts: &[Stmt], boost_fns: &BTreeSet<String>) -> bool {
    ast_stmts_check_boost_ops(stmts, Some(boost_fns), true)
}

pub(super) fn ast_expr_has_inbox_recv(expr: &Expr, boost_fns: &BTreeSet<String>) -> bool {
    ast_expr_checks_boost_ops(expr, Some(boost_fns), true)
}

pub(super) fn ast_stmts_check_boost_ops(
    stmts: &[Stmt],
    boost_fns: Option<&BTreeSet<String>>,
    include_wait_chunk: bool,
) -> bool {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::VarDecl { init: Some(e), .. }
            | StmtKind::Return(Some(e))
            | StmtKind::Expr(e) => {
                if ast_expr_checks_boost_ops(e, boost_fns, include_wait_chunk) {
                    return true;
                }
            }
            StmtKind::Assign { target, value, .. } => {
                if ast_expr_checks_boost_ops(target, boost_fns, include_wait_chunk)
                    || ast_expr_checks_boost_ops(value, boost_fns, include_wait_chunk)
                {
                    return true;
                }
            }
            StmtKind::Error { reason, partial } => {
                if reason
                    .as_ref()
                    .is_some_and(|r| ast_expr_checks_boost_ops(r, boost_fns, include_wait_chunk))
                    || partial.as_ref().is_some_and(|p| {
                        ast_expr_checks_boost_ops(p, boost_fns, include_wait_chunk)
                    })
                {
                    return true;
                }
            }
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                if ast_expr_checks_boost_ops(cond, boost_fns, include_wait_chunk)
                    || ast_stmts_check_boost_ops(&then_block.stmts, boost_fns, include_wait_chunk)
                    || else_block.as_ref().is_some_and(|eb| {
                        ast_stmts_check_boost_ops(&eb.stmts, boost_fns, include_wait_chunk)
                    })
                {
                    return true;
                }
            }
            StmtKind::Switch { scrutinee, arms } => {
                if ast_expr_checks_boost_ops(scrutinee, boost_fns, include_wait_chunk)
                    || arms.iter().any(|arm| {
                        ast_stmts_check_boost_ops(&arm.body.stmts, boost_fns, include_wait_chunk)
                    })
                {
                    return true;
                }
            }
            StmtKind::Wait { arms, timer } => {
                if arms.iter().any(|arm| {
                    ast_expr_checks_boost_ops(&arm.call, boost_fns, include_wait_chunk)
                        || ast_stmts_check_boost_ops(&arm.body.stmts, boost_fns, include_wait_chunk)
                }) || timer.as_ref().is_some_and(|t| {
                    ast_expr_checks_boost_ops(&t.expr, boost_fns, include_wait_chunk)
                        || ast_stmts_check_boost_ops(&t.body.stmts, boost_fns, include_wait_chunk)
                }) {
                    return true;
                }
            }
            StmtKind::While { cond, body } => {
                if ast_expr_checks_boost_ops(cond, boost_fns, include_wait_chunk)
                    || ast_stmts_check_boost_ops(&body.stmts, boost_fns, include_wait_chunk)
                {
                    return true;
                }
            }
            StmtKind::Loop { body }
            | StmtKind::NoYield { body }
            | StmtKind::Critical { body }
            | StmtKind::Block(body) => {
                if ast_stmts_check_boost_ops(&body.stmts, boost_fns, include_wait_chunk) {
                    return true;
                }
            }
            StmtKind::LoopEvery { period, body } => {
                if ast_expr_checks_boost_ops(period, boost_fns, include_wait_chunk)
                    || ast_stmts_check_boost_ops(&body.stmts, boost_fns, include_wait_chunk)
                {
                    return true;
                }
            }
            StmtKind::ForRange {
                start, end, body, ..
            } => {
                if ast_expr_checks_boost_ops(start, boost_fns, include_wait_chunk)
                    || ast_expr_checks_boost_ops(end, boost_fns, include_wait_chunk)
                    || ast_stmts_check_boost_ops(&body.stmts, boost_fns, include_wait_chunk)
                {
                    return true;
                }
            }
            StmtKind::ForEach { iter, body, .. } => {
                if ast_expr_checks_boost_ops(iter, boost_fns, include_wait_chunk)
                    || ast_stmts_check_boost_ops(&body.stmts, boost_fns, include_wait_chunk)
                {
                    return true;
                }
            }
            StmtKind::Defer(inner) => {
                if ast_stmts_check_boost_ops(
                    std::slice::from_ref(inner.as_ref()),
                    boost_fns,
                    include_wait_chunk,
                ) {
                    return true;
                }
            }
            StmtKind::Spawn(sp) => {
                if sp
                    .args
                    .iter()
                    .any(|a| ast_expr_checks_boost_ops(&a.value, boost_fns, include_wait_chunk))
                {
                    return true;
                }
            }
            StmtKind::VarDecl { init: None, .. }
            | StmtKind::StateVar { .. }
            | StmtKind::Return(None)
            | StmtKind::Restart
            | StmtKind::Continue
            | StmtKind::Break
            | StmtKind::Checkpoint => {}
        }
    }
    false
}

pub(super) fn ast_expr_checks_boost_ops(
    expr: &Expr,
    boost_fns: Option<&BTreeSet<String>>,
    include_wait_chunk: bool,
) -> bool {
    match &expr.kind {
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => {
            let m = method.name.as_str();
            if matches!(
                m,
                "recv"
                    | "wait"
                    | "alloc"
                    | "wait_read"
                    | "try_recv"
                    | "try_wait"
                    | "pop"
                    | "try_pop_byte"
            ) || (include_wait_chunk && m == "wait_chunk")
            {
                return true;
            }
            ast_expr_checks_boost_ops(receiver, boost_fns, include_wait_chunk)
                || args
                    .iter()
                    .any(|a| ast_expr_checks_boost_ops(&a.value, boost_fns, include_wait_chunk))
        }
        ExprKind::Call { callee, args, .. } => {
            if let (Some(fns), ExprKind::Ident(id)) = (boost_fns, &callee.kind) {
                if fns.contains(&id.name) {
                    return true;
                }
            }
            ast_expr_checks_boost_ops(callee, boost_fns, include_wait_chunk)
                || args
                    .iter()
                    .any(|a| ast_expr_checks_boost_ops(&a.value, boost_fns, include_wait_chunk))
        }
        ExprKind::Or {
            expr: l,
            fallback: r,
        }
        | ExprKind::Binary { lhs: l, rhs: r, .. }
        | ExprKind::Index {
            target: l,
            index: r,
        } => {
            ast_expr_checks_boost_ops(l, boost_fns, include_wait_chunk)
                || ast_expr_checks_boost_ops(r, boost_fns, include_wait_chunk)
        }
        ExprKind::Catch {
            expr: inner,
            clause,
        } => {
            ast_expr_checks_boost_ops(inner, boost_fns, include_wait_chunk)
                || ast_stmts_check_boost_ops(&clause.body.stmts, boost_fns, include_wait_chunk)
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::Field { target: inner, .. } => {
            ast_expr_checks_boost_ops(inner, boost_fns, include_wait_chunk)
        }
        ExprKind::Slice { target, start, end } => {
            ast_expr_checks_boost_ops(target, boost_fns, include_wait_chunk)
                || start
                    .as_ref()
                    .is_some_and(|s| ast_expr_checks_boost_ops(s, boost_fns, include_wait_chunk))
                || end
                    .as_ref()
                    .is_some_and(|e| ast_expr_checks_boost_ops(e, boost_fns, include_wait_chunk))
        }
        ExprKind::StructLit { fields, .. } => fields
            .iter()
            .any(|f| ast_expr_checks_boost_ops(&f.value, boost_fns, include_wait_chunk)),
        ExprKind::ArrayLit(elems) => elems
            .iter()
            .any(|e| ast_expr_checks_boost_ops(e, boost_fns, include_wait_chunk)),
        ExprKind::FormatString(parts) => parts.iter().any(|part| match part {
            FormatPart::Hole { expr: he, .. } => {
                ast_expr_checks_boost_ops(he, boost_fns, include_wait_chunk)
            }
            FormatPart::Literal(_) => false,
        }),
        ExprKind::Literal(_) | ExprKind::Ident(_) | ExprKind::Priority(_) => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure / Leaf / Non-blocking Shared Lowering Routines
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn flatten_or_operands<'e>(expr: &'e Expr, out: &mut Vec<&'e Expr>) {
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

pub(super) fn extract_result_types(ty: &MirType) -> (MirType, MirType) {
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

pub(super) fn unwrap_ok_result_expr(expr: MirExpr) -> MirExpr {
    if let (MirType::Result { ok, .. }, MirExprKind::Var(vid)) = (&expr.ty, &expr.kind) {
        return MirExpr::new(MirExprKind::ResultVal(*vid), *ok.clone());
    }
    expr
}

pub(super) fn extract_subsystem_place_chain(
    place: &MirPlace,
) -> Option<(String, Vec<(String, String)>)> {
    let mut rev = Vec::new();
    let mut cur = place;
    loop {
        match cur {
            MirPlace::Field {
                base,
                field,
                field_ty,
                ..
            } => {
                rev.push((field.clone(), field_ty.display_name()));
                cur = base;
            }
            MirPlace::Deref(inner, _) => {
                if let MirExprKind::SubsystemAddr(root) = &inner.kind {
                    rev.reverse();
                    return Some((root.clone(), rev));
                }
                return None;
            }
            _ => return None,
        }
    }
}

pub(super) fn consume_moved_lease_be<B: LowerBackend>(expr: MirExpr, be: &mut B) -> MirExpr {
    let MirType::Lease(elem_ty) = &expr.ty else {
        return expr;
    };
    let elem_ty = elem_ty.as_ref().clone();
    let ty = expr.ty.clone();
    let target = match &expr.kind {
        MirExprKind::Var(vid) => MirPlace::Var(*vid),
        MirExprKind::FieldAccess {
            base,
            struct_name,
            packed,
            is_ptr: false,
            field,
            raw_field_ty,
        } => {
            if let MirExprKind::Var(bvid) = &base.kind {
                MirPlace::Field {
                    base: Box::new(MirPlace::Var(*bvid)),
                    struct_name: struct_name.clone(),
                    packed: *packed,
                    field: field.clone(),
                    field_ty: raw_field_ty.clone(),
                }
            } else {
                return expr;
            }
        }
        _ => return expr,
    };
    let tmp = be.alloc_temp_be("lease_mv", ty.clone());
    be.emit_stmt_be(MirStmt::Assign {
        target: MirPlace::Var(tmp),
        op: MirAssignOp::Assign,
        value: expr,
    });
    be.emit_stmt_be(MirStmt::PrimOp(MirPrimNonBlockingOp::LeaseNullify {
        target,
        elem_ty,
    }));
    MirExpr::new(MirExprKind::Var(tmp), ty)
}

pub(super) fn is_rodata_const_mir_expr(expr: &MirExpr) -> bool {
    match &expr.kind {
        MirExprKind::ConstRef(_) => true,
        MirExprKind::FieldAccess {
            base,
            is_ptr: false,
            ..
        }
        | MirExprKind::Index {
            base,
            is_slice: false,
            ..
        } => is_rodata_const_mir_expr(base),
        _ => false,
    }
}

pub(super) fn coerce_expr_if_needed(expr: MirExpr, target_ty: &MirType) -> MirExpr {
    if &expr.ty == target_ty || target_ty.is_void() {
        return expr;
    }
    let host_target = target_ty.host_value_type();
    if is_scalar_int_mir_type(&expr.ty)
        && is_scalar_int_mir_type(&host_target)
        && can_widen_mir_int(&expr.ty.host_value_type(), &host_target)
        && expr.ty.host_value_type() != host_target
    {
        return widen_arith_mir_expr(expr, &host_target);
    }
    if let (MirType::Result { ok, .. }, MirExprKind::Var(vid)) = (&expr.ty, &expr.kind) {
        if **ok == *target_ty || ok.host_value_type() == host_target {
            return MirExpr::new(MirExprKind::ResultVal(*vid), ok.as_ref().clone());
        }
    }
    match (&expr.ty, target_ty) {
        (MirType::Ptr(inner), MirType::PrimView(MirPrimViewKind::ByteRing))
            if matches!(**inner, MirType::Primitive(MirPrimKind::ByteRing { .. })) =>
        {
            let cap = match **inner {
                MirType::Primitive(MirPrimKind::ByteRing { cap }) => cap,
                _ => 0,
            };
            return MirExpr::new(
                MirExprKind::MakePrimView {
                    ptr: Box::new(expr),
                    cap,
                    kind: MirPrimViewKind::ByteRing,
                },
                target_ty.clone(),
            );
        }
        (MirType::Ptr(inner), MirType::PrimView(MirPrimViewKind::Inbox { elem }))
            if matches!(**inner, MirType::Primitive(MirPrimKind::Inbox { .. })) =>
        {
            let cap = match &**inner {
                MirType::Primitive(MirPrimKind::Inbox { cap, .. }) => *cap,
                _ => 0,
            };
            return MirExpr::new(
                MirExprKind::MakePrimView {
                    ptr: Box::new(expr),
                    cap,
                    kind: MirPrimViewKind::Inbox { elem: elem.clone() },
                },
                target_ty.clone(),
            );
        }
        (MirType::Array { elem, len }, MirType::Slice(_)) if is_rodata_const_mir_expr(&expr) => {
            let e_ty = *elem.clone();
            let n = *len;
            return MirExpr::new(
                MirExprKind::SubSlice {
                    base: Box::new(expr),
                    start: Box::new(MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)),
                    end: Box::new(MirExpr::new(MirExprKind::IntLit(n as i128), MirType::Usize)),
                    elem_ty: e_ty,
                    base_is_array: true,
                },
                target_ty.clone(),
            );
        }
        _ => {}
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
        MirExprKind::StringLit(s)
            if expr.ty == MirType::Str
                && matches!(&host_target, MirType::Slice(elem) if **elem == MirType::U8) =>
        {
            MirExpr::new(
                MirExprKind::StringLit(s),
                MirType::Slice(Box::new(MirType::U8)),
            )
        }
        other_kind => MirExpr::new(other_kind, expr.ty),
    }
}

pub(super) fn coerce_call_arg_expr(arg: MirExpr, param_ty: &MirType) -> MirExpr {
    if let (MirType::Array { elem, len }, MirType::Slice(_), MirExprKind::Var(vid)) =
        (&arg.ty, param_ty, &arg.kind)
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
    if let (MirType::Array { elem, len }, MirType::Slice(_)) = (&arg.ty, param_ty) {
        if is_rodata_const_mir_expr(&arg) {
            let e_ty = *elem.clone();
            let n = *len;
            return MirExpr::new(
                MirExprKind::SubSlice {
                    base: Box::new(arg),
                    start: Box::new(MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)),
                    end: Box::new(MirExpr::new(MirExprKind::IntLit(n as i128), MirType::Usize)),
                    elem_ty: e_ty,
                    base_is_array: true,
                },
                param_ty.clone(),
            );
        }
    }
    coerce_expr_if_needed(arg, param_ty)
}

fn is_pure_untyped_int_ast(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Literal(Literal::Int { suffix: None, .. }) => true,
        ExprKind::Unary {
            op: UnaryOp::Neg | UnaryOp::BitNot,
            expr: inner,
        } => is_pure_untyped_int_ast(inner),
        ExprKind::Binary { op, lhs, rhs } => {
            if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
                is_pure_untyped_int_ast(lhs) && is_pure_untyped_int_ast(rhs)
            } else {
                !matches!(
                    op,
                    BinaryOp::Eq
                        | BinaryOp::Ne
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge
                        | BinaryOp::And
                        | BinaryOp::Or
                ) && is_pure_untyped_int_ast(lhs)
                    && is_pure_untyped_int_ast(rhs)
            }
        }
        _ => false,
    }
}

fn is_pure_untyped_float_ast(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Literal(Literal::Float { suffix: None, .. }) => true,
        ExprKind::Unary {
            op: UnaryOp::Neg,
            expr: inner,
        } => is_pure_untyped_float_ast(inner),
        ExprKind::Binary { op, lhs, rhs } => {
            !matches!(
                op,
                BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::And
                    | BinaryOp::Or
            ) && is_pure_untyped_float_ast(lhs)
                && is_pure_untyped_float_ast(rhs)
        }
        _ => false,
    }
}

fn unify_array_mir_types(lhs: &MirType, rhs: &MirType) -> MirType {
    if is_scalar_int_mir_type(lhs) && is_scalar_int_mir_type(rhs) {
        unify_mir_int_types(lhs, rhs)
    } else if matches!(
        (lhs.host_value_type(), rhs.host_value_type()),
        (MirType::F32, MirType::F64) | (MirType::F64, MirType::F32)
    ) {
        MirType::F64
    } else {
        lhs.host_value_type()
    }
}

pub(super) fn lower_leaf_expr(
    expr: &Expr,
    expected: Option<&MirType>,
    syms: &ModuleSymbols,
    vars: &[MirVar],
    scopes: &VarScopes,
) -> Result<MirExpr, MirLowerError> {
    match &expr.kind {
        ExprKind::Literal(lit) => match lit {
            Literal::Int { value, suffix } => {
                let ty = if let Some(p) = suffix {
                    convert_prim_type(*p).host_value_type()
                } else {
                    expected
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
                        .unwrap_or(if *value > u32::MAX as u64 {
                            MirType::U64
                        } else {
                            MirType::U32
                        })
                };
                Ok(MirExpr::new(MirExprKind::IntLit(*value as i128), ty))
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
            Literal::String(s) => {
                let ty = match expected.map(|t| t.host_value_type()) {
                    Some(MirType::Slice(elem)) if *elem == MirType::U8 => {
                        MirType::Slice(Box::new(MirType::U8))
                    }
                    _ => MirType::Str,
                };
                Ok(MirExpr::new(MirExprKind::StringLit(s.clone()), ty))
            }
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
            if let Some((sname, _)) = syms.subsystems.get(&id.name) {
                return Ok(MirExpr::new(
                    MirExprKind::SubsystemAddr(id.name.clone()),
                    MirType::Ptr(Box::new(MirType::Struct(sname.clone()))),
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

pub(super) fn lower_common_expr<B: LowerBackend>(
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
                    if let Some((sname, _)) = be.syms().subsystems.get(&id.name) {
                        return Ok(MirExpr::new(
                            MirExprKind::SubsystemAddr(id.name.clone()),
                            MirType::Ptr(Box::new(MirType::Struct(sname.clone()))),
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
            } else if let Some(inst_name) =
                be.syms().resolve_mmio_instance_field_be(inner, be.scopes())
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
            if *op == UnaryOp::Neg {
                if let ExprKind::Literal(Literal::Int {
                    value,
                    suffix: None,
                }) = &inner.kind
                {
                    let neg_val = -i128::from(*value);
                    let ty = expected
                        .map(|t| t.host_value_type())
                        .filter(is_scalar_int_mir_type)
                        .unwrap_or(if neg_val >= i32::MIN as i128 {
                            MirType::I32
                        } else {
                            MirType::I64
                        });
                    return Ok(MirExpr::new(MirExprKind::IntLit(neg_val), ty));
                }
            }
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
            let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
            let op_expected = if is_cmp_or_logic { None } else { expected };
            let (l, r) = if !is_shift && ast_is_untyped_int_lit(lhs) && !ast_is_untyped_int_lit(rhs)
            {
                let r = be.lower_sub(rhs, op_expected)?;
                let l = be.lower_sub(lhs, Some(&r.ty.host_value_type()))?;
                (l, r)
            } else {
                let l = be.lower_sub(lhs, op_expected)?;
                let rhs_exp = if is_shift {
                    None
                } else {
                    Some(l.ty.host_value_type())
                };
                let r = be.lower_sub(rhs, rhs_exp.as_ref())?;
                (l, r)
            };
            Ok(finish_binary(*op, l, r))
        }
        ExprKind::Cast { expr: inner, ty } => {
            let target_ty = be.syms().resolve_type(ty);
            let sub = be.lower_sub(inner, None)?;
            if matches!(&target_ty, MirType::Slice(elem) if **elem == MirType::U8)
                && matches!(&sub.kind, MirExprKind::StringLit(_))
            {
                return Ok(MirExpr::new(sub.kind, target_ty));
            }
            Ok(MirExpr::new(
                MirExprKind::Cast {
                    expr: Box::new(sub),
                    target_ty: target_ty.clone(),
                },
                target_ty,
            ))
        }
        ExprKind::Field { target, field } => {
            if let Some(inst_name) = be.syms().resolve_mmio_instance_field_be(expr, be.scopes()) {
                let layout = be.syms().mmio_instances[&inst_name].layout.clone();
                return Ok(MirExpr::new(
                    MirExprKind::MmioAddr(inst_name),
                    MirType::MmioPtr(layout),
                ));
            }
            if let ExprKind::Ident(id) = &target.kind {
                if be.scopes().lookup(&id.name).is_none() && be.syms().enums.contains_key(&id.name)
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
            let raw_idx = be.lower_sub(index, Some(&MirType::Usize))?;
            let idx = coerce_expr_if_needed(raw_idx, &MirType::Usize);
            finish_index(base, idx, expr.span)
        }
        ExprKind::Slice { target, start, end } => {
            let s = if let Some(se) = start {
                let raw_s = be.lower_sub(se, Some(&MirType::Usize))?;
                coerce_expr_if_needed(raw_s, &MirType::Usize)
            } else {
                MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)
            };
            if let Ok((place, MirType::Array { elem, len })) = be.lower_place_be(target) {
                let e = if let Some(ee) = end {
                    let raw_e = be.lower_sub(ee, Some(&MirType::Usize))?;
                    coerce_expr_if_needed(raw_e, &MirType::Usize)
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
            let (elem_ty, array_len) = match &base.ty {
                MirType::Slice(inner) => (*inner.clone(), None),
                MirType::Array { elem, len } => (*elem.clone(), Some(*len)),
                _ => (MirType::U8, None),
            };
            let base_is_array = array_len.is_some();
            let e = if let Some(ee) = end {
                let raw_e = be.lower_sub(ee, Some(&MirType::Usize))?;
                coerce_expr_if_needed(raw_e, &MirType::Usize)
            } else if let Some(len) = array_len {
                MirExpr::new(MirExprKind::IntLit(len as i128), MirType::Usize)
            } else {
                MirExpr::new(
                    MirExprKind::SliceLen(Box::new(base.clone())),
                    MirType::Usize,
                )
            };
            Ok(MirExpr::new(
                MirExprKind::SubSlice {
                    base: Box::new(base),
                    start: Box::new(s),
                    end: Box::new(e),
                    elem_ty: elem_ty.clone(),
                    base_is_array,
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
            let packed = st.as_ref().is_some_and(|s| s.packed);
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
            let (elem_ty, melems) =
                if let Some(MirType::Array { elem, .. } | MirType::Slice(elem)) = expected {
                    let elem_ty = *elem.clone();
                    let mut out = Vec::with_capacity(elems.len());
                    for el in elems {
                        let me = be.lower_sub(el, Some(&elem_ty))?;
                        out.push(coerce_expr_if_needed(me, &elem_ty));
                    }
                    (elem_ty, out)
                } else if elems.is_empty() {
                    (MirType::U8, Vec::new())
                } else {
                    let mut raw_elems = Vec::with_capacity(elems.len());
                    let mut inferred: Option<MirType> = None;
                    for el in elems {
                        let me = be.lower_sub(el, None)?;
                        if !is_pure_untyped_int_ast(el) && !is_pure_untyped_float_ast(el) {
                            let hty = me.ty.host_value_type();
                            inferred = Some(match &inferred {
                                None => hty,
                                Some(cur) => unify_array_mir_types(cur, &hty),
                            });
                        }
                        raw_elems.push(me);
                    }
                    if inferred.is_none() {
                        for (el, me) in elems.iter().zip(raw_elems.iter()) {
                            if is_pure_untyped_float_ast(el) {
                                inferred = Some(MirType::F64);
                                break;
                            }
                            let hty = me.ty.host_value_type();
                            inferred = Some(match &inferred {
                                None => hty,
                                Some(cur) => unify_array_mir_types(cur, &hty),
                            });
                        }
                    }
                    let elem_ty = inferred.unwrap_or(MirType::U8);
                    let mut out = Vec::with_capacity(elems.len());
                    for (el, me) in elems.iter().zip(raw_elems) {
                        if is_pure_untyped_int_ast(el) {
                            let re = be.lower_sub(el, Some(&elem_ty))?;
                            out.push(coerce_expr_if_needed(re, &elem_ty));
                        } else {
                            out.push(coerce_expr_if_needed(me, &elem_ty));
                        }
                    }
                    (elem_ty, out)
                };
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
                match BuiltinFn::from_name(&id.name) {
                    Some(BuiltinFn::FmtBuf) => {
                        return lower_fmt_buf_call_be(args, expr.span, be);
                    }
                    Some(BuiltinFn::FmtRing) => {
                        return lower_fmt_ring_call_be(args, expr.span, be);
                    }
                    Some(BuiltinFn::NowTicks | BuiltinFn::NowMs) => {
                        return Ok(MirExpr::new(MirExprKind::NowTicks, MirType::U32));
                    }
                    Some(b) if b.is_blocking_delay() => {
                        return Err(MirLowerError {
                            message: format!(
                                "`{}()` yields cooperatively and cannot be lowered in a direct non-yielding or boot-helper context",
                                id.name
                            ),
                            span: expr.span,
                        });
                    }
                    Some(BuiltinFn::Checkpoint) if be.in_noyield() => {
                        return Ok(MirExpr::new(MirExprKind::IntLit(0), MirType::Void));
                    }
                    _ => {}
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
                        let coerced = if let Some(pty) = sig.params.get(idx) {
                            coerce_call_arg_expr(me, pty)
                        } else {
                            me
                        };
                        largs.push(consume_moved_lease_be(coerced, be));
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
                    if be.syms().boost_capable_fns.contains(&id.name) {
                        be.mark_may_hold_boost_be();
                    }
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
                if id.name == SYS_NAMESPACE {
                    match SysMethod::from_name(&method.name) {
                        Some(SysMethod::NowMs | SysMethod::NowTicks) => {
                            return Ok(MirExpr::new(MirExprKind::NowTicks, MirType::U32));
                        }
                        Some(SysMethod::CurrentPrio) => {
                            return Ok(MirExpr::new(
                                MirExprKind::SysQuery(MirSysQuery::CurrentPrio),
                                MirType::U32,
                            ));
                        }
                        Some(SysMethod::SleepDepth) => {
                            return Ok(MirExpr::new(
                                MirExprKind::SysQuery(MirSysQuery::SleepDepth),
                                MirType::U32,
                            ));
                        }
                        Some(SysMethod::WfiCount) => {
                            return Ok(MirExpr::new(
                                MirExprKind::SysQuery(MirSysQuery::WfiCount),
                                MirType::U32,
                            ));
                        }
                        Some(SysMethod::RomCallU32) => {
                            let (addr, stack_bound, abi_args) =
                                lower_rom_call_args_be(args, true, expr.span, be)?;
                            return Ok(MirExpr::new(
                                MirExprKind::RomCallU32 {
                                    addr: Box::new(addr),
                                    stack_bound,
                                    args: abi_args,
                                },
                                MirType::U32,
                            ));
                        }
                        _ => {}
                    }
                }
            }
            if let Some(recv_kind) = resolve_prim_receiver_kind_be(receiver, be)? {
                return match recv_kind {
                    LoweredPrimReceiver::Sized(target, pkind) => {
                        finish_prim_read_method(target, pkind, &method.name, args, expr.span, be)
                    }
                    LoweredPrimReceiver::ByteRingView(view) => {
                        finish_byte_ring_view_read_method(view, &method.name, args, expr.span, be)
                    }
                    LoweredPrimReceiver::InboxView(view, elem) => {
                        finish_inbox_view_read_method(view, elem, &method.name, args, expr.span, be)
                    }
                };
            }
            if let Ok(recv_me) = be.lower_sub(receiver, None) {
                if let MirType::Lease(elem) = recv_me.ty.clone() {
                    match method.name.as_str() {
                        "get" => {
                            return Ok(MirExpr::new(
                                MirExprKind::LeaseGet {
                                    lease: Box::new(recv_me),
                                    elem_ty: *elem.clone(),
                                },
                                MirType::Ptr(elem),
                            ));
                        }
                        "addr" => {
                            return Ok(MirExpr::new(
                                MirExprKind::LeaseAddr {
                                    lease: Box::new(recv_me),
                                    elem_ty: *elem,
                                },
                                MirType::Usize,
                            ));
                        }
                        _ => {}
                    }
                }
                let (st_name_opt, is_ptr) = match &recv_me.ty {
                    MirType::Struct(s) => (Some(s.clone()), false),
                    MirType::Ptr(inner) => match inner.as_ref() {
                        MirType::Struct(s) => (Some(s.clone()), true),
                        _ => (None, false),
                    },
                    _ => (None, false),
                };
                if let Some(sname) = st_name_opt {
                    if let Some((fty, is_packed)) = be.syms().structs.get(&sname).and_then(|st| {
                        st.fields
                            .iter()
                            .find(|f| f.name == method.name)
                            .map(|f| (f.ty.clone(), st.packed))
                    }) {
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

pub(super) fn lower_indirect_fnptr_call_be<B: LowerBackend>(
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
        largs.push(consume_moved_lease_be(coerced, be));
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

pub(super) fn finish_unary(op: UnaryOp, sub: MirExpr) -> Result<MirExpr, MirLowerError> {
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
            let ty = sub.ty.host_value_type();
            if let MirExprKind::IntLit(v) = sub.kind {
                return Ok(MirExpr::new(MirExprKind::IntLit(-v), ty));
            }
            Ok(MirExpr::new(
                MirExprKind::Unary {
                    op: MirUnaryOp::Neg,
                    expr: Box::new(sub),
                },
                ty,
            ))
        }
        UnaryOp::AddrOf => unreachable!(),
    }
}

pub(super) fn ast_is_untyped_int_lit(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Literal(Literal::Int { suffix: None, .. })
        | ExprKind::Literal(Literal::Float { suffix: None, .. }) => true,
        ExprKind::Unary {
            op: UnaryOp::Neg,
            expr: inner,
        } => matches!(
            inner.kind,
            ExprKind::Literal(Literal::Int { suffix: None, .. })
                | ExprKind::Literal(Literal::Float { suffix: None, .. })
        ),
        _ => false,
    }
}

pub(super) fn is_scalar_int_mir_type(ty: &MirType) -> bool {
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

pub(super) fn int_lit_fits_mir_type(v: i128, target: &MirType) -> bool {
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

pub(super) fn coerce_or_cast_int(expr: MirExpr, target_ty: &MirType) -> MirExpr {
    let target = target_ty.host_value_type();
    if expr.ty.host_value_type() == target || target.is_void() {
        return expr;
    }
    if let MirExprKind::IntLit(v) = expr.kind {
        if is_scalar_int_mir_type(&target) && int_lit_fits_mir_type(v, &target) {
            return MirExpr::new(MirExprKind::IntLit(v), target);
        }
        return MirExpr::new(
            MirExprKind::Cast {
                expr: Box::new(expr),
                target_ty: target.clone(),
            },
            target,
        );
    }
    MirExpr::new(
        MirExprKind::Cast {
            expr: Box::new(expr),
            target_ty: target.clone(),
        },
        target,
    )
}

pub(super) fn can_widen_mir_int(src: &MirType, dst: &MirType) -> bool {
    use MirType::*;
    matches!(
        (src.host_value_type(), dst.host_value_type()),
        (U8, U8 | U16 | U32 | U64 | Usize | I16 | I32 | I64 | Isize)
            | (U16, U16 | U32 | U64 | Usize | I32 | I64)
            | (U32, U32 | U64 | I64)
            | (U64, U64)
            | (Usize, Usize | U64)
            | (I8, I8 | I16 | I32 | I64 | Isize)
            | (I16, I16 | I32 | I64 | Isize)
            | (I32, I32 | I64)
            | (I64, I64)
            | (Isize, Isize | I64)
    )
}

pub(super) fn unify_mir_int_types(lhs: &MirType, rhs: &MirType) -> MirType {
    if can_widen_mir_int(lhs, rhs) {
        rhs.host_value_type()
    } else {
        lhs.host_value_type()
    }
}

pub(super) fn widen_arith_mir_expr(expr: MirExpr, target_ty: &MirType) -> MirExpr {
    let target = target_ty.host_value_type();
    let cur_host = expr.ty.host_value_type();
    if !is_scalar_int_mir_type(&target)
        || !is_scalar_int_mir_type(&expr.ty)
        || cur_host == target
        || !can_widen_mir_int(&cur_host, &target)
    {
        return expr;
    }
    let orig_ty = expr.ty;
    match expr.kind {
        MirExprKind::IntLit(v) => MirExpr::new(MirExprKind::IntLit(v), target),
        MirExprKind::Unary {
            op: op @ (MirUnaryOp::Neg | MirUnaryOp::BitNot),
            expr: inner,
        } => {
            let widened_inner = widen_arith_mir_expr(*inner, &target);
            MirExpr::new(
                MirExprKind::Unary {
                    op,
                    expr: Box::new(widened_inner),
                },
                target,
            )
        }
        MirExprKind::Binary {
            op:
                op @ (MirBinOp::Add
                | MirBinOp::Sub
                | MirBinOp::Mul
                | MirBinOp::Div
                | MirBinOp::Rem
                | MirBinOp::BitAnd
                | MirBinOp::BitOr
                | MirBinOp::BitXor),
            lhs,
            rhs,
        } => {
            let wl = widen_arith_mir_expr(*lhs, &target);
            let wr = widen_arith_mir_expr(*rhs, &target);
            MirExpr::new(
                MirExprKind::Binary {
                    op,
                    lhs: Box::new(wl),
                    rhs: Box::new(wr),
                },
                target,
            )
        }
        MirExprKind::Binary {
            op: op @ (MirBinOp::Shl | MirBinOp::Shr),
            lhs,
            rhs,
        } => {
            let wl = widen_arith_mir_expr(*lhs, &target);
            MirExpr::new(
                MirExprKind::Binary {
                    op,
                    lhs: Box::new(wl),
                    rhs,
                },
                target,
            )
        }
        other_kind => MirExpr::new(
            MirExprKind::Cast {
                expr: Box::new(MirExpr::new(other_kind, orig_ty)),
                target_ty: target.clone(),
            },
            target,
        ),
    }
}

pub(super) fn finish_binary(op: BinaryOp, mut lhs: MirExpr, mut rhs: MirExpr) -> MirExpr {
    let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
    // Unify float literal precision if one side is `f32` and the other is a `FloatLit`
    if lhs.ty == MirType::F32 && matches!(rhs.kind, MirExprKind::FloatLit(_)) {
        rhs = coerce_expr_if_needed(rhs, &MirType::F32);
    } else if rhs.ty == MirType::F32 && matches!(lhs.kind, MirExprKind::FloatLit(_)) {
        lhs = coerce_expr_if_needed(lhs, &MirType::F32);
    } else if !is_shift && is_scalar_int_mir_type(&lhs.ty) && is_scalar_int_mir_type(&rhs.ty) {
        let common = unify_mir_int_types(&lhs.ty, &rhs.ty);
        lhs = widen_arith_mir_expr(lhs, &common);
        rhs = widen_arith_mir_expr(rhs, &common);
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
    let rhs_coerced = if is_shift {
        rhs
    } else {
        coerce_expr_if_needed(rhs, &lhs.ty.host_value_type())
    };
    MirExpr::new(
        MirExprKind::Binary {
            op: mop,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs_coerced),
        },
        out_ty,
    )
}

pub(super) fn finish_field_access(
    base: MirExpr,
    field: &str,
    syms: &ModuleSymbols,
    span: Span,
) -> Result<MirExpr, MirLowerError> {
    let base = unwrap_ok_result_expr(base);
    if field == "len" {
        match &base.ty {
            MirType::Slice(_) => {
                return Ok(MirExpr::new(
                    MirExprKind::SliceLen(Box::new(base)),
                    MirType::Usize,
                ));
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
                    message: format!(
                        "cannot access field `{field}` on pointer `{}`",
                        base.ty.display_name()
                    ),
                    span,
                });
            }
        },
        _ => {
            return Err(MirLowerError {
                message: format!(
                    "cannot access field `{field}` on `{}`",
                    base.ty.display_name()
                ),
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

pub(super) fn finish_index(
    base: MirExpr,
    idx: MirExpr,
    span: Span,
) -> Result<MirExpr, MirLowerError> {
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

pub(super) enum LoweredPrimReceiver {
    Sized(MirPrimTarget, MirPrimKind),
    ByteRingView(MirExpr),
    InboxView(MirExpr, MirType),
}

pub(super) fn resolve_prim_receiver_kind_be<B: LowerBackend>(
    receiver: &Expr,
    be: &mut B,
) -> Result<Option<LoweredPrimReceiver>, MirLowerError> {
    if let ExprKind::Ident(id) = &receiver.kind {
        if let Some(pinfo) = be.syms().primitives.get(&id.name) {
            return Ok(Some(LoweredPrimReceiver::Sized(
                MirPrimTarget::Global(id.name.clone()),
                pinfo.kind.clone(),
            )));
        }
        if let Some(vid) = be.scopes().lookup(&id.name) {
            let vty = be.vars()[vid.0].ty.clone();
            let vexpr = MirExpr::new(MirExprKind::Var(vid), vty.clone());
            match vty {
                MirType::Ptr(inner) => {
                    if let MirType::Primitive(pk) = *inner {
                        return Ok(Some(LoweredPrimReceiver::Sized(
                            MirPrimTarget::PtrExpr(vexpr),
                            pk,
                        )));
                    }
                }
                MirType::PrimView(MirPrimViewKind::ByteRing) => {
                    return Ok(Some(LoweredPrimReceiver::ByteRingView(vexpr)));
                }
                MirType::PrimView(MirPrimViewKind::Inbox { elem }) => {
                    return Ok(Some(LoweredPrimReceiver::InboxView(vexpr, *elem)));
                }
                _ => {}
            }
        }
        return Ok(None);
    }
    if let ExprKind::Field { .. }
    | ExprKind::Unary {
        op: UnaryOp::Deref, ..
    }
    | ExprKind::Index { .. }
    | ExprKind::Call { .. }
    | ExprKind::MethodCall { .. } = &receiver.kind
    {
        if let Ok(rv) = be.lower_sub(receiver, None) {
            match &rv.ty {
                MirType::Ptr(inner) => {
                    if let MirType::Primitive(pk) = inner.as_ref() {
                        let pk_c = pk.clone();
                        return Ok(Some(LoweredPrimReceiver::Sized(
                            MirPrimTarget::PtrExpr(rv),
                            pk_c,
                        )));
                    }
                }
                MirType::PrimView(MirPrimViewKind::ByteRing) => {
                    return Ok(Some(LoweredPrimReceiver::ByteRingView(rv)));
                }
                MirType::PrimView(MirPrimViewKind::Inbox { elem }) => {
                    let elem_c = *elem.clone();
                    return Ok(Some(LoweredPrimReceiver::InboxView(rv, elem_c)));
                }
                MirType::Primitive(pk) => {
                    let pk_c = pk.clone();
                    if let Ok((place, pty)) = be.lower_place_be(receiver) {
                        let ptr = MirExpr::new(
                            MirExprKind::AddrOfPlace(Box::new(place)),
                            MirType::Ptr(Box::new(pty)),
                        );
                        return Ok(Some(LoweredPrimReceiver::Sized(
                            MirPrimTarget::PtrExpr(ptr),
                            pk_c,
                        )));
                    }
                }
                _ => {}
            }
        }
    }
    Ok(None)
}

pub(super) fn is_checkpoint_call_expr(expr: &Expr) -> bool {
    matches!(
        &expr.kind,
        ExprKind::Call { callee, args, .. }
            if args.is_empty() && matches!(&callee.kind, ExprKind::Ident(id) if id.name == "checkpoint")
    )
}

pub(super) fn extract_sys_exit_arg(expr: &Expr) -> Option<Option<&Expr>> {
    match &expr.kind {
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } if matches!(&receiver.kind, ExprKind::Ident(id) if id.name == SYS_NAMESPACE)
            && SysMethod::from_name(&method.name) == Some(SysMethod::Exit) =>
        {
            Some(first_positional_arg(args))
        }
        ExprKind::Call { callee, args, .. } => match &callee.kind {
            ExprKind::Field { target, field }
                if matches!(&target.kind, ExprKind::Ident(id) if id.name == SYS_NAMESPACE)
                    && SysMethod::from_name(&field.name) == Some(SysMethod::Exit) =>
            {
                Some(first_positional_arg(args))
            }
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn ast_stmts_have_return(stmts: &[Stmt]) -> bool {
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
            StmtKind::Wait { arms, timer } => {
                if arms.iter().any(|a| ast_stmts_have_return(&a.body.stmts))
                    || timer
                        .as_ref()
                        .is_some_and(|t| ast_stmts_have_return(&t.body.stmts))
                {
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
            | StmtKind::Block(body)
                if ast_stmts_have_return(&body.stmts) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

pub(super) fn finish_prim_read_method<B: LowerBackend>(
    target: MirPrimTarget,
    pkind: MirPrimKind,
    method: &str,
    args: &[CallArg],
    span: Span,
    be: &mut B,
) -> Result<MirExpr, MirLowerError> {
    match (method, pkind) {
        ("read", MirPrimKind::Watch { elem }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::WatchRead {
                target,
                elem_ty: *elem.clone(),
            })),
            *elem,
        )),
        ("get", MirPrimKind::Guarded { elem }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::GuardedGet {
                target,
                elem_ty: *elem.clone(),
            })),
            *elem,
        )),
        ("try_recv", MirPrimKind::Inbox { elem, cap }) => {
            be.mark_may_hold_boost_be();
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxTryRecv {
                    target,
                    elem_ty: *elem.clone(),
                    cap,
                })),
                MirType::Result {
                    ok: elem,
                    err: None,
                },
            ))
        }
        ("try_send", MirPrimKind::Inbox { elem, cap }) => {
            let v = if let Some(first) = first_positional_arg(args) {
                let raw = be.lower_sub(first, Some(&elem))?;
                coerce_expr_if_needed(raw, &elem)
            } else {
                MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
            };
            let v = consume_moved_lease_be(v, be);
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
        ("cap", MirPrimKind::Inbox { cap, .. }) => Ok(MirExpr::new(
            MirExprKind::IntLit(cap as i128),
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
        ("try_wait", MirPrimKind::Signal) => {
            be.mark_may_hold_boost_be();
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::SignalTryWait { target })),
                MirType::Bool,
            ))
        }
        ("wake_one", MirPrimKind::WaitQueue) => {
            let boost = extract_boost_arg(args);
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::WaitQueueWakeOne { target, boost })),
                MirType::Bool,
            ))
        }
        ("half_len" | "cap", MirPrimKind::PingPong { cap, .. }) => Ok(MirExpr::new(
            MirExprKind::IntLit(cap as i128),
            MirType::Usize,
        )),
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
        ("base_addr", MirPrimKind::PingPong { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::PingPongBaseAddr {
                target,
                elem_ty: *elem,
                cap,
            })),
            MirType::Usize,
        )),
        ("push", MirPrimKind::ByteRing { cap }) => {
            let src = if let Some(first) = first_positional_arg(args) {
                let mut s = lower_fmt_buf_dst_be(first, be)?;
                if matches!(s.kind, MirExprKind::StringLit(_)) && s.ty == MirType::Str {
                    s.ty = MirType::Slice(Box::new(MirType::U8));
                }
                s
            } else {
                MirExpr::new(
                    MirExprKind::StringLit(String::new()),
                    MirType::Slice(Box::new(MirType::U8)),
                )
            };
            let boost = extract_boost_arg(args);
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingPush {
                    target,
                    cap,
                    src: Box::new(src),
                    boost,
                })),
                MirType::Usize,
            ))
        }
        ("push_byte", MirPrimKind::ByteRing { cap }) => {
            let b = if let Some(first) = first_positional_arg(args) {
                let raw = be.lower_sub(first, Some(&MirType::U8))?;
                coerce_expr_if_needed(raw, &MirType::U8)
            } else {
                MirExpr::new(MirExprKind::IntLit(0), MirType::U8)
            };
            let boost = extract_boost_arg(args);
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingPushByte {
                    target,
                    cap,
                    byte: Box::new(b),
                    boost,
                })),
                MirType::Bool,
            ))
        }
        ("grant_write", MirPrimKind::ByteRing { cap }) => {
            let len = if let Some(first) = first_positional_arg(args) {
                let raw = be.lower_sub(first, Some(&MirType::Usize))?;
                coerce_expr_if_needed(raw, &MirType::Usize)
            } else {
                MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)
            };
            let slice_ty = MirType::Slice(Box::new(MirType::U8));
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingGrantWrite {
                    target,
                    cap,
                    len: Box::new(len),
                })),
                MirType::Result {
                    ok: Box::new(slice_ty),
                    err: None,
                },
            ))
        }
        ("read_slice", MirPrimKind::ByteRing { cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingReadSlice { target, cap })),
            MirType::Slice(Box::new(MirType::U8)),
        )),
        ("pop", MirPrimKind::ByteRing { cap }) => {
            be.mark_may_hold_boost_be();
            let dst = if let Some(first) = first_positional_arg(args) {
                lower_fmt_buf_dst_be(first, be)?
            } else {
                MirExpr::new(
                    MirExprKind::StringLit(String::new()),
                    MirType::Slice(Box::new(MirType::U8)),
                )
            };
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingPop {
                    target,
                    cap,
                    dst: Box::new(dst),
                })),
                MirType::Usize,
            ))
        }
        ("try_pop_byte", MirPrimKind::ByteRing { cap }) => {
            be.mark_may_hold_boost_be();
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingTryPopByte { target, cap })),
                MirType::Result {
                    ok: Box::new(MirType::U8),
                    err: None,
                },
            ))
        }
        ("len", MirPrimKind::ByteRing { cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingLen { target, cap })),
            MirType::Usize,
        )),
        ("free_space", MirPrimKind::ByteRing { cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingFreeSpace { target, cap })),
            MirType::Usize,
        )),
        ("cap", MirPrimKind::ByteRing { cap }) => Ok(MirExpr::new(
            MirExprKind::IntLit(cap as i128),
            MirType::Usize,
        )),
        ("is_empty", MirPrimKind::ByteRing { cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingIsEmpty { target, cap })),
            MirType::Bool,
        )),
        ("is_full", MirPrimKind::ByteRing { cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingIsFull { target, cap })),
            MirType::Bool,
        )),
        ("try_alloc", MirPrimKind::Pool { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::PoolTryAlloc {
                target,
                elem_ty: *elem.clone(),
                cap,
            })),
            MirType::Result {
                ok: Box::new(MirType::Lease(elem)),
                err: None,
            },
        )),
        ("avail", MirPrimKind::Pool { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::PoolAvail {
                target,
                elem_ty: *elem,
                cap,
            })),
            MirType::Usize,
        )),
        ("in_use", MirPrimKind::Pool { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::PoolInUse {
                target,
                elem_ty: *elem,
                cap,
            })),
            MirType::Usize,
        )),
        ("cap", MirPrimKind::Pool { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::PoolCap {
                target,
                elem_ty: *elem,
                cap,
            })),
            MirType::Usize,
        )),
        ("is_exhausted", MirPrimKind::Pool { elem, cap }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::PoolIsExhausted {
                target,
                elem_ty: *elem,
                cap,
            })),
            MirType::Bool,
        )),
        ("park", MirPrimKind::DmaSlot { elem }) => {
            let lease_ty = MirType::Lease(elem.clone());
            let l = if let Some(first) = first_positional_arg(args) {
                let raw = be.lower_sub(first, Some(&lease_ty))?;
                coerce_expr_if_needed(raw, &lease_ty)
            } else {
                MirExpr::new(MirExprKind::IntLit(0), lease_ty.clone())
            };
            let l = consume_moved_lease_be(l, be);
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::DmaSlotPark {
                    target,
                    elem_ty: *elem,
                    lease: Box::new(l),
                })),
                MirType::Result {
                    ok: Box::new(MirType::Void),
                    err: None,
                },
            ))
        }
        ("take", MirPrimKind::DmaSlot { elem }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::DmaSlotTake {
                target,
                elem_ty: *elem.clone(),
            })),
            MirType::Result {
                ok: Box::new(MirType::Lease(elem)),
                err: None,
            },
        )),
        ("release", MirPrimKind::DmaSlot { elem }) => {
            let boost = extract_boost_arg(args);
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::DmaSlotRelease {
                    target,
                    elem_ty: *elem,
                    boost,
                })),
                MirType::Bool,
            ))
        }
        ("is_busy", MirPrimKind::DmaSlot { elem }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::DmaSlotIsBusy {
                target,
                elem_ty: *elem,
            })),
            MirType::Bool,
        )),
        ("is_empty", MirPrimKind::DmaSlot { elem }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::DmaSlotIsEmpty {
                target,
                elem_ty: *elem,
            })),
            MirType::Bool,
        )),
        ("addr", MirPrimKind::DmaSlot { elem }) => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::DmaSlotAddr {
                target,
                elem_ty: *elem,
            })),
            MirType::Usize,
        )),
        _ => Err(MirLowerError {
            message: format!("unknown primitive read method `.{method}()`"),
            span,
        }),
    }
}

pub(super) fn finish_inbox_view_read_method<B: LowerBackend>(
    view: MirExpr,
    elem_ty: MirType,
    method: &str,
    args: &[CallArg],
    span: Span,
    be: &mut B,
) -> Result<MirExpr, MirLowerError> {
    match method {
        "try_recv" => {
            be.mark_may_hold_boost_be();
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxViewTryRecv {
                    view,
                    elem_ty: elem_ty.clone(),
                })),
                MirType::Result {
                    ok: Box::new(elem_ty),
                    err: None,
                },
            ))
        }
        "try_send" => {
            let v = if let Some(first) = first_positional_arg(args) {
                let raw = be.lower_sub(first, Some(&elem_ty))?;
                coerce_expr_if_needed(raw, &elem_ty)
            } else {
                MirExpr::new(MirExprKind::IntLit(0), elem_ty.clone())
            };
            let v = consume_moved_lease_be(v, be);
            let boost = extract_boost_arg(args);
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxViewTrySend {
                    view,
                    elem_ty,
                    value: Box::new(v),
                    boost,
                })),
                MirType::Bool,
            ))
        }
        "len" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxViewLen { view, elem_ty })),
            MirType::Usize,
        )),
        "cap" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxViewCap { view, elem_ty })),
            MirType::Usize,
        )),
        "is_empty" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxViewIsEmpty { view, elem_ty })),
            MirType::Bool,
        )),
        "is_full" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::InboxViewIsFull { view, elem_ty })),
            MirType::Bool,
        )),
        _ => Err(MirLowerError {
            message: format!("unknown `*inbox` view read method `.{method}()`"),
            span,
        }),
    }
}

pub(super) fn finish_byte_ring_view_read_method<B: LowerBackend>(
    view: MirExpr,
    method: &str,
    args: &[CallArg],
    span: Span,
    be: &mut B,
) -> Result<MirExpr, MirLowerError> {
    match method {
        "push" => {
            let src = if let Some(first) = first_positional_arg(args) {
                let mut s = lower_fmt_buf_dst_be(first, be)?;
                if matches!(s.kind, MirExprKind::StringLit(_)) && s.ty == MirType::Str {
                    s.ty = MirType::Slice(Box::new(MirType::U8));
                }
                s
            } else {
                MirExpr::new(
                    MirExprKind::StringLit(String::new()),
                    MirType::Slice(Box::new(MirType::U8)),
                )
            };
            let boost = extract_boost_arg(args);
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewPush {
                    view,
                    src: Box::new(src),
                    boost,
                })),
                MirType::Usize,
            ))
        }
        "push_byte" => {
            let b = if let Some(first) = first_positional_arg(args) {
                let raw = be.lower_sub(first, Some(&MirType::U8))?;
                coerce_expr_if_needed(raw, &MirType::U8)
            } else {
                MirExpr::new(MirExprKind::IntLit(0), MirType::U8)
            };
            let boost = extract_boost_arg(args);
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewPushByte {
                    view,
                    byte: Box::new(b),
                    boost,
                })),
                MirType::Bool,
            ))
        }
        "grant_write" => {
            let len = if let Some(first) = first_positional_arg(args) {
                let raw = be.lower_sub(first, Some(&MirType::Usize))?;
                coerce_expr_if_needed(raw, &MirType::Usize)
            } else {
                MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)
            };
            let slice_ty = MirType::Slice(Box::new(MirType::U8));
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewGrantWrite {
                    view,
                    len: Box::new(len),
                })),
                MirType::Result {
                    ok: Box::new(slice_ty),
                    err: None,
                },
            ))
        }
        "read_slice" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewReadSlice { view })),
            MirType::Slice(Box::new(MirType::U8)),
        )),
        "pop" => {
            be.mark_may_hold_boost_be();
            let dst = if let Some(first) = first_positional_arg(args) {
                lower_fmt_buf_dst_be(first, be)?
            } else {
                MirExpr::new(
                    MirExprKind::StringLit(String::new()),
                    MirType::Slice(Box::new(MirType::U8)),
                )
            };
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewPop {
                    view,
                    dst: Box::new(dst),
                })),
                MirType::Usize,
            ))
        }
        "try_pop_byte" => {
            be.mark_may_hold_boost_be();
            Ok(MirExpr::new(
                MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewTryPopByte { view })),
                MirType::Result {
                    ok: Box::new(MirType::U8),
                    err: None,
                },
            ))
        }
        "len" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewLen { view })),
            MirType::Usize,
        )),
        "cap" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewCap { view })),
            MirType::Usize,
        )),
        "free_space" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewFreeSpace { view })),
            MirType::Usize,
        )),
        "is_empty" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewIsEmpty { view })),
            MirType::Bool,
        )),
        "is_full" => Ok(MirExpr::new(
            MirExprKind::PrimRead(Box::new(MirPrimReadOp::ByteRingViewIsFull { view })),
            MirType::Bool,
        )),
        _ => Err(MirLowerError {
            message: format!("unknown `*byte_ring` view read method `.{method}()`"),
            span,
        }),
    }
}

pub(super) fn lower_place_common<B: LowerBackend>(
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
                    (
                        MirPlace::Deref(Box::new(ptr_expr), pointee.clone()),
                        pointee,
                    )
                }
            };
            let sname = match &base_ty {
                MirType::Struct(n) => n.clone(),
                _ => {
                    return Err(MirLowerError {
                        message: format!(
                            "field assignment on non-struct `{}`",
                            base_ty.display_name()
                        ),
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
            let raw_idx = be.lower_sub(index, Some(&MirType::Usize))?;
            let idx = coerce_expr_if_needed(raw_idx, &MirType::Usize);
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

pub(super) fn try_lower_nonblocking_call_or_method<B: LowerBackend>(
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
            // 1. `sys.*` intrinsics (excluding expression queries `now_ms`, `now_ticks`, `current_prio`, `sleep_depth`, `wfi_count`, `rom_call_u32`)
            if let ExprKind::Ident(id) = &receiver.kind {
                let sys_method = SysMethod::from_name(&method.name);
                if id.name == SYS_NAMESPACE && !sys_method.is_some_and(SysMethod::is_expr_query) {
                    if sys_method == Some(SysMethod::Log)
                        || matches!(method.name.as_str(), "println" | "print" | "puts")
                    {
                        let newline =
                            sys_method == Some(SysMethod::Log) || method.name == "println";
                        if let Some(first) = first_positional_arg(args) {
                            if let Some(pfmt) = try_lower_print_arg_be(first, newline, be)? {
                                return Ok(Some(pfmt));
                            }
                        }
                    }
                    match sys_method {
                        Some(SysMethod::UseSystick) => {
                            let hz = if let Some(first) = first_positional_arg(args) {
                                let raw = be.lower_sub(first, Some(&MirType::U32))?;
                                coerce_expr_if_needed(raw, &MirType::U32)
                            } else {
                                MirExpr::new(MirExprKind::IntLit(1_000_000), MirType::U32)
                            };
                            return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::UseSystick { hz })));
                        }
                        Some(SysMethod::PendIrq) if args.len() == 1 => {
                            if let Some((_, num)) = resolve_irq_token_be(&args[0].value, be) {
                                return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                                    method: SysMethod::PendIrq.canonical_name().to_string(),
                                    args: vec![MirExpr::new(
                                        MirExprKind::IntLit(num as i128),
                                        MirType::U32,
                                    )],
                                })));
                            }
                        }
                        Some(SysMethod::RomCallVoid) => {
                            let (addr, stack_bound, mut abi_args) =
                                lower_rom_call_args_be(args, true, expr.span, be)?;
                            let mut sys_args = Vec::with_capacity(2 + abi_args.len());
                            sys_args.push(addr);
                            sys_args.push(MirExpr::new(
                                MirExprKind::IntLit(stack_bound as i128),
                                MirType::U32,
                            ));
                            sys_args.append(&mut abi_args);
                            return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                                method: SysMethod::RomCallVoid.canonical_name().to_string(),
                                args: sys_args,
                            })));
                        }
                        Some(SysMethod::RomCallNoreturn) => {
                            let (addr, _, mut abi_args) =
                                lower_rom_call_args_be(args, false, expr.span, be)?;
                            let mut sys_args = Vec::with_capacity(1 + abi_args.len());
                            sys_args.push(addr);
                            sys_args.append(&mut abi_args);
                            return Ok(Some(MirStmt::Intrinsic(MirIntrinsicOp::SysOp {
                                method: SysMethod::RomCallNoreturn.canonical_name().to_string(),
                                args: sys_args,
                            })));
                        }
                        _ => {}
                    }
                    let mut largs = Vec::new();
                    for (idx, a) in args.iter().enumerate() {
                        let exp = match (sys_method, idx) {
                            (Some(SysMethod::Log | SysMethod::LogKv | SysMethod::LogHex), 0) => {
                                Some(&MirType::Str)
                            }
                            (Some(SysMethod::LogKv), 1) | (Some(SysMethod::Exit), 0) => {
                                Some(&MirType::I32)
                            }
                            (Some(SysMethod::LogHex), 1)
                            | (Some(SysMethod::PendIrq), 0)
                            | (Some(SysMethod::FuzzIrqs), 0 | 1)
                            | (Some(SysMethod::TickBy), 0) => Some(&MirType::U32),
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
            if let Some(recv_kind) = resolve_prim_receiver_kind_be(receiver, be)? {
                let m = method.name.as_str();
                match recv_kind {
                    LoweredPrimReceiver::InboxView(view, elem) => {
                        if m == "post" {
                            let v = if let Some(first) = first_positional_arg(args) {
                                be.lower_sub(first, Some(&elem))?
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), elem.clone())
                            };
                            let boost = extract_boost_arg(args);
                            let v = consume_moved_lease_be(coerce_expr_if_needed(v, &elem), be);
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::InboxViewPost {
                                    view,
                                    elem_ty: elem.clone(),
                                    value: v,
                                    boost,
                                },
                            )));
                        }
                    }
                    LoweredPrimReceiver::ByteRingView(view) => match m {
                        "consume" => {
                            let len = if let Some(first) = first_positional_arg(args) {
                                let raw = be.lower_sub(first, Some(&MirType::Usize))?;
                                coerce_expr_if_needed(raw, &MirType::Usize)
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)
                            };
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::ByteRingViewConsume { view, len },
                            )));
                        }
                        "commit_write" => {
                            let len = if let Some(first) = first_positional_arg(args) {
                                let raw = be.lower_sub(first, Some(&MirType::Usize))?;
                                coerce_expr_if_needed(raw, &MirType::Usize)
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)
                            };
                            let boost = extract_boost_arg(args);
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, boost },
                            )));
                        }
                        "clear" => {
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::ByteRingViewClear { view },
                            )));
                        }
                        _ => {}
                    },
                    LoweredPrimReceiver::Sized(target, pkind) => match (m, pkind) {
                        ("publish", MirPrimKind::Watch { elem }) => {
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
                        ("set", MirPrimKind::Guarded { elem }) => {
                            let v = if let Some(first) = first_positional_arg(args) {
                                be.lower_sub(first, Some(&elem))?
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                            };
                            return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::GuardedSet {
                                target,
                                elem_ty: *elem.clone(),
                                value: coerce_expr_if_needed(v, &elem),
                            })));
                        }
                        ("post", MirPrimKind::Inbox { elem, cap }) => {
                            let v = if let Some(first) = first_positional_arg(args) {
                                be.lower_sub(first, Some(&elem))?
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), *elem.clone())
                            };
                            let boost = extract_boost_arg(args);
                            let v = consume_moved_lease_be(coerce_expr_if_needed(v, &elem), be);
                            return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::InboxPost {
                                target,
                                elem_ty: *elem.clone(),
                                cap,
                                value: v,
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
                        ("prepare_wait", MirPrimKind::WaitQueue) => {
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::WaitQueuePrepare { target },
                            )));
                        }
                        ("cancel_wait", MirPrimKind::WaitQueue) => {
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::WaitQueueCancel { target },
                            )));
                        }
                        ("wake_all", MirPrimKind::WaitQueue) => {
                            let boost = extract_boost_arg(args);
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::WaitQueueWakeAll { target, boost },
                            )));
                        }
                        ("flip", MirPrimKind::PingPong { elem, cap }) => {
                            return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::PingPongFlip {
                                target,
                                elem_ty: *elem,
                                cap,
                            })));
                        }
                        ("complete_half", MirPrimKind::PingPong { elem, cap }) => {
                            let half = if let Some(first) = first_positional_arg(args) {
                                let raw = be.lower_sub(first, Some(&MirType::U8))?;
                                coerce_expr_if_needed(raw, &MirType::U8)
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), MirType::U8)
                            };
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::PingPongCompleteHalf {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    half,
                                },
                            )));
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
                        (
                            "stop" | "deactivate" | "release",
                            MirPrimKind::PingPong { elem, cap },
                        ) => {
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::PingPongSetActive {
                                    target,
                                    elem_ty: *elem,
                                    cap,
                                    active: false,
                                },
                            )));
                        }
                        ("release", MirPrimKind::DmaSlot { elem }) => {
                            let boost = extract_boost_arg(args);
                            return Ok(Some(MirStmt::Expr(MirExpr::new(
                                MirExprKind::PrimRead(Box::new(MirPrimReadOp::DmaSlotRelease {
                                    target,
                                    elem_ty: *elem,
                                    boost,
                                })),
                                MirType::Bool,
                            ))));
                        }
                        ("consume", MirPrimKind::ByteRing { cap }) => {
                            let len = if let Some(first) = first_positional_arg(args) {
                                let raw = be.lower_sub(first, Some(&MirType::Usize))?;
                                coerce_expr_if_needed(raw, &MirType::Usize)
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)
                            };
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::ByteRingConsume { target, cap, len },
                            )));
                        }
                        ("commit_write", MirPrimKind::ByteRing { cap }) => {
                            let len = if let Some(first) = first_positional_arg(args) {
                                let raw = be.lower_sub(first, Some(&MirType::Usize))?;
                                coerce_expr_if_needed(raw, &MirType::Usize)
                            } else {
                                MirExpr::new(MirExprKind::IntLit(0), MirType::Usize)
                            };
                            let boost = extract_boost_arg(args);
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::ByteRingCommitWrite {
                                    target,
                                    cap,
                                    len,
                                    boost,
                                },
                            )));
                        }
                        ("clear", MirPrimKind::ByteRing { cap }) => {
                            return Ok(Some(MirStmt::PrimOp(
                                MirPrimNonBlockingOp::ByteRingClear { target, cap },
                            )));
                        }
                        _ => {}
                    },
                }
            }

            if method.name == "release" {
                if let Ok(recv_me) = be.lower_sub(receiver, None) {
                    if let MirType::Lease(elem) = recv_me.ty.clone() {
                        let moved = consume_moved_lease_be(recv_me, be);
                        let boost = extract_boost_arg(args);
                        return Ok(Some(MirStmt::PrimOp(MirPrimNonBlockingOp::LeaseRelease {
                            value: moved,
                            elem_ty: *elem,
                            boost,
                        })));
                    }
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

pub(super) fn push_fmt_lit(chunks: &mut Vec<MirFmtChunk>, s: &str) {
    if s.is_empty() {
        return;
    }
    if let Some(MirFmtChunk::Literal(prev)) = chunks.last_mut() {
        prev.push_str(s);
    } else {
        chunks.push(MirFmtChunk::Literal(s.to_string()));
    }
}

pub(super) fn try_lower_print_arg_be<B: LowerBackend>(
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
                if matches!(&lowered.ty, MirType::Str | MirType::Ptr(_)) {
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

pub(super) fn array_expr_to_slice_be<B: LowerBackend>(
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

pub(super) fn lower_fmt_buf_dst_be<B: LowerBackend>(
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

pub(super) fn lower_fmt_buf_call_be<B: LowerBackend>(
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

pub(super) fn lower_fmt_ring_call_be<B: LowerBackend>(
    args: &[CallArg],
    span: Span,
    be: &mut B,
) -> Result<MirExpr, MirLowerError> {
    let pos_args: Vec<&CallArg> = args.iter().filter(|a| a.name.is_none()).collect();
    let ring_arg = pos_args.first().ok_or_else(|| MirLowerError {
        message: "`fmt_ring` expects `(tx, f\"...\" [, boost: @prio])`".to_string(),
        span,
    })?;
    let fmt_arg = pos_args.get(1).ok_or_else(|| MirLowerError {
        message: "`fmt_ring` expects a second `f\"...\"` or value argument".to_string(),
        span,
    })?;
    let raw_ring = be.lower_sub(&ring_arg.value, None)?;
    let ring = coerce_expr_if_needed(raw_ring, &MirType::PrimView(MirPrimViewKind::ByteRing));
    let boost = extract_boost_arg(args);
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
        ok: Box::new(MirType::Usize),
        err: Some(Box::new(MirType::Enum("FmtErr".to_string()))),
    };
    let dst_res = be.alloc_temp_be("fmt_ring_res", res_ty.clone());
    be.emit_stmt_be(MirStmt::FmtRing {
        dst_res,
        ring,
        chunks,
        boost,
    });
    Ok(MirExpr::new(MirExprKind::Var(dst_res), res_ty))
}

pub(super) fn lower_format_parts_be<B: LowerBackend>(
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

pub(super) fn emit_ptr_hex_debug(val: MirExpr, chunks: &mut Vec<MirFmtChunk>) {
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

pub(super) fn emit_fmt_value_be<B: LowerBackend>(
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

pub(super) fn expand_array_debug_be<B: LowerBackend>(
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
                index: Box::new(MirExpr::new(
                    MirExprKind::IntLit(idx as i128),
                    MirType::Usize,
                )),
                is_slice: false,
            },
            elem_ty.host_value_type(),
        );
        emit_fmt_value_be(elem_expr, FormatSpec::Debug, span, chunks, be)?;
    }
    push_fmt_lit(chunks, "]");
    Ok(())
}

pub(super) fn expand_struct_debug_be<B: LowerBackend>(
    val: MirExpr,
    sname: &str,
    span: Span,
    chunks: &mut Vec<MirFmtChunk>,
    be: &mut B,
) -> Result<(), MirLowerError> {
    let st = be
        .syms()
        .structs
        .get(sname)
        .cloned()
        .ok_or_else(|| MirLowerError {
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

pub(super) fn collect_formatted_enums(
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

pub(super) fn scan_stmt_formatted_enums(stmt: &MirStmt, out: &mut BTreeSet<String>) {
    match stmt {
        MirStmt::PrintFmt { chunks, .. }
        | MirStmt::FmtBuf { chunks, .. }
        | MirStmt::FmtRing { chunks, .. } => {
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
