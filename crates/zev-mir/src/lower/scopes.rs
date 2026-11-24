use super::*;

pub(super) struct VarScopes {
    pub(super) frames: Vec<BTreeMap<String, VarId>>,
}

impl VarScopes {
    pub(super) fn new() -> Self {
        Self {
            frames: vec![BTreeMap::new()],
        }
    }

    pub(super) fn push(&mut self) {
        self.frames.push(BTreeMap::new());
    }

    pub(super) fn pop(&mut self) {
        self.frames.pop();
    }

    pub(super) fn insert(&mut self, name: String, vid: VarId) {
        if let Some(top) = self.frames.last_mut() {
            top.insert(name, vid);
        }
    }

    pub(super) fn lookup(&self, name: &str) -> Option<VarId> {
        for frame in self.frames.iter().rev() {
            if let Some(&vid) = frame.get(name) {
                return Some(vid);
            }
        }
        None
    }
}

pub(super) fn convert_assign_op(op: AssignOp) -> MirAssignOp {
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

pub(super) fn convert_bin_op(op: BinaryOp) -> MirBinOp {
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

pub(super) fn sanitize_ident(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn eval_const_u32(
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

pub(super) fn eval_const_u64(
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

pub(super) fn mmio_elem_bits_from_type(ty: &TypeExpr, span: Span) -> Result<u8, MirLowerError> {
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

pub(super) fn lower_mmio_regs(
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
        let end = (offset as usize)
            .checked_add(reg_bytes)
            .ok_or_else(|| MirLowerError {
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

pub(super) fn mmio_elem_mir_type(elem_bits: u8) -> MirType {
    match elem_bits {
        8 => MirType::U8,
        16 => MirType::U16,
        _ => MirType::U32,
    }
}

pub(super) fn find_named_arg<'a>(args: &'a [CallArg], name: &str) -> Option<&'a Expr> {
    args.iter()
        .find(|a| a.name.as_ref().is_some_and(|n| n.name == name))
        .map(|a| &a.value)
}

pub(super) fn first_positional_arg(args: &[CallArg]) -> Option<&Expr> {
    args.iter().find(|a| a.name.is_none()).map(|a| &a.value)
}

pub(super) fn extract_boost_arg(args: &[CallArg]) -> Option<MirPriority> {
    if let Some(be) = find_named_arg(args, "boost") {
        if let ExprKind::Priority(p) = &be.kind {
            return Some(convert_priority(*p));
        }
    }
    None
}

pub(super) fn lower_rom_call_args_be<B: LowerBackend>(
    args: &[CallArg],
    has_stack_bound: bool,
    span: Span,
    be: &mut B,
) -> Result<(MirExpr, u32, Vec<MirExpr>), MirLowerError> {
    let fixed = if has_stack_bound { 2 } else { 1 };
    if args.len() < fixed {
        return Err(MirLowerError {
            message: format!("sys.rom_call_* expects at least {fixed} argument(s)"),
            span,
        });
    }
    let stack_bound = if has_stack_bound {
        match eval_const_expr(&args[1].value, be.syms()) {
            Some(ConstVal::Int(v)) => u32::try_from(v).map_err(|_| MirLowerError {
                message: format!("sys.rom_call_* stack_bound ({v}) out of u32 range"),
                span: args[1].span,
            })?,
            _ => {
                return Err(MirLowerError {
                    message: "sys.rom_call_* stack_bound must be a compile-time constant integer"
                        .to_string(),
                    span: args[1].span,
                });
            }
        }
    } else {
        0
    };
    let raw_addr = be.lower_sub(&args[0].value, Some(&MirType::Usize))?;
    let addr = coerce_expr_if_needed(raw_addr, &MirType::Usize);
    let mut abi_args = Vec::with_capacity(args.len() - fixed);
    for a in &args[fixed..] {
        abi_args.push(be.lower_sub(&a.value, None)?);
    }
    Ok((addr, stack_bound, abi_args))
}

pub(super) fn eval_const_expr(expr: &Expr, syms: &ModuleSymbols) -> Option<ConstVal> {
    match &expr.kind {
        ExprKind::Literal(Literal::Int { value, .. }) => Some(ConstVal::Int(*value as i128)),
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
                (BinaryOp::Add, ConstVal::Int(a), ConstVal::Int(b)) => {
                    Some(ConstVal::Int(a.wrapping_add(b)))
                }
                (BinaryOp::Sub, ConstVal::Int(a), ConstVal::Int(b)) => {
                    Some(ConstVal::Int(a.wrapping_sub(b)))
                }
                (BinaryOp::Mul, ConstVal::Int(a), ConstVal::Int(b)) => {
                    Some(ConstVal::Int(a.wrapping_mul(b)))
                }
                (BinaryOp::Div, ConstVal::Int(a), ConstVal::Int(b)) if b != 0 => {
                    Some(ConstVal::Int(a / b))
                }
                (BinaryOp::Rem, ConstVal::Int(a), ConstVal::Int(b)) if b != 0 => {
                    Some(ConstVal::Int(a % b))
                }
                (BinaryOp::BitOr, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a | b)),
                (BinaryOp::BitAnd, ConstVal::Int(a), ConstVal::Int(b)) => {
                    Some(ConstVal::Int(a & b))
                }
                (BinaryOp::BitXor, ConstVal::Int(a), ConstVal::Int(b)) => {
                    Some(ConstVal::Int(a ^ b))
                }
                (BinaryOp::Shl, ConstVal::Int(a), ConstVal::Int(b)) => {
                    Some(ConstVal::Int(a.wrapping_shl(b as u32)))
                }
                (BinaryOp::Shr, ConstVal::Int(a), ConstVal::Int(b)) => {
                    Some(ConstVal::Int(a.wrapping_shr(b as u32)))
                }
                (BinaryOp::Eq, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a == b)),
                (BinaryOp::Ne, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a != b)),
                (BinaryOp::Lt, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a < b)),
                (BinaryOp::Le, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a <= b)),
                (BinaryOp::Gt, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a > b)),
                (BinaryOp::Ge, ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Bool(a >= b)),
                (BinaryOp::And, ConstVal::Bool(a), ConstVal::Bool(b)) => {
                    Some(ConstVal::Bool(a && b))
                }
                (BinaryOp::Or, ConstVal::Bool(a), ConstVal::Bool(b)) => {
                    Some(ConstVal::Bool(a || b))
                }
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

pub(super) fn convert_switch_pat(
    pat: &SwitchPat,
    scrutinee_ty: &MirType,
    syms: &ModuleSymbols,
) -> Option<MirSwitchPattern> {
    match pat {
        SwitchPat::Wildcard(_) | SwitchPat::Qualified { .. } => None,
        SwitchPat::Literal(lit, _) => match lit {
            Literal::Int { value, .. } => Some(MirSwitchPattern::Int(*value as i128)),
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
