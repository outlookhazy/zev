use super::*;

pub(super) const VALID_SYS_METHODS_NOTE: &str = "valid portable `sys.*` methods: `sys.use_systick(u32 hz)`, `sys.tick()`, `sys.tick_ms()`, `sys.tick_by(u32 ms)`, `sys.set_putc(fn(u8))`, `sys.set_clock_ms(fn() -> u32)`, `sys.set_exit(fn(i32))`, `sys.set_fault_handler(fn())`, `sys.set_idle(fn(u32, u32))`, `sys.set_irq_router(fn(u32, u32), fn(u32))`, `sys.rom_call_u32(usize fn_addr, usize stack_bound, ...) -> u32`, `sys.rom_call_void(usize fn_addr, usize stack_bound, ...)`, `sys.rom_call_noreturn(usize fn_addr, ...)`, `sys.now_ms() -> u32`, `sys.now_ticks() -> u32`, `sys.current_prio() -> u32`, `sys.sleep_depth() -> u32`, `sys.wfi_count() -> u32`, `sys.trace_sleep(bool on)`, `sys.pend_irq(u32 irq)`, `sys.fuzz_irqs(u32 seed, u32 irq_mask)`, `sys.nop()`, `sys.exit(i32 code)`, `sys.log(*u8 msg)`, `sys.log_kv(*u8 label, i32 val)`, `sys.log_hex(*u8 label, u32 val)`, `sys.delay(dur)`";

pub(super) fn is_string_arg_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::Str)
        || matches!(ty, Ty::Ptr(inner) if matches!(**inner, Ty::Prim(PrimType::U8 | PrimType::I8)))
}

pub(super) fn is_print_arg_ty(ty: &Ty) -> bool {
    is_string_arg_ty(ty)
        || matches!(ty, Ty::Slice(inner) | Ty::Array { elem: inner, .. } if **inner == Ty::u8())
}

pub(super) fn has_attr_dma_safe(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| matches!(a.kind, AttrKind::DmaSafe))
}

pub(super) fn has_attr_manual_yield(attrs: &[Attr]) -> bool {
    attrs
        .iter()
        .any(|a| matches!(a.kind, AttrKind::ManualYield))
}

pub(super) fn has_attr_noyield(attrs: &[Attr]) -> bool {
    attrs
        .iter()
        .any(|a| matches!(&a.kind, AttrKind::Custom { name, .. } if name == "noyield"))
}

pub(super) fn root_ident_of(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Ident(id) => Some(id.name.clone()),
        ExprKind::Field { target, .. }
        | ExprKind::Index { target, .. }
        | ExprKind::Slice { target, .. } => root_ident_of(target),
        ExprKind::Unary {
            op: UnaryOp::Deref | UnaryOp::AddrOf,
            expr: inner,
        } => root_ident_of(inner),
        _ => None,
    }
}

pub(super) fn extract_monomorphized_attr(attrs: &[Attr]) -> Option<String> {
    for a in attrs {
        if let AttrKind::Custom {
            name,
            arg: Some(sig),
        } = &a.kind
        {
            if name == "monomorphized" {
                return Some(sig.clone());
            }
        }
    }
    None
}

pub(super) fn extract_instantiated_from_attr(attrs: &[Attr]) -> Option<String> {
    for a in attrs {
        if let AttrKind::Custom {
            name,
            arg: Some(note),
        } = &a.kind
        {
            if name == "instantiated_from" {
                return Some(note.clone());
            }
        }
    }
    None
}

pub(super) fn ty_contains_os_prim_view(
    ty: &Ty,
    structs: &BTreeMap<String, StructInfo>,
    visiting: &mut BTreeSet<String>,
) -> bool {
    match ty {
        Ty::OsPrimitive(_) => true,
        Ty::Ptr(inner) | Ty::Slice(inner) | Ty::Array { elem: inner, .. } | Ty::Lease(inner) => {
            ty_contains_os_prim_view(inner, structs, visiting)
        }
        Ty::Struct(name) => {
            if !visiting.insert(name.clone()) {
                return false;
            }
            let has = structs.get(name).is_some_and(|st| {
                st.fields
                    .iter()
                    .any(|f| ty_contains_os_prim_view(&f.ty, structs, visiting))
            });
            visiting.remove(name);
            has
        }
        _ => false,
    }
}

pub(super) fn is_valid_cast(src: &Ty, dst: &Ty) -> bool {
    if ty_contains_linear_lease(src) || ty_contains_linear_lease(dst) {
        return false;
    }
    if src.contains_mmio() || dst.contains_mmio() {
        return match (src, dst) {
            (Ty::Ptr(a), Ty::Ptr(b)) => {
                matches!((&**a, &**b), (Ty::MmioLayout(la), Ty::MmioLayout(lb)) if la == lb)
            }
            (Ty::MmioInstance { layout: la, .. }, Ty::Ptr(b)) => {
                matches!(&**b, Ty::MmioLayout(lb) if la == lb)
            }
            (Ty::MmioInstance { .. }, b) => b.is_integer(),
            _ => false,
        };
    }
    if src.contains_os_primitive() || dst.contains_os_primitive() {
        return match (src, dst) {
            (Ty::Ptr(a), Ty::Ptr(b)) => match (&**a, &**b) {
                (Ty::OsPrimitive(sa), Ty::OsPrimitive(sb)) => {
                    sa == sb
                        || matches!(
                            (sa, sb),
                            (
                                ResolvedOsPrimitive::ByteRing { .. },
                                ResolvedOsPrimitive::ByteRingUnsized
                            )
                        )
                        || matches!(
                            (sa, sb),
                            (
                                ResolvedOsPrimitive::Inbox { elem: ea, .. },
                                ResolvedOsPrimitive::InboxUnsized { elem: eb }
                            ) if ea == eb
                        )
                }
                _ => false,
            },
            _ => false,
        };
    }
    if src == dst {
        return true;
    }
    match (src, dst) {
        // Any numeric <-> numeric (`int <-> int`, `int <-> float`, `f32 <-> f64`).
        (a, b) if a.is_numeric() && b.is_numeric() => true,
        // `bool` -> integer or float.
        (Ty::Prim(PrimType::Bool), b) if b.is_numeric() => true,
        // `enum` -> integer.
        (Ty::Enum(_), b) if b.is_integer() => true,
        // Pointer <-> pointer or pointer <-> integer.
        (Ty::Ptr(_), Ty::Ptr(_)) => true,
        (Ty::Ptr(_), b) if b.is_integer() => true,
        (a, Ty::Ptr(_)) if a.is_integer() => true,
        (Ty::IrqToken, b) if b.is_integer() => true,
        (Ty::Str, Ty::Ptr(inner))
            if inner.is_void() || matches!(**inner, Ty::Prim(PrimType::U8 | PrimType::I8)) =>
        {
            true
        }
        (Ty::FnPtr { .. }, Ty::Ptr(inner)) if inner.is_void() => true,
        _ => false,
    }
}

pub(super) fn unify_float_arith(lhs: &Ty, rhs: &Ty) -> Option<Ty> {
    match (lhs, rhs) {
        (Ty::FloatLit, Ty::FloatLit) => Some(Ty::FloatLit),
        (Ty::FloatLit, Ty::Prim(PrimType::F32)) | (Ty::Prim(PrimType::F32), Ty::FloatLit) => {
            Some(Ty::f32())
        }
        (Ty::FloatLit, Ty::Prim(PrimType::F64)) | (Ty::Prim(PrimType::F64), Ty::FloatLit) => {
            Some(Ty::f64())
        }
        (Ty::Prim(PrimType::F32), Ty::Prim(PrimType::F32)) => Some(Ty::f32()),
        (Ty::Prim(PrimType::F64), Ty::Prim(PrimType::F64)) => Some(Ty::f64()),
        _ => None,
    }
}

pub(super) const fn int_decimal_worst_case(p: PrimType) -> usize {
    match p {
        PrimType::U8 => 3,
        PrimType::I8 => 4,
        PrimType::U16 | PrimType::U16Be | PrimType::U16Le => 5,
        PrimType::I16 => 6,
        PrimType::U32 | PrimType::U32Be | PrimType::U32Le => 10,
        PrimType::I32 => 11,
        PrimType::U64
        | PrimType::U64Be
        | PrimType::U64Le
        | PrimType::Usize
        | PrimType::I64
        | PrimType::Isize => 20,
        _ => 11,
    }
}

pub(super) const fn int_hex_worst_case(p: PrimType) -> usize {
    match p {
        PrimType::U8 | PrimType::I8 => 2,
        PrimType::U16 | PrimType::U16Be | PrimType::U16Le | PrimType::I16 => 4,
        PrimType::U32 | PrimType::U32Be | PrimType::U32Le | PrimType::I32 => 8,
        _ => 16,
    }
}

pub(super) const fn int_bin_worst_case(p: PrimType) -> usize {
    match p {
        PrimType::U8 | PrimType::I8 => 8,
        PrimType::U16 | PrimType::U16Be | PrimType::U16Le | PrimType::I16 => 16,
        PrimType::U32 | PrimType::U32Be | PrimType::U32Le | PrimType::I32 => 32,
        _ => 64,
    }
}

pub(super) fn try_unify_integer(lhs: &Ty, rhs: &Ty) -> Option<Ty> {
    match (lhs, rhs) {
        (Ty::IntLit, Ty::IntLit) => Some(Ty::IntLit),
        (Ty::IntLit, Ty::Prim(p)) | (Ty::Prim(p), Ty::IntLit) if p.is_integer() => {
            Some(Ty::Prim(Ty::host_prim(*p)))
        }
        (Ty::Prim(a), Ty::Prim(b)) if a.is_integer() && b.is_integer() => {
            if Ty::can_implicitly_widen_int(*a, *b) {
                Some(Ty::Prim(Ty::host_prim(*b)))
            } else if Ty::can_implicitly_widen_int(*b, *a) {
                Some(Ty::Prim(Ty::host_prim(*a)))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) const fn iter_span_none() -> Option<Span> {
    None
}

pub(super) fn is_sys_exit_expr(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::MethodCall {
            receiver, method, ..
        } => {
            matches!(&receiver.kind, ExprKind::Ident(id) if id.name == SYS_NAMESPACE)
                && SysMethod::from_name(&method.name).is_some_and(SysMethod::is_diverging_exit)
        }
        ExprKind::Call { callee, .. } => match &callee.kind {
            ExprKind::Field { target, field } => {
                matches!(&target.kind, ExprKind::Ident(id) if id.name == SYS_NAMESPACE)
                    && SysMethod::from_name(&field.name).is_some_and(SysMethod::is_diverging_exit)
            }
            _ => false,
        },
        _ => false,
    }
}

/// Returns `true` if `block` contains a `break` targeting the enclosing loop (not nested inside an inner loop).
pub(super) fn block_has_break(block: &Block) -> bool {
    stmts_have_break(&block.stmts)
}

pub(super) fn stmts_have_break(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Break => return true,
            StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                if block_has_break(then_block) || else_block.as_ref().is_some_and(block_has_break) {
                    return true;
                }
            }
            StmtKind::Switch { arms, .. } => {
                if arms.iter().any(|a| block_has_break(&a.body)) {
                    return true;
                }
            }
            StmtKind::Wait { arms, timer } => {
                if arms.iter().any(|a| block_has_break(&a.body))
                    || timer.as_ref().is_some_and(|t| block_has_break(&t.body))
                {
                    return true;
                }
            }
            StmtKind::NoYield { body } | StmtKind::Critical { body } | StmtKind::Block(body) => {
                if block_has_break(body) {
                    return true;
                }
            }
            StmtKind::Expr(expr)
            | StmtKind::VarDecl {
                init: Some(expr), ..
            } => {
                if expr_has_break(expr) {
                    return true;
                }
            }
            // Inner loops consume their own `break`.
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

pub(super) fn expr_has_break(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Catch {
            expr: inner,
            clause,
        } => expr_has_break(inner) || block_has_break(&clause.body),
        ExprKind::Or {
            expr: a,
            fallback: b,
        } => expr_has_break(a) || expr_has_break(b),
        _ => false,
    }
}

pub(super) fn is_mmio_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::MmioLayout(_) | Ty::MmioInstance { .. })
        || matches!(ty, Ty::Ptr(inner) if matches!(&**inner, Ty::MmioLayout(_)))
}

pub(super) fn ty_contains_linear_lease(ty: &Ty) -> bool {
    match ty {
        Ty::Lease(_) => true,
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Ptr(elem) => ty_contains_linear_lease(elem),
        Ty::OsPrimitive(os) => os.elem_ty().is_some_and(ty_contains_linear_lease),
        Ty::FnPtr { params, ret } => {
            params.iter().any(ty_contains_linear_lease) || ty_contains_linear_lease(ret)
        }
        _ => false,
    }
}

pub(super) fn ty_contains_byval_mmio(ty: &Ty) -> Option<&str> {
    match ty {
        Ty::MmioLayout(name) => Some(name.as_str()),
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Lease(elem) => ty_contains_byval_mmio(elem),
        Ty::OsPrimitive(os) => os.elem_ty().and_then(ty_contains_byval_mmio),
        Ty::FnPtr { params, ret } => params
            .iter()
            .find_map(ty_contains_byval_mmio)
            .or_else(|| ty_contains_byval_mmio(ret)),
        _ => None,
    }
}

pub(super) fn ty_contains_bare_unsized_prim(ty: &Ty) -> Option<String> {
    match ty {
        Ty::OsPrimitive(os) if os.is_unsized_view() => Some(os.to_string()),
        Ty::OsPrimitive(os) => os.elem_ty().and_then(ty_contains_bare_unsized_prim),
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Lease(elem) => {
            ty_contains_bare_unsized_prim(elem)
        }
        Ty::FnPtr { params, ret } => params
            .iter()
            .find_map(ty_contains_bare_unsized_prim)
            .or_else(|| ty_contains_bare_unsized_prim(ret)),
        _ => None,
    }
}

pub(super) fn ty_embeds_os_primitive(ty: &Ty) -> bool {
    match ty {
        Ty::OsPrimitive(os) => !os.is_unsized_view(),
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Lease(elem) => ty_embeds_os_primitive(elem),
        _ => false,
    }
}

pub(super) fn ty_contains_byval_subsys<'a>(
    ty: &'a Ty,
    structs: &'a BTreeMap<String, StructInfo>,
) -> Option<&'a str> {
    match ty {
        Ty::Struct(name) if structs.get(name).is_some_and(|s| s.is_subsystem) => {
            Some(name.as_str())
        }
        Ty::Array { elem, .. } | Ty::Slice(elem) | Ty::Lease(elem) => {
            ty_contains_byval_subsys(elem, structs)
        }
        Ty::OsPrimitive(os) => os
            .elem_ty()
            .and_then(|e| ty_contains_byval_subsys(e, structs)),
        Ty::FnPtr { params, ret } => params
            .iter()
            .find_map(|p| ty_contains_byval_subsys(p, structs))
            .or_else(|| ty_contains_byval_subsys(ret, structs)),
        _ => None,
    }
}
