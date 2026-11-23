use super::*;

impl<'a> BodyChecker<'a> {
    pub(super) fn check_fmt_buf_call(&mut self, args: &[CallArg], span: Span) -> ExprVal {
        let ret_slice_ty = Ty::Slice(Box::new(Ty::u8()));
        let fmt_err_spec = ResolvedErrorSpec::Typed(Ty::Enum("FmtErr".to_string()));

        if args.len() != 2 {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "'fmt_buf' expects 2 arguments (dst, f\"...\"), found {}",
                    args.len()
                ),
            ));
            for a in args {
                if let ExprKind::FormatString(parts) = &a.value.kind {
                    let _ = self.check_format_string(parts);
                } else {
                    let _ = self.check_expr_infallible(&a.value);
                }
            }
            self.proven_infallible_spans.insert((span.start, span.end));
            return ExprVal {
                ty: ret_slice_ty,
                error_spec: fmt_err_spec,
                prov: Provenance::pure_static(),
                endpoint_name: None,
                field_endpoints: BTreeMap::new(),
                span,
            };
        }

        let dst_expr = &args[0].value;
        let dv = self.check_expr_infallible(dst_expr);
        let is_valid_dst = match &dv.ty {
            Ty::Slice(elem) | Ty::Array { elem, .. } => **elem == Ty::u8(),
            Ty::Ptr(inner) => match &**inner {
                Ty::Slice(elem) | Ty::Array { elem, .. } => **elem == Ty::u8(),
                _ => false,
            },
            Ty::Error => true,
            _ => false,
        };
        if !is_valid_dst {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                args[0].span,
                format!(
                    "'fmt_buf' destination must be '[]u8', '[N]u8', or '*[N]u8', found '{}'",
                    dv.ty
                ),
            ));
        }

        let mut ret_prov = dv.prov.clone();
        if matches!(dv.ty, Ty::Array { .. }) {
            if let Some(root) = root_ident_of(dst_expr) {
                if let Some(var) = self.lookup_local(&root) {
                    ret_prov = ret_prov.merge(&Provenance::addr_of_local(&var.name, var.decl_depth));
                }
            }
        }
        let mut dst_alias_prov = ret_prov.clone();
        if let Some(root) = root_ident_of(dst_expr) {
            if let Some(var) = self.lookup_local(&root) {
                if var.ty.contains_reference(&self.sema.structs)
                    || matches!(var.ty, Ty::Array { .. })
                {
                    dst_alias_prov = dst_alias_prov
                        .merge(&Provenance::addr_of_local(&var.name, var.decl_depth));
                }
            }
        }

        let worst_len = if let ExprKind::FormatString(parts) = &args[1].value.kind {
            self.check_format_string_with_dst(parts, Some(&dst_alias_prov))
        } else {
            let _ = self.check_expr_infallible(&args[1].value);
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                args[1].span,
                "'fmt_buf' second argument must be an 'f\"...\"' format string",
            ));
            None
        };

        if let (Some(cap), Some(worst)) = (self.static_dst_capacity(dst_expr, &dv.ty), worst_len) {
            if worst > cap {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::FormatBufferTooSmall,
                        span,
                        format!(
                            "format string worst-case length ({} bytes) exceeds destination buffer capacity ({} bytes)",
                            worst, cap
                        ),
                    )
                    .with_note(format!(
                        "increase the destination buffer to at least `[{}]u8`, or format into a `[]u8` slice and handle truncation with `catch (FmtErr err, []u8 clamped) {{ ... }}`",
                        worst
                    )),
                );
                self.proven_infallible_spans.insert((span.start, span.end));
            } else {
                // Worst-case length is statically proven to fit inside `cap`!
                self.proven_infallible_spans.insert((span.start, span.end));
            }
        }

        ExprVal {
            ty: ret_slice_ty,
            error_spec: fmt_err_spec,
            prov: ret_prov,
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }

    pub(super) fn check_fmt_ring_call(&mut self, args: &[CallArg], span: Span) -> ExprVal {
        let fmt_err_spec = ResolvedErrorSpec::Typed(Ty::Enum("FmtErr".to_string()));
        let mut pos_args = Vec::new();
        let mut boost_prio: Option<(Priority, Span)> = None;

        for a in args {
            if let Some(arg_name) = &a.name {
                if arg_name.name == "boost" {
                    let bv = self.check_expr_infallible(&a.value);
                    if bv.ty != Ty::Priority && !bv.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            a.span,
                            format!(
                                "'boost:' must be a priority (@critical..@low), found '{}'",
                                bv.ty
                            ),
                        ));
                    }
                    if let ExprKind::Priority(prio) = a.value.kind {
                        boost_prio = Some((prio, a.span));
                    }
                } else {
                    let _ = self.check_expr_infallible(&a.value);
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        arg_name.span,
                        format!(
                            "unknown named argument '{}:' in 'fmt_ring'",
                            arg_name.name
                        ),
                    ));
                }
            } else {
                pos_args.push(a);
            }
        }

        if pos_args.len() != 2 {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "'fmt_ring' expects 2 positional arguments (tx, f\"...\" [, boost: @prio]), found {}",
                    pos_args.len()
                ),
            ));
            for a in &pos_args {
                if let ExprKind::FormatString(parts) = &a.value.kind {
                    let _ = self.check_format_string(parts);
                } else {
                    let _ = self.check_expr_infallible(&a.value);
                }
            }
            return ExprVal {
                ty: Ty::usize(),
                error_spec: fmt_err_spec,
                prov: Provenance::pure_static(),
                endpoint_name: None,
                field_endpoints: BTreeMap::new(),
                span,
            };
        }

        let ring_arg = pos_args[0];
        let fmt_arg = pos_args[1];
        let rv = self.check_expr_infallible(&ring_arg.value);
        let (is_byte_ring, ring_cap) = match &rv.ty {
            Ty::Ptr(inner) => match &**inner {
                Ty::OsPrimitive(ResolvedOsPrimitive::ByteRing { cap }) => (true, Some(*cap)),
                Ty::OsPrimitive(ResolvedOsPrimitive::ByteRingUnsized) => (true, None),
                Ty::Error => (false, None),
                _ => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        ring_arg.span,
                        format!(
                            "'fmt_ring' first argument must be '*byte_ring<N>' or '*byte_ring', found '{}'",
                            rv.ty
                        ),
                    ));
                    (false, None)
                }
            },
            Ty::Error => (false, None),
            _ => {
                let mut diag = Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    ring_arg.span,
                    format!(
                        "'fmt_ring' first argument must be '*byte_ring<N>' or '*byte_ring', found '{}'",
                        rv.ty
                    ),
                );
                if matches!(
                    rv.ty,
                    Ty::OsPrimitive(ResolvedOsPrimitive::ByteRing { .. })
                ) {
                    diag = diag.with_note("pass `&ring` or a `*byte_ring` view pointer");
                }
                self.sema.emit(diag);
                (false, None)
            }
        };

        if is_byte_ring {
            let source_key = rv
                .endpoint_name
                .clone()
                .or_else(|| root_ident_of(&ring_arg.value))
                .unwrap_or_else(|| "<anon_endpoint>".to_string());
            if let Some((prio, boost_span)) = boost_prio {
                self.recorded_boosts.push(PriorityBoostSite {
                    primitive_name: source_key.clone(),
                    boost_priority: prio,
                    span: boost_span,
                });
            }
            self.record_byte_ring_op(ByteRingOpRole::Producer, "fmt_ring", &source_key, span);
            let wkey = format!("{source_key}#write");
            let _ = self.leases.invalidate(&wkey, span, "fmt_ring");
        }

        let min_len = if let ExprKind::FormatString(parts) = &fmt_arg.value.kind {
            let _ = self.check_format_string(parts);
            self.min_format_string_len(parts)
        } else {
            let fv = self.check_expr_infallible(&fmt_arg.value);
            if !fv.ty.is_error() {
                let _ = self.check_format_hole(
                    &fmt_arg.value,
                    &fv.ty,
                    FormatSpec::Default,
                    fmt_arg.span,
                );
            }
            self.min_format_value_len(&fv.ty, &FormatSpec::Default, &fmt_arg.value)
        };

        if let Some(cap) = ring_cap {
            if min_len > cap {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::FormatBufferTooSmall,
                        span,
                        format!(
                            "format output minimum length ({} bytes) exceeds byte_ring<{}> capacity ({} bytes)",
                            min_len, cap, cap
                        ),
                    )
                    .with_note(format!(
                        "increase the byte_ring capacity to at least `byte_ring<{}>`, or use an unsized `*byte_ring` view if capacity varies",
                        min_len
                    )),
                );
            }
        }

        ExprVal {
            ty: Ty::usize(),
            error_spec: fmt_err_spec,
            prov: Provenance::pure_static(),
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }

    pub(super) fn min_format_string_len(&self, parts: &[FormatPart]) -> usize {
        let mut total = 0usize;
        for part in parts {
            let step = match part {
                FormatPart::Literal(s) => zev_ast::decoded_string_byte_len(s),
                FormatPart::Hole { expr, spec, .. } => {
                    let Some(ty) = self.structural_expr_ty(expr).or_else(|| match &expr.kind {
                        ExprKind::Literal(Literal::String(_)) => Some(Ty::Str),
                        ExprKind::Literal(Literal::Int(_)) => Some(Ty::IntLit),
                        ExprKind::Literal(Literal::Float { .. }) => Some(Ty::FloatLit),
                        ExprKind::Literal(Literal::Bool(_)) => Some(Ty::bool()),
                        _ => None,
                    }) else {
                        continue;
                    };
                    self.min_format_value_len(&ty, spec, expr)
                }
            };
            total = total.saturating_add(step);
        }
        total
    }

    pub(super) fn min_format_value_len(&self, ty: &Ty, spec: &FormatSpec, expr: &Expr) -> usize {
        if let ExprKind::Literal(Literal::String(s)) = &expr.kind {
            return zev_ast::decoded_string_byte_len(s);
        }
        if ty.is_bool() {
            return if let ExprKind::Literal(Literal::Bool(b)) = &expr.kind {
                if *b { 4 } else { 5 }
            } else {
                4
            };
        }
        if ty.is_integer() {
            return match spec {
                FormatSpec::Hex { width, .. } | FormatSpec::Bin { width } => {
                    width.unwrap_or(1).max(1) as usize
                }
                _ => match &expr.kind {
                    ExprKind::Literal(Literal::Int(n)) => {
                        if *n == 0 {
                            1
                        } else {
                            n.ilog10() as usize + 1
                        }
                    }
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        expr: inner,
                    } => {
                        if let ExprKind::Literal(Literal::Int(n)) = &inner.kind {
                            let d = if *n == 0 { 1 } else { n.ilog10() as usize + 1 };
                            d + 1
                        } else {
                            1
                        }
                    }
                    _ => 1,
                },
            };
        }
        if ty.is_float() {
            return match spec {
                FormatSpec::Fixed { precision: p } => {
                    if *p == 0 {
                        1
                    } else {
                        2 + (*p as usize)
                    }
                }
                FormatSpec::Sci { .. } => 3,
                _ => 1,
            };
        }
        if let Ty::Enum(en_name) = ty {
            return self
                .sema
                .enums
                .get(en_name)
                .and_then(|en| en.variants.iter().map(|v| v.len()).min())
                .unwrap_or(1);
        }
        match (ty, spec) {
            (Ty::Array { elem, len }, FormatSpec::Default) if **elem == Ty::u8() => *len,
            (Ty::Struct(_) | Ty::Slice(_) | Ty::Array { .. }, FormatSpec::Debug) => {
                let mut visiting = BTreeSet::new();
                self.debug_min_case_ty(ty, &mut visiting)
            }
            (Ty::Ptr(_), FormatSpec::Debug) => 3,
            _ => 0,
        }
    }

    pub(super) fn debug_min_case_ty(&self, ty: &Ty, visiting: &mut BTreeSet<String>) -> usize {
        match ty {
            Ty::Prim(PrimType::Bool) => 4,
            Ty::Prim(p) if p.is_integer() => 1,
            Ty::Prim(p) if p.is_float() => 1,
            Ty::IntLit | Ty::FloatLit => 1,
            Ty::Enum(en_name) => self
                .sema
                .enums
                .get(en_name)
                .and_then(|en| en.variants.iter().map(|v| v.len()).min())
                .unwrap_or(1),
            Ty::Struct(st_name) => {
                if !visiting.insert(st_name.clone()) {
                    return 0;
                }
                let Some(st_info) = self.sema.structs.get(st_name) else {
                    visiting.remove(st_name);
                    return 0;
                };
                if st_info.fields.is_empty() {
                    visiting.remove(st_name);
                    return st_name.len() + 4;
                }
                let mut acc = st_name.len() + 3;
                for (i, field) in st_info.fields.iter().enumerate() {
                    if i > 0 {
                        acc = acc.saturating_add(2);
                    }
                    acc = acc.saturating_add(field.name.len() + 2);
                    acc = acc.saturating_add(self.debug_min_case_ty(&field.ty, visiting));
                }
                visiting.remove(st_name);
                acc.saturating_add(2)
            }
            Ty::Array { elem, len } => {
                if *len == 0 {
                    2
                } else {
                    2usize
                        .saturating_add(2 * (*len - 1))
                        .saturating_add(len.saturating_mul(self.debug_min_case_ty(elem, visiting)))
                }
            }
            Ty::Slice(_) => 2,
            Ty::Ptr(_) => 3,
            _ => 0,
        }
    }

    pub(super) fn static_dst_capacity(&self, dst_expr: &Expr, dst_ty: &Ty) -> Option<usize> {
        if matches!(&dst_expr.kind, ExprKind::Slice { .. }) {
            return self.structural_u8_capacity(dst_expr);
        }

        match dst_ty {
            Ty::Array { elem, len } if **elem == Ty::u8() => Some(*len),
            Ty::Ptr(inner) => match &**inner {
                Ty::Array { elem, len } if **elem == Ty::u8() => Some(*len),
                _ => None,
            },
            _ => None,
        }
    }

    pub(super) fn structural_u8_capacity(&self, expr: &Expr) -> Option<usize> {
        if let ExprKind::Slice { target, start, end } = &expr.kind {
            let base_cap = self.structural_u8_capacity(target)?;
            let s_raw = match start {
                None => 0usize,
                Some(s) => self.eval_const_usize_expr(s)?,
            };
            let e_raw = match end {
                None => base_cap,
                Some(e) => self.eval_const_usize_expr(e)?,
            };
            let e = e_raw.min(base_cap);
            let s = s_raw.min(e);
            return Some(e - s);
        }

        match self.structural_expr_ty(expr)? {
            Ty::Array { elem, len } if *elem == Ty::u8() => Some(len),
            Ty::Ptr(inner) => match *inner {
                Ty::Array { elem, len } if *elem == Ty::u8() => Some(len),
                _ => None,
            },
            _ => None,
        }
    }

    pub(super) fn structural_expr_ty(&self, expr: &Expr) -> Option<Ty> {
        match &expr.kind {
            ExprKind::Ident(id) => {
                if let Some(var) = self.lookup_local(&id.name) {
                    return Some(var.ty.clone());
                }
                if let Some(c) = self.sema.consts.get(&id.name) {
                    return Some(c.ty.clone());
                }
                None
            }
            ExprKind::Field { target, field } => {
                let base_ty = match self.structural_expr_ty(target)? {
                    Ty::Ptr(inner) => *inner,
                    other => other,
                };
                if let Ty::Struct(st_name) = base_ty {
                    return self
                        .sema
                        .structs
                        .get(&st_name)
                        .and_then(|st| st.field(&field.name))
                        .map(|f| f.ty.clone());
                }
                None
            }
            ExprKind::Index { target, .. } => match self.structural_expr_ty(target)? {
                Ty::Array { elem, .. } | Ty::Slice(elem) => Some(*elem),
                Ty::Ptr(inner) => match *inner {
                    Ty::Array { elem, .. } | Ty::Slice(elem) => Some(*elem),
                    _ => None,
                },
                _ => None,
            },
            ExprKind::Slice { target, .. } => match self.structural_expr_ty(target)? {
                Ty::Array { elem, .. } | Ty::Slice(elem) => Some(Ty::Slice(elem)),
                Ty::Ptr(inner) => match *inner {
                    Ty::Array { elem, .. } | Ty::Slice(elem) => Some(Ty::Slice(elem)),
                    _ => None,
                },
                _ => None,
            },
            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: inner,
            } => match self.structural_expr_ty(inner)? {
                Ty::Ptr(pointee) => Some(*pointee),
                _ => None,
            },
            ExprKind::Unary {
                op: UnaryOp::AddrOf,
                expr: inner,
            } => Some(Ty::Ptr(Box::new(self.structural_expr_ty(inner)?))),
            _ => None,
        }
    }

    pub(super) fn eval_const_usize_expr(&self, expr: &Expr) -> Option<usize> {
        let shadowed = self.active_local_names();
        self.sema
            .eval_const_expr_i128_shadowed(expr, &mut BTreeSet::new(), &shadowed)
            .and_then(|v| usize::try_from(v).ok())
    }

    /// Typecheck all holes in an `f"..."` format string and compute its exact worst-case byte width
    /// (`Some(bytes)` when every hole is statically bounded, or `None` when any hole is a dynamic slice/string).
    pub(super) fn check_format_string(&mut self, parts: &[FormatPart]) -> Option<usize> {
        self.check_format_string_with_dst(parts, None)
    }

    pub(super) fn check_format_string_with_dst(
        &mut self,
        parts: &[FormatPart],
        alias_dst: Option<&Provenance>,
    ) -> Option<usize> {
        let mut worst_total: Option<usize> = Some(0);
        for part in parts {
            match part {
                FormatPart::Literal(s) => {
                    let byte_len = zev_ast::decoded_string_byte_len(s);
                    worst_total = worst_total.and_then(|acc| acc.checked_add(byte_len));
                }
                FormatPart::Hole {
                    expr,
                    spec,
                    span: hole_span,
                } => {
                    let hv = self.check_expr_infallible(expr);
                    if let Some(dst_prov) = alias_dst {
                        if !hv.ty.is_error()
                            && (hv.ty.contains_reference(&self.sema.structs)
                                || matches!(hv.ty, Ty::Array { .. }))
                        {
                            let mut hole_prov = hv.prov.clone();
                            if let Some(root) = root_ident_of(expr) {
                                if let Some(var) = self.lookup_local(&root) {
                                    if var.ty.contains_reference(&self.sema.structs)
                                        || matches!(var.ty, Ty::Array { .. })
                                    {
                                        hole_prov = hole_prov.merge(&Provenance::addr_of_local(
                                            &var.name,
                                            var.decl_depth,
                                        ));
                                    }
                                }
                            }
                            if !dst_prov.local_origins.is_disjoint(&hole_prov.local_origins)
                                || !dst_prov.param_origins.is_disjoint(&hole_prov.param_origins)
                            {
                                self.sema.emit(
                                    Diagnostic::new(
                                        DiagnosticKind::InvalidOperation,
                                        *hole_span,
                                        "cannot format slice/buffer into the same destination buffer in 'fmt_buf' (in-place aliasing would corrupt output)",
                                    )
                                    .with_note(
                                        "format into a separate `[N]u8` scratch buffer first, then copy or use the resulting slice",
                                    ),
                                );
                            }
                        }
                    }
                    let hw = self.check_format_hole(expr, &hv.ty, *spec, *hole_span);
                    worst_total = match (worst_total, hw) {
                        (Some(acc), Some(w)) => acc.checked_add(w),
                        _ => None,
                    };
                }
            }
        }
        worst_total
    }

    pub(super) fn check_format_hole(
        &mut self,
        expr: &Expr,
        ty: &Ty,
        spec: FormatSpec,
        span: Span,
    ) -> Option<usize> {
        if ty.is_error() {
            return None;
        }

        // 1. String literal hole (`"ok"`).
        if let ExprKind::Literal(Literal::String(s)) = &expr.kind {
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => {
                    Some(zev_ast::decoded_string_byte_len(s))
                }
                _ => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        "numeric format specifier cannot be applied to a string literal",
                    ));
                    None
                }
            };
        }

        // 2. Boolean (`true` / `false` -> max 5 bytes).
        if ty.is_bool() {
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => Some(5),
                _ => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        "numeric format specifier cannot be applied to 'bool'",
                    ));
                    None
                }
            };
        }

        // 3. Integers (`u8`..`u64`, `i8`..`i64`, `usize`, `isize`, endian ints, `IntLit`).
        if ty.is_integer() {
            let int_prim = match ty {
                Ty::Prim(p) => *p,
                Ty::IntLit => match &expr.kind {
                    ExprKind::Literal(Literal::Int(v)) if *v > i32::MAX as u64 => PrimType::U64,
                    _ => PrimType::I32,
                },
                _ => PrimType::I32,
            };
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => Some(int_decimal_worst_case(int_prim)),
                FormatSpec::Hex { width, .. } => {
                    let w = width.unwrap_or(0) as usize;
                    Some(int_hex_worst_case(int_prim).max(w))
                }
                FormatSpec::Bin { width } => {
                    let w = width.unwrap_or(0) as usize;
                    Some(int_bin_worst_case(int_prim).max(w))
                }
                FormatSpec::Fixed { .. } | FormatSpec::Sci { .. } => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "floating-point format specifier requires 'f32' or 'f64', found '{}'",
                            ty
                        ),
                    ));
                    None
                }
            };
        }

        // 4. Floating-point (`f32`, `f64`, `FloatLit`).
        if ty.is_float() {
            let is_f32 = matches!(ty, Ty::Prim(PrimType::F32));
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => Some(14),
                FormatSpec::Fixed { precision: p } => {
                    if p > 9 {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "unsupported float format precision '.{}'; supported float precision is 0..=9",
                                p
                            ),
                        ));
                        None
                    } else {
                        Some(if p == 0 { 11 } else { 12 + p as usize })
                    }
                }
                FormatSpec::Sci { precision } => {
                    let p_u8 = precision.unwrap_or(2);
                    if p_u8 > 9 {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "unsupported scientific float format precision '.{}e'; supported float precision is 0..=9",
                                p_u8
                            ),
                        ));
                        None
                    } else {
                        let p = p_u8 as usize;
                        Some(if is_f32 { 9 + p } else { 10 + p })
                    }
                }
                FormatSpec::Hex { .. } | FormatSpec::Bin { .. } => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "hex/binary format specifier requires an integer type, found '{}'",
                            ty
                        ),
                    ));
                    None
                }
            };
        }

        // 5. Enum variant names (`{err}` or `{err:?}`), bounded by `<EnumName>(?)` fallback.
        if let Ty::Enum(enum_name) = ty {
            return match spec {
                FormatSpec::Default | FormatSpec::Debug => {
                    self.sema.enums.get(enum_name).map(|en| {
                        let variants_max = en.variants.iter().map(|v| v.len()).max().unwrap_or(0);
                        variants_max.max(enum_name.len() + 3)
                    })
                }
                _ => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "numeric format specifier cannot be applied to enum '{}'",
                            enum_name
                        ),
                    ));
                    None
                }
            };
        }

        // 6. Byte strings/slices, structs under `:?`, and uniform pointer addresses under `:?`.
        match (ty, spec) {
            (Ty::Str, FormatSpec::Default | FormatSpec::Debug) => None,
            (Ty::Slice(elem), FormatSpec::Default) if **elem == Ty::u8() => None,
            (Ty::Ptr(inner), FormatSpec::Default)
                if matches!(**inner, Ty::Prim(PrimType::U8 | PrimType::I8)) =>
            {
                None
            }
            (Ty::Array { elem, len }, FormatSpec::Default) if **elem == Ty::u8() => Some(*len),

            (Ty::Struct(st_name), FormatSpec::Debug) => {
                let mut visiting = BTreeSet::new();
                self.debug_worst_case_ty(&Ty::Struct(st_name.clone()), span, &mut visiting)
            }
            (Ty::Ptr(_), FormatSpec::Debug) => Some(18),
            (Ty::Struct(st_name), FormatSpec::Default) => {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "struct '{}' requires debug specifier '{{expr:?}}' or explicit field formatting",
                            st_name
                        ),
                    )
                    .with_note(format!(
                        "use `{{val:?}}` for compile-time struct debug expansion or format fields explicitly (`f\"{{val.field}}\"`)"
                    )),
                );
                None
            }
            (Ty::Slice(_) | Ty::Array { .. }, FormatSpec::Debug) => {
                let mut visiting = BTreeSet::new();
                self.debug_worst_case_ty(ty, span, &mut visiting)
            }
            _ => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    span,
                    format!(
                        "type '{}' cannot be formatted with specifier '{:?}' in 'f\"...\"'",
                        ty, spec
                    ),
                ));
                None
            }
        }
    }

    pub(super) fn debug_worst_case_ty(
        &mut self,
        ty: &Ty,
        span: Span,
        visiting: &mut BTreeSet<String>,
    ) -> Option<usize> {
        match ty {
            Ty::Error => None,
            Ty::Prim(PrimType::Bool) => Some(5),
            Ty::Prim(p) if p.is_integer() => Some(int_decimal_worst_case(*p)),
            Ty::Prim(p) if p.is_float() => Some(14),
            Ty::IntLit => Some(11),
            Ty::FloatLit => Some(14),
            Ty::Str => None,
            Ty::Enum(en_name) => self.sema.enums.get(en_name).map(|en| {
                let variants_max = en.variants.iter().map(|v| v.len()).max().unwrap_or(0);
                variants_max.max(en_name.len() + 3)
            }),
            Ty::Struct(st_name) => {
                if !visiting.insert(st_name.clone()) {
                    return None;
                }
                let Some(st_info) = self.sema.structs.get(st_name).cloned() else {
                    visiting.remove(st_name);
                    return None;
                };
                if st_info.fields.is_empty() {
                    visiting.remove(st_name);
                    // `"Name { }"` (matches `zev-mir` `expand_struct_debug_be`)
                    return Some(st_name.len() + 4);
                }
                // `"Name { f1: <v1>, f2: <v2> }"`
                let mut acc: Option<usize> = Some(st_name.len() + 3); // `"Name { "`
                for (i, field) in st_info.fields.iter().enumerate() {
                    if i > 0 {
                        acc = acc.and_then(|a| a.checked_add(2)); // `", "`
                    }
                    acc = acc.and_then(|a| a.checked_add(field.name.len() + 2)); // `"field: "`
                    let fw = self.debug_worst_case_ty(&field.ty, span, visiting);
                    acc = match (acc, fw) {
                        (Some(a), Some(w)) => a.checked_add(w),
                        _ => None,
                    };
                }
                visiting.remove(st_name);
                acc.and_then(|a| a.checked_add(2)) // `" }"`
            }
            Ty::Slice(elem) => {
                if matches!(&**elem, Ty::Struct(_) | Ty::Slice(_) | Ty::Array { .. }) {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "dynamic slice '{}' of composite elements cannot be formatted with ':?'",
                                ty
                            ),
                        )
                        .with_note(
                            "iterate over the slice explicitly (`for (Elem x in slice) { println(f\"{x:?}\"); }`) or format a fixed-size `[N]T` array",
                        ),
                    );
                    return None;
                }
                let _ = self.debug_worst_case_ty(elem, span, visiting);
                None
            }
            Ty::Array { elem, len } => {
                let ew = self.debug_worst_case_ty(elem, span, visiting)?;
                let items = ew.checked_mul(*len)?;
                let seps = len.saturating_sub(1).checked_mul(2)?;
                items.checked_add(seps)?.checked_add(2)
            }
            Ty::Ptr(_) => {
                // Pointer addresses formatted uniformly as hex (`0x` + up to 16 hex digits).
                Some(18)
            }
            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    span,
                    format!("field/element of type '{}' cannot be formatted with ':?'", other),
                ));
                None
            }
        }
    }
}
