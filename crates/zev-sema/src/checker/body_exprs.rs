use super::*;

type AddrOfPlaceResult = (
    ExprVal,
    Provenance,
    Option<String>,
    Option<(String, String, usize)>,
);

impl<'a> BodyChecker<'a> {
    pub(super) fn take_mmio_array_tag(&mut self, span: Span) -> Option<(String, String, usize)> {
        self.mmio_array_expr_spans.remove(&(span.start, span.end))
    }

    pub(super) fn reject_unindexed_mmio_array(&mut self, span: Span) -> bool {
        if let Some((layout_name, reg_name, _)) = self.take_mmio_array_tag(span) {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                span,
                format!(
                    "cannot use unindexed MMIO register array '{}.{}'; index an element with '[i]' or read '.len'",
                    layout_name, reg_name
                ),
            ));
            true
        } else {
            false
        }
    }

    pub(super) fn check_expr_infallible_allow_mmio_array(
        &mut self,
        expr: &Expr,
    ) -> (ExprVal, Option<(String, String, usize)>) {
        let mut val = self.check_expr_raw(expr);
        let mmio_arr = self.take_mmio_array_tag(expr.span);
        if val.error_spec.is_fallible() {
            if self
                .proven_infallible_spans
                .remove(&(expr.span.start, expr.span.end))
            {
                val.error_spec = ResolvedErrorSpec::None;
                return (val, mmio_arr);
            }
            let note = match (&val.error_spec, val.ty.is_void()) {
                (ResolvedErrorSpec::Typed(err_ty), false) => format!(
                    "unwrap with `expr or <fallback>` (e.g. `{} x = expr or default_val;`) or `expr catch ({} err) {{ ... }}` / `expr catch ({} err, {} partial) {{ ... }}`",
                    val.ty, err_ty, err_ty, val.ty
                ),
                (ResolvedErrorSpec::Typed(err_ty), true) => format!(
                    "handle failure with `expr catch ({} err) {{ ... }};`",
                    err_ty
                ),
                (_, false) => format!(
                    "unwrap with `expr or <fallback>` (e.g. `{} x = expr or default_val;`) or `expr catch {{ ... }}`",
                    val.ty
                ),
                (_, true) => "handle failure with `expr catch { ... };`".to_string(),
            };
            self.sema.emit(
                Diagnostic::new(
                    DiagnosticKind::UnhandledFallibleExpr,
                    expr.span,
                    "unhandled fallible expression (may exit via 'error'); handle failure explicitly with 'catch' or 'or'",
                )
                .with_note(note),
            );
            val.error_spec = ResolvedErrorSpec::None;
        }
        (val, mmio_arr)
    }

    pub(super) fn check_expr_infallible(&mut self, expr: &Expr) -> ExprVal {
        let (val, mmio_arr) = self.check_expr_infallible_allow_mmio_array(expr);
        if let Some((layout_name, reg_name, _)) = mmio_arr {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                expr.span,
                format!(
                    "cannot use unindexed MMIO register array '{}.{}'; index an element with '[i]' or read '.len'",
                    layout_name, reg_name
                ),
            ));
            return ExprVal::error(expr.span);
        }
        val
    }

    pub(super) fn check_expr_raw(&mut self, expr: &Expr) -> ExprVal {
        let val = self.check_expr_raw_inner(expr);
        self.expr_tys
            .insert(expr as *const Expr as usize, val.ty.clone());
        val
    }

    fn check_expr_raw_inner(&mut self, expr: &Expr) -> ExprVal {
        match &expr.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Int {
                    value,
                    suffix: Some(p),
                } => {
                    self.check_int_lit_range_or_emit(i128::from(*value), *p, expr.span);
                    ExprVal::scalar(Ty::Prim(*p), expr.span)
                }
                Literal::Int { suffix: None, .. } => ExprVal::scalar(Ty::IntLit, expr.span),
                Literal::Float {
                    suffix: Some(p), ..
                } => ExprVal::scalar(Ty::Prim(*p), expr.span),
                Literal::Float { suffix: None, .. } => ExprVal::scalar(Ty::FloatLit, expr.span),
                Literal::Bool(_) => ExprVal::scalar(Ty::bool(), expr.span),
                Literal::String(_) => ExprVal::scalar(Ty::Str, expr.span),
                Literal::Duration(dur) => {
                    self.sema.validate_duration(dur);
                    ExprVal::scalar(Ty::Duration, expr.span)
                }
            },

            ExprKind::Priority(_) => ExprVal::scalar(Ty::Priority, expr.span),

            ExprKind::Ident(ident) => self.check_ident_expr(ident),

            ExprKind::Unary { op, expr: inner } => self.check_unary_expr(*op, inner, expr.span),

            ExprKind::Binary { op, lhs, rhs } => self.check_binary_expr(*op, lhs, rhs, expr.span),

            ExprKind::Cast { expr: inner, ty } => {
                let iv = self.check_expr_infallible(inner);
                self.validate_int_expr_at_type(inner, None);
                let target_ty = self.sema.resolve_type(ty);
                self.sema
                    .check_type_no_byval_mmio(&target_ty, ty.span, "cast target type");
                if !iv.ty.is_error()
                    && !target_ty.is_error()
                    && ty_contains_byval_mmio(&target_ty).is_none()
                    && !is_valid_cast(&iv.ty, &target_ty)
                {
                    let kind = if iv.ty.contains_mmio() || target_ty.contains_mmio() {
                        DiagnosticKind::MmioViolation
                    } else {
                        DiagnosticKind::TypeMismatch
                    };
                    self.sema.emit(Diagnostic::new(
                        kind,
                        expr.span,
                        format!("cannot cast value of type '{}' to '{}'", iv.ty, target_ty),
                    ));
                }
                let mut prov = iv.prov;
                if !target_ty.contains_reference(&self.sema.structs)
                    && !target_ty.contains_fn_ptr(&self.sema.structs)
                {
                    prov = prov.into_scalar_value();
                }
                ExprVal {
                    ty: target_ty,
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name: None,
                    field_endpoints: BTreeMap::new(),
                    span: expr.span,
                }
            }

            ExprKind::Field { target, field } => self.check_field_expr(target, field, expr.span),

            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                if !generic_args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::GenericInstantiationError,
                        expr.span,
                        "unmonomorphized generic call arguments",
                    ));
                }
                self.check_call_expr(callee, args, expr.span)
            }

            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => self.check_method_call(receiver, method, args, expr.span),

            ExprKind::Index { target, index } => {
                let (tv, mmio_arr) = self.check_expr_infallible_allow_mmio_array(target);
                let iv = self.check_expr_infallible(index);
                self.validate_int_expr_at_type(index, Some(PrimType::Usize));
                if !iv.ty.is_integer() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        index.span,
                        format!("index must be an integer, found '{}'", iv.ty),
                    ));
                }
                if let Some((layout_name, reg_name, len)) = mmio_arr {
                    let shadowed = self.active_local_names();
                    if let Some(idx) = self.sema.eval_const_expr_i128_shadowed(
                        index,
                        &mut BTreeSet::new(),
                        &shadowed,
                    ) {
                        if idx < 0 || idx as u128 >= len as u128 {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::MmioViolation,
                                index.span,
                                format!(
                                    "MMIO register array '{}.{}' index {} out of bounds (length {})",
                                    layout_name, reg_name, idx, len
                                ),
                            ));
                        }
                    }
                }
                match &tv.ty {
                    Ty::Slice(elem) | Ty::Array { elem, .. } => {
                        let elem_ty = (**elem).clone();
                        let prov = if elem_ty.contains_reference(&self.sema.structs) {
                            tv.prov
                        } else {
                            Provenance::pure_static()
                        };
                        ExprVal {
                            ty: elem_ty,
                            error_spec: ResolvedErrorSpec::None,
                            prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span: expr.span,
                        }
                    }
                    _ => {
                        if !tv.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                target.span,
                                format!("cannot index into type '{}'", tv.ty),
                            ));
                        }
                        ExprVal::error(expr.span)
                    }
                }
            }

            ExprKind::Slice { target, start, end } => {
                if let Some(s) = start {
                    let sv = self.check_expr_infallible(s);
                    self.validate_int_expr_at_type(s, Some(PrimType::Usize));
                    if !sv.ty.is_integer() && !sv.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            s.span,
                            format!("slice start index must be an integer, found '{}'", sv.ty),
                        ));
                    }
                }
                if let Some(e) = end {
                    let ev = self.check_expr_infallible(e);
                    self.validate_int_expr_at_type(e, Some(PrimType::Usize));
                    if !ev.ty.is_integer() && !ev.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            e.span,
                            format!("slice end index must be an integer, found '{}'", ev.ty),
                        ));
                    }
                }
                let (tv, place_prov, _) = self.check_addr_of_place(target, expr.span);
                match &tv.ty {
                    // Sub-slice of an existing slice `buf[0..5]` inherits `buf`'s exact scope depth and DMA lease!
                    Ty::Slice(elem) => ExprVal {
                        ty: Ty::Slice(elem.clone()),
                        error_spec: ResolvedErrorSpec::None,
                        prov: tv.prov,
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span: expr.span,
                    },
                    // Slicing a fixed array `arr[0..5]` / `s.arr[0..5]` / `(*p).arr[0..5]` preserves backing provenance!
                    Ty::Array { elem, .. } => ExprVal {
                        ty: Ty::Slice(elem.clone()),
                        error_spec: ResolvedErrorSpec::None,
                        prov: place_prov,
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span: expr.span,
                    },
                    Ty::Ptr(inner) => match &**inner {
                        Ty::Array { elem, .. } => ExprVal {
                            ty: Ty::Slice(elem.clone()),
                            error_spec: ResolvedErrorSpec::None,
                            prov: tv.prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span: expr.span,
                        },
                        _ => {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                target.span,
                                format!("cannot slice pointer type '{}'", tv.ty),
                            ));
                            ExprVal::error(expr.span)
                        }
                    },
                    _ => {
                        if !tv.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                target.span,
                                format!("cannot take sub-slice of type '{}'", tv.ty),
                            ));
                        }
                        ExprVal::error(expr.span)
                    }
                }
            }

            ExprKind::StructLit {
                module: _,
                name,
                generic_args,
                fields,
            } => {
                if !generic_args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::GenericInstantiationError,
                        expr.span,
                        format!("unmonomorphized generic struct literal '{}'", name.name),
                    ));
                }
                let Some(st_info) = self.sema.structs.get(&name.name).cloned() else {
                    let struct_names: Vec<&str> =
                        self.sema.structs.keys().map(String::as_str).collect();
                    let mut diag = Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        name.span,
                        format!("unknown struct '{}'", name.name),
                    );
                    if let Some(best) = closest_match(&name.name, struct_names, 2) {
                        diag = diag.with_note(format!("did you mean struct `{}`?", best));
                    }
                    self.sema.emit(diag);
                    for fi in fields {
                        let _ = self.check_expr_infallible(&fi.value);
                    }
                    return ExprVal::error(expr.span);
                };

                if st_info.is_subsystem {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        expr.span,
                        format!(
                            "cannot construct subsystem struct '{}' via struct literal; declare a top-level static instance instead",
                            name.name
                        ),
                    ));
                    for fi in fields {
                        let _ = self.check_expr_infallible(&fi.value);
                    }
                    return ExprVal::scalar(Ty::Struct(name.name.clone()), expr.span);
                }

                let mut seen = BTreeSet::new();
                let mut combined_prov = Provenance::pure_static();
                let mut per_field_prov = BTreeMap::new();
                let mut field_endpoints = BTreeMap::new();

                for fi in fields {
                    if !seen.insert(fi.name.name.clone()) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::DuplicateDefinition,
                            fi.name.span,
                            format!(
                                "duplicate field '{}' in '{}' struct literal",
                                fi.name.name, name.name
                            ),
                        ));
                    }
                    let fv = self.check_expr_infallible(&fi.value);
                    if let Some(ep) = &fv.endpoint_name {
                        field_endpoints.insert(fi.name.name.clone(), ep.clone());
                    }
                    for (sub_k, sub_ep) in &fv.field_endpoints {
                        field_endpoints
                            .insert(format!("{}.{}", fi.name.name, sub_k), sub_ep.clone());
                    }
                    if let Some(decl_field) = st_info.field(&fi.name.name) {
                        if self.sema.check_assignable_or_emit(
                            &fv.ty,
                            &decl_field.ty,
                            fi.value.span,
                            DiagnosticKind::TypeMismatch,
                            || {
                                format!(
                                    "field '{}.{}' expects '{}', found '{}'",
                                    name.name, fi.name.name, decl_field.ty, fv.ty
                                )
                            },
                        ) {
                            self.check_coerced_expr(&fi.value, &decl_field.ty);
                        }
                        if decl_field.ty.contains_reference(&self.sema.structs)
                            || decl_field.ty.contains_fn_ptr(&self.sema.structs)
                        {
                            let fprov = self.coerce_array_to_slice_prov(
                                &fi.value,
                                &fv.ty,
                                &decl_field.ty,
                                &fv.prov,
                            );
                            combined_prov = combined_prov.merge(&fprov);
                            per_field_prov.insert(fi.name.name.clone(), fprov);
                        }
                    } else {
                        let hint = self.suggest_struct_field_hint(&name.name, &fi.name.name);
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::UndefinedSymbol,
                                fi.name.span,
                                format!("struct '{}' has no field '{}'", name.name, fi.name.name),
                            )
                            .with_note(hint),
                        );
                    }
                }

                self.last_struct_lit_fields
                    .insert((expr.span.start, expr.span.end), per_field_prov);

                for decl_field in &st_info.fields {
                    if !seen.contains(&decl_field.name) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            expr.span,
                            format!(
                                "missing field '{}' in '{}' struct literal",
                                decl_field.name, name.name
                            ),
                        ));
                    }
                }

                ExprVal {
                    ty: Ty::Struct(name.name.clone()),
                    error_spec: ResolvedErrorSpec::None,
                    prov: combined_prov,
                    endpoint_name: None,
                    field_endpoints,
                    span: expr.span,
                }
            }

            ExprKind::ArrayLit(elems) => {
                if elems.is_empty() {
                    return ExprVal::error(expr.span);
                }
                let mut elem_ty: Option<Ty> = None;
                let mut combined_prov = Provenance::pure_static();
                for e in elems {
                    let ev = self.check_expr_infallible(e);
                    if ev.ty.is_linear_lease() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::LinearLeaseViolation,
                            e.span,
                            "linear 'Lease<T>' handles cannot be stored inside an array literal",
                        ));
                    }
                    if ev.ty.contains_reference(&self.sema.structs) {
                        combined_prov = combined_prov.merge(&ev.prov);
                    }
                    match &elem_ty {
                        None => elem_ty = Some(ev.ty),
                        Some(cur) if matches!(cur, Ty::IntLit) && ev.ty.is_integer() => {
                            elem_ty = Some(ev.ty);
                        }
                        Some(cur) if matches!(cur, Ty::FloatLit) && ev.ty.is_float() => {
                            elem_ty = Some(ev.ty);
                        }
                        Some(cur) if cur.is_integer() && ev.ty.is_integer() => {
                            if let Some(unified) = try_unify_integer(cur, &ev.ty) {
                                elem_ty = Some(unified);
                            } else {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    e.span,
                                    format!(
                                        "array element has type '{}', expected '{}'",
                                        ev.ty, cur
                                    ),
                                ));
                            }
                        }
                        Some(cur) => {
                            let _ = self.sema.check_assignable_or_emit(
                                &ev.ty,
                                cur,
                                e.span,
                                DiagnosticKind::TypeMismatch,
                                || {
                                    format!(
                                        "array element has type '{}', expected '{}'",
                                        ev.ty, cur
                                    )
                                },
                            );
                        }
                    }
                }
                let final_elem = elem_ty.unwrap_or(Ty::Error);
                if final_elem == Ty::f32() {
                    for e in elems {
                        self.sema.check_f32_coercion_expr(e, &final_elem);
                    }
                } else if let Ty::Prim(ep) = final_elem {
                    if ep.is_integer() {
                        for e in elems {
                            self.validate_int_expr_at_type(e, Some(Ty::host_prim(ep)));
                        }
                    }
                } else if matches!(final_elem, Ty::IntLit) {
                    for e in elems {
                        self.validate_int_expr_at_type(e, None);
                    }
                }
                ExprVal {
                    ty: Ty::Array {
                        elem: Box::new(final_elem),
                        len: elems.len(),
                    },
                    error_spec: ResolvedErrorSpec::None,
                    prov: combined_prov,
                    endpoint_name: None,
                    field_endpoints: BTreeMap::new(),
                    span: expr.span,
                }
            }

            ExprKind::FormatString(parts) => {
                let _ = self.check_format_string(parts);
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        expr.span,
                        "zero-heap format string 'f\"...\"' cannot allocate a standalone string value",
                    )
                    .with_note(
                        "pass `f\"...\"` directly into a sink: `println(f\"...\");`, `print(f\"...\");`, or `fmt_buf(dst, f\"...\")`",
                    ),
                );
                ExprVal::error(expr.span)
            }

            ExprKind::Or {
                expr: inner,
                fallback,
            } => {
                let iv = self.check_expr_raw(inner);
                let _ = self.reject_unindexed_mmio_array(inner.span);
                let _ = self
                    .proven_infallible_spans
                    .remove(&(inner.span.start, inner.span.end));
                if !iv.error_spec.is_fallible() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOrFallback,
                        expr.span,
                        "cannot apply 'or' fallback to an infallible expression",
                    ));
                }
                let base_vars = self.snapshot_outer_vars();
                let base_leases = self.leases.clone();
                let fv = self.check_expr_raw(fallback);
                let _ = self.reject_unindexed_mmio_array(fallback.span);
                if fv.ty.is_linear_lease() {
                    self.consume_lease_value(fallback, &fv, "`or` fallback", fallback.span);
                }
                let fallback_vars = self.snapshot_outer_vars();
                let fallback_leases = self.leases.clone();
                self.scopes = base_vars.clone();
                self.merge_branch_vars(&base_vars, &[base_vars.clone(), fallback_vars], expr.span);
                self.leases = base_leases.clone();
                self.leases.join_branches(&[base_leases, fallback_leases]);
                let int_unified = if iv.ty.is_integer() && fv.ty.is_integer() {
                    try_unify_integer(&iv.ty, &fv.ty)
                } else {
                    None
                };
                if !fv.ty.is_assignable_to(&iv.ty)
                    && !iv.ty.is_assignable_to(&fv.ty)
                    && int_unified.is_none()
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOrFallback,
                        fallback.span,
                        format!("'or' fallback has type '{}', expected '{}'", fv.ty, iv.ty),
                    ));
                }
                let res_ty = if let Some(u) = int_unified {
                    u
                } else if matches!(iv.ty, Ty::IntLit | Ty::FloatLit) {
                    fv.ty.clone()
                } else {
                    iv.ty.clone()
                };
                self.check_coerced_expr(inner, &res_ty);
                self.check_coerced_expr(fallback, &res_ty);
                let prov = if res_ty.contains_reference(&self.sema.structs) {
                    iv.prov.merge(&fv.prov)
                } else {
                    Provenance::pure_static()
                };
                ExprVal {
                    ty: res_ty,
                    error_spec: fv.error_spec,
                    prov,
                    endpoint_name: None,
                    field_endpoints: BTreeMap::new(),
                    span: expr.span,
                }
            }

            ExprKind::Catch {
                expr: inner,
                clause,
            } => self.check_catch_expr(inner, clause, expr.span),
        }
    }

    pub(super) fn check_ident_expr(&mut self, ident: &Ident) -> ExprVal {
        // 1. Local variable or parameter.
        if let Some(var) = self.lookup_local(&ident.name).cloned() {
            // Check for invalidated DMA / `byte_ring` lease on read!
            if let Some((stale_tok, inv_span)) = self.leases.find_stale_lease(&var.prov) {
                let inv_method = self.leases.last_invalidator(&stale_tok.source);
                let display_src = stale_tok
                    .source
                    .strip_suffix("#write")
                    .unwrap_or(&stale_tok.source);
                let mut diag = if stale_tok.acq_method == "wait_chunk" {
                    let mut d = Diagnostic::new(
                        DiagnosticKind::StaleDmaLease,
                        ident.span,
                        format!(
                            "cannot read '{}' derived from '{}.wait_chunk()' after a subsequent 'wait_chunk()' invalidated its DMA buffer lease",
                            ident.name, stale_tok.source
                        ),
                    );
                    if let Some(sp) = inv_span {
                        d = d.with_note(format!(
                            "DMA lease acquired at {} was invalidated by subsequent `{}.wait_chunk()` at {}; finish decoding/parsing `{}` (e.g. `Type.from_bytes(slice)`) or copy needed bytes into a local `[N]u8` buffer before calling `wait_chunk()` again",
                            stale_tok.acquired_at, stale_tok.source, sp, ident.name
                        ));
                    } else {
                        d = d.with_note(format!(
                            "finish reading/decoding `{}` or copy needed bytes out before calling `{}.wait_chunk()` again",
                            ident.name, stale_tok.source
                        ));
                    }
                    d
                } else if stale_tok.acq_method == "get" {
                    let lease_name = Self::display_lease_source(&stale_tok.source);
                    let mut d = Diagnostic::new(
                        DiagnosticKind::StaleDmaLease,
                        ident.span,
                        format!(
                            "cannot read '{}' derived from '{}.get()' after '{}' moved or released its Lease",
                            ident.name, lease_name, lease_name
                        ),
                    );
                    if let Some(sp) = inv_span {
                        d = d.with_note(format!(
                            "pointer borrowed from `{}.get()` at {} was invalidated by `{}` at {}; finish dereferencing `{}` before calling `.release()`, `.park()`, `.post()`, or moving `{}`",
                            lease_name, stale_tok.acquired_at, inv_method, sp, ident.name, lease_name
                        ));
                    } else {
                        d = d.with_note(format!(
                            "finish reading `{}` before `{}` is moved or released",
                            ident.name, lease_name
                        ));
                    }
                    d
                } else {
                    let mut d = Diagnostic::new(
                        DiagnosticKind::StaleDmaLease,
                        ident.span,
                        format!(
                            "cannot read '{}' derived from '{}.{}()' after '{}.{}()' invalidated its byte_ring lease",
                            ident.name, display_src, stale_tok.acq_method, display_src, inv_method
                        ),
                    );
                    if let Some(sp) = inv_span {
                        d = d.with_note(format!(
                            "byte_ring slice acquired at {} was invalidated by `{}.{}()` at {}; finish reading/copying bytes from `{}` before calling `consume()`, `pop()`, `commit_write()`, or `grant_write()`",
                            stale_tok.acquired_at, display_src, inv_method, sp, ident.name
                        ));
                    } else {
                        d = d.with_note(format!(
                            "finish reading/copying bytes from `{}` before calling `{}.{}()`",
                            ident.name, display_src, inv_method
                        ));
                    }
                    d
                };
                let _ = &mut diag;
                self.sema.emit(diag);
            } else if !var.prov.leases.is_empty() && !self.loop_outer_reads.is_empty() {
                // Record reads of variables declared OUTSIDE the enclosing loop so cross-iteration
                // DMA lease invalidations are caught too.
                let loop_entry = self.loop_entry_depths.last().copied().unwrap_or(0);
                if var.decl_depth <= loop_entry {
                    if let Some(top_loop) = self.loop_outer_reads.last_mut() {
                        top_loop.push((
                            var.name.clone(),
                            self.leases.clone(),
                            var.prov.clone(),
                            ident.span,
                        ));
                    }
                }
            }

            if var.ty.is_linear_lease() {
                match var.lease_state {
                    Some(LinearLeaseState::Moved { moved_at, reason }) => {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::LinearLeaseViolation,
                                ident.span,
                                format!("use of moved linear Lease<T> variable '{}'", ident.name),
                            )
                            .with_note(format!(
                                "`{}` was previously consumed via {} at {}",
                                ident.name, reason, moved_at
                            )),
                        );
                        return ExprVal::error(ident.span);
                    }
                    Some(LinearLeaseState::Uninit) => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::LinearLeaseViolation,
                            ident.span,
                            format!(
                                "use of uninitialized linear Lease<T> variable '{}'",
                                ident.name
                            ),
                        ));
                        return ExprVal::error(ident.span);
                    }
                    _ => {}
                }
            }

            return ExprVal {
                ty: var.ty,
                error_spec: ResolvedErrorSpec::None,
                prov: var.prov,
                endpoint_name: var.endpoint_alias,
                field_endpoints: var.field_endpoints,
                span: ident.span,
            };
        }

        // 2. Illegal mutable raw global -> report Task/ISR isolation violation at access site!
        if let Some(ig) = self.sema.illegal_globals.get(&ident.name).cloned() {
            let ty_str = if ig.ty.is_error() {
                "u32".to_string()
            } else {
                ig.ty.to_string()
            };
            self.sema.emit(
                Diagnostic::new(
                    DiagnosticKind::GlobalIsolationViolation,
                    ident.span,
                    format!(
                        "task/ISR isolation violation: '{}' cannot access raw mutable global '{}'",
                        self.display_caller_name(),
                        ident.name
                    ),
                )
                .with_note(format!(
                    "raw global declared at {}; replace with `watch<{ty}> {name} = 0;` (single writer via `{name}.read()`/`{name}.publish(val)`), `guarded<{ty}> {name} = 0;` (multi-writer inside `critical {{ ... }}` via `{name}.get()`/`{name}.set(val)`), or `inbox<{ty}, 8> {name};` (queued events via `{name}.post(ev)` / `{name}.recv()`)",
                    ig.decl_span,
                    ty = ty_str,
                    name = ident.name
                )),
            );
            return ExprVal::scalar(ig.ty, ident.span);
        }

        // 3. Static OS primitive endpoint.
        if let Some(p) = self.sema.primitives.get(&ident.name).cloned() {
            return ExprVal {
                ty: Ty::OsPrimitive(p.primitive),
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::pure_static(),
                endpoint_name: Some(ident.name.clone()),
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }

        // 3b. Static subsystem struct instance (`uart0`).
        if let Some(sub) = self.sema.subsystems.get(&ident.name).cloned() {
            if self.in_isr() {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::GlobalIsolationViolation,
                        ident.span,
                        format!(
                            "ISR isolation violation: '{}' cannot access global subsystem instance '{}' directly",
                            self.display_caller_name(),
                            ident.name
                        ),
                    )
                    .with_note(format!(
                        "pass `&{name}` through `irq.attach({isr}(&{name}))` as a `*{st}` ISR parameter instead",
                        name = ident.name,
                        isr = self.display_caller_name(),
                        st = sub.struct_name,
                    )),
                );
            }
            return ExprVal {
                ty: Ty::Struct(sub.struct_name),
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::addr_of_static(),
                endpoint_name: Some(ident.name.clone()),
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }

        // 4. Immutable `const` global.
        if let Some(c) = self.sema.consts.get(&ident.name).cloned() {
            return ExprVal::scalar(c.ty, ident.span);
        }

        // 5. Unqualified enum variant (`Overcurrent`, `BadPacket`, `TooShort`, `BadCrc`, ...),
        //    scoped to enums visible in the current module.
        let visible_enums = self
            .sema
            .visible_enums_for_variant(self.prog_idx, &ident.name);
        if !visible_enums.is_empty() {
            if visible_enums.len() == 1 {
                return ExprVal::scalar(Ty::Enum(visible_enums[0].clone()), ident.span);
            }
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                ident.span,
                format!(
                    "ambiguous enum variant '{}'; qualify with enum name (candidates: {})",
                    ident.name,
                    visible_enums.join(", ")
                ),
            ));
            return ExprVal::error(ident.span);
        }

        // 6. Declared function (e.g. callback reference `register_cb(my_fn)`),
        //    MMIO peripheral instance (`USART1`), or hardware IRQ token (`IRQ_USART1`).
        if let Some(inst) = self.sema.mmio_instances.get(&ident.name).cloned() {
            return ExprVal {
                ty: Ty::MmioInstance {
                    name: inst.name.clone(),
                    layout: inst.layout,
                },
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::addr_of_static(),
                endpoint_name: Some(inst.name),
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }
        if let Some(iq) = self.sema.irqs.get(&ident.name).cloned() {
            return ExprVal {
                ty: Ty::IrqToken,
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::pure_static(),
                endpoint_name: Some(iq.name),
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }
        if let Some(fn_info) = self.sema.functions.get(&ident.name).cloned() {
            self.sema.check_extern_c_access(self.prog_idx, ident);
            let fn_ty = Ty::FnPtr {
                params: fn_info.params.iter().map(|p| p.ty.clone()).collect(),
                ret: Box::new(fn_info.ret_ty),
            };
            return ExprVal {
                ty: fn_ty,
                error_spec: ResolvedErrorSpec::None,
                prov: Provenance::fn_callback(ident.name.clone()),
                endpoint_name: None,
                field_endpoints: BTreeMap::new(),
                span: ident.span,
            };
        }

        // 7. Built-in or type namespaces (`sys`, `sim`, `CmdFrame`, `ParseErr`).
        if ident.name == SYS_NAMESPACE {
            return ExprVal::scalar(Ty::Namespace(NamespaceKind::Sys), ident.span);
        }
        if ident.name == "sim" {
            return ExprVal::scalar(Ty::Namespace(NamespaceKind::Sim), ident.span);
        }
        if self.sema.structs.contains_key(&ident.name) {
            return ExprVal::scalar(
                Ty::Namespace(NamespaceKind::Type(ident.name.clone())),
                ident.span,
            );
        }
        if self.sema.enums.contains_key(&ident.name) {
            return ExprVal::scalar(
                Ty::Namespace(NamespaceKind::Enum(ident.name.clone())),
                ident.span,
            );
        }

        let mut diag = Diagnostic::new(
            DiagnosticKind::UndefinedSymbol,
            ident.span,
            format!("undeclared identifier '{}'", ident.name),
        );
        if let Some(hint) = self.suggest_ident_hint(&ident.name) {
            diag = diag.with_note(hint);
        }
        self.sema.emit(diag);
        ExprVal::error(ident.span)
    }

    pub(super) fn check_unary_expr(&mut self, op: UnaryOp, inner: &Expr, span: Span) -> ExprVal {
        match op {
            UnaryOp::AddrOf => {
                // `&fn_name`, `&MMIO_PERIPH`, `&irq`, `&const`.
                if let ExprKind::Ident(id) = &inner.kind {
                    if let Some(inst) = self.sema.mmio_instances.get(&id.name).cloned() {
                        return ExprVal {
                            ty: Ty::Ptr(Box::new(Ty::MmioLayout(inst.layout))),
                            error_spec: ResolvedErrorSpec::None,
                            prov: Provenance::addr_of_static(),
                            endpoint_name: Some(id.name.clone()),
                            field_endpoints: BTreeMap::new(),
                            span,
                        };
                    }
                    if self.sema.irqs.contains_key(&id.name) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::ImmutableAssignment,
                            span,
                            format!("cannot take address '&{}' of hardware IRQ token", id.name),
                        ));
                        return ExprVal::error(span);
                    }
                    if let Some(fn_info) = self.sema.functions.get(&id.name).cloned() {
                        self.sema.check_extern_c_access(self.prog_idx, id);
                        let fn_ty = Ty::FnPtr {
                            params: fn_info.params.iter().map(|p| p.ty.clone()).collect(),
                            ret: Box::new(fn_info.ret_ty),
                        };
                        return ExprVal {
                            ty: fn_ty,
                            error_spec: ResolvedErrorSpec::None,
                            prov: Provenance::fn_callback(id.name.clone()),
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        };
                    }
                    if self.sema.consts.contains_key(&id.name) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::ImmutableAssignment,
                            span,
                            format!(
                                "cannot take mutable pointer '&{}' to immutable 'const'",
                                id.name
                            ),
                        ));
                        return ExprVal::error(span);
                    }
                }
                if let Some(const_name) = self.root_const_of_place(inner) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        span,
                        format!(
                            "cannot take mutable pointer to immutable 'const' '{}'",
                            const_name
                        ),
                    ));
                    return ExprVal::error(span);
                }

                let (iv, prov, endpoint_name) = self.check_addr_of_place(inner, span);
                if iv.ty.is_error() {
                    return ExprVal::error(span);
                }
                if iv.ty.is_linear_lease() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        span,
                        "cannot take address `&` of linear Lease<T> variable (use `.get()` to obtain a tracked `*T` buffer pointer)",
                    ));
                    return ExprVal::error(span);
                }

                ExprVal {
                    ty: Ty::Ptr(Box::new(iv.ty)),
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name,
                    field_endpoints: iv.field_endpoints,
                    span,
                }
            }

            UnaryOp::Deref => {
                let iv = self.check_expr_infallible(inner);
                self.finish_deref_expr(iv, span, true)
            }

            UnaryOp::Not => {
                let iv = self.check_expr_infallible(inner);
                if !iv.ty.is_bool() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("logical '!' requires 'bool', found '{}'", iv.ty),
                    ));
                }
                ExprVal::scalar(Ty::bool(), span)
            }

            UnaryOp::BitNot => {
                let iv = self.check_expr_infallible(inner);
                if !iv.ty.is_integer() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("bitwise '~' requires an integer, found '{}'", iv.ty),
                    ));
                }
                let out_ty = match iv.ty {
                    Ty::Prim(p) if p.is_integer() => Ty::Prim(Ty::host_prim(p)),
                    other => other,
                };
                ExprVal::scalar(out_ty, span)
            }

            UnaryOp::Neg => {
                if let ExprKind::Literal(Literal::Int { value, suffix }) = &inner.kind {
                    let inner_ty = suffix.map(Ty::Prim).unwrap_or(Ty::IntLit);
                    self.expr_tys
                        .insert(inner as *const Expr as usize, inner_ty);
                    return match suffix {
                        Some(p) if p.is_signed() => {
                            self.check_int_lit_range_or_emit(-i128::from(*value), *p, span);
                            ExprVal::scalar(Ty::Prim(*p), span)
                        }
                        Some(p) => {
                            self.check_int_lit_range_or_emit(i128::from(*value), *p, inner.span);
                            ExprVal::scalar(Ty::Prim(Ty::host_prim(*p)), span)
                        }
                        None => {
                            if *value > (i64::MAX as u64) + 1
                                && self
                                    .sema
                                    .f32_overflow_reported_spans
                                    .insert((span.start, span.end))
                            {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    span,
                                    format!(
                                        "integer literal -{} out of range for any integer type",
                                        value
                                    ),
                                ));
                            }
                            ExprVal::scalar(Ty::IntLit, span)
                        }
                    };
                }
                let iv = self.check_expr_infallible(inner);
                if !iv.ty.is_numeric() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("unary '-' requires an integer or float, found '{}'", iv.ty),
                    ));
                }
                let out_ty = match iv.ty {
                    Ty::IntLit => Ty::IntLit,
                    Ty::Prim(p) if p.is_integer() => Ty::Prim(Ty::host_prim(p)),
                    other => other,
                };
                ExprVal::scalar(out_ty, span)
            }
        }
    }

    pub(super) fn check_binary_expr(
        &mut self,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        span: Span,
    ) -> ExprVal {
        // S-H2: Short-circuit `&&` and `||` conditionally evaluate `rhs` only when `lhs` does not
        // short-circuit, so snapshot/fork variable move states & lease generations around `rhs`!
        if matches!(op, BinaryOp::And | BinaryOp::Or) {
            let lv = self.check_expr_infallible(lhs);
            let after_lhs_vars = self.snapshot_outer_vars();
            let after_lhs_leases = self.leases.clone();
            self.branch_depth += 1;
            let rv = self.check_expr_infallible(rhs);
            self.branch_depth -= 1;
            let after_rhs_vars = self.snapshot_outer_vars();
            let after_rhs_leases = self.leases.clone();

            self.scopes = after_lhs_vars.clone();
            self.merge_branch_vars(
                &after_lhs_vars,
                &[after_lhs_vars.clone(), after_rhs_vars],
                span,
            );
            self.leases = after_lhs_leases.clone();
            self.leases
                .join_branches(&[after_lhs_leases, after_rhs_leases]);

            if lv.ty.is_error() || rv.ty.is_error() {
                return ExprVal::error(span);
            }
            if !lv.ty.is_bool() || !rv.ty.is_bool() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    span,
                    format!(
                        "logical '{}' requires 'bool' operands, found '{}' and '{}'",
                        op.as_str(),
                        lv.ty,
                        rv.ty
                    ),
                ));
            }
            return ExprVal::scalar(Ty::bool(), span);
        }

        let lv = self.check_expr_infallible(lhs);
        let rv = self.check_expr_infallible(rhs);
        if lv.ty.is_error() || rv.ty.is_error() {
            return ExprVal::error(span);
        }

        match op {
            BinaryOp::And | BinaryOp::Or => unreachable!(),

            BinaryOp::Eq | BinaryOp::Ne => {
                if lv.ty.is_linear_lease() || rv.ty.is_linear_lease() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        span,
                        "linear Lease<T> values cannot be compared directly with `==` / `!=` (compare `.addr()` instead)",
                    ));
                    return ExprVal::error(span);
                }
                if lv.ty.is_integer() && rv.ty.is_integer() {
                    if let Some(common) = try_unify_integer(&lv.ty, &rv.ty) {
                        let dst = match common {
                            Ty::Prim(p) => Some(p),
                            _ => None,
                        };
                        self.validate_int_expr_at_type(lhs, dst);
                        self.validate_int_expr_at_type(rhs, dst);
                    } else {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "cannot compare incompatible types '{}' and '{}' with '{}' (use an explicit `as` cast)",
                                lv.ty,
                                rv.ty,
                                op.as_str()
                            ),
                        ));
                    }
                    return ExprVal::scalar(Ty::bool(), span);
                }
                let unified_float = unify_float_arith(&lv.ty, &rv.ty);
                let compat = lv.ty.is_assignable_to(&rv.ty)
                    || rv.ty.is_assignable_to(&lv.ty)
                    || unified_float.is_some();
                if !compat {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "cannot compare incompatible types '{}' and '{}' with '{}'",
                            lv.ty,
                            rv.ty,
                            op.as_str()
                        ),
                    ));
                } else if unified_float == Some(Ty::f32()) {
                    self.sema.check_f32_coercion_expr(lhs, &Ty::f32());
                    self.sema.check_f32_coercion_expr(rhs, &Ty::f32());
                }
                ExprVal::scalar(Ty::bool(), span)
            }

            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                if lv.ty.is_integer() && rv.ty.is_integer() {
                    if let Some(common) = try_unify_integer(&lv.ty, &rv.ty) {
                        let dst = match common {
                            Ty::Prim(p) => Some(p),
                            _ => None,
                        };
                        self.validate_int_expr_at_type(lhs, dst);
                        self.validate_int_expr_at_type(rhs, dst);
                    } else {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "comparison '{}' requires compatible integer operands, found '{}' and '{}' (use an explicit `as` cast)",
                                op.as_str(),
                                lv.ty,
                                rv.ty
                            ),
                        ));
                    }
                    return ExprVal::scalar(Ty::bool(), span);
                }
                let unified_float = unify_float_arith(&lv.ty, &rv.ty);
                if unified_float.is_none() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "comparison '{}' requires matching numeric operands, found '{}' and '{}'",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                } else if unified_float == Some(Ty::f32()) {
                    self.sema.check_f32_coercion_expr(lhs, &Ty::f32());
                    self.sema.check_f32_coercion_expr(rhs, &Ty::f32());
                }
                ExprVal::scalar(Ty::bool(), span)
            }

            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                if lv.ty.is_integer() && rv.ty.is_integer() {
                    if let Some(res_ty) = try_unify_integer(&lv.ty, &rv.ty) {
                        ExprVal::scalar(res_ty, span)
                    } else {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "operator '{}' requires compatible integer operands, found '{}' and '{}' (use an explicit `as` cast)",
                                op.as_str(),
                                lv.ty,
                                rv.ty
                            ),
                        ));
                        ExprVal::error(span)
                    }
                } else if let Some(float_ty) = unify_float_arith(&lv.ty, &rv.ty) {
                    if float_ty == Ty::f32() {
                        self.sema.check_f32_coercion_expr(lhs, &Ty::f32());
                        self.sema.check_f32_coercion_expr(rhs, &Ty::f32());
                    }
                    ExprVal::scalar(float_ty, span)
                } else {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "operator '{}' requires matching integer or float operands, found '{}' and '{}'",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                    ExprVal::error(span)
                }
            }

            BinaryOp::BitOr | BinaryOp::BitAnd | BinaryOp::BitXor => {
                if !lv.ty.is_integer() || !rv.ty.is_integer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "operator '{}' requires integer operands, found '{}' and '{}'",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                    return ExprVal::error(span);
                }
                if let Some(res_ty) = try_unify_integer(&lv.ty, &rv.ty) {
                    ExprVal::scalar(res_ty, span)
                } else {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "operator '{}' requires compatible integer operands, found '{}' and '{}' (use an explicit `as` cast)",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                    ExprVal::error(span)
                }
            }

            BinaryOp::Shl | BinaryOp::Shr => {
                if !lv.ty.is_integer() || !rv.ty.is_integer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "operator '{}' requires integer operands, found '{}' and '{}'",
                            op.as_str(),
                            lv.ty,
                            rv.ty
                        ),
                    ));
                    return ExprVal::error(span);
                }
                self.validate_int_expr_at_type(rhs, None);
                // L-H1: Shift result takes the host integer type of the shifted left operand
                // rather than promoting to a wider right-hand shift amount (`0x8000_0000u32 >> 4i32` -> `u32`).
                // Constant shift range check runs in `validate_int_expr_at_type` so destination-widened shifts
                // like `u32 bit = a << 8` (where `a: u8`) validate at `u32`.
                let res_ty = match &lv.ty {
                    Ty::Prim(p) => Ty::Prim(Ty::host_prim(*p)),
                    Ty::IntLit => Ty::IntLit,
                    _ => lv.ty.clone(),
                };
                ExprVal::scalar(res_ty, span)
            }
        }
    }

    pub(super) fn check_const_shift_amount(&mut self, op_str: &str, lhs_ty: &Ty, rhs_expr: &Expr) {
        let shadowed = self.active_local_names();
        let Some(sh) =
            self.sema
                .eval_const_expr_i128_shadowed(rhs_expr, &mut BTreeSet::new(), &shadowed)
        else {
            return;
        };
        let max_bits = lhs_ty.fixed_int_bit_width().unwrap_or(64);
        if (sh < 0 || sh >= i128::from(max_bits))
            && self
                .sema
                .f32_overflow_reported_spans
                .insert((rhs_expr.span.start, rhs_expr.span.end))
        {
            let ty_desc = if matches!(lhs_ty, Ty::IntLit) {
                "integer".to_string()
            } else {
                format!("'{lhs_ty}'")
            };
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                rhs_expr.span,
                format!(
                    "shift count {sh} for '{op_str}' is out of range for {ty_desc} (must be in 0..{max_bits})"
                ),
            ));
        }
    }

    pub(super) fn finish_deref_expr(
        &mut self,
        iv: ExprVal,
        span: Span,
        reject_byval: bool,
    ) -> ExprVal {
        match iv.ty {
            Ty::Ptr(pointee) => {
                if reject_byval {
                    if let Some(ln) = ty_contains_byval_mmio(&pointee) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::MmioViolation,
                            span,
                            format!(
                                "cannot dereference MMIO layout pointer '*{}' by value (access its registers via '.reg' projection)",
                                ln
                            ),
                        ));
                        return ExprVal::error(span);
                    }
                    if let Some(sub) = ty_contains_byval_subsys(&pointee, &self.sema.structs) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!(
                                "cannot dereference subsystem struct '{}' by value (access its fields or methods via pointer)",
                                sub
                            ),
                        ));
                        return ExprVal::error(span);
                    }
                    if ty_embeds_os_primitive(&pointee)
                        || ty_contains_bare_unsized_prim(&pointee).is_some()
                    {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!(
                                "cannot dereference OS primitive pointer '*{}' by value (use its methods via pointer)",
                                pointee
                            ),
                        ));
                        return ExprVal::error(span);
                    }
                }
                let prov = if pointee.contains_reference(&self.sema.structs)
                    || pointee.contains_fn_ptr(&self.sema.structs)
                {
                    iv.prov
                } else {
                    Provenance::pure_static()
                };
                ExprVal {
                    ty: *pointee,
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name: iv.endpoint_name,
                    field_endpoints: iv.field_endpoints,
                    span,
                }
            }
            Ty::Error => ExprVal::error(span),
            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    span,
                    format!("cannot dereference non-pointer type '{}'", other),
                ));
                ExprVal::error(span)
            }
        }
    }

    pub(super) fn check_projection_base_allow_mmio_array(
        &mut self,
        expr: &Expr,
    ) -> (ExprVal, Option<(String, String, usize)>) {
        if let ExprKind::Unary {
            op: UnaryOp::Deref,
            expr: inner,
        } = &expr.kind
        {
            let iv = self.check_expr_infallible(inner);
            (self.finish_deref_expr(iv, expr.span, false), None)
        } else {
            self.check_expr_infallible_allow_mmio_array(expr)
        }
    }

    pub(super) fn check_projection_base(&mut self, expr: &Expr) -> ExprVal {
        if let ExprKind::Unary {
            op: UnaryOp::Deref,
            expr: inner,
        } = &expr.kind
        {
            let iv = self.check_expr_infallible(inner);
            self.finish_deref_expr(iv, expr.span, false)
        } else {
            self.check_expr_infallible(expr)
        }
    }

    pub(super) fn root_const_of_place(&self, place: &Expr) -> Option<String> {
        let mut cur = place;
        loop {
            match &cur.kind {
                ExprKind::Ident(id) => {
                    if self.lookup_local(&id.name).is_none()
                        && self.sema.consts.contains_key(&id.name)
                    {
                        return Some(id.name.clone());
                    }
                    return None;
                }
                ExprKind::Field { target, .. } | ExprKind::Index { target, .. } => {
                    cur = target;
                }
                _ => return None,
            }
        }
    }

    pub(super) fn coerce_array_to_slice_prov(
        &self,
        expr: &Expr,
        actual_ty: &Ty,
        expected_ty: &Ty,
        val_prov: &Provenance,
    ) -> Provenance {
        if !matches!(actual_ty, Ty::Array { .. }) || !matches!(expected_ty, Ty::Slice(_)) {
            return val_prov.clone();
        }
        self.place_provenance_readonly(expr)
            .merge_place_with_val_origins(val_prov)
    }

    pub(super) fn place_provenance_readonly(&self, place: &Expr) -> Provenance {
        match &place.kind {
            ExprKind::Ident(id) => {
                if let Some(local) = self.lookup_local(&id.name) {
                    if local.is_isr_state {
                        Provenance::addr_of_isr_state(&local.name, local.decl_depth)
                    } else {
                        Provenance::addr_of_local(&local.name, local.decl_depth)
                    }
                } else if self.sema.primitives.contains_key(&id.name)
                    || self.sema.subsystems.contains_key(&id.name)
                    || self.sema.mmio_instances.contains_key(&id.name)
                    || self.sema.consts.contains_key(&id.name)
                {
                    Provenance::addr_of_static()
                } else {
                    Provenance::addr_of_local("<temp>", self.current_depth())
                }
            }
            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: inner,
            } => {
                if let ExprKind::Ident(id) = &inner.kind {
                    if let Some(local) = self.lookup_local(&id.name) {
                        return local.prov.clone().as_backing_ptr();
                    }
                }
                self.place_provenance_readonly(inner)
            }
            ExprKind::Field { target, .. } => {
                if let ExprKind::Ident(id) = &target.kind {
                    if let Some(local) = self.lookup_local(&id.name) {
                        if matches!(local.ty, Ty::Ptr(_)) {
                            return local.prov.clone().as_backing_ptr();
                        }
                    }
                }
                self.place_provenance_readonly(target)
            }
            ExprKind::Index { target, .. } => {
                if let ExprKind::Ident(id) = &target.kind {
                    if let Some(local) = self.lookup_local(&id.name) {
                        if matches!(local.ty, Ty::Ptr(_) | Ty::Slice(_)) {
                            return local.prov.clone().as_backing_ptr();
                        }
                    }
                }
                self.place_provenance_readonly(target)
            }
            _ => Provenance::addr_of_local("<temp>", self.current_depth()),
        }
    }

    pub(super) fn check_addr_of_place(
        &mut self,
        place: &Expr,
        addr_span: Span,
    ) -> (ExprVal, Provenance, Option<String>) {
        let (val, prov, ep, _) = self.check_addr_of_place_impl(place, addr_span, false);
        (val, prov, ep)
    }

    pub(super) fn check_addr_of_place_impl(
        &mut self,
        place: &Expr,
        addr_span: Span,
        allow_mmio_array: bool,
    ) -> AddrOfPlaceResult {
        match &place.kind {
            ExprKind::Ident(id) => {
                let iv = self.check_expr_infallible(place);
                if iv.ty.is_error() {
                    return (iv, Provenance::pure_static(), None, None);
                }
                let mut endpoint_name = iv.endpoint_name.clone();
                let base_place_prov = if let Some(local) = self.lookup_local(&id.name) {
                    if endpoint_name.is_none() {
                        endpoint_name = local.endpoint_alias.clone();
                    }
                    if local.is_isr_state {
                        Provenance::addr_of_isr_state(&local.name, local.decl_depth)
                    } else {
                        Provenance::addr_of_local(&local.name, local.decl_depth)
                    }
                } else if self.sema.primitives.contains_key(&id.name)
                    || self.sema.subsystems.contains_key(&id.name)
                    || self.sema.mmio_instances.contains_key(&id.name)
                    || self.sema.consts.contains_key(&id.name)
                {
                    if endpoint_name.is_none() {
                        endpoint_name = Some(id.name.clone());
                    }
                    Provenance::addr_of_static()
                } else {
                    Provenance::addr_of_local("<temp>", self.current_depth())
                };
                let prov = base_place_prov.merge_place_with_val_origins(&iv.prov);
                (iv, prov, endpoint_name, None)
            }

            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: inner,
            } => {
                // `&(*p)` yields `p` directly without triggering by-value deref rejection and
                // preserves `p`'s backing pointer provenance (`leases`, depth, origins, endpoint).
                let ptr_v = self.check_expr_infallible(inner);
                let val = self.finish_deref_expr(ptr_v.clone(), place.span, false);
                if val.ty.is_error() {
                    return (val, Provenance::pure_static(), None, None);
                }
                let prov = ptr_v.prov.as_backing_ptr();
                let ep = val.endpoint_name.clone();
                (val, prov, ep, None)
            }

            ExprKind::Field { target, field } => {
                let (bv, target_place_prov, target_ep, target_mmio_arr) =
                    self.check_addr_of_place_impl(target, addr_span, true);
                if let Some((layout_name, reg_name, _)) = target_mmio_arr {
                    if field.name != "len" {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::MmioViolation,
                            place.span,
                            format!(
                                "cannot access '.{}' on MMIO register array '{}.{}' (only element indexing '[i]' and '.len' are allowed)",
                                field.name, layout_name, reg_name
                            ),
                        ));
                        return (
                            ExprVal::error(place.span),
                            Provenance::pure_static(),
                            None,
                            None,
                        );
                    }
                }
                if bv.ty.is_error() {
                    return (
                        ExprVal::error(place.span),
                        Provenance::pure_static(),
                        None,
                        None,
                    );
                }
                if is_mmio_ty(&bv.ty) {
                    let layout_name = match &bv.ty {
                        Ty::MmioLayout(ln) | Ty::MmioInstance { layout: ln, .. } => {
                            Some(ln.as_str())
                        }
                        Ty::Ptr(inner) => match &**inner {
                            Ty::MmioLayout(ln) => Some(ln.as_str()),
                            _ => None,
                        },
                        _ => None,
                    };
                    if allow_mmio_array {
                        if let Some(ln) = layout_name {
                            if let Some(reg) = self
                                .sema
                                .mmio_layouts
                                .get(ln)
                                .and_then(|l| l.regs.iter().find(|r| r.name == field.name))
                                .cloned()
                            {
                                if let Some(len) = reg.array_len {
                                    let reg_ty = Ty::Array {
                                        elem: Box::new(Ty::Prim(reg.elem_ty)),
                                        len,
                                    };
                                    return (
                                        ExprVal::scalar(reg_ty, place.span),
                                        Provenance::pure_static(),
                                        None,
                                        Some((ln.to_string(), reg.name, len)),
                                    );
                                }
                            }
                        }
                    }
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        addr_span,
                        format!(
                            "cannot take raw address `&` of MMIO register '{}'; access MMIO registers through `*Layout` pointers instead",
                            field.name
                        ),
                    ));
                    return (
                        ExprVal::error(place.span),
                        Provenance::pure_static(),
                        None,
                        None,
                    );
                }
                let fv = self.check_resolved_field_val_with_place(
                    bv.clone(),
                    Some(target_place_prov.clone()),
                    target,
                    field,
                    place.span,
                );
                if fv.ty.is_error() {
                    return (fv, Provenance::pure_static(), None, None);
                }
                let base_place_prov = if matches!(bv.ty, Ty::Ptr(_)) {
                    bv.prov.as_backing_ptr()
                } else {
                    target_place_prov
                };
                let prov = base_place_prov.merge_place_with_val_origins(&fv.prov);
                let ep = fv.endpoint_name.clone().or(target_ep);
                (fv, prov, ep, None)
            }

            ExprKind::Index { target: seq, index } => {
                let (seq_v, seq_place_prov, seq_ep, mmio_arr) =
                    self.check_addr_of_place_impl(seq, addr_span, true);
                let idx_v = self.check_expr_infallible(index);
                self.validate_int_expr_at_type(index, Some(PrimType::Usize));
                if !idx_v.ty.is_integer() && !idx_v.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        index.span,
                        format!("index must be an integer, found '{}'", idx_v.ty),
                    ));
                }
                if let Some((_, reg_name, _)) = mmio_arr {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        addr_span,
                        format!(
                            "cannot take raw address `&` of MMIO register '{}'; access MMIO registers through `*Layout` pointers instead",
                            reg_name
                        ),
                    ));
                    return (
                        ExprVal::error(place.span),
                        Provenance::pure_static(),
                        None,
                        None,
                    );
                }
                if seq_v.ty.is_error() {
                    return (
                        ExprVal::error(place.span),
                        Provenance::pure_static(),
                        None,
                        None,
                    );
                }
                let elem_v =
                    self.check_resolved_index_val(seq_v.clone(), seq, index.span, place.span);
                if elem_v.ty.is_error() {
                    return (elem_v, Provenance::pure_static(), None, None);
                }
                let base_place_prov = match &seq_v.ty {
                    Ty::Slice(_) | Ty::Ptr(_) => seq_v.prov.as_backing_ptr(),
                    _ => seq_place_prov,
                };
                let prov = base_place_prov.merge_place_with_val_origins(&elem_v.prov);
                let ep = elem_v.endpoint_name.clone().or(seq_ep);
                (elem_v, prov, ep, None)
            }

            _ => {
                let iv = self.check_expr_infallible(place);
                if iv.ty.is_error() {
                    return (iv, Provenance::pure_static(), None, None);
                }
                let base = Provenance::addr_of_local("<temp>", self.current_depth());
                let prov = base.merge_place_with_val_origins(&iv.prov);
                let ep = iv.endpoint_name.clone();
                (iv, prov, ep, None)
            }
        }
    }

    pub(super) fn check_field_expr(&mut self, target: &Expr, field: &Ident, span: Span) -> ExprVal {
        if field.name == "ptr" {
            let (tv, target_place_prov, _) = self.check_addr_of_place(target, span);
            if tv.ty.is_error() {
                return ExprVal::error(span);
            }
            return self.check_resolved_field_val_with_place(
                tv,
                Some(target_place_prov),
                target,
                field,
                span,
            );
        }
        let (tv, mmio_arr) = self.check_projection_base_allow_mmio_array(target);
        if let Some((layout_name, reg_name, _)) = mmio_arr {
            if field.name != "len" {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    span,
                    format!(
                        "cannot access '.{}' on MMIO register array '{}.{}' (only element indexing '[i]' and '.len' are allowed)",
                        field.name, layout_name, reg_name
                    ),
                ));
                return ExprVal::error(span);
            }
        }
        self.check_resolved_field_val_with_place(tv, None, target, field, span)
    }

    pub(super) fn check_resolved_field_val_with_place(
        &mut self,
        tv: ExprVal,
        target_place_prov: Option<Provenance>,
        target: &Expr,
        field: &Ident,
        span: Span,
    ) -> ExprVal {
        if tv.ty.is_error() {
            return ExprVal::error(span);
        }

        // Auto-dereference `*Struct` / `*Slice` / `*MmioLayout` on field access.
        let deref_ty = match &tv.ty {
            Ty::Ptr(inner) => (**inner).clone(),
            other => other.clone(),
        };

        match deref_ty {
            Ty::Slice(_) => {
                if field.name == "len" {
                    ExprVal::scalar(Ty::usize(), span)
                } else if field.name == "ptr" {
                    if let Ty::Slice(elem) = deref_ty {
                        ExprVal {
                            ty: Ty::Ptr(elem),
                            error_spec: ResolvedErrorSpec::None,
                            prov: tv.prov.as_backing_ptr(),
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    } else {
                        unreachable!()
                    }
                } else {
                    let note = if matches!(field.name.as_str(), "length" | "size" | "count" | "cap")
                    {
                        "did you mean `.len`? (slices provide `.len` of type `usize` and `.ptr` of type `*T`)".to_string()
                    } else if let Some(best) = closest_match(&field.name, ["len", "ptr"], 2) {
                        format!("did you mean `.{}`?", best)
                    } else {
                        "slices provide `.len` (`usize`) and `.ptr` (`*T`)".to_string()
                    };
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UndefinedSymbol,
                            field.span,
                            format!(
                                "slice has no property '{}' (supported: 'len', 'ptr')",
                                field.name
                            ),
                        )
                        .with_note(note),
                    );
                    ExprVal::error(span)
                }
            }

            Ty::Array { len: _, elem } => {
                if field.name == "len" {
                    ExprVal::scalar(Ty::usize(), span)
                } else if field.name == "ptr" {
                    let prov = if matches!(tv.ty, Ty::Ptr(_)) {
                        tv.prov.as_backing_ptr()
                    } else {
                        target_place_prov.unwrap_or(tv.prov)
                    };
                    ExprVal {
                        ty: Ty::Ptr(elem),
                        error_spec: ResolvedErrorSpec::None,
                        prov,
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                } else {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        field.span,
                        format!("array has no property '{}'", field.name),
                    ));
                    ExprVal::error(span)
                }
            }

            Ty::Struct(struct_name) => {
                let Some(f_info) = self
                    .sema
                    .structs
                    .get(&struct_name)
                    .and_then(|s| s.field(&field.name))
                    .cloned()
                else {
                    let hint = self.suggest_struct_field_hint(&struct_name, &field.name);
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UndefinedSymbol,
                            field.span,
                            format!("struct '{}' has no field '{}'", struct_name, field.name),
                        )
                        .with_note(hint),
                    );
                    return ExprVal::error(span);
                };

                let is_subsys_field = matches!(&f_info.ty, Ty::OsPrimitive(_))
                    || matches!(&f_info.ty, Ty::Ptr(inner) if matches!(&**inner, Ty::OsPrimitive(_)))
                    || matches!(
                        &f_info.ty,
                        Ty::Struct(s) if self.sema.structs.get(s).is_some_and(|st| st.is_subsystem)
                    )
                    || matches!(
                        &f_info.ty,
                        Ty::Ptr(inner) if matches!(&**inner, Ty::Struct(s) if self.sema.structs.get(s).is_some_and(|st| st.is_subsystem))
                    );
                let is_view_field = is_subsys_field
                    || ty_contains_os_prim_view(
                        &f_info.ty,
                        &self.sema.structs,
                        &mut BTreeSet::new(),
                    );
                let prov = if is_view_field
                    || f_info.ty.contains_reference(&self.sema.structs)
                    || f_info.ty.contains_fn_ptr(&self.sema.structs)
                {
                    if let Some(root) = root_ident_of(target) {
                        if let Some(v) = self.lookup_local(&root) {
                            if !matches!(v.ty, Ty::Ptr(_)) {
                                v.field_prov
                                    .get(&field.name)
                                    .cloned()
                                    .unwrap_or_else(|| tv.prov.clone())
                            } else {
                                tv.prov.clone()
                            }
                        } else {
                            tv.prov.clone()
                        }
                    } else {
                        tv.prov.clone()
                    }
                } else {
                    Provenance::pure_static()
                };

                let mut out_field_endpoints = BTreeMap::new();
                let prefix = format!("{}.", field.name);
                for (k, v) in &tv.field_endpoints {
                    if let Some(rest) = k.strip_prefix(&prefix) {
                        out_field_endpoints.insert(rest.to_string(), v.clone());
                    }
                }

                let endpoint_name = if let Some(ep) = tv.field_endpoints.get(&field.name) {
                    Some(ep.clone())
                } else if is_view_field {
                    tv.endpoint_name
                        .as_ref()
                        .cloned()
                        .or_else(|| root_ident_of(target))
                        .map(|parent| format!("{parent}.{}", field.name))
                } else {
                    None
                };

                ExprVal {
                    ty: f_info.ty,
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name,
                    field_endpoints: out_field_endpoints,
                    span,
                }
            }

            Ty::MmioLayout(layout_name)
            | Ty::MmioInstance {
                layout: layout_name,
                ..
            } => {
                let Some(layout_info) = self.sema.mmio_layouts.get(&layout_name).cloned() else {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        field.span,
                        format!("unknown MMIO register layout '{}'", layout_name),
                    ));
                    return ExprVal::error(span);
                };
                let Some(reg) = layout_info
                    .regs
                    .iter()
                    .find(|r| r.name == field.name)
                    .cloned()
                else {
                    let candidates = layout_info.regs.iter().map(|r| r.name.as_str());
                    let mut diag = Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        field.span,
                        format!(
                            "MMIO layout '{}' has no register '{}'",
                            layout_name, field.name
                        ),
                    );
                    if let Some(best) = closest_match(&field.name, candidates, 2) {
                        diag = diag.with_note(format!("did you mean register `.{}`?", best));
                    }
                    self.sema.emit(diag);
                    return ExprVal::error(span);
                };
                if reg.access == MmioAccess::Wo {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        span,
                        format!(
                            "cannot read write-only `@wo` MMIO register '{}.{}'",
                            layout_name, reg.name
                        ),
                    ));
                }
                let reg_ty = if let Some(len) = reg.array_len {
                    self.mmio_array_expr_spans
                        .insert((span.start, span.end), (layout_name, reg.name, len));
                    Ty::Array {
                        elem: Box::new(Ty::Prim(reg.elem_ty)),
                        len,
                    }
                } else {
                    Ty::Prim(reg.elem_ty)
                };
                ExprVal::scalar(reg_ty, span)
            }

            Ty::Namespace(NamespaceKind::Enum(enum_name)) => {
                if self
                    .sema
                    .enums
                    .get(&enum_name)
                    .is_some_and(|e| e.has_variant(&field.name))
                {
                    ExprVal::scalar(Ty::Enum(enum_name), span)
                } else {
                    let hint = self.suggest_enum_variant_hint(&enum_name, &field.name);
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UndefinedSymbol,
                            field.span,
                            format!("enum '{}' has no variant '{}'", enum_name, field.name),
                        )
                        .with_note(hint),
                    );
                    ExprVal::error(span)
                }
            }

            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    field.span,
                    format!("type '{}' has no field '{}'", other, field.name),
                ));
                ExprVal::error(span)
            }
        }
    }

    pub(super) fn check_call_expr(
        &mut self,
        callee: &Expr,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        // Normalize `a.b(args)` if parser represented it as `Call { callee: Field { target, field }, args }`.
        if let ExprKind::Field { target, field } = &callee.kind {
            // Unless `target` is a struct whose `field` is a `FnPtr`! Check if `target` has a `FnPtr` field first.
            let is_struct_fn_ptr = if let Some(root) = root_ident_of(target) {
                if let Some(v) = self.lookup_local(&root) {
                    let st_ty = match &v.ty {
                        Ty::Struct(s) => Some(s.as_str()),
                        Ty::Ptr(inner) => match &**inner {
                            Ty::Struct(s) => Some(s.as_str()),
                            _ => None,
                        },
                        _ => None,
                    };
                    st_ty
                        .and_then(|sn| self.sema.structs.get(sn))
                        .and_then(|s| s.field(&field.name))
                        .is_some_and(|f| matches!(f.ty, Ty::FnPtr { .. }))
                } else {
                    false
                }
            } else {
                false
            };
            if !is_struct_fn_ptr {
                return self.check_method_call(target, field, args, span);
            }
        }

        if let ExprKind::Ident(ident) = &callee.kind {
            // 1. User-declared or `extern "C"` function.
            if let Some(fn_info) = self.sema.functions.get(&ident.name).cloned() {
                self.sema.check_extern_c_access(self.prog_idx, ident);
                return self.check_direct_fn_call(&fn_info, args, span);
            }

            // 2. Calling a `task` blueprint like a regular function is an error (use `spawn`!).
            if self.sema.tasks.contains_key(&ident.name) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    ident.span,
                    format!(
                        "cannot call task blueprint '{}' directly; instantiate it with 'spawn {}(...)'",
                        ident.name, ident.name
                    ),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                return ExprVal::error(span);
            }

            // 3. Built-in runtime functions (`set_priority`, `checkpoint`, `delay`, `sleep`, `print`, `assert`, ...).
            if let Some(res) = self.try_check_builtin_fn(&ident.name, args, span) {
                return res;
            }
        }

        // 4. Indirect call through a local `FnPtr` variable or expression.
        let cv = self.check_expr_infallible(callee);
        match cv.ty {
            Ty::FnPtr { params, ret } => {
                if args.len() != params.len() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "function pointer expects {} argument(s), found {}",
                            params.len(),
                            args.len()
                        ),
                    ));
                }
                for (i, arg) in args.iter().enumerate() {
                    let av = self.check_expr_infallible(&arg.value);
                    if av.ty.is_linear_lease() {
                        self.consume_lease_value(
                            &arg.value,
                            &av,
                            "function pointer argument",
                            arg.span,
                        );
                    }
                    if let Some(expected) = params.get(i) {
                        if self.sema.check_assignable_or_emit(
                            &av.ty,
                            expected,
                            arg.span,
                            DiagnosticKind::TypeMismatch,
                            || {
                                format!(
                                    "argument {} has type '{}', expected '{}'",
                                    i + 1,
                                    av.ty,
                                    expected
                                )
                            },
                        ) {
                            self.check_coerced_expr(&arg.value, expected);
                        }
                    }
                }
                for cb_target in &cv.prov.fn_callbacks {
                    self.sema.call_graph.add_edge(CallEdge {
                        caller: self.caller_key.clone(),
                        caller_kind: self.caller_kind,
                        callee: cb_target.clone(),
                        span,
                        ctx: self.call_context(),
                    });
                }
                ExprVal::scalar(*ret, span)
            }
            Ty::Error => {
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                ExprVal::error(span)
            }
            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    callee.span,
                    format!("expression of type '{}' is not callable", other),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                ExprVal::error(span)
            }
        }
    }

    pub(super) fn check_direct_fn_call(
        &mut self,
        fn_info: &FunctionInfo,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        // Record call-graph edge for whole-program yield inference and recursion cycle detection.
        self.sema.call_graph.add_edge(CallEdge {
            caller: self.caller_key.clone(),
            caller_kind: self.caller_kind,
            callee: fn_info.name.clone(),
            span,
            ctx: self.call_context(),
        });

        if (!fn_info.is_variadic && args.len() != fn_info.params.len())
            || (fn_info.is_variadic && args.len() < fn_info.params.len())
        {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "function '{}' expects {}{} argument(s), found {}",
                    fn_info.name,
                    if fn_info.is_variadic { "at least " } else { "" },
                    fn_info.params.len(),
                    args.len()
                ),
            ));
        }

        let mut arg_vals = Vec::with_capacity(args.len());
        let mut boot_call_args = Vec::with_capacity(args.len());
        let mut boot_attach_args = Vec::with_capacity(args.len());
        for (i, arg) in args.iter().enumerate() {
            let mut av = self.check_expr_infallible(&arg.value);
            if av.ty.is_linear_lease() {
                self.consume_lease_value(&arg.value, &av, "function argument", arg.span);
            }
            if let Some(param) = fn_info.params.get(i) {
                if self.sema.check_assignable_or_emit(
                    &av.ty,
                    &param.ty,
                    arg.span,
                    DiagnosticKind::TypeMismatch,
                    || {
                        format!(
                            "argument {} ('{}') to '{}' has type '{}', expected '{}'",
                            i + 1,
                            param.name,
                            fn_info.name,
                            av.ty,
                            param.ty
                        )
                    },
                ) {
                    self.check_coerced_expr(&arg.value, &param.ty);
                }
                av.prov = self.coerce_array_to_slice_prov(&arg.value, &av.ty, &param.ty, &av.prov);
            }

            // If calling an `extern "C"` function, record any user `fn` callbacks passed in `arg`!
            if fn_info.is_extern_c {
                for cb_fn in &av.prov.fn_callbacks {
                    self.sema
                        .call_graph
                        .add_extern_callback(ExternCallbackSite {
                            extern_fn: fn_info.name.clone(),
                            callback_fn: cb_fn.clone(),
                            span: arg.span,
                        });
                }
            }

            let endpoint_from_param = self.resolve_param_endpoint_index(&av.endpoint_name);
            let static_endpoint = av
                .endpoint_name
                .clone()
                .filter(|ep| !ep.starts_with("param:"));
            let field_endpoints = self.extract_field_endpoint_specs(&av);
            let has_non_static_local_ref = av.prov.has_non_returning_origin();
            let has_isr_state_ref = !av.prov.isr_state_origins.is_empty();
            boot_call_args.push(BootCallArg {
                static_endpoint,
                endpoint_from_param,
                field_endpoints,
                has_non_static_local_ref,
                has_isr_state_ref,
                ref_param_origins: av.prov.param_origins.clone(),
                span: arg.span,
            });
            boot_attach_args.push(BootAttachCallArg {
                spec: self.classify_irq_attach_arg(&arg.value, &av),
            });
            arg_vals.push(av);
        }

        if !fn_info.is_extern_c {
            let in_main_post_boot = self.in_main_post_boot();
            let in_main_after_spawn = self.in_main_after_spawn();
            let in_main_after_irq_attach = self.in_main_after_irq_attach();
            self.sema.recorded_call_args.push(RecordedCallArgs {
                caller: self.caller_key.clone(),
                callee: fn_info.name.clone(),
                args: boot_call_args.clone(),
                in_main_post_boot,
                in_main_after_spawn,
                in_main_after_irq_attach,
            });
            if self.caller_key == "main" {
                let (spawns, attaches) = self.sema.fn_boot_flags(&fn_info.name);
                if spawns {
                    self.has_spawned = true;
                }
                if attaches {
                    self.has_attached_irq = true;
                }
            }
            if self.can_record_boot_step() {
                self.boot_steps.push(BootStep::HelperCall {
                    callee: fn_info.name.clone(),
                    conditional: self.branch_depth > 0,
                    args: boot_call_args,
                });
                if let Some(top) = self.boot_block_stack.last_mut() {
                    top.push(IrqBootNode::Call {
                        callee: fn_info.name.clone(),
                        args: boot_attach_args,
                    });
                }
            }
        }

        // Compute return value provenance if `fn_info.ret_ty` carries references/slices.
        let mut ret_prov = Provenance::pure_static();
        if fn_info.ret_ty.contains_reference(&self.sema.structs) {
            if !fn_info.return_param_origins.is_empty() {
                for &idx in &fn_info.return_param_origins {
                    if let Some(av) = arg_vals.get(idx) {
                        ret_prov = ret_prov.merge(&av.prov);
                    }
                }
            } else {
                // Conservative fallback for forward/mutual non-yielding helpers not yet analyzed:
                // propagate reference-bearing arguments into reference-bearing return value.
                for (i, av) in arg_vals.iter().enumerate() {
                    if fn_info
                        .params
                        .get(i)
                        .is_some_and(|p| p.ty.contains_reference(&self.sema.structs))
                    {
                        ret_prov = ret_prov.merge(&av.prov);
                    }
                }
            }
        }

        ExprVal {
            ty: fn_info.ret_ty.clone(),
            error_spec: fn_info.error_spec.clone(),
            prov: ret_prov,
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }

    pub(super) fn try_check_builtin_fn(
        &mut self,
        name: &str,
        args: &[CallArg],
        span: Span,
    ) -> Option<ExprVal> {
        let builtin = BuiltinFn::from_name(name)?;
        match builtin {
            BuiltinFn::SetPriority => {
                if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        span,
                        format!(
                            "cannot call 'set_priority' inside ISR '{}'",
                            self.display_caller_name()
                        ),
                    ));
                }
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'set_priority' expects 1 priority argument, found {}",
                            args.len()
                        ),
                    ));
                }
                for a in args {
                    let av = self.check_expr_infallible(&a.value);
                    if av.ty != Ty::Priority && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            a.span,
                            format!("'set_priority' expects a priority literal (@critical..@low), found '{}'", av.ty),
                        ));
                    }
                    if self.caller_key == "main" {
                        if let ExprKind::Priority(prio) = a.value.kind {
                            self.sema.main_supervisor_priority = Some(prio);
                        }
                    }
                }
                Some(ExprVal::scalar(Ty::void(), span))
            }

            BuiltinFn::Checkpoint => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        "'checkpoint()' takes 0 arguments",
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                self.record_direct_yield(span, FnEffect::SoftYielding, "'checkpoint()'");
                Some(ExprVal::scalar(Ty::void(), span))
            }

            BuiltinFn::Delay | BuiltinFn::Sleep => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'{}' expects 1 duration argument, found {}",
                            name,
                            args.len()
                        ),
                    ));
                }
                for a in args {
                    let av = self.check_expr_infallible(&a.value);
                    if av.ty != Ty::Duration && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            a.span,
                            format!(
                                "'{}' expects a duration literal (e.g. 10ms), found '{}'",
                                name, av.ty
                            ),
                        ));
                    }
                }
                self.record_direct_yield(span, FnEffect::HardBlocking, "'delay()'");
                Some(ExprVal::scalar(Ty::void(), span))
            }

            BuiltinFn::DelayMs => None,

            BuiltinFn::NowTicks | BuiltinFn::NowMs => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'{}()' takes 0 arguments, found {}", name, args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                Some(ExprVal::scalar(Ty::u32(), span))
            }

            BuiltinFn::Print | BuiltinFn::Println => {
                if args.len() != 1 {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "'{}' expects 1 argument (`*u8`, string literal, or `f\"...\"` format string), found {}",
                                name,
                                args.len()
                            ),
                        )
                        .with_note(format!(
                            "pass `{}(f\"val={{x}}\");` or `{}(\"boot ok\");`",
                            name, name
                        )),
                    );
                    for a in args {
                        if let ExprKind::FormatString(parts) = &a.value.kind {
                            let _ = self.check_format_string(parts);
                        } else {
                            let _ = self.check_expr_infallible(&a.value);
                        }
                    }
                } else if let ExprKind::FormatString(parts) = &args[0].value.kind {
                    let _ = self.check_format_string(parts);
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !is_print_arg_ty(&av.ty) && !av.ty.is_error() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                args[0].span,
                                format!(
                                    "'{}' expects a string or byte slice (`*u8`, `[]u8`, string literal, or `f\"...\"`), found '{}'",
                                    name, av.ty
                                ),
                            )
                            .with_note(format!(
                                "use `{}(f\"val={{expr}}\");`, `sys.log_kv(\"val=\", expr);`, or `sys.log_hex(\"val=\", expr);` to print values",
                                name
                            )),
                        );
                    }
                }
                Some(ExprVal::scalar(Ty::void(), span))
            }

            BuiltinFn::FmtBuf => Some(self.check_fmt_buf_call(args, span)),
            BuiltinFn::FmtRing => Some(self.check_fmt_ring_call(args, span)),

            BuiltinFn::Assert => {
                if args.len() != 1 {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "'assert' expects 1 'bool' condition argument, found {}",
                                args.len()
                            ),
                        )
                        .with_note("pass a boolean comparison such as `assert(count > 0);`"),
                    );
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !av.ty.is_bool() && !av.ty.is_error() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                args[0].span,
                                format!("'assert' expects a 'bool' condition, found '{}'", av.ty),
                            )
                            .with_note(
                                "pass a boolean comparison such as `assert(val == expected);`",
                            ),
                        );
                    }
                }
                Some(ExprVal::scalar(Ty::void(), span))
            }
        }
    }

    pub(super) fn check_method_call(
        &mut self,
        receiver: &Expr,
        method: &Ident,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        let rv = self.check_projection_base(receiver);
        if rv.ty.is_error() {
            for a in args {
                let _ = self.check_expr_infallible(&a.value);
            }
            return ExprVal::error(span);
        }

        // Auto-deref `*OsPrimitive` / `*Struct` on method calls (`rx_dma.wait_chunk()`, `target_rpm.read()`).
        let deref_ty = match &rv.ty {
            Ty::Ptr(inner) => (**inner).clone(),
            other => other.clone(),
        };

        match deref_ty {
            Ty::Namespace(NamespaceKind::Sys) => self.check_sys_method(method, args, span),

            Ty::Namespace(NamespaceKind::Sim) => {
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                let suggestion = match method.name.as_str() {
                    "log" => "did you mean `sys.log_kv(\"label=\", val)` or `sys.log(\"msg\")`?",
                    "now_ms" | "now_ticks" => "did you mean `sys.now_ms()` or `sys.now_ticks()`?",
                    "exit" | "stop" => "did you mean `sys.exit(0)`?",
                    "pend_irq" | "fire_isr" => "did you mean `sys.pend_irq(irq_num)`?",
                    _ => "`sim.*` is not a built-in Zev namespace; use portable `sys.*` runtime methods instead",
                };
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        method.span,
                        format!("unknown method 'sim.{}' — {}", method.name, suggestion),
                    )
                    .with_note(VALID_SYS_METHODS_NOTE),
                );
                ExprVal::error(span)
            }

            Ty::Namespace(NamespaceKind::Type(type_name)) => {
                if method.name == "from_bytes" {
                    if args.len() != 1 {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "'{}.from_bytes' expects 1 '[]u8' argument, found {}",
                                type_name,
                                args.len()
                            ),
                        ));
                    }
                    let expected_slice = Ty::Slice(Box::new(Ty::u8()));
                    for a in args {
                        let av = self.check_expr_infallible(&a.value);
                        if !av.ty.is_assignable_to(&expected_slice) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                a.span,
                                format!(
                                    "'{}.from_bytes' expects '[]u8', found '{}'",
                                    type_name, av.ty
                                ),
                            ));
                        }
                    }
                    let struct_ty = Ty::Struct(type_name.clone());
                    if self
                        .sema
                        .structs
                        .get(&type_name)
                        .is_some_and(|s| s.is_subsystem)
                    {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!(
                                "'{}.from_bytes' cannot construct subsystem struct '{}' (requires a wire-value struct)",
                                type_name, type_name
                            ),
                        ));
                    } else if struct_ty.contains_reference(&self.sema.structs) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!(
                                "'{}.from_bytes' requires a wire-value struct without pointer or slice fields",
                                type_name
                            ),
                        ));
                    }
                    ExprVal {
                        ty: struct_ty,
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                } else {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UndefinedSymbol,
                            method.span,
                            format!(
                                "unknown associated function '{}.{}'",
                                type_name, method.name
                            ),
                        )
                        .with_note(format!(
                            "did you mean `{}.from_bytes(slice)`? (decodes a `[]u8` wire slice into `{}`)",
                            type_name, type_name
                        )),
                    );
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                    ExprVal::error(span)
                }
            }

            Ty::OsPrimitive(os_prim) => {
                let source_key = rv
                    .endpoint_name
                    .clone()
                    .or_else(|| root_ident_of(receiver))
                    .unwrap_or_else(|| "<anon_endpoint>".to_string());
                self.check_os_primitive_method(&os_prim, &source_key, method, args, span)
            }

            Ty::Lease(elem) => {
                let mut pos_vals = Vec::new();
                let mut has_boost = false;
                for a in args {
                    let av = self.check_expr_infallible(&a.value);
                    if let Some(arg_name) = &a.name {
                        match arg_name.name.as_str() {
                            "timeout" => {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::InvalidOperation,
                                    a.span,
                                    format!("`timeout:` is not valid on `Lease.{}`", method.name),
                                ));
                            }
                            "boost" => {
                                has_boost = true;
                                if av.ty != Ty::Priority && !av.ty.is_error() {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::TypeMismatch,
                                        a.span,
                                        format!(
                                            "'boost:' must be a priority (@critical..@low), found '{}'",
                                            av.ty
                                        ),
                                    ));
                                }
                                if let ExprKind::Priority(prio) = a.value.kind {
                                    self.recorded_boosts.push(PriorityBoostSite {
                                        primitive_name: "<Lease>".to_string(),
                                        boost_priority: prio,
                                        span: a.span,
                                    });
                                }
                            }
                            other => {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    arg_name.span,
                                    format!(
                                        "unknown named argument '{}:' on `Lease.{}`",
                                        other, method.name
                                    ),
                                ));
                            }
                        }
                    } else {
                        pos_vals.push((av, a.value.clone(), a.span));
                    }
                }
                match method.name.as_str() {
                    "get" => {
                        self.expect_positional_arity("get", &pos_vals, 0, span);
                        if has_boost {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                span,
                                "`boost:` is only valid on `Lease.release(...)`",
                            ));
                        }
                        if let ExprKind::Ident(id) = &receiver.kind {
                            if let Some(var) = self.lookup_local(&id.name).cloned() {
                                if let Some(key) = var.lease_gen_key {
                                    let gen = self.leases.acquire_next(&key, span);
                                    let prov =
                                        Provenance::dma_lease_get(key, gen, var.decl_depth, span);
                                    return ExprVal {
                                        ty: Ty::Ptr(elem),
                                        error_spec: ResolvedErrorSpec::None,
                                        prov,
                                        endpoint_name: None,
                                        field_endpoints: BTreeMap::new(),
                                        span,
                                    };
                                }
                            }
                            ExprVal::scalar(Ty::Ptr(elem), span)
                        } else {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::LinearLeaseViolation,
                                span,
                                "cannot borrow `.get()` from a temporary unnamed `Lease<T>` (bind it to a local variable and release it after use)",
                            ));
                            ExprVal::error(span)
                        }
                    }
                    "addr" => {
                        self.expect_positional_arity("addr", &pos_vals, 0, span);
                        if has_boost {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                span,
                                "`boost:` is only valid on `Lease.release(...)`",
                            ));
                        }
                        if !matches!(receiver.kind, ExprKind::Ident(_)) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::LinearLeaseViolation,
                                span,
                                "cannot call `.addr()` on a temporary unnamed `Lease<T>` (bind it to a local variable first)",
                            ));
                            return ExprVal::error(span);
                        }
                        ExprVal::scalar(Ty::usize(), span)
                    }
                    "release" => {
                        self.expect_positional_arity("release", &pos_vals, 0, span);
                        self.consume_lease_value(receiver, &rv, "release", span);
                        ExprVal::scalar(Ty::void(), span)
                    }
                    _ => {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::UndefinedSymbol,
                                method.span,
                                format!("linear `Lease<{}>` has no method '{}'", elem, method.name),
                            )
                            .with_note("valid methods on `Lease<T>`: `.get() -> *T`, `.addr() -> usize`, `.release([boost: @prio])`"),
                        );
                        ExprVal::error(span)
                    }
                }
            }

            Ty::Struct(struct_name) => {
                if let Some(FieldInfo {
                    ty: Ty::FnPtr { params, ret },
                    ..
                }) = self
                    .sema
                    .structs
                    .get(&struct_name)
                    .and_then(|s| s.field(&method.name))
                    .cloned()
                {
                    let field_prov = if let Some(root) = root_ident_of(receiver) {
                        if let Some(v) = self.lookup_local(&root) {
                            v.field_prov
                                .get(&method.name)
                                .cloned()
                                .unwrap_or_else(|| rv.prov.clone())
                        } else {
                            rv.prov.clone()
                        }
                    } else {
                        rv.prov.clone()
                    };
                    if args.len() != params.len() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!(
                                "function pointer field '{}.{}' expects {} argument(s), found {}",
                                struct_name,
                                method.name,
                                params.len(),
                                args.len()
                            ),
                        ));
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let av = self.check_expr_infallible(&arg.value);
                        if av.ty.is_linear_lease() {
                            self.consume_lease_value(
                                &arg.value,
                                &av,
                                "function pointer field argument",
                                arg.span,
                            );
                        }
                        if let Some(expected) = params.get(i) {
                            if !av.ty.is_assignable_to(expected) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    arg.span,
                                    format!(
                                        "argument {} to '{}.{}' has type '{}', expected '{}'",
                                        i + 1,
                                        struct_name,
                                        method.name,
                                        av.ty,
                                        expected
                                    ),
                                ));
                            } else {
                                self.check_coerced_expr(&arg.value, expected);
                            }
                        }
                    }
                    for cb_target in &field_prov.fn_callbacks {
                        self.sema.call_graph.add_edge(CallEdge {
                            caller: self.caller_key.clone(),
                            caller_kind: self.caller_kind,
                            callee: cb_target.clone(),
                            span,
                            ctx: self.call_context(),
                        });
                    }
                    return ExprVal::scalar(*ret, span);
                }

                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    method.span,
                    format!("type '{}' has no method '{}'", struct_name, method.name),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                ExprVal::error(span)
            }

            Ty::IrqToken => self.check_irq_method(&rv, method, args, span),

            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    method.span,
                    format!("type '{}' has no method '{}'", other, method.name),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                ExprVal::error(span)
            }
        }
    }

    pub(super) fn check_sys_method(
        &mut self,
        method: &Ident,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        let m = method.name.as_str();
        match SysMethod::from_name(m) {
            Some(SysMethod::UseSystick) => {
                if args.is_empty() || args.len() > 2 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.use_systick' expects 1 argument (u32 hz), found {}",
                            args.len()
                        ),
                    ));
                }
                if let Some(hz_arg) = args.first() {
                    let hv = self.check_expr_infallible(&hz_arg.value);
                    if !hv.ty.is_integer() && !hv.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            hz_arg.span,
                            format!(
                                "'sys.use_systick' clock frequency must be an integer (`u32`), found '{}'",
                                hv.ty
                            ),
                        ));
                    } else {
                        self.validate_int_expr_at_type(&hz_arg.value, Some(PrimType::U32));
                    }
                }
                // Optional second argument configures custom tick period (`sys.use_systick(hz, 500us)`).
                if let Some(tick_arg) = args.get(1) {
                    if !matches!(&tick_arg.value.kind, ExprKind::Literal(Literal::Duration(d)) if d.as_micros() > 0)
                    {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidDurationResolution,
                            tick_arg.span,
                            "'sys.use_systick' tick period must be a non-zero duration literal",
                        ));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::SetTick | SysMethod::SetTickPeriod) => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.{}' expects 1 duration literal argument", m),
                    ));
                } else if !matches!(&args[0].value.kind, ExprKind::Literal(Literal::Duration(d)) if d.as_micros() > 0)
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidDurationResolution,
                        args[0].span,
                        format!("'sys.{}' requires a non-zero duration literal", m),
                    ));
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::SetTickUs) => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        "'sys.set_tick_us' expects 1 integer microseconds argument",
                    ));
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !av.ty.is_integer() && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!(
                                "'sys.set_tick_us' expects integer microseconds, found '{}'",
                                av.ty
                            ),
                        ));
                    } else {
                        self.validate_int_expr_at_type(&args[0].value, None);
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::Delay) => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.delay' expects 1 duration argument, found {}",
                            args.len()
                        ),
                    ));
                }
                for a in args {
                    let av = self.check_expr_infallible(&a.value);
                    if av.ty != Ty::Duration && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            a.span,
                            format!("'sys.delay' expects duration literal, found '{}'", av.ty),
                        ));
                    }
                }
                self.record_direct_yield(span, FnEffect::HardBlocking, "sys.delay()");
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::Tick | SysMethod::TickMs) => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.{}()' takes 0 arguments, found {}", m, args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::TickBy) => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.tick_by(u32 ms)' expects 1 argument, found {}",
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !av.ty.is_assignable_to(&Ty::u32()) && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!(
                                "'sys.tick_by' expects `u32` milliseconds, found '{}'",
                                av.ty
                            ),
                        ));
                    } else {
                        self.check_coerced_expr(&args[0].value, &Ty::u32());
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::SetPutc) => {
                if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        span,
                        "cannot call 'sys.set_putc' inside an ISR",
                    ));
                }
                let expected_ty = Ty::FnPtr {
                    params: vec![Ty::u8()],
                    ret: Box::new(Ty::void()),
                };
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.set_putc(fn(u8))' expects 1 argument, found {}",
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    self.check_sys_callback_arg(
                        "set_putc",
                        "console byte-sink",
                        &args[0],
                        &expected_ty,
                    );
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(
                hook @ (SysMethod::SetClockMs
                | SysMethod::SetExit
                | SysMethod::SetFaultHandler
                | SysMethod::SetIdle),
            ) => {
                if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        span,
                        format!("cannot call 'sys.{}' inside an ISR", m),
                    ));
                }
                let (sig_desc, role_desc, expected_ty) = match hook {
                    SysMethod::SetClockMs => (
                        "fn() -> u32",
                        "clock-source",
                        Ty::FnPtr {
                            params: vec![],
                            ret: Box::new(Ty::u32()),
                        },
                    ),
                    SysMethod::SetExit => (
                        "fn(i32)",
                        "exit-handler",
                        Ty::FnPtr {
                            params: vec![Ty::Prim(PrimType::I32)],
                            ret: Box::new(Ty::void()),
                        },
                    ),
                    SysMethod::SetFaultHandler => (
                        "fn()",
                        "fault-handler",
                        Ty::FnPtr {
                            params: vec![],
                            ret: Box::new(Ty::void()),
                        },
                    ),
                    SysMethod::SetIdle => (
                        "fn(u32, u32)",
                        "idle-hook",
                        Ty::FnPtr {
                            params: vec![Ty::u32(), Ty::u32()],
                            ret: Box::new(Ty::void()),
                        },
                    ),
                    _ => unreachable!(),
                };
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.{}({})' expects 1 argument, found {}",
                            m,
                            sig_desc,
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    self.check_sys_callback_arg(m, role_desc, &args[0], &expected_ty);
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::SetIrqRouter) => {
                if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        span,
                        "cannot call 'sys.set_irq_router' inside an ISR",
                    ));
                }
                let route_ty = Ty::FnPtr {
                    params: vec![Ty::u32(), Ty::u32()],
                    ret: Box::new(Ty::void()),
                };
                let ack_ty = Ty::FnPtr {
                    params: vec![Ty::u32()],
                    ret: Box::new(Ty::void()),
                };
                if args.len() != 2 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.set_irq_router(fn(u32, u32), fn(u32))' expects 2 arguments, found {}",
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    self.check_sys_callback_arg("set_irq_router", "IRQ-route", &args[0], &route_ty);
                    self.check_sys_callback_arg("set_irq_router", "IRQ-ack", &args[1], &ack_ty);
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::RomCallU32) => self.check_sys_rom_call(m, args, span, true, Ty::u32()),
            Some(SysMethod::RomCallVoid) => {
                self.check_sys_rom_call(m, args, span, true, Ty::void())
            }
            Some(SysMethod::RomCallNoreturn) => {
                self.check_sys_rom_call(m, args, span, false, Ty::void())
            }

            Some(
                SysMethod::NowTicks
                | SysMethod::NowMs
                | SysMethod::CurrentPrio
                | SysMethod::SleepDepth
                | SysMethod::WfiCount,
            ) => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.{}()' takes 0 arguments, found {}", m, args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                ExprVal::scalar(Ty::u32(), span)
            }

            Some(SysMethod::TraceSleep) => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.trace_sleep(bool on)' expects 1 argument, found {}",
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if av.ty != Ty::bool() && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!("'sys.trace_sleep' expects 'bool', found '{}'", av.ty),
                        ));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::Nop) => {
                if !args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!("'sys.{}()' takes 0 arguments, found {}", m, args.len()),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::PendIrq) => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.pend_irq(u32 irq)' expects 1 argument, found {}",
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if av.ty != Ty::IrqToken
                        && !av.ty.is_assignable_to(&Ty::u32())
                        && !av.ty.is_error()
                    {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!(
                                "'sys.pend_irq' expects `u32` or `irq` vector token, found '{}'",
                                av.ty
                            ),
                        ));
                    } else if av.ty != Ty::IrqToken {
                        self.check_coerced_expr(&args[0].value, &Ty::u32());
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::FuzzIrqs) => {
                if args.len() != 2 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.fuzz_irqs(u32 seed, u32 irq_mask)' expects 2 arguments, found {}",
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let s_val = self.check_expr_infallible(&args[0].value);
                    if !s_val.ty.is_assignable_to(&Ty::u32()) && !s_val.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!(
                                "'sys.fuzz_irqs' seed argument expects `u32`, found '{}'",
                                s_val.ty
                            ),
                        ));
                    } else {
                        self.check_coerced_expr(&args[0].value, &Ty::u32());
                    }
                    let m_val = self.check_expr_infallible(&args[1].value);
                    if !m_val.ty.is_assignable_to(&Ty::u32()) && !m_val.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[1].span,
                            format!(
                                "'sys.fuzz_irqs' irq_mask argument expects `u32`, found '{}'",
                                m_val.ty
                            ),
                        ));
                    } else {
                        self.check_coerced_expr(&args[1].value, &Ty::u32());
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::Exit) => {
                if args.len() != 1 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.exit(i32 code)' expects 1 argument, found {}",
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !av.ty.is_assignable_to(&Ty::Prim(PrimType::I32)) && !av.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!("'sys.exit' expects `i32` exit code, found '{}'", av.ty),
                        ));
                    } else {
                        self.check_coerced_expr(&args[0].value, &Ty::Prim(PrimType::I32));
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(SysMethod::Log) => {
                if args.len() != 1 {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            span,
                            format!("'sys.log(*u8 msg)' expects 1 string argument, found {}", args.len()),
                        )
                        .with_note("for labeled integer telemetry, use `sys.log_kv(\"label=\", val);` or `sys.log_hex(\"reg=\", val);`"),
                    );
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let av = self.check_expr_infallible(&args[0].value);
                    if !is_string_arg_ty(&av.ty) && !av.ty.is_error() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                args[0].span,
                                format!("'sys.log' expects a string (`*u8`), found '{}'", av.ty),
                            )
                            .with_note("use `sys.log_kv(\"label=\", val);` or `sys.log_hex(\"label=\", val);` to log numeric values"),
                        );
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            Some(kind @ (SysMethod::LogKv | SysMethod::LogHex)) => {
                let val_ty = if kind == SysMethod::LogKv {
                    Ty::Prim(PrimType::I32)
                } else {
                    Ty::u32()
                };
                if args.len() != 2 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "'sys.{}(*u8 label, {} val)' expects 2 arguments, found {}",
                            m,
                            val_ty,
                            args.len()
                        ),
                    ));
                    for a in args {
                        let _ = self.check_expr_infallible(&a.value);
                    }
                } else {
                    let label_val = self.check_expr_infallible(&args[0].value);
                    if !is_string_arg_ty(&label_val.ty) && !label_val.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[0].span,
                            format!(
                                "argument 1 ('label') to 'sys.{}' must be a string (`*u8`), found '{}'",
                                m, label_val.ty
                            ),
                        ));
                    }
                    let num_val = self.check_expr_infallible(&args[1].value);
                    if !num_val.ty.is_assignable_to(&val_ty) && !num_val.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            args[1].span,
                            format!(
                                "argument 2 ('val') to 'sys.{}' must be '{}', found '{}'",
                                m, val_ty, num_val.ty
                            ),
                        ));
                    } else {
                        self.check_coerced_expr(&args[1].value, &val_ty);
                    }
                }
                ExprVal::scalar(Ty::void(), span)
            }

            None => {
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                let msg = if matches!(m, "print" | "println" | "puts") {
                    format!(
                        "unknown runtime method 'sys.{}' — did you mean top-level `println(\"...\")` or `sys.log(\"...\")`?",
                        m
                    )
                } else if let Some(best) = closest_match(m, SysMethod::ALL_NAMES.iter().copied(), 2)
                {
                    format!(
                        "unknown runtime method 'sys.{}' — did you mean `sys.{}(...)`?",
                        m, best
                    )
                } else {
                    format!("unknown runtime method 'sys.{}'", m)
                };
                self.sema.emit(
                    Diagnostic::new(DiagnosticKind::UndefinedSymbol, method.span, msg)
                        .with_note(VALID_SYS_METHODS_NOTE),
                );
                ExprVal::error(span)
            }
        }
    }

    pub(super) fn check_sys_callback_arg(
        &mut self,
        method_name: &str,
        role_desc: &str,
        arg: &CallArg,
        expected_ty: &Ty,
    ) {
        let av = self.check_expr_infallible(&arg.value);
        if av.ty != *expected_ty && !av.ty.is_error() {
            let msg = if method_name == "set_putc" {
                format!(
                    "'sys.set_putc' expects callback of type '{}', found '{}'",
                    expected_ty, av.ty
                )
            } else {
                format!(
                    "'sys.{}' expects {} callback of type '{}', found '{}'",
                    method_name, role_desc, expected_ty, av.ty
                )
            };
            self.sema
                .emit(Diagnostic::new(DiagnosticKind::TypeMismatch, arg.span, msg));
        }
        for cb_fn in &av.prov.fn_callbacks {
            if let Some(fn_info) = self.sema.functions.get(cb_fn) {
                if fn_info.error_spec.is_fallible() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        arg.span,
                        format!(
                            "cannot pass fallible function '{}' (`error`) to 'sys.{}': {} callbacks must not fail",
                            cb_fn, method_name, role_desc
                        ),
                    ));
                }
            }
            self.sema
                .call_graph
                .add_extern_callback(ExternCallbackSite {
                    extern_fn: format!("sys.{}", method_name),
                    callback_fn: cb_fn.clone(),
                    span: arg.span,
                });
        }
    }

    pub(super) fn check_sys_rom_call(
        &mut self,
        method_name: &str,
        args: &[CallArg],
        span: Span,
        has_stack_bound: bool,
        ret_ty: Ty,
    ) -> ExprVal {
        let fixed = if has_stack_bound { 2 } else { 1 };
        let max_args = fixed + 4;
        if args.len() < fixed || args.len() > max_args {
            let sig = if has_stack_bound {
                format!(
                    "'sys.{}(usize fn_addr, usize stack_bound, ...args)'",
                    method_name
                )
            } else {
                format!("'sys.{}(usize fn_addr, ...args)'", method_name)
            };
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "{} expects {}..={} arguments (up to 4 ABI register args), found {}",
                    sig,
                    fixed,
                    max_args,
                    args.len()
                ),
            ));
            for a in args {
                let _ = self.check_expr_infallible(&a.value);
            }
            return ExprVal::scalar(ret_ty, span);
        }

        let addr_val = self.check_expr_infallible(&args[0].value);
        if !addr_val.ty.is_assignable_to(&Ty::usize()) && !addr_val.ty.is_error() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                args[0].span,
                format!(
                    "'sys.{}' argument 1 (`fn_addr`) expects `usize` address, found '{}'",
                    method_name, addr_val.ty
                ),
            ));
        } else {
            self.check_coerced_expr(&args[0].value, &Ty::usize());
        }

        if has_stack_bound {
            let bound_val = self.check_expr_infallible(&args[1].value);
            if !bound_val.ty.is_assignable_to(&Ty::usize()) && !bound_val.ty.is_error() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    args[1].span,
                    format!(
                        "'sys.{}' argument 2 (`stack_bound`) expects `usize` bytes, found '{}'",
                        method_name, bound_val.ty
                    ),
                ));
            } else {
                self.check_coerced_expr(&args[1].value, &Ty::usize());
            }
            let const_bound = self
                .eval_const_usize_expr(&args[1].value)
                .and_then(|v| u32::try_from(v).ok());
            if const_bound.is_none() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    args[1].span,
                    format!(
                        "'sys.{}' argument 2 (`stack_bound`) must be a non-negative compile-time constant integer (`0..={}`)",
                        method_name,
                        u32::MAX
                    ),
                ));
            }
        }

        for (idx, a) in args[fixed..].iter().enumerate() {
            let av = self.check_expr_infallible(&a.value);
            if !av.ty.is_integer() && !matches!(av.ty, Ty::Ptr(_)) && !av.ty.is_error() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    a.span,
                    format!(
                        "'sys.{}' ABI argument {} must be an integer or pointer, found '{}'",
                        method_name,
                        fixed + idx + 1,
                        av.ty
                    ),
                ));
            }
        }

        ExprVal::scalar(ret_ty, span)
    }

    pub(super) fn check_os_primitive_method(
        &mut self,
        prim: &ResolvedOsPrimitive,
        source_key: &str,
        method: &Ident,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        // Partition positional vs named arguments (`timeout: <dur>`, `boost: @prio`).
        let mut pos_vals = Vec::new();
        let mut has_timeout = false;
        let mut has_boost = false;

        for a in args {
            let av = self.check_expr_infallible(&a.value);
            if let Some(arg_name) = &a.name {
                match arg_name.name.as_str() {
                    "timeout" => {
                        has_timeout = true;
                        if av.ty != Ty::Duration && !av.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                a.span,
                                format!("'timeout:' must be a duration literal, found '{}'", av.ty),
                            ));
                        }
                    }
                    "boost" => {
                        has_boost = true;
                        if av.ty != Ty::Priority && !av.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                a.span,
                                format!(
                                    "'boost:' must be a priority (@critical..@low), found '{}'",
                                    av.ty
                                ),
                            ));
                        }
                        if let ExprKind::Priority(prio) = a.value.kind {
                            self.recorded_boosts.push(PriorityBoostSite {
                                primitive_name: source_key.to_string(),
                                boost_priority: prio,
                                span: a.span,
                            });
                        }
                    }
                    other => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            arg_name.span,
                            format!("unknown named argument '{}:' on '{}'", other, prim),
                        ));
                    }
                }
            } else {
                pos_vals.push((av, a.value.clone(), a.span));
            }
        }

        let m = method.name.as_str();
        match prim {
            ResolvedOsPrimitive::Signal => match m {
                "wait" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.wait()", source_key),
                    );
                    ExprVal {
                        ty: Ty::void(),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "notify" | "clear" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "try_wait" | "is_set" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::bool(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::WaitQueue => match m {
                "prepare_wait" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "'wait_queue.prepare_wait()' does not accept 'timeout:' or 'boost:' arguments",
                        ));
                    }
                    if self.in_isr() {
                        let isr_name = self.display_caller_name().to_string();
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::YieldInIsr,
                            span,
                            format!(
                                "cannot call '{}.prepare_wait()' inside ISR '{}'",
                                source_key, isr_name
                            ),
                        ));
                    } else if !self.in_critical() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                span,
                                "'wait_queue.prepare_wait()' must be called inside a 'critical { ... }' block alongside the condition check",
                            )
                            .with_note("arm the waiter bit inside `critical { ... }` while testing the condition so an ISR wake between check and `commit_wait()` cannot be lost"),
                        );
                    }
                    ExprVal::scalar(Ty::void(), span)
                }
                "cancel_wait" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "'wait_queue.cancel_wait()' does not accept 'timeout:' or 'boost:' arguments",
                        ));
                    }
                    if self.in_isr() {
                        let isr_name = self.display_caller_name().to_string();
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::YieldInIsr,
                            span,
                            format!(
                                "cannot call '{}.cancel_wait()' inside ISR '{}'",
                                source_key, isr_name
                            ),
                        ));
                    }
                    ExprVal::scalar(Ty::void(), span)
                }
                "commit_wait" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "'wait_queue.commit_wait()' does not accept 'boost:'",
                        ));
                    }
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.commit_wait()", source_key),
                    );
                    ExprVal {
                        ty: Ty::void(),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "wake_one" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "'wait_queue.wake_one()' does not accept 'timeout:'",
                        ));
                    }
                    ExprVal::scalar(Ty::bool(), span)
                }
                "wake_all" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "'wait_queue.wake_all()' does not accept 'timeout:'",
                        ));
                    }
                    ExprVal::scalar(Ty::void(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::Watch { elem } => match m {
                "read" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar((**elem).clone(), span)
                }
                "publish" => {
                    self.record_watch_publish(source_key, span);
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "wait_change" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.wait_change()", source_key),
                    );
                    ExprVal {
                        ty: (**elem).clone(),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::Guarded { elem } => match m {
                "get" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "'guarded.get()' does not accept 'timeout:' or 'boost:' arguments",
                        ));
                    }
                    if !self.in_critical() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                span,
                                format!(
                                    "'{}.get()' on `guarded<T>` may only be called inside a 'critical {{ ... }}' block",
                                    source_key
                                ),
                            )
                            .with_note("wrap multi-writer `guarded<T>` reads and writes in `critical { ... }` to prevent task/ISR races"),
                        );
                    }
                    ExprVal::scalar((**elem).clone(), span)
                }
                "set" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "'guarded.set()' does not accept 'timeout:' or 'boost:' arguments",
                        ));
                    }
                    if !self.in_critical() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                span,
                                format!(
                                    "'{}.set()' on `guarded<T>` may only be called inside a 'critical {{ ... }}' block",
                                    source_key
                                ),
                            )
                            .with_note("wrap multi-writer `guarded<T>` reads and writes in `critical { ... }` to prevent task/ISR races"),
                        );
                    }
                    ExprVal::scalar(Ty::void(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::Inbox { elem, .. }
            | ResolvedOsPrimitive::InboxUnsized { elem } => match m {
                "post" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    if elem.is_linear_lease() {
                        if let Some((av, ae, asp)) = pos_vals.first() {
                            self.consume_lease_value(ae, av, "inbox.post", *asp);
                        }
                    }
                    ExprVal::scalar(Ty::void(), span)
                }
                "try_send" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    if elem.is_linear_lease() {
                        if let Some((av, ae, asp)) = pos_vals.first() {
                            self.consume_lease_value(ae, av, "inbox.try_send", *asp);
                        }
                    }
                    ExprVal::scalar(Ty::bool(), span)
                }
                "send" => {
                    self.expect_single_elem(m, &pos_vals, elem, span);
                    if elem.is_linear_lease() {
                        if let Some((av, ae, asp)) = pos_vals.first() {
                            self.consume_lease_value(ae, av, "inbox.send", *asp);
                        }
                    }
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.send()", source_key),
                    );
                    ExprVal {
                        ty: Ty::void(),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "recv" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.recv()", source_key),
                    );
                    ExprVal {
                        ty: (**elem).clone(),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "try_recv" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal {
                        ty: (**elem).clone(),
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "len" | "cap" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::usize(), span)
                }
                "is_empty" | "is_full" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::bool(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::PingPong { elem, .. } => match m {
                "flip" | "start" | "activate" | "stop" | "deactivate" | "release" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "complete_half" => {
                    self.expect_single_elem(m, &pos_vals, &Ty::u8(), span);
                    ExprVal::scalar(Ty::void(), span)
                }
                "wait_chunk" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.wait_chunk()", source_key),
                    );
                    let gen = self.leases.acquire_next(source_key, span);
                    let prov = Provenance::dma_lease(
                        source_key.to_string(),
                        gen,
                        self.current_depth(),
                        span,
                    );
                    ExprVal {
                        ty: Ty::Slice(elem.clone()),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov,
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "active_buf" | "dma_buf" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::Slice(elem.clone()), span)
                }
                "half_len" | "cap" | "base_addr" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::usize(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::Pool { elem, .. } => match m {
                "try_alloc" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`pool.try_alloc()` is non-blocking and does not accept `timeout:` (use `pool.alloc(timeout: <dur>)`)",
                        ));
                    }
                    if has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`pool.try_alloc()` does not accept `boost:`",
                        ));
                    }
                    ExprVal {
                        ty: Ty::Lease(elem.clone()),
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "alloc" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`pool.alloc()` does not accept `boost:`",
                        ));
                    }
                    self.record_direct_yield(
                        span,
                        FnEffect::HardBlocking,
                        &format!("{}.alloc()", source_key),
                    );
                    ExprVal {
                        ty: Ty::Lease(elem.clone()),
                        error_spec: if has_timeout {
                            ResolvedErrorSpec::Untyped
                        } else {
                            ResolvedErrorSpec::None
                        },
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "avail" | "in_use" | "cap" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            format!("`pool.{}()` does not accept `timeout:` or `boost:`", m),
                        ));
                    }
                    ExprVal::scalar(Ty::usize(), span)
                }
                "is_exhausted" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`pool.is_exhausted()` does not accept `timeout:` or `boost:`",
                        ));
                    }
                    ExprVal::scalar(Ty::bool(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::DmaSlot { elem } => match m {
                "park" => {
                    self.expect_single_elem(m, &pos_vals, &Ty::Lease(elem.clone()), span);
                    if has_timeout {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`dma_slot.park()` does not accept `timeout:`",
                        ));
                    }
                    if let Some((av, ae, asp)) = pos_vals.first() {
                        self.consume_lease_value(ae, av, "dma_slot.park", *asp);
                    }
                    ExprVal {
                        ty: Ty::void(),
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "take" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout || has_boost {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`dma_slot.take()` does not accept `timeout:` or `boost:`",
                        ));
                    }
                    ExprVal {
                        ty: Ty::Lease(elem.clone()),
                        error_spec: ResolvedErrorSpec::Untyped,
                        prov: Provenance::pure_static(),
                        endpoint_name: None,
                        field_endpoints: BTreeMap::new(),
                        span,
                    }
                }
                "release" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    if has_timeout {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            span,
                            "`dma_slot.release()` does not accept `timeout:`",
                        ));
                    }
                    ExprVal::scalar(Ty::bool(), span)
                }
                "is_busy" | "is_empty" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::bool(), span)
                }
                "addr" => {
                    self.expect_positional_arity(m, &pos_vals, 0, span);
                    ExprVal::scalar(Ty::usize(), span)
                }
                _ => self.unknown_prim_method(prim, method, span),
            },

            ResolvedOsPrimitive::ByteRing { .. } | ResolvedOsPrimitive::ByteRingUnsized => {
                let spsc_role = match m {
                    "push" | "push_byte" | "grant_write" | "commit_write" => {
                        Some(ByteRingOpRole::Producer)
                    }
                    "wait_read" | "read_slice" | "pop" | "try_pop_byte" | "consume" => {
                        Some(ByteRingOpRole::Consumer)
                    }
                    "clear" => Some(ByteRingOpRole::Both),
                    _ => None,
                };
                if let Some(role) = spsc_role {
                    self.record_byte_ring_op(role, m, source_key, span);
                }
                let u8_slice = Ty::Slice(Box::new(Ty::u8()));
                match m {
                    "push" => {
                        self.expect_single_elem(m, &pos_vals, &u8_slice, span);
                        let wkey = format!("{source_key}#write");
                        let _ = self.leases.invalidate(&wkey, span, "push");
                        ExprVal::scalar(Ty::usize(), span)
                    }
                    "push_byte" => {
                        self.expect_single_elem(m, &pos_vals, &Ty::u8(), span);
                        let wkey = format!("{source_key}#write");
                        let _ = self.leases.invalidate(&wkey, span, "push_byte");
                        ExprVal::scalar(Ty::bool(), span)
                    }
                    "grant_write" => {
                        self.expect_single_elem(m, &pos_vals, &Ty::usize(), span);
                        let wkey = format!("{source_key}#write");
                        let gen = self.leases.invalidate(&wkey, span, "grant_write");
                        let prov = Provenance::ring_lease(
                            wkey,
                            gen,
                            self.current_depth(),
                            span,
                            "grant_write",
                        );
                        ExprVal {
                            ty: u8_slice,
                            error_spec: ResolvedErrorSpec::Untyped,
                            prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    }
                    "commit_write" => {
                        self.expect_single_elem(m, &pos_vals, &Ty::usize(), span);
                        let wkey = format!("{source_key}#write");
                        let _ = self.leases.invalidate(&wkey, span, "commit_write");
                        ExprVal::scalar(Ty::void(), span)
                    }
                    "read_slice" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        let gen = self.leases.acquire_current(source_key);
                        let prov = Provenance::ring_lease(
                            source_key.to_string(),
                            gen,
                            self.current_depth(),
                            span,
                            "read_slice",
                        );
                        ExprVal {
                            ty: u8_slice,
                            error_spec: ResolvedErrorSpec::None,
                            prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    }
                    "wait_read" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        self.record_direct_yield(
                            span,
                            FnEffect::HardBlocking,
                            &format!("{}.wait_read()", source_key),
                        );
                        let gen = self.leases.acquire_current(source_key);
                        let prov = Provenance::ring_lease(
                            source_key.to_string(),
                            gen,
                            self.current_depth(),
                            span,
                            "wait_read",
                        );
                        ExprVal {
                            ty: u8_slice,
                            error_spec: if has_timeout {
                                ResolvedErrorSpec::Untyped
                            } else {
                                ResolvedErrorSpec::None
                            },
                            prov,
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    }
                    "consume" => {
                        self.expect_single_elem(m, &pos_vals, &Ty::usize(), span);
                        let _ = self.leases.invalidate(source_key, span, "consume");
                        ExprVal::scalar(Ty::void(), span)
                    }
                    "pop" => {
                        self.expect_single_elem(m, &pos_vals, &u8_slice, span);
                        if pos_vals
                            .first()
                            .is_some_and(|(v, _, _)| matches!(v.ty, Ty::Str))
                        {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                pos_vals[0].2,
                                "cannot call 'pop' into an immutable string literal; pass a mutable `[N]u8` or `[]u8` buffer",
                            ));
                        }
                        let _ = self.leases.invalidate(source_key, span, "pop");
                        ExprVal::scalar(Ty::usize(), span)
                    }
                    "try_pop_byte" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        let _ = self.leases.invalidate(source_key, span, "try_pop_byte");
                        ExprVal {
                            ty: Ty::u8(),
                            error_spec: ResolvedErrorSpec::Untyped,
                            prov: Provenance::pure_static(),
                            endpoint_name: None,
                            field_endpoints: BTreeMap::new(),
                            span,
                        }
                    }
                    "clear" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        let _ = self.leases.invalidate(source_key, span, "clear");
                        let wkey = format!("{source_key}#write");
                        let _ = self.leases.invalidate(&wkey, span, "clear");
                        ExprVal::scalar(Ty::void(), span)
                    }
                    "len" | "cap" | "free_space" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        ExprVal::scalar(Ty::usize(), span)
                    }
                    "is_empty" | "is_full" => {
                        self.expect_positional_arity(m, &pos_vals, 0, span);
                        ExprVal::scalar(Ty::bool(), span)
                    }
                    _ => self.unknown_prim_method(prim, method, span),
                }
            }
        }
    }

    pub(super) fn expect_positional_arity(
        &mut self,
        method: &str,
        pos_vals: &[(ExprVal, Expr, Span)],
        expected: usize,
        span: Span,
    ) {
        if pos_vals.len() != expected {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "method '{}' expects {} positional argument(s), found {}",
                    method,
                    expected,
                    pos_vals.len()
                ),
            ));
        }
    }

    pub(super) fn expect_single_elem(
        &mut self,
        method: &str,
        pos_vals: &[(ExprVal, Expr, Span)],
        elem: &Ty,
        span: Span,
    ) {
        if pos_vals.len() != 1 {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "method '{}' expects 1 value argument of type '{}', found {}",
                    method,
                    elem,
                    pos_vals.len()
                ),
            ));
            return;
        }
        let (val, arg_expr, arg_span) = &pos_vals[0];
        if !val.prov.isr_state_origins.is_empty() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidIsrState,
                *arg_span,
                format!(
                    "cannot pass `&` of ISR `state` variable to OS primitive method '{}'",
                    method
                ),
            ));
        }
        if !val.prov.param_origins.is_empty() {
            self.sema
                .prim_escaping_ref_params
                .entry(self.caller_key.clone())
                .or_default()
                .extend(val.prov.param_origins.iter().copied());
        }
        if self.sema.check_assignable_or_emit(
            &val.ty,
            elem,
            *arg_span,
            DiagnosticKind::TypeMismatch,
            || {
                format!(
                    "argument to '{}' has type '{}', expected '{}'",
                    method, val.ty, elem
                )
            },
        ) {
            self.check_coerced_expr(arg_expr, elem);
        }
    }

    pub(super) fn unknown_prim_method(
        &mut self,
        prim: &ResolvedOsPrimitive,
        method: &Ident,
        span: Span,
    ) -> ExprVal {
        let (did_you_mean, valid_table) = advise_primitive_method(prim, &method.name);
        let msg = if let Some(suggestion) = &did_you_mean {
            format!(
                "OS primitive '{}' has no method '{}' — did you mean {}?",
                prim, method.name, suggestion
            )
        } else {
            format!("OS primitive '{}' has no method '{}'", prim, method.name)
        };
        self.sema.emit(
            Diagnostic::new(DiagnosticKind::UndefinedSymbol, method.span, msg)
                .with_note(valid_table),
        );
        ExprVal::error(span)
    }

    pub(super) fn check_catch_expr(
        &mut self,
        inner: &Expr,
        clause: &CatchClause,
        span: Span,
    ) -> ExprVal {
        let iv = self.check_expr_raw(inner);
        let _ = self.reject_unindexed_mmio_array(inner.span);
        let _ = self
            .proven_infallible_spans
            .remove(&(inner.span.start, inner.span.end));

        if !iv.error_spec.is_fallible() && !iv.ty.is_error() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidCatchSignature,
                clause.span,
                "cannot attach 'catch' handler to an infallible expression",
            ));
        }

        // Validate catch parameter count (0, 1, or 2) and types against `(iv.error_spec, iv.ty)`.
        let mut bound_err: Option<(String, Ty, Span)> = None;
        let mut bound_partial: Option<(String, Ty, Span)> = None;

        if let Some(ep) = &clause.err_param {
            let decl_err_ty = self.sema.resolve_type(&ep.ty);
            self.sema
                .check_type_no_byval_mmio(&decl_err_ty, ep.ty.span, "'catch' error parameter");
            self.sema.check_disallowed_lease_type(
                &decl_err_ty,
                ep.ty.span,
                "'catch' error parameter",
                false,
            );
            match &iv.error_spec {
                ResolvedErrorSpec::None => {}
                ResolvedErrorSpec::Untyped => {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidCatchSignature,
                        ep.span,
                        format!(
                            "'catch' binds error parameter '{}', but expression has untyped bare 'error' (use 'catch {{ ... }}')",
                            ep.name.name
                        ),
                    ));
                }
                ResolvedErrorSpec::Typed(actual_err_ty) => {
                    if !actual_err_ty.is_assignable_to(&decl_err_ty) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidCatchSignature,
                            ep.span,
                            format!(
                                "'catch' error parameter '{}' declared as '{}', but expression fails with '{}'",
                                ep.name.name, decl_err_ty, actual_err_ty
                            ),
                        ));
                    }
                }
            }
            bound_err = Some((ep.name.name.clone(), decl_err_ty, ep.span));
        }

        if let Some(pp) = &clause.partial_param {
            if clause.err_param.is_none() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidCatchSignature,
                    pp.span,
                    "'catch' partial-progress parameter requires an error parameter first",
                ));
            }
            let decl_part_ty = self.sema.resolve_type(&pp.ty);
            self.sema.check_type_no_byval_mmio(
                &decl_part_ty,
                pp.ty.span,
                "'catch' partial-progress parameter",
            );
            if decl_part_ty.is_linear_lease()
                || iv.ty.is_linear_lease()
                || ty_contains_linear_lease(&decl_part_ty)
            {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::LinearLeaseViolation,
                    pp.ty.span,
                    "cannot bind linear Lease<T> in 'catch' partial-progress parameter (linear leases cannot be partially produced on error)",
                ));
            } else {
                self.sema.check_disallowed_lease_type(
                    &decl_part_ty,
                    pp.ty.span,
                    "'catch' partial-progress parameter",
                    false,
                );
            }
            if matches!(iv.error_spec, ResolvedErrorSpec::Untyped) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidCatchSignature,
                    pp.span,
                    "'catch' cannot bind partial-progress parameter on untyped bare 'error'",
                ));
            } else if iv.ty.is_void() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidCatchSignature,
                    pp.span,
                    "'catch' cannot bind partial-progress parameter on a 'void' expression",
                ));
            } else if !iv.ty.is_assignable_to(&decl_part_ty) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidCatchSignature,
                    pp.span,
                    format!(
                        "'catch' partial parameter '{}' declared as '{}', expected '{}'",
                        pp.name.name, decl_part_ty, iv.ty
                    ),
                ));
            }
            bound_partial = Some((pp.name.name.clone(), decl_part_ty, pp.span));
        }

        let base_vars = self.snapshot_outer_vars();
        let base_leases = self.leases.clone();

        let (catch_div, trailing_val) = self.with_nested_scope(|this| {
            let depth = this.current_depth();
            if let Some((ename, ety, espan)) = bound_err {
                let init_lease_state = if ety.is_linear_lease() {
                    Some(LinearLeaseState::Moved {
                        moved_at: espan,
                        reason: "invalid catch error binding",
                    })
                } else {
                    None
                };
                if let Some(scope) = this.scopes.last_mut() {
                    scope.insert(
                        ename.clone(),
                        LocalVar {
                            name: ename,
                            ty: ety,
                            decl_depth: depth,
                            prov: Provenance::pure_static(),
                            field_prov: BTreeMap::new(),
                            endpoint_alias: None,
                            field_endpoints: BTreeMap::new(),
                            is_isr_state: false,
                            lease_state: init_lease_state,
                            lease_gen_key: None,
                        },
                    );
                }
            }
            if let Some((pname, pty, pspan)) = bound_partial {
                if let Some(ep) = &clause.err_param {
                    if ep.name.name == pname {
                        this.sema.emit(Diagnostic::new(
                            DiagnosticKind::DuplicateDefinition,
                            pspan,
                            format!("duplicate parameter '{}' in 'catch' clause", pname),
                        ));
                    }
                }
                let init_lease_state = if pty.is_linear_lease() {
                    Some(LinearLeaseState::Moved {
                        moved_at: pspan,
                        reason: "invalid catch partial binding",
                    })
                } else {
                    None
                };
                if let Some(scope) = this.scopes.last_mut() {
                    scope.insert(
                        pname.clone(),
                        LocalVar {
                            name: pname,
                            ty: pty,
                            decl_depth: depth,
                            prov: iv.prov.clone(),
                            field_prov: BTreeMap::new(),
                            endpoint_alias: None,
                            field_endpoints: BTreeMap::new(),
                            is_isr_state: false,
                            lease_state: init_lease_state,
                            lease_gen_key: None,
                        },
                    );
                }
            }

            if !iv.ty.is_void() && !clause.body.stmts.is_empty() {
                let (prefix, last) = clause.body.stmts.split_at(clause.body.stmts.len() - 1);
                let (mut div, warned) = this.check_stmt_slice(prefix);
                let prev_after = this.after_diverge;
                if div {
                    if !warned {
                        this.sema.emit(Diagnostic::new(
                            DiagnosticKind::UnreachableCode,
                            last[0].span,
                            "unreachable statement following diverging control flow",
                        ));
                    }
                    this.after_diverge = true;
                }
                if let StmtKind::Expr(last_expr) = &last[0].kind {
                    let lv = this.check_expr_infallible(last_expr);
                    if lv.ty.is_linear_lease() {
                        this.consume_lease_value(last_expr, &lv, "catch fallback", last_expr.span);
                    }
                    let sys_exit =
                        this.lookup_local(SYS_NAMESPACE).is_none() && is_sys_exit_expr(last_expr);
                    let res_div = div || sys_exit;
                    this.finish_lexical_block(res_div, clause.span);
                    if sys_exit {
                        this.after_diverge = prev_after;
                        return (true, None);
                    }
                    this.after_diverge = prev_after;
                    return (div, Some(lv));
                }
                if this.check_stmt(&last[0]) {
                    div = true;
                }
                this.finish_lexical_block(div, clause.span);
                this.after_diverge = prev_after;
                (div, None)
            } else {
                let div = this.check_block_stmts(&clause.body.stmts);
                this.finish_lexical_block(div, clause.span);
                (div, None)
            }
        });

        let catch_vars = self.snapshot_outer_vars();
        let catch_leases = self.leases.clone();

        let mut merged_prov = iv.prov.clone();
        if !catch_div {
            self.scopes = base_vars.clone();
            self.merge_branch_vars(&base_vars, &[base_vars.clone(), catch_vars], clause.span);
            self.leases = base_leases.clone();
            self.leases.join_branches(&[base_leases, catch_leases]);

            if !iv.ty.is_void() && !iv.ty.is_error() {
                if let Some(tv) = trailing_val {
                    if !tv.ty.is_assignable_to(&iv.ty) && !tv.ty.is_error() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::InvalidCatchSignature,
                                tv.span,
                                format!(
                                    "'catch' fallback expression has type '{}', expected '{}'",
                                    tv.ty, iv.ty
                                ),
                            )
                            .with_note(format!(
                                "a non-diverging 'catch' block for a '{}' expression must end with an expression assignable to '{}'",
                                iv.ty, iv.ty
                            )),
                        );
                    } else if let Some(Stmt {
                        kind: StmtKind::Expr(last_expr),
                        ..
                    }) = clause.body.stmts.last()
                    {
                        self.check_coerced_expr(last_expr, &iv.ty);
                    }
                    merged_prov = merged_prov.merge(&tv.prov);
                } else {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidCatchSignature,
                            clause.span,
                            format!(
                                "'catch' block for non-void '{}' expression must either diverge ('return'/'error'/'continue'/'break') or end with a fallback expression of type '{}'",
                                iv.ty, iv.ty
                            ),
                        )
                        .with_note(format!(
                            "either diverge inside `catch` (`return;`, `continue;`, `break;`, `error ...;`) or end the block with a replacement `{}` value (e.g. `catch (...) {{ count += 1; fallback_val }}`)",
                            iv.ty
                        )),
                    );
                }
            }
        } else {
            self.scopes = base_vars;
            self.leases = base_leases;
        }

        ExprVal {
            ty: iv.ty,
            error_spec: ResolvedErrorSpec::None,
            prov: merged_prov,
            endpoint_name: None,
            field_endpoints: BTreeMap::new(),
            span,
        }
    }
}

impl<'a> BodyChecker<'a> {
    pub(super) fn check_resolved_index_val(
        &mut self,
        tv: ExprVal,
        target: &Expr,
        _index_span: Span,
        span: Span,
    ) -> ExprVal {
        match &tv.ty {
            Ty::Slice(elem) | Ty::Array { elem, .. } => {
                let elem_ty = (**elem).clone();
                let prov = if elem_ty.contains_reference(&self.sema.structs) {
                    tv.prov
                } else {
                    Provenance::pure_static()
                };
                ExprVal {
                    ty: elem_ty,
                    error_spec: ResolvedErrorSpec::None,
                    prov,
                    endpoint_name: None,
                    field_endpoints: BTreeMap::new(),
                    span,
                }
            }
            _ => {
                if !tv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        target.span,
                        format!("cannot index into type '{}'", tv.ty),
                    ));
                }
                ExprVal::error(span)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn check_mmio_reg_write_access(
        &mut self,
        layout_name: &str,
        field: &Ident,
        elem_index: Option<&Expr>,
        op: AssignOp,
        value_expr: &Expr,
        rhs: &ExprVal,
        stmt_span: Span,
    ) {
        if let Some(idx_expr) = elem_index {
            let iv = self.check_expr_infallible(idx_expr);
            self.validate_int_expr_at_type(idx_expr, Some(PrimType::Usize));
            if !iv.ty.is_integer() && !iv.ty.is_error() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    idx_expr.span,
                    format!("array index must be an integer, found '{}'", iv.ty),
                ));
            }
        }

        let Some(layout_info) = self.sema.mmio_layouts.get(layout_name).cloned() else {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::UndefinedSymbol,
                field.span,
                format!("unknown MMIO register layout '{}'", layout_name),
            ));
            return;
        };
        let Some(reg) = layout_info
            .regs
            .iter()
            .find(|r| r.name == field.name)
            .cloned()
        else {
            let candidates = layout_info.regs.iter().map(|r| r.name.as_str());
            let mut diag = Diagnostic::new(
                DiagnosticKind::UndefinedSymbol,
                field.span,
                format!(
                    "MMIO layout '{}' has no register '{}'",
                    layout_name, field.name
                ),
            );
            if let Some(best) = closest_match(&field.name, candidates, 2) {
                diag = diag.with_note(format!("did you mean register `.{}`?", best));
            }
            self.sema.emit(diag);
            return;
        };

        let target_ty = if let Some(idx_expr) = elem_index {
            let Some(len) = reg.array_len else {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    field.span,
                    format!(
                        "cannot index scalar MMIO register '{}.{}' of type '{}'",
                        layout_name, reg.name, reg.elem_ty
                    ),
                ));
                return;
            };
            let shadowed = self.active_local_names();
            if let Some(idx) =
                self.sema
                    .eval_const_expr_i128_shadowed(idx_expr, &mut BTreeSet::new(), &shadowed)
            {
                if idx < 0 || idx as u128 >= len as u128 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        idx_expr.span,
                        format!(
                            "MMIO register array '{}.{}' index {} out of bounds (length {})",
                            layout_name, reg.name, idx, len
                        ),
                    ));
                }
            }
            Ty::Prim(reg.elem_ty)
        } else if reg.array_len.is_some() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                stmt_span,
                format!(
                    "cannot assign whole MMIO register array '{}.{}'; assign an element with '{}.{}[i] = ...' instead",
                    layout_name, reg.name, layout_name, reg.name
                ),
            ));
            return;
        } else {
            Ty::Prim(reg.elem_ty)
        };

        match reg.access {
            MmioAccess::Rw => {}
            MmioAccess::Ro => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    stmt_span,
                    format!(
                        "cannot write to read-only `@ro` MMIO register '{}.{}'",
                        layout_name, reg.name
                    ),
                ));
                return;
            }
            MmioAccess::Wo => {
                if op != AssignOp::Assign {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        stmt_span,
                        format!(
                            "cannot perform compound read-modify-write `{}` on write-only `@wo` MMIO register '{}.{}'; use plain `=`",
                            op.as_str(),
                            layout_name,
                            reg.name
                        ),
                    ));
                    return;
                }
            }
            MmioAccess::W1c => {
                if op != AssignOp::Assign {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        stmt_span,
                        format!(
                            "cannot perform compound read-modify-write `{}` on write-1-to-clear `@w1c` MMIO register '{}.{}'; write masks directly using `=`",
                            op.as_str(),
                            layout_name,
                            reg.name
                        ),
                    ));
                    return;
                }
            }
        }

        if op == AssignOp::Assign {
            if !rhs.ty.is_assignable_to(&target_ty) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    rhs.span,
                    format!(
                        "cannot assign '{}' to MMIO register '{}.{}' of type '{}'",
                        rhs.ty, layout_name, reg.name, target_ty
                    ),
                ));
            } else {
                self.check_coerced_expr(value_expr, &target_ty);
            }
        } else {
            self.check_compound_op_types(op, &target_ty, value_expr, &rhs.ty, stmt_span);
        }
    }

    pub(super) fn build_subsystem_field_chain(
        &self,
        mut base_ty: Ty,
        field_names: &[String],
    ) -> Option<Vec<(String, String)>> {
        let mut chain = Vec::with_capacity(field_names.len());
        for fname in field_names {
            let Ty::Struct(sname) = base_ty else {
                return None;
            };
            let f = self.sema.structs.get(&sname)?.field(fname)?;
            chain.push((fname.clone(), f.ty.to_string()));
            base_ty = f.ty.clone();
        }
        Some(chain)
    }

    pub(super) fn classify_irq_attach_arg(
        &self,
        expr: &Expr,
        val: &ExprVal,
    ) -> Option<BootAttachArgSpec> {
        match &expr.kind {
            ExprKind::Unary {
                op: UnaryOp::AddrOf,
                expr: inner,
            } => {
                let mut rev_fields = Vec::new();
                let mut cur = &**inner;
                while let ExprKind::Field { target, field } = &cur.kind {
                    rev_fields.push(field.name.clone());
                    cur = target;
                }
                rev_fields.reverse();
                if let ExprKind::Ident(id) = &cur.kind {
                    if rev_fields.is_empty() {
                        if self.sema.primitives.contains_key(&id.name) {
                            return Some(BootAttachArgSpec::Static(IrqBindingArg::PrimAddr(
                                id.name.clone(),
                            )));
                        }
                        if self.sema.mmio_instances.contains_key(&id.name) {
                            return Some(BootAttachArgSpec::Static(IrqBindingArg::MmioAddr(
                                id.name.clone(),
                            )));
                        }
                    }
                    if let Some(sub) = self.sema.subsystems.get(&id.name) {
                        let base_ty = Ty::Struct(sub.struct_name.clone());
                        if let Some(field_chain) =
                            self.build_subsystem_field_chain(base_ty, &rev_fields)
                        {
                            return Some(BootAttachArgSpec::Static(IrqBindingArg::SubsystemPath {
                                root: id.name.clone(),
                                root_struct: sub.struct_name.clone(),
                                field_chain,
                            }));
                        }
                    }
                    if !rev_fields.is_empty() {
                        if let Some(idx) = self.param_names.iter().position(|p| p == &id.name) {
                            if let Some(local) = self.lookup_local(&id.name) {
                                let base_ty = match &local.ty {
                                    Ty::Ptr(inner_ty) => (**inner_ty).clone(),
                                    other => other.clone(),
                                };
                                if let Some(field_chain) =
                                    self.build_subsystem_field_chain(base_ty, &rev_fields)
                                {
                                    return Some(BootAttachArgSpec::Param {
                                        index: idx,
                                        field_chain,
                                    });
                                }
                            }
                        }
                    }
                }
                None
            }
            ExprKind::Ident(id) => {
                if let Some(idx) = self.param_names.iter().position(|p| p == &id.name) {
                    return Some(BootAttachArgSpec::Param {
                        index: idx,
                        field_chain: Vec::new(),
                    });
                }
                if self.sema.mmio_instances.contains_key(&id.name) {
                    return Some(BootAttachArgSpec::Static(IrqBindingArg::MmioAddr(
                        id.name.clone(),
                    )));
                }
                let shadowed = self.active_local_names();
                self.sema
                    .eval_compile_time_irq_arg_with_shadowed(
                        expr,
                        &val.ty,
                        &shadowed,
                        self.prog_idx,
                    )
                    .map(BootAttachArgSpec::Static)
            }
            _ => {
                let shadowed = self.active_local_names();
                self.sema
                    .eval_compile_time_irq_arg_with_shadowed(
                        expr,
                        &val.ty,
                        &shadowed,
                        self.prog_idx,
                    )
                    .map(BootAttachArgSpec::Static)
            }
        }
    }

    pub(super) fn check_irq_method(
        &mut self,
        rv: &ExprVal,
        method: &Ident,
        args: &[CallArg],
        span: Span,
    ) -> ExprVal {
        let is_shared = match method.name.as_str() {
            "attach" => false,
            "attach_shared" => true,
            other => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    method.span,
                    format!(
                        "unknown method '{}' on hardware `irq` token; supported: `.attach(...)` and `.attach_shared(...)`",
                        other
                    ),
                ));
                for a in args {
                    let _ = self.check_expr_infallible(&a.value);
                }
                return ExprVal::error(span);
            }
        };

        let Some(irq_name) = rv.endpoint_name.clone() else {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                "`.attach(...)` requires a named top-level `irq` declaration",
            ));
            return ExprVal::error(span);
        };
        let irq_num = self
            .sema
            .irqs
            .get(&irq_name)
            .map(|q| q.irq_num)
            .unwrap_or(0);

        let mut invalid_ctx = false;
        if self.in_defer() {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' inside a 'defer' cleanup statement",
                    irq_name
                ),
            ));
        } else if self.in_isr() {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' inside ISR '{}'; IRQ bindings must happen during sequential boot",
                    irq_name,
                    self.display_caller_name()
                ),
            ));
        } else if matches!(self.body_kind, ActiveBodyKind::Task) {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' inside task '{}'; IRQ bindings must happen during sequential boot",
                    irq_name,
                    self.display_caller_name()
                ),
            ));
        } else if self.loop_depth > 0 {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' inside a loop; IRQ bindings must be static in the boot tree",
                    irq_name
                ),
            ));
        } else if self.after_diverge || self.main_supervisor_started {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "cannot attach IRQ handler on '{}' after 'main()' supervisor loop or diverging control flow",
                    irq_name
                ),
            ));
        }

        if args.len() != 1 || args[0].name.is_some() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                span,
                format!(
                    "`{}.{}(...)` expects 1 positional ISR handler argument (`handler(args...)` or `handler`)",
                    irq_name, method.name
                ),
            ));
            for a in args {
                let _ = self.check_expr_infallible(&a.value);
            }
            return ExprVal::scalar(Ty::void(), span);
        }

        let handler_arg = &args[0];
        let (isr_ident, isr_call_args): (&Ident, &[CallArg]) = match &handler_arg.value.kind {
            ExprKind::Ident(id) => (id, &[]),
            ExprKind::Call {
                callee,
                generic_args,
                args: call_args,
            } => {
                if !generic_args.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::IrqAttachmentError,
                        handler_arg.span,
                        "`isr` handlers cannot accept generic type arguments",
                    ));
                }
                match &callee.kind {
                    ExprKind::Ident(id) => (id, call_args.as_slice()),
                    _ => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            callee.span,
                            "expected named `isr` handler in `.attach(...)`",
                        ));
                        for a in call_args {
                            let _ = self.check_expr_infallible(&a.value);
                        }
                        return ExprVal::scalar(Ty::void(), span);
                    }
                }
            }
            _ => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    handler_arg.span,
                    format!(
                        "expected `isr` handler invocation `isr_name(args...)` or `isr_name` in `{}.{}(...)`",
                        irq_name, method.name
                    ),
                ));
                let _ = self.check_expr_infallible(&handler_arg.value);
                return ExprVal::scalar(Ty::void(), span);
            }
        };

        let Some(isr_info) = self.sema.isrs.get(&isr_ident.name).cloned() else {
            if self.sema.functions.contains_key(&isr_ident.name) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    isr_ident.span,
                    format!(
                        "'{}' is a `fn`, not an `isr` handler; declare `isr {}(...) {{ ... }}` instead",
                        isr_ident.name, isr_ident.name
                    ),
                ));
            } else if self.sema.tasks.contains_key(&isr_ident.name) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    isr_ident.span,
                    format!("'{}' is a `task`, not an `isr` handler", isr_ident.name),
                ));
            } else {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    isr_ident.span,
                    format!("undeclared `isr` handler '{}'", isr_ident.name),
                ));
            }
            for a in isr_call_args {
                let _ = self.check_expr_infallible(&a.value);
            }
            return ExprVal::scalar(Ty::void(), span);
        };

        if isr_call_args.len() != isr_info.params.len() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::IrqAttachmentError,
                handler_arg.span,
                format!(
                    "ISR handler '{}' expects {} argument(s), found {}",
                    isr_info.vector,
                    isr_info.params.len(),
                    isr_call_args.len()
                ),
            ));
        }

        let mut arg_specs = Vec::with_capacity(isr_call_args.len());
        for (idx, a) in isr_call_args.iter().enumerate() {
            let av = self.check_expr_infallible(&a.value);
            let param_name = isr_info
                .params
                .get(idx)
                .map(|p| p.name.clone())
                .unwrap_or_else(|| format!("arg{idx}"));
            if let Some(param) = isr_info.params.get(idx) {
                if !av.ty.is_assignable_to(&param.ty) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        a.span,
                        format!(
                            "argument {} ('{}') to ISR handler '{}' has type '{}', expected '{}'",
                            idx + 1,
                            param.name,
                            isr_info.vector,
                            av.ty,
                            param.ty
                        ),
                    ));
                } else {
                    self.check_coerced_expr(&a.value, &param.ty);
                }
            }
            let spec = self.classify_irq_attach_arg(&a.value, &av);
            if spec.is_none() {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    a.span,
                    format!(
                        "argument {} ('{}') passed to `{}.{}` handler '{}' must be a static OS primitive `&prim`, `mmio` instance, or compile-time constant expression",
                        idx + 1,
                        param_name,
                        irq_name,
                        method.name,
                        isr_info.vector
                    ),
                ));
            }
            let forwarded_pname = match &spec {
                Some(BootAttachArgSpec::Param { index, .. }) => {
                    self.param_names.get(*index).cloned().unwrap_or(param_name)
                }
                _ => param_name,
            };
            arg_specs.push((spec, a.span, forwarded_pname));
        }

        if !invalid_ctx && self.can_record_boot_step() {
            self.sema
                .call_graph
                .mark_direct_attach(&self.caller_key, &irq_name);
        }
        if self.can_record_boot_step() {
            let in_caller = self.display_caller_name().to_string();
            let file = self.sema.current_file.clone();
            if let Some(top) = self.boot_block_stack.last_mut() {
                top.push(IrqBootNode::Attach(PendingIrqAttach {
                    irq_name,
                    irq_num,
                    is_shared,
                    isr_name: isr_info.vector,
                    args: arg_specs,
                    span,
                    file,
                    in_caller,
                    invalid_ctx,
                }));
            }
        }

        if self.caller_key == "main" {
            self.has_attached_irq = true;
        }
        ExprVal::scalar(Ty::void(), span)
    }

    pub(super) fn check_coerced_expr(&mut self, expr: &Expr, target_ty: &Ty) {
        self.sema.check_f32_coercion_expr(expr, target_ty);
        self.validate_int_coercion(expr, target_ty);
    }

    pub(super) fn validate_int_coercion(&mut self, expr: &Expr, target_ty: &Ty) {
        match target_ty {
            Ty::Prim(p) if p.is_integer() => {
                self.validate_int_expr_at_type(expr, Some(Ty::host_prim(*p)));
            }
            Ty::Array { elem, .. } | Ty::Slice(elem) => {
                if let ExprKind::ArrayLit(elems) = &expr.kind {
                    for e in elems {
                        self.validate_int_coercion(e, elem);
                    }
                }
            }
            _ => {}
        }
    }

    pub(super) fn validate_int_expr_at_type(&mut self, expr: &Expr, dest: Option<PrimType>) {
        self.validate_int_expr_at_type_inner(expr, dest, dest.is_some());
    }

    fn validate_int_expr_at_type_inner(
        &mut self,
        expr: &Expr,
        dest: Option<PrimType>,
        check_const_target_range: bool,
    ) {
        let nat_ty = self.expr_tys.get(&(expr as *const Expr as usize)).cloned();
        let eff_prim = match (dest, &nat_ty) {
            (Some(dp), Some(Ty::IntLit)) | (Some(dp), None) => Some(Ty::host_prim(dp)),
            (Some(dp), Some(nt)) if nt.is_assignable_to(&Ty::Prim(dp)) => Some(Ty::host_prim(dp)),
            _ => match &nat_ty {
                Some(Ty::Prim(p)) if p.is_integer() => Some(Ty::host_prim(*p)),
                _ => None,
            },
        };
        match &expr.kind {
            ExprKind::Literal(Literal::Int {
                value,
                suffix: None,
            }) => {
                if let Some(dp) = eff_prim {
                    self.check_int_lit_range_or_emit(i128::from(*value), dp, expr.span);
                }
            }
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => {
                if let ExprKind::Literal(Literal::Int {
                    value,
                    suffix: None,
                }) = &inner.kind
                {
                    if let Some(dp) = eff_prim {
                        self.check_int_lit_range_or_emit(-i128::from(*value), dp, expr.span);
                    }
                } else if nat_ty
                    .as_ref()
                    .is_some_and(|t| t.is_integer() || matches!(t, Ty::IntLit))
                {
                    self.validate_int_expr_at_type_inner(inner, eff_prim, false);
                }
            }
            ExprKind::Unary {
                op: UnaryOp::BitNot,
                expr: inner,
            } => {
                if nat_ty
                    .as_ref()
                    .is_some_and(|t| t.is_integer() || matches!(t, Ty::IntLit))
                {
                    self.validate_int_expr_at_type_inner(inner, eff_prim, false);
                }
            }
            ExprKind::Binary {
                op: op @ (BinaryOp::Shl | BinaryOp::Shr),
                lhs,
                rhs,
            } => {
                if nat_ty
                    .as_ref()
                    .is_some_and(|t| t.is_integer() || matches!(t, Ty::IntLit))
                {
                    let check_ty = eff_prim
                        .map(Ty::Prim)
                        .or_else(|| nat_ty.clone())
                        .unwrap_or(Ty::IntLit);
                    self.check_const_shift_amount(op.as_str(), &check_ty, rhs);
                    self.validate_int_expr_at_type_inner(lhs, eff_prim, false);
                }
            }
            ExprKind::Binary {
                op:
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem
                    | BinaryOp::BitOr
                    | BinaryOp::BitAnd
                    | BinaryOp::BitXor,
                lhs,
                rhs,
            } if nat_ty
                .as_ref()
                .is_some_and(|t| t.is_integer() || matches!(t, Ty::IntLit)) =>
            {
                self.validate_int_expr_at_type_inner(lhs, eff_prim, false);
                self.validate_int_expr_at_type_inner(rhs, eff_prim, false);
            }
            _ => {}
        }

        if check_const_target_range {
            if let Some(dp @ (PrimType::Usize | PrimType::Isize)) = eff_prim {
                let is_direct_lit_or_cast = match &expr.kind {
                    ExprKind::Literal(Literal::Int { .. }) | ExprKind::Cast { .. } => true,
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        expr: inner,
                    } => matches!(inner.kind, ExprKind::Literal(Literal::Int { .. })),
                    _ => false,
                };
                if !is_direct_lit_or_cast && !self.sema.has_reported_span_within(expr.span) {
                    let shadowed = self.active_local_names();
                    if let Some((min_v, max_v)) =
                        Ty::int_prim_range_for_ptr_bits(dp, self.sema.target_ptr_bits)
                    {
                        if let Some(val) = self.sema.eval_const_expr_with_dest_i128(
                            expr,
                            Some(dp),
                            &mut BTreeSet::new(),
                            &shadowed,
                        ) {
                            if (val < min_v || val > max_v)
                                && self
                                    .sema
                                    .f32_overflow_reported_spans
                                    .insert((expr.span.start, expr.span.end))
                            {
                                self.sema.emit(
                                    Diagnostic::new(
                                        DiagnosticKind::TypeMismatch,
                                        expr.span,
                                        format!(
                                            "constant expression {} out of range for '{}' on {}-bit target '{}' ({min_v}..={max_v})",
                                            val,
                                            dp.as_str(),
                                            self.sema.target_ptr_bits,
                                            self.sema.target_name
                                        ),
                                    )
                                    .with_note(format!(
                                        "'{}' has {}-bit pointer width ({min_v}..={max_v}) on target '{}'",
                                        dp.as_str(),
                                        self.sema.target_ptr_bits,
                                        self.sema.target_name
                                    )),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    pub(super) fn check_int_lit_range_or_emit(&mut self, val: i128, prim: PrimType, span: Span) {
        let Some((min_v, max_v)) = Ty::int_prim_range_for_ptr_bits(prim, self.sema.target_ptr_bits)
        else {
            return;
        };
        if (val < min_v || val > max_v)
            && self
                .sema
                .f32_overflow_reported_spans
                .insert((span.start, span.end))
        {
            if matches!(prim, PrimType::Usize | PrimType::Isize) {
                let msg = format!(
                    "integer literal {} out of range for '{}' on {}-bit target '{}' ({min_v}..={max_v})",
                    val,
                    prim.as_str(),
                    self.sema.target_ptr_bits,
                    self.sema.target_name
                );
                let note = format!(
                    "'{}' has {}-bit pointer width ({min_v}..={max_v}) on target '{}' (use `--board`/`--chip` for a wider target or a concrete fixed-width type)",
                    prim.as_str(),
                    self.sema.target_ptr_bits,
                    self.sema.target_name
                );
                self.sema
                    .emit(Diagnostic::new(DiagnosticKind::TypeMismatch, span, msg).with_note(note));
            } else {
                let note = format!("valid range for `{}` is {min_v}..={max_v}", prim.as_str());
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        span,
                        format!(
                            "integer literal {} out of range for '{}'",
                            val,
                            prim.as_str()
                        ),
                    )
                    .with_note(note),
                );
            }
        }
    }
}
