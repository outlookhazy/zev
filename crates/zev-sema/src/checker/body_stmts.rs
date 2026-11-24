use super::*;

impl<'a> BodyChecker<'a> {
    pub(super) fn new(
        sema: &'a mut Analyzer,
        caller_key: &str,
        caller_kind: CallableKind,
        body_kind: ActiveBodyKind,
    ) -> Self {
        let prog_idx = sema.current_prog_idx;
        Self {
            sema,
            caller_key: caller_key.to_string(),
            caller_kind,
            body_kind,
            prog_idx,
            scopes: vec![BTreeMap::new()], // depth 1 = top-level fn/task/isr body
            leases: LeaseEnv::new(),
            loop_depth: 0,
            noyield_depth: 0,
            critical_depth: 0,
            defer_depth: 0,
            branch_depth: 0,
            after_diverge: false,
            main_supervisor_started: false,
            in_main_supervisor_loop: false,
            has_spawned: false,
            has_attached_irq: false,
            param_names: Vec::new(),
            isr_state_names: BTreeSet::new(),
            boot_steps: Vec::new(),
            boot_block_stack: vec![Vec::new()],
            return_param_origins: BTreeSet::new(),
            return_endpoint: None,
            return_field_endpoints: BTreeMap::new(),
            recorded_boosts: Vec::new(),
            proven_infallible_spans: BTreeSet::new(),
            last_struct_lit_fields: BTreeMap::new(),
            mmio_array_expr_spans: BTreeMap::new(),
            loop_entry_depths: Vec::new(),
            loop_outer_reads: Vec::new(),
            loop_lease_frames: Vec::new(),
            active_defer_owner_depth: None,
            next_lease_var_id: 0,
            expr_tys: std::collections::HashMap::new(),
        }
    }

    pub(super) fn current_depth(&self) -> u32 {
        self.scopes.len() as u32
    }

    pub(super) fn active_local_names(&self) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for scope in &self.scopes {
            names.extend(scope.keys().cloned());
        }
        names
    }

    pub(super) fn in_isr(&self) -> bool {
        self.caller_kind == CallableKind::Isr
    }

    pub(super) fn in_critical(&self) -> bool {
        self.critical_depth > 0
    }

    pub(super) fn in_noyield(&self) -> bool {
        self.noyield_depth > 0
    }

    pub(super) fn in_defer(&self) -> bool {
        self.defer_depth > 0
    }

    pub(super) fn can_record_boot_step(&self) -> bool {
        self.caller_kind == CallableKind::Fn
            && self.loop_depth == 0
            && self.defer_depth == 0
            && !self.after_diverge
            && !self.main_supervisor_started
    }

    pub(super) fn in_main_post_boot(&self) -> bool {
        self.caller_key == "main" && (self.has_spawned || self.has_attached_irq)
    }

    pub(super) fn in_main_after_spawn(&self) -> bool {
        self.caller_key == "main"
            && (self.has_spawned || self.in_main_supervisor_loop || self.main_supervisor_started)
    }

    pub(super) fn in_main_after_irq_attach(&self) -> bool {
        self.caller_key == "main" && self.has_attached_irq
    }

    pub(super) fn resolve_endpoint_spec(&self, ep: &str) -> EndpointSpec {
        if let Some(rest) = ep.strip_prefix("param:") {
            let (pname, suffix) = match rest.split_once('.') {
                Some((head, tail)) => (head, Some(tail.to_string())),
                None => (rest, None),
            };
            if let Some(idx) = self.param_names.iter().position(|n| n == pname) {
                return EndpointSpec::Param { index: idx, suffix };
            }
        }
        EndpointSpec::Static(ep.to_string())
    }

    pub(super) fn extract_field_endpoint_specs(
        &self,
        val: &ExprVal,
    ) -> BTreeMap<String, EndpointSpec> {
        let mut out = BTreeMap::new();
        for (fname, ep) in &val.field_endpoints {
            if ep != "<anon_endpoint>" {
                out.insert(fname.clone(), self.resolve_endpoint_spec(ep));
            }
        }
        out
    }

    pub(super) fn record_byte_ring_op(
        &mut self,
        role: ByteRingOpRole,
        method: &str,
        source_key: &str,
        span: Span,
    ) {
        if source_key == "<anon_endpoint>" {
            return;
        }
        let caller = if self.caller_key == "main" {
            if !self.in_main_post_boot() {
                return;
            }
            "main::post_boot".to_string()
        } else {
            self.caller_key.clone()
        };
        let target = self.resolve_endpoint_spec(source_key);
        self.sema.recorded_byte_ring_ops.push(RecordedByteRingOp {
            caller,
            role,
            method: method.to_string(),
            target,
            span,
            file: self.sema.current_file.clone(),
        });
    }

    pub(super) fn record_watch_publish(&mut self, source_key: &str, span: Span) {
        if source_key == "<anon_endpoint>" {
            return;
        }
        let caller = if self.caller_key == "main" {
            if self.in_main_after_spawn() {
                "main::post_spawn".to_string()
            } else if self.in_main_after_irq_attach() {
                "main::post_irq_pre_spawn".to_string()
            } else {
                // Sequential boot initialization in `main()` before `spawn` / `irq.attach`
                // and before `main()`'s supervisor loop does not race with concurrent actors.
                return;
            }
        } else {
            self.caller_key.clone()
        };
        let target = self.resolve_endpoint_spec(source_key);
        self.sema.recorded_watch_writes.push(RecordedWatchWrite {
            caller,
            target,
            span,
            file: self.sema.current_file.clone(),
        });
    }

    pub(super) fn resolve_param_endpoint_index(
        &self,
        endpoint_name: &Option<String>,
    ) -> Option<(usize, Option<String>)> {
        let ep = endpoint_name.as_deref()?;
        let rest = ep.strip_prefix("param:")?;
        let (pname, suffix) = match rest.split_once('.') {
            Some((head, tail)) => (head, Some(tail.to_string())),
            None => (rest, None),
        };
        self.param_names
            .iter()
            .position(|n| n == pname)
            .map(|idx| (idx, suffix))
    }

    pub(super) fn call_context(&self) -> CallContext {
        CallContext {
            in_noyield: self.in_noyield(),
            in_critical: self.in_critical(),
            in_isr: self.in_isr(),
            in_defer: self.in_defer(),
            in_loop: self.loop_depth > 0,
            after_diverge: self.after_diverge,
            after_main_supervisor: self.main_supervisor_started,
        }
    }

    pub(super) fn display_caller_name(&self) -> &str {
        self.caller_key
            .strip_prefix("task::")
            .or_else(|| self.caller_key.strip_prefix("isr::"))
            .unwrap_or(&self.caller_key)
    }

    pub(super) fn bind_params(&mut self, params: &[ParamInfo]) {
        self.param_names = params.iter().map(|p| p.name.clone()).collect();
        for (idx, p) in params.iter().enumerate() {
            let is_ptr = matches!(p.ty, Ty::Ptr(_));
            let endpoint_alias =
                if ty_contains_os_prim_view(&p.ty, &self.sema.structs, &mut BTreeSet::new()) {
                    Some(format!("param:{}", p.name))
                } else {
                    None
                };
            let prov = if p.ty.contains_reference(&self.sema.structs) {
                Provenance::for_param(idx, is_ptr, endpoint_alias.clone())
            } else {
                // Non-reference scalars and function pointers both carry pure_static provenance.
                Provenance::pure_static()
            };
            let (lease_state, lease_gen_key) = if p.ty.is_linear_lease() {
                if self.caller_kind == CallableKind::Fn {
                    let key = self.fresh_lease_gen_key(&p.name);
                    (Some(LinearLeaseState::Live), Some(key))
                } else {
                    (
                        Some(LinearLeaseState::Moved {
                            moved_at: p.span,
                            reason: "invalid param",
                        }),
                        None,
                    )
                }
            } else {
                (None, None)
            };
            self.scopes[0].insert(
                p.name.clone(),
                LocalVar {
                    name: p.name.clone(),
                    ty: p.ty.clone(),
                    decl_depth: 1,
                    prov,
                    field_prov: BTreeMap::new(),
                    endpoint_alias,
                    field_endpoints: BTreeMap::new(),
                    is_isr_state: false,
                    lease_state,
                    lease_gen_key,
                },
            );
        }
    }

    pub(super) fn lookup_local(&self, name: &str) -> Option<&LocalVar> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v);
            }
        }
        None
    }

    pub(super) fn lookup_local_mut(&mut self, name: &str) -> Option<&mut LocalVar> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(v) = scope.get_mut(name) {
                return Some(v);
            }
        }
        None
    }

    pub(super) fn suggest_ident_hint(&self, needle: &str) -> Option<String> {
        match needle {
            "sleep_ms" | "usleep" | "msleep" => {
                return Some("did you mean `delay(10ms)`? (Zev uses typed duration literals like `1ms`, `500us`, `5s`)".to_string());
            }
            "yield_now" | "task_yield" | "co_yield" | "sched_yield" => {
                return Some("did you mean `checkpoint()`?".to_string());
            }
            "printf" | "log" => {
                return Some(
                    "did you mean `println(\"...\")`, `sys.log(\"...\")`, or `sys.log_kv(\"label=\", val)`?"
                        .to_string(),
                );
            }
            _ => {}
        }
        let mut pool: Vec<&str> = Vec::new();
        for scope in &self.scopes {
            pool.extend(scope.keys().map(String::as_str));
        }
        pool.extend(self.sema.primitives.keys().map(String::as_str));
        pool.extend(self.sema.consts.keys().map(String::as_str));
        pool.extend(
            self.sema
                .variant_to_enums
                .keys()
                .filter(|v| {
                    !self
                        .sema
                        .visible_enums_for_variant(self.prog_idx, v)
                        .is_empty()
                })
                .map(String::as_str),
        );
        pool.extend(self.sema.functions.keys().map(String::as_str));
        pool.extend(["set_priority", "checkpoint", "delay", "sys", "sim"]);
        closest_match(needle, pool, 2).map(|m| format!("did you mean `{}`?", m))
    }

    pub(super) fn suggest_struct_field_hint(&self, struct_name: &str, needle: &str) -> String {
        if let Some(st) = self.sema.structs.get(struct_name) {
            let names: Vec<&str> = st.fields.iter().map(|f| f.name.as_str()).collect();
            let avail = names.join(", ");
            if let Some(best) = closest_match(needle, names, 2) {
                return format!(
                    "did you mean `{}.{}`? (available fields on `{}`: {})",
                    struct_name, best, struct_name, avail
                );
            }
            return format!("available fields on `{}`: {}", struct_name, avail);
        }
        format!("check field definitions on struct `{}`", struct_name)
    }

    pub(super) fn suggest_enum_variant_hint(&self, enum_name: &str, needle: &str) -> String {
        if let Some(en) = self.sema.enums.get(enum_name) {
            let names: Vec<&str> = en.variants.iter().map(String::as_str).collect();
            let avail = names.join(", ");
            if let Some(best) = closest_match(needle, names, 2) {
                return format!(
                    "did you mean `{}`? (variants of `{}`: {})",
                    best, enum_name, avail
                );
            }
            return format!("valid variants of `{}`: {}", enum_name, avail);
        }
        format!("check variant definitions on enum `{}`", enum_name)
    }

    pub(super) fn record_direct_yield(&mut self, span: Span, tier: FnEffect, construct_desc: &str) {
        if self.in_isr() {
            let isr_name = self.display_caller_name().to_string();
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::YieldInIsr,
                span,
                format!(
                    "cannot yield via {} inside ISR '{}'",
                    construct_desc, isr_name
                ),
            ));
        } else if self.in_critical() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::YieldInCritical,
                span,
                format!(
                    "cannot yield via {} inside 'critical' block while hardware IRQs are masked",
                    construct_desc
                ),
            ));
        } else if self.in_defer() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::YieldInNoYield,
                span,
                format!(
                    "cannot yield via {} inside 'defer' cleanup statement",
                    construct_desc
                ),
            ));
        } else if self.in_noyield() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::YieldInNoYield,
                span,
                format!("cannot yield via {} inside 'noyield' block", construct_desc),
            ));
        } else {
            self.sema
                .call_graph
                .mark_direct_yield(&self.caller_key, tier, construct_desc);
        }
    }

    pub(super) fn fresh_lease_gen_key(&mut self, var_name: &str) -> String {
        self.next_lease_var_id += 1;
        format!("{}#{}", var_name, self.next_lease_var_id)
    }

    pub(super) fn display_lease_source(source: &str) -> &str {
        source.split('#').next().unwrap_or(source)
    }

    pub(super) fn finish_lexical_block(&mut self, diverged: bool, block_span: Span) {
        let closing_depth = self.current_depth();
        if diverged {
            return;
        }
        if let Some(closing_scope) = self.scopes.last() {
            let unconsumed: Vec<String> = closing_scope
                .values()
                .filter(|v| matches!(v.lease_state, Some(LinearLeaseState::Live)))
                .map(|v| v.name.clone())
                .collect();
            for vname in unconsumed {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::LinearLeaseViolation,
                    block_span,
                    format!(
                        "linear Lease<T> variable '{}' was not consumed before end of scope",
                        vname
                    ),
                ));
            }
        }
        // Any outer variable whose `defer` was registered at `closing_depth` is released right here!
        let mut to_invalidate = Vec::new();
        for scope in &mut self.scopes {
            for var in scope.values_mut() {
                if let Some(LinearLeaseState::DeferredRelease { span, scope_depth }) =
                    var.lease_state
                {
                    if scope_depth == closing_depth {
                        var.lease_state = Some(LinearLeaseState::Moved {
                            moved_at: span,
                            reason: "defer release",
                        });
                        if let Some(ref key) = var.lease_gen_key {
                            to_invalidate.push((key.clone(), span));
                        }
                    }
                }
            }
        }
        for (key, sp) in to_invalidate {
            let _ = self.leases.invalidate(&key, sp, "defer release");
        }
    }

    pub(super) fn check_unconsumed_leases_on_exit(
        &mut self,
        exit_span: Span,
        exit_kind: &str,
        min_depth_exclusive: u32,
    ) {
        let mut offenders = Vec::new();
        for scope in &self.scopes {
            for var in scope.values() {
                if var.decl_depth > min_depth_exclusive
                    && matches!(var.lease_state, Some(LinearLeaseState::Live))
                {
                    offenders.push(var.name.clone());
                }
            }
        }
        for name in offenders {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                exit_span,
                format!(
                    "cannot '{}' while linear Lease<T> variable '{}' is still live and unconsumed",
                    exit_kind, name
                ),
            ));
        }
    }

    pub(super) fn consume_lease_value(
        &mut self,
        expr: &Expr,
        val: &ExprVal,
        reason: &'static str,
        use_span: Span,
    ) {
        if !val.ty.is_linear_lease() {
            return;
        }
        let ExprKind::Ident(ident) = &expr.kind else {
            return;
        };
        let Some(var) = self.lookup_local(&ident.name).cloned() else {
            return;
        };
        if !var.ty.is_linear_lease() {
            return;
        }
        match var.lease_state {
            Some(LinearLeaseState::Live) => {
                // Reject moving/releasing an outer variable that was already Live/Deferred at loop entry.
                for (entry_depth, entry_vars) in &self.loop_lease_frames {
                    if var.decl_depth <= *entry_depth {
                        let was_outer_live = entry_vars
                            .iter()
                            .rev()
                            .find_map(|s| s.get(&var.name))
                            .is_some_and(|ev| {
                                matches!(
                                    ev.lease_state,
                                    Some(
                                        LinearLeaseState::Live
                                            | LinearLeaseState::DeferredRelease { .. }
                                    )
                                )
                            });
                        if was_outer_live {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::LinearLeaseViolation,
                                    use_span,
                                    format!(
                                        "cannot move/consume outer live Lease<T> variable '{}' inside a loop",
                                        var.name
                                    ),
                                )
                                .with_note("release the outer Lease<T> after the loop exits (or allocate/release inside each loop iteration)"),
                            );
                            return;
                        }
                    }
                }

                if let Some(defer_owner) = self.active_defer_owner_depth {
                    if var.decl_depth <= defer_owner {
                        if let Some(v) = self.lookup_local_mut(&ident.name) {
                            v.lease_state = Some(LinearLeaseState::DeferredRelease {
                                span: use_span,
                                scope_depth: defer_owner,
                            });
                        }
                        return;
                    }
                }

                if let Some(v) = self.lookup_local_mut(&ident.name) {
                    v.lease_state = Some(LinearLeaseState::Moved {
                        moved_at: use_span,
                        reason,
                    });
                }
                if let Some(key) = var.lease_gen_key {
                    let _ = self.leases.invalidate(&key, use_span, reason);
                }
            }
            Some(LinearLeaseState::DeferredRelease { span, .. }) => {
                self.sema.emit(
                    Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        use_span,
                        format!(
                            "cannot move/consume Lease<T> variable '{}' via {}: `defer` release was already scheduled at {}",
                            var.name, reason, span
                        ),
                    )
                    .with_note("remove either the manual consume or `defer lease.release();` so it is released once"),
                );
            }
            Some(LinearLeaseState::Moved { .. }) | Some(LinearLeaseState::Uninit) | None => {
                // `check_ident_expr` already reported use-after-move / uninit if accessed as rvalue.
            }
        }
    }

    /// Push a nested lexical block (`depth + 1`), execute `f`, and pop the nested block.
    pub(super) fn with_nested_scope<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.scopes.push(BTreeMap::new());
        let res = f(self);
        self.scopes.pop();
        res
    }

    pub(super) fn snapshot_outer_vars(&self) -> Vec<BTreeMap<String, LocalVar>> {
        self.scopes.clone()
    }

    pub(super) fn merge_branch_vars(
        &mut self,
        base: &[BTreeMap<String, LocalVar>],
        branches: &[Vec<BTreeMap<String, LocalVar>>],
        join_span: Span,
    ) {
        if branches.is_empty() {
            return;
        }
        for (depth_idx, scope) in self.scopes.iter_mut().enumerate() {
            if depth_idx >= base.len() {
                break;
            }
            for (name, var) in scope.iter_mut() {
                let mut merged_prov: Option<Provenance> = None;
                let mut merged_lease: Option<LinearLeaseState> = None;
                let mut lease_mismatch = false;
                for b in branches {
                    if let Some(b_var) = b.get(depth_idx).and_then(|s| s.get(name)) {
                        merged_prov = Some(match merged_prov {
                            None => b_var.prov.clone(),
                            Some(acc) => acc.merge(&b_var.prov),
                        });
                        if let Some(bs) = b_var.lease_state {
                            match merged_lease {
                                None => merged_lease = Some(bs),
                                Some(prev) => {
                                    let same = match (prev, bs) {
                                        (LinearLeaseState::Live, LinearLeaseState::Live)
                                        | (LinearLeaseState::Uninit, LinearLeaseState::Uninit)
                                        | (
                                            LinearLeaseState::Moved { .. },
                                            LinearLeaseState::Moved { .. },
                                        ) => true,
                                        (
                                            LinearLeaseState::DeferredRelease {
                                                scope_depth: d1,
                                                ..
                                            },
                                            LinearLeaseState::DeferredRelease {
                                                scope_depth: d2,
                                                ..
                                            },
                                        ) => d1 == d2,
                                        _ => false,
                                    };
                                    if !same {
                                        lease_mismatch = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(mp) = merged_prov {
                    var.prov = mp;
                }
                if lease_mismatch {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::LinearLeaseViolation,
                            join_span,
                            format!(
                                "conditional branches disagree on linear ownership state of Lease<T> variable '{}'",
                                name
                            ),
                        )
                        .with_note("every control-flow path reaching the branch merge must either leave the Lease<T> unconsumed or consume it along all branches"),
                    );
                    var.lease_state = Some(LinearLeaseState::Moved {
                        moved_at: join_span,
                        reason: "branch mismatch",
                    });
                } else if let Some(ml) = merged_lease {
                    var.lease_state = Some(ml);
                }
            }
        }
    }

    /// Analyze a slice of statements, returning `(diverged, warned_unreachable)`.
    pub(super) fn check_stmt_slice(&mut self, stmts: &[Stmt]) -> (bool, bool) {
        let prev_after_diverge = self.after_diverge;
        let mut diverged = false;
        let mut warned_unreachable = false;
        for stmt in stmts {
            if diverged {
                if !warned_unreachable {
                    warned_unreachable = true;
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::UnreachableCode,
                        stmt.span,
                        "unreachable statement following diverging control flow",
                    ));
                }
                self.after_diverge = true;
            }
            if self.check_stmt(stmt) {
                diverged = true;
            }
        }
        self.after_diverge = prev_after_diverge;
        (diverged, warned_unreachable)
    }

    /// Analyze a list of statements in the current scope. Returns `true` if control flow diverges.
    pub(super) fn check_block_stmts(&mut self, stmts: &[Stmt]) -> bool {
        self.check_stmt_slice(stmts).0
    }

    pub(super) fn check_nested_block(&mut self, block: &Block) -> bool {
        self.with_nested_scope(|this| {
            let div = this.check_block_stmts(&block.stmts);
            this.finish_lexical_block(div, block.span);
            div
        })
    }

    pub(super) fn check_stmt(&mut self, stmt: &Stmt) -> bool {
        match &stmt.kind {
            StmtKind::VarDecl { ty, name, init } => {
                let resolved_ty = self.sema.resolve_type(ty);
                if resolved_ty.is_void() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        ty.span,
                        format!("local variable '{}' cannot have type 'void'", name.name),
                    ));
                }
                self.sema.check_type_no_byval_mmio(
                    &resolved_ty,
                    ty.span,
                    &format!("local variable '{}'", name.name),
                );
                self.sema.check_disallowed_lease_type(
                    &resolved_ty,
                    ty.span,
                    &format!("local variable '{}'", name.name),
                    true,
                );
                if matches!(
                    &resolved_ty,
                    Ty::OsPrimitive(
                        ResolvedOsPrimitive::Pool { .. } | ResolvedOsPrimitive::DmaSlot { .. }
                    )
                ) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        ty.span,
                        format!(
                            "static OS primitive '{}' cannot be declared as local variable '{}' (declare at module scope or in a subsystem struct)",
                            resolved_ty, name.name
                        ),
                    ));
                }
                if self
                    .scopes
                    .last()
                    .is_some_and(|s| s.contains_key(&name.name))
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::DuplicateDefinition,
                        name.span,
                        format!("duplicate variable '{}' in the same scope", name.name),
                    ));
                }

                let decl_depth = self.current_depth();
                let mut prov = Provenance::pure_static();
                let mut field_prov = BTreeMap::new();
                let mut endpoint_alias = None;
                let mut field_endpoints = BTreeMap::new();
                let mut lease_state = if resolved_ty.is_linear_lease() {
                    Some(LinearLeaseState::Uninit)
                } else {
                    None
                };
                let mut lease_gen_key = None;

                if let Some(init_expr) = init {
                    let val = self.check_expr_infallible(init_expr);
                    if val.ty.is_linear_lease() {
                        self.consume_lease_value(
                            init_expr,
                            &val,
                            "variable initialization",
                            init_expr.span,
                        );
                    }
                    if matches!(init_expr.kind, ExprKind::StructLit { .. }) {
                        if let Some(fp) = self
                            .last_struct_lit_fields
                            .remove(&(init_expr.span.start, init_expr.span.end))
                        {
                            field_prov = fp;
                        }
                    }
                    field_endpoints = val.field_endpoints.clone();
                    if self.sema.check_assignable_or_emit(
                        &val.ty,
                        &resolved_ty,
                        init_expr.span,
                        DiagnosticKind::TypeMismatch,
                        || {
                            format!(
                                "variable '{}' declared as '{}', but initialized with '{}'",
                                name.name, resolved_ty, val.ty
                            )
                        },
                    ) {
                        self.check_coerced_expr(init_expr, &resolved_ty);
                    }
                    if resolved_ty.is_linear_lease() {
                        if val.ty.is_error() {
                            lease_state = Some(LinearLeaseState::Moved {
                                moved_at: init_expr.span,
                                reason: "error init",
                            });
                        } else {
                            lease_state = Some(LinearLeaseState::Live);
                            lease_gen_key = Some(self.fresh_lease_gen_key(&name.name));
                        }
                    }
                    if resolved_ty.contains_reference(&self.sema.structs)
                        || resolved_ty.contains_fn_ptr(&self.sema.structs)
                    {
                        let init_prov = self.coerce_array_to_slice_prov(
                            init_expr,
                            &val.ty,
                            &resolved_ty,
                            &val.prov,
                        );
                        if decl_depth < init_prov.origin_depth {
                            if !init_prov.isr_state_origins.is_empty() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::InvalidIsrState,
                                    init_expr.span,
                                    format!(
                                        "cannot escape `&` of ISR `state` variable into outer variable '{}'",
                                        name.name
                                    ),
                                ));
                            } else {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ScopeDepthEscape,
                                    init_expr.span,
                                    format!(
                                        "cannot store reference/slice of inner scope depth {} into variable '{}' at outer scope depth {}",
                                        init_prov.origin_depth, name.name, decl_depth
                                    ),
                                ));
                            }
                        }
                        prov = init_prov;
                    }
                    endpoint_alias = val.endpoint_name;
                }

                if let Some(scope) = self.scopes.last_mut() {
                    scope.insert(
                        name.name.clone(),
                        LocalVar {
                            name: name.name.clone(),
                            ty: resolved_ty,
                            decl_depth,
                            prov,
                            field_prov,
                            endpoint_alias,
                            field_endpoints,
                            is_isr_state: false,
                            lease_state,
                            lease_gen_key,
                        },
                    );
                }
                false
            }

            StmtKind::StateVar { ty, name, init } => {
                if !matches!(self.body_kind, ActiveBodyKind::Isr) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        stmt.span,
                        format!(
                            "`state` variable '{}' is only allowed inside an `isr` handler",
                            name.name
                        ),
                    ));
                }

                let dup_in_isr = if matches!(self.body_kind, ActiveBodyKind::Isr) {
                    !self.isr_state_names.insert(name.name.clone())
                } else {
                    self.scopes
                        .last()
                        .is_some_and(|s| s.contains_key(&name.name))
                };
                if dup_in_isr {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        name.span,
                        format!(
                            "duplicate `state` variable '{}' inside ISR '{}'",
                            name.name,
                            self.display_caller_name()
                        ),
                    ));
                } else if matches!(self.body_kind, ActiveBodyKind::Isr)
                    && self.param_names.iter().any(|p| p == &name.name)
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        name.span,
                        format!(
                            "`state` variable '{}' shadows parameter of ISR '{}'",
                            name.name,
                            self.display_caller_name()
                        ),
                    ));
                }

                let resolved_ty = self.sema.resolve_type(ty);
                if resolved_ty.is_void() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        ty.span,
                        format!("`state` variable '{}' cannot have type 'void'", name.name),
                    ));
                }
                self.sema.check_type_no_byval_mmio(
                    &resolved_ty,
                    ty.span,
                    &format!("`state` variable '{}'", name.name),
                );
                self.sema.check_disallowed_lease_type(
                    &resolved_ty,
                    ty.span,
                    &format!("ISR `state` variable '{}'", name.name),
                    false,
                );
                if resolved_ty.contains_reference(&self.sema.structs)
                    || resolved_ty.contains_fn_ptr(&self.sema.structs)
                    || matches!(resolved_ty, Ty::OsPrimitive(_) | Ty::MmioInstance { .. })
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidIsrState,
                        ty.span,
                        format!(
                            "`state` variable '{}' cannot have reference-bearing or handle type '{}'",
                            name.name, resolved_ty
                        ),
                    ));
                }

                match init {
                    None => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidIsrState,
                            stmt.span,
                            format!(
                                "`state` variable '{}' inside ISR requires a compile-time constant initializer",
                                name.name
                            ),
                        ));
                    }
                    Some(init_expr) => {
                        let val = self.check_expr_infallible(init_expr);
                        if !val.ty.is_assignable_to(&resolved_ty) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                init_expr.span,
                                format!(
                                    "`state` variable '{}' declared as '{}', but initialized with '{}'",
                                    name.name, resolved_ty, val.ty
                                ),
                            ));
                        } else {
                            self.check_coerced_expr(init_expr, &resolved_ty);
                        }
                        let shadowed = self.active_local_names();
                        if !self.sema.is_compile_time_const_expr_with_shadowed(
                            init_expr,
                            &shadowed,
                            self.prog_idx,
                        ) {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidIsrState,
                                init_expr.span,
                                format!(
                                    "`state` variable '{}' initializer must be a compile-time constant expression",
                                    name.name
                                ),
                            ));
                        }
                    }
                }

                let decl_depth = self.current_depth();
                if let Some(scope) = self.scopes.last_mut() {
                    scope.insert(
                        name.name.clone(),
                        LocalVar {
                            name: name.name.clone(),
                            ty: resolved_ty.clone(),
                            decl_depth,
                            prov: Provenance::pure_static(),
                            field_prov: BTreeMap::new(),
                            endpoint_alias: None,
                            field_endpoints: BTreeMap::new(),
                            is_isr_state: true,
                            lease_state: None,
                            lease_gen_key: None,
                        },
                    );
                }
                if matches!(self.body_kind, ActiveBodyKind::Isr) {
                    self.sema.isr_states.push(IsrStateInfo {
                        isr_name: self.display_caller_name().to_string(),
                        name: name.name.clone(),
                        ty: resolved_ty,
                        span: stmt.span,
                    });
                }
                false
            }
            StmtKind::Assign { target, op, value } => {
                let rhs = self.check_expr_infallible(value);
                self.check_assignment(target, *op, value, rhs, stmt.span);
                false
            }

            StmtKind::Return(opt_expr) => {
                if self.in_defer() {
                    if let Some(e) = opt_expr {
                        let _ = self.check_expr_infallible(e);
                    }
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'return' cannot be used inside a 'defer' cleanup statement",
                    ));
                    return true;
                }
                match &self.body_kind.clone() {
                    ActiveBodyKind::Isr => {
                        if let Some(e) = opt_expr {
                            let _ = self.check_expr_infallible(e);
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::ReturnMismatch,
                                stmt.span,
                                "ISR handlers cannot return a value",
                            ));
                        }
                        self.check_unconsumed_leases_on_exit(stmt.span, "return", 0);
                    }
                    ActiveBodyKind::Task => {
                        if let Some(e) = opt_expr {
                            let _ = self.check_expr_infallible(e);
                        }
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::NonDivergingTask,
                            stmt.span,
                            format!(
                                "cannot 'return' from task '{}'; tasks are permanent actors (keep running inside 'loop' / 'loop every', park on 'inbox.recv()' / 'signal.wait()', reset via 'restart;', or move finite workflows into yielding 'fn's)",
                                self.display_caller_name()
                            ),
                        ));
                    }
                    ActiveBodyKind::Fn { ret_ty, .. } => {
                        match (ret_ty.is_void(), opt_expr) {
                            (true, None) => {}
                            (true, Some(e)) => {
                                let val = self.check_expr_infallible(e);
                                if val.ty.is_linear_lease() {
                                    self.consume_lease_value(e, &val, "return", e.span);
                                }
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ReturnMismatch,
                                    stmt.span,
                                    format!(
                                        "void function '{}' cannot return value of type '{}'",
                                        self.caller_key, val.ty
                                    ),
                                ));
                            }
                            (false, None) => {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ReturnMismatch,
                                    stmt.span,
                                    format!(
                                        "function '{}' must return a value of type '{}'",
                                        self.caller_key, ret_ty
                                    ),
                                ));
                            }
                            (false, Some(e)) => {
                                let val = self.check_expr_infallible(e);
                                if val.ty.is_linear_lease() {
                                    self.consume_lease_value(e, &val, "return", e.span);
                                }
                                if self.sema.check_assignable_or_emit(
                                    &val.ty,
                                    ret_ty,
                                    e.span,
                                    DiagnosticKind::ReturnMismatch,
                                    || {
                                        format!(
                                            "return value has type '{}', expected '{}'",
                                            val.ty, ret_ty
                                        )
                                    },
                                ) {
                                    self.check_coerced_expr(e, ret_ty);
                                }
                                if let Some(ep) = &val.endpoint_name {
                                    if ep != "<anon_endpoint>"
                                        && !ep.starts_with("param:")
                                        && self.return_endpoint.is_none()
                                    {
                                        self.return_endpoint = Some(ep.clone());
                                    }
                                }
                                for (k, v) in &val.field_endpoints {
                                    if v != "<anon_endpoint>" && !v.starts_with("param:") {
                                        self.return_field_endpoints
                                            .entry(k.clone())
                                            .or_insert_with(|| v.clone());
                                    }
                                }
                                if ret_ty.contains_reference(&self.sema.structs) {
                                    let ret_prov = self
                                        .coerce_array_to_slice_prov(e, &val.ty, ret_ty, &val.prov);
                                    if ret_prov.has_non_returning_origin() {
                                        self.sema.emit(
                                            Diagnostic::new(
                                                DiagnosticKind::ReturnLocalReference,
                                                e.span,
                                                format!(
                                                    "cannot return pointer/slice/wrapper of type '{}' referencing local variable or temporary DMA lease from '{}'",
                                                    ret_ty, self.caller_key
                                                ),
                                            )
                                            .with_note(format!(
                                                "referenced origins: {:?}",
                                                ret_prov.local_origins
                                            )),
                                        );
                                    } else {
                                        for &idx in &ret_prov.param_origins {
                                            self.return_param_origins.insert(idx);
                                        }
                                    }
                                }
                            }
                        }
                        self.check_unconsumed_leases_on_exit(stmt.span, "return", 0);
                    }
                }
                true
            }

            StmtKind::Error { reason, partial } => {
                let r_val = reason.as_ref().map(|r| self.check_expr_infallible(r));
                let p_val = partial.as_ref().map(|p| self.check_expr_infallible(p));

                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'error' exit cannot be used inside a 'defer' cleanup statement",
                    ));
                    return true;
                }

                match &self.body_kind.clone() {
                    ActiveBodyKind::Task | ActiveBodyKind::Isr => {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::ErrorSpecMismatch,
                            stmt.span,
                            format!(
                                "'error' exit statement is only valid inside a function declared with 'error', not inside '{}'",
                                self.display_caller_name()
                            ),
                        ));
                    }
                    ActiveBodyKind::Fn { ret_ty, error_spec } => match (error_spec, &r_val, &p_val)
                    {
                        (ResolvedErrorSpec::None, _, _) => {
                            let sig_example = if ret_ty.is_void() {
                                format!("fn {}(...) error ErrEnum", self.caller_key)
                            } else {
                                format!("{} {}(...) error ErrEnum", ret_ty, self.caller_key)
                            };
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    stmt.span,
                                    format!(
                                        "function '{}' is not declared with 'error' exit specification",
                                        self.caller_key
                                    ),
                                )
                                .with_note(format!(
                                    "add `error` (for `error;`) or `error ErrEnum` (for `error Reason;` / `error Reason, partial;`) to signature: `{}`",
                                    sig_example
                                )),
                            );
                        }
                        // 0-arg `error;`
                        (ResolvedErrorSpec::Untyped, None, None) => {}
                        (ResolvedErrorSpec::Untyped, Some(_), _) => {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    stmt.span,
                                    format!(
                                        "function '{}' declares bare 'error' without an error payload type; use 'error;'",
                                        self.caller_key
                                    ),
                                )
                                .with_note(format!(
                                    "either exit with `error;` (0 args), or declare typed `{}` to exit with `error Reason;`",
                                    if ret_ty.is_void() {
                                        format!("fn {}(...) error ErrEnum", self.caller_key)
                                    } else {
                                        format!("{} {}(...) error ErrEnum", ret_ty, self.caller_key)
                                    }
                                )),
                            );
                        }
                        (ResolvedErrorSpec::Typed(err_ty), None, _) => {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    stmt.span,
                                    format!(
                                        "function '{}' declares typed 'error {}' and requires an error reason ('error <{}>;')",
                                        self.caller_key, err_ty, err_ty
                                    ),
                                )
                                .with_note(format!(
                                    "use `error <{} variant>;` (or `error <{} variant>, partial_val;` to report partial progress)",
                                    err_ty, err_ty
                                )),
                            );
                        }
                        // 1-arg `error Reason;`
                        (ResolvedErrorSpec::Typed(err_ty), Some(rv), None) => {
                            if !rv.ty.is_assignable_to(err_ty) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    rv.span,
                                    format!(
                                        "error reason has type '{}', expected '{}'",
                                        rv.ty, err_ty
                                    ),
                                ));
                            }
                        }
                        // 2-arg `error Reason, partial_val;`
                        (ResolvedErrorSpec::Typed(err_ty), Some(rv), Some(pv)) => {
                            if pv.ty.is_linear_lease() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::LinearLeaseViolation,
                                    pv.span,
                                    "cannot return linear 'Lease<T>' as partial-progress payload on 'error' exit",
                                ));
                            }
                            if !rv.ty.is_assignable_to(err_ty) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    rv.span,
                                    format!(
                                        "error reason has type '{}', expected '{}'",
                                        rv.ty, err_ty
                                    ),
                                ));
                            }
                            if ret_ty.is_void() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    pv.span,
                                    format!(
                                        "void function '{}' cannot return a partial-progress value on 'error'",
                                        self.caller_key
                                    ),
                                ));
                            } else if !pv.ty.is_assignable_to(ret_ty) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ErrorSpecMismatch,
                                    pv.span,
                                    format!(
                                        "error partial-progress value has type '{}', expected '{}'",
                                        pv.ty, ret_ty
                                    ),
                                ));
                            } else {
                                if let Some(p_expr) = partial {
                                    self.check_coerced_expr(p_expr, ret_ty);
                                    let partial_prov = self.coerce_array_to_slice_prov(
                                        p_expr, &pv.ty, ret_ty, &pv.prov,
                                    );
                                    if ret_ty.contains_reference(&self.sema.structs)
                                        && partial_prov.has_non_returning_origin()
                                    {
                                        self.sema.emit(Diagnostic::new(
                                            DiagnosticKind::ReturnLocalReference,
                                            pv.span,
                                            format!(
                                                "cannot return partial-progress value of type '{}' referencing local variable or DMA lease",
                                                ret_ty
                                            ),
                                        ));
                                    }
                                }
                            }
                        }
                        (ResolvedErrorSpec::Untyped, None, Some(_)) => {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::ErrorSpecMismatch,
                                stmt.span,
                                "partial-progress error requires a typed 'error ErrType' signature",
                            ));
                        }
                    },
                }
                self.check_unconsumed_leases_on_exit(stmt.span, "error", 0);
                true
            }

            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                let cv = self.check_expr_infallible(cond);
                if !cv.ty.is_bool() && !cv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        cond.span,
                        format!("'if' condition must be 'bool', found '{}'", cv.ty),
                    ));
                }

                let base_vars = self.snapshot_outer_vars();
                let base_leases = self.leases.clone();

                self.branch_depth += 1;
                self.boot_block_stack.push(Vec::new());
                let then_div = self.check_nested_block(then_block);
                let then_irq_nodes = self.boot_block_stack.pop().unwrap_or_default();
                self.branch_depth -= 1;
                let then_vars = self.snapshot_outer_vars();
                let then_leases = self.leases.clone();

                self.scopes = base_vars.clone();
                self.leases = base_leases.clone();

                if let Some(eb) = else_block {
                    self.branch_depth += 1;
                    self.boot_block_stack.push(Vec::new());
                    let else_div = self.check_nested_block(eb);
                    let else_irq_nodes = self.boot_block_stack.pop().unwrap_or_default();
                    self.branch_depth -= 1;
                    let else_vars = self.snapshot_outer_vars();
                    let else_leases = self.leases.clone();

                    if self.can_record_boot_step()
                        && (!then_irq_nodes.is_empty() || !else_irq_nodes.is_empty())
                    {
                        if let Some(top) = self.boot_block_stack.last_mut() {
                            top.push(IrqBootNode::BranchGroup {
                                arms: vec![(then_div, then_irq_nodes), (else_div, else_irq_nodes)],
                            });
                        }
                    }

                    self.scopes = base_vars.clone();
                    let mut live_var_branches = Vec::new();
                    let mut live_lease_branches = Vec::new();
                    if !then_div {
                        live_var_branches.push(then_vars);
                        live_lease_branches.push(then_leases);
                    }
                    if !else_div {
                        live_var_branches.push(else_vars);
                        live_lease_branches.push(else_leases);
                    }
                    self.merge_branch_vars(&base_vars, &live_var_branches, stmt.span);
                    self.leases = base_leases;
                    self.leases.join_branches(&live_lease_branches);
                    then_div && else_div
                } else {
                    if self.can_record_boot_step() && !then_irq_nodes.is_empty() {
                        if let Some(top) = self.boot_block_stack.last_mut() {
                            top.push(IrqBootNode::BranchGroup {
                                arms: vec![(then_div, then_irq_nodes), (false, Vec::new())],
                            });
                        }
                    }
                    let mut live_var_branches = vec![base_vars.clone()];
                    let mut live_lease_branches = vec![base_leases.clone()];
                    if !then_div {
                        live_var_branches.push(then_vars);
                        live_lease_branches.push(then_leases);
                    }
                    self.merge_branch_vars(&base_vars, &live_var_branches, stmt.span);
                    self.leases = base_leases;
                    self.leases.join_branches(&live_lease_branches);
                    false
                }
            }

            StmtKind::While { cond, body } => {
                let is_while_true = matches!(cond.kind, ExprKind::Literal(Literal::Bool(true)));
                let manual_yield =
                    has_attr_manual_yield(&stmt.attrs) || has_attr_noyield(&stmt.attrs);
                let has_break = block_has_break(body);

                let cv = self.check_expr_infallible(cond);
                if !cv.ty.is_bool() && !cv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        cond.span,
                        format!("'while' condition must be 'bool', found '{}'", cv.ty),
                    ));
                }

                if is_while_true {
                    if self.in_critical() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::UnboundedLoopInCritical,
                            stmt.span,
                            "unbounded loop ('while (true)') is forbidden inside 'critical' block while hardware IRQs are masked",
                        ));
                    } else if self.in_isr() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::YieldInIsr,
                            stmt.span,
                            format!(
                                "unbounded 'while (true)' loop is forbidden inside ISR '{}'",
                                self.display_caller_name()
                            ),
                        ));
                    } else if self.in_defer() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::YieldInNoYield,
                            stmt.span,
                            "unbounded 'while (true)' loop is forbidden inside 'defer' cleanup statement",
                        ));
                    } else if self.in_noyield() {
                        if !manual_yield {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::YieldInNoYield,
                                stmt.span,
                                "auto-yield 'while (true)' loop is forbidden inside 'noyield' block",
                            ));
                        } else if !has_break {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::YieldInNoYield,
                                stmt.span,
                                "infinite 'while (true)' without 'break' inside 'noyield' block would permanently starve the cooperative scheduler",
                            ));
                        }
                    } else if !has_break {
                        self.sema.call_graph.mark_direct_yield(
                            &self.caller_key,
                            FnEffect::HardBlocking,
                            "while (true) (infinite loop)",
                        );
                    } else if !manual_yield {
                        self.sema.call_graph.mark_direct_yield(
                            &self.caller_key,
                            FnEffect::SoftYielding,
                            "while (true) (should_share)",
                        );
                    }
                } else if !manual_yield
                    && !self.in_noyield()
                    && !self.in_critical()
                    && !self.in_isr()
                    && !self.in_defer()
                    && self.caller_key != "main"
                {
                    self.sema.call_graph.mark_direct_yield(
                        &self.caller_key,
                        FnEffect::SoftYielding,
                        "while (cond) (should_share)",
                    );
                }

                if is_while_true && !has_break {
                    self.check_unconsumed_leases_on_exit(stmt.span, "enter infinite while loop", 0);
                }
                let prev_sup_loop = self.in_main_supervisor_loop;
                if is_while_true && !has_break && self.caller_key == "main" {
                    self.in_main_supervisor_loop = true;
                }
                self.check_loop_body(body, is_while_true && !has_break);
                self.in_main_supervisor_loop = prev_sup_loop;
                if is_while_true && self.caller_key == "main" {
                    self.main_supervisor_started = true;
                }
                is_while_true && !has_break
            }

            StmtKind::Loop { body } => {
                let manual_yield =
                    has_attr_manual_yield(&stmt.attrs) || has_attr_noyield(&stmt.attrs);
                let has_break = block_has_break(body);

                if self.in_critical() {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::UnboundedLoopInCritical,
                            stmt.span,
                            "unbounded 'loop' is forbidden inside 'critical' block (IRQ-masked blocks must stay bounded)",
                        )
                        .with_note("move the loop outside 'critical' or use a bounded 'for..in' / 'while' loop"),
                    );
                } else if self.in_isr() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::YieldInIsr,
                        stmt.span,
                        format!(
                            "unbounded / auto-yield 'loop' is forbidden inside ISR '{}'",
                            self.display_caller_name()
                        ),
                    ));
                } else if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::YieldInNoYield,
                        stmt.span,
                        "'loop' is forbidden inside 'defer' cleanup statement",
                    ));
                } else if self.in_noyield() {
                    if !manual_yield {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::YieldInNoYield,
                                stmt.span,
                                "auto-yield 'loop' is forbidden inside 'noyield' block",
                            )
                            .with_note(
                                "use a bounded 'for..in' / 'while' loop or '@manual_yield' with an explicit 'break'",
                            ),
                        );
                    } else if !has_break {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::YieldInNoYield,
                            stmt.span,
                            "infinite 'loop' without 'break' inside 'noyield' block would permanently starve the cooperative scheduler",
                        ));
                    }
                } else if !has_break {
                    self.sema.call_graph.mark_direct_yield(
                        &self.caller_key,
                        FnEffect::HardBlocking,
                        "loop (infinite)",
                    );
                } else if !manual_yield {
                    self.sema.call_graph.mark_direct_yield(
                        &self.caller_key,
                        FnEffect::SoftYielding,
                        "loop (should_share)",
                    );
                }

                if !has_break {
                    self.check_unconsumed_leases_on_exit(stmt.span, "enter infinite loop", 0);
                }
                let prev_sup_loop = self.in_main_supervisor_loop;
                if !has_break && self.caller_key == "main" {
                    self.in_main_supervisor_loop = true;
                }
                self.check_loop_body(body, !has_break);
                self.in_main_supervisor_loop = prev_sup_loop;
                if self.caller_key == "main" {
                    self.main_supervisor_started = true;
                }
                !has_break
            }

            StmtKind::LoopEvery { period, body } => {
                let pv = self.check_expr_infallible(period);
                if pv.ty.is_integer() {
                    let dest_prim = match pv.ty {
                        Ty::Prim(PrimType::U8 | PrimType::U16 | PrimType::U32) => {
                            Some(PrimType::U32)
                        }
                        Ty::Prim(PrimType::I8 | PrimType::I16 | PrimType::I32) => {
                            Some(PrimType::I32)
                        }
                        _ => None,
                    };
                    self.validate_int_expr_at_type(period, dest_prim);
                    let shadowed = self.active_local_names();
                    let const_ticks = self.sema.eval_const_expr_i128_shadowed(
                        period,
                        &mut BTreeSet::new(),
                        &shadowed,
                    );
                    if const_ticks.is_some_and(|t| t <= 0) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidDurationResolution,
                            period.span,
                            "periodic 'loop every' interval must be greater than 0",
                        ));
                    } else if const_ticks.is_some_and(|t| t > zev_ast::MAX_DURATION_TICKS as i128) {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::InvalidDurationResolution,
                            period.span,
                            format!(
                                "periodic 'loop every' interval exceeds the scheduler deadline horizon of {} ticks",
                                zev_ast::MAX_DURATION_TICKS
                            ),
                        ));
                    }
                } else if pv.ty != Ty::Duration && !pv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        period.span,
                        format!(
                            "'loop every' period must be a duration (e.g. 1ms) or integer expression, found '{}'",
                            pv.ty
                        ),
                    ));
                }

                self.record_direct_yield(
                    stmt.span,
                    FnEffect::HardBlocking,
                    "'loop every' periodic deadline",
                );
                let has_break = block_has_break(body);
                if !has_break {
                    self.check_unconsumed_leases_on_exit(stmt.span, "enter infinite loop every", 0);
                }
                let prev_sup_loop = self.in_main_supervisor_loop;
                if !has_break && self.caller_key == "main" {
                    self.in_main_supervisor_loop = true;
                }
                self.check_loop_body(body, !has_break);
                self.in_main_supervisor_loop = prev_sup_loop;
                if self.caller_key == "main" {
                    self.main_supervisor_started = true;
                }
                !has_break
            }

            StmtKind::ForRange {
                var_ty,
                var_name,
                start,
                end,
                inclusive: _,
                body,
            } => {
                let resolved_var_ty = self.sema.resolve_type(var_ty);
                if !resolved_var_ty.is_integer() && !resolved_var_ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        var_ty.span,
                        format!(
                            "'for' range loop variable '{}' must have an integer type, found '{}'",
                            var_name.name, resolved_var_ty
                        ),
                    ));
                }
                let sv = self.check_expr_infallible(start);
                if !sv.ty.is_assignable_to(&resolved_var_ty)
                    && !resolved_var_ty.is_error()
                    && !sv.ty.is_error()
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        start.span,
                        format!(
                            "'for' range start has type '{}', expected '{}'",
                            sv.ty, resolved_var_ty
                        ),
                    ));
                } else {
                    self.check_coerced_expr(start, &resolved_var_ty);
                }
                let ev = self.check_expr_infallible(end);
                if !ev.ty.is_assignable_to(&resolved_var_ty)
                    && !resolved_var_ty.is_error()
                    && !ev.ty.is_error()
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        end.span,
                        format!(
                            "'for' range end has type '{}', expected '{}'",
                            ev.ty, resolved_var_ty
                        ),
                    ));
                } else {
                    self.check_coerced_expr(end, &resolved_var_ty);
                }

                self.check_for_loop_with_binding(
                    &var_name.name,
                    resolved_var_ty,
                    Provenance::pure_static(),
                    iter_span_none(),
                    body,
                );
                false
            }

            StmtKind::ForEach {
                elem_ty,
                elem_name,
                iter,
                body,
            } => {
                let resolved_elem_ty = self.sema.resolve_type(elem_ty);
                let iv = self.check_expr_infallible(iter);
                let deref_iter_ty = match &iv.ty {
                    Ty::Ptr(inner) => (**inner).clone(),
                    other => other.clone(),
                };
                match &deref_iter_ty {
                    Ty::Slice(seq_elem) | Ty::Array { elem: seq_elem, .. } => {
                        if !seq_elem.is_assignable_to(&resolved_elem_ty)
                            && !resolved_elem_ty.is_error()
                            && !seq_elem.is_error()
                        {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                elem_ty.span,
                                format!(
                                    "'for..in' element '{}' declared as '{}', but iterable yields '{}'",
                                    elem_name.name, resolved_elem_ty, seq_elem
                                ),
                            ));
                        }
                    }
                    _ => {
                        if !iv.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                iter.span,
                                format!(
                                    "cannot iterate over type '{}' in 'for..in' loop (expected slice '[]T' or array '[N]T')",
                                    iv.ty
                                ),
                            ));
                        }
                    }
                }

                self.check_for_loop_with_binding(
                    &elem_name.name,
                    resolved_elem_ty,
                    iv.prov,
                    Some(iter.span),
                    body,
                );
                false
            }

            StmtKind::Defer(inner) => {
                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "nested 'defer' inside a 'defer' statement is not allowed",
                    ));
                }
                if matches!(inner.kind, StmtKind::VarDecl { .. }) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        inner.span,
                        "cannot declare a standalone variable directly inside 'defer'; wrap cleanup statements in 'defer { ... }' if temporaries are needed",
                    ));
                }
                self.defer_depth += 1;
                self.noyield_depth += 1;
                let prev_defer_owner = self.active_defer_owner_depth;
                self.active_defer_owner_depth = Some(self.current_depth());
                let _ = self.with_nested_scope(|this| {
                    let div = this.check_stmt(inner);
                    this.finish_lexical_block(div, inner.span);
                    div
                });
                self.active_defer_owner_depth = prev_defer_owner;
                self.noyield_depth -= 1;
                self.defer_depth -= 1;
                false
            }

            StmtKind::Restart => {
                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'restart;' cannot be used inside a 'defer' cleanup statement",
                    ));
                } else if !matches!(self.body_kind, ActiveBodyKind::Task) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        format!(
                            "'restart;' is only valid inside a 'task' body, not inside '{}'",
                            self.display_caller_name()
                        ),
                    ));
                } else {
                    self.record_direct_yield(stmt.span, FnEffect::SoftYielding, "'restart;'");
                }
                self.check_unconsumed_leases_on_exit(stmt.span, "restart", 0);
                true
            }

            StmtKind::NoYield { body } => {
                self.noyield_depth += 1;
                let div = self.check_nested_block(body);
                self.noyield_depth -= 1;
                div
            }

            StmtKind::Critical { body } => {
                self.critical_depth += 1;
                self.noyield_depth += 1;
                let div = self.check_nested_block(body);
                self.noyield_depth -= 1;
                self.critical_depth -= 1;
                div
            }

            StmtKind::Switch { scrutinee, arms } => {
                let mut sv = self.check_expr_infallible(scrutinee);
                self.validate_int_expr_at_type(scrutinee, None);
                if !sv.ty.is_error()
                    && !sv.ty.is_integer()
                    && !sv.ty.is_bool()
                    && !matches!(sv.ty, Ty::Enum(_))
                {
                    let mut diag = Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        scrutinee.span,
                        format!(
                            "'switch' scrutinee must be an integer, 'bool', or 'enum', found '{}'",
                            sv.ty
                        ),
                    );
                    if matches!(sv.ty, Ty::Str)
                        || matches!(&sv.ty, Ty::Slice(elem) if **elem == Ty::u8())
                    {
                        diag = diag.with_note("compare strings using `if (s == \"...\")` instead");
                    }
                    self.sema.emit(diag);
                    sv.ty = Ty::Error;
                }
                let base_vars = self.snapshot_outer_vars();
                let base_leases = self.leases.clone();
                let mut live_var_branches = Vec::new();
                let mut live_lease_branches = Vec::new();
                let mut has_wildcard = false;
                let mut covered_variants = BTreeSet::new();
                let mut all_arms_diverge = !arms.is_empty();
                let mut arm_irq_nodes: Vec<(bool, Vec<IrqBootNode>)> =
                    Vec::with_capacity(arms.len());

                for arm in arms {
                    match &arm.pat {
                        SwitchPat::Wildcard(_) => {
                            has_wildcard = true;
                        }
                        SwitchPat::Literal(lit, pat_span) => {
                            let lit_ty = match lit {
                                Literal::Int {
                                    suffix: Some(p), ..
                                } => Ty::Prim(*p),
                                Literal::Int { suffix: None, .. } => Ty::IntLit,
                                Literal::Float {
                                    suffix: Some(p), ..
                                } => Ty::Prim(*p),
                                Literal::Float { suffix: None, .. } => Ty::FloatLit,
                                Literal::Bool(_) => Ty::bool(),
                                Literal::Duration(d) => {
                                    self.sema.validate_duration(d);
                                    Ty::Duration
                                }
                                Literal::String(_) => Ty::Str,
                            };
                            if !lit_ty.is_assignable_to(&sv.ty) {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    *pat_span,
                                    format!(
                                        "switch pattern has type '{}', expected '{}'",
                                        lit_ty, sv.ty
                                    ),
                                ));
                            } else if let (
                                Literal::Int {
                                    value,
                                    suffix: None,
                                },
                                Ty::Prim(p),
                            ) = (lit, &sv.ty)
                            {
                                self.check_int_lit_range_or_emit(i128::from(*value), *p, *pat_span);
                            } else if let Literal::Int {
                                value,
                                suffix: Some(p),
                            } = lit
                            {
                                self.check_int_lit_range_or_emit(i128::from(*value), *p, *pat_span);
                            }
                        }
                        SwitchPat::Ident {
                            prefix,
                            name,
                            span: pat_span,
                        } => {
                            if let Ty::Enum(enum_name) = &sv.ty {
                                if let Some(pref) = prefix {
                                    if pref.name != *enum_name {
                                        self.sema.emit(Diagnostic::new(
                                            DiagnosticKind::TypeMismatch,
                                            pref.span,
                                            format!(
                                                "switch pattern enum '{}' does not match scrutinee enum '{}'",
                                                pref.name, enum_name
                                            ),
                                        ));
                                    }
                                }
                                if self
                                    .sema
                                    .enums
                                    .get(enum_name)
                                    .is_some_and(|e| e.has_variant(&name.name))
                                {
                                    covered_variants.insert(name.name.clone());
                                } else {
                                    let hint =
                                        self.suggest_enum_variant_hint(enum_name, &name.name);
                                    self.sema.emit(
                                        Diagnostic::new(
                                            DiagnosticKind::UndefinedSymbol,
                                            *pat_span,
                                            format!(
                                                "unknown variant '{}' for enum '{}'",
                                                name.name, enum_name
                                            ),
                                        )
                                        .with_note(hint),
                                    );
                                }
                            } else if let Some(pref) = prefix {
                                if !sv.ty.is_error() {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::UndefinedSymbol,
                                        *pat_span,
                                        format!(
                                            "unknown switch pattern '{}.{}'",
                                            pref.name, name.name
                                        ),
                                    ));
                                }
                            } else if let Some(c) = self.sema.consts.get(&name.name) {
                                if !c.ty.is_assignable_to(&sv.ty) {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::TypeMismatch,
                                        *pat_span,
                                        format!(
                                            "switch constant '{}' has type '{}', expected '{}'",
                                            name.name, c.ty, sv.ty
                                        ),
                                    ));
                                }
                            } else if !sv.ty.is_error() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::UndefinedSymbol,
                                    *pat_span,
                                    format!("unknown switch pattern '{}'", name.name),
                                ));
                            }
                        }
                        SwitchPat::Qualified {
                            path,
                            span: pat_span,
                        } => {
                            if !sv.ty.is_error() {
                                let rendered = path
                                    .iter()
                                    .map(|s| s.name.as_str())
                                    .collect::<Vec<_>>()
                                    .join(".");
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::UndefinedSymbol,
                                    *pat_span,
                                    format!("unknown qualified switch pattern '{rendered}'"),
                                ));
                            }
                        }
                    }

                    self.scopes = base_vars.clone();
                    self.leases = base_leases.clone();
                    self.branch_depth += 1;
                    self.boot_block_stack.push(Vec::new());
                    let arm_div = self.check_nested_block(&arm.body);
                    let this_arm_irq_nodes = self.boot_block_stack.pop().unwrap_or_default();
                    self.branch_depth -= 1;
                    arm_irq_nodes.push((arm_div, this_arm_irq_nodes));
                    if !arm_div {
                        all_arms_diverge = false;
                        live_var_branches.push(self.snapshot_outer_vars());
                        live_lease_branches.push(self.leases.clone());
                    }
                }

                let exhaustive = has_wildcard
                    || matches!(&sv.ty, Ty::Enum(en) if self.sema.enums.get(en).is_some_and(|e| e.variants.iter().all(|v| covered_variants.contains(v))));

                if !exhaustive {
                    all_arms_diverge = false;
                    live_var_branches.push(base_vars.clone());
                    live_lease_branches.push(base_leases.clone());
                    arm_irq_nodes.push((false, Vec::new()));
                }

                if self.can_record_boot_step() && arm_irq_nodes.iter().any(|(_, n)| !n.is_empty()) {
                    if let Some(top) = self.boot_block_stack.last_mut() {
                        top.push(IrqBootNode::BranchGroup {
                            arms: arm_irq_nodes,
                        });
                    }
                }

                self.scopes = base_vars.clone();
                self.merge_branch_vars(&base_vars, &live_var_branches, stmt.span);
                self.leases = base_leases;
                self.leases.join_branches(&live_lease_branches);
                all_arms_diverge
            }

            StmtKind::Wait { arms, timer } => {
                self.record_direct_yield(stmt.span, FnEffect::HardBlocking, "'wait { ... }'");
                if arms.is_empty() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        stmt.span,
                        "'wait { ... }' requires at least one event arm",
                    ));
                }

                let base_vars = self.snapshot_outer_vars();
                let base_leases = self.leases.clone();
                let mut live_var_branches = Vec::new();
                let mut live_lease_branches = Vec::new();
                let mut all_arms_diverge = !arms.is_empty();

                for arm in arms {
                    self.scopes = base_vars.clone();
                    self.leases = base_leases.clone();
                    self.branch_depth += 1;
                    self.scopes.push(BTreeMap::new());

                    let mut valid_consumer_call = false;
                    let mut is_boost_arm = false;
                    match &arm.call.kind {
                        ExprKind::MethodCall {
                            receiver,
                            method,
                            args,
                        } => {
                            let has_timeout_arg = args
                                .iter()
                                .any(|a| a.name.as_ref().is_some_and(|n| n.name == "timeout"));
                            if !args.is_empty() {
                                let mut d = Diagnostic::new(
                                    DiagnosticKind::InvalidOperation,
                                    arm.call.span,
                                    "'wait { ... }' event arms must be direct zero-argument blocking primitive consumer calls",
                                );
                                if has_timeout_arg {
                                    d = d.with_note("move `timeout:` out of the arm into a single `timeout <dur> => { ... }` or `deadline <u32> => { ... }` arm on the `wait` block");
                                }
                                self.sema.emit(d);
                            }
                            let rv = self.check_expr_infallible(receiver);
                            let prim_opt = match &rv.ty {
                                Ty::OsPrimitive(p) => Some(p),
                                Ty::Ptr(inner) => match inner.as_ref() {
                                    Ty::OsPrimitive(p) => Some(p),
                                    _ => None,
                                },
                                _ => None,
                            };
                            if let Some(prim) = prim_opt {
                                let m = method.name.as_str();
                                valid_consumer_call = matches!(
                                    (prim, m),
                                    (
                                        ResolvedOsPrimitive::Inbox { .. }
                                            | ResolvedOsPrimitive::InboxUnsized { .. },
                                        "recv"
                                    ) | (ResolvedOsPrimitive::Signal, "wait")
                                        | (
                                            ResolvedOsPrimitive::ByteRing { .. }
                                                | ResolvedOsPrimitive::ByteRingUnsized,
                                            "wait_read" | "wait_write"
                                        )
                                        | (ResolvedOsPrimitive::PingPong { .. }, "wait_chunk")
                                        | (ResolvedOsPrimitive::Watch { .. }, "wait_change")
                                        | (ResolvedOsPrimitive::Pool { .. }, "alloc")
                                );
                                is_boost_arm = matches!(
                                    (prim, m),
                                    (
                                        ResolvedOsPrimitive::Inbox { .. }
                                            | ResolvedOsPrimitive::InboxUnsized { .. },
                                        "recv"
                                    ) | (ResolvedOsPrimitive::Signal, "wait")
                                        | (
                                            ResolvedOsPrimitive::ByteRing { .. }
                                                | ResolvedOsPrimitive::ByteRingUnsized,
                                            "wait_read"
                                        )
                                );
                            }
                            if !valid_consumer_call && !rv.ty.is_error() {
                                let mut d = Diagnostic::new(
                                    DiagnosticKind::InvalidOperation,
                                    arm.call.span,
                                    "unsupported 'wait { ... }' arm: arms must be direct blocking consumer calls (`inbox.recv()`, `signal.wait()`, `byte_ring.wait_read()`, `ping_pong.wait_chunk()`, `watch.wait_change()`, `pool.alloc()`)",
                                );
                                if matches!(prim_opt, Some(ResolvedOsPrimitive::Guarded { .. })) {
                                    d = d.with_note("`guarded<T>` is not a wait source; pair it with a `wait_queue` (`prepare_wait()` inside `critical`, then `commit_wait()` or sleep outside `wait`)");
                                }
                                self.sema.emit(d);
                            }
                        }
                        _ => {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                arm.call.span,
                                "'wait { ... }' arms must be direct zero-argument consumer primitive calls (`inbox.recv()`, `signal.wait()`, `byte_ring.wait_read()`, `ping_pong.wait_chunk()`, `watch.wait_change()`, `pool.alloc()`); arbitrary calls are not cancellation-safe",
                            ));
                        }
                    }

                    let call_val = self.check_expr_infallible(&arm.call);
                    let _ = is_boost_arm;
                    if let Some((bind_ty, bind_name)) = &arm.bind {
                        let resolved_ty = self.sema.resolve_type(bind_ty);
                        self.sema.check_disallowed_lease_type(
                            &resolved_ty,
                            bind_ty.span,
                            &format!("wait arm binding '{}'", bind_name.name),
                            true,
                        );
                        if call_val.ty.is_void() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                bind_name.span,
                                format!(
                                    "cannot bind variable '{}' to 'void' wait arm '{}'",
                                    bind_name.name, arm.call.span
                                ),
                            ));
                        } else if !call_val.ty.is_error()
                            && !resolved_ty.is_error()
                            && self.sema.check_assignable_or_emit(
                                &call_val.ty,
                                &resolved_ty,
                                arm.call.span,
                                DiagnosticKind::TypeMismatch,
                                || {
                                    format!(
                                        "wait arm binding '{}' declared as '{}', but arm produces '{}'",
                                        bind_name.name, resolved_ty, call_val.ty
                                    )
                                },
                            )
                        {
                            self.check_coerced_expr(&arm.call, &resolved_ty);
                        }
                        let decl_depth = self.current_depth();
                        let (lease_state, lease_gen_key) = if resolved_ty.is_linear_lease() {
                            (
                                Some(LinearLeaseState::Live),
                                Some(self.fresh_lease_gen_key(&bind_name.name)),
                            )
                        } else {
                            (None, None)
                        };
                        if let Some(scope) = self.scopes.last_mut() {
                            scope.insert(
                                bind_name.name.clone(),
                                LocalVar {
                                    name: bind_name.name.clone(),
                                    ty: resolved_ty,
                                    decl_depth,
                                    prov: call_val.prov.clone(),
                                    field_prov: BTreeMap::new(),
                                    endpoint_alias: call_val.endpoint_name.clone(),
                                    field_endpoints: call_val.field_endpoints.clone(),
                                    is_isr_state: false,
                                    lease_state,
                                    lease_gen_key,
                                },
                            );
                        }
                    } else if call_val.ty.is_linear_lease() {
                        self.sema.emit(
                            Diagnostic::new(
                                DiagnosticKind::LinearLeaseViolation,
                                arm.call.span,
                                format!(
                                    "unused linear '{}' returned by 'wait' arm leaks pool slot",
                                    call_val.ty
                                ),
                            )
                            .with_note("bind the `Lease<T>` in the arm header (`Lease<T> l = pool.alloc() => { ... }`) and consume it before leaving the arm"),
                        );
                    }

                    let arm_div = self.check_block_stmts(&arm.body.stmts);
                    self.finish_lexical_block(arm_div, arm.body.span);
                    self.scopes.pop();
                    self.branch_depth -= 1;
                    if !arm_div {
                        all_arms_diverge = false;
                        live_var_branches.push(self.snapshot_outer_vars());
                        live_lease_branches.push(self.leases.clone());
                    }
                }

                if let Some(t) = timer {
                    self.scopes = base_vars.clone();
                    self.leases = base_leases.clone();
                    let tv = self.check_expr_infallible(&t.expr);
                    match t.kind {
                        zev_ast::WaitTimerKind::Timeout => {
                            if tv.ty != Ty::Duration && !tv.ty.is_error() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    t.expr.span,
                                    format!(
                                        "'wait' block `timeout` arm must be a duration literal, found '{}'",
                                        tv.ty
                                    ),
                                ));
                            }
                        }
                        zev_ast::WaitTimerKind::Deadline => {
                            if !tv.ty.is_assignable_to(&Ty::u32()) && !tv.ty.is_error() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    t.expr.span,
                                    format!(
                                        "'wait' block `deadline` arm must be 'u32' (`sys.now_ms()` tick timestamp), found '{}'",
                                        tv.ty
                                    ),
                                ));
                            } else {
                                self.check_coerced_expr(&t.expr, &Ty::u32());
                            }
                        }
                    }
                    self.branch_depth += 1;
                    let t_div = self.check_nested_block(&t.body);
                    self.branch_depth -= 1;
                    if !t_div {
                        all_arms_diverge = false;
                        live_var_branches.push(self.snapshot_outer_vars());
                        live_lease_branches.push(self.leases.clone());
                    }
                }

                self.scopes = base_vars.clone();
                self.merge_branch_vars(&base_vars, &live_var_branches, stmt.span);
                self.leases = base_leases;
                self.leases.join_branches(&live_lease_branches);
                all_arms_diverge
            }

            StmtKind::Spawn(sp) => {
                self.check_spawn(sp);
                false
            }

            StmtKind::Continue => {
                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'continue' cannot be used inside a 'defer' cleanup statement",
                    ));
                } else if self.loop_depth == 0 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'continue' may only be used inside a 'loop', 'loop every', 'while', or 'for' body",
                    ));
                } else {
                    let loop_entry = self.loop_lease_frames.last().map(|(d, _)| *d).unwrap_or(0);
                    self.check_unconsumed_leases_on_exit(stmt.span, "continue", loop_entry);
                    if let Some((entry_d, entry_vars)) = self.loop_lease_frames.last().cloned() {
                        self.check_loop_reassigned_outer_live(
                            &entry_vars,
                            entry_d,
                            stmt.span,
                            "continue",
                        );
                    }
                }
                true
            }

            StmtKind::Break => {
                if self.in_defer() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'break' cannot be used inside a 'defer' cleanup statement",
                    ));
                } else if self.loop_depth == 0 {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidControlFlow,
                        stmt.span,
                        "'break' may only be used inside a 'loop', 'loop every', 'while', or 'for' body",
                    ));
                } else {
                    let loop_entry = self.loop_lease_frames.last().map(|(d, _)| *d).unwrap_or(0);
                    self.check_unconsumed_leases_on_exit(stmt.span, "break", loop_entry);
                    if let Some((entry_d, entry_vars)) = self.loop_lease_frames.last().cloned() {
                        self.check_loop_reassigned_outer_live(
                            &entry_vars,
                            entry_d,
                            stmt.span,
                            "break",
                        );
                    }
                }
                true
            }

            StmtKind::Checkpoint => {
                self.record_direct_yield(stmt.span, FnEffect::SoftYielding, "'checkpoint()'");
                false
            }

            StmtKind::Block(b) => self.check_nested_block(b),

            StmtKind::Expr(e) => {
                let ev = self.check_expr_infallible(e);
                self.validate_int_expr_at_type(e, None);
                if ev.ty.is_linear_lease() {
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::LinearLeaseViolation,
                            e.span,
                            format!(
                                "unused expression of linear type '{}' drops ownership and leaks pool slot",
                                ev.ty
                            ),
                        )
                        .with_note("bind the `Lease<T>` to a local variable (`Lease<T> l = ...;`) or consume it via `.release()`, `dma_slot.park(...)`, or `inbox.post(...)`"),
                    );
                }
                let exits = self.lookup_local(SYS_NAMESPACE).is_none() && is_sys_exit_expr(e);
                if exits {
                    self.check_unconsumed_leases_on_exit(e.span, "sys.exit", 0);
                }
                exits
            }
        }
    }

    pub(super) fn check_loop_reassigned_outer_live(
        &mut self,
        entry_vars: &[BTreeMap<String, LocalVar>],
        entry_depth: u32,
        span: Span,
        ctx: &str,
    ) {
        let mut offenders = Vec::new();
        for (idx, scope) in self.scopes.iter().enumerate() {
            for (name, var) in scope {
                if var.decl_depth <= entry_depth
                    && matches!(var.lease_state, Some(LinearLeaseState::Live))
                {
                    let was_live =
                        entry_vars
                            .get(idx)
                            .and_then(|s| s.get(name))
                            .is_some_and(|ev| {
                                matches!(
                                    ev.lease_state,
                                    Some(
                                        LinearLeaseState::Live
                                            | LinearLeaseState::DeferredRelease { .. }
                                    )
                                )
                            });
                    if !was_live {
                        offenders.push(name.clone());
                    }
                }
            }
        }
        for name in offenders {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!(
                    "Lease<T> variable '{}' acquired inside loop must be consumed before '{}'",
                    name, ctx
                ),
            ));
            if let Some(v) = self.lookup_local_mut(&name) {
                v.lease_state = Some(LinearLeaseState::Moved {
                    moved_at: span,
                    reason: "loop leak",
                });
            }
        }
    }

    pub(super) fn check_for_loop_with_binding(
        &mut self,
        var_name: &str,
        var_ty: Ty,
        var_prov: Provenance,
        iter_span: Option<Span>,
        body: &Block,
    ) {
        let leases_before = self.leases.clone();
        let entry_depth = self.current_depth();
        let loop_entry_vars = self.snapshot_outer_vars();
        self.loop_depth += 1;
        self.loop_entry_depths.push(entry_depth);
        self.loop_outer_reads.push(Vec::new());
        self.loop_lease_frames
            .push((entry_depth, loop_entry_vars.clone()));

        let body_div = self.with_nested_scope(|this| {
            let bind_depth = this.current_depth();
            if let Some(scope) = this.scopes.last_mut() {
                scope.insert(
                    var_name.to_string(),
                    LocalVar {
                        name: var_name.to_string(),
                        ty: var_ty,
                        decl_depth: bind_depth,
                        prov: var_prov.clone(),
                        field_prov: BTreeMap::new(),
                        endpoint_alias: None,
                        field_endpoints: BTreeMap::new(),
                        is_isr_state: false,
                        lease_state: None,
                        lease_gen_key: None,
                    },
                );
            }
            let div = this.check_block_stmts(&body.stmts);
            this.finish_lexical_block(div, body.span);
            div
        });
        if !body_div {
            self.check_loop_reassigned_outer_live(
                &loop_entry_vars,
                entry_depth,
                body.span,
                "end of loop iteration",
            );
        }

        let _ = self.loop_lease_frames.pop();
        let outer_reads = self.loop_outer_reads.pop().unwrap_or_default();
        self.loop_entry_depths.pop();
        self.loop_depth -= 1;

        let modified = self.leases.modified_sources_since(&leases_before);
        if !modified.is_empty() {
            if let Some(isp) = iter_span {
                for tok in &var_prov.leases {
                    if let Some(inv_span) = modified.get(&tok.source) {
                        if leases_before.check_valid(tok).is_ok() {
                            let inv_method = self.leases.last_invalidator(&tok.source);
                            let display_src =
                                tok.source.strip_suffix("#write").unwrap_or(&tok.source);
                            let (msg, note_msg) = if tok.acq_method == "wait_chunk" {
                                (
                                    format!(
                                        "cannot iterate over slice derived from '{}.wait_chunk()' across a loop body that calls 'wait_chunk()'",
                                        tok.source
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "DMA lease invalidated inside 'for..in' body via 'wait_chunk()' at {}",
                                            sp
                                        )
                                    }),
                                )
                            } else {
                                (
                                    format!(
                                        "cannot iterate over slice derived from '{}.{}()' across a loop body that calls '{}.{}()'",
                                        display_src, tok.acq_method, display_src, inv_method
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "byte_ring lease invalidated inside 'for..in' body via '{}()' at {}",
                                            inv_method, sp
                                        )
                                    }),
                                )
                            };
                            let mut diag = Diagnostic::new(DiagnosticKind::StaleDmaLease, isp, msg);
                            if let Some(nm) = note_msg {
                                diag = diag.with_note(nm);
                            }
                            self.sema.emit(diag);
                        }
                    }
                }
            }

            for (r_name, read_env, r_prov, read_span) in outer_reads {
                for tok in &r_prov.leases {
                    if let Some(inv_span) = modified.get(&tok.source) {
                        if read_env.check_valid(tok).is_ok() {
                            let inv_method = self.leases.last_invalidator(&tok.source);
                            let display_src =
                                tok.source.strip_suffix("#write").unwrap_or(&tok.source);
                            let (msg, note_msg) = if tok.acq_method == "wait_chunk" {
                                (
                                    format!(
                                        "cannot read outer variable '{}' derived from '{}.wait_chunk()' across loop iterations that call 'wait_chunk()'",
                                        r_name, tok.source
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "subsequent loop iteration invalidates lease via 'wait_chunk()' at {}",
                                            sp
                                        )
                                    }),
                                )
                            } else {
                                (
                                    format!(
                                        "cannot read outer variable '{}' derived from '{}.{}()' across loop iterations that call '{}.{}()'",
                                        r_name, display_src, tok.acq_method, display_src, inv_method
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "subsequent loop iteration invalidates lease via '{}()' at {}",
                                            inv_method, sp
                                        )
                                    }),
                                )
                            };
                            let mut diag =
                                Diagnostic::new(DiagnosticKind::StaleDmaLease, read_span, msg);
                            if let Some(nm) = note_msg {
                                diag = diag.with_note(nm);
                            }
                            self.sema.emit(diag);
                        }
                    }
                }
            }
        }
    }

    pub(super) fn check_loop_body(&mut self, body: &Block, _infinite: bool) {
        let leases_before = self.leases.clone();
        let entry_depth = self.current_depth();
        let loop_entry_vars = self.snapshot_outer_vars();
        self.loop_depth += 1;
        self.loop_entry_depths.push(entry_depth);
        self.loop_outer_reads.push(Vec::new());
        self.loop_lease_frames
            .push((entry_depth, loop_entry_vars.clone()));

        let body_div = self.check_nested_block(body);
        if !body_div {
            self.check_loop_reassigned_outer_live(
                &loop_entry_vars,
                entry_depth,
                body.span,
                "end of loop iteration",
            );
        }

        let _ = self.loop_lease_frames.pop();
        let outer_reads = self.loop_outer_reads.pop().unwrap_or_default();
        self.loop_entry_depths.pop();
        self.loop_depth -= 1;

        let modified = self.leases.modified_sources_since(&leases_before);
        if !modified.is_empty() {
            // Check if any outer variable holding a pre-loop DMA/ring lease on a source mutated inside
            // the loop was read in the loop (which would read a stale chunk on iteration 2).
            for (var_name, read_env, var_prov, read_span) in outer_reads {
                for tok in &var_prov.leases {
                    if let Some(inv_span) = modified.get(&tok.source) {
                        // Only report if `read_env` hadn't already reported it stale on iteration 1.
                        if read_env.check_valid(tok).is_ok() {
                            let inv_method = self.leases.last_invalidator(&tok.source);
                            let display_src =
                                tok.source.strip_suffix("#write").unwrap_or(&tok.source);
                            let (msg, note_msg) = if tok.acq_method == "wait_chunk" {
                                (
                                    format!(
                                        "cannot read outer variable '{}' derived from '{}.wait_chunk()' across loop iterations that call 'wait_chunk()'",
                                        var_name, tok.source
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "subsequent loop iteration invalidates lease via 'wait_chunk()' at {}",
                                            sp
                                        )
                                    }),
                                )
                            } else {
                                (
                                    format!(
                                        "cannot read outer variable '{}' derived from '{}.{}()' across loop iterations that call '{}.{}()'",
                                        var_name, display_src, tok.acq_method, display_src, inv_method
                                    ),
                                    inv_span.map(|sp| {
                                        format!(
                                            "subsequent loop iteration invalidates lease via '{}()' at {}",
                                            inv_method, sp
                                        )
                                    }),
                                )
                            };
                            let mut diag =
                                Diagnostic::new(DiagnosticKind::StaleDmaLease, read_span, msg);
                            if let Some(nm) = note_msg {
                                diag = diag.with_note(nm);
                            }
                            self.sema.emit(diag);
                        }
                    }
                }
            }
            let _ = entry_depth;
        }
    }

    pub(super) fn check_spawn(&mut self, sp: &SpawnStmt) {
        let mut invalid_ctx = false;
        if self.in_defer() {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidControlFlow,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' inside a 'defer' cleanup statement",
                    sp.task.name
                ),
            ));
        } else if self.in_isr() {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidSpawnContext,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' inside ISR '{}'; tasks must be statically spawned in 'main()' boot setup",
                    sp.task.name,
                    self.display_caller_name()
                ),
            ));
        } else if matches!(self.body_kind, ActiveBodyKind::Task) {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidSpawnContext,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' inside task '{}'; dynamic runtime task spawning is forbidden—wire all tasks in 'main()' boot setup",
                    sp.task.name,
                    self.display_caller_name()
                ),
            ));
        } else if self.loop_depth > 0 {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidSpawnContext,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' inside a loop; every static boot-tree path must map 1:1 to a task slot",
                    sp.task.name
                ),
            ));
        } else if self.after_diverge || self.main_supervisor_started {
            invalid_ctx = true;
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::InvalidSpawnContext,
                sp.span,
                format!(
                    "cannot 'spawn' task '{}' after 'main()' supervisor loop or diverging control flow",
                    sp.task.name
                ),
            ));
        }

        let Some(task_info) = self.sema.tasks.get(&sp.task.name).cloned() else {
            if self.sema.functions.contains_key(&sp.task.name) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    sp.task.span,
                    format!(
                        "cannot 'spawn' function '{}'; only 'task' blueprints may be spawned",
                        sp.task.name
                    ),
                ));
            } else {
                let task_names: Vec<&str> = self.sema.tasks.keys().map(String::as_str).collect();
                let mut diag = Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    sp.task.span,
                    format!("undeclared task blueprint '{}'", sp.task.name),
                );
                if let Some(best) = closest_match(&sp.task.name, task_names, 2) {
                    diag = diag.with_note(format!("did you mean `spawn {}(...)`?", best));
                }
                self.sema.emit(diag);
            }
            for arg in &sp.args {
                let _ = self.check_expr_infallible(&arg.value);
            }
            return;
        };

        if sp.args.len() != task_info.params.len() {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                sp.span,
                format!(
                    "task '{}' expects {} argument(s), found {}",
                    sp.task.name,
                    task_info.params.len(),
                    sp.args.len()
                ),
            ));
        }

        let mut arg_bindings = Vec::with_capacity(sp.args.len());
        for (i, arg) in sp.args.iter().enumerate() {
            let val = self.check_expr_infallible(&arg.value);
            let mut ref_param_origins = BTreeSet::new();
            if let Some(param) = task_info.params.get(i) {
                if self.sema.check_assignable_or_emit(
                    &val.ty,
                    &param.ty,
                    arg.span,
                    DiagnosticKind::TypeMismatch,
                    || {
                        format!(
                            "argument {} to 'spawn {}' has type '{}', expected '{}'",
                            i + 1,
                            sp.task.name,
                            val.ty,
                            param.ty
                        )
                    },
                ) {
                    self.check_coerced_expr(&arg.value, &param.ty);
                }
                if param.ty.contains_reference(&self.sema.structs) {
                    let arg_prov =
                        self.coerce_array_to_slice_prov(&arg.value, &val.ty, &param.ty, &val.prov);
                    if arg_prov.has_non_returning_origin() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::SpawnNonStaticReference,
                            arg.span,
                            format!(
                                "cannot pass non-static reference/slice into 'spawn {}' (argument {}): spawned tasks outlive local stack variables and DMA leases",
                                sp.task.name,
                                i + 1
                            ),
                        ));
                    } else {
                        ref_param_origins = arg_prov.param_origins.clone();
                    }
                }
            }
            let field_endpoints = self.extract_field_endpoint_specs(&val);
            let endpoint_from_param = self.resolve_param_endpoint_index(&val.endpoint_name);
            let static_endpoint = val.endpoint_name.filter(|ep| !ep.starts_with("param:"));
            arg_bindings.push(BootArgBinding {
                static_endpoint,
                endpoint_from_param,
                field_endpoints,
                ref_param_origins,
            });
        }

        if self.caller_key == "main" {
            self.has_spawned = true;
        }

        if !invalid_ctx && self.can_record_boot_step() {
            self.sema
                .call_graph
                .mark_direct_spawn(&self.caller_key, &sp.task.name);
            self.boot_steps.push(BootStep::DirectSpawn {
                task_name: sp.task.name.clone(),
                priority: sp.priority.unwrap_or(Priority::Normal),
                core: sp.core,
                conditional: self.branch_depth > 0,
                arg_bindings,
                span: sp.span,
            });
        }
    }

    pub(super) fn check_assignment(
        &mut self,
        target: &Expr,
        op: AssignOp,
        value_expr: &Expr,
        rhs: ExprVal,
        stmt_span: Span,
    ) {
        match &target.kind {
            ExprKind::Ident(ident) => {
                if self.sema.consts.contains_key(&ident.name) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!("cannot assign to immutable const '{}'", ident.name),
                    ));
                    return;
                }
                if self.sema.primitives.contains_key(&ident.name) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!(
                            "cannot overwrite OS primitive '{}' directly; use its methods (e.g. .publish(), .post())",
                            ident.name
                        ),
                    ));
                    return;
                }
                if self.sema.subsystems.contains_key(&ident.name) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!(
                            "cannot overwrite subsystem instance '{}' directly; operate on its fields via methods",
                            ident.name
                        ),
                    ));
                    return;
                }
                if self.sema.mmio_instances.contains_key(&ident.name)
                    || self.sema.irqs.contains_key(&ident.name)
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!("cannot reassign hardware handle '{}' directly", ident.name),
                    ));
                    return;
                }
                if let Some(ig) = self.sema.illegal_globals.get(&ident.name).cloned() {
                    let ty_str = if ig.ty.is_error() {
                        "u32".to_string()
                    } else {
                        ig.ty.to_string()
                    };
                    self.sema.emit(
                        Diagnostic::new(
                            DiagnosticKind::GlobalIsolationViolation,
                            target.span,
                            format!(
                                "task/ISR isolation violation: '{}' cannot write to raw mutable global '{}'",
                                self.display_caller_name(),
                                ident.name
                            ),
                        )
                        .with_note(format!(
                            "declared at {}; replace raw global `{name}` with `watch<{ty}> {name} = 0;` (single writer via `{name}.publish(val);`), `guarded<{ty}> {name} = 0;` (multi-writer inside `critical {{ ... }}` via `{name}.set(val);`), or `inbox<{ty}, 8> {name};` (queued events via `{name}.post(ev);`)",
                            ig.decl_span,
                            name = ident.name,
                            ty = ty_str
                        )),
                    );
                    return;
                }

                // If compound assignment (`+=`, etc.), first validate reading `ident`.
                if op != AssignOp::Assign {
                    let cur = self.check_expr_infallible(target);
                    self.check_compound_op_types(op, &cur.ty, value_expr, &rhs.ty, stmt_span);
                    return;
                }

                let Some(var_info) = self.lookup_local(&ident.name).cloned() else {
                    let mut diag = Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        ident.span,
                        format!("undeclared variable '{}'", ident.name),
                    );
                    if let Some(hint) = self.suggest_ident_hint(&ident.name) {
                        diag = diag.with_note(hint);
                    }
                    self.sema.emit(diag);
                    return;
                };

                if self.sema.check_assignable_or_emit(
                    &rhs.ty,
                    &var_info.ty,
                    rhs.span,
                    DiagnosticKind::TypeMismatch,
                    || {
                        format!(
                            "cannot assign '{}' to variable '{}' of type '{}'",
                            rhs.ty, ident.name, var_info.ty
                        )
                    },
                ) {
                    self.check_coerced_expr(value_expr, &var_info.ty);
                }

                if var_info.ty.is_linear_lease() {
                    match var_info.lease_state {
                        Some(LinearLeaseState::Live) => {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::LinearLeaseViolation,
                                    stmt_span,
                                    format!(
                                        "cannot overwrite live Lease<T> variable '{}' before consuming its current handle",
                                        ident.name
                                    ),
                                )
                                .with_note("call `.release()`, `dma_slot.park(...)`, or `inbox.post(...)` on it before reassigning"),
                            );
                        }
                        Some(LinearLeaseState::DeferredRelease { span, .. }) => {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::LinearLeaseViolation,
                                    stmt_span,
                                    format!(
                                        "cannot overwrite Lease<T> variable '{}' while `defer` release from {} is pending",
                                        ident.name, span
                                    ),
                                ),
                            );
                        }
                        Some(LinearLeaseState::Moved { .. })
                        | Some(LinearLeaseState::Uninit)
                        | None => {}
                    }
                    if rhs.ty.is_linear_lease() {
                        self.consume_lease_value(value_expr, &rhs, "assignment", stmt_span);
                    }
                    let new_key = self.fresh_lease_gen_key(&ident.name);
                    if let Some(v) = self.lookup_local_mut(&ident.name) {
                        v.lease_state = Some(if rhs.ty.is_error() {
                            LinearLeaseState::Moved {
                                moved_at: stmt_span,
                                reason: "error assign",
                            }
                        } else {
                            LinearLeaseState::Live
                        });
                        v.lease_gen_key = Some(new_key);
                    }
                }

                if var_info.ty.contains_reference(&self.sema.structs)
                    || var_info.ty.contains_fn_ptr(&self.sema.structs)
                {
                    let rhs_prov = self.coerce_array_to_slice_prov(
                        value_expr,
                        &rhs.ty,
                        &var_info.ty,
                        &rhs.prov,
                    );
                    if var_info.decl_depth < rhs_prov.origin_depth {
                        if !rhs_prov.isr_state_origins.is_empty() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidIsrState,
                                stmt_span,
                                format!(
                                    "cannot escape `&` of ISR `state` variable into outer variable '{}' at depth {}",
                                    ident.name, var_info.decl_depth
                                ),
                            ));
                        } else {
                            self.sema.emit(
                                Diagnostic::new(
                                    DiagnosticKind::ScopeDepthEscape,
                                    stmt_span,
                                    format!(
                                        "cannot escape temporary/leased value of scope depth {} into outer variable '{}' declared at scope depth {}",
                                        rhs_prov.origin_depth, ident.name, var_info.decl_depth
                                    ),
                                )
                                .with_note(format!(
                                    "origin depth is {} (nested block/DMA lease) while '{}' lives at outer depth {} (0=static, 1=fn/task body, 2+=nested block); pass the slice/struct down into a helper `fn`, decode into a value struct (`Type.from_bytes(slice)`), or copy bytes into a fixed `[N]u8` buffer",
                                    rhs_prov.origin_depth, ident.name, var_info.decl_depth
                                )),
                            );
                        }
                    }
                    if let Some(v) = self.lookup_local_mut(&ident.name) {
                        v.prov = rhs_prov;
                        v.field_prov.clear();
                        v.endpoint_alias = rhs.endpoint_name;
                        v.field_endpoints = rhs.field_endpoints;
                    }
                }
            }

            ExprKind::Field {
                target: base,
                field,
            } => {
                if let Some(const_name) = self.root_const_of_place(target) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!("cannot assign to field of immutable const '{}'", const_name),
                    ));
                    return;
                }
                // Check field or MMIO register write (`r.buf = half;`, `USART1.icr = 1;`, `regs.cr1 |= 1;`).
                let base_val = self.check_projection_base(base);
                let (struct_name, via_ptr) = match &base_val.ty {
                    Ty::Struct(s) => (s.clone(), false),
                    Ty::MmioLayout(layout_name)
                    | Ty::MmioInstance {
                        layout: layout_name,
                        ..
                    } => {
                        let ln = layout_name.clone();
                        self.check_mmio_reg_write_access(
                            &ln, field, None, op, value_expr, &rhs, stmt_span,
                        );
                        return;
                    }
                    Ty::Ptr(inner) => match &**inner {
                        Ty::Struct(s) => (s.clone(), true),
                        Ty::MmioLayout(layout_name) => {
                            let ln = layout_name.clone();
                            self.check_mmio_reg_write_access(
                                &ln, field, None, op, value_expr, &rhs, stmt_span,
                            );
                            return;
                        }
                        _ => {
                            if !base_val.ty.is_error() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::TypeMismatch,
                                    base.span,
                                    format!("type '{}' has no field '{}'", base_val.ty, field.name),
                                ));
                            }
                            return;
                        }
                    },
                    _ => {
                        if !base_val.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                base.span,
                                format!(
                                    "type '{}' has no assignable field '{}'",
                                    base_val.ty, field.name
                                ),
                            ));
                        }
                        return;
                    }
                };

                let Some(field_ty) = self
                    .sema
                    .structs
                    .get(&struct_name)
                    .and_then(|s| s.field(&field.name))
                    .map(|f| f.ty.clone())
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
                    return;
                };

                if self
                    .sema
                    .structs
                    .get(&struct_name)
                    .is_some_and(|s| s.is_subsystem)
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!(
                            "cannot overwrite field '{}.{}' of subsystem struct directly; operate on it via its methods",
                            struct_name, field.name
                        ),
                    ));
                    return;
                }

                if op == AssignOp::Assign {
                    if self.sema.check_assignable_or_emit(
                        &rhs.ty,
                        &field_ty,
                        rhs.span,
                        DiagnosticKind::TypeMismatch,
                        || {
                            format!(
                                "cannot assign '{}' to field '{}.{}' of type '{}'",
                                rhs.ty, struct_name, field.name, field_ty
                            )
                        },
                    ) {
                        self.check_coerced_expr(value_expr, &field_ty);
                    }
                } else {
                    self.check_compound_op_types(op, &field_ty, value_expr, &rhs.ty, stmt_span);
                }

                if field_ty.contains_reference(&self.sema.structs)
                    || field_ty.contains_fn_ptr(&self.sema.structs)
                {
                    let rhs_prov =
                        self.coerce_array_to_slice_prov(value_expr, &rhs.ty, &field_ty, &rhs.prov);
                    if via_ptr {
                        let dest_depth = base_val.prov.pointee_depth.unwrap_or(0);
                        if (base_val.prov.pointee_is_caller_or_static
                            && rhs_prov.has_non_returning_origin())
                            || dest_depth < rhs_prov.origin_depth
                        {
                            if !rhs_prov.isr_state_origins.is_empty() {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::InvalidIsrState,
                                    stmt_span,
                                    format!(
                                        "cannot escape `&` of ISR `state` variable through pointer field '{}'",
                                        field.name
                                    ),
                                ));
                            } else {
                                self.sema.emit(Diagnostic::new(
                                    DiagnosticKind::ScopeDepthEscape,
                                    stmt_span,
                                    format!(
                                        "cannot store inner/leased reference of depth {} into struct field '{}' through outer pointer (depth {})",
                                        rhs_prov.origin_depth, field.name, dest_depth
                                    ),
                                ));
                            }
                        }
                    } else if let Some(root_name) = root_ident_of(base) {
                        if let Some(root_var) = self.lookup_local(&root_name).cloned() {
                            if root_var.decl_depth < rhs_prov.origin_depth {
                                if !rhs_prov.isr_state_origins.is_empty() {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::InvalidIsrState,
                                        stmt_span,
                                        format!(
                                            "cannot escape `&` of ISR `state` variable into outer field '{}.{}'",
                                            root_name, field.name
                                        ),
                                    ));
                                } else {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::ScopeDepthEscape,
                                        stmt_span,
                                        format!(
                                            "cannot escape value of scope depth {} into field '{}' of outer struct '{}' declared at scope depth {}",
                                            rhs_prov.origin_depth, field.name, root_name, root_var.decl_depth
                                        ),
                                    ));
                                }
                            }
                            if let Some(v) = self.lookup_local_mut(&root_name) {
                                v.field_prov.insert(field.name.clone(), rhs_prov.clone());
                                let mut combined = Provenance::pure_static();
                                for fp in v.field_prov.values() {
                                    combined = combined.merge(fp);
                                }
                                v.prov = combined;
                                if matches!(&base.kind, ExprKind::Ident(_)) {
                                    if let Some(ep) = &rhs.endpoint_name {
                                        v.field_endpoints.insert(field.name.clone(), ep.clone());
                                    } else {
                                        v.field_endpoints.remove(&field.name);
                                    }
                                    let prefix = format!("{}.", field.name);
                                    v.field_endpoints.retain(|k, _| !k.starts_with(&prefix));
                                    for (sub_k, sub_ep) in &rhs.field_endpoints {
                                        v.field_endpoints.insert(
                                            format!("{}.{}", field.name, sub_k),
                                            sub_ep.clone(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: ptr_expr,
            } => {
                let pv = self.check_expr_infallible(ptr_expr);
                let Ty::Ptr(pointee_ty) = &pv.ty else {
                    if !pv.ty.is_error() {
                        self.sema.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            target.span,
                            format!("cannot dereference non-pointer type '{}'", pv.ty),
                        ));
                    }
                    return;
                };
                if pointee_ty.is_void() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        target.span,
                        "cannot dereference opaque '*void' pointer; cast it to a concrete pointer type first",
                    ));
                    return;
                }
                if let Some(ln) = ty_contains_byval_mmio(pointee_ty) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        target.span,
                        format!(
                            "cannot overwrite MMIO register block '{}' by value; assign individual registers instead",
                            ln
                        ),
                    ));
                    return;
                }
                if let Some(sub) = ty_contains_byval_subsys(pointee_ty, &self.sema.structs) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        target.span,
                        format!(
                            "cannot overwrite subsystem struct '{}' by value; operate on its fields via methods",
                            sub
                        ),
                    ));
                    return;
                }
                if ty_embeds_os_primitive(pointee_ty)
                    || ty_contains_bare_unsized_prim(pointee_ty).is_some()
                {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        target.span,
                        format!(
                            "cannot overwrite static OS primitive '{}' by value; use its methods instead",
                            pointee_ty
                        ),
                    ));
                    return;
                }
                if op == AssignOp::Assign {
                    if self.sema.check_assignable_or_emit(
                        &rhs.ty,
                        pointee_ty,
                        rhs.span,
                        DiagnosticKind::TypeMismatch,
                        || {
                            format!(
                                "cannot assign '{}' to dereferenced '*{}'",
                                rhs.ty, pointee_ty
                            )
                        },
                    ) {
                        self.check_coerced_expr(value_expr, pointee_ty);
                    }
                } else {
                    self.check_compound_op_types(op, pointee_ty, value_expr, &rhs.ty, stmt_span);
                }
                if pointee_ty.contains_reference(&self.sema.structs) {
                    let rhs_prov =
                        self.coerce_array_to_slice_prov(value_expr, &rhs.ty, pointee_ty, &rhs.prov);
                    let dest_depth = pv.prov.pointee_depth.unwrap_or(0);
                    if (pv.prov.pointee_is_caller_or_static && rhs_prov.has_non_returning_origin())
                        || dest_depth < rhs_prov.origin_depth
                    {
                        if !rhs_prov.isr_state_origins.is_empty() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::InvalidIsrState,
                                stmt_span,
                                "cannot escape `&` of ISR `state` variable through outer or caller pointer",
                            ));
                        } else {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::ScopeDepthEscape,
                                stmt_span,
                                format!(
                                    "cannot store inner/leased reference of depth {} through pointer targeting outer depth {}",
                                    rhs_prov.origin_depth, dest_depth
                                ),
                            ));
                        }
                    }
                }
            }

            ExprKind::Index {
                target: seq_expr,
                index,
            } => {
                if let Some(const_name) = self.root_const_of_place(target) {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::ImmutableAssignment,
                        target.span,
                        format!(
                            "cannot assign to element of immutable const '{}'",
                            const_name
                        ),
                    ));
                    return;
                }
                // Handle indexed MMIO array register writes (`regs.brr[i] = val;`, `regs.brr[i] |= mask;`)
                // directly so `@wo` array registers are never misread before checking write qualifiers.
                if let ExprKind::Field {
                    target: mmio_base,
                    field,
                } = &seq_expr.kind
                {
                    let base_val = self.check_projection_base(mmio_base);
                    let mmio_layout = match &base_val.ty {
                        Ty::MmioLayout(ln) | Ty::MmioInstance { layout: ln, .. } => {
                            Some(ln.clone())
                        }
                        Ty::Ptr(inner) => match &**inner {
                            Ty::MmioLayout(ln) => Some(ln.clone()),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(ln) = mmio_layout {
                        self.check_mmio_reg_write_access(
                            &ln,
                            field,
                            Some(index),
                            op,
                            value_expr,
                            &rhs,
                            stmt_span,
                        );
                        return;
                    }
                }

                let sv = self.check_expr_infallible(seq_expr);
                let iv = self.check_expr_infallible(index);
                self.validate_int_expr_at_type(index, Some(PrimType::Usize));
                if !iv.ty.is_integer() && !iv.ty.is_error() {
                    self.sema.emit(Diagnostic::new(
                        DiagnosticKind::TypeMismatch,
                        index.span,
                        format!("array/slice index must be an integer, found '{}'", iv.ty),
                    ));
                }
                let elem_ty = match &sv.ty {
                    Ty::Slice(e) | Ty::Array { elem: e, .. } => (**e).clone(),
                    _ => {
                        if !sv.ty.is_error() {
                            self.sema.emit(Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                seq_expr.span,
                                format!("cannot index into type '{}'", sv.ty),
                            ));
                        }
                        return;
                    }
                };
                if op == AssignOp::Assign {
                    if self.sema.check_assignable_or_emit(
                        &rhs.ty,
                        &elem_ty,
                        rhs.span,
                        DiagnosticKind::TypeMismatch,
                        || {
                            format!(
                                "cannot assign '{}' to element of type '{}'",
                                rhs.ty, elem_ty
                            )
                        },
                    ) {
                        self.check_coerced_expr(value_expr, &elem_ty);
                    }
                } else {
                    self.check_compound_op_types(op, &elem_ty, value_expr, &rhs.ty, stmt_span);
                }
                if elem_ty.contains_reference(&self.sema.structs) {
                    let rhs_prov =
                        self.coerce_array_to_slice_prov(value_expr, &rhs.ty, &elem_ty, &rhs.prov);
                    if let Some(root_name) = root_ident_of(seq_expr) {
                        if let Some(root_var) = self.lookup_local(&root_name) {
                            if root_var.decl_depth < rhs_prov.origin_depth {
                                if !rhs_prov.isr_state_origins.is_empty() {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::InvalidIsrState,
                                        stmt_span,
                                        format!(
                                            "cannot escape `&` of ISR `state` variable into outer array/slice '{}'",
                                            root_name
                                        ),
                                    ));
                                } else {
                                    self.sema.emit(Diagnostic::new(
                                        DiagnosticKind::ScopeDepthEscape,
                                        stmt_span,
                                        format!(
                                            "cannot escape value of scope depth {} into array/slice '{}' at scope depth {}",
                                            rhs_prov.origin_depth, root_name, root_var.decl_depth
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            _ => {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::ImmutableAssignment,
                    target.span,
                    "invalid assignment target",
                ));
            }
        }
    }

    pub(super) fn check_compound_op_types(
        &mut self,
        op: AssignOp,
        lhs: &Ty,
        rhs_expr: &Expr,
        rhs: &Ty,
        span: Span,
    ) {
        if lhs.is_error() || rhs.is_error() {
            return;
        }
        if matches!(op, AssignOp::ShlAssign | AssignOp::ShrAssign) {
            if !(lhs.is_integer() && rhs.is_integer()) {
                self.sema.emit(Diagnostic::new(
                    DiagnosticKind::TypeMismatch,
                    span,
                    format!(
                        "compound shift '{}' requires integer operands, found '{}' and '{}'",
                        op.as_str(),
                        lhs,
                        rhs
                    ),
                ));
            } else {
                self.validate_int_expr_at_type(rhs_expr, None);
                self.check_const_shift_amount(op.as_str(), lhs, rhs_expr);
            }
            return;
        }
        let float_arith_ok = matches!(
            op,
            AssignOp::AddAssign
                | AssignOp::SubAssign
                | AssignOp::MulAssign
                | AssignOp::DivAssign
                | AssignOp::RemAssign
        ) && lhs.is_float()
            && rhs.is_float()
            && rhs.is_assignable_to(lhs);
        let int_arith_ok = lhs.is_integer() && rhs.is_integer() && rhs.is_assignable_to(lhs);
        if !int_arith_ok && !float_arith_ok {
            self.sema.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                span,
                format!(
                    "compound assignment '{}' requires compatible numeric operands, found '{}' and '{}'",
                    op.as_str(),
                    lhs,
                    rhs
                ),
            ));
        } else {
            self.check_coerced_expr(rhs_expr, lhs);
        }
    }
}
