use super::*;

impl Analyzer {
    pub fn new(opts: &AnalyzeOptions) -> Self {
        Self {
            tick_period_us: opts.default_tick_period_us.max(1),
            tick_configured_explicitly: false,
            systick_hz: None,
            display_names: BTreeMap::new(),
            program_extern_c_scopes: Vec::new(),
            rewritten_pub_extern_sites: BTreeSet::new(),
            non_root_extern_c_names: BTreeSet::new(),
            non_root_prog_count: 0,
            structs: BTreeMap::new(),
            struct_files: BTreeMap::new(),
            enums: BTreeMap::new(),
            enum_prog_idx: BTreeMap::new(),
            variant_to_enums: BTreeMap::new(),
            consts: BTreeMap::new(),
            raw_const_exprs: BTreeMap::new(),
            primitives: BTreeMap::new(),
            subsystems: BTreeMap::new(),
            mmio_layout_names: BTreeSet::new(),
            mmio_layouts: BTreeMap::new(),
            mmio_instances: BTreeMap::new(),
            irqs: BTreeMap::new(),
            irq_bindings: Vec::new(),
            isr_states: Vec::new(),
            fn_irq_boot_nodes: BTreeMap::new(),
            illegal_globals: BTreeMap::new(),
            functions: BTreeMap::new(),
            fn_decl_order: Vec::new(),
            fn_boot_steps: BTreeMap::new(),
            recorded_call_args: Vec::new(),
            recorded_byte_ring_ops: Vec::new(),
            spawned_task_field_endpoints: BTreeMap::new(),
            prim_escaping_ref_params: BTreeMap::new(),
            tasks: BTreeMap::new(),
            isrs: BTreeMap::new(),
            spawned_tasks: Vec::new(),
            soft_yielding_in_coro: BTreeSet::new(),
            soft_yielding_in_noyield: BTreeSet::new(),
            dual_specialized_fns: BTreeSet::new(),
            main_supervisor_priority: None,
            call_graph: CallGraph::new(),
            current_file: None,
            current_prog_idx: 0,
            current_instantiation_note: None,
            mono_instantiation_notes: BTreeMap::new(),
            f32_overflow_reported_spans: BTreeSet::new(),
            node_files: BTreeMap::new(),
            diagnostics: Vec::new(),
        }
    }

    pub(crate) fn with_linked_context(mut self, linked: &LinkedModulesOutput) -> Self {
        self.display_names = linked.display_names.clone();
        self.program_extern_c_scopes = linked.program_extern_c_scopes.clone();
        self.rewritten_pub_extern_sites = linked.rewritten_pub_extern_sites.clone();
        self.non_root_extern_c_names = linked.non_root_extern_c_names.clone();
        self.non_root_prog_count = linked.non_root_program_count;
        self
    }

    pub(super) fn emit(&mut self, mut diag: Diagnostic) {
        if diag.file.is_none() {
            diag.file = self.current_file.clone();
        }
        if diag.note.is_none() {
            diag.note = Some(diag.kind.doc_hint().to_string());
        }
        if let Some(origin) = &self.current_instantiation_note {
            if let Some(existing) = &mut diag.note {
                if !existing.contains("instantiated from ") {
                    existing.push_str(&format!(" (instantiated from `{}`)", origin));
                }
            } else {
                diag.note = Some(format!("instantiated from `{}`", origin));
            }
        }
        self.diagnostics.push(diag);
    }

    pub fn run(mut self, programs: &[(&Program, Option<&str>)]) -> AnalysisOutput {
        // Merge items while tracking optional file names and program index per item.
        let mut all_items: Vec<(&Item, Option<String>, usize)> = Vec::new();
        for (prog_idx, (prog, file)) in programs.iter().enumerate() {
            for item in &prog.items {
                all_items.push((item, file.map(str::to_string), prog_idx));
            }
        }

        // Pass 1a: Imports, struct names, enum declarations, and MMIO layout type names.
        let mut mono_instances_from_attrs: BTreeSet<MonomorphizedInstance> = BTreeSet::new();
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            match &item.kind {
                ItemKind::Import(imp) => {
                    let dot = imp
                        .path
                        .iter()
                        .map(|id| id.name.as_str())
                        .collect::<Vec<_>>()
                        .join(".");
                    self.emit(Diagnostic::new(
                        DiagnosticKind::ModuleError,
                        imp.span,
                        format!("unresolved module import `{dot}`"),
                    ));
                }
                ItemKind::Struct(st) => {
                    if !st.generic_params.is_empty() {
                        continue;
                    }
                    if let Some(sig) = extract_monomorphized_attr(&item.attrs) {
                        mono_instances_from_attrs.insert(MonomorphizedInstance {
                            kind: MonomorphizedKind::Struct,
                            signature: sig,
                            mangled_name: st.name.name.clone(),
                        });
                    }
                    if let Some(orig) = extract_instantiated_from_attr(&item.attrs) {
                        self.mono_instantiation_notes
                            .insert(st.name.name.clone(), orig);
                    }
                    self.struct_files
                        .insert(st.name.name.clone(), file.clone());
                    self.register_struct_header(st);
                }
                ItemKind::Enum(en) => self.collect_enum(en),
                ItemKind::Mmio(mm)
                    if matches!(
                        mm.kind,
                        MmioDeclKind::Layout { .. } | MmioDeclKind::Inline { .. }
                    ) =>
                {
                    self.register_mmio_layout_name(mm);
                }
                ItemKind::Fn(f) if f.generic_params.is_empty() => {
                    if let Some(sig) = extract_monomorphized_attr(&item.attrs) {
                        mono_instances_from_attrs.insert(MonomorphizedInstance {
                            kind: MonomorphizedKind::Fn,
                            signature: sig,
                            mangled_name: f.name.name.clone(),
                        });
                    }
                    if let Some(orig) = extract_instantiated_from_attr(&item.attrs) {
                        self.mono_instantiation_notes
                            .insert(f.name.name.clone(), orig);
                    }
                }
                _ => {}
            }
        }

        // Pre-register built-in `enum FmtErr { Truncated }` if not user-declared.
        if !self.enums.contains_key("FmtErr")
            && !self.structs.contains_key("FmtErr")
            && !self.mmio_layout_names.contains("FmtErr")
        {
            self.enums.insert(
                "FmtErr".to_string(),
                EnumInfo {
                    name: "FmtErr".to_string(),
                    variants: vec!["Truncated".to_string()],
                    span: Span::dummy(),
                },
            );
            self.enum_prog_idx.insert("FmtErr".to_string(), None);
            if !self.variant_to_enums.contains_key("Truncated") {
                self.variant_to_enums
                    .insert("Truncated".to_string(), vec!["FmtErr".to_string()]);
            }
        }

        // Pass 1b: Resolve struct fields and verify packed struct & recursive value invariants.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            if let ItemKind::Struct(st) = &item.kind {
                if st.generic_params.is_empty() {
                    self.current_instantiation_note =
                        self.mono_instantiation_notes.get(&st.name.name).cloned();
                    self.resolve_struct_fields(st);
                    self.current_instantiation_note = None;
                }
            }
        }
        self.current_file = None;
        self.check_recursive_struct_values();
        self.finalize_subsystem_structs();

        // Pass 1c: Pre-scan for `sys.use_systick` / `sys.set_tick` so custom tick period is known
        // before validating duration literals anywhere in the program.
        for (item, _, _) in &all_items {
            if let ItemKind::Fn(f) = &item.kind {
                if f.generic_params.is_empty() {
                    if let Some(body) = &f.body {
                        self.prescan_tick_config(body);
                    }
                }
            }
        }

        // Pass 1d: Collect and evaluate `const` declarations first so `mmio` offsets/base addresses,
        // `irq` vector numbers, and `[CAP]T` array sizes can reference any top-level `const`.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            if let ItemKind::Const(c) = &item.kind {
                self.collect_const(c);
            }
        }
        self.evaluate_all_const_int_values();
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            if let ItemKind::Const(c) = &item.kind {
                if !self.is_compile_time_const_expr_with_shadowed(
                    &c.value,
                    &BTreeSet::new(),
                    *prog_idx,
                ) {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        c.value.span,
                        format!(
                            "`const` '{}' initializer must be a compile-time constant expression",
                            c.name.name
                        ),
                    ));
                }
            }
        }

        // Collect `mmio`, `irq`, OS primitives, illegal mutable globals, and callable signatures.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            match &item.kind {
                ItemKind::Mmio(mm) => self.collect_mmio(mm),
                ItemKind::Irq(iq) => self.collect_irq(iq),
                ItemKind::OsPrimitive(p) => self.collect_os_primitive(&item.attrs, p),
                ItemKind::GlobalVar(g) => self.collect_global_var(&item.attrs, g),
                ItemKind::Fn(f) if f.generic_params.is_empty() => {
                    self.current_instantiation_note =
                        self.mono_instantiation_notes.get(&f.name.name).cloned();
                    self.collect_fn_sig(&item.attrs, f);
                    self.current_instantiation_note = None;
                }
                ItemKind::Task(t) => self.collect_task_sig(&item.attrs, t),
                ItemKind::Isr(i) => self.collect_isr_sig(&item.attrs, i),
                _ => {}
            }
        }

        // Validate every MMIO instance's [base_addr .. base_addr + byte_size] fits within 32-bit address space
        // after all layouts have been collected (handles forward-declared layouts).
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            if let ItemKind::Mmio(mm) = &item.kind {
                if matches!(
                    mm.kind,
                    MmioDeclKind::Instance { .. } | MmioDeclKind::Inline { .. }
                ) {
                    if let Some(inst) = self.mmio_instances.get(&mm.name.name) {
                        if let Some(layout_info) = self.mmio_layouts.get(&inst.layout) {
                            let size_u32 =
                                u32::try_from(layout_info.byte_size).unwrap_or(u32::MAX);
                            if inst.base_addr.checked_add(size_u32).is_none() {
                                self.emit(Diagnostic::new(
                                    DiagnosticKind::MmioViolation,
                                    mm.span,
                                    format!(
                                        "MMIO instance '{}' at base 0x{:08X} with layout '{}' ({} bytes) overflows 32-bit address space",
                                        inst.name, inst.base_addr, inst.layout, layout_info.byte_size
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Pass 2a: Analyze non-`main` helper functions first so interprocedural `return_param_origins`
        // summaries are ready before `main` and `task` bodies call them.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            if let ItemKind::Fn(f) = &item.kind {
                if f.generic_params.is_empty() && f.name.name != "main" {
                    self.check_fn_body(&item.attrs, f);
                }
            }
        }
        // Pass 2b: Analyze `main()`, `task` blueprints, and `isr` handlers.
        for (item, file, prog_idx) in &all_items {
            self.current_file = file.clone();
            self.current_prog_idx = *prog_idx;
            match &item.kind {
                ItemKind::Fn(f) if f.generic_params.is_empty() && f.name.name == "main" => {
                    self.check_fn_body(&item.attrs, f);
                }
                ItemKind::Task(t) => self.check_task_body(&item.attrs, t),
                ItemKind::Isr(i) => self.check_isr_body(&item.attrs, i),
                _ => {}
            }
        }
        self.current_file = None;

        // Pass 3: Whole-program transitive yield inference, `noyield`/`critical`/`isr` call checks,
        // `extern "C"` callback checks, boot-tree spawn & IRQ attach unrolling, and cycle detection.
        self.run_whole_program_verification();

        let main_decl_opt = all_items.iter().find_map(|(item, _, _)| match &item.kind {
            ItemKind::Fn(f) if f.name.name == "main" => Some(f),
            _ => None,
        });
        self.check_task_slot_limit(main_decl_opt);

        let main_can_yield = self
            .functions
            .get("main")
            .is_some_and(|m| m.can_yield);

        let mut monomorphized_instances: Vec<MonomorphizedInstance> =
            mono_instances_from_attrs.into_iter().collect();
        monomorphized_instances.sort();

        let model = SemanticModel {
            tick_period_us: self.tick_period_us,
            systick_hz: self.systick_hz,
            structs: self.structs,
            enums: self.enums,
            consts: self.consts,
            primitives: self.primitives,
            subsystems: self.subsystems,
            functions: self.functions,
            tasks: self.tasks,
            isrs: self.isrs,
            mmio_layouts: self.mmio_layouts,
            mmio_instances: self.mmio_instances,
            irqs: self.irqs,
            irq_bindings: self.irq_bindings,
            isr_states: self.isr_states,
            display_names: self.display_names,
            spawned_tasks: self.spawned_tasks,
            monomorphized_instances,
            soft_yielding_in_coro: self.soft_yielding_in_coro,
            soft_yielding_in_noyield: self.soft_yielding_in_noyield,
            dual_specialized_fns: self.dual_specialized_fns,
            main_supervisor_priority: self.main_supervisor_priority,
            main_can_yield,
            call_graph: self.call_graph,
        };

        AnalysisOutput {
            model,
            diagnostics: self.diagnostics,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass 1 Helpers
    // ─────────────────────────────────────────────────────────────────────────

    pub(super) fn register_mmio_layout_name(&mut self, mm: &MmioDecl) {
        if self.structs.contains_key(&mm.name.name)
            || self.enums.contains_key(&mm.name.name)
            || !self.mmio_layout_names.insert(mm.name.name.clone())
        {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                mm.name.span,
                format!("duplicate type definition '{}'", mm.name.name),
            ));
        }
    }

    pub(super) fn register_struct_header(&mut self, st: &StructDecl) {
        if self.structs.contains_key(&st.name.name)
            || self.enums.contains_key(&st.name.name)
            || self.mmio_layout_names.contains(&st.name.name)
        {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                st.name.span,
                format!("duplicate type definition '{}'", st.name.name),
            ));
            return;
        }
        self.structs.insert(
            st.name.name.clone(),
            StructInfo {
                name: st.name.name.clone(),
                is_packed: st.is_packed,
                is_subsystem: false,
                fields: Vec::new(),
                span: st.span,
            },
        );
    }

    pub(super) fn collect_enum(&mut self, en: &EnumDecl) {
        if self.enums.contains_key(&en.name.name)
            || self.structs.contains_key(&en.name.name)
            || self.mmio_layout_names.contains(&en.name.name)
        {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                en.name.span,
                format!("duplicate type definition '{}'", en.name.name),
            ));
            return;
        }
        let mut seen = BTreeSet::new();
        let mut variants = Vec::new();
        for v in &en.variants {
            if !seen.insert(v.name.name.clone()) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::DuplicateDefinition,
                    v.name.span,
                    format!(
                        "duplicate variant '{}' in enum '{}'",
                        v.name.name, en.name.name
                    ),
                ));
                continue;
            }
            variants.push(v.name.name.clone());
            self.variant_to_enums
                .entry(v.name.name.clone())
                .or_default()
                .push(en.name.name.clone());
        }
        self.enums.insert(
            en.name.name.clone(),
            EnumInfo {
                name: en.name.name.clone(),
                variants,
                span: en.span,
            },
        );
        let scope_idx = if self.current_prog_idx < self.non_root_prog_count {
            Some(self.current_prog_idx)
        } else {
            None
        };
        self.enum_prog_idx.insert(en.name.name.clone(), scope_idx);
    }

    pub(super) fn visible_enums_for_variant(&self, prog_idx: usize, variant: &str) -> Vec<String> {
        let Some(all) = self.variant_to_enums.get(variant) else {
            return Vec::new();
        };
        all.iter()
            .filter(|en| match self.enum_prog_idx.get(*en).copied().flatten() {
                Some(decl_idx) => decl_idx == prog_idx,
                None => prog_idx >= self.non_root_prog_count,
            })
            .cloned()
            .collect()
    }

    pub(super) fn check_assignable_or_emit(
        &mut self,
        actual: &Ty,
        expected: &Ty,
        span: Span,
        default_kind: DiagnosticKind,
        default_msg: impl FnOnce() -> String,
    ) -> bool {
        if actual.is_assignable_to(expected) {
            return true;
        }
        if matches!(expected, Ty::Ptr(inner) if inner.is_void()) && actual.contains_mmio() {
            self.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                span,
                format!(
                    "cannot coerce MMIO handle '{}' to raw '*void' pointer",
                    actual
                ),
            ));
        } else {
            self.emit(Diagnostic::new(default_kind, span, default_msg()));
        }
        false
    }

    pub(super) fn resolve_struct_fields(&mut self, st: &StructDecl) {
        let mut seen = BTreeSet::new();
        let mut fields = Vec::new();
        for f in &st.fields {
            if !seen.insert(f.name.name.clone()) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::DuplicateDefinition,
                    f.name.span,
                    format!(
                        "duplicate field '{}' in struct '{}'",
                        f.name.name, st.name.name
                    ),
                ));
                continue;
            }
            let ty = self.resolve_type(&f.ty);
            if ty.is_void() {
                self.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    f.ty.span,
                    format!(
                        "field '{}' in struct '{}' cannot have type 'void'",
                        f.name.name, st.name.name
                    ),
                ));
            }
            self.check_type_no_byval_mmio_or_unsized(
                &ty,
                f.ty.span,
                &format!("field '{}' of struct '{}'", f.name.name, st.name.name),
            );
            if st.is_packed && matches!(ty, Ty::Ptr(_) | Ty::Slice(_) | Ty::FnPtr { .. }) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    f.span,
                    format!(
                        "packed struct '{}' must have wire-exact layout and cannot contain pointer or slice field '{}'",
                        st.name.name, f.name.name
                    ),
                ));
            }
            let dma_safe = has_attr_dma_safe(&f.attrs);
            for attr in &f.attrs {
                match &attr.kind {
                    AttrKind::DmaSafe => {
                        if !matches!(
                            ty,
                            Ty::OsPrimitive(
                                ResolvedOsPrimitive::Pool { .. }
                                    | ResolvedOsPrimitive::PingPong { .. }
                            )
                        ) {
                            self.emit(Diagnostic::new(
                                DiagnosticKind::InvalidOperation,
                                attr.span,
                                format!(
                                    "`@dma_safe` on field '{}' of struct '{}' is only allowed on `pool<T, K>` or `ping_pong<T, N>` fields",
                                    f.name.name, st.name.name
                                ),
                            ));
                        }
                    }
                    other => {
                        self.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            attr.span,
                            format!(
                                "attribute `{:?}` is not valid on field '{}' of struct '{}'",
                                other, f.name.name, st.name.name
                            ),
                        ));
                    }
                }
            }
            self.check_disallowed_lease_type(
                &ty,
                f.ty.span,
                &format!("field '{}' of struct '{}'", f.name.name, st.name.name),
                false,
            );
            fields.push(FieldInfo {
                name: f.name.name.clone(),
                ty,
                dma_safe,
                span: f.span,
            });
        }
        if let Some(info) = self.structs.get_mut(&st.name.name) {
            info.fields = fields;
        }
    }

    pub(super) fn check_recursive_struct_values(&mut self) {
        let names: Vec<String> = self.structs.keys().cloned().collect();
        for name in names {
            let mut visiting = BTreeSet::new();
            if self.has_value_cycle(&name, &name, &mut visiting) {
                let span = self.structs[&name].span;
                let file = self.struct_files.get(&name).cloned().flatten();
                self.current_instantiation_note =
                    self.mono_instantiation_notes.get(&name).cloned();
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        span,
                        format!(
                            "recursive struct '{}' has infinite size without indirection",
                            name
                        ),
                    )
                    .with_file(file),
                );
                self.current_instantiation_note = None;
            }
        }
    }

    pub(super) fn has_value_cycle(
        &self,
        target: &str,
        curr: &str,
        visiting: &mut BTreeSet<String>,
    ) -> bool {
        let Some(st) = self.structs.get(curr) else {
            return false;
        };
        for f in &st.fields {
            let mut inner = &f.ty;
            while let Ty::Array { elem, .. } = inner {
                inner = elem;
            }
            if let Ty::Struct(next) = inner {
                if next == target {
                    return true;
                }
                if visiting.insert(next.clone()) && self.has_value_cycle(target, next, visiting) {
                    return true;
                }
            }
        }
        false
    }

    pub(super) fn finalize_subsystem_structs(&mut self) {
        let names: Vec<String> = self.structs.keys().cloned().collect();
        // Fixed-point propagate `is_subsystem` across nested subsystem structs.
        loop {
            let mut changed = false;
            for name in &names {
                if self.structs[name].is_subsystem {
                    continue;
                }
                let has_subsys_content = self.structs[name].fields.iter().any(|f| {
                    ty_embeds_os_primitive(&f.ty)
                        || ty_contains_byval_subsys(&f.ty, &self.structs).is_some()
                });
                if has_subsys_content {
                    if let Some(info) = self.structs.get_mut(name) {
                        info.is_subsystem = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        for name in &names {
            let Some(st) = self.structs.get(name).cloned() else {
                continue;
            };
            if !st.is_subsystem {
                continue;
            }
            let file = self.struct_files.get(name).cloned().flatten();
            self.current_instantiation_note = self.mono_instantiation_notes.get(name).cloned();
            if st.is_packed {
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        st.span,
                        format!(
                            "packed struct '{}' cannot embed OS primitives or act as a subsystem struct",
                            st.name
                        ),
                    )
                    .with_file(file.clone()),
                );
            }
            for f in &st.fields {
                let valid_field = match &f.ty {
                    Ty::OsPrimitive(os) => !os.is_unsized_view(),
                    Ty::Struct(sub) => self.structs.get(sub).is_some_and(|s| s.is_subsystem),
                    _ => false,
                };
                if !valid_field {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            f.span,
                            format!(
                                "subsystem struct '{}' cannot mix non-primitive field '{}' of type '{}' (all fields must be concrete OS primitives or nested subsystem structs)",
                                st.name, f.name, f.ty
                            ),
                        )
                        .with_file(file.clone())
                        .with_note("move raw scalar/buffer fields into `watch<T>`, `inbox<T, N>`, `pool<T, K>`, `dma_slot<T>`, or `byte_ring<N>` fields"),
                    );
                }
                match &f.ty {
                    Ty::OsPrimitive(ResolvedOsPrimitive::Pool { elem, .. }) => {
                        self.validate_pool_slot_elem(
                            elem,
                            f.span,
                            &format!("pool field '{}.{}'", st.name, f.name),
                        );
                    }
                    Ty::OsPrimitive(ResolvedOsPrimitive::DmaSlot { elem }) => {
                        self.validate_pool_slot_elem(
                            elem,
                            f.span,
                            &format!("dma_slot field '{}.{}'", st.name, f.name),
                        );
                    }
                    _ => {}
                }
            }
            self.current_instantiation_note = None;
        }
    }

    pub(super) fn prescan_tick_config(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.prescan_tick_stmt(stmt);
        }
    }

    pub(super) fn prescan_tick_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Expr(expr) => self.prescan_tick_expr(expr),
            StmtKind::Block(b) | StmtKind::NoYield { body: b } | StmtKind::Critical { body: b } => {
                self.prescan_tick_config(b);
            }
            StmtKind::Defer(inner) => self.prescan_tick_stmt(inner),
            StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                self.prescan_tick_config(then_block);
                if let Some(eb) = else_block {
                    self.prescan_tick_config(eb);
                }
            }
            _ => {}
        }
    }

    pub(super) fn prescan_tick_expr(&mut self, expr: &Expr) {
        if let ExprKind::MethodCall {
            receiver,
            method,
            args,
        } = &expr.kind
        {
            if matches!(&receiver.kind, ExprKind::Ident(id) if id.name == SYS_NAMESPACE) {
                match SysMethod::from_name(&method.name) {
                    Some(SysMethod::UseSystick) => {
                        if let Some(first) = args.first() {
                            if let ExprKind::Literal(Literal::Int(hz)) = &first.value.kind {
                                self.systick_hz = Some(*hz);
                            }
                        }
                        if let Some(second) = args.get(1) {
                            if let ExprKind::Literal(Literal::Duration(dur)) = &second.value.kind {
                                let us = dur.as_micros();
                                if us > 0 {
                                    self.tick_period_us = us;
                                    self.tick_configured_explicitly = true;
                                }
                            }
                        }
                    }
                    Some(SysMethod::SetTick | SysMethod::SetTickPeriod) => {
                        if let Some(first) = args.first() {
                            if let ExprKind::Literal(Literal::Duration(dur)) = &first.value.kind {
                                let us = dur.as_micros();
                                if us > 0 {
                                    self.tick_period_us = us;
                                    self.tick_configured_explicitly = true;
                                }
                            }
                        }
                    }
                    Some(SysMethod::SetTickUs) => {
                        if let Some(first) = args.first() {
                            if let ExprKind::Literal(Literal::Int(us)) = &first.value.kind {
                                if *us > 0 {
                                    self.tick_period_us = *us;
                                    self.tick_configured_explicitly = true;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub(super) fn check_top_level_name_unique(&mut self, name: &Ident) -> bool {
        if self.consts.contains_key(&name.name)
            || self.primitives.contains_key(&name.name)
            || self.subsystems.contains_key(&name.name)
            || self.mmio_instances.contains_key(&name.name)
            || self.irqs.contains_key(&name.name)
            || self.illegal_globals.contains_key(&name.name)
            || self.functions.contains_key(&name.name)
            || self.tasks.contains_key(&name.name)
        {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                name.span,
                format!("duplicate top-level symbol '{}'", name.name),
            ));
            return false;
        }
        true
    }

    pub(super) fn collect_mmio(&mut self, mm: &MmioDecl) {
        match &mm.kind {
            MmioDeclKind::Layout { regs } => {
                self.collect_mmio_layout(&mm.name, regs, mm.span);
            }
            MmioDeclKind::Instance { layout, base_addr } => {
                if !self.check_top_level_name_unique(&mm.name) {
                    return;
                }
                let resolved_layout = self.resolve_type(layout);
                let layout_name = match resolved_layout {
                    Ty::MmioLayout(l) => l,
                    Ty::Error => return,
                    other => {
                        self.emit(Diagnostic::new(
                            DiagnosticKind::MmioViolation,
                            layout.span,
                            format!(
                                "MMIO instance '{}' must use a `mmio` register layout type, found '{}'",
                                mm.name.name, other
                            ),
                        ));
                        return;
                    }
                };
                let base_u32 = self.eval_u32_const(
                    base_addr,
                    DiagnosticKind::MmioViolation,
                    &format!("MMIO instance '{}' base address", mm.name.name),
                );
                self.mmio_instances.insert(
                    mm.name.name.clone(),
                    MmioInstanceInfo {
                        name: mm.name.name.clone(),
                        layout: layout_name,
                        base_addr: base_u32,
                        span: mm.span,
                    },
                );
            }
            MmioDeclKind::Inline { base_addr, regs } => {
                if !self.check_top_level_name_unique(&mm.name) {
                    return;
                }
                let layout_id = Ident::new(format!("{}__Layout", mm.name.name), mm.name.span);
                self.collect_mmio_layout(&layout_id, regs, mm.span);
                let base_u32 = self.eval_u32_const(
                    base_addr,
                    DiagnosticKind::MmioViolation,
                    &format!("MMIO instance '{}' base address", mm.name.name),
                );
                self.mmio_instances.insert(
                    mm.name.name.clone(),
                    MmioInstanceInfo {
                        name: mm.name.name.clone(),
                        layout: layout_id.name,
                        base_addr: base_u32,
                        span: mm.span,
                    },
                );
            }
        }
    }

    pub(super) fn collect_mmio_layout(
        &mut self,
        name: &Ident,
        regs: &[zev_ast::MmioRegDecl],
        span: Span,
    ) {
        if self.mmio_layouts.contains_key(&name.name) {
            return;
        }
        let mut seen_regs = BTreeSet::new();
        let mut resolved_regs = Vec::new();
        let mut ranges: Vec<(u64, u64, String)> = Vec::new();
        let mut max_end: u64 = 0;

        for reg in regs {
            if !seen_regs.insert(reg.name.name.clone()) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.name.span,
                    format!(
                        "duplicate MMIO register '{}' in layout '{}'",
                        reg.name.name, name.name
                    ),
                ));
                continue;
            }

            let reg_ty = self.resolve_type(&reg.ty);
            let (elem_ty, array_len, elem_bytes) = match &reg_ty {
                Ty::Prim(p @ (PrimType::U8 | PrimType::U16 | PrimType::U32)) => {
                    (*p, None, p.byte_size(4) as u64)
                }
                Ty::Array { elem, len } => match &**elem {
                    Ty::Prim(p @ (PrimType::U8 | PrimType::U16 | PrimType::U32)) => {
                        if *len == 0 {
                            self.emit(Diagnostic::new(
                                DiagnosticKind::MmioViolation,
                                reg.ty.span,
                                format!(
                                    "MMIO register array '{}.{}' must have length >= 1",
                                    name.name, reg.name.name
                                ),
                            ));
                        }
                        (*p, Some((*len).max(1)), p.byte_size(4) as u64)
                    }
                    _ => {
                        self.emit(Diagnostic::new(
                            DiagnosticKind::MmioViolation,
                            reg.ty.span,
                            format!(
                                "MMIO register '{}.{}' has unsupported type '{}'; only `u8`, `u16`, `u32`, and fixed arrays `[N]u8`/`[N]u16`/`[N]u32` are supported",
                                name.name, reg.name.name, reg_ty
                            ),
                        ));
                        continue;
                    }
                },
                Ty::Error => continue,
                other => {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        reg.ty.span,
                        format!(
                            "MMIO register '{}.{}' has unsupported type '{}'; only `u8`, `u16`, `u32`, and fixed arrays `[N]u8`/`[N]u16`/`[N]u32` are supported",
                            name.name, reg.name.name, other
                        ),
                    ));
                    continue;
                }
            };

            let offset_u32 = self.eval_u32_const(
                &reg.offset,
                DiagnosticKind::MmioViolation,
                &format!("MMIO register '{}.{}' offset", name.name, reg.name.name),
            );
            let offset_u64 = u64::from(offset_u32);

            if offset_u64 % elem_bytes != 0 {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.offset.span,
                    format!(
                        "MMIO register '{}.{}' at offset 0x{:X} is not naturally aligned to {}-byte `{}` boundary",
                        name.name, reg.name.name, offset_u64, elem_bytes, elem_ty
                    ),
                ));
            }

            let Ok(count_u32) = u32::try_from(array_len.unwrap_or(1)) else {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.ty.span,
                    format!(
                        "MMIO register array '{}.{}' length {} exceeds `u32` address space",
                        name.name,
                        reg.name.name,
                        array_len.unwrap_or(1)
                    ),
                ));
                continue;
            };
            let elem_bytes_u32 = elem_bytes as u32;
            let Some(reg_bytes_u32) = elem_bytes_u32.checked_mul(count_u32) else {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.ty.span,
                    format!(
                        "MMIO register '{}.{}' byte size overflows 32-bit MMIO address space",
                        name.name, reg.name.name
                    ),
                ));
                continue;
            };
            let Some(end_u32) = offset_u32.checked_add(reg_bytes_u32) else {
                self.emit(Diagnostic::new(
                    DiagnosticKind::MmioViolation,
                    reg.span,
                    format!(
                        "MMIO register '{}.{}' at offset 0x{:X} (size {} bytes) overflows 32-bit MMIO address space",
                        name.name, reg.name.name, offset_u32, reg_bytes_u32
                    ),
                ));
                continue;
            };
            let end = u64::from(end_u32);

            for (prev_start, prev_end, prev_name) in &ranges {
                if offset_u64 < *prev_end && *prev_start < end {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::MmioViolation,
                        reg.span,
                        format!(
                            "MMIO register '{}.{}' (0x{:X}..0x{:X}) overlaps register '{}' (0x{:X}..0x{:X})",
                            name.name, reg.name.name, offset_u64, end, prev_name, prev_start, prev_end
                        ),
                    ));
                    break;
                }
            }

            ranges.push((offset_u64, end, reg.name.name.clone()));
            max_end = max_end.max(end);
            resolved_regs.push(MmioRegInfo {
                name: reg.name.name.clone(),
                access: reg.access,
                elem_ty,
                array_len,
                offset: offset_u32,
                span: reg.span,
            });
        }

        self.mmio_layouts.insert(
            name.name.clone(),
            MmioLayoutInfo {
                name: name.name.clone(),
                byte_size: usize::try_from(max_end).unwrap_or(usize::MAX),
                regs: resolved_regs,
                span,
            },
        );
    }

    pub(super) fn collect_irq(&mut self, iq: &IrqDecl) {
        if !self.check_top_level_name_unique(&iq.name) {
            return;
        }
        let irq_num = self.eval_u32_const(
            &iq.number,
            DiagnosticKind::IrqAttachmentError,
            &format!("IRQ '{}' vector number", iq.name.name),
        );
        self.irqs.insert(
            iq.name.name.clone(),
            IrqInfo {
                name: iq.name.name.clone(),
                irq_num,
                span: iq.span,
            },
        );
    }

    pub(super) fn eval_u32_const(&mut self, expr: &Expr, diag_kind: DiagnosticKind, what: &str) -> u32 {
        let mut visiting = BTreeSet::new();
        match self.eval_const_expr_i128(expr, &mut visiting) {
            Some(v) => match u32::try_from(v) {
                Ok(u) => u,
                Err(_) => {
                    self.emit(Diagnostic::new(
                        diag_kind,
                        expr.span,
                        format!("{what} value {v} is outside valid `u32` range (0..=0xFFFF_FFFF)"),
                    ));
                    0
                }
            },
            None => {
                self.emit(Diagnostic::new(
                    diag_kind,
                    expr.span,
                    format!("{what} must be a compile-time `u32` constant expression"),
                ));
                0
            }
        }
    }

    pub(super) fn collect_const(&mut self, c: &ConstDecl) {
        if !self.check_top_level_name_unique(&c.name) {
            return;
        }

        let ty = self.resolve_type(&c.ty);
        self.raw_const_exprs
            .insert(c.name.name.clone(), (c.value.clone(), ty.clone(), self.current_prog_idx));
        let int_value = match &c.value.kind {
            ExprKind::Literal(Literal::Int(v)) => Some(*v),
            _ => None,
        };
        let mut body_checker = BodyChecker::new(
            self,
            "<const>",
            CallableKind::Fn,
            ActiveBodyKind::Fn {
                ret_ty: Ty::void(),
                error_spec: ResolvedErrorSpec::None,
            },
        );
        let val = body_checker.check_expr_infallible(&c.value);
        if !val.ty.is_assignable_to(&ty) {
            self.emit(Diagnostic::new(
                DiagnosticKind::TypeMismatch,
                c.value.span,
                format!(
                    "const '{}' expected initializer of type '{}', found '{}'",
                    c.name.name, ty, val.ty
                ),
            ));
        } else {
            self.check_f32_coercion_expr(&c.value, &ty);
            self.check_const_untyped_shift_coercion(&c.value, &ty, true);
        }

        self.consts.insert(
            c.name.name.clone(),
            ConstInfo {
                name: c.name.name.clone(),
                ty,
                int_value,
                span: c.span,
            },
        );
    }

    pub(super) fn evaluate_all_const_int_values(&mut self) {
        let names: Vec<String> = self.consts.keys().cloned().collect();
        for name in names {
            let mut visiting = BTreeSet::new();
            if let Some(v) = self.eval_named_const_i128(&name, &mut visiting) {
                if let Ok(u) = u64::try_from(v) {
                    if let Some(info) = self.consts.get_mut(&name) {
                        info.int_value = Some(u);
                    }
                }
            }
        }
    }

    pub(super) fn eval_named_const_i128(
        &self,
        name: &str,
        visiting: &mut BTreeSet<String>,
    ) -> Option<i128> {
        let (expr, ty, _) = self.raw_const_exprs.get(name)?;
        if !visiting.insert(name.to_string()) {
            return None;
        }
        let raw = self.eval_const_expr_i128(expr, visiting);
        visiting.remove(name);
        let val = raw?;
        if let Ty::Prim(p) = ty {
            if p.is_integer() {
                return Some(crate::monomorphize::truncate_int_prim(val, *p));
            }
        }
        Some(val)
    }

    pub(super) fn infer_const_prim_ty(&self, expr: &Expr) -> Option<PrimType> {
        match &expr.kind {
            ExprKind::Ident(id) => match self.consts.get(&id.name).map(|c| &c.ty) {
                Some(Ty::Prim(p)) if p.is_integer() => Some(*p),
                _ => None,
            },
            ExprKind::Cast { ty, .. } => match &ty.kind {
                TypeKind::Prim(p) if p.is_integer() => Some(*p),
                TypeKind::Named(id) => PrimType::from_name(&id.name).filter(|p| p.is_integer()),
                _ => None,
            },
            ExprKind::Unary { expr: inner, .. } => self.infer_const_prim_ty(inner),
            ExprKind::Binary { lhs, rhs, .. } => self
                .infer_const_prim_ty(lhs)
                .or_else(|| self.infer_const_prim_ty(rhs)),
            _ => None,
        }
    }

    pub(super) fn eval_const_expr_i128(
        &self,
        expr: &Expr,
        visiting: &mut BTreeSet<String>,
    ) -> Option<i128> {
        self.eval_const_expr_i128_shadowed(expr, visiting, &BTreeSet::new())
    }

    pub(super) fn eval_const_expr_i128_shadowed(
        &self,
        expr: &Expr,
        visiting: &mut BTreeSet<String>,
        shadowed: &BTreeSet<String>,
    ) -> Option<i128> {
        match &expr.kind {
            ExprKind::Literal(Literal::Int(v)) => Some(*v as i128),
            ExprKind::Ident(id) => {
                if shadowed.contains(&id.name) {
                    return None;
                }
                self.eval_named_const_i128(&id.name, visiting).or_else(|| {
                    self.consts
                        .get(&id.name)
                        .and_then(|c| c.int_value)
                        .map(|u| u as i128)
                })
            }
            ExprKind::Cast { expr: inner, ty } => {
                let v = self.eval_const_expr_i128_shadowed(inner, visiting, shadowed)?;
                let target_prim = match &ty.kind {
                    TypeKind::Prim(p) if p.is_integer() => Some(*p),
                    TypeKind::Named(id) => PrimType::from_name(&id.name).filter(|p| p.is_integer()),
                    _ => None,
                }?;
                Some(crate::monomorphize::truncate_int_prim(v, target_prim))
            }
            ExprKind::Unary { op, expr: inner } => {
                let v = self.eval_const_expr_i128_shadowed(inner, visiting, shadowed)?;
                match op {
                    UnaryOp::Neg => {
                        let neg = v.checked_neg()?;
                        if let Some(p) = self.infer_const_prim_ty(inner) {
                            if p.is_signed() {
                                return Some(crate::monomorphize::truncate_int_prim(neg, p));
                            }
                        }
                        Some(neg)
                    }
                    UnaryOp::BitNot => {
                        let inv = !v;
                        if let Some(p) = self.infer_const_prim_ty(inner) {
                            Some(crate::monomorphize::truncate_int_prim(inv, p))
                        } else {
                            Some(inv)
                        }
                    }
                    _ => None,
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.eval_const_expr_i128_shadowed(lhs, visiting, shadowed)?;
                let r = self.eval_const_expr_i128_shadowed(rhs, visiting, shadowed)?;
                match op {
                    BinaryOp::Add => l.checked_add(r),
                    BinaryOp::Sub => l.checked_sub(r),
                    BinaryOp::Mul => l.checked_mul(r),
                    BinaryOp::Div if r != 0 => l.checked_div(r),
                    BinaryOp::Rem if r != 0 => l.checked_rem(r),
                    BinaryOp::BitOr => Some(l | r),
                    BinaryOp::BitAnd => Some(l & r),
                    BinaryOp::BitXor => Some(l ^ r),
                    BinaryOp::Shl => u32::try_from(r).ok().and_then(|sh| l.checked_shl(sh)),
                    BinaryOp::Shr => u32::try_from(r).ok().and_then(|sh| l.checked_shr(sh)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub(super) fn check_f32_coercion_expr(&mut self, expr: &Expr, target_ty: &Ty) {
        match target_ty {
            Ty::Prim(PrimType::F32) => match &expr.kind {
                ExprKind::Literal(Literal::Float { bits, suffix: None }) => {
                    if !(f64::from_bits(*bits) as f32).is_finite()
                        && self
                            .f32_overflow_reported_spans
                            .insert((expr.span.start, expr.span.end))
                    {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::TypeMismatch,
                                expr.span,
                                "float literal out of finite 'f32' range",
                            )
                            .with_note(
                                "use `f64` for double-precision exponents above ~3.40e38, or write an explicit `(f32)` cast if saturation to infinity is intended",
                            ),
                        );
                    }
                }
                ExprKind::Unary {
                    op: UnaryOp::Neg,
                    expr: inner,
                } => self.check_f32_coercion_expr(inner, target_ty),
                ExprKind::Binary {
                    op:
                        BinaryOp::Add
                        | BinaryOp::Sub
                        | BinaryOp::Mul
                        | BinaryOp::Div
                        | BinaryOp::Rem,
                    lhs,
                    rhs,
                } => {
                    self.check_f32_coercion_expr(lhs, target_ty);
                    self.check_f32_coercion_expr(rhs, target_ty);
                }
                ExprKind::Or {
                    expr: a,
                    fallback: b,
                } => {
                    self.check_f32_coercion_expr(a, target_ty);
                    self.check_f32_coercion_expr(b, target_ty);
                }
                ExprKind::Catch { expr: a, clause } => {
                    self.check_f32_coercion_expr(a, target_ty);
                    if let Some(Stmt {
                        kind: StmtKind::Expr(tail),
                        ..
                    }) = clause.body.stmts.last()
                    {
                        self.check_f32_coercion_expr(tail, target_ty);
                    }
                }
                _ => {}
            },
            Ty::Array { elem, .. } | Ty::Slice(elem) => {
                if let ExprKind::ArrayLit(elems) = &expr.kind {
                    for e in elems {
                        self.check_f32_coercion_expr(e, elem);
                    }
                }
            }
            _ => {
                if target_ty.fixed_int_bit_width().is_some_and(|b| b < 64) {
                    self.check_const_untyped_shift_coercion(expr, target_ty, false);
                }
            }
        }
    }

    pub(super) fn check_const_untyped_shift_coercion(
        &mut self,
        expr: &Expr,
        target_ty: &Ty,
        allow_ident_consts: bool,
    ) {
        let Some(bits) = target_ty.fixed_int_bit_width() else {
            return;
        };
        if bits >= 64 {
            return;
        }
        match &expr.kind {
            ExprKind::Binary {
                op: op @ (BinaryOp::Shl | BinaryOp::Shr),
                lhs,
                rhs,
            } if Self::expr_is_untyped_int_tree(lhs) => {
                let sh_opt = if allow_ident_consts {
                    self.eval_const_expr_i128(rhs, &mut BTreeSet::new())
                } else {
                    Self::eval_literal_int_expr(rhs)
                };
                if let Some(sh) = sh_opt {
                    if sh >= i128::from(bits)
                        && sh < 64
                        && self
                            .f32_overflow_reported_spans
                            .insert((rhs.span.start, rhs.span.end))
                    {
                        self.emit(Diagnostic::new(
                            DiagnosticKind::InvalidOperation,
                            rhs.span,
                            format!(
                                "shift count {sh} for '{}' is out of range for '{target_ty}' (must be in 0..{bits})",
                                op.as_str()
                            ),
                        ));
                    }
                }
            }
            ExprKind::Unary {
                op: UnaryOp::Neg | UnaryOp::BitNot,
                expr: inner,
            } => self.check_const_untyped_shift_coercion(inner, target_ty, allow_ident_consts),
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
            } if Self::expr_is_untyped_int_tree(expr) => {
                self.check_const_untyped_shift_coercion(lhs, target_ty, allow_ident_consts);
                self.check_const_untyped_shift_coercion(rhs, target_ty, allow_ident_consts);
            }
            _ => {}
        }
    }

    fn expr_is_untyped_int_tree(expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Literal(Literal::Int(_)) => true,
            ExprKind::Unary {
                op: UnaryOp::BitNot,
                expr: inner,
            } => Self::expr_is_untyped_int_tree(inner),
            ExprKind::Binary {
                op:
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem
                    | BinaryOp::BitOr
                    | BinaryOp::BitAnd
                    | BinaryOp::BitXor
                    | BinaryOp::Shl
                    | BinaryOp::Shr,
                lhs,
                rhs,
            } => {
                if matches!(expr.kind, ExprKind::Binary { op: BinaryOp::Shl | BinaryOp::Shr, .. }) {
                    Self::expr_is_untyped_int_tree(lhs)
                } else {
                    Self::expr_is_untyped_int_tree(lhs) && Self::expr_is_untyped_int_tree(rhs)
                }
            }
            _ => false,
        }
    }

    fn eval_literal_int_expr(expr: &Expr) -> Option<i128> {
        match &expr.kind {
            ExprKind::Literal(Literal::Int(v)) => Some(i128::from(*v)),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => Self::eval_literal_int_expr(inner)?.checked_neg(),
            ExprKind::Unary {
                op: UnaryOp::BitNot,
                expr: inner,
            } => Some(!Self::eval_literal_int_expr(inner)?),
            ExprKind::Binary { op, lhs, rhs } => {
                let l = Self::eval_literal_int_expr(lhs)?;
                let r = Self::eval_literal_int_expr(rhs)?;
                match op {
                    BinaryOp::Add => l.checked_add(r),
                    BinaryOp::Sub => l.checked_sub(r),
                    BinaryOp::Mul => l.checked_mul(r),
                    BinaryOp::Div if r != 0 => l.checked_div(r),
                    BinaryOp::Rem if r != 0 => l.checked_rem(r),
                    BinaryOp::BitOr => Some(l | r),
                    BinaryOp::BitAnd => Some(l | r).map(|_| l & r),
                    BinaryOp::BitXor => Some(l ^ r),
                    BinaryOp::Shl => u32::try_from(r).ok().and_then(|sh| l.checked_shl(sh)),
                    BinaryOp::Shr => u32::try_from(r).ok().and_then(|sh| l.checked_shr(sh)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub(super) fn collect_os_primitive(&mut self, item_attrs: &[Attr], p: &OsPrimitiveDecl) {
        if !self.check_top_level_name_unique(&p.name) {
            return;
        }
        let dma_safe = has_attr_dma_safe(item_attrs) || has_attr_dma_safe(&p.attrs);
        let resolved = self.resolve_os_primitive(&p.ty, p.span);
        if resolved.is_unsized_view() {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                p.span,
                format!(
                    "top-level OS primitive '{}' requires a concrete capacity `<N>` (use `{}` view only behind pointer `*`)",
                    resolved, resolved
                ),
            ));
            return;
        }

        if let Some(init_expr) = &p.init {
            let mut body_checker = BodyChecker::new(
                self,
                "<primitive-init>",
                CallableKind::Fn,
                ActiveBodyKind::Fn {
                    ret_ty: Ty::void(),
                    error_spec: ResolvedErrorSpec::None,
                },
            );
            let val = body_checker.check_expr_infallible(init_expr);
            match &resolved {
                ResolvedOsPrimitive::Watch { elem } => {
                    if !val.ty.is_assignable_to(elem) {
                        self.emit(Diagnostic::new(
                            DiagnosticKind::TypeMismatch,
                            init_expr.span,
                            format!(
                                "watch '{}' initializer has type '{}', expected '{}'",
                                p.name.name, val.ty, elem
                            ),
                        ));
                    } else {
                        self.check_f32_coercion_expr(init_expr, elem);
                    }
                }
                other => {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        init_expr.span,
                        format!(
                            "OS primitive '{}' of kind '{}' does not take a scalar initializer",
                            p.name.name,
                            other.kind()
                        ),
                    ));
                }
            }
        }

        let kind = resolved.kind();
        let elem_ty = resolved.elem_ty().cloned();
        if let Some(et) = &elem_ty {
            self.check_type_no_byval_mmio(
                et,
                p.span,
                &format!("element type of OS primitive '{}'", p.name.name),
            );
            match &resolved {
                ResolvedOsPrimitive::Pool { .. } => {
                    self.validate_pool_slot_elem(
                        et,
                        p.span,
                        &format!("pool '{}'", p.name.name),
                    );
                }
                ResolvedOsPrimitive::DmaSlot { .. } => {
                    self.validate_pool_slot_elem(
                        et,
                        p.span,
                        &format!("dma_slot '{}'", p.name.name),
                    );
                }
                ResolvedOsPrimitive::Inbox { elem, .. }
                    if matches!(&**elem, Ty::Lease(_)) =>
                {
                    if let Ty::Lease(inner) = &**elem {
                        self.validate_pool_slot_elem(
                            inner,
                            p.span,
                            &format!("Lease<T> in inbox '{}'", p.name.name),
                        );
                    }
                }
                _ => {
                    self.check_disallowed_lease_type(
                        et,
                        p.span,
                        &format!("OS primitive '{}'", p.name.name),
                        false,
                    );
                }
            }
        }
        let capacity = resolved.capacity();
        let clamps_sleep_depth = matches!(kind, PrimitiveKind::PingPong)
            || (dma_safe && matches!(kind, PrimitiveKind::Pool));
        self.primitives.insert(
            p.name.name.clone(),
            PrimitiveInfo {
                name: p.name.name.clone(),
                primitive: resolved,
                kind,
                elem_ty,
                capacity,
                dma_safe,
                has_init: p.init.is_some(),
                clamps_sleep_depth,
                span: p.span,
            },
        );
    }

    pub(super) fn collect_global_var(&mut self, item_attrs: &[Attr], g: &GlobalVarDecl) {
        // Resilient fallback if a parser ever produces GlobalVar with an OsPrimitive TypeKind.
        if let TypeKind::OsPrimitive(os_ty) = &g.ty.kind {
            let synth = OsPrimitiveDecl {
                attrs: Vec::new(),
                ty: os_ty.clone(),
                name: g.name.clone(),
                init: g.init.clone(),
                span: g.span,
            };
            self.collect_os_primitive(item_attrs, &synth);
            return;
        }

        if !self.check_top_level_name_unique(&g.name) {
            return;
        }
        let ty = self.resolve_type(&g.ty);
        if let Ty::Struct(struct_name) = &ty {
            if self.structs.get(struct_name).is_some_and(|s| s.is_subsystem) {
                if let Some(init_expr) = &g.init {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        init_expr.span,
                        format!(
                            "subsystem instance '{}' of type '{}' cannot have a runtime initializer; embedded OS primitives are statically zero-initialized",
                            g.name.name, struct_name
                        ),
                    ));
                }
                self.subsystems.insert(
                    g.name.name.clone(),
                    SubsystemInfo {
                        name: g.name.name.clone(),
                        struct_name: struct_name.clone(),
                        dma_safe: has_attr_dma_safe(item_attrs),
                        span: g.span,
                    },
                );
                return;
            }
        }
        let ty_str = if ty.is_error() {
            "u32".to_string()
        } else {
            ty.to_string()
        };
        self.emit(
            Diagnostic::new(
                DiagnosticKind::IllegalMutableGlobal,
                g.span,
                format!(
                    "mutable raw global variable '{}' is forbidden; top-level declarations must be 'const' or an OS primitive (signal, inbox, watch, ping_pong, pool, dma_slot, byte_ring)",
                    g.name.name
                ),
            )
            .with_note(format!(
                "replace with `const {ty} {name} = ...;` (immutable), `watch<{ty}> {name} = 0;` (single-writer latest value via `.read()`/`.publish(val)`), or `inbox<{ty}, 8> {name};` (queued events via `.post(ev)`/`.recv()`)",
                ty = ty_str,
                name = g.name.name
            )),
        );
        self.check_disallowed_lease_type(
            &ty,
            g.ty.span,
            &format!("global declaration '{}'", g.name.name),
            false,
        );

        self.illegal_globals.insert(
            g.name.name.clone(),
            IllegalGlobalInfo {
                ty,
                decl_span: g.span,
            },
        );
    }

    pub(super) fn collect_fn_sig(&mut self, _item_attrs: &[Attr], f: &FnDecl) {
        if !self.check_top_level_name_unique(&f.name) {
            return;
        }
        let ret_ty = self.resolve_type(&f.ret_type);
        self.check_type_no_byval_mmio(
            &ret_ty,
            f.ret_type.span,
            &format!("return type of function '{}'", f.name.name),
        );
        if f.is_extern_c {
            self.check_disallowed_lease_type(
                &ret_ty,
                f.ret_type.span,
                &format!("return type of extern \"C\" function '{}'", f.name.name),
                false,
            );
        } else {
            self.check_disallowed_lease_type(
                &ret_ty,
                f.ret_type.span,
                &format!("return type of function '{}'", f.name.name),
                true,
            );
        }
        let error_spec = match &f.error_spec {
            None => ResolvedErrorSpec::None,
            Some(ErrorSpec::Bare(_)) => ResolvedErrorSpec::Untyped,
            Some(ErrorSpec::Typed(err_ty_expr)) => {
                let err_ty = self.resolve_type(err_ty_expr);
                self.check_type_no_byval_mmio(
                    &err_ty,
                    err_ty_expr.span,
                    &format!("error specification of function '{}'", f.name.name),
                );
                self.check_disallowed_lease_type(
                    &err_ty,
                    err_ty_expr.span,
                    &format!("error specification of function '{}'", f.name.name),
                    false,
                );
                if err_ty.is_void() {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::ErrorSpecMismatch,
                        err_ty_expr.span,
                        "typed 'error' specification cannot use 'void'; use bare 'error' instead",
                    ));
                }
                ResolvedErrorSpec::Typed(err_ty)
            }
        };

        if !f.is_extern_c && f.body.is_none() {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                f.span,
                format!(
                    "function '{}' has no body (only 'extern \"C\"' declarations may omit a body)",
                    f.name.name
                ),
            ));
        }

        let params = self.resolve_params(&f.params, &f.name.name, !f.is_extern_c);
        let callable_kind = if f.is_extern_c {
            CallableKind::ExternCFn
        } else {
            CallableKind::Fn
        };
        self.call_graph
            .add_node(f.name.name.clone(), callable_kind, f.span);
        if let Some(cf) = &self.current_file {
            self.node_files.insert(f.name.name.clone(), cf.clone());
        }

        self.fn_decl_order.push(f.name.name.clone());
        self.functions.insert(
            f.name.name.clone(),
            FunctionInfo {
                name: f.name.name.clone(),
                params,
                is_variadic: f.is_variadic,
                ret_ty,
                error_spec,
                is_extern_c: f.is_extern_c,
                direct_yield: false,
                can_yield: false,
                effect: FnEffect::NonYielding,
                spawns_in_boot: false,
                yield_chain: Vec::new(),
                callees: Vec::new(),
                yielding_callees: Vec::new(),
                return_param_origins: Vec::new(),
                span: f.span,
            },
        );
    }

    pub(super) fn collect_task_sig(&mut self, _item_attrs: &[Attr], t: &TaskDecl) {
        if !self.check_top_level_name_unique(&t.name) {
            return;
        }
        let params = self.resolve_params(&t.params, &t.name.name, false);
        let node_key = format!("task::{}", t.name.name);
        if let Some(cf) = &self.current_file {
            self.node_files.insert(node_key.clone(), cf.clone());
        }
        self.call_graph
            .add_node(node_key, CallableKind::Task, t.span);

        self.tasks.insert(
            t.name.name.clone(),
            TaskInfo {
                name: t.name.name.clone(),
                params,
                direct_yield: false,
                can_yield: false,
                effect: FnEffect::NonYielding,
                callees: Vec::new(),
                yielding_callees: Vec::new(),
                instance_count: 0,
                span: t.span,
            },
        );
    }

    pub(super) fn collect_isr_sig(&mut self, _item_attrs: &[Attr], i: &IsrDecl) {
        if self.isrs.contains_key(&i.vector.name) {
            self.emit(Diagnostic::new(
                DiagnosticKind::DuplicateDefinition,
                i.vector.span,
                format!("duplicate ISR handler for vector '{}'", i.vector.name),
            ));
            return;
        }
        let params = self.resolve_params(&i.params, &format!("isr {}", i.vector.name), false);
        let node_key = format!("isr::{}", i.vector.name);
        if let Some(cf) = &self.current_file {
            self.node_files.insert(node_key.clone(), cf.clone());
        }
        self.call_graph
            .add_node(node_key, CallableKind::Isr, i.span);

        self.isrs.insert(
            i.vector.name.clone(),
            IsrInfo {
                vector: i.vector.name.clone(),
                params,
                callees: Vec::new(),
                priority_boosts: Vec::new(),
                span: i.span,
            },
        );
    }

    pub(super) fn resolve_params(
        &mut self,
        params: &[Param],
        owner: &str,
        allow_top_level_lease: bool,
    ) -> Vec<ParamInfo> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for p in params {
            if !seen.insert(p.name.name.clone()) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::DuplicateDefinition,
                    p.name.span,
                    format!("duplicate parameter '{}' in '{}'", p.name.name, owner),
                ));
            }
            let ty = self.resolve_type(&p.ty);
            if ty.is_void() {
                self.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    p.ty.span,
                    format!("parameter '{}' in '{}' cannot have type 'void'", p.name.name, owner),
                ));
            }
            self.check_type_no_byval_mmio(
                &ty,
                p.ty.span,
                &format!("parameter '{}' of '{}'", p.name.name, owner),
            );
            self.check_disallowed_lease_type(
                &ty,
                p.ty.span,
                &format!("parameter '{}' of '{}'", p.name.name, owner),
                allow_top_level_lease,
            );
            if matches!(
                &ty,
                Ty::OsPrimitive(
                    ResolvedOsPrimitive::Pool { .. } | ResolvedOsPrimitive::DmaSlot { .. }
                )
            ) {
                self.emit(Diagnostic::new(
                    DiagnosticKind::InvalidOperation,
                    p.ty.span,
                    format!(
                        "cannot pass static OS primitive '{}' by value in parameter '{}' of '{}' (pass a pointer `*{}`)",
                        ty, p.name.name, owner, ty
                    ),
                ));
            }
            out.push(ParamInfo {
                name: p.name.name.clone(),
                ty,
                span: p.span,
            });
        }
        out
    }

    pub(super) fn check_type_no_byval_mmio_or_unsized(&mut self, ty: &Ty, span: Span, context: &str) {
        if let Some(layout_name) = ty_contains_byval_mmio(ty) {
            self.emit(Diagnostic::new(
                DiagnosticKind::MmioViolation,
                span,
                format!(
                    "cannot pass or store MMIO register layout '{}' by value in {} (use pointer `*{}`)",
                    layout_name, context, layout_name
                ),
            ));
        }
        if let Some(unsized_ty) = ty_contains_bare_unsized_prim(ty) {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                span,
                format!(
                    "unsized primitive type '{}' in {} requires a concrete capacity `<N>` (or pointer view `*{}`)",
                    unsized_ty, context, unsized_ty
                ),
            ));
        }
    }

    pub(super) fn check_type_no_byval_mmio(&mut self, ty: &Ty, span: Span, context: &str) {
        self.check_type_no_byval_mmio_or_unsized(ty, span, context);
        if let Some(subsys_name) = ty_contains_byval_subsys(ty, &self.structs) {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                span,
                format!(
                    "cannot pass, store, or return subsystem struct '{}' by value in {} (use pointer `*{}`)",
                    subsys_name, context, subsys_name
                ),
            ));
        }
    }

    pub(super) fn resolve_type(&mut self, te: &TypeExpr) -> Ty {
        match &te.kind {
            TypeKind::Prim(p) => Ty::Prim(*p),
            TypeKind::Named(ident) => {
                if let Some(p) = PrimType::from_name(&ident.name) {
                    return Ty::Prim(p);
                }
                if self.structs.contains_key(&ident.name) {
                    return Ty::Struct(ident.name.clone());
                }
                if self.enums.contains_key(&ident.name) {
                    return Ty::Enum(ident.name.clone());
                }
                if self.mmio_layouts.contains_key(&ident.name)
                    || self.mmio_layout_names.contains(&ident.name)
                {
                    return Ty::MmioLayout(ident.name.clone());
                }
                let hint = if let Some(z_ty) = suggest_foreign_type_name(&ident.name) {
                    format!(
                        "did you mean Zev built-in type `{}`? (Zev uses explicit fixed-width types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `usize`, `bool`, `u16be`, `u32le`, `[]u8`)",
                        z_ty
                    )
                } else {
                    let prim_names = [
                        "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize",
                        "f32", "f64", "u16be", "u16le", "u32be", "u32le", "u64be", "u64le",
                        "bool", "void",
                    ];
                    let candidates = prim_names
                        .into_iter()
                        .chain(self.structs.keys().map(String::as_str))
                        .chain(self.enums.keys().map(String::as_str));
                    if let Some(best) = closest_match(&ident.name, candidates, 2) {
                        format!("did you mean type `{}`?", best)
                    } else {
                        "built-in Zev types: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `usize`, `bool`, `void`, `u16be`, `u32le`, `[]T`, `*T`".to_string()
                    }
                };
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::UndefinedSymbol,
                        ident.span,
                        format!("unknown type '{}'", ident.name),
                    )
                    .with_note(hint),
                );
                Ty::Error
            }
            TypeKind::Path(segments) => {
                let joined = segments
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                self.emit(Diagnostic::new(
                    DiagnosticKind::UndefinedSymbol,
                    te.span,
                    format!("unknown qualified type '{}'", joined),
                ));
                Ty::Error
            }
            TypeKind::Generic { base, .. } => {
                self.emit(Diagnostic::new(
                    DiagnosticKind::GenericInstantiationError,
                    te.span,
                    format!("unmonomorphized generic type '{}'", base.name),
                ));
                Ty::Error
            }
            TypeKind::GenericPath { path, .. } => {
                let joined = path
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                self.emit(Diagnostic::new(
                    DiagnosticKind::GenericInstantiationError,
                    te.span,
                    format!("unmonomorphized generic type '{}'", joined),
                ));
                Ty::Error
            }
            TypeKind::Slice(elem) => Ty::Slice(Box::new(self.resolve_type(elem))),
            TypeKind::Array { elem, len } => {
                let resolved_len = match len {
                    ArrayLen::Fixed(n) => *n,
                    ArrayLen::Expr(expr) => {
                        let mut visiting = BTreeSet::new();
                        if let Some(v) = self.eval_const_expr_i128(expr, &mut visiting) {
                            usize::try_from(v).unwrap_or_else(|_| {
                                self.emit(Diagnostic::new(
                                    DiagnosticKind::GenericInstantiationError,
                                    expr.span,
                                    "array length must be a non-negative compile-time constant 'usize'",
                                ));
                                1
                            })
                        } else {
                            self.emit(Diagnostic::new(
                                DiagnosticKind::GenericInstantiationError,
                                expr.span,
                                "array length must be a compile-time constant 'usize'",
                            ));
                            1
                        }
                    }
                };
                if resolved_len == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        te.span,
                        "fixed array length must be greater than 0",
                    ));
                }
                Ty::Array {
                    elem: Box::new(self.resolve_type(elem)),
                    len: resolved_len,
                }
            }
            TypeKind::Ptr(inner) => Ty::Ptr(Box::new(self.resolve_type(inner))),
            TypeKind::OsPrimitive(os) => Ty::OsPrimitive(self.resolve_os_primitive(os, te.span)),
            TypeKind::Lease(inner) => {
                let elem = self.resolve_non_void_elem(inner);
                Ty::Lease(Box::new(elem))
            }
            TypeKind::FnPtr { params, ret } => {
                let ps = params.iter().map(|p| self.resolve_type(p)).collect();
                let r = self.resolve_type(ret);
                Ty::FnPtr {
                    params: ps,
                    ret: Box::new(r),
                }
            }
        }
    }

    pub(super) fn resolve_os_primitive(&mut self, os: &OsPrimitiveType, span: Span) -> ResolvedOsPrimitive {
        match os {
            OsPrimitiveType::Signal => ResolvedOsPrimitive::Signal,
            OsPrimitiveType::WaitQueue => ResolvedOsPrimitive::WaitQueue,
            OsPrimitiveType::Inbox { elem, cap } => {
                if *cap == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        span,
                        "inbox capacity must be greater than 0",
                    ));
                }
                ResolvedOsPrimitive::Inbox {
                    elem: Box::new(self.resolve_non_void_elem(elem)),
                    cap: *cap,
                }
            }
            OsPrimitiveType::InboxUnsized { elem } => ResolvedOsPrimitive::InboxUnsized {
                elem: Box::new(self.resolve_non_void_elem(elem)),
            },
            OsPrimitiveType::Watch { elem } => ResolvedOsPrimitive::Watch {
                elem: Box::new(self.resolve_non_void_elem(elem)),
            },
            OsPrimitiveType::PingPong { elem, cap } => {
                if *cap == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        span,
                        "ping_pong buffer capacity must be greater than 0",
                    ));
                }
                ResolvedOsPrimitive::PingPong {
                    elem: Box::new(self.resolve_non_void_elem(elem)),
                    cap: *cap,
                }
            }
            OsPrimitiveType::Pool { elem, cap } => {
                if *cap == 0 || *cap > 32 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        span,
                        format!(
                            "pool capacity must be between 1 and 32 slots (32-bit bitmask), found {}",
                            cap
                        ),
                    ));
                }
                ResolvedOsPrimitive::Pool {
                    elem: Box::new(self.resolve_non_void_elem(elem)),
                    cap: *cap,
                }
            }
            OsPrimitiveType::DmaSlot { elem } => ResolvedOsPrimitive::DmaSlot {
                elem: Box::new(self.resolve_non_void_elem(elem)),
            },
            OsPrimitiveType::ByteRing { cap } => {
                if *cap == 0 {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::InvalidOperation,
                        span,
                        "byte_ring capacity must be greater than 0",
                    ));
                }
                ResolvedOsPrimitive::ByteRing { cap: *cap }
            }
            OsPrimitiveType::ByteRingUnsized => ResolvedOsPrimitive::ByteRingUnsized,
        }
    }

    pub(super) fn resolve_non_void_elem(&mut self, elem: &TypeExpr) -> Ty {
        let t = self.resolve_type(elem);
        if t.is_void() {
            self.emit(Diagnostic::new(
                DiagnosticKind::InvalidOperation,
                elem.span,
                "OS primitive element type cannot be 'void'",
            ));
        }
        t
    }

    pub(super) fn validate_duration(&mut self, dur: &DurationLit) {
        let us = dur.as_micros();
        if us == 0 {
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::InvalidDurationResolution,
                    dur.span,
                    format!("duration '{}' must be greater than 0", dur),
                )
                .with_note(format!(
                    "scheduler tick period is {}us; use at least 1 tick (e.g. `1ms`)",
                    self.tick_period_us
                )),
            );
            return;
        }
        if us < self.tick_period_us {
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::InvalidDurationResolution,
                    dur.span,
                    format!(
                        "sub-tick duration '{}' ({}us) is smaller than configured tick resolution {}us and would truncate to 0 ticks",
                        dur, us, self.tick_period_us
                    ),
                )
                .with_note(format!(
                    "increase the duration to at least `{}us` (`1ms` by default) or configure a finer tick in `main()` via `sys.use_systick(hz, {});`",
                    self.tick_period_us, dur
                )),
            );
            return;
        }
        if us % self.tick_period_us != 0 {
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::InvalidDurationResolution,
                    dur.span,
                    format!(
                        "duration '{}' ({}us) is not an exact multiple of configured tick resolution {}us",
                        dur, us, self.tick_period_us
                    ),
                )
                .with_note(format!(
                    "use an exact multiple of {}us or configure a finer tick period via `sys.use_systick(hz, 500us);`",
                    self.tick_period_us
                )),
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass 2 Entry Points
    // ─────────────────────────────────────────────────────────────────────────

    pub(super) fn check_fn_body(&mut self, item_attrs: &[Attr], f: &FnDecl) {
        let Some(body) = &f.body else {
            return;
        };
        let Some(fn_info) = self.functions.get(&f.name.name).cloned() else {
            return;
        };
        let prev_note = self.current_instantiation_note.take();
        self.current_instantiation_note =
            self.mono_instantiation_notes.get(&f.name.name).cloned();
        let force_noyield = has_attr_noyield(item_attrs) || has_attr_noyield(&f.attrs);
        let mut checker =
            BodyChecker::new(self, &f.name.name, CallableKind::Fn, ActiveBodyKind::Fn {
                ret_ty: fn_info.ret_ty.clone(),
                error_spec: fn_info.error_spec.clone(),
            });
        if force_noyield {
            checker.noyield_depth += 1;
        }
        checker.bind_params(&fn_info.params);
        let diverges = checker.check_block_stmts(&body.stmts);
        if !diverges {
            checker.finish_lexical_block(false, f.name.span);
        }
        let ret_origins: Vec<usize> = checker.return_param_origins.into_iter().collect();
        let boot_steps = checker.boot_steps;
        let irq_nodes = checker
            .boot_block_stack
            .into_iter()
            .next()
            .unwrap_or_default();

        if !fn_info.ret_ty.is_void() && !fn_info.ret_ty.is_error() && !diverges {
            let ret_ty = fn_info.ret_ty.clone();
            self.emit(Diagnostic::new(
                DiagnosticKind::ReturnMismatch,
                f.name.span,
                format!(
                    "function '{}' declared with return type '{}' may exit without returning a value",
                    f.name.name, ret_ty
                ),
            ));
        }
        self.current_instantiation_note = prev_note;

        self.fn_boot_steps.insert(f.name.name.clone(), boot_steps);
        self.fn_irq_boot_nodes
            .insert(f.name.name.clone(), irq_nodes);
        if let Some(info) = self.functions.get_mut(&f.name.name) {
            info.return_param_origins = ret_origins;
        }
    }

    pub(super) fn check_task_body(&mut self, item_attrs: &[Attr], t: &TaskDecl) {
        let Some(task_info) = self.tasks.get(&t.name.name).cloned() else {
            return;
        };
        let node_key = format!("task::{}", t.name.name);
        let force_noyield = has_attr_noyield(item_attrs) || has_attr_noyield(&t.attrs);
        let mut checker =
            BodyChecker::new(self, &node_key, CallableKind::Task, ActiveBodyKind::Task);
        if force_noyield {
            checker.noyield_depth += 1;
        }
        checker.bind_params(&task_info.params);
        let diverges = checker.check_block_stmts(&t.body.stmts);
        if !diverges {
            self.emit(Diagnostic::new(
                DiagnosticKind::NonDivergingTask,
                t.name.span,
                format!(
                    "task '{}' may exit without diverging; tasks are permanent actors and must run inside 'loop' / 'loop every', park on 'inbox.recv()' / 'signal.wait()', reset via 'restart;', or move finite workflows into yielding 'fn's",
                    t.name.name
                ),
            ));
        }
    }

    pub(super) fn check_isr_body(&mut self, _item_attrs: &[Attr], i: &IsrDecl) {
        let Some(isr_info) = self.isrs.get(&i.vector.name).cloned() else {
            return;
        };
        let node_key = format!("isr::{}", i.vector.name);
        let mut checker =
            BodyChecker::new(self, &node_key, CallableKind::Isr, ActiveBodyKind::Isr);
        checker.bind_params(&isr_info.params);
        let diverges = checker.check_block_stmts(&i.body.stmts);
        if !diverges {
            checker.finish_lexical_block(false, i.vector.span);
        }
        let boosts = checker.recorded_boosts;
        if let Some(info) = self.isrs.get_mut(&i.vector.name) {
            info.priority_boosts = boosts;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass 3: Whole-Program Call-Graph Verification & Boot-Tree Unrolling
    // ─────────────────────────────────────────────────────────────────────────

    pub(super) fn run_whole_program_verification(&mut self) {
        self.call_graph.infer_yields();

        // 1. Check all recorded call edges against `isr`, `critical`, `defer`, `noyield`, and boot-spawn contexts.
        let edges = self.call_graph.edges.clone();
        for edge in &edges {
            let Some(callee_node) = self.call_graph.nodes.get(&edge.callee).cloned() else {
                continue;
            };
            let caller_file = self.node_files.get(&edge.caller).cloned();

            // 1a. Yielding call checks (3-tier: `SoftYielding` allowed inside `noyield`, forbidden in `critical`/`isr`/`defer`).
            if edge.ctx.in_isr {
                if callee_node.any_yield_effect.can_yield() {
                    let isr_name = edge.caller.strip_prefix("isr::").unwrap_or(&edge.caller);
                    let sub_chain = self
                        .call_graph
                        .shortest_any_yield_chain(&edge.callee)
                        .join(" -> ");
                    let note = format!(
                        "yield chain: isr {} -> {}; ISRs cannot block or yield—signal a task via `inbox.post(Ev, boost: @critical);`, `signal.notify();`, or `dma.flip();`",
                        isr_name, sub_chain
                    );
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::YieldInIsr,
                            edge.span,
                            format!(
                                "cannot call yielding function '{}' inside ISR '{}'",
                                edge.callee, isr_name
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(note),
                    );
                }
            } else if edge.ctx.in_critical {
                if callee_node.any_yield_effect.can_yield() {
                    let caller_disp = edge
                        .caller
                        .strip_prefix("task::")
                        .unwrap_or(&edge.caller);
                    let sub_chain = self
                        .call_graph
                        .shortest_any_yield_chain(&edge.callee)
                        .join(" -> ");
                    let note = format!(
                        "yield chain: critical block in {} -> {}; `critical {{ ... }}` masks hardware IRQs—move yielding calls outside `critical`",
                        caller_disp, sub_chain
                    );
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::YieldInCritical,
                            edge.span,
                            format!(
                                "cannot call yielding function '{}' inside 'critical' block (IRQs are masked)",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(note),
                    );
                }
            } else if edge.ctx.in_defer {
                if callee_node.any_yield_effect.can_yield() {
                    let caller_disp = edge
                        .caller
                        .strip_prefix("task::")
                        .unwrap_or(&edge.caller);
                    let sub_chain = self
                        .call_graph
                        .shortest_any_yield_chain(&edge.callee)
                        .join(" -> ");
                    let note = format!(
                        "yield chain: defer cleanup in {} -> {}; `defer` statements execute synchronously on scope exit and cannot yield",
                        caller_disp, sub_chain
                    );
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::YieldInNoYield,
                            edge.span,
                            format!(
                                "cannot call yielding function '{}' inside 'defer' cleanup statement",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(note),
                    );
                }
            } else if edge.ctx.in_noyield && callee_node.effect == FnEffect::HardBlocking {
                let caller_disp = edge
                    .caller
                    .strip_prefix("task::")
                    .unwrap_or(&edge.caller);
                let sub_chain = self
                    .call_graph
                    .shortest_hard_yield_chain(&edge.callee)
                    .join(" -> ");
                let note = format!(
                    "yield chain: noyield block in {} -> {}; move hard-blocking calls outside `noyield {{ ... }}` (`SoftYielding` functions that only call `checkpoint()` ARE allowed inside `noyield`)",
                    caller_disp, sub_chain
                );
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::YieldInNoYield,
                        edge.span,
                        format!(
                            "cannot call hard-blocking function '{}' inside 'noyield' scope",
                            edge.callee
                        ),
                    )
                    .with_file(caller_file.clone())
                    .with_note(note),
                );
            }

            // 1b. Boot-only `spawns_in_boot` call-site checks.
            if callee_node.spawns_in_boot {
                let spawn_chain = callee_node.spawn_chain.join(" -> ");
                if edge.ctx.in_defer {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidControlFlow,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' inside 'defer' cleanup statement",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                } else if edge.caller_kind == CallableKind::Task {
                    let task_name = edge.caller.strip_prefix("task::").unwrap_or(&edge.caller);
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' inside task '{}'; tasks must be statically spawned in 'main()' boot setup",
                                edge.callee, task_name
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                } else if edge.caller_kind == CallableKind::Isr || edge.ctx.in_isr {
                    let isr_name = edge.caller.strip_prefix("isr::").unwrap_or(&edge.caller);
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' inside ISR '{}'",
                                edge.callee, isr_name
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                } else if edge.ctx.in_loop {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' inside a loop; every static boot-tree path must map 1:1 to a task slot",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                } else if edge.ctx.after_diverge || edge.ctx.after_main_supervisor {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            edge.span,
                            format!(
                                "cannot call boot-spawning function '{}' after 'main()' supervisor loop or diverging control flow",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file.clone())
                        .with_note(format!("spawn chain: {}", spawn_chain)),
                    );
                }
            }

            // 1c. Boot-only `attaches_in_boot` call-site checks (`E0032`).
            if callee_node.attaches_in_boot {
                let attach_chain = callee_node.attach_chain.join(" -> ");
                if edge.ctx.in_defer {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' inside 'defer' cleanup statement",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                } else if edge.caller_kind == CallableKind::Task {
                    let task_name = edge.caller.strip_prefix("task::").unwrap_or(&edge.caller);
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' inside task '{}'; IRQs must be attached during sequential boot",
                                edge.callee, task_name
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                } else if edge.caller_kind == CallableKind::Isr || edge.ctx.in_isr {
                    let isr_name = edge.caller.strip_prefix("isr::").unwrap_or(&edge.caller);
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' inside ISR '{}'",
                                edge.callee, isr_name
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                } else if edge.ctx.in_loop {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' inside a loop; IRQ attachments must be deterministic in boot",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                } else if edge.ctx.after_diverge || edge.ctx.after_main_supervisor {
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            edge.span,
                            format!(
                                "cannot call IRQ-attaching function '{}' after 'main()' supervisor loop or diverging control flow",
                                edge.callee
                            ),
                        )
                        .with_file(caller_file)
                        .with_note(format!("attach chain: {}", attach_chain)),
                    );
                }
            }
        }

        // 1d. Reject yielding inside boot-only (`spawns_in_boot` / `attaches_in_boot`) helpers.
        let boot_nodes: Vec<_> = self
            .call_graph
            .nodes
            .iter()
            .filter(|(name, node)| {
                node.kind == CallableKind::Fn
                    && *name != "main"
                    && (node.spawns_in_boot || node.attaches_in_boot)
                    && node.any_yield_effect.can_yield()
            })
            .map(|(name, node)| (name.clone(), node.clone()))
            .collect();
        for (name, node) in boot_nodes {
            let fn_file = self.node_files.get(&name).cloned();
            let yield_chain = self.call_graph.shortest_any_yield_chain(&name).join(" -> ");
            if node.spawns_in_boot {
                let spawn_chain = node.spawn_chain.join(" -> ");
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::InvalidSpawnContext,
                        node.span,
                        format!(
                            "boot-spawning helper '{}' cannot yield: boot helpers run sequentially before the cooperative scheduler starts",
                            name
                        ),
                    )
                    .with_file(fn_file)
                    .with_note(format!(
                        "yield chain: {} (spawn chain: {}); move yielding work into a spawned `task` or run it in `main()` after `spawn`",
                        yield_chain, spawn_chain
                    )),
                );
            } else {
                let attach_chain = node.attach_chain.join(" -> ");
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::IrqAttachmentError,
                        node.span,
                        format!(
                            "IRQ-attaching boot helper '{}' cannot yield: boot helpers run sequentially before the cooperative scheduler starts",
                            name
                        ),
                    )
                    .with_file(fn_file)
                    .with_note(format!(
                        "yield chain: {} (attach chain: {}); move yielding calls outside the boot IRQ-attaching helper",
                        yield_chain, attach_chain
                    )),
                );
            }
        }

        // 2. Check all `extern "C"` and `sys.*` hook callback arguments for yielding, boot-spawning, or boot-attaching functions.
        let cb_sites = self.call_graph.extern_callbacks.clone();
        for site in &cb_sites {
            if let Some(cb_node) = self.call_graph.nodes.get(&site.callback_fn).cloned() {
                let is_sys_hook = site.extern_fn.starts_with("sys.");
                if cb_node.any_yield_effect.can_yield() {
                    let cb_file = self.node_files.get(&site.callback_fn).cloned();
                    let sub_chain = self
                        .call_graph
                        .shortest_any_yield_chain(&site.callback_fn)
                        .join(" -> ");
                    let (msg, note) = if site.extern_fn == "sys.set_putc" {
                        (
                            format!(
                                "cannot pass yielding function '{}' to 'sys.set_putc': console byte-sink callbacks must be non-yielding",
                                site.callback_fn
                            ),
                            format!(
                                "yield chain: sys.set_putc(...) callback -> {}; pass a non-yielding `fn(u8)` callback",
                                sub_chain
                            ),
                        )
                    } else if is_sys_hook {
                        (
                            format!(
                                "cannot pass yielding function '{}' to '{}': system hook callbacks must be non-yielding",
                                site.callback_fn, site.extern_fn
                            ),
                            format!(
                                "yield chain: {}(...) callback -> {}; pass a non-yielding `fn` callback",
                                site.extern_fn, sub_chain
                            ),
                        )
                    } else {
                        (
                            format!(
                                "cannot pass yielding function '{}' as callback to extern \"C\" function '{}': stackless coroutine yields cannot unwind through C stack frames",
                                site.callback_fn, site.extern_fn
                            ),
                            format!(
                                "yield chain: extern \"C\" {}(...) callback -> {}; stackless coroutines cannot suspend across C stack frames—pass a non-yielding `fn` callback",
                                site.extern_fn, sub_chain
                            ),
                        )
                    };
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::YieldingCallbackToExternC,
                            site.span,
                            msg,
                        )
                        .with_file(cb_file)
                        .with_note(note),
                    );
                }
                if cb_node.spawns_in_boot {
                    let cb_file = self.node_files.get(&site.callback_fn).cloned();
                    let target_desc = if is_sys_hook {
                        format!("'{}'", site.extern_fn)
                    } else {
                        format!("extern \"C\" function '{}'", site.extern_fn)
                    };
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::InvalidSpawnContext,
                            site.span,
                            format!(
                                "cannot pass boot-spawning function '{}' as callback to {}",
                                site.callback_fn, target_desc
                            ),
                        )
                        .with_file(cb_file),
                    );
                }
                if cb_node.attaches_in_boot {
                    let cb_file = self.node_files.get(&site.callback_fn).cloned();
                    let target_desc = if is_sys_hook {
                        format!("'{}'", site.extern_fn)
                    } else {
                        format!("extern \"C\" function '{}'", site.extern_fn)
                    };
                    self.emit(
                        Diagnostic::new(
                            DiagnosticKind::IrqAttachmentError,
                            site.span,
                            format!(
                                "cannot pass IRQ-attaching function '{}' as callback to {}",
                                site.callback_fn, target_desc
                            ),
                        )
                        .with_file(cb_file),
                    );
                }
            }
        }

        // 3. Detect recursive call cycles across boot-spawning (`E0027`), boot-attaching (`E0032`), and yielding (`E0012`) functions.
        let spawning_cycles = self.call_graph.find_spawning_cycles();
        for (cycle, span) in &spawning_cycles {
            let cycle_file = cycle
                .first()
                .and_then(|f| self.node_files.get(f))
                .cloned();
            let cycle_str = cycle.join(" -> ");
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::InvalidSpawnContext,
                    *span,
                    format!(
                        "recursive call cycle across boot-spawning function: {}",
                        cycle_str
                    ),
                )
                .with_file(cycle_file)
                .with_note(
                    "boot-tree `spawn` functions must form a static DAG so task slots are statically bounded at compile time",
                ),
            );
        }
        let attaching_cycles = self.call_graph.find_attaching_cycles();
        for (cycle, span) in &attaching_cycles {
            let cycle_file = cycle
                .first()
                .and_then(|f| self.node_files.get(f))
                .cloned();
            let cycle_str = cycle.join(" -> ");
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::IrqAttachmentError,
                    *span,
                    format!(
                        "recursive call cycle across IRQ-attaching function: {}",
                        cycle_str
                    ),
                )
                .with_file(cycle_file)
                .with_note(
                    "boot-tree `irq.attach(...)` functions must form a static acyclic boot tree",
                ),
            );
        }

        for (cycle, span) in self.call_graph.find_yielding_cycles() {
            let cycle_file = cycle
                .first()
                .and_then(|f| self.node_files.get(f))
                .cloned();
            let cycle_str = cycle.join(" -> ");
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::YieldingRecursion,
                    span,
                    format!("recursive yielding call cycle detected: {}", cycle_str),
                )
                .with_file(cycle_file)
                .with_note(format!(
                    "100% static coroutine frame allocation requires an acyclic call graph across yielding functions; rewrite `{}` as an iterative `loop {{ ... }}` or `while` state machine",
                    cycle_str
                )),
            );
        }

        // 4. Verify interprocedural reference provenance for `spawn` (`E0010`) and ISR `state`
        //    escapes into OS primitives (`E0033`), then unroll `main()`'s static boot tree into
        //    `self.spawned_tasks` and `self.irq_bindings` (`E0032`), and verify `byte_ring` SPSC (`E0002`).
        self.verify_boot_spawn_arg_provenance();
        self.verify_isr_state_escape_provenance();
        self.unroll_boot_spawn_tree();
        self.unroll_irq_boot_tree();
        self.verify_byte_ring_spsc();

        // 5. Compute 3-tier `SoftYielding` coroutine vs `noyield` dual specializations.
        let (soft_coro, soft_ny, dual_spec) = self.call_graph.compute_soft_specializations();
        self.soft_yielding_in_coro = soft_coro;
        self.soft_yielding_in_noyield = soft_ny;
        self.dual_specialized_fns = dual_spec;

        // 6. Synchronize resolved `effect`, `can_yield`, `spawns_in_boot`, and callee lists into `FunctionInfo`, `TaskInfo`, and `IsrInfo`.
        let fn_names: Vec<String> = self.functions.keys().cloned().collect();
        for name in fn_names {
            let callees = self.call_graph.callees_of(&name);
            let yielding_callees = self.call_graph.yielding_callees_of(&name);
            if let (Some(info), Some(node)) = (
                self.functions.get_mut(&name),
                self.call_graph.nodes.get(&name),
            ) {
                info.direct_yield = node.direct_yield;
                info.can_yield = node.can_yield;
                info.effect = node.effect;
                info.spawns_in_boot = node.spawns_in_boot;
                info.yield_chain = node.yield_chain.clone();
                info.callees = callees;
                info.yielding_callees = yielding_callees;
            }
        }

        let task_names: Vec<String> = self.tasks.keys().cloned().collect();
        for name in task_names {
            let key = format!("task::{}", name);
            let callees = self.call_graph.callees_of(&key);
            let yielding_callees = self.call_graph.yielding_callees_of(&key);
            if let (Some(info), Some(node)) =
                (self.tasks.get_mut(&name), self.call_graph.nodes.get(&key))
            {
                info.direct_yield = node.direct_yield;
                info.can_yield = node.can_yield;
                info.effect = node.effect;
                info.callees = callees;
                info.yielding_callees = yielding_callees;
            }
        }

        let isr_names: Vec<String> = self.isrs.keys().cloned().collect();
        for name in isr_names {
            let key = format!("isr::{}", name);
            let callees = self.call_graph.callees_of(&key);
            if let Some(info) = self.isrs.get_mut(&name) {
                info.callees = callees;
            }
        }
    }

    pub(super) fn propagate_escaping_ref_params(
        escaping: &mut BTreeMap<String, BTreeSet<usize>>,
        recorded_call_args: &[RecordedCallArgs],
    ) {
        loop {
            let mut changed = false;
            for call in recorded_call_args {
                let Some(callee_params) = escaping.get(&call.callee).cloned() else {
                    continue;
                };
                for p_idx in callee_params {
                    if let Some(arg) = call.args.get(p_idx) {
                        for &caller_p in &arg.ref_param_origins {
                            if escaping
                                .entry(call.caller.clone())
                                .or_default()
                                .insert(caller_p)
                            {
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    pub(super) fn verify_boot_spawn_arg_provenance(&mut self) {
        let mut spawning_ref_params: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
        for (fn_name, steps) in &self.fn_boot_steps {
            for step in steps {
                if let BootStep::DirectSpawn { arg_bindings, .. } = step {
                    for b in arg_bindings {
                        if !b.ref_param_origins.is_empty() {
                            spawning_ref_params
                                .entry(fn_name.clone())
                                .or_default()
                                .extend(b.ref_param_origins.iter().copied());
                        }
                    }
                }
            }
        }

        Self::propagate_escaping_ref_params(&mut spawning_ref_params, &self.recorded_call_args);

        let calls = self.recorded_call_args.clone();
        for call in &calls {
            let Some(callee_params) = spawning_ref_params.get(&call.callee) else {
                continue;
            };
            let caller_file = self.node_files.get(&call.caller).cloned();
            for &p_idx in callee_params {
                if let Some(arg) = call.args.get(p_idx) {
                    if arg.has_non_static_local_ref {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::SpawnNonStaticReference,
                                arg.span,
                                format!(
                                    "cannot pass non-static reference/slice into boot helper '{}' (argument {}): parameter is forwarded into 'spawn' and outlives local stack variables",
                                    call.callee,
                                    p_idx + 1
                                ),
                            )
                            .with_file(caller_file.clone()),
                        );
                    }
                }
            }
        }
    }

    pub(super) fn verify_isr_state_escape_provenance(&mut self) {
        let mut escaping = self.prim_escaping_ref_params.clone();
        Self::propagate_escaping_ref_params(&mut escaping, &self.recorded_call_args);

        let calls = self.recorded_call_args.clone();
        for call in &calls {
            let Some(callee_params) = escaping.get(&call.callee) else {
                continue;
            };
            let caller_file = self.node_files.get(&call.caller).cloned();
            for &p_idx in callee_params {
                if let Some(arg) = call.args.get(p_idx) {
                    if arg.has_isr_state_ref {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::InvalidIsrState,
                                arg.span,
                                format!(
                                    "cannot pass `&` of ISR `state` variable into helper '{}' (argument {}): parameter escapes into an OS primitive",
                                    call.callee,
                                    p_idx + 1
                                ),
                            )
                            .with_file(caller_file.clone()),
                        );
                    }
                }
            }
        }
    }

    pub(super) fn fn_boot_flags(&self, fn_name: &str) -> (bool, bool) {
        let mut visiting = BTreeSet::new();
        let mut stack = vec![fn_name.to_string()];
        visiting.insert(fn_name.to_string());
        let mut spawns = false;
        let mut attaches = false;
        while let Some(cur) = stack.pop() {
            if let Some(node) = self.call_graph.nodes.get(&cur) {
                spawns |= node.spawns_in_boot;
                attaches |= node.attaches_in_boot;
            }
            for e in &self.call_graph.edges {
                if e.caller == cur && visiting.insert(e.callee.clone()) {
                    stack.push(e.callee.clone());
                }
            }
        }
        (spawns, attaches)
    }

    pub(super) fn resolve_field_chain_map(
        fields: &BTreeMap<String, String>,
        chain: &str,
    ) -> Option<String> {
        if let Some(exact) = fields.get(chain) {
            return Some(exact.clone());
        }
        let mut prefix_len = 0;
        let mut best: Option<(usize, String)> = None;
        for (k, v) in fields {
            if let Some(rest) = chain.strip_prefix(k).and_then(|r| r.strip_prefix('.')) {
                if k.len() > prefix_len {
                    prefix_len = k.len();
                    best = Some((k.len(), format!("{v}.{rest}")));
                }
            }
        }
        best.map(|(_, s)| s)
    }

    pub(super) fn resolve_endpoint_spec_with_bindings(
        spec: &EndpointSpec,
        param_endpoints: &[Option<String>],
        param_field_endpoints: &[BTreeMap<String, String>],
    ) -> Option<String> {
        match spec {
            EndpointSpec::Static(s) => Some(s.clone()),
            EndpointSpec::Param { index, suffix } => match suffix {
                Some(suf) => param_field_endpoints
                    .get(*index)
                    .and_then(|m| Self::resolve_field_chain_map(m, suf))
                    .or_else(|| {
                        param_endpoints
                            .get(*index)
                            .cloned()
                            .flatten()
                            .map(|base| format!("{base}.{suf}"))
                    }),
                None => param_endpoints.get(*index).cloned().flatten(),
            },
        }
    }

    pub(super) fn resolve_boot_arg_endpoint_and_fields(
        static_endpoint: &Option<String>,
        endpoint_from_param: &Option<(usize, Option<String>)>,
        field_endpoints: &BTreeMap<String, EndpointSpec>,
        param_endpoints: &[Option<String>],
        param_field_endpoints: &[BTreeMap<String, String>],
    ) -> (Option<String>, BTreeMap<String, String>) {
        let ep = static_endpoint.clone().or_else(|| {
            endpoint_from_param.as_ref().and_then(|(idx, suffix)| {
                Self::resolve_endpoint_spec_with_bindings(
                    &EndpointSpec::Param {
                        index: *idx,
                        suffix: suffix.clone(),
                    },
                    param_endpoints,
                    param_field_endpoints,
                )
            })
        });
        let mut f_map = BTreeMap::new();
        if let Some((idx, suffix)) = endpoint_from_param {
            if let Some(src_fields) = param_field_endpoints.get(*idx) {
                match suffix {
                    None => {
                        for (k, v) in src_fields {
                            f_map.insert(k.clone(), v.clone());
                        }
                    }
                    Some(suf) => {
                        let prefix = format!("{suf}.");
                        for (k, v) in src_fields {
                            if let Some(rest) = k.strip_prefix(&prefix) {
                                f_map.insert(rest.to_string(), v.clone());
                            }
                        }
                    }
                }
            }
        }
        for (fname, fspec) in field_endpoints {
            if let Some(resolved_f) = Self::resolve_endpoint_spec_with_bindings(
                fspec,
                param_endpoints,
                param_field_endpoints,
            ) {
                f_map.insert(fname.clone(), resolved_f);
            }
        }
        (ep, f_map)
    }

    pub(super) fn verify_byte_ring_spsc(&mut self) {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        enum ActorId {
            Task(usize),
            Isr(usize),
            MainPostBoot,
        }

        #[derive(Clone, Debug)]
        struct ActorAccess {
            actor_id: ActorId,
            actor_desc: String,
            method: String,
            span: Span,
            file: Option<String>,
        }

        let mut ops_by_caller: BTreeMap<String, Vec<RecordedByteRingOp>> = BTreeMap::new();
        for op in &self.recorded_byte_ring_ops {
            ops_by_caller
                .entry(op.caller.clone())
                .or_default()
                .push(op.clone());
        }
        let mut calls_by_caller: BTreeMap<String, Vec<RecordedCallArgs>> = BTreeMap::new();
        for call in &self.recorded_call_args {
            calls_by_caller
                .entry(call.caller.clone())
                .or_default()
                .push(call.clone());
        }

        type RingActorsMap = BTreeMap<String, Vec<ActorAccess>>;
        let mut producers_by_ring: RingActorsMap = BTreeMap::new();
        let mut consumers_by_ring: RingActorsMap = BTreeMap::new();

        let record_access =
            |map: &mut RingActorsMap, ring: String, access: ActorAccess| {
                let list = map.entry(ring).or_default();
                if !list.iter().any(|a| a.actor_id == access.actor_id) {
                    list.push(access);
                }
            };

        let mut actors: Vec<(
            ActorId,
            String,
            String,
            Vec<Option<String>>,
            Vec<BTreeMap<String, String>>,
            bool,
        )> = Vec::new();

        for inst in &self.spawned_tasks {
            let caller_key = format!("task::{}", inst.task_name);
            let field_eps = self
                .spawned_task_field_endpoints
                .get(&inst.instance_id)
                .cloned()
                .unwrap_or_else(|| vec![BTreeMap::new(); inst.endpoint_bindings.len()]);
            actors.push((
                ActorId::Task(inst.instance_id),
                format!("task '{}' (spawned via {})", inst.task_name, inst.format_provenance()),
                caller_key,
                inst.endpoint_bindings.clone(),
                field_eps,
                false,
            ));
        }

        for (idx, binding) in self.irq_bindings.iter().enumerate() {
            let caller_key = format!("isr::{}", binding.isr_name);
            let mut param_eps = Vec::with_capacity(binding.args.len());
            for arg in &binding.args {
                let ep = match arg {
                    IrqBindingArg::PrimAddr(name) => Some(name.clone()),
                    IrqBindingArg::SubsystemPath {
                        root, field_chain, ..
                    } => {
                        if field_chain.is_empty() {
                            Some(root.clone())
                        } else {
                            let chain = field_chain
                                .iter()
                                .map(|(f, _)| f.as_str())
                                .collect::<Vec<_>>()
                                .join(".");
                            Some(format!("{root}.{chain}"))
                        }
                    }
                    _ => None,
                };
                param_eps.push(ep);
            }
            let field_eps = vec![BTreeMap::new(); param_eps.len()];
            actors.push((
                ActorId::Isr(idx),
                format!("ISR '{}' (IRQ '{}')", binding.isr_name, binding.irq_name),
                caller_key,
                param_eps,
                field_eps,
                false,
            ));
        }

        let has_main_post_boot = ops_by_caller.contains_key("main::post_boot")
            || self
                .recorded_call_args
                .iter()
                .any(|c| c.caller == "main" && c.in_main_post_boot);
        if has_main_post_boot {
            actors.push((
                ActorId::MainPostBoot,
                "main() (post-boot)".to_string(),
                "main".to_string(),
                Vec::new(),
                Vec::new(),
                true,
            ));
        }

        for (actor_id, actor_desc, root_caller, root_params, root_fields, is_main_post) in actors {
            let mut visited: BTreeSet<(
                String,
                Vec<Option<String>>,
                Vec<BTreeMap<String, String>>,
            )> = BTreeSet::new();
            let mut worklist = vec![(root_caller.clone(), root_params, root_fields, is_main_post)];
            visited.insert((worklist[0].0.clone(), worklist[0].1.clone(), worklist[0].2.clone()));

            while let Some((caller, param_eps, param_fields, post_boot_only)) = worklist.pop() {
                let op_lookup_key = if post_boot_only {
                    "main::post_boot"
                } else {
                    caller.as_str()
                };
                if let Some(ops) = ops_by_caller.get(op_lookup_key) {
                    for op in ops {
                        let Some(concrete_ring) = Self::resolve_endpoint_spec_with_bindings(
                            &op.target,
                            &param_eps,
                            &param_fields,
                        ) else {
                            continue;
                        };
                        if concrete_ring == "<anon_endpoint>" {
                            continue;
                        }
                        let access = ActorAccess {
                            actor_id: actor_id.clone(),
                            actor_desc: actor_desc.clone(),
                            method: op.method.clone(),
                            span: op.span,
                            file: op.file.clone(),
                        };
                        match op.role {
                            ByteRingOpRole::Producer => {
                                record_access(&mut producers_by_ring, concrete_ring, access);
                            }
                            ByteRingOpRole::Consumer => {
                                record_access(&mut consumers_by_ring, concrete_ring, access);
                            }
                            ByteRingOpRole::Both => {
                                record_access(
                                    &mut producers_by_ring,
                                    concrete_ring.clone(),
                                    access.clone(),
                                );
                                record_access(&mut consumers_by_ring, concrete_ring, access);
                            }
                        }
                    }
                }

                if visited.len() >= 4096 {
                    continue;
                }
                if let Some(calls) = calls_by_caller.get(&caller) {
                    for call in calls {
                        if post_boot_only && !call.in_main_post_boot {
                            continue;
                        }
                        let mut child_eps = Vec::with_capacity(call.args.len());
                        let mut child_fields = Vec::with_capacity(call.args.len());
                        for arg in &call.args {
                            let (ep, f_map) = Self::resolve_boot_arg_endpoint_and_fields(
                                &arg.static_endpoint,
                                &arg.endpoint_from_param,
                                &arg.field_endpoints,
                                &param_eps,
                                &param_fields,
                            );
                            child_eps.push(ep);
                            child_fields.push(f_map);
                        }
                        let state = (call.callee.clone(), child_eps.clone(), child_fields.clone());
                        if visited.insert(state) {
                            worklist.push((call.callee.clone(), child_eps, child_fields, false));
                        }
                    }
                }
            }
        }

        let mut all_rings: BTreeSet<String> = producers_by_ring.keys().cloned().collect();
        all_rings.extend(consumers_by_ring.keys().cloned());

        for ring in all_rings {
            let disp_ring = self
                .display_names
                .get(&ring)
                .cloned()
                .unwrap_or_else(|| ring.clone());
            if let Some(prods) = producers_by_ring.get(&ring) {
                if prods.len() > 1 {
                    let first = &prods[0];
                    for extra in &prods[1..] {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::GlobalIsolationViolation,
                                extra.span,
                                format!(
                                    "byte_ring '{}' SPSC violation: multiple concurrent producers ('{}' and '{}' both call `{}`)",
                                    disp_ring, first.actor_desc, extra.actor_desc, extra.method
                                ),
                            )
                            .with_file(extra.file.clone())
                            .with_note(format!(
                                "first producer '{}' called `{}` at {}; `byte_ring` is strictly single-producer single-consumer (SPSC) — use a separate `byte_ring` per producer or an `inbox<T, N>` for MPSC",
                                first.actor_desc, first.method, first.span
                            )),
                        );
                    }
                }
            }
            if let Some(cons) = consumers_by_ring.get(&ring) {
                if cons.len() > 1 {
                    let first = &cons[0];
                    for extra in &cons[1..] {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::GlobalIsolationViolation,
                                extra.span,
                                format!(
                                    "byte_ring '{}' SPSC violation: multiple concurrent consumers ('{}' and '{}' both call `{}`)",
                                    disp_ring, first.actor_desc, extra.actor_desc, extra.method
                                ),
                            )
                            .with_file(extra.file.clone())
                            .with_note(format!(
                                "first consumer '{}' called `{}` at {}; `byte_ring` is strictly single-producer single-consumer (SPSC) — use a separate `byte_ring` per consumer",
                                first.actor_desc, first.method, first.span
                            )),
                        );
                    }
                }
            }
        }
    }

    pub(super) fn unroll_boot_spawn_tree(&mut self) {
        self.spawned_tasks.clear();
        for t in self.tasks.values_mut() {
            t.instance_count = 0;
        }

        let has_main = self.functions.contains_key("main");
        let roots: Vec<String> = if has_main {
            vec!["main".to_string()]
        } else {
            let mut called_by_fn = BTreeSet::new();
            for e in &self.call_graph.edges {
                if e.caller_kind == CallableKind::Fn {
                    called_by_fn.insert(e.callee.clone());
                }
            }
            self.fn_decl_order
                .iter()
                .filter(|name| {
                    !called_by_fn.contains(*name)
                        && self
                            .call_graph
                            .nodes
                            .get(*name)
                            .is_some_and(|n| n.spawns_in_boot)
                })
                .cloned()
                .collect()
        };

        for root in roots {
            let mut path = vec![root.clone()];
            let mut visiting = BTreeSet::new();
            visiting.insert(root.clone());
            self.unroll_boot_fn(&root, &mut path, false, &[], &[], &mut visiting);
        }
    }

    pub(super) fn unroll_irq_boot_tree(&mut self) {
        let mut called_by_fn = BTreeSet::new();
        for e in &self.call_graph.edges {
            if e.caller_kind == CallableKind::Fn {
                called_by_fn.insert(e.callee.clone());
            }
        }
        let has_main = self.functions.contains_key("main");
        let roots: Vec<String> = if has_main {
            vec!["main".to_string()]
        } else {
            self.fn_decl_order
                .iter()
                .filter(|name| {
                    !called_by_fn.contains(*name)
                        && self
                            .call_graph
                            .nodes
                            .get(*name)
                            .is_some_and(|n| n.attaches_in_boot)
                })
                .cloned()
                .collect()
        };

        let mut irq_mode_seen: BTreeMap<u32, (bool, String, Span)> = BTreeMap::new();
        let mut mixed_warned: BTreeSet<u32> = BTreeSet::new();
        let mut next_branch_id: usize = 0;
        let mut reached_fns: BTreeSet<String> = BTreeSet::new();

        for root in roots {
            let Some(nodes) = self.fn_irq_boot_nodes.get(&root).cloned() else {
                continue;
            };
            let mut visiting = BTreeSet::new();
            visiting.insert(root.clone());
            reached_fns.insert(root.clone());
            let initial_excl = BTreeSet::new();
            let _ = self.unroll_irq_nodes(
                &nodes,
                &[],
                None,
                &initial_excl,
                &mut visiting,
                &mut reached_fns,
                &mut irq_mode_seen,
                &mut mixed_warned,
                &mut next_branch_id,
            );
        }

        // Also validate any uncalled helper functions that directly/transitively call `attach`
        // (for instance `fn bad_helper(u32 x) { IRQ.attach(h(x)); } fn main() {}`).
        if has_main {
            let unreached: Vec<String> = self
                .fn_decl_order
                .iter()
                .filter(|name| {
                    !reached_fns.contains(*name)
                        && !called_by_fn.contains(*name)
                        && self
                            .call_graph
                            .nodes
                            .get(*name)
                            .is_some_and(|n| n.attaches_in_boot)
                })
                .cloned()
                .collect();
            for fname in unreached {
                if let Some(nodes) = self.fn_irq_boot_nodes.get(&fname).cloned() {
                    let mut visiting = BTreeSet::new();
                    visiting.insert(fname.clone());
                    let initial_excl = BTreeSet::new();
                    let _ = self.unroll_irq_nodes(
                        &nodes,
                        &[],
                        None,
                        &initial_excl,
                        &mut visiting,
                        &mut reached_fns,
                        &mut irq_mode_seen,
                        &mut mixed_warned,
                        &mut next_branch_id,
                    );
                }
            }
        }
    }

    pub(super) fn irq_nodes_contain_attach(&self, nodes: &[IrqBootNode]) -> bool {
        for n in nodes {
            match n {
                IrqBootNode::Attach(_) => return true,
                IrqBootNode::Call { callee, .. } => {
                    if self
                        .call_graph
                        .nodes
                        .get(callee)
                        .is_some_and(|node| node.attaches_in_boot)
                    {
                        return true;
                    }
                }
                IrqBootNode::BranchGroup { arms } => {
                    if arms
                        .iter()
                        .any(|(_, sub)| self.irq_nodes_contain_attach(sub))
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn unroll_irq_nodes(
        &mut self,
        nodes: &[IrqBootNode],
        param_args: &[Option<IrqBindingArg>],
        cond_branch: Option<usize>,
        active_exclusive: &BTreeSet<u32>,
        visiting: &mut BTreeSet<String>,
        reached_fns: &mut BTreeSet<String>,
        irq_mode_seen: &mut BTreeMap<u32, (bool, String, Span)>,
        mixed_warned: &mut BTreeSet<u32>,
        next_branch_id: &mut usize,
    ) -> (bool, BTreeSet<u32>, BTreeSet<u32>) {
        let mut cur_excl = active_exclusive.clone();
        let mut ret_union = BTreeSet::new();
        for node in nodes {
            match node {
                IrqBootNode::Attach(att) => {
                    let mut resolved_args = Vec::with_capacity(att.args.len());
                    let mut args_ok = true;
                    for (spec_opt, arg_span, pname) in &att.args {
                        match spec_opt {
                            Some(BootAttachArgSpec::Static(arg)) => {
                                resolved_args.push(arg.clone());
                            }
                            Some(BootAttachArgSpec::Param { index, field_chain }) => {
                                if let Some(Some(forwarded)) = param_args.get(*index) {
                                    if field_chain.is_empty() {
                                        resolved_args.push(forwarded.clone());
                                    } else if let IrqBindingArg::SubsystemPath {
                                        root,
                                        root_struct,
                                        field_chain: mut base_chain,
                                    } = forwarded.clone()
                                    {
                                        base_chain.extend(field_chain.iter().cloned());
                                        resolved_args.push(IrqBindingArg::SubsystemPath {
                                            root,
                                            root_struct,
                                            field_chain: base_chain,
                                        });
                                    } else {
                                        self.emit(
                                            Diagnostic::new(
                                                DiagnosticKind::IrqAttachmentError,
                                                *arg_span,
                                                format!(
                                                    "parameter '{}' forwarded into '{}.attach({}(...))' in '{}' must resolve to a static subsystem `&inst`",
                                                    pname, att.irq_name, att.isr_name, att.in_caller
                                                ),
                                            )
                                            .with_file(att.file.clone()),
                                        );
                                        args_ok = false;
                                    }
                                } else {
                                    self.emit(
                                        Diagnostic::new(
                                            DiagnosticKind::IrqAttachmentError,
                                            *arg_span,
                                            format!(
                                                "parameter '{}' forwarded into '{}.attach({}(...))' in '{}' must be bound to a static primitive `&prim`, MMIO handle, or compile-time constant at every boot call site",
                                                pname, att.irq_name, att.isr_name, att.in_caller
                                            ),
                                        )
                                        .with_file(att.file.clone()),
                                    );
                                    args_ok = false;
                                }
                            }
                            None => {
                                args_ok = false;
                            }
                        }
                    }

                    if !att.invalid_ctx {
                        if let Some((prev_shared, prev_name, prev_span)) =
                            irq_mode_seen.get(&att.irq_num).cloned()
                        {
                            if prev_shared != att.is_shared && mixed_warned.insert(att.irq_num) {
                                self.emit(
                                    Diagnostic::new(
                                        DiagnosticKind::IrqAttachmentError,
                                        att.span,
                                        format!(
                                            "IRQ '{}' (#{}) cannot mix exclusive `.attach(...)` and shared `.attach_shared(...)`",
                                            att.irq_name, att.irq_num
                                        ),
                                    )
                                    .with_file(att.file.clone())
                                    .with_note(format!(
                                        "previously bound via `{}.{}(...)` at {}",
                                        prev_name,
                                        if prev_shared { "attach_shared" } else { "attach" },
                                        prev_span
                                    )),
                                );
                            }
                        } else {
                            irq_mode_seen.insert(
                                att.irq_num,
                                (att.is_shared, att.irq_name.clone(), att.span),
                            );
                        }

                        if !att.is_shared && !cur_excl.insert(att.irq_num) {
                            self.emit(
                                Diagnostic::new(
                                    DiagnosticKind::IrqAttachmentError,
                                    att.span,
                                    format!(
                                        "IRQ '{}' (#{}) already has an exclusive `.attach(...)` handler on this boot path",
                                        att.irq_name, att.irq_num
                                    ),
                                )
                                .with_file(att.file.clone())
                                .with_note(
                                    "use `.attach_shared(...)` for multi-handler interrupts or guard mutually exclusive variants in `if`/`else`/`switch` branches",
                                ),
                            );
                        } else if args_ok {
                            self.irq_bindings.push(IrqBindingInfo {
                                irq_name: att.irq_name.clone(),
                                irq_num: att.irq_num,
                                is_shared: att.is_shared,
                                isr_name: att.isr_name.clone(),
                                args: resolved_args,
                                conditional_branch: cond_branch,
                                span: att.span,
                            });
                        }
                    }
                }
                IrqBootNode::Call {
                    callee,
                    args,
                } => {
                    let attaches = self
                        .call_graph
                        .nodes
                        .get(callee)
                        .is_some_and(|n| n.attaches_in_boot);
                    if !attaches || !visiting.insert(callee.clone()) {
                        continue;
                    }
                    reached_fns.insert(callee.clone());
                    let child_params: Vec<Option<IrqBindingArg>> = args
                        .iter()
                        .map(|a| match &a.spec {
                            Some(BootAttachArgSpec::Static(s)) => Some(s.clone()),
                            Some(BootAttachArgSpec::Param { index, field_chain }) => {
                                let base = param_args.get(*index).cloned().flatten()?;
                                if field_chain.is_empty() {
                                    Some(base)
                                } else if let IrqBindingArg::SubsystemPath {
                                    root,
                                    root_struct,
                                    field_chain: mut base_chain,
                                } = base
                                {
                                    base_chain.extend(field_chain.iter().cloned());
                                    Some(IrqBindingArg::SubsystemPath {
                                        root,
                                        root_struct,
                                        field_chain: base_chain,
                                    })
                                } else {
                                    None
                                }
                            }
                            None => None,
                        })
                        .collect();
                    if let Some(child_nodes) = self.fn_irq_boot_nodes.get(callee).cloned() {
                        let (_child_div, mut child_fall, child_ret) = self.unroll_irq_nodes(
                            &child_nodes,
                            &child_params,
                            cond_branch,
                            &cur_excl,
                            visiting,
                            reached_fns,
                            irq_mode_seen,
                            mixed_warned,
                            next_branch_id,
                        );
                        // Returning inside a helper exits only that helper; execution resumes
                        // in this caller with the union of IRQs attached on all returning paths.
                        child_fall.extend(child_ret);
                        cur_excl = child_fall;
                        visiting.remove(callee);
                    } else {
                        visiting.remove(callee);
                    }
                }
                IrqBootNode::BranchGroup { arms } => {
                    let group_has_attach = arms
                        .iter()
                        .any(|(_, sub)| self.irq_nodes_contain_attach(sub));
                    if !group_has_attach {
                        continue;
                    }
                    let mut post_union = cur_excl.clone();
                    let mut all_arms_div = true;
                    let mut any_non_div = false;
                    for (arm_div, arm_nodes) in arms {
                        let arm_branch = if cond_branch.is_some() {
                            cond_branch
                        } else {
                            let id = *next_branch_id;
                            *next_branch_id += 1;
                            Some(id)
                        };
                        let (sub_div, arm_fall, arm_ret) = self.unroll_irq_nodes(
                            arm_nodes,
                            param_args,
                            arm_branch,
                            &cur_excl,
                            visiting,
                            reached_fns,
                            irq_mode_seen,
                            mixed_warned,
                            next_branch_id,
                        );
                        ret_union.extend(arm_ret);
                        if *arm_div || sub_div {
                            ret_union.extend(arm_fall);
                        } else {
                            all_arms_div = false;
                            if !any_non_div {
                                post_union = arm_fall;
                                any_non_div = true;
                            } else {
                                post_union.extend(arm_fall);
                            }
                        }
                    }
                    if all_arms_div {
                        return (true, BTreeSet::new(), ret_union);
                    }
                    cur_excl = post_union;
                }
            }
        }
        (false, cur_excl, ret_union)
    }

    pub(super) fn unroll_boot_fn(
        &mut self,
        fn_name: &str,
        path: &mut Vec<String>,
        cond_active: bool,
        param_endpoints: &[Option<String>],
        param_field_endpoints: &[BTreeMap<String, String>],
        visiting: &mut BTreeSet<String>,
    ) {
        let Some(steps) = self.fn_boot_steps.get(fn_name).cloned() else {
            return;
        };
        for step in steps {
            match step {
                BootStep::DirectSpawn {
                    task_name,
                    priority,
                    core,
                    conditional,
                    arg_bindings,
                    span,
                } => {
                    let mut endpoint_bindings = Vec::with_capacity(arg_bindings.len());
                    let mut resolved_arg_fields = Vec::with_capacity(arg_bindings.len());
                    for b in &arg_bindings {
                        let (resolved, f_map) = Self::resolve_boot_arg_endpoint_and_fields(
                            &b.static_endpoint,
                            &b.endpoint_from_param,
                            &b.field_endpoints,
                            param_endpoints,
                            param_field_endpoints,
                        );
                        endpoint_bindings.push(resolved);
                        resolved_arg_fields.push(f_map);
                    }
                    let mut provenance_path = path.clone();
                    provenance_path.push(task_name.clone());
                    let instance_id = self.spawned_tasks.len();
                    if resolved_arg_fields.iter().any(|m| !m.is_empty()) {
                        self.spawned_task_field_endpoints
                            .insert(instance_id, resolved_arg_fields);
                    }
                    self.spawned_tasks.push(SpawnedTaskInstance {
                        instance_id,
                        task_name: task_name.clone(),
                        priority,
                        core,
                        spawned_in: fn_name.to_string(),
                        provenance_path,
                        conditional: cond_active || conditional,
                        endpoint_bindings,
                        span,
                    });
                    if let Some(t) = self.tasks.get_mut(&task_name) {
                        t.instance_count += 1;
                    }
                }
                BootStep::HelperCall {
                    callee,
                    conditional,
                    args,
                } => {
                    let spawns = self
                        .call_graph
                        .nodes
                        .get(&callee)
                        .is_some_and(|n| n.spawns_in_boot);
                    if !spawns || !visiting.insert(callee.clone()) {
                        continue;
                    }
                    let mut child_endpoints = Vec::with_capacity(args.len());
                    let mut child_field_endpoints = Vec::with_capacity(args.len());
                    for a in &args {
                        let (resolved, f_map) = Self::resolve_boot_arg_endpoint_and_fields(
                            &a.static_endpoint,
                            &a.endpoint_from_param,
                            &a.field_endpoints,
                            param_endpoints,
                            param_field_endpoints,
                        );
                        child_endpoints.push(resolved);
                        child_field_endpoints.push(f_map);
                    }
                    path.push(callee.clone());
                    self.unroll_boot_fn(
                        &callee,
                        path,
                        cond_active || conditional,
                        &child_endpoints,
                        &child_field_endpoints,
                        visiting,
                    );
                    path.pop();
                    visiting.remove(&callee);
                }
            }
        }
    }

    fn check_task_slot_limit(&mut self, main_decl: Option<&FnDecl>) {
        const MAX_TASKS: usize = 32;
        fn expr_can_yield(expr: &Expr, cg: &CallGraph) -> bool {
            match &expr.kind {
                ExprKind::Call { callee, args, .. } => {
                    if let ExprKind::Ident(id) = &callee.kind {
                        if let Some(node) = cg.nodes.get(&id.name) {
                            if node.can_yield {
                                return true;
                            }
                        }
                        if matches!(id.name.as_str(), "delay" | "sleep" | "checkpoint") {
                            return true;
                        }
                    }
                    expr_can_yield(callee, cg)
                        || args.iter().any(|a| expr_can_yield(&a.value, cg))
                }
                ExprKind::MethodCall {
                    receiver,
                    method,
                    args,
                } => {
                    if matches!(
                        method.name.as_str(),
                        "wait_chunk"
                            | "wait_read"
                            | "recv"
                            | "send"
                            | "wait"
                            | "wait_change"
                            | "alloc"
                            | "commit_wait"
                            | "delay"
                    ) {
                        return true;
                    }
                    expr_can_yield(receiver, cg)
                        || args.iter().any(|a| expr_can_yield(&a.value, cg))
                }
                ExprKind::Unary { expr: inner, .. } | ExprKind::Cast { expr: inner, .. }
                | ExprKind::Field { target: inner, .. } => expr_can_yield(inner, cg),
                ExprKind::Binary { lhs, rhs, .. }
                | ExprKind::Index {
                    target: lhs,
                    index: rhs,
                }
                | ExprKind::Or {
                    expr: lhs,
                    fallback: rhs,
                } => expr_can_yield(lhs, cg) || expr_can_yield(rhs, cg),
                ExprKind::Slice { target, start, end } => {
                    expr_can_yield(target, cg)
                        || start.as_ref().is_some_and(|e| expr_can_yield(e, cg))
                        || end.as_ref().is_some_and(|e| expr_can_yield(e, cg))
                }
                ExprKind::StructLit { fields, .. } => {
                    fields.iter().any(|f| expr_can_yield(&f.value, cg))
                }
                ExprKind::ArrayLit(elems) => elems.iter().any(|e| expr_can_yield(e, cg)),
                ExprKind::FormatString(parts) => parts.iter().any(|p| match p {
                    FormatPart::Hole { expr: e, .. } => expr_can_yield(e, cg),
                    _ => false,
                }),
                ExprKind::Catch { expr: inner, clause } => {
                    expr_can_yield(inner, cg)
                        || clause.body.stmts.iter().any(|s| stmt_can_yield(s, false, cg))
                }
                ExprKind::Literal(_) | ExprKind::Ident(_) | ExprKind::Priority(_) => false,
            }
        }
        fn stmt_has_boot_spawn(stmt: &Stmt, cg: &CallGraph) -> bool {
            match &stmt.kind {
                StmtKind::Spawn(_) => true,
                StmtKind::Expr(e) => expr_has_boot_spawn(e, cg),
                StmtKind::VarDecl { init: Some(e), .. } => expr_has_boot_spawn(e, cg),
                StmtKind::Assign { value: e, .. } => expr_has_boot_spawn(e, cg),
                StmtKind::If {
                    then_block,
                    else_block,
                    ..
                } => {
                    then_block.stmts.iter().any(|s| stmt_has_boot_spawn(s, cg))
                        || else_block
                            .as_ref()
                            .is_some_and(|b| b.stmts.iter().any(|s| stmt_has_boot_spawn(s, cg)))
                }
                StmtKind::Switch { arms, .. } => arms
                    .iter()
                    .any(|a| a.body.stmts.iter().any(|s| stmt_has_boot_spawn(s, cg))),
                StmtKind::Wait { .. } => false,
                StmtKind::Block(b) | StmtKind::NoYield { body: b } | StmtKind::Critical { body: b } => {
                    b.stmts.iter().any(|s| stmt_has_boot_spawn(s, cg))
                }
                _ => false,
            }
        }
        fn expr_has_boot_spawn(expr: &Expr, cg: &CallGraph) -> bool {
            match &expr.kind {
                ExprKind::Call { callee, args, .. } => {
                    let direct = if let ExprKind::Ident(id) = &callee.kind {
                        cg.nodes
                            .get(&id.name)
                            .is_some_and(|n| n.spawns_in_boot || n.attaches_in_boot)
                    } else {
                        false
                    };
                    direct || args.iter().any(|a| expr_has_boot_spawn(&a.value, cg))
                }
                ExprKind::MethodCall { method, .. } => {
                    matches!(method.name.as_str(), "attach" | "attach_shared")
                }
                _ => false,
            }
        }
        fn stmt_can_yield(stmt: &Stmt, manual_yield: bool, cg: &CallGraph) -> bool {
            match &stmt.kind {
                StmtKind::Loop { body } => {
                    !manual_yield
                        || body
                            .stmts
                            .iter()
                            .any(|s| stmt_can_yield(s, manual_yield, cg))
                }
                StmtKind::LoopEvery { .. } => true,
                StmtKind::Expr(e)
                | StmtKind::VarDecl { init: Some(e), .. }
                | StmtKind::Assign { value: e, .. }
                | StmtKind::Return(Some(e)) => expr_can_yield(e, cg),
                StmtKind::If {
                    cond,
                    then_block,
                    else_block,
                } => {
                    expr_can_yield(cond, cg)
                        || then_block
                            .stmts
                            .iter()
                            .any(|s| stmt_can_yield(s, manual_yield, cg))
                        || else_block.as_ref().is_some_and(|b| {
                            b.stmts.iter().any(|s| stmt_can_yield(s, manual_yield, cg))
                        })
                }
                StmtKind::Switch { scrutinee, arms } => {
                    expr_can_yield(scrutinee, cg)
                        || arms.iter().any(|a| {
                            a.body
                                .stmts
                                .iter()
                                .any(|s| stmt_can_yield(s, manual_yield, cg))
                        })
                }
                StmtKind::Wait { .. } => true,
                StmtKind::While { cond, body } => {
                    let is_true = matches!(&cond.kind, ExprKind::Literal(Literal::Bool(true)));
                    (is_true && !manual_yield)
                        || expr_can_yield(cond, cg)
                        || body
                            .stmts
                            .iter()
                            .any(|s| stmt_can_yield(s, manual_yield, cg))
                }
                StmtKind::ForRange {
                    start, end, body, ..
                } => {
                    expr_can_yield(start, cg)
                        || expr_can_yield(end, cg)
                        || body
                            .stmts
                            .iter()
                            .any(|s| stmt_can_yield(s, manual_yield, cg))
                }
                StmtKind::ForEach { iter, body, .. } => {
                    expr_can_yield(iter, cg)
                        || body
                            .stmts
                            .iter()
                            .any(|s| stmt_can_yield(s, manual_yield, cg))
                }
                StmtKind::Critical { body } | StmtKind::Block(body) => {
                    body.stmts.iter().any(|s| stmt_can_yield(s, manual_yield, cg))
                }
                StmtKind::NoYield { .. } => false,
                _ => false,
            }
        }
        let main_becomes_task = main_decl.is_some_and(|md| {
            let Some(body) = &md.body else {
                return false;
            };
            let manual_yield = has_attr_manual_yield(&md.attrs);
            let stmts = &body.stmts;
            let tail = match stmts.iter().rposition(|s| stmt_has_boot_spawn(s, &self.call_graph)) {
                Some(idx) => &stmts[idx + 1..],
                None => stmts.as_slice(),
            };
            tail.iter().any(|s| stmt_can_yield(s, manual_yield, &self.call_graph))
        });
        let total = self.spawned_tasks.len() + usize::from(main_becomes_task);
        if total > MAX_TASKS {
            let (span, msg) = if self.spawned_tasks.len() > MAX_TASKS {
                (
                    self.spawned_tasks[MAX_TASKS].span,
                    format!(
                        "program instantiates {} static task slots (max {} supported by scheduler bitmasks)",
                        total, MAX_TASKS
                    ),
                )
            } else {
                let sp = main_decl.map_or_else(Span::dummy, |m| m.name.span);
                (
                    sp,
                    format!(
                        "program instantiates {} tasks plus the 'main()' supervisor task ({} total, max {} supported by scheduler bitmasks)",
                        self.spawned_tasks.len(),
                        total,
                        MAX_TASKS
                    ),
                )
            };
            self.emit(Diagnostic::new(DiagnosticKind::InvalidSpawnContext, span, msg));
        }
    }
}


impl Analyzer {
    pub(super) fn validate_pool_slot_elem(&mut self, elem: &Ty, span: Span, owner_desc: &str) {
        if elem.is_error() || elem.is_void() {
            return;
        }
        if ty_contains_linear_lease(elem) {
            self.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!("{} cannot wrap linear 'Lease<...>' inside its payload element type '{}'", owner_desc, elem),
            ));
            return;
        }
        if let Some(mmio) = ty_contains_byval_mmio(elem) {
            self.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!("{} cannot store MMIO register layout '{}' by value", owner_desc, mmio),
            ));
            return;
        }
        if ty_embeds_os_primitive(elem) || ty_contains_byval_subsys(elem, &self.structs).is_some() {
            self.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!("{} cannot embed OS primitives or subsystem structs inside element type '{}'", owner_desc, elem),
            ));
            return;
        }
        if elem.contains_reference(&self.structs) || elem.contains_fn_ptr(&self.structs) {
            self.emit(Diagnostic::new(
                DiagnosticKind::LinearLeaseViolation,
                span,
                format!("{} requires a plain non-reference value type (no '*T', '[]T', or 'fn(...)' fields in '{}')", owner_desc, elem),
            ));
        }
    }

    pub(super) fn check_disallowed_lease_type(
        &mut self,
        ty: &Ty,
        span: Span,
        ctx: &str,
        allow_top_level_lease: bool,
    ) {
        match ty {
            Ty::Lease(elem) => {
                if !allow_top_level_lease {
                    self.emit(Diagnostic::new(
                        DiagnosticKind::LinearLeaseViolation,
                        span,
                        format!(
                            "linear type '{}' is not allowed in {} (only permitted as locals, `fn` params/returns, `inbox<Lease<T>, N>`, `*inbox<Lease<T>>`, and managed via `dma_slot<T>`)",
                            ty, ctx
                        ),
                    ));
                } else {
                    self.validate_pool_slot_elem(elem, span, "Lease<T>");
                }
            }
            Ty::OsPrimitive(
                ResolvedOsPrimitive::Inbox { elem, .. }
                | ResolvedOsPrimitive::InboxUnsized { elem },
            ) => {
                if let Ty::Lease(inner) = &**elem {
                    self.validate_pool_slot_elem(inner, span, "Lease<T>");
                } else {
                    self.check_disallowed_lease_type(elem, span, ctx, false);
                }
            }
            Ty::OsPrimitive(os) => {
                if let Some(elem) = os.elem_ty() {
                    self.check_disallowed_lease_type(elem, span, ctx, false);
                }
            }
            Ty::Ptr(inner) => match &**inner {
                Ty::OsPrimitive(
                    ResolvedOsPrimitive::Inbox { .. }
                    | ResolvedOsPrimitive::InboxUnsized { .. }
                    | ResolvedOsPrimitive::Pool { .. }
                    | ResolvedOsPrimitive::DmaSlot { .. },
                ) => {
                    self.check_disallowed_lease_type(inner, span, ctx, false);
                }
                _ => self.check_disallowed_lease_type(inner, span, ctx, false),
            },
            Ty::Array { elem, .. } | Ty::Slice(elem) => {
                self.check_disallowed_lease_type(elem, span, ctx, false);
            }
            Ty::FnPtr { params, ret } => {
                for p in params {
                    self.check_disallowed_lease_type(p, span, ctx, false);
                }
                self.check_disallowed_lease_type(ret, span, ctx, false);
            }
            _ => {}
        }
    }
    pub(super) fn check_extern_c_access(&mut self, prog_idx: usize, ident: &Ident) {
        if self.program_extern_c_scopes.is_empty()
            || !self.non_root_extern_c_names.contains(&ident.name)
        {
            return;
        }
        let Some(fn_info) = self.functions.get(&ident.name) else {
            return;
        };
        if !fn_info.is_extern_c {
            return;
        }
        let declared_locally = self
            .program_extern_c_scopes
            .get(prog_idx)
            .is_some_and(|s| s.contains(&ident.name));
        let imported_pub = self
            .rewritten_pub_extern_sites
            .contains(&(prog_idx, ident.span.start, ident.span.end));
        if !declared_locally && !imported_pub {
            self.emit(Diagnostic::new(
                DiagnosticKind::ModuleError,
                ident.span,
                format!(
                    "`extern \"C\"` function '{}' belongs to another module and is not declared here or imported as `pub`",
                    ident.name
                ),
            ));
        }
    }

    pub(super) fn is_compile_time_const_expr_with_shadowed(
        &self,
        expr: &Expr,
        shadowed: &BTreeSet<String>,
        prog_idx: usize,
    ) -> bool {
        let mut visiting = BTreeSet::new();
        self.is_const_expr_inner(expr, &mut visiting, shadowed, prog_idx)
    }

    pub(super) fn is_const_expr_inner(
        &self,
        expr: &Expr,
        visiting: &mut BTreeSet<String>,
        shadowed: &BTreeSet<String>,
        prog_idx: usize,
    ) -> bool {
        match &expr.kind {
            ExprKind::Literal(_) | ExprKind::Priority(_) => true,
            ExprKind::Ident(id) => {
                if shadowed.contains(&id.name) {
                    return false;
                }
                if self.consts.contains_key(&id.name) {
                    if let Some((raw, _, const_prog)) = self.raw_const_exprs.get(&id.name) {
                        if visiting.insert(id.name.clone()) {
                            let ok = self.is_const_expr_inner(
                                raw,
                                visiting,
                                &BTreeSet::new(),
                                *const_prog,
                            );
                            visiting.remove(&id.name);
                            return ok;
                        }
                    }
                    true
                } else {
                    self.visible_enums_for_variant(prog_idx, &id.name).len() == 1
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(en) = &target.kind {
                    if !shadowed.contains(&en.name)
                        && self
                            .enums
                            .get(&en.name)
                            .is_some_and(|e| e.has_variant(&field.name))
                    {
                        return true;
                    }
                }
                false
            }
            ExprKind::Unary { op, expr: inner } => match op {
                UnaryOp::Neg | UnaryOp::Not | UnaryOp::BitNot => {
                    self.is_const_expr_inner(inner, visiting, shadowed, prog_idx)
                }
                UnaryOp::AddrOf | UnaryOp::Deref => false,
            },
            ExprKind::Binary { lhs, rhs, .. } => {
                self.is_const_expr_inner(lhs, visiting, shadowed, prog_idx)
                    && self.is_const_expr_inner(rhs, visiting, shadowed, prog_idx)
            }
            ExprKind::Cast { expr: inner, .. } => {
                self.is_const_expr_inner(inner, visiting, shadowed, prog_idx)
            }
            ExprKind::ArrayLit(elems) => elems
                .iter()
                .all(|e| self.is_const_expr_inner(e, visiting, shadowed, prog_idx)),
            ExprKind::StructLit { fields, .. } => fields
                .iter()
                .all(|f| self.is_const_expr_inner(&f.value, visiting, shadowed, prog_idx)),
            _ => false,
        }
    }

    pub(super) fn eval_const_bool_shadowed(
        &self,
        expr: &Expr,
        visiting: &mut BTreeSet<String>,
        shadowed: &BTreeSet<String>,
    ) -> Option<bool> {
        match &expr.kind {
            ExprKind::Literal(Literal::Bool(b)) => Some(*b),
            ExprKind::Ident(id) => {
                if shadowed.contains(&id.name) {
                    return None;
                }
                if !visiting.insert(id.name.clone()) {
                    return None;
                }
                let res = self
                    .raw_const_exprs
                    .get(&id.name)
                    .and_then(|(raw, _, _)| {
                        self.eval_const_bool_shadowed(raw, visiting, &BTreeSet::new())
                    });
                visiting.remove(&id.name);
                res
            }
            ExprKind::Unary {
                op: UnaryOp::Not,
                expr: inner,
            } => Some(!self.eval_const_bool_shadowed(inner, visiting, shadowed)?),
            ExprKind::Binary { op, lhs, rhs } => match op {
                BinaryOp::And => Some(
                    self.eval_const_bool_shadowed(lhs, visiting, shadowed)?
                        && self.eval_const_bool_shadowed(rhs, visiting, shadowed)?,
                ),
                BinaryOp::Or => Some(
                    self.eval_const_bool_shadowed(lhs, visiting, shadowed)?
                        || self.eval_const_bool_shadowed(rhs, visiting, shadowed)?,
                ),
                BinaryOp::Eq | BinaryOp::Ne => {
                    if let (Some(a), Some(b)) = (
                        self.eval_const_bool_shadowed(lhs, visiting, shadowed),
                        self.eval_const_bool_shadowed(rhs, visiting, shadowed),
                    ) {
                        return Some(if matches!(op, BinaryOp::Eq) { a == b } else { a != b });
                    }
                    let a = self.eval_const_expr_i128_shadowed(lhs, visiting, shadowed)?;
                    let b = self.eval_const_expr_i128_shadowed(rhs, visiting, shadowed)?;
                    Some(if matches!(op, BinaryOp::Eq) { a == b } else { a != b })
                }
                BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                    let a = self.eval_const_expr_i128_shadowed(lhs, visiting, shadowed)?;
                    let b = self.eval_const_expr_i128_shadowed(rhs, visiting, shadowed)?;
                    Some(match op {
                        BinaryOp::Lt => a < b,
                        BinaryOp::Le => a <= b,
                        BinaryOp::Gt => a > b,
                        BinaryOp::Ge => a >= b,
                        _ => unreachable!(),
                    })
                }
                _ => None,
            },
            _ => None,
        }
    }

    pub(super) fn eval_compile_time_irq_arg_with_shadowed(
        &self,
        expr: &Expr,
        ty: &Ty,
        shadowed: &BTreeSet<String>,
        prog_idx: usize,
    ) -> Option<IrqBindingArg> {
        if ty.is_bool() {
            let mut visiting = BTreeSet::new();
            if let Some(b) = self.eval_const_bool_shadowed(expr, &mut visiting, shadowed) {
                return Some(IrqBindingArg::ConstBool(b));
            }
        }
        if ty.is_integer() || matches!(ty, Ty::IntLit) {
            let mut visiting = BTreeSet::new();
            if let Some(value) = self.eval_const_expr_i128_shadowed(expr, &mut visiting, shadowed)
            {
                let final_ty = if matches!(ty, Ty::IntLit) {
                    Ty::u32()
                } else {
                    ty.clone()
                };
                return Some(IrqBindingArg::ConstInt {
                    ty: final_ty,
                    value,
                });
            }
        }
        if self.is_compile_time_const_expr_with_shadowed(expr, shadowed, prog_idx) {
            return Some(IrqBindingArg::ConstExpr {
                ty: ty.clone(),
                expr: expr.clone(),
            });
        }
        None
    }
}
