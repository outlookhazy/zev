use super::*;

impl<'ctx, 'm> FnIrBuilder<'ctx, 'm> {
    pub(super) fn new(
        ctx: &'ctx mut ModuleCtx<'m>,
        vars: &[MirVar],
        coroutine: Option<&CoroutineFrameLayout>,
        ret_ty: &MirType,
        error_spec: &MirErrorSpec,
    ) -> Self {
        Self {
            ctx,
            vars: vars.to_vec(),
            coroutine: coroutine.cloned(),
            isr_state_ptrs: HashMap::new(),
            ret_ty: ret_ty.clone(),
            error_spec: error_spec.clone(),
            buf: String::with_capacity(4096),
            entry_allocas: String::new(),
            entry_alloca_insert_pos: None,
            temp_alloca_id: 0,
            tmp_id: 0,
            lbl_id: 0,
            cur_label: "entry".to_string(),
            terminated: false,
            loop_targets: Vec::new(),
        }
    }

    pub(super) fn finish(mut self) -> String {
        if !self.entry_allocas.is_empty() {
            if let Some(pos) = self.entry_alloca_insert_pos {
                self.buf.insert_str(pos, &self.entry_allocas);
            }
        }
        self.buf
    }

    pub(super) fn require_coro(&self) -> Result<CoroutineFrameLayout, CodegenError> {
        self.coroutine.clone().ok_or_else(|| {
            CodegenError::new(
                "llvm-codegen",
                "coroutine state requested in non-coroutine function",
            )
        })
    }

    pub(super) fn stack_align(&self, lty: &str) -> u32 {
        if self.ctx.spec.isa.is_avr() {
            return 1;
        }
        let ptr_bits = self.ctx.spec.isa.ptr_bits();
        if lty == "double" || lty == "i64" || (lty == "ptr" && ptr_bits == 64) {
            8
        } else {
            4
        }
    }

    pub(super) fn emit_temp_alloca(&mut self, lty: &str) -> (String, u32) {
        let id = self.temp_alloca_id;
        self.temp_alloca_id += 1;
        let name = format!("%__a{id}");
        let al = self.stack_align(lty);
        let _ = writeln!(self.entry_allocas, "  {name} = alloca {lty}, align {al}");
        (name, al)
    }

    pub(super) fn next_tmp(&mut self) -> String {
        let id = self.tmp_id;
        self.tmp_id += 1;
        format!("%t{id}")
    }

    pub(super) fn next_label(&mut self, prefix: &str) -> String {
        let id = self.lbl_id;
        self.lbl_id += 1;
        format!("{prefix}_{id}")
    }

    pub(super) fn start_block(&mut self, label: &str) {
        let _ = writeln!(self.buf, "{label}:");
        self.cur_label = label.to_string();
        self.terminated = false;
    }

    pub(super) fn emit_line(&mut self, line: impl AsRef<str>) {
        if !self.terminated {
            let _ = writeln!(self.buf, "  {}", line.as_ref());
        }
    }

    pub(super) fn emit_term(&mut self, line: impl AsRef<str>) {
        if !self.terminated {
            let _ = writeln!(self.buf, "  {}", line.as_ref());
            self.terminated = true;
        }
    }

    pub(super) fn emit_entry_allocas(&mut self, param_vids: &[VarId]) {
        self.start_block("entry");
        self.entry_alloca_insert_pos = Some(self.buf.len());
        let crit_al = self.stack_align("i32");
        self.emit_line(format!("%__crit_pm = alloca i32, align {crit_al}"));
        self.emit_line(format!("store i32 0, ptr %__crit_pm, align {crit_al}"));
        if self.ctx.rt_features.uses_fmt {
            self.emit_line(format!("%__fmt_sink = alloca %SynFmtSink, align {crit_al}"));
        }
        for v in &self.vars.clone() {
            if v.role == VarRole::IsrState
                || self.isr_state_ptrs.contains_key(&v.id)
                || v.ty.is_void()
                || (self.coroutine.is_some() && v.role == VarRole::Param)
            {
                continue;
            }
            if let Some(coro) = &self.coroutine {
                if coro.hoisted_var_gep.contains_key(&v.id) {
                    continue;
                }
            }
            let lty = self.ctx.llvm_ty(&v.ty);
            let al = self.stack_align(&lty);
            self.emit_line(format!("%v_{} = alloca {lty}, align {al}", v.id.0));
            self.emit_line(format!(
                "store {lty} zeroinitializer, ptr %v_{}, align {al}",
                v.id.0
            ));
        }
        for (idx, &vid) in param_vids.iter().enumerate() {
            let lty = self.ctx.llvm_ty(&self.vars[vid.0].ty);
            let al = self.stack_align(&lty);
            self.emit_line(format!(
                "store {lty} %arg{idx}, ptr %v_{}, align {al}",
                vid.0
            ));
        }
    }

    pub(super) fn call_kw(&self) -> &'static str {
        self.ctx.call_kw()
    }

    pub(super) fn emit_avr_safe_i32_load(&mut self, ptr_repr: &str, contended: bool) -> String {
        let val = self.next_tmp();
        if self.ctx.spec.isa.is_avr() && contended {
            let call_kw = self.call_kw();
            let pm = self.next_tmp();
            self.emit_line(format!("{pm} = {call_kw} i32 @__syn_crit_enter()"));
            self.emit_line(format!(
                "{val} = load volatile i32, ptr {ptr_repr}, align 4"
            ));
            self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
        } else {
            self.emit_line(format!(
                "{val} = load volatile i32, ptr {ptr_repr}, align 4"
            ));
        }
        val
    }

    pub(super) fn var_ptr(&mut self, vid: VarId) -> String {
        if let Some(state_ptr) = self.isr_state_ptrs.get(&vid) {
            return state_ptr.clone();
        }
        let hoisted_info = self.coroutine.as_ref().and_then(|coro| {
            coro.hoisted_var_gep
                .get(&vid)
                .map(|&idx| (coro.llvm_name.clone(), idx))
        });
        if let Some((cname, gep_idx)) = hoisted_info {
            let p = self.next_tmp();
            self.emit_line(format!(
                "{p} = getelementptr inbounds {cname}, ptr %__frame, i32 0, i32 {gep_idx}"
            ));
            return p;
        }
        format!("%v_{}", vid.0)
    }

    pub(super) fn emit_zero_task_locals(&mut self) {
        let vars_snapshot = self.vars.clone();
        for v in &vars_snapshot {
            if v.role == VarRole::Param
                || v.role == VarRole::IsrState
                || self.isr_state_ptrs.contains_key(&v.id)
                || v.ty.is_void()
            {
                continue;
            }
            let vp = self.var_ptr(v.id);
            let lty = self.ctx.llvm_ty(&v.ty);
            self.emit_line(format!("store {lty} zeroinitializer, ptr {vp}, align 1"));
        }
        if let Some(coro) = self.coroutine.clone() {
            for &(gep_idx, _) in coro.union_group_gep.values() {
                if let Some(fty) = coro.field_llvm_tys.get(gep_idx) {
                    let up = self.next_tmp();
                    self.emit_line(format!(
                        "{up} = getelementptr inbounds {}, ptr %__frame, i32 0, i32 {gep_idx}",
                        coro.llvm_name
                    ));
                    self.emit_line(format!("store {fty} zeroinitializer, ptr {up}, align 1"));
                }
            }
        }
    }

    pub(super) fn coerce(&mut self, val: SsaVal, dst_ty: &str, signed: bool) -> SsaVal {
        if val.ty == dst_ty {
            return val;
        }
        if val.ty == "float" && dst_ty == "double" {
            let t = self.next_tmp();
            self.emit_line(format!("{t} = fpext float {} to double", val.repr));
            return SsaVal::new("double", t);
        }
        if val.ty == "double" && dst_ty == "float" {
            let t = self.next_tmp();
            self.emit_line(format!("{t} = fptrunc double {} to float", val.repr));
            return SsaVal::new("float", t);
        }
        if (val.ty == "float" || val.ty == "double") && dst_ty.starts_with('i') {
            if dst_ty == "i1" {
                let t = self.next_tmp();
                self.emit_line(format!("{t} = fcmp une {} {}, 0.0", val.ty, val.repr));
                return SsaVal::new("i1", t);
            }
            let int_ty = match dst_ty {
                "i8" | "i16" | "i32" | "i64" => dst_ty,
                _ => "i32",
            };
            let f_sfx = if val.ty == "float" { "f32" } else { "f64" };
            let intr = if signed { "fptosi.sat" } else { "fptoui.sat" };
            let ti = self.next_tmp();
            self.emit_line(format!(
                "{ti} = call {int_ty} @llvm.{intr}.{int_ty}.{f_sfx}({} {})",
                val.ty, val.repr
            ));
            return SsaVal::new(int_ty, ti);
        }
        if val.ty.starts_with('i') && (dst_ty == "float" || dst_ty == "double") {
            let sb = llvm_int_bits(&val.ty);
            let widened = if sb < 32 {
                let wop = if signed && sb > 1 { "sext" } else { "zext" };
                let wt = self.next_tmp();
                self.emit_line(format!("{wt} = {wop} {} {} to i32", val.ty, val.repr));
                SsaVal::new("i32", wt)
            } else {
                val
            };
            let fop = if signed && sb > 1 { "sitofp" } else { "uitofp" };
            let tf = self.next_tmp();
            self.emit_line(format!(
                "{tf} = {fop} {} {} to {dst_ty}",
                widened.ty, widened.repr
            ));
            return SsaVal::new(dst_ty, tf);
        }
        if (val.ty == "ptr addrspace(1)" && dst_ty == "ptr")
            || (val.ty == "ptr" && dst_ty == "ptr addrspace(1)")
        {
            let usize_ty = self.ctx.spec.isa.usize_llvm_ty();
            let it = self.next_tmp();
            let pt = self.next_tmp();
            self.emit_line(format!(
                "{it} = ptrtoint {} {} to {usize_ty}",
                val.ty, val.repr
            ));
            self.emit_line(format!("{pt} = inttoptr {usize_ty} {it} to {dst_ty}"));
            return SsaVal::new(dst_ty, pt);
        }
        if (dst_ty == "ptr" || dst_ty == "ptr addrspace(1)") && val.ty.starts_with('i') {
            let t = self.next_tmp();
            self.emit_line(format!(
                "{t} = inttoptr {} {} to {dst_ty}",
                val.ty, val.repr
            ));
            return SsaVal::new(dst_ty, t);
        }
        if (val.ty == "ptr" || val.ty == "ptr addrspace(1)") && dst_ty.starts_with('i') {
            let t = self.next_tmp();
            self.emit_line(format!(
                "{t} = ptrtoint {} {} to {dst_ty}",
                val.ty, val.repr
            ));
            return SsaVal::new(dst_ty, t);
        }
        if val.ty.starts_with('i') && dst_ty.starts_with('i') {
            let sb = llvm_int_bits(&val.ty);
            let db = llvm_int_bits(dst_ty);
            let t = self.next_tmp();
            if sb < db {
                let op = if signed && sb > 1 { "sext" } else { "zext" };
                self.emit_line(format!("{t} = {op} {} {} to {dst_ty}", val.ty, val.repr));
            } else {
                self.emit_line(format!("{t} = trunc {} {} to {dst_ty}", val.ty, val.repr));
            }
            return SsaVal::new(dst_ty, t);
        }
        if !dst_ty.starts_with('i')
            && dst_ty != "ptr"
            && dst_ty != "ptr addrspace(1)"
            && dst_ty != "void"
            && dst_ty != "float"
            && dst_ty != "double"
        {
            return SsaVal::new(dst_ty, "zeroinitializer");
        }
        val
    }

    pub(super) fn coerce_to_i1(&mut self, val: SsaVal) -> String {
        if val.ty == "i1" {
            return val.repr;
        }
        let t = self.next_tmp();
        if val.ty == "ptr" || val.ty == "ptr addrspace(1)" {
            self.emit_line(format!("{t} = icmp ne {} {}, null", val.ty, val.repr));
        } else if val.ty == "float" || val.ty == "double" {
            self.emit_line(format!("{t} = fcmp une {} {}, 0.0", val.ty, val.repr));
        } else {
            self.emit_line(format!("{t} = icmp ne {} {}, 0", val.ty, val.repr));
        }
        t
    }

    // ── Coroutine State-Machine Lowering ─────────────────────────────────────

    pub(super) fn emit_coroutine_cfg(&mut self, blocks: &[MirBlock]) -> Result<(), CodegenError> {
        let coro = self.require_coro()?;
        self.emit_entry_allocas(&[]);

        let pc_ptr = self.next_tmp();
        self.emit_line(format!(
            "{pc_ptr} = getelementptr inbounds {}, ptr %__frame, i32 0, i32 0",
            coro.llvm_name
        ));
        let pc_val = self.next_tmp();
        self.emit_line(format!("{pc_val} = load i32, ptr {pc_ptr}, align 4"));

        let mut switch_cases = String::new();
        let entry_bb = blocks.first().map(|b| b.id.0).unwrap_or(0);
        let _ = writeln!(switch_cases, "    i32 {entry_bb}, label %bb_{entry_bb}");
        for b in blocks {
            if b.id.0 != entry_bb && b.is_resume_entry {
                let _ = writeln!(switch_cases, "    i32 {}, label %bb_{}", b.id.0, b.id.0);
            }
        }
        self.emit_term(format!(
            "switch i32 {pc_val}, label %coro_done [\n{switch_cases}  ]"
        ));

        let mut every_resume_map: BTreeMap<BlockId, (VarId, VarId)> = BTreeMap::new();
        for b in blocks {
            if let MirTerminator::Yield(MirYieldKind::EveryLatch {
                next_deadline_var,
                period_var,
                resume_bb,
            }) = &b.terminator
            {
                every_resume_map.insert(*resume_bb, (*next_deadline_var, *period_var));
            }
        }

        let entry_is_restart_target = blocks.first().is_some_and(|b| b.is_resume_entry);

        for b in blocks {
            let lbl = format!("bb_{}", b.id.0);
            self.start_block(&lbl);

            // On `restart;` back into `bb_0`, zero non-Param locals and child frames while keeping `Param` fields intact!
            if b.id.0 == entry_bb && entry_is_restart_target {
                self.emit_zero_task_locals();
            }

            // Resync while-sleeping overruns at `loop every` body entry (`wdiff >= eff_p`)
            if let Some(&(dv_id, pv_id)) = every_resume_map.get(&b.id) {
                let dl_ptr = self.var_ptr(dv_id);
                let eff_p = self.emit_clamped_every_period(pv_id);
                let call_kw = self.call_kw();
                let dv = self.next_tmp();
                let now = self.next_tmp();
                let wdiff = self.next_tmp();
                let is_over = self.next_tmp();
                self.emit_line(format!("{dv} = load i32, ptr {dl_ptr}, align 4"));
                self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                self.emit_line(format!("{wdiff} = sub i32 {now}, {dv}"));
                self.emit_line(format!("{is_over} = icmp sge i32 {wdiff}, {eff_p}"));
                let resync_lbl = self.next_label("every_wresync");
                let cont_lbl = self.next_label("every_wcont");
                self.emit_term(format!(
                    "br i1 {is_over}, label %{resync_lbl}, label %{cont_lbl}"
                ));
                self.start_block(&resync_lbl);
                let q = self.next_tmp();
                let skip = self.next_tmp();
                let dv_new = self.next_tmp();
                self.emit_line(format!("{q} = udiv i32 {wdiff}, {eff_p}"));
                self.emit_line(format!("{skip} = mul i32 {q}, {eff_p}"));
                self.emit_line(format!("{dv_new} = add i32 {dv}, {skip}"));
                self.emit_line(format!("store i32 {dv_new}, ptr {dl_ptr}, align 4"));
                self.emit_term(format!("br label %{cont_lbl}"));
                self.start_block(&cont_lbl);
            }

            for stmt in &b.stmts {
                self.emit_stmt(stmt)?;
            }
            self.emit_coroutine_terminator(&b.terminator)?;
        }

        self.start_block("coro_done");
        self.emit_term("ret i32 2");
        Ok(())
    }

    pub(super) fn store_coro_pc(&mut self, next_pc: i32) {
        let Some(cname) = self.coroutine.as_ref().map(|c| c.llvm_name.clone()) else {
            return;
        };
        let p = self.next_tmp();
        self.emit_line(format!(
            "{p} = getelementptr inbounds {cname}, ptr %__frame, i32 0, i32 0"
        ));
        self.emit_line(format!("store i32 {next_pc}, ptr {p}, align 4"));
    }

    pub(super) fn park_task_deadline(&mut self, wake_tick_val: &SsaVal) {
        let call_kw = self.call_kw();
        let wt = self.coerce(wake_tick_val.clone(), "i32", false);
        let is_null = self.next_tmp();
        let park_lbl = self.next_label("park_dl");
        let after_lbl = self.next_label("after_dl");
        self.emit_line(format!("{is_null} = icmp eq ptr %__task, null"));
        self.emit_term(format!(
            "br i1 {is_null}, label %{after_lbl}, label %{park_lbl}"
        ));
        self.start_block(&park_lbl);
        let pm = self.next_tmp();
        self.emit_line(format!("{pm} = {call_kw} i32 @__syn_crit_enter()"));
        let st_p = self.next_tmp();
        let wt_p = self.next_tmp();
        let ht_p = self.next_tmp();
        let wo_p = self.next_tmp();
        self.emit_line(format!(
            "{st_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 2"
        ));
        self.emit_line(format!(
            "{ht_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 7"
        ));
        self.emit_line(format!(
            "{wt_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 11"
        ));
        self.emit_line(format!(
            "{wo_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 12"
        ));
        self.emit_line(format!("store i8 2, ptr {st_p}, align 1"));
        self.emit_line(format!("store i8 0, ptr {ht_p}, align 1"));
        self.emit_line(format!("store i32 {}, ptr {wt_p}, align 4", wt.repr));
        self.emit_line(format!("store ptr null, ptr {wo_p}, align 4"));
        self.emit_line(format!(
            "{call_kw} void @__syn_ready_clear_task(ptr %__task)"
        ));
        self.emit_line(format!(
            "{call_kw} void @__syn_note_deadline(i32 {})",
            wt.repr
        ));
        self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
        self.emit_term(format!("br label %{after_lbl}"));
        self.start_block(&after_lbl);
    }

    pub(super) fn emit_coroutine_terminator(
        &mut self,
        term: &MirTerminator,
    ) -> Result<(), CodegenError> {
        if self.terminated {
            return Ok(());
        }
        match term {
            MirTerminator::Goto(b) => {
                self.emit_term(format!("br label %bb_{}", b.0));
            }
            MirTerminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                let cv = self.emit_expr(cond)?;
                let b1 = self.coerce_to_i1(cv);
                self.emit_term(format!(
                    "br i1 {b1}, label %bb_{}, label %bb_{}",
                    then_bb.0, else_bb.0
                ));
            }
            MirTerminator::Switch {
                discr,
                arms,
                default_bb,
            } => {
                let dv = self.emit_expr(discr)?;
                let sw_ty = if llvm_int_bits(&dv.ty) > 32 {
                    "i64"
                } else {
                    "i32"
                };
                let dc = self.coerce(dv, sw_ty, !discr.ty.is_unsigned());
                let mut cases = String::new();
                for (pat, bb) in arms {
                    let pv = self.pattern_int(pat);
                    let _ = writeln!(cases, "    {sw_ty} {pv}, label %bb_{}", bb.0);
                }
                self.emit_term(format!(
                    "switch {sw_ty} {}, label %bb_{} [\n{cases}  ]",
                    dc.repr, default_bb.0
                ));
            }
            MirTerminator::Return(val_opt) => {
                let coro = self.require_coro()?;
                if coro.has_ret_slot {
                    let env = self.build_return_envelope(val_opt.as_ref())?;
                    let rp = self.next_tmp();
                    self.emit_line(format!(
                        "{rp} = getelementptr inbounds {}, ptr %__frame, i32 0, i32 1",
                        coro.llvm_name
                    ));
                    self.emit_line(format!("store {} {}, ptr {rp}, align 4", env.ty, env.repr));
                }
                self.store_coro_pc(-1);
                self.emit_term("ret i32 2");
            }
            MirTerminator::Error { reason, partial } => {
                let coro = self.require_coro()?;
                if coro.has_ret_slot {
                    let env = self.build_error_envelope(reason.as_ref(), partial.as_ref())?;
                    let rp = self.next_tmp();
                    self.emit_line(format!(
                        "{rp} = getelementptr inbounds {}, ptr %__frame, i32 0, i32 1",
                        coro.llvm_name
                    ));
                    self.emit_line(format!("store {} {}, ptr {rp}, align 4", env.ty, env.repr));
                }
                self.store_coro_pc(-1);
                self.emit_term("ret i32 2");
            }
            MirTerminator::Yield(yk) => self.emit_yield(yk)?,
        }
        Ok(())
    }

    fn emit_clamped_every_period(&mut self, pv_id: VarId) -> String {
        let pv_ptr = self.var_ptr(pv_id);
        let vty = self.vars[pv_id.0].ty.host_value_type();
        let pv = self.load_from_place_ptr(&pv_ptr, &vty, false);
        let ptr_bits = self.ctx.spec.isa.ptr_bits();
        let (is_signed, bits) = match vty {
            MirType::I8 => (true, 8),
            MirType::I16 => (true, 16),
            MirType::I32 | MirType::Enum(_) => (true, 32),
            MirType::I64 => (true, 64),
            MirType::Isize => (true, ptr_bits),
            MirType::U8 => (false, 8),
            MirType::U16 => (false, 16),
            MirType::U32 => (false, 32),
            MirType::U64 => (false, 64),
            MirType::Usize => (false, ptr_bits),
            _ => (!vty.is_unsigned(), 32),
        };
        let lty = pv.ty;
        if bits < 32 {
            let ext32 = self
                .coerce(SsaVal::new(lty, pv.repr), "i32", is_signed)
                .repr;
            let low = self.next_tmp();
            let clamped = self.next_tmp();
            if is_signed {
                self.emit_line(format!("{low} = icmp sle i32 {ext32}, 0"));
            } else {
                self.emit_line(format!("{low} = icmp eq i32 {ext32}, 0"));
            }
            self.emit_line(format!("{clamped} = select i1 {low}, i32 1, i32 {ext32}"));
            return clamped;
        }
        if bits == 64 || lty == "i64" {
            let low = self.next_tmp();
            let c1 = self.next_tmp();
            let high = self.next_tmp();
            let c2 = self.next_tmp();
            let tr = self.next_tmp();
            if is_signed {
                self.emit_line(format!("{low} = icmp sle i64 {}, 0", pv.repr));
                self.emit_line(format!("{c1} = select i1 {low}, i64 1, i64 {}", pv.repr));
                self.emit_line(format!(
                    "{high} = icmp sgt i64 {c1}, {}",
                    zev_ast::MAX_DURATION_TICKS
                ));
            } else {
                self.emit_line(format!("{low} = icmp eq i64 {}, 0", pv.repr));
                self.emit_line(format!("{c1} = select i1 {low}, i64 1, i64 {}", pv.repr));
                self.emit_line(format!(
                    "{high} = icmp ugt i64 {c1}, {}",
                    zev_ast::MAX_DURATION_TICKS
                ));
            }
            self.emit_line(format!(
                "{c2} = select i1 {high}, i64 {}, i64 {c1}",
                zev_ast::MAX_DURATION_TICKS
            ));
            self.emit_line(format!("{tr} = trunc i64 {c2} to i32"));
            return tr;
        }
        // 32-bit (`i32` / `u32`): extend/normalize in case `Enum`/`Bool`, then clamp `[1, i32::MAX]`.
        let pv32 = if lty == "i32" {
            pv.repr
        } else {
            self.coerce(SsaVal::new(lty, pv.repr), "i32", is_signed)
                .repr
        };
        let low = self.next_tmp();
        let c1 = self.next_tmp();
        if is_signed {
            self.emit_line(format!("{low} = icmp sle i32 {pv32}, 0"));
            self.emit_line(format!("{c1} = select i1 {low}, i32 1, i32 {pv32}"));
            c1
        } else {
            self.emit_line(format!("{low} = icmp eq i32 {pv32}, 0"));
            self.emit_line(format!("{c1} = select i1 {low}, i32 1, i32 {pv32}"));
            let high = self.next_tmp();
            let c2 = self.next_tmp();
            self.emit_line(format!(
                "{high} = icmp ugt i32 {c1}, {}",
                zev_ast::MAX_DURATION_TICKS
            ));
            self.emit_line(format!(
                "{c2} = select i1 {high}, i32 {}, i32 {c1}",
                zev_ast::MAX_DURATION_TICKS
            ));
            c2
        }
    }

    pub(super) fn emit_yield(&mut self, yk: &MirYieldKind) -> Result<(), CodegenError> {
        match yk {
            MirYieldKind::EveryLatch {
                next_deadline_var,
                period_var,
                resume_bb,
            } => {
                let dl_ptr = self.var_ptr(*next_deadline_var);
                let eff_p = self.emit_clamped_every_period(*period_var);

                let cur_dl = self.next_tmp();
                self.emit_line(format!("{cur_dl} = load i32, ptr {dl_ptr}, align 4"));
                let next_dl = self.next_tmp();
                self.emit_line(format!("{next_dl} = add i32 {cur_dl}, {eff_p}"));
                self.emit_line(format!("store i32 {next_dl}, ptr {dl_ptr}, align 4"));

                let call_kw = self.call_kw();
                let now = self.next_tmp();
                self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                let diff = self.next_tmp();
                self.emit_line(format!("{diff} = sub i32 {now}, {next_dl}"));
                let is_early = self.next_tmp();
                self.emit_line(format!("{is_early} = icmp slt i32 {diff}, 0"));

                let due_lbl = self.next_label("every_due");
                let sleep_lbl = self.next_label("every_wait");
                self.emit_term(format!(
                    "br i1 {is_early}, label %{sleep_lbl}, label %{due_lbl}"
                ));

                self.start_block(&due_lbl);
                let q = self.next_tmp();
                let skip = self.next_tmp();
                let resync_dl = self.next_tmp();
                self.emit_line(format!("{q} = udiv i32 {diff}, {eff_p}"));
                self.emit_line(format!("{skip} = mul i32 {q}, {eff_p}"));
                self.emit_line(format!("{resync_dl} = add i32 {next_dl}, {skip}"));
                self.emit_line(format!("store i32 {resync_dl}, ptr {dl_ptr}, align 4"));

                let share = self.next_tmp();
                self.emit_line(format!(
                    "{share} = {call_kw} i1 @__syn_should_share(ptr %__task)"
                ));
                let yield_more = self.next_label("every_more");
                self.emit_term(format!(
                    "br i1 {share}, label %{yield_more}, label %bb_{}",
                    resume_bb.0
                ));
                self.start_block(&yield_more);
                self.store_coro_pc(resume_bb.0 as i32);
                self.emit_term("ret i32 0");

                self.start_block(&sleep_lbl);
                let dl_ssa = SsaVal::new("i32", next_dl);
                self.park_task_deadline(&dl_ssa);
                self.store_coro_pc(resume_bb.0 as i32);
                self.emit_term("ret i32 1");
            }
            MirYieldKind::DelayPoll {
                deadline_var,
                poll_bb,
                resume_bb,
            } => {
                let call_kw = self.call_kw();
                let dl_ptr = self.var_ptr(*deadline_var);
                let dl = self.next_tmp();
                self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 4"));
                let now = self.next_tmp();
                self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                let diff = self.next_tmp();
                self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
                let done = self.next_tmp();
                self.emit_line(format!("{done} = icmp sge i32 {diff}, 0"));

                let wait_lbl = self.next_label("delay_wait");
                self.emit_term(format!(
                    "br i1 {done}, label %bb_{}, label %{wait_lbl}",
                    resume_bb.0
                ));
                self.start_block(&wait_lbl);
                let dl_ssa = SsaVal::new("i32", dl);
                self.park_task_deadline(&dl_ssa);
                self.store_coro_pc(poll_bb.0 as i32);
                self.emit_term("ret i32 1");
            }
            MirYieldKind::ShareCheckpoint { resume_bb } => {
                let call_kw = self.call_kw();
                let share = self.next_tmp();
                self.emit_line(format!(
                    "{share} = {call_kw} i1 @__syn_should_share(ptr %__task)"
                ));
                let yield_lbl = self.next_label("share_yield");
                self.emit_term(format!(
                    "br i1 {share}, label %{yield_lbl}, label %bb_{}",
                    resume_bb.0
                ));
                self.start_block(&yield_lbl);
                self.store_coro_pc(resume_bb.0 as i32);
                self.emit_term("ret i32 0");
            }
            MirYieldKind::Wait {
                arms,
                timer,
                poll_bb,
            } => {
                self.emit_wait_yield(arms, timer.as_ref(), *poll_bb)?;
            }
            MirYieldKind::CallYielding {
                callee,
                slot_name,
                args,
                poll_bb,
                ..
            } => {
                let coro = self.require_coro()?;
                let ug = coro.slot_to_group[slot_name];
                let (ug_gep, _) = coro.union_group_gep[&ug];
                let child_layout = self.ctx.fn_frames[callee].clone();
                let child_sig = self.ctx.fn_sigs[callee].clone();
                let callee_fn = self
                    .ctx
                    .module
                    .functions
                    .iter()
                    .find(|f| &f.name == callee)
                    .cloned()
                    .ok_or_else(|| {
                        CodegenError::new(
                            "llvm-codegen",
                            format!("missing yielding callee `{callee}`"),
                        )
                    })?;

                let mut eval_args = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    let av = self.emit_expr(a)?;
                    let pty = self.ctx.llvm_ty(&child_sig.params[i]);
                    let signed = !a.ty.is_unsigned();
                    eval_args.push(self.coerce(av, &pty, signed));
                }

                let cptr = self.next_tmp();
                self.emit_line(format!(
                    "{cptr} = getelementptr inbounds {}, ptr %__frame, i32 0, i32 {ug_gep}",
                    coro.llvm_name
                ));
                self.emit_line(format!(
                    "store {} zeroinitializer, ptr {cptr}, align 8",
                    child_layout.llvm_name
                ));
                for (i, &pvid) in callee_fn.params.iter().enumerate() {
                    if let Some(&pgep) = child_layout.hoisted_var_gep.get(&pvid) {
                        let pp = self.next_tmp();
                        self.emit_line(format!(
                            "{pp} = getelementptr inbounds {}, ptr {cptr}, i32 0, i32 {pgep}",
                            child_layout.llvm_name
                        ));
                        self.emit_line(format!(
                            "store {} {}, ptr {pp}, align 4",
                            eval_args[i].ty, eval_args[i].repr
                        ));
                    }
                }
                self.emit_term(format!("br label %bb_{}", poll_bb.0));
            }
            MirYieldKind::PollChild {
                callee,
                slot_name,
                dst,
                poll_bb,
                resume_bb,
            } => {
                let coro = self.require_coro()?;
                let ug = coro.slot_to_group[slot_name];
                let (ug_gep, _) = coro.union_group_gep[&ug];
                let child_layout = self.ctx.fn_frames[callee].clone();
                let child_sig = self.ctx.fn_sigs[callee].clone();

                let cptr = self.next_tmp();
                self.emit_line(format!(
                    "{cptr} = getelementptr inbounds {}, ptr %__frame, i32 0, i32 {ug_gep}",
                    coro.llvm_name
                ));

                // Uncontended back-to-back MORE execution loop (mirroring PT_SPAWN_IF_NEEDED!)
                let loop_lbl = self.next_label("poll_child_loop");
                let chk_more_lbl = self.next_label("poll_child_more");
                let yield_lbl = self.next_label("poll_child_yield");
                let done_lbl = self.next_label("poll_child_done");
                self.emit_term(format!("br label %{loop_lbl}"));

                let call_kw = self.call_kw();
                self.start_block(&loop_lbl);
                let st = self.next_tmp();
                self.emit_line(format!(
                    "{st} = {call_kw} i32 {}(ptr {cptr}, ptr %__task)",
                    child_sig.symbol
                ));
                let is_done = self.next_tmp();
                self.emit_line(format!("{is_done} = icmp eq i32 {st}, 2"));
                self.emit_term(format!(
                    "br i1 {is_done}, label %{done_lbl}, label %{chk_more_lbl}"
                ));

                self.start_block(&chk_more_lbl);
                let is_more = self.next_tmp();
                self.emit_line(format!("{is_more} = icmp eq i32 {st}, 0"));
                let share = self.next_tmp();
                self.emit_line(format!(
                    "{share} = {call_kw} i1 @__syn_should_share(ptr %__task)"
                ));
                let not_share = self.next_tmp();
                self.emit_line(format!("{not_share} = xor i1 {share}, true"));
                let keep_going = self.next_tmp();
                self.emit_line(format!("{keep_going} = and i1 {is_more}, {not_share}"));
                self.emit_term(format!(
                    "br i1 {keep_going}, label %{loop_lbl}, label %{yield_lbl}"
                ));

                self.start_block(&yield_lbl);
                self.store_coro_pc(poll_bb.0 as i32);
                self.emit_term(format!("ret i32 {st}"));

                self.start_block(&done_lbl);
                if let Some(d) = dst {
                    if child_layout.has_ret_slot {
                        let rp = self.next_tmp();
                        self.emit_line(format!(
                            "{rp} = getelementptr inbounds {}, ptr {cptr}, i32 0, i32 1",
                            child_layout.llvm_name
                        ));
                        let rv = self.next_tmp();
                        self.emit_line(format!(
                            "{rv} = load {}, ptr {rp}, align 4",
                            child_layout.ret_llvm_ty
                        ));
                        let dvid = match d {
                            MirCallDst::Value(v) | MirCallDst::Result(v) => *v,
                        };
                        let dp = self.var_ptr(dvid);
                        let dty = self.ctx.llvm_ty(&self.vars[dvid.0].ty);
                        let signed = !self.vars[dvid.0].ty.is_unsigned();
                        let coerced =
                            self.coerce(SsaVal::new(&child_layout.ret_llvm_ty, rv), &dty, signed);
                        self.emit_line(format!(
                            "store {} {}, ptr {dp}, align 4",
                            coerced.ty, coerced.repr
                        ));
                    }
                }
                self.emit_term(format!("br label %bb_{}", resume_bb.0));
            }
        }
        Ok(())
    }

    pub(super) fn emit_wait_yield(
        &mut self,
        arms: &[MirWaitArm],
        timer: Option<&MirWaitTimer>,
        poll_bb: BlockId,
    ) -> Result<(), CodegenError> {
        let call_kw = self.call_kw();
        let ptr_al = self.ctx.ptr_align();
        let i32_al = self.ctx.llvm_scalar_abi_align(4);
        let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();

        enum ResolvedArm {
            InboxRecv {
                qptr: String,
                wm_p: String,
                id: String,
                elem_ty: MirType,
                dst_val: Option<VarId>,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            InboxViewRecv {
                hdr: String,
                items_base: String,
                cap32: String,
                wm_p: String,
                key: String,
                elem_ty: MirType,
                dst_val: Option<VarId>,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            InboxSend {
                qptr: String,
                wm_p: String,
                id: String,
                elem_ty: MirType,
                val_var: VarId,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            InboxViewSend {
                hdr: String,
                items_base: String,
                cap32: String,
                wm_p: String,
                key: String,
                elem_ty: MirType,
                val_var: VarId,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            SignalWait {
                sptr: String,
                wm_p: String,
                id: String,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            WaitQueueCommit {
                wm_p: String,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            WatchWaitChange {
                wptr: String,
                wm_p: String,
                pty: String,
                elem_ty: MirType,
                last_gen_var: VarId,
                dst_val: Option<VarId>,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            PingPongWaitChunk {
                pptr: String,
                wm_p: String,
                id: String,
                cap: usize,
                dst_slice: VarId,
                target_bb: BlockId,
            },
            PoolAlloc {
                pptr: String,
                wm_p: String,
                id: String,
                dst_val: Option<VarId>,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            ByteRingWaitRead {
                rptr: String,
                wm_p: String,
                id: String,
                dst_val: Option<VarId>,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            ByteRingViewWaitRead {
                hdr: String,
                cap32: String,
                wm_p: String,
                dst_val: Option<VarId>,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            ByteRingWaitWrite {
                rptr: String,
                wm_p: String,
                id: String,
                dst_val: Option<VarId>,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
            ByteRingViewWaitWrite {
                hdr: String,
                cap32: String,
                wm_p: String,
                dst_val: Option<VarId>,
                dst_res: Option<VarId>,
                target_bb: BlockId,
            },
        }

        impl ResolvedArm {
            fn wm_ptr(&self) -> &str {
                match self {
                    ResolvedArm::InboxRecv { wm_p, .. }
                    | ResolvedArm::InboxViewRecv { wm_p, .. }
                    | ResolvedArm::InboxSend { wm_p, .. }
                    | ResolvedArm::InboxViewSend { wm_p, .. }
                    | ResolvedArm::SignalWait { wm_p, .. }
                    | ResolvedArm::WaitQueueCommit { wm_p, .. }
                    | ResolvedArm::WatchWaitChange { wm_p, .. }
                    | ResolvedArm::PingPongWaitChunk { wm_p, .. }
                    | ResolvedArm::PoolAlloc { wm_p, .. }
                    | ResolvedArm::ByteRingWaitRead { wm_p, .. }
                    | ResolvedArm::ByteRingViewWaitRead { wm_p, .. }
                    | ResolvedArm::ByteRingWaitWrite { wm_p, .. }
                    | ResolvedArm::ByteRingViewWaitWrite { wm_p, .. } => wm_p,
                }
            }
        }

        // Single outer critical section covers poll + timeout check + mask/park atomically.
        let pm = self.next_tmp();
        self.emit_line(format!("{pm} = {call_kw} i32 @__syn_crit_enter()"));

        let is_null = self.next_tmp();
        self.emit_line(format!("{is_null} = icmp eq ptr %__task, null"));
        let has_slot_lbl = self.next_label("wait_has_slot");
        let no_slot_lbl = self.next_label("wait_no_slot");
        let bit_ready_lbl = self.next_label("wait_bit_ready");
        self.emit_term(format!(
            "br i1 {is_null}, label %{no_slot_lbl}, label %{has_slot_lbl}"
        ));

        self.start_block(&has_slot_lbl);
        let sl_p = self.next_tmp();
        let sl8 = self.next_tmp();
        let sl32 = self.next_tmp();
        let sh = self.next_tmp();
        let tbit = self.next_tmp();
        self.emit_line(format!(
            "{sl_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 9"
        ));
        self.emit_line(format!("{sl8} = load i8, ptr {sl_p}, align 1"));
        self.emit_line(format!("{sl32} = zext i8 {sl8} to i32"));
        self.emit_line(format!("{sh} = and i32 {sl32}, 31"));
        self.emit_line(format!("{tbit} = shl i32 1, {sh}"));
        self.emit_term(format!("br label %{bit_ready_lbl}"));

        self.start_block(&no_slot_lbl);
        self.emit_term(format!("br label %{bit_ready_lbl}"));

        self.start_block(&bit_ready_lbl);
        let task_bit = self.next_tmp();
        self.emit_line(format!(
            "{task_bit} = phi i32 [ {tbit}, %{has_slot_lbl} ], [ 0, %{no_slot_lbl} ]"
        ));
        let clr_mask = self.next_tmp();
        self.emit_line(format!("{clr_mask} = xor i32 {task_bit}, -1"));

        let mut resolved = Vec::with_capacity(arms.len());
        for arm in arms {
            let r = match &arm.op {
                MirWaitArmOp::InboxRecv {
                    target,
                    elem_ty,
                    cap,
                    dst_val,
                    dst_res,
                } => {
                    let pk = MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    };
                    let id = sanitize_ident(&pk.mangle_name());
                    let pty = format!("%prim.{id}");
                    let qptr = self.resolve_prim_ptr(target)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 3"
                    ));
                    ResolvedArm::InboxRecv {
                        qptr,
                        wm_p,
                        id,
                        elem_ty: elem_ty.clone(),
                        dst_val: *dst_val,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::InboxViewRecv {
                    view,
                    elem_ty,
                    dst_val,
                    dst_res,
                } => {
                    let (hdr, items_base, cap32) = self.emit_inbox_view_parts(view)?;
                    let key = sanitize_ident(&elem_ty.mangle_name());
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 3"
                    ));
                    ResolvedArm::InboxViewRecv {
                        hdr,
                        items_base,
                        cap32,
                        wm_p,
                        key,
                        elem_ty: elem_ty.clone(),
                        dst_val: *dst_val,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::InboxSend {
                    target,
                    elem_ty,
                    cap,
                    val_var,
                    dst_res,
                } => {
                    let pk = MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    };
                    let id = sanitize_ident(&pk.mangle_name());
                    let pty = format!("%prim.{id}");
                    let qptr = self.resolve_prim_ptr(target)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 4"
                    ));
                    ResolvedArm::InboxSend {
                        qptr,
                        wm_p,
                        id,
                        elem_ty: elem_ty.clone(),
                        val_var: *val_var,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::InboxViewSend {
                    view,
                    elem_ty,
                    val_var,
                    dst_res,
                } => {
                    let (hdr, items_base, cap32) = self.emit_inbox_view_parts(view)?;
                    let key = sanitize_ident(&elem_ty.mangle_name());
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 4"
                    ));
                    ResolvedArm::InboxViewSend {
                        hdr,
                        items_base,
                        cap32,
                        wm_p,
                        key,
                        elem_ty: elem_ty.clone(),
                        val_var: *val_var,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::SignalWait { target, dst_res } => {
                    let id = sanitize_ident(&MirPrimKind::Signal.mangle_name());
                    let sptr = self.resolve_prim_ptr(target)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds %prim.{id}, ptr {sptr}, i32 0, i32 2"
                    ));
                    ResolvedArm::SignalWait {
                        sptr,
                        wm_p,
                        id,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::WaitQueueCommit { target, dst_res } => {
                    let wptr = self.resolve_prim_ptr(target)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds %prim.wait_queue, ptr {wptr}, i32 0, i32 0"
                    ));
                    ResolvedArm::WaitQueueCommit {
                        wm_p,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::WatchWaitChange {
                    target,
                    elem_ty,
                    last_gen_var,
                    dst_val,
                    dst_res,
                    ..
                } => {
                    let pk = MirPrimKind::Watch {
                        elem: Box::new(elem_ty.clone()),
                    };
                    let id = sanitize_ident(&pk.mangle_name());
                    let pty = format!("%prim.{id}");
                    let wptr = self.resolve_prim_ptr(target)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds {pty}, ptr {wptr}, i32 0, i32 2"
                    ));
                    ResolvedArm::WatchWaitChange {
                        wptr,
                        wm_p,
                        pty,
                        elem_ty: elem_ty.clone(),
                        last_gen_var: *last_gen_var,
                        dst_val: *dst_val,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::PingPongWaitChunk {
                    target,
                    elem_ty,
                    cap,
                    dst_slice,
                } => {
                    let pk = MirPrimKind::PingPong {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    };
                    let id = sanitize_ident(&pk.mangle_name());
                    let pty = format!("%prim.{id}");
                    let pptr = self.resolve_prim_ptr(target)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds {pty}, ptr {pptr}, i32 0, i32 4"
                    ));
                    ResolvedArm::PingPongWaitChunk {
                        pptr,
                        wm_p,
                        id,
                        cap: *cap,
                        dst_slice: *dst_slice,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::PoolAlloc {
                    target,
                    elem_ty,
                    cap,
                    dst_val,
                    dst_res,
                } => {
                    let pk = MirPrimKind::Pool {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    };
                    let id = sanitize_ident(&pk.mangle_name());
                    let pptr = self.resolve_prim_ptr(target)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds %prim.pool_hdr, ptr {pptr}, i32 0, i32 1"
                    ));
                    ResolvedArm::PoolAlloc {
                        pptr,
                        wm_p,
                        id,
                        dst_val: *dst_val,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::ByteRingWaitRead {
                    target,
                    cap,
                    dst_val,
                    dst_res,
                } => {
                    let pk = MirPrimKind::ByteRing { cap: *cap };
                    let id = sanitize_ident(&pk.mangle_name());
                    let pty = format!("%prim.{id}");
                    let rptr = self.resolve_prim_ptr(target)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds {pty}, ptr {rptr}, i32 0, i32 6"
                    ));
                    ResolvedArm::ByteRingWaitRead {
                        rptr,
                        wm_p,
                        id,
                        dst_val: *dst_val,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::ByteRingViewWaitRead {
                    view,
                    dst_val,
                    dst_res,
                } => {
                    let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds %prim.byte_ring_hdr, ptr {hdr}, i32 0, i32 6"
                    ));
                    ResolvedArm::ByteRingViewWaitRead {
                        hdr,
                        cap32,
                        wm_p,
                        dst_val: *dst_val,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::ByteRingWaitWrite {
                    target,
                    cap,
                    dst_val,
                    dst_res,
                } => {
                    let pk = MirPrimKind::ByteRing { cap: *cap };
                    let id = sanitize_ident(&pk.mangle_name());
                    let pty = format!("%prim.{id}");
                    let rptr = self.resolve_prim_ptr(target)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds {pty}, ptr {rptr}, i32 0, i32 6"
                    ));
                    ResolvedArm::ByteRingWaitWrite {
                        rptr,
                        wm_p,
                        id,
                        dst_val: *dst_val,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
                MirWaitArmOp::ByteRingViewWaitWrite {
                    view,
                    dst_val,
                    dst_res,
                } => {
                    let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                    let wm_p = self.next_tmp();
                    self.emit_line(format!(
                        "{wm_p} = getelementptr inbounds %prim.byte_ring_hdr, ptr {hdr}, i32 0, i32 6"
                    ));
                    ResolvedArm::ByteRingViewWaitWrite {
                        hdr,
                        cap32,
                        wm_p,
                        dst_val: *dst_val,
                        dst_res: *dst_res,
                        target_bb: arm.target_bb,
                    }
                }
            };
            resolved.push(r);
        }

        // Clear any leftover task bit before polling so stale bits never accumulate across arms.
        // (`WaitQueueCommit` already armed its bit inside `prepare_wait()`, so leave it intact!)
        for r in &resolved {
            if matches!(r, ResolvedArm::WaitQueueCommit { .. }) {
                continue;
            }
            let wm_p = r.wm_ptr();
            let old_m = self.next_tmp();
            let new_m = self.next_tmp();
            self.emit_line(format!(
                "{old_m} = load volatile i32, ptr {wm_p}, align {i32_al}"
            ));
            self.emit_line(format!("{new_m} = and i32 {old_m}, {clr_mask}"));
            self.emit_line(format!(
                "store volatile i32 {new_m}, ptr {wm_p}, align {i32_al}"
            ));
        }

        let after_polls_lbl = self.next_label("wait_after_polls");
        for (idx, arm) in resolved.iter().enumerate() {
            let next_lbl = if idx + 1 < resolved.len() {
                self.next_label(&format!("wait_arm_{}", idx + 1))
            } else {
                after_polls_lbl.clone()
            };
            let ok_lbl = self.next_label(&format!("wait_ok_{idx}"));
            match arm {
                ResolvedArm::InboxRecv {
                    qptr,
                    id,
                    elem_ty,
                    dst_val,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let ety = self.ctx.llvm_ty(elem_ty);
                    let (tmp_elem, al) = self.emit_temp_alloca(&ety);
                    self.emit_line(format!(
                        "store {ety} zeroinitializer, ptr {tmp_elem}, align {al}"
                    ));
                    let ok = self.next_tmp();
                    self.emit_line(format!(
                        "{ok} = {call_kw} i8 @__syn_q_pop_{id}(ptr {qptr}, ptr {tmp_elem}, ptr %__task)"
                    ));
                    let got = self.next_tmp();
                    self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    let val = self.next_tmp();
                    self.emit_line(format!("{val} = load {ety}, ptr {tmp_elem}, align {al}"));
                    if let Some(vvid) = dst_val {
                        let vp = self.var_ptr(*vvid);
                        self.emit_line(format!("store {ety} {val}, ptr {vp}, align 1"));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let r0 = self.next_tmp();
                        let r1 = self.next_tmp();
                        self.emit_line(format!("{r0} = insertvalue {rty} poison, {ety} {val}, 0"));
                        self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 0, 1"));
                        self.emit_line(format!("store {rty} {r1}, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::InboxViewRecv {
                    hdr,
                    items_base,
                    cap32,
                    key,
                    elem_ty,
                    dst_val,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let ety = self.ctx.llvm_ty(elem_ty);
                    let (tmp_elem, al) = self.emit_temp_alloca(&ety);
                    self.emit_line(format!(
                        "store {ety} zeroinitializer, ptr {tmp_elem}, align {al}"
                    ));
                    let ok = self.next_tmp();
                    self.emit_line(format!(
                        "{ok} = {call_kw} i8 @__syn_q_view_pop_{key}(ptr {hdr}, ptr {items_base}, i32 {cap32}, ptr {tmp_elem}, ptr %__task)"
                    ));
                    let got = self.next_tmp();
                    self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    let val = self.next_tmp();
                    self.emit_line(format!("{val} = load {ety}, ptr {tmp_elem}, align {al}"));
                    if let Some(vvid) = dst_val {
                        let vp = self.var_ptr(*vvid);
                        self.emit_line(format!("store {ety} {val}, ptr {vp}, align 1"));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let r0 = self.next_tmp();
                        let r1 = self.next_tmp();
                        self.emit_line(format!("{r0} = insertvalue {rty} poison, {ety} {val}, 0"));
                        self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 0, 1"));
                        self.emit_line(format!("store {rty} {r1}, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::InboxSend {
                    qptr,
                    id,
                    elem_ty,
                    val_var,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let vp = self.var_ptr(*val_var);
                    let ety = self.ctx.llvm_ty(elem_ty);
                    let is_lease = matches!(elem_ty, MirType::Lease(_));
                    let push_fn = if is_lease {
                        format!("@__syn_q_push_keep_{id}")
                    } else {
                        format!("@__syn_q_push_{id}")
                    };
                    let v = self.next_tmp();
                    self.emit_line(format!("{v} = load {ety}, ptr {vp}, align 1"));
                    let ok = self.next_tmp();
                    self.emit_line(format!(
                        "{ok} = {call_kw} i8 {push_fn}(ptr {qptr}, {ety} {v}, i32 -1)"
                    ));
                    let ok_b = self.next_tmp();
                    self.emit_line(format!("{ok_b} = icmp ne i8 {ok}, 0"));
                    self.emit_term(format!("br i1 {ok_b}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    if is_lease {
                        let vp_ph = self.next_tmp();
                        self.emit_line(format!(
                            "{vp_ph} = getelementptr inbounds {ety}, ptr {vp}, i32 0, i32 1"
                        ));
                        self.emit_line(format!("store ptr null, ptr {vp_ph}, align {ptr_al}"));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        self.emit_line(format!("store {rty} zeroinitializer, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::InboxViewSend {
                    hdr,
                    items_base,
                    cap32,
                    key,
                    elem_ty,
                    val_var,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let vp = self.var_ptr(*val_var);
                    let ety = self.ctx.llvm_ty(elem_ty);
                    let is_lease = matches!(elem_ty, MirType::Lease(_));
                    let push_fn = if is_lease {
                        format!("@__syn_q_view_push_keep_{key}")
                    } else {
                        format!("@__syn_q_view_push_{key}")
                    };
                    let v = self.next_tmp();
                    self.emit_line(format!("{v} = load {ety}, ptr {vp}, align 1"));
                    let ok = self.next_tmp();
                    self.emit_line(format!(
                        "{ok} = {call_kw} i8 {push_fn}(ptr {hdr}, ptr {items_base}, i32 {cap32}, {ety} {v}, i32 -1)"
                    ));
                    let ok_b = self.next_tmp();
                    self.emit_line(format!("{ok_b} = icmp ne i8 {ok}, 0"));
                    self.emit_term(format!("br i1 {ok_b}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    if is_lease {
                        let vp_ph = self.next_tmp();
                        self.emit_line(format!(
                            "{vp_ph} = getelementptr inbounds {ety}, ptr {vp}, i32 0, i32 1"
                        ));
                        self.emit_line(format!("store ptr null, ptr {vp_ph}, align {ptr_al}"));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        self.emit_line(format!("store {rty} zeroinitializer, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::SignalWait {
                    sptr,
                    id,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let ok = self.next_tmp();
                    self.emit_line(format!(
                        "{ok} = {call_kw} i8 @__syn_sig_poll_{id}(ptr {sptr}, ptr %__task)"
                    ));
                    let got = self.next_tmp();
                    self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        self.emit_line(format!("store {rty} zeroinitializer, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::WaitQueueCommit {
                    wm_p,
                    dst_res,
                    target_bb,
                } => {
                    let cur_m = self.next_tmp();
                    let masked = self.next_tmp();
                    let cleared = self.next_tmp();
                    let not_null = self.next_tmp();
                    let got = self.next_tmp();
                    self.emit_line(format!(
                        "{cur_m} = load volatile i32, ptr {wm_p}, align {i32_al}"
                    ));
                    self.emit_line(format!("{masked} = and i32 {cur_m}, {task_bit}"));
                    self.emit_line(format!("{cleared} = icmp eq i32 {masked}, 0"));
                    self.emit_line(format!("{not_null} = xor i1 {is_null}, true"));
                    self.emit_line(format!("{got} = and i1 {not_null}, {cleared}"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        self.emit_line(format!("store {rty} zeroinitializer, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::WatchWaitChange {
                    wptr,
                    pty,
                    elem_ty,
                    last_gen_var,
                    dst_val,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let gp = self.next_tmp();
                    let cur_gen = self.next_tmp();
                    let lgp = self.var_ptr(*last_gen_var);
                    let last_gen = self.next_tmp();
                    let chg = self.next_tmp();
                    self.emit_line(format!(
                        "{gp} = getelementptr inbounds {pty}, ptr {wptr}, i32 0, i32 1"
                    ));
                    self.emit_line(format!("{cur_gen} = load i32, ptr {gp}, align {i32_al}"));
                    self.emit_line(format!("{last_gen} = load i32, ptr {lgp}, align 1"));
                    self.emit_line(format!("{chg} = icmp ne i32 {cur_gen}, {last_gen}"));
                    self.emit_term(format!("br i1 {chg}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    self.emit_line(format!("store i32 {cur_gen}, ptr {lgp}, align 1"));
                    if dst_val.is_some() || dst_res.is_some() {
                        let ety = self.ctx.llvm_ty(elem_ty);
                        let vp_src = self.next_tmp();
                        let val = self.next_tmp();
                        self.emit_line(format!(
                            "{vp_src} = getelementptr inbounds {pty}, ptr {wptr}, i32 0, i32 0"
                        ));
                        self.emit_line(format!("{val} = load {ety}, ptr {vp_src}, align 1"));
                        if let Some(vvid) = dst_val {
                            let vp = self.var_ptr(*vvid);
                            self.emit_line(format!("store {ety} {val}, ptr {vp}, align 1"));
                        }
                        if let Some(rvid) = dst_res {
                            let rp = self.var_ptr(*rvid);
                            let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                            let r0 = self.next_tmp();
                            let r1 = self.next_tmp();
                            self.emit_line(format!(
                                "{r0} = insertvalue {rty} poison, {ety} {val}, 0"
                            ));
                            self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 0, 1"));
                            self.emit_line(format!("store {rty} {r1}, ptr {rp}, align 1"));
                        }
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::PingPongWaitChunk {
                    pptr,
                    id,
                    cap,
                    dst_slice,
                    target_bb,
                    ..
                } => {
                    let chunk_ptr = self.next_tmp();
                    self.emit_line(format!(
                        "{chunk_ptr} = {call_kw} ptr @__syn_pp_poll_{id}(ptr {pptr})"
                    ));
                    let got = self.next_tmp();
                    self.emit_line(format!("{got} = icmp ne ptr {chunk_ptr}, null"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    let slice_ty = format!("{{ ptr, {usize_ty} }}");
                    let s0 = self.next_tmp();
                    let s1 = self.next_tmp();
                    self.emit_line(format!(
                        "{s0} = insertvalue {slice_ty} poison, ptr {chunk_ptr}, 0"
                    ));
                    self.emit_line(format!(
                        "{s1} = insertvalue {slice_ty} {s0}, {usize_ty} {cap}, 1"
                    ));
                    let dp = self.var_ptr(*dst_slice);
                    self.emit_line(format!("store {slice_ty} {s1}, ptr {dp}, align 1"));
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::PoolAlloc {
                    pptr,
                    id,
                    dst_val,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let (tmp, al) = self.emit_temp_alloca("{ ptr, ptr, i8 }");
                    self.emit_line(format!(
                        "store {{ ptr, ptr, i8 }} zeroinitializer, ptr {tmp}, align {al}"
                    ));
                    let ok = self.next_tmp();
                    self.emit_line(format!(
                        "{ok} = {call_kw} i8 @__syn_pool_try_alloc_raw_{id}(ptr {pptr}, ptr {tmp})"
                    ));
                    let got = self.next_tmp();
                    self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    let ls = self.next_tmp();
                    self.emit_line(format!(
                        "{ls} = load {{ ptr, ptr, i8 }}, ptr {tmp}, align {al}"
                    ));
                    if let Some(vid) = dst_val {
                        let vp = self.var_ptr(*vid);
                        self.emit_line(format!("store {{ ptr, ptr, i8 }} {ls}, ptr {vp}, align 1"));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let r0 = self.next_tmp();
                        let r1 = self.next_tmp();
                        self.emit_line(format!(
                            "{r0} = insertvalue {rty} poison, {{ ptr, ptr, i8 }} {ls}, 0"
                        ));
                        self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 0, 1"));
                        self.emit_line(format!("store {rty} {r1}, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::ByteRingWaitRead {
                    rptr,
                    id,
                    dst_val,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let slice_ty = format!("{{ ptr, {usize_ty} }}");
                    let sl_val = self.next_tmp();
                    self.emit_line(format!(
                        "{sl_val} = {call_kw} {slice_ty} @__syn_ring_poll_read_{id}(ptr {rptr}, ptr %__task)"
                    ));
                    let sl_len = self.next_tmp();
                    self.emit_line(format!("{sl_len} = extractvalue {slice_ty} {sl_val}, 1"));
                    let got = self.next_tmp();
                    self.emit_line(format!("{got} = icmp ne {usize_ty} {sl_len}, 0"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    if let Some(vid) = dst_val {
                        let dp = self.var_ptr(*vid);
                        self.emit_line(format!("store {slice_ty} {sl_val}, ptr {dp}, align 1"));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let res0 = self.next_tmp();
                        self.emit_line(format!(
                            "{res0} = insertvalue {rty} zeroinitializer, {slice_ty} {sl_val}, 0"
                        ));
                        self.emit_line(format!("store {rty} {res0}, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::ByteRingViewWaitRead {
                    hdr,
                    cap32,
                    dst_val,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let slice_ty = format!("{{ ptr, {usize_ty} }}");
                    let sl_val = self.next_tmp();
                    self.emit_line(format!(
                        "{sl_val} = {call_kw} {slice_ty} @__syn_ring_view_poll_read(ptr {hdr}, i32 {cap32}, ptr %__task)"
                    ));
                    let sl_len = self.next_tmp();
                    self.emit_line(format!("{sl_len} = extractvalue {slice_ty} {sl_val}, 1"));
                    let got = self.next_tmp();
                    self.emit_line(format!("{got} = icmp ne {usize_ty} {sl_len}, 0"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    if let Some(vid) = dst_val {
                        let dp = self.var_ptr(*vid);
                        self.emit_line(format!("store {slice_ty} {sl_val}, ptr {dp}, align 1"));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let res0 = self.next_tmp();
                        self.emit_line(format!(
                            "{res0} = insertvalue {rty} zeroinitializer, {slice_ty} {sl_val}, 0"
                        ));
                        self.emit_line(format!("store {rty} {res0}, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::ByteRingWaitWrite {
                    rptr,
                    id,
                    dst_val,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let fs = self.next_tmp();
                    self.emit_line(format!(
                        "{fs} = {call_kw} i32 @__syn_ring_free_space_{id}(ptr {rptr})"
                    ));
                    let got = self.next_tmp();
                    self.emit_line(format!("{got} = icmp ugt i32 {fs}, 0"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    let fs_u = self.coerce(SsaVal::new("i32".to_string(), fs), &usize_ty, false);
                    if let Some(vid) = dst_val {
                        let dp = self.var_ptr(*vid);
                        self.emit_line(format!(
                            "store {usize_ty} {}, ptr {dp}, align 1",
                            fs_u.repr
                        ));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let res0 = self.next_tmp();
                        self.emit_line(format!(
                            "{res0} = insertvalue {rty} zeroinitializer, {usize_ty} {}, 0",
                            fs_u.repr
                        ));
                        self.emit_line(format!("store {rty} {res0}, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
                ResolvedArm::ByteRingViewWaitWrite {
                    hdr,
                    cap32,
                    dst_val,
                    dst_res,
                    target_bb,
                    ..
                } => {
                    let fs = self.next_tmp();
                    self.emit_line(format!(
                        "{fs} = {call_kw} i32 @__syn_ring_view_free_space(ptr {hdr}, i32 {cap32})"
                    ));
                    let got = self.next_tmp();
                    self.emit_line(format!("{got} = icmp ugt i32 {fs}, 0"));
                    self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{next_lbl}"));

                    self.start_block(&ok_lbl);
                    let fs_u = self.coerce(SsaVal::new("i32".to_string(), fs), &usize_ty, false);
                    if let Some(vid) = dst_val {
                        let dp = self.var_ptr(*vid);
                        self.emit_line(format!(
                            "store {usize_ty} {}, ptr {dp}, align 1",
                            fs_u.repr
                        ));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let res0 = self.next_tmp();
                        self.emit_line(format!(
                            "{res0} = insertvalue {rty} zeroinitializer, {usize_ty} {}, 0",
                            fs_u.repr
                        ));
                        self.emit_line(format!("store {rty} {res0}, ptr {rp}, align 1"));
                    }
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                    self.emit_term(format!("br label %bb_{}", target_bb.0));
                }
            }
            if idx + 1 < resolved.len() {
                self.start_block(&next_lbl);
            }
        }

        // All arms missed: check optional deadline, then atomically register waiter_masks and park.
        self.start_block(&after_polls_lbl);
        let park_lbl = self.next_label("wait_park");
        let dl_ssa = if let Some(t) = timer {
            let dl_ptr = self.var_ptr(t.deadline_var);
            let dl = self.next_tmp();
            let now = self.next_tmp();
            let diff = self.next_tmp();
            let exp = self.next_tmp();
            self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 1"));
            self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
            self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
            self.emit_line(format!("{exp} = icmp sge i32 {diff}, 0"));
            let to_lbl = self.next_label("wait_to");
            self.emit_term(format!("br i1 {exp}, label %{to_lbl}, label %{park_lbl}"));

            self.start_block(&to_lbl);
            for r in &resolved {
                if let ResolvedArm::WaitQueueCommit { wm_p, .. } = r {
                    let old_m = self.next_tmp();
                    let new_m = self.next_tmp();
                    self.emit_line(format!(
                        "{old_m} = load volatile i32, ptr {wm_p}, align {i32_al}"
                    ));
                    self.emit_line(format!("{new_m} = and i32 {old_m}, {clr_mask}"));
                    self.emit_line(format!(
                        "store volatile i32 {new_m}, ptr {wm_p}, align {i32_al}"
                    ));
                }
            }
            self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
            if let Some(rvid) = t.timeout_err_res {
                let rp = self.var_ptr(rvid);
                let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                let e0 = self.next_tmp();
                self.emit_line(format!(
                    "{e0} = insertvalue {rty} zeroinitializer, i32 1, 1"
                ));
                self.emit_line(format!("store {rty} {e0}, ptr {rp}, align 1"));
            }
            if let Some((lv, lty)) = &t.release_lease_on_timeout {
                let vp = self.var_ptr(*lv);
                let ety = self.ctx.llvm_ty(lty);
                let v = self.next_tmp();
                let ph = self.next_tmp();
                let sidx = self.next_tmp();
                let vp_ph = self.next_tmp();
                self.emit_line(format!("{v} = load {ety}, ptr {vp}, align 1"));
                self.emit_line(format!("{ph} = extractvalue {ety} {v}, 1"));
                self.emit_line(format!("{sidx} = extractvalue {ety} {v}, 2"));
                self.emit_line(format!(
                    "{vp_ph} = getelementptr inbounds {ety}, ptr {vp}, i32 0, i32 1"
                ));
                self.emit_line(format!("store ptr null, ptr {vp_ph}, align {ptr_al}"));
                self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                self.emit_line(format!(
                    "{call_kw} void @__syn_pool_release(ptr {ph}, i8 {sidx}, i8 -1)"
                ));
            } else {
                self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
            }
            self.emit_term(format!("br label %bb_{}", t.timeout_bb.0));

            self.start_block(&park_lbl);
            Some(dl)
        } else {
            self.emit_term(format!("br label %{park_lbl}"));
            self.start_block(&park_lbl);
            None
        };

        let do_park_lbl = self.next_label("wait_do_park");
        let finish_yield_lbl = self.next_label("wait_ret");
        self.emit_term(format!(
            "br i1 {is_null}, label %{finish_yield_lbl}, label %{do_park_lbl}"
        ));

        self.start_block(&do_park_lbl);
        self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
        for r in &resolved {
            let wm_p = r.wm_ptr();
            let cur_m = self.next_tmp();
            let next_m = self.next_tmp();
            self.emit_line(format!(
                "{cur_m} = load volatile i32, ptr {wm_p}, align {i32_al}"
            ));
            self.emit_line(format!("{next_m} = or i32 {cur_m}, {task_bit}"));
            self.emit_line(format!(
                "store volatile i32 {next_m}, ptr {wm_p}, align {i32_al}"
            ));
        }
        let st_p = self.next_tmp();
        let ht_p = self.next_tmp();
        let wo_p = self.next_tmp();
        self.emit_line(format!(
            "{st_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 2"
        ));
        self.emit_line(format!(
            "{ht_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 7"
        ));
        self.emit_line(format!(
            "{wo_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 12"
        ));
        self.emit_line(format!("store i8 3, ptr {st_p}, align 1"));
        self.emit_line(format!("store ptr null, ptr {wo_p}, align {ptr_al}"));
        self.emit_line(format!(
            "{call_kw} void @__syn_ready_clear_task(ptr %__task)"
        ));
        if let Some(dl) = dl_ssa {
            let wt_p = self.next_tmp();
            self.emit_line(format!(
                "{wt_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 11"
            ));
            self.emit_line(format!("store i8 1, ptr {ht_p}, align 1"));
            self.emit_line(format!("store i32 {dl}, ptr {wt_p}, align 4"));
            self.emit_line(format!("{call_kw} void @__syn_note_deadline(i32 {dl})"));
        } else {
            self.emit_line(format!("store i8 0, ptr {ht_p}, align 1"));
        }
        self.emit_term(format!("br label %{finish_yield_lbl}"));

        self.start_block(&finish_yield_lbl);
        self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
        self.store_coro_pc(poll_bb.0 as i32);
        self.emit_term("ret i32 1");
        Ok(())
    }

    // ── Boot Helpers ─────────────────────────────────────────────────────────

    pub(super) fn emit_boot_run_yielding(
        &mut self,
        callee: &str,
        args: &[MirExpr],
        dst: Option<&MirCallDst>,
    ) -> Result<(), CodegenError> {
        if self.terminated {
            return Ok(());
        }
        let child_layout = self.ctx.fn_frames[callee].clone();
        let child_sig = self.ctx.fn_sigs[callee].clone();
        let callee_fn = self
            .ctx
            .module
            .functions
            .iter()
            .find(|f| f.name == callee)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(
                    "llvm-codegen",
                    format!("missing boot yielding callee `{callee}`"),
                )
            })?;
        let fsym = format!("@__syn_boot_frame_{}", sanitize_ident(callee));

        let mut eval_args = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let av = self.emit_expr(a)?;
            let pty = self.ctx.llvm_ty(&child_sig.params[i]);
            let signed = !a.ty.is_unsigned();
            eval_args.push(self.coerce(av, &pty, signed));
        }

        self.emit_line(format!(
            "store {} zeroinitializer, ptr {fsym}, align 8",
            child_layout.llvm_name
        ));
        for (i, &pvid) in callee_fn.params.iter().enumerate() {
            if let Some(&pgep) = child_layout.hoisted_var_gep.get(&pvid) {
                let pp = self.next_tmp();
                self.emit_line(format!(
                    "{pp} = getelementptr inbounds {}, ptr {fsym}, i32 0, i32 {pgep}",
                    child_layout.llvm_name
                ));
                self.emit_line(format!(
                    "store {} {}, ptr {pp}, align 4",
                    eval_args[i].ty, eval_args[i].repr
                ));
            }
        }

        let loop_lbl = self.next_label("boot_step");
        let wait_lbl = self.next_label("boot_wait");
        let sleep_lbl = self.next_label("boot_sleep");
        let done_lbl = self.next_label("boot_done");
        self.emit_term(format!("br label %{loop_lbl}"));

        let call_kw = self.call_kw();
        self.start_block(&loop_lbl);
        let st = self.next_tmp();
        self.emit_line(format!(
            "{st} = {call_kw} i32 {}(ptr {fsym}, ptr null)",
            child_sig.symbol
        ));
        let done = self.next_tmp();
        self.emit_line(format!("{done} = icmp eq i32 {st}, 2"));
        self.emit_term(format!(
            "br i1 {done}, label %{done_lbl}, label %{wait_lbl}"
        ));

        self.start_block(&wait_lbl);
        let is_wait = self.next_tmp();
        self.emit_line(format!("{is_wait} = icmp eq i32 {st}, 1"));
        self.emit_term(format!(
            "br i1 {is_wait}, label %{sleep_lbl}, label %{loop_lbl}"
        ));

        self.start_block(&sleep_lbl);
        self.emit_line(format!("{call_kw} void @__syn_idle_sleep()"));
        self.emit_term(format!("br label %{loop_lbl}"));

        self.start_block(&done_lbl);
        if let Some(d) = dst {
            if child_layout.has_ret_slot {
                let rp = self.next_tmp();
                self.emit_line(format!(
                    "{rp} = getelementptr inbounds {}, ptr {fsym}, i32 0, i32 1",
                    child_layout.llvm_name
                ));
                let rv = self.next_tmp();
                self.emit_line(format!(
                    "{rv} = load {}, ptr {rp}, align 4",
                    child_layout.ret_llvm_ty
                ));
                let dvid = match d {
                    MirCallDst::Value(v) | MirCallDst::Result(v) => *v,
                };
                let dp = self.var_ptr(dvid);
                let dty = self.ctx.llvm_ty(&self.vars[dvid.0].ty);
                let signed = !self.vars[dvid.0].ty.is_unsigned();
                let coerced = self.coerce(SsaVal::new(&child_layout.ret_llvm_ty, rv), &dty, signed);
                self.emit_line(format!(
                    "store {} {}, ptr {dp}, align 4",
                    coerced.ty, coerced.repr
                ));
            }
        }
        Ok(())
    }

    pub(super) fn emit_store_pool_hdr(&mut self, hdr_ptr: &str, cap: usize, dma_flag: u8) {
        let c = cap.max(1);
        let mask: u32 = if c >= 32 { u32::MAX } else { (1u32 << c) - 1 };
        let mask_i32 = mask as i32;
        let dma8 = dma_flag as i8;
        let cap8 = c as u8;
        let ptr_al = self.ctx.ptr_align();
        let f0 = self.next_tmp();
        let f1 = self.next_tmp();
        let f2 = self.next_tmp();
        let f3 = self.next_tmp();
        let f4 = self.next_tmp();
        self.emit_line(format!(
            "{f0} = getelementptr inbounds %prim.pool_hdr, ptr {hdr_ptr}, i32 0, i32 0"
        ));
        self.emit_line(format!("store i32 {mask_i32}, ptr {f0}, align 4"));
        self.emit_line(format!(
            "{f1} = getelementptr inbounds %prim.pool_hdr, ptr {hdr_ptr}, i32 0, i32 1"
        ));
        self.emit_line(format!("store i32 0, ptr {f1}, align 4"));
        self.emit_line(format!(
            "{f2} = getelementptr inbounds %prim.pool_hdr, ptr {hdr_ptr}, i32 0, i32 2"
        ));
        self.emit_line(format!("store ptr null, ptr {f2}, align {ptr_al}"));
        self.emit_line(format!(
            "{f3} = getelementptr inbounds %prim.pool_hdr, ptr {hdr_ptr}, i32 0, i32 3"
        ));
        self.emit_line(format!("store i8 {dma8}, ptr {f3}, align 1"));
        self.emit_line(format!(
            "{f4} = getelementptr inbounds %prim.pool_hdr, ptr {hdr_ptr}, i32 0, i32 4"
        ));
        self.emit_line(format!("store i8 {cap8}, ptr {f4}, align 1"));
    }

    pub(super) fn emit_subsys_pool_inits(
        &mut self,
        struct_name: &str,
        base_ptr: &str,
        parent_dma_safe: bool,
    ) {
        let Some(si) = self.ctx.structs.get(struct_name).cloned() else {
            return;
        };
        for f in &si.decl.fields {
            let Some(&(fidx, _)) = si.field_indices.get(&f.name) else {
                continue;
            };
            let eff_dma = parent_dma_safe || f.dma_safe;
            match &f.ty {
                MirType::Primitive(MirPrimKind::Pool { cap, .. }) => {
                    let fp = self.next_tmp();
                    self.emit_line(format!(
                        "{fp} = getelementptr inbounds {}, ptr {base_ptr}, i32 0, i32 {fidx}",
                        si.llvm_name
                    ));
                    let dma_flag = if eff_dma {
                        1u8
                    } else if self.ctx.dma_layout_structs.contains(struct_name) {
                        2u8
                    } else {
                        0u8
                    };
                    self.emit_store_pool_hdr(&fp, *cap, dma_flag);
                }
                MirType::Struct(sub_name) => {
                    let fp = self.next_tmp();
                    self.emit_line(format!(
                        "{fp} = getelementptr inbounds {}, ptr {base_ptr}, i32 0, i32 {fidx}",
                        si.llvm_name
                    ));
                    self.emit_subsys_pool_inits(sub_name, &fp, eff_dma);
                }
                _ => {}
            }
        }
    }

    pub(super) fn emit_pool_boot_inits(&mut self) {
        let prims: Vec<(String, usize, bool)> = self
            .ctx
            .module
            .primitives
            .iter()
            .filter_map(|p| {
                if !p.dead_stripped {
                    if let MirPrimKind::Pool { cap, .. } = &p.kind {
                        return Some((p.name.clone(), *cap, p.dma_safe));
                    }
                }
                None
            })
            .collect();
        for (name, cap, dma_safe) in prims {
            let gsym = format!("@__syn_prim_{}", sanitize_ident(&name));
            self.emit_store_pool_hdr(&gsym, cap, u8::from(dma_safe));
        }
        let subs: Vec<(String, String, bool)> = self
            .ctx
            .module
            .subsystems
            .iter()
            .map(|s| {
                (
                    s.struct_name.clone(),
                    format!("@__zev_subsys_{}", sanitize_ident(&s.name)),
                    s.dma_safe,
                )
            })
            .collect();
        for (sname, gsym, dma_safe) in subs {
            self.emit_subsys_pool_inits(&sname, &gsym, dma_safe);
        }
    }

    pub(super) fn emit_boot_spawn_activate(
        &mut self,
        slot_id: usize,
        args: &[MirExpr],
    ) -> Result<(), CodegenError> {
        if self.terminated {
            return Ok(());
        }
        let sp = self
            .ctx
            .module
            .spawned_tasks
            .iter()
            .find(|s| s.slot_id == slot_id)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(
                    "llvm-codegen",
                    format!("missing spawned task slot {slot_id}"),
                )
            })?;
        let bp = self
            .ctx
            .module
            .task_blueprints
            .iter()
            .find(|b| b.name == sp.blueprint_name)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(
                    "llvm-codegen",
                    format!("missing task blueprint `{}`", sp.blueprint_name),
                )
            })?;
        let bp_layout = self.ctx.bp_frames[&sp.blueprint_name].clone();
        let fsym = format!("@__syn_task_frame_{slot_id}");
        let step_sym = format!("@__syn_task_step_{}", sanitize_ident(&sp.blueprint_name));

        let mut eval_args = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let av = self.emit_expr(a)?;
            let pvid = bp.params[i];
            let pty = self.ctx.llvm_ty(&bp.vars[pvid.0].ty);
            let signed = !a.ty.is_unsigned();
            eval_args.push(self.coerce(av, &pty, signed));
        }

        self.emit_line(format!(
            "store {} zeroinitializer, ptr {fsym}, align 8",
            bp_layout.llvm_name
        ));
        for (i, &pvid) in bp.params.iter().enumerate() {
            if let Some(&pgep) = bp_layout.hoisted_var_gep.get(&pvid) {
                let pp = self.next_tmp();
                self.emit_line(format!(
                    "{pp} = getelementptr inbounds {}, ptr {fsym}, i32 0, i32 {pgep}",
                    bp_layout.llvm_name
                ));
                self.emit_line(format!(
                    "store {} {}, ptr {pp}, align 4",
                    eval_args[i].ty, eval_args[i].repr
                ));
            }
        }

        let prio = sp.priority.as_u8();
        let core = sp.core_pin.map_or(-1i8, |c| c as i8);
        let fn_ptr_ty = self.ctx.spec.isa.fn_ptr_llvm_ty();
        let call_kw = self.call_kw();
        self.emit_line(format!(
            "{call_kw} void @__syn_activate_task(i32 {slot_id}, {fn_ptr_ty} {step_sym}, ptr {fsym}, i8 {prio}, i8 {core})"
        ));
        Ok(())
    }

    // ── Statements & Dual-Exit Envelopes ─────────────────────────────────────

    pub(super) fn ensure_terminated_void(&mut self) {
        if !self.terminated {
            self.emit_term("ret void");
        }
    }

    pub(super) fn ensure_terminated_default(&mut self) {
        if self.terminated {
            return;
        }
        if self.error_spec.has_error() {
            let env_ty = MirType::Result {
                ok: Box::new(self.ret_ty.clone()),
                err: self.error_spec.err_type().map(Box::new),
            };
            let lty = self.ctx.llvm_ty(&env_ty);
            self.emit_term(format!("ret {lty} zeroinitializer"));
        } else if self.ret_ty.is_void() {
            self.emit_term("ret void");
        } else {
            let lty = self.ctx.llvm_ty(&self.ret_ty);
            self.emit_term(format!("ret {lty} zeroinitializer"));
        }
    }

    pub(super) fn build_return_envelope(
        &mut self,
        val_opt: Option<&MirExpr>,
    ) -> Result<SsaVal, CodegenError> {
        if self.error_spec.has_error() {
            let env_ty = MirType::Result {
                ok: Box::new(self.ret_ty.clone()),
                err: self.error_spec.err_type().map(Box::new),
            };
            let lty = self.ctx.llvm_ty(&env_ty);
            let ok_lty = if self.ret_ty.is_void() {
                "i32".to_string()
            } else {
                self.ctx.llvm_ty(&self.ret_ty)
            };
            let ok_val = if let Some(ve) = val_opt {
                let raw = self.emit_expr(ve)?;
                self.coerce(raw, &ok_lty, !ve.ty.is_unsigned())
            } else {
                SsaVal::new(&ok_lty, "zeroinitializer")
            };
            let r0 = self.next_tmp();
            let r1 = self.next_tmp();
            self.emit_line(format!(
                "{r0} = insertvalue {lty} poison, {} {}, 0",
                ok_val.ty, ok_val.repr
            ));
            self.emit_line(format!("{r1} = insertvalue {lty} {r0}, i32 0, 1"));
            Ok(SsaVal::new(lty, r1))
        } else {
            let lty = self.ctx.llvm_ty(&self.ret_ty);
            if let Some(ve) = val_opt {
                let raw = self.emit_expr(ve)?;
                Ok(self.coerce(raw, &lty, !ve.ty.is_unsigned()))
            } else {
                Ok(SsaVal::new(lty, "zeroinitializer"))
            }
        }
    }

    pub(super) fn build_error_envelope(
        &mut self,
        reason: Option<&MirExpr>,
        partial: Option<&MirExpr>,
    ) -> Result<SsaVal, CodegenError> {
        let env_ty = MirType::Result {
            ok: Box::new(self.ret_ty.clone()),
            err: self.error_spec.err_type().map(Box::new),
        };
        let lty = self.ctx.llvm_ty(&env_ty);
        let ok_lty = if self.ret_ty.is_void() {
            "i32".to_string()
        } else {
            self.ctx.llvm_ty(&self.ret_ty)
        };
        let pval = if let Some(pe) = partial {
            let raw = self.emit_expr(pe)?;
            self.coerce(raw, &ok_lty, !pe.ty.is_unsigned())
        } else {
            SsaVal::new(&ok_lty, "zeroinitializer")
        };

        let err_tag = if let Some(re) = reason {
            let rv = self.emit_expr(re)?;
            let r32 = self.coerce(rv, "i32", !re.ty.is_unsigned());
            let plus1 = self.next_tmp();
            self.emit_line(format!("{plus1} = add i32 {}, 1", r32.repr));
            let is_z = self.next_tmp();
            let safe = self.next_tmp();
            self.emit_line(format!("{is_z} = icmp eq i32 {plus1}, 0"));
            self.emit_line(format!("{safe} = select i1 {is_z}, i32 1, i32 {plus1}"));
            safe
        } else {
            "1".to_string()
        };

        let r0 = self.next_tmp();
        let r1 = self.next_tmp();
        self.emit_line(format!(
            "{r0} = insertvalue {lty} poison, {} {}, 0",
            pval.ty, pval.repr
        ));
        self.emit_line(format!("{r1} = insertvalue {lty} {r0}, i32 {err_tag}, 1"));
        Ok(SsaVal::new(lty, r1))
    }

    pub(super) fn emit_stmt(&mut self, stmt: &MirStmt) -> Result<(), CodegenError> {
        if self.terminated {
            return Ok(());
        }
        match stmt {
            MirStmt::Assign { target, op, value } => {
                let (ptr, raw_ty, packed) = self.emit_place_ptr(target)?;
                let host_ty = raw_ty.host_value_type();
                let lty = self.ctx.llvm_ty(&host_ty);
                let dst_signed = !host_ty.is_unsigned();
                let rhs = self.emit_expr(value)?;
                let rhs_signed = if matches!(op, MirAssignOp::ShlAssign | MirAssignOp::ShrAssign) {
                    false
                } else {
                    !value.ty.is_unsigned()
                };
                let rhs_c = self.coerce(rhs, &lty, rhs_signed);

                let final_host = if *op == MirAssignOp::Assign {
                    rhs_c
                } else {
                    let cur = self.load_from_place_ptr(&ptr, &raw_ty, packed);
                    self.emit_compound_op(*op, &cur, &rhs_c, dst_signed)
                };
                self.store_to_place_ptr(&ptr, &raw_ty, packed, final_host);
            }
            MirStmt::Expr(e) => {
                let _ = self.emit_expr(e)?;
            }
            MirStmt::PrimOp(pop) => self.emit_prim_nonblocking(pop)?,
            MirStmt::Intrinsic(iop) => self.emit_intrinsic(iop)?,
            MirStmt::MmioWrite {
                base,
                offset,
                elem_bits,
                index,
                array_len,
                value,
            } => {
                let (p, lty, align) = self.emit_mmio_elem_ptr(
                    base,
                    *offset,
                    *elem_bits,
                    index.as_deref(),
                    *array_len,
                )?;
                let vv = self.emit_expr(value)?;
                let vc = self.coerce(vv, lty, !value.ty.is_unsigned());
                self.emit_line(format!(
                    "store volatile {lty} {}, ptr {p}, align {align}",
                    vc.repr
                ));
            }
            MirStmt::SetIrqSelector { symbol, value } => {
                let sym = sanitize_ident(symbol.trim_start_matches('@'));
                self.emit_line(format!("store i8 {value}, ptr @{sym}, align 1"));
            }
            MirStmt::EnterCritical => {
                let call_kw = self.call_kw();
                let pm = self.next_tmp();
                self.emit_line(format!("{pm} = {call_kw} i32 @__syn_crit_enter()"));
                self.emit_line(format!("store i32 {pm}, ptr %__crit_pm, align 4"));
            }
            MirStmt::ExitCritical => {
                let call_kw = self.call_kw();
                let pm = self.next_tmp();
                self.emit_line(format!("{pm} = load i32, ptr %__crit_pm, align 4"));
                self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
            }
            MirStmt::EndInboxTurn => {
                if self.coroutine.is_some() {
                    let call_kw = self.call_kw();
                    self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
                }
            }
            MirStmt::BoundedForLatch => {}
            MirStmt::ZeroTaskLocals => {
                self.emit_zero_task_locals();
            }
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let cv = self.emit_expr(cond)?;
                let c1 = self.coerce_to_i1(cv);
                let then_lbl = self.next_label("if_then");
                let else_lbl = self.next_label("if_else");
                let end_lbl = self.next_label("if_end");
                if else_body.is_empty() {
                    self.emit_term(format!("br i1 {c1}, label %{then_lbl}, label %{end_lbl}"));
                    self.start_block(&then_lbl);
                    for s in then_body {
                        self.emit_stmt(s)?;
                    }
                    self.emit_term(format!("br label %{end_lbl}"));
                    self.start_block(&end_lbl);
                } else {
                    self.emit_term(format!("br i1 {c1}, label %{then_lbl}, label %{else_lbl}"));
                    self.start_block(&then_lbl);
                    for s in then_body {
                        self.emit_stmt(s)?;
                    }
                    self.emit_term(format!("br label %{end_lbl}"));
                    self.start_block(&else_lbl);
                    for s in else_body {
                        self.emit_stmt(s)?;
                    }
                    self.emit_term(format!("br label %{end_lbl}"));
                    self.start_block(&end_lbl);
                }
            }
            MirStmt::While { cond, body } => {
                let cond_lbl = self.next_label("wh_cond");
                let body_lbl = self.next_label("wh_body");
                let end_lbl = self.next_label("wh_end");
                self.emit_term(format!("br label %{cond_lbl}"));
                self.start_block(&cond_lbl);
                let cv = self.emit_expr(cond)?;
                let c1 = self.coerce_to_i1(cv);
                self.emit_term(format!("br i1 {c1}, label %{body_lbl}, label %{end_lbl}"));
                self.start_block(&body_lbl);
                self.loop_targets.push((cond_lbl.clone(), end_lbl.clone()));
                for s in body {
                    self.emit_stmt(s)?;
                }
                self.loop_targets.pop();
                self.emit_term(format!("br label %{cond_lbl}"));
                self.start_block(&end_lbl);
            }
            MirStmt::Switch {
                discr,
                arms,
                default,
            } => {
                let dv = self.emit_expr(discr)?;
                let sw_ty = if llvm_int_bits(&dv.ty) > 32 {
                    "i64"
                } else {
                    "i32"
                };
                let dc = self.coerce(dv, sw_ty, !discr.ty.is_unsigned());
                let def_lbl = self.next_label("sw_def");
                let end_lbl = self.next_label("sw_end");
                let mut arm_lbls = Vec::new();
                let mut cases = String::new();
                for (pat, _) in arms {
                    let al = self.next_label("sw_arm");
                    let pv = self.pattern_int(pat);
                    let _ = writeln!(cases, "    {sw_ty} {pv}, label %{al}");
                    arm_lbls.push(al);
                }
                self.emit_term(format!(
                    "switch {sw_ty} {}, label %{def_lbl} [\n{cases}  ]",
                    dc.repr
                ));
                for ((_, abody), al) in arms.iter().zip(arm_lbls.iter()) {
                    self.start_block(al);
                    for s in abody {
                        self.emit_stmt(s)?;
                    }
                    self.emit_term(format!("br label %{end_lbl}"));
                }
                self.start_block(&def_lbl);
                for s in default {
                    self.emit_stmt(s)?;
                }
                self.emit_term(format!("br label %{end_lbl}"));
                self.start_block(&end_lbl);
            }
            MirStmt::Return(val_opt) => {
                if !self.error_spec.has_error() && self.ret_ty.is_void() {
                    self.emit_term("ret void");
                } else {
                    let env = self.build_return_envelope(val_opt.as_ref())?;
                    self.emit_term(format!("ret {} {}", env.ty, env.repr));
                }
            }
            MirStmt::Error { reason, partial } => {
                let env = self.build_error_envelope(reason.as_ref(), partial.as_ref())?;
                self.emit_term(format!("ret {} {}", env.ty, env.repr));
            }
            MirStmt::Break => {
                if let Some((_, brk)) = self.loop_targets.last().cloned() {
                    self.emit_term(format!("br label %{brk}"));
                }
            }
            MirStmt::Continue => {
                if let Some((cnt, _)) = self.loop_targets.last().cloned() {
                    self.emit_term(format!("br label %{cnt}"));
                }
            }
            MirStmt::SpawnActivate { slot_id, args } => {
                self.emit_boot_spawn_activate(*slot_id, args)?;
            }
            MirStmt::PrintFmt { chunks, newline } => {
                let call_kw = self.call_kw();
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_init_uart(ptr %__fmt_sink)"
                ));
                for ch in chunks {
                    self.emit_fmt_chunk("%__fmt_sink", ch)?;
                }
                if *newline {
                    self.emit_line(format!(
                        "{call_kw} void @__syn_fmt_putc(ptr %__fmt_sink, i8 10)"
                    ));
                }
            }
            MirStmt::FmtBuf {
                dst_res,
                buf,
                chunks,
            } => {
                let call_kw = self.call_kw();
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let bv = self.emit_expr(buf)?;
                let bptr = self.next_tmp();
                let blen = self.next_tmp();
                self.emit_line(format!("{bptr} = extractvalue {sl_ty} {}, 0", bv.repr));
                self.emit_line(format!("{blen} = extractvalue {sl_ty} {}, 1", bv.repr));
                let cap32 = self.coerce(SsaVal::new(&usize_ty, blen), "i32", false);
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_init_buf(ptr %__fmt_sink, ptr {bptr}, i32 {})",
                    cap32.repr
                ));
                for ch in chunks {
                    self.emit_fmt_chunk("%__fmt_sink", ch)?;
                }
                let wp = self.next_tmp();
                let tp = self.next_tmp();
                let written = self.next_tmp();
                let trunc = self.next_tmp();
                let is_tr = self.next_tmp();
                let out_len32 = self.next_tmp();
                self.emit_line(format!(
                    "{wp} = getelementptr inbounds %SynFmtSink, ptr %__fmt_sink, i32 0, i32 2"
                ));
                self.emit_line(format!(
                    "{tp} = getelementptr inbounds %SynFmtSink, ptr %__fmt_sink, i32 0, i32 3"
                ));
                self.emit_line(format!("{written} = load i32, ptr {wp}, align 4"));
                self.emit_line(format!("{trunc} = load i8, ptr {tp}, align 1"));
                self.emit_line(format!("{is_tr} = icmp ne i8 {trunc}, 0"));
                self.emit_line(format!(
                    "{out_len32} = select i1 {is_tr}, i32 {}, i32 {written}",
                    cap32.repr
                ));
                let out_len_u = self.coerce(SsaVal::new("i32", out_len32), &usize_ty, false);
                let s0 = self.next_tmp();
                let s1 = self.next_tmp();
                self.emit_line(format!("{s0} = insertvalue {sl_ty} poison, ptr {bptr}, 0"));
                self.emit_line(format!(
                    "{s1} = insertvalue {sl_ty} {s0}, {usize_ty} {}, 1",
                    out_len_u.repr
                ));
                let err_tag = self.next_tmp();
                self.emit_line(format!("{err_tag} = select i1 {is_tr}, i32 1, i32 0"));
                let rty = self.ctx.llvm_ty(&self.vars[dst_res.0].ty);
                let r0 = self.next_tmp();
                let r1 = self.next_tmp();
                self.emit_line(format!("{r0} = insertvalue {rty} poison, {sl_ty} {s1}, 0"));
                self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 {err_tag}, 1"));
                let dp = self.var_ptr(*dst_res);
                self.emit_line(format!("store {rty} {r1}, ptr {dp}, align 4"));
            }
            MirStmt::FmtRing {
                dst_res,
                ring,
                chunks,
                boost,
            } => {
                let call_kw = self.call_kw();
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let (hdr, cap32) = self.emit_byte_ring_view_parts(ring)?;
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_init_ring(ptr %__fmt_sink, ptr {hdr}, i32 {cap32})"
                ));
                for ch in chunks {
                    self.emit_fmt_chunk("%__fmt_sink", ch)?;
                }
                let wp = self.next_tmp();
                let tp = self.next_tmp();
                let written = self.next_tmp();
                let trunc = self.next_tmp();
                self.emit_line(format!(
                    "{wp} = getelementptr inbounds %SynFmtSink, ptr %__fmt_sink, i32 0, i32 2"
                ));
                self.emit_line(format!(
                    "{tp} = getelementptr inbounds %SynFmtSink, ptr %__fmt_sink, i32 0, i32 3"
                ));
                self.emit_line(format!("{written} = load i32, ptr {wp}, align 4"));
                self.emit_line(format!("{trunc} = load i8, ptr {tp}, align 1"));
                let has_w = self.next_tmp();
                self.emit_line(format!("{has_w} = icmp sgt i32 {written}, 0"));
                let wake_lbl = self.next_label("fring_wake");
                let done_lbl = self.next_label("fring_done");
                self.emit_term(format!(
                    "br i1 {has_w}, label %{wake_lbl}, label %{done_lbl}"
                ));
                self.start_block(&wake_lbl);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                self.emit_line(format!(
                    "{call_kw} void @__syn_ring_hdr_wake(ptr {hdr}, i32 {b})"
                ));
                self.emit_term(format!("br label %{done_lbl}"));
                self.start_block(&done_lbl);
                let is_tr = self.next_tmp();
                self.emit_line(format!("{is_tr} = icmp ne i8 {trunc}, 0"));
                let written_u = self.coerce(SsaVal::new("i32", written), &usize_ty, false);
                let err_tag = self.next_tmp();
                self.emit_line(format!("{err_tag} = select i1 {is_tr}, i32 1, i32 0"));
                let rty = self.ctx.llvm_ty(&self.vars[dst_res.0].ty);
                let r0 = self.next_tmp();
                let r1 = self.next_tmp();
                self.emit_line(format!(
                    "{r0} = insertvalue {rty} poison, {usize_ty} {}, 0",
                    written_u.repr
                ));
                self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 {err_tag}, 1"));
                let dp = self.var_ptr(*dst_res);
                self.emit_line(format!("store {rty} {r1}, ptr {dp}, align 4"));
            }
        }
        Ok(())
    }

    pub(super) fn emit_fmt_chunk(
        &mut self,
        sink: &str,
        ch: &MirFmtChunk,
    ) -> Result<(), CodegenError> {
        let call_kw = self.call_kw();
        let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
        match ch {
            MirFmtChunk::Literal(s) => {
                let bytes = zev_ast::decoded_string_bytes(s);
                if bytes.is_empty() {
                    return Ok(());
                }
                if bytes.len() == 1 {
                    let b = bytes[0] as i8;
                    self.emit_line(format!(
                        "{call_kw} void @__syn_fmt_putc(ptr {sink}, i8 {b})"
                    ));
                } else {
                    let idx = self.ctx.intern_string(s);
                    let slen = bytes.len() as i32;
                    self.emit_line(format!(
                        "{call_kw} void @__syn_fmt_bytes(ptr {sink}, ptr @.str.{idx}, i32 {slen})"
                    ));
                }
            }
            MirFmtChunk::Int {
                value,
                signed,
                bits,
                fmt,
            } => {
                let vv = self.emit_expr(value)?;
                match fmt {
                    MirIntFmt::Dec => {
                        let is64 =
                            *bits == 64 || (*bits == 0 && self.ctx.spec.isa.ptr_bits() == 64);
                        if is64 {
                            let v64 = self.coerce(vv, "i64", *signed);
                            let fn_sym = if *signed {
                                "@__syn_fmt_i64_dec"
                            } else {
                                "@__syn_fmt_u64_dec"
                            };
                            self.emit_line(format!(
                                "{call_kw} void {fn_sym}(ptr {sink}, i64 {})",
                                v64.repr
                            ));
                        } else {
                            let v32 = self.coerce(vv, "i32", *signed);
                            let fn_sym = if *signed {
                                "@__syn_fmt_i32_dec"
                            } else {
                                "@__syn_fmt_u32_dec"
                            };
                            self.emit_line(format!(
                                "{call_kw} void {fn_sym}(ptr {sink}, i32 {})",
                                v32.repr
                            ));
                        }
                    }
                    MirIntFmt::Hex {
                        upper,
                        zero_pad_width,
                    } => {
                        let narrow_ty = match *bits {
                            8 => "i8",
                            16 => "i16",
                            32 => "i32",
                            64 => "i64",
                            _ => usize_ty.as_str(),
                        };
                        let nv = self.coerce(vv, narrow_ty, false);
                        let up = i32::from(*upper);
                        let w = i32::from(*zero_pad_width);
                        if narrow_ty == "i64" {
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_u64_hex(ptr {sink}, i64 {}, i8 {up}, i32 {w})",
                                nv.repr
                            ));
                        } else {
                            let u32v = self.coerce(nv, "i32", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_u32_hex(ptr {sink}, i32 {}, i8 {up}, i32 {w})",
                                u32v.repr
                            ));
                        }
                    }
                    MirIntFmt::Bin { zero_pad_width } => {
                        let narrow_ty = match *bits {
                            8 => "i8",
                            16 => "i16",
                            32 => "i32",
                            64 => "i64",
                            _ => usize_ty.as_str(),
                        };
                        let nv = self.coerce(vv, narrow_ty, false);
                        let w = i32::from(*zero_pad_width);
                        if narrow_ty == "i64" {
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_u64_bin(ptr {sink}, i64 {}, i32 {w})",
                                nv.repr
                            ));
                        } else {
                            let u32v = self.coerce(nv, "i32", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_u32_bin(ptr {sink}, i32 {}, i32 {w})",
                                u32v.repr
                            ));
                        }
                    }
                    MirIntFmt::Padded { base, upper, pad } => {
                        let up = i32::from(*upper);
                        let w = i32::from(pad.width);
                        let la = i32::from(pad.left_align);
                        let zp = i32::from(pad.zero_pad);
                        let lf = if pad.zero_pad { 48 } else { 32 };
                        let b = i32::from(*base);
                        let ptr64 = self.ctx.spec.isa.ptr_bits() == 64;
                        let is64 = *bits == 64 || (*bits == 0 && ptr64);
                        if *base == 10 {
                            let sg = i32::from(*signed);
                            if is64 {
                                let v64 = self.coerce(vv, "i64", *signed);
                                self.emit_line(format!(
                                    "{call_kw} void @__syn_fmt_int64_pad(ptr {sink}, i64 {}, i8 {sg}, i32 {b}, i8 {up}, i32 {w}, i8 {la}, i8 {zp}, i8 {lf})",
                                    v64.repr
                                ));
                            } else {
                                let v32 = self.coerce(vv, "i32", *signed);
                                self.emit_line(format!(
                                    "{call_kw} void @__syn_fmt_int32_pad(ptr {sink}, i32 {}, i8 {sg}, i32 {b}, i8 {up}, i32 {w}, i8 {la}, i8 {zp}, i8 {lf})",
                                    v32.repr
                                ));
                            }
                        } else {
                            let narrow_ty = match *bits {
                                8 => "i8",
                                16 => "i16",
                                32 => "i32",
                                64 => "i64",
                                _ => usize_ty.as_str(),
                            };
                            let nv = self.coerce(vv, narrow_ty, false);
                            if narrow_ty == "i64" || (narrow_ty == usize_ty && ptr64) {
                                let v64 = self.coerce(nv, "i64", false);
                                self.emit_line(format!(
                                    "{call_kw} void @__syn_fmt_int64_pad(ptr {sink}, i64 {}, i8 0, i32 {b}, i8 {up}, i32 {w}, i8 {la}, i8 {zp}, i8 {lf})",
                                    v64.repr
                                ));
                            } else {
                                let u32v = self.coerce(nv, "i32", false);
                                self.emit_line(format!(
                                    "{call_kw} void @__syn_fmt_int32_pad(ptr {sink}, i32 {}, i8 0, i32 {b}, i8 {up}, i32 {w}, i8 {la}, i8 {zp}, i8 {lf})",
                                    u32v.repr
                                ));
                            }
                        }
                    }
                }
            }
            MirFmtChunk::Float { value, is_f64, fmt } => {
                let vv = self.emit_expr(value)?;
                match fmt {
                    MirFloatFmt::Fixed { precision } => {
                        let p = i32::from(*precision);
                        if *is_f64 {
                            let dv = self.coerce(vv, "double", true);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_f64_fixed(ptr {sink}, double {}, i32 {p})",
                                dv.repr
                            ));
                        } else {
                            let fv = self.coerce(vv, "float", true);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_f32_fixed(ptr {sink}, float {}, i32 {p})",
                                fv.repr
                            ));
                        }
                    }
                    MirFloatFmt::Sci { precision } => {
                        let p = i32::from(*precision);
                        if *is_f64 {
                            let dv = self.coerce(vv, "double", true);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_f64_sci(ptr {sink}, double {}, i32 {p})",
                                dv.repr
                            ));
                        } else {
                            let fv = self.coerce(vv, "float", true);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_f32_sci(ptr {sink}, float {}, i32 {p})",
                                fv.repr
                            ));
                        }
                    }
                }
            }
            MirFmtChunk::Bool(value) => {
                let vv = self.emit_expr(value)?;
                let b1 = self.coerce_to_i1(vv);
                let b8 = self.next_tmp();
                self.emit_line(format!("{b8} = zext i1 {b1} to i8"));
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_bool(ptr {sink}, i8 {b8})"
                ));
            }
            MirFmtChunk::Str(value) => {
                let vv = self.emit_expr(value)?;
                let p = self.coerce_str_ptr(vv);
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_cstr(ptr {sink}, ptr {})",
                    p.repr
                ));
            }
            MirFmtChunk::PaddedStr { cstr, pad } => {
                let vv = self.emit_expr(cstr)?;
                let p = self.coerce_str_ptr(vv);
                let w = i32::from(pad.width);
                let la = i32::from(pad.left_align);
                let fill = if pad.zero_pad { 48 } else { 32 };
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_cstr_pad(ptr {sink}, ptr {}, i32 {w}, i8 {la}, i8 {fill})",
                    p.repr
                ));
            }
            MirFmtChunk::Bytes(value) => {
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(value)?;
                let dp = self.next_tmp();
                let dl = self.next_tmp();
                self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", sv.repr));
                self.emit_line(format!("{dl} = extractvalue {sl_ty} {}, 1", sv.repr));
                let dl32 = self.coerce(SsaVal::new(&usize_ty, dl), "i32", false);
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_bytes(ptr {sink}, ptr {dp}, i32 {})",
                    dl32.repr
                ));
            }
            MirFmtChunk::PaddedBytes { slice, pad } => {
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(slice)?;
                let dp = self.next_tmp();
                let dl = self.next_tmp();
                self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", sv.repr));
                self.emit_line(format!("{dl} = extractvalue {sl_ty} {}, 1", sv.repr));
                let dl32 = self.coerce(SsaVal::new(&usize_ty, dl), "i32", false);
                let w = i32::from(pad.width);
                let la = i32::from(pad.left_align);
                let fill = if pad.zero_pad { 48 } else { 32 };
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_bytes_pad(ptr {sink}, ptr {dp}, i32 {}, i32 {w}, i8 {la}, i8 {fill})",
                    dl32.repr
                ));
            }
            MirFmtChunk::EnumName { enum_name, value } => {
                let vv = self.emit_expr(value)?;
                let v32 = self.coerce(vv, "i32", true);
                let eid = sanitize_ident(enum_name);
                let np = self.next_tmp();
                self.emit_line(format!(
                    "{np} = {call_kw} ptr @__syn_enum_name_{eid}(i32 {})",
                    v32.repr
                ));
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_cstr(ptr {sink}, ptr {np})"
                ));
            }
            MirFmtChunk::DebugSlice { elem_ty, slice } => {
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(slice)?;
                let raw_slice_elem_ty = match &slice.ty {
                    MirType::Slice(inner) => (**inner).clone(),
                    _ => elem_ty.clone(),
                };
                let el_lty = self.ctx.llvm_ty(&raw_slice_elem_ty);
                let dp = self.next_tmp();
                let dl = self.next_tmp();
                self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", sv.repr));
                self.emit_line(format!("{dl} = extractvalue {sl_ty} {}, 1", sv.repr));
                self.emit_line(format!("{call_kw} void @__syn_fmt_putc(ptr {sink}, i8 91)"));

                let (idx_slot, idx_al) = self.emit_temp_alloca(&usize_ty);
                self.emit_line(format!(
                    "store {usize_ty} 0, ptr {idx_slot}, align {idx_al}"
                ));

                let cond_lbl = self.next_label("dsl_cond");
                let body_lbl = self.next_label("dsl_body");
                let sep_lbl = self.next_label("dsl_sep");
                let val_lbl = self.next_label("dsl_val");
                let end_lbl = self.next_label("dsl_end");
                let sep_idx = self.ctx.intern_string(", ");

                self.emit_term(format!("br label %{cond_lbl}"));
                self.start_block(&cond_lbl);
                let idx_v = self.next_tmp();
                let more = self.next_tmp();
                self.emit_line(format!(
                    "{idx_v} = load {usize_ty}, ptr {idx_slot}, align {idx_al}"
                ));
                self.emit_line(format!("{more} = icmp ult {usize_ty} {idx_v}, {dl}"));
                self.emit_term(format!("br i1 {more}, label %{body_lbl}, label %{end_lbl}"));

                self.start_block(&body_lbl);
                let gt0 = self.next_tmp();
                self.emit_line(format!("{gt0} = icmp ugt {usize_ty} {idx_v}, 0"));
                self.emit_term(format!("br i1 {gt0}, label %{sep_lbl}, label %{val_lbl}"));

                self.start_block(&sep_lbl);
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_bytes(ptr {sink}, ptr @.str.{sep_idx}, i32 2)"
                ));
                self.emit_term(format!("br label %{val_lbl}"));

                self.start_block(&val_lbl);
                let ep = self.next_tmp();
                self.emit_line(format!(
                    "{ep} = getelementptr {el_lty}, ptr {dp}, {usize_ty} {idx_v}"
                ));
                let ev = self.load_from_place_ptr(&ep, &raw_slice_elem_ty, false);
                match elem_ty {
                    MirType::Bool => {
                        let b1 = self.coerce_to_i1(ev);
                        let b8 = self.next_tmp();
                        self.emit_line(format!("{b8} = zext i1 {b1} to i8"));
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_bool(ptr {sink}, i8 {b8})"
                        ));
                    }
                    MirType::F32 => {
                        let fv = self.coerce(ev, "float", true);
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_f32_fixed(ptr {sink}, float {}, i32 2)",
                            fv.repr
                        ));
                    }
                    MirType::F64 => {
                        let dv = self.coerce(ev, "double", true);
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_f64_fixed(ptr {sink}, double {}, i32 2)",
                            dv.repr
                        ));
                    }
                    MirType::Enum(ename) => {
                        let v32 = self.coerce(ev, "i32", true);
                        let eid = sanitize_ident(ename);
                        let np = self.next_tmp();
                        self.emit_line(format!(
                            "{np} = {call_kw} ptr @__syn_enum_name_{eid}(i32 {})",
                            v32.repr
                        ));
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_cstr(ptr {sink}, ptr {np})"
                        ));
                    }
                    MirType::Str => {
                        let p = self.coerce_str_ptr(ev);
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_cstr(ptr {sink}, ptr {})",
                            p.repr
                        ));
                    }
                    MirType::Ptr(_) => {
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_putc(ptr {sink}, i8 48)"
                        ));
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_putc(ptr {sink}, i8 120)"
                        ));
                        if self.ctx.spec.isa.ptr_bits() == 64 {
                            let addr64 = self.coerce(ev, "i64", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_u64_hex(ptr {sink}, i64 {}, i8 0, i32 1)",
                                addr64.repr
                            ));
                        } else {
                            let addr32 = self.coerce(ev, "i32", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_u32_hex(ptr {sink}, i32 {}, i8 0, i32 1)",
                                addr32.repr
                            ));
                        }
                    }
                    MirType::Slice(inner) if **inner == MirType::U8 => {
                        let sp = self.next_tmp();
                        let sln = self.next_tmp();
                        self.emit_line(format!("{sp} = extractvalue {sl_ty} {}, 0", ev.repr));
                        self.emit_line(format!("{sln} = extractvalue {sl_ty} {}, 1", ev.repr));
                        let sln32 = self.coerce(SsaVal::new(&usize_ty, sln), "i32", false);
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_bytes(ptr {sink}, ptr {sp}, i32 {})",
                            sln32.repr
                        ));
                    }
                    MirType::I64 => {
                        let v64 = self.coerce(ev, "i64", true);
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_i64_dec(ptr {sink}, i64 {})",
                            v64.repr
                        ));
                    }
                    MirType::U64 => {
                        let v64 = self.coerce(ev, "i64", false);
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_u64_dec(ptr {sink}, i64 {})",
                            v64.repr
                        ));
                    }
                    MirType::I8
                    | MirType::I16
                    | MirType::I32
                    | MirType::Isize
                    | MirType::EndianInt { signed: true, .. } => {
                        if matches!(elem_ty, MirType::Isize) && self.ctx.spec.isa.ptr_bits() == 64 {
                            let v64 = self.coerce(ev, "i64", true);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_i64_dec(ptr {sink}, i64 {})",
                                v64.repr
                            ));
                        } else {
                            let v32 = self.coerce(ev, "i32", true);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_i32_dec(ptr {sink}, i32 {})",
                                v32.repr
                            ));
                        }
                    }
                    _ => {
                        if matches!(elem_ty, MirType::Usize) && self.ctx.spec.isa.ptr_bits() == 64 {
                            let v64 = self.coerce(ev, "i64", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_u64_dec(ptr {sink}, i64 {})",
                                v64.repr
                            ));
                        } else {
                            let v32 = self.coerce(ev, "i32", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fmt_u32_dec(ptr {sink}, i32 {})",
                                v32.repr
                            ));
                        }
                    }
                }
                let idx_next = self.next_tmp();
                self.emit_line(format!("{idx_next} = add {usize_ty} {idx_v}, 1"));
                self.emit_line(format!(
                    "store {usize_ty} {idx_next}, ptr {idx_slot}, align {idx_al}"
                ));
                self.emit_term(format!("br label %{cond_lbl}"));

                self.start_block(&end_lbl);
                self.emit_line(format!("{call_kw} void @__syn_fmt_putc(ptr {sink}, i8 93)"));
            }
        }
        Ok(())
    }

    pub(super) fn emit_compound_op(
        &mut self,
        op: MirAssignOp,
        lhs: &SsaVal,
        rhs: &SsaVal,
        signed: bool,
    ) -> SsaVal {
        let ty = &lhs.ty;
        if ty == "float" || ty == "double" {
            let finst = match op {
                MirAssignOp::Assign => unreachable!(),
                MirAssignOp::AddAssign => "fadd",
                MirAssignOp::SubAssign => "fsub",
                MirAssignOp::MulAssign => "fmul",
                MirAssignOp::DivAssign => "fdiv",
                MirAssignOp::RemAssign => "frem",
                _ => "fadd",
            };
            let t = self.next_tmp();
            self.emit_line(format!("{t} = {finst} {ty} {}, {}", lhs.repr, rhs.repr));
            return SsaVal::new(ty, t);
        }
        let rv = if matches!(op, MirAssignOp::ShlAssign | MirAssignOp::ShrAssign) {
            let coerced = self.coerce(rhs.clone(), ty, false);
            let mask = llvm_int_bits(ty).saturating_sub(1);
            if let Ok(c) = coerced.repr.parse::<i128>() {
                SsaVal::new(ty, ((c as u128) & u128::from(mask)).to_string())
            } else {
                let m = self.next_tmp();
                self.emit_line(format!("{m} = and {ty} {}, {mask}", coerced.repr));
                SsaVal::new(ty, m)
            }
        } else {
            rhs.clone()
        };
        let inst = match op {
            MirAssignOp::Assign => unreachable!(),
            MirAssignOp::AddAssign => "add",
            MirAssignOp::SubAssign => "sub",
            MirAssignOp::MulAssign => "mul",
            MirAssignOp::DivAssign => {
                let res = self.emit_checked_int_div_rem(ty, &lhs.repr, &rv.repr, signed, false);
                return SsaVal::new(ty, res);
            }
            MirAssignOp::RemAssign => {
                let res = self.emit_checked_int_div_rem(ty, &lhs.repr, &rv.repr, signed, true);
                return SsaVal::new(ty, res);
            }
            MirAssignOp::BitOrAssign => "or",
            MirAssignOp::BitAndAssign => "and",
            MirAssignOp::BitXorAssign => "xor",
            MirAssignOp::ShlAssign => "shl",
            MirAssignOp::ShrAssign => {
                if signed {
                    "ashr"
                } else {
                    "lshr"
                }
            }
        };
        let t = self.next_tmp();
        self.emit_line(format!("{t} = {inst} {ty} {}, {}", lhs.repr, rv.repr));
        SsaVal::new(ty, t)
    }

    pub(super) fn emit_bounds_check(&mut self, in_bounds_i1: &str, reason: u8) {
        let ok_lbl = self.next_label("chk_ok");
        let trap_lbl = self.next_label("chk_trap");
        let call_kw = self.call_kw();
        self.emit_term(format!(
            "br i1 {in_bounds_i1}, label %{ok_lbl}, label %{trap_lbl}"
        ));
        self.start_block(&trap_lbl);
        self.emit_line(format!("{call_kw} void @__syn_panic(i8 {reason})"));
        self.emit_term("unreachable");
        self.start_block(&ok_lbl);
    }

    pub(super) fn emit_checked_int_div_rem(
        &mut self,
        ty: &str,
        lhs: &str,
        rhs: &str,
        signed: bool,
        is_rem: bool,
    ) -> String {
        let nz = self.next_tmp();
        self.emit_line(format!("{nz} = icmp ne {ty} {rhs}, 0"));
        self.emit_bounds_check(&nz, 4);
        let t = self.next_tmp();
        if !signed {
            let inst = if is_rem { "urem" } else { "udiv" };
            self.emit_line(format!("{t} = {inst} {ty} {lhs}, {rhs}"));
            return t;
        }
        let min_lit = match ty {
            "i8" => Some("-128"),
            "i16" => Some("-32768"),
            "i32" => Some("-2147483648"),
            "i64" => Some("-9223372036854775808"),
            _ => None,
        };
        if let Some(int_min) = min_lit {
            let is_min = self.next_tmp();
            let is_neg1 = self.next_tmp();
            let ovf = self.next_tmp();
            let safe_rhs = self.next_tmp();
            self.emit_line(format!("{is_min} = icmp eq {ty} {lhs}, {int_min}"));
            self.emit_line(format!("{is_neg1} = icmp eq {ty} {rhs}, -1"));
            self.emit_line(format!("{ovf} = and i1 {is_min}, {is_neg1}"));
            self.emit_line(format!("{safe_rhs} = select i1 {ovf}, {ty} 1, {ty} {rhs}"));
            if is_rem {
                let r0 = self.next_tmp();
                self.emit_line(format!("{r0} = srem {ty} {lhs}, {safe_rhs}"));
                self.emit_line(format!("{t} = select i1 {ovf}, {ty} 0, {ty} {r0}"));
            } else {
                self.emit_line(format!("{t} = sdiv {ty} {lhs}, {safe_rhs}"));
            }
        } else {
            let inst = if is_rem { "srem" } else { "sdiv" };
            self.emit_line(format!("{t} = {inst} {ty} {lhs}, {rhs}"));
        }
        t
    }

    pub(super) fn emit_byte_ring_view_parts(
        &mut self,
        view: &MirExpr,
    ) -> Result<(String, String), CodegenError> {
        let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
        let vty = format!("{{ ptr, {usize_ty} }}");
        let vv = self.emit_expr(view)?;
        let hdr = self.next_tmp();
        let cap_u = self.next_tmp();
        self.emit_line(format!("{hdr} = extractvalue {vty} {}, 0", vv.repr));
        self.emit_line(format!("{cap_u} = extractvalue {vty} {}, 1", vv.repr));
        let cap32 = self.coerce(SsaVal::new(&usize_ty, cap_u), "i32", false);
        Ok((hdr, cap32.repr))
    }

    pub(super) fn emit_inbox_view_parts(
        &mut self,
        view: &MirExpr,
    ) -> Result<(String, String, String), CodegenError> {
        let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
        let vty = format!("{{ ptr, ptr, {usize_ty} }}");
        let vv = self.emit_expr(view)?;
        let hdr = self.next_tmp();
        let items_base = self.next_tmp();
        let cap_u = self.next_tmp();
        self.emit_line(format!("{hdr} = extractvalue {vty} {}, 0", vv.repr));
        self.emit_line(format!("{items_base} = extractvalue {vty} {}, 1", vv.repr));
        self.emit_line(format!("{cap_u} = extractvalue {vty} {}, 2", vv.repr));
        let cap32 = self.coerce(SsaVal::new(&usize_ty, cap_u), "i32", false);
        Ok((hdr, items_base, cap32.repr))
    }

    pub(super) fn resolve_prim_ptr(
        &mut self,
        target: &MirPrimTarget,
    ) -> Result<String, CodegenError> {
        match target {
            MirPrimTarget::Global(name) => Ok(format!("@__syn_prim_{}", sanitize_ident(name))),
            MirPrimTarget::PtrExpr(e) => {
                let v = self.emit_expr(e)?;
                let p = self.coerce(v, "ptr", false);
                Ok(p.repr)
            }
        }
    }

    pub(super) fn emit_prim_nonblocking(
        &mut self,
        pop: &MirPrimNonBlockingOp,
    ) -> Result<(), CodegenError> {
        let call_kw = self.call_kw();
        match pop {
            MirPrimNonBlockingOp::WatchPublish {
                target,
                elem_ty,
                value,
            } => {
                if let MirPrimTarget::Global(name) = target {
                    if self
                        .ctx
                        .module
                        .primitives
                        .iter()
                        .any(|p| p.name == *name && p.dead_stripped)
                    {
                        let _ = self.emit_expr(value)?;
                        return Ok(());
                    }
                }
                let pk = MirPrimKind::Watch {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let wptr = self.resolve_prim_ptr(target)?;
                let ety = self.ctx.llvm_ty(elem_ty);
                let vv = self.emit_expr(value)?;
                let vc = self.coerce(vv, &ety, !value.ty.is_unsigned());
                self.emit_line(format!(
                    "{call_kw} void @__syn_watch_pub_{id}(ptr {wptr}, {ety} {})",
                    vc.repr
                ));
            }
            MirPrimNonBlockingOp::GuardedSet {
                target,
                elem_ty,
                value,
            } => {
                let pk = MirPrimKind::Guarded {
                    elem: Box::new(elem_ty.clone()),
                };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let gptr = self.resolve_prim_ptr(target)?;
                let ety = self.ctx.llvm_ty(elem_ty);
                let vv = self.emit_expr(value)?;
                let vc = self.coerce(vv, &ety, !value.ty.is_unsigned());
                let vp = self.next_tmp();
                self.emit_line(format!(
                    "{vp} = getelementptr inbounds {pty}, ptr {gptr}, i32 0, i32 0"
                ));
                self.emit_line(format!(
                    "store volatile {ety} {}, ptr {vp}, align 1",
                    vc.repr
                ));
            }
            MirPrimNonBlockingOp::InboxPost {
                target,
                elem_ty,
                cap,
                value,
                boost,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let qptr = self.resolve_prim_ptr(target)?;
                let ety = self.ctx.llvm_ty(elem_ty);
                let vv = self.emit_expr(value)?;
                let vc = self.coerce(vv, &ety, !value.ty.is_unsigned());
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let ign = self.next_tmp();
                self.emit_line(format!(
                    "{ign} = {call_kw} i8 @__syn_q_push_{id}(ptr {qptr}, {ety} {}, i32 {b})",
                    vc.repr
                ));
            }
            MirPrimNonBlockingOp::SignalNotify { target, boost } => {
                let id = sanitize_ident(&MirPrimKind::Signal.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                self.emit_line(format!(
                    "{call_kw} void @__syn_sig_notify_{id}(ptr {sptr}, i32 {b})"
                ));
            }
            MirPrimNonBlockingOp::SignalClear { target } => {
                let sptr = self.resolve_prim_ptr(target)?;
                let fp = self.next_tmp();
                self.emit_line(format!(
                    "{fp} = getelementptr inbounds %prim.signal, ptr {sptr}, i32 0, i32 0"
                ));
                self.emit_line(format!("store volatile i8 0, ptr {fp}, align 1"));
            }
            MirPrimNonBlockingOp::WaitQueuePrepare { target } => {
                let wptr = self.resolve_prim_ptr(target)?;
                self.emit_line(format!("{call_kw} void @__syn_wq_prepare(ptr {wptr})"));
            }
            MirPrimNonBlockingOp::WaitQueueCancel { target } => {
                let wptr = self.resolve_prim_ptr(target)?;
                self.emit_line(format!("{call_kw} void @__syn_wq_cancel(ptr {wptr})"));
            }
            MirPrimNonBlockingOp::WaitQueueWakeAll { target, boost } => {
                let wptr = self.resolve_prim_ptr(target)?;
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                self.emit_line(format!(
                    "{call_kw} void @__syn_wake_all_mask(ptr {wptr}, i32 {b})"
                ));
            }
            MirPrimNonBlockingOp::PingPongFlip {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                self.emit_line(format!("{call_kw} void @__syn_pp_flip_{id}(ptr {pptr})"));
            }
            MirPrimNonBlockingOp::PingPongCompleteHalf {
                target,
                elem_ty,
                cap,
                half,
            } => {
                let pk = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let hv = self.emit_expr(half)?;
                let h8 = self.coerce(hv, "i8", false);
                self.emit_line(format!(
                    "{call_kw} void @__syn_pp_complete_half_{id}(ptr {pptr}, i8 {})",
                    h8.repr
                ));
            }
            MirPrimNonBlockingOp::PingPongSetActive {
                target,
                elem_ty,
                cap,
                active,
            } => {
                let pk = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let act = i8::from(*active);
                self.emit_line(format!(
                    "{call_kw} void @__syn_pp_set_active_{id}(ptr {pptr}, i8 {act})"
                ));
            }
            MirPrimNonBlockingOp::LeaseRelease { value, boost, .. } => {
                let lv = self.emit_expr(value)?;
                let ph = self.next_tmp();
                let idx = self.next_tmp();
                self.emit_line(format!(
                    "{ph} = extractvalue {{ ptr, ptr, i8 }} {}, 1",
                    lv.repr
                ));
                self.emit_line(format!(
                    "{idx} = extractvalue {{ ptr, ptr, i8 }} {}, 2",
                    lv.repr
                ));
                let b = boost.map_or(-1i8, |p| p.as_u8() as i8);
                self.emit_line(format!(
                    "{call_kw} void @__syn_pool_release(ptr {ph}, i8 {idx}, i8 {b})"
                ));
            }
            MirPrimNonBlockingOp::LeaseNullify { target, .. } => {
                let (lp, _, _) = self.emit_place_ptr(target)?;
                let php = self.next_tmp();
                let ptr_al = self.ctx.ptr_align();
                self.emit_line(format!(
                    "{php} = getelementptr inbounds {{ ptr, ptr, i8 }}, ptr {lp}, i32 0, i32 1"
                ));
                self.emit_line(format!("store ptr null, ptr {php}, align {ptr_al}"));
            }
            MirPrimNonBlockingOp::ByteRingConsume { target, cap, len } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let lv = self.emit_expr(len)?;
                let l32 = self.coerce(lv, "i32", false);
                self.emit_line(format!(
                    "{call_kw} void @__syn_ring_consume_{id}(ptr {rptr}, i32 {})",
                    l32.repr
                ));
            }
            MirPrimNonBlockingOp::ByteRingCommitWrite {
                target,
                cap,
                len,
                boost,
            } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let lv = self.emit_expr(len)?;
                let l32 = self.coerce(lv, "i32", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                self.emit_line(format!(
                    "{call_kw} void @__syn_ring_commit_write_{id}(ptr {rptr}, i32 {}, i32 {b})",
                    l32.repr
                ));
            }
            MirPrimNonBlockingOp::ByteRingClear { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                self.emit_line(format!("{call_kw} void @__syn_ring_clear_{id}(ptr {rptr})"));
            }
            MirPrimNonBlockingOp::InboxViewPost {
                view,
                elem_ty,
                value,
                boost,
            } => {
                let (hdr, items_base, cap32) = self.emit_inbox_view_parts(view)?;
                let key = sanitize_ident(&elem_ty.mangle_name());
                let ety = self.ctx.llvm_ty(elem_ty);
                let vv = self.emit_expr(value)?;
                let vc = self.coerce(vv, &ety, !value.ty.is_unsigned());
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let ign = self.next_tmp();
                self.emit_line(format!(
                    "{ign} = {call_kw} i8 @__syn_q_view_push_{key}(ptr {hdr}, ptr {items_base}, i32 {cap32}, {ety} {}, i32 {b})",
                    vc.repr
                ));
            }
            MirPrimNonBlockingOp::ByteRingViewConsume { view, len } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let lv = self.emit_expr(len)?;
                let l32 = self.coerce(lv, "i32", false);
                self.emit_line(format!(
                    "{call_kw} void @__syn_ring_view_consume(ptr {hdr}, i32 {cap32}, i32 {})",
                    l32.repr
                ));
            }
            MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, boost } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let lv = self.emit_expr(len)?;
                let l32 = self.coerce(lv, "i32", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                self.emit_line(format!(
                    "{call_kw} void @__syn_ring_view_commit_write(ptr {hdr}, i32 {cap32}, i32 {}, i32 {b})",
                    l32.repr
                ));
            }
            MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                let (hdr, _) = self.emit_byte_ring_view_parts(view)?;
                self.emit_line(format!("{call_kw} void @__syn_ring_view_clear(ptr {hdr})"));
            }
        }
        Ok(())
    }

    pub(super) fn emit_intrinsic(&mut self, iop: &MirIntrinsicOp) -> Result<(), CodegenError> {
        let call_kw = self.call_kw();
        match iop {
            MirIntrinsicOp::UseSystick { hz } => {
                let hv = self.emit_expr(hz)?;
                let h32 = self.coerce(hv, "i32", false);
                self.emit_line(format!(
                    "{call_kw} void @__syn_use_systick(i32 {})",
                    h32.repr
                ));
            }
            MirIntrinsicOp::SetPriority { prio } => {
                let p = prio.as_u8() as i32;
                self.emit_line(format!("{call_kw} void @__syn_set_priority(i32 {p})"));
            }
            MirIntrinsicOp::SysOp { method, args } => {
                match method.as_str() {
                    "set_putc" => return self.emit_hook_registration(HookKind::Putc, args.first()),
                    "set_clock_ms" => {
                        return self.emit_hook_registration(HookKind::ClockMs, args.first());
                    }
                    "set_exit" => return self.emit_hook_registration(HookKind::Exit, args.first()),
                    "set_fault_handler" => {
                        return self.emit_hook_registration(HookKind::FaultHandler, args.first());
                    }
                    "set_idle" => return self.emit_hook_registration(HookKind::Idle, args.first()),
                    "set_irq_router" => {
                        self.emit_hook_registration(HookKind::IrqRoute, args.first())?;
                        return self.emit_hook_registration(HookKind::IrqAck, args.get(1));
                    }
                    "rom_call_void" => {
                        if args.len() >= 2 {
                            let sb = u32::try_from(self.ctx.eval_const_int(&args[1]).max(0))
                                .unwrap_or(0);
                            let _ = self.emit_rom_call_invocation(
                                RomWrapperKind::Void,
                                &args[0],
                                sb,
                                &args[2..],
                            )?;
                        }
                        return Ok(());
                    }
                    "rom_call_noreturn" => {
                        if let Some(addr_expr) = args.first() {
                            let _ = self.emit_rom_call_invocation(
                                RomWrapperKind::Noreturn,
                                addr_expr,
                                0,
                                &args[1..],
                            )?;
                        }
                        return Ok(());
                    }
                    _ => {}
                }
                let mut ev = Vec::new();
                for a in args {
                    ev.push((self.emit_expr(a)?, a.ty.clone()));
                }
                match method.as_str() {
                    "exit" | "stop" => {
                        let code = if let Some((c, cty)) = ev.first() {
                            self.coerce(c.clone(), "i32", !cty.is_unsigned()).repr
                        } else {
                            "0".to_string()
                        };
                        self.emit_line(format!("{call_kw} void @__syn_platform_exit(i32 {code})"));
                        self.emit_term("unreachable");
                    }
                    "pend_irq" => {
                        if let Some((irq, _)) = ev.first() {
                            let i32v = self.coerce(irq.clone(), "i32", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_nvic_pend_irq(i32 {})",
                                i32v.repr
                            ));
                        }
                    }
                    "fuzz_irqs" => {
                        if let (Some((seed, _)), Some((mask, _))) = (ev.first(), ev.get(1)) {
                            let s32 = self.coerce(seed.clone(), "i32", false);
                            let m32 = self.coerce(mask.clone(), "i32", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_fuzz_irqs(i32 {}, i32 {})",
                                s32.repr, m32.repr
                            ));
                        }
                    }
                    "use_systick" => {
                        if let Some((hz, _)) = ev.first() {
                            let h32 = self.coerce(hz.clone(), "i32", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_use_systick(i32 {})",
                                h32.repr
                            ));
                        }
                    }
                    "now_ms" | "now_ticks" => {
                        let ign = self.next_tmp();
                        self.emit_line(format!("{ign} = {call_kw} i32 @__syn_now_ms()"));
                    }
                    "puts" | "print" => {
                        if let Some((s, _)) = ev.first() {
                            let p = self.coerce_str_ptr(s.clone());
                            self.emit_line(format!(
                                "{call_kw} void @__syn_uart_puts(ptr {})",
                                p.repr
                            ));
                        }
                    }
                    "println" | "log" => {
                        if let Some((s, _)) = ev.first() {
                            let p = self.coerce_str_ptr(s.clone());
                            self.emit_line(format!(
                                "{call_kw} void @__syn_uart_println(ptr {})",
                                p.repr
                            ));
                        } else {
                            self.emit_line(format!("{call_kw} void @__syn_uart_putc(i8 10)"));
                        }
                    }
                    "log_kv" => {
                        if let (Some((lbl, _)), Some((val, vty))) = (ev.first(), ev.get(1)) {
                            let p = self.coerce_str_ptr(lbl.clone());
                            let v32 = self.coerce(val.clone(), "i32", !vty.is_unsigned());
                            self.emit_line(format!(
                                "{call_kw} void @__syn_uart_kv(ptr {}, i32 {})",
                                p.repr, v32.repr
                            ));
                        }
                    }
                    "log_hex" => {
                        if let (Some((lbl, _)), Some((val, _))) = (ev.first(), ev.get(1)) {
                            let p = self.coerce_str_ptr(lbl.clone());
                            let v32 = self.coerce(val.clone(), "i32", false);
                            self.emit_line(format!(
                                "{call_kw} void @__syn_uart_log_hex(ptr {}, i32 {})",
                                p.repr, v32.repr
                            ));
                        }
                    }
                    "tick" | "tick_ms" => {
                        self.emit_line(format!("{call_kw} void @__syn_sys_tick()"));
                    }
                    "tick_by" => {
                        if let Some((ms, mty)) = ev.first() {
                            let m32 = self.coerce(ms.clone(), "i32", !mty.is_unsigned());
                            self.emit_line(format!(
                                "{call_kw} void @__syn_sys_tick_by(i32 {})",
                                m32.repr
                            ));
                        }
                    }
                    "nop" => {
                        self.emit_line("call void asm sideeffect \"nop\", \"~{memory}\"()");
                    }
                    "assert" => {
                        if let Some((cond, _)) = ev.first() {
                            self.emit_runtime_assert(cond.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    pub(super) fn emit_hook_registration(
        &mut self,
        kind: HookKind,
        expr_opt: Option<&MirExpr>,
    ) -> Result<(), CodegenError> {
        let Some(mut expr) = expr_opt else {
            return Ok(());
        };
        let mode = self.ctx.hooks.mode(kind).clone();
        match mode {
            StaticHookMode::Unset | StaticHookMode::DirectUnconditional { .. } => Ok(()),
            StaticHookMode::TagSwitch { ref global_sym, .. } => {
                while let MirExprKind::Cast { expr: inner, .. } = &expr.kind {
                    expr = inner;
                }
                if let MirExprKind::FnAddr(fn_name) = &expr.kind {
                    if let Some(tag) = mode.tag_index_for(fn_name) {
                        self.emit_line(format!("store i8 {tag}, ptr {global_sym}, align 1"));
                    }
                }
                Ok(())
            }
            StaticHookMode::Indirect { global_sym } => {
                let fp_val = self.emit_expr(expr)?;
                let fn_ptr_ty = self.ctx.spec.isa.fn_ptr_llvm_ty();
                let ptr_al = self.ctx.ptr_align();
                let coerced = self.coerce(fp_val, fn_ptr_ty, false);
                self.emit_line(format!(
                    "store {} {}, ptr {global_sym}, align {ptr_al}",
                    coerced.ty, coerced.repr
                ));
                Ok(())
            }
        }
    }

    pub(super) fn emit_rom_call_invocation(
        &mut self,
        kind: RomWrapperKind,
        addr_expr: &MirExpr,
        stack_bound: u32,
        abi_args: &[MirExpr],
    ) -> Result<Option<SsaVal>, CodegenError> {
        let call_kw = self.call_kw();
        let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
        let addr_raw = self.emit_expr(addr_expr)?;
        let addr_v = self.coerce(addr_raw, &usize_ty, false);

        let mut param_llvm_tys = Vec::with_capacity(abi_args.len());
        let mut call_args = Vec::with_capacity(1 + abi_args.len());
        call_args.push(format!("{usize_ty} {}", addr_v.repr));
        for arg in abi_args {
            let pty = self.ctx.llvm_ty(&arg.ty);
            let av = self.emit_expr(arg)?;
            let cv = self.coerce(av, &pty, !arg.ty.is_unsigned());
            param_llvm_tys.push(pty);
            call_args.push(format!("{} {}", cv.ty, cv.repr));
        }

        let wrapper = RomWrapperSpec {
            kind,
            stack_bound,
            param_llvm_tys,
        };
        let sym = wrapper.symbol();
        self.ctx.rom_wrappers.insert(wrapper);
        let args_str = call_args.join(", ");
        match kind {
            RomWrapperKind::U32 => {
                let rv = self.next_tmp();
                self.emit_line(format!("{rv} = {call_kw} i32 {sym}({args_str})"));
                Ok(Some(SsaVal::new("i32", rv)))
            }
            RomWrapperKind::Void => {
                self.emit_line(format!("{call_kw} void {sym}({args_str})"));
                Ok(None)
            }
            RomWrapperKind::Noreturn => {
                self.emit_line(format!("{call_kw} void {sym}({args_str})"));
                self.emit_term("unreachable");
                Ok(None)
            }
        }
    }
}
