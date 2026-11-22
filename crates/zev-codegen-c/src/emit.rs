//! C99 Code Generator lowering `zev_mir::MirModule` into clean, readable,
//! warning-free freestanding C99 source code.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write;

use zev_mir::{
    BlockId, MirAssignOp, MirBakedArg, MirBinOp, MirBlock, MirBootStep, MirCallDst,
    MirChildCallSlot, MirErrorSpec, MirExpr, MirExprKind, MirFloatFmt, MirFmtChunk, MirFnBody,
    MirFunction, MirIntFmt, MirIntrinsicOp, MirIsr, MirModule, MirPlace, MirPrimKind,
    MirPrimNonBlockingOp, MirPrimReadOp, MirPrimTarget, MirStmt, MirStruct, MirSwitchPattern,
    MirSysQuery, MirTerminator, MirType, MirUnaryOp, MirVar, MirYieldKind, VarId, VarRole,
};

use crate::runtime::c99_runtime_preamble;

/// Configuration options for C99 code generation.
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    /// Emit `int main(void)` entry point (`true` by default; can also be overridden via `-DSYN_NO_MAIN`).
    pub emit_main: bool,
    /// Default `SYN_SIM_MAX_TICKS` value in generated C99 (`50` ms by default if not overridden).
    pub default_sim_max_ticks: u32,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            emit_main: true,
            default_sim_max_ticks: 100,
        }
    }
}

/// Emit freestanding C99 code for `module` with default options.
pub fn emit_c(module: &MirModule) -> String {
    emit_c_with_options(module, &CodegenOptions::default())
}

/// Emit freestanding C99 code for `module` with explicit `options`.
pub fn emit_c_with_options(module: &MirModule, options: &CodegenOptions) -> String {
    let mut emitter = C99Emitter::new(module, options);
    emitter.generate()
}

struct C99Emitter<'a> {
    m: &'a MirModule,
    opts: &'a CodegenOptions,
    out: String,
    slices: BTreeMap<String, MirType>,
    arrays: BTreeMap<String, (MirType, usize)>,
    fn_ptrs: BTreeMap<String, (Vec<MirType>, MirType)>,
    results: BTreeMap<String, (MirType, Option<MirType>)>,
    prims: BTreeMap<String, MirPrimKind>,
    fn_param_names: BTreeMap<String, Vec<(VarId, MirType)>>,
    task_param_names: BTreeMap<String, Vec<(VarId, MirType)>>,
    builtin_c_syms: BTreeSet<String>,
    uses_float_sci: bool,
}

impl<'a> C99Emitter<'a> {
    fn new(m: &'a MirModule, opts: &'a CodegenOptions) -> Self {
        let mut builtin_c_syms: BTreeSet<String> = [
            "assert",
            "printf",
            "puts",
            "putchar",
            "print",
            "println",
            "syn_sim_enable_trace",
            "syn_sim_set_max_ticks",
            "syn_sim_stop",
            "syn_sim_exit",
            "syn_sim_log",
            "syn_sim_now_ms",
            "syn_sim_last_sleep_depth",
            "syn_sim_allowed_sleep_depth",
            "syn_sim_sleep_depth",
            "syn_sim_sleep_enter_count",
            "syn_sim_set_pp_dma_active",
            "syn_sim_set_dma_active",
            "syn_sim_push_dma_bytes",
            "syn_sim_push_dma_half",
            "syn_sim_pump_ready",
            "syn_sim_advance_ms",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        for isr in &m.isrs {
            if isr.params.is_empty() && isr.state_vars.is_empty() {
                builtin_c_syms.insert(format!("syn_sim_fire_isr_{}", sanitize_ident(&isr.name)));
            }
        }

        let mut fn_param_names = BTreeMap::new();
        for f in &m.functions {
            let ps: Vec<(VarId, MirType)> = f
                .params
                .iter()
                .map(|&vid| (vid, f.vars[vid.0].ty.clone()))
                .collect();
            fn_param_names.insert(f.name.clone(), ps);
        }

        let mut task_param_names = BTreeMap::new();
        for bp in &m.task_blueprints {
            let ps: Vec<(VarId, MirType)> = bp
                .params
                .iter()
                .map(|&vid| (vid, bp.vars[vid.0].ty.clone()))
                .collect();
            task_param_names.insert(bp.name.clone(), ps);
        }

        Self {
            m,
            opts,
            out: String::with_capacity(32 * 1024),
            slices: BTreeMap::new(),
            arrays: BTreeMap::new(),
            fn_ptrs: BTreeMap::new(),
            results: BTreeMap::new(),
            prims: BTreeMap::new(),
            fn_param_names,
            task_param_names,
            builtin_c_syms,
            uses_float_sci: false,
        }
    }

    fn generate(&mut self) -> String {
        self.collect_all_types();

        // 1. Embedded C99 cooperative kernel & simulation preamble
        self.out
            .push_str(&c99_runtime_preamble(self.opts.default_sim_max_ticks));
        if self.uses_float_sci {
            self.out.push_str(
                r#"
static SYN_UNUSED inline void syn_sci_mul_u32(uint64_t *pm, int32_t *pe2, uint32_t k) {
    uint64_t m_lo = (*pm) & 0xFFFFFFFFULL;
    uint64_t m_hi = (*pm) >> 32;
    uint64_t p0 = m_lo * (uint64_t)k;
    uint64_t p1 = m_hi * (uint64_t)k + (p0 >> 32);
    uint32_t top_bits = 0U;
    uint64_t tmp = p1 >> 32;
    while (tmp != 0ULL) { top_bits++; tmp >>= 1; }
    uint64_t full_lo = (p0 & 0xFFFFFFFFULL) | (p1 << 32);
    uint64_t full_hi = p1 >> 32;
    uint64_t norm = (full_lo >> top_bits) | (full_hi << (64U - top_bits));
    uint64_t rem = full_lo & ((1ULL << top_bits) - 1ULL);
    uint64_t half = 1ULL << (top_bits - 1U);
    if (rem > half || (rem == half && (norm & 1ULL) != 0ULL)) {
        norm += 1ULL;
        if (norm == 0ULL) {
            norm = 1ULL << 63;
            top_bits += 1U;
        }
    }
    *pm = norm;
    *pe2 += (int32_t)top_bits;
}

static SYN_UNUSED inline void syn_sci_div_u32(uint64_t *pm, int32_t *pe2, uint32_t k) {
    uint32_t kb = 0U;
    uint32_t kt = k;
    while (kt != 0U) { kb++; kt >>= 1; }
    uint64_t k_norm = ((uint64_t)k) << (64U - kb);
    uint32_t sh = ((*pm) >= k_norm) ? (kb - 1U) : kb;
    uint64_t top = (*pm) >> (64U - sh);
    uint64_t rest = (*pm) << sh;
    uint64_t d1 = (top << 32) | (rest >> 32);
    uint64_t q_hi = d1 / (uint64_t)k;
    uint64_t r1 = d1 % (uint64_t)k;
    uint64_t d0 = (r1 << 32) | (rest & 0xFFFFFFFFULL);
    uint64_t q_lo = d0 / (uint64_t)k;
    uint64_t rem = d0 % (uint64_t)k;
    uint64_t q = (q_hi << 32) | q_lo;
    if (rem * 2ULL > (uint64_t)k || (rem * 2ULL == (uint64_t)k && (q & 1ULL) != 0ULL)) {
        q += 1ULL;
        if (q == 0ULL) {
            q = 1ULL << 63;
            sh -= 1U;
        }
    }
    *pm = q;
    *pe2 -= (int32_t)sh;
}

static SYN_UNUSED void syn_fmt_float_sci(syn_fmt_sink_t *s, double val, uint8_t prec) {
    static const uint64_t pow10_tbl[10] = {
        1ULL, 10ULL, 100ULL, 1000ULL, 10000ULL,
        100000ULL, 1000000ULL, 10000000ULL, 100000000ULL, 1000000000ULL
    };
    uint64_t bits = syn_f64_to_bits(val);
    uint64_t abs_bits = bits & 0x7FFFFFFFFFFFFFFFULL;
    bool neg_bit = (bits >> 63) != 0ULL;
    if (abs_bits > 0x7FF0000000000000ULL) {
        syn_fmt_write(s, "nan", 3U);
        return;
    }
    if (abs_bits == 0x7FF0000000000000ULL) {
        if (neg_bit) { syn_fmt_write(s, "-inf", 4U); }
        else         { syn_fmt_write(s, "inf", 3U); }
        return;
    }
    uint8_t p = (prec > 9U) ? 9U : prec;
    if (abs_bits == 0ULL) {
        syn_fmt_putc(s, '0');
        if (p > 0U) {
            syn_fmt_putc(s, '.');
            syn_fmt_u64_dec_pad(s, 0ULL, p);
        }
        syn_fmt_write(s, "e+00", 4U);
        return;
    }
    if (neg_bit) {
        syn_fmt_putc(s, '-');
    }
    uint32_t raw_exp = (uint32_t)(abs_bits >> 52);
    uint64_t raw_frac = abs_bits & 0x000FFFFFFFFFFFFFULL;
    uint64_t m = 0ULL;
    int32_t e2 = 0;
    if (raw_exp == 0U) {
        m = raw_frac;
        e2 = -1074;
        while ((m & (1ULL << 63)) == 0ULL) {
            m <<= 1;
            e2 -= 1;
        }
    } else {
        m = (raw_frac | (1ULL << 52)) << 11;
        e2 = (int32_t)raw_exp - 1086;
    }
    int32_t exp10 = 0;
    while (e2 + 63 >= 30) {
        syn_sci_div_u32(&m, &e2, 1000000000U);
        exp10 += 9;
    }
    while (e2 + 64 <= -30) {
        syn_sci_mul_u32(&m, &e2, 1000000000U);
        exp10 -= 9;
    }
    while (e2 > -60 || (e2 == -60 && m >= (10ULL << 60))) {
        syn_sci_div_u32(&m, &e2, 10U);
        exp10 += 1;
    }
    while (e2 < -63) {
        syn_sci_mul_u32(&m, &e2, 10U);
        exp10 -= 1;
    }
    uint64_t scale = pow10_tbl[p];
    if (p > 0U) {
        syn_sci_mul_u32(&m, &e2, (uint32_t)scale);
    }
    uint32_t rsh = (uint32_t)(-e2);
    uint64_t half = 1ULL << (rsh - 1U);
    uint64_t scaled = (m >> rsh) + (((m & ((1ULL << rsh) - 1ULL)) >= half) ? 1ULL : 0ULL);
    if (scaled >= scale * 10ULL) {
        scaled = scale;
        exp10 += 1;
    }
    uint64_t lead = scaled / scale;
    uint64_t frac = scaled % scale;
    syn_fmt_putc(s, (char)('0' + (uint32_t)lead));
    if (p > 0U) {
        syn_fmt_putc(s, '.');
        syn_fmt_u64_dec_pad(s, frac, p);
    }
    syn_fmt_putc(s, 'e');
    if (exp10 < 0) {
        syn_fmt_putc(s, '-');
        exp10 = -exp10;
    } else {
        syn_fmt_putc(s, '+');
    }
    syn_fmt_u64_dec_pad(s, (uint64_t)(uint32_t)exp10, 2U);
}
"#,
            );
        }
        self.out.push_str(
            r#"
#if !defined(SYN_FREESTANDING)
static SYN_UNUSED inline void print(const void *s) { if (s != NULL) { fputs((const char *)s, stdout); fflush(stdout); } }
static SYN_UNUSED inline void println(const void *s) { if (s != NULL) { puts((const char *)s); fflush(stdout); } }
static SYN_UNUSED inline void syn_builtin_assert(bool cond) { if (!cond) { fputs("Zev assertion failed!\n", stderr); fflush(stderr); abort(); } }
#else
static SYN_UNUSED inline void print(const void *s) { (void)s; }
static SYN_UNUSED inline void println(const void *s) { (void)s; }
static SYN_UNUSED inline void syn_builtin_assert(bool cond) { if (!cond) { for (;;) {} } }
#endif

"#,
        );

        // 2. Built-in default hardware bit constants (guarded by #ifndef)
        self.out.push_str(
            r#"#ifndef MOTOR_EN
#define MOTOR_EN (1U << 0)
#endif
#ifndef HALF_TRANSFER
#define HALF_TRANSFER (1U << 1)
#endif
#ifndef FULL_TRANSFER
#define FULL_TRANSFER (1U << 2)
#endif

"#,
        );

        // 3. Enums & Structs (plus forward declarations, slices, arrays, results, `from_bytes`)
        self.emit_enums_and_structs();

        // 4. Concrete OS Primitive Types & Helpers
        self.emit_primitive_types();

        // 5. Top-level Constants, Static OS Primitive Instances, and `extern "C"` Prototypes
        self.emit_globals_and_externs();

        // 6. Coroutine Frame Structs (topologically ordered) + Static Task Slots & TCB Table
        self.emit_frames_and_task_table();

        // 7. Forward Prototypes for ISRs & User Functions
        self.emit_fn_prototypes();

        // 8. Hardware ISRs + Deterministic Simulation Fire Hooks
        self.emit_isrs();

        // 9. User Functions (Direct & Yielding State Machines) + Task Blueprint Step Functions
        self.emit_functions_and_tasks();

        // 10. Sequential Boot Initializer + `main()`
        self.emit_boot_and_main();

        std::mem::take(&mut self.out)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass A: Type & Specialization Discovery
    // ─────────────────────────────────────────────────────────────────────────

    fn collect_all_types(&mut self) {
        let structs = self.m.structs.clone();
        for st in &structs {
            for f in &st.fields {
                self.register_type(&f.ty);
            }
            // Register Result type for `<Struct>.from_bytes(...)`
            self.register_type(&MirType::Result {
                ok: Box::new(MirType::Struct(st.name.clone())),
                err: None,
            });
        }
        for c in &self.m.consts {
            self.register_type(&c.ty);
            self.register_expr(&c.init);
        }
        for p in &self.m.primitives {
            self.register_prim_kind(&p.kind);
            if let Some(ie) = &p.init {
                self.register_expr(ie);
            }
        }
        for ef in &self.m.extern_fns {
            for (_, pty) in &ef.params {
                self.register_type(pty);
            }
            self.register_type(&ef.ret_ty);
        }
        for isr in &self.m.isrs {
            for sv in &isr.state_vars {
                self.register_type(&sv.ty);
                self.register_expr(&sv.init);
            }
            for v in &isr.vars {
                self.register_type(&v.ty);
            }
            for s in &isr.body {
                self.register_stmt(s);
            }
        }
        for vec in &self.m.irq_vectors {
            for b in &vec.bindings {
                for sv in &b.state_vars {
                    self.register_type(&sv.ty);
                    self.register_expr(&sv.init);
                }
                for ba in &b.baked_args {
                    if let MirBakedArg::Const(e) = ba {
                        self.register_expr(e);
                    }
                }
            }
        }
        for func in &self.m.functions {
            let env_ret = func.envelope_return_type();
            self.register_type(&env_ret);
            for v in &func.vars {
                self.register_type(&v.ty);
            }
            match &func.body {
                MirFnBody::Direct(stmts) => {
                    for s in stmts {
                        self.register_stmt(s);
                    }
                }
                MirFnBody::Coroutine { blocks, .. } => {
                    for b in blocks {
                        for s in &b.stmts {
                            self.register_stmt(s);
                        }
                        self.register_terminator(&b.terminator);
                    }
                }
            }
        }
        for bp in &self.m.task_blueprints {
            for v in &bp.vars {
                self.register_type(&v.ty);
            }
            for b in &bp.blocks {
                for s in &b.stmts {
                    self.register_stmt(s);
                }
                self.register_terminator(&b.terminator);
            }
        }
        for v in &self.m.boot.vars {
            self.register_type(&v.ty);
        }
        for step in &self.m.boot.steps {
            match step {
                MirBootStep::Stmt(s) => self.register_stmt(s),
                MirBootStep::RunYieldingFn { args, .. } | MirBootStep::SpawnActivate { args, .. } => {
                    for a in args {
                        self.register_expr(a);
                    }
                }
            }
        }
    }

    fn register_type(&mut self, ty: &MirType) {
        match ty {
            MirType::Slice(inner) => {
                self.register_type(inner);
                if **inner != MirType::U8 {
                    self.slices.insert(ty.mangle_name(), *inner.clone());
                }
            }
            MirType::Array { elem, len } => {
                self.register_type(elem);
                self.arrays
                    .insert(ty.mangle_name(), (*elem.clone(), *len));
            }
            MirType::Ptr(inner) => self.register_type(inner),
            MirType::Primitive(pk) => self.register_prim_kind(pk),
            MirType::FnPtr { params, ret } => {
                for p in params {
                    self.register_type(p);
                }
                self.register_type(ret);
                self.fn_ptrs
                    .insert(ty.mangle_name(), (params.clone(), *ret.clone()));
            }
            MirType::Result { ok, err } => {
                self.register_type(ok);
                if let Some(e) = err {
                    self.register_type(e);
                }
                self.results.insert(
                    ty.mangle_name(),
                    (*ok.clone(), err.as_deref().cloned()),
                );
            }
            _ => {}
        }
    }

    fn register_prim_kind(&mut self, pk: &MirPrimKind) {
        match pk {
            MirPrimKind::Signal => {}
            MirPrimKind::Watch { elem } => {
                self.register_type(elem);
            }
            MirPrimKind::Inbox { elem, .. } => {
                self.register_type(elem);
                self.register_type(&MirType::Result {
                    ok: elem.clone(),
                    err: None,
                });
            }
            MirPrimKind::PingPong { elem, .. } | MirPrimKind::Lease { elem, .. } => {
                self.register_type(elem);
                self.register_type(&MirType::Slice(elem.clone()));
            }
        }
        self.prims.insert(pk.mangle_name(), pk.clone());
    }

    fn register_prim_target(&mut self, target: &MirPrimTarget) {
        if let MirPrimTarget::PtrExpr(pe) = target {
            self.register_expr(pe);
        }
    }

    fn register_stmt(&mut self, stmt: &MirStmt) {
        match stmt {
            MirStmt::Assign { value, .. } | MirStmt::Expr(value) => self.register_expr(value),
            MirStmt::SpawnActivate { args, .. } => {
                for a in args {
                    self.register_expr(a);
                }
            }
            MirStmt::PrintFmt { chunks, .. } => {
                for ch in chunks {
                    self.register_fmt_chunk(ch);
                }
            }
            MirStmt::FmtBuf { buf, chunks, .. } => {
                self.register_expr(buf);
                self.register_type(&MirType::Result {
                    ok: Box::new(MirType::Slice(Box::new(MirType::U8))),
                    err: Some(Box::new(MirType::Enum("FmtErr".to_string()))),
                });
                for ch in chunks {
                    self.register_fmt_chunk(ch);
                }
            }
            MirStmt::PrimOp(op) => match op {
                MirPrimNonBlockingOp::WatchPublish {
                    target,
                    elem_ty,
                    value,
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Watch {
                        elem: Box::new(elem_ty.clone()),
                    });
                    self.register_expr(value);
                }
                MirPrimNonBlockingOp::InboxPost {
                    target,
                    elem_ty,
                    cap,
                    value,
                    ..
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                    self.register_expr(value);
                }
                MirPrimNonBlockingOp::SignalNotify { target, .. }
                | MirPrimNonBlockingOp::SignalClear { target } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Signal);
                }
                MirPrimNonBlockingOp::PingPongFlip {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimNonBlockingOp::PingPongSetActive {
                    target,
                    elem_ty,
                    cap,
                    ..
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::PingPong {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
                MirPrimNonBlockingOp::LeaseComplete {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimNonBlockingOp::LeaseSetActive {
                    target,
                    elem_ty,
                    cap,
                    ..
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Lease {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
            },
            MirStmt::Intrinsic(intr) => match intr {
                MirIntrinsicOp::UseSystick { hz } => self.register_expr(hz),
                MirIntrinsicOp::SetPriority { .. } => {}
                MirIntrinsicOp::SysOp { args, .. } => {
                    for a in args {
                        self.register_expr(a);
                    }
                }
            },
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.register_expr(cond);
                for s in then_body {
                    self.register_stmt(s);
                }
                for s in else_body {
                    self.register_stmt(s);
                }
            }
            MirStmt::While { cond, body } => {
                self.register_expr(cond);
                for s in body {
                    self.register_stmt(s);
                }
            }
            MirStmt::Switch {
                discr,
                arms,
                default,
            } => {
                self.register_expr(discr);
                for (_, body) in arms {
                    for s in body {
                        self.register_stmt(s);
                    }
                }
                for s in default {
                    self.register_stmt(s);
                }
            }
            MirStmt::Return(Some(e)) => self.register_expr(e),
            MirStmt::Error { reason, partial } => {
                if let Some(r) = reason {
                    self.register_expr(r);
                }
                if let Some(p) = partial {
                    self.register_expr(p);
                }
            }
            MirStmt::MmioWrite {
                base,
                index,
                value,
                ..
            } => {
                self.register_expr(base);
                if let Some(idx) = index {
                    self.register_expr(idx);
                }
                self.register_expr(value);
            }
            _ => {}
        }
    }

    fn register_terminator(&mut self, term: &MirTerminator) {
        match term {
            MirTerminator::Branch { cond, .. } | MirTerminator::Switch { discr: cond, .. } => {
                self.register_expr(cond);
            }
            MirTerminator::Return(Some(e)) => self.register_expr(e),
            MirTerminator::Error { reason, partial } => {
                if let Some(r) = reason {
                    self.register_expr(r);
                }
                if let Some(p) = partial {
                    self.register_expr(p);
                }
            }
            MirTerminator::Yield(yk) => match yk {
                MirYieldKind::InboxRecv {
                    target,
                    elem_ty,
                    cap,
                    ..
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
                MirYieldKind::InboxSend {
                    target,
                    elem_ty,
                    cap,
                    dst_res,
                    ..
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                    if dst_res.is_some() {
                        self.register_type(&MirType::Result {
                            ok: Box::new(MirType::Void),
                            err: None,
                        });
                    }
                }
                MirYieldKind::SignalWait { target, dst_res, .. } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Signal);
                    if dst_res.is_some() {
                        self.register_type(&MirType::Result {
                            ok: Box::new(MirType::Void),
                            err: None,
                        });
                    }
                }
                MirYieldKind::PingPongWaitChunk {
                    target,
                    elem_ty,
                    cap,
                    ..
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::PingPong {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
                MirYieldKind::LeaseWaitChunk {
                    target,
                    elem_ty,
                    cap,
                    ..
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Lease {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
                MirYieldKind::CallYielding { args, .. } => {
                    for a in args {
                        self.register_expr(a);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn register_expr(&mut self, expr: &MirExpr) {
        self.register_type(&expr.ty);
        match &expr.kind {
            MirExprKind::Unary { expr: inner, .. }
            | MirExprKind::Cast { expr: inner, .. }
            | MirExprKind::FieldAccess { base: inner, .. }
            | MirExprKind::SliceLen(inner)
            | MirExprKind::FromBytes { slice: inner, .. } => self.register_expr(inner),
            MirExprKind::Binary { lhs, rhs, .. }
            | MirExprKind::Index {
                base: lhs,
                index: rhs,
                ..
            } => {
                self.register_expr(lhs);
                self.register_expr(rhs);
            }
            MirExprKind::SubSlice {
                base, start, end, ..
            } => {
                self.register_expr(base);
                self.register_expr(start);
                self.register_expr(end);
            }
            MirExprKind::StructLit { fields, .. } => {
                for (_, fe, _) in fields {
                    self.register_expr(fe);
                }
            }
            MirExprKind::ArrayLit { elems, .. } | MirExprKind::CallDirect { args: elems, .. } => {
                for e in elems {
                    self.register_expr(e);
                }
            }
            MirExprKind::PrimRead(ro) => match ro.as_ref() {
                MirPrimReadOp::WatchRead { target, elem_ty } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Watch {
                        elem: Box::new(elem_ty.clone()),
                    });
                }
                MirPrimReadOp::InboxTryRecv {
                    target,
                    elem_ty,
                    cap,
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                    self.register_type(&MirType::Result {
                        ok: Box::new(elem_ty.clone()),
                        err: None,
                    });
                }
                MirPrimReadOp::InboxTrySend {
                    target,
                    elem_ty,
                    cap,
                    value,
                    ..
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                    self.register_expr(value);
                }
                MirPrimReadOp::InboxLen {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::InboxIsEmpty {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::InboxIsFull {
                    target,
                    elem_ty,
                    cap,
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
                MirPrimReadOp::SignalIsSet { target } | MirPrimReadOp::SignalTryWait { target } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::Signal);
                }
                MirPrimReadOp::PingPongDmaBuf {
                    target,
                    elem_ty,
                    cap,
                } => {
                    self.register_prim_target(target);
                    self.register_prim_kind(&MirPrimKind::PingPong {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                    self.register_type(&MirType::Slice(Box::new(elem_ty.clone())));
                }
            },
            MirExprKind::MmioRead { base, index, .. } => {
                self.register_expr(base);
                if let Some(idx) = index {
                    self.register_expr(idx);
                }
            }
            _ => {}
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // C Type Formatting Helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn register_fmt_chunk(&mut self, ch: &MirFmtChunk) {
        match ch {
            MirFmtChunk::Literal(_) => {}
            MirFmtChunk::Float { value, fmt, .. } => {
                if matches!(fmt, MirFloatFmt::Sci { .. }) {
                    self.uses_float_sci = true;
                }
                self.register_expr(value);
            }
            MirFmtChunk::Int { value, .. }
            | MirFmtChunk::Bool(value)
            | MirFmtChunk::Str(value)
            | MirFmtChunk::Bytes(value)
            | MirFmtChunk::EnumName { value, .. } => {
                self.register_expr(value);
            }
            MirFmtChunk::DebugSlice { elem_ty, slice } => {
                self.register_type(elem_ty);
                self.register_expr(slice);
            }
        }
    }

    fn c_type(&self, ty: &MirType) -> String {
        match ty {
            MirType::Void => "void".to_string(),
            MirType::Bool => "bool".to_string(),
            MirType::U8 => "uint8_t".to_string(),
            MirType::U16 => "uint16_t".to_string(),
            MirType::U32 => "uint32_t".to_string(),
            MirType::U64 => "uint64_t".to_string(),
            MirType::I8 => "int8_t".to_string(),
            MirType::I16 => "int16_t".to_string(),
            MirType::I32 => "int32_t".to_string(),
            MirType::I64 => "int64_t".to_string(),
            MirType::Usize => "size_t".to_string(),
            MirType::Isize => "ptrdiff_t".to_string(),
            MirType::F32 => "float".to_string(),
            MirType::F64 => "double".to_string(),
            MirType::Str => "const char *".to_string(),
            MirType::EndianInt {
                bits, big_endian, ..
            } => {
                let end = if *big_endian { "be" } else { "le" };
                format!("syn_u{bits}{end}_t")
            }
            MirType::Enum(name) => format!("syn_enum_{name}"),
            MirType::Struct(name) => format!("syn_struct_{name}"),
            MirType::Slice(inner) => {
                if **inner == MirType::U8 {
                    "syn_slice_u8".to_string()
                } else {
                    format!("syn_{}", ty.mangle_name())
                }
            }
            MirType::Array { .. } => format!("syn_{}_t", ty.mangle_name()),
            MirType::Ptr(inner) => format!("{} *", self.c_type(inner)),
            MirType::Primitive(pk) => format!("syn_{}_t", pk.mangle_name()),
            MirType::MmioPtr(_) => "volatile uint8_t *".to_string(),
            MirType::FnPtr { .. } => format!("syn_{}_t", ty.mangle_name()),
            MirType::Result { .. } => format!("syn_{}_t", ty.mangle_name()),
        }
    }

    fn c_decl(&self, ty: &MirType, ident: &str) -> String {
        format!("{} {ident}", self.c_type(ty))
    }

    fn zero_init_expr(&self, ty: &MirType) -> String {
        match ty {
            MirType::Void => String::new(),
            MirType::Bool => "false".to_string(),
            MirType::U8
            | MirType::U16
            | MirType::U32
            | MirType::U64
            | MirType::Usize => format!("(({})0U)", self.c_type(ty)),
            MirType::I8 | MirType::I16 | MirType::I32 | MirType::I64 | MirType::Isize => {
                format!("(({})0)", self.c_type(ty))
            }
            MirType::F32 => "0.0f".to_string(),
            MirType::F64 => "0.0".to_string(),
            MirType::Str | MirType::Ptr(_) | MirType::MmioPtr(_) | MirType::FnPtr { .. } => {
                "NULL".to_string()
            }
            MirType::Enum(name) => format!("((syn_enum_{name})0)"),
            MirType::EndianInt {
                bits, big_endian, ..
            } => {
                let end = if *big_endian { "be" } else { "le" };
                format!("((syn_u{bits}{end}_t){{ .b = {{ 0U }} }})")
            }
            MirType::Slice(_) => format!("(({}){{ .ptr = NULL, .len = 0U }})", self.c_type(ty)),
            MirType::Array { .. } => format!("(({}){{ .data = {{ 0 }} }})", self.c_type(ty)),
            MirType::Struct(sname) => {
                if let Some(st) = self.m.structs.iter().find(|s| s.name == *sname) {
                    if let Some(first) = st.fields.first() {
                        return format!(
                            "(({}){{ .{} = {} }})",
                            self.c_type(ty),
                            c_ident(&first.name),
                            self.zero_init_expr(&first.ty)
                        );
                    }
                }
                format!("(({}){{ ._empty = 0U }})", self.c_type(ty))
            }
            MirType::Result { .. } => format!("(({}){{ .is_err = false }})", self.c_type(ty)),
            MirType::Primitive(_) => format!("(({}){{ 0 }})", self.c_type(ty)),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 3: Enums, Structs, Slices, Arrays, Dual-Exit Results, `from_bytes`
    // ─────────────────────────────────────────────────────────────────────────

    fn emit_enums_and_structs(&mut self) {
        self.out
            .push_str("/* ── User Enums, Wire-Exact Structs, Slices, & Dual-Exit Envelopes ────── */\n\n");

        for en in &self.m.enums {
            let _ = writeln!(self.out, "typedef enum syn_enum_{} {{", en.name);
            for (idx, (vname, disc)) in en.variants.iter().enumerate() {
                let comma = if idx + 1 < en.variants.len() { "," } else { "" };
                let _ = writeln!(self.out, "    syn_{}_{} = {}{}", en.name, vname, disc, comma);
            }
            let _ = writeln!(self.out, "}} syn_enum_{};\n", en.name);

            // Emit `.rodata` variant-name lookup table ONLY for enums used in `f"..."` formatting!
            if self.m.formatted_enums.iter().any(|fe| fe == &en.name) {
                let _ = writeln!(
                    self.out,
                    "static SYN_UNUSED inline const char *syn_enum_name_{0}(syn_enum_{0} _v) {{\n    switch (_v) {{",
                    en.name
                );
                let mut seen_discs = BTreeSet::new();
                for (vname, disc) in &en.variants {
                    if seen_discs.insert(*disc) {
                        let _ = writeln!(
                            self.out,
                            "        case syn_{}_{}: return {};",
                            en.name,
                            vname,
                            escape_c_string(vname)
                        );
                    }
                }
                let _ = writeln!(
                    self.out,
                    "        default: return {};\n    }}\n}}\n",
                    escape_c_string(&format!("{}(?)", en.name))
                );
            }
        }

        // Forward-declare structs, fixed-size arrays, slices, results, & primitives so pointer/fnptr/slice typedefs can reference any of them
        let arrays_snapshot = self.arrays.clone();
        let slices_snapshot = self.slices.clone();
        let fn_ptrs_snapshot = self.fn_ptrs.clone();
        let results_snapshot = self.results.clone();
        let prims_snapshot = self.prims.clone();

        for st in &self.m.structs {
            let _ = writeln!(
                self.out,
                "typedef struct syn_struct_{0} syn_struct_{0};",
                st.name
            );
        }
        for mangle in arrays_snapshot.keys() {
            let _ = writeln!(
                self.out,
                "typedef struct syn_{0}_t syn_{0}_t;",
                mangle
            );
        }
        for mangle in slices_snapshot.keys() {
            let _ = writeln!(
                self.out,
                "typedef struct syn_{0} syn_{0};",
                mangle
            );
        }
        for mangle in results_snapshot.keys() {
            let _ = writeln!(
                self.out,
                "typedef struct syn_{0}_t syn_{0}_t;",
                mangle
            );
        }
        for mangle in prims_snapshot.keys() {
            let _ = writeln!(
                self.out,
                "typedef struct syn_{0}_t syn_{0}_t;",
                mangle
            );
        }
        if !self.m.structs.is_empty()
            || !arrays_snapshot.is_empty()
            || !slices_snapshot.is_empty()
            || !results_snapshot.is_empty()
            || !prims_snapshot.is_empty()
        {
            self.out.push('\n');
        }

        // Emit function-pointer typedefs in post-order DFS over nested FnPtr signatures
        let mut emitted_fn_ptrs: BTreeSet<String> = BTreeSet::new();
        for (mangle, (params, ret)) in &fn_ptrs_snapshot {
            self.emit_fn_ptr_topo(mangle, params, ret, &fn_ptrs_snapshot, &mut emitted_fn_ptrs);
        }
        if !fn_ptrs_snapshot.is_empty() {
            self.out.push('\n');
        }

        // Emit non-u8 slice struct bodies (inner-first so nested slices `[][]T` are clean)
        let mut emitted_slices: BTreeSet<String> = BTreeSet::new();
        for (mangle, elem) in &slices_snapshot {
            self.emit_slice_topo(mangle, elem, &slices_snapshot, &mut emitted_slices);
        }

        // Joint topological emission over `{ struct bodies ∪ fixed-size array wrappers }` so every
        // by-value dependency (`[N]SomeStruct`, `FixedVec<FixedVec<u8, 2>, 2>`, reverse-declared
        // outer-before-inner structs) is defined before its dependent type!
        let structs_by_name: BTreeMap<&str, &MirStruct> =
            self.m.structs.iter().map(|s| (s.name.as_str(), s)).collect();
        let mut emitted_structs: BTreeSet<String> = BTreeSet::new();
        let mut emitted_arrays: BTreeSet<String> = BTreeSet::new();

        for (mangle, (elem, len)) in &arrays_snapshot {
            self.emit_array_topo(
                mangle,
                elem,
                *len,
                &structs_by_name,
                &arrays_snapshot,
                &mut emitted_structs,
                &mut emitted_arrays,
            );
        }
        for st in &self.m.structs {
            self.emit_struct_topo(
                st,
                &structs_by_name,
                &arrays_snapshot,
                &mut emitted_structs,
                &mut emitted_arrays,
            );
        }

        // Emit slice subslice helpers AFTER all struct/array bodies are complete (so `s.ptr + start` has complete sizeof!)
        for (mangle, _) in &slices_snapshot {
            let _ = writeln!(
                self.out,
                "static SYN_UNUSED inline syn_{mangle} syn_sub_{mangle}(syn_{mangle} s, size_t start, size_t end) {{\n    if (start > s.len) {{ start = s.len; }}\n    if (end > s.len) {{ end = s.len; }}\n    if (end < start) {{ end = start; }}\n    syn_{mangle} out;\n    out.ptr = (s.ptr != NULL) ? (s.ptr + start) : NULL;\n    out.len = end - start;\n    return out;\n}}\n"
            );
        }

        // Emit dual-exit result envelopes (`{ bool is_err; [Err err;] [Val val;] }`)
        for (mangle, (ok_ty, err_ty_opt)) in &results_snapshot {
            let _ = writeln!(self.out, "struct syn_{mangle}_t {{");
            self.out.push_str("    bool is_err;\n");
            if let Some(et) = err_ty_opt {
                if !et.is_void() {
                    let _ = writeln!(self.out, "    {};", self.c_decl(et, "err"));
                }
            }
            if !ok_ty.is_void() {
                let _ = writeln!(self.out, "    {};", self.c_decl(ok_ty, "val"));
            }
            let _ = writeln!(self.out, "}};\n");
        }

        // Emit `<Struct>_from_bytes(syn_slice_u8)` helpers
        for st in &self.m.structs {
            let res_mangle = MirType::Result {
                ok: Box::new(MirType::Struct(st.name.clone())),
                err: None,
            }
            .mangle_name();
            let _ = writeln!(
                self.out,
                "static SYN_UNUSED inline syn_{res_mangle}_t syn_{s}_from_bytes(syn_slice_u8 _s) {{\n    syn_{res_mangle}_t _r = {{ .is_err = true }};\n    if (_s.ptr == NULL || _s.len < sizeof(syn_struct_{s})) {{\n        return _r;\n    }}\n    _r.is_err = false;\n    memcpy(&_r.val, _s.ptr, sizeof(syn_struct_{s}));\n    return _r;\n}}\n",
                s = st.name
            );
        }
    }

    fn emit_fn_ptr_dep_topo(
        &mut self,
        ty: &MirType,
        fn_ptrs_snapshot: &BTreeMap<String, (Vec<MirType>, MirType)>,
        emitted_fn_ptrs: &mut BTreeSet<String>,
    ) {
        match ty {
            MirType::FnPtr { params, ret } => {
                let mangle = ty.mangle_name();
                if let Some((ps, r)) = fn_ptrs_snapshot.get(&mangle) {
                    self.emit_fn_ptr_topo(&mangle, ps, r, fn_ptrs_snapshot, emitted_fn_ptrs);
                } else {
                    self.emit_fn_ptr_topo(&mangle, params, ret, fn_ptrs_snapshot, emitted_fn_ptrs);
                }
            }
            MirType::Ptr(inner) | MirType::Slice(inner) | MirType::Array { elem: inner, .. } => {
                self.emit_fn_ptr_dep_topo(inner, fn_ptrs_snapshot, emitted_fn_ptrs);
            }
            MirType::Result { ok, err } => {
                self.emit_fn_ptr_dep_topo(ok, fn_ptrs_snapshot, emitted_fn_ptrs);
                if let Some(e) = err {
                    self.emit_fn_ptr_dep_topo(e, fn_ptrs_snapshot, emitted_fn_ptrs);
                }
            }
            _ => {}
        }
    }

    fn emit_fn_ptr_topo(
        &mut self,
        mangle: &str,
        params: &[MirType],
        ret: &MirType,
        fn_ptrs_snapshot: &BTreeMap<String, (Vec<MirType>, MirType)>,
        emitted_fn_ptrs: &mut BTreeSet<String>,
    ) {
        if !emitted_fn_ptrs.insert(mangle.to_string()) {
            return;
        }
        for p in params {
            self.emit_fn_ptr_dep_topo(p, fn_ptrs_snapshot, emitted_fn_ptrs);
        }
        self.emit_fn_ptr_dep_topo(ret, fn_ptrs_snapshot, emitted_fn_ptrs);
        let ps = if params.is_empty() {
            "void".to_string()
        } else {
            params
                .iter()
                .map(|p| self.c_type(p))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let rt = self.c_type(ret);
        let _ = writeln!(self.out, "typedef {rt} (*syn_{mangle}_t)({ps});");
    }

    fn emit_slice_topo(
        &mut self,
        mangle: &str,
        elem: &MirType,
        slices_snapshot: &BTreeMap<String, MirType>,
        emitted_slices: &mut BTreeSet<String>,
    ) {
        if !emitted_slices.insert(mangle.to_string()) {
            return;
        }
        if let MirType::Slice(inner) = elem {
            if **inner != MirType::U8 {
                let child_mangle = elem.mangle_name();
                if let Some(child_elem) = slices_snapshot.get(&child_mangle) {
                    self.emit_slice_topo(&child_mangle, child_elem, slices_snapshot, emitted_slices);
                }
            }
        }
        let ec = self.c_type(elem);
        let _ = writeln!(
            self.out,
            "struct syn_{mangle} {{\n    {ec} *ptr;\n    size_t len;\n}};\n"
        );
    }

    fn emit_value_dep_topo(
        &mut self,
        ty: &MirType,
        structs_by_name: &BTreeMap<&str, &MirStruct>,
        arrays_snapshot: &BTreeMap<String, (MirType, usize)>,
        emitted_structs: &mut BTreeSet<String>,
        emitted_arrays: &mut BTreeSet<String>,
    ) {
        match ty {
            MirType::Struct(sname) => {
                if let Some(st) = structs_by_name.get(sname.as_str()).copied() {
                    self.emit_struct_topo(
                        st,
                        structs_by_name,
                        arrays_snapshot,
                        emitted_structs,
                        emitted_arrays,
                    );
                }
            }
            MirType::Array { elem, len } => {
                let mangle = ty.mangle_name();
                let (e_ty, arr_len) = arrays_snapshot
                    .get(&mangle)
                    .map(|(e, l)| (e, *l))
                    .unwrap_or((elem.as_ref(), *len));
                self.emit_array_topo(
                    &mangle,
                    e_ty,
                    arr_len,
                    structs_by_name,
                    arrays_snapshot,
                    emitted_structs,
                    emitted_arrays,
                );
            }
            _ => {}
        }
    }

    fn emit_array_topo(
        &mut self,
        mangle: &str,
        elem: &MirType,
        len: usize,
        structs_by_name: &BTreeMap<&str, &MirStruct>,
        arrays_snapshot: &BTreeMap<String, (MirType, usize)>,
        emitted_structs: &mut BTreeSet<String>,
        emitted_arrays: &mut BTreeSet<String>,
    ) {
        if !emitted_arrays.insert(mangle.to_string()) {
            return;
        }
        self.emit_value_dep_topo(
            elem,
            structs_by_name,
            arrays_snapshot,
            emitted_structs,
            emitted_arrays,
        );
        let ec = self.c_type(elem);
        let n = len.max(1);
        let _ = writeln!(self.out, "struct syn_{mangle}_t {{\n    {ec} data[{n}];\n}};\n");
    }

    fn emit_struct_topo(
        &mut self,
        st: &MirStruct,
        structs_by_name: &BTreeMap<&str, &MirStruct>,
        arrays_snapshot: &BTreeMap<String, (MirType, usize)>,
        emitted_structs: &mut BTreeSet<String>,
        emitted_arrays: &mut BTreeSet<String>,
    ) {
        if !emitted_structs.insert(st.name.clone()) {
            return;
        }
        for f in &st.fields {
            self.emit_value_dep_topo(
                &f.ty,
                structs_by_name,
                arrays_snapshot,
                emitted_structs,
                emitted_arrays,
            );
        }
        let packed_attr = if st.packed { " SYN_PACKED" } else { "" };
        let _ = writeln!(self.out, "struct{packed_attr} syn_struct_{} {{", st.name);
        if st.fields.is_empty() {
            self.out.push_str("    uint8_t _empty;\n");
        } else {
            for f in &st.fields {
                let fc = self.c_type(&f.ty);
                let _ = writeln!(self.out, "    {} {};", fc, c_ident(&f.name));
            }
        }
        let _ = writeln!(self.out, "}};\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 4: Concrete OS Primitive Structs & Inline Methods
    // ─────────────────────────────────────────────────────────────────────────

    fn emit_primitive_types(&mut self) {
        if self.prims.is_empty() {
            return;
        }
        self.out
            .push_str("/* ── Static OS & DMA Primitives (Zero-Heap, ISR- & Scheduler-Aware) ───── */\n\n");

        let prims_snapshot = self.prims.clone();
        for (mangle, pk) in &prims_snapshot {
            match pk {
                MirPrimKind::Signal => {
                    self.out.push_str(
                        r#"struct syn_signal_t {
    syn_task_t *waiter;
    volatile bool flag;
    uint8_t boost_prio;
};

static SYN_UNUSED inline bool syn_signal_is_ready(void *obj) {
    const syn_signal_t *s = (const syn_signal_t *)obj;
    return s != NULL && s->flag;
}
static SYN_UNUSED inline void syn_signal_notify(syn_signal_t *s, uint8_t boost_prio) {
    syn_enter_critical();
    s->flag = true;
    s->boost_prio = boost_prio;
    syn_task_t *w = s->waiter;
    s->waiter = NULL;
    if (w != NULL) {
        syn_task_wake_from_isr_or_task(w, boost_prio);
    }
    syn_exit_critical();
}
static SYN_UNUSED inline void syn_signal_clear(syn_signal_t *s) {
    syn_enter_critical();
    s->flag = false;
    syn_exit_critical();
}
static SYN_UNUSED inline bool syn_signal_try_consume(syn_signal_t *s, syn_task_t *task) {
    syn_enter_critical();
    if (s->flag) {
        s->flag = false;
        uint8_t bp = s->boost_prio;
        s->boost_prio = SYN_NO_BOOST;
        if (s->waiter == task) { s->waiter = NULL; }
        syn_exit_critical();
        if (bp < SYN_PRIO_LEVELS) { syn_task_apply_boost(task, bp); }
        return true;
    }
    syn_exit_critical();
    return false;
}
static SYN_UNUSED inline void syn_signal_arm_wait(syn_signal_t *s, syn_task_t *task, bool has_to, uint32_t dl) {
    syn_enter_critical();
    if (s->flag) {
        task->state = SYN_TASK_READY;
        task->sched_flags |= SYN_FLAG_MORE;
        syn_exit_critical();
        return;
    }
    s->waiter = task;
    task->wait_check = syn_signal_is_ready;
    task->wait_obj = s;
    task->has_deadline = has_to;
    task->delay_until = dl;
    task->state = SYN_TASK_BLOCKED;
    syn_exit_critical();
}
static SYN_UNUSED inline void syn_signal_cancel_wait(syn_signal_t *s, const syn_task_t *task) {
    syn_enter_critical();
    if (s->waiter == task) { s->waiter = NULL; }
    syn_exit_critical();
}

"#,
                    );
                }
                MirPrimKind::Watch { elem } => {
                    let ec = self.c_type(elem);
                    let _ = writeln!(
                        self.out,
                        "struct syn_{mangle}_t {{\n    {ec} value;\n    volatile uint32_t seq;\n}};\n\nstatic SYN_UNUSED inline {ec} syn_{mangle}_read(const syn_{mangle}_t *w) {{\n    syn_enter_critical();\n    {ec} v = w->value;\n    syn_exit_critical();\n    return v;\n}}\nstatic SYN_UNUSED inline void syn_{mangle}_publish(syn_{mangle}_t *w, {ec} v) {{\n    syn_enter_critical();\n    w->value = v;\n    w->seq++;\n    syn_exit_critical();\n}}\n"
                    );
                }
                MirPrimKind::Inbox { elem, cap } => {
                    let ec = self.c_type(elem);
                    let n = (*cap).max(1);
                    let res_mangle = MirType::Result {
                        ok: elem.clone(),
                        err: None,
                    }
                    .mangle_name();
                    let _ = writeln!(
                        self.out,
                        "struct syn_{mangle}_t {{\n    {ec} buf[{n}];\n    uint8_t boost_buf[{n}];\n    syn_task_t *rx_waiter;\n    syn_task_t *tx_waiter;\n    uint16_t head;\n    uint16_t tail;\n    volatile uint16_t count;\n}};\n\nstatic SYN_UNUSED inline bool syn_{mangle}_rx_ready(void *obj) {{\n    const syn_{mangle}_t *q = (const syn_{mangle}_t *)obj;\n    return q != NULL && q->count > 0U;\n}}\nstatic SYN_UNUSED inline bool syn_{mangle}_tx_ready(void *obj) {{\n    const syn_{mangle}_t *q = (const syn_{mangle}_t *)obj;\n    return q != NULL && q->count < {n}U;\n}}\nstatic SYN_UNUSED inline bool syn_{mangle}_post(syn_{mangle}_t *q, {ec} item, uint8_t boost_prio) {{\n    syn_enter_critical();\n    if (q->count >= {n}U) {{\n        syn_exit_critical();\n        return false;\n    }}\n    q->buf[q->tail] = item;\n    q->boost_buf[q->tail] = boost_prio;\n    q->tail = (uint16_t)((q->tail + 1U) % {n}U);\n    q->count++;\n    syn_task_t *rw = q->rx_waiter;\n    q->rx_waiter = NULL;\n    if (rw != NULL) {{\n        syn_task_wake_from_isr_or_task(rw, boost_prio);\n    }}\n    syn_exit_critical();\n    return true;\n}}\nstatic SYN_UNUSED inline bool syn_{mangle}_try_pop(syn_{mangle}_t *q, syn_task_t *task, {ec} *out_item) {{\n    syn_enter_critical();\n    if (q->count == 0U) {{\n        syn_exit_critical();\n        return false;\n    }}\n    *out_item = q->buf[q->head];\n    uint8_t bp = q->boost_buf[q->head];\n    q->head = (uint16_t)((q->head + 1U) % {n}U);\n    q->count--;\n    if (q->rx_waiter == task) {{ q->rx_waiter = NULL; }}\n    syn_task_t *tw = q->tx_waiter;\n    q->tx_waiter = NULL;\n    if (tw != NULL) {{\n        syn_task_wake_from_isr_or_task(tw, SYN_NO_BOOST);\n    }}\n    syn_exit_critical();\n    if (task != NULL && bp < SYN_PRIO_LEVELS) {{\n        syn_task_apply_boost(task, bp);\n    }}\n    return true;\n}}\nstatic SYN_UNUSED inline syn_{res_mangle}_t syn_{mangle}_try_recv(syn_{mangle}_t *q) {{\n    {ec} _item;\n    if (syn_{mangle}_try_pop(q, g_syn_sched.running, &_item)) {{\n        return (syn_{res_mangle}_t){{ .is_err = false, .val = _item }};\n    }}\n    return (syn_{res_mangle}_t){{ .is_err = true }};\n}}\nstatic SYN_UNUSED inline void syn_{mangle}_arm_rx(syn_{mangle}_t *q, syn_task_t *task, bool has_to, uint32_t dl) {{\n    syn_enter_critical();\n    if (q->count > 0U) {{\n        task->state = SYN_TASK_READY;\n        task->sched_flags |= SYN_FLAG_MORE;\n        syn_exit_critical();\n        return;\n    }}\n    q->rx_waiter = task;\n    task->wait_check = syn_{mangle}_rx_ready;\n    task->wait_obj = q;\n    task->has_deadline = has_to;\n    task->delay_until = dl;\n    task->state = SYN_TASK_BLOCKED;\n    syn_exit_critical();\n}}\nstatic SYN_UNUSED inline void syn_{mangle}_cancel_rx(syn_{mangle}_t *q, const syn_task_t *task) {{\n    syn_enter_critical();\n    if (q->rx_waiter == task) {{ q->rx_waiter = NULL; }}\n    syn_exit_critical();\n}}\nstatic SYN_UNUSED inline void syn_{mangle}_arm_tx(syn_{mangle}_t *q, syn_task_t *task, bool has_to, uint32_t dl) {{\n    syn_enter_critical();\n    if (q->count < {n}U) {{\n        task->state = SYN_TASK_READY;\n        task->sched_flags |= SYN_FLAG_MORE;\n        syn_exit_critical();\n        return;\n    }}\n    q->tx_waiter = task;\n    task->wait_check = syn_{mangle}_tx_ready;\n    task->wait_obj = q;\n    task->has_deadline = has_to;\n    task->delay_until = dl;\n    task->state = SYN_TASK_BLOCKED;\n    syn_exit_critical();\n}}\nstatic SYN_UNUSED inline void syn_{mangle}_cancel_tx(syn_{mangle}_t *q, const syn_task_t *task) {{\n    syn_enter_critical();\n    if (q->tx_waiter == task) {{ q->tx_waiter = NULL; }}\n    syn_exit_critical();\n}}\n"
                    );
                }
                MirPrimKind::PingPong { elem, cap } => {
                    let ec = self.c_type(elem);
                    let slice_c = self.c_type(&MirType::Slice(elem.clone()));
                    let n = (*cap).max(1);
                    let _ = writeln!(
                        self.out,
                        "struct syn_{mangle}_t {{\n    syn_pp_hdr_t hdr;\n    SYN_DMA_ALIGN {ec} buf[2][{n}];\n}};\n\nstatic SYN_UNUSED inline {slice_c} syn_{mangle}_dma_buf(syn_{mangle}_t *pp) {{\n    return ({slice_c}){{ .ptr = &pp->buf[pp->hdr.write_idx & 1U][0], .len = (size_t){n}U }};\n}}\nstatic SYN_UNUSED inline bool syn_{mangle}_try_take(syn_{mangle}_t *pp, {slice_c} *out_slice) {{\n    syn_enter_critical();\n    if (!pp->hdr.chunk_ready) {{\n        syn_exit_critical();\n        return false;\n    }}\n    uint8_t ridx = pp->hdr.ready_idx;\n    uint16_t rlen = pp->hdr.ready_len;\n    pp->hdr.chunk_ready = false;\n    syn_exit_critical();\n    out_slice->ptr = &pp->buf[ridx][0];\n    out_slice->len = (size_t)rlen;\n    return true;\n}}\nstatic SYN_UNUSED inline void syn_{mangle}_arm_wait(syn_{mangle}_t *pp, syn_task_t *task) {{\n    syn_enter_critical();\n    if (pp->hdr.chunk_ready) {{\n        task->state = SYN_TASK_READY;\n        task->sched_flags |= SYN_FLAG_MORE;\n        syn_exit_critical();\n        return;\n    }}\n    pp->hdr.waiter = task;\n    task->wait_check = syn_pp_hdr_is_ready;\n    task->wait_obj = &pp->hdr;\n    task->has_deadline = false;\n    task->state = SYN_TASK_BLOCKED;\n    syn_exit_critical();\n}}\n"
                    );
                }
                MirPrimKind::Lease { elem, cap } => {
                    let ec = self.c_type(elem);
                    let slice_c = self.c_type(&MirType::Slice(elem.clone()));
                    let n = (*cap).max(1);
                    let _ = writeln!(
                        self.out,
                        "struct syn_{mangle}_t {{\n    SYN_DMA_ALIGN {ec} buf[{n}];\n    syn_task_t *waiter;\n    uint32_t sleep_lock_bit;\n    volatile uint16_t len;\n    volatile bool leased;\n    volatile bool ready;\n}};\n\nstatic SYN_UNUSED inline bool syn_{mangle}_is_ready(void *obj) {{\n    const syn_{mangle}_t *l = (const syn_{mangle}_t *)obj;\n    return l != NULL && l->ready;\n}}\nstatic SYN_UNUSED inline void syn_{mangle}_set_active(syn_{mangle}_t *l, bool active) {{\n    l->leased = active;\n    if (active) {{ syn_sleep_lock_dma(l->sleep_lock_bit); }}\n    else {{ syn_sleep_unlock_dma(l->sleep_lock_bit); }}\n}}\nstatic SYN_UNUSED inline void syn_{mangle}_complete(syn_{mangle}_t *l) {{\n    syn_enter_critical();\n    if (l->len == 0U) {{ l->len = {n}U; }}\n    l->ready = true;\n    syn_task_t *w = l->waiter;\n    l->waiter = NULL;\n    if (w != NULL) {{ syn_task_wake_from_isr_or_task(w, SYN_NO_BOOST); }}\n    syn_exit_critical();\n}}\nstatic SYN_UNUSED inline bool syn_{mangle}_try_take(syn_{mangle}_t *l, {slice_c} *out_slice) {{\n    syn_enter_critical();\n    if (!l->ready) {{ syn_exit_critical(); return false; }}\n    uint16_t rlen = l->len;\n    l->ready = false;\n    l->len = 0U;\n    syn_exit_critical();\n    out_slice->ptr = &l->buf[0];\n    out_slice->len = (size_t)rlen;\n    return true;\n}}\nstatic SYN_UNUSED inline void syn_{mangle}_arm_wait(syn_{mangle}_t *l, syn_task_t *task) {{\n    syn_enter_critical();\n    if (l->ready) {{ task->state = SYN_TASK_READY; task->sched_flags |= SYN_FLAG_MORE; syn_exit_critical(); return; }}\n    l->waiter = task;\n    task->wait_check = syn_{mangle}_is_ready;\n    task->wait_obj = l;\n    task->has_deadline = false;\n    task->state = SYN_TASK_BLOCKED;\n    syn_exit_critical();\n}}\n"
                    );
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 5: Constants, Static Primitive Instances, & `extern "C"` Decls
    // ─────────────────────────────────────────────────────────────────────────

    fn emit_globals_and_externs(&mut self) {
        if !self.m.mmio_instances.is_empty() {
            self.out.push_str("/* ── Typed MMIO Peripheral Addresses / Host Simulation Buffers ────────── */\n\n");
            for inst in &self.m.mmio_instances {
                let sym = sanitize_ident(&inst.name);
                let macro_name = mmio_c_macro_name(&inst.name);
                let hex_lit = if inst.base_addr > u32::MAX as u64 {
                    format!("0x{:X}ULL", inst.base_addr)
                } else {
                    format!("0x{:X}UL", inst.base_addr)
                };
                let buf_bytes = inst.byte_size.max(4);
                let _ = writeln!(self.out, "#if defined(SYN_FREESTANDING)");
                let _ = writeln!(
                    self.out,
                    "#define {macro_name} ((volatile uint8_t *)(uintptr_t){hex_lit})"
                );
                let _ = writeln!(self.out, "#else");
                let _ = writeln!(
                    self.out,
                    "static SYN_UNUSED volatile uint8_t __zev_mmio_buf_{sym}[{buf_bytes}] __attribute__((aligned(4)));"
                );
                let _ = writeln!(self.out, "#define {macro_name} (&__zev_mmio_buf_{sym}[0])");
                let _ = writeln!(self.out, "#endif\n");
            }
        }

        if !self.m.consts.is_empty() {
            self.out.push_str("/* ── Immutable Global Constants ─────────────────────────────────────────── */\n\n");
            for c in &self.m.consts {
                let ct = self.c_type(&c.ty);
                let val = self.fmt_const_init_expr(&c.init);
                // `#undef` first if it collided with a default hardware bit macro like `MOTOR_EN`
                if matches!(c.name.as_str(), "MOTOR_EN" | "HALF_TRANSFER" | "FULL_TRANSFER") {
                    let _ = writeln!(self.out, "#undef {}", c.name);
                }
                let _ = writeln!(self.out, "static SYN_UNUSED const {ct} {} = {val};", c_ident(&c.name));
            }
            self.out.push('\n');
        }

        if !self.m.primitives.is_empty() {
            self.out.push_str("/* ── Static OS & DMA Endpoint Instances ─────────────────────────────────── */\n\n");
            for p in &self.m.primitives {
                let dma_attr = if p.dma_safe { "SYN_DMA_ALIGN " } else { "" };
                let pt = format!("syn_{}_t", p.kind.mangle_name());
                let pname = c_ident(&p.name);
                match &p.kind {
                    MirPrimKind::Watch { elem } => {
                        let init_str = match &p.init {
                            Some(ie) => self.fmt_const_init_expr(ie),
                            None => self.zero_init_expr(elem),
                        };
                        let _ = writeln!(
                            self.out,
                            "static SYN_UNUSED {dma_attr}{pt} {pname} = {{ .value = {init_str}, .seq = 0U }};"
                        );
                    }
                    MirPrimKind::PingPong { elem, cap } => {
                        let bit = p.sleep_lock_bit.unwrap_or(0);
                        let ec = self.c_type(elem);
                        let _ = writeln!(
                            self.out,
                            "static SYN_UNUSED {dma_attr}{pt} {pname} = {{ .hdr = {{ .buf0 = {pname}.buf[0], .buf1 = {pname}.buf[1], .waiter = NULL, .sleep_lock_bit = (1U << {bit}U), .cap = {cap}U, .elem_size = (uint16_t)sizeof({ec}), .pending_write_len = 0U, .ready_len = 0U, .write_idx = 0U, .ready_idx = 0U, .chunk_ready = false, .dma_active = false }} }};"
                        );
                    }
                    MirPrimKind::Lease { .. } => {
                        let bit = p.sleep_lock_bit.unwrap_or(0);
                        let _ = writeln!(
                            self.out,
                            "static SYN_UNUSED {dma_attr}{pt} {pname} = {{ .waiter = NULL, .sleep_lock_bit = (1U << {bit}U), .len = 0U, .leased = false, .ready = false }};"
                        );
                    }
                    MirPrimKind::Signal => {
                        let _ = writeln!(
                            self.out,
                            "static SYN_UNUSED {dma_attr}{pt} {pname} = {{ .waiter = NULL, .flag = false, .boost_prio = SYN_NO_BOOST }};"
                        );
                    }
                    MirPrimKind::Inbox { .. } => {
                        let _ = writeln!(
                            self.out,
                            "static SYN_UNUSED {dma_attr}{pt} {pname} = {{ .rx_waiter = NULL, .tx_waiter = NULL, .head = 0U, .tail = 0U, .count = 0U }};"
                        );
                    }
                }
            }
            self.out.push('\n');
        }

        let has_irq_state = self.m.irq_vectors.iter().any(|v| {
            v.selector_symbol.is_some() || v.bindings.iter().any(|b| !b.state_vars.is_empty())
        });
        if has_irq_state {
            self.out.push_str("/* ── Per-Attachment ISR Static State & Boot IRQ Selectors ────────────── */\n\n");
            for vec in &self.m.irq_vectors {
                if let Some(sel_sym) = &vec.selector_symbol {
                    let _ = writeln!(
                        self.out,
                        "static SYN_UNUSED uint8_t {sel_sym} = 0U;"
                    );
                }
                for b in &vec.bindings {
                    for sv in &b.state_vars {
                        let decl = self.c_decl(&sv.ty, &sv.storage_symbol);
                        let init_s = self.fmt_const_init_expr(&sv.init);
                        let _ = writeln!(self.out, "static SYN_UNUSED {decl} = {init_s};");
                    }
                }
            }
            self.out.push('\n');
        }

        // Extern "C" declarations (skipping symbols already provided by C99 headers or sim runtime)
        let mut any_ext = false;
        for ef in &self.m.extern_fns {
            if self.builtin_c_syms.contains(&ef.name) {
                continue;
            }
            if !any_ext {
                self.out.push_str("/* ── External C Prototypes ──────────────────────────────────────────────── */\n\n");
                any_ext = true;
            }
            let rt = self.c_type(&ef.ret_ty);
            let mut params: Vec<String> = ef
                .params
                .iter()
                .map(|(pn, pt)| self.c_decl(pt, &c_ident(pn)))
                .collect();
            if ef.variadic {
                params.push("...".to_string());
            } else if params.is_empty() {
                params.push("void".to_string());
            }
            let _ = writeln!(self.out, "extern {rt} {}({});", ef.name, params.join(", "));
        }
        if any_ext {
            self.out.push('\n');
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 6: Static Coroutine Frame Structs & Task Table
    // ─────────────────────────────────────────────────────────────────────────

    fn emit_frames_and_task_table(&mut self) {
        self.out
            .push_str("/* ── Packed Coroutine Frames (Union-Aliased Disjoint Child Calls) ───────── */\n\n");

        // Topologically sort yielding functions so callees precede callers!
        let ordered_yielding = self.toposort_yielding_fns();
        for fname in &ordered_yielding {
            let func = self
                .m
                .functions
                .iter()
                .find(|f| f.name == *fname)
                .unwrap();
            if let MirFnBody::Coroutine { child_slots, .. } = &func.body {
                let env_ret = func.envelope_return_type();
                let ret_opt = if env_ret.is_void() {
                    None
                } else {
                    Some(&env_ret)
                };
                self.emit_single_frame_struct(
                    &format!("syn_frame_fn_{}", func.name),
                    &func.vars,
                    child_slots,
                    ret_opt,
                );
            }
        }

        // Task blueprint frame structs
        for bp in &self.m.task_blueprints {
            self.emit_single_frame_struct(
                &format!("syn_frame_task_{}", bp.name),
                &bp.vars,
                &bp.child_slots,
                None,
            );
        }

        // Static frame instances for sequential boot yielding calls (if any)
        for callee in &self.m.boot.boot_yielding_callees {
            let _ = writeln!(
                self.out,
                "static syn_frame_fn_{callee}_t g_syn_boot_frame_{callee} = {{ ._pc = 0U }};"
            );
        }

        // Static frame instances per `spawn`ed task slot + TCB array
        if !self.m.spawned_tasks.is_empty() {
            for sp in &self.m.spawned_tasks {
                let _ = writeln!(
                    self.out,
                    "static syn_frame_task_{}_t g_syn_frame_slot_{} = {{ ._pc = 0U }};",
                    sp.blueprint_name, sp.slot_id
                );
            }
            let _ = writeln!(
                self.out,
                "static syn_task_t g_syn_tasks[{}];\n",
                self.m.spawned_tasks.len()
            );
        }
    }

    fn emit_single_frame_struct(
        &mut self,
        struct_name: &str,
        vars: &[MirVar],
        child_slots: &[MirChildCallSlot],
        ret_ty: Option<&MirType>,
    ) {
        let _ = writeln!(self.out, "typedef struct {struct_name} {{");
        self.out.push_str("    uint16_t _pc;\n");
        for v in vars {
            if v.hoisted && !v.ty.is_void() {
                let vc = self.c_type(&v.ty);
                let _ = writeln!(self.out, "    {vc} {};", frame_var_name(v));
            }
        }
        if let Some(rt) = ret_ty {
            if !rt.is_void() {
                let rc = self.c_type(rt);
                let _ = writeln!(self.out, "    {rc} _ret;");
            }
        }
        if !child_slots.is_empty() {
            let mut groups: BTreeMap<usize, Vec<&MirChildCallSlot>> = BTreeMap::new();
            for s in child_slots {
                groups.entry(s.union_group).or_default().push(s);
            }
            for (g, slots) in groups {
                let _ = writeln!(self.out, "    union {{");
                for s in slots {
                    let _ = writeln!(
                        self.out,
                        "        syn_frame_fn_{}_t {};",
                        s.callee, s.slot_name
                    );
                }
                let _ = writeln!(self.out, "    }} _child_u{g};");
            }
        }
        let _ = writeln!(self.out, "}} {struct_name}_t;\n");
    }

    fn toposort_yielding_fns(&self) -> Vec<String> {
        let mut visited = BTreeSet::new();
        let mut order = Vec::new();
        for f in &self.m.functions {
            if f.is_yielding {
                self.dfs_yielding_fn(&f.name, &mut visited, &mut order);
            }
        }
        order
    }

    fn dfs_yielding_fn(
        &self,
        name: &str,
        visited: &mut BTreeSet<String>,
        order: &mut Vec<String>,
    ) {
        if !visited.insert(name.to_string()) {
            return;
        }
        if let Some(func) = self.m.functions.iter().find(|f| f.name == name) {
            if let MirFnBody::Coroutine { child_slots, .. } = &func.body {
                for s in child_slots {
                    self.dfs_yielding_fn(&s.callee, visited, order);
                }
            }
            order.push(name.to_string());
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 7 & 8: Prototypes & Hardware ISRs
    // ─────────────────────────────────────────────────────────────────────────

    fn emit_fn_prototypes(&mut self) {
        self.out
            .push_str("static SYN_UNUSED inline void syn_sim_pend_irq(uint32_t irq);\n");
        for f in &self.m.functions {
            if f.is_yielding {
                let _ = writeln!(
                    self.out,
                    "static SYN_UNUSED syn_step_t syn_coro_{0}(syn_task_t *_task, syn_frame_fn_{0}_t *_f);",
                    f.name
                );
            } else {
                let sig = self.direct_fn_signature(f);
                let _ = writeln!(self.out, "static SYN_UNUSED {sig};");
            }
        }
        for bp in &self.m.task_blueprints {
            let _ = writeln!(
                self.out,
                "static SYN_UNUSED syn_step_t syn_task_step_{}(syn_task_t *_task, void *_raw_f);",
                bp.name
            );
        }
        self.out.push('\n');
    }

    fn direct_fn_signature(&self, f: &MirFunction) -> String {
        let env_ret = f.envelope_return_type();
        let rt = self.c_type(&env_ret);
        let params = if f.params.is_empty() {
            "void".to_string()
        } else {
            f.params
                .iter()
                .map(|&vid| {
                    let v = &f.vars[vid.0];
                    format!("{} {}", self.c_type(&v.ty), stack_var_name(v))
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!("{rt} syn_fn_{}({params})", f.name)
    }

    fn isr_signature(&self, isr: &MirIsr) -> String {
        let mut params: Vec<String> = isr
            .params
            .iter()
            .map(|&vid| {
                let v = &isr.vars[vid.0];
                format!("{} {}", self.c_type(&v.ty), stack_var_name(v))
            })
            .collect();
        for sv in &isr.state_vars {
            params.push(format!("{} *__state_{}", self.c_type(&sv.ty), sv.name));
        }
        if params.is_empty() {
            params.push("void".to_string());
        }
        format!(
            "void syn_isr_{}({})",
            sanitize_ident(&isr.name),
            params.join(", ")
        )
    }

    fn fmt_irq_binding_call(&self, b: &zev_mir::MirIrqBinding) -> String {
        let mut call_args: Vec<String> = b
            .baked_args
            .iter()
            .map(|ba| match ba {
                MirBakedArg::PrimAddr(p) => format!("(&{})", c_ident(p)),
                MirBakedArg::MmioAddr(m) => mmio_c_macro_name(m),
                MirBakedArg::Const(e) => self.fmt_const_init_expr(e),
            })
            .collect();
        for sv in &b.state_vars {
            call_args.push(format!("&{}", sv.storage_symbol));
        }
        format!(
            "syn_isr_{}({});",
            sanitize_ident(&b.isr_name),
            call_args.join(", ")
        )
    }

    fn emit_isrs(&mut self) {
        if !self.m.isrs.is_empty() {
            self.out
                .push_str("/* ── Hardware Interrupt Service Routines & Sim Trigger Hooks ────────────── */\n\n");
            for isr in &self.m.isrs {
                let sig = self.isr_signature(isr);
                let _ = writeln!(self.out, "static SYN_UNUSED {sig} {{");
                for &vid in &isr.params {
                    let _ = writeln!(
                        self.out,
                        "    (void){};",
                        stack_var_name(&isr.vars[vid.0])
                    );
                }
                for sv in &isr.state_vars {
                    let _ = writeln!(self.out, "    (void)__state_{};", sv.name);
                }
                self.emit_stack_local_decls(&isr.vars, true);
                for s in &isr.body {
                    self.emit_direct_stmt(
                        s,
                        &isr.vars,
                        &MirType::Void,
                        &MirErrorSpec::None,
                        1,
                        false,
                    );
                }
                self.out.push_str("}\n\n");

                if isr.params.is_empty() && isr.state_vars.is_empty() {
                    let sname = sanitize_ident(&isr.name);
                    let _ = writeln!(
                        self.out,
                        "void syn_sim_fire_isr_{0}(void) {{\n    if (g_syn_sim_trace) {{\n#if !defined(SYN_FREESTANDING)\n        printf(\"[t=%ums] ISR {0}\\n\", (unsigned)syn_now_ms());\n#endif\n    }}\n    syn_isr_{0}();\n}}\n",
                        sname
                    );
                }
            }
        }

        if !self.m.irq_vectors.is_empty() {
            self.out
                .push_str("/* ── Synthesized Vector Dispatch Wrappers (__zev_irq_vec_<N>) ──────────── */\n\n");
            for vec in &self.m.irq_vectors {
                let _ = writeln!(
                    self.out,
                    "static SYN_UNUSED void __zev_irq_vec_{}(void) {{",
                    vec.irq_num
                );
                if let Some(sel_sym) = &vec.selector_symbol {
                    if !vec.is_shared {
                        let _ = writeln!(self.out, "    switch ({sel_sym}) {{");
                        for b in &vec.bindings {
                            if let Some((_, val)) = &b.cond_selector {
                                let call_s = self.fmt_irq_binding_call(b);
                                let _ = writeln!(self.out, "        case {val}U: {call_s} break;");
                            }
                        }
                        let _ = writeln!(self.out, "        default: break;");
                        let _ = writeln!(self.out, "    }}");
                    } else {
                        for b in &vec.bindings {
                            let call_s = self.fmt_irq_binding_call(b);
                            if let Some((_, val)) = &b.cond_selector {
                                let _ =
                                    writeln!(self.out, "    if ({sel_sym} == {val}U) {{ {call_s} }}");
                            } else {
                                let _ = writeln!(self.out, "    {call_s}");
                            }
                        }
                    }
                } else {
                    for b in &vec.bindings {
                        let call_s = self.fmt_irq_binding_call(b);
                        let _ = writeln!(self.out, "    {call_s}");
                    }
                }
                self.out.push_str("}\n\n");
            }
        }

        // Synthesize `syn_sim_pend_irq(uint32_t irq)` mapping NVIC vector IDs to user ISRs / vectors
        self.out
            .push_str("static SYN_UNUSED inline void syn_sim_pend_irq(uint32_t irq) {\n");
        let legacy_isrs: Vec<&MirIsr> = {
            let mut attached: BTreeSet<&str> = BTreeSet::new();
            for vec in &self.m.irq_vectors {
                for b in &vec.bindings {
                    attached.insert(b.isr_name.as_str());
                }
            }
            self.m
                .isrs
                .iter()
                .filter(|isr| {
                    !attached.contains(isr.name.as_str())
                        && isr.params.is_empty()
                        && isr.state_vars.is_empty()
                })
                .collect()
        };
        if self.m.irq_vectors.is_empty() && legacy_isrs.is_empty() {
            self.out.push_str("    (void)irq;\n");
        } else {
            self.out.push_str("    switch (irq) {\n");
            let mut used_irqs = BTreeSet::new();
            for vec in &self.m.irq_vectors {
                if used_irqs.insert(vec.irq_num) {
                    let _ = writeln!(
                        self.out,
                        "        case {}U: __zev_irq_vec_{}(); break;",
                        vec.irq_num, vec.irq_num
                    );
                }
            }
            for (idx, isr) in legacy_isrs.iter().enumerate() {
                let irq_num: u32 = match isr.name.as_str() {
                    "EXTI0" | "GPIOA" => 0,
                    "EXTI1" | "GPIOB" => 1,
                    "EXTI2" | "GPIOC" => 2,
                    "EXTI3" => 3,
                    "EXTI4" => 4,
                    "DMA1_CH5" | "UART0" | "USART1" => 5,
                    "DMA1_CH6" | "UART1" | "USART2" => 6,
                    _ => (16 + idx) as u32,
                };
                if used_irqs.insert(irq_num) {
                    let _ = writeln!(
                        self.out,
                        "        case {irq_num}U: syn_sim_fire_isr_{}(); break;",
                        sanitize_ident(&isr.name)
                    );
                }
            }
            self.out.push_str("        default: break;\n    }\n");
        }
        self.out.push_str("}\n\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 9: User Functions & Coroutine Task Step Implementations
    // ─────────────────────────────────────────────────────────────────────────

    fn emit_functions_and_tasks(&mut self) {
        self.out
            .push_str("/* ── Functions & Stackless Coroutine State Machines ─────────────────────── */\n\n");

        for f in &self.m.functions {
            match &f.body {
                MirFnBody::Direct(stmts) => {
                    let sig = self.direct_fn_signature(f);
                    let _ = writeln!(self.out, "static SYN_UNUSED {sig} {{");
                    for &pid in &f.params {
                        let _ = writeln!(self.out, "    (void){};", stack_var_name(&f.vars[pid.0]));
                    }
                    self.emit_stack_local_decls(&f.vars, true);
                    for s in stmts {
                        self.emit_direct_stmt(s, &f.vars, &f.ret_ty, &f.error_spec, 1, false);
                    }
                    let env_ret = f.envelope_return_type();
                    if !env_ret.is_void() {
                        let _ = writeln!(self.out, "    return {};", self.zero_init_expr(&env_ret));
                    }
                    self.out.push_str("}\n\n");
                }
                MirFnBody::Coroutine {
                    blocks,
                    child_slots,
                } => {
                    let _ = writeln!(
                        self.out,
                        "static SYN_UNUSED syn_step_t syn_coro_{0}(syn_task_t *_task, syn_frame_fn_{0}_t *_f) {{",
                        f.name
                    );
                    self.emit_coroutine_body(
                        blocks,
                        &f.vars,
                        child_slots,
                        &f.ret_ty,
                        &f.error_spec,
                        true,
                    );
                    self.out.push_str("}\n\n");
                }
            }
        }

        for bp in &self.m.task_blueprints {
            let _ = writeln!(
                self.out,
                "static SYN_UNUSED syn_step_t syn_task_step_{0}(syn_task_t *_task, void *_raw_f) {{\n    syn_frame_task_{0}_t *_f = (syn_frame_task_{0}_t *)_raw_f;",
                bp.name
            );
            self.emit_coroutine_body(
                &bp.blocks,
                &bp.vars,
                &bp.child_slots,
                &MirType::Void,
                &MirErrorSpec::None,
                false,
            );
            self.out.push_str("}\n\n");
        }
    }

    fn emit_stack_local_decls(&mut self, vars: &[MirVar], skip_params: bool) {
        for v in vars {
            if v.hoisted || v.ty.is_void() {
                continue;
            }
            if v.role == VarRole::IsrState {
                continue;
            }
            if skip_params && v.role == VarRole::Param {
                continue;
            }
            let vc = self.c_type(&v.ty);
            let vn = stack_var_name(v);
            let zi = self.zero_init_expr(&v.ty);
            let _ = writeln!(self.out, "    {vc} {vn} = {zi};");
            let _ = writeln!(self.out, "    (void){vn};");
        }
    }

    fn emit_coroutine_body(
        &mut self,
        blocks: &[MirBlock],
        vars: &[MirVar],
        child_slots: &[MirChildCallSlot],
        ret_ty: &MirType,
        error_spec: &MirErrorSpec,
        has_ret_slot: bool,
    ) {
        self.out.push_str("    (void)_task;\n    (void)_f;\n");
        self.emit_stack_local_decls(vars, true);

        let reachable = compute_reachable_blocks(blocks);
        let slot_group_map: BTreeMap<&str, usize> = child_slots
            .iter()
            .map(|s| (s.slot_name.as_str(), s.union_group))
            .collect();

        // Top-level state machine dispatch switch
        self.out.push_str("    switch (_f->_pc) {\n");
        for blk in blocks {
            if reachable.contains(&blk.id) && (blk.id.0 == 0 || blk.is_resume_entry) {
                let _ = writeln!(self.out, "        case {}U: goto _bb{};", blk.id.0, blk.id.0);
            }
        }
        self.out
            .push_str("        default: return SYN_STEP_DONE;\n    }\n");

        let env_ret = if error_spec.has_error() {
            MirType::Result {
                ok: Box::new(ret_ty.clone()),
                err: error_spec.err_type().map(Box::new),
            }
        } else {
            ret_ty.clone()
        };

        let mut every_resume_map: BTreeMap<BlockId, (VarId, VarId)> = BTreeMap::new();
        for blk in blocks {
            if reachable.contains(&blk.id) {
                if let MirTerminator::Yield(MirYieldKind::EveryLatch {
                    next_deadline_var,
                    period_var,
                    resume_bb,
                }) = &blk.terminator
                {
                    every_resume_map.insert(*resume_bb, (*next_deadline_var, *period_var));
                }
            }
        }

        for blk in blocks {
            if !reachable.contains(&blk.id) {
                continue;
            }
            let _ = writeln!(self.out, "\n_bb{}: ;", blk.id.0);
            if let Some(&(dv_id, pv_id)) = every_resume_map.get(&blk.id) {
                let dv = self.var_ref(dv_id, vars);
                let pv = self.var_ref(pv_id, vars);
                let _ = writeln!(
                    self.out,
                    "    {{\n        uint32_t _eff_p = ({pv} == 0U) ? 1U : (uint32_t){pv};\n        int32_t _wdiff = (int32_t)(syn_now_ms() - {dv});\n        if (_wdiff >= (int32_t)_eff_p) {{\n            {dv} += ((uint32_t)_wdiff / _eff_p) * _eff_p;\n        }}\n    }}"
                );
            }
            for s in &blk.stmts {
                self.emit_direct_stmt(s, vars, ret_ty, error_spec, 1, true);
            }
            self.emit_terminator(
                &blk.terminator,
                vars,
                &slot_group_map,
                ret_ty,
                error_spec,
                &env_ret,
                has_ret_slot,
            );
        }
    }

    fn emit_terminator(
        &mut self,
        term: &MirTerminator,
        vars: &[MirVar],
        slot_groups: &BTreeMap<&str, usize>,
        ret_ty: &MirType,
        error_spec: &MirErrorSpec,
        env_ret: &MirType,
        has_ret_slot: bool,
    ) {
        match term {
            MirTerminator::Goto(target) => {
                let _ = writeln!(self.out, "    goto _bb{};", target.0);
            }
            MirTerminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                let c_raw = self.fmt_expr(cond, vars);
                let c = strip_outer_parens(&c_raw);
                let _ = writeln!(
                    self.out,
                    "    if ({c}) {{ goto _bb{}; }} else {{ goto _bb{}; }}",
                    then_bb.0, else_bb.0
                );
            }
            MirTerminator::Switch {
                discr,
                arms,
                default_bb,
            } => {
                let d = self.fmt_expr(discr, vars);
                let _ = writeln!(self.out, "    switch ({d}) {{");
                for (pat, target) in arms {
                    let pstr = self.fmt_switch_pat(pat);
                    let _ = writeln!(self.out, "        case {pstr}: goto _bb{};", target.0);
                }
                let _ = writeln!(self.out, "        default: goto _bb{};\n    }}", default_bb.0);
            }
            MirTerminator::Return(expr_opt) => {
                if has_ret_slot && !env_ret.is_void() {
                    if error_spec.has_error() {
                        let rc = self.c_type(env_ret);
                        if let Some(e) = expr_opt {
                            let v = self.fmt_expr(e, vars);
                            let _ = writeln!(
                                self.out,
                                "    _f->_ret = ({rc}){{ .is_err = false, .val = {v} }};"
                            );
                        } else {
                            let _ = writeln!(self.out, "    _f->_ret = ({rc}){{ .is_err = false }};");
                        }
                    } else if let Some(e) = expr_opt {
                        let v = self.fmt_expr(e, vars);
                        let _ = writeln!(self.out, "    _f->_ret = {v};");
                    }
                }
                self.out
                    .push_str("    _f->_pc = 0xFFFFU;\n    return SYN_STEP_DONE;\n");
            }
            MirTerminator::Error { reason, partial } => {
                if has_ret_slot && !env_ret.is_void() {
                    let rc = self.c_type(env_ret);
                    let mut fields = vec![".is_err = true".to_string()];
                    if let Some(r) = reason {
                        fields.push(format!(".err = {}", self.fmt_expr(r, vars)));
                    }
                    if let Some(p) = partial {
                        if !ret_ty.is_void() {
                            fields.push(format!(".val = {}", self.fmt_expr(p, vars)));
                        }
                    }
                    let _ = writeln!(
                        self.out,
                        "    _f->_ret = ({rc}){{ {} }};",
                        fields.join(", ")
                    );
                }
                self.out
                    .push_str("    _f->_pc = 0xFFFFU;\n    return SYN_STEP_DONE;\n");
            }
            MirTerminator::Yield(yk) => self.emit_yield(yk, vars, slot_groups),
        }
    }

    fn emit_yield(
        &mut self,
        yk: &MirYieldKind,
        vars: &[MirVar],
        slot_groups: &BTreeMap<&str, usize>,
    ) {
        match yk {
            MirYieldKind::EveryLatch {
                next_deadline_var,
                period_var,
                resume_bb,
            } => {
                let dv = self.var_ref(*next_deadline_var, vars);
                let pv = self.var_ref(*period_var, vars);
                let r = resume_bb.0;
                let _ = writeln!(
                    self.out,
                    "    {{\n        uint32_t _eff_p = ({pv} == 0U) ? 1U : (uint32_t){pv};\n        {dv} += _eff_p;\n        int32_t _diff = (int32_t)(syn_now_ms() - {dv});\n        if (_diff < 0) {{\n            syn_task_sleep_until(_task, {dv});\n            _f->_pc = {r}U;\n            return SYN_STEP_IDLE;\n        }}\n        uint32_t _late = (uint32_t)_diff;\n        {dv} += (_late / _eff_p) * _eff_p;\n        if (syn_task_should_share(_task)) {{\n            _f->_pc = {r}U;\n            return SYN_STEP_MORE;\n        }}\n        goto _bb{r};\n    }}"
                );
            }
            MirYieldKind::DelayPoll {
                deadline_var,
                poll_bb,
                resume_bb,
            } => {
                let dv = self.var_ref(*deadline_var, vars);
                let p = poll_bb.0;
                let r = resume_bb.0;
                let _ = writeln!(
                    self.out,
                    "    if ((int32_t)(syn_now_ms() - {dv}) < 0) {{\n        syn_task_sleep_until(_task, {dv});\n        _f->_pc = {p}U;\n        return SYN_STEP_IDLE;\n    }}\n    syn_task_clear_wait(_task);\n    goto _bb{r};"
                );
            }
            MirYieldKind::ShareCheckpoint { resume_bb } => {
                let r = resume_bb.0;
                let _ = writeln!(
                    self.out,
                    "    if (syn_task_should_share(_task)) {{\n        _f->_pc = {r}U;\n        return SYN_STEP_MORE;\n    }}\n    goto _bb{r};"
                );
            }
            MirYieldKind::InboxRecv {
                target,
                elem_ty,
                cap,
                timeout_deadline_var,
                dst_val,
                dst_res,
                poll_bb,
                resume_bb,
            } => {
                let mangle = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                }
                .mangle_name();
                let ptr = self.fmt_prim_ptr(target, vars);
                let ec = self.c_type(elem_ty);
                let p = poll_bb.0;
                let r = resume_bb.0;

                self.out.push_str("    {\n");
                let _ = writeln!(self.out, "        {ec} _rx_item;");
                let _ = writeln!(
                    self.out,
                    "        if (syn_{mangle}_try_pop({ptr}, _task, &_rx_item)) {{"
                );
                self.out.push_str("            syn_task_clear_wait(_task);\n");
                if let Some(dv) = dst_val {
                    let _ = writeln!(self.out, "            {} = _rx_item;", self.var_ref(*dv, vars));
                }
                if let Some(dr) = dst_res {
                    let rc = self.c_type(&vars[dr.0].ty);
                    let _ = writeln!(
                        self.out,
                        "            {} = ({rc}){{ .is_err = false, .val = _rx_item }};",
                        self.var_ref(*dr, vars)
                    );
                }
                let _ = writeln!(self.out, "            goto _bb{r};\n        }}");

                if let Some(tv) = timeout_deadline_var {
                    let dl = self.var_ref(*tv, vars);
                    let _ = writeln!(
                        self.out,
                        "        if ((int32_t)(syn_now_ms() - {dl}) >= 0) {{\n            syn_{mangle}_cancel_rx({ptr}, _task);\n            syn_task_clear_wait(_task);"
                    );
                    if let Some(dr) = dst_res {
                        let rc = self.c_type(&vars[dr.0].ty);
                        let _ = writeln!(
                            self.out,
                            "            {} = ({rc}){{ .is_err = true }};",
                            self.var_ref(*dr, vars)
                        );
                    }
                    let _ = writeln!(
                        self.out,
                        "            goto _bb{r};\n        }}\n        syn_{mangle}_arm_rx({ptr}, _task, true, {dl});"
                    );
                } else {
                    let _ = writeln!(
                        self.out,
                        "        syn_{mangle}_arm_rx({ptr}, _task, false, 0U);"
                    );
                }
                let _ = writeln!(
                    self.out,
                    "        _f->_pc = {p}U;\n        return SYN_STEP_IDLE;\n    }}"
                );
            }
            MirYieldKind::InboxSend {
                target,
                elem_ty,
                cap,
                val_var,
                timeout_deadline_var,
                dst_res,
                poll_bb,
                resume_bb,
            } => {
                let mangle = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                }
                .mangle_name();
                let ptr = self.fmt_prim_ptr(target, vars);
                let vv = self.var_ref(*val_var, vars);
                let p = poll_bb.0;
                let r = resume_bb.0;
                let _ = writeln!(
                    self.out,
                    "    if (syn_{mangle}_post({ptr}, {vv}, SYN_NO_BOOST)) {{\n        syn_task_clear_wait(_task);"
                );
                if let Some(dr) = dst_res {
                    let rc = self.c_type(&vars[dr.0].ty);
                    let _ = writeln!(
                        self.out,
                        "        {} = ({rc}){{ .is_err = false }};",
                        self.var_ref(*dr, vars)
                    );
                }
                let _ = writeln!(self.out, "        goto _bb{r};\n    }}");
                if let Some(tv) = timeout_deadline_var {
                    let dl = self.var_ref(*tv, vars);
                    let _ = writeln!(
                        self.out,
                        "    if ((int32_t)(syn_now_ms() - {dl}) >= 0) {{\n        syn_{mangle}_cancel_tx({ptr}, _task);\n        syn_task_clear_wait(_task);"
                    );
                    if let Some(dr) = dst_res {
                        let rc = self.c_type(&vars[dr.0].ty);
                        let _ = writeln!(
                            self.out,
                            "        {} = ({rc}){{ .is_err = true }};",
                            self.var_ref(*dr, vars)
                        );
                    }
                    let _ = writeln!(
                        self.out,
                        "        goto _bb{r};\n    }}\n    syn_{mangle}_arm_tx({ptr}, _task, true, {dl});"
                    );
                } else {
                    let _ = writeln!(self.out, "    syn_{mangle}_arm_tx({ptr}, _task, false, 0U);");
                }
                let _ = writeln!(self.out, "    _f->_pc = {p}U;\n    return SYN_STEP_IDLE;");
            }
            MirYieldKind::SignalWait {
                target,
                timeout_deadline_var,
                dst_res,
                poll_bb,
                resume_bb,
            } => {
                let ptr = self.fmt_prim_ptr(target, vars);
                let p = poll_bb.0;
                let r = resume_bb.0;
                let _ = writeln!(
                    self.out,
                    "    if (syn_signal_try_consume({ptr}, _task)) {{\n        syn_task_clear_wait(_task);"
                );
                if let Some(dr) = dst_res {
                    let rc = self.c_type(&vars[dr.0].ty);
                    let _ = writeln!(
                        self.out,
                        "        {} = ({rc}){{ .is_err = false }};",
                        self.var_ref(*dr, vars)
                    );
                }
                let _ = writeln!(self.out, "        goto _bb{r};\n    }}");
                if let Some(tv) = timeout_deadline_var {
                    let dl = self.var_ref(*tv, vars);
                    let _ = writeln!(
                        self.out,
                        "    if ((int32_t)(syn_now_ms() - {dl}) >= 0) {{\n        syn_signal_cancel_wait({ptr}, _task);\n        syn_task_clear_wait(_task);"
                    );
                    if let Some(dr) = dst_res {
                        let rc = self.c_type(&vars[dr.0].ty);
                        let _ = writeln!(
                            self.out,
                            "        {} = ({rc}){{ .is_err = true }};",
                            self.var_ref(*dr, vars)
                        );
                    }
                    let _ = writeln!(
                        self.out,
                        "        goto _bb{r};\n    }}\n    syn_signal_arm_wait({ptr}, _task, true, {dl});"
                    );
                } else {
                    let _ = writeln!(self.out, "    syn_signal_arm_wait({ptr}, _task, false, 0U);");
                }
                let _ = writeln!(self.out, "    _f->_pc = {p}U;\n    return SYN_STEP_IDLE;");
            }
            MirYieldKind::PingPongWaitChunk {
                target,
                elem_ty,
                cap,
                dst_slice,
                poll_bb,
                resume_bb,
            } => {
                let mangle = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                }
                .mangle_name();
                let ptr = self.fmt_prim_ptr(target, vars);
                let sc = self.c_type(&MirType::Slice(Box::new(elem_ty.clone())));
                let dv = self.var_ref(*dst_slice, vars);
                let p = poll_bb.0;
                let r = resume_bb.0;
                let _ = writeln!(
                    self.out,
                    "    {{\n        {sc} _chunk;\n        if (syn_{mangle}_try_take({ptr}, &_chunk)) {{\n            syn_task_clear_wait(_task);\n            {dv} = _chunk;\n            goto _bb{r};\n        }}\n        syn_{mangle}_arm_wait({ptr}, _task);\n        _f->_pc = {p}U;\n        return SYN_STEP_IDLE;\n    }}"
                );
            }
            MirYieldKind::LeaseWaitChunk {
                target,
                elem_ty,
                cap,
                dst_slice,
                poll_bb,
                resume_bb,
            } => {
                let mangle = MirPrimKind::Lease {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                }
                .mangle_name();
                let ptr = self.fmt_prim_ptr(target, vars);
                let sc = self.c_type(&MirType::Slice(Box::new(elem_ty.clone())));
                let dv = self.var_ref(*dst_slice, vars);
                let p = poll_bb.0;
                let r = resume_bb.0;
                let _ = writeln!(
                    self.out,
                    "    {{\n        {sc} _chunk;\n        if (syn_{mangle}_try_take({ptr}, &_chunk)) {{\n            syn_task_clear_wait(_task);\n            {dv} = _chunk;\n            goto _bb{r};\n        }}\n        syn_{mangle}_arm_wait({ptr}, _task);\n        _f->_pc = {p}U;\n        return SYN_STEP_IDLE;\n    }}"
                );
            }
            MirYieldKind::CallYielding {
                callee,
                slot_name,
                args,
                poll_bb,
                ..
            } => {
                let g = slot_groups.get(slot_name.as_str()).copied().unwrap_or(0);
                let child_path = format!("_f->_child_u{g}.{slot_name}");
                let mut init_fields = vec!["._pc = 0U".to_string()];
                if let Some(callee_fn) = self.m.functions.iter().find(|f| f.name == *callee) {
                    for (idx, &pvid) in callee_fn.params.iter().enumerate() {
                        if let Some(arg_expr) = args.get(idx) {
                            let pvar = &callee_fn.vars[pvid.0];
                            init_fields.push(format!(
                                ".{} = {}",
                                frame_var_name(pvar),
                                self.fmt_expr(arg_expr, vars)
                            ));
                        }
                    }
                }
                let _ = writeln!(
                    self.out,
                    "    {child_path} = (syn_frame_fn_{callee}_t){{ {} }};\n    goto _bb{};",
                    init_fields.join(", "),
                    poll_bb.0
                );
            }
            MirYieldKind::PollChild {
                callee,
                slot_name,
                dst,
                poll_bb,
                resume_bb,
            } => {
                let g = slot_groups.get(slot_name.as_str()).copied().unwrap_or(0);
                let child_path = format!("_f->_child_u{g}.{slot_name}");
                let p = poll_bb.0;
                let r = resume_bb.0;
                let _ = writeln!(
                    self.out,
                    "    for (;;) {{\n        syn_step_t _cst = syn_coro_{callee}(_task, &{child_path});\n        if (_cst == SYN_STEP_DONE) {{"
                );
                if let Some(d) = dst {
                    match d {
                        MirCallDst::Value(v) | MirCallDst::Result(v) => {
                            if !vars[v.0].ty.is_void() {
                                let _ = writeln!(
                                    self.out,
                                    "            {} = {child_path}._ret;",
                                    self.var_ref(*v, vars)
                                );
                            }
                        }
                    }
                }
                let _ = writeln!(
                    self.out,
                    "            goto _bb{r};\n        }}\n        if (_cst == SYN_STEP_IDLE || syn_task_should_share(_task)) {{\n            _f->_pc = {p}U;\n            return _cst;\n        }}\n    }}"
                );
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Statements & Expressions Formatting
    // ─────────────────────────────────────────────────────────────────────────

    fn emit_direct_stmt(
        &mut self,
        stmt: &MirStmt,
        vars: &[MirVar],
        ret_ty: &MirType,
        error_spec: &MirErrorSpec,
        indent: usize,
        in_coro_block: bool,
    ) {
        let pad = "    ".repeat(indent);
        match stmt {
            MirStmt::Assign { target, op, value } => {
                if let MirPlace::Field {
                    field_ty:
                        MirType::EndianInt {
                            bits, big_endian, ..
                        },
                    ..
                } = target
                {
                    let lhs = self.fmt_place_raw(target, vars);
                    let rhs = self.fmt_expr(value, vars);
                    let end = if *big_endian { "be" } else { "le" };
                    let host_c = format!("uint{bits}_t");
                    if *op == MirAssignOp::Assign {
                        let _ = writeln!(
                            self.out,
                            "{pad}{lhs} = syn_make_u{bits}{end}(({host_c})({rhs}));"
                        );
                    } else {
                        let bop = assign_op_bin_str(*op);
                        let _ = writeln!(
                            self.out,
                            "{pad}{lhs} = syn_make_u{bits}{end}(({host_c})(syn_load_u{bits}{end}({lhs}) {bop} ({rhs})));"
                        );
                    }
                    return;
                }
                let lhs = self.fmt_place_raw(target, vars);
                let rhs = self.fmt_expr(value, vars);
                if *op == MirAssignOp::RemAssign && value.ty.is_float() {
                    let fty = value.ty.host_value_type();
                    let fn_name = if fty == MirType::F32 {
                        "syn_fmod_f32"
                    } else {
                        "syn_fmod_f64"
                    };
                    let c_ty = self.c_type(&fty);
                    let _ = writeln!(
                        self.out,
                        "{pad}{{ {c_ty} *_rem_p = &({lhs}); *_rem_p = {fn_name}(*_rem_p, {rhs}); }}"
                    );
                    return;
                }
                let op_str = assign_op_str(*op);
                let _ = writeln!(self.out, "{pad}{lhs} {op_str} {rhs};");
            }
            MirStmt::Expr(e) => {
                let s = self.fmt_expr(e, vars);
                if e.ty.is_void() {
                    let _ = writeln!(self.out, "{pad}{s};");
                } else {
                    let _ = writeln!(self.out, "{pad}(void)({s});");
                }
            }
            MirStmt::PrimOp(op) => self.emit_prim_op(op, vars, &pad),
            MirStmt::Intrinsic(intr) => {
                self.emit_intrinsic(intr, vars, &pad);
                if !in_coro_block
                    && matches!(intr, MirIntrinsicOp::SysOp { method, .. } if method == "exit")
                {
                    let env_ret = if error_spec.has_error() {
                        MirType::Result {
                            ok: Box::new(ret_ty.clone()),
                            err: error_spec.err_type().map(Box::new),
                        }
                    } else {
                        ret_ty.clone()
                    };
                    if env_ret.is_void() {
                        let _ = writeln!(self.out, "{pad}return;");
                    } else {
                        let _ = writeln!(self.out, "{pad}return {};", self.zero_init_expr(&env_ret));
                    }
                }
            }
            MirStmt::SpawnActivate { slot_id, args } => {
                self.emit_spawn_activation(*slot_id, args, vars, &pad);
            }
            MirStmt::PrintFmt { chunks, newline } => {
                let _ = writeln!(self.out, "{pad}{{");
                let _ = writeln!(
                    self.out,
                    "{pad}    syn_fmt_sink_t _sink = {{ .dst = NULL, .cap = 0U, .pos = 0U, .truncated = false, .is_buf = false }};"
                );
                self.emit_fmt_chunks("_sink", chunks, vars, indent + 1);
                if *newline {
                    let _ = writeln!(self.out, "{pad}    syn_fmt_putc(&_sink, '\\n');");
                }
                let _ = writeln!(
                    self.out,
                    "#if !defined(SYN_FREESTANDING)\n{pad}    fflush(stdout);\n#endif\n{pad}}}"
                );
            }
            MirStmt::FmtBuf {
                dst_res,
                buf,
                chunks,
            } => {
                let buf_s = self.fmt_expr(buf, vars);
                let dst_s = self.var_ref(*dst_res, vars);
                let res_c = self.c_type(&vars[dst_res.0].ty);
                let _ = writeln!(self.out, "{pad}{{");
                let _ = writeln!(self.out, "{pad}    syn_slice_u8 _fbuf = {buf_s};");
                let _ = writeln!(
                    self.out,
                    "{pad}    syn_fmt_sink_t _sink = {{ .dst = _fbuf.ptr, .cap = _fbuf.len, .pos = 0U, .truncated = false, .is_buf = true }};"
                );
                self.emit_fmt_chunks("_sink", chunks, vars, indent + 1);
                let _ = writeln!(
                    self.out,
                    "{pad}    syn_slice_u8 _written = {{ .ptr = _fbuf.ptr, .len = _sink.pos }};"
                );
                let _ = writeln!(
                    self.out,
                    "{pad}    if (_sink.truncated) {{\n{pad}        {dst_s} = ({res_c}){{ .is_err = true, .err = syn_FmtErr_Truncated, .val = _written }};\n{pad}    }} else {{\n{pad}        {dst_s} = ({res_c}){{ .is_err = false, .val = _written }};\n{pad}    }}\n{pad}}}"
                );
            }
            MirStmt::MmioWrite {
                base,
                offset,
                elem_bits,
                index,
                array_len,
                value,
            } => {
                let base_s = self.fmt_expr(base, vars);
                let val_s = self.fmt_expr(value, vars);
                match index {
                    None => {
                        let _ = writeln!(
                            self.out,
                            "{pad}syn_mmio_write_u{elem_bits}({base_s}, {offset}U, (uint{elem_bits}_t)({val_s}));"
                        );
                    }
                    Some(idx_expr) => {
                        let idx_s = self.fmt_expr(idx_expr, vars);
                        let len_val = array_len.unwrap_or(1);
                        let _ = writeln!(
                            self.out,
                            "{pad}syn_mmio_write_arr_u{elem_bits}({base_s}, {offset}U, (size_t)({idx_s}), (size_t){len_val}U, (uint{elem_bits}_t)({val_s}));"
                        );
                    }
                }
            }
            MirStmt::SetIrqSelector { symbol, value } => {
                let _ = writeln!(self.out, "{pad}{symbol} = (uint8_t){value}U;");
            }
            MirStmt::EnterCritical => {
                let _ = writeln!(self.out, "{pad}syn_enter_critical();");
            }
            MirStmt::ExitCritical => {
                let _ = writeln!(self.out, "{pad}syn_exit_critical();");
            }
            MirStmt::EndInboxTurn => {
                let _ = writeln!(self.out, "{pad}syn_task_end_inbox_turn(_task);");
            }
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let c_raw = self.fmt_expr(cond, vars);
                let c = strip_outer_parens(&c_raw);
                let _ = writeln!(self.out, "{pad}if ({c}) {{");
                for s in then_body {
                    self.emit_direct_stmt(s, vars, ret_ty, error_spec, indent + 1, in_coro_block);
                }
                if else_body.is_empty() {
                    let _ = writeln!(self.out, "{pad}}}");
                } else {
                    let _ = writeln!(self.out, "{pad}}} else {{");
                    for s in else_body {
                        self.emit_direct_stmt(
                            s,
                            vars,
                            ret_ty,
                            error_spec,
                            indent + 1,
                            in_coro_block,
                        );
                    }
                    let _ = writeln!(self.out, "{pad}}}");
                }
            }
            MirStmt::While { cond, body } => {
                let c_raw = self.fmt_expr(cond, vars);
                let c = strip_outer_parens(&c_raw);
                let _ = writeln!(self.out, "{pad}while ({c}) {{");
                for s in body {
                    self.emit_direct_stmt(s, vars, ret_ty, error_spec, indent + 1, in_coro_block);
                }
                let _ = writeln!(self.out, "{pad}}}");
            }
            MirStmt::Switch {
                discr,
                arms,
                default,
            } => {
                let has_loop_break = arms
                    .iter()
                    .any(|(_, b)| direct_stmts_have_unnested_break(b))
                    || direct_stmts_have_unnested_break(default);
                let d = self.fmt_expr(discr, vars);
                if has_loop_break {
                    let dt = self.c_type(&discr.ty.host_value_type());
                    let _ = writeln!(self.out, "{pad}{{");
                    let _ = writeln!(self.out, "{pad}    {dt} _sw_d = {d};");
                    if arms.is_empty() {
                        let _ = writeln!(self.out, "{pad}    (void)_sw_d;");
                        for s in default {
                            self.emit_direct_stmt(
                                s,
                                vars,
                                ret_ty,
                                error_spec,
                                indent + 1,
                                in_coro_block,
                            );
                        }
                    } else {
                        for (idx, (pat, body)) in arms.iter().enumerate() {
                            let ps = self.fmt_switch_pat(pat);
                            let kw = if idx == 0 { "if" } else { "} else if" };
                            let _ = writeln!(self.out, "{pad}    {kw} (_sw_d == {ps}) {{");
                            for s in body {
                                self.emit_direct_stmt(
                                    s,
                                    vars,
                                    ret_ty,
                                    error_spec,
                                    indent + 2,
                                    in_coro_block,
                                );
                            }
                        }
                        if default.is_empty() {
                            let _ = writeln!(self.out, "{pad}    }}");
                        } else {
                            let _ = writeln!(self.out, "{pad}    }} else {{");
                            for s in default {
                                self.emit_direct_stmt(
                                    s,
                                    vars,
                                    ret_ty,
                                    error_spec,
                                    indent + 2,
                                    in_coro_block,
                                );
                            }
                            let _ = writeln!(self.out, "{pad}    }}");
                        }
                    }
                    let _ = writeln!(self.out, "{pad}}}");
                } else {
                    let _ = writeln!(self.out, "{pad}switch ({d}) {{");
                    for (pat, body) in arms {
                        let ps = self.fmt_switch_pat(pat);
                        let _ = writeln!(self.out, "{pad}    case {ps}: {{");
                        for s in body {
                            self.emit_direct_stmt(
                                s,
                                vars,
                                ret_ty,
                                error_spec,
                                indent + 2,
                                in_coro_block,
                            );
                        }
                        let _ = writeln!(self.out, "{pad}        break;\n{pad}    }}");
                    }
                    let _ = writeln!(self.out, "{pad}    default: {{");
                    for s in default {
                        self.emit_direct_stmt(
                            s,
                            vars,
                            ret_ty,
                            error_spec,
                            indent + 2,
                            in_coro_block,
                        );
                    }
                    let _ = writeln!(self.out, "{pad}        break;\n{pad}    }}\n{pad}}}");
                }
            }
            MirStmt::Return(expr_opt) => {
                if error_spec.has_error() {
                    let env = MirType::Result {
                        ok: Box::new(ret_ty.clone()),
                        err: error_spec.err_type().map(Box::new),
                    };
                    let rc = self.c_type(&env);
                    if let Some(e) = expr_opt {
                        let v = self.fmt_expr(e, vars);
                        let _ = writeln!(
                            self.out,
                            "{pad}return ({rc}){{ .is_err = false, .val = {v} }};"
                        );
                    } else {
                        let _ = writeln!(self.out, "{pad}return ({rc}){{ .is_err = false }};");
                    }
                } else if let Some(e) = expr_opt {
                    let v = self.fmt_expr(e, vars);
                    let _ = writeln!(self.out, "{pad}return {v};");
                } else {
                    let _ = writeln!(self.out, "{pad}return;");
                }
            }
            MirStmt::Error { reason, partial } => {
                let env = MirType::Result {
                    ok: Box::new(ret_ty.clone()),
                    err: error_spec.err_type().map(Box::new),
                };
                let rc = self.c_type(&env);
                let mut fields = vec![".is_err = true".to_string()];
                if let Some(r) = reason {
                    fields.push(format!(".err = {}", self.fmt_expr(r, vars)));
                }
                if let Some(p) = partial {
                    if !ret_ty.is_void() {
                        fields.push(format!(".val = {}", self.fmt_expr(p, vars)));
                    }
                }
                let _ = writeln!(self.out, "{pad}return ({rc}){{ {} }};", fields.join(", "));
            }
            MirStmt::Break => {
                let _ = writeln!(self.out, "{pad}break;");
            }
            MirStmt::Continue => {
                let _ = writeln!(self.out, "{pad}continue;");
            }
        }
    }

    fn emit_prim_op(&mut self, op: &MirPrimNonBlockingOp, vars: &[MirVar], pad: &str) {
        match op {
            MirPrimNonBlockingOp::WatchPublish {
                target,
                elem_ty,
                value,
            } => {
                let mangle = MirPrimKind::Watch {
                    elem: Box::new(elem_ty.clone()),
                }
                .mangle_name();
                let ptr = self.fmt_prim_ptr(target, vars);
                let v = self.fmt_expr(value, vars);
                let _ = writeln!(self.out, "{pad}syn_{mangle}_publish({ptr}, {v});");
            }
            MirPrimNonBlockingOp::InboxPost {
                target,
                elem_ty,
                cap,
                value,
                boost,
            } => {
                let mangle = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                }
                .mangle_name();
                let ptr = self.fmt_prim_ptr(target, vars);
                let v = self.fmt_expr(value, vars);
                let b = boost.map_or("SYN_NO_BOOST", |p| p.c_const_name());
                let _ = writeln!(self.out, "{pad}(void)syn_{mangle}_post({ptr}, {v}, {b});");
            }
            MirPrimNonBlockingOp::SignalNotify { target, boost } => {
                let ptr = self.fmt_prim_ptr(target, vars);
                let b = boost.map_or("SYN_NO_BOOST", |p| p.c_const_name());
                let _ = writeln!(self.out, "{pad}syn_signal_notify({ptr}, {b});");
            }
            MirPrimNonBlockingOp::SignalClear { target } => {
                let ptr = self.fmt_prim_ptr(target, vars);
                let _ = writeln!(self.out, "{pad}syn_signal_clear({ptr});");
            }
            MirPrimNonBlockingOp::PingPongFlip { target, .. } => {
                let ptr = self.fmt_prim_ptr(target, vars);
                let _ = writeln!(self.out, "{pad}syn_pp_hdr_flip(&({ptr})->hdr);");
            }
            MirPrimNonBlockingOp::PingPongSetActive { target, active, .. } => {
                let ptr = self.fmt_prim_ptr(target, vars);
                let _ = writeln!(
                    self.out,
                    "{pad}syn_pp_hdr_set_active(&({ptr})->hdr, {active});"
                );
            }
            MirPrimNonBlockingOp::LeaseComplete {
                target,
                elem_ty,
                cap,
            } => {
                let mangle = MirPrimKind::Lease {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                }
                .mangle_name();
                let ptr = self.fmt_prim_ptr(target, vars);
                let _ = writeln!(self.out, "{pad}syn_{mangle}_complete({ptr});");
            }
            MirPrimNonBlockingOp::LeaseSetActive {
                target,
                elem_ty,
                cap,
                active,
            } => {
                let mangle = MirPrimKind::Lease {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                }
                .mangle_name();
                let ptr = self.fmt_prim_ptr(target, vars);
                let _ = writeln!(self.out, "{pad}syn_{mangle}_set_active({ptr}, {active});");
            }
        }
    }

    fn emit_intrinsic(&mut self, intr: &MirIntrinsicOp, vars: &[MirVar], pad: &str) {
        match intr {
            MirIntrinsicOp::UseSystick { hz } => {
                let h = self.fmt_expr(hz, vars);
                let _ = writeln!(self.out, "{pad}syn_sys_use_systick((uint32_t)({h}));");
            }
            MirIntrinsicOp::SetPriority { prio } => {
                let _ = writeln!(
                    self.out,
                    "{pad}syn_task_set_base_priority(g_syn_sched.running, {});",
                    prio.c_const_name()
                );
            }
            MirIntrinsicOp::SysOp { method, args } => match method.as_str() {
                "log" | "println" => {
                    let s = args
                        .first()
                        .map(|a| self.fmt_expr(a, vars))
                        .unwrap_or_else(|| "\"\"".to_string());
                    let _ = writeln!(self.out, "{pad}syn_sys_log({s});");
                }
                "print" | "puts" => {
                    let s = args
                        .first()
                        .map(|a| self.fmt_expr(a, vars))
                        .unwrap_or_else(|| "\"\"".to_string());
                    let _ = writeln!(self.out, "{pad}print({s});");
                }
                "log_kv" => {
                    let label = args
                        .first()
                        .map(|a| self.fmt_expr(a, vars))
                        .unwrap_or_else(|| "\"\"".to_string());
                    let val = args
                        .get(1)
                        .map(|a| self.fmt_expr(a, vars))
                        .unwrap_or_else(|| "0".to_string());
                    let _ = writeln!(self.out, "{pad}syn_sys_log_kv({label}, (int32_t)({val}));");
                }
                "log_hex" => {
                    let label = args
                        .first()
                        .map(|a| self.fmt_expr(a, vars))
                        .unwrap_or_else(|| "\"\"".to_string());
                    let val = args
                        .get(1)
                        .map(|a| self.fmt_expr(a, vars))
                        .unwrap_or_else(|| "0U".to_string());
                    let _ = writeln!(
                        self.out,
                        "{pad}syn_sys_log_hex({label}, (uint32_t)({val}));"
                    );
                }
                "exit" => {
                    let code = args
                        .first()
                        .map(|a| self.fmt_expr(a, vars))
                        .unwrap_or_else(|| "0".to_string());
                    let _ = writeln!(self.out, "{pad}syn_sim_exit((int32_t)({code}));");
                }
                "stop" => {
                    for a in args {
                        let s = self.fmt_expr(a, vars);
                        let _ = writeln!(self.out, "{pad}(void)({s});");
                    }
                    let _ = writeln!(self.out, "{pad}syn_sim_stop();");
                }
                "pend_irq" => {
                    let irq = args
                        .first()
                        .map(|a| self.fmt_expr(a, vars))
                        .unwrap_or_else(|| "0U".to_string());
                    let _ = writeln!(self.out, "{pad}syn_sim_pend_irq((uint32_t)({irq}));");
                }
                "trace_sleep" => {
                    let on = args
                        .first()
                        .map(|a| self.fmt_expr(a, vars))
                        .unwrap_or_else(|| "false".to_string());
                    let _ = writeln!(self.out, "{pad}syn_sim_enable_trace((bool)({on}));");
                }
                _ => {
                    for a in args {
                        let s = self.fmt_expr(a, vars);
                        let _ = writeln!(self.out, "{pad}(void)({s});");
                    }
                }
            },
        }
    }

    fn fmt_prim_ptr(&self, target: &MirPrimTarget, vars: &[MirVar]) -> String {
        match target {
            MirPrimTarget::Global(name) => format!("&{}", c_ident(name)),
            MirPrimTarget::PtrExpr(e) => self.fmt_expr(e, vars),
        }
    }

    fn var_ref(&self, vid: VarId, vars: &[MirVar]) -> String {
        if let Some(v) = vars.get(vid.0) {
            if v.role == VarRole::IsrState {
                format!("(*__state_{})", v.name)
            } else if v.hoisted {
                format!("_f->{}", frame_var_name(v))
            } else {
                stack_var_name(v)
            }
        } else {
            format!("_v{}", vid.0)
        }
    }

    fn fmt_place_raw(&self, place: &MirPlace, vars: &[MirVar]) -> String {
        match place {
            MirPlace::Var(vid) => self.var_ref(*vid, vars),
            MirPlace::Deref(ptr, _) => format!("(*({}))", self.fmt_expr(ptr, vars)),
            MirPlace::Field { base, field, .. } => {
                if let MirPlace::Deref(ptr, _) = base.as_ref() {
                    format!("({})->{}", self.fmt_expr(ptr, vars), c_ident(field))
                } else {
                    format!("{}.{}", self.fmt_place_raw(base, vars), c_ident(field))
                }
            }
            MirPlace::Index {
                base,
                index,
                is_slice,
                ..
            } => {
                let b = self.fmt_place_raw(base, vars);
                let idx = self.fmt_expr(index, vars);
                if *is_slice {
                    format!("{b}.ptr[{idx}]")
                } else {
                    format!("{b}.data[{idx}]")
                }
            }
        }
    }

    fn fmt_switch_pat(&self, pat: &MirSwitchPattern) -> String {
        match pat {
            MirSwitchPattern::EnumVariant { enum_name, variant } => {
                format!("syn_{enum_name}_{variant}")
            }
            MirSwitchPattern::Int(v) => format!("{v}"),
            MirSwitchPattern::Bool(b) => format!("{b}"),
        }
    }

    fn fmt_expr(&self, expr: &MirExpr, vars: &[MirVar]) -> String {
        match &expr.kind {
            MirExprKind::IntLit(v) => {
                let ct = self.c_type(&expr.ty.host_value_type());
                if expr.ty.is_unsigned() && *v >= 0 {
                    format!("(({ct}){v}ULL)")
                } else {
                    format!("(({ct}){v}LL)")
                }
            }
            MirExprKind::FloatLit(bits) => {
                format_c99_float_lit(*bits, expr.ty.host_value_type() == MirType::F32)
            }
            MirExprKind::BoolLit(b) => {
                if *b {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            MirExprKind::StringLit(s) => escape_c_string(s),
            MirExprKind::TicksLit(t) => format!("((uint32_t){t}U)"),
            MirExprKind::EnumVariant { enum_name, variant } => {
                format!("syn_{enum_name}_{variant}")
            }
            MirExprKind::ConstRef(name) => c_ident(name),
            MirExprKind::GlobalPrimAddr(name) => format!("(&{})", c_ident(name)),
            MirExprKind::MmioAddr(name) => mmio_c_macro_name(name),
            MirExprKind::MmioRead {
                base,
                offset,
                elem_bits,
                index,
                array_len,
            } => {
                let base_s = self.fmt_expr(base, vars);
                match index {
                    None => format!("syn_mmio_read_u{elem_bits}({base_s}, {offset}U)"),
                    Some(idx_expr) => {
                        let idx_s = self.fmt_expr(idx_expr, vars);
                        let len_val = array_len.unwrap_or(1);
                        format!(
                            "syn_mmio_read_arr_u{elem_bits}({base_s}, {offset}U, (size_t)({idx_s}), (size_t){len_val}U)"
                        )
                    }
                }
            }
            MirExprKind::FnAddr(fname) => {
                if self.m.functions.iter().any(|f| f.name == *fname) {
                    format!("syn_fn_{fname}")
                } else {
                    fname.clone()
                }
            }
            MirExprKind::Var(vid) => self.var_ref(*vid, vars),
            MirExprKind::Unary { op, expr: inner } => {
                let sub = self.fmt_expr(inner, vars);
                match op {
                    MirUnaryOp::Neg => format!("(-({sub}))"),
                    MirUnaryOp::Not => format!("(!({sub}))"),
                    MirUnaryOp::BitNot => format!("(~({sub}))"),
                    MirUnaryOp::Deref => format!("(*({sub}))"),
                }
            }
            MirExprKind::AddrOfPlace(place) => {
                if let MirPlace::Var(vid) = place.as_ref() {
                    if let Some(v) = vars.get(vid.0) {
                        if v.role == VarRole::IsrState {
                            return format!("__state_{}", v.name);
                        }
                    }
                }
                format!("(&({}))", self.fmt_place_raw(place, vars))
            }
            MirExprKind::Binary { op, lhs, rhs } => {
                let l = self.fmt_expr(lhs, vars);
                let mut r = self.fmt_expr(rhs, vars);
                if *op == MirBinOp::Rem && (lhs.ty.is_float() || rhs.ty.is_float()) {
                    if lhs.ty.host_value_type() == MirType::F32 {
                        return format!("syn_fmod_f32({l}, {r})");
                    }
                    return format!("syn_fmod_f64({l}, {r})");
                }
                if *op == MirBinOp::Div && (lhs.ty.is_float() || rhs.ty.is_float()) {
                    if lhs.ty.host_value_type() == MirType::F32 {
                        return format!("syn_fdiv_f32({l}, {r})");
                    }
                    return format!("syn_fdiv_f64({l}, {r})");
                }
                // Prevent any `-Wsign-compare` if signedness differs on integer operands
                if !lhs.ty.is_float()
                    && !rhs.ty.is_float()
                    && lhs.ty.is_unsigned() != rhs.ty.is_unsigned()
                    && !lhs.ty.is_void()
                {
                    let lct = self.c_type(&lhs.ty.host_value_type());
                    r = format!("(({lct})({r}))");
                }
                let op_s = bin_op_str(*op);
                format!("(({l}) {op_s} ({r}))")
            }
            MirExprKind::Cast {
                expr: inner,
                target_ty,
            } => {
                let sub = self.fmt_expr(inner, vars);
                let host_target = target_ty.host_value_type();
                if inner.ty.is_float() {
                    let sat_fn = match host_target {
                        MirType::U8 => Some("syn_f64_to_u8_sat"),
                        MirType::I8 => Some("syn_f64_to_i8_sat"),
                        MirType::U16 => Some("syn_f64_to_u16_sat"),
                        MirType::I16 => Some("syn_f64_to_i16_sat"),
                        MirType::U32 => Some("syn_f64_to_u32_sat"),
                        MirType::I32 => Some("syn_f64_to_i32_sat"),
                        MirType::U64 => Some("syn_f64_to_u64_sat"),
                        MirType::I64 => Some("syn_f64_to_i64_sat"),
                        MirType::Usize => Some("syn_f64_to_usize_sat"),
                        MirType::Isize => Some("syn_f64_to_isize_sat"),
                        _ => None,
                    };
                    if let Some(fname) = sat_fn {
                        return format!("{fname}((double)({sub}))");
                    }
                }
                let ct = self.c_type(&host_target);
                format!("(({ct})({sub}))")
            }
            MirExprKind::StructLit {
                struct_name,
                fields,
                ..
            } => {
                if fields.is_empty() {
                    return format!("((syn_struct_{struct_name}){{ ._empty = 0U }})");
                }
                let fs: Vec<String> = fields
                    .iter()
                    .map(|(fname, fval, raw_fty)| {
                        let vstr = self.fmt_expr(fval, vars);
                        let wrapped = match raw_fty {
                            MirType::EndianInt {
                                bits, big_endian, ..
                            } => {
                                let end = if *big_endian { "be" } else { "le" };
                                format!("syn_make_u{bits}{end}((uint{bits}_t)({vstr}))")
                            }
                            _ => vstr,
                        };
                        format!(".{} = {wrapped}", c_ident(fname))
                    })
                    .collect();
                format!("((syn_struct_{struct_name}){{ {} }})", fs.join(", "))
            }
            MirExprKind::ArrayLit { elems, .. } => {
                let ct = self.c_type(&expr.ty);
                if elems.is_empty() {
                    format!("(({ct}){{ .data = {{ 0 }} }})")
                } else {
                    let es: Vec<String> = elems.iter().map(|e| self.fmt_expr(e, vars)).collect();
                    format!("(({ct}){{ .data = {{ {} }} }})", es.join(", "))
                }
            }
            MirExprKind::FieldAccess {
                base,
                is_ptr,
                field,
                raw_field_ty,
                ..
            } => {
                let b = self.fmt_expr(base, vars);
                let raw = if *is_ptr {
                    format!("({b})->{}", c_ident(field))
                } else {
                    format!("({b}).{}", c_ident(field))
                };
                match raw_field_ty {
                    MirType::EndianInt {
                        bits, big_endian, ..
                    } => {
                        let end = if *big_endian { "be" } else { "le" };
                        format!("syn_load_u{bits}{end}({raw})")
                    }
                    _ => raw,
                }
            }
            MirExprKind::SliceLen(inner) => format!("(({}).len)", self.fmt_expr(inner, vars)),
            MirExprKind::ArrayLen(n) => format!("((size_t){n}U)"),
            MirExprKind::Index {
                base,
                index,
                is_slice,
            } => {
                let b = self.fmt_expr(base, vars);
                let idx = self.fmt_expr(index, vars);
                if *is_slice {
                    format!("({b}).ptr[{idx}]")
                } else {
                    format!("({b}).data[{idx}]")
                }
            }
            MirExprKind::SubSlice {
                base,
                start,
                end,
                elem_ty,
                ..
            } => {
                let b = self.fmt_expr(base, vars);
                let s = self.fmt_expr(start, vars);
                let e = self.fmt_expr(end, vars);
                if *elem_ty == MirType::U8 {
                    format!("syn_subslice_u8({b}, (size_t)({s}), (size_t)({e}))")
                } else {
                    let mangle = MirType::Slice(Box::new(elem_ty.clone())).mangle_name();
                    format!("syn_sub_{mangle}({b}, (size_t)({s}), (size_t)({e}))")
                }
            }
            MirExprKind::ArrayToSlice {
                array_place,
                len,
                elem_ty,
            } => {
                let sc = self.c_type(&MirType::Slice(Box::new(elem_ty.clone())));
                let p = self.fmt_place_raw(array_place, vars);
                format!("(({sc}){{ .ptr = &({p}).data[0], .len = (size_t){len}U }})")
            }
            MirExprKind::FromBytes { struct_name, slice } => {
                format!(
                    "syn_{struct_name}_from_bytes({})",
                    self.fmt_expr(slice, vars)
                )
            }
            MirExprKind::PrimRead(ro) => match ro.as_ref() {
                MirPrimReadOp::WatchRead { target, elem_ty } => {
                    let mangle = MirPrimKind::Watch {
                        elem: Box::new(elem_ty.clone()),
                    }
                    .mangle_name();
                    format!("syn_{mangle}_read({})", self.fmt_prim_ptr(target, vars))
                }
                MirPrimReadOp::InboxTryRecv {
                    target,
                    elem_ty,
                    cap,
                } => {
                    let mangle = MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    }
                    .mangle_name();
                    format!("syn_{mangle}_try_recv({})", self.fmt_prim_ptr(target, vars))
                }
                MirPrimReadOp::InboxTrySend {
                    target,
                    elem_ty,
                    cap,
                    value,
                    boost,
                } => {
                    let mangle = MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    }
                    .mangle_name();
                    let ptr = self.fmt_prim_ptr(target, vars);
                    let v = self.fmt_expr(value, vars);
                    let b = boost.map_or("SYN_NO_BOOST", |p| p.c_const_name());
                    format!("syn_{mangle}_post({ptr}, {v}, {b})")
                }
                MirPrimReadOp::InboxLen { target, .. } => {
                    format!("((size_t)(({})->count))", self.fmt_prim_ptr(target, vars))
                }
                MirPrimReadOp::InboxIsEmpty { target, .. } => {
                    format!("((({})->count) == 0U)", self.fmt_prim_ptr(target, vars))
                }
                MirPrimReadOp::InboxIsFull { target, cap, .. } => {
                    let n = (*cap).max(1);
                    format!("((({})->count) >= {n}U)", self.fmt_prim_ptr(target, vars))
                }
                MirPrimReadOp::SignalIsSet { target } => {
                    format!("(({})->flag)", self.fmt_prim_ptr(target, vars))
                }
                MirPrimReadOp::SignalTryWait { target } => {
                    format!(
                        "syn_signal_try_consume({}, g_syn_sched.running)",
                        self.fmt_prim_ptr(target, vars)
                    )
                }
                MirPrimReadOp::PingPongDmaBuf {
                    target,
                    elem_ty,
                    cap,
                } => {
                    let mangle = MirPrimKind::PingPong {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    }
                    .mangle_name();
                    format!("syn_{mangle}_dma_buf({})", self.fmt_prim_ptr(target, vars))
                }
            },
            MirExprKind::CallDirect {
                callee,
                is_extern_c,
                args,
            } => {
                let args_s: Vec<String> = args.iter().map(|a| self.fmt_expr(a, vars)).collect();
                let cname = if *is_extern_c {
                    if callee == "assert" {
                        "syn_builtin_assert".to_string()
                    } else if let Some(v) = vars
                        .iter()
                        .rev()
                        .find(|v| v.name == *callee && matches!(v.ty, MirType::FnPtr { .. }))
                    {
                        self.var_ref(v.id, vars)
                    } else {
                        callee.clone()
                    }
                } else {
                    format!("syn_fn_{callee}")
                };
                format!("{cname}({})", args_s.join(", "))
            }
            MirExprKind::ResultIsErr(vid) => format!("({}.is_err)", self.var_ref(*vid, vars)),
            MirExprKind::ResultErr(vid) => format!("({}.err)", self.var_ref(*vid, vars)),
            MirExprKind::ResultVal(vid) => format!("({}.val)", self.var_ref(*vid, vars)),
            MirExprKind::NowTicks => "syn_now_ms()".to_string(),
            MirExprKind::SysQuery(q) => match q {
                MirSysQuery::CurrentPrio => {
                    "(g_syn_sched.running != NULL ? (uint32_t)g_syn_sched.running->priority : 255U)"
                        .to_string()
                }
                MirSysQuery::SleepDepth => "(uint32_t)syn_allowed_sleep_depth()".to_string(),
                MirSysQuery::WfiCount => "g_syn_sleep.enter_count".to_string(),
            },
        }
    }

    fn fmt_const_endian_bytes(bits: u8, big_endian: bool, val: i128) -> String {
        let nbytes = (bits / 8) as usize;
        let u = val as u64;
        let bytes: Vec<String> = (0..nbytes)
            .map(|i| {
                let shift = if big_endian { (nbytes - 1 - i) * 8 } else { i * 8 };
                format!("0x{:02X}U", (u >> shift) & 0xFF)
            })
            .collect();
        let end = if big_endian { "be" } else { "le" };
        format!("((syn_u{bits}{end}_t){{ .b = {{ {} }} }})", bytes.join(", "))
    }

    fn fmt_const_init_expr(&self, expr: &MirExpr) -> String {
        if let Some(cv) = self.eval_const_val(expr, 0) {
            return match cv {
                ConstInitVal::Bool(b) => {
                    if b {
                        "true".to_string()
                    } else {
                        "false".to_string()
                    }
                }
                ConstInitVal::Float(f) => {
                    format_c99_float_lit(f.to_bits(), expr.ty.host_value_type() == MirType::F32)
                }
                ConstInitVal::Int(v) => {
                    if let MirType::EndianInt {
                        bits, big_endian, ..
                    } = &expr.ty
                    {
                        return Self::fmt_const_endian_bytes(*bits, *big_endian, v);
                    }
                    let ct = self.c_type(&expr.ty.host_value_type());
                    if expr.ty.is_unsigned() {
                        format!("(({ct}){}ULL)", v as u128 as u64)
                    } else {
                        let sv = v as i64;
                        if sv == i64::MIN {
                            format!("(({ct})INT64_MIN)")
                        } else {
                            format!("(({ct}){sv}LL)")
                        }
                    }
                }
            };
        }
        match &expr.kind {
            MirExprKind::ConstRef(name) => {
                if let Some(c) = self.m.consts.iter().find(|c| c.name == *name) {
                    self.fmt_const_init_expr(&c.init)
                } else {
                    c_ident(name)
                }
            }
            MirExprKind::StructLit {
                struct_name,
                fields,
                ..
            } => {
                if fields.is_empty() {
                    return format!("((syn_struct_{struct_name}){{ ._empty = 0U }})");
                }
                let fs: Vec<String> = fields
                    .iter()
                    .map(|(fname, fval, raw_fty)| {
                        let wrapped = match raw_fty {
                            MirType::EndianInt {
                                bits, big_endian, ..
                            } => {
                                if let Some(ConstInitVal::Int(raw)) = self.eval_const_val(fval, 0) {
                                    Self::fmt_const_endian_bytes(*bits, *big_endian, raw)
                                } else {
                                    let vstr = self.fmt_const_init_expr(fval);
                                    let end = if *big_endian { "be" } else { "le" };
                                    format!("syn_make_u{bits}{end}((uint{bits}_t)({vstr}))")
                                }
                            }
                            _ => self.fmt_const_init_expr(fval),
                        };
                        format!(".{} = {wrapped}", c_ident(fname))
                    })
                    .collect();
                format!("((syn_struct_{struct_name}){{ {} }})", fs.join(", "))
            }
            MirExprKind::ArrayLit { elems, .. } => {
                let ct = self.c_type(&expr.ty);
                if elems.is_empty() {
                    format!("(({ct}){{ .data = {{ 0 }} }})")
                } else {
                    let es: Vec<String> = elems
                        .iter()
                        .map(|e| self.fmt_const_init_expr(e))
                        .collect();
                    format!("(({ct}){{ .data = {{ {} }} }})", es.join(", "))
                }
            }
            _ => self.fmt_expr(expr, &[]),
        }
    }

    fn eval_const_val(&self, expr: &MirExpr, depth: usize) -> Option<ConstInitVal> {
        if depth > 32 {
            return None;
        }
        let norm_float = |f: f64, ty: &MirType| -> f64 {
            if ty.host_value_type() == MirType::F32 {
                (f as f32) as f64
            } else {
                f
            }
        };
        let trunc_int = |v: i128, ty: &MirType| -> i128 {
            match ty.host_value_type() {
                MirType::U8 => (v as u8) as i128,
                MirType::I8 => (v as i8) as i128,
                MirType::U16 => (v as u16) as i128,
                MirType::I16 => (v as i16) as i128,
                MirType::U32 => (v as u32) as i128,
                MirType::I32 => (v as i32) as i128,
                MirType::U64 | MirType::Usize => (v as u64) as i128,
                MirType::I64 | MirType::Isize => (v as i64) as i128,
                _ => v,
            }
        };
        match &expr.kind {
            MirExprKind::IntLit(v) => Some(ConstInitVal::Int(trunc_int(*v, &expr.ty))),
            MirExprKind::TicksLit(t) => Some(ConstInitVal::Int(*t as i128)),
            MirExprKind::FloatLit(bits) => {
                Some(ConstInitVal::Float(norm_float(f64::from_bits(*bits), &expr.ty)))
            }
            MirExprKind::BoolLit(b) => Some(ConstInitVal::Bool(*b)),
            MirExprKind::ConstRef(name) => {
                let c = self.m.consts.iter().find(|c| c.name == *name)?;
                self.eval_const_val(&c.init, depth + 1)
            }
            MirExprKind::Unary { op, expr: inner } => {
                let sub = self.eval_const_val(inner, depth + 1)?;
                match (op, sub) {
                    (MirUnaryOp::Neg, ConstInitVal::Int(v)) => {
                        Some(ConstInitVal::Int(trunc_int(v.wrapping_neg(), &expr.ty)))
                    }
                    (MirUnaryOp::Neg, ConstInitVal::Float(f)) => {
                        Some(ConstInitVal::Float(norm_float(-f, &expr.ty)))
                    }
                    (MirUnaryOp::Not, ConstInitVal::Bool(b)) => Some(ConstInitVal::Bool(!b)),
                    (MirUnaryOp::BitNot, ConstInitVal::Int(v)) => {
                        Some(ConstInitVal::Int(trunc_int(!v, &expr.ty)))
                    }
                    _ => None,
                }
            }
            MirExprKind::Cast {
                expr: inner,
                target_ty,
            } => {
                let sub = self.eval_const_val(inner, depth + 1)?;
                let ht = target_ty.host_value_type();
                match sub {
                    ConstInitVal::Float(f) => match ht {
                        MirType::F32 | MirType::F64 => Some(ConstInitVal::Float(norm_float(f, &ht))),
                        MirType::U8 => Some(ConstInitVal::Int((f as u8) as i128)),
                        MirType::I8 => Some(ConstInitVal::Int((f as i8) as i128)),
                        MirType::U16 => Some(ConstInitVal::Int((f as u16) as i128)),
                        MirType::I16 => Some(ConstInitVal::Int((f as i16) as i128)),
                        MirType::U32 => Some(ConstInitVal::Int((f as u32) as i128)),
                        MirType::I32 => Some(ConstInitVal::Int((f as i32) as i128)),
                        MirType::U64 | MirType::Usize => {
                            Some(ConstInitVal::Int((f as u64) as i128))
                        }
                        MirType::I64 | MirType::Isize => {
                            Some(ConstInitVal::Int((f as i64) as i128))
                        }
                        _ => None,
                    },
                    ConstInitVal::Int(v) => match ht {
                        MirType::F32 | MirType::F64 => {
                            let fv = if inner.ty.is_unsigned() {
                                (v as u128) as f64
                            } else {
                                v as f64
                            };
                            Some(ConstInitVal::Float(norm_float(fv, &ht)))
                        }
                        MirType::Bool => Some(ConstInitVal::Bool(v != 0)),
                        MirType::U8
                        | MirType::I8
                        | MirType::U16
                        | MirType::I16
                        | MirType::U32
                        | MirType::I32
                        | MirType::U64
                        | MirType::I64
                        | MirType::Usize
                        | MirType::Isize => Some(ConstInitVal::Int(trunc_int(v, &ht))),
                        _ => None,
                    },
                    ConstInitVal::Bool(b) => match ht {
                        MirType::Bool => Some(ConstInitVal::Bool(b)),
                        MirType::U8
                        | MirType::I8
                        | MirType::U16
                        | MirType::I16
                        | MirType::U32
                        | MirType::I32
                        | MirType::U64
                        | MirType::I64
                        | MirType::Usize
                        | MirType::Isize => Some(ConstInitVal::Int(if b { 1 } else { 0 })),
                        _ => None,
                    },
                }
            }
            MirExprKind::Binary { op, lhs, rhs } => {
                let lv = self.eval_const_val(lhs, depth + 1)?;
                let rv = self.eval_const_val(rhs, depth + 1)?;
                match (lv, rv) {
                    (ConstInitVal::Float(a), ConstInitVal::Float(b)) => match op {
                        MirBinOp::Add => Some(ConstInitVal::Float(norm_float(a + b, &expr.ty))),
                        MirBinOp::Sub => Some(ConstInitVal::Float(norm_float(a - b, &expr.ty))),
                        MirBinOp::Mul => Some(ConstInitVal::Float(norm_float(a * b, &expr.ty))),
                        MirBinOp::Div => Some(ConstInitVal::Float(norm_float(a / b, &expr.ty))),
                        MirBinOp::Rem => Some(ConstInitVal::Float(norm_float(a % b, &expr.ty))),
                        MirBinOp::Eq => Some(ConstInitVal::Bool(a == b)),
                        MirBinOp::Ne => Some(ConstInitVal::Bool(a != b)),
                        MirBinOp::Lt => Some(ConstInitVal::Bool(a < b)),
                        MirBinOp::Le => Some(ConstInitVal::Bool(a <= b)),
                        MirBinOp::Gt => Some(ConstInitVal::Bool(a > b)),
                        MirBinOp::Ge => Some(ConstInitVal::Bool(a >= b)),
                        _ => None,
                    },
                    (ConstInitVal::Bool(a), ConstInitVal::Bool(b)) => match op {
                        MirBinOp::LogicalAnd => Some(ConstInitVal::Bool(a && b)),
                        MirBinOp::LogicalOr => Some(ConstInitVal::Bool(a || b)),
                        MirBinOp::Eq => Some(ConstInitVal::Bool(a == b)),
                        MirBinOp::Ne => Some(ConstInitVal::Bool(a != b)),
                        _ => None,
                    },
                    (ConstInitVal::Int(a), ConstInitVal::Int(b)) => {
                        let uns = lhs.ty.is_unsigned() && rhs.ty.is_unsigned();
                        match op {
                            MirBinOp::Add => {
                                Some(ConstInitVal::Int(trunc_int(a.wrapping_add(b), &expr.ty)))
                            }
                            MirBinOp::Sub => {
                                Some(ConstInitVal::Int(trunc_int(a.wrapping_sub(b), &expr.ty)))
                            }
                            MirBinOp::Mul => {
                                Some(ConstInitVal::Int(trunc_int(a.wrapping_mul(b), &expr.ty)))
                            }
                            MirBinOp::Div if b != 0 => {
                                let q = if uns {
                                    ((a as u128) / (b as u128)) as i128
                                } else {
                                    a.wrapping_div(b)
                                };
                                Some(ConstInitVal::Int(trunc_int(q, &expr.ty)))
                            }
                            MirBinOp::Rem if b != 0 => {
                                let r = if uns {
                                    ((a as u128) % (b as u128)) as i128
                                } else {
                                    a.wrapping_rem(b)
                                };
                                Some(ConstInitVal::Int(trunc_int(r, &expr.ty)))
                            }
                            MirBinOp::BitOr => {
                                Some(ConstInitVal::Int(trunc_int(a | b, &expr.ty)))
                            }
                            MirBinOp::BitAnd => {
                                Some(ConstInitVal::Int(trunc_int(a & b, &expr.ty)))
                            }
                            MirBinOp::BitXor => {
                                Some(ConstInitVal::Int(trunc_int(a ^ b, &expr.ty)))
                            }
                            MirBinOp::Shl => Some(ConstInitVal::Int(trunc_int(
                                a.wrapping_shl(b as u32),
                                &expr.ty,
                            ))),
                            MirBinOp::Shr => {
                                let s = if uns {
                                    ((a as u128).wrapping_shr(b as u32)) as i128
                                } else {
                                    a.wrapping_shr(b as u32)
                                };
                                Some(ConstInitVal::Int(trunc_int(s, &expr.ty)))
                            }
                            MirBinOp::Eq => Some(ConstInitVal::Bool(a == b)),
                            MirBinOp::Ne => Some(ConstInitVal::Bool(a != b)),
                            MirBinOp::Lt => Some(ConstInitVal::Bool(if uns {
                                (a as u128) < (b as u128)
                            } else {
                                a < b
                            })),
                            MirBinOp::Le => Some(ConstInitVal::Bool(if uns {
                                (a as u128) <= (b as u128)
                            } else {
                                a <= b
                            })),
                            MirBinOp::Gt => Some(ConstInitVal::Bool(if uns {
                                (a as u128) > (b as u128)
                            } else {
                                a > b
                            })),
                            MirBinOp::Ge => Some(ConstInitVal::Bool(if uns {
                                (a as u128) >= (b as u128)
                            } else {
                                a >= b
                            })),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 10: Sequential Boot Initializer & `int main(void)`
    // ─────────────────────────────────────────────────────────────────────────

    fn emit_boot_and_main(&mut self) {
        let _ = &self.fn_param_names;
        let _ = &self.task_param_names;

        self.out
            .push_str("/* ── Phased Sequential Boot & Cooperative Kernel Entry ──────────────────── */\n\n");

        self.out.push_str("void syn_boot_init(void) {\n");
        let n_tasks = self.m.spawned_tasks.len();
        if n_tasks > 0 {
            let _ = writeln!(
                self.out,
                "    g_syn_sched.tasks = g_syn_tasks;\n    g_syn_sched.task_count = {n_tasks}U;"
            );
            for sp in &self.m.spawned_tasks {
                let core_c = sp.core_pin.map_or(0xFFu8, |c| c);
                let _ = writeln!(
                    self.out,
                    "    g_syn_tasks[{id}] = (syn_task_t){{ .step = syn_task_step_{bp}, .frame = &g_syn_frame_slot_{id}, .name = \"{name}\", .wait_check = NULL, .wait_obj = NULL, .delay_until = 0U, .due_ms = 0U, .id = {id}U, .priority = {prio}, .base_priority = {prio}, .boost_priority = SYN_NO_BOOST, .core_pin = {core_c}U, .state = SYN_TASK_INACTIVE, .sched_flags = 0U, .has_deadline = false }};",
                    id = sp.slot_id,
                    bp = sp.blueprint_name,
                    name = sp.instance_name,
                    prio = sp.priority.c_const_name(),
                );
            }
        }

        self.emit_stack_local_decls(&self.m.boot.vars, false);

        // Dummy boot task descriptor for driving sequential boot-time yielding self-tests
        if !self.m.boot.boot_yielding_callees.is_empty() {
            self.out.push_str(
                "    syn_task_t _boot_task = { .step = NULL, .frame = NULL, .name = \"boot\", .priority = SYN_PRIO_CRITICAL, .base_priority = SYN_PRIO_CRITICAL, .boost_priority = SYN_NO_BOOST, .core_pin = 0xFFU, .state = SYN_TASK_READY, .sched_flags = SYN_FLAG_MORE };\n",
            );
        }

        for step in &self.m.boot.steps {
            match step {
                MirBootStep::Stmt(s) => {
                    self.emit_direct_stmt(
                        s,
                        &self.m.boot.vars,
                        &MirType::Void,
                        &MirErrorSpec::None,
                        1,
                        false,
                    );
                }
                MirBootStep::RunYieldingFn { callee, args, dst } => {
                    let mut init_fields = vec!["._pc = 0U".to_string()];
                    if let Some(cf) = self.m.functions.iter().find(|f| f.name == *callee) {
                        for (idx, &pvid) in cf.params.iter().enumerate() {
                            if let Some(ae) = args.get(idx) {
                                init_fields.push(format!(
                                    ".{} = {}",
                                    frame_var_name(&cf.vars[pvid.0]),
                                    self.fmt_expr(ae, &self.m.boot.vars)
                                ));
                            }
                        }
                    }
                    let _ = writeln!(
                        self.out,
                        "    g_syn_boot_frame_{callee} = (syn_frame_fn_{callee}_t){{ {} }};\n    while (!g_syn_sim_stopped && syn_coro_{callee}(&_boot_task, &g_syn_boot_frame_{callee}) != SYN_STEP_DONE) {{\n        if (_boot_task.has_deadline && (int32_t)(syn_now_ms() - _boot_task.delay_until) < 0) {{\n            g_syn_tick_ms = _boot_task.delay_until;\n            _boot_task.has_deadline = false;\n        }}\n    }}\n    if (g_syn_sim_stopped) {{ return; }}",
                        init_fields.join(", ")
                    );
                    if let Some(d) = dst {
                        match d {
                            MirCallDst::Value(v) | MirCallDst::Result(v) => {
                                if !self.m.boot.vars[v.0].ty.is_void() {
                                    let _ = writeln!(
                                        self.out,
                                        "    {} = g_syn_boot_frame_{callee}._ret;",
                                        self.var_ref(*v, &self.m.boot.vars)
                                    );
                                }
                            }
                        }
                    }
                }
                MirBootStep::SpawnActivate { slot_id, args } => {
                    self.emit_spawn_activation(*slot_id, args, &self.m.boot.vars, "    ");
                }
            }
        }
        self.out.push_str("}\n\n");

        if self.opts.emit_main {
            self.out.push_str(
                r#"#if !defined(SYN_NO_MAIN)
int main(void) {
    syn_boot_init();
    if (!g_syn_sim_stopped && !g_syn_sim_manual_driven && g_syn_sched.task_count > 0U) {
        syn_sched_run_tickless_loop();
    }
    return g_syn_sim_exit_code;
}
#endif
"#,
            );
        }
    }

    fn emit_spawn_activation(
        &mut self,
        slot_id: usize,
        args: &[MirExpr],
        vars: &[MirVar],
        pad: &str,
    ) {
        let sp = &self.m.spawned_tasks[slot_id];
        let mut init_fields = vec!["._pc = 0U".to_string()];
        if let Some(bp) = self
            .m
            .task_blueprints
            .iter()
            .find(|b| b.name == sp.blueprint_name)
        {
            for (idx, &pvid) in bp.params.iter().enumerate() {
                if let Some(ae) = args.get(idx) {
                    init_fields.push(format!(
                        ".{} = {}",
                        frame_var_name(&bp.vars[pvid.0]),
                        self.fmt_expr(ae, vars)
                    ));
                }
            }
        }
        let _ = writeln!(
            self.out,
            "{pad}g_syn_frame_slot_{slot_id} = (syn_frame_task_{bp}_t){{ {fields} }};\n{pad}g_syn_tasks[{slot_id}].state = SYN_TASK_READY;\n{pad}g_syn_tasks[{slot_id}].sched_flags = SYN_FLAG_MORE;",
            bp = sp.blueprint_name,
            fields = init_fields.join(", ")
        );
    }

    fn emit_fmt_chunks(
        &mut self,
        sink_var: &str,
        chunks: &[MirFmtChunk],
        vars: &[MirVar],
        indent: usize,
    ) {
        let pad = "    ".repeat(indent);
        for ch in chunks {
            match ch {
                MirFmtChunk::Literal(s) => {
                    let byte_len = zev_mir::decoded_string_byte_len(s);
                    if byte_len > 0 {
                        let _ = writeln!(
                            self.out,
                            "{pad}syn_fmt_write(&{sink_var}, {}, {byte_len}U);",
                            escape_c_string(s)
                        );
                    }
                }
                MirFmtChunk::Int {
                    value,
                    signed,
                    bits,
                    fmt,
                } => {
                    let v = self.fmt_expr(value, vars);
                    match fmt {
                        MirIntFmt::Dec => {
                            if *signed {
                                let _ = writeln!(
                                    self.out,
                                    "{pad}syn_fmt_i64_dec(&{sink_var}, (int64_t)({v}));"
                                );
                            } else {
                                let _ = writeln!(
                                    self.out,
                                    "{pad}syn_fmt_u64_dec(&{sink_var}, (uint64_t)({v}));"
                                );
                            }
                        }
                        MirIntFmt::Hex {
                            upper,
                            zero_pad_width,
                        } => {
                            let ucast = match *bits {
                                8 => format!("(uint64_t)(uint8_t)({v})"),
                                16 => format!("(uint64_t)(uint16_t)({v})"),
                                32 => format!("(uint64_t)(uint32_t)({v})"),
                                64 => format!("(uint64_t)({v})"),
                                _ => format!("(uint64_t)(size_t)({v})"),
                            };
                            let _ = writeln!(
                                self.out,
                                "{pad}syn_fmt_u64_hex(&{sink_var}, {ucast}, {upper}, {zero_pad_width}U);"
                            );
                        }
                        MirIntFmt::Bin { zero_pad_width } => {
                            let ucast = match *bits {
                                8 => format!("(uint64_t)(uint8_t)({v})"),
                                16 => format!("(uint64_t)(uint16_t)({v})"),
                                32 => format!("(uint64_t)(uint32_t)({v})"),
                                64 => format!("(uint64_t)({v})"),
                                _ => format!("(uint64_t)(size_t)({v})"),
                            };
                            let _ = writeln!(
                                self.out,
                                "{pad}syn_fmt_u64_bin(&{sink_var}, {ucast}, {zero_pad_width}U);"
                            );
                        }
                    }
                }
                MirFmtChunk::Float { value, fmt, .. } => {
                    let v = self.fmt_expr(value, vars);
                    match fmt {
                        MirFloatFmt::Fixed { precision } => {
                            let _ = writeln!(
                                self.out,
                                "{pad}syn_fmt_float_fixed(&{sink_var}, (double)({v}), {precision}U);"
                            );
                        }
                        MirFloatFmt::Sci { precision } => {
                            let _ = writeln!(
                                self.out,
                                "{pad}syn_fmt_float_sci(&{sink_var}, (double)({v}), {precision}U);"
                            );
                        }
                    }
                }
                MirFmtChunk::Bool(value) => {
                    let v = self.fmt_expr(value, vars);
                    let _ = writeln!(self.out, "{pad}syn_fmt_bool(&{sink_var}, (bool)({v}));");
                }
                MirFmtChunk::Str(value) => {
                    let v = self.fmt_expr(value, vars);
                    let _ = writeln!(
                        self.out,
                        "{pad}syn_fmt_cstr(&{sink_var}, (const char *)({v}));"
                    );
                }
                MirFmtChunk::Bytes(value) => {
                    let v = self.fmt_expr(value, vars);
                    let _ = writeln!(self.out, "{pad}syn_fmt_bytes(&{sink_var}, {v});");
                }
                MirFmtChunk::EnumName { enum_name, value } => {
                    let v = self.fmt_expr(value, vars);
                    let _ = writeln!(
                        self.out,
                        "{pad}syn_fmt_cstr(&{sink_var}, syn_enum_name_{enum_name}({v}));"
                    );
                }
                MirFmtChunk::DebugSlice { elem_ty, slice } => {
                    let sl_c = self.c_type(&slice.ty);
                    let sl_v = self.fmt_expr(slice, vars);
                    let raw_elem = match &slice.ty {
                        MirType::Slice(inner) => match inner.as_ref() {
                            MirType::EndianInt {
                                bits,
                                big_endian,
                                signed,
                            } => {
                                let end = if *big_endian { "be" } else { "le" };
                                if *signed {
                                    format!("((int{bits}_t)syn_load_u{bits}{end}(_dsl.ptr[_di]))")
                                } else {
                                    format!("syn_load_u{bits}{end}(_dsl.ptr[_di])")
                                }
                            }
                            _ => "_dsl.ptr[_di]".to_string(),
                        },
                        _ => "_dsl.ptr[_di]".to_string(),
                    };
                    let _ = writeln!(self.out, "{pad}{{");
                    let _ = writeln!(self.out, "{pad}    {sl_c} _dsl = {sl_v};");
                    let _ = writeln!(self.out, "{pad}    syn_fmt_putc(&{sink_var}, '[');");
                    let _ = writeln!(
                        self.out,
                        "{pad}    for (size_t _di = 0U; _di < _dsl.len; _di++) {{"
                    );
                    let _ = writeln!(
                        self.out,
                        "{pad}        if (_di > 0U) {{ syn_fmt_write(&{sink_var}, \", \", 2U); }}"
                    );
                    match elem_ty {
                        MirType::Bool => {
                            let _ = writeln!(
                                self.out,
                                "{pad}        syn_fmt_bool(&{sink_var}, (bool)({raw_elem}));"
                            );
                        }
                        MirType::F32 | MirType::F64 => {
                            let _ = writeln!(
                                self.out,
                                "{pad}        syn_fmt_float_fixed(&{sink_var}, (double)({raw_elem}), 2U);"
                            );
                        }
                        MirType::Enum(ename) => {
                            let _ = writeln!(
                                self.out,
                                "{pad}        syn_fmt_cstr(&{sink_var}, syn_enum_name_{ename}({raw_elem}));"
                            );
                        }
                        MirType::Str => {
                            let _ = writeln!(
                                self.out,
                                "{pad}        syn_fmt_cstr(&{sink_var}, (const char *)({raw_elem}));"
                            );
                        }
                        MirType::Ptr(_) | MirType::FnPtr { .. } => {
                            let _ = writeln!(
                                self.out,
                                "{pad}        syn_fmt_write(&{sink_var}, \"0x\", 2U);\n{pad}        syn_fmt_u64_hex(&{sink_var}, (uint64_t)(uintptr_t)({raw_elem}), false, 0U);"
                            );
                        }
                        MirType::Slice(inner) if **inner == MirType::U8 => {
                            let _ = writeln!(
                                self.out,
                                "{pad}        syn_fmt_bytes(&{sink_var}, {raw_elem});"
                            );
                        }
                        MirType::I8
                        | MirType::I16
                        | MirType::I32
                        | MirType::I64
                        | MirType::Isize => {
                            let _ = writeln!(
                                self.out,
                                "{pad}        syn_fmt_i64_dec(&{sink_var}, (int64_t)({raw_elem}));"
                            );
                        }
                        _ => {
                            let _ = writeln!(
                                self.out,
                                "{pad}        syn_fmt_u64_dec(&{sink_var}, (uint64_t)({raw_elem}));"
                            );
                        }
                    }
                    let _ = writeln!(self.out, "{pad}    }}");
                    let _ = writeln!(self.out, "{pad}    syn_fmt_putc(&{sink_var}, ']');");
                    let _ = writeln!(self.out, "{pad}}}");
                }
            }
        }
    }
}

fn format_c99_float_lit(bits: u64, is_f32: bool) -> String {
    if is_f32 {
        let f32_bits = (f64::from_bits(bits) as f32).to_bits();
        let neg = (f32_bits >> 31) != 0;
        let sign = if neg { "-" } else { "" };
        let exp_bits = ((f32_bits >> 23) & 0xFF) as i32;
        let mant = f32_bits & 0x007F_FFFFu32;
        if exp_bits == 0xFF {
            if mant == 0 {
                format!("({sign}__builtin_inff())")
            } else {
                format!("({sign}__builtin_nanf(\"\"))")
            }
        } else if exp_bits == 0 {
            if mant == 0 {
                format!("({sign}0x0.0p+0f)")
            } else {
                format!("({sign}0x0.{:06x}p-126f)", mant << 1)
            }
        } else {
            let exp = exp_bits - 127;
            format!("({sign}0x1.{:06x}p{exp:+}f)", mant << 1)
        }
    } else {
        let neg = (bits >> 63) != 0;
        let sign = if neg { "-" } else { "" };
        let exp_bits = ((bits >> 52) & 0x7FF) as i32;
        let mant = bits & 0x000F_FFFF_FFFF_FFFFu64;
        if exp_bits == 0x7FF {
            if mant == 0 {
                format!("({sign}__builtin_inf())")
            } else {
                format!("({sign}__builtin_nan(\"\"))")
            }
        } else if exp_bits == 0 {
            if mant == 0 {
                format!("({sign}0x0.0p+0)")
            } else {
                format!("({sign}0x0.{mant:013x}p-1022)")
            }
        } else {
            let exp = exp_bits - 1023;
            format!("({sign}0x1.{mant:013x}p{exp:+})")
        }
    }
}

fn direct_stmts_have_unnested_break(stmts: &[MirStmt]) -> bool {
    stmts.iter().any(|s| match s {
        MirStmt::Break => true,
        MirStmt::If {
            then_body,
            else_body,
            ..
        } => {
            direct_stmts_have_unnested_break(then_body)
                || direct_stmts_have_unnested_break(else_body)
        }
        MirStmt::Switch { arms, default, .. } => {
            arms.iter().any(|(_, b)| direct_stmts_have_unnested_break(b))
                || direct_stmts_have_unnested_break(default)
        }
        MirStmt::While { .. } => false,
        _ => false,
    })
}

fn compute_reachable_blocks(blocks: &[MirBlock]) -> BTreeSet<BlockId> {
    let map: BTreeMap<BlockId, &MirBlock> = blocks.iter().map(|b| (b.id, b)).collect();
    let mut visited = BTreeSet::new();
    let mut q = VecDeque::new();
    if let Some(first) = blocks.first() {
        visited.insert(first.id);
        q.push_back(first.id);
    }
    while let Some(bid) = q.pop_front() {
        if let Some(blk) = map.get(&bid) {
            for succ in blk.terminator.successors() {
                if visited.insert(succ) {
                    q.push_back(succ);
                }
            }
        }
    }
    visited
}

fn c_ident(name: &str) -> String {
    match name {
        "auto" | "register" | "volatile" | "extern" | "static" | "inline" | "restrict"
        | "default" | "case" | "union" | "typedef" | "sizeof" | "goto" | "signed" | "unsigned"
        | "short" | "long" | "float" | "double" | "char" | "int" => format!("syn_id_{name}"),
        _ => name.to_string(),
    }
}

fn frame_var_name(v: &MirVar) -> String {
    format!("v{}_{}", v.id.0, sanitize_ident(&v.name))
}

fn stack_var_name(v: &MirVar) -> String {
    format!("_lv{}_{}", v.id.0, sanitize_ident(&v.name))
}

fn sanitize_ident(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

fn mmio_c_macro_name(name: &str) -> String {
    format!("ZEV_MMIO_{}", sanitize_ident(name))
}

fn assign_op_str(op: MirAssignOp) -> &'static str {
    match op {
        MirAssignOp::Assign => "=",
        MirAssignOp::AddAssign => "+=",
        MirAssignOp::SubAssign => "-=",
        MirAssignOp::MulAssign => "*=",
        MirAssignOp::DivAssign => "/=",
        MirAssignOp::RemAssign => "%=",
        MirAssignOp::BitOrAssign => "|=",
        MirAssignOp::BitAndAssign => "&=",
        MirAssignOp::BitXorAssign => "^=",
        MirAssignOp::ShlAssign => "<<=",
        MirAssignOp::ShrAssign => ">>=",
    }
}

fn assign_op_bin_str(op: MirAssignOp) -> &'static str {
    match op {
        MirAssignOp::AddAssign => "+",
        MirAssignOp::SubAssign => "-",
        MirAssignOp::MulAssign => "*",
        MirAssignOp::DivAssign => "/",
        MirAssignOp::RemAssign => "%",
        MirAssignOp::BitOrAssign => "|",
        MirAssignOp::BitAndAssign => "&",
        MirAssignOp::BitXorAssign => "^",
        MirAssignOp::ShlAssign => "<<",
        MirAssignOp::ShrAssign => ">>",
        MirAssignOp::Assign => "+",
    }
}

fn bin_op_str(op: MirBinOp) -> &'static str {
    match op {
        MirBinOp::Add => "+",
        MirBinOp::Sub => "-",
        MirBinOp::Mul => "*",
        MirBinOp::Div => "/",
        MirBinOp::Rem => "%",
        MirBinOp::BitOr => "|",
        MirBinOp::BitAnd => "&",
        MirBinOp::BitXor => "^",
        MirBinOp::Shl => "<<",
        MirBinOp::Shr => ">>",
        MirBinOp::Eq => "==",
        MirBinOp::Ne => "!=",
        MirBinOp::Lt => "<",
        MirBinOp::Le => "<=",
        MirBinOp::Gt => ">",
        MirBinOp::Ge => ">=",
        MirBinOp::LogicalAnd => "&&",
        MirBinOp::LogicalOr => "||",
    }
}

fn strip_outer_parens(s: &str) -> &str {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'(' && bytes[bytes.len() - 1] == b')' {
        let mut depth = 0i32;
        let mut wraps_all = true;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'(' {
                depth += 1;
            } else if b == b')' {
                depth -= 1;
                if depth == 0 && i + 1 < bytes.len() {
                    wraps_all = false;
                    break;
                }
            }
        }
        if wraps_all && depth == 0 {
            return &s[1..s.len() - 1];
        }
    }
    s
}

#[derive(Clone, Copy)]
enum ConstInitVal {
    Int(i128),
    Float(f64),
    Bool(bool),
}

fn escape_c_string(s: &str) -> String {
    let bytes = zev_mir::decoded_string_bytes(s);
    let mut out = String::with_capacity(bytes.len() + 2);
    out.push('"');
    for b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7E => out.push(b as char),
            other => {
                let _ = write!(out, "\\{:03o}", other);
            }
        }
    }
    out.push('"');
    out
}
