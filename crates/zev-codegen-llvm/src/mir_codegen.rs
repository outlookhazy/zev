//! Direct `zev-mir` -> LLVM 23 Opaque-Pointer SSA IR Emitter (`mir_codegen`).
//!
//! Lowers a [`zev_mir::MirModule`] directly into textual LLVM 23 IR (`.ll`) with:
//! - True aggregate dual-exit register returns (`{ T, i32 }` -> `r0, r1` on ARM Cortex-M3)
//! - Wire-exact `packed struct` (`<{ ... }>`) + `llvm.bswap.*` big/little-endian field access
//! - Stackless coroutine state machines (`switch i32 %pc, ...`) over static `%TaskFrame_*` /
//!   `%FnFrame_*` structs with `[K x i64]` byte-aligned union storage for disjoint child frames
//! - Scheduler-integrated `watch`, `inbox` (with `timeout` + `or` + ISR `boost`), `chan`,
//!   `signal`, `ping_pong`, `@dma_safe pool`/`Lease`/`dma_slot` (with DMA wake-lock sleep clamping) primitives
//! - Synthesized bare-metal Cortex-M3 `.isr_vector` + user hardware `isr` handlers.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;

use zev_mir::{
    BlockId, MirAssignOp, MirBakedArg, MirBinOp, MirBlock, MirBootStep, MirCallDst,
    MirChildCallSlot, MirErrorSpec, MirExpr, MirExprKind, MirFloatFmt, MirFmtChunk, MirFnBody,
    MirFunction, MirIntFmt, MirIntrinsicOp, MirIrqBinding, MirIrqVector, MirIsr,
    MirMmioInstance, MirMmioLayout, MirModule, MirPlace, MirPrimKind, MirPrimNonBlockingOp,
    MirPrimReadOp, MirPrimTarget, MirPrimViewKind, MirStmt, MirStruct, MirSwitchPattern,
    MirSysQuery, MirTaskBlueprint, MirTerminator, MirType, MirUnaryOp, MirVar, MirYieldKind, VarId,
    VarRole,
};

use crate::arm_cortex_m::{NvicMap, build_nvic_map, emit_arm_cortex_m_startup_ir, sanitize_ident};
use crate::riscv32::{RvIrqRoute, assign_rv_irq_routes_for_spec, emit_riscv32_startup_ir};
use crate::runtime_ir::{RuntimeIrFeatures, emit_kernel_runtime_ir_with_features};
use crate::target_spec::{CpuIsa, IrqModel, TargetSpec};
use crate::toolchain::{CodegenError, CodegenOptions};

// ─────────────────────────────────────────────────────────────────────────────
// Module-wide Lowering Metadata
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct StructInfo {
    decl: MirStruct,
    llvm_name: String,
    byte_size: usize,
    align: usize,
    llvm_abi_size: usize,
    llvm_nat_align: usize,
    field_indices: HashMap<String, (usize, MirType)>,
    llvm_fields: Vec<(Option<usize>, String)>,
}

#[derive(Clone)]
struct CoroutineFrameLayout {
    llvm_name: String,
    has_ret_slot: bool,
    ret_llvm_ty: String,
    /// Map from `VarId` -> GEP field index inside `%TaskFrame_*` / `%FnFrame_*`.
    hoisted_var_gep: HashMap<VarId, usize>,
    /// Map from `union_group` -> `(gep_field_idx, num_i64_words)`.
    union_group_gep: BTreeMap<usize, (usize, usize)>,
    /// Map from `slot_name` -> `union_group`.
    slot_to_group: HashMap<String, usize>,
    /// Approximate byte size (8-byte aligned conservative upper bound for parent union sizing).
    byte_size_bound: usize,
    /// Ordered field LLVM types for `type { ... }`.
    field_llvm_tys: Vec<String>,
}

#[derive(Clone, Debug)]
struct SsaVal {
    ty: String,
    repr: String,
}

impl SsaVal {
    fn new(ty: impl Into<String>, repr: impl Into<String>) -> Self {
        Self {
            ty: ty.into(),
            repr: repr.into(),
        }
    }
}

struct ModuleCtx<'a> {
    module: &'a MirModule,
    opts: &'a CodegenOptions,
    spec: TargetSpec,
    nvic: NvicMap,
    rv_irq_routes: Vec<RvIrqRoute>,
    avr_irq_vectors: BTreeSet<u32>,
    mmio_layouts: HashMap<String, MirMmioLayout>,
    mmio_instances: HashMap<String, MirMmioInstance>,
    structs: HashMap<String, StructInfo>,
    struct_order: Vec<String>,
    enum_variants: HashMap<(String, String), i64>,
    bare_variants: HashMap<String, i64>,
    consts: HashMap<String, i128>,
    float_consts: HashMap<String, u64>,
    fn_sigs: HashMap<String, FnLlvmSig>,
    extern_sigs: HashMap<String, ExternLlvmSig>,
    fn_frames: HashMap<String, CoroutineFrameLayout>,
    bp_frames: HashMap<String, CoroutineFrameLayout>,
    prim_globals: HashMap<String, MirPrimKind>,
    distinct_prims: BTreeMap<String, MirPrimKind>,
    distinct_inbox_views: BTreeMap<String, MirType>,
    uses_byte_ring_view: bool,
    uses_pool_lease: bool,
    dma_layout_structs: BTreeSet<String>,
    vis: zev_mir::opt::VisibilityReport,
    rt_features: RuntimeIrFeatures,
    hooks: ResolvedSysHooks,
    rom_wrappers: BTreeSet<RomWrapperSpec>,
    strings: Vec<String>,
    string_dedup: HashMap<String, usize>,
    const_blobs: Vec<String>,
    const_blob_dedup: HashMap<String, usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum HookKind {
    Putc,
    ClockMs,
    Exit,
    FaultHandler,
    Idle,
    IrqRoute,
    IrqAck,
}

impl HookKind {
    const fn suffix(self) -> &'static str {
        match self {
            Self::Putc => "putc",
            Self::ClockMs => "clock_ms",
            Self::Exit => "exit",
            Self::FaultHandler => "fault",
            Self::Idle => "idle",
            Self::IrqRoute => "rv_irq_route",
            Self::IrqAck => "rv_irq_ack",
        }
    }

    const fn wrapper_sym(self) -> &'static str {
        match self {
            Self::Putc => "@__syn_on_putc",
            Self::ClockMs => "@__syn_now_ms",
            Self::Exit => "@__syn_on_exit",
            Self::FaultHandler => "@__syn_on_fault",
            Self::Idle => "@__syn_on_idle",
            Self::IrqRoute => "@__syn_on_rv_irq_route",
            Self::IrqAck => "@__syn_on_rv_irq_ack",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum StaticHookMode {
    Unset,
    DirectUnconditional {
        fn_name: String,
        symbol: String,
    },
    TagSwitch {
        arms: Vec<(String, String)>,
        global_sym: String,
    },
    Indirect {
        global_sym: String,
    },
}

impl StaticHookMode {
    fn tag_index_for(&self, fn_name: &str) -> Option<usize> {
        if let Self::TagSwitch { arms, .. } = self {
            arms.iter()
                .position(|(name, _)| name == fn_name)
                .map(|idx| idx + 1)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct ResolvedSysHooks {
    putc: StaticHookMode,
    clock_ms: StaticHookMode,
    exit: StaticHookMode,
    fault_handler: StaticHookMode,
    idle: StaticHookMode,
    irq_route: StaticHookMode,
    irq_ack: StaticHookMode,
}

impl Default for ResolvedSysHooks {
    fn default() -> Self {
        Self {
            putc: StaticHookMode::Unset,
            clock_ms: StaticHookMode::Unset,
            exit: StaticHookMode::Unset,
            fault_handler: StaticHookMode::Unset,
            idle: StaticHookMode::Unset,
            irq_route: StaticHookMode::Unset,
            irq_ack: StaticHookMode::Unset,
        }
    }
}

impl ResolvedSysHooks {
    fn mode(&self, kind: HookKind) -> &StaticHookMode {
        match kind {
            HookKind::Putc => &self.putc,
            HookKind::ClockMs => &self.clock_ms,
            HookKind::Exit => &self.exit,
            HookKind::FaultHandler => &self.fault_handler,
            HookKind::Idle => &self.idle,
            HookKind::IrqRoute => &self.irq_route,
            HookKind::IrqAck => &self.irq_ack,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RomWrapperKind {
    U32,
    Void,
    Noreturn,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RomWrapperSpec {
    kind: RomWrapperKind,
    stack_bound: u32,
    param_llvm_tys: Vec<String>,
}

impl RomWrapperSpec {
    fn symbol(&self) -> String {
        let mut sig_suffix = String::new();
        for ty in &self.param_llvm_tys {
            sig_suffix.push('_');
            sig_suffix.push_str(&sanitize_ident(ty));
        }
        let n = self.param_llvm_tys.len();
        match self.kind {
            RomWrapperKind::U32 => {
                format!("@__syn_rom_call_u32_a{n}{sig_suffix}_sb{}", self.stack_bound)
            }
            RomWrapperKind::Void => {
                format!("@__syn_rom_call_void_a{n}{sig_suffix}_sb{}", self.stack_bound)
            }
            RomWrapperKind::Noreturn => {
                format!("@__syn_rom_noreturn_a{n}{sig_suffix}")
            }
        }
    }
}

#[derive(Clone)]
struct FnLlvmSig {
    symbol: String,
    params: Vec<MirType>,
    ret_llvm_ty: String,
}

#[derive(Clone)]
struct ExternLlvmSig {
    name: String,
    params: Vec<MirType>,
    ret_ty: MirType,
    variadic: bool,
}

impl<'a> ModuleCtx<'a> {
    const fn fn_as(&self) -> &'static str {
        self.spec.isa.fn_addrspace_attr()
    }

    const fn call_kw(&self) -> &'static str {
        self.spec.isa.call_kw()
    }
}

const fn align_up(offset: usize, align: usize) -> usize {
    if align <= 1 {
        offset
    } else {
        (offset + align - 1) & !(align - 1)
    }
}

/// Emit complete LLVM 23 `.ll` IR for `module` under `opts`.
pub fn emit_llvm_ir_with_options(
    module: &MirModule,
    opts: &CodegenOptions,
) -> Result<String, CodegenError> {
    let spec = opts.effective_spec()?;
    let nvic = if spec.isa == CpuIsa::ArmCortexM {
        build_nvic_map(&module.irq_vectors, &spec)?
    } else {
        NvicMap::default()
    };
    let rv_irq_routes = if spec.isa == CpuIsa::Riscv32 {
        assign_rv_irq_routes_for_spec(&module.irq_vectors, &spec)?
    } else {
        Vec::new()
    };
    let avr_irq_vectors = match &spec.irq {
        IrqModel::AvrVectored { vector_slots, .. } if spec.isa == CpuIsa::Avr => {
            crate::avr::validate_avr_irq_vectors(&module.irq_vectors, *vector_slots)?
        }
        _ => BTreeSet::new(),
    };
    let mut rt_features = scan_module_runtime_features(module);
    rt_features.uses_rv_irqs = spec.isa == CpuIsa::Riscv32 && !module.irq_vectors.is_empty();
    let mut ctx = ModuleCtx {
        module,
        opts,
        spec,
        nvic,
        rv_irq_routes,
        avr_irq_vectors,
        mmio_layouts: HashMap::new(),
        mmio_instances: HashMap::new(),
        structs: HashMap::new(),
        struct_order: Vec::new(),
        enum_variants: HashMap::new(),
        bare_variants: HashMap::new(),
        consts: HashMap::new(),
        float_consts: HashMap::new(),
        fn_sigs: HashMap::new(),
        extern_sigs: HashMap::new(),
        fn_frames: HashMap::new(),
        bp_frames: HashMap::new(),
        prim_globals: HashMap::new(),
        distinct_prims: BTreeMap::new(),
        distinct_inbox_views: BTreeMap::new(),
        uses_byte_ring_view: false,
        uses_pool_lease: false,
        dma_layout_structs: zev_mir::collect_dma_layout_structs(module),
        vis: zev_mir::opt::VisibilityReport::effective_for(module),
        rt_features,
        hooks: ResolvedSysHooks::default(),
        rom_wrappers: BTreeSet::new(),
        strings: Vec::new(),
        string_dedup: HashMap::new(),
        const_blobs: Vec::new(),
        const_blob_dedup: HashMap::new(),
    };

    ctx.prepare()?;
    ctx.hooks = ctx.resolve_sys_hooks();
    ctx.rt_features.uses_clock_hook = !matches!(ctx.hooks.clock_ms, StaticHookMode::Unset);

    let mut out = String::with_capacity(64 * 1024);

    // 1. Target header + cooperative OS kernel & HAL (emitting soft-float & fmt helpers on demand)
    emit_kernel_runtime_ir_with_features(&mut out, module.spawned_tasks.len(), opts, &ctx.rt_features)?;

    // 2. Pure-ISA vector tables, startup reset routines, and software/boot IRQ dispatchers
    match ctx.spec.isa {
        CpuIsa::ArmCortexM => {
            emit_arm_cortex_m_startup_ir(&mut out, &ctx.spec, opts.fn_size_attrs(), &ctx.nvic);
            emit_cm_irq_dispatch_ir(&mut out, &ctx);
        }
        CpuIsa::Riscv32 => {
            emit_riscv32_startup_ir(&mut out, &ctx.spec, opts.fn_size_attrs(), &ctx.rv_irq_routes);
            emit_rv_irq_dispatch_ir(&mut out, &ctx);
        }
        CpuIsa::Avr => {
            crate::avr::emit_avr_startup_ir(
                &mut out,
                &ctx.spec,
                opts.fn_size_attrs(),
                &ctx.avr_irq_vectors,
            );
            emit_avr_irq_dispatch_ir(&mut out, &ctx);
        }
        CpuIsa::HostX86_64Linux => {
            emit_host_entry_ir(&mut out, &ctx);
        }
    }

    // 3. User struct type declarations + Host MMIO backing buffers + ISR / hook static state globals
    ctx.emit_struct_types_ir(&mut out);
    ctx.emit_mmio_host_buffers_ir(&mut out);
    ctx.emit_isr_state_and_selector_globals_ir(&mut out);
    ctx.emit_sys_hook_globals_ir(&mut out);

    // 4. OS primitive type declarations, global instances, and typed helper functions
    ctx.emit_primitives_ir(&mut out);

    // 5. Coroutine frame struct types + static task/boot frame globals
    ctx.emit_coroutine_frame_globals_ir(&mut out);

    // 6. Struct `FromBytes` helpers + on-demand `formatted_enums` `.rodata` lookup helpers
    ctx.emit_from_bytes_helpers_ir(&mut out);
    ctx.emit_formatted_enum_names_ir(&mut out);

    // 7. `extern "C"` declarations + synthesized bare-metal `@syn_*` intrinsics
    ctx.emit_extern_fns_ir(&mut out);

    // 8. User hardware ISRs (`@__syn_isr_<NAME>`) + Flash IRQ vector dispatchers (`@__syn_vec_irq_<N>`)
    for isr in &module.isrs {
        ctx.emit_isr_ir(&mut out, isr)?;
    }
    ctx.emit_irq_vector_dispatchers_ir(&mut out)?;

    // 9. User functions (both Direct dual-exit register fns and Coroutine state-machine fns)
    for func in &module.functions {
        ctx.emit_function_ir(&mut out, func)?;
    }

    // 10. Task blueprints (`@__syn_task_step_<NAME>`)
    for bp in &module.task_blueprints {
        ctx.emit_task_blueprint_ir(&mut out, bp)?;
    }

    // 11. Sequential `main()` boot sequence (`@__syn_boot`)
    ctx.emit_boot_ir(&mut out)?;

    // 12. Explicit `sys.*` hook wrappers and stack-bounded ROM call wrappers
    ctx.emit_sys_hooks_ir(&mut out);
    ctx.emit_rom_wrappers_ir(&mut out);

    // 13. Pooled string literal constants (`@.str.<k>`)
    ctx.emit_string_constants_ir(&mut out);

    Ok(out)
}

fn emit_sw_irq_dispatch_ir(
    out: &mut String,
    fn_name: &str,
    comment: &str,
    ctx: &ModuleCtx<'_>,
) {
    let fn_as = ctx.fn_as();
    let call_kw = ctx.call_kw();
    let _ = writeln!(out, "{comment}");
    let _ = writeln!(out, "define void {fn_name}(i32 %irq){fn_as} nounwind {{");
    let _ = writeln!(out, "entry:");
    let mut irqs = BTreeSet::new();
    for vec in &ctx.module.irq_vectors {
        irqs.insert(vec.irq_num);
    }
    let _ = writeln!(out, "  switch i32 %irq, label %done [");
    for &irq in &irqs {
        let _ = writeln!(out, "    i32 {irq}, label %irq_{irq}");
    }
    let _ = writeln!(out, "  ]");
    for &irq in &irqs {
        let _ = writeln!(out, "irq_{irq}:");
        let _ = writeln!(out, "  {call_kw} void @__syn_vec_irq_{irq}()");
        let _ = writeln!(out, "  br label %done");
    }
    let _ = writeln!(out, "done:");
    let _ = writeln!(out, "  ret void");
    let _ = writeln!(out, "}}\n");
}

fn emit_cm_irq_dispatch_ir(out: &mut String, ctx: &ModuleCtx<'_>) {
    emit_sw_irq_dispatch_ir(
        out,
        "@__syn_cm_dispatch_irq",
        "; ── ARM Cortex-M Software/Boot IRQ Dispatcher ──",
        ctx,
    );
}

fn emit_rv_irq_dispatch_ir(out: &mut String, ctx: &ModuleCtx<'_>) {
    emit_sw_irq_dispatch_ir(
        out,
        "@__syn_rv_dispatch_irq",
        "; ── RISC-V Software/Boot IRQ Dispatcher ──",
        ctx,
    );
}

fn emit_avr_irq_dispatch_ir(out: &mut String, ctx: &ModuleCtx<'_>) {
    emit_sw_irq_dispatch_ir(
        out,
        "@__syn_avr_dispatch_irq",
        "; ── AVR Software/Vector IRQ Dispatcher ──",
        ctx,
    );
}

fn emit_host_entry_ir(out: &mut String, ctx: &ModuleCtx<'_>) {
    emit_sw_irq_dispatch_ir(
        out,
        "@__syn_host_dispatch_irq",
        "; ── Host Simulation IRQ Dispatcher & Entry Point ──",
        ctx,
    );

    let _ = writeln!(
        out,
        r#"define i32 @main() nounwind {{
entry:
  call void @__syn_boot()
  call void @__syn_sched_loop()
  ret i32 0
}}
"#
    );
}

fn scan_module_runtime_features(module: &MirModule) -> RuntimeIrFeatures {
    let mut feat = RuntimeIrFeatures {
        uses_i64: false,
        uses_floats: false,
        uses_fmt: false,
        uses_fmt_float_sci: false,
        uses_rv_irqs: false,
        uses_sys_tick: false,
        uses_clock_hook: false,
    };

    fn scan_ty(ty: &MirType, feat: &mut RuntimeIrFeatures) {
        match ty {
            MirType::F32 | MirType::F64 => feat.uses_floats = true,
            MirType::U64 | MirType::I64 | MirType::EndianInt { bits: 64, .. } => {
                feat.uses_i64 = true;
            }
            MirType::Ptr(inner) | MirType::Slice(inner) => scan_ty(inner, feat),
            MirType::FnPtr { params, ret } => {
                for p in params {
                    scan_ty(p, feat);
                }
                scan_ty(ret, feat);
            }
            MirType::Array { elem, .. } => scan_ty(elem, feat),
            MirType::Result { ok, err } => {
                scan_ty(ok, feat);
                if let Some(e) = err {
                    scan_ty(e, feat);
                }
            }
            MirType::Primitive(pk) => match pk {
                MirPrimKind::Signal | MirPrimKind::ByteRing { .. } => {}
                MirPrimKind::Watch { elem }
                | MirPrimKind::Inbox { elem, .. }
                | MirPrimKind::PingPong { elem, .. }
                | MirPrimKind::Pool { elem, .. }
                | MirPrimKind::DmaSlot { elem } => scan_ty(elem, feat),
            },
            MirType::Lease(elem) => scan_ty(elem, feat),
            MirType::PrimView(MirPrimViewKind::Inbox { elem }) => scan_ty(elem, feat),
            MirType::PrimView(MirPrimViewKind::ByteRing) => {}
            _ => {}
        }
    }

    fn scan_prim_target(t: &MirPrimTarget, feat: &mut RuntimeIrFeatures) {
        match t {
            MirPrimTarget::Global(_) => {}
            MirPrimTarget::PtrExpr(expr) => scan_expr(expr, feat),
        }
    }

    fn scan_place(p: &MirPlace, feat: &mut RuntimeIrFeatures) {
        match p {
            MirPlace::Var(_) => {}
            MirPlace::Deref(expr, ty) => {
                scan_ty(ty, feat);
                scan_expr(expr, feat);
            }
            MirPlace::Field { base, field_ty, .. } => {
                scan_ty(field_ty, feat);
                scan_place(base, feat);
            }
            MirPlace::Index {
                base,
                index,
                elem_ty,
                ..
            } => {
                scan_ty(elem_ty, feat);
                scan_place(base, feat);
                scan_expr(index, feat);
            }
        }
    }

    fn scan_expr(e: &MirExpr, feat: &mut RuntimeIrFeatures) {
        scan_ty(&e.ty, feat);
        match &e.kind {
            MirExprKind::FloatLit(_) => feat.uses_floats = true,
            MirExprKind::Unary { expr: sub, .. } => scan_expr(sub, feat),
            MirExprKind::AddrOfPlace(place) => scan_place(place, feat),
            MirExprKind::Binary { lhs, rhs, .. } => {
                scan_expr(lhs, feat);
                scan_expr(rhs, feat);
            }
            MirExprKind::Cast {
                expr: sub,
                target_ty,
            } => {
                scan_ty(target_ty, feat);
                scan_expr(sub, feat);
            }
            MirExprKind::StructLit { fields, .. } => {
                for (_, fe, ft) in fields {
                    scan_ty(ft, feat);
                    scan_expr(fe, feat);
                }
            }
            MirExprKind::ArrayLit { elems, elem_ty } => {
                scan_ty(elem_ty, feat);
                for el in elems {
                    scan_expr(el, feat);
                }
            }
            MirExprKind::FieldAccess {
                base, raw_field_ty, ..
            } => {
                scan_ty(raw_field_ty, feat);
                scan_expr(base, feat);
            }
            MirExprKind::SliceLen(sub) | MirExprKind::FromBytes { slice: sub, .. } => {
                scan_expr(sub, feat);
            }
            MirExprKind::Index { base, index, .. } => {
                scan_expr(base, feat);
                scan_expr(index, feat);
            }
            MirExprKind::SubSlice {
                base,
                start,
                end,
                elem_ty,
                ..
            } => {
                scan_ty(elem_ty, feat);
                scan_expr(base, feat);
                scan_expr(start, feat);
                scan_expr(end, feat);
            }
            MirExprKind::ArrayToSlice {
                array_place,
                elem_ty,
                ..
            } => {
                scan_ty(elem_ty, feat);
                scan_place(array_place, feat);
            }
            MirExprKind::CallDirect { args, .. } => {
                for a in args {
                    scan_expr(a, feat);
                }
            }
            MirExprKind::RomCallU32 { addr, args, .. } => {
                scan_expr(addr, feat);
                for a in args {
                    scan_expr(a, feat);
                }
            }
            MirExprKind::PrimRead(prop) => match &**prop {
                MirPrimReadOp::WatchRead { target, elem_ty }
                | MirPrimReadOp::InboxTryRecv { target, elem_ty, .. }
                | MirPrimReadOp::InboxLen { target, elem_ty, .. }
                | MirPrimReadOp::InboxIsEmpty { target, elem_ty, .. }
                | MirPrimReadOp::InboxIsFull { target, elem_ty, .. } => {
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                }
                MirPrimReadOp::InboxTrySend {
                    target,
                    elem_ty,
                    value,
                    ..
                } => {
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                    scan_expr(value, feat);
                }
                MirPrimReadOp::SignalIsSet { target } | MirPrimReadOp::SignalTryWait { target } => {
                    scan_prim_target(target, feat);
                }
                MirPrimReadOp::PingPongDmaBuf {
                    target,
                    elem_ty,
                    cap,
                } => {
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                    let _ = cap;
                }
                MirPrimReadOp::PoolTryAlloc {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::PoolAvail {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::PoolInUse {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::PoolIsExhausted {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::PoolCap {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::DmaSlotTake { target, elem_ty }
                | MirPrimReadOp::DmaSlotRelease {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::DmaSlotIsBusy { target, elem_ty }
                | MirPrimReadOp::DmaSlotIsEmpty { target, elem_ty }
                | MirPrimReadOp::DmaSlotAddr { target, elem_ty } => {
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                }
                MirPrimReadOp::DmaSlotPark {
                    target,
                    elem_ty,
                    lease,
                } => {
                    scan_prim_target(target, feat);
                    scan_ty(elem_ty, feat);
                    scan_expr(lease, feat);
                }
                MirPrimReadOp::ByteRingPush {
                    target, src: val, ..
                }
                | MirPrimReadOp::ByteRingPop {
                    target, dst: val, ..
                }
                | MirPrimReadOp::ByteRingPushByte {
                    target, byte: val, ..
                }
                | MirPrimReadOp::ByteRingGrantWrite {
                    target, len: val, ..
                } => {
                    scan_prim_target(target, feat);
                    scan_expr(val, feat);
                }
                MirPrimReadOp::ByteRingReadSlice { target, .. }
                | MirPrimReadOp::ByteRingTryPopByte { target, .. }
                | MirPrimReadOp::ByteRingLen { target, .. }
                | MirPrimReadOp::ByteRingFreeSpace { target, .. }
                | MirPrimReadOp::ByteRingIsEmpty { target, .. }
                | MirPrimReadOp::ByteRingIsFull { target, .. } => {
                    scan_prim_target(target, feat);
                }
                MirPrimReadOp::InboxViewTryRecv { view, elem_ty }
                | MirPrimReadOp::InboxViewLen { view, elem_ty }
                | MirPrimReadOp::InboxViewCap { view, elem_ty }
                | MirPrimReadOp::InboxViewIsEmpty { view, elem_ty }
                | MirPrimReadOp::InboxViewIsFull { view, elem_ty } => {
                    scan_expr(view, feat);
                    scan_ty(elem_ty, feat);
                }
                MirPrimReadOp::InboxViewTrySend {
                    view,
                    elem_ty,
                    value,
                    ..
                } => {
                    scan_expr(view, feat);
                    scan_ty(elem_ty, feat);
                    scan_expr(value, feat);
                }
                MirPrimReadOp::ByteRingViewPush { view, src: val, .. }
                | MirPrimReadOp::ByteRingViewPop { view, dst: val }
                | MirPrimReadOp::ByteRingViewPushByte { view, byte: val, .. }
                | MirPrimReadOp::ByteRingViewGrantWrite { view, len: val } => {
                    scan_expr(view, feat);
                    scan_expr(val, feat);
                }
                MirPrimReadOp::ByteRingViewReadSlice { view }
                | MirPrimReadOp::ByteRingViewTryPopByte { view }
                | MirPrimReadOp::ByteRingViewLen { view }
                | MirPrimReadOp::ByteRingViewCap { view }
                | MirPrimReadOp::ByteRingViewFreeSpace { view }
                | MirPrimReadOp::ByteRingViewIsEmpty { view }
                | MirPrimReadOp::ByteRingViewIsFull { view } => {
                    scan_expr(view, feat);
                }
            },
            MirExprKind::MakePrimView { ptr, kind, .. } => {
                scan_expr(ptr, feat);
                if let MirPrimViewKind::Inbox { elem } = kind {
                    scan_ty(elem, feat);
                }
            }
            MirExprKind::MmioRead { base, index, .. } => {
                scan_expr(base, feat);
                if let Some(idx) = index {
                    scan_expr(idx, feat);
                }
            }
            MirExprKind::LeaseGet { lease, elem_ty } | MirExprKind::LeaseAddr { lease, elem_ty } => {
                scan_ty(elem_ty, feat);
                scan_expr(lease, feat);
            }
            _ => {}
        }
    }

    fn scan_fmt_chunks(chunks: &[MirFmtChunk], feat: &mut RuntimeIrFeatures) {
        feat.uses_fmt = true;
        for ch in chunks {
            match ch {
                MirFmtChunk::Literal(_) => {}
                MirFmtChunk::Int { value, bits, .. } => {
                    if *bits == 64 {
                        feat.uses_i64 = true;
                    }
                    scan_expr(value, feat);
                }
                MirFmtChunk::Float { value, fmt, .. } => {
                    feat.uses_floats = true;
                    if matches!(fmt, MirFloatFmt::Sci { .. }) {
                        feat.uses_fmt_float_sci = true;
                    }
                    scan_expr(value, feat);
                }
                MirFmtChunk::Bool(v) | MirFmtChunk::Str(v) | MirFmtChunk::Bytes(v) => {
                    scan_expr(v, feat);
                }
                MirFmtChunk::EnumName { value, .. } => scan_expr(value, feat),
                MirFmtChunk::DebugSlice { elem_ty, slice } => {
                    scan_ty(elem_ty, feat);
                    scan_expr(slice, feat);
                }
            }
        }
    }

    fn scan_stmts(stmts: &[MirStmt], feat: &mut RuntimeIrFeatures) {
        for s in stmts {
            match s {
                MirStmt::Assign { target, value, .. } => {
                    scan_place(target, feat);
                    scan_expr(value, feat);
                }
                MirStmt::Expr(value) => scan_expr(value, feat),
                MirStmt::PrimOp(pop) => match pop {
                    MirPrimNonBlockingOp::WatchPublish {
                        target,
                        elem_ty,
                        value,
                    }
                    | MirPrimNonBlockingOp::InboxPost {
                        target,
                        elem_ty,
                        value,
                        ..
                    } => {
                        scan_prim_target(target, feat);
                        scan_ty(elem_ty, feat);
                        scan_expr(value, feat);
                    }
                    MirPrimNonBlockingOp::SignalNotify { target, .. }
                    | MirPrimNonBlockingOp::SignalClear { target }
                    | MirPrimNonBlockingOp::ByteRingClear { target, .. } => {
                        scan_prim_target(target, feat);
                    }
                    MirPrimNonBlockingOp::ByteRingConsume { target, len, .. }
                    | MirPrimNonBlockingOp::ByteRingCommitWrite { target, len, .. } => {
                        scan_prim_target(target, feat);
                        scan_expr(len, feat);
                    }
                    MirPrimNonBlockingOp::PingPongFlip {
                        target, elem_ty, ..
                    }
                    | MirPrimNonBlockingOp::PingPongSetActive {
                        target, elem_ty, ..
                    } => {
                        scan_prim_target(target, feat);
                        scan_ty(elem_ty, feat);
                    }
                    MirPrimNonBlockingOp::LeaseRelease {
                        value, elem_ty, ..
                    } => {
                        scan_ty(elem_ty, feat);
                        scan_expr(value, feat);
                    }
                    MirPrimNonBlockingOp::LeaseNullify { target, elem_ty } => {
                        scan_ty(elem_ty, feat);
                        scan_place(target, feat);
                    }
                    MirPrimNonBlockingOp::InboxViewPost {
                        view,
                        elem_ty,
                        value,
                        ..
                    } => {
                        scan_expr(view, feat);
                        scan_ty(elem_ty, feat);
                        scan_expr(value, feat);
                    }
                    MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
                    | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                        scan_expr(view, feat);
                        scan_expr(len, feat);
                    }
                    MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                        scan_expr(view, feat);
                    }
                },
                MirStmt::Intrinsic(iop) => match iop {
                    MirIntrinsicOp::UseSystick { hz } => scan_expr(hz, feat),
                    MirIntrinsicOp::SysOp { method, args } => {
                        if matches!(method.as_str(), "tick" | "tick_ms" | "tick_by") {
                            feat.uses_sys_tick = true;
                        }
                        for a in args {
                            scan_expr(a, feat);
                        }
                    }
                    MirIntrinsicOp::SetPriority { .. } => {}
                },
                MirStmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    scan_expr(cond, feat);
                    scan_stmts(then_body, feat);
                    scan_stmts(else_body, feat);
                }
                MirStmt::While { cond, body } => {
                    scan_expr(cond, feat);
                    scan_stmts(body, feat);
                }
                MirStmt::Switch {
                    discr,
                    arms,
                    default,
                } => {
                    scan_expr(discr, feat);
                    for (_, ab) in arms {
                        scan_stmts(ab, feat);
                    }
                    scan_stmts(default, feat);
                }
                MirStmt::Return(Some(e)) => scan_expr(e, feat),
                MirStmt::Error { reason, partial } => {
                    if let Some(r) = reason {
                        scan_expr(r, feat);
                    }
                    if let Some(p) = partial {
                        scan_expr(p, feat);
                    }
                }
                MirStmt::SpawnActivate { args, .. } => {
                    for a in args {
                        scan_expr(a, feat);
                    }
                }
                MirStmt::PrintFmt { chunks, .. } => scan_fmt_chunks(chunks, feat),
                MirStmt::FmtBuf { buf, chunks, .. } => {
                    scan_expr(buf, feat);
                    scan_fmt_chunks(chunks, feat);
                }
                MirStmt::FmtRing { ring, chunks, .. } => {
                    scan_expr(ring, feat);
                    scan_fmt_chunks(chunks, feat);
                }
                MirStmt::MmioWrite {
                    base, index, value, ..
                } => {
                    scan_expr(base, feat);
                    if let Some(idx) = index {
                        scan_expr(idx, feat);
                    }
                    scan_expr(value, feat);
                }
                _ => {}
            }
        }
    }

    fn scan_blocks(blocks: &[MirBlock], feat: &mut RuntimeIrFeatures) {
        for b in blocks {
            scan_stmts(&b.stmts, feat);
            match &b.terminator {
                MirTerminator::Branch { cond, .. } => scan_expr(cond, feat),
                MirTerminator::Switch { discr, .. } => scan_expr(discr, feat),
                MirTerminator::Return(Some(e)) => scan_expr(e, feat),
                MirTerminator::Error { reason, partial } => {
                    if let Some(r) = reason {
                        scan_expr(r, feat);
                    }
                    if let Some(p) = partial {
                        scan_expr(p, feat);
                    }
                }
                MirTerminator::Yield(yk) => match yk {
                    MirYieldKind::CallYielding { args, .. } => {
                        for a in args {
                            scan_expr(a, feat);
                        }
                    }
                    MirYieldKind::InboxRecv {
                        target, elem_ty, ..
                    }
                    | MirYieldKind::InboxSend {
                        target, elem_ty, ..
                    }
                    | MirYieldKind::PingPongWaitChunk {
                        target, elem_ty, ..
                    }
                    | MirYieldKind::PoolAlloc {
                        target, elem_ty, ..
                    } => {
                        scan_prim_target(target, feat);
                        scan_ty(elem_ty, feat);
                    }
                    MirYieldKind::SignalWait { target, .. }
                    | MirYieldKind::ByteRingWaitRead { target, .. } => {
                        scan_prim_target(target, feat);
                    }
                    MirYieldKind::InboxViewRecv { view, elem_ty, .. }
                    | MirYieldKind::InboxViewSend { view, elem_ty, .. } => {
                        scan_expr(view, feat);
                        scan_ty(elem_ty, feat);
                    }
                    MirYieldKind::ByteRingViewWaitRead { view, .. } => {
                        scan_expr(view, feat);
                    }
                    MirYieldKind::EveryLatch { .. }
                    | MirYieldKind::DelayPoll { .. }
                    | MirYieldKind::ShareCheckpoint { .. }
                    | MirYieldKind::PollChild { .. } => {}
                },
                MirTerminator::Goto(_) | MirTerminator::Return(None) => {}
            }
        }
    }

    for c in &module.consts {
        scan_expr(&c.init, &mut feat);
    }
    for s in &module.structs {
        for f in &s.fields {
            scan_ty(&f.ty, &mut feat);
        }
    }
    for p in &module.primitives {
        scan_ty(&MirType::Primitive(p.kind.clone()), &mut feat);
        if let Some(init) = &p.init {
            scan_expr(init, &mut feat);
        }
    }
    for ext in &module.extern_fns {
        scan_ty(&ext.ret_ty, &mut feat);
        for (_, pt) in &ext.params {
            scan_ty(pt, &mut feat);
        }
    }
    for isr in &module.isrs {
        for v in &isr.vars {
            scan_ty(&v.ty, &mut feat);
        }
        for sv in &isr.state_vars {
            scan_ty(&sv.ty, &mut feat);
            scan_expr(&sv.init, &mut feat);
        }
        scan_stmts(&isr.body, &mut feat);
    }
    for vec in &module.irq_vectors {
        for b in &vec.bindings {
            for barg in &b.baked_args {
                if let MirBakedArg::Const(e) = barg {
                    scan_expr(e, &mut feat);
                }
            }
            for sv in &b.state_vars {
                scan_ty(&sv.ty, &mut feat);
                scan_expr(&sv.init, &mut feat);
            }
        }
    }
    for func in &module.functions {
        scan_ty(&func.ret_ty, &mut feat);
        for v in &func.vars {
            scan_ty(&v.ty, &mut feat);
        }
        match &func.body {
            MirFnBody::Direct(stmts) => scan_stmts(stmts, &mut feat),
            MirFnBody::Coroutine { blocks, .. } => scan_blocks(blocks, &mut feat),
        }
    }
    for bp in &module.task_blueprints {
        for v in &bp.vars {
            scan_ty(&v.ty, &mut feat);
        }
        scan_blocks(&bp.blocks, &mut feat);
    }
    for v in &module.boot.vars {
        scan_ty(&v.ty, &mut feat);
    }
    for step in &module.boot.steps {
        match step {
            MirBootStep::Stmt(s) => scan_stmts(std::slice::from_ref(s), &mut feat),
            MirBootStep::RunYieldingFn { args, .. } | MirBootStep::SpawnActivate { args, .. } => {
                for a in args {
                    scan_expr(a, &mut feat);
                }
            }
        }
    }
    feat
}

impl<'a> ModuleCtx<'a> {
    fn prepare(&mut self) -> Result<(), CodegenError> {
        // Enums
        for en in &self.module.enums {
            for (vname, disc) in &en.variants {
                self.enum_variants
                    .insert((en.name.clone(), vname.clone()), *disc);
                self.bare_variants.entry(vname.clone()).or_insert(*disc);
            }
        }

        // Consts
        for c in &self.module.consts {
            if c.init.ty.is_float() {
                let bits = self.eval_const_float_bits(&c.init);
                self.float_consts.insert(c.name.clone(), bits);
            } else {
                let v = self.eval_const_int(&c.init);
                self.consts.insert(c.name.clone(), v);
            }
        }

        // Structs in topological order
        let mut visited = BTreeSet::new();
        let by_name: HashMap<&str, &MirStruct> = self
            .module
            .structs
            .iter()
            .map(|s| (s.name.as_str(), s))
            .collect();
        for s in &self.module.structs {
            self.visit_struct(s.name.as_str(), &by_name, &mut visited);
        }

        // MMIO layouts & peripheral instances
        for layout in &self.module.mmio_layouts {
            self.mmio_layouts.insert(layout.name.clone(), layout.clone());
        }
        for inst in &self.module.mmio_instances {
            self.mmio_instances.insert(inst.name.clone(), inst.clone());
        }

        // Collect all distinct OS primitive types across structs, globals, ISRs, functions, tasks, and boot
        let structs = self.module.structs.clone();
        for st in &structs {
            for f in &st.fields {
                self.scan_mir_type_for_prims(&f.ty);
            }
        }
        for p in &self.module.primitives {
            if p.dead_stripped {
                continue;
            }
            self.prim_globals.insert(p.name.clone(), p.kind.clone());
            self.register_prim_kind(&p.kind);
        }
        for c in &self.module.consts {
            self.scan_expr_for_prims(&c.init);
        }
        for isr in &self.module.isrs {
            for v in &isr.vars {
                self.scan_mir_type_for_prims(&v.ty);
            }
            for sv in &isr.state_vars {
                self.scan_mir_type_for_prims(&sv.ty);
            }
            for s in &isr.body {
                self.scan_stmt_for_prims(s);
            }
        }
        for vec in &self.module.irq_vectors {
            for b in &vec.bindings {
                for ba in &b.baked_args {
                    if let MirBakedArg::Const(e) = ba {
                        self.scan_expr_for_prims(e);
                    }
                }
                for sv in &b.state_vars {
                    self.scan_mir_type_for_prims(&sv.ty);
                    self.scan_expr_for_prims(&sv.init);
                }
            }
        }
        for f in &self.module.functions {
            for v in &f.vars {
                self.scan_mir_type_for_prims(&v.ty);
            }
            self.scan_mir_type_for_prims(&f.ret_ty);
            match &f.body {
                MirFnBody::Direct(stmts) => {
                    for s in stmts {
                        self.scan_stmt_for_prims(s);
                    }
                }
                MirFnBody::Coroutine { blocks, .. } => self.scan_blocks_for_prims(blocks),
            }
        }
        for bp in &self.module.task_blueprints {
            for v in &bp.vars {
                self.scan_mir_type_for_prims(&v.ty);
            }
            self.scan_blocks_for_prims(&bp.blocks);
        }
        for ext in &self.module.extern_fns {
            for (_, pty) in &ext.params {
                self.scan_mir_type_for_prims(pty);
            }
            self.scan_mir_type_for_prims(&ext.ret_ty);
        }
        for v in &self.module.boot.vars {
            self.scan_mir_type_for_prims(&v.ty);
        }
        for step in &self.module.boot.steps {
            match step {
                MirBootStep::Stmt(s) => self.scan_stmt_for_prims(s),
                MirBootStep::RunYieldingFn { args, .. } | MirBootStep::SpawnActivate { args, .. } => {
                    for a in args {
                        self.scan_expr_for_prims(a);
                    }
                }
            }
        }

        // Function signatures
        for f in &self.module.functions {
            let params: Vec<MirType> = f.params.iter().map(|vid| f.vars[vid.0].ty.clone()).collect();
            let env_mir_ty = f.envelope_return_type();
            let ret_llvm_ty = if f.is_yielding {
                "i32".to_string()
            } else if env_mir_ty.is_void() {
                "void".to_string()
            } else {
                self.llvm_ty(&env_mir_ty)
            };
            let symbol = if f.is_yielding {
                format!("@__syn_fn_step_{}", sanitize_ident(&f.name))
            } else {
                format!("@__syn_fn_{}", sanitize_ident(&f.name))
            };
            self.fn_sigs.insert(
                f.name.clone(),
                FnLlvmSig {
                    symbol,
                    params,
                    ret_llvm_ty,
                },
            );
        }

        // Extern function signatures
        for ext in &self.module.extern_fns {
            self.extern_sigs.insert(
                ext.name.clone(),
                ExternLlvmSig {
                    name: ext.name.clone(),
                    params: ext.params.iter().map(|(_, t)| t.clone()).collect(),
                    ret_ty: ext.ret_ty.clone(),
                    variadic: ext.variadic,
                },
            );
        }

        // Yielding function frames in post-order (children before parents!)
        let fn_map: HashMap<&str, &MirFunction> = self
            .module
            .functions
            .iter()
            .filter(|f| f.is_yielding)
            .map(|f| (f.name.as_str(), f))
            .collect();
        let mut done_fns = BTreeSet::new();
        for f in &self.module.functions {
            if f.is_yielding {
                self.compute_fn_frame_layout(f.name.as_str(), &fn_map, &mut done_fns);
            }
        }

        // Task blueprint frames
        for bp in &self.module.task_blueprints {
            let layout = self.build_coroutine_layout(
                &format!("%TaskFrame_{}", sanitize_ident(&bp.name)),
                false,
                &MirType::Void,
                &bp.vars,
                &bp.child_slots,
            );
            self.bp_frames.insert(bp.name.clone(), layout);
        }

        Ok(())
    }

    fn resolve_sys_hooks(&self) -> ResolvedSysHooks {
        fn unwrap_fn_addr(mut e: &MirExpr) -> Option<&str> {
            while let MirExprKind::Cast { expr, .. } = &e.kind {
                e = expr;
            }
            if let MirExprKind::FnAddr(name) = &e.kind {
                Some(name.as_str())
            } else {
                None
            }
        }

        fn record_hook_call<'a>(
            method: &str,
            args: &'a [MirExpr],
            sites: &mut BTreeMap<HookKind, Vec<Option<&'a str>>>,
        ) {
            match method {
                "set_putc" => sites
                    .entry(HookKind::Putc)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_clock_ms" => sites
                    .entry(HookKind::ClockMs)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_exit" => sites
                    .entry(HookKind::Exit)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_fault_handler" => sites
                    .entry(HookKind::FaultHandler)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_idle" => sites
                    .entry(HookKind::Idle)
                    .or_default()
                    .push(args.first().and_then(unwrap_fn_addr)),
                "set_irq_router" => {
                    sites
                        .entry(HookKind::IrqRoute)
                        .or_default()
                        .push(args.first().and_then(unwrap_fn_addr));
                    sites
                        .entry(HookKind::IrqAck)
                        .or_default()
                        .push(args.get(1).and_then(unwrap_fn_addr));
                }
                _ => {}
            }
        }

        fn collect_stmts<'a>(
            stmts: &'a [MirStmt],
            sites: &mut BTreeMap<HookKind, Vec<Option<&'a str>>>,
        ) {
            for s in stmts {
                match s {
                    MirStmt::Intrinsic(MirIntrinsicOp::SysOp { method, args }) => {
                        record_hook_call(method, args, sites);
                    }
                    MirStmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        collect_stmts(then_body, sites);
                        collect_stmts(else_body, sites);
                    }
                    MirStmt::While { body, .. } => collect_stmts(body, sites),
                    MirStmt::Switch { arms, default, .. } => {
                        for (_, b) in arms {
                            collect_stmts(b, sites);
                        }
                        collect_stmts(default, sites);
                    }
                    _ => {}
                }
            }
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum WalkStop {
            Continue,
            FnReturns,
            ProgramExits,
        }

        fn stmts_may_return_or_exit(stmts: &[MirStmt]) -> bool {
            for s in stmts {
                match s {
                    MirStmt::Return(_) | MirStmt::Error { .. } => return true,
                    MirStmt::Intrinsic(MirIntrinsicOp::SysOp { method, .. })
                        if matches!(method.as_str(), "exit" | "stop" | "rom_call_noreturn") =>
                    {
                        return true;
                    }
                    MirStmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        if stmts_may_return_or_exit(then_body)
                            || stmts_may_return_or_exit(else_body)
                        {
                            return true;
                        }
                    }
                    MirStmt::While { body, .. } => {
                        if stmts_may_return_or_exit(body) {
                            return true;
                        }
                    }
                    MirStmt::Switch { arms, default, .. } => {
                        if arms.iter().any(|(_, b)| stmts_may_return_or_exit(b))
                            || stmts_may_return_or_exit(default)
                        {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        }

        fn walk_expr_uncond(
            expr: &MirExpr,
            fn_map: &HashMap<&str, &MirFnBody>,
            active: &mut BTreeSet<String>,
            uncond: &mut BTreeSet<HookKind>,
        ) -> WalkStop {
            match &expr.kind {
                MirExprKind::CallDirect {
                    callee,
                    is_extern_c: false,
                    args,
                } => {
                    for a in args {
                        if walk_expr_uncond(a, fn_map, active, uncond) == WalkStop::ProgramExits {
                            return WalkStop::ProgramExits;
                        }
                    }
                    if !active.contains(callee.as_str()) {
                        if let Some(MirFnBody::Direct(body)) = fn_map.get(callee.as_str()) {
                            active.insert(callee.clone());
                            let st = walk_stmts_uncond(body, fn_map, active, uncond);
                            active.remove(callee.as_str());
                            if st == WalkStop::ProgramExits {
                                return WalkStop::ProgramExits;
                            }
                        }
                    }
                    WalkStop::Continue
                }
                MirExprKind::Unary { expr: sub, .. } | MirExprKind::Cast { expr: sub, .. } => {
                    walk_expr_uncond(sub, fn_map, active, uncond)
                }
                _ => WalkStop::Continue,
            }
        }

        fn walk_stmt_uncond(
            stmt: &MirStmt,
            fn_map: &HashMap<&str, &MirFnBody>,
            active: &mut BTreeSet<String>,
            uncond: &mut BTreeSet<HookKind>,
        ) -> WalkStop {
            match stmt {
                MirStmt::Intrinsic(MirIntrinsicOp::SysOp { method, .. }) => {
                    match method.as_str() {
                        "set_putc" => {
                            uncond.insert(HookKind::Putc);
                        }
                        "set_clock_ms" => {
                            uncond.insert(HookKind::ClockMs);
                        }
                        "set_exit" => {
                            uncond.insert(HookKind::Exit);
                        }
                        "set_fault_handler" => {
                            uncond.insert(HookKind::FaultHandler);
                        }
                        "set_idle" => {
                            uncond.insert(HookKind::Idle);
                        }
                        "set_irq_router" => {
                            uncond.insert(HookKind::IrqRoute);
                            uncond.insert(HookKind::IrqAck);
                        }
                        "exit" | "stop" | "rom_call_noreturn" => return WalkStop::ProgramExits,
                        _ => {}
                    }
                    WalkStop::Continue
                }
                MirStmt::Return(_) | MirStmt::Error { .. } => WalkStop::FnReturns,
                MirStmt::Expr(e) | MirStmt::Assign { value: e, .. } => {
                    walk_expr_uncond(e, fn_map, active, uncond)
                }
                MirStmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    if stmts_may_return_or_exit(then_body) || stmts_may_return_or_exit(else_body) {
                        WalkStop::FnReturns
                    } else {
                        WalkStop::Continue
                    }
                }
                MirStmt::While { body, .. } => {
                    if stmts_may_return_or_exit(body) {
                        WalkStop::FnReturns
                    } else {
                        WalkStop::Continue
                    }
                }
                MirStmt::Switch { arms, default, .. } => {
                    if arms.iter().any(|(_, b)| stmts_may_return_or_exit(b))
                        || stmts_may_return_or_exit(default)
                    {
                        WalkStop::FnReturns
                    } else {
                        WalkStop::Continue
                    }
                }
                _ => WalkStop::Continue,
            }
        }

        fn walk_stmts_uncond(
            stmts: &[MirStmt],
            fn_map: &HashMap<&str, &MirFnBody>,
            active: &mut BTreeSet<String>,
            uncond: &mut BTreeSet<HookKind>,
        ) -> WalkStop {
            for s in stmts {
                let st = walk_stmt_uncond(s, fn_map, active, uncond);
                if st != WalkStop::Continue {
                    return st;
                }
            }
            WalkStop::Continue
        }

        let mut sites: BTreeMap<HookKind, Vec<Option<&str>>> = BTreeMap::new();
        for step in &self.module.boot.steps {
            if let MirBootStep::Stmt(s) = step {
                collect_stmts(std::slice::from_ref(s), &mut sites);
            }
        }
        let mut fn_map: HashMap<&str, &MirFnBody> = HashMap::new();
        for f in &self.module.functions {
            fn_map.insert(f.name.as_str(), &f.body);
            match &f.body {
                MirFnBody::Direct(stmts) => collect_stmts(stmts, &mut sites),
                MirFnBody::Coroutine { blocks, .. } => {
                    for b in blocks {
                        collect_stmts(&b.stmts, &mut sites);
                    }
                }
            }
        }
        for bp in &self.module.task_blueprints {
            for b in &bp.blocks {
                collect_stmts(&b.stmts, &mut sites);
            }
        }
        for isr in &self.module.isrs {
            collect_stmts(&isr.body, &mut sites);
        }

        let mut uncond_hooks = BTreeSet::new();
        if self.module.boot.present {
            let mut active = BTreeSet::new();
            for step in &self.module.boot.steps {
                if let MirBootStep::Stmt(s) = step {
                    if walk_stmt_uncond(s, &fn_map, &mut active, &mut uncond_hooks)
                        == WalkStop::ProgramExits
                    {
                        break;
                    }
                }
            }
        }

        let resolve_one = |kind: HookKind| -> StaticHookMode {
            let Some(entries) = sites.get(&kind) else {
                return StaticHookMode::Unset;
            };
            if entries.is_empty() {
                return StaticHookMode::Unset;
            }
            if entries.iter().all(Option::is_some) {
                let mut arms: Vec<(String, String)> = Vec::new();
                let mut seen = BTreeSet::new();
                let mut all_resolved = true;
                for name_opt in entries {
                    let fname = name_opt.expect("checked is_some");
                    if seen.insert(fname.to_string()) {
                        if let Some(sig) = self.fn_sigs.get(fname) {
                            arms.push((fname.to_string(), sig.symbol.clone()));
                        } else {
                            all_resolved = false;
                            break;
                        }
                    }
                }
                if all_resolved {
                    if arms.len() == 1 && uncond_hooks.contains(&kind) {
                        let (fn_name, symbol) = arms.into_iter().next().expect("len 1");
                        return StaticHookMode::DirectUnconditional { fn_name, symbol };
                    }
                    return StaticHookMode::TagSwitch {
                        arms,
                        global_sym: format!("@__syn_hook_tag_{}", kind.suffix()),
                    };
                }
            }
            StaticHookMode::Indirect {
                global_sym: format!("@__syn_hook_fn_{}", kind.suffix()),
            }
        };

        ResolvedSysHooks {
            putc: resolve_one(HookKind::Putc),
            clock_ms: resolve_one(HookKind::ClockMs),
            exit: resolve_one(HookKind::Exit),
            fault_handler: resolve_one(HookKind::FaultHandler),
            idle: resolve_one(HookKind::Idle),
            irq_route: resolve_one(HookKind::IrqRoute),
            irq_ack: resolve_one(HookKind::IrqAck),
        }
    }

    fn emit_sys_hook_globals_ir(&self, out: &mut String) {
        let fn_ptr_ty = self.spec.isa.fn_ptr_llvm_ty();
        let ptr_al = self.ptr_align();
        let all_kinds = [
            HookKind::Putc,
            HookKind::ClockMs,
            HookKind::Exit,
            HookKind::FaultHandler,
            HookKind::Idle,
            HookKind::IrqRoute,
            HookKind::IrqAck,
        ];
        let mut any = false;
        for kind in all_kinds {
            match self.hooks.mode(kind) {
                StaticHookMode::TagSwitch { ref global_sym, .. } => {
                    if !any {
                        let _ = writeln!(out, "; ── Explicit `sys.*` Hook Static Dispatch State ──");
                        any = true;
                    }
                    let _ = writeln!(out, "{global_sym} = internal global i8 0, align 1");
                }
                StaticHookMode::Indirect { global_sym } => {
                    if !any {
                        let _ = writeln!(out, "; ── Explicit `sys.*` Hook Static Dispatch State ──");
                        any = true;
                    }
                    let _ = writeln!(
                        out,
                        "{global_sym} = internal global {fn_ptr_ty} null, align {ptr_al}"
                    );
                }
                StaticHookMode::Unset | StaticHookMode::DirectUnconditional { .. } => {}
            }
        }
        if any {
            let _ = writeln!(out);
        }
    }

    fn emit_hook_wrapper(
        &self,
        out: &mut String,
        kind: HookKind,
        params_sig: &str,
        ret_ty: &str,
        section_attr: &str,
        fallback_body: &str,
        pre_call_ir: &str,
        pass_args: &str,
    ) {
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let call_kw = self.call_kw();
        let fn_ptr_ty = self.spec.isa.fn_ptr_llvm_ty();
        let ptr_al = self.ptr_align();
        let sym = kind.wrapper_sym();
        let emit_call_and_ret = |callee: &str| -> String {
            if ret_ty == "void" {
                format!("{pre_call_ir}  {call_kw} void {callee}({pass_args})\n  ret void")
            } else {
                format!("{pre_call_ir}  %rv = {call_kw} {ret_ty} {callee}({pass_args})\n  ret {ret_ty} %rv")
            }
        };

        match self.hooks.mode(kind) {
            StaticHookMode::Unset => {
                let _ = writeln!(
                    out,
                    "define {ret_ty} {sym}({params_sig}){fn_as} nounwind{fn_sz}{section_attr} {{\nentry:\n{fallback_body}\n}}\n"
                );
            }
            StaticHookMode::DirectUnconditional { symbol, .. } => {
                let body = emit_call_and_ret(symbol);
                let _ = writeln!(
                    out,
                    "define {ret_ty} {sym}({params_sig}){fn_as} nounwind{fn_sz}{section_attr} {{\nentry:\n{body}\n}}\n"
                );
            }
            StaticHookMode::TagSwitch { arms, global_sym } => {
                let _ = writeln!(
                    out,
                    "define {ret_ty} {sym}({params_sig}){fn_as} nounwind{fn_sz}{section_attr} {{\nentry:"
                );
                let _ = writeln!(out, "  %tag = load i8, ptr {global_sym}, align 1");
                let _ = writeln!(out, "  switch i8 %tag, label %fallback [");
                for (i, _) in arms.iter().enumerate() {
                    let tag = i + 1;
                    let _ = writeln!(out, "    i8 {tag}, label %arm_{tag}");
                }
                let _ = writeln!(out, "  ]");
                for (i, (_, fn_sym)) in arms.iter().enumerate() {
                    let tag = i + 1;
                    let _ = writeln!(out, "arm_{tag}:");
                    let _ = writeln!(out, "{}", emit_call_and_ret(fn_sym));
                }
                let _ = writeln!(out, "fallback:\n{fallback_body}\n}}\n");
            }
            StaticHookMode::Indirect { global_sym } => {
                let call_body = emit_call_and_ret("%fp");
                let _ = writeln!(
                    out,
                    "define {ret_ty} {sym}({params_sig}){fn_as} nounwind{fn_sz}{section_attr} {{\nentry:\n  %fp = load {fn_ptr_ty}, ptr {global_sym}, align {ptr_al}\n  %has = icmp ne {fn_ptr_ty} %fp, null\n  br i1 %has, label %do_call, label %fallback\ndo_call:\n{call_body}\nfallback:\n{fallback_body}\n}}\n"
                );
            }
        }
    }

    fn emit_sys_hooks_ir(&self, out: &mut String) {
        let _ = writeln!(out, "; ── Explicit `sys.*` Hook Wrappers (Zero-RAM when Unconditional/Unset) ──");
        let putc_fallback = if self.spec.isa == CpuIsa::HostX86_64Linux {
            "  %z = zext i8 %c to i32\n  %r = call i32 @putchar(i32 %z)\n  ret void"
        } else if self.spec.isa == CpuIsa::ArmCortexM && self.spec.semihosting && self.spec.has_thumb2 {
            "  %ch = alloca i8, align 1\n  store i8 %c, ptr %ch, align 1\n  call void asm sideeffect \"mov r0, #0x03\\0Abkpt #0xab\", \"{r1},~{r0},~{memory}\"(ptr %ch)\n  ret void"
        } else {
            "  ret void"
        };
        let iram_sec = if self.spec.isa.is_riscv() && self.spec.iram_in_data {
            " section \".iram1\""
        } else {
            ""
        };
        self.emit_hook_wrapper(
            out,
            HookKind::Putc,
            "i8 %c",
            "void",
            "",
            putc_fallback,
            "",
            "i8 %c",
        );
        self.emit_hook_wrapper(
            out,
            HookKind::ClockMs,
            "",
            "i32",
            "",
            "  %t = load volatile i32, ptr @__syn_tick_ms, align 4\n  ret i32 %t",
            "",
            "",
        );
        self.emit_hook_wrapper(
            out,
            HookKind::Exit,
            "i32 %code",
            "void",
            "",
            "  ret void",
            "",
            "i32 %code",
        );
        self.emit_hook_wrapper(
            out,
            HookKind::FaultHandler,
            "",
            "void",
            iram_sec,
            "  ret void",
            "",
            "",
        );
        self.emit_hook_wrapper(
            out,
            HookKind::Idle,
            "i32 %in_sim, i32 %depth",
            "void",
            "",
            "  ret void",
            "",
            "i32 %in_sim, i32 %depth",
        );
        if self.spec.isa.is_riscv() {
            self.emit_hook_wrapper(
                out,
                HookKind::IrqRoute,
                "i32 %src, i32 %slot",
                "void",
                "",
                "  ret void",
                "",
                "i32 %src, i32 %slot",
            );
            self.emit_hook_wrapper(
                out,
                HookKind::IrqAck,
                "i32 %slot",
                "void",
                iram_sec,
                "  ret void",
                "",
                "i32 %slot",
            );
        }
    }

    fn emit_rom_wrappers_ir(&self, out: &mut String) {
        if self.rom_wrappers.is_empty() {
            return;
        }
        let usize_ty = self.spec.isa.usize_llvm_ty();
        let fn_ptr_ty = self.spec.isa.fn_ptr_llvm_ty();
        let fn_as = self.fn_as();
        let call_kw = self.call_kw();
        let fn_sz = self.opts.fn_size_attrs();
        let _ = writeln!(out, "; ── Stack-Bounded ROM / Firmware ABI Wrappers ──");
        for w in &self.rom_wrappers {
            let sym = w.symbol();
            let mut def_params = vec![format!("{usize_ty} %addr")];
            let mut call_args = Vec::with_capacity(w.param_llvm_tys.len());
            for (i, pty) in w.param_llvm_tys.iter().enumerate() {
                def_params.push(format!("{pty} %a{i}"));
                call_args.push(format!("{pty} %a{i}"));
            }
            let def_sig = def_params.join(", ");
            let pass_args = call_args.join(", ");
            match w.kind {
                RomWrapperKind::U32 => {
                    let _ = writeln!(
                        out,
                        "define i32 {sym}({def_sig}){fn_as} nounwind noinline{fn_sz} {{\nentry:\n  %fp = inttoptr {usize_ty} %addr to {fn_ptr_ty}\n  %rv = {call_kw} i32 %fp({pass_args})\n  ret i32 %rv\n}}\n"
                    );
                }
                RomWrapperKind::Void => {
                    let _ = writeln!(
                        out,
                        "define void {sym}({def_sig}){fn_as} nounwind noinline{fn_sz} {{\nentry:\n  %fp = inttoptr {usize_ty} %addr to {fn_ptr_ty}\n  {call_kw} void %fp({pass_args})\n  ret void\n}}\n"
                    );
                }
                RomWrapperKind::Noreturn => {
                    let _ = writeln!(
                        out,
                        "define void {sym}({def_sig}){fn_as} noreturn nounwind noinline{fn_sz} {{\nentry:\n  %fp = inttoptr {usize_ty} %addr to {fn_ptr_ty}\n  {call_kw} void %fp({pass_args})\n  unreachable\n}}\n"
                    );
                }
            }
        }
    }

    fn visit_struct<'b>(
        &mut self,
        name: &str,
        by_name: &HashMap<&'b str, &'b MirStruct>,
        visited: &mut BTreeSet<String>,
    ) {
        if visited.contains(name) {
            return;
        }
        visited.insert(name.to_string());
        let Some(&decl) = by_name.get(name) else {
            return;
        };
        for fld in &decl.fields {
            if let MirType::Struct(sub) = &fld.ty {
                self.visit_struct(sub, by_name, visited);
            }
        }
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let mut offset = 0usize;
        let mut max_align = 1usize;
        let mut llvm_off = 0usize;
        let mut llvm_max_nat_al = 1usize;
        let mut field_indices = HashMap::new();
        let mut llvm_fields = Vec::new();
        let struct_dma = self.dma_layout_structs.contains(&decl.name);
        for (idx, fld) in decl.fields.iter().enumerate() {
            let eff_fld_dma = fld.dma_safe || struct_dma;
            let is_dma_pool = eff_fld_dma
                && matches!(fld.ty, MirType::Primitive(MirPrimKind::Pool { .. }));
            let (fsize, mut falign) = self.size_align(&fld.ty, ptr_bytes);
            let fsize = if is_dma_pool {
                if let MirType::Primitive(MirPrimKind::Pool { elem, cap }) = &fld.ty {
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let stride = align_up(esz.max(1), eal.max(32));
                    32 + (*cap).max(1) * stride
                } else {
                    fsize
                }
            } else {
                fsize
            };
            if is_dma_pool {
                falign = falign.max(32);
            }
            if !decl.packed {
                offset = align_up(offset, falign);
                max_align = max_align.max(falign);
            }
            offset += fsize;

            let (nat_sz, nat_al) = self.llvm_abi_size_align(&fld.ty, eff_fld_dma);
            let req_al = if decl.is_subsystem {
                falign
            } else if decl.packed {
                1
            } else {
                nat_al
            };
            if decl.is_subsystem && req_al > nat_al {
                let aligned = align_up(llvm_off, req_al);
                if aligned > llvm_off {
                    llvm_fields.push((None, format!("[{} x i8]", aligned - llvm_off)));
                    llvm_off = aligned;
                }
            }
            if !decl.packed {
                llvm_off = align_up(llvm_off, nat_al);
                llvm_max_nat_al = llvm_max_nat_al.max(nat_al);
            }
            let llvm_idx = llvm_fields.len();
            llvm_fields.push((Some(idx), self.llvm_field_ty(&decl.name, fld)));
            llvm_off += nat_sz;
            field_indices.insert(fld.name.clone(), (llvm_idx, fld.ty.clone()));
        }
        if !decl.packed {
            offset = align_up(offset, max_align);
            llvm_off = align_up(llvm_off, llvm_max_nat_al);
        }
        let llvm_name = format!("%struct.{}", sanitize_ident(&decl.name));
        self.structs.insert(
            decl.name.clone(),
            StructInfo {
                decl: decl.clone(),
                llvm_name,
                byte_size: offset.max(1),
                align: if decl.packed { 1 } else { max_align.max(1) },
                llvm_abi_size: llvm_off.max(1),
                llvm_nat_align: if decl.packed {
                    1
                } else {
                    llvm_max_nat_al.max(1)
                },
                field_indices,
                llvm_fields,
            },
        );
        self.struct_order.push(decl.name.clone());
    }

    fn scan_mir_type_for_prims(&mut self, ty: &MirType) {
        match ty {
            MirType::Primitive(pk) => self.register_prim_kind(pk),
            MirType::Lease(elem) => {
                self.uses_pool_lease = true;
                self.scan_mir_type_for_prims(elem);
            }
            MirType::PrimView(MirPrimViewKind::ByteRing) => self.uses_byte_ring_view = true,
            MirType::PrimView(MirPrimViewKind::Inbox { elem }) => self.register_inbox_view(elem),
            MirType::Ptr(inner) | MirType::Slice(inner) => self.scan_mir_type_for_prims(inner),
            MirType::Array { elem, .. } => self.scan_mir_type_for_prims(elem),
            MirType::Result { ok, err } => {
                self.scan_mir_type_for_prims(ok);
                if let Some(e) = err {
                    self.scan_mir_type_for_prims(e);
                }
            }
            _ => {}
        }
    }

    fn register_inbox_view(&mut self, elem: &MirType) {
        self.scan_mir_type_for_prims(elem);
        let key = sanitize_ident(&elem.mangle_name());
        if !self.distinct_inbox_views.contains_key(&key) {
            self.distinct_inbox_views.insert(key, elem.clone());
        }
    }

    fn register_prim_kind(&mut self, pk: &MirPrimKind) {
        if matches!(pk, MirPrimKind::Pool { .. } | MirPrimKind::DmaSlot { .. }) {
            self.uses_pool_lease = true;
        }
        match pk {
            MirPrimKind::Watch { elem }
            | MirPrimKind::Inbox { elem, .. }
            | MirPrimKind::PingPong { elem, .. }
            | MirPrimKind::Pool { elem, .. }
            | MirPrimKind::DmaSlot { elem } => self.scan_mir_type_for_prims(elem),
            MirPrimKind::Signal | MirPrimKind::ByteRing { .. } => {}
        }
        let key = pk.mangle_name();
        if !self.distinct_prims.contains_key(&key) {
            self.distinct_prims.insert(key, pk.clone());
        }
    }

    fn scan_prim_target_for_prims(&mut self, target: &MirPrimTarget) {
        if let MirPrimTarget::PtrExpr(e) = target {
            self.scan_expr_for_prims(e);
        }
    }

    fn scan_blocks_for_prims(&mut self, blocks: &[MirBlock]) {
        for b in blocks {
            for s in &b.stmts {
                self.scan_stmt_for_prims(s);
            }
            match &b.terminator {
                MirTerminator::Branch { cond, .. } | MirTerminator::Switch { discr: cond, .. } => {
                    self.scan_expr_for_prims(cond);
                }
                MirTerminator::Return(Some(e)) => self.scan_expr_for_prims(e),
                MirTerminator::Error { reason, partial } => {
                    if let Some(r) = reason {
                        self.scan_expr_for_prims(r);
                    }
                    if let Some(p) = partial {
                        self.scan_expr_for_prims(p);
                    }
                }
                MirTerminator::Yield(yk) => match yk {
                    MirYieldKind::CallYielding { args, .. } => {
                        for a in args {
                            self.scan_expr_for_prims(a);
                        }
                    }
                    MirYieldKind::InboxViewRecv { view, elem_ty, .. }
                    | MirYieldKind::InboxViewSend { view, elem_ty, .. } => {
                        self.scan_expr_for_prims(view);
                        self.register_inbox_view(elem_ty);
                    }
                    MirYieldKind::ByteRingViewWaitRead { view, .. } => {
                        self.scan_expr_for_prims(view);
                        self.uses_byte_ring_view = true;
                    }
                    MirYieldKind::PoolAlloc {
                        target,
                        elem_ty,
                        cap,
                        ..
                    } => {
                        self.scan_prim_target_for_prims(target);
                        self.register_prim_kind(&MirPrimKind::Pool {
                            elem: Box::new(elem_ty.clone()),
                            cap: *cap,
                        });
                    }
                    MirYieldKind::InboxRecv {
                        target,
                        elem_ty,
                        cap,
                        ..
                    }
                    | MirYieldKind::InboxSend {
                        target,
                        elem_ty,
                        cap,
                        ..
                    } => {
                        self.scan_prim_target_for_prims(target);
                        self.register_prim_kind(&MirPrimKind::Inbox {
                            elem: Box::new(elem_ty.clone()),
                            cap: *cap,
                        });
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    fn scan_stmt_for_prims(&mut self, stmt: &MirStmt) {
        match stmt {
            MirStmt::Assign { value, .. } | MirStmt::Expr(value) | MirStmt::Return(Some(value)) => {
                self.scan_expr_for_prims(value);
            }
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.scan_expr_for_prims(cond);
                for s in then_body {
                    self.scan_stmt_for_prims(s);
                }
                for s in else_body {
                    self.scan_stmt_for_prims(s);
                }
            }
            MirStmt::While { cond, body } => {
                self.scan_expr_for_prims(cond);
                for s in body {
                    self.scan_stmt_for_prims(s);
                }
            }
            MirStmt::Switch {
                discr,
                arms,
                default,
            } => {
                self.scan_expr_for_prims(discr);
                for (_, ab) in arms {
                    for s in ab {
                        self.scan_stmt_for_prims(s);
                    }
                }
                for s in default {
                    self.scan_stmt_for_prims(s);
                }
            }
            MirStmt::FmtRing { ring, chunks, .. } => {
                self.scan_expr_for_prims(ring);
                self.uses_byte_ring_view = true;
                for ch in chunks {
                    self.scan_fmt_chunk_for_prims(ch);
                }
            }
            MirStmt::PrintFmt { chunks, .. } | MirStmt::FmtBuf { chunks, .. } => {
                if let MirStmt::FmtBuf { buf, .. } = stmt {
                    self.scan_expr_for_prims(buf);
                }
                for ch in chunks {
                    self.scan_fmt_chunk_for_prims(ch);
                }
            }
            MirStmt::PrimOp(pop) => match pop {
                MirPrimNonBlockingOp::InboxViewPost {
                    view,
                    elem_ty,
                    value,
                    ..
                } => {
                    self.scan_expr_for_prims(view);
                    self.register_inbox_view(elem_ty);
                    self.scan_expr_for_prims(value);
                }
                MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
                | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                    self.scan_expr_for_prims(view);
                    self.uses_byte_ring_view = true;
                    self.scan_expr_for_prims(len);
                }
                MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                    self.scan_expr_for_prims(view);
                    self.uses_byte_ring_view = true;
                }
                MirPrimNonBlockingOp::WatchPublish { target, value, .. }
                | MirPrimNonBlockingOp::InboxPost { target, value, .. } => {
                    self.scan_prim_target_for_prims(target);
                    self.scan_expr_for_prims(value);
                }
                MirPrimNonBlockingOp::LeaseRelease { value, elem_ty, .. } => {
                    self.uses_pool_lease = true;
                    self.scan_mir_type_for_prims(elem_ty);
                    self.scan_expr_for_prims(value);
                }
                MirPrimNonBlockingOp::LeaseNullify { elem_ty, .. } => {
                    self.uses_pool_lease = true;
                    self.scan_mir_type_for_prims(elem_ty);
                }
                MirPrimNonBlockingOp::ByteRingConsume { len, .. }
                | MirPrimNonBlockingOp::ByteRingCommitWrite { len, .. } => {
                    self.scan_expr_for_prims(len);
                }
                _ => {}
            },
            MirStmt::SpawnActivate { args, .. } => {
                for a in args {
                    self.scan_expr_for_prims(a);
                }
            }
            MirStmt::Intrinsic(iop) => match iop {
                MirIntrinsicOp::UseSystick { hz } => self.scan_expr_for_prims(hz),
                MirIntrinsicOp::SysOp { args, .. } => {
                    for a in args {
                        self.scan_expr_for_prims(a);
                    }
                }
                MirIntrinsicOp::SetPriority { .. } => {}
            },
            MirStmt::MmioWrite {
                base, index, value, ..
            } => {
                self.scan_expr_for_prims(base);
                if let Some(i) = index {
                    self.scan_expr_for_prims(i);
                }
                self.scan_expr_for_prims(value);
            }
            MirStmt::Error { reason, partial } => {
                if let Some(r) = reason {
                    self.scan_expr_for_prims(r);
                }
                if let Some(p) = partial {
                    self.scan_expr_for_prims(p);
                }
            }
            _ => {}
        }
    }

    fn scan_fmt_chunk_for_prims(&mut self, ch: &MirFmtChunk) {
        match ch {
            MirFmtChunk::Int { value, .. }
            | MirFmtChunk::Float { value, .. }
            | MirFmtChunk::Bool(value)
            | MirFmtChunk::Str(value)
            | MirFmtChunk::Bytes(value)
            | MirFmtChunk::EnumName { value, .. } => self.scan_expr_for_prims(value),
            MirFmtChunk::DebugSlice { slice, .. } => self.scan_expr_for_prims(slice),
            MirFmtChunk::Literal(_) => {}
        }
    }

    fn scan_expr_for_prims(&mut self, expr: &MirExpr) {
        self.scan_mir_type_for_prims(&expr.ty);
        match &expr.kind {
            MirExprKind::Unary { expr: e, .. }
            | MirExprKind::Cast { expr: e, .. }
            | MirExprKind::FieldAccess { base: e, .. }
            | MirExprKind::SliceLen(e)
            | MirExprKind::FromBytes { slice: e, .. } => self.scan_expr_for_prims(e),
            MirExprKind::Binary { lhs, rhs, .. }
            | MirExprKind::Index {
                base: lhs,
                index: rhs,
                ..
            } => {
                self.scan_expr_for_prims(lhs);
                self.scan_expr_for_prims(rhs);
            }
            MirExprKind::SubSlice {
                base, start, end, ..
            } => {
                self.scan_expr_for_prims(base);
                self.scan_expr_for_prims(start);
                self.scan_expr_for_prims(end);
            }
            MirExprKind::StructLit { fields, .. } => {
                for (_, fe, _) in fields {
                    self.scan_expr_for_prims(fe);
                }
            }
            MirExprKind::ArrayLit { elems, .. } | MirExprKind::CallDirect { args: elems, .. } => {
                for a in elems {
                    self.scan_expr_for_prims(a);
                }
            }
            MirExprKind::RomCallU32 { addr, args, .. } => {
                self.scan_expr_for_prims(addr);
                for a in args {
                    self.scan_expr_for_prims(a);
                }
            }
            MirExprKind::MakePrimView { ptr, cap, kind } => {
                self.scan_expr_for_prims(ptr);
                let n = (*cap).max(1);
                match kind {
                    MirPrimViewKind::ByteRing => {
                        self.uses_byte_ring_view = true;
                        self.register_prim_kind(&MirPrimKind::ByteRing { cap: n });
                    }
                    MirPrimViewKind::Inbox { elem } => {
                        self.register_inbox_view(elem);
                        self.register_prim_kind(&MirPrimKind::Inbox {
                            elem: elem.clone(),
                            cap: n,
                        });
                    }
                }
            }
            MirExprKind::PrimRead(ro) => match ro.as_ref() {
                MirPrimReadOp::InboxViewTryRecv { view, elem_ty }
                | MirPrimReadOp::InboxViewLen { view, elem_ty }
                | MirPrimReadOp::InboxViewCap { view, elem_ty }
                | MirPrimReadOp::InboxViewIsEmpty { view, elem_ty }
                | MirPrimReadOp::InboxViewIsFull { view, elem_ty } => {
                    self.scan_expr_for_prims(view);
                    self.register_inbox_view(elem_ty);
                }
                MirPrimReadOp::InboxViewTrySend {
                    view,
                    elem_ty,
                    value,
                    ..
                } => {
                    self.scan_expr_for_prims(view);
                    self.register_inbox_view(elem_ty);
                    self.scan_expr_for_prims(value);
                }
                MirPrimReadOp::ByteRingViewPush { view, src: val, .. }
                | MirPrimReadOp::ByteRingViewPop { view, dst: val }
                | MirPrimReadOp::ByteRingViewPushByte {
                    view, byte: val, ..
                }
                | MirPrimReadOp::ByteRingViewGrantWrite { view, len: val } => {
                    self.scan_expr_for_prims(view);
                    self.uses_byte_ring_view = true;
                    self.scan_expr_for_prims(val);
                }
                MirPrimReadOp::ByteRingViewReadSlice { view }
                | MirPrimReadOp::ByteRingViewTryPopByte { view }
                | MirPrimReadOp::ByteRingViewLen { view }
                | MirPrimReadOp::ByteRingViewCap { view }
                | MirPrimReadOp::ByteRingViewFreeSpace { view }
                | MirPrimReadOp::ByteRingViewIsEmpty { view }
                | MirPrimReadOp::ByteRingViewIsFull { view } => {
                    self.scan_expr_for_prims(view);
                    self.uses_byte_ring_view = true;
                }
                MirPrimReadOp::ByteRingPush {
                    target, src: val, ..
                }
                | MirPrimReadOp::ByteRingPop {
                    target, dst: val, ..
                }
                | MirPrimReadOp::ByteRingPushByte {
                    target, byte: val, ..
                }
                | MirPrimReadOp::ByteRingGrantWrite {
                    target, len: val, ..
                } => {
                    self.scan_prim_target_for_prims(target);
                    self.scan_expr_for_prims(val);
                }
                MirPrimReadOp::InboxTrySend {
                    target,
                    elem_ty,
                    cap,
                    value: val,
                    ..
                } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::Inbox {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                    self.scan_expr_for_prims(val);
                }
                MirPrimReadOp::PoolTryAlloc {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::PoolAvail {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::PoolInUse {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::PoolIsExhausted {
                    target,
                    elem_ty,
                    cap,
                }
                | MirPrimReadOp::PoolCap {
                    target,
                    elem_ty,
                    cap,
                } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::Pool {
                        elem: Box::new(elem_ty.clone()),
                        cap: *cap,
                    });
                }
                MirPrimReadOp::DmaSlotPark {
                    target,
                    elem_ty,
                    lease,
                } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::DmaSlot {
                        elem: Box::new(elem_ty.clone()),
                    });
                    self.scan_expr_for_prims(lease);
                }
                MirPrimReadOp::DmaSlotTake { target, elem_ty }
                | MirPrimReadOp::DmaSlotRelease {
                    target, elem_ty, ..
                }
                | MirPrimReadOp::DmaSlotIsBusy { target, elem_ty }
                | MirPrimReadOp::DmaSlotIsEmpty { target, elem_ty }
                | MirPrimReadOp::DmaSlotAddr { target, elem_ty } => {
                    self.scan_prim_target_for_prims(target);
                    self.register_prim_kind(&MirPrimKind::DmaSlot {
                        elem: Box::new(elem_ty.clone()),
                    });
                }
                _ => {}
            },
            MirExprKind::LeaseGet { lease, elem_ty }
            | MirExprKind::LeaseAddr { lease, elem_ty } => {
                self.uses_pool_lease = true;
                self.scan_mir_type_for_prims(elem_ty);
                self.scan_expr_for_prims(lease);
            }
            MirExprKind::MmioRead { base, index, .. } => {
                self.scan_expr_for_prims(base);
                if let Some(i) = index {
                    self.scan_expr_for_prims(i);
                }
            }
            _ => {}
        }
    }

    fn compute_fn_frame_layout<'b>(
        &mut self,
        name: &str,
        fn_map: &HashMap<&'b str, &'b MirFunction>,
        done: &mut BTreeSet<String>,
    ) {
        if done.contains(name) {
            return;
        }
        done.insert(name.to_string());
        let Some(&func) = fn_map.get(name) else {
            return;
        };
        if let MirFnBody::Coroutine { child_slots, .. } = &func.body {
            for slot in child_slots {
                self.compute_fn_frame_layout(&slot.callee, fn_map, done);
            }
            let env_ty = func.envelope_return_type();
            let has_ret = !env_ty.is_void();
            let layout = self.build_coroutine_layout(
                &format!("%FnFrame_{}", sanitize_ident(&func.name)),
                has_ret,
                &env_ty,
                &func.vars,
                child_slots,
            );
            self.fn_frames.insert(func.name.clone(), layout);
        }
    }

    fn build_coroutine_layout(
        &self,
        llvm_name: &str,
        has_ret_slot: bool,
        ret_env_ty: &MirType,
        vars: &[MirVar],
        child_slots: &[MirChildCallSlot],
    ) -> CoroutineFrameLayout {
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let mut field_llvm_tys = Vec::new();
        // Field 0 is always `i32` (`_pc`)
        field_llvm_tys.push("i32".to_string());
        let mut offset = 4usize;
        let mut max_align = 4usize;

        let ret_llvm_ty = if has_ret_slot {
            let (sz, al) = self.size_align(ret_env_ty, ptr_bytes);
            offset = align_up(offset, al);
            offset += sz;
            max_align = max_align.max(al);
            let rty = self.llvm_ty(ret_env_ty);
            field_llvm_tys.push(rty.clone());
            rty
        } else {
            "void".to_string()
        };

        let mut hoisted_var_gep = HashMap::new();
        for v in vars {
            if v.hoisted && !v.ty.is_void() {
                let gep_idx = field_llvm_tys.len();
                let (sz, al) = self.size_align(&v.ty, ptr_bytes);
                offset = align_up(offset, al);
                offset += sz;
                max_align = max_align.max(al);
                field_llvm_tys.push(self.llvm_ty(&v.ty));
                hoisted_var_gep.insert(v.id, gep_idx);
            }
        }

        // Group disjoint child slots by `union_group`
        let mut group_max_bytes: BTreeMap<usize, usize> = BTreeMap::new();
        let mut slot_to_group = HashMap::new();
        for slot in child_slots {
            let child_bytes = self
                .fn_frames
                .get(&slot.callee)
                .map_or(8, |cf| cf.byte_size_bound);
            let entry = group_max_bytes.entry(slot.union_group).or_insert(8);
            *entry = (*entry).max(child_bytes);
            slot_to_group.insert(slot.slot_name.clone(), slot.union_group);
        }

        let mut union_group_gep = BTreeMap::new();
        for (ug, max_b) in group_max_bytes {
            let words_i64 = ((max_b + 7) / 8).max(1);
            let gep_idx = field_llvm_tys.len();
            offset = align_up(offset, 8);
            offset += words_i64 * 8;
            max_align = max_align.max(8);
            field_llvm_tys.push(format!("[{words_i64} x i64]"));
            union_group_gep.insert(ug, (gep_idx, words_i64));
        }

        let byte_size_bound = align_up(offset, max_align).max(8);
        CoroutineFrameLayout {
            llvm_name: llvm_name.to_string(),
            has_ret_slot,
            ret_llvm_ty,
            hoisted_var_gep,
            union_group_gep,
            slot_to_group,
            byte_size_bound,
            field_llvm_tys,
        }
    }

    fn size_align(&self, ty: &MirType, ptr_bytes: usize) -> (usize, usize) {
        match ty {
            MirType::Void => (0, 1),
            MirType::Bool | MirType::U8 | MirType::I8 => (1, 1),
            MirType::U16 | MirType::I16 | MirType::EndianInt { bits: 16, .. } => (2, 2),
            MirType::U32
            | MirType::I32
            | MirType::F32
            | MirType::EndianInt { bits: 32, .. }
            | MirType::Enum(_) => (4, 4),
            MirType::U64 | MirType::I64 | MirType::F64 | MirType::EndianInt { bits: 64, .. } => {
                (8, 8)
            }
            MirType::EndianInt { .. } => (4, 4),
            MirType::Usize
            | MirType::Isize
            | MirType::Ptr(_)
            | MirType::MmioPtr(_)
            | MirType::FnPtr { .. }
            | MirType::Str => (ptr_bytes, ptr_bytes),
            MirType::Slice(_) => (ptr_bytes * 2, ptr_bytes),
            MirType::Array { elem, len } => {
                let (esz, eal) = self.size_align(elem, ptr_bytes);
                (esz * (*len).max(1), eal)
            }
            MirType::Struct(name) => {
                if let Some(si) = self.structs.get(name) {
                    (si.byte_size, si.align)
                } else {
                    (4, 4)
                }
            }
            MirType::Primitive(pk) => match pk {
                MirPrimKind::Signal => (2, 1),
                MirPrimKind::Watch { elem } => {
                    let (esz, eal) = self.size_align(elem, ptr_bytes);
                    let al = eal.max(4);
                    (align_up(esz, 4) + 4, al)
                }
                MirPrimKind::Inbox { elem, cap } => {
                    let (esz, eal) = self.size_align(elem, ptr_bytes);
                    let al = eal.max(4);
                    (align_up(13, eal) + esz * (*cap).max(1), al)
                }
                MirPrimKind::PingPong { elem, cap } => {
                    let (esz, eal) = self.size_align(elem, ptr_bytes);
                    (align_up(4, eal) + 2 * esz * (*cap).max(1), eal.max(4))
                }
                MirPrimKind::Pool { elem, cap } => {
                    let (hdr_sz, hdr_al) = self.pool_hdr_llvm_size();
                    let (esz, eal) = self.size_align(elem, ptr_bytes);
                    let al = eal.max(hdr_al);
                    (align_up(hdr_sz, eal) + esz * (*cap).max(1), al)
                }
                MirPrimKind::DmaSlot { .. } => {
                    let al = self.ptr_align();
                    (align_up(2 * ptr_bytes + 2, al.max(1)), al.max(1))
                }
                MirPrimKind::ByteRing { cap } => (28 + (*cap).max(1), 4),
            },
            MirType::Lease(_) => {
                let al = self.ptr_align();
                (align_up(2 * ptr_bytes + 1, al.max(1)), al.max(1))
            }
            MirType::PrimView(MirPrimViewKind::ByteRing) => (2 * ptr_bytes, ptr_bytes),
            MirType::PrimView(MirPrimViewKind::Inbox { .. }) => (3 * ptr_bytes, ptr_bytes),
            MirType::Result { ok, .. } => {
                let (osz, oal) = if ok.is_void() {
                    (4, 4)
                } else {
                    self.size_align(ok, ptr_bytes)
                };
                let al = oal.max(4);
                (align_up(osz, 4) + 4, al)
            }
        }
    }

    fn ptr_align(&self) -> usize {
        if self.spec.isa.is_avr() {
            1
        } else {
            (self.spec.isa.ptr_bits() / 8) as usize
        }
    }

    fn llvm_scalar_abi_align(&self, natural_align: usize) -> usize {
        if self.spec.isa.is_avr() {
            1
        } else {
            natural_align
        }
    }

    fn pool_hdr_llvm_size(&self) -> (usize, usize) {
        match self.spec.isa {
            CpuIsa::HostX86_64Linux => (24, 8),
            CpuIsa::ArmCortexM | CpuIsa::Riscv32 => (16, 4),
            CpuIsa::Avr => (12, 1),
        }
    }

    fn llvm_abi_size_align(&self, ty: &MirType, dma_safe: bool) -> (usize, usize) {
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let ptr_al = self.ptr_align();
        match ty {
            MirType::Void => (0, 1),
            MirType::Bool | MirType::U8 | MirType::I8 => (1, 1),
            MirType::U16 | MirType::I16 | MirType::EndianInt { bits: 16, .. } => {
                (2, self.llvm_scalar_abi_align(2))
            }
            MirType::U32
            | MirType::I32
            | MirType::F32
            | MirType::EndianInt { bits: 32, .. }
            | MirType::Enum(_) => (4, self.llvm_scalar_abi_align(4)),
            MirType::U64 | MirType::I64 | MirType::F64 | MirType::EndianInt { bits: 64, .. } => {
                (8, self.llvm_scalar_abi_align(8))
            }
            MirType::EndianInt { .. } => (4, self.llvm_scalar_abi_align(4)),
            MirType::Usize
            | MirType::Isize
            | MirType::Ptr(_)
            | MirType::MmioPtr(_)
            | MirType::FnPtr { .. }
            | MirType::Str => (ptr_bytes, ptr_al),
            MirType::Slice(_) | MirType::PrimView(MirPrimViewKind::ByteRing) => {
                (2 * ptr_bytes, ptr_al)
            }
            MirType::PrimView(MirPrimViewKind::Inbox { .. }) => (3 * ptr_bytes, ptr_al),
            MirType::Lease(_) => (align_up(2 * ptr_bytes + 1, ptr_al), ptr_al),
            MirType::Array { elem, len } => {
                let (esz, eal) = self.llvm_abi_size_align(elem, false);
                (esz * (*len).max(1), eal)
            }
            MirType::Struct(name) => {
                if let Some(si) = self.structs.get(name) {
                    (si.llvm_abi_size, si.llvm_nat_align)
                } else {
                    (4, self.llvm_scalar_abi_align(4))
                }
            }
            MirType::Primitive(pk) => match pk {
                MirPrimKind::Signal => (2, 1),
                MirPrimKind::Watch { elem } => {
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let i32_al = self.llvm_scalar_abi_align(4);
                    let al = eal.max(i32_al);
                    (align_up(align_up(esz, i32_al) + 4, al), al)
                }
                MirPrimKind::Inbox { elem, cap } => {
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let i32_al = self.llvm_scalar_abi_align(4);
                    let c = (*cap).max(1);
                    let al = eal.max(i32_al);
                    (align_up(align_up(12 + c, eal) + c * esz, al), al)
                }
                MirPrimKind::PingPong { elem, cap } => {
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let c = (*cap).max(1);
                    (align_up(4, eal) + 2 * c * esz, eal.max(1))
                }
                MirPrimKind::Pool { elem, cap } => {
                    let (hdr_sz, hdr_al) = self.pool_hdr_llvm_size();
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let c = (*cap).max(1);
                    if dma_safe {
                        let slot_stride = align_up(esz.max(1), eal.max(32));
                        let al = hdr_al.max(eal);
                        (align_up(32 + c * slot_stride, al), al)
                    } else {
                        let al = hdr_al.max(eal);
                        (align_up(align_up(hdr_sz, eal) + c * esz, al), al)
                    }
                }
                MirPrimKind::DmaSlot { .. } => (align_up(2 * ptr_bytes + 2, ptr_al), ptr_al),
                MirPrimKind::ByteRing { cap } => {
                    let i32_al = self.llvm_scalar_abi_align(4);
                    (align_up(25 + (*cap).max(1), i32_al), i32_al)
                }
            },
            MirType::Result { ok, .. } => {
                let i32_al = self.llvm_scalar_abi_align(4);
                let (osz, oal) = if ok.is_void() {
                    (4, i32_al)
                } else {
                    self.llvm_abi_size_align(ok, false)
                };
                let al = oal.max(i32_al);
                (align_up(align_up(osz, i32_al) + 4, al), al)
            }
        }
    }

    fn llvm_field_ty(&self, struct_name: &str, fld: &zev_mir::MirField) -> String {
        let eff_dma = fld.dma_safe || self.dma_layout_structs.contains(struct_name);
        if eff_dma {
            if let MirType::Primitive(pk @ MirPrimKind::Pool { .. }) = &fld.ty {
                return format!("%prim.{}_dma", sanitize_ident(&pk.mangle_name()));
            }
        }
        self.llvm_ty(&fld.ty)
    }

    fn llvm_ty(&self, ty: &MirType) -> String {
        match ty {
            MirType::Void => "void".to_string(),
            MirType::Bool | MirType::U8 | MirType::I8 => "i8".to_string(),
            MirType::U16 | MirType::I16 | MirType::EndianInt { bits: 16, .. } => "i16".to_string(),
            MirType::U32
            | MirType::I32
            | MirType::EndianInt { bits: 32, .. }
            | MirType::Enum(_) => "i32".to_string(),
            MirType::U64 | MirType::I64 | MirType::EndianInt { bits: 64, .. } => "i64".to_string(),
            MirType::F32 => "float".to_string(),
            MirType::F64 => "double".to_string(),
            MirType::EndianInt { .. } => "i32".to_string(),
            MirType::Usize | MirType::Isize => self.spec.isa.usize_llvm_ty().to_string(),
            MirType::Ptr(_) | MirType::MmioPtr(_) | MirType::Str => "ptr".to_string(),
            MirType::FnPtr { .. } => self.spec.isa.fn_ptr_llvm_ty().to_string(),
            MirType::Slice(_) => format!("{{ ptr, {} }}", self.spec.isa.usize_llvm_ty()),
            MirType::Array { elem, len } => {
                let n = (*len).max(1);
                format!("[{n} x {}]", self.llvm_ty(elem))
            }
            MirType::Struct(name) => format!("%struct.{}", sanitize_ident(name)),
            MirType::Primitive(pk) => format!("%prim.{}", sanitize_ident(&pk.mangle_name())),
            MirType::Lease(_) => "{ ptr, ptr, i8 }".to_string(),
            MirType::PrimView(MirPrimViewKind::ByteRing) => {
                format!("{{ ptr, {} }}", self.spec.isa.usize_llvm_ty())
            }
            MirType::PrimView(MirPrimViewKind::Inbox { .. }) => {
                format!("{{ ptr, ptr, {} }}", self.spec.isa.usize_llvm_ty())
            }
            MirType::Result { ok, .. } => {
                let ok_ty = if ok.is_void() {
                    "i32".to_string()
                } else {
                    self.llvm_ty(ok)
                };
                format!("{{ {ok_ty}, i32 }}")
            }
        }
    }

    fn intern_string(&mut self, s: &str) -> usize {
        if let Some(&idx) = self.string_dedup.get(s) {
            return idx;
        }
        let idx = self.strings.len();
        self.strings.push(s.to_string());
        self.string_dedup.insert(s.to_string(), idx);
        idx
    }

    fn intern_const_blob(&mut self, decl_body: &str) -> usize {
        if let Some(&idx) = self.const_blob_dedup.get(decl_body) {
            return idx;
        }
        let idx = self.const_blobs.len();
        self.const_blobs.push(decl_body.to_string());
        self.const_blob_dedup.insert(decl_body.to_string(), idx);
        idx
    }

    fn eval_const_int(&self, expr: &MirExpr) -> i128 {
        match &expr.kind {
            MirExprKind::IntLit(v) => *v,
            MirExprKind::BoolLit(b) => i128::from(*b),
            MirExprKind::TicksLit(t) => *t as i128,
            MirExprKind::EnumVariant {
                enum_name,
                variant,
            } => self
                .enum_variants
                .get(&(enum_name.clone(), variant.clone()))
                .or_else(|| self.bare_variants.get(variant))
                .copied()
                .unwrap_or(0) as i128,
            MirExprKind::ConstRef(name) => self.consts.get(name).copied().unwrap_or(1),
            MirExprKind::Unary { op, expr: sub } => {
                let v = self.eval_const_int(sub);
                match op {
                    MirUnaryOp::Neg => -v,
                    MirUnaryOp::Not => i128::from(v == 0),
                    MirUnaryOp::BitNot => !v,
                    MirUnaryOp::Deref => v,
                }
            }
            MirExprKind::Binary { op, lhs, rhs } => {
                if lhs.ty.is_float() || rhs.ty.is_float() {
                    let lf = f64::from_bits(self.eval_const_float_bits(lhs));
                    let rf = f64::from_bits(self.eval_const_float_bits(rhs));
                    return match op {
                        MirBinOp::Eq => i128::from(lf == rf),
                        MirBinOp::Ne => i128::from(lf != rf),
                        MirBinOp::Lt => i128::from(lf < rf),
                        MirBinOp::Le => i128::from(lf <= rf),
                        MirBinOp::Gt => i128::from(lf > rf),
                        MirBinOp::Ge => i128::from(lf >= rf),
                        _ => 0,
                    };
                }
                let l = self.eval_const_int(lhs);
                let r = self.eval_const_int(rhs);
                match op {
                    MirBinOp::Add => l.wrapping_add(r),
                    MirBinOp::Sub => l.wrapping_sub(r),
                    MirBinOp::Mul => l.wrapping_mul(r),
                    MirBinOp::Div => {
                        if r != 0 {
                            l / r
                        } else {
                            0
                        }
                    }
                    MirBinOp::Rem => {
                        if r != 0 {
                            l % r
                        } else {
                            0
                        }
                    }
                    MirBinOp::BitOr => l | r,
                    MirBinOp::BitAnd => l & r,
                    MirBinOp::BitXor => l ^ r,
                    MirBinOp::Shl => l << (r & 63),
                    MirBinOp::Shr => {
                        let sh = (r as u32) & 63;
                        if lhs.ty.is_unsigned() {
                            let ptr_bits = self.spec.isa.ptr_bits();
                            let masked = if lhs.ty == MirType::Bool {
                                (l as u128) & 1
                            } else {
                                match target_int_signed_and_bits(&lhs.ty, ptr_bits) {
                                    Some((_, 8)) => (l as u8) as u128,
                                    Some((_, 16)) => (l as u16) as u128,
                                    Some((_, 32)) => (l as u32) as u128,
                                    Some((_, 64)) => (l as u64) as u128,
                                    _ => l as u128,
                                }
                            };
                            (masked >> sh) as i128
                        } else {
                            l >> sh
                        }
                    }
                    MirBinOp::Eq => i128::from(l == r),
                    MirBinOp::Ne => i128::from(l != r),
                    MirBinOp::Lt => i128::from(l < r),
                    MirBinOp::Le => i128::from(l <= r),
                    MirBinOp::Gt => i128::from(l > r),
                    MirBinOp::Ge => i128::from(l >= r),
                    MirBinOp::LogicalAnd => i128::from(l != 0 && r != 0),
                    MirBinOp::LogicalOr => i128::from(l != 0 || r != 0),
                }
            }
            MirExprKind::Cast {
                expr: sub,
                target_ty,
            } => {
                let ptr_bits = self.spec.isa.ptr_bits();
                if *target_ty == MirType::Bool {
                    if sub.ty.is_float() {
                        let f = f64::from_bits(self.eval_const_float_bits(sub));
                        return i128::from(f != 0.0);
                    }
                    return i128::from(self.eval_const_int(sub) != 0);
                }
                if sub.ty.is_float() {
                    let raw = f64::from_bits(self.eval_const_float_bits(sub));
                    let f = if sub.ty == MirType::F32 {
                        (raw as f32) as f64
                    } else {
                        raw
                    };
                    return saturating_float_to_int_i128(f, target_ty, ptr_bits);
                }
                mask_int_to_target_i128(self.eval_const_int(sub), target_ty, ptr_bits)
            }
            _ => 0,
        }
    }

    fn eval_const_float_bits(&self, expr: &MirExpr) -> u64 {
        let raw_f = match &expr.kind {
            MirExprKind::FloatLit(bits) => f64::from_bits(*bits),
            MirExprKind::IntLit(v) => *v as f64,
            MirExprKind::ConstRef(name) => self
                .float_consts
                .get(name)
                .copied()
                .map(f64::from_bits)
                .or_else(|| self.consts.get(name).copied().map(|v| v as f64))
                .unwrap_or(0.0),
            MirExprKind::Unary {
                op: MirUnaryOp::Neg,
                expr: sub,
            } => -f64::from_bits(self.eval_const_float_bits(sub)),
            MirExprKind::Binary { op, lhs, rhs } => {
                let l = f64::from_bits(self.eval_const_float_bits(lhs));
                let r = f64::from_bits(self.eval_const_float_bits(rhs));
                match op {
                    MirBinOp::Add => l + r,
                    MirBinOp::Sub => l - r,
                    MirBinOp::Mul => l * r,
                    MirBinOp::Div => l / r,
                    MirBinOp::Rem => l % r,
                    _ => 0.0,
                }
            }
            MirExprKind::Cast { expr: sub, .. } => {
                if sub.ty.is_float() {
                    f64::from_bits(self.eval_const_float_bits(sub))
                } else {
                    self.eval_const_int(sub) as f64
                }
            }
            _ => 0.0,
        };
        if expr.ty == MirType::F32 {
            ((raw_f as f32) as f64).to_bits()
        } else {
            raw_f.to_bits()
        }
    }

    // ── Section Emitters ─────────────────────────────────────────────────────

    fn emit_struct_types_ir(&self, out: &mut String) {
        let has_val_structs = self
            .struct_order
            .iter()
            .any(|name| !self.structs[name].decl.is_subsystem);
        if !has_val_structs {
            return;
        }
        let _ = writeln!(out, "; ── User Struct & Wire-Exact Packed Struct Types ──");
        for name in &self.struct_order {
            let si = &self.structs[name];
            if si.decl.is_subsystem {
                continue;
            }
            let fields_str: Vec<String> = if si.decl.fields.is_empty() {
                vec!["i8".to_string()]
            } else {
                si.decl.fields.iter().map(|f| self.llvm_ty(&f.ty)).collect()
            };
            let joined = fields_str.join(", ");
            if si.decl.packed {
                let _ = writeln!(out, "{} = type <{{ {joined} }}>", si.llvm_name);
            } else {
                let _ = writeln!(out, "{} = type {{ {joined} }}", si.llvm_name);
            }
        }
        let _ = writeln!(out);
    }

    fn emit_mmio_host_buffers_ir(&self, out: &mut String) {
        if self.spec.isa != CpuIsa::HostX86_64Linux || self.module.mmio_instances.is_empty() {
            return;
        }
        let _ = writeln!(out, "; ── Host Simulation MMIO Backing Buffers ──");
        for inst in &self.module.mmio_instances {
            let layout_sz = self
                .mmio_layouts
                .get(&inst.layout)
                .map_or(0, |l| l.byte_size);
            let buf_len = inst.byte_size.max(layout_sz).max(4);
            let sym = format!("@__zev_mmio_buf_{}", sanitize_ident(&inst.name));
            let _ = writeln!(
                out,
                "{sym} = internal global [{buf_len} x i8] zeroinitializer, align 4"
            );
        }
        let _ = writeln!(out);
    }

    fn emit_isr_state_and_selector_globals_ir(&mut self, out: &mut String) {
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let mut emitted_state = BTreeSet::new();
        let mut state_lines = Vec::new();
        let mut selector_syms = BTreeSet::new();

        for vec in &self.module.irq_vectors.clone() {
            if let Some(sel) = &vec.selector_symbol {
                selector_syms.insert(sel.trim_start_matches('@').to_string());
            }
            for b in &vec.bindings {
                if let Some((sel, _)) = &b.cond_selector {
                    selector_syms.insert(sel.trim_start_matches('@').to_string());
                }
                for sv in &b.state_vars {
                    let sym = sanitize_ident(sv.storage_symbol.trim_start_matches('@'));
                    if emitted_state.insert(sym.clone()) {
                        let lty = self.llvm_ty(&sv.ty);
                        let (_, align) = self.size_align(&sv.ty, ptr_bytes);
                        let init_val = self.const_expr_to_llvm(&sv.init, &sv.ty);
                        state_lines.push(format!(
                            "@{sym} = internal global {lty} {init_val}, align {}",
                            align.max(1)
                        ));
                    }
                }
            }
        }

        fn scan_stmt_selectors(stmts: &[MirStmt], out: &mut BTreeSet<String>) {
            for s in stmts {
                match s {
                    MirStmt::SetIrqSelector { symbol, .. } => {
                        out.insert(symbol.trim_start_matches('@').to_string());
                    }
                    MirStmt::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        scan_stmt_selectors(then_body, out);
                        scan_stmt_selectors(else_body, out);
                    }
                    MirStmt::While { body, .. } => scan_stmt_selectors(body, out),
                    MirStmt::Switch { arms, default, .. } => {
                        for (_, ab) in arms {
                            scan_stmt_selectors(ab, out);
                        }
                        scan_stmt_selectors(default, out);
                    }
                    _ => {}
                }
            }
        }
        for step in &self.module.boot.steps {
            if let MirBootStep::Stmt(s) = step {
                scan_stmt_selectors(std::slice::from_ref(s), &mut selector_syms);
            }
        }
        for f in &self.module.functions {
            if let MirFnBody::Direct(stmts) = &f.body {
                scan_stmt_selectors(stmts, &mut selector_syms);
            }
        }

        if !state_lines.is_empty() || !selector_syms.is_empty() {
            let _ = writeln!(out, "; ── Per-Binding ISR Static State & Conditional IRQ Selectors ──");
            for line in state_lines {
                let _ = writeln!(out, "{line}");
            }
            for raw_sel in selector_syms {
                let sym = sanitize_ident(&raw_sel);
                let _ = writeln!(out, "@{sym} = internal global i8 0, align 1");
            }
            let _ = writeln!(out);
        }
    }

    fn const_expr_to_llvm(&mut self, expr: &MirExpr, target_ty: &MirType) -> String {
        if let MirExprKind::ConstRef(cname) = &expr.kind {
            if let Some(init) = self
                .module
                .consts
                .iter()
                .find(|c| c.name == *cname)
                .map(|c| c.init.clone())
            {
                return self.const_expr_to_llvm(&init, target_ty);
            }
        }
        let ptr_bits = self.spec.isa.ptr_bits();
        match target_ty {
            MirType::Bool => {
                if self.eval_const_int(expr) != 0 {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
            MirType::U8 => format!("{}", self.eval_const_int(expr) as u8),
            MirType::I8 => format!("{}", self.eval_const_int(expr) as i8),
            MirType::U16
            | MirType::EndianInt {
                bits: 16,
                signed: false,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as u16;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                format!("{v}")
            }
            MirType::I16
            | MirType::EndianInt {
                bits: 16,
                signed: true,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as i16;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                format!("{v}")
            }
            MirType::U32
            | MirType::EndianInt {
                bits: 32,
                signed: false,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as u32;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                format!("{v}")
            }
            MirType::I32
            | MirType::Enum(_)
            | MirType::EndianInt {
                bits: 32,
                signed: true,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as i32;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                format!("{v}")
            }
            MirType::U64
            | MirType::EndianInt {
                bits: 64,
                signed: false,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as u64;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                format!("{v}")
            }
            MirType::I64
            | MirType::EndianInt {
                bits: 64,
                signed: true,
                ..
            } => {
                let mut v = self.eval_const_int(expr) as i64;
                if matches!(
                    target_ty,
                    MirType::EndianInt {
                        big_endian: true,
                        ..
                    }
                ) {
                    v = v.swap_bytes();
                }
                format!("{v}")
            }
            MirType::Usize => {
                if ptr_bits == 64 {
                    format!("{}", self.eval_const_int(expr) as u64)
                } else if ptr_bits == 16 {
                    format!("{}", self.eval_const_int(expr) as u16)
                } else {
                    format!("{}", self.eval_const_int(expr) as u32)
                }
            }
            MirType::Isize => {
                if ptr_bits == 64 {
                    format!("{}", self.eval_const_int(expr) as i64)
                } else if ptr_bits == 16 {
                    format!("{}", self.eval_const_int(expr) as i16)
                } else {
                    format!("{}", self.eval_const_int(expr) as i32)
                }
            }
            MirType::F32 => {
                let f = f64::from_bits(self.eval_const_float_bits(expr)) as f32;
                let norm = if f.is_nan() {
                    0x7FF8_0000_0000_0000u64
                } else {
                    (f as f64).to_bits()
                };
                format!("0x{norm:016X}")
            }
            MirType::F64 => {
                let bits = self.eval_const_float_bits(expr);
                format!("0x{bits:016X}")
            }
            MirType::Ptr(_) | MirType::MmioPtr(_) | MirType::FnPtr { .. } | MirType::Str => {
                let inner = match &expr.kind {
                    MirExprKind::Cast { expr: sub, .. } => sub.as_ref(),
                    _ => expr,
                };
                if let MirExprKind::ConstRef(cname) = &inner.kind {
                    if let Some(init) = self
                        .module
                        .consts
                        .iter()
                        .find(|c| c.name == *cname)
                        .map(|c| c.init.clone())
                    {
                        return self.const_expr_to_llvm(&init, target_ty);
                    }
                }
                match &inner.kind {
                    MirExprKind::GlobalPrimAddr(name) => {
                        format!("@__syn_prim_{}", sanitize_ident(name))
                    }
                    MirExprKind::SubsystemAddr(name) => {
                        format!("@__zev_subsys_{}", sanitize_ident(name))
                    }
                    MirExprKind::AddrOfPlace(place) => {
                        self.const_place_to_llvm_ptr(place).unwrap_or_else(|| "null".to_string())
                    }
                    MirExprKind::MmioAddr(inst) => {
                        if self.spec.isa == CpuIsa::HostX86_64Linux {
                            format!("@__zev_mmio_buf_{}", sanitize_ident(inst))
                        } else {
                            let addr = self
                                .mmio_instances
                                .get(inst)
                                .map_or(0, |m| m.base_addr as i32);
                            format!("inttoptr (i32 {addr} to ptr)")
                        }
                    }
                    MirExprKind::StringLit(s) => {
                        let idx = self.intern_string(s);
                        format!("@.str.{idx}")
                    }
                    MirExprKind::FnAddr(fname) => {
                        if let Some(sig) = self.fn_sigs.get(fname) {
                            sig.symbol.clone()
                        } else {
                            format!("@{}", sanitize_ident(fname))
                        }
                    }
                    MirExprKind::IntLit(0) => "null".to_string(),
                    MirExprKind::IntLit(v) => {
                        let dst_ptr = if matches!(target_ty, MirType::FnPtr { .. }) {
                            self.spec.isa.fn_ptr_llvm_ty()
                        } else {
                            "ptr"
                        };
                        if ptr_bits == 64 {
                            format!("inttoptr (i64 {} to {dst_ptr})", *v as i64)
                        } else if ptr_bits == 16 {
                            format!("inttoptr (i16 {} to {dst_ptr})", *v as i16)
                        } else {
                            format!("inttoptr (i32 {} to {dst_ptr})", *v as i32)
                        }
                    }
                    _ => "null".to_string(),
                }
            }
            MirType::Slice(elem) => {
                let usize_ty = self.spec.isa.usize_llvm_ty();
                match &expr.kind {
                    MirExprKind::Cast { expr: sub, .. } => self.const_expr_to_llvm(sub, target_ty),
                    MirExprKind::StringLit(s) => {
                        let idx = self.intern_string(s);
                        let slen = zev_ast::decoded_string_byte_len(s);
                        format!("{{ ptr @.str.{idx}, {usize_ty} {slen} }}")
                    }
                    MirExprKind::ArrayLit { elems, .. } => {
                        let n = elems.len();
                        if n == 0 {
                            format!("{{ ptr null, {usize_ty} 0 }}")
                        } else {
                            let ety = self.llvm_ty(elem);
                            let (_, eal) = self.size_align(elem, (ptr_bits / 8) as usize);
                            let mut parts = Vec::with_capacity(n);
                            for el in elems {
                                let cv = self.const_expr_to_llvm(el, elem);
                                parts.push(format!("{ety} {cv}"));
                            }
                            let decl_body = format!(
                                "private unnamed_addr constant [{n} x {ety}] [{}], align {eal}",
                                parts.join(", ")
                            );
                            let idx = self.intern_const_blob(&decl_body);
                            format!("{{ ptr @.cblob.{idx}, {usize_ty} {n} }}")
                        }
                    }
                    MirExprKind::ArrayToSlice {
                        array_place, len, ..
                    } => {
                        if let Some(aptr) = self.const_place_to_llvm_ptr(array_place) {
                            format!("{{ ptr {aptr}, {usize_ty} {len} }}")
                        } else {
                            "zeroinitializer".to_string()
                        }
                    }
                    _ => "zeroinitializer".to_string(),
                }
            }
            MirType::Array { elem, len } => {
                let n = (*len).max(1);
                if let MirExprKind::ArrayLit { elems, .. } = &expr.kind {
                    let ety = self.llvm_ty(elem);
                    let mut parts = Vec::with_capacity(n);
                    for i in 0..n {
                        if let Some(el) = elems.get(i) {
                            let cv = self.const_expr_to_llvm(el, elem);
                            parts.push(format!("{ety} {cv}"));
                        } else {
                            parts.push(format!("{ety} zeroinitializer"));
                        }
                    }
                    format!("[{}]", parts.join(", "))
                } else {
                    "zeroinitializer".to_string()
                }
            }
            MirType::Struct(sname) => {
                if let MirExprKind::StructLit { fields, .. } = &expr.kind {
                    if let Some(si) = self.structs.get(sname).cloned() {
                        if si.decl.fields.is_empty() {
                            return "zeroinitializer".to_string();
                        }
                        let by_field: HashMap<&str, &MirExpr> = fields
                            .iter()
                            .map(|(fname, fexpr, _)| (fname.as_str(), fexpr))
                            .collect();
                        let mut parts = Vec::with_capacity(si.decl.fields.len());
                        for fld in &si.decl.fields {
                            let fty = self.llvm_ty(&fld.ty);
                            if let Some(&fexpr) = by_field.get(fld.name.as_str()) {
                                let cv = self.const_expr_to_llvm(fexpr, &fld.ty);
                                parts.push(format!("{fty} {cv}"));
                            } else {
                                parts.push(format!("{fty} zeroinitializer"));
                            }
                        }
                        if si.decl.packed {
                            format!("<{{ {} }}>", parts.join(", "))
                        } else {
                            format!("{{ {} }}", parts.join(", "))
                        }
                    } else {
                        "zeroinitializer".to_string()
                    }
                } else {
                    "zeroinitializer".to_string()
                }
            }
            MirType::PrimView(vk) => {
                if let MirExprKind::MakePrimView { ptr, cap, .. } = &expr.kind {
                    let usize_ty = self.spec.isa.usize_llvm_ty();
                    let pconst = self.const_expr_to_llvm(ptr, &ptr.ty);
                    let n = (*cap).max(1);
                    match vk {
                        MirPrimViewKind::ByteRing => format!("{{ ptr {pconst}, {usize_ty} {n} }}"),
                        MirPrimViewKind::Inbox { elem } => {
                            let pty = format!(
                                "%prim.{}",
                                sanitize_ident(
                                    &MirPrimKind::Inbox {
                                        elem: elem.clone(),
                                        cap: n,
                                    }
                                    .mangle_name()
                                )
                            );
                            format!(
                                "{{ ptr {pconst}, ptr getelementptr inbounds ({pty}, ptr {pconst}, i32 0, i32 4, i32 0), {usize_ty} {n} }}"
                            )
                        }
                    }
                } else {
                    "zeroinitializer".to_string()
                }
            }
            _ => "zeroinitializer".to_string(),
        }
    }

    fn const_place_to_llvm_ptr(&self, place: &MirPlace) -> Option<String> {
        match place {
            MirPlace::Deref(ptr_expr, _) => match &ptr_expr.kind {
                MirExprKind::SubsystemAddr(name) => {
                    Some(format!("@__zev_subsys_{}", sanitize_ident(name)))
                }
                MirExprKind::GlobalPrimAddr(name) => {
                    Some(format!("@__syn_prim_{}", sanitize_ident(name)))
                }
                MirExprKind::AddrOfPlace(inner) => self.const_place_to_llvm_ptr(inner),
                _ => None,
            },
            MirPlace::Field {
                base,
                struct_name,
                field,
                ..
            } => {
                let bptr = self.const_place_to_llvm_ptr(base)?;
                let si = self.structs.get(struct_name)?;
                let (fidx, _) = si.field_indices.get(field)?;
                Some(format!(
                    "getelementptr inbounds ({}, ptr {bptr}, i32 0, i32 {fidx})",
                    si.llvm_name
                ))
            }
            MirPlace::Index {
                base,
                index,
                elem_ty,
                is_slice: false,
            } => {
                let bptr = self.const_place_to_llvm_ptr(base)?;
                let idx = self.eval_const_int(index);
                let ety = self.llvm_ty(elem_ty);
                Some(format!("getelementptr ({ety}, ptr {bptr}, i32 {idx})"))
            }
            _ => None,
        }
    }

    fn struct_needs_const_init(&self, struct_name: &str) -> bool {
        let Some(si) = self.structs.get(struct_name) else {
            return false;
        };
        si.decl.fields.iter().any(|f| match &f.ty {
            MirType::Primitive(MirPrimKind::Signal | MirPrimKind::PingPong { .. }) => true,
            MirType::Struct(sub) => self.struct_needs_const_init(sub),
            _ => false,
        })
    }

    fn subsystem_zero_init_const(&self, struct_name: &str, next_pp_bit: &mut u8) -> String {
        if !self.struct_needs_const_init(struct_name) {
            return "zeroinitializer".to_string();
        }
        let Some(si) = self.structs.get(struct_name) else {
            return "zeroinitializer".to_string();
        };
        if si.llvm_fields.is_empty() {
            return "zeroinitializer".to_string();
        }
        let parts: Vec<String> = si
            .llvm_fields
            .iter()
            .map(|(decl_idx, lty)| match decl_idx {
                Some(idx) => {
                    let f = &si.decl.fields[*idx];
                    match &f.ty {
                        MirType::Primitive(MirPrimKind::Signal) => {
                            format!("{lty} {{ i8 0, i8 127 }}")
                        }
                        MirType::Primitive(MirPrimKind::PingPong { elem, cap }) => {
                            let ety = self.llvm_ty(elem);
                            let c = (*cap).max(1);
                            let lbit = *next_pp_bit;
                            *next_pp_bit = next_pp_bit.saturating_add(1).min(31);
                            format!(
                                "{lty} {{ i8 0, i8 0, i8 0, i8 {lbit}, [{c} x {ety}] zeroinitializer, [{c} x {ety}] zeroinitializer }}"
                            )
                        }
                        MirType::Struct(sub) => {
                            format!("{lty} {}", self.subsystem_zero_init_const(sub, next_pp_bit))
                        }
                        _ => format!("{lty} zeroinitializer"),
                    }
                }
                None => format!("{lty} zeroinitializer"),
            })
            .collect();
        format!("{{ {} }}", parts.join(", "))
    }

    fn wrap_const_ptr_as_view(&self, raw_ptr: &str, pk: &MirPrimKind) -> Option<String> {
        let usize_ty = self.spec.isa.usize_llvm_ty();
        match pk {
            MirPrimKind::ByteRing { cap } => {
                let n = (*cap).max(1);
                Some(format!("{{ ptr, {usize_ty} }} {{ ptr {raw_ptr}, {usize_ty} {n} }}"))
            }
            MirPrimKind::Inbox { elem, cap } => {
                let n = (*cap).max(1);
                let pty = format!(
                    "%prim.{}",
                    sanitize_ident(
                        &MirPrimKind::Inbox {
                            elem: elem.clone(),
                            cap: n,
                        }
                        .mangle_name()
                    )
                );
                Some(format!(
                    "{{ ptr, ptr, {usize_ty} }} {{ ptr {raw_ptr}, ptr getelementptr inbounds ({pty}, ptr {raw_ptr}, i32 0, i32 4, i32 0), {usize_ty} {n} }}"
                ))
            }
            _ => None,
        }
    }

    fn emit_from_bytes_helpers_ir(&self, out: &mut String) {
        if self.struct_order.is_empty() {
            return;
        }
        let usize_ty = self.spec.isa.usize_llvm_ty();
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let tmp_al = if self.spec.isa.is_avr() {
            1
        } else {
            4
        };
        for name in &self.struct_order {
            let si = &self.structs[name];
            if si.decl.is_subsystem {
                continue;
            }
            let sname = sanitize_ident(name);
            let sty = &si.llvm_name;
            let sz = si.byte_size;
            let _ = writeln!(
                out,
                r#"define {{ {sty}, i32 }} @__syn_from_bytes_{sname}({{ ptr, {usize_ty} }} %slice){fn_as} nounwind{fn_sz} {{
entry:
  %tmp = alloca {sty}, align {tmp_al}
  %ptr = extractvalue {{ ptr, {usize_ty} }} %slice, 0
  %len = extractvalue {{ ptr, {usize_ty} }} %slice, 1
  %short = icmp ult {usize_ty} %len, {sz}
  br i1 %short, label %err_bb, label %ok_bb
err_bb:
  ret {{ {sty}, i32 }} {{ {sty} zeroinitializer, i32 1 }}
ok_bb:
  call void @llvm.memcpy.p0.p0.i32(ptr %tmp, ptr %ptr, i32 {sz}, i1 false)
  %val = load {sty}, ptr %tmp, align 1
  %r0 = insertvalue {{ {sty}, i32 }} poison, {sty} %val, 0
  %r1 = insertvalue {{ {sty}, i32 }} %r0, i32 0, 1
  ret {{ {sty}, i32 }} %r1
}}
"#
            );
        }
    }

    fn emit_formatted_enum_names_ir(&mut self, out: &mut String) {
        if self.module.formatted_enums.is_empty() {
            return;
        }
        let fn_sz = self.opts.fn_size_attrs().to_string();
        let fn_as = self.fn_as();
        let formatted: Vec<String> = self.module.formatted_enums.clone();
        let _ = writeln!(out, "; ── On-Demand Enum Variant-Name `.rodata` Lookup Helpers ──");
        for ename in &formatted {
            let Some(en) = self.module.enums.iter().find(|e| &e.name == ename).cloned() else {
                continue;
            };
            let eid = sanitize_ident(&en.name);
            let unk_idx = self.intern_string(&format!("{}(?)", en.name));
            let mut seen_discs = BTreeSet::new();
            let mut arms: Vec<(i64, usize)> = Vec::new();
            for (vname, disc) in &en.variants {
                if seen_discs.insert(*disc) {
                    let sidx = self.intern_string(vname);
                    arms.push((*disc, sidx));
                }
            }
            let _ = writeln!(
                out,
                "define ptr @__syn_enum_name_{eid}(i32 %v){fn_as} nounwind{fn_sz} {{"
            );
            let _ = writeln!(out, "entry:");
            let _ = writeln!(out, "  switch i32 %v, label %def [");
            for (idx, (disc, _)) in arms.iter().enumerate() {
                let _ = writeln!(out, "    i32 {disc}, label %arm_{idx}");
            }
            let _ = writeln!(out, "  ]");
            for (idx, (_, sidx)) in arms.iter().enumerate() {
                let _ = writeln!(out, "arm_{idx}:");
                let _ = writeln!(out, "  ret ptr @.str.{sidx}");
            }
            let _ = writeln!(out, "def:");
            let _ = writeln!(out, "  ret ptr @.str.{unk_idx}");
            let _ = writeln!(out, "}}\n");
        }
    }

    fn emit_primitives_ir(&mut self, out: &mut String) {
        if self.distinct_prims.is_empty()
            && self.module.primitives.is_empty()
            && !self.uses_pool_lease
            && self.module.subsystems.is_empty()
            && self.distinct_inbox_views.is_empty()
        {
            return;
        }
        let fn_as = self.fn_as();
        let call_kw = self.call_kw();
        let vis = self.vis.clone();
        let _ = writeln!(out, "; ── Static OS & Zero-Copy DMA Primitive Endpoints ──");
        if self.uses_pool_lease {
            let _ = writeln!(out, "%prim.pool_hdr = type {{ i32, i32, ptr, i8, i8 }}");
        }
        for (mangled, pk) in &self.distinct_prims {
            let id = sanitize_ident(mangled);
            match pk {
                MirPrimKind::Signal => {
                    let _ = writeln!(out, "%prim.{id} = type {{ i8, i8 }}");
                }
                MirPrimKind::Watch { elem } => {
                    let ety = self.llvm_ty(elem);
                    let _ = writeln!(out, "%prim.{id} = type {{ {ety}, i32 }}");
                }
                MirPrimKind::Inbox { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1);
                    let _ = writeln!(
                        out,
                        "%prim.{id} = type {{ i32, i32, i32, [{c} x i8], [{c} x {ety}] }}"
                    );
                }
                MirPrimKind::PingPong { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1);
                    let _ = writeln!(
                        out,
                        "%prim.{id} = type {{ i8, i8, i8, i8, [{c} x {ety}], [{c} x {ety}] }}"
                    );
                }
                MirPrimKind::Pool { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1);
                    let _ = writeln!(out, "%prim.{id} = type {{ %prim.pool_hdr, [{c} x {ety}] }}");
                    let (esz, eal) = self.llvm_abi_size_align(elem, false);
                    let (hdr_sz, _) = self.pool_hdr_llvm_size();
                    let stride = align_up(esz.max(1), eal.max(32));
                    let pad = stride.saturating_sub(esz);
                    if pad > 0 {
                        let _ = writeln!(
                            out,
                            "%prim.{id}_dma_slot = type {{ {ety}, [{pad} x i8] }}"
                        );
                    } else {
                        let _ = writeln!(out, "%prim.{id}_dma_slot = type {{ {ety} }}");
                    }
                    let hdr_pad = 32usize.saturating_sub(hdr_sz);
                    if hdr_pad > 0 {
                        let _ = writeln!(
                            out,
                            "%prim.{id}_dma = type {{ %prim.pool_hdr, [{hdr_pad} x i8], [{c} x %prim.{id}_dma_slot] }}"
                        );
                    } else {
                        let _ = writeln!(
                            out,
                            "%prim.{id}_dma = type {{ %prim.pool_hdr, [{c} x %prim.{id}_dma_slot] }}"
                        );
                    }
                }
                MirPrimKind::DmaSlot { .. } => {
                    let _ = writeln!(out, "%prim.{id} = type {{ ptr, ptr, i8, i8 }}");
                }
                MirPrimKind::ByteRing { cap } => {
                    let c = (*cap).max(1);
                    let _ = writeln!(
                        out,
                        "%prim.{id} = type {{ i32, i32, i32, i32, i32, i32, i8, [{c} x i8] }}"
                    );
                }
            }
        }
        if self.uses_byte_ring_view {
            let _ = writeln!(
                out,
                "%prim.byte_ring_hdr = type {{ i32, i32, i32, i32, i32, i32, i8, [0 x i8] }}"
            );
        }
        if !self.distinct_inbox_views.is_empty() {
            let _ = writeln!(out, "%prim.inbox_hdr = type {{ i32, i32, i32, [0 x i8] }}");
        }
        for name in &self.struct_order {
            let si = &self.structs[name];
            if !si.decl.is_subsystem {
                continue;
            }
            let fields_str: Vec<String> = if si.llvm_fields.is_empty() {
                vec!["i8".to_string()]
            } else {
                si.llvm_fields.iter().map(|(_, lty)| lty.clone()).collect()
            };
            let joined = fields_str.join(", ");
            let _ = writeln!(out, "{} = type {{ {joined} }}", si.llvm_name);
        }
        let _ = writeln!(out);

        // Emit static subsystem instances
        let mut next_pp_bit: u8 = self
            .module
            .primitives
            .iter()
            .filter_map(|p| p.sleep_lock_bit)
            .map(|b| b.saturating_add(1))
            .max()
            .unwrap_or(0)
            .min(31);
        for sub in &self.module.subsystems {
            let gsym = format!("@__zev_subsys_{}", sanitize_ident(&sub.name));
            let si = &self.structs[&sub.struct_name];
            let init = self.subsystem_zero_init_const(&sub.struct_name, &mut next_pp_bit);
            let align = if sub.dma_safe {
                si.align.max(16)
            } else {
                si.align.max(4)
            };
            let _ = writeln!(
                out,
                "{gsym} = internal global {} {init}, align {align}",
                si.llvm_name
            );
        }

        // Emit static global primitive instances
        let ptr_bytes = (self.spec.isa.ptr_bits() / 8) as usize;
        let prims = self.module.primitives.clone();
        for p in &prims {
            if p.dead_stripped {
                let _ = writeln!(out, "; dead watch stripped: @__syn_prim_{}", sanitize_ident(&p.name));
                continue;
            }
            let gsym = format!("@__syn_prim_{}", sanitize_ident(&p.name));
            let pty = format!("%prim.{}", sanitize_ident(&p.kind.mangle_name()));
            let align = if p.dma_safe {
                if matches!(p.kind, MirPrimKind::Pool { .. }) {
                    32
                } else {
                    16
                }
            } else {
                4
            };
            match &p.kind {
                MirPrimKind::Signal => {
                    let _ = writeln!(out, "{gsym} = global {pty} {{ i8 0, i8 127 }}, align {align}");
                }
                MirPrimKind::Watch { elem } => {
                    let ety = self.llvm_ty(elem);
                    let (_, elem_align) = self.size_align(elem, ptr_bytes);
                    let watch_align = align.max(elem_align);
                    if let Some(init_expr) = p.init.clone() {
                        let iv = self.const_expr_to_llvm(&init_expr, elem);
                        let _ = writeln!(
                            out,
                            "{gsym} = global {pty} {{ {ety} {iv}, i32 0 }}, align {watch_align}"
                        );
                    } else {
                        let _ = writeln!(
                            out,
                            "{gsym} = global {pty} zeroinitializer, align {watch_align}"
                        );
                    }
                }
                MirPrimKind::Inbox { .. } | MirPrimKind::ByteRing { .. } => {
                    let _ = writeln!(out, "{gsym} = global {pty} zeroinitializer, align {align}");
                }
                MirPrimKind::PingPong { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1);
                    let lbit = p.sleep_lock_bit.unwrap_or(0);
                    let _ = writeln!(
                        out,
                        "{gsym} = global {pty} {{ i8 0, i8 0, i8 0, i8 {lbit}, [{c} x {ety}] zeroinitializer, [{c} x {ety}] zeroinitializer }}, align {align}"
                    );
                }
                MirPrimKind::Pool { .. } => {
                    let gty = if p.dma_safe { format!("{pty}_dma") } else { pty };
                    let _ = writeln!(out, "{gsym} = global {gty} zeroinitializer, align {align}");
                }
                MirPrimKind::DmaSlot { .. } => {
                    let _ = writeln!(out, "{gsym} = global {pty} zeroinitializer, align {align}");
                }
            }
        }
        let _ = writeln!(out);

        // Emit shared pool lease release helper if pools, dma_slots, or Lease<T> values exist.
        let fn_sz = self.opts.fn_size_attrs();
        if self.uses_pool_lease {
            let (crit_enter, crit_exit) = crit_pair(vis.any_pool_or_lease_contended(), call_kw);
            let _ = write!(
                out,
                r#"define void @__syn_pool_release(ptr %pool_hdr, i8 %slot_idx, i8 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %is_null = icmp eq ptr %pool_hdr, null
  br i1 %is_null, label %done_null, label %chk_bounds
done_null:
  ret void
chk_bounds:
{crit_enter}  %cap_p = getelementptr inbounds %prim.pool_hdr, ptr %pool_hdr, i32 0, i32 4
  %cap = load i8, ptr %cap_p, align 1
  %valid = icmp ult i8 %slot_idx, %cap
  br i1 %valid, label %chk_bit, label %unlock_exit
chk_bit:
  %idx32 = zext i8 %slot_idx to i32
  %bit = shl i32 1, %idx32
  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %pool_hdr, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
  %was = and i32 %fm, %bit
  %is_used = icmp eq i32 %was, 0
  br i1 %is_used, label %free_slot, label %unlock_exit
free_slot:
  %new_fm = or i32 %fm, %bit
  store i32 %new_fm, ptr %fm_p, align 4
  %dma_p = getelementptr inbounds %prim.pool_hdr, ptr %pool_hdr, i32 0, i32 3
  %dma = load i8, ptr %dma_p, align 1
  %dma_lock = and i8 %dma, 1
  %is_dma = icmp ne i8 %dma_lock, 0
  br i1 %is_dma, label %dec_dma, label %wake
dec_dma:
  %dl = load volatile i32, ptr @__syn_dma_locks, align 4
  %has_dl = icmp ugt i32 %dl, 0
  br i1 %has_dl, label %do_dec, label %wake
do_dec:
  %dl1 = sub i32 %dl, 1
  store volatile i32 %dl1, ptr @__syn_dma_locks, align 4
  br label %wake
wake:
  %b32 = sext i8 %boost to i32
  {call_kw} void @__syn_wake_one_waiter(ptr %pool_hdr, i32 %b32)
  br label %unlock_exit
unlock_exit:
{crit_exit}  ret void
}}

"#
            );
        }
        // Emit typed primitive runtime helpers for each distinct primitive kind
        for (mangled, pk) in &self.distinct_prims {
            let id = sanitize_ident(mangled);
            let pty = format!("%prim.{id}");
            let (crit_enter, crit_exit) = crit_pair(vis.is_prim_family_contended(pk), call_kw);
            match pk {
                MirPrimKind::Signal => {
                    let _ = writeln!(
                        out,
                        r#"define void @__syn_sig_notify_{id}(ptr %s, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fp = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  store i8 1, ptr %fp, align 1
  %has_b = icmp sge i32 %boost, 0
  br i1 %has_b, label %upd_b, label %wake
upd_b:
  %pb_ptr = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %cur_pb = load i8, ptr %pb_ptr, align 1
  %b8 = trunc i32 %boost to i8
  %lt = icmp slt i8 %b8, %cur_pb
  %min_b = select i1 %lt, i8 %b8, i8 %cur_pb
  store i8 %min_b, ptr %pb_ptr, align 1
  br label %wake
wake:
  {call_kw} void @__syn_wake_waiters(ptr %s, i32 %boost)
{crit_exit}  ret void
}}

define i8 @__syn_sig_poll_{id}(ptr %s, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fp = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %f = load i8, ptr %fp, align 1
  %set = icmp ne i8 %f, 0
  br i1 %set, label %take, label %empty
take:
  store i8 0, ptr %fp, align 1
  %pb_ptr = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %pb = load i8, ptr %pb_ptr, align 1
  %has_pb = icmp slt i8 %pb, 127
  br i1 %has_pb, label %apply_pb, label %done_ok
apply_pb:
  store i8 127, ptr %pb_ptr, align 1
  %now = {call_kw} i32 @__syn_now_ms()
  %pb32 = zext i8 %pb to i32
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %pb32, i32 %now)
  br label %done_ok
done_ok:
{crit_exit}  ret i8 1
empty:
{crit_exit}  ret i8 0
}}"#
                    );
                }
                MirPrimKind::Watch { elem } => {
                    let ety = self.llvm_ty(elem);
                    let _ = writeln!(
                        out,
                        r#"define void @__syn_watch_pub_{id}(ptr %w, {ety} %val){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %vp = getelementptr inbounds {pty}, ptr %w, i32 0, i32 0
  store {ety} %val, ptr %vp, align 1
  %gp = getelementptr inbounds {pty}, ptr %w, i32 0, i32 1
  %g = load i32, ptr %gp, align 4
  %g1 = add i32 %g, 1
  store i32 %g1, ptr %gp, align 4
{crit_exit}  ret void
}}

define {ety} @__syn_watch_read_{id}(ptr %w){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %vp = getelementptr inbounds {pty}, ptr %w, i32 0, i32 0
  %v = load {ety}, ptr %vp, align 1
{crit_exit}  ret {ety} %v
}}"#
                    );
                }
                MirPrimKind::Inbox { elem, cap } => {
                    let ety = self.llvm_ty(elem);
                    let c = (*cap).max(1) as i32;
                    let is_lease = matches!(elem.as_ref(), MirType::Lease(_));
                    let push_fn = if is_lease {
                        format!("@__syn_q_push_keep_{id}")
                    } else {
                        format!("@__syn_q_push_{id}")
                    };
                    let clear_slot = if is_lease {
                        format!("\n  store {ety} zeroinitializer, ptr %slot_p, align 1")
                    } else {
                        String::new()
                    };
                    let _ = writeln!(
                        out,
                        r#"define i8 {push_fn}(ptr %q, {ety} %val, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cnt_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 2
  %cnt = load i32, ptr %cnt_p, align 4
  %space = icmp slt i32 %cnt, {c}
  br i1 %space, label %do_push, label %full
do_push:
  %tail_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 1
  %tail = load i32, ptr %tail_p, align 4
  %slot_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 4, i32 %tail
  store {ety} %val, ptr %slot_p, align 1
  %bp = getelementptr inbounds {pty}, ptr %q, i32 0, i32 3, i32 %tail
  %has_b = icmp sge i32 %boost, 0
  %b_p1 = add i32 %boost, 1
  %b8 = trunc i32 %b_p1 to i8
  %sb = select i1 %has_b, i8 %b8, i8 0
  store i8 %sb, ptr %bp, align 1
  %t1 = add i32 %tail, 1
  %twrap = icmp uge i32 %t1, {c}
  %tsub = sub i32 %t1, {c}
  %tnext = select i1 %twrap, i32 %tsub, i32 %t1
  store i32 %tnext, ptr %tail_p, align 4
  %c1 = add i32 %cnt, 1
  store i32 %c1, ptr %cnt_p, align 4
  {call_kw} void @__syn_wake_waiters(ptr %q, i32 %boost)
{crit_exit}  ret i8 1
full:
{crit_exit}  ret i8 0
}}

define i8 @__syn_q_pop_{id}(ptr %q, ptr %out_val, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cnt_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 2
  %cnt = load i32, ptr %cnt_p, align 4
  %avail = icmp sgt i32 %cnt, 0
  br i1 %avail, label %do_pop, label %empty
do_pop:
  %head_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 0
  %head = load i32, ptr %head_p, align 4
  %slot_p = getelementptr inbounds {pty}, ptr %q, i32 0, i32 4, i32 %head
  %val = load {ety}, ptr %slot_p, align 1
  store {ety} %val, ptr %out_val, align 1{clear_slot}
  %bp = getelementptr inbounds {pty}, ptr %q, i32 0, i32 3, i32 %head
  %sb = load i8, ptr %bp, align 1
  store i8 0, ptr %bp, align 1
  %h1 = add i32 %head, 1
  %hwrap = icmp uge i32 %h1, {c}
  %hsub = sub i32 %h1, {c}
  %hnext = select i1 %hwrap, i32 %hsub, i32 %h1
  store i32 %hnext, ptr %head_p, align 4
  %c1 = sub i32 %cnt, 1
  store i32 %c1, ptr %cnt_p, align 4
  %has_pb = icmp ne i8 %sb, 0
  br i1 %has_pb, label %apply_pb, label %wake_senders
apply_pb:
  %now = {call_kw} i32 @__syn_now_ms()
  %sb32 = zext i8 %sb to i32
  %pb32 = sub i32 %sb32, 1
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %pb32, i32 %now)
  br label %wake_senders
wake_senders:
  %q_tx = getelementptr inbounds {pty}, ptr %q, i32 0, i32 1
  {call_kw} void @__syn_wake_waiters(ptr %q_tx, i32 -1)
{crit_exit}  ret i8 1
empty:
{crit_exit}  ret i8 0
}}"#
                    );
                    if is_lease {
                        let _ = writeln!(
                            out,
                            r#"define i8 @__syn_q_push_{id}(ptr %q, {ety} %val, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %ok = {call_kw} i8 @__syn_q_push_keep_{id}(ptr %q, {ety} %val, i32 %boost)
  %ok_b = icmp ne i8 %ok, 0
  br i1 %ok_b, label %done, label %drop_lease
drop_lease:
  %ph = extractvalue {ety} %val, 1
  %idx = extractvalue {ety} %val, 2
  {call_kw} void @__syn_pool_release(ptr %ph, i8 %idx, i8 -1)
  ret i8 0
done:
  ret i8 1
}}"#
                        );
                    }
                }
                MirPrimKind::PingPong { .. } => {
                    let _ = writeln!(
                        out,
                        r#"define void @__syn_pp_set_active_{id}(ptr %pp, i8 %active){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %lb_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 3
  %lb = load i8, ptr %lb_p, align 1
  %lb32 = zext i8 %lb to i32
  %sh = and i32 %lb32, 31
  %bit = shl i32 1, %sh
  %old = load volatile i32, ptr @__syn_sleep_lock_mask, align 4
  %is_on = icmp ne i8 %active, 0
  %inv = xor i32 %bit, -1
  %cleared = and i32 %old, %inv
  %setv = or i32 %old, %bit
  %newv = select i1 %is_on, i32 %setv, i32 %cleared
  store volatile i32 %newv, ptr @__syn_sleep_lock_mask, align 4
{crit_exit}  ret void
}}

define void @__syn_pp_flip_{id}(ptr %pp){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %wh_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 0
  %rh_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 1
  %cr_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 2
  %wh = load i8, ptr %wh_p, align 1
  store i8 %wh, ptr %rh_p, align 1
  %next = xor i8 %wh, 1
  store i8 %next, ptr %wh_p, align 1
  store i8 1, ptr %cr_p, align 1
  {call_kw} void @__syn_wake_waiters(ptr %pp, i32 -1)
{crit_exit}  ret void
}}

define ptr @__syn_pp_poll_{id}(ptr %pp){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cr_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 2
  %cr = load i8, ptr %cr_p, align 1
  %rdy = icmp ne i8 %cr, 0
  br i1 %rdy, label %take, label %none
take:
  store i8 0, ptr %cr_p, align 1
  %rh_p = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 1
  %rh = load i8, ptr %rh_p, align 1
  %is_b0 = icmp eq i8 %rh, 0
  %b0 = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 4, i32 0
  %b1 = getelementptr inbounds {pty}, ptr %pp, i32 0, i32 5, i32 0
  %buf = select i1 %is_b0, ptr %b0, ptr %b1
{crit_exit}  ret ptr %buf
none:
{crit_exit}  ret ptr null
}}"#
                    );
                }
                MirPrimKind::Pool { .. } => {
                    let ptr_al = self.ptr_align();
                    let usize_ty = self.spec.isa.usize_llvm_ty();
                    let (hdr_sz, _) = self.pool_hdr_llvm_size();
                    let slots_fidx = if 32usize.saturating_sub(hdr_sz) > 0 {
                        2
                    } else {
                        1
                    };
                    let conv_usize = |src: &str, dst: &str| -> String {
                        match usize_ty {
                            "i16" => format!("  {dst} = trunc i32 {src} to i16"),
                            "i64" => format!("  {dst} = zext i32 {src} to i64"),
                            _ => format!("  {dst} = add i32 {src}, 0"),
                        }
                    };
                    let conv_av = conv_usize("%cnt", "%cnt_u");
                    let conv_used = conv_usize("%used", "%used_u");
                    let conv_cap = conv_usize("%cap32", "%cap_u");
                    let _ = writeln!(
                        out,
                        r#"define ptr @__syn_pool_slot_ptr_{id}(ptr %p, i8 %idx){fn_as} nounwind{fn_sz} {{
entry:
  %dma_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 3
  %dma = load i8, ptr %dma_p, align 1
  %is_dma = icmp ne i8 %dma, 0
  %idx32 = zext i8 %idx to i32
  %p_dma = getelementptr inbounds {pty}_dma, ptr %p, i32 0, i32 {slots_fidx}, i32 %idx32, i32 0
  %p_plain = getelementptr inbounds {pty}, ptr %p, i32 0, i32 1, i32 %idx32
  %res = select i1 %is_dma, ptr %p_dma, ptr %p_plain
  ret ptr %res
}}

define i8 @__syn_pool_try_alloc_raw_{id}(ptr %p, ptr %out_lease){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
  %has = icmp ne i32 %fm, 0
  br i1 %has, label %take, label %empty
take:
  %idx32 = call i32 @llvm.cttz.i32(i32 %fm, i1 true)
  %bit = shl i32 1, %idx32
  %inv = xor i32 %bit, -1
  %new_fm = and i32 %fm, %inv
  store i32 %new_fm, ptr %fm_p, align 4
  %dma_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 3
  %dma = load i8, ptr %dma_p, align 1
  %dma_lock = and i8 %dma, 1
  %is_dma = icmp ne i8 %dma_lock, 0
  br i1 %is_dma, label %inc_dma, label %finish
inc_dma:
  %dl = load volatile i32, ptr @__syn_dma_locks, align 4
  %dl1 = add i32 %dl, 1
  store volatile i32 %dl1, ptr @__syn_dma_locks, align 4
  br label %finish
finish:
  %idx8 = trunc i32 %idx32 to i8
{crit_exit}  %dp = {call_kw} ptr @__syn_pool_slot_ptr_{id}(ptr %p, i8 %idx8)
  %f0 = getelementptr inbounds {{ ptr, ptr, i8 }}, ptr %out_lease, i32 0, i32 0
  store ptr %dp, ptr %f0, align {ptr_al}
  %f1 = getelementptr inbounds {{ ptr, ptr, i8 }}, ptr %out_lease, i32 0, i32 1
  store ptr %p, ptr %f1, align {ptr_al}
  %f2 = getelementptr inbounds {{ ptr, ptr, i8 }}, ptr %out_lease, i32 0, i32 2
  store i8 %idx8, ptr %f2, align 1
  ret i8 1
empty:
{crit_exit}  ret i8 0
}}

define {{ {{ ptr, ptr, i8 }}, i32 }} @__syn_pool_try_alloc_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
  %tmp = alloca {{ ptr, ptr, i8 }}, align {ptr_al}
  store {{ ptr, ptr, i8 }} zeroinitializer, ptr %tmp, align {ptr_al}
  %ok = {call_kw} i8 @__syn_pool_try_alloc_raw_{id}(ptr %p, ptr %tmp)
  %got = icmp ne i8 %ok, 0
  %ls = load {{ ptr, ptr, i8 }}, ptr %tmp, align {ptr_al}
  %tag = select i1 %got, i32 0, i32 1
  %r0 = insertvalue {{ {{ ptr, ptr, i8 }}, i32 }} poison, {{ ptr, ptr, i8 }} %ls, 0
  %r1 = insertvalue {{ {{ ptr, ptr, i8 }}, i32 }} %r0, i32 %tag, 1
  ret {{ {{ ptr, ptr, i8 }}, i32 }} %r1
}}

define {usize_ty} @__syn_pool_avail_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
{crit_exit}  %cnt = call i32 @llvm.ctpop.i32(i32 %fm)
{conv_av}
  ret {usize_ty} %cnt_u
}}

define {usize_ty} @__syn_pool_in_use_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cap_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 4
  %cap = load i8, ptr %cap_p, align 1
  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
{crit_exit}  %avail = call i32 @llvm.ctpop.i32(i32 %fm)
  %cap32 = zext i8 %cap to i32
  %used = sub i32 %cap32, %avail
{conv_used}
  ret {usize_ty} %used_u
}}

define {usize_ty} @__syn_pool_cap_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
  %cap_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 4
  %cap = load i8, ptr %cap_p, align 1
  %cap32 = zext i8 %cap to i32
{conv_cap}
  ret {usize_ty} %cap_u
}}

define i8 @__syn_pool_is_exhausted_{id}(ptr %p){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fm_p = getelementptr inbounds %prim.pool_hdr, ptr %p, i32 0, i32 0
  %fm = load i32, ptr %fm_p, align 4
{crit_exit}  %empty = icmp eq i32 %fm, 0
  %b8 = zext i1 %empty to i8
  ret i8 %b8
}}"#
                    );
                }
                MirPrimKind::DmaSlot { .. } => {
                    let usize_ty = self.spec.isa.usize_llvm_ty();
                    let _ = writeln!(
                        out,
                        r#"define {{ i32, i32 }} @__syn_slot_park_{id}(ptr %s, {{ ptr, ptr, i8 }} %lease){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
  %busy = icmp ne i8 %occ, 0
  br i1 %busy, label %is_busy, label %do_park
is_busy:
{crit_exit}  %ph_b = extractvalue {{ ptr, ptr, i8 }} %lease, 1
  %idx_b = extractvalue {{ ptr, ptr, i8 }} %lease, 2
  {call_kw} void @__syn_pool_release(ptr %ph_b, i8 %idx_b, i8 -1)
  ret {{ i32, i32 }} {{ i32 0, i32 1 }}
do_park:
  %dp = extractvalue {{ ptr, ptr, i8 }} %lease, 0
  %ph = extractvalue {{ ptr, ptr, i8 }} %lease, 1
  %idx = extractvalue {{ ptr, ptr, i8 }} %lease, 2
  %f0 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %f1 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %f2 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 2
  store ptr %dp, ptr %f0, align 1
  store ptr %ph, ptr %f1, align 1
  store i8 %idx, ptr %f2, align 1
  store i8 1, ptr %occ_p, align 1
{crit_exit}  ret {{ i32, i32 }} zeroinitializer
}}

define {{ {{ ptr, ptr, i8 }}, i32 }} @__syn_slot_take_{id}(ptr %s){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
  %empty = icmp eq i8 %occ, 0
  br i1 %empty, label %is_empty, label %do_take
is_empty:
{crit_exit}  ret {{ {{ ptr, ptr, i8 }}, i32 }} {{ {{ ptr, ptr, i8 }} zeroinitializer, i32 1 }}
do_take:
  %f0 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %f1 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %f2 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 2
  %dp = load ptr, ptr %f0, align 1
  %ph = load ptr, ptr %f1, align 1
  %idx = load i8, ptr %f2, align 1
  store ptr null, ptr %f0, align 1
  store ptr null, ptr %f1, align 1
  store i8 0, ptr %f2, align 1
  store i8 0, ptr %occ_p, align 1
{crit_exit}  %l0 = insertvalue {{ ptr, ptr, i8 }} poison, ptr %dp, 0
  %l1 = insertvalue {{ ptr, ptr, i8 }} %l0, ptr %ph, 1
  %l2 = insertvalue {{ ptr, ptr, i8 }} %l1, i8 %idx, 2
  %r0 = insertvalue {{ {{ ptr, ptr, i8 }}, i32 }} poison, {{ ptr, ptr, i8 }} %l2, 0
  %r1 = insertvalue {{ {{ ptr, ptr, i8 }}, i32 }} %r0, i32 0, 1
  ret {{ {{ ptr, ptr, i8 }}, i32 }} %r1
}}

define i8 @__syn_slot_release_{id}(ptr %s, i8 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
  %empty = icmp eq i8 %occ, 0
  br i1 %empty, label %is_empty, label %do_rel
is_empty:
{crit_exit}  ret i8 0
do_rel:
  %f0 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %f1 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 1
  %f2 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 2
  %ph = load ptr, ptr %f1, align 1
  %idx = load i8, ptr %f2, align 1
  store ptr null, ptr %f0, align 1
  store ptr null, ptr %f1, align 1
  store i8 0, ptr %f2, align 1
  store i8 0, ptr %occ_p, align 1
{crit_exit}  {call_kw} void @__syn_pool_release(ptr %ph, i8 %idx, i8 %boost)
  ret i8 1
}}

define i8 @__syn_slot_is_busy_{id}(ptr %s){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
{crit_exit}  %b = icmp ne i8 %occ, 0
  %b8 = zext i1 %b to i8
  ret i8 %b8
}}

define i8 @__syn_slot_is_empty_{id}(ptr %s){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
{crit_exit}  %e = icmp eq i8 %occ, 0
  %b8 = zext i1 %e to i8
  ret i8 %b8
}}

define {usize_ty} @__syn_slot_addr_{id}(ptr %s){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %occ_p = getelementptr inbounds {pty}, ptr %s, i32 0, i32 3
  %occ = load i8, ptr %occ_p, align 1
  %f0 = getelementptr inbounds {pty}, ptr %s, i32 0, i32 0
  %dp = load ptr, ptr %f0, align 1
{crit_exit}  %busy = icmp ne i8 %occ, 0
  %p_sel = select i1 %busy, ptr %dp, ptr null
  %addr = ptrtoint ptr %p_sel to {usize_ty}
  ret {usize_ty} %addr
}}"#
                    );
                }
                MirPrimKind::ByteRing { cap } => {
                    let c = (*cap).max(1) as i32;
                    let usize_ty = self.spec.isa.usize_llvm_ty();
                    let slice_ty = format!("{{ ptr, {usize_ty} }}");
                    let res_slice_ty = format!("{{ {slice_ty}, i32 }}");
                    let to_usize = |src: &str, dst: &str| -> String {
                        match usize_ty {
                            "i16" => format!("  {dst} = trunc i32 {src} to i16"),
                            "i64" => format!("  {dst} = zext i32 {src} to i64"),
                            _ => format!("  {dst} = add i32 {src}, 0"),
                        }
                    };
                    let conv_req = to_usize("%req", "%req_u");
                    let conv_s1_r = to_usize("%s1", "%s1_u");
                    let conv_s1_p = to_usize("%s1", "%s1_u");
                    let mk_slice_ok = |tag: &str, ptr_v: &str, len_u: &str| -> String {
                        format!(
                            "  %sl0_{tag} = insertvalue {slice_ty} poison, ptr {ptr_v}, 0\n  %sl1_{tag} = insertvalue {slice_ty} %sl0_{tag}, {usize_ty} {len_u}, 1\n  %ok0_{tag} = insertvalue {res_slice_ty} zeroinitializer, {slice_ty} %sl1_{tag}, 0\n  ret {res_slice_ty} %ok0_{tag}"
                        )
                    };
                    let ok_z = mk_slice_ok("z", "%pz", "0");
                    let ok_back = mk_slice_ok("b", "%pb", "%req_u");
                    let ok_wrap = mk_slice_ok("w", "%pw", "%req_u");
                    let ok_gap = mk_slice_ok("g", "%pg", "%req_u");
                    let _ = writeln!(
                        out,
                        r#"define i32 @__syn_ring_seg1_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_z = icmp eq i32 %cnt, 0
  br i1 %is_z, label %z, label %nz
z:
  ret i32 0
nz:
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %has_se = icmp ugt i32 %se, 0
  br i1 %has_se, label %use_se, label %no_se
use_se:
  %d0 = sub i32 %se, %h
  ret i32 %d0
no_se:
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %h_lt_t = icmp ult i32 %h, %t
  %d1 = sub i32 %t, %h
  %d2 = sub i32 {c}, %h
  %res = select i1 %h_lt_t, i32 %d1, i32 %d2
  ret i32 %res
}}

define i32 @__syn_ring_free_nolock_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  br i1 %has_se, label %se_p, label %no_se
se_p:
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %h = load i32, ptr %hp, align 4
  %t = load i32, ptr %tp, align 4
  %fs0 = sub i32 %h, %t
  ret i32 %fs0
no_se:
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %fs1 = sub i32 {c}, %cnt
  ret i32 %fs1
}}

define i32 @__syn_ring_free_space_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fs = {call_kw} i32 @__syn_ring_free_nolock_{id}(ptr %r)
{crit_exit}  ret i32 %fs
}}

define void @__syn_ring_advance_{id}(ptr %r, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %lt = icmp ult i32 %req, %cnt
  %rem = select i1 %lt, i32 %req, i32 %cnt
  %is_z = icmp eq i32 %rem, 0
  br i1 %is_z, label %done, label %chk_all
chk_all:
  %s1 = {call_kw} i32 @__syn_ring_seg1_{id}(ptr %r)
  %is_all = icmp eq i32 %rem, %cnt
  br i1 %is_all, label %all_empty, label %partial
all_empty:
  %hp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %sp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %gp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  %t0 = load i32, ptr %tp0, align 4
  %gc = load i32, ptr %gp0, align 4
  %has_g = icmp ugt i32 %gc, 0
  %nh0 = select i1 %has_g, i32 %t0, i32 0
  store i32 %nh0, ptr %hp0, align 4
  br i1 %has_g, label %skip_tail_rst, label %do_tail_rst
do_tail_rst:
  store i32 0, ptr %tp0, align 4
  br label %skip_tail_rst
skip_tail_rst:
  store i32 0, ptr %cp, align 4
  store i32 0, ptr %sp0, align 4
  ret void
partial:
  %hp1 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h1 = load i32, ptr %hp1, align 4
  %in_s1 = icmp ult i32 %rem, %s1
  br i1 %in_s1, label %adv_h, label %wrap_h
adv_h:
  %nh1 = add i32 %h1, %rem
  %nc1 = sub i32 %cnt, %rem
  store i32 %nh1, ptr %hp1, align 4
  store i32 %nc1, ptr %cp, align 4
  ret void
wrap_h:
  %nh2 = sub i32 %rem, %s1
  %nc2 = sub i32 %cnt, %rem
  %sp2 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  store i32 %nh2, ptr %hp1, align 4
  store i32 %nc2, ptr %cp, align 4
  store i32 0, ptr %sp2, align 4
  ret void
done:
  ret void
}}

define void @__syn_ring_wake_{id}(ptr %r, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %has_b = icmp sge i32 %boost, 0
  br i1 %has_b, label %chk_b, label %do_wake
chk_b:
  %b_p1 = add i32 %boost, 1
  %enc = trunc i32 %b_p1 to i8
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 6
  %cur = load i8, ptr %bp, align 1
  %is_z = icmp eq i8 %cur, 0
  %better = icmp ult i8 %enc, %cur
  %upd = or i1 %is_z, %better
  br i1 %upd, label %set_b, label %do_wake
set_b:
  store i8 %enc, ptr %bp, align 1
  br label %do_wake
do_wake:
  {call_kw} void @__syn_wake_waiters(ptr %r, i32 %boost)
  ret void
}}

define void @__syn_ring_apply_boost_{id}(ptr %r, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 6
  %enc = load i8, ptr %bp, align 1
  %has_b = icmp ne i8 %enc, 0
  %has_t = icmp ne ptr %task, null
  %go = and i1 %has_b, %has_t
  br i1 %go, label %apply, label %done
apply:
  store i8 0, ptr %bp, align 1
  %enc32 = zext i8 %enc to i32
  %prio = sub i32 %enc32, 1
  %now = {call_kw} i32 @__syn_now_ms()
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %prio, i32 %now)
  br label %done
done:
  ret void
}}

define i8 @__syn_ring_push_byte_{id}(ptr %r, i8 %b, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 0, ptr %gp, align 4
  %fs = {call_kw} i32 @__syn_ring_free_nolock_{id}(ptr %r)
  %has = icmp ugt i32 %fs, 0
  br i1 %has, label %do_wr, label %full
do_wr:
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %dp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 %t
  store i8 %b, ptr %dp, align 1
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  %t1 = add i32 %t, 1
  %wrap = icmp uge i32 %t1, {c}
  %nt_circ = select i1 %wrap, i32 0, i32 %t1
  %nt = select i1 %has_se, i32 %t1, i32 %nt_circ
  store i32 %nt, ptr %tp, align 4
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %nc = add i32 %cnt, 1
  store i32 %nc, ptr %cp, align 4
  {call_kw} void @__syn_ring_wake_{id}(ptr %r, i32 %boost)
{crit_exit}  ret i8 1
full:
{crit_exit}  ret i8 0
}}

define i32 @__syn_ring_push_{id}(ptr %r, ptr %src_ptr, i32 %src_len, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 0, ptr %gp, align 4
  %fs = {call_kw} i32 @__syn_ring_free_nolock_{id}(ptr %r)
  %is_null = icmp eq ptr %src_ptr, null
  %lt = icmp ult i32 %src_len, %fs
  %nw_raw = select i1 %lt, i32 %src_len, i32 %fs
  %nw = select i1 %is_null, i32 0, i32 %nw_raw
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %t_init = load i32, ptr %tp, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %body ]
  %t_cur = phi i32 [ %t_init, %entry ], [ %t_next, %body ]
  %done = icmp uge i32 %i, %nw
  br i1 %done, label %finish, label %body
body:
  %sp_in = getelementptr i8, ptr %src_ptr, i32 %i
  %bv = load i8, ptr %sp_in, align 1
  %dp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 %t_cur
  store i8 %bv, ptr %dp, align 1
  %t1 = add i32 %t_cur, 1
  %wrap = icmp uge i32 %t1, {c}
  %t_circ = select i1 %wrap, i32 0, i32 %t1
  %t_next = select i1 %has_se, i32 %t1, i32 %t_circ
  %inext = add i32 %i, 1
  br label %loop
finish:
  store i32 %t_cur, ptr %tp, align 4
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %nc = add i32 %cnt, %nw
  store i32 %nc, ptr %cp, align 4
  %any = icmp ugt i32 %nw, 0
  br i1 %any, label %wake, label %exit
wake:
  {call_kw} void @__syn_ring_wake_{id}(ptr %r, i32 %boost)
  br label %exit
exit:
{crit_exit}  ret i32 %nw
}}

define {res_slice_ty} @__syn_ring_grant_write_{id}(ptr %r, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}{conv_req}
  %is_z = icmp eq i32 %req, 0
  br i1 %is_z, label %ret_z, label %chk_cap
ret_z:
  %tp_z = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %tz = load i32, ptr %tp_z, align 4
  %gop_z = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp_z = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 %tz, ptr %gop_z, align 4
  store i32 0, ptr %gcp_z, align 4
  %pz = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 0
{crit_exit}{ok_z}
chk_cap:
  %too_big = icmp ugt i32 %req, {c}
  br i1 %too_big, label %fail, label %chk_empty
chk_empty:
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_emp = icmp eq i32 %cnt, 0
  br i1 %is_emp, label %empty_ok, label %non_empty
empty_ok:
  %hp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %sp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %gop0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp0 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 0, ptr %hp0, align 4
  store i32 0, ptr %tp0, align 4
  store i32 0, ptr %sp0, align 4
  store i32 0, ptr %gop0, align 4
  store i32 %req, ptr %gcp0, align 4
  %pz_ok = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 0
{crit_exit}  %sl0_e = insertvalue {slice_ty} poison, ptr %pz_ok, 0
  %sl1_e = insertvalue {slice_ty} %sl0_e, {usize_ty} %req_u, 1
  %ok_e = insertvalue {res_slice_ty} zeroinitializer, {slice_ty} %sl1_e, 0
  ret {res_slice_ty} %ok_e
non_empty:
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %h = load i32, ptr %hp, align 4
  %t = load i32, ptr %tp, align 4
  %se = load i32, ptr %sp, align 4
  %no_se = icmp eq i32 %se, 0
  %h_lt_t = icmp ult i32 %h, %t
  %unwrapped = and i1 %no_se, %h_lt_t
  br i1 %unwrapped, label %back_or_wrap, label %mid_gap
back_or_wrap:
  %rem_back = sub i32 {c}, %t
  %fit_back = icmp ule i32 %req, %rem_back
  br i1 %fit_back, label %ok_b, label %try_wrap
ok_b:
  %gop_b = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp_b = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 %t, ptr %gop_b, align 4
  store i32 %req, ptr %gcp_b, align 4
  %pb = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 %t
{crit_exit}{ok_back}
try_wrap:
  %fit_wrap = icmp ule i32 %req, %h
  br i1 %fit_wrap, label %ok_w, label %fail
ok_w:
  %gop_w = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp_w = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 0, ptr %gop_w, align 4
  store i32 %req, ptr %gcp_w, align 4
  %pw = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 0
{crit_exit}{ok_wrap}
mid_gap:
  %has_se = icmp ugt i32 %se, 0
  %h_gt_t = icmp ugt i32 %h, %t
  %has_gap = or i1 %has_se, %h_gt_t
  %raw_gap = sub i32 %h, %t
  %gap = select i1 %has_gap, i32 %raw_gap, i32 0
  %fit_gap = icmp ule i32 %req, %gap
  br i1 %fit_gap, label %ok_g, label %fail
ok_g:
  %gop_g = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp_g = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  store i32 %t, ptr %gop_g, align 4
  store i32 %req, ptr %gcp_g, align 4
  %pg = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 %t
{crit_exit}{ok_gap}
fail:
{crit_exit}  %err_r = insertvalue {res_slice_ty} zeroinitializer, i32 1, 1
  ret {res_slice_ty} %err_r
}}

define void @__syn_ring_commit_write_{id}(ptr %r, i32 %req, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gcp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  %gc = load i32, ptr %gcp, align 4
  store i32 0, ptr %gcp, align 4
  %lt = icmp ult i32 %req, %gc
  %nw = select i1 %lt, i32 %req, i32 %gc
  %is_z = icmp eq i32 %nw, 0
  br i1 %is_z, label %done, label %apply
apply:
  %gop = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %goff = load i32, ptr %gop, align 4
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_w0 = icmp eq i32 %goff, 0
  %t_gt0 = icmp ugt i32 %t, 0
  %se_eq0 = icmp eq i32 %se, 0
  %c0 = and i1 %is_w0, %t_gt0
  %is_early = and i1 %c0, %se_eq0
  br i1 %is_early, label %early_wrap, label %linear
early_wrap:
  %cnt_z = icmp eq i32 %cnt, 0
  br i1 %cnt_z, label %ew_empty, label %ew_split
ew_empty:
  %hp_e = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  store i32 0, ptr %hp_e, align 4
  store i32 %nw, ptr %tp, align 4
  store i32 %nw, ptr %cp, align 4
  br label %wake
ew_split:
  store i32 %t, ptr %sp, align 4
  store i32 %nw, ptr %tp, align 4
  %nc_ew = add i32 %cnt, %nw
  store i32 %nc_ew, ptr %cp, align 4
  br label %wake
linear:
  %nt = add i32 %t, %nw
  %wrap = icmp uge i32 %nt, {c}
  %do_wrap = and i1 %se_eq0, %wrap
  %nt_final = select i1 %do_wrap, i32 0, i32 %nt
  store i32 %nt_final, ptr %tp, align 4
  %nc_l = add i32 %cnt, %nw
  store i32 %nc_l, ptr %cp, align 4
  br label %wake
wake:
  {call_kw} void @__syn_ring_wake_{id}(ptr %r, i32 %boost)
  br label %done
done:
{crit_exit}  ret void
}}

define {slice_ty} @__syn_ring_read_slice_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %s1 = {call_kw} i32 @__syn_ring_seg1_{id}(ptr %r)
  %has = icmp ugt i32 %s1, 0
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %idx = select i1 %has, i32 %h, i32 0
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 %idx
{crit_exit}{conv_s1_r}
  %sl0 = insertvalue {slice_ty} poison, ptr %bp, 0
  %sl1 = insertvalue {slice_ty} %sl0, {usize_ty} %s1_u, 1
  ret {slice_ty} %sl1
}}

define {slice_ty} @__syn_ring_poll_read_{id}(ptr %r, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %s1 = {call_kw} i32 @__syn_ring_seg1_{id}(ptr %r)
  %has = icmp ugt i32 %s1, 0
  br i1 %has, label %ready, label %empty
ready:
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 %h
  {call_kw} void @__syn_ring_apply_boost_{id}(ptr %r, ptr %task)
{crit_exit}{conv_s1_p}
  %sl0 = insertvalue {slice_ty} poison, ptr %bp, 0
  %sl1 = insertvalue {slice_ty} %sl0, {usize_ty} %s1_u, 1
  ret {slice_ty} %sl1
empty:
{crit_exit}  ret {slice_ty} zeroinitializer
}}

define void @__syn_ring_consume_{id}(ptr %r, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  {call_kw} void @__syn_ring_advance_{id}(ptr %r, i32 %req)
{crit_exit}  ret void
}}

define i32 @__syn_ring_pop_{id}(ptr %r, ptr %dst_ptr, i32 %dst_len, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %lt = icmp ult i32 %dst_len, %cnt
  %rem_raw = select i1 %lt, i32 %dst_len, i32 %cnt
  %is_null = icmp eq ptr %dst_ptr, null
  %rem = select i1 %is_null, i32 0, i32 %rem_raw
  %has = icmp ugt i32 %rem, 0
  br i1 %has, label %do_pop, label %empty
do_pop:
  %s1 = {call_kw} i32 @__syn_ring_seg1_{id}(ptr %r)
  %lt_s1 = icmp ult i32 %rem, %s1
  %c1 = select i1 %lt_s1, i32 %rem, i32 %s1
  %c2 = sub i32 %rem, %c1
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  br label %l1
l1:
  %i1 = phi i32 [ 0, %do_pop ], [ %i1n, %b1 ]
  %d1 = icmp uge i32 %i1, %c1
  br i1 %d1, label %l2_init, label %b1
b1:
  %src1_i = add i32 %h, %i1
  %sp1 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 %src1_i
  %v1 = load i8, ptr %sp1, align 1
  %dp1 = getelementptr i8, ptr %dst_ptr, i32 %i1
  store i8 %v1, ptr %dp1, align 1
  %i1n = add i32 %i1, 1
  br label %l1
l2_init:
  br label %l2
l2:
  %i2 = phi i32 [ 0, %l2_init ], [ %i2n, %b2 ]
  %d2 = icmp uge i32 %i2, %c2
  br i1 %d2, label %adv, label %b2
b2:
  %sp2 = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 %i2
  %v2 = load i8, ptr %sp2, align 1
  %dst2_i = add i32 %c1, %i2
  %dp2 = getelementptr i8, ptr %dst_ptr, i32 %dst2_i
  store i8 %v2, ptr %dp2, align 1
  %i2n = add i32 %i2, 1
  br label %l2
adv:
  {call_kw} void @__syn_ring_advance_{id}(ptr %r, i32 %rem)
  {call_kw} void @__syn_ring_apply_boost_{id}(ptr %r, ptr %task)
{crit_exit}  ret i32 %rem
empty:
{crit_exit}  ret i32 0
}}

define {{ i8, i32 }} @__syn_ring_try_pop_byte_{id}(ptr %r, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %has = icmp ugt i32 %cnt, 0
  br i1 %has, label %ok, label %empty
ok:
  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 7, i32 %h
  %b = load i8, ptr %bp, align 1
  {call_kw} void @__syn_ring_advance_{id}(ptr %r, i32 1)
  {call_kw} void @__syn_ring_apply_boost_{id}(ptr %r, ptr %task)
{crit_exit}  %r0 = insertvalue {{ i8, i32 }} zeroinitializer, i8 %b, 0
  ret {{ i8, i32 }} %r0
empty:
{crit_exit}  %re = insertvalue {{ i8, i32 }} zeroinitializer, i32 1, 1
  ret {{ i8, i32 }} %re
}}

define void @__syn_ring_clear_{id}(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %hp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 1
  %cp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 2
  %sp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 3
  %gop = getelementptr inbounds {pty}, ptr %r, i32 0, i32 4
  %gcp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 5
  %bp = getelementptr inbounds {pty}, ptr %r, i32 0, i32 6
  store i32 0, ptr %hp, align 4
  store i32 0, ptr %tp, align 4
  store i32 0, ptr %cp, align 4
  store i32 0, ptr %sp, align 4
  store i32 0, ptr %gop, align 4
  store i32 0, ptr %gcp, align 4
  store i8 0, ptr %bp, align 1
{crit_exit}  ret void
}}"#
                    );
                }
            }
        }

        // Emit capacity-erased unsized `*inbox<T>` view helpers
        for (key, elem) in &self.distinct_inbox_views {
            let ety = self.llvm_ty(elem);
            let is_lease = matches!(elem, MirType::Lease(_));
            let push_fn = if is_lease {
                format!("@__syn_q_view_push_keep_{key}")
            } else {
                format!("@__syn_q_view_push_{key}")
            };
            let clear_slot = if is_lease {
                format!("\n  store {ety} zeroinitializer, ptr %slot_p, align 1")
            } else {
                String::new()
            };
            let (crit_enter, crit_exit) = crit_pair(vis.is_inbox_view_contended(elem), call_kw);
            let _ = writeln!(
                out,
                r#"define i8 {push_fn}(ptr %hdr, ptr %items_base, i32 %cap, {ety} %val, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cnt_p = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 2
  %cnt = load i32, ptr %cnt_p, align 4
  %has_c = icmp sgt i32 %cap, 0
  %space = icmp slt i32 %cnt, %cap
  %ok = and i1 %has_c, %space
  br i1 %ok, label %do_push, label %full
do_push:
  %tail_p = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 1
  %tail = load i32, ptr %tail_p, align 4
  %slot_p = getelementptr inbounds {ety}, ptr %items_base, i32 %tail
  store {ety} %val, ptr %slot_p, align 1
  %bp = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 3, i32 %tail
  %has_b = icmp sge i32 %boost, 0
  %b_p1 = add i32 %boost, 1
  %b8 = trunc i32 %b_p1 to i8
  %sb = select i1 %has_b, i8 %b8, i8 0
  store i8 %sb, ptr %bp, align 1
  %t1 = add i32 %tail, 1
  %twrap = icmp uge i32 %t1, %cap
  %tsub = sub i32 %t1, %cap
  %tnext = select i1 %twrap, i32 %tsub, i32 %t1
  store i32 %tnext, ptr %tail_p, align 4
  %c1 = add i32 %cnt, 1
  store i32 %c1, ptr %cnt_p, align 4
  {call_kw} void @__syn_wake_waiters(ptr %hdr, i32 %boost)
{crit_exit}  ret i8 1
full:
{crit_exit}  ret i8 0
}}

define i8 @__syn_q_view_pop_{key}(ptr %hdr, ptr %items_base, i32 %cap, ptr %out_val, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cnt_p = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 2
  %cnt = load i32, ptr %cnt_p, align 4
  %has_c = icmp sgt i32 %cap, 0
  %avail = icmp sgt i32 %cnt, 0
  %ok = and i1 %has_c, %avail
  br i1 %ok, label %do_pop, label %empty
do_pop:
  %head_p = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 0
  %head = load i32, ptr %head_p, align 4
  %slot_p = getelementptr inbounds {ety}, ptr %items_base, i32 %head
  %val = load {ety}, ptr %slot_p, align 1
  store {ety} %val, ptr %out_val, align 1{clear_slot}
  %bp = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 3, i32 %head
  %sb = load i8, ptr %bp, align 1
  store i8 0, ptr %bp, align 1
  %h1 = add i32 %head, 1
  %hwrap = icmp uge i32 %h1, %cap
  %hsub = sub i32 %h1, %cap
  %hnext = select i1 %hwrap, i32 %hsub, i32 %h1
  store i32 %hnext, ptr %head_p, align 4
  %c1 = sub i32 %cnt, 1
  store i32 %c1, ptr %cnt_p, align 4
  %has_pb = icmp ne i8 %sb, 0
  br i1 %has_pb, label %apply_pb, label %wake_senders
apply_pb:
  %now = {call_kw} i32 @__syn_now_ms()
  %sb32 = zext i8 %sb to i32
  %pb32 = sub i32 %sb32, 1
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %pb32, i32 %now)
  br label %wake_senders
wake_senders:
  %q_tx = getelementptr inbounds %prim.inbox_hdr, ptr %hdr, i32 0, i32 1
  {call_kw} void @__syn_wake_waiters(ptr %q_tx, i32 -1)
{crit_exit}  ret i8 1
empty:
{crit_exit}  ret i8 0
}}"#
            );
            if is_lease {
                let _ = writeln!(
                    out,
                    r#"define i8 @__syn_q_view_push_{key}(ptr %hdr, ptr %items_base, i32 %cap, {ety} %val, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %ok = {call_kw} i8 @__syn_q_view_push_keep_{key}(ptr %hdr, ptr %items_base, i32 %cap, {ety} %val, i32 %boost)
  %ok_b = icmp ne i8 %ok, 0
  br i1 %ok_b, label %done, label %drop_lease
drop_lease:
  %ph = extractvalue {ety} %val, 1
  %idx = extractvalue {ety} %val, 2
  {call_kw} void @__syn_pool_release(ptr %ph, i8 %idx, i8 -1)
  ret i8 0
done:
  ret i8 1
}}"#
                );
            }
        }

        // Emit capacity-erased unsized `*byte_ring` view helpers (pure SSA, zero non-entry allocas)
        if self.uses_byte_ring_view {
            let (crit_enter, crit_exit) = crit_pair(vis.is_byte_ring_view_contended(), call_kw);
            let usize_ty = self.spec.isa.usize_llvm_ty();
            let slice_ty = format!("{{ ptr, {usize_ty} }}");
            let res_slice_ty = format!("{{ {slice_ty}, i32 }}");
            let to_usize = |src: &str, dst: &str| -> String {
                match usize_ty {
                    "i16" => format!("  {dst} = trunc i32 {src} to i16"),
                    "i64" => format!("  {dst} = zext i32 {src} to i64"),
                    _ => format!("  {dst} = add i32 {src}, 0"),
                }
            };
            let conv_req = to_usize("%req", "%req_u");
            let conv_s1_r = to_usize("%s1", "%s1_u");
            let conv_s1_p = to_usize("%s1", "%s1_u");
            let mk_slice_ok = |tag: &str, ptr_v: &str, len_u: &str| -> String {
                format!(
                    "  %sl0_{tag} = insertvalue {slice_ty} poison, ptr {ptr_v}, 0\n  %sl1_{tag} = insertvalue {slice_ty} %sl0_{tag}, {usize_ty} {len_u}, 1\n  %ok0_{tag} = insertvalue {res_slice_ty} zeroinitializer, {slice_ty} %sl1_{tag}, 0\n  ret {res_slice_ty} %ok0_{tag}"
                )
            };
            let ok_z = mk_slice_ok("z", "%pz", "0");
            let ok_back = mk_slice_ok("b", "%pb", "%req_u");
            let ok_wrap = mk_slice_ok("w", "%pw", "%req_u");
            let ok_gap = mk_slice_ok("g", "%pg", "%req_u");
            let _ = writeln!(
                out,
                r#"define i32 @__syn_ring_view_seg1(ptr %r, i32 %cap){fn_as} nounwind{fn_sz} {{
entry:
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_z = icmp eq i32 %cnt, 0
  br i1 %is_z, label %z, label %nz
z:
  ret i32 0
nz:
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %has_se = icmp ugt i32 %se, 0
  br i1 %has_se, label %use_se, label %no_se
use_se:
  %d0 = sub i32 %se, %h
  ret i32 %d0
no_se:
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %h_lt_t = icmp ult i32 %h, %t
  %d1 = sub i32 %t, %h
  %d2 = sub i32 %cap, %h
  %res = select i1 %h_lt_t, i32 %d1, i32 %d2
  ret i32 %res
}}

define i32 @__syn_ring_view_free_nolock(ptr %r, i32 %cap){fn_as} nounwind{fn_sz} {{
entry:
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  br i1 %has_se, label %se_p, label %no_se
se_p:
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %h = load i32, ptr %hp, align 4
  %t = load i32, ptr %tp, align 4
  %fs0 = sub i32 %h, %t
  ret i32 %fs0
no_se:
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %fs1 = sub i32 %cap, %cnt
  ret i32 %fs1
}}

define i32 @__syn_ring_view_free_space(ptr %r, i32 %cap){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %fs = {call_kw} i32 @__syn_ring_view_free_nolock(ptr %r, i32 %cap)
{crit_exit}  ret i32 %fs
}}

define void @__syn_ring_view_advance(ptr %r, i32 %cap, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %lt = icmp ult i32 %req, %cnt
  %rem = select i1 %lt, i32 %req, i32 %cnt
  %is_z = icmp eq i32 %rem, 0
  br i1 %is_z, label %done, label %chk_all
chk_all:
  %s1 = {call_kw} i32 @__syn_ring_view_seg1(ptr %r, i32 %cap)
  %is_all = icmp eq i32 %rem, %cnt
  br i1 %is_all, label %all_empty, label %partial
all_empty:
  %hp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %sp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %gp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  %t0 = load i32, ptr %tp0, align 4
  %gc = load i32, ptr %gp0, align 4
  %has_g = icmp ugt i32 %gc, 0
  %nh0 = select i1 %has_g, i32 %t0, i32 0
  store i32 %nh0, ptr %hp0, align 4
  br i1 %has_g, label %skip_tail_rst, label %do_tail_rst
do_tail_rst:
  store i32 0, ptr %tp0, align 4
  br label %skip_tail_rst
skip_tail_rst:
  store i32 0, ptr %cp, align 4
  store i32 0, ptr %sp0, align 4
  ret void
partial:
  %hp1 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h1 = load i32, ptr %hp1, align 4
  %in_s1 = icmp ult i32 %rem, %s1
  br i1 %in_s1, label %adv_h, label %wrap_h
adv_h:
  %nh1 = add i32 %h1, %rem
  %nc1 = sub i32 %cnt, %rem
  store i32 %nh1, ptr %hp1, align 4
  store i32 %nc1, ptr %cp, align 4
  ret void
wrap_h:
  %nh2 = sub i32 %rem, %s1
  %nc2 = sub i32 %cnt, %rem
  %sp2 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  store i32 %nh2, ptr %hp1, align 4
  store i32 %nc2, ptr %cp, align 4
  store i32 0, ptr %sp2, align 4
  ret void
done:
  ret void
}}

define void @__syn_ring_view_wake(ptr %r, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
  %has_b = icmp sge i32 %boost, 0
  br i1 %has_b, label %chk_b, label %do_wake
chk_b:
  %b_p1 = add i32 %boost, 1
  %enc = trunc i32 %b_p1 to i8
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 6
  %cur = load i8, ptr %bp, align 1
  %is_z = icmp eq i8 %cur, 0
  %better = icmp ult i8 %enc, %cur
  %upd = or i1 %is_z, %better
  br i1 %upd, label %set_b, label %do_wake
set_b:
  store i8 %enc, ptr %bp, align 1
  br label %do_wake
do_wake:
  {call_kw} void @__syn_wake_waiters(ptr %r, i32 %boost)
  ret void
}}

define void @__syn_ring_hdr_wake(ptr %r, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  {call_kw} void @__syn_ring_view_wake(ptr %r, i32 %boost)
{crit_exit}  ret void
}}

define void @__syn_ring_view_apply_boost(ptr %r, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 6
  %enc = load i8, ptr %bp, align 1
  %has_b = icmp ne i8 %enc, 0
  %has_t = icmp ne ptr %task, null
  %go = and i1 %has_b, %has_t
  br i1 %go, label %apply, label %done
apply:
  store i8 0, ptr %bp, align 1
  %enc32 = zext i8 %enc to i32
  %prio = sub i32 %enc32, 1
  %now = {call_kw} i32 @__syn_now_ms()
  {call_kw} void @__syn_boost_task_ptr(ptr %task, i32 %prio, i32 %now)
  br label %done
done:
  ret void
}}

define i8 @__syn_ring_view_push_byte(ptr %r, i32 %cap, i8 %b, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 0, ptr %gp, align 4
  %fs = {call_kw} i32 @__syn_ring_view_free_nolock(ptr %r, i32 %cap)
  %has = icmp ugt i32 %fs, 0
  br i1 %has, label %do_wr, label %full
do_wr:
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %dp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 %t
  store i8 %b, ptr %dp, align 1
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  %t1 = add i32 %t, 1
  %wrap = icmp uge i32 %t1, %cap
  %nt_circ = select i1 %wrap, i32 0, i32 %t1
  %nt = select i1 %has_se, i32 %t1, i32 %nt_circ
  store i32 %nt, ptr %tp, align 4
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %nc = add i32 %cnt, 1
  store i32 %nc, ptr %cp, align 4
  {call_kw} void @__syn_ring_view_wake(ptr %r, i32 %boost)
{crit_exit}  ret i8 1
full:
{crit_exit}  ret i8 0
}}

define i32 @__syn_ring_view_push(ptr %r, i32 %cap, ptr %src_ptr, i32 %src_len, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 0, ptr %gp, align 4
  %fs = {call_kw} i32 @__syn_ring_view_free_nolock(ptr %r, i32 %cap)
  %is_null = icmp eq ptr %src_ptr, null
  %lt = icmp ult i32 %src_len, %fs
  %nw_raw = select i1 %lt, i32 %src_len, i32 %fs
  %nw = select i1 %is_null, i32 0, i32 %nw_raw
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %has_se = icmp ugt i32 %se, 0
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %t_init = load i32, ptr %tp, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %body ]
  %t_cur = phi i32 [ %t_init, %entry ], [ %t_next, %body ]
  %done = icmp uge i32 %i, %nw
  br i1 %done, label %finish, label %body
body:
  %sp_in = getelementptr i8, ptr %src_ptr, i32 %i
  %bv = load i8, ptr %sp_in, align 1
  %dp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 %t_cur
  store i8 %bv, ptr %dp, align 1
  %t1 = add i32 %t_cur, 1
  %wrap = icmp uge i32 %t1, %cap
  %t_circ = select i1 %wrap, i32 0, i32 %t1
  %t_next = select i1 %has_se, i32 %t1, i32 %t_circ
  %inext = add i32 %i, 1
  br label %loop
finish:
  store i32 %t_cur, ptr %tp, align 4
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %nc = add i32 %cnt, %nw
  store i32 %nc, ptr %cp, align 4
  %any = icmp ugt i32 %nw, 0
  br i1 %any, label %wake, label %exit
wake:
  {call_kw} void @__syn_ring_view_wake(ptr %r, i32 %boost)
  br label %exit
exit:
{crit_exit}  ret i32 %nw
}}

define {res_slice_ty} @__syn_ring_view_grant_write(ptr %r, i32 %cap, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}{conv_req}
  %is_z = icmp eq i32 %req, 0
  br i1 %is_z, label %ret_z, label %chk_cap
ret_z:
  %tp_z = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %tz = load i32, ptr %tp_z, align 4
  %gop_z = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp_z = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 %tz, ptr %gop_z, align 4
  store i32 0, ptr %gcp_z, align 4
  %pz = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 0
{crit_exit}{ok_z}
chk_cap:
  %too_big = icmp ugt i32 %req, %cap
  br i1 %too_big, label %fail, label %chk_empty
chk_empty:
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_emp = icmp eq i32 %cnt, 0
  br i1 %is_emp, label %empty_ok, label %non_empty
empty_ok:
  %hp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %sp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %gop0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp0 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 0, ptr %hp0, align 4
  store i32 0, ptr %tp0, align 4
  store i32 0, ptr %sp0, align 4
  store i32 0, ptr %gop0, align 4
  store i32 %req, ptr %gcp0, align 4
  %pz_ok = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 0
{crit_exit}  %sl0_e = insertvalue {slice_ty} poison, ptr %pz_ok, 0
  %sl1_e = insertvalue {slice_ty} %sl0_e, {usize_ty} %req_u, 1
  %ok_e = insertvalue {res_slice_ty} zeroinitializer, {slice_ty} %sl1_e, 0
  ret {res_slice_ty} %ok_e
non_empty:
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %h = load i32, ptr %hp, align 4
  %t = load i32, ptr %tp, align 4
  %se = load i32, ptr %sp, align 4
  %no_se = icmp eq i32 %se, 0
  %h_lt_t = icmp ult i32 %h, %t
  %unwrapped = and i1 %no_se, %h_lt_t
  br i1 %unwrapped, label %back_or_wrap, label %mid_gap
back_or_wrap:
  %rem_back = sub i32 %cap, %t
  %fit_back = icmp ule i32 %req, %rem_back
  br i1 %fit_back, label %ok_b, label %try_wrap
ok_b:
  %gop_b = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp_b = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 %t, ptr %gop_b, align 4
  store i32 %req, ptr %gcp_b, align 4
  %pb = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 %t
{crit_exit}{ok_back}
try_wrap:
  %fit_wrap = icmp ule i32 %req, %h
  br i1 %fit_wrap, label %ok_w, label %fail
ok_w:
  %gop_w = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp_w = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 0, ptr %gop_w, align 4
  store i32 %req, ptr %gcp_w, align 4
  %pw = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 0
{crit_exit}{ok_wrap}
mid_gap:
  %has_se = icmp ugt i32 %se, 0
  %h_gt_t = icmp ugt i32 %h, %t
  %has_gap = or i1 %has_se, %h_gt_t
  %raw_gap = sub i32 %h, %t
  %gap = select i1 %has_gap, i32 %raw_gap, i32 0
  %fit_gap = icmp ule i32 %req, %gap
  br i1 %fit_gap, label %ok_g, label %fail
ok_g:
  %gop_g = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp_g = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  store i32 %t, ptr %gop_g, align 4
  store i32 %req, ptr %gcp_g, align 4
  %pg = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 %t
{crit_exit}{ok_gap}
fail:
{crit_exit}  %err_r = insertvalue {res_slice_ty} zeroinitializer, i32 1, 1
  ret {res_slice_ty} %err_r
}}

define void @__syn_ring_view_commit_write(ptr %r, i32 %cap, i32 %req, i32 %boost){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %gcp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  %gc = load i32, ptr %gcp, align 4
  store i32 0, ptr %gcp, align 4
  %lt = icmp ult i32 %req, %gc
  %nw = select i1 %lt, i32 %req, i32 %gc
  %is_z = icmp eq i32 %nw, 0
  br i1 %is_z, label %done, label %apply
apply:
  %gop = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %goff = load i32, ptr %gop, align 4
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %t = load i32, ptr %tp, align 4
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %se = load i32, ptr %sp, align 4
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %is_w0 = icmp eq i32 %goff, 0
  %t_gt0 = icmp ugt i32 %t, 0
  %se_eq0 = icmp eq i32 %se, 0
  %c0 = and i1 %is_w0, %t_gt0
  %is_early = and i1 %c0, %se_eq0
  br i1 %is_early, label %early_wrap, label %linear
early_wrap:
  %cnt_z = icmp eq i32 %cnt, 0
  br i1 %cnt_z, label %ew_empty, label %ew_split
ew_empty:
  %hp_e = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  store i32 0, ptr %hp_e, align 4
  store i32 %nw, ptr %tp, align 4
  store i32 %nw, ptr %cp, align 4
  br label %wake
ew_split:
  store i32 %t, ptr %sp, align 4
  store i32 %nw, ptr %tp, align 4
  %nc_ew = add i32 %cnt, %nw
  store i32 %nc_ew, ptr %cp, align 4
  br label %wake
linear:
  %nt = add i32 %t, %nw
  %wrap = icmp uge i32 %nt, %cap
  %do_wrap = and i1 %se_eq0, %wrap
  %nt_final = select i1 %do_wrap, i32 0, i32 %nt
  store i32 %nt_final, ptr %tp, align 4
  %nc_l = add i32 %cnt, %nw
  store i32 %nc_l, ptr %cp, align 4
  br label %wake
wake:
  {call_kw} void @__syn_ring_view_wake(ptr %r, i32 %boost)
  br label %done
done:
{crit_exit}  ret void
}}

define {slice_ty} @__syn_ring_view_read_slice(ptr %r, i32 %cap){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %s1 = {call_kw} i32 @__syn_ring_view_seg1(ptr %r, i32 %cap)
  %has = icmp ugt i32 %s1, 0
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %idx = select i1 %has, i32 %h, i32 0
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 %idx
{crit_exit}{conv_s1_r}
  %sl0 = insertvalue {slice_ty} poison, ptr %bp, 0
  %sl1 = insertvalue {slice_ty} %sl0, {usize_ty} %s1_u, 1
  ret {slice_ty} %sl1
}}

define {slice_ty} @__syn_ring_view_poll_read(ptr %r, i32 %cap, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %s1 = {call_kw} i32 @__syn_ring_view_seg1(ptr %r, i32 %cap)
  %has = icmp ugt i32 %s1, 0
  br i1 %has, label %ready, label %empty
ready:
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 %h
  {call_kw} void @__syn_ring_view_apply_boost(ptr %r, ptr %task)
{crit_exit}{conv_s1_p}
  %sl0 = insertvalue {slice_ty} poison, ptr %bp, 0
  %sl1 = insertvalue {slice_ty} %sl0, {usize_ty} %s1_u, 1
  ret {slice_ty} %sl1
empty:
{crit_exit}  ret {slice_ty} zeroinitializer
}}

define void @__syn_ring_view_consume(ptr %r, i32 %cap, i32 %req){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  {call_kw} void @__syn_ring_view_advance(ptr %r, i32 %cap, i32 %req)
{crit_exit}  ret void
}}

define i32 @__syn_ring_view_pop(ptr %r, i32 %cap, ptr %dst_ptr, i32 %dst_len, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %lt = icmp ult i32 %dst_len, %cnt
  %rem_raw = select i1 %lt, i32 %dst_len, i32 %cnt
  %is_null = icmp eq ptr %dst_ptr, null
  %rem = select i1 %is_null, i32 0, i32 %rem_raw
  %has = icmp ugt i32 %rem, 0
  br i1 %has, label %do_pop, label %empty
do_pop:
  %s1 = {call_kw} i32 @__syn_ring_view_seg1(ptr %r, i32 %cap)
  %lt_s1 = icmp ult i32 %rem, %s1
  %c1 = select i1 %lt_s1, i32 %rem, i32 %s1
  %c2 = sub i32 %rem, %c1
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  br label %l1
l1:
  %i1 = phi i32 [ 0, %do_pop ], [ %i1n, %b1 ]
  %d1 = icmp uge i32 %i1, %c1
  br i1 %d1, label %l2_init, label %b1
b1:
  %src1_i = add i32 %h, %i1
  %sp1 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 %src1_i
  %v1 = load i8, ptr %sp1, align 1
  %dp1 = getelementptr i8, ptr %dst_ptr, i32 %i1
  store i8 %v1, ptr %dp1, align 1
  %i1n = add i32 %i1, 1
  br label %l1
l2_init:
  br label %l2
l2:
  %i2 = phi i32 [ 0, %l2_init ], [ %i2n, %b2 ]
  %d2 = icmp uge i32 %i2, %c2
  br i1 %d2, label %adv, label %b2
b2:
  %sp2 = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 %i2
  %v2 = load i8, ptr %sp2, align 1
  %dst2_i = add i32 %c1, %i2
  %dp2 = getelementptr i8, ptr %dst_ptr, i32 %dst2_i
  store i8 %v2, ptr %dp2, align 1
  %i2n = add i32 %i2, 1
  br label %l2
adv:
  {call_kw} void @__syn_ring_view_advance(ptr %r, i32 %cap, i32 %rem)
  {call_kw} void @__syn_ring_view_apply_boost(ptr %r, ptr %task)
{crit_exit}  ret i32 %rem
empty:
{crit_exit}  ret i32 0
}}

define {{ i8, i32 }} @__syn_ring_view_try_pop_byte(ptr %r, i32 %cap, ptr %task){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %cnt = load i32, ptr %cp, align 4
  %has = icmp ugt i32 %cnt, 0
  br i1 %has, label %ok, label %empty
ok:
  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %h = load i32, ptr %hp, align 4
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 7, i32 %h
  %b = load i8, ptr %bp, align 1
  {call_kw} void @__syn_ring_view_advance(ptr %r, i32 %cap, i32 1)
  {call_kw} void @__syn_ring_view_apply_boost(ptr %r, ptr %task)
{crit_exit}  %r0 = insertvalue {{ i8, i32 }} zeroinitializer, i8 %b, 0
  ret {{ i8, i32 }} %r0
empty:
{crit_exit}  %re = insertvalue {{ i8, i32 }} zeroinitializer, i32 1, 1
  ret {{ i8, i32 }} %re
}}

define void @__syn_ring_view_clear(ptr %r){fn_as} nounwind{fn_sz} {{
entry:
{crit_enter}  %hp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 0
  %tp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 1
  %cp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 2
  %sp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 3
  %gop = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 4
  %gcp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 5
  %bp = getelementptr inbounds %prim.byte_ring_hdr, ptr %r, i32 0, i32 6
  store i32 0, ptr %hp, align 4
  store i32 0, ptr %tp, align 4
  store i32 0, ptr %cp, align 4
  store i32 0, ptr %sp, align 4
  store i32 0, ptr %gop, align 4
  store i32 0, ptr %gcp, align 4
  store i8 0, ptr %bp, align 1
{crit_exit}  ret void
}}"#
            );
        }
    }

    fn emit_coroutine_frame_globals_ir(&self, out: &mut String) {
        let _ = writeln!(
            out,
            "; ── Packed Static Coroutine Frames (with Union-Aliased Child Storage) ──"
        );
        for f in &self.module.functions {
            if let Some(layout) = self.fn_frames.get(&f.name) {
                let joined = layout.field_llvm_tys.join(", ");
                let _ = writeln!(out, "{} = type {{ {joined} }}", layout.llvm_name);
            }
        }
        for bp in &self.module.task_blueprints {
            if let Some(layout) = self.bp_frames.get(&bp.name) {
                let joined = layout.field_llvm_tys.join(", ");
                let _ = writeln!(out, "{} = type {{ {joined} }}", layout.llvm_name);
            }
        }
        for callee in &self.module.boot.boot_yielding_callees {
            if let Some(layout) = self.fn_frames.get(callee) {
                let sym = format!("@__syn_boot_frame_{}", sanitize_ident(callee));
                let _ = writeln!(out, "{sym} = global {} zeroinitializer, align 8", layout.llvm_name);
            }
        }
        for sp in &self.module.spawned_tasks {
            if let Some(layout) = self.bp_frames.get(&sp.blueprint_name) {
                let sym = format!("@__syn_task_frame_{}", sp.slot_id);
                let _ = writeln!(
                    out,
                    "{sym} = global {} zeroinitializer, align 8 ; instance `{}`",
                    layout.llvm_name, sp.instance_name
                );
            }
        }
        let _ = writeln!(out);
    }

    fn emit_extern_fns_ir(&self, out: &mut String) {
        if self.module.extern_fns.is_empty() {
            return;
        }
        let fn_as = self.fn_as();
        let _ = writeln!(out, "; ── Genuine External \"C\" FFI Declarations ──");
        for ext in &self.module.extern_fns {
            let ret = self.llvm_ty(&ext.ret_ty);
            let mut ps: Vec<String> =
                ext.params.iter().map(|(_, t)| self.llvm_ty(t)).collect();
            if ext.variadic {
                ps.push("...".to_string());
            }
            let _ = writeln!(
                out,
                "declare {ret} @{}({}){fn_as} nounwind",
                sanitize_ident(&ext.name),
                ps.join(", ")
            );
        }
        let _ = writeln!(out);
    }

    fn emit_isr_ir(&mut self, out: &mut String, isr: &MirIsr) -> Result<(), CodegenError> {
        let sym = format!("@__syn_isr_{}", sanitize_ident(&isr.name));
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let mut param_sig = Vec::new();
        for (idx, &vid) in isr.params.iter().enumerate() {
            let pty = self.llvm_ty(&isr.vars[vid.0].ty);
            param_sig.push(format!("{pty} %arg{idx}"));
        }
        for (sidx, _) in isr.state_vars.iter().enumerate() {
            param_sig.push(format!("ptr %__state_{sidx}"));
        }

        let mut fb = FnIrBuilder::new(self, &isr.vars, None, &MirType::Void, &MirErrorSpec::None);
        for (sidx, sd) in isr.state_vars.iter().enumerate() {
            fb.isr_state_ptrs
                .insert(sd.var_id, format!("%__state_{sidx}"));
        }
        fb.emit_entry_allocas(&isr.params);
        for stmt in &isr.body {
            fb.emit_stmt(stmt)?;
        }
        fb.ensure_terminated_void();
        let body = fb.finish();
        let _ = writeln!(out, "; ── Hardware ISR `{}` ──", isr.name);
        let _ = writeln!(
            out,
            "define void {sym}({}){fn_as} nounwind{fn_sz} {{\n{body}}}\n",
            param_sig.join(", ")
        );
        Ok(())
    }

    fn format_irq_binding_call(&mut self, binding: &MirIrqBinding) -> Result<String, CodegenError> {
        let isr_opt = self
            .module
            .isrs
            .iter()
            .find(|i| i.name == binding.isr_name)
            .cloned();
        let isr_sym = format!("@__syn_isr_{}", sanitize_ident(&binding.isr_name));
        let mut args = Vec::new();

        for (idx, barg) in binding.baked_args.iter().enumerate() {
            let expected_ty = isr_opt
                .as_ref()
                .and_then(|isr| isr.params.get(idx).map(|vid| isr.vars[vid.0].ty.clone()));
            match barg {
                MirBakedArg::PrimAddr(prim_name) => {
                    let raw = format!("@__syn_prim_{}", sanitize_ident(prim_name));
                    if matches!(expected_ty, Some(MirType::PrimView(_))) {
                        if let Some(pk) = self.prim_globals.get(prim_name) {
                            if let Some(wrapped) = self.wrap_const_ptr_as_view(&raw, pk) {
                                args.push(wrapped);
                                continue;
                            }
                        }
                    }
                    args.push(format!("ptr {raw}"));
                }
                MirBakedArg::SubsystemPath {
                    root,
                    root_struct,
                    field_chain,
                } => {
                    let mut cur_ptr = format!("@__zev_subsys_{}", sanitize_ident(root));
                    let mut cur_struct = root_struct.as_str();
                    let mut leaf_ty: Option<MirType> = Some(MirType::Struct(root_struct.clone()));
                    for (fname, _) in field_chain {
                        let si = &self.structs[cur_struct];
                        let (fidx, fty) = &si.field_indices[fname];
                        cur_ptr = format!(
                            "getelementptr inbounds ({}, ptr {cur_ptr}, i32 0, i32 {fidx})",
                            si.llvm_name
                        );
                        leaf_ty = Some(fty.clone());
                        if let MirType::Struct(next_s) = fty {
                            cur_struct = next_s.as_str();
                        }
                    }
                    if matches!(expected_ty, Some(MirType::PrimView(_))) {
                        if let Some(MirType::Primitive(ref pk)) = leaf_ty {
                            if let Some(wrapped) = self.wrap_const_ptr_as_view(&cur_ptr, pk) {
                                args.push(wrapped);
                                continue;
                            }
                        }
                    }
                    args.push(format!("ptr {cur_ptr}"));
                }
                MirBakedArg::MmioAddr(inst_name) => {
                    if self.spec.isa == CpuIsa::HostX86_64Linux {
                        args.push(format!("ptr @__zev_mmio_buf_{}", sanitize_ident(inst_name)));
                    } else {
                        let addr = self
                            .mmio_instances
                            .get(inst_name)
                            .map_or(0, |m| m.base_addr as i32);
                        args.push(format!("ptr inttoptr (i32 {addr} to ptr)"));
                    }
                }
                MirBakedArg::Const(expr) => {
                    let target_mir_ty = expected_ty.unwrap_or_else(|| expr.ty.clone());
                    let lty = self.llvm_ty(&target_mir_ty);
                    let cval = self.const_expr_to_llvm(expr, &target_mir_ty);
                    args.push(format!("{lty} {cval}"));
                }
            }
        }

        for sv in &binding.state_vars {
            let sym = sanitize_ident(sv.storage_symbol.trim_start_matches('@'));
            args.push(format!("ptr @{sym}"));
        }

        let call_kw = self.call_kw();
        Ok(format!("{call_kw} void {isr_sym}({})", args.join(", ")))
    }

    fn emit_irq_vector_dispatchers_ir(&mut self, out: &mut String) -> Result<(), CodegenError> {
        if self.module.irq_vectors.is_empty() {
            return Ok(());
        }
        let fn_sz = self.opts.fn_size_attrs().to_string();
        let fn_as = self.fn_as();
        let vectors: Vec<MirIrqVector> = self.module.irq_vectors.clone();
        let _ = writeln!(out, "; ── Zero-RAM Static Flash IRQ Vector Dispatchers ──");
        for vec in &vectors {
            let irq_num = vec.irq_num;
            let _ = writeln!(
                out,
                "define void @__syn_vec_irq_{irq_num}(){fn_as} nounwind{fn_sz} {{"
            );
            let _ = writeln!(out, "entry:");

            let eff_selector: Option<String> = vec.selector_symbol.clone().or_else(|| {
                vec.bindings
                    .iter()
                    .find_map(|b| b.cond_selector.as_ref().map(|(s, _)| s.clone()))
            });

            if let Some(sel_raw) = eff_selector {
                let sel_sym = sanitize_ident(sel_raw.trim_start_matches('@'));
                let _ = writeln!(out, "  %sel = load i8, ptr @{sel_sym}, align 1");
                if !vec.is_shared {
                    let mut uncond_calls = Vec::new();
                    let mut grouped: BTreeMap<u8, Vec<String>> = BTreeMap::new();
                    for b in &vec.bindings {
                        let call_ir = self.format_irq_binding_call(b)?;
                        if let Some((_, val)) = &b.cond_selector {
                            grouped.entry(*val).or_default().push(call_ir);
                        } else {
                            uncond_calls.push(call_ir);
                        }
                    }
                    for call_ir in uncond_calls {
                        let _ = writeln!(out, "  {call_ir}");
                    }
                    let _ = writeln!(out, "  switch i8 %sel, label %done [");
                    for &val in grouped.keys() {
                        let _ = writeln!(out, "    i8 {val}, label %arm_{val}");
                    }
                    let _ = writeln!(out, "  ]");
                    for (val, calls) in grouped {
                        let _ = writeln!(out, "arm_{val}:");
                        for call_ir in calls {
                            let _ = writeln!(out, "  {call_ir}");
                        }
                        let _ = writeln!(out, "  br label %done");
                    }
                    let _ = writeln!(out, "done:");
                    let _ = writeln!(out, "  ret void");
                } else {
                    for (idx, b) in vec.bindings.iter().enumerate() {
                        let call_ir = self.format_irq_binding_call(b)?;
                        if let Some((_, val)) = &b.cond_selector {
                            let _ = writeln!(out, "  %cmp_{idx} = icmp eq i8 %sel, {val}");
                            let _ = writeln!(
                                out,
                                "  br i1 %cmp_{idx}, label %shared_exec_{idx}, label %shared_next_{idx}"
                            );
                            let _ = writeln!(out, "shared_exec_{idx}:");
                            let _ = writeln!(out, "  {call_ir}");
                            let _ = writeln!(out, "  br label %shared_next_{idx}");
                            let _ = writeln!(out, "shared_next_{idx}:");
                        } else {
                            let _ = writeln!(out, "  {call_ir}");
                        }
                    }
                    let _ = writeln!(out, "  ret void");
                }
            } else {
                for b in &vec.bindings {
                    let call_ir = self.format_irq_binding_call(b)?;
                    let _ = writeln!(out, "  {call_ir}");
                }
                let _ = writeln!(out, "  ret void");
            }

            let _ = writeln!(out, "}}\n");
        }
        Ok(())
    }

    fn emit_function_ir(
        &mut self,
        out: &mut String,
        func: &MirFunction,
    ) -> Result<(), CodegenError> {
        let sig = self.fn_sigs[&func.name].clone();
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        match &func.body {
            MirFnBody::Direct(stmts) => {
                let mut param_sig = Vec::new();
                for (idx, &vid) in func.params.iter().enumerate() {
                    let pty = self.llvm_ty(&func.vars[vid.0].ty);
                    param_sig.push(format!("{pty} %arg{idx}"));
                }
                let mut fb = FnIrBuilder::new(
                    self,
                    &func.vars,
                    None,
                    &func.ret_ty,
                    &func.error_spec,
                );
                fb.emit_entry_allocas(&func.params);
                for stmt in stmts {
                    fb.emit_stmt(stmt)?;
                }
                fb.ensure_terminated_default();
                let body = fb.finish();
                let _ = writeln!(
                    out,
                    "define {} {}({}){fn_as} nounwind{fn_sz} {{\n{body}}}\n",
                    sig.ret_llvm_ty,
                    sig.symbol,
                    param_sig.join(", ")
                );
            }
            MirFnBody::Coroutine { blocks, .. } => {
                let layout = self.fn_frames[&func.name].clone();
                let mut fb = FnIrBuilder::new(
                    self,
                    &func.vars,
                    Some(&layout),
                    &func.ret_ty,
                    &func.error_spec,
                );
                fb.emit_coroutine_cfg(blocks)?;
                let body = fb.finish();
                let _ = writeln!(
                    out,
                    "define i32 {}(ptr %__frame, ptr %__task){fn_as} nounwind{fn_sz} {{\n{body}}}\n",
                    sig.symbol
                );
            }
        }
        Ok(())
    }

    fn emit_task_blueprint_ir(
        &mut self,
        out: &mut String,
        bp: &MirTaskBlueprint,
    ) -> Result<(), CodegenError> {
        let layout = self.bp_frames[&bp.name].clone();
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let sym = format!("@__syn_task_step_{}", sanitize_ident(&bp.name));
        let mut fb = FnIrBuilder::new(
            self,
            &bp.vars,
            Some(&layout),
            &MirType::Void,
            &MirErrorSpec::None,
        );
        fb.emit_coroutine_cfg(&bp.blocks)?;
        let body = fb.finish();
        let _ = writeln!(
            out,
            "define i32 {sym}(ptr %__frame, ptr %__task){fn_as} nounwind{fn_sz} {{\n{body}}}\n"
        );
        Ok(())
    }

    fn emit_boot_ir(&mut self, out: &mut String) -> Result<(), CodegenError> {
        let boot_vars = self.module.boot.vars.clone();
        let boot_present = self.module.boot.present;
        let steps = self.module.boot.steps.clone();
        let fn_sz = self.opts.fn_size_attrs();
        let fn_as = self.fn_as();
        let mut fb = FnIrBuilder::new(
            self,
            &boot_vars,
            None,
            &MirType::Void,
            &MirErrorSpec::None,
        );
        fb.emit_entry_allocas(&[]);
        fb.emit_pool_boot_inits();

        if boot_present {
            for step in &steps {
                match step {
                    MirBootStep::Stmt(s) => fb.emit_stmt(s)?,
                    MirBootStep::RunYieldingFn { callee, args, dst } => {
                        fb.emit_boot_run_yielding(callee, args, dst.as_ref())?;
                    }
                    MirBootStep::SpawnActivate { slot_id, args } => {
                        fb.emit_boot_spawn_activate(*slot_id, args)?;
                    }
                }
            }
        }
        fb.ensure_terminated_void();
        let body = fb.finish();
        let _ = writeln!(out, "; ── Sequential `fn main()` Boot Phase ──");
        let _ = writeln!(
            out,
            "define void @__syn_boot(){fn_as} nounwind{fn_sz} {{\n{body}}}\n"
        );
        Ok(())
    }

    fn emit_string_constants_ir(&self, out: &mut String) {
        if self.strings.is_empty() && self.const_blobs.is_empty() {
            return;
        }
        let _ = writeln!(out, "; ── Pooled String & Slice Constants (Flash .rodata) ──");
        for (idx, s) in self.strings.iter().enumerate() {
            let bytes = zev_ast::decoded_string_bytes(s);
            let len = bytes.len() + 1;
            let mut esc = String::new();
            for &b in &bytes {
                if (0x20..=0x7E).contains(&b) && b != b'"' && b != b'\\' {
                    esc.push(b as char);
                } else {
                    let _ = write!(esc, "\\{b:02X}");
                }
            }
            esc.push_str("\\00");
            let _ = writeln!(
                out,
                "@.str.{idx} = private unnamed_addr constant [{len} x i8] c\"{esc}\", align 1"
            );
        }
        for (idx, decl_body) in self.const_blobs.iter().enumerate() {
            let _ = writeln!(out, "@.cblob.{idx} = {decl_body}");
        }
    }
}

fn target_int_signed_and_bits(target_ty: &MirType, ptr_bits: u32) -> Option<(bool, u32)> {
    match target_ty.host_value_type() {
        MirType::U8 => Some((false, 8)),
        MirType::I8 => Some((true, 8)),
        MirType::U16 => Some((false, 16)),
        MirType::I16 => Some((true, 16)),
        MirType::U32 => Some((false, 32)),
        MirType::I32 | MirType::Enum(_) => Some((true, 32)),
        MirType::U64 => Some((false, 64)),
        MirType::I64 => Some((true, 64)),
        MirType::Usize | MirType::Ptr(_) | MirType::FnPtr { .. } => Some((false, ptr_bits)),
        MirType::Isize => Some((true, ptr_bits)),
        _ => None,
    }
}

fn saturating_float_to_int_i128(f: f64, target_ty: &MirType, ptr_bits: u32) -> i128 {
    match target_int_signed_and_bits(target_ty, ptr_bits) {
        Some((false, 8)) => (f as u8) as i128,
        Some((true, 8)) => (f as i8) as i128,
        Some((false, 16)) => (f as u16) as i128,
        Some((true, 16)) => (f as i16) as i128,
        Some((false, 32)) => (f as u32) as i128,
        Some((true, 32)) => (f as i32) as i128,
        Some((false, 64)) => (f as u64) as i128,
        Some((true, 64)) => (f as i64) as i128,
        _ => (f as i32) as i128,
    }
}

fn mask_int_to_target_i128(v: i128, target_ty: &MirType, ptr_bits: u32) -> i128 {
    match target_int_signed_and_bits(target_ty, ptr_bits) {
        Some((false, 8)) => (v as u8) as i128,
        Some((true, 8)) => (v as i8) as i128,
        Some((false, 16)) => (v as u16) as i128,
        Some((true, 16)) => (v as i16) as i128,
        Some((false, 32)) => (v as u32) as i128,
        Some((true, 32)) => (v as i32) as i128,
        Some((false, 64)) => (v as u64) as i128,
        Some((true, 64)) => (v as i64) as i128,
        _ => v,
    }
}

fn llvm_int_bits(ty: &str) -> u32 {
    match ty {
        "i1" => 1,
        "i8" => 8,
        "i16" => 16,
        "i32" => 32,
        "i64" => 64,
        _ => 32,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-Function / Coroutine LLVM 23 SSA IR Builder
// ─────────────────────────────────────────────────────────────────────────────

struct FnIrBuilder<'ctx, 'm> {
    ctx: &'ctx mut ModuleCtx<'m>,
    vars: Vec<MirVar>,
    coroutine: Option<CoroutineFrameLayout>,
    isr_state_ptrs: HashMap<VarId, String>,
    ret_ty: MirType,
    error_spec: MirErrorSpec,
    buf: String,
    entry_allocas: String,
    entry_alloca_insert_pos: Option<usize>,
    temp_alloca_id: usize,
    tmp_id: usize,
    lbl_id: usize,
    cur_label: String,
    terminated: bool,
    loop_targets: Vec<(String, String)>, // (continue_lbl, break_lbl) for Direct fns
}

impl<'ctx, 'm> FnIrBuilder<'ctx, 'm> {
    fn new(
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

    fn finish(mut self) -> String {
        if !self.entry_allocas.is_empty() {
            let pos = self
                .entry_alloca_insert_pos
                .expect("entry block must exist before finish");
            self.buf.insert_str(pos, &self.entry_allocas);
        }
        self.buf
    }

    fn stack_align(&self, lty: &str) -> u32 {
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

    fn emit_temp_alloca(&mut self, lty: &str) -> (String, u32) {
        let id = self.temp_alloca_id;
        self.temp_alloca_id += 1;
        let name = format!("%__a{id}");
        let al = self.stack_align(lty);
        let _ = writeln!(self.entry_allocas, "  {name} = alloca {lty}, align {al}");
        (name, al)
    }

    fn next_tmp(&mut self) -> String {
        let id = self.tmp_id;
        self.tmp_id += 1;
        format!("%t{id}")
    }

    fn next_label(&mut self, prefix: &str) -> String {
        let id = self.lbl_id;
        self.lbl_id += 1;
        format!("{prefix}_{id}")
    }

    fn start_block(&mut self, label: &str) {
        let _ = writeln!(self.buf, "{label}:");
        self.cur_label = label.to_string();
        self.terminated = false;
    }

    fn emit_line(&mut self, line: impl AsRef<str>) {
        if !self.terminated {
            let _ = writeln!(self.buf, "  {}", line.as_ref());
        }
    }

    fn emit_term(&mut self, line: impl AsRef<str>) {
        if !self.terminated {
            let _ = writeln!(self.buf, "  {}", line.as_ref());
            self.terminated = true;
        }
    }

    fn emit_entry_allocas(&mut self, param_vids: &[VarId]) {
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
            self.emit_line(format!("store {lty} zeroinitializer, ptr %v_{}, align {al}", v.id.0));
        }
        for (idx, &vid) in param_vids.iter().enumerate() {
            let lty = self.ctx.llvm_ty(&self.vars[vid.0].ty);
            let al = self.stack_align(&lty);
            self.emit_line(format!("store {lty} %arg{idx}, ptr %v_{}, align {al}", vid.0));
        }
    }

    fn call_kw(&self) -> &'static str {
        self.ctx.call_kw()
    }

    fn emit_avr_safe_i32_load(&mut self, ptr_repr: &str, contended: bool) -> String {
        let val = self.next_tmp();
        if self.ctx.spec.isa.is_avr() && contended {
            let call_kw = self.call_kw();
            let pm = self.next_tmp();
            self.emit_line(format!("{pm} = {call_kw} i32 @__syn_crit_enter()"));
            self.emit_line(format!("{val} = load volatile i32, ptr {ptr_repr}, align 4"));
            self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
        } else {
            self.emit_line(format!("{val} = load volatile i32, ptr {ptr_repr}, align 4"));
        }
        val
    }

    fn var_ptr(&mut self, vid: VarId) -> String {
        if let Some(state_ptr) = self.isr_state_ptrs.get(&vid) {
            return state_ptr.clone();
        }
        let hoisted_info = self
            .coroutine
            .as_ref()
            .and_then(|coro| coro.hoisted_var_gep.get(&vid).map(|&idx| (coro.llvm_name.clone(), idx)));
        if let Some((cname, gep_idx)) = hoisted_info {
            let p = self.next_tmp();
            self.emit_line(format!(
                "{p} = getelementptr inbounds {cname}, ptr %__frame, i32 0, i32 {gep_idx}"
            ));
            return p;
        }
        format!("%v_{}", vid.0)
    }

    fn coerce(&mut self, val: SsaVal, dst_ty: &str, signed: bool) -> SsaVal {
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
        if (dst_ty == "ptr" || dst_ty == "ptr addrspace(1)") && val.ty.starts_with('i') {
            let t = self.next_tmp();
            self.emit_line(format!("{t} = inttoptr {} {} to {dst_ty}", val.ty, val.repr));
            return SsaVal::new(dst_ty, t);
        }
        if (val.ty == "ptr" || val.ty == "ptr addrspace(1)") && dst_ty.starts_with('i') {
            let t = self.next_tmp();
            self.emit_line(format!("{t} = ptrtoint {} {} to {dst_ty}", val.ty, val.repr));
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

    fn to_i1(&mut self, val: SsaVal) -> String {
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

    fn emit_coroutine_cfg(&mut self, blocks: &[MirBlock]) -> Result<(), CodegenError> {
        let coro = self.coroutine.clone().unwrap();
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

            // On `restart;` back into `bb_0`, zero hoisted non-Param locals while keeping `Param` fields intact!
            if b.id.0 == entry_bb && entry_is_restart_target {
                let vars_snapshot = self.vars.clone();
                for v in &vars_snapshot {
                    if v.hoisted && v.role != VarRole::Param {
                        let vp = self.var_ptr(v.id);
                        let lty = self.ctx.llvm_ty(&v.ty);
                        self.emit_line(format!("store {lty} zeroinitializer, ptr {vp}, align 4"));
                    }
                }
            }

            // Resync while-sleeping overruns at `loop every` body entry (`wdiff >= eff_p`)
            if let Some(&(dv_id, pv_id)) = every_resume_map.get(&b.id) {
                let dl_ptr = self.var_ptr(dv_id);
                let pv_ptr = self.var_ptr(pv_id);
                let pv_lty = self.ctx.llvm_ty(&self.vars[pv_id.0].ty);
                let pv_raw = self.next_tmp();
                self.emit_line(format!("{pv_raw} = load {pv_lty}, ptr {pv_ptr}, align 4"));
                let pv32 = self.coerce(SsaVal::new(pv_lty, pv_raw), "i32", false);
                let p_zero = self.next_tmp();
                let eff_p = self.next_tmp();
                self.emit_line(format!("{p_zero} = icmp eq i32 {}, 0", pv32.repr));
                self.emit_line(format!("{eff_p} = select i1 {p_zero}, i32 1, i32 {}", pv32.repr));
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

    fn store_coro_pc(&mut self, next_pc: i32) {
        let cname = self.coroutine.as_ref().unwrap().llvm_name.clone();
        let p = self.next_tmp();
        self.emit_line(format!(
            "{p} = getelementptr inbounds {cname}, ptr %__frame, i32 0, i32 0"
        ));
        self.emit_line(format!("store i32 {next_pc}, ptr {p}, align 4"));
    }

    fn park_task_deadline(&mut self, wake_tick_val: &SsaVal) {
        let wt = self.coerce(wake_tick_val.clone(), "i32", false);
        let is_null = self.next_tmp();
        let park_lbl = self.next_label("park_dl");
        let after_lbl = self.next_label("after_dl");
        self.emit_line(format!("{is_null} = icmp eq ptr %__task, null"));
        self.emit_term(format!("br i1 {is_null}, label %{after_lbl}, label %{park_lbl}"));
        self.start_block(&park_lbl);
        let st_p = self.next_tmp();
        let wt_p = self.next_tmp();
        let ht_p = self.next_tmp();
        let wo_p = self.next_tmp();
        self.emit_line(format!("{st_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 2"));
        self.emit_line(format!("{ht_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 7"));
        self.emit_line(format!("{wt_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 11"));
        self.emit_line(format!("{wo_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 12"));
        self.emit_line(format!("store i8 2, ptr {st_p}, align 1"));
        self.emit_line(format!("store i8 0, ptr {ht_p}, align 1"));
        self.emit_line(format!("store i32 {}, ptr {wt_p}, align 4", wt.repr));
        self.emit_line(format!("store ptr null, ptr {wo_p}, align 4"));
        self.emit_term(format!("br label %{after_lbl}"));
        self.start_block(&after_lbl);
    }

    fn park_task_prim(&mut self, prim_ptr: &str, timeout_tick: Option<&SsaVal>) {
        let call_kw = self.call_kw();
        let is_null = self.next_tmp();
        let park_lbl = self.next_label("park_prim");
        let after_lbl = self.next_label("after_prim");
        self.emit_line(format!("{is_null} = icmp eq ptr %__task, null"));
        self.emit_term(format!("br i1 {is_null}, label %{after_lbl}, label %{park_lbl}"));
        self.start_block(&park_lbl);
        self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
        let st_p = self.next_tmp();
        let ht_p = self.next_tmp();
        let wt_p = self.next_tmp();
        let wo_p = self.next_tmp();
        self.emit_line(format!("{st_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 2"));
        self.emit_line(format!("{ht_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 7"));
        self.emit_line(format!("{wt_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 11"));
        self.emit_line(format!("{wo_p} = getelementptr inbounds %SynTask, ptr %__task, i32 0, i32 12"));
        self.emit_line(format!("store i8 3, ptr {st_p}, align 1"));
        self.emit_line(format!("store ptr {prim_ptr}, ptr {wo_p}, align 4"));
        if let Some(dl) = timeout_tick {
            let wt = self.coerce(dl.clone(), "i32", false);
            self.emit_line(format!("store i8 1, ptr {ht_p}, align 1"));
            self.emit_line(format!("store i32 {}, ptr {wt_p}, align 4", wt.repr));
        } else {
            self.emit_line(format!("store i8 0, ptr {ht_p}, align 1"));
        }
        self.emit_term(format!("br label %{after_lbl}"));
        self.start_block(&after_lbl);
    }

    fn emit_coroutine_terminator(&mut self, term: &MirTerminator) -> Result<(), CodegenError> {
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
                let b1 = self.to_i1(cv);
                self.emit_term(format!("br i1 {b1}, label %bb_{}, label %bb_{}", then_bb.0, else_bb.0));
            }
            MirTerminator::Switch {
                discr,
                arms,
                default_bb,
            } => {
                let dv = self.emit_expr(discr)?;
                let sw_ty = if llvm_int_bits(&dv.ty) > 32 { "i64" } else { "i32" };
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
                let coro = self.coroutine.clone().unwrap();
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
                let coro = self.coroutine.clone().unwrap();
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

    fn emit_yield(&mut self, yk: &MirYieldKind) -> Result<(), CodegenError> {
        match yk {
            MirYieldKind::EveryLatch {
                next_deadline_var,
                period_var,
                resume_bb,
            } => {
                let dl_ptr = self.var_ptr(*next_deadline_var);
                let pv_ptr = self.var_ptr(*period_var);
                let pv_lty = self.ctx.llvm_ty(&self.vars[period_var.0].ty);
                let pv_raw = self.next_tmp();
                self.emit_line(format!("{pv_raw} = load {pv_lty}, ptr {pv_ptr}, align 4"));
                let pv32 = self.coerce(SsaVal::new(pv_lty, pv_raw), "i32", false);
                let p_zero = self.next_tmp();
                let eff_p = self.next_tmp();
                self.emit_line(format!("{p_zero} = icmp eq i32 {}, 0", pv32.repr));
                self.emit_line(format!("{eff_p} = select i1 {p_zero}, i32 1, i32 {}", pv32.repr));

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
                self.emit_term(format!("br i1 {is_early}, label %{sleep_lbl}, label %{due_lbl}"));

                self.start_block(&due_lbl);
                let q = self.next_tmp();
                let skip = self.next_tmp();
                let resync_dl = self.next_tmp();
                self.emit_line(format!("{q} = udiv i32 {diff}, {eff_p}"));
                self.emit_line(format!("{skip} = mul i32 {q}, {eff_p}"));
                self.emit_line(format!("{resync_dl} = add i32 {next_dl}, {skip}"));
                self.emit_line(format!("store i32 {resync_dl}, ptr {dl_ptr}, align 4"));

                let share = self.next_tmp();
                self.emit_line(format!("{share} = {call_kw} i1 @__syn_should_share(ptr %__task)"));
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
                self.emit_line(format!("{share} = {call_kw} i1 @__syn_should_share(ptr %__task)"));
                let yield_lbl = self.next_label("share_yield");
                self.emit_term(format!(
                    "br i1 {share}, label %{yield_lbl}, label %bb_{}",
                    resume_bb.0
                ));
                self.start_block(&yield_lbl);
                self.store_coro_pc(resume_bb.0 as i32);
                self.emit_term("ret i32 0");
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
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                self.emit_queue_recv_yield(
                    target,
                    &pk,
                    elem_ty,
                    *timeout_deadline_var,
                    *dst_val,
                    *dst_res,
                    *poll_bb,
                    *resume_bb,
                )?;
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
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let qptr = self.resolve_prim_ptr(target)?;
                let vp = self.var_ptr(*val_var);
                let ety = self.ctx.llvm_ty(elem_ty);
                let is_lease = matches!(elem_ty, MirType::Lease(_));
                let push_fn = if is_lease {
                    format!("@__syn_q_push_keep_{id}")
                } else {
                    format!("@__syn_q_push_{id}")
                };
                let call_kw = self.call_kw();
                let ptr_al = self.ctx.ptr_align();
                let v = self.next_tmp();
                self.emit_line(format!("{v} = load {ety}, ptr {vp}, align 1"));
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 {push_fn}(ptr {qptr}, {ety} {v}, i32 -1)"
                ));
                let ok_b = self.next_tmp();
                self.emit_line(format!("{ok_b} = icmp ne i8 {ok}, 0"));
                let ok_lbl = self.next_label("q_tx_ok");
                let full_lbl = self.next_label("q_tx_full");
                self.emit_term(format!("br i1 {ok_b}, label %{ok_lbl}, label %{full_lbl}"));

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
                    self.emit_line(format!("store {rty} zeroinitializer, ptr {rp}, align 4"));
                }
                self.emit_term(format!("br label %bb_{}", resume_bb.0));

                self.start_block(&full_lbl);
                if let Some(dl_var) = timeout_deadline_var {
                    let dl_ptr = self.var_ptr(*dl_var);
                    let dl = self.next_tmp();
                    self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 4"));
                    let now = self.next_tmp();
                    self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                    let diff = self.next_tmp();
                    self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
                    let exp = self.next_tmp();
                    self.emit_line(format!("{exp} = icmp sge i32 {diff}, 0"));
                    let to_lbl = self.next_label("q_tx_to");
                    let park_lbl = self.next_label("q_tx_park");
                    self.emit_term(format!("br i1 {exp}, label %{to_lbl}, label %{park_lbl}"));

                    self.start_block(&to_lbl);
                    if is_lease {
                        let ph = self.next_tmp();
                        let sidx = self.next_tmp();
                        let vp_ph = self.next_tmp();
                        self.emit_line(format!("{ph} = extractvalue {ety} {v}, 1"));
                        self.emit_line(format!("{sidx} = extractvalue {ety} {v}, 2"));
                        self.emit_line(format!(
                            "{call_kw} void @__syn_pool_release(ptr {ph}, i8 {sidx}, i8 -1)"
                        ));
                        self.emit_line(format!(
                            "{vp_ph} = getelementptr inbounds {ety}, ptr {vp}, i32 0, i32 1"
                        ));
                        self.emit_line(format!("store ptr null, ptr {vp_ph}, align {ptr_al}"));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let e0 = self.next_tmp();
                        self.emit_line(format!(
                            "{e0} = insertvalue {rty} zeroinitializer, i32 1, 1"
                        ));
                        self.emit_line(format!("store {rty} {e0}, ptr {rp}, align 4"));
                    }
                    self.emit_term(format!("br label %bb_{}", resume_bb.0));

                    self.start_block(&park_lbl);
                    let pty = format!("%prim.{id}");
                    let q_tx = self.next_tmp();
                    self.emit_line(format!(
                        "{q_tx} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 1"
                    ));
                    let dl_ssa = SsaVal::new("i32", dl);
                    self.park_task_prim(&q_tx, Some(&dl_ssa));
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                } else {
                    let pty = format!("%prim.{id}");
                    let q_tx = self.next_tmp();
                    self.emit_line(format!(
                        "{q_tx} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 1"
                    ));
                    self.park_task_prim(&q_tx, None);
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                }
            }
            MirYieldKind::SignalWait {
                target,
                timeout_deadline_var,
                dst_res,
                poll_bb,
                resume_bb,
            } => {
                let call_kw = self.call_kw();
                let id = sanitize_ident(&MirPrimKind::Signal.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_sig_poll_{id}(ptr {sptr}, ptr %__task)"
                ));
                let got = self.next_tmp();
                self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
                let ok_lbl = self.next_label("sig_ok");
                let miss_lbl = self.next_label("sig_miss");
                self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{miss_lbl}"));

                self.start_block(&ok_lbl);
                if let Some(rvid) = dst_res {
                    let rp = self.var_ptr(*rvid);
                    let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                    self.emit_line(format!("store {rty} zeroinitializer, ptr {rp}, align 4"));
                }
                self.emit_term(format!("br label %bb_{}", resume_bb.0));

                self.start_block(&miss_lbl);
                if let Some(dl_var) = timeout_deadline_var {
                    let dl_ptr = self.var_ptr(*dl_var);
                    let dl = self.next_tmp();
                    self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 4"));
                    let now = self.next_tmp();
                    self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                    let diff = self.next_tmp();
                    self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
                    let exp = self.next_tmp();
                    self.emit_line(format!("{exp} = icmp sge i32 {diff}, 0"));
                    let to_lbl = self.next_label("sig_to");
                    let park_lbl = self.next_label("sig_park");
                    self.emit_term(format!("br i1 {exp}, label %{to_lbl}, label %{park_lbl}"));

                    self.start_block(&to_lbl);
                    self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let e0 = self.next_tmp();
                        self.emit_line(format!("{e0} = insertvalue {rty} zeroinitializer, i32 1, 1"));
                        self.emit_line(format!("store {rty} {e0}, ptr {rp}, align 4"));
                    }
                    self.emit_term(format!("br label %bb_{}", resume_bb.0));

                    self.start_block(&park_lbl);
                    let dl_ssa = SsaVal::new("i32", dl);
                    self.park_task_prim(&sptr, Some(&dl_ssa));
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                } else {
                    self.park_task_prim(&sptr, None);
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                }
            }
            MirYieldKind::ByteRingWaitRead {
                target,
                cap,
                timeout_deadline_var,
                dst_val,
                dst_res,
                poll_bb,
                resume_bb,
            } => {
                let call_kw = self.call_kw();
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty();
                let slice_ty = format!("{{ ptr, {usize_ty} }}");
                let sl_val = self.next_tmp();
                self.emit_line(format!(
                    "{sl_val} = {call_kw} {slice_ty} @__syn_ring_poll_read_{id}(ptr {rptr}, ptr %__task)"
                ));
                let sl_len = self.next_tmp();
                self.emit_line(format!("{sl_len} = extractvalue {slice_ty} {sl_val}, 1"));
                let got = self.next_tmp();
                self.emit_line(format!("{got} = icmp ne {usize_ty} {sl_len}, 0"));
                let ok_lbl = self.next_label("ring_ok");
                let miss_lbl = self.next_label("ring_miss");
                self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{miss_lbl}"));

                self.start_block(&ok_lbl);
                if let Some(vid) = dst_val {
                    let dp = self.var_ptr(*vid);
                    self.emit_line(format!("store {slice_ty} {sl_val}, ptr {dp}, align 4"));
                }
                if let Some(rvid) = dst_res {
                    let rp = self.var_ptr(*rvid);
                    let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                    let res0 = self.next_tmp();
                    self.emit_line(format!(
                        "{res0} = insertvalue {rty} zeroinitializer, {slice_ty} {sl_val}, 0"
                    ));
                    self.emit_line(format!("store {rty} {res0}, ptr {rp}, align 4"));
                }
                self.emit_term(format!("br label %bb_{}", resume_bb.0));

                self.start_block(&miss_lbl);
                if let Some(dl_var) = timeout_deadline_var {
                    let dl_ptr = self.var_ptr(*dl_var);
                    let dl = self.next_tmp();
                    self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 4"));
                    let now = self.next_tmp();
                    self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                    let diff = self.next_tmp();
                    self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
                    let exp = self.next_tmp();
                    self.emit_line(format!("{exp} = icmp sge i32 {diff}, 0"));
                    let to_lbl = self.next_label("ring_to");
                    let park_lbl = self.next_label("ring_park");
                    self.emit_term(format!("br i1 {exp}, label %{to_lbl}, label %{park_lbl}"));

                    self.start_block(&to_lbl);
                    self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let e0 = self.next_tmp();
                        self.emit_line(format!("{e0} = insertvalue {rty} zeroinitializer, i32 1, 1"));
                        self.emit_line(format!("store {rty} {e0}, ptr {rp}, align 4"));
                    }
                    self.emit_term(format!("br label %bb_{}", resume_bb.0));

                    self.start_block(&park_lbl);
                    let dl_ssa = SsaVal::new("i32", dl);
                    self.park_task_prim(&rptr, Some(&dl_ssa));
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                } else {
                    self.park_task_prim(&rptr, None);
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                }
            }
            MirYieldKind::InboxViewRecv {
                view,
                elem_ty,
                timeout_deadline_var,
                dst_val,
                dst_res,
                poll_bb,
                resume_bb,
            } => {
                let call_kw = self.call_kw();
                let (hdr, items_base, cap32) = self.emit_inbox_view_parts(view)?;
                let key = sanitize_ident(&elem_ty.mangle_name());
                let ety = self.ctx.llvm_ty(elem_ty);
                let (tmp_elem, al) = self.emit_temp_alloca(&ety);
                self.emit_line(format!("store {ety} zeroinitializer, ptr {tmp_elem}, align {al}"));

                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_q_view_pop_{key}(ptr {hdr}, ptr {items_base}, i32 {cap32}, ptr {tmp_elem}, ptr %__task)"
                ));
                let got = self.next_tmp();
                self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
                let ok_lbl = self.next_label("qv_ok");
                let miss_lbl = self.next_label("qv_miss");
                self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{miss_lbl}"));

                self.start_block(&ok_lbl);
                let val = self.next_tmp();
                self.emit_line(format!("{val} = load {ety}, ptr {tmp_elem}, align {al}"));
                if let Some(vvid) = dst_val {
                    let vp = self.var_ptr(*vvid);
                    self.emit_line(format!("store {ety} {val}, ptr {vp}, align 4"));
                }
                if let Some(rvid) = dst_res {
                    let rp = self.var_ptr(*rvid);
                    let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                    let r0 = self.next_tmp();
                    let r1 = self.next_tmp();
                    self.emit_line(format!("{r0} = insertvalue {rty} poison, {ety} {val}, 0"));
                    self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 0, 1"));
                    self.emit_line(format!("store {rty} {r1}, ptr {rp}, align 4"));
                }
                self.emit_term(format!("br label %bb_{}", resume_bb.0));

                self.start_block(&miss_lbl);
                if let Some(dl_var) = timeout_deadline_var {
                    let dl_ptr = self.var_ptr(*dl_var);
                    let dl = self.next_tmp();
                    self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 4"));
                    let now = self.next_tmp();
                    self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                    let diff = self.next_tmp();
                    self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
                    let exp = self.next_tmp();
                    self.emit_line(format!("{exp} = icmp sge i32 {diff}, 0"));
                    let to_lbl = self.next_label("qv_to");
                    let park_lbl = self.next_label("qv_park");
                    self.emit_term(format!("br i1 {exp}, label %{to_lbl}, label %{park_lbl}"));

                    self.start_block(&to_lbl);
                    self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let e0 = self.next_tmp();
                        self.emit_line(format!("{e0} = insertvalue {rty} zeroinitializer, i32 1, 1"));
                        self.emit_line(format!("store {rty} {e0}, ptr {rp}, align 4"));
                    }
                    self.emit_term(format!("br label %bb_{}", resume_bb.0));

                    self.start_block(&park_lbl);
                    let dl_ssa = SsaVal::new("i32", dl);
                    self.park_task_prim(&hdr, Some(&dl_ssa));
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                } else {
                    self.park_task_prim(&hdr, None);
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                }
            }
            MirYieldKind::InboxViewSend {
                view,
                elem_ty,
                val_var,
                timeout_deadline_var,
                dst_res,
                poll_bb,
                resume_bb,
            } => {
                let (hdr, items_base, cap32) = self.emit_inbox_view_parts(view)?;
                let key = sanitize_ident(&elem_ty.mangle_name());
                let vp = self.var_ptr(*val_var);
                let ety = self.ctx.llvm_ty(elem_ty);
                let is_lease = matches!(elem_ty, MirType::Lease(_));
                let push_fn = if is_lease {
                    format!("@__syn_q_view_push_keep_{key}")
                } else {
                    format!("@__syn_q_view_push_{key}")
                };
                let call_kw = self.call_kw();
                let ptr_al = self.ctx.ptr_align();
                let v = self.next_tmp();
                self.emit_line(format!("{v} = load {ety}, ptr {vp}, align 1"));
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 {push_fn}(ptr {hdr}, ptr {items_base}, i32 {cap32}, {ety} {v}, i32 -1)"
                ));
                let ok_b = self.next_tmp();
                self.emit_line(format!("{ok_b} = icmp ne i8 {ok}, 0"));
                let ok_lbl = self.next_label("qv_tx_ok");
                let full_lbl = self.next_label("qv_tx_full");
                self.emit_term(format!("br i1 {ok_b}, label %{ok_lbl}, label %{full_lbl}"));

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
                    self.emit_line(format!("store {rty} zeroinitializer, ptr {rp}, align 4"));
                }
                self.emit_term(format!("br label %bb_{}", resume_bb.0));

                self.start_block(&full_lbl);
                if let Some(dl_var) = timeout_deadline_var {
                    let dl_ptr = self.var_ptr(*dl_var);
                    let dl = self.next_tmp();
                    self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 4"));
                    let now = self.next_tmp();
                    self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                    let diff = self.next_tmp();
                    self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
                    let exp = self.next_tmp();
                    self.emit_line(format!("{exp} = icmp sge i32 {diff}, 0"));
                    let to_lbl = self.next_label("qv_tx_to");
                    let park_lbl = self.next_label("qv_tx_park");
                    self.emit_term(format!("br i1 {exp}, label %{to_lbl}, label %{park_lbl}"));

                    self.start_block(&to_lbl);
                    if is_lease {
                        let ph = self.next_tmp();
                        let sidx = self.next_tmp();
                        let vp_ph = self.next_tmp();
                        self.emit_line(format!("{ph} = extractvalue {ety} {v}, 1"));
                        self.emit_line(format!("{sidx} = extractvalue {ety} {v}, 2"));
                        self.emit_line(format!(
                            "{call_kw} void @__syn_pool_release(ptr {ph}, i8 {sidx}, i8 -1)"
                        ));
                        self.emit_line(format!(
                            "{vp_ph} = getelementptr inbounds {ety}, ptr {vp}, i32 0, i32 1"
                        ));
                        self.emit_line(format!("store ptr null, ptr {vp_ph}, align {ptr_al}"));
                    }
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let e0 = self.next_tmp();
                        self.emit_line(format!("{e0} = insertvalue {rty} zeroinitializer, i32 1, 1"));
                        self.emit_line(format!("store {rty} {e0}, ptr {rp}, align 4"));
                    }
                    self.emit_term(format!("br label %bb_{}", resume_bb.0));

                    self.start_block(&park_lbl);
                    let q_tx = self.next_tmp();
                    self.emit_line(format!(
                        "{q_tx} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 1"
                    ));
                    let dl_ssa = SsaVal::new("i32", dl);
                    self.park_task_prim(&q_tx, Some(&dl_ssa));
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                } else {
                    let q_tx = self.next_tmp();
                    self.emit_line(format!(
                        "{q_tx} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 1"
                    ));
                    self.park_task_prim(&q_tx, None);
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                }
            }
            MirYieldKind::ByteRingViewWaitRead {
                view,
                timeout_deadline_var,
                dst_val,
                dst_res,
                poll_bb,
                resume_bb,
            } => {
                let call_kw = self.call_kw();
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let slice_ty = format!("{{ ptr, {usize_ty} }}");
                let sl_val = self.next_tmp();
                self.emit_line(format!(
                    "{sl_val} = {call_kw} {slice_ty} @__syn_ring_view_poll_read(ptr {hdr}, i32 {cap32}, ptr %__task)"
                ));
                let sl_len = self.next_tmp();
                self.emit_line(format!("{sl_len} = extractvalue {slice_ty} {sl_val}, 1"));
                let got = self.next_tmp();
                self.emit_line(format!("{got} = icmp ne {usize_ty} {sl_len}, 0"));
                let ok_lbl = self.next_label("rv_ok");
                let miss_lbl = self.next_label("rv_miss");
                self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{miss_lbl}"));

                self.start_block(&ok_lbl);
                if let Some(vid) = dst_val {
                    let dp = self.var_ptr(*vid);
                    self.emit_line(format!("store {slice_ty} {sl_val}, ptr {dp}, align 4"));
                }
                if let Some(rvid) = dst_res {
                    let rp = self.var_ptr(*rvid);
                    let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                    let res0 = self.next_tmp();
                    self.emit_line(format!(
                        "{res0} = insertvalue {rty} zeroinitializer, {slice_ty} {sl_val}, 0"
                    ));
                    self.emit_line(format!("store {rty} {res0}, ptr {rp}, align 4"));
                }
                self.emit_term(format!("br label %bb_{}", resume_bb.0));

                self.start_block(&miss_lbl);
                if let Some(dl_var) = timeout_deadline_var {
                    let dl_ptr = self.var_ptr(*dl_var);
                    let dl = self.next_tmp();
                    self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 4"));
                    let now = self.next_tmp();
                    self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                    let diff = self.next_tmp();
                    self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
                    let exp = self.next_tmp();
                    self.emit_line(format!("{exp} = icmp sge i32 {diff}, 0"));
                    let to_lbl = self.next_label("rv_to");
                    let park_lbl = self.next_label("rv_park");
                    self.emit_term(format!("br i1 {exp}, label %{to_lbl}, label %{park_lbl}"));

                    self.start_block(&to_lbl);
                    self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let e0 = self.next_tmp();
                        self.emit_line(format!("{e0} = insertvalue {rty} zeroinitializer, i32 1, 1"));
                        self.emit_line(format!("store {rty} {e0}, ptr {rp}, align 4"));
                    }
                    self.emit_term(format!("br label %bb_{}", resume_bb.0));

                    self.start_block(&park_lbl);
                    let dl_ssa = SsaVal::new("i32", dl);
                    self.park_task_prim(&hdr, Some(&dl_ssa));
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                } else {
                    self.park_task_prim(&hdr, None);
                    self.store_coro_pc(poll_bb.0 as i32);
                    self.emit_term("ret i32 1");
                }
            }
            MirYieldKind::PingPongWaitChunk {
                target,
                elem_ty,
                cap,
                dst_slice,
                poll_bb,
                resume_bb,
            } => {
                let call_kw = self.call_kw();
                let pk = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let chunk_ptr = self.next_tmp();
                self.emit_line(format!("{chunk_ptr} = {call_kw} ptr @__syn_pp_poll_{id}(ptr {pptr})"));
                let got = self.next_tmp();
                self.emit_line(format!("{got} = icmp ne ptr {chunk_ptr}, null"));
                let ok_lbl = self.next_label("pp_ok");
                let wait_lbl = self.next_label("pp_wait");
                self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{wait_lbl}"));

                self.start_block(&ok_lbl);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty();
                let slice_ty = format!("{{ ptr, {usize_ty} }}");
                let s0 = self.next_tmp();
                let s1 = self.next_tmp();
                self.emit_line(format!("{s0} = insertvalue {slice_ty} poison, ptr {chunk_ptr}, 0"));
                self.emit_line(format!("{s1} = insertvalue {slice_ty} {s0}, {usize_ty} {cap}, 1"));
                let dp = self.var_ptr(*dst_slice);
                self.emit_line(format!("store {slice_ty} {s1}, ptr {dp}, align 4"));
                self.emit_term(format!("br label %bb_{}", resume_bb.0));

                self.start_block(&wait_lbl);
                self.park_task_prim(&pptr, None);
                self.store_coro_pc(poll_bb.0 as i32);
                self.emit_term("ret i32 1");
            }
            MirYieldKind::PoolAlloc {
                target,
                elem_ty,
                cap,
                timeout_deadline_var,
                dst_val,
                dst_res,
                poll_bb,
                resume_bb,
            } => {
                let call_kw = self.call_kw();
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
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
                let ok_lbl = self.next_label("pool_ok");
                let empty_lbl = self.next_label("pool_empty");
                self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{empty_lbl}"));

                self.start_block(&ok_lbl);
                let ls = self.next_tmp();
                self.emit_line(format!("{ls} = load {{ ptr, ptr, i8 }}, ptr {tmp}, align {al}"));
                let ptr_al = self.ctx.ptr_align();
                if let Some(vid) = dst_val {
                    let vp = self.var_ptr(*vid);
                    self.emit_line(format!(
                        "store {{ ptr, ptr, i8 }} {ls}, ptr {vp}, align {ptr_al}"
                    ));
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
                    self.emit_line(format!("store {rty} {r1}, ptr {rp}, align {ptr_al}"));
                }
                self.emit_term(format!("br label %bb_{}", resume_bb.0));

                self.start_block(&empty_lbl);
                let dl_ssa = if let Some(dl_var) = timeout_deadline_var {
                    let dl_ptr = self.var_ptr(*dl_var);
                    let dl = self.next_tmp();
                    self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 4"));
                    let now = self.next_tmp();
                    self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
                    let diff = self.next_tmp();
                    self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
                    let exp = self.next_tmp();
                    self.emit_line(format!("{exp} = icmp sge i32 {diff}, 0"));
                    let to_lbl = self.next_label("pool_to");
                    let park_lbl = self.next_label("pool_park");
                    self.emit_term(format!("br i1 {exp}, label %{to_lbl}, label %{park_lbl}"));

                    self.start_block(&to_lbl);
                    self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
                    if let Some(rvid) = dst_res {
                        let rp = self.var_ptr(*rvid);
                        let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                        let e0 = self.next_tmp();
                        self.emit_line(format!(
                            "{e0} = insertvalue {rty} zeroinitializer, i32 1, 1"
                        ));
                        self.emit_line(format!("store {rty} {e0}, ptr {rp}, align {ptr_al}"));
                    }
                    self.emit_term(format!("br label %bb_{}", resume_bb.0));

                    self.start_block(&park_lbl);
                    Some(SsaVal::new("i32", dl))
                } else {
                    None
                };
                let pm = self.next_tmp();
                self.emit_line(format!("{pm} = {call_kw} i32 @__syn_crit_enter()"));
                let fm_p = self.next_tmp();
                let fm = self.next_tmp();
                let has_free = self.next_tmp();
                self.emit_line(format!(
                    "{fm_p} = getelementptr inbounds %prim.pool_hdr, ptr {pptr}, i32 0, i32 0"
                ));
                self.emit_line(format!("{fm} = load i32, ptr {fm_p}, align 4"));
                self.emit_line(format!("{has_free} = icmp ne i32 {fm}, 0"));
                let retry_lbl = self.next_label("pool_retry");
                let do_park_lbl = self.next_label("pool_do_park");
                self.emit_term(format!(
                    "br i1 {has_free}, label %{retry_lbl}, label %{do_park_lbl}"
                ));

                self.start_block(&retry_lbl);
                self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                self.emit_term(format!("br label %bb_{}", poll_bb.0));

                self.start_block(&do_park_lbl);
                self.park_task_prim(&pptr, dl_ssa.as_ref());
                self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                self.store_coro_pc(poll_bb.0 as i32);
                self.emit_term("ret i32 1");
            }
            MirYieldKind::CallYielding {
                callee,
                slot_name,
                args,
                poll_bb,
                ..
            } => {
                let coro = self.coroutine.clone().unwrap();
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
                    .unwrap()
                    .clone();

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
                let coro = self.coroutine.clone().unwrap();
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
                self.emit_line(format!("{share} = {call_kw} i1 @__syn_should_share(ptr %__task)"));
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
                        let coerced = self.coerce(
                            SsaVal::new(&child_layout.ret_llvm_ty, rv),
                            &dty,
                            signed,
                        );
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

    #[allow(clippy::too_many_arguments)]
    fn emit_queue_recv_yield(
        &mut self,
        target: &MirPrimTarget,
        pk: &MirPrimKind,
        elem_ty: &MirType,
        timeout_deadline_var: Option<VarId>,
        dst_val: Option<VarId>,
        dst_res: Option<VarId>,
        poll_bb: BlockId,
        resume_bb: BlockId,
    ) -> Result<(), CodegenError> {
        let id = sanitize_ident(&pk.mangle_name());
        let qptr = self.resolve_prim_ptr(target)?;
        let ety = self.ctx.llvm_ty(elem_ty);
        let (tmp_elem, al) = self.emit_temp_alloca(&ety);
        self.emit_line(format!("store {ety} zeroinitializer, ptr {tmp_elem}, align {al}"));

        let call_kw = self.call_kw();
        let ok = self.next_tmp();
        self.emit_line(format!(
            "{ok} = {call_kw} i8 @__syn_q_pop_{id}(ptr {qptr}, ptr {tmp_elem}, ptr %__task)"
        ));
        let got = self.next_tmp();
        self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
        let ok_lbl = self.next_label("q_ok");
        let miss_lbl = self.next_label("q_miss");
        self.emit_term(format!("br i1 {got}, label %{ok_lbl}, label %{miss_lbl}"));

        self.start_block(&ok_lbl);
        let val = self.next_tmp();
        self.emit_line(format!("{val} = load {ety}, ptr {tmp_elem}, align {al}"));
        if let Some(vvid) = dst_val {
            let vp = self.var_ptr(vvid);
            self.emit_line(format!("store {ety} {val}, ptr {vp}, align 4"));
        }
        if let Some(rvid) = dst_res {
            let rp = self.var_ptr(rvid);
            let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
            let r0 = self.next_tmp();
            let r1 = self.next_tmp();
            self.emit_line(format!("{r0} = insertvalue {rty} poison, {ety} {val}, 0"));
            self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 0, 1"));
            self.emit_line(format!("store {rty} {r1}, ptr {rp}, align 4"));
        }
        self.emit_term(format!("br label %bb_{}", resume_bb.0));

        self.start_block(&miss_lbl);
        if let Some(dl_var) = timeout_deadline_var {
            let dl_ptr = self.var_ptr(dl_var);
            let dl = self.next_tmp();
            self.emit_line(format!("{dl} = load i32, ptr {dl_ptr}, align 4"));
            let now = self.next_tmp();
            self.emit_line(format!("{now} = {call_kw} i32 @__syn_now_ms()"));
            let diff = self.next_tmp();
            self.emit_line(format!("{diff} = sub i32 {now}, {dl}"));
            let exp = self.next_tmp();
            self.emit_line(format!("{exp} = icmp sge i32 {diff}, 0"));
            let to_lbl = self.next_label("q_to");
            let park_lbl = self.next_label("q_park");
            self.emit_term(format!("br i1 {exp}, label %{to_lbl}, label %{park_lbl}"));

            self.start_block(&to_lbl);
            self.emit_line(format!("{call_kw} void @__syn_end_inbox_turn(ptr %__task)"));
            if let Some(rvid) = dst_res {
                let rp = self.var_ptr(rvid);
                let rty = self.ctx.llvm_ty(&self.vars[rvid.0].ty);
                let e0 = self.next_tmp();
                self.emit_line(format!("{e0} = insertvalue {rty} zeroinitializer, i32 1, 1"));
                self.emit_line(format!("store {rty} {e0}, ptr {rp}, align 4"));
            }
            self.emit_term(format!("br label %bb_{}", resume_bb.0));

            self.start_block(&park_lbl);
            let dl_ssa = SsaVal::new("i32", dl);
            self.park_task_prim(&qptr, Some(&dl_ssa));
            self.store_coro_pc(poll_bb.0 as i32);
            self.emit_term("ret i32 1");
        } else {
            self.park_task_prim(&qptr, None);
            self.store_coro_pc(poll_bb.0 as i32);
            self.emit_term("ret i32 1");
        }
        Ok(())
    }

    // ── Boot Helpers ─────────────────────────────────────────────────────────

    fn emit_boot_run_yielding(
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
            .unwrap()
            .clone();
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
        self.emit_term(format!("br i1 {done}, label %{done_lbl}, label %{wait_lbl}"));

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
                let coerced =
                    self.coerce(SsaVal::new(&child_layout.ret_llvm_ty, rv), &dty, signed);
                self.emit_line(format!(
                    "store {} {}, ptr {dp}, align 4",
                    coerced.ty, coerced.repr
                ));
            }
        }
        Ok(())
    }

    fn emit_store_pool_hdr(&mut self, hdr_ptr: &str, cap: usize, dma_flag: u8) {
        let c = cap.max(1);
        let mask: u32 = if c >= 32 {
            u32::MAX
        } else {
            (1u32 << c) - 1
        };
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
        self.emit_line(format!("store i32 {mask_i32}, ptr {f1}, align 4"));
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

    fn emit_subsys_pool_inits(&mut self, struct_name: &str, base_ptr: &str, parent_dma_safe: bool) {
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

    fn emit_pool_boot_inits(&mut self) {
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

    fn emit_boot_spawn_activate(
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
            .unwrap()
            .clone();
        let bp = self
            .ctx
            .module
            .task_blueprints
            .iter()
            .find(|b| b.name == sp.blueprint_name)
            .unwrap()
            .clone();
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

    fn ensure_terminated_void(&mut self) {
        if !self.terminated {
            self.emit_term("ret void");
        }
    }

    fn ensure_terminated_default(&mut self) {
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

    fn build_return_envelope(&mut self, val_opt: Option<&MirExpr>) -> Result<SsaVal, CodegenError> {
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

    fn build_error_envelope(
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

    fn emit_stmt(&mut self, stmt: &MirStmt) -> Result<(), CodegenError> {
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
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let cv = self.emit_expr(cond)?;
                let c1 = self.to_i1(cv);
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
                let c1 = self.to_i1(cv);
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
                let sw_ty = if llvm_int_bits(&dv.ty) > 32 { "i64" } else { "i32" };
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
                self.emit_line(format!("{call_kw} void @__syn_fmt_init_uart(ptr %__fmt_sink)"));
                for ch in chunks {
                    self.emit_fmt_chunk("%__fmt_sink", ch)?;
                }
                if *newline {
                    self.emit_line(format!("{call_kw} void @__syn_fmt_putc(ptr %__fmt_sink, i8 10)"));
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
                self.emit_term(format!("br i1 {has_w}, label %{wake_lbl}, label %{done_lbl}"));
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

    fn emit_fmt_chunk(&mut self, sink: &str, ch: &MirFmtChunk) -> Result<(), CodegenError> {
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
                    self.emit_line(format!("{call_kw} void @__syn_fmt_putc(ptr {sink}, i8 {b})"));
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
                        let is64 = *bits == 64
                            || (*bits == 0 && self.ctx.spec.isa.ptr_bits() == 64);
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
                let b1 = self.to_i1(vv);
                let b8 = self.next_tmp();
                self.emit_line(format!("{b8} = zext i1 {b1} to i8"));
                self.emit_line(format!("{call_kw} void @__syn_fmt_bool(ptr {sink}, i8 {b8})"));
            }
            MirFmtChunk::Str(value) => {
                let vv = self.emit_expr(value)?;
                let p = self.coerce_str_ptr(vv);
                self.emit_line(format!(
                    "{call_kw} void @__syn_fmt_cstr(ptr {sink}, ptr {})",
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
            MirFmtChunk::EnumName { enum_name, value } => {
                let vv = self.emit_expr(value)?;
                let v32 = self.coerce(vv, "i32", true);
                let eid = sanitize_ident(enum_name);
                let np = self.next_tmp();
                self.emit_line(format!(
                    "{np} = {call_kw} ptr @__syn_enum_name_{eid}(i32 {})",
                    v32.repr
                ));
                self.emit_line(format!("{call_kw} void @__syn_fmt_cstr(ptr {sink}, ptr {np})"));
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
                self.emit_line(format!("store {usize_ty} 0, ptr {idx_slot}, align {idx_al}"));

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
                self.emit_line(format!("{idx_v} = load {usize_ty}, ptr {idx_slot}, align {idx_al}"));
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
                        let b1 = self.to_i1(ev);
                        let b8 = self.next_tmp();
                        self.emit_line(format!("{b8} = zext i1 {b1} to i8"));
                        self.emit_line(format!("{call_kw} void @__syn_fmt_bool(ptr {sink}, i8 {b8})"));
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
                        self.emit_line(format!("{call_kw} void @__syn_fmt_cstr(ptr {sink}, ptr {np})"));
                    }
                    MirType::Str => {
                        let p = self.coerce_str_ptr(ev);
                        self.emit_line(format!(
                            "{call_kw} void @__syn_fmt_cstr(ptr {sink}, ptr {})",
                            p.repr
                        ));
                    }
                    MirType::Ptr(_) => {
                        self.emit_line(format!("{call_kw} void @__syn_fmt_putc(ptr {sink}, i8 48)"));
                        self.emit_line(format!("{call_kw} void @__syn_fmt_putc(ptr {sink}, i8 120)"));
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
                        if matches!(elem_ty, MirType::Isize)
                            && self.ctx.spec.isa.ptr_bits() == 64
                        {
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
                        if matches!(elem_ty, MirType::Usize)
                            && self.ctx.spec.isa.ptr_bits() == 64
                        {
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
                self.emit_line(format!("store {usize_ty} {idx_next}, ptr {idx_slot}, align {idx_al}"));
                self.emit_term(format!("br label %{cond_lbl}"));

                self.start_block(&end_lbl);
                self.emit_line(format!("{call_kw} void @__syn_fmt_putc(ptr {sink}, i8 93)"));
            }
        }
        Ok(())
    }

    fn emit_compound_op(
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
            self.coerce(rhs.clone(), ty, false)
        } else {
            rhs.clone()
        };
        let inst = match op {
            MirAssignOp::Assign => unreachable!(),
            MirAssignOp::AddAssign => "add",
            MirAssignOp::SubAssign => "sub",
            MirAssignOp::MulAssign => "mul",
            MirAssignOp::DivAssign => {
                if signed {
                    "sdiv"
                } else {
                    "udiv"
                }
            }
            MirAssignOp::RemAssign => {
                if signed {
                    "srem"
                } else {
                    "urem"
                }
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

    fn emit_byte_ring_view_parts(&mut self, view: &MirExpr) -> Result<(String, String), CodegenError> {
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

    fn emit_inbox_view_parts(
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

    fn resolve_prim_ptr(&mut self, target: &MirPrimTarget) -> Result<String, CodegenError> {
        match target {
            MirPrimTarget::Global(name) => Ok(format!("@__syn_prim_{}", sanitize_ident(name))),
            MirPrimTarget::PtrExpr(e) => {
                let v = self.emit_expr(e)?;
                let p = self.coerce(v, "ptr", false);
                Ok(p.repr)
            }
        }
    }

    fn emit_prim_nonblocking(&mut self, pop: &MirPrimNonBlockingOp) -> Result<(), CodegenError> {
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
                self.emit_line(format!("{call_kw} void @__syn_sig_notify_{id}(ptr {sptr}, i32 {b})"));
            }
            MirPrimNonBlockingOp::SignalClear { target } => {
                let sptr = self.resolve_prim_ptr(target)?;
                let fp = self.next_tmp();
                self.emit_line(format!(
                    "{fp} = getelementptr inbounds %prim.signal, ptr {sptr}, i32 0, i32 0"
                ));
                self.emit_line(format!("store volatile i8 0, ptr {fp}, align 1"));
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
                self.emit_line(format!("{ph} = extractvalue {{ ptr, ptr, i8 }} {}, 1", lv.repr));
                self.emit_line(format!("{idx} = extractvalue {{ ptr, ptr, i8 }} {}, 2", lv.repr));
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

    fn emit_intrinsic(&mut self, iop: &MirIntrinsicOp) -> Result<(), CodegenError> {
        let call_kw = self.call_kw();
        match iop {
            MirIntrinsicOp::UseSystick { hz } => {
                let hv = self.emit_expr(hz)?;
                let h32 = self.coerce(hv, "i32", false);
                self.emit_line(format!("{call_kw} void @__syn_use_systick(i32 {})", h32.repr));
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
                    "use_systick" => {
                        if let Some((hz, _)) = ev.first() {
                            let h32 = self.coerce(hz.clone(), "i32", false);
                            self.emit_line(format!("{call_kw} void @__syn_use_systick(i32 {})", h32.repr));
                        }
                    }
                    "now_ms" | "now_ticks" => {
                        let ign = self.next_tmp();
                        self.emit_line(format!("{ign} = {call_kw} i32 @__syn_now_ms()"));
                    }
                    "puts" | "print" => {
                        if let Some((s, _)) = ev.first() {
                            let p = self.coerce_str_ptr(s.clone());
                            self.emit_line(format!("{call_kw} void @__syn_uart_puts(ptr {})", p.repr));
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

    fn emit_hook_registration(
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

    fn emit_rom_call_invocation(
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

    // ── Places, Typed MMIO Addressing & Wire-Endian Load/Store ───────────────

    fn emit_mmio_elem_ptr(
        &mut self,
        base: &MirExpr,
        offset: u32,
        elem_bits: u8,
        index: Option<&MirExpr>,
        array_len: Option<usize>,
    ) -> Result<(String, &'static str, usize), CodegenError> {
        let bv = self.emit_expr(base)?;
        let base_ptr = self.coerce(bv, "ptr", false).repr;
        let (lty, stride, align) = match elem_bits {
            8 => ("i8", 1u32, 1usize),
            16 => ("i16", 2u32, 2usize),
            _ => ("i32", 4u32, 4usize),
        };

        let is_host64 = self.ctx.spec.isa == CpuIsa::HostX86_64Linux;
        let idx_ty = if is_host64 { "i64" } else { "i32" };
        let ptr_tmp = if let Some(idx_expr) = index {
            let iv = self.emit_expr(idx_expr)?;
            let idx_val = self.coerce(iv, idx_ty, !idx_expr.ty.is_unsigned()).repr;
            let eff_idx = if let Some(alen) = array_len {
                if alen == 0 {
                    "0".to_string()
                } else {
                    let inb = self.next_tmp();
                    let clamped = self.next_tmp();
                    self.emit_line(format!("{inb} = icmp ult {idx_ty} {idx_val}, {alen}"));
                    self.emit_line(format!(
                        "{clamped} = select i1 {inb}, {idx_ty} {idx_val}, {idx_ty} 0"
                    ));
                    clamped
                }
            } else {
                idx_val
            };

            let scaled = if stride == 1 {
                eff_idx
            } else {
                let s = self.next_tmp();
                self.emit_line(format!("{s} = mul {idx_ty} {eff_idx}, {stride}"));
                s
            };
            let total_off = if offset == 0 {
                scaled
            } else {
                let sum = self.next_tmp();
                let off_lit = if is_host64 {
                    i64::from(offset)
                } else {
                    i64::from(offset as i32)
                };
                self.emit_line(format!("{sum} = add {idx_ty} {off_lit}, {scaled}"));
                sum
            };
            let p = self.next_tmp();
            self.emit_line(format!(
                "{p} = getelementptr i8, ptr {base_ptr}, {idx_ty} {total_off}"
            ));
            p
        } else {
            let p = self.next_tmp();
            let off_lit = if is_host64 {
                i64::from(offset)
            } else {
                i64::from(offset as i32)
            };
            self.emit_line(format!(
                "{p} = getelementptr i8, ptr {base_ptr}, {idx_ty} {off_lit}"
            ));
            p
        };

        Ok((ptr_tmp, lty, align))
    }

    fn emit_place_ptr(&mut self, place: &MirPlace) -> Result<(String, MirType, bool), CodegenError> {
        match place {
            MirPlace::Var(vid) => {
                let p = self.var_ptr(*vid);
                let ty = self.vars[vid.0].ty.clone();
                Ok((p, ty, false))
            }
            MirPlace::Deref(ptr_expr, pointee_ty) => {
                let pv = self.emit_expr(ptr_expr)?;
                let pc = self.coerce(pv, "ptr", false);
                Ok((pc.repr, pointee_ty.clone(), false))
            }
            MirPlace::Field {
                base,
                struct_name,
                packed,
                field,
                field_ty,
            } => {
                let (bptr, _, bpacked) = self.emit_place_ptr(base)?;
                let si = &self.ctx.structs[struct_name];
                let (fidx, _) = si.field_indices[field];
                let sty = si.llvm_name.clone();
                let gp = self.next_tmp();
                self.emit_line(format!(
                    "{gp} = getelementptr inbounds {sty}, ptr {bptr}, i32 0, i32 {fidx}"
                ));
                Ok((gp, field_ty.clone(), *packed || bpacked))
            }
            MirPlace::Index {
                base,
                index,
                elem_ty,
                is_slice,
            } => {
                let (bptr, bty, bpacked) = self.emit_place_ptr(base)?;
                let iv = self.emit_expr(index)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let ic = self.coerce(iv, &usize_ty, false);
                let ety = self.ctx.llvm_ty(elem_ty);
                if *is_slice {
                    let sl_ty = format!("{{ ptr, {usize_ty} }}");
                    let sl = self.next_tmp();
                    let data_p = self.next_tmp();
                    let ep = self.next_tmp();
                    self.emit_line(format!("{sl} = load {sl_ty}, ptr {bptr}, align 4"));
                    self.emit_line(format!("{data_p} = extractvalue {sl_ty} {sl}, 0"));
                    self.emit_line(format!(
                        "{ep} = getelementptr {ety}, ptr {data_p}, {usize_ty} {}",
                        ic.repr
                    ));
                    Ok((ep, elem_ty.clone(), bpacked))
                } else {
                    let aty = self.ctx.llvm_ty(&bty);
                    let ep = self.next_tmp();
                    self.emit_line(format!(
                        "{ep} = getelementptr inbounds {aty}, ptr {bptr}, {usize_ty} 0, {usize_ty} {}",
                        ic.repr
                    ));
                    Ok((ep, elem_ty.clone(), bpacked))
                }
            }
        }
    }

    fn load_from_place_ptr(&mut self, ptr: &str, raw_ty: &MirType, _packed: bool) -> SsaVal {
        let lty = self.ctx.llvm_ty(raw_ty);
        // `align 1` keeps unaligned `packed struct` / wire-cast pointer loads fault-safe while
        // `llc` folds known-aligned stack/frame accesses to natural load instructions.
        let raw = self.next_tmp();
        self.emit_line(format!("{raw} = load {lty}, ptr {ptr}, align 1"));
        if let MirType::EndianInt {
            bits,
            big_endian: true,
            ..
        } = raw_ty
        {
            let swapped = self.next_tmp();
            self.emit_line(format!("{swapped} = call i{bits} @llvm.bswap.i{bits}(i{bits} {raw})"));
            SsaVal::new(lty, swapped)
        } else {
            SsaVal::new(lty, raw)
        }
    }

    fn store_to_place_ptr(
        &mut self,
        ptr: &str,
        raw_ty: &MirType,
        _packed: bool,
        host_val: SsaVal,
    ) {
        let lty = self.ctx.llvm_ty(raw_ty);
        let coerced = self.coerce(host_val, &lty, !raw_ty.is_unsigned());
        let wire_repr = if let MirType::EndianInt {
            bits,
            big_endian: true,
            ..
        } = raw_ty
        {
            let sw = self.next_tmp();
            self.emit_line(format!(
                "{sw} = call i{bits} @llvm.bswap.i{bits}(i{bits} {})",
                coerced.repr
            ));
            sw
        } else {
            coerced.repr
        };
        self.emit_line(format!("store {lty} {wire_repr}, ptr {ptr}, align 1"));
    }

    fn pattern_int(&self, pat: &MirSwitchPattern) -> i64 {
        match pat {
            MirSwitchPattern::EnumVariant {
                enum_name,
                variant,
            } => self
                .ctx
                .enum_variants
                .get(&(enum_name.clone(), variant.clone()))
                .or_else(|| self.ctx.bare_variants.get(variant))
                .copied()
                .unwrap_or(0),
            MirSwitchPattern::Int(v) => *v as i64,
            MirSwitchPattern::Bool(b) => i64::from(*b),
        }
    }

    // ── Pure / Direct Expressions ────────────────────────────────────────────

    fn emit_expr(&mut self, expr: &MirExpr) -> Result<SsaVal, CodegenError> {
        let dst_lty = self.ctx.llvm_ty(&expr.ty.host_value_type());
        match &expr.kind {
            MirExprKind::IntLit(v) => {
                if dst_lty == "ptr" || dst_lty == "ptr addrspace(1)" {
                    if *v == 0 {
                        Ok(SsaVal::new(dst_lty, "null"))
                    } else if self.ctx.spec.isa.ptr_bits() == 16 {
                        let t = self.next_tmp();
                        let v16 = *v as i16;
                        self.emit_line(format!("{t} = inttoptr i16 {v16} to {dst_lty}"));
                        Ok(SsaVal::new(dst_lty, t))
                    } else {
                        let t = self.next_tmp();
                        let v32 = *v as i32;
                        self.emit_line(format!("{t} = inttoptr i32 {v32} to {dst_lty}"));
                        Ok(SsaVal::new(dst_lty, t))
                    }
                } else if dst_lty == "void" {
                    Ok(SsaVal::new("void", "0"))
                } else if dst_lty == "float" {
                    let fb = ((*v as f32) as f64).to_bits();
                    Ok(SsaVal::new("float", format!("0x{fb:016X}")))
                } else if dst_lty == "double" {
                    let db = (*v as f64).to_bits();
                    Ok(SsaVal::new("double", format!("0x{db:016X}")))
                } else if !dst_lty.starts_with('i') {
                    Ok(SsaVal::new(dst_lty, "zeroinitializer"))
                } else {
                    Ok(SsaVal::new(dst_lty, format!("{v}")))
                }
            }
            MirExprKind::FloatLit(bits) => {
                let target_fty = if dst_lty == "float" || expr.ty == MirType::F32 {
                    "float"
                } else {
                    "double"
                };
                let norm_bits = if target_fty == "float" {
                    let f = f64::from_bits(*bits) as f32;
                    if f.is_nan() {
                        0x7FF8_0000_0000_0000u64
                    } else {
                        (f as f64).to_bits()
                    }
                } else {
                    *bits
                };
                Ok(SsaVal::new(target_fty, format!("0x{norm_bits:016X}")))
            }
            MirExprKind::BoolLit(b) => Ok(SsaVal::new("i8", if *b { "1" } else { "0" })),
            MirExprKind::TicksLit(t) => Ok(SsaVal::new(dst_lty, format!("{t}"))),
            MirExprKind::StringLit(s) => {
                let idx = self.ctx.intern_string(s);
                let slen = zev_ast::decoded_string_byte_len(s);
                if matches!(expr.ty, MirType::Slice(_)) {
                    let usize_ty = self.ctx.spec.isa.usize_llvm_ty();
                    let sl_ty = format!("{{ ptr, {usize_ty} }}");
                    let r0 = self.next_tmp();
                    let r1 = self.next_tmp();
                    self.emit_line(format!(
                        "{r0} = insertvalue {sl_ty} poison, ptr @.str.{idx}, 0"
                    ));
                    self.emit_line(format!(
                        "{r1} = insertvalue {sl_ty} {r0}, {usize_ty} {slen}, 1"
                    ));
                    Ok(SsaVal::new(sl_ty, r1))
                } else {
                    Ok(SsaVal::new("ptr", format!("@.str.{idx}")))
                }
            }
            MirExprKind::EnumVariant {
                enum_name,
                variant,
            } => {
                let disc = self
                    .ctx
                    .enum_variants
                    .get(&(enum_name.clone(), variant.clone()))
                    .or_else(|| self.ctx.bare_variants.get(variant))
                    .copied()
                    .unwrap_or(0);
                Ok(SsaVal::new("i32", format!("{disc}")))
            }
            MirExprKind::ConstRef(name) => {
                if expr.ty.is_float() || dst_lty == "float" || dst_lty == "double" {
                    let raw = self
                        .ctx
                        .float_consts
                        .get(name)
                        .copied()
                        .or_else(|| self.ctx.consts.get(name).copied().map(|v| (v as f64).to_bits()))
                        .unwrap_or(0);
                    let target_fty = if dst_lty == "float" || expr.ty == MirType::F32 {
                        "float"
                    } else {
                        "double"
                    };
                    let norm = if target_fty == "float" {
                        let f = f64::from_bits(raw) as f32;
                        if f.is_nan() {
                            0x7FF8_0000_0000_0000u64
                        } else {
                            (f as f64).to_bits()
                        }
                    } else {
                        raw
                    };
                    Ok(SsaVal::new(target_fty, format!("0x{norm:016X}")))
                } else {
                    let v = self.ctx.consts.get(name).copied().unwrap_or(1);
                    Ok(SsaVal::new(dst_lty, format!("{v}")))
                }
            }
            MirExprKind::GlobalPrimAddr(name) => {
                Ok(SsaVal::new("ptr", format!("@__syn_prim_{}", sanitize_ident(name))))
            }
            MirExprKind::SubsystemAddr(name) => {
                Ok(SsaVal::new("ptr", format!("@__zev_subsys_{}", sanitize_ident(name))))
            }
            MirExprKind::MakePrimView { ptr, cap, kind } => {
                let pv = self.emit_expr(ptr)?;
                let pc = self.coerce(pv, "ptr", false);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let n = (*cap).max(1);
                match kind {
                    MirPrimViewKind::ByteRing => {
                        let vty = format!("{{ ptr, {usize_ty} }}");
                        let v0 = self.next_tmp();
                        let v1 = self.next_tmp();
                        self.emit_line(format!(
                            "{v0} = insertvalue {vty} poison, ptr {}, 0",
                            pc.repr
                        ));
                        self.emit_line(format!(
                            "{v1} = insertvalue {vty} {v0}, {usize_ty} {n}, 1"
                        ));
                        Ok(SsaVal::new(vty, v1))
                    }
                    MirPrimViewKind::Inbox { elem } => {
                        let vty = format!("{{ ptr, ptr, {usize_ty} }}");
                        let pty = format!(
                            "%prim.{}",
                            sanitize_ident(
                                &MirPrimKind::Inbox {
                                    elem: elem.clone(),
                                    cap: n,
                                }
                                .mangle_name()
                            )
                        );
                        let items_base = self.next_tmp();
                        self.emit_line(format!(
                            "{items_base} = getelementptr inbounds {pty}, ptr {}, i32 0, i32 4, i32 0",
                            pc.repr
                        ));
                        let v0 = self.next_tmp();
                        let v1 = self.next_tmp();
                        let v2 = self.next_tmp();
                        self.emit_line(format!(
                            "{v0} = insertvalue {vty} poison, ptr {}, 0",
                            pc.repr
                        ));
                        self.emit_line(format!(
                            "{v1} = insertvalue {vty} {v0}, ptr {items_base}, 1"
                        ));
                        self.emit_line(format!(
                            "{v2} = insertvalue {vty} {v1}, {usize_ty} {n}, 2"
                        ));
                        Ok(SsaVal::new(vty, v2))
                    }
                }
            }
            MirExprKind::MmioAddr(inst) => {
                if !self.ctx.mmio_instances.contains_key(inst) {
                    return Err(CodegenError::new(
                        "mir_codegen",
                        format!("unknown MMIO peripheral instance `{inst}`"),
                    ));
                }
                if self.ctx.spec.isa == CpuIsa::HostX86_64Linux {
                    Ok(SsaVal::new(
                        "ptr",
                        format!("@__zev_mmio_buf_{}", sanitize_ident(inst)),
                    ))
                } else {
                    let addr = self.ctx.mmio_instances[inst].base_addr as i32;
                    Ok(SsaVal::new("ptr", format!("inttoptr (i32 {addr} to ptr)")))
                }
            }
            MirExprKind::MmioRead {
                base,
                offset,
                elem_bits,
                index,
                array_len,
            } => {
                let (p, lty, align) = self.emit_mmio_elem_ptr(
                    base,
                    *offset,
                    *elem_bits,
                    index.as_deref(),
                    *array_len,
                )?;
                let val = self.next_tmp();
                self.emit_line(format!("{val} = load volatile {lty}, ptr {p}, align {align}"));
                Ok(self.coerce(SsaVal::new(lty, val), &dst_lty, !expr.ty.is_unsigned()))
            }
            MirExprKind::FnAddr(fname) => {
                let fn_ptr_ty = self.ctx.spec.isa.fn_ptr_llvm_ty();
                if let Some(sig) = self.ctx.fn_sigs.get(fname) {
                    Ok(SsaVal::new(fn_ptr_ty, sig.symbol.clone()))
                } else {
                    Ok(SsaVal::new(fn_ptr_ty, format!("@{}", sanitize_ident(fname))))
                }
            }
            MirExprKind::Var(vid) => {
                let p = self.var_ptr(*vid);
                let vty = self.vars[vid.0].ty.clone();
                Ok(self.load_from_place_ptr(&p, &vty, false))
            }
            MirExprKind::AddrOfPlace(place) => {
                let (p, _, _) = self.emit_place_ptr(place)?;
                Ok(SsaVal::new("ptr", p))
            }
            MirExprKind::Unary { op, expr: sub } => {
                let sv = self.emit_expr(sub)?;
                match op {
                    MirUnaryOp::Neg => {
                        let sc = self.coerce(sv, &dst_lty, true);
                        let t = self.next_tmp();
                        if dst_lty == "float" || dst_lty == "double" {
                            self.emit_line(format!("{t} = fneg {dst_lty} {}", sc.repr));
                        } else {
                            self.emit_line(format!("{t} = sub {dst_lty} 0, {}", sc.repr));
                        }
                        Ok(SsaVal::new(dst_lty, t))
                    }
                    MirUnaryOp::Not => {
                        let i1v = self.to_i1(sv);
                        let inv = self.next_tmp();
                        let b8 = self.next_tmp();
                        self.emit_line(format!("{inv} = xor i1 {i1v}, true"));
                        self.emit_line(format!("{b8} = zext i1 {inv} to i8"));
                        Ok(SsaVal::new("i8", b8))
                    }
                    MirUnaryOp::BitNot => {
                        let sc = self.coerce(sv, &dst_lty, false);
                        let t = self.next_tmp();
                        self.emit_line(format!("{t} = xor {dst_lty} {}, -1", sc.repr));
                        Ok(SsaVal::new(dst_lty, t))
                    }
                    MirUnaryOp::Deref => {
                        let pc = self.coerce(sv, "ptr", false);
                        Ok(self.load_from_place_ptr(&pc.repr, &expr.ty, false))
                    }
                }
            }
            MirExprKind::Binary { op, lhs, rhs } => self.emit_binary(*op, lhs, rhs, &expr.ty),
            MirExprKind::Cast {
                expr: sub,
                target_ty,
            } => {
                let sv = self.emit_expr(sub)?;
                if *target_ty == MirType::Bool {
                    let i1v = self.to_i1(sv);
                    let b8 = self.next_tmp();
                    self.emit_line(format!("{b8} = zext i1 {i1v} to i8"));
                    return Ok(SsaVal::new("i8", b8));
                }
                let t_lty = self.ctx.llvm_ty(&target_ty.host_value_type());
                let signed = if sub.ty.is_float() {
                    !target_ty.is_unsigned()
                } else {
                    !sub.ty.is_unsigned()
                };
                Ok(self.coerce(sv, &t_lty, signed))
            }
            MirExprKind::StructLit {
                struct_name,
                fields,
                ..
            } => {
                let si = self.ctx.structs[struct_name].clone();
                let sty = si.llvm_name.clone();
                let mut cur = SsaVal::new(&sty, "zeroinitializer");
                for (fname, fexpr, fty) in fields {
                    let (fidx, _) = si.field_indices[fname];
                    let raw_v = self.emit_expr(fexpr)?;
                    let flty = self.ctx.llvm_ty(fty);
                    let mut coerced = self.coerce(raw_v, &flty, !fexpr.ty.is_unsigned());
                    if let MirType::EndianInt {
                        bits,
                        big_endian: true,
                        ..
                    } = fty
                    {
                        let sw = self.next_tmp();
                        self.emit_line(format!(
                            "{sw} = call i{bits} @llvm.bswap.i{bits}(i{bits} {})",
                            coerced.repr
                        ));
                        coerced = SsaVal::new(flty, sw);
                    }
                    let nxt = self.next_tmp();
                    self.emit_line(format!(
                        "{nxt} = insertvalue {sty} {}, {} {}, {fidx}",
                        cur.repr, coerced.ty, coerced.repr
                    ));
                    cur = SsaVal::new(&sty, nxt);
                }
                Ok(cur)
            }
            MirExprKind::ArrayLit { elems, elem_ty } => {
                let n = elems.len().max(1);
                let ety = self.ctx.llvm_ty(elem_ty);
                let aty = format!("[{n} x {ety}]");
                let mut cur = SsaVal::new(&aty, "zeroinitializer");
                for (i, e) in elems.iter().enumerate() {
                    let ev = self.emit_expr(e)?;
                    let ec = self.coerce(ev, &ety, !e.ty.is_unsigned());
                    let nxt = self.next_tmp();
                    self.emit_line(format!(
                        "{nxt} = insertvalue {aty} {}, {ety} {}, {i}",
                        cur.repr, ec.repr
                    ));
                    cur = SsaVal::new(&aty, nxt);
                }
                Ok(cur)
            }
            MirExprKind::FieldAccess {
                base,
                struct_name,
                packed,
                is_ptr,
                field,
                raw_field_ty,
            } => {
                let si = self.ctx.structs[struct_name].clone();
                let (fidx, _) = si.field_indices[field];
                let sty = si.llvm_name.clone();
                if *is_ptr {
                    let bv = self.emit_expr(base)?;
                    let bp = self.coerce(bv, "ptr", false);
                    let gp = self.next_tmp();
                    self.emit_line(format!(
                        "{gp} = getelementptr inbounds {sty}, ptr {}, i32 0, i32 {fidx}",
                        bp.repr
                    ));
                    Ok(self.load_from_place_ptr(&gp, raw_field_ty, *packed))
                } else {
                    let bv = self.emit_expr(base)?;
                    let ext = self.next_tmp();
                    let flty = self.ctx.llvm_ty(raw_field_ty);
                    self.emit_line(format!("{ext} = extractvalue {sty} {}, {fidx}", bv.repr));
                    if let MirType::EndianInt {
                        bits,
                        big_endian: true,
                        ..
                    } = raw_field_ty
                    {
                        let sw = self.next_tmp();
                        self.emit_line(format!(
                            "{sw} = call i{bits} @llvm.bswap.i{bits}(i{bits} {ext})"
                        ));
                        Ok(SsaVal::new(flty, sw))
                    } else {
                        Ok(SsaVal::new(flty, ext))
                    }
                }
            }
            MirExprKind::SliceLen(sl_expr) => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(sl_expr)?;
                let len = self.next_tmp();
                self.emit_line(format!("{len} = extractvalue {sl_ty} {}, 1", sv.repr));
                Ok(SsaVal::new(usize_ty, len))
            }
            MirExprKind::ArrayLen(n) => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(SsaVal::new(usize_ty, format!("{n}")))
            }
            MirExprKind::Index {
                base,
                index,
                is_slice,
            } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let iv = self.emit_expr(index)?;
                let ic = self.coerce(iv, &usize_ty, false);
                let ety = self.ctx.llvm_ty(&expr.ty);
                if *is_slice {
                    let sl_ty = format!("{{ ptr, {usize_ty} }}");
                    let bv = self.emit_expr(base)?;
                    let dp = self.next_tmp();
                    let ep = self.next_tmp();
                    self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", bv.repr));
                    self.emit_line(format!(
                        "{ep} = getelementptr {ety}, ptr {dp}, {usize_ty} {}",
                        ic.repr
                    ));
                    Ok(self.load_from_place_ptr(&ep, &expr.ty, false))
                } else {
                    let bv = self.emit_expr(base)?;
                    let (tmp, al) = self.emit_temp_alloca(&bv.ty);
                    let ep = self.next_tmp();
                    self.emit_line(format!("store {} {}, ptr {tmp}, align {al}", bv.ty, bv.repr));
                    self.emit_line(format!(
                        "{ep} = getelementptr inbounds {}, ptr {tmp}, {usize_ty} 0, {usize_ty} {}",
                        bv.ty, ic.repr
                    ));
                    Ok(self.load_from_place_ptr(&ep, &expr.ty, false))
                }
            }
            MirExprKind::SubSlice {
                base,
                start,
                end,
                elem_ty,
                base_is_array,
            } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let ety = self.ctx.llvm_ty(elem_ty);
                let st_v = self.emit_expr(start)?;
                let en_v = self.emit_expr(end)?;
                let st_c = self.coerce(st_v, &usize_ty, false);
                let en_c = self.coerce(en_v, &usize_ty, false);

                let (base_ptr, base_len) =
                    if *base_is_array || matches!(base.ty, MirType::Array { .. }) {
                        let arr_len = match &base.ty {
                            MirType::Array { len, .. } => *len,
                            _ => 0,
                        };
                        let p = if let MirExprKind::Var(vid) = &base.kind {
                            let ap = self.var_ptr(*vid);
                            let aty = self.ctx.llvm_ty(&base.ty);
                            let gp = self.next_tmp();
                            self.emit_line(format!(
                                "{gp} = getelementptr inbounds {aty}, ptr {ap}, {usize_ty} 0, {usize_ty} 0"
                            ));
                            gp
                        } else {
                            let bv = self.emit_expr(base)?;
                            let (tmp, al) = self.emit_temp_alloca(&bv.ty);
                            self.emit_line(format!(
                                "store {} {}, ptr {tmp}, align {al}",
                                bv.ty, bv.repr
                            ));
                            tmp
                        };
                        (p, format!("{arr_len}"))
                    } else {
                        let bv = self.emit_expr(base)?;
                        let p = self.next_tmp();
                        let bl = self.next_tmp();
                        self.emit_line(format!("{p} = extractvalue {sl_ty} {}, 0", bv.repr));
                        self.emit_line(format!("{bl} = extractvalue {sl_ty} {}, 1", bv.repr));
                        (p, bl)
                    };

                // Clamp `start` and `end` to `[0, base_len]` with `end >= start` (matching `syn_subslice_u8`)
                let st_gt = self.next_tmp();
                let st_cl = self.next_tmp();
                let en_gt = self.next_tmp();
                let en_cl0 = self.next_tmp();
                let en_lt_st = self.next_tmp();
                let en_cl = self.next_tmp();
                let is_null = self.next_tmp();
                let gep_p = self.next_tmp();
                let sub_p = self.next_tmp();
                let cl_len = self.next_tmp();
                let r0 = self.next_tmp();
                let r1 = self.next_tmp();

                self.emit_line(format!(
                    "{st_gt} = icmp ugt {usize_ty} {}, {base_len}",
                    st_c.repr
                ));
                self.emit_line(format!(
                    "{st_cl} = select i1 {st_gt}, {usize_ty} {base_len}, {usize_ty} {}",
                    st_c.repr
                ));
                self.emit_line(format!(
                    "{en_gt} = icmp ugt {usize_ty} {}, {base_len}",
                    en_c.repr
                ));
                self.emit_line(format!(
                    "{en_cl0} = select i1 {en_gt}, {usize_ty} {base_len}, {usize_ty} {}",
                    en_c.repr
                ));
                self.emit_line(format!("{en_lt_st} = icmp ult {usize_ty} {en_cl0}, {st_cl}"));
                self.emit_line(format!(
                    "{en_cl} = select i1 {en_lt_st}, {usize_ty} {st_cl}, {usize_ty} {en_cl0}"
                ));
                self.emit_line(format!("{is_null} = icmp eq ptr {base_ptr}, null"));
                self.emit_line(format!(
                    "{gep_p} = getelementptr {ety}, ptr {base_ptr}, {usize_ty} {st_cl}"
                ));
                self.emit_line(format!("{sub_p} = select i1 {is_null}, ptr null, ptr {gep_p}"));
                self.emit_line(format!("{cl_len} = sub {usize_ty} {en_cl}, {st_cl}"));
                self.emit_line(format!("{r0} = insertvalue {sl_ty} poison, ptr {sub_p}, 0"));
                self.emit_line(format!("{r1} = insertvalue {sl_ty} {r0}, {usize_ty} {cl_len}, 1"));
                Ok(SsaVal::new(sl_ty, r1))
            }
            MirExprKind::ArrayToSlice {
                array_place,
                len,
                ..
            } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let (ap, aty_mir, _) = self.emit_place_ptr(array_place)?;
                let aty = self.ctx.llvm_ty(&aty_mir);
                let ep = self.next_tmp();
                let r0 = self.next_tmp();
                let r1 = self.next_tmp();
                self.emit_line(format!(
                    "{ep} = getelementptr inbounds {aty}, ptr {ap}, {usize_ty} 0, {usize_ty} 0"
                ));
                self.emit_line(format!("{r0} = insertvalue {sl_ty} poison, ptr {ep}, 0"));
                self.emit_line(format!("{r1} = insertvalue {sl_ty} {r0}, {usize_ty} {len}, 1"));
                Ok(SsaVal::new(sl_ty, r1))
            }
            MirExprKind::FromBytes { struct_name, slice } => {
                let call_kw = self.call_kw();
                let sv = self.emit_expr(slice)?;
                let sname = sanitize_ident(struct_name);
                let sty = format!("%struct.{sname}");
                let rty = format!("{{ {sty}, i32 }}");
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {rty} @__syn_from_bytes_{sname}({} {})",
                    sv.ty, sv.repr
                ));
                Ok(SsaVal::new(rty, res))
            }
            MirExprKind::PrimRead(prop) => self.emit_prim_read(prop),
            MirExprKind::CallDirect {
                callee,
                is_extern_c,
                args,
            } => self.emit_direct_call(callee, *is_extern_c, args, &expr.ty),
            MirExprKind::ResultIsErr(vid) => {
                let p = self.var_ptr(*vid);
                let rty = self.ctx.llvm_ty(&self.vars[vid.0].ty);
                let rv = self.next_tmp();
                let tag = self.next_tmp();
                let ne = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{rv} = load {rty}, ptr {p}, align 4"));
                self.emit_line(format!("{tag} = extractvalue {rty} {rv}, 1"));
                self.emit_line(format!("{ne} = icmp ne i32 {tag}, 0"));
                self.emit_line(format!("{b8} = zext i1 {ne} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirExprKind::ResultErr(vid) => {
                let p = self.var_ptr(*vid);
                let rty = self.ctx.llvm_ty(&self.vars[vid.0].ty);
                let rv = self.next_tmp();
                let tag = self.next_tmp();
                let raw_e = self.next_tmp();
                self.emit_line(format!("{rv} = load {rty}, ptr {p}, align 4"));
                self.emit_line(format!("{tag} = extractvalue {rty} {rv}, 1"));
                self.emit_line(format!("{raw_e} = sub i32 {tag}, 1"));
                Ok(self.coerce(SsaVal::new("i32", raw_e), &dst_lty, true))
            }
            MirExprKind::ResultVal(vid) => {
                let p = self.var_ptr(*vid);
                let vmir = self.vars[vid.0].ty.clone();
                let rty = self.ctx.llvm_ty(&vmir);
                let (ok_lty, ok_signed) = match &vmir {
                    MirType::Result { ok, .. } => {
                        if ok.is_void() {
                            ("i32".to_string(), false)
                        } else {
                            (self.ctx.llvm_ty(ok), !ok.is_unsigned())
                        }
                    }
                    _ => (dst_lty.clone(), !expr.ty.is_unsigned()),
                };
                let rv = self.next_tmp();
                let val = self.next_tmp();
                self.emit_line(format!("{rv} = load {rty}, ptr {p}, align 4"));
                self.emit_line(format!("{val} = extractvalue {rty} {rv}, 0"));
                Ok(self.coerce(SsaVal::new(ok_lty, val), &dst_lty, ok_signed))
            }
            MirExprKind::NowTicks => {
                let call_kw = self.call_kw();
                let t = self.next_tmp();
                self.emit_line(format!("{t} = {call_kw} i32 @__syn_now_ms()"));
                Ok(self.coerce(SsaVal::new("i32", t), &dst_lty, false))
            }
            MirExprKind::SysQuery(q) => {
                let call_kw = self.call_kw();
                let v = match q {
                    MirSysQuery::CurrentPrio => {
                        let tmp = self.next_tmp();
                        self.emit_line(format!("{tmp} = {call_kw} i32 @__syn_current_prio()"));
                        tmp
                    }
                    MirSysQuery::SleepDepth => {
                        let m = self.next_tmp();
                        let dl = self.next_tmp();
                        let any = self.next_tmp();
                        let locked = self.next_tmp();
                        let out_v = self.next_tmp();
                        let pm = if self.ctx.spec.isa.is_avr() {
                            let pm = self.next_tmp();
                            self.emit_line(format!("{pm} = {call_kw} i32 @__syn_crit_enter()"));
                            Some(pm)
                        } else {
                            None
                        };
                        self.emit_line(format!(
                            "{m} = load volatile i32, ptr @__syn_sleep_lock_mask, align 4"
                        ));
                        self.emit_line(format!(
                            "{dl} = load volatile i32, ptr @__syn_dma_locks, align 4"
                        ));
                        if let Some(pm) = pm {
                            self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                        }
                        self.emit_line(format!("{any} = or i32 {m}, {dl}"));
                        self.emit_line(format!("{locked} = icmp ne i32 {any}, 0"));
                        self.emit_line(format!("{out_v} = select i1 {locked}, i32 1, i32 2"));
                        out_v
                    }
                    MirSysQuery::WfiCount => self.emit_avr_safe_i32_load("@__syn_wfi_count", true),
                };
                Ok(self.coerce(SsaVal::new("i32", v), &dst_lty, false))
            }
            MirExprKind::LeaseGet { lease, .. } => {
                let lv = self.emit_expr(lease)?;
                let p = self.next_tmp();
                self.emit_line(format!("{p} = extractvalue {{ ptr, ptr, i8 }} {}, 0", lv.repr));
                Ok(SsaVal::new("ptr", p))
            }
            MirExprKind::LeaseAddr { lease, .. } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let lv = self.emit_expr(lease)?;
                let p = self.next_tmp();
                let a = self.next_tmp();
                self.emit_line(format!("{p} = extractvalue {{ ptr, ptr, i8 }} {}, 0", lv.repr));
                self.emit_line(format!("{a} = ptrtoint ptr {p} to {usize_ty}"));
                Ok(SsaVal::new(usize_ty, a))
            }
            MirExprKind::RomCallU32 {
                addr,
                stack_bound,
                args,
            } => {
                let rv = self
                    .emit_rom_call_invocation(RomWrapperKind::U32, addr, *stack_bound, args)?
                    .expect("u32 rom_call returns i32");
                Ok(self.coerce(rv, &dst_lty, false))
            }
        }
    }

    fn emit_prim_read(&mut self, prop: &MirPrimReadOp) -> Result<SsaVal, CodegenError> {
        let call_kw = self.call_kw();
        match prop {
            MirPrimReadOp::WatchRead { target, elem_ty } => {
                let pk = MirPrimKind::Watch {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let wptr = self.resolve_prim_ptr(target)?;
                let ety = self.ctx.llvm_ty(elem_ty);
                let v = self.next_tmp();
                self.emit_line(format!("{v} = {call_kw} {ety} @__syn_watch_read_{id}(ptr {wptr})"));
                Ok(SsaVal::new(ety, v))
            }
            MirPrimReadOp::InboxTryRecv {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                self.emit_try_recv_expr(target, &pk, elem_ty)
            }
            MirPrimReadOp::InboxTrySend {
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
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_q_push_{id}(ptr {qptr}, {ety} {}, i32 {b})",
                    vc.repr
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::InboxLen {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let qptr = self.resolve_prim_ptr(target)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", cnt), &usize_ty, false))
            }
            MirPrimReadOp::InboxIsEmpty {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let qptr = self.resolve_prim_ptr(target)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let emp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{emp} = icmp eq i32 {cnt}, 0"));
                self.emit_line(format!("{b8} = zext i1 {emp} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::InboxIsFull {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let qptr = self.resolve_prim_ptr(target)?;
                let c = (*cap).max(1) as i32;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let fl = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{fl} = icmp sge i32 {cnt}, {c}"));
                self.emit_line(format!("{b8} = zext i1 {fl} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::SignalIsSet { target } => {
                let sptr = self.resolve_prim_ptr(target)?;
                let fp = self.next_tmp();
                let f = self.next_tmp();
                self.emit_line(format!(
                    "{fp} = getelementptr inbounds %prim.signal, ptr {sptr}, i32 0, i32 0"
                ));
                self.emit_line(format!("{f} = load volatile i8, ptr {fp}, align 1"));
                Ok(SsaVal::new("i8", f))
            }
            MirPrimReadOp::SignalTryWait { target } => {
                let id = sanitize_ident(&MirPrimKind::Signal.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_sig_poll_{id}(ptr {sptr}, ptr {task_arg})"
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::PingPongDmaBuf {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let pp_ty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let pptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let n = (*cap).max(1);
                let wh_p = self.next_tmp();
                let wh = self.next_tmp();
                let is_b0 = self.next_tmp();
                let b0 = self.next_tmp();
                let b1 = self.next_tmp();
                let buf_p = self.next_tmp();
                let s0 = self.next_tmp();
                let s1 = self.next_tmp();
                self.emit_line(format!(
                    "{wh_p} = getelementptr inbounds {pp_ty}, ptr {pptr}, i32 0, i32 0"
                ));
                self.emit_line(format!("{wh} = load i8, ptr {wh_p}, align 1"));
                self.emit_line(format!("{is_b0} = icmp eq i8 {wh}, 0"));
                self.emit_line(format!(
                    "{b0} = getelementptr inbounds {pp_ty}, ptr {pptr}, i32 0, i32 4, i32 0"
                ));
                self.emit_line(format!(
                    "{b1} = getelementptr inbounds {pp_ty}, ptr {pptr}, i32 0, i32 5, i32 0"
                ));
                self.emit_line(format!("{buf_p} = select i1 {is_b0}, ptr {b0}, ptr {b1}"));
                self.emit_line(format!("{s0} = insertvalue {sl_ty} poison, ptr {buf_p}, 0"));
                self.emit_line(format!("{s1} = insertvalue {sl_ty} {s0}, {usize_ty} {n}, 1"));
                Ok(SsaVal::new(sl_ty, s1))
            }
            MirPrimReadOp::PoolTryAlloc {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let rty = "{ { ptr, ptr, i8 }, i32 }".to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {rty} @__syn_pool_try_alloc_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new(rty, res))
            }
            MirPrimReadOp::PoolAvail {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {usize_ty} @__syn_pool_avail_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new(usize_ty, res))
            }
            MirPrimReadOp::PoolInUse {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {usize_ty} @__syn_pool_in_use_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new(usize_ty, res))
            }
            MirPrimReadOp::PoolIsExhausted {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} i8 @__syn_pool_is_exhausted_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new("i8", res))
            }
            MirPrimReadOp::PoolCap {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {usize_ty} @__syn_pool_cap_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new(usize_ty, res))
            }
            MirPrimReadOp::DmaSlotPark {
                target,
                elem_ty,
                lease,
            } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let lv = self.emit_expr(lease)?;
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {{ i32, i32 }} @__syn_slot_park_{id}(ptr {sptr}, {{ ptr, ptr, i8 }} {})",
                    lv.repr
                ));
                Ok(SsaVal::new("{ i32, i32 }", res))
            }
            MirPrimReadOp::DmaSlotTake { target, elem_ty } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let rty = "{ { ptr, ptr, i8 }, i32 }".to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {rty} @__syn_slot_take_{id}(ptr {sptr})"
                ));
                Ok(SsaVal::new(rty, res))
            }
            MirPrimReadOp::DmaSlotRelease {
                target,
                elem_ty,
                boost,
            } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let b = boost.map_or(-1i8, |p| p.as_u8() as i8);
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} i8 @__syn_slot_release_{id}(ptr {sptr}, i8 {b})"
                ));
                Ok(SsaVal::new("i8", res))
            }
            MirPrimReadOp::DmaSlotIsBusy { target, elem_ty } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} i8 @__syn_slot_is_busy_{id}(ptr {sptr})"
                ));
                Ok(SsaVal::new("i8", res))
            }
            MirPrimReadOp::DmaSlotIsEmpty { target, elem_ty } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} i8 @__syn_slot_is_empty_{id}(ptr {sptr})"
                ));
                Ok(SsaVal::new("i8", res))
            }
            MirPrimReadOp::DmaSlotAddr { target, elem_ty } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {usize_ty} @__syn_slot_addr_{id}(ptr {sptr})"
                ));
                Ok(SsaVal::new(usize_ty, res))
            }
            MirPrimReadOp::ByteRingPush {
                target,
                cap,
                src,
                boost,
            } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(src)?;
                let sc = self.coerce(sv, &sl_ty, false);
                let sp = self.next_tmp();
                let sl = self.next_tmp();
                self.emit_line(format!("{sp} = extractvalue {sl_ty} {}, 0", sc.repr));
                self.emit_line(format!("{sl} = extractvalue {sl_ty} {}, 1", sc.repr));
                let sl32 = self.coerce(SsaVal::new(usize_ty.clone(), sl), "i32", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let w32 = self.next_tmp();
                self.emit_line(format!(
                    "{w32} = {call_kw} i32 @__syn_ring_push_{id}(ptr {rptr}, ptr {sp}, i32 {}, i32 {b})",
                    sl32.repr
                ));
                Ok(self.coerce(SsaVal::new("i32", w32), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingPushByte {
                target,
                cap,
                byte,
                boost,
            } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let bv = self.emit_expr(byte)?;
                let b8 = self.coerce(bv, "i8", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_ring_push_byte_{id}(ptr {rptr}, i8 {}, i32 {b})",
                    b8.repr
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::ByteRingGrantWrite { target, cap, len } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let lv = self.emit_expr(len)?;
                let l32 = self.coerce(lv, "i32", false);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty();
                let res_ty = format!("{{ {{ ptr, {usize_ty} }}, i32 }}");
                let rv = self.next_tmp();
                self.emit_line(format!(
                    "{rv} = {call_kw} {res_ty} @__syn_ring_grant_write_{id}(ptr {rptr}, i32 {})",
                    l32.repr
                ));
                Ok(SsaVal::new(res_ty, rv))
            }
            MirPrimReadOp::ByteRingReadSlice { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sl = self.next_tmp();
                self.emit_line(format!(
                    "{sl} = {call_kw} {sl_ty} @__syn_ring_read_slice_{id}(ptr {rptr})"
                ));
                Ok(SsaVal::new(sl_ty, sl))
            }
            MirPrimReadOp::ByteRingPop { target, cap, dst } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let dv = self.emit_expr(dst)?;
                let dc = self.coerce(dv, &sl_ty, false);
                let dp = self.next_tmp();
                let dl = self.next_tmp();
                self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", dc.repr));
                self.emit_line(format!("{dl} = extractvalue {sl_ty} {}, 1", dc.repr));
                let dl32 = self.coerce(SsaVal::new(usize_ty.clone(), dl), "i32", false);
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let n32 = self.next_tmp();
                self.emit_line(format!(
                    "{n32} = {call_kw} i32 @__syn_ring_pop_{id}(ptr {rptr}, ptr {dp}, i32 {}, ptr {task_arg})",
                    dl32.repr
                ));
                Ok(self.coerce(SsaVal::new("i32", n32), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingTryPopByte { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let rv = self.next_tmp();
                self.emit_line(format!(
                    "{rv} = {call_kw} {{ i8, i32 }} @__syn_ring_try_pop_byte_{id}(ptr {rptr}, ptr {task_arg})"
                ));
                Ok(SsaVal::new("{ i8, i32 }", rv))
            }
            MirPrimReadOp::ByteRingLen { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let rptr = self.resolve_prim_ptr(target)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {rptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", cnt), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingFreeSpace { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let fs = self.next_tmp();
                self.emit_line(format!(
                    "{fs} = {call_kw} i32 @__syn_ring_free_space_{id}(ptr {rptr})"
                ));
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", fs), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingIsEmpty { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let rptr = self.resolve_prim_ptr(target)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {rptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let emp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{emp} = icmp eq i32 {cnt}, 0"));
                self.emit_line(format!("{b8} = zext i1 {emp} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::ByteRingIsFull { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let fs = self.next_tmp();
                let full = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!(
                    "{fs} = {call_kw} i32 @__syn_ring_free_space_{id}(ptr {rptr})"
                ));
                self.emit_line(format!("{full} = icmp eq i32 {fs}, 0"));
                self.emit_line(format!("{b8} = zext i1 {full} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::InboxViewTryRecv { view, elem_ty } => {
                let (hdr, items_base, cap32) = self.emit_inbox_view_parts(view)?;
                let key = sanitize_ident(&elem_ty.mangle_name());
                let ety = self.ctx.llvm_ty(elem_ty);
                let rty = format!("{{ {ety}, i32 }}");
                let (tmp, al) = self.emit_temp_alloca(&ety);
                self.emit_line(format!("store {ety} zeroinitializer, ptr {tmp}, align {al}"));
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_q_view_pop_{key}(ptr {hdr}, ptr {items_base}, i32 {cap32}, ptr {tmp}, ptr {task_arg})"
                ));
                let val = self.next_tmp();
                let got = self.next_tmp();
                let tag = self.next_tmp();
                let r0 = self.next_tmp();
                let r1 = self.next_tmp();
                self.emit_line(format!("{val} = load {ety}, ptr {tmp}, align {al}"));
                self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
                self.emit_line(format!("{tag} = select i1 {got}, i32 0, i32 1"));
                self.emit_line(format!("{r0} = insertvalue {rty} poison, {ety} {val}, 0"));
                self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 {tag}, 1"));
                Ok(SsaVal::new(rty, r1))
            }
            MirPrimReadOp::InboxViewTrySend {
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
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_q_view_push_{key}(ptr {hdr}, ptr {items_base}, i32 {cap32}, {ety} {}, i32 {b})",
                    vc.repr
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::InboxViewLen { view, elem_ty } => {
                let (hdr, _, _) = self.emit_inbox_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_inbox_view_contended(elem_ty);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", cnt), &usize_ty, false))
            }
            MirPrimReadOp::InboxViewCap { view, .. } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let vty = format!("{{ ptr, ptr, {usize_ty} }}");
                let vv = self.emit_expr(view)?;
                let cap_u = self.next_tmp();
                self.emit_line(format!("{cap_u} = extractvalue {vty} {}, 2", vv.repr));
                Ok(SsaVal::new(usize_ty, cap_u))
            }
            MirPrimReadOp::InboxViewIsEmpty { view, elem_ty } => {
                let (hdr, _, _) = self.emit_inbox_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_inbox_view_contended(elem_ty);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let emp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{emp} = icmp eq i32 {cnt}, 0"));
                self.emit_line(format!("{b8} = zext i1 {emp} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::InboxViewIsFull { view, elem_ty } => {
                let (hdr, _, cap32) = self.emit_inbox_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_inbox_view_contended(elem_ty);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let fl = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{fl} = icmp sge i32 {cnt}, {cap32}"));
                self.emit_line(format!("{b8} = zext i1 {fl} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::ByteRingViewPush { view, src, boost } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(src)?;
                let sc = self.coerce(sv, &sl_ty, false);
                let sp = self.next_tmp();
                let sl = self.next_tmp();
                self.emit_line(format!("{sp} = extractvalue {sl_ty} {}, 0", sc.repr));
                self.emit_line(format!("{sl} = extractvalue {sl_ty} {}, 1", sc.repr));
                let sl32 = self.coerce(SsaVal::new(usize_ty.clone(), sl), "i32", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let w32 = self.next_tmp();
                self.emit_line(format!(
                    "{w32} = {call_kw} i32 @__syn_ring_view_push(ptr {hdr}, i32 {cap32}, ptr {sp}, i32 {}, i32 {b})",
                    sl32.repr
                ));
                Ok(self.coerce(SsaVal::new("i32", w32), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingViewPushByte { view, byte, boost } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let bv = self.emit_expr(byte)?;
                let b8 = self.coerce(bv, "i8", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_ring_view_push_byte(ptr {hdr}, i32 {cap32}, i8 {}, i32 {b})",
                    b8.repr
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::ByteRingViewGrantWrite { view, len } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res_slice_ty = format!("{{ {{ ptr, {usize_ty} }}, i32 }}");
                let lv = self.emit_expr(len)?;
                let l32 = self.coerce(lv, "i32", false);
                let rv = self.next_tmp();
                self.emit_line(format!(
                    "{rv} = {call_kw} {res_slice_ty} @__syn_ring_view_grant_write(ptr {hdr}, i32 {cap32}, i32 {})",
                    l32.repr
                ));
                Ok(SsaVal::new(res_slice_ty, rv))
            }
            MirPrimReadOp::ByteRingViewReadSlice { view } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.next_tmp();
                self.emit_line(format!(
                    "{sv} = {call_kw} {sl_ty} @__syn_ring_view_read_slice(ptr {hdr}, i32 {cap32})"
                ));
                Ok(SsaVal::new(sl_ty, sv))
            }
            MirPrimReadOp::ByteRingViewPop { view, dst } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let dv = self.emit_expr(dst)?;
                let dc = self.coerce(dv, &sl_ty, false);
                let dp = self.next_tmp();
                let dl = self.next_tmp();
                self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", dc.repr));
                self.emit_line(format!("{dl} = extractvalue {sl_ty} {}, 1", dc.repr));
                let dl32 = self.coerce(SsaVal::new(usize_ty.clone(), dl), "i32", false);
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let n32 = self.next_tmp();
                self.emit_line(format!(
                    "{n32} = {call_kw} i32 @__syn_ring_view_pop(ptr {hdr}, i32 {cap32}, ptr {dp}, i32 {}, ptr {task_arg})",
                    dl32.repr
                ));
                Ok(self.coerce(SsaVal::new("i32", n32), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingViewTryPopByte { view } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let rv = self.next_tmp();
                self.emit_line(format!(
                    "{rv} = {call_kw} {{ i8, i32 }} @__syn_ring_view_try_pop_byte(ptr {hdr}, i32 {cap32}, ptr {task_arg})"
                ));
                Ok(SsaVal::new("{ i8, i32 }", rv))
            }
            MirPrimReadOp::ByteRingViewLen { view } => {
                let (hdr, _) = self.emit_byte_ring_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.byte_ring_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_byte_ring_view_contended();
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", cnt), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingViewCap { view } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let vty = format!("{{ ptr, {usize_ty} }}");
                let vv = self.emit_expr(view)?;
                let cap_u = self.next_tmp();
                self.emit_line(format!("{cap_u} = extractvalue {vty} {}, 1", vv.repr));
                Ok(SsaVal::new(usize_ty, cap_u))
            }
            MirPrimReadOp::ByteRingViewFreeSpace { view } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let fs = self.next_tmp();
                self.emit_line(format!(
                    "{fs} = {call_kw} i32 @__syn_ring_view_free_space(ptr {hdr}, i32 {cap32})"
                ));
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", fs), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingViewIsEmpty { view } => {
                let (hdr, _) = self.emit_byte_ring_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.byte_ring_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_byte_ring_view_contended();
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let emp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{emp} = icmp eq i32 {cnt}, 0"));
                self.emit_line(format!("{b8} = zext i1 {emp} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::ByteRingViewIsFull { view } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let fs = self.next_tmp();
                let full = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!(
                    "{fs} = {call_kw} i32 @__syn_ring_view_free_space(ptr {hdr}, i32 {cap32})"
                ));
                self.emit_line(format!("{full} = icmp eq i32 {fs}, 0"));
                self.emit_line(format!("{b8} = zext i1 {full} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
        }
    }

    fn emit_try_recv_expr(
        &mut self,
        target: &MirPrimTarget,
        pk: &MirPrimKind,
        elem_ty: &MirType,
    ) -> Result<SsaVal, CodegenError> {
        let id = sanitize_ident(&pk.mangle_name());
        let qptr = self.resolve_prim_ptr(target)?;
        let ety = self.ctx.llvm_ty(elem_ty);
        let rty = format!("{{ {ety}, i32 }}");
        let (tmp, al) = self.emit_temp_alloca(&ety);
        self.emit_line(format!("store {ety} zeroinitializer, ptr {tmp}, align {al}"));
        let task_arg = if self.coroutine.is_some() {
            "%__task"
        } else {
            "null"
        };
        let call_kw = self.call_kw();
        let ok = self.next_tmp();
        self.emit_line(format!(
            "{ok} = {call_kw} i8 @__syn_q_pop_{id}(ptr {qptr}, ptr {tmp}, ptr {task_arg})"
        ));
        let val = self.next_tmp();
        let got = self.next_tmp();
        let tag = self.next_tmp();
        let r0 = self.next_tmp();
        let r1 = self.next_tmp();
        self.emit_line(format!("{val} = load {ety}, ptr {tmp}, align {al}"));
        self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
        self.emit_line(format!("{tag} = select i1 {got}, i32 0, i32 1"));
        self.emit_line(format!("{r0} = insertvalue {rty} poison, {ety} {val}, 0"));
        self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 {tag}, 1"));
        Ok(SsaVal::new(rty, r1))
    }

    fn coerce_str_ptr(&mut self, val: SsaVal) -> SsaVal {
        if val.ty.starts_with("{ ptr,") {
            let p = self.next_tmp();
            self.emit_line(format!("{p} = extractvalue {} {}, 0", val.ty, val.repr));
            SsaVal::new("ptr", p)
        } else {
            self.coerce(val, "ptr", false)
        }
    }

    fn emit_runtime_assert(&mut self, cond_val: SsaVal) {
        let call_kw = self.call_kw();
        let c1 = self.to_i1(cond_val);
        let ok_lbl = self.next_label("assert_ok");
        let fail_lbl = self.next_label("assert_fail");
        self.emit_term(format!("br i1 {c1}, label %{ok_lbl}, label %{fail_lbl}"));
        self.start_block(&fail_lbl);
        let idx = self.ctx.intern_string("ASSERTION FAILED\n");
        self.emit_line(format!("{call_kw} void @__syn_uart_puts(ptr @.str.{idx})"));
        self.emit_line(format!("{call_kw} void @__syn_platform_exit(i32 1)"));
        self.emit_term("unreachable");
        self.start_block(&ok_lbl);
    }

    fn emit_direct_call(
        &mut self,
        callee: &str,
        is_extern_c: bool,
        args: &[MirExpr],
        ret_mir_ty: &MirType,
    ) -> Result<SsaVal, CodegenError> {
        let call_kw = self.call_kw();
        // Indirect call through a local/parameter/temp `MirVar` of `MirType::FnPtr`
        if callee != "assert" {
            if let Some((vid, params, ret)) =
                self.vars.iter().rev().find_map(|v| match &v.ty {
                    MirType::FnPtr { params, ret } if v.name == callee => {
                        Some((v.id, params.clone(), (**ret).clone()))
                    }
                    _ => None,
                })
            {
                let fp_ty = MirType::FnPtr {
                    params: params.clone(),
                    ret: Box::new(ret.clone()),
                };
                let fp_addr = self.var_ptr(vid);
                let fp_val = self.load_from_place_ptr(&fp_addr, &fp_ty, false);
                let fn_ptr_ty = self.ctx.spec.isa.fn_ptr_llvm_ty();
                let fp_ptr = self.coerce(fp_val, fn_ptr_ty, false);
                let ret_lty = self.ctx.llvm_ty(&ret.host_value_type());
                let mut call_args = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    let av = self.emit_expr(a)?;
                    if let Some(pty_mir) = params.get(i) {
                        let pty = self.ctx.llvm_ty(&pty_mir.host_value_type());
                        let c = self.coerce(av, &pty, !a.ty.is_unsigned());
                        call_args.push(format!("{} {}", c.ty, c.repr));
                    } else {
                        call_args.push(format!("{} {}", av.ty, av.repr));
                    }
                }
                let joined = call_args.join(", ");
                if ret.is_void() || ret_lty == "void" {
                    self.emit_line(format!("{call_kw} void {}({joined})", fp_ptr.repr));
                    return Ok(SsaVal::new("void", "0"));
                } else {
                    let r = self.next_tmp();
                    self.emit_line(format!("{r} = {call_kw} {ret_lty} {}({joined})", fp_ptr.repr));
                    return Ok(SsaVal::new(ret_lty, r));
                }
            }
        }

        if !is_extern_c {
            if let Some(sig) = self.ctx.fn_sigs.get(callee).cloned() {
                let mut call_args = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    let av = self.emit_expr(a)?;
                    if let Some(pty_mir) = sig.params.get(i) {
                        let pty = self.ctx.llvm_ty(pty_mir);
                        let c = self.coerce(av, &pty, !a.ty.is_unsigned());
                        call_args.push(format!("{} {}", c.ty, c.repr));
                    } else {
                        call_args.push(format!("{} {}", av.ty, av.repr));
                    }
                }
                let joined = call_args.join(", ");
                if sig.ret_llvm_ty == "void" {
                    self.emit_line(format!("{call_kw} void {}({joined})", sig.symbol));
                    return Ok(SsaVal::new("void", "0"));
                } else {
                    let r = self.next_tmp();
                    self.emit_line(format!(
                        "{r} = {call_kw} {} {}({joined})",
                        sig.ret_llvm_ty, sig.symbol
                    ));
                    return Ok(SsaVal::new(sig.ret_llvm_ty, r));
                }
            }
        }

        // Extern "C" call (if explicitly declared)
        if let Some(esig) = self.ctx.extern_sigs.get(callee).cloned() {
            let sym = format!("@{}", sanitize_ident(&esig.name));
            let ret_lty = self.ctx.llvm_ty(&esig.ret_ty);
            let mut call_args = Vec::new();
            for (i, a) in args.iter().enumerate() {
                let av = self.emit_expr(a)?;
                if let Some(pty_mir) = esig.params.get(i) {
                    let pty = self.ctx.llvm_ty(pty_mir);
                    let c = self.coerce(av, &pty, !a.ty.is_unsigned());
                    call_args.push(format!("{} {}", c.ty, c.repr));
                } else {
                    call_args.push(format!("{} {}", av.ty, av.repr));
                }
            }
            let sig_prefix = if esig.variadic {
                let fixed: Vec<String> =
                    esig.params.iter().map(|t| self.ctx.llvm_ty(t)).collect();
                let mut fp = fixed;
                fp.push("...".to_string());
                format!("{ret_lty} ({})", fp.join(", "))
            } else {
                ret_lty.clone()
            };
            let joined = call_args.join(", ");
            if ret_lty == "void" {
                self.emit_line(format!("{call_kw} {sig_prefix} {sym}({joined})"));
                return Ok(SsaVal::new("void", "0"));
            } else {
                let r = self.next_tmp();
                self.emit_line(format!("{r} = {call_kw} {sig_prefix} {sym}({joined})"));
                return Ok(SsaVal::new(ret_lty, r));
            }
        }

        // Portable built-in functions (`print`, `println`, `assert`, `now_ms`, `now_ticks`)
        match callee {
            "print" | "puts" => {
                if let Some(a0) = args.first() {
                    let sv = self.emit_expr(a0)?;
                    let p = self.coerce_str_ptr(sv);
                    self.emit_line(format!("{call_kw} void @__syn_uart_puts(ptr {})", p.repr));
                }
                return Ok(SsaVal::new("void", "0"));
            }
            "println" => {
                if let Some(a0) = args.first() {
                    let sv = self.emit_expr(a0)?;
                    let p = self.coerce_str_ptr(sv);
                    self.emit_line(format!("{call_kw} void @__syn_uart_println(ptr {})", p.repr));
                } else {
                    self.emit_line(format!("{call_kw} void @__syn_uart_putc(i8 10)"));
                }
                return Ok(SsaVal::new("void", "0"));
            }
            "assert" => {
                if let Some(a0) = args.first() {
                    let cv = self.emit_expr(a0)?;
                    self.emit_runtime_assert(cv);
                }
                return Ok(SsaVal::new("void", "0"));
            }
            "now_ms" | "now_ticks" => {
                let t = self.next_tmp();
                self.emit_line(format!("{t} = {call_kw} i32 @__syn_now_ms()"));
                let rlty = self.ctx.llvm_ty(ret_mir_ty);
                return Ok(self.coerce(SsaVal::new("i32", t), &rlty, false));
            }
            _ => {}
        }

        // Fallback if unknown pure helper
        let rlty = self.ctx.llvm_ty(ret_mir_ty);
        for a in args {
            let _ = self.emit_expr(a)?;
        }
        if rlty == "void" {
            Ok(SsaVal::new("void", "0"))
        } else {
            Ok(SsaVal::new(rlty, "zeroinitializer"))
        }
    }

    fn emit_binary(
        &mut self,
        op: MirBinOp,
        lhs: &MirExpr,
        rhs: &MirExpr,
        res_ty: &MirType,
    ) -> Result<SsaVal, CodegenError> {
        match op {
            MirBinOp::LogicalAnd => {
                let lv = self.emit_expr(lhs)?;
                let l1 = self.to_i1(lv);
                let lhs_lbl = self.cur_label.clone();
                let rhs_lbl = self.next_label("and_rhs");
                let end_lbl = self.next_label("and_end");
                self.emit_term(format!("br i1 {l1}, label %{rhs_lbl}, label %{end_lbl}"));

                self.start_block(&rhs_lbl);
                let rv = self.emit_expr(rhs)?;
                let r1 = self.to_i1(rv);
                let r8 = self.next_tmp();
                self.emit_line(format!("{r8} = zext i1 {r1} to i8"));
                let rhs_done_lbl = self.cur_label.clone();
                self.emit_term(format!("br label %{end_lbl}"));

                self.start_block(&end_lbl);
                let phi = self.next_tmp();
                self.emit_line(format!(
                    "{phi} = phi i8 [ 0, %{lhs_lbl} ], [ {r8}, %{rhs_done_lbl} ]"
                ));
                return Ok(SsaVal::new("i8", phi));
            }
            MirBinOp::LogicalOr => {
                let lv = self.emit_expr(lhs)?;
                let l1 = self.to_i1(lv);
                let lhs_lbl = self.cur_label.clone();
                let rhs_lbl = self.next_label("or_rhs");
                let end_lbl = self.next_label("or_end");
                self.emit_term(format!("br i1 {l1}, label %{end_lbl}, label %{rhs_lbl}"));

                self.start_block(&rhs_lbl);
                let rv = self.emit_expr(rhs)?;
                let r1 = self.to_i1(rv);
                let r8 = self.next_tmp();
                self.emit_line(format!("{r8} = zext i1 {r1} to i8"));
                let rhs_done_lbl = self.cur_label.clone();
                self.emit_term(format!("br label %{end_lbl}"));

                self.start_block(&end_lbl);
                let phi = self.next_tmp();
                self.emit_line(format!(
                    "{phi} = phi i8 [ 1, %{lhs_lbl} ], [ {r8}, %{rhs_done_lbl} ]"
                ));
                return Ok(SsaVal::new("i8", phi));
            }
            _ => {}
        }

        let lv = self.emit_expr(lhs)?;
        let rv = self.emit_expr(rhs)?;
        let is_fp = lhs.ty.is_float()
            || rhs.ty.is_float()
            || lv.ty == "float"
            || lv.ty == "double"
            || rv.ty == "float"
            || rv.ty == "double";

        if is_fp {
            let res_lty = self.ctx.llvm_ty(&res_ty.host_value_type());
            let common_fty = if lv.ty == "double"
                || rv.ty == "double"
                || lhs.ty == MirType::F64
                || rhs.ty == MirType::F64
                || res_lty == "double"
            {
                "double"
            } else {
                "float"
            };
            let lc = self.coerce(lv, common_fty, !lhs.ty.is_unsigned());
            let rc = self.coerce(rv, common_fty, !rhs.ty.is_unsigned());

            let is_cmp = matches!(
                op,
                MirBinOp::Eq
                    | MirBinOp::Ne
                    | MirBinOp::Lt
                    | MirBinOp::Le
                    | MirBinOp::Gt
                    | MirBinOp::Ge
            );
            if is_cmp {
                let fpred = match op {
                    MirBinOp::Eq => "oeq",
                    MirBinOp::Ne => "une",
                    MirBinOp::Lt => "olt",
                    MirBinOp::Le => "ole",
                    MirBinOp::Gt => "ogt",
                    MirBinOp::Ge => "oge",
                    _ => unreachable!(),
                };
                let cmp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!(
                    "{cmp} = fcmp {fpred} {common_fty} {}, {}",
                    lc.repr, rc.repr
                ));
                self.emit_line(format!("{b8} = zext i1 {cmp} to i8"));
                return Ok(SsaVal::new("i8", b8));
            }

            let finst = match op {
                MirBinOp::Add => "fadd",
                MirBinOp::Sub => "fsub",
                MirBinOp::Mul => "fmul",
                MirBinOp::Div => "fdiv",
                MirBinOp::Rem => "frem",
                _ => "fadd",
            };
            let t = self.next_tmp();
            self.emit_line(format!("{t} = {finst} {common_fty} {}, {}", lc.repr, rc.repr));
            return Ok(self.coerce(SsaVal::new(common_fty, t), &res_lty, true));
        }

        if matches!(op, MirBinOp::Shl | MirBinOp::Shr) {
            let lhs_host = lhs.ty.host_value_type();
            let shift_ty = self.ctx.llvm_ty(&lhs_host);
            let lc = self.coerce(lv, &shift_ty, !lhs.ty.is_unsigned());
            let rc = self.coerce(rv, &shift_ty, false);
            let inst = match op {
                MirBinOp::Shl => "shl",
                MirBinOp::Shr => {
                    if !lhs.ty.is_unsigned() {
                        "ashr"
                    } else {
                        "lshr"
                    }
                }
                _ => unreachable!(),
            };
            let t = self.next_tmp();
            self.emit_line(format!("{t} = {inst} {shift_ty} {}, {}", lc.repr, rc.repr));
            let dst_lty = self.ctx.llvm_ty(&res_ty.host_value_type());
            return Ok(self.coerce(SsaVal::new(shift_ty, t), &dst_lty, !res_ty.is_unsigned()));
        }

        let signed = !lhs.ty.is_unsigned() || !rhs.ty.is_unsigned();

        // Pick common operand type
        let common_ty = if lv.ty == "ptr addrspace(1)" || rv.ty == "ptr addrspace(1)" {
            "ptr addrspace(1)".to_string()
        } else if lv.ty == "ptr" || rv.ty == "ptr" {
            "ptr".to_string()
        } else {
            let lb = llvm_int_bits(&lv.ty);
            let rb = llvm_int_bits(&rv.ty);
            let res_lty = self.ctx.llvm_ty(&res_ty.host_value_type());
            let mut max_b = lb.max(rb);
            if !matches!(
                op,
                MirBinOp::Eq
                    | MirBinOp::Ne
                    | MirBinOp::Lt
                    | MirBinOp::Le
                    | MirBinOp::Gt
                    | MirBinOp::Ge
            ) && res_lty.starts_with('i')
            {
                max_b = max_b.max(llvm_int_bits(&res_lty));
            }
            format!("i{max_b}")
        };

        let lc = self.coerce(lv, &common_ty, !lhs.ty.is_unsigned());
        let rc = self.coerce(rv, &common_ty, !rhs.ty.is_unsigned());

        let is_cmp = matches!(
            op,
            MirBinOp::Eq
                | MirBinOp::Ne
                | MirBinOp::Lt
                | MirBinOp::Le
                | MirBinOp::Gt
                | MirBinOp::Ge
        );
        if is_cmp {
            let pred = match (op, signed) {
                (MirBinOp::Eq, _) => "eq",
                (MirBinOp::Ne, _) => "ne",
                (MirBinOp::Lt, true) => "slt",
                (MirBinOp::Lt, false) => "ult",
                (MirBinOp::Le, true) => "sle",
                (MirBinOp::Le, false) => "ule",
                (MirBinOp::Gt, true) => "sgt",
                (MirBinOp::Gt, false) => "ugt",
                (MirBinOp::Ge, true) => "sge",
                (MirBinOp::Ge, false) => "uge",
                _ => unreachable!(),
            };
            let cmp = self.next_tmp();
            let b8 = self.next_tmp();
            self.emit_line(format!(
                "{cmp} = icmp {pred} {common_ty} {}, {}",
                lc.repr, rc.repr
            ));
            self.emit_line(format!("{b8} = zext i1 {cmp} to i8"));
            return Ok(SsaVal::new("i8", b8));
        }

        let inst = match (op, signed) {
            (MirBinOp::Add, _) => "add",
            (MirBinOp::Sub, _) => "sub",
            (MirBinOp::Mul, _) => "mul",
            (MirBinOp::Div, true) => "sdiv",
            (MirBinOp::Div, false) => "udiv",
            (MirBinOp::Rem, true) => "srem",
            (MirBinOp::Rem, false) => "urem",
            (MirBinOp::BitOr, _) => "or",
            (MirBinOp::BitAnd, _) => "and",
            (MirBinOp::BitXor, _) => "xor",
            (MirBinOp::Shl, _) => "shl",
            (MirBinOp::Shr, true) => "ashr",
            (MirBinOp::Shr, false) => "lshr",
            _ => unreachable!(),
        };
        let t = self.next_tmp();
        self.emit_line(format!("{t} = {inst} {common_ty} {}, {}", lc.repr, rc.repr));
        let dst_lty = self.ctx.llvm_ty(&res_ty.host_value_type());
        Ok(self.coerce(SsaVal::new(common_ty, t), &dst_lty, !res_ty.is_unsigned()))
    }
}

fn crit_pair(contended: bool, call_kw: &str) -> (String, String) {
    if contended {
        (
            format!("  %pm = {call_kw} i32 @__syn_crit_enter()\n"),
            format!("  {call_kw} void @__syn_crit_exit(i32 %pm)\n"),
        )
    } else {
        (
            "  ; irq-mask elided (task-to-task)\n".to_string(),
            String::new(),
        )
    }
}
