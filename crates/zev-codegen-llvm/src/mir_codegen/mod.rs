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

pub(super) use std::collections::{BTreeMap, BTreeSet, HashMap};
pub(super) use std::fmt::Write;

pub(super) use zev_mir::{
    BlockId, MirAssignOp, MirBakedArg, MirBinOp, MirBlock, MirBootStep, MirCallDst,
    MirChildCallSlot, MirErrorSpec, MirExpr, MirExprKind, MirFloatFmt, MirFmtChunk, MirFnBody,
    MirFunction, MirIntFmt, MirIntrinsicOp, MirIrqBinding, MirIrqVector, MirIsr, MirMmioInstance,
    MirMmioLayout, MirModule, MirPlace, MirPrimKind, MirPrimNonBlockingOp, MirPrimReadOp,
    MirPrimTarget, MirPrimViewKind, MirStmt, MirStruct, MirSwitchPattern, MirSysQuery,
    MirTaskBlueprint, MirTerminator, MirType, MirUnaryOp, MirVar, MirWaitArm, MirWaitArmOp,
    MirWaitTimer, MirYieldKind, VarId, VarRole,
};

pub(super) use crate::arm_cortex_m::{
    build_nvic_map, emit_arm_cortex_m_startup_ir, sanitize_ident, NvicMap,
};
pub(super) use crate::riscv32::{
    assign_rv_irq_routes_for_spec, emit_riscv32_startup_ir, RvIrqRoute,
};
pub(super) use crate::runtime_ir::{emit_kernel_runtime_ir_with_features, RuntimeIrFeatures};
pub(super) use crate::target_spec::{CpuIsa, IrqModel, TargetSpec};
pub(super) use crate::toolchain::{CodegenError, CodegenOptions};

// ─────────────────────────────────────────────────────────────────────────────
// Module-wide Lowering Metadata
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub(super) struct StructInfo {
    pub(super) decl: MirStruct,
    pub(super) llvm_name: String,
    pub(super) byte_size: usize,
    pub(super) align: usize,
    pub(super) llvm_abi_size: usize,
    pub(super) llvm_nat_align: usize,
    pub(super) field_indices: HashMap<String, (usize, MirType)>,
    pub(super) llvm_fields: Vec<(Option<usize>, String)>,
}

#[derive(Clone)]
pub(super) struct CoroutineFrameLayout {
    pub(super) llvm_name: String,
    pub(super) has_ret_slot: bool,
    pub(super) ret_llvm_ty: String,
    /// Map from `VarId` -> GEP field index inside `%TaskFrame_*` / `%FnFrame_*`.
    pub(super) hoisted_var_gep: HashMap<VarId, usize>,
    /// Map from `union_group` -> `(gep_field_idx, num_i64_words)`.
    pub(super) union_group_gep: BTreeMap<usize, (usize, usize)>,
    /// Map from `slot_name` -> `union_group`.
    pub(super) slot_to_group: HashMap<String, usize>,
    /// Approximate byte size (8-byte aligned conservative upper bound for parent union sizing).
    pub(super) byte_size_bound: usize,
    /// Ordered field LLVM types for `type { ... }`.
    pub(super) field_llvm_tys: Vec<String>,
}

#[derive(Clone, Debug)]
pub(super) struct SsaVal {
    pub(super) ty: String,
    pub(super) repr: String,
}

impl SsaVal {
    pub(super) fn new(ty: impl Into<String>, repr: impl Into<String>) -> Self {
        Self {
            ty: ty.into(),
            repr: repr.into(),
        }
    }
}

pub(super) struct ModuleCtx<'a> {
    pub(super) module: &'a MirModule,
    pub(super) opts: &'a CodegenOptions,
    pub(super) spec: TargetSpec,
    pub(super) nvic: NvicMap,
    pub(super) rv_irq_routes: Vec<RvIrqRoute>,
    pub(super) avr_irq_vectors: BTreeSet<u32>,
    pub(super) mmio_layouts: HashMap<String, MirMmioLayout>,
    pub(super) mmio_instances: HashMap<String, MirMmioInstance>,
    pub(super) structs: HashMap<String, StructInfo>,
    pub(super) struct_order: Vec<String>,
    pub(super) enum_variants: HashMap<(String, String), i64>,
    pub(super) bare_variants: HashMap<String, i64>,
    pub(super) consts: HashMap<String, i128>,
    pub(super) float_consts: HashMap<String, u64>,
    pub(super) fnptr_consts: HashMap<String, String>,
    pub(super) rodata_consts: HashMap<String, (String, MirType)>,
    pub(super) fn_sigs: HashMap<String, FnLlvmSig>,
    pub(super) extern_sigs: HashMap<String, ExternLlvmSig>,
    pub(super) fn_frames: HashMap<String, CoroutineFrameLayout>,
    pub(super) bp_frames: HashMap<String, CoroutineFrameLayout>,
    pub(super) prim_globals: HashMap<String, MirPrimKind>,
    pub(super) distinct_prims: BTreeMap<String, MirPrimKind>,
    pub(super) distinct_inbox_views: BTreeMap<String, MirType>,
    pub(super) uses_byte_ring_view: bool,
    pub(super) uses_byte_ring_wait_write: bool,
    pub(super) uses_pool_lease: bool,
    pub(super) dma_layout_structs: BTreeSet<String>,
    pub(super) vis: zev_mir::opt::VisibilityReport,
    pub(super) rt_features: RuntimeIrFeatures,
    pub(super) hooks: ResolvedSysHooks,
    pub(super) rom_wrappers: BTreeSet<RomWrapperSpec>,
    pub(super) strings: Vec<String>,
    pub(super) string_dedup: HashMap<String, usize>,
    pub(super) const_blobs: Vec<String>,
    pub(super) const_blob_dedup: HashMap<String, usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum HookKind {
    Putc,
    ClockMs,
    Exit,
    FaultHandler,
    Idle,
    IrqRoute,
    IrqAck,
}

impl HookKind {
    pub(super) const fn suffix(self) -> &'static str {
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

    pub(super) const fn wrapper_sym(self) -> &'static str {
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
pub(super) enum StaticHookMode {
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
    pub(super) fn tag_index_for(&self, fn_name: &str) -> Option<usize> {
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
pub(super) struct ResolvedSysHooks {
    pub(super) putc: StaticHookMode,
    pub(super) clock_ms: StaticHookMode,
    pub(super) exit: StaticHookMode,
    pub(super) fault_handler: StaticHookMode,
    pub(super) idle: StaticHookMode,
    pub(super) irq_route: StaticHookMode,
    pub(super) irq_ack: StaticHookMode,
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
    pub(super) fn mode(&self, kind: HookKind) -> &StaticHookMode {
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
pub(super) enum RomWrapperKind {
    U32,
    Void,
    Noreturn,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct RomWrapperSpec {
    pub(super) kind: RomWrapperKind,
    pub(super) stack_bound: u32,
    pub(super) param_llvm_tys: Vec<String>,
}

impl RomWrapperSpec {
    pub(super) fn symbol(&self) -> String {
        let mut sig_suffix = String::new();
        for ty in &self.param_llvm_tys {
            sig_suffix.push('_');
            sig_suffix.push_str(&sanitize_ident(ty));
        }
        let n = self.param_llvm_tys.len();
        match self.kind {
            RomWrapperKind::U32 => {
                format!(
                    "@__syn_rom_call_u32_a{n}{sig_suffix}_sb{}",
                    self.stack_bound
                )
            }
            RomWrapperKind::Void => {
                format!(
                    "@__syn_rom_call_void_a{n}{sig_suffix}_sb{}",
                    self.stack_bound
                )
            }
            RomWrapperKind::Noreturn => {
                format!("@__syn_rom_noreturn_a{n}{sig_suffix}")
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct FnLlvmSig {
    pub(super) symbol: String,
    pub(super) params: Vec<MirType>,
    pub(super) ret_mir_ty: MirType,
    pub(super) ret_llvm_ty: String,
}

#[derive(Clone)]
pub(super) struct ExternLlvmSig {
    pub(super) name: String,
    pub(super) params: Vec<MirType>,
    pub(super) ret_ty: MirType,
    pub(super) variadic: bool,
}

impl<'a> ModuleCtx<'a> {
    pub(super) const fn fn_as(&self) -> &'static str {
        self.spec.isa.fn_addrspace_attr()
    }

    pub(super) const fn call_kw(&self) -> &'static str {
        self.spec.isa.call_kw()
    }
}

pub(super) const fn align_up(offset: usize, align: usize) -> usize {
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
        fnptr_consts: HashMap::new(),
        rodata_consts: HashMap::new(),
        fn_sigs: HashMap::new(),
        extern_sigs: HashMap::new(),
        fn_frames: HashMap::new(),
        bp_frames: HashMap::new(),
        prim_globals: HashMap::new(),
        distinct_prims: BTreeMap::new(),
        distinct_inbox_views: BTreeMap::new(),
        uses_byte_ring_view: false,
        uses_byte_ring_wait_write: false,
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
    emit_kernel_runtime_ir_with_features(
        &mut out,
        module.spawned_tasks.len(),
        opts,
        &ctx.rt_features,
    )?;

    // 2. Pure-ISA vector tables, startup reset routines, and software/boot IRQ dispatchers
    match ctx.spec.isa {
        CpuIsa::ArmCortexM => {
            emit_arm_cortex_m_startup_ir(&mut out, &ctx.spec, opts.fn_size_attrs(), &ctx.nvic);
            emit_cm_irq_dispatch_ir(&mut out, &ctx);
        }
        CpuIsa::Riscv32 => {
            emit_riscv32_startup_ir(
                &mut out,
                &ctx.spec,
                opts.fn_size_attrs(),
                &ctx.rv_irq_routes,
            );
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
    ctx.emit_isr_state_and_selector_globals_ir(&mut out)?;
    ctx.emit_sys_hook_globals_ir(&mut out);

    // 4. OS primitive type declarations, global instances, and typed helper functions
    ctx.emit_primitives_ir(&mut out)?;

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

    // 13. Immutable aggregate / slice / str constants and pooled string literals (.rodata)
    ctx.emit_rodata_consts_ir(&mut out)?;
    ctx.emit_string_constants_ir(&mut out);

    Ok(out)
}

/// Format an integer constant for LLVM type `llvm_ty`, raising an internal compiler error
/// (`CodegenError::new("ice", ...)`) if `value` does not fit the signed-or-unsigned bit range.
pub(super) fn format_checked_llvm_int_const(
    llvm_ty: &str,
    value: i128,
) -> Result<String, CodegenError> {
    let fits = match llvm_ty {
        "i1" => (-1..=1).contains(&value),
        "i8" => (i128::from(i8::MIN)..=i128::from(u8::MAX)).contains(&value),
        "i16" => (i128::from(i16::MIN)..=i128::from(u16::MAX)).contains(&value),
        "i32" => (i128::from(i32::MIN)..=i128::from(u32::MAX)).contains(&value),
        "i64" => (i128::from(i64::MIN)..=i128::from(u64::MAX)).contains(&value),
        _ => {
            return Err(CodegenError::new(
                "ice",
                format!(
                    "internal compiler error: expected LLVM integer type, found `{llvm_ty}` for constant {value}"
                ),
            ));
        }
    };
    if !fits {
        return Err(CodegenError::new(
            "ice",
            format!(
                "internal compiler error: integer constant {value} does not fit LLVM type `{llvm_ty}`"
            ),
        ));
    }
    Ok(value.to_string())
}

mod expr_emit;
mod fn_builder;
mod module_emit;

pub(in crate::mir_codegen) use expr_emit::*;
pub(in crate::mir_codegen) use module_emit::*;

pub(super) struct FnIrBuilder<'ctx, 'm> {
    pub(super) ctx: &'ctx mut ModuleCtx<'m>,
    pub(super) vars: Vec<MirVar>,
    pub(super) coroutine: Option<CoroutineFrameLayout>,
    pub(super) isr_state_ptrs: HashMap<VarId, String>,
    pub(super) ret_ty: MirType,
    pub(super) error_spec: MirErrorSpec,
    pub(super) in_isr: bool,
    pub(super) crit_depth: u32,
    pub(super) buf: String,
    pub(super) entry_allocas: String,
    pub(super) entry_alloca_insert_pos: Option<usize>,
    pub(super) temp_alloca_id: usize,
    pub(super) tmp_id: usize,
    pub(super) lbl_id: usize,
    pub(super) cur_label: String,
    pub(super) terminated: bool,
    pub(super) loop_targets: Vec<(String, String)>, // (continue_lbl, break_lbl) for Direct fns
}
