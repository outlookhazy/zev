//! Static Zero-Heap RAM Layout & Budget Calculator (`RamReport`).
//!
//! Computes exact static byte sizes and alignments under both:
//! - **32-bit Bare-Metal MCU ABI** (ARM Cortex-M EABI / RISC-V ILP32: 4-byte pointers & `usize`)
//! - **64-bit Host Simulation ABI** (x86_64 LP64: 8-byte pointers & `usize`)

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::ir::{
    MirChildCallSlot, MirFnBody, MirFunction, MirIrqVector, MirModule, MirPrimKind,
    MirPrimViewKind, MirPrimitive, MirPriority, MirSpawnInstance, MirStruct, MirSubsystemGlobal,
    MirTaskBlueprint, MirType, MirVar, MonomorphizedItemKind, MonomorphizedMirItem, VarRole,
};

/// Target data-model parameters for C/ABI struct layout calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetAbi {
    Mcu32,
    Host64,
}

impl TargetAbi {
    pub const fn ptr_bytes(self) -> usize {
        match self {
            TargetAbi::Mcu32 => 4,
            TargetAbi::Host64 => 8,
        }
    }
}

/// Computed `(size, align)` pair in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SizeAlign {
    pub size: usize,
    pub align: usize,
}

impl SizeAlign {
    pub const fn new(size: usize, align: usize) -> Self {
        Self { size, align }
    }
}

const fn align_up(offset: usize, align: usize) -> usize {
    if align <= 1 {
        offset
    } else {
        (offset + align - 1) & !(align - 1)
    }
}

/// Variable placement record for diagnostics & `zevc ram` inspection.
#[derive(Debug, Clone)]
pub struct VarRamInfo {
    pub name: String,
    pub ty_display: String,
    pub role: VarRole,
    pub hoisted: bool,
    pub in_noyield_scope: bool,
    pub bytes_32: usize,
    pub bytes_64: usize,
}

/// Child coroutine slot inside a parent's disjoint `union`.
#[derive(Debug, Clone)]
pub struct ChildSlotRamInfo {
    pub slot_name: String,
    pub callee: String,
    pub union_group: usize,
    pub bytes_32: usize,
    pub bytes_64: usize,
}

/// Frame layout report for a yielding `fn` or `task` blueprint.
#[derive(Debug, Clone)]
pub struct FrameLayoutReport {
    pub name: String,
    pub packed_bytes_32: usize,
    pub naive_bytes_32: usize,
    pub union_saved_bytes_32: usize,
    pub child_union_bytes_32: usize,
    pub child_naive_bytes_32: usize,
    pub packed_bytes_64: usize,
    pub naive_bytes_64: usize,
    pub union_saved_bytes_64: usize,
    pub hoisted_vars: Vec<VarRamInfo>,
    pub stack_vars: Vec<VarRamInfo>,
    pub child_slots: Vec<ChildSlotRamInfo>,
}

/// Static RAM allocation for a single `spawn`ed task slot (or `main` supervisor).
#[derive(Debug, Clone)]
pub struct SpawnedTaskRam {
    pub slot_id: usize,
    pub instance_name: String,
    pub blueprint_name: String,
    pub priority: MirPriority,
    pub core_pin: Option<u8>,
    pub is_main_supervisor: bool,
    pub provenance_path: Vec<String>,
    pub conditional: bool,
    pub frame_bytes_32: usize,
    pub naive_frame_bytes_32: usize,
    pub union_saved_bytes_32: usize,
    pub tcb_bytes_32: usize,
    pub frame_bytes_64: usize,
    pub naive_frame_bytes_64: usize,
    pub union_saved_bytes_64: usize,
    pub tcb_bytes_64: usize,
    pub hoisted_vars: Vec<VarRamInfo>,
    pub stack_vars: Vec<VarRamInfo>,
    pub child_slots: Vec<ChildSlotRamInfo>,
}

/// Static RAM visibility for a `SoftYielding` function compiled into both a coroutine
/// state machine (`name`) and a zero-frame direct stack clone (`noyield_name`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualSpecializedFnRam {
    pub name: String,
    pub noyield_name: String,
    pub frame_bytes_32: usize,
    pub frame_bytes_64: usize,
}

/// Static RAM allocation for a top-level OS or DMA primitive.
#[derive(Debug, Clone)]
pub struct PrimitiveRam {
    pub name: String,
    pub kind_display: String,
    pub dma_safe: bool,
    pub holds_sleep_lock: bool,
    pub dead_stripped: bool,
    pub irq_mask_elided: bool,
    pub bytes_32: usize,
    pub bytes_64: usize,
}

/// Architecture-wide SRAM totals.
#[derive(Debug, Clone, Default)]
pub struct ArchRamTotals {
    pub task_frames_bytes: usize,
    pub naive_task_frames_bytes: usize,
    pub union_bytes_saved: usize,
    pub boot_frames_bytes: usize,
    pub tcb_table_bytes: usize,
    pub primitives_bytes: usize,
    pub isr_state_bytes: usize,
    pub scheduler_bytes: usize,
    pub total_sram_bytes: usize,
    pub naive_total_sram_bytes: usize,
    pub heap_bytes: usize,
}

/// Complete zero-heap static SRAM report for a lowered `MirModule`.
#[derive(Debug, Clone, Default)]
pub struct RamReport {
    pub fn_frames: Vec<FrameLayoutReport>,
    pub blueprint_frames: Vec<FrameLayoutReport>,
    pub spawned_tasks: Vec<SpawnedTaskRam>,
    pub primitives: Vec<PrimitiveRam>,
    pub boot_frames: Vec<(String, usize, usize)>,
    pub isr_states: Vec<(String, String, usize, usize)>,
    pub dual_specialized_fns: Vec<DualSpecializedFnRam>,
    pub monomorphized_instances: Vec<MonomorphizedMirItem>,
    pub mcu32: ArchRamTotals,
    pub host64: ArchRamTotals,
}

impl RamReport {
    /// Human-readable static SRAM budget table suitable for `zevc ram`.
    pub fn format_table(&self) -> String {
        let mut out = String::new();
        out.push_str("================================================================================\n");
        out.push_str("  STATIC ZERO-HEAP SRAM REPORT (32-bit MCU Target / 64-bit Host Sim)\n");
        out.push_str("================================================================================\n\n");

        out.push_str("1. Spawned Task Instances (Static Coroutine Frames + TCBs)\n");
        out.push_str("--------------------------------------------------------------------------------\n");
        out.push_str(&format!(
            "  {:<4} {:<24} {:<10} {:>10} {:>10} {:>11} {:>9}\n",
            "Slot", "Instance (Blueprint)", "Priority", "Frame(32b)", "TCB(32b)", "UnionSaved", "Frame(64b)"
        ));
        out.push_str("  ------------------------------------------------------------------------------\n");
        if self.spawned_tasks.is_empty() {
            out.push_str("  (no spawned tasks)\n");
        } else {
            for t in &self.spawned_tasks {
                let core_tag = match t.core_pin {
                    Some(c) => format!(" @core({c})"),
                    None => String::new(),
                };
                let cond_tag = if t.conditional { " [conditional]" } else { "" };
                let label = if t.is_main_supervisor {
                    "main (supervisor)".to_string()
                } else {
                    format!("{} ({})", t.instance_name, t.blueprint_name)
                };
                out.push_str(&format!(
                    "  #{:<3} {:<24} {:<10} {:>8} B {:>8} B {:>9} B {:>7} B{}{}\n",
                    t.slot_id,
                    label,
                    t.priority.label(),
                    t.frame_bytes_32,
                    t.tcb_bytes_32,
                    t.union_saved_bytes_32,
                    t.frame_bytes_64,
                    core_tag,
                    cond_tag
                ));

                if t.provenance_path.len() > 2 || t.conditional {
                    out.push_str(&format!(
                        "       boot provenance  : {}{}\n",
                        t.provenance_path.join(" -> "),
                        cond_tag
                    ));
                }

                let user_hoisted: Vec<String> = t
                    .hoisted_vars
                    .iter()
                    .filter(|v| v.role != VarRole::CompilerTemp)
                    .map(|v| format!("{}: {} ({}B)", v.name, v.ty_display, v.bytes_32))
                    .collect();
                if !user_hoisted.is_empty() {
                    out.push_str(&format!(
                        "       hoisted in frame : {}\n",
                        user_hoisted.join(", ")
                    ));
                }

                let user_stack: Vec<String> = t
                    .stack_vars
                    .iter()
                    .filter(|v| v.role == VarRole::UserLocal)
                    .map(|v| {
                        let tag = if v.in_noyield_scope { " [noyield]" } else { "" };
                        format!("{}: {}{}", v.name, v.ty_display, tag)
                    })
                    .collect();
                if !user_stack.is_empty() {
                    out.push_str(&format!(
                        "       stack-only local : {}\n",
                        user_stack.join(", ")
                    ));
                }

                if !t.child_slots.is_empty() {
                    let members: Vec<String> = t
                        .child_slots
                        .iter()
                        .map(|c| format!("{} ({}B)", c.callee, c.bytes_32))
                        .collect();
                    out.push_str(&format!(
                        "       child union      : union {{ {} }} -> saved {} B\n",
                        members.join(" | "),
                        t.union_saved_bytes_32
                    ));
                }
            }
        }
        out.push('\n');

        if !self.fn_frames.is_empty() {
            out.push_str("2. Yielding Helper Function Frames (Embedded in Caller Unions)\n");
            out.push_str("--------------------------------------------------------------------------------\n");
            for f in &self.fn_frames {
                let hoisted: Vec<String> = f
                    .hoisted_vars
                    .iter()
                    .filter(|v| v.role != VarRole::CompilerTemp)
                    .map(|v| format!("{}: {}", v.name, v.ty_display))
                    .collect();
                let vars_desc = if hoisted.is_empty() {
                    "_pc only".to_string()
                } else {
                    hoisted.join(", ")
                };
                out.push_str(&format!(
                    "  {:<28} {:>6} B (32b) / {:>6} B (64b)  [{}]\n",
                    f.name, f.packed_bytes_32, f.packed_bytes_64, vars_desc
                ));
            }
            out.push('\n');
        }

        out.push_str("3. Static OS & DMA Endpoints\n");
        out.push_str("--------------------------------------------------------------------------------\n");
        if self.primitives.is_empty() {
            out.push_str("  (none)\n");
        } else {
            for p in &self.primitives {
                let mut flags = Vec::new();
                if p.dma_safe {
                    flags.push("@dma_safe");
                }
                if p.holds_sleep_lock {
                    flags.push("clamps sleep->LIGHT");
                }
                if p.dead_stripped {
                    flags.push("dead watch stripped -> 0 B");
                } else if p.irq_mask_elided {
                    flags.push("irq-mask elided");
                }
                let flag_str = if flags.is_empty() {
                    String::new()
                } else {
                    format!("  [{}]", flags.join(", "))
                };
                out.push_str(&format!(
                    "  {:<18} {:<26} {:>6} B (32b) / {:>6} B (64b){}\n",
                    p.name, p.kind_display, p.bytes_32, p.bytes_64, flag_str
                ));
            }
        }
        out.push('\n');

        out.push_str("4. Total Static SRAM Summary\n");
        out.push_str("--------------------------------------------------------------------------------\n");
        out.push_str(&format!(
            "  {:<34} {:>12} {:>14}\n",
            "Component", "32-bit MCU", "64-bit Host"
        ));
        out.push_str("  ------------------------------------------------------------------------------\n");
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "Task Coroutine Frames (packed)",
            self.mcu32.task_frames_bytes,
            self.host64.task_frames_bytes
        ));
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "Child-Frame Union Bytes Saved",
            self.mcu32.union_bytes_saved,
            self.host64.union_bytes_saved
        ));
        if self.mcu32.boot_frames_bytes > 0 || self.host64.boot_frames_bytes > 0 {
            out.push_str(&format!(
                "  {:<34} {:>10} B {:>12} B\n",
                "Sequential Boot Frames",
                self.mcu32.boot_frames_bytes,
                self.host64.boot_frames_bytes
            ));
        }
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "Scheduler Task Table (TCBs)",
            self.mcu32.tcb_table_bytes,
            self.host64.tcb_table_bytes
        ));
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "OS & DMA Primitives",
            self.mcu32.primitives_bytes,
            self.host64.primitives_bytes
        ));
        if self.mcu32.isr_state_bytes > 0 || self.host64.isr_state_bytes > 0 {
            out.push_str(&format!(
                "  {:<34} {:>10} B {:>12} B\n",
                "ISR Static State",
                self.mcu32.isr_state_bytes,
                self.host64.isr_state_bytes
            ));
        }
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "Scheduler + Sleep Coordinator",
            self.mcu32.scheduler_bytes,
            self.host64.scheduler_bytes
        ));
        out.push_str("  ------------------------------------------------------------------------------\n");
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "TOTAL STATIC SRAM (Packed)",
            self.mcu32.total_sram_bytes,
            self.host64.total_sram_bytes
        ));
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "Naive SRAM (Without Child Unions)",
            self.mcu32.naive_total_sram_bytes,
            self.host64.naive_total_sram_bytes
        ));
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "Dynamic Heap Allocation", self.mcu32.heap_bytes, self.host64.heap_bytes
        ));

        if !self.isr_states.is_empty() {
            out.push_str("\n5. ISR Static State\n");
            out.push_str("--------------------------------------------------------------------------------\n");
            for (sym, ty_disp, b32, b64) in &self.isr_states {
                out.push_str(&format!(
                    "  {:<34} {:<18} {:>6} B (32b) / {:>6} B (64b)\n",
                    sym, ty_disp, b32, b64
                ));
            }
        }

        if !self.dual_specialized_fns.is_empty() {
            out.push_str("\n6. Dual-Specialized Functions\n");
            out.push_str("--------------------------------------------------------------------------------\n");
            for ds in &self.dual_specialized_fns {
                out.push_str(&format!(
                    "  {:<28} coro: {:>4} B (32b) / {:>4} B (64b) | noyield clone `{}`: 0 B frame\n",
                    ds.name, ds.frame_bytes_32, ds.frame_bytes_64, ds.noyield_name
                ));
            }
        }

        if !self.monomorphized_instances.is_empty() {
            out.push_str("\n7. Monomorphized Generic Instantiations\n");
            out.push_str("--------------------------------------------------------------------------------\n");
            for m in &self.monomorphized_instances {
                match m.kind {
                    MonomorphizedItemKind::Struct => {
                        out.push_str(&format!(
                            "  struct {:<28} -> `{}` ({} B 32b / {} B 64b)\n",
                            m.signature, m.mangled_name, m.size_32, m.size_64
                        ));
                    }
                    MonomorphizedItemKind::Fn => {
                        if m.size_32 > 0 || m.size_64 > 0 {
                            out.push_str(&format!(
                                "  fn     {:<28} -> `{}` (coro frame: {} B 32b / {} B 64b)\n",
                                m.signature, m.mangled_name, m.size_32, m.size_64
                            ));
                        } else {
                            out.push_str(&format!(
                                "  fn     {:<28} -> `{}` (direct, 0 B static SRAM)\n",
                                m.signature, m.mangled_name
                            ));
                        }
                    }
                }
            }
        }

        out.push_str("================================================================================\n");
        out
    }
}

impl fmt::Display for RamReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_table())
    }
}

struct LayoutCtx<'a> {
    structs: BTreeMap<&'a str, &'a MirStruct>,
    enums: BTreeSet<&'a str>,
}

impl<'a> LayoutCtx<'a> {
    fn struct_layout(&self, fields: &[SizeAlign], packed: bool) -> SizeAlign {
        if fields.is_empty() {
            return SizeAlign::new(1, 1);
        }
        if packed {
            let total: usize = fields.iter().map(|f| f.size).sum();
            return SizeAlign::new(total.max(1), 1);
        }
        let mut offset = 0usize;
        let mut max_align = 1usize;
        for f in fields {
            let a = f.align.max(1);
            if a > max_align {
                max_align = a;
            }
            offset = align_up(offset, a);
            offset += f.size;
        }
        let size = align_up(offset, max_align).max(1);
        SizeAlign::new(size, max_align)
    }

    fn union_layout(&self, members: &[SizeAlign]) -> SizeAlign {
        if members.is_empty() {
            return SizeAlign::new(0, 1);
        }
        let mut max_size = 0usize;
        let mut max_align = 1usize;
        for m in members {
            if m.size > max_size {
                max_size = m.size;
            }
            if m.align > max_align {
                max_align = m.align;
            }
        }
        SizeAlign::new(align_up(max_size, max_align), max_align)
    }

    fn type_layout(&self, ty: &MirType, abi: TargetAbi) -> SizeAlign {
        let ptr = abi.ptr_bytes();
        match ty {
            MirType::Void => SizeAlign::new(0, 1),
            MirType::Bool | MirType::U8 | MirType::I8 => SizeAlign::new(1, 1),
            MirType::U16 | MirType::I16 => SizeAlign::new(2, 2),
            MirType::U32 | MirType::I32 | MirType::F32 => {
                SizeAlign::new(4, 4)
            }
            MirType::U64 | MirType::I64 | MirType::F64 => SizeAlign::new(8, 8),
            MirType::Usize
            | MirType::Isize
            | MirType::Str
            | MirType::Ptr(_)
            | MirType::MmioPtr(_)
            | MirType::FnPtr { .. } => {
                SizeAlign::new(ptr, ptr)
            }
            MirType::EndianInt { bits, .. } => {
                let bytes = (*bits as usize) / 8;
                // Wire-endian wrapper `struct { uint8_t b[N]; }` has byte alignment 1!
                SizeAlign::new(bytes, 1)
            }
            MirType::Enum(_) => SizeAlign::new(4, 4),
            MirType::Struct(name) => {
                if let Some(st) = self.structs.get(name.as_str()) {
                    let fl: Vec<SizeAlign> = st
                        .fields
                        .iter()
                        .map(|f| self.type_layout(&f.ty, abi))
                        .collect();
                    self.struct_layout(&fl, st.packed)
                } else {
                    let _ = &self.enums;
                    SizeAlign::new(4, 4)
                }
            }
            MirType::Slice(_) => {
                // `{ T *ptr; size_t len; }`
                SizeAlign::new(ptr * 2, ptr)
            }
            MirType::Array { elem, len } => {
                let el = self.type_layout(elem, abi);
                SizeAlign::new(el.size * len, el.align)
            }
            MirType::Primitive(pk) => self.prim_layout(pk, false, abi),
            MirType::PrimView(MirPrimViewKind::ByteRing) => SizeAlign::new(ptr * 2, ptr),
            MirType::PrimView(MirPrimViewKind::Inbox { .. }) => SizeAlign::new(ptr * 3, ptr),
            MirType::Result { ok, err } => {
                let mut fields = vec![SizeAlign::new(1, 1)]; // bool is_err
                if let Some(et) = err {
                    if !et.is_void() {
                        fields.push(self.type_layout(et, abi));
                    }
                }
                if !ok.is_void() {
                    fields.push(self.type_layout(ok, abi));
                }
                self.struct_layout(&fields, false)
            }
        }
    }

    fn prim_layout(&self, kind: &MirPrimKind, dma_safe: bool, abi: TargetAbi) -> SizeAlign {
        let ptr = abi.ptr_bytes();
        let raw = match kind {
            MirPrimKind::Signal => {
                // `{ syn_task_t *waiter; volatile bool flag; uint8_t boost_prio; }`
                self.struct_layout(
                    &[
                        SizeAlign::new(ptr, ptr),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                    ],
                    false,
                )
            }
            MirPrimKind::Watch { elem } => {
                // `{ T value; volatile uint32_t seq; }`
                let el = self.type_layout(elem, abi);
                self.struct_layout(&[el, SizeAlign::new(4, 4)], false)
            }
            MirPrimKind::Inbox { elem, cap } => {
                // `{ syn_inbox_hdr_t hdr; T buf[N]; uint8_t boost_buf[N]; }`
                let hdr = self.struct_layout(
                    &[
                        SizeAlign::new(ptr, ptr),
                        SizeAlign::new(ptr, ptr),
                        SizeAlign::new(2, 2),
                        SizeAlign::new(2, 2),
                        SizeAlign::new(2, 2),
                        SizeAlign::new(2, 2),
                    ],
                    false,
                );
                let el = self.type_layout(elem, abi);
                self.struct_layout(
                    &[
                        hdr,
                        SizeAlign::new(el.size * cap, el.align),
                        SizeAlign::new(*cap, 1),
                    ],
                    false,
                )
            }
            MirPrimKind::PingPong { elem, cap } => {
                // `{ T buf[2][N]; syn_task_t *waiter; uint32_t sleep_lock_bit; uint16_t ready_len; uint8_t write_idx, ready_idx; bool chunk_ready, dma_active; }`
                let el = self.type_layout(elem, abi);
                self.struct_layout(
                    &[
                        SizeAlign::new(el.size * cap * 2, el.align),
                        SizeAlign::new(ptr, ptr),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(2, 2),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                    ],
                    false,
                )
            }
            MirPrimKind::Lease { elem, cap } => {
                // `{ T buf[N]; syn_task_t *waiter; uint32_t sleep_lock_bit; uint16_t len; bool leased, ready; }`
                let el = self.type_layout(elem, abi);
                self.struct_layout(
                    &[
                        SizeAlign::new(el.size * cap, el.align),
                        SizeAlign::new(ptr, ptr),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(2, 2),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                    ],
                    false,
                )
            }
            MirPrimKind::ByteRing { cap } => {
                // `{ syn_ring_hdr_t hdr; uint8_t buf[N]; }`
                let hdr = self.struct_layout(
                    &[
                        SizeAlign::new(ptr, ptr),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(1, 1),
                    ],
                    false,
                );
                self.struct_layout(&[hdr, SizeAlign::new(*cap, 1)], false)
            }
        };

        if dma_safe {
            let align = raw.align.max(4);
            SizeAlign::new(align_up(raw.size, align), align)
        } else {
            raw
        }
    }
}

#[derive(Clone, Copy, Default)]
struct DualFrameMetrics {
    packed_32: SizeAlign,
    naive_32: SizeAlign,
    child_union_32: usize,
    child_naive_32: usize,
    packed_64: SizeAlign,
    naive_64: SizeAlign,
}

fn tcb_bytes(abi: TargetAbi) -> usize {
    match abi {
        TargetAbi::Mcu32 => 36,
        TargetAbi::Host64 => 56,
    }
}

fn scheduler_bytes(abi: TargetAbi) -> usize {
    match abi {
        TargetAbi::Mcu32 => 52,
        TargetAbi::Host64 => 64,
    }
}

fn struct_holds_dma_sleep_lock(name: &str, structs: &BTreeMap<&str, &MirStruct>) -> bool {
    let Some(st) = structs.get(name) else {
        return false;
    };
    st.fields.iter().any(|f| match &f.ty {
        MirType::Primitive(pk) => pk.holds_dma_sleep_lock(),
        MirType::Struct(sub) => struct_holds_dma_sleep_lock(sub, structs),
        _ => false,
    })
}

/// Compute the complete static `RamReport` for `module` and store it in `module.ram_report`.
pub fn compute_ram_report(
    structs: &[MirStruct],
    functions: &[MirFunction],
    blueprints: &[MirTaskBlueprint],
    spawned: &[MirSpawnInstance],
    primitives: &[MirPrimitive],
    irq_vectors: &[MirIrqVector],
    boot_yielding_callees: &[String],
) -> RamReport {
    compute_ram_report_with_subsystems(
        structs,
        functions,
        blueprints,
        spawned,
        primitives,
        &[],
        irq_vectors,
        boot_yielding_callees,
    )
}

pub fn compute_ram_report_with_subsystems(
    structs: &[MirStruct],
    functions: &[MirFunction],
    blueprints: &[MirTaskBlueprint],
    spawned: &[MirSpawnInstance],
    primitives: &[MirPrimitive],
    subsystems: &[MirSubsystemGlobal],
    irq_vectors: &[MirIrqVector],
    boot_yielding_callees: &[String],
) -> RamReport {
    let ctx = LayoutCtx {
        structs: structs.iter().map(|s| (s.name.as_str(), s)).collect(),
        enums: BTreeSet::new(),
    };

    let fn_map: BTreeMap<&str, &MirFunction> =
        functions.iter().map(|f| (f.name.as_str(), f)).collect();

    let mut fn_cache: BTreeMap<String, DualFrameMetrics> = BTreeMap::new();
    let mut visiting: BTreeSet<String> = BTreeSet::new();

    for f in functions {
        if f.is_yielding {
            resolve_fn_metrics(f.name.as_str(), &fn_map, &ctx, &mut fn_cache, &mut visiting);
        }
    }

    let mut fn_frames = Vec::new();
    for f in functions {
        if let MirFnBody::Coroutine { child_slots, .. } = &f.body {
            let m = fn_cache.get(&f.name).copied().unwrap_or_default();
            fn_frames.push(build_frame_report(
                &f.name,
                &f.vars,
                child_slots,
                m,
                &fn_cache,
                &ctx,
            ));
        }
    }

    let mut bp_reports: BTreeMap<String, (FrameLayoutReport, DualFrameMetrics)> = BTreeMap::new();
    for bp in blueprints {
        let m = compute_frame_metrics(&bp.vars, &bp.child_slots, None, &fn_cache, &ctx);
        let rep = build_frame_report(&bp.name, &bp.vars, &bp.child_slots, m, &fn_cache, &ctx);
        bp_reports.insert(bp.name.clone(), (rep, m));
    }

    let blueprint_frames: Vec<FrameLayoutReport> =
        bp_reports.values().map(|(r, _)| r.clone()).collect();

    let mut spawned_tasks = Vec::new();
    for sp in spawned {
        let (bp_rep, m) = bp_reports
            .get(&sp.blueprint_name)
            .cloned()
            .unwrap_or_else(|| {
                (
                    FrameLayoutReport {
                        name: sp.blueprint_name.clone(),
                        packed_bytes_32: 2,
                        naive_bytes_32: 2,
                        union_saved_bytes_32: 0,
                        child_union_bytes_32: 0,
                        child_naive_bytes_32: 0,
                        packed_bytes_64: 2,
                        naive_bytes_64: 2,
                        union_saved_bytes_64: 0,
                        hoisted_vars: vec![],
                        stack_vars: vec![],
                        child_slots: vec![],
                    },
                    DualFrameMetrics::default(),
                )
            });

        let saved_32 = m.naive_32.size.saturating_sub(m.packed_32.size);
        let saved_64 = m.naive_64.size.saturating_sub(m.packed_64.size);

        spawned_tasks.push(SpawnedTaskRam {
            slot_id: sp.slot_id,
            instance_name: sp.instance_name.clone(),
            blueprint_name: sp.blueprint_name.clone(),
            priority: sp.priority,
            core_pin: sp.core_pin,
            is_main_supervisor: sp.is_main_supervisor,
            provenance_path: sp.provenance_path.clone(),
            conditional: sp.conditional,
            frame_bytes_32: m.packed_32.size,
            naive_frame_bytes_32: m.naive_32.size,
            union_saved_bytes_32: saved_32,
            tcb_bytes_32: tcb_bytes(TargetAbi::Mcu32),
            frame_bytes_64: m.packed_64.size,
            naive_frame_bytes_64: m.naive_64.size,
            union_saved_bytes_64: saved_64,
            tcb_bytes_64: tcb_bytes(TargetAbi::Host64),
            hoisted_vars: bp_rep.hoisted_vars,
            stack_vars: bp_rep.stack_vars,
            child_slots: bp_rep.child_slots,
        });
    }

    let mut dual_specialized_fns = Vec::new();
    for f in functions {
        if f.is_yielding {
            let ny_name = format!("{}__noyield", f.name);
            if fn_map.contains_key(ny_name.as_str()) {
                let m = fn_cache.get(&f.name).copied().unwrap_or_default();
                dual_specialized_fns.push(DualSpecializedFnRam {
                    name: f.name.clone(),
                    noyield_name: ny_name,
                    frame_bytes_32: m.packed_32.size,
                    frame_bytes_64: m.packed_64.size,
                });
            }
        }
    }

    let mut prim_rams = Vec::new();
    for p in primitives {
        let (s32, s64) = if p.dead_stripped {
            (0, 0)
        } else {
            (
                ctx.prim_layout(&p.kind, p.dma_safe, TargetAbi::Mcu32).size,
                ctx.prim_layout(&p.kind, p.dma_safe, TargetAbi::Host64).size,
            )
        };
        prim_rams.push(PrimitiveRam {
            name: p.name.clone(),
            kind_display: p.kind.display_name(),
            dma_safe: p.dma_safe,
            holds_sleep_lock: p.kind.holds_dma_sleep_lock(),
            dead_stripped: p.dead_stripped,
            irq_mask_elided: p.irq_mask_elided,
            bytes_32: s32,
            bytes_64: s64,
        });
    }
    for sub in subsystems {
        let ty = MirType::Struct(sub.struct_name.clone());
        let l32 = ctx.type_layout(&ty, TargetAbi::Mcu32);
        let l64 = ctx.type_layout(&ty, TargetAbi::Host64);
        let s32 = if sub.dma_safe {
            align_up(l32.size, l32.align.max(4))
        } else {
            l32.size
        };
        let s64 = if sub.dma_safe {
            align_up(l64.size, l64.align.max(4))
        } else {
            l64.size
        };
        prim_rams.push(PrimitiveRam {
            name: sub.name.clone(),
            kind_display: sub.struct_name.clone(),
            dma_safe: sub.dma_safe,
            holds_sleep_lock: struct_holds_dma_sleep_lock(&sub.struct_name, &ctx.structs),
            dead_stripped: false,
            irq_mask_elided: false,
            bytes_32: s32,
            bytes_64: s64,
        });
    }

    let mut boot_frames = Vec::new();
    for callee in boot_yielding_callees {
        if let Some(m) = fn_cache.get(callee) {
            boot_frames.push((callee.clone(), m.packed_32.size, m.packed_64.size));
        }
    }

    let mut isr_states = Vec::new();
    for vec in irq_vectors {
        if let Some(sel_sym) = &vec.selector_symbol {
            isr_states.push((sel_sym.clone(), "u8 (irq selector)".to_string(), 1, 1));
        }
        for b in &vec.bindings {
            for sv in &b.state_vars {
                let b32 = ctx.type_layout(&sv.ty, TargetAbi::Mcu32).size;
                let b64 = ctx.type_layout(&sv.ty, TargetAbi::Host64).size;
                isr_states.push((sv.storage_symbol.clone(), sv.ty.display_name(), b32, b64));
            }
        }
    }

    let mcu32 = summarize_arch(
        &spawned_tasks,
        &prim_rams,
        &boot_frames,
        &isr_states,
        TargetAbi::Mcu32,
    );
    let host64 = summarize_arch(
        &spawned_tasks,
        &prim_rams,
        &boot_frames,
        &isr_states,
        TargetAbi::Host64,
    );

    RamReport {
        fn_frames,
        blueprint_frames,
        spawned_tasks,
        primitives: prim_rams,
        boot_frames,
        isr_states,
        dual_specialized_fns,
        monomorphized_instances: Vec::new(),
        mcu32,
        host64,
    }
}

fn summarize_arch(
    spawned: &[SpawnedTaskRam],
    prims: &[PrimitiveRam],
    boot_frames: &[(String, usize, usize)],
    isr_states: &[(String, String, usize, usize)],
    abi: TargetAbi,
) -> ArchRamTotals {
    let task_frames_bytes: usize = spawned
        .iter()
        .map(|t| match abi {
            TargetAbi::Mcu32 => t.frame_bytes_32,
            TargetAbi::Host64 => t.frame_bytes_64,
        })
        .sum();
    let naive_task_frames_bytes: usize = spawned
        .iter()
        .map(|t| match abi {
            TargetAbi::Mcu32 => t.naive_frame_bytes_32,
            TargetAbi::Host64 => t.naive_frame_bytes_64,
        })
        .sum();
    let union_bytes_saved = naive_task_frames_bytes.saturating_sub(task_frames_bytes);
    let boot_frames_bytes: usize = boot_frames
        .iter()
        .map(|(_, b32, b64)| match abi {
            TargetAbi::Mcu32 => *b32,
            TargetAbi::Host64 => *b64,
        })
        .sum();
    let tcb_table_bytes = spawned.len() * tcb_bytes(abi);
    let primitives_bytes: usize = prims
        .iter()
        .map(|p| match abi {
            TargetAbi::Mcu32 => p.bytes_32,
            TargetAbi::Host64 => p.bytes_64,
        })
        .sum();
    let isr_state_bytes: usize = isr_states
        .iter()
        .map(|(_, _, b32, b64)| match abi {
            TargetAbi::Mcu32 => *b32,
            TargetAbi::Host64 => *b64,
        })
        .sum();
    let sched = scheduler_bytes(abi);
    let total_sram_bytes = task_frames_bytes
        + boot_frames_bytes
        + tcb_table_bytes
        + primitives_bytes
        + isr_state_bytes
        + sched;
    let naive_total_sram_bytes = naive_task_frames_bytes
        + boot_frames_bytes
        + tcb_table_bytes
        + primitives_bytes
        + isr_state_bytes
        + sched;

    ArchRamTotals {
        task_frames_bytes,
        naive_task_frames_bytes,
        union_bytes_saved,
        boot_frames_bytes,
        tcb_table_bytes,
        primitives_bytes,
        isr_state_bytes,
        scheduler_bytes: sched,
        total_sram_bytes,
        naive_total_sram_bytes,
        heap_bytes: 0,
    }
}

fn resolve_fn_metrics<'a>(
    name: &str,
    fn_map: &BTreeMap<&'a str, &'a MirFunction>,
    ctx: &LayoutCtx<'a>,
    cache: &mut BTreeMap<String, DualFrameMetrics>,
    visiting: &mut BTreeSet<String>,
) -> DualFrameMetrics {
    if let Some(&m) = cache.get(name) {
        return m;
    }
    if !visiting.insert(name.to_string()) {
        return DualFrameMetrics {
            packed_32: SizeAlign::new(2, 2),
            naive_32: SizeAlign::new(2, 2),
            packed_64: SizeAlign::new(2, 2),
            naive_64: SizeAlign::new(2, 2),
            ..Default::default()
        };
    }

    let metrics = if let Some(func) = fn_map.get(name) {
        match &func.body {
            MirFnBody::Direct(_) => DualFrameMetrics::default(),
            MirFnBody::Coroutine { child_slots, .. } => {
                for slot in child_slots {
                    resolve_fn_metrics(&slot.callee, fn_map, ctx, cache, visiting);
                }
                let ret_ty = func.envelope_return_type();
                let ret_opt = if ret_ty.is_void() { None } else { Some(&ret_ty) };
                compute_frame_metrics(&func.vars, child_slots, ret_opt, cache, ctx)
            }
        }
    } else {
        DualFrameMetrics {
            packed_32: SizeAlign::new(2, 2),
            naive_32: SizeAlign::new(2, 2),
            packed_64: SizeAlign::new(2, 2),
            naive_64: SizeAlign::new(2, 2),
            ..Default::default()
        }
    };

    visiting.remove(name);
    cache.insert(name.to_string(), metrics);
    metrics
}

fn compute_frame_metrics(
    vars: &[MirVar],
    child_slots: &[MirChildCallSlot],
    ret_ty: Option<&MirType>,
    fn_cache: &BTreeMap<String, DualFrameMetrics>,
    ctx: &LayoutCtx<'_>,
) -> DualFrameMetrics {
    let (packed_32, naive_32, cu32, cn32) =
        compute_single_abi_frame(vars, child_slots, ret_ty, fn_cache, ctx, TargetAbi::Mcu32);
    let (packed_64, naive_64, _, _) =
        compute_single_abi_frame(vars, child_slots, ret_ty, fn_cache, ctx, TargetAbi::Host64);
    DualFrameMetrics {
        packed_32,
        naive_32,
        child_union_32: cu32,
        child_naive_32: cn32,
        packed_64,
        naive_64,
    }
}

fn compute_single_abi_frame(
    vars: &[MirVar],
    child_slots: &[MirChildCallSlot],
    ret_ty: Option<&MirType>,
    fn_cache: &BTreeMap<String, DualFrameMetrics>,
    ctx: &LayoutCtx<'_>,
    abi: TargetAbi,
) -> (SizeAlign, SizeAlign, usize, usize) {
    // Base fields: `uint16_t _pc;` + hoisted vars + optional `_ret`
    let mut base_fields = vec![SizeAlign::new(2, 2)];
    for v in vars {
        if v.hoisted && !v.ty.is_void() {
            base_fields.push(ctx.type_layout(&v.ty, abi));
        }
    }
    if let Some(rt) = ret_ty {
        if !rt.is_void() {
            base_fields.push(ctx.type_layout(rt, abi));
        }
    }

    let mut packed_fields = base_fields.clone();
    let mut naive_fields = base_fields;

    let mut child_union_bytes = 0usize;
    let mut child_naive_bytes = 0usize;

    if !child_slots.is_empty() {
        let mut groups: BTreeMap<usize, Vec<SizeAlign>> = BTreeMap::new();
        for slot in child_slots {
            let cm = fn_cache.get(&slot.callee).copied().unwrap_or_default();
            let packed_child = match abi {
                TargetAbi::Mcu32 => cm.packed_32,
                TargetAbi::Host64 => cm.packed_64,
            };
            let naive_child = match abi {
                TargetAbi::Mcu32 => cm.naive_32,
                TargetAbi::Host64 => cm.naive_64,
            };
            groups.entry(slot.union_group).or_default().push(packed_child);
            naive_fields.push(naive_child);
            child_naive_bytes += naive_child.size;
        }
        for members in groups.values() {
            let u = ctx.union_layout(members);
            packed_fields.push(u);
            child_union_bytes += u.size;
        }
    }

    let packed = ctx.struct_layout(&packed_fields, false);
    let naive = ctx.struct_layout(&naive_fields, false);
    (packed, naive, child_union_bytes, child_naive_bytes)
}

fn build_frame_report(
    name: &str,
    vars: &[MirVar],
    child_slots: &[MirChildCallSlot],
    metrics: DualFrameMetrics,
    fn_cache: &BTreeMap<String, DualFrameMetrics>,
    ctx: &LayoutCtx<'_>,
) -> FrameLayoutReport {
    let mut hoisted_vars = Vec::new();
    let mut stack_vars = Vec::new();
    for v in vars {
        let b32 = ctx.type_layout(&v.ty, TargetAbi::Mcu32).size;
        let b64 = ctx.type_layout(&v.ty, TargetAbi::Host64).size;
        let info = VarRamInfo {
            name: v.name.clone(),
            ty_display: v.ty.display_name(),
            role: v.role,
            hoisted: v.hoisted,
            in_noyield_scope: v.in_noyield_scope,
            bytes_32: b32,
            bytes_64: b64,
        };
        if v.hoisted {
            hoisted_vars.push(info);
        } else {
            stack_vars.push(info);
        }
    }

    let mut slot_infos = Vec::new();
    for slot in child_slots {
        let cm = fn_cache.get(&slot.callee).copied().unwrap_or_default();
        slot_infos.push(ChildSlotRamInfo {
            slot_name: slot.slot_name.clone(),
            callee: slot.callee.clone(),
            union_group: slot.union_group,
            bytes_32: cm.packed_32.size,
            bytes_64: cm.packed_64.size,
        });
    }

    FrameLayoutReport {
        name: name.to_string(),
        packed_bytes_32: metrics.packed_32.size,
        naive_bytes_32: metrics.naive_32.size,
        union_saved_bytes_32: metrics.naive_32.size.saturating_sub(metrics.packed_32.size),
        child_union_bytes_32: metrics.child_union_32,
        child_naive_bytes_32: metrics.child_naive_32,
        packed_bytes_64: metrics.packed_64.size,
        naive_bytes_64: metrics.naive_64.size,
        union_saved_bytes_64: metrics.naive_64.size.saturating_sub(metrics.packed_64.size),
        hoisted_vars,
        stack_vars,
        child_slots: slot_infos,
    }
}

/// Recompute `module.ram_report` in-place (including `monomorphized_instances` ABI sizes).
pub fn populate_module_ram_report(module: &mut MirModule) {
    let mut rep = compute_ram_report_with_subsystems(
        &module.structs,
        &module.functions,
        &module.task_blueprints,
        &module.spawned_tasks,
        &module.primitives,
        &module.subsystems,
        &module.irq_vectors,
        &module.boot.boot_yielding_callees,
    );
    if !module.monomorphized_instances.is_empty() {
        let ctx = LayoutCtx {
            structs: module
                .structs
                .iter()
                .map(|s| (s.name.as_str(), s))
                .collect(),
            enums: BTreeSet::new(),
        };
        for item in &mut module.monomorphized_instances {
            match item.kind {
                MonomorphizedItemKind::Struct => {
                    let st_ty = MirType::Struct(item.mangled_name.clone());
                    item.size_32 = ctx.type_layout(&st_ty, TargetAbi::Mcu32).size;
                    item.size_64 = ctx.type_layout(&st_ty, TargetAbi::Host64).size;
                }
                MonomorphizedItemKind::Fn => {
                    if let Some(fr) = rep.fn_frames.iter().find(|r| r.name == item.mangled_name) {
                        item.size_32 = fr.packed_bytes_32;
                        item.size_64 = fr.packed_bytes_64;
                    } else {
                        item.size_32 = 0;
                        item.size_64 = 0;
                    }
                }
            }
        }
        rep.monomorphized_instances = module.monomorphized_instances.clone();
    }
    module.ram_report = rep;
}
