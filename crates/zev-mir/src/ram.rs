//! Static Zero-Heap RAM Layout & Budget Calculator (`RamReport`).
//!
//! Computes exact static byte sizes and alignments under both:
//! - **32-bit Bare-Metal MCU ABI** (ARM Cortex-M EABI / RISC-V ILP32: 4-byte pointers & `usize`)
//! - **64-bit Host Simulation ABI** (x86_64 LP64: 8-byte pointers & `usize`)

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::ir::{
    MirBlock, MirChildCallSlot, MirExpr, MirExprKind, MirFmtChunk, MirFnBody, MirFunction,
    MirIntrinsicOp, MirIrqVector, MirModule, MirPlace, MirPrimKind, MirPrimNonBlockingOp,
    MirPrimReadOp, MirPrimTarget, MirPrimViewKind, MirPrimitive, MirPriority, MirSpawnInstance,
    MirStmt, MirStruct, MirSubsystemGlobal, MirTaskBlueprint, MirTerminator, MirType, MirVar,
    MirYieldKind, MonomorphizedItemKind, MonomorphizedMirItem, VarRole,
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
    pub singleton_folded: bool,
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
    pub singleton_folded_vars: Vec<String>,
}

/// Static classification of a task's longest non-yielding run between suspension points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskRunKind {
    /// Only straight-line / branching blocks (plus direct non-looping calls) between yields.
    StraightLine,
    /// Contains at least one bounded `for..in` loop without yields, and no unbounded non-yielding spin loops.
    BoundedFor,
    /// Contains an un-yielded `while`/spin cycle (e.g., `noyield { while (cond) {} }`, `@manual_yield`, or recursion cycle).
    NoyieldSpin,
}

impl TaskRunKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StraightLine => "straight-line",
            Self::BoundedFor => "bounded-for",
            Self::NoyieldSpin => "noyield-spin",
        }
    }
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
    pub max_run_kind: TaskRunKind,
    pub max_run_blocks: usize,
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
    pub singleton_folded_vars: Vec<String>,
    pub singleton_saved_bytes_32: usize,
    pub singleton_saved_bytes_64: usize,
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
    pub singleton_folded: bool,
    pub bytes_32: usize,
    pub bytes_64: usize,
}

/// Architecture-wide SRAM totals.
#[derive(Debug, Clone, Default)]
pub struct ArchRamTotals {
    pub task_frames_bytes: usize,
    pub naive_task_frames_bytes: usize,
    pub union_bytes_saved: usize,
    pub singleton_bytes_saved: usize,
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
        out.push_str(
            "================================================================================\n",
        );
        out.push_str("  STATIC ZERO-HEAP SRAM REPORT (32-bit MCU Target / 64-bit Host Sim)\n");
        out.push_str(
            "================================================================================\n\n",
        );

        out.push_str("1. Spawned Task Instances (Static Coroutine Frames + TCBs)\n");
        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
        out.push_str(&format!(
            "  {:<4} {:<24} {:<10} {:>10} {:>10} {:>11} {:>9}\n",
            "Slot",
            "Instance (Blueprint)",
            "Priority",
            "Frame(32b)",
            "TCB(32b)",
            "UnionSaved",
            "Frame(64b)"
        ));
        out.push_str(
            "  ------------------------------------------------------------------------------\n",
        );
        if self.spawned_tasks.is_empty() {
            out.push_str("  (no spawned tasks)\n");
        } else {
            for t in &self.spawned_tasks {
                let core_tag = match t.core_pin {
                    Some(c) => format!(" @core({c})"),
                    None => String::new(),
                };
                let cond_tag = if t.conditional { " [conditional]" } else { "" };
                let singleton_tag =
                    if !t.singleton_folded_vars.is_empty() || t.singleton_saved_bytes_32 > 0 {
                        " [singleton -> folded]"
                    } else {
                        ""
                    };
                let label = if t.is_main_supervisor {
                    "main (supervisor)".to_string()
                } else {
                    format!("{} ({})", t.instance_name, t.blueprint_name)
                };
                out.push_str(&format!(
                    "  #{:<3} {:<24} {:<10} {:>8} B {:>8} B {:>9} B {:>7} B{}{}{}\n",
                    t.slot_id,
                    label,
                    t.priority.label(),
                    t.frame_bytes_32,
                    t.tcb_bytes_32,
                    t.union_saved_bytes_32,
                    t.frame_bytes_64,
                    core_tag,
                    cond_tag,
                    singleton_tag
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

                if !t.singleton_folded_vars.is_empty() || t.singleton_saved_bytes_32 > 0 {
                    let vars_list = if t.singleton_folded_vars.is_empty() {
                        "child-frame".to_string()
                    } else {
                        t.singleton_folded_vars.join(", ")
                    };
                    out.push_str(&format!(
                        "       singleton folded : {} (0 B in frame, saved {} B 32b / {} B 64b)\n",
                        vars_list, t.singleton_saved_bytes_32, t.singleton_saved_bytes_64
                    ));
                }

                let user_stack: Vec<String> = t
                    .stack_vars
                    .iter()
                    .filter(|v| v.role == VarRole::UserLocal && !v.singleton_folded)
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

                out.push_str(&format!(
                    "       max non-yield run: {} ({} block(s))\n",
                    t.max_run_kind.as_str(),
                    t.max_run_blocks
                ));
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
                let sf_tag = if f.singleton_folded_vars.is_empty() {
                    String::new()
                } else {
                    let hoisted_names: BTreeSet<&str> =
                        f.hoisted_vars.iter().map(|v| v.name.as_str()).collect();
                    let eliminated: Vec<&str> = f
                        .singleton_folded_vars
                        .iter()
                        .filter(|n| !hoisted_names.contains(n.as_str()))
                        .map(String::as_str)
                        .collect();
                    let inlined: Vec<&str> = f
                        .singleton_folded_vars
                        .iter()
                        .filter(|n| hoisted_names.contains(n.as_str()))
                        .map(String::as_str)
                        .collect();
                    let mut parts = Vec::new();
                    if !eliminated.is_empty() {
                        parts.push(format!("{} -> 0 B", eliminated.join(", ")));
                    }
                    if !inlined.is_empty() {
                        parts.push(format!("inlined: {}", inlined.join(", ")));
                    }
                    format!("  [singleton: {}]", parts.join("; "))
                };
                out.push_str(&format!(
                    "  {:<28} {:>6} B (32b) / {:>6} B (64b)  [{}]{}\n",
                    f.name, f.packed_bytes_32, f.packed_bytes_64, vars_desc, sf_tag
                ));
            }
            out.push('\n');
        }

        out.push_str("3. Static OS & DMA Endpoints\n");
        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
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
                if p.singleton_folded {
                    flags.push("singleton -> folded");
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
        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
        out.push_str(&format!(
            "  {:<34} {:>12} {:>14}\n",
            "Component", "32-bit MCU", "64-bit Host"
        ));
        out.push_str(
            "  ------------------------------------------------------------------------------\n",
        );
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
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "Singleton Folding Bytes Saved",
            self.mcu32.singleton_bytes_saved,
            self.host64.singleton_bytes_saved
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
            "Scheduler Task Table (TCBs)", self.mcu32.tcb_table_bytes, self.host64.tcb_table_bytes
        ));
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "OS & DMA Primitives", self.mcu32.primitives_bytes, self.host64.primitives_bytes
        ));
        if self.mcu32.isr_state_bytes > 0 || self.host64.isr_state_bytes > 0 {
            out.push_str(&format!(
                "  {:<34} {:>10} B {:>12} B\n",
                "ISR Static State", self.mcu32.isr_state_bytes, self.host64.isr_state_bytes
            ));
        }
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "Scheduler + Sleep Coordinator",
            self.mcu32.scheduler_bytes,
            self.host64.scheduler_bytes
        ));
        out.push_str(
            "  ------------------------------------------------------------------------------\n",
        );
        out.push_str(&format!(
            "  {:<34} {:>10} B {:>12} B\n",
            "TOTAL STATIC SRAM (Packed)", self.mcu32.total_sram_bytes, self.host64.total_sram_bytes
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

        out.push_str(
            "================================================================================\n",
        );
        out
    }
}

impl fmt::Display for RamReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_table())
    }
}

fn collect_struct_names_in_ty(ty: &MirType, out: &mut Vec<String>) {
    match ty {
        MirType::Struct(name) => out.push(name.clone()),
        MirType::Array { elem, .. } => collect_struct_names_in_ty(elem, out),
        _ => {}
    }
}

/// Collect the set of struct names whose layout requires 32-byte DMA alignment for
/// embedded `pool<T, K>` fields (seeded from `@dma_safe` subsystem globals and
/// `@dma_safe` struct fields, then transitively closed over nested struct fields).
pub fn collect_dma_layout_structs_from(
    structs: &[MirStruct],
    subsystems: &[MirSubsystemGlobal],
) -> BTreeSet<String> {
    let struct_map: BTreeMap<&str, &MirStruct> =
        structs.iter().map(|s| (s.name.as_str(), s)).collect();
    let mut dma_structs = BTreeSet::new();
    let mut queue = std::collections::VecDeque::new();

    for sub in subsystems {
        if sub.dma_safe && dma_structs.insert(sub.struct_name.clone()) {
            queue.push_back(sub.struct_name.clone());
        }
    }
    for st in structs {
        for f in &st.fields {
            if f.dma_safe {
                let mut names = Vec::new();
                collect_struct_names_in_ty(&f.ty, &mut names);
                for name in names {
                    if dma_structs.insert(name.clone()) {
                        queue.push_back(name);
                    }
                }
            }
        }
    }
    while let Some(sname) = queue.pop_front() {
        if let Some(st) = struct_map.get(sname.as_str()) {
            for f in &st.fields {
                let mut names = Vec::new();
                collect_struct_names_in_ty(&f.ty, &mut names);
                for child in names {
                    if dma_structs.insert(child.clone()) {
                        queue.push_back(child);
                    }
                }
            }
        }
    }
    dma_structs
}

/// Collect the set of struct names in `module` that require 32-byte DMA pool field layouts.
pub fn collect_dma_layout_structs(module: &MirModule) -> BTreeSet<String> {
    collect_dma_layout_structs_from(&module.structs, &module.subsystems)
}

struct LayoutCtx<'a> {
    structs: BTreeMap<&'a str, &'a MirStruct>,
    enums: BTreeSet<&'a str>,
    dma_layout_structs: BTreeSet<String>,
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
        self.type_layout_with_dma(ty, false, abi)
    }

    fn type_layout_with_dma(
        &self,
        ty: &MirType,
        parent_dma_safe: bool,
        abi: TargetAbi,
    ) -> SizeAlign {
        let ptr = abi.ptr_bytes();
        match ty {
            MirType::Void => SizeAlign::new(0, 1),
            MirType::Bool | MirType::U8 | MirType::I8 => SizeAlign::new(1, 1),
            MirType::U16 | MirType::I16 => SizeAlign::new(2, 2),
            MirType::U32 | MirType::I32 | MirType::F32 => SizeAlign::new(4, 4),
            MirType::U64 | MirType::I64 | MirType::F64 => SizeAlign::new(8, 8),
            MirType::Usize
            | MirType::Isize
            | MirType::Str
            | MirType::Ptr(_)
            | MirType::MmioPtr(_)
            | MirType::FnPtr { .. } => SizeAlign::new(ptr, ptr),
            MirType::Lease(_) => SizeAlign::new(ptr * 3, ptr),
            MirType::EndianInt { bits, .. } => {
                let bytes = (*bits as usize) / 8;
                // Wire-endian wrapper `struct { uint8_t b[N]; }` has byte alignment 1!
                SizeAlign::new(bytes, 1)
            }
            MirType::Enum(_) => SizeAlign::new(4, 4),
            MirType::Struct(name) => {
                if let Some(st) = self.structs.get(name.as_str()) {
                    let struct_dma =
                        parent_dma_safe || self.dma_layout_structs.contains(name.as_str());
                    let fl: Vec<SizeAlign> = st
                        .fields
                        .iter()
                        .map(|f| {
                            let eff_dma = struct_dma || f.dma_safe;
                            match &f.ty {
                                MirType::Primitive(pk) => self.prim_layout(pk, eff_dma, abi),
                                other => self.type_layout_with_dma(other, eff_dma, abi),
                            }
                        })
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
                let el = self.type_layout_with_dma(elem, parent_dma_safe, abi);
                SizeAlign::new(el.size * len, el.align)
            }
            MirType::Primitive(pk) => self.prim_layout(pk, parent_dma_safe, abi),
            MirType::PrimView(MirPrimViewKind::ByteRing) => SizeAlign::new(ptr * 2, ptr),
            MirType::PrimView(MirPrimViewKind::Inbox { .. }) => SizeAlign::new(ptr * 3, ptr),
            MirType::Result { ok, err } => {
                let mut fields = vec![SizeAlign::new(1, 1)]; // bool is_err
                if let Some(et) = err {
                    if !et.is_void() {
                        fields.push(self.type_layout_with_dma(et, false, abi));
                    }
                }
                if !ok.is_void() {
                    fields.push(self.type_layout_with_dma(ok, false, abi));
                }
                self.struct_layout(&fields, false)
            }
        }
    }

    fn prim_layout(&self, kind: &MirPrimKind, dma_safe: bool, abi: TargetAbi) -> SizeAlign {
        let ptr = abi.ptr_bytes();
        let raw = match kind {
            MirPrimKind::Signal => {
                // `{ u8 flag; u8 pending_boost; u32 waiter_mask; }`
                self.struct_layout(
                    &[
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(4, 4),
                    ],
                    false,
                )
            }
            MirPrimKind::WaitQueue => self.struct_layout(&[SizeAlign::new(4, 4)], false),
            MirPrimKind::Watch { elem } => {
                // `{ T value; u32 gen; u32 waiter_mask; }`
                let el = self.type_layout(elem, abi);
                self.struct_layout(&[el, SizeAlign::new(4, 4), SizeAlign::new(4, 4)], false)
            }
            MirPrimKind::Guarded { elem } => {
                // `{ T value; }`
                let el = self.type_layout(elem, abi);
                self.struct_layout(&[el], false)
            }
            MirPrimKind::Inbox { elem, cap } => {
                // `{ u32 head, tail, count, rx_waiters, tx_waiters; u8 boosts[N]; T items[N]; }`
                let hdr = self.struct_layout(
                    &[
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                    ],
                    false,
                );
                let el = self.type_layout(elem, abi);
                self.struct_layout(
                    &[
                        hdr,
                        SizeAlign::new(*cap, 1),
                        SizeAlign::new(el.size * cap, el.align),
                    ],
                    false,
                )
            }
            MirPrimKind::PingPong { elem, cap } => {
                // `{ u8 write_half, read_half, chunk_ready, sleep_lock_bit; u32 waiter_mask; T buf0[N]; T buf1[N]; }`
                let el = self.type_layout(elem, abi);
                self.struct_layout(
                    &[
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(el.size * cap * 2, el.align),
                    ],
                    false,
                )
            }
            MirPrimKind::Pool { elem, cap } => {
                // Header `{ u32 free_mask; u32 waiter_mask; ptr base_ptr; u8 dma_safe; u8 cap; }`
                let hdr = self.struct_layout(
                    &[
                        SizeAlign::new(4, 4),
                        SizeAlign::new(4, 4),
                        SizeAlign::new(ptr, ptr),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                    ],
                    false,
                );
                let el = self.type_layout(elem, abi);
                let elem_sz = el.size.max(1);
                let elem_al = el.align.max(1);
                if dma_safe {
                    let slot_al = elem_al.max(32);
                    let slot_stride = align_up(elem_sz, slot_al);
                    let hdr_aligned = align_up(hdr.size, slot_al);
                    return SizeAlign::new(hdr_aligned + slot_stride * *cap, slot_al);
                }
                self.struct_layout(&[hdr, SizeAlign::new(elem_sz * *cap, elem_al)], false)
            }
            MirPrimKind::DmaSlot { .. } => {
                // `{ ptr data; ptr pool_hdr; u8 slot_idx; u8 occupied; }`
                self.struct_layout(
                    &[
                        SizeAlign::new(ptr, ptr),
                        SizeAlign::new(ptr, ptr),
                        SizeAlign::new(1, 1),
                        SizeAlign::new(1, 1),
                    ],
                    false,
                )
            }
            MirPrimKind::ByteRing { cap } => {
                // `{ u32 head, tail, count, split_end, grant_off, grant_cap, waiter_mask; u8 boost; u8 buf[N]; }`
                let hdr = self.struct_layout(
                    &[
                        SizeAlign::new(4, 4),
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
        TargetAbi::Mcu32 => 76,
        TargetAbi::Host64 => 88,
    }
}

fn struct_holds_dma_sleep_lock(
    name: &str,
    parent_dma_safe: bool,
    structs: &BTreeMap<&str, &MirStruct>,
) -> bool {
    let Some(st) = structs.get(name) else {
        return false;
    };
    st.fields.iter().any(|f| {
        let field_dma_safe = parent_dma_safe || f.dma_safe;
        match &f.ty {
            MirType::Primitive(pk) => pk.holds_dma_sleep_lock(field_dma_safe),
            MirType::Struct(sub) => struct_holds_dma_sleep_lock(sub, field_dma_safe, structs),
            _ => false,
        }
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

#[allow(clippy::too_many_arguments)]
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
        dma_layout_structs: collect_dma_layout_structs_from(structs, subsystems),
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

    let mut run_fn_cache: BTreeMap<String, (TaskRunKind, usize)> = BTreeMap::new();
    let mut run_visiting: BTreeSet<String> = BTreeSet::new();
    let mut bp_run_cache: BTreeMap<String, (TaskRunKind, usize)> = BTreeMap::new();
    for bp in blueprints {
        let stats = analyze_coro_blocks(&bp.blocks, &fn_map, &mut run_fn_cache, &mut run_visiting);
        bp_run_cache.insert(bp.name.clone(), stats);
    }

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
                        singleton_folded_vars: vec![],
                    },
                    DualFrameMetrics::default(),
                )
            });

        let (max_run_kind, max_run_blocks) = bp_run_cache
            .get(&sp.blueprint_name)
            .copied()
            .unwrap_or((TaskRunKind::StraightLine, 0));

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
            max_run_kind,
            max_run_blocks,
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
            singleton_folded_vars: Vec::new(),
            singleton_saved_bytes_32: 0,
            singleton_saved_bytes_64: 0,
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
            holds_sleep_lock: p.kind.holds_dma_sleep_lock(p.dma_safe),
            dead_stripped: p.dead_stripped,
            irq_mask_elided: p.irq_mask_elided,
            singleton_folded: false,
            bytes_32: s32,
            bytes_64: s64,
        });
    }
    for sub in subsystems {
        let ty = MirType::Struct(sub.struct_name.clone());
        let l32 = ctx.type_layout_with_dma(&ty, sub.dma_safe, TargetAbi::Mcu32);
        let l64 = ctx.type_layout_with_dma(&ty, sub.dma_safe, TargetAbi::Host64);
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
            holds_sleep_lock: struct_holds_dma_sleep_lock(
                &sub.struct_name,
                sub.dma_safe,
                &ctx.structs,
            ),
            dead_stripped: false,
            irq_mask_elided: false,
            singleton_folded: false,
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
        singleton_bytes_saved: 0,
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
                let ret_opt = if ret_ty.is_void() {
                    None
                } else {
                    Some(&ret_ty)
                };
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
            groups
                .entry(slot.union_group)
                .or_default()
                .push(packed_child);
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
            singleton_folded: false,
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
        singleton_folded_vars: Vec::new(),
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
    rep.mcu32.singleton_bytes_saved = module.visibility.singleton_bytes_saved_32;
    rep.host64.singleton_bytes_saved = module.visibility.singleton_bytes_saved_64;
    for t in &mut rep.spawned_tasks {
        if let Some(vars) = module.visibility.folded_task_vars.get(&t.blueprint_name) {
            t.singleton_folded_vars = vars.clone();
        }
        if let Some(&(s32, s64)) = module
            .visibility
            .singleton_task_bytes_saved
            .get(&t.instance_name)
        {
            t.singleton_saved_bytes_32 = s32;
            t.singleton_saved_bytes_64 = s64;
        }
        if let Some(folded_ids) = module.visibility.folded_task_var_ids.get(&t.blueprint_name) {
            if let Some(bp) = module
                .task_blueprints
                .iter()
                .find(|b| b.name == t.blueprint_name)
            {
                for (sv, mv) in t
                    .stack_vars
                    .iter_mut()
                    .zip(bp.vars.iter().filter(|v| !v.hoisted))
                {
                    if folded_ids.contains(&mv.id.0) {
                        sv.singleton_folded = true;
                    }
                }
            }
        }
    }
    for f in &mut rep.fn_frames {
        if let Some(vars) = module.visibility.folded_fn_vars.get(&f.name) {
            f.singleton_folded_vars = vars.clone();
        }
    }
    for f in &mut rep.blueprint_frames {
        if let Some(vars) = module.visibility.folded_task_vars.get(&f.name) {
            f.singleton_folded_vars = vars.clone();
        }
    }
    for p in &mut rep.primitives {
        if module.visibility.singleton_subsystems.contains(&p.name) {
            p.singleton_folded = true;
        }
    }
    if !module.monomorphized_instances.is_empty() {
        let ctx = LayoutCtx {
            structs: module
                .structs
                .iter()
                .map(|s| (s.name.as_str(), s))
                .collect(),
            enums: BTreeSet::new(),
            dma_layout_structs: collect_dma_layout_structs(module),
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

fn analyze_fn_run(
    fn_name: &str,
    fn_map: &BTreeMap<&str, &MirFunction>,
    cache: &mut BTreeMap<String, (TaskRunKind, usize)>,
    visiting: &mut BTreeSet<String>,
) -> (TaskRunKind, usize) {
    if let Some(&cached) = cache.get(fn_name) {
        return cached;
    }
    if !visiting.insert(fn_name.to_string()) {
        return (TaskRunKind::NoyieldSpin, 1);
    }
    let res = if let Some(func) = fn_map.get(fn_name) {
        match &func.body {
            MirFnBody::Direct(stmts) => analyze_direct_stmts(stmts, fn_map, cache, visiting),
            MirFnBody::Coroutine { blocks, .. } => {
                analyze_coro_blocks(blocks, fn_map, cache, visiting)
            }
        }
    } else {
        (TaskRunKind::StraightLine, 1)
    };
    visiting.remove(fn_name);
    cache.insert(fn_name.to_string(), res);
    res
}

fn fold_place_run(
    place: &MirPlace,
    fn_map: &BTreeMap<&str, &MirFunction>,
    cache: &mut BTreeMap<String, (TaskRunKind, usize)>,
    visiting: &mut BTreeSet<String>,
    kind: &mut TaskRunKind,
    extra_blocks: &mut usize,
) {
    match place {
        MirPlace::Var(_) => {}
        MirPlace::Deref(expr, _) => {
            fold_expr_run(expr, fn_map, cache, visiting, kind, extra_blocks);
        }
        MirPlace::Field { base, .. } => {
            fold_place_run(base, fn_map, cache, visiting, kind, extra_blocks);
        }
        MirPlace::Index { base, index, .. } => {
            fold_place_run(base, fn_map, cache, visiting, kind, extra_blocks);
            fold_expr_run(index, fn_map, cache, visiting, kind, extra_blocks);
        }
    }
}

fn fold_prim_target_run(
    target: &MirPrimTarget,
    fn_map: &BTreeMap<&str, &MirFunction>,
    cache: &mut BTreeMap<String, (TaskRunKind, usize)>,
    visiting: &mut BTreeSet<String>,
    kind: &mut TaskRunKind,
    extra_blocks: &mut usize,
) {
    if let MirPrimTarget::PtrExpr(expr) = target {
        fold_expr_run(expr, fn_map, cache, visiting, kind, extra_blocks);
    }
}

fn fold_fmt_chunks_run(
    chunks: &[MirFmtChunk],
    fn_map: &BTreeMap<&str, &MirFunction>,
    cache: &mut BTreeMap<String, (TaskRunKind, usize)>,
    visiting: &mut BTreeSet<String>,
    kind: &mut TaskRunKind,
    extra_blocks: &mut usize,
) {
    for ch in chunks {
        match ch {
            MirFmtChunk::Literal(_) => {}
            MirFmtChunk::Int { value, .. }
            | MirFmtChunk::Float { value, .. }
            | MirFmtChunk::Bool(value)
            | MirFmtChunk::Str(value)
            | MirFmtChunk::Bytes(value)
            | MirFmtChunk::EnumName { value, .. } => {
                fold_expr_run(value, fn_map, cache, visiting, kind, extra_blocks);
            }
            MirFmtChunk::DebugSlice { slice, .. } => {
                fold_expr_run(slice, fn_map, cache, visiting, kind, extra_blocks);
            }
        }
    }
}

fn fold_expr_run(
    expr: &MirExpr,
    fn_map: &BTreeMap<&str, &MirFunction>,
    cache: &mut BTreeMap<String, (TaskRunKind, usize)>,
    visiting: &mut BTreeSet<String>,
    kind: &mut TaskRunKind,
    extra_blocks: &mut usize,
) {
    match &expr.kind {
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::StringLit(_)
        | MirExprKind::TicksLit(_)
        | MirExprKind::EnumVariant { .. }
        | MirExprKind::ConstRef(_)
        | MirExprKind::GlobalPrimAddr(_)
        | MirExprKind::SubsystemAddr(_)
        | MirExprKind::MmioAddr(_)
        | MirExprKind::FnAddr(_)
        | MirExprKind::Var(_)
        | MirExprKind::ArrayLen(_)
        | MirExprKind::ResultIsErr(_)
        | MirExprKind::ResultErr(_)
        | MirExprKind::ResultVal(_)
        | MirExprKind::NowTicks
        | MirExprKind::SysQuery(_) => {}
        MirExprKind::MakePrimView { ptr, .. } => {
            fold_expr_run(ptr, fn_map, cache, visiting, kind, extra_blocks);
        }
        MirExprKind::MmioRead { base, index, .. } => {
            fold_expr_run(base, fn_map, cache, visiting, kind, extra_blocks);
            if let Some(idx) = index {
                fold_expr_run(idx, fn_map, cache, visiting, kind, extra_blocks);
            }
        }
        MirExprKind::Unary { expr: inner, .. }
        | MirExprKind::Cast { expr: inner, .. }
        | MirExprKind::FieldAccess { base: inner, .. }
        | MirExprKind::SliceLen(inner)
        | MirExprKind::FromBytes { slice: inner, .. }
        | MirExprKind::LeaseGet { lease: inner, .. }
        | MirExprKind::LeaseAddr { lease: inner, .. } => {
            fold_expr_run(inner, fn_map, cache, visiting, kind, extra_blocks);
        }
        MirExprKind::AddrOfPlace(place)
        | MirExprKind::ArrayToSlice {
            array_place: place, ..
        } => {
            fold_place_run(place, fn_map, cache, visiting, kind, extra_blocks);
        }
        MirExprKind::Binary { lhs, rhs, .. }
        | MirExprKind::Index {
            base: lhs,
            index: rhs,
            ..
        } => {
            fold_expr_run(lhs, fn_map, cache, visiting, kind, extra_blocks);
            fold_expr_run(rhs, fn_map, cache, visiting, kind, extra_blocks);
        }
        MirExprKind::SubSlice {
            base, start, end, ..
        } => {
            fold_expr_run(base, fn_map, cache, visiting, kind, extra_blocks);
            fold_expr_run(start, fn_map, cache, visiting, kind, extra_blocks);
            fold_expr_run(end, fn_map, cache, visiting, kind, extra_blocks);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, val, _) in fields {
                fold_expr_run(val, fn_map, cache, visiting, kind, extra_blocks);
            }
        }
        MirExprKind::ArrayLit { elems, .. } => {
            for elem in elems {
                fold_expr_run(elem, fn_map, cache, visiting, kind, extra_blocks);
            }
        }
        MirExprKind::PrimRead(op) => match op.as_ref() {
            MirPrimReadOp::WatchRead { target, .. }
            | MirPrimReadOp::WatchGen { target, .. }
            | MirPrimReadOp::GuardedGet { target, .. }
            | MirPrimReadOp::InboxTryRecv { target, .. }
            | MirPrimReadOp::InboxLen { target, .. }
            | MirPrimReadOp::InboxIsEmpty { target, .. }
            | MirPrimReadOp::InboxIsFull { target, .. }
            | MirPrimReadOp::SignalIsSet { target }
            | MirPrimReadOp::SignalTryWait { target }
            | MirPrimReadOp::WaitQueueWakeOne { target, .. }
            | MirPrimReadOp::PingPongDmaBuf { target, .. }
            | MirPrimReadOp::PingPongBaseAddr { target, .. }
            | MirPrimReadOp::PoolTryAlloc { target, .. }
            | MirPrimReadOp::PoolAvail { target, .. }
            | MirPrimReadOp::PoolInUse { target, .. }
            | MirPrimReadOp::PoolIsExhausted { target, .. }
            | MirPrimReadOp::PoolCap { target, .. }
            | MirPrimReadOp::DmaSlotTake { target, .. }
            | MirPrimReadOp::DmaSlotRelease { target, .. }
            | MirPrimReadOp::DmaSlotIsBusy { target, .. }
            | MirPrimReadOp::DmaSlotIsEmpty { target, .. }
            | MirPrimReadOp::DmaSlotAddr { target, .. }
            | MirPrimReadOp::ByteRingReadSlice { target, .. }
            | MirPrimReadOp::ByteRingTryPopByte { target, .. }
            | MirPrimReadOp::ByteRingLen { target, .. }
            | MirPrimReadOp::ByteRingFreeSpace { target, .. }
            | MirPrimReadOp::ByteRingIsEmpty { target, .. }
            | MirPrimReadOp::ByteRingIsFull { target, .. } => {
                fold_prim_target_run(target, fn_map, cache, visiting, kind, extra_blocks);
            }
            MirPrimReadOp::InboxTrySend { target, value, .. }
            | MirPrimReadOp::DmaSlotPark {
                target,
                lease: value,
                ..
            }
            | MirPrimReadOp::ByteRingPush {
                target, src: value, ..
            }
            | MirPrimReadOp::ByteRingPushByte {
                target,
                byte: value,
                ..
            }
            | MirPrimReadOp::ByteRingGrantWrite {
                target, len: value, ..
            }
            | MirPrimReadOp::ByteRingPop {
                target, dst: value, ..
            } => {
                fold_prim_target_run(target, fn_map, cache, visiting, kind, extra_blocks);
                fold_expr_run(value, fn_map, cache, visiting, kind, extra_blocks);
            }
            MirPrimReadOp::InboxViewTryRecv { view, .. }
            | MirPrimReadOp::InboxViewLen { view, .. }
            | MirPrimReadOp::InboxViewCap { view, .. }
            | MirPrimReadOp::InboxViewIsEmpty { view, .. }
            | MirPrimReadOp::InboxViewIsFull { view, .. }
            | MirPrimReadOp::ByteRingViewReadSlice { view }
            | MirPrimReadOp::ByteRingViewTryPopByte { view }
            | MirPrimReadOp::ByteRingViewLen { view }
            | MirPrimReadOp::ByteRingViewCap { view }
            | MirPrimReadOp::ByteRingViewFreeSpace { view }
            | MirPrimReadOp::ByteRingViewIsEmpty { view }
            | MirPrimReadOp::ByteRingViewIsFull { view } => {
                fold_expr_run(view, fn_map, cache, visiting, kind, extra_blocks);
            }
            MirPrimReadOp::InboxViewTrySend { view, value, .. }
            | MirPrimReadOp::ByteRingViewPush {
                view, src: value, ..
            }
            | MirPrimReadOp::ByteRingViewPushByte {
                view, byte: value, ..
            }
            | MirPrimReadOp::ByteRingViewGrantWrite { view, len: value }
            | MirPrimReadOp::ByteRingViewPop { view, dst: value } => {
                fold_expr_run(view, fn_map, cache, visiting, kind, extra_blocks);
                fold_expr_run(value, fn_map, cache, visiting, kind, extra_blocks);
            }
        },
        MirExprKind::CallDirect { callee, args, .. } => {
            for a in args {
                fold_expr_run(a, fn_map, cache, visiting, kind, extra_blocks);
            }
            let (ck, cb) = analyze_fn_run(callee, fn_map, cache, visiting);
            *kind = (*kind).max(ck);
            *extra_blocks += cb;
        }
        MirExprKind::RomCallU32 { addr, args, .. } => {
            fold_expr_run(addr, fn_map, cache, visiting, kind, extra_blocks);
            for a in args {
                fold_expr_run(a, fn_map, cache, visiting, kind, extra_blocks);
            }
        }
    }
}

fn analyze_direct_stmts(
    stmts: &[MirStmt],
    fn_map: &BTreeMap<&str, &MirFunction>,
    cache: &mut BTreeMap<String, (TaskRunKind, usize)>,
    visiting: &mut BTreeSet<String>,
) -> (TaskRunKind, usize) {
    let mut kind = TaskRunKind::StraightLine;
    let mut blocks = 1usize;

    for s in stmts {
        match s {
            MirStmt::Assign { target, value, .. } => {
                fold_place_run(target, fn_map, cache, visiting, &mut kind, &mut blocks);
                fold_expr_run(value, fn_map, cache, visiting, &mut kind, &mut blocks);
            }
            MirStmt::Expr(expr) | MirStmt::Intrinsic(MirIntrinsicOp::UseSystick { hz: expr }) => {
                fold_expr_run(expr, fn_map, cache, visiting, &mut kind, &mut blocks);
            }
            MirStmt::PrimOp(op) => match op {
                MirPrimNonBlockingOp::SignalNotify { target, .. }
                | MirPrimNonBlockingOp::SignalClear { target }
                | MirPrimNonBlockingOp::WaitQueuePrepare { target }
                | MirPrimNonBlockingOp::WaitQueueCancel { target }
                | MirPrimNonBlockingOp::WaitQueueWakeAll { target, .. }
                | MirPrimNonBlockingOp::PingPongFlip { target, .. }
                | MirPrimNonBlockingOp::PingPongSetActive { target, .. }
                | MirPrimNonBlockingOp::ByteRingClear { target, .. } => {
                    fold_prim_target_run(target, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
                MirPrimNonBlockingOp::WatchPublish { target, value, .. }
                | MirPrimNonBlockingOp::GuardedSet { target, value, .. }
                | MirPrimNonBlockingOp::InboxPost { target, value, .. }
                | MirPrimNonBlockingOp::PingPongCompleteHalf {
                    target,
                    half: value,
                    ..
                }
                | MirPrimNonBlockingOp::ByteRingConsume {
                    target, len: value, ..
                }
                | MirPrimNonBlockingOp::ByteRingCommitWrite {
                    target, len: value, ..
                } => {
                    fold_prim_target_run(target, fn_map, cache, visiting, &mut kind, &mut blocks);
                    fold_expr_run(value, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
                MirPrimNonBlockingOp::LeaseRelease { value, .. } => {
                    fold_expr_run(value, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
                MirPrimNonBlockingOp::LeaseNullify { target, .. } => {
                    fold_place_run(target, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
                MirPrimNonBlockingOp::InboxViewPost { view, value, .. }
                | MirPrimNonBlockingOp::ByteRingViewConsume { view, len: value }
                | MirPrimNonBlockingOp::ByteRingViewCommitWrite {
                    view, len: value, ..
                } => {
                    fold_expr_run(view, fn_map, cache, visiting, &mut kind, &mut blocks);
                    fold_expr_run(value, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
                MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                    fold_expr_run(view, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
            },
            MirStmt::Intrinsic(MirIntrinsicOp::SetPriority { .. })
            | MirStmt::EnterCritical
            | MirStmt::ExitCritical
            | MirStmt::EndInboxTurn
            | MirStmt::Break
            | MirStmt::Continue
            | MirStmt::SetIrqSelector { .. }
            | MirStmt::BoundedForLatch
            | MirStmt::ZeroTaskLocals => {}
            MirStmt::Intrinsic(MirIntrinsicOp::SysOp { args, .. })
            | MirStmt::SpawnActivate { args, .. } => {
                for a in args {
                    fold_expr_run(a, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
            }
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                fold_expr_run(cond, fn_map, cache, visiting, &mut kind, &mut blocks);
                let (tk, tb) = analyze_direct_stmts(then_body, fn_map, cache, visiting);
                let (ek, eb) = if else_body.is_empty() {
                    (TaskRunKind::StraightLine, 0)
                } else {
                    analyze_direct_stmts(else_body, fn_map, cache, visiting)
                };
                kind = kind.max(tk).max(ek);
                blocks += tb.max(eb);
            }
            MirStmt::While { cond, body } => {
                fold_expr_run(cond, fn_map, cache, visiting, &mut kind, &mut blocks);
                let is_bounded_for = body.iter().any(|st| matches!(st, MirStmt::BoundedForLatch));
                let loop_kind = if is_bounded_for {
                    TaskRunKind::BoundedFor
                } else {
                    TaskRunKind::NoyieldSpin
                };
                let (bk, bb) = analyze_direct_stmts(body, fn_map, cache, visiting);
                kind = kind.max(loop_kind).max(bk);
                blocks += 1 + bb;
            }
            MirStmt::Switch {
                discr,
                arms,
                default,
            } => {
                fold_expr_run(discr, fn_map, cache, visiting, &mut kind, &mut blocks);
                let mut max_arm_blocks = 0usize;
                for (_, arm_body) in arms {
                    let (ak, ab) = analyze_direct_stmts(arm_body, fn_map, cache, visiting);
                    kind = kind.max(ak);
                    max_arm_blocks = max_arm_blocks.max(ab);
                }
                if !default.is_empty() {
                    let (dk, db) = analyze_direct_stmts(default, fn_map, cache, visiting);
                    kind = kind.max(dk);
                    max_arm_blocks = max_arm_blocks.max(db);
                }
                blocks += max_arm_blocks;
            }
            MirStmt::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    fold_expr_run(expr, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
            }
            MirStmt::Error { reason, partial } => {
                if let Some(expr) = reason {
                    fold_expr_run(expr, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
                if let Some(expr) = partial {
                    fold_expr_run(expr, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
            }
            MirStmt::PrintFmt { chunks, .. } => {
                fold_fmt_chunks_run(chunks, fn_map, cache, visiting, &mut kind, &mut blocks);
            }
            MirStmt::FmtBuf { buf, chunks, .. } => {
                fold_expr_run(buf, fn_map, cache, visiting, &mut kind, &mut blocks);
                fold_fmt_chunks_run(chunks, fn_map, cache, visiting, &mut kind, &mut blocks);
            }
            MirStmt::FmtRing { ring, chunks, .. } => {
                fold_expr_run(ring, fn_map, cache, visiting, &mut kind, &mut blocks);
                fold_fmt_chunks_run(chunks, fn_map, cache, visiting, &mut kind, &mut blocks);
            }
            MirStmt::MmioWrite {
                base, index, value, ..
            } => {
                fold_expr_run(base, fn_map, cache, visiting, &mut kind, &mut blocks);
                if let Some(idx) = index {
                    fold_expr_run(idx, fn_map, cache, visiting, &mut kind, &mut blocks);
                }
                fold_expr_run(value, fn_map, cache, visiting, &mut kind, &mut blocks);
            }
        }
    }

    (kind, blocks)
}

fn tarjan_sccs(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct State {
        index: usize,
        indices: Vec<Option<usize>>,
        lowlink: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        sccs: Vec<Vec<usize>>,
    }

    fn strongconnect(v: usize, adj: &[Vec<usize>], st: &mut State) {
        st.indices[v] = Some(st.index);
        st.lowlink[v] = st.index;
        st.index += 1;
        st.stack.push(v);
        st.on_stack[v] = true;

        for &w in &adj[v] {
            if st.indices[w].is_none() {
                strongconnect(w, adj, st);
                st.lowlink[v] = st.lowlink[v].min(st.lowlink[w]);
            } else if st.on_stack[w] {
                if let Some(w_idx) = st.indices[w] {
                    st.lowlink[v] = st.lowlink[v].min(w_idx);
                }
            }
        }

        if Some(st.lowlink[v]) == st.indices[v] {
            let mut scc = Vec::new();
            while let Some(w) = st.stack.pop() {
                st.on_stack[w] = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            st.sccs.push(scc);
        }
    }

    let n = adj.len();
    let mut st = State {
        index: 0,
        indices: vec![None; n],
        lowlink: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        sccs: Vec::new(),
    };
    for v in 0..n {
        if st.indices[v].is_none() {
            strongconnect(v, adj, &mut st);
        }
    }
    st.sccs
}

fn sccs_have_cycle(sccs: &[Vec<usize>], adj: &[Vec<usize>]) -> bool {
    sccs.iter()
        .any(|scc| scc.len() > 1 || scc.first().is_some_and(|&u| adj[u].contains(&u)))
}

fn analyze_coro_blocks(
    blocks: &[MirBlock],
    fn_map: &BTreeMap<&str, &MirFunction>,
    cache: &mut BTreeMap<String, (TaskRunKind, usize)>,
    visiting: &mut BTreeSet<String>,
) -> (TaskRunKind, usize) {
    if blocks.is_empty() {
        return (TaskRunKind::StraightLine, 0);
    }

    let n = blocks.len();
    let mut id_to_idx: BTreeMap<u16, usize> = BTreeMap::new();
    for (idx, b) in blocks.iter().enumerate() {
        id_to_idx.insert(b.id.0, idx);
    }

    let mut block_kinds = Vec::with_capacity(n);
    let mut block_weights = Vec::with_capacity(n);
    let mut adj = vec![Vec::new(); n];
    let mut adj_free = vec![Vec::new(); n];

    for (idx, b) in blocks.iter().enumerate() {
        let (mut k, mut w) = analyze_direct_stmts(&b.stmts, fn_map, cache, visiting);
        match &b.terminator {
            MirTerminator::Goto(target) => {
                if let Some(&t_idx) = id_to_idx.get(&target.0) {
                    adj[idx].push(t_idx);
                }
            }
            MirTerminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                fold_expr_run(cond, fn_map, cache, visiting, &mut k, &mut w);
                if let Some(&t_idx) = id_to_idx.get(&then_bb.0) {
                    adj[idx].push(t_idx);
                }
                if let Some(&e_idx) = id_to_idx.get(&else_bb.0) {
                    adj[idx].push(e_idx);
                }
            }
            MirTerminator::Switch {
                discr,
                arms,
                default_bb,
            } => {
                fold_expr_run(discr, fn_map, cache, visiting, &mut k, &mut w);
                for (_, arm_bb) in arms {
                    if let Some(&a_idx) = id_to_idx.get(&arm_bb.0) {
                        adj[idx].push(a_idx);
                    }
                }
                if let Some(&d_idx) = id_to_idx.get(&default_bb.0) {
                    adj[idx].push(d_idx);
                }
            }
            MirTerminator::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    fold_expr_run(expr, fn_map, cache, visiting, &mut k, &mut w);
                }
            }
            MirTerminator::Error { reason, partial } => {
                if let Some(expr) = reason {
                    fold_expr_run(expr, fn_map, cache, visiting, &mut k, &mut w);
                }
                if let Some(expr) = partial {
                    fold_expr_run(expr, fn_map, cache, visiting, &mut k, &mut w);
                }
            }
            MirTerminator::Yield(yk) => match yk {
                MirYieldKind::EveryLatch { .. } | MirYieldKind::ShareCheckpoint { .. } => {}
                MirYieldKind::DelayPoll { .. } => {}
                MirYieldKind::Wait { arms, .. } => {
                    for arm in arms {
                        match &arm.op {
                            crate::ir::MirWaitArmOp::InboxRecv { target, .. }
                            | crate::ir::MirWaitArmOp::InboxSend { target, .. }
                            | crate::ir::MirWaitArmOp::SignalWait { target, .. }
                            | crate::ir::MirWaitArmOp::WaitQueueCommit { target, .. }
                            | crate::ir::MirWaitArmOp::WatchWaitChange { target, .. }
                            | crate::ir::MirWaitArmOp::PingPongWaitChunk { target, .. }
                            | crate::ir::MirWaitArmOp::PoolAlloc { target, .. }
                            | crate::ir::MirWaitArmOp::ByteRingWaitRead { target, .. } => {
                                fold_prim_target_run(
                                    target, fn_map, cache, visiting, &mut k, &mut w,
                                );
                            }
                            crate::ir::MirWaitArmOp::InboxViewRecv { view, .. }
                            | crate::ir::MirWaitArmOp::InboxViewSend { view, .. }
                            | crate::ir::MirWaitArmOp::ByteRingViewWaitRead { view, .. } => {
                                fold_expr_run(view, fn_map, cache, visiting, &mut k, &mut w);
                            }
                        }
                    }
                }
                MirYieldKind::CallYielding { callee, args, .. } => {
                    for a in args {
                        fold_expr_run(a, fn_map, cache, visiting, &mut k, &mut w);
                    }
                    let (ck, cb) = analyze_fn_run(callee, fn_map, cache, visiting);
                    k = k.max(ck);
                    w += cb;
                }
                MirYieldKind::PollChild { callee, .. } => {
                    let (ck, cb) = analyze_fn_run(callee, fn_map, cache, visiting);
                    k = k.max(ck);
                    w += cb;
                }
            },
        }
        let has_for_latch = b
            .stmts
            .iter()
            .any(|st| matches!(st, MirStmt::BoundedForLatch));
        if !has_for_latch {
            adj_free[idx] = adj[idx].clone();
        }
        block_kinds.push(k);
        block_weights.push(w);
    }

    let sccs = tarjan_sccs(&adj);
    let cycle_kind = if !sccs_have_cycle(&sccs, &adj) {
        TaskRunKind::StraightLine
    } else {
        let free_sccs = tarjan_sccs(&adj_free);
        if !sccs_have_cycle(&free_sccs, &adj_free) {
            TaskRunKind::BoundedFor
        } else {
            TaskRunKind::NoyieldSpin
        }
    };

    let mut kind = cycle_kind;
    for bk in &block_kinds {
        kind = kind.max(*bk);
    }

    let mut scc_of = vec![0usize; n];
    for (scc_id, scc) in sccs.iter().enumerate() {
        for &u in scc {
            scc_of[u] = scc_id;
        }
    }

    let mut scc_dp = vec![0usize; sccs.len()];
    for (scc_id, scc) in sccs.iter().enumerate() {
        let own_weight: usize = scc.iter().map(|&u| block_weights[u]).sum();
        let mut best_succ = 0usize;
        for &u in scc {
            for &v in &adj[u] {
                let sv = scc_of[v];
                if sv != scc_id {
                    best_succ = best_succ.max(scc_dp[sv]);
                }
            }
        }
        scc_dp[scc_id] = own_weight + best_succ;
    }

    let max_blocks = scc_dp.into_iter().max().unwrap_or(0);
    (kind, max_blocks)
}
