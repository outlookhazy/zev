//! Hardware Single-Stack + `.data`/`.bss` High-Water Mark Proof (`stack_sizes`).
//!
//! Computes exact worst-case bare-metal SRAM consumption across pure ISA targets:
//! - **ARM Cortex-M (`arm-cortex-m`)**:
//!   `/bin/opt` -> `/bin/llc --stack-size-section` -> `/bin/ld.lld` -> `/bin/llvm-size`
//!   + `/bin/llvm-readobj --stack-sizes` + `/bin/llvm-objdump -d` + `32 B` NVIC HW exception frame.
//! - **RISC-V (`riscv32`)**:
//!   `/bin/opt` -> `/bin/llc --stack-size-section` -> `/bin/ld.lld` -> `/bin/llvm-size`
//!   + `/bin/llvm-readobj --stack-sizes` + `/bin/llvm-objdump -d` + `64 B` trap context frame.
//! - **AVR (`avr`)**:
//!   `/bin/opt` -> `/bin/llc` -> `/bin/ld.lld` -> `/bin/llvm-size`
//!   + `/bin/llvm-objdump -d` AVR prologue/call-edge analysis (`push` depth + `2 * rcall .+0`
//!   + `sbiw r28, K` / `subi+sbci` SP subtracts + `2 B` return-PC per call + `2 B` vector push).
//!
//! Stack-bounded ROM ABI wrappers (`@__syn_rom_call_*_sb<N>` / `@__syn_rom_noreturn_*`) supply
//! compile-time verified constant bounds directly via their `_sb<N>` suffix.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use zev_mir::{
    MirBakedArg, MirBootStep, MirErrorSpec, MirExpr, MirExprKind, MirFmtChunk, MirFnBody,
    MirIntrinsicOp, MirModule, MirPlace, MirPrimNonBlockingOp, MirPrimReadOp, MirPrimTarget,
    MirStmt, MirTerminator, MirType, MirVar, MirWaitArmOp, MirYieldKind,
};

use crate::arm_cortex_m::sanitize_ident;
use crate::riscv32::assign_rv_irq_routes_for_spec;
use crate::target_spec::{CpuIsa, TargetSpec};
use crate::toolchain::{
    inspect_elf_sizes, link_obj_to_elf, optimize_ll_with_opt, require_llvm_tool,
    synthesize_target_linker_script, CodegenError, CodegenOptions,
};

fn parse_rom_call_stack_bound(sym: &str) -> Option<u64> {
    let s = sym.trim_start_matches('@');
    if !s.starts_with("__syn_rom_call_") {
        return None;
    }
    let (_, tail) = s.rsplit_once("_sb")?;
    tail.parse::<u64>().ok()
}

fn is_exempt_indirect_wrapper(sym: &str) -> bool {
    let s = sym.trim_start_matches('@');
    s == "__syn_sched_loop"
        || s.starts_with("__syn_rom_call_")
        || s.starts_with("__syn_rom_noreturn_")
}

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Hardware single-stack + `.data`/`.bss` high-water proof for a compiled bare-metal ELF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareStackProof {
    pub isa: CpuIsa,
    pub chip_name: String,
    pub board_name: Option<String>,
    pub code_in_ram: bool,
    pub opt_level: String,
    pub text_bytes: u64,
    pub data_bytes: u64,
    pub bss_bytes: u64,
    pub static_sram_bytes: u64,
    pub sram_budget_bytes: u64,
    pub boot_stack_bytes: u64,
    pub boot_worst_path: Vec<String>,
    pub sched_task_stack_bytes: u64,
    pub sched_task_worst_path: Vec<String>,
    pub worst_isr_stack_bytes: u64,
    pub worst_isr_name: String,
    pub worst_isr_path: Vec<String>,
    pub isr_hw_overhead_bytes: u64,
    pub peak_stack_bytes: u64,
    pub total_high_water_bytes: u64,
    pub within_budget: bool,
    pub frame_sizes: BTreeMap<String, u64>,
    pub has_cycle: bool,
}

impl HardwareStackProof {
    /// Human-readable hardware stack + `.data`/`.bss` high-water proof table for `zevc ram`.
    pub fn format_table(&self) -> String {
        let target_label = match &self.board_name {
            Some(board) => format!("{} / {} ({})", board, self.chip_name, self.isa.as_str()),
            None => format!("{} ({})", self.chip_name, self.isa.as_str()),
        };
        let analyzer_label = match self.isa {
            CpuIsa::ArmCortexM if self.isr_hw_overhead_bytes == 104 => {
                "LLVM .stack_sizes + ELF call graph + 104B NVIC FPU frame"
            }
            CpuIsa::ArmCortexM => "LLVM .stack_sizes + ELF call graph + 32B NVIC frame",
            CpuIsa::Riscv32 => "LLVM .stack_sizes + ELF call graph + 64B trap context frame",
            CpuIsa::Avr => {
                "AVR ELF disassembly (push + sbiw/rcall.+0 + 2B return PC + 2B IRQ push)"
            }
            CpuIsa::HostX86_64Linux => "unsupported",
        };

        let mut out = String::new();
        out.push_str(
            "\n================================================================================\n",
        );
        let _ = writeln!(out, "  HARDWARE SINGLE-STACK + .BSS HIGH-WATER PROOF");
        let _ = writeln!(out, "  Target   : {target_label} [{}]", self.opt_level);
        let _ = writeln!(out, "  Analyzer : {analyzer_label}");
        out.push_str(
            "================================================================================\n\n",
        );

        out.push_str("1. Linked ELF Section Footprint (llvm-size)\n");
        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
        let text_label = if self.code_in_ram {
            "SRAM Code (.text + .rodata)"
        } else {
            "Flash Code (.text)"
        };
        let static_label = if self.code_in_ram {
            "Static SRAM (.text + .data + .bss)"
        } else {
            "Static SRAM (.data + .bss)"
        };
        let _ = writeln!(out, "  {:<36} {:>10} B", text_label, self.text_bytes);
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "Initialized SRAM (.data)", self.data_bytes
        );
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "Zero-Init SRAM (.bss)", self.bss_bytes
        );
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            static_label, self.static_sram_bytes
        );
        out.push('\n');

        out.push_str("2. Worst-Case Hardware Call Stack Proof\n");
        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "Sequential Boot Phase (boot_stack)", self.boot_stack_bytes
        );
        if !self.boot_worst_path.is_empty() {
            let _ = writeln!(out, "    path: {}", self.boot_worst_path.join(" -> "));
        }
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "Scheduler + Task Step (sched_task)", self.sched_task_stack_bytes
        );
        if !self.sched_task_worst_path.is_empty() {
            let _ = writeln!(out, "    path: {}", self.sched_task_worst_path.join(" -> "));
        }
        let isr_label = if self.worst_isr_name == "(none)" {
            "Worst-Case ISR (none attached)".to_string()
        } else {
            format!("Worst-Case ISR ({})", self.worst_isr_name)
        };
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B  (HW overhead: {} B)",
            isr_label, self.worst_isr_stack_bytes, self.isr_hw_overhead_bytes
        );
        if !self.worst_isr_path.is_empty() {
            let _ = writeln!(out, "    path: {}", self.worst_isr_path.join(" -> "));
        }
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B  [max(boot, sched_task) + worst_isr]",
            "PEAK HARDWARE STACK", self.peak_stack_bytes
        );
        if self.has_cycle {
            out.push_str("  note: call graph cycle guarded during DAG traversal\n");
        }
        out.push('\n');

        out.push_str("3. Total SRAM High-Water Budget Verdict\n");
        out.push_str(
            "--------------------------------------------------------------------------------\n",
        );
        let static_sram_lbl = if self.code_in_ram {
            "Static SRAM (.text + .data + .bss)"
        } else {
            "Static SRAM (.data + .bss)"
        };
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            static_sram_lbl, self.static_sram_bytes
        );
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "Peak Hardware Stack", self.peak_stack_bytes
        );
        out.push_str(
            "  ------------------------------------------------------------------------------\n",
        );
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "TOTAL HIGH-WATER SRAM", self.total_high_water_bytes
        );
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "Hardware SRAM Budget", self.sram_budget_bytes
        );
        let margin: i64 = (self.sram_budget_bytes as i64) - (self.total_high_water_bytes as i64);
        let verdict = if self.has_cycle {
            "UNBOUNDED (recursion cycle detected in call graph; static stack bound cannot be proven)"
                .to_string()
        } else if self.within_budget {
            format!("PASS ({margin} B headroom remaining)")
        } else {
            format!("FAIL (exceeds SRAM budget by {} B!)", -margin)
        };
        let _ = writeln!(out, "  {:<36}   {verdict}", "Budget Status");
        out.push_str(
            "================================================================================\n",
        );
        out
    }
}

struct TempProofDir {
    dir: PathBuf,
}

impl TempProofDir {
    fn create(prefix: &str) -> Result<Self, CodegenError> {
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("{prefix}-{}-{seq}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|e| {
            CodegenError::new("io", format!("failed to create {}: {e}", dir.display()))
        })?;
        Ok(Self { dir })
    }
}

impl Drop for TempProofDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn compile_ll_to_obj_with_stack_sizes(
    ll_path: &Path,
    opt_ll_path: &Path,
    obj_path: &Path,
    opts: &CodegenOptions,
    spec: &TargetSpec,
) -> Result<PathBuf, CodegenError> {
    let llc_flag = opts.llc_opt_flag()?;
    let llc_input: PathBuf = if opts.is_opt_o0() {
        ll_path.to_path_buf()
    } else {
        optimize_ll_with_opt(ll_path, opt_ll_path, opts)?;
        opt_ll_path.to_path_buf()
    };

    let llc = require_llvm_tool("/bin/llc", "llc")?;
    let mut cmd = Command::new(&llc);
    cmd.arg(format!("-mtriple={}", spec.triple));
    if spec.isa != CpuIsa::HostX86_64Linux {
        if !spec.cpu.is_empty() {
            cmd.arg(format!("-mcpu={}", spec.cpu));
        }
        if !spec.features.is_empty() {
            cmd.arg(format!("-mattr={}", spec.features));
        }
        if spec.isa == CpuIsa::Riscv32 {
            cmd.arg("-relocation-model=static");
        }
        cmd.arg("-function-sections").arg("-data-sections");
        if !spec.isa.is_avr() {
            cmd.arg("--stack-size-section");
        }
    }
    cmd.arg(llc_flag)
        .arg("-filetype=obj")
        .arg(&llc_input)
        .arg("-o")
        .arg(obj_path);

    let out = cmd
        .output()
        .map_err(|e| CodegenError::new("llc", format!("failed to invoke `{llc}`: {e}")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "llc",
            format!("llc exited with {}: {}", out.status, stderr.trim()),
        ));
    }
    Ok(llc_input)
}

fn inspect_elf_text_data_bss(elf_path: &Path) -> Result<(u64, u64, u64), CodegenError> {
    let llvm_size = require_llvm_tool("/bin/llvm-size", "llvm-size")?;
    let out = Command::new(&llvm_size)
        .arg("-A")
        .arg(elf_path)
        .output()
        .map_err(|e| {
            CodegenError::new("llvm-size", format!("failed to invoke `{llvm_size}`: {e}"))
        })?;
    if out.status.success() {
        let raw = String::from_utf8_lossy(&out.stdout);
        let mut text = 0u64;
        let mut data = 0u64;
        let mut bss = 0u64;
        let mut found_any = false;
        for line in raw.lines() {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() >= 2 {
                if let Ok(sz) = cols[1].parse::<u64>() {
                    match cols[0] {
                        ".text" => {
                            text = text.saturating_add(sz);
                            found_any = true;
                        }
                        ".data" | ".sdata" | ".iram1" => {
                            data = data.saturating_add(sz);
                            found_any = true;
                        }
                        ".bss" | ".sbss" => {
                            bss = bss.saturating_add(sz);
                            found_any = true;
                        }
                        _ => {}
                    }
                }
            }
        }
        if found_any {
            return Ok((text, data, bss));
        }
    }
    let fallback = inspect_elf_sizes(elf_path)?;
    Ok((fallback.text, fallback.data, fallback.bss))
}

fn run_readobj_stack_sizes(obj_path: &Path) -> Result<String, CodegenError> {
    let readobj = require_llvm_tool("/bin/llvm-readobj", "llvm-readobj")?;
    let out = Command::new(&readobj)
        .arg("--stack-sizes")
        .arg(obj_path)
        .output()
        .map_err(|e| {
            CodegenError::new("llvm-readobj", format!("failed to invoke `{readobj}`: {e}"))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "llvm-readobj",
            format!("llvm-readobj exited with {}: {}", out.status, stderr.trim()),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn run_objdump_disasm(elf_path: &Path) -> Result<String, CodegenError> {
    let objdump = require_llvm_tool("/bin/llvm-objdump", "llvm-objdump")?;
    let out = Command::new(&objdump)
        .arg("-d")
        .arg(elf_path)
        .output()
        .map_err(|e| {
            CodegenError::new("llvm-objdump", format!("failed to invoke `{objdump}`: {e}"))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "llvm-objdump",
            format!("llvm-objdump exited with {}: {}", out.status, stderr.trim()),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Parse `/bin/llvm-readobj --stack-sizes <obj>` output into a `symbol -> stack_bytes` map.
pub fn parse_llvm_readobj_stack_sizes(raw: &str) -> BTreeMap<String, u64> {
    let mut map = BTreeMap::new();
    let mut cur_fns: Vec<String> = Vec::new();

    for line in raw.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("Functions:") {
            let inner = rest.trim().trim_start_matches('[').trim_end_matches(']');
            cur_fns = inner
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        } else if let Some(rest) = t.strip_prefix("Size:") {
            let val_str = rest.trim();
            let parsed = if let Some(hex) = val_str
                .strip_prefix("0x")
                .or_else(|| val_str.strip_prefix("0X"))
            {
                u64::from_str_radix(hex, 16).ok()
            } else {
                val_str.parse::<u64>().ok()
            };
            if let Some(sz) = parsed {
                for f in cur_fns.drain(..) {
                    map.entry(f)
                        .and_modify(|prev: &mut u64| *prev = (*prev).max(sz))
                        .or_insert(sz);
                }
            }
        } else if t == "}" {
            cur_fns.clear();
        }
    }
    map
}

#[derive(Default)]
struct DisasmData {
    elf_symbols: BTreeSet<String>,
    addr_to_symbol: BTreeMap<u64, String>,
    fn_insns: BTreeMap<String, Vec<(u64, String, String)>>,
}

fn parse_objdump_disassembly(disasm: &str) -> DisasmData {
    let mut data = DisasmData::default();
    let mut current_fn: Option<String> = None;

    for line in disasm.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.ends_with(">:") {
            if let Some(lt) = trimmed.find('<') {
                let addr_str = trimmed[..lt].trim();
                let sym = &trimmed[lt + 1..trimmed.len() - 2];
                if let Ok(addr) = u64::from_str_radix(addr_str, 16) {
                    if sym.starts_with('.') {
                        current_fn = None;
                    } else {
                        data.addr_to_symbol.insert(addr, sym.to_string());
                        data.elf_symbols.insert(sym.to_string());
                        data.fn_insns.entry(sym.to_string()).or_default();
                        current_fn = Some(sym.to_string());
                    }
                    continue;
                }
            }
        }
        let Some(ref fn_name) = current_fn else {
            continue;
        };
        let Some(colon) = trimmed.find(':') else {
            continue;
        };
        let Ok(addr) = u64::from_str_radix(trimmed[..colon].trim(), 16) else {
            continue;
        };
        let after_colon = &trimmed[colon + 1..];
        let Some(tab) = after_colon.find('\t') else {
            continue;
        };
        let insn = after_colon[tab + 1..].trim();
        if insn.is_empty() || insn.starts_with('<') || insn.starts_with('.') {
            continue;
        }
        let (mnemonic, operands) = match insn.find(|c: char| c.is_ascii_whitespace()) {
            Some(pos) => (
                insn[..pos].to_ascii_lowercase(),
                insn[pos..].trim().to_string(),
            ),
            None => (insn.to_ascii_lowercase(), String::new()),
        };
        data.fn_insns
            .entry(fn_name.clone())
            .or_default()
            .push((addr, mnemonic, operands));
    }
    data
}

fn extract_angle_symbol(operands: &str) -> Option<&str> {
    let start = operands.find('<')?;
    let end_rel = operands[start + 1..].find('>')?;
    let sym = &operands[start + 1..start + 1 + end_rel];
    if sym.is_empty() || sym.contains('+') {
        None
    } else {
        Some(sym)
    }
}

fn parse_ll_define_header(t: &str) -> Option<String> {
    if t.starts_with("define ") && t.ends_with('{') {
        if let Some(at) = t.find('@') {
            let after = &t[at + 1..];
            if let Some(paren) = after.find('(') {
                return Some(after[..paren].trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

fn is_ll_call_line(t: &str) -> bool {
    t.starts_with("call ")
        || t.starts_with("tail call ")
        || t.starts_with("musttail call ")
        || t.starts_with("notail call ")
        || t.contains(" = call ")
        || t.contains(" = tail call ")
        || t.contains(" = musttail call ")
        || t.contains(" = notail call ")
}

fn parse_ll_call_edges(ll_text: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut cur_fn: Option<String> = None;

    for line in ll_text.lines() {
        let t = line.trim();
        if let Some(name) = parse_ll_define_header(t) {
            cur_fn = Some(name);
            continue;
        }
        if t == "}" {
            cur_fn = None;
            continue;
        }
        let Some(ref caller) = cur_fn else {
            continue;
        };
        if t.starts_with(';') || t.contains(" asm ") || !is_ll_call_line(t) {
            continue;
        }
        // Locate `@callee(` after `call `
        if let Some(call_pos) = t.find("call ") {
            let after_call = &t[call_pos + 5..];
            let mut search_from = 0usize;
            while let Some(rel_at) = after_call[search_from..].find('@') {
                let abs_at = search_from + rel_at;
                let ident_part = &after_call[abs_at + 1..];
                let end = ident_part
                    .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.' && c != '"')
                    .unwrap_or(ident_part.len());
                let raw_name = ident_part[..end].trim_matches('"');
                let rest = ident_part[end..].trim_start();
                if rest.starts_with('(') && !raw_name.is_empty() && !raw_name.starts_with("llvm.") {
                    edges
                        .entry(caller.clone())
                        .or_default()
                        .insert(raw_name.to_string());
                    break;
                }
                search_from = abs_at + 1;
            }
        }
    }
    edges
}

fn has_indirect_ll_call(after_call: &str) -> bool {
    let bytes = after_call.as_bytes();
    let mut i = 0usize;
    let mut paren_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut in_quotes = false;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' {
            in_quotes = !in_quotes;
            i += 1;
            continue;
        }
        if in_quotes {
            i += 1;
            continue;
        }
        match b {
            b'(' => paren_depth += 1,
            b')' => paren_depth = paren_depth.saturating_sub(1),
            b'{' => brace_depth += 1,
            b'}' => brace_depth = brace_depth.saturating_sub(1),
            b'[' => bracket_depth += 1,
            b']' => bracket_depth = bracket_depth.saturating_sub(1),
            b'@' | b'%' if paren_depth == 0 && brace_depth == 0 && bracket_depth == 0 => {
                let mut j = i + 1;
                let mut q = false;
                while j < bytes.len() {
                    let c = bytes[j] as char;
                    if c == '"' {
                        q = !q;
                        j += 1;
                    } else if q || c.is_ascii_alphanumeric() || c == '_' || c == '.' {
                        j += 1;
                    } else {
                        break;
                    }
                }
                let mut k = j;
                while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                    k += 1;
                }
                if k < bytes.len() && bytes[k] == b'(' && j > i + 1 {
                    return b == b'%';
                }
                i = j;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    false
}

fn parse_ll_indirect_callers(ll_text: &str) -> BTreeSet<String> {
    let mut callers = BTreeSet::new();
    let mut cur_fn: Option<String> = None;

    for line in ll_text.lines() {
        let t = line.trim();
        if let Some(name) = parse_ll_define_header(t) {
            cur_fn = Some(name);
            continue;
        }
        if t == "}" {
            cur_fn = None;
            continue;
        }
        let Some(ref caller) = cur_fn else {
            continue;
        };
        if is_exempt_indirect_wrapper(caller) {
            continue;
        }
        if t.starts_with(';') || t.contains(" asm ") || !is_ll_call_line(t) {
            continue;
        }
        if let Some(call_pos) = t.find("call ") {
            if has_indirect_ll_call(&t[call_pos + 5..]) {
                callers.insert(caller.clone());
            }
        }
    }
    callers
}

fn parse_imm_u64(tok: &str) -> Option<u64> {
    let s = tok.trim().trim_end_matches(',');
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

fn parse_signed_rel_offset(tok: &str) -> Option<i64> {
    let s = tok.split_whitespace().next()?;
    if let Some(rest) = s.strip_prefix(".+") {
        if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
            i64::from_str_radix(hex, 16).ok()
        } else {
            rest.parse::<i64>().ok()
        }
    } else if let Some(rest) = s.strip_prefix(".-") {
        if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
            i64::from_str_radix(hex, 16).ok().map(|v| -v)
        } else {
            rest.parse::<i64>().ok().map(|v| -v)
        }
    } else {
        None
    }
}

fn extract_rv_tramp_sp_alloc(insns: &[(u64, String, String)]) -> Option<u64> {
    for (_, mnem, ops) in insns {
        if (mnem == "addi" || mnem == "c.addi16sp") && ops.starts_with("sp,") {
            let last = ops.rsplit(',').next()?.trim();
            if let Some(neg) = last.strip_prefix('-') {
                return parse_imm_u64(neg);
            }
        }
    }
    None
}

type EdgeGraph = BTreeMap<String, Vec<(String, u64)>>;

fn add_edge(graph: &mut EdgeGraph, caller: &str, callee: &str, cost: u64) {
    let list = graph.entry(caller.to_string()).or_default();
    if let Some(existing) = list.iter_mut().find(|(c, _)| c == callee) {
        existing.1 = existing.1.max(cost);
    } else {
        list.push((callee.to_string(), cost));
    }
}

fn avr_two_ops(ops: &str) -> Option<(&str, &str)> {
    let clean = ops.split(';').next().unwrap_or("").trim();
    let (lhs, rhs) = clean.split_once(',')?;
    Some((lhs.trim(), rhs.trim()))
}

fn is_riscv_ra_target(ops: &str) -> bool {
    let clean = ops.split(';').next().unwrap_or("").trim();
    clean == "ra" || clean == "x1" || clean.ends_with("(ra)") || clean.ends_with("(x1)")
}

fn analyze_avr_disasm(
    disasm: &DisasmData,
    ll_indirect_callers: &BTreeSet<String>,
) -> (BTreeMap<String, u64>, EdgeGraph, BTreeMap<String, u64>) {
    let mut frame_sizes = BTreeMap::new();
    let mut graph = EdgeGraph::new();
    let mut indirect_callers: BTreeMap<String, u64> = BTreeMap::new();

    for (fn_name, insns) in &disasm.fn_insns {
        let mut frame_bytes = 0u64;
        let mut in_prologue = true;
        let mut pending_subi_lo: Option<u8> = None;
        let mut pending_sbci_hi: Option<u8> = None;
        let mut saw_sp_read = false;

        let mut z_tracks_sp = false;
        let mut z_pending_delta: i64 = 0;
        let mut z_subi_lo: Option<u8> = None;
        let mut z_sbci_hi: Option<u8> = None;
        let mut active_call_arg_bytes = 0u64;
        let mut max_call_arg_bytes = 0u64;
        let mut max_recorded_call_arg_bytes = 0u64;

        for (addr, mnem, ops) in insns {
            // 1. Prologue stack frame sizing
            if in_prologue {
                match mnem.as_str() {
                    "push" => {
                        frame_bytes += 1;
                    }
                    "rcall" if parse_signed_rel_offset(ops) == Some(0) => {
                        frame_bytes += 2;
                    }
                    "sbiw" if ops.starts_with("r28,") => {
                        if let Some(k_tok) = ops.split(',').nth(1) {
                            if let Some(k) = parse_imm_u64(k_tok) {
                                frame_bytes += k;
                            }
                        }
                    }
                    "in" if ops.starts_with("r28, 0x3d") || ops.starts_with("r29, 0x3e") => {
                        saw_sp_read = true;
                    }
                    "subi" if saw_sp_read && ops.starts_with("r28,") => {
                        if let Some(k_tok) = ops.split(',').nth(1) {
                            if let Some(v) = parse_imm_u64(k_tok).and_then(|x| u8::try_from(x).ok())
                            {
                                pending_subi_lo = Some(v);
                            }
                        }
                    }
                    "sbci" if saw_sp_read && ops.starts_with("r29,") => {
                        if let Some(k_tok) = ops.split(',').nth(1) {
                            if let Some(v) = parse_imm_u64(k_tok).and_then(|x| u8::try_from(x).ok())
                            {
                                pending_sbci_hi = Some(v);
                            }
                        }
                    }
                    "out" if ops.starts_with("0x3d, r28") => {
                        if let (Some(lo), Some(hi)) = (pending_subi_lo, pending_sbci_hi) {
                            let delta = ((hi as u16) << 8) | (lo as u16);
                            let signed = delta as i16;
                            if signed > 0 {
                                frame_bytes += signed as u64;
                            }
                        }
                        pending_subi_lo = None;
                        pending_sbci_hi = None;
                    }
                    "in" | "out" | "clr" | "eor" | "cli" => {}
                    _ => {
                        in_prologue = false;
                    }
                }
            }

            // 2. Z-register (`r30:r31`) outgoing stack arg SP adjust across the entire body
            match mnem.as_str() {
                "in" => {
                    if let Some((dst, src)) = avr_two_ops(ops) {
                        if dst == "r30" && parse_imm_u64(src) == Some(0x3d) {
                            z_tracks_sp = true;
                            z_pending_delta = 0;
                            z_subi_lo = None;
                            z_sbci_hi = None;
                        }
                    }
                }
                "sbiw" if z_tracks_sp => {
                    if let Some((dst, imm)) = avr_two_ops(ops) {
                        if matches!(dst, "r30" | "Z" | "r31:r30") {
                            if let Some(k) = parse_imm_u64(imm) {
                                z_pending_delta = z_pending_delta.saturating_add(k as i64);
                            }
                        }
                    }
                }
                "adiw" if z_tracks_sp => {
                    if let Some((dst, imm)) = avr_two_ops(ops) {
                        if matches!(dst, "r30" | "Z" | "r31:r30") {
                            if let Some(k) = parse_imm_u64(imm) {
                                z_pending_delta = z_pending_delta.saturating_sub(k as i64);
                            }
                        }
                    }
                }
                "subi" if z_tracks_sp => {
                    if let Some((dst, imm)) = avr_two_ops(ops) {
                        if dst == "r30" {
                            if let Some(v) = parse_imm_u64(imm).and_then(|x| u8::try_from(x).ok()) {
                                z_subi_lo = Some(v);
                            }
                        }
                    }
                }
                "sbci" if z_tracks_sp => {
                    if let Some((dst, imm)) = avr_two_ops(ops) {
                        if dst == "r31" {
                            if let Some(v) = parse_imm_u64(imm).and_then(|x| u8::try_from(x).ok()) {
                                z_sbci_hi = Some(v);
                            }
                        }
                    }
                }
                "out" if z_tracks_sp => {
                    if let Some((dst, src)) = avr_two_ops(ops) {
                        if parse_imm_u64(dst) == Some(0x3d) && src == "r30" {
                            let mut net_delta = z_pending_delta;
                            if let (Some(lo), Some(hi)) = (z_subi_lo.take(), z_sbci_hi.take()) {
                                let raw = ((hi as u16) << 8) | (lo as u16);
                                net_delta = net_delta.saturating_add((raw as i16) as i64);
                            }
                            if net_delta >= 0 {
                                active_call_arg_bytes =
                                    active_call_arg_bytes.saturating_add(net_delta as u64);
                                max_call_arg_bytes = max_call_arg_bytes.max(active_call_arg_bytes);
                            } else {
                                active_call_arg_bytes =
                                    active_call_arg_bytes.saturating_sub((-net_delta) as u64);
                            }
                            z_tracks_sp = false;
                            z_pending_delta = 0;
                        }
                    }
                }
                _ => {}
            }

            // 3. Call & tail-call edges (skip vector table `_start` jumps)
            if fn_name == "_start" {
                continue;
            }
            match mnem.as_str() {
                "call" => {
                    if let Some(target_addr) =
                        parse_imm_u64(ops.split_whitespace().next().unwrap_or(""))
                    {
                        if let Some((_, sym)) =
                            disasm.addr_to_symbol.range(..=target_addr).next_back()
                        {
                            if !sym.starts_with('.') {
                                let cost = 2u64.saturating_add(active_call_arg_bytes);
                                max_recorded_call_arg_bytes =
                                    max_recorded_call_arg_bytes.max(active_call_arg_bytes);
                                add_edge(&mut graph, fn_name, sym, cost);
                            }
                        }
                    }
                }
                "rcall" => {
                    if let Some(off) = parse_signed_rel_offset(ops) {
                        if off != 0 {
                            let target_addr = ((*addr as i64) + 2 + off) as u64;
                            if let Some((_, sym)) =
                                disasm.addr_to_symbol.range(..=target_addr).next_back()
                            {
                                if !sym.starts_with('.') {
                                    let cost = 2u64.saturating_add(active_call_arg_bytes);
                                    max_recorded_call_arg_bytes =
                                        max_recorded_call_arg_bytes.max(active_call_arg_bytes);
                                    add_edge(&mut graph, fn_name, sym, cost);
                                }
                            }
                        }
                    }
                }
                "icall" | "eicall" => {
                    let cost = 2u64.saturating_add(active_call_arg_bytes);
                    max_recorded_call_arg_bytes =
                        max_recorded_call_arg_bytes.max(active_call_arg_bytes);
                    if fn_name != "__syn_sched_loop" {
                        indirect_callers
                            .entry(fn_name.clone())
                            .and_modify(|c| *c = (*c).max(cost))
                            .or_insert(cost);
                    }
                }
                "ijmp" | "eijmp" => {
                    if fn_name != "__syn_sched_loop" && ll_indirect_callers.contains(fn_name) {
                        let cost = active_call_arg_bytes;
                        max_recorded_call_arg_bytes =
                            max_recorded_call_arg_bytes.max(active_call_arg_bytes);
                        indirect_callers
                            .entry(fn_name.clone())
                            .and_modify(|c| *c = (*c).max(cost))
                            .or_insert(cost);
                    }
                }
                "jmp" => {
                    if let Some(target_addr) =
                        parse_imm_u64(ops.split_whitespace().next().unwrap_or(""))
                    {
                        if let Some(sym) = disasm.addr_to_symbol.get(&target_addr) {
                            if sym != fn_name && !sym.starts_with('.') {
                                add_edge(&mut graph, fn_name, sym, 0);
                            }
                        }
                    }
                }
                "rjmp" => {
                    if let Some(off) = parse_signed_rel_offset(ops) {
                        let target_addr = ((*addr as i64) + 2 + off) as u64;
                        if let Some(sym) = disasm.addr_to_symbol.get(&target_addr) {
                            if sym != fn_name && !sym.starts_with('.') {
                                add_edge(&mut graph, fn_name, sym, 0);
                            }
                        }
                    }
                }
                _ => {}
            }

            if matches!(
                mnem.as_str(),
                "ret" | "reti" | "jmp" | "rjmp" | "ijmp" | "eijmp"
            ) {
                active_call_arg_bytes = 0;
                z_tracks_sp = false;
                z_pending_delta = 0;
                z_subi_lo = None;
                z_sbci_hi = None;
            }
        }

        if max_call_arg_bytes > max_recorded_call_arg_bytes {
            frame_bytes =
                frame_bytes.saturating_add(max_call_arg_bytes - max_recorded_call_arg_bytes);
        }
        frame_sizes.insert(fn_name.clone(), frame_bytes);
    }

    (frame_sizes, graph, indirect_callers)
}

fn build_arm_or_rv_call_graph(
    isa: CpuIsa,
    disasm: &DisasmData,
    ll_text: &str,
    frame_sizes: &BTreeMap<String, u64>,
    ll_indirect_callers: &BTreeSet<String>,
) -> (EdgeGraph, BTreeSet<String>) {
    let mut graph = EdgeGraph::new();
    let mut indirect_callers = BTreeSet::new();

    for (fn_name, insns) in &disasm.fn_insns {
        if fn_name == "__syn_rv_vector_table" {
            continue;
        }
        let check_indirect = !is_exempt_indirect_wrapper(fn_name);
        for (_, mnem, ops) in insns {
            let is_call = match isa {
                CpuIsa::ArmCortexM => matches!(mnem.as_str(), "bl" | "bl.w" | "blx"),
                CpuIsa::Riscv32 => {
                    matches!(mnem.as_str(), "jal" | "jalr" | "c.jal" | "c.jalr" | "call")
                }
                _ => false,
            };
            let is_tail_branch = match isa {
                CpuIsa::ArmCortexM => matches!(mnem.as_str(), "b" | "b.w" | "b.n" | "bx"),
                CpuIsa::Riscv32 => {
                    matches!(mnem.as_str(), "j" | "c.j" | "jr" | "c.jr" | "tail")
                }
                _ => false,
            };
            if is_call {
                if let Some(sym) = extract_angle_symbol(ops) {
                    add_edge(&mut graph, fn_name, sym, 0);
                } else if check_indirect {
                    match isa {
                        CpuIsa::ArmCortexM if mnem == "blx" => {
                            indirect_callers.insert(fn_name.clone());
                        }
                        CpuIsa::Riscv32
                            if matches!(mnem.as_str(), "jalr" | "c.jalr")
                                && !is_riscv_ra_target(ops) =>
                        {
                            indirect_callers.insert(fn_name.clone());
                        }
                        _ => {}
                    }
                }
            } else if is_tail_branch {
                if let Some(sym) = extract_angle_symbol(ops) {
                    if sym != fn_name && !sym.starts_with('.') && disasm.elf_symbols.contains(sym) {
                        add_edge(&mut graph, fn_name, sym, 0);
                    }
                } else if check_indirect && ll_indirect_callers.contains(fn_name) {
                    let op_clean = ops.split(';').next().unwrap_or("").trim();
                    let is_indirect_tail = match isa {
                        CpuIsa::ArmCortexM => mnem == "bx" && op_clean != "lr",
                        CpuIsa::Riscv32 => {
                            matches!(mnem.as_str(), "jr" | "c.jr") && !is_riscv_ra_target(op_clean)
                        }
                        _ => false,
                    };
                    if is_indirect_tail {
                        indirect_callers.insert(fn_name.clone());
                    }
                }
            }
        }
    }

    // Merge optimized `.ll` call edges (catches tail calls / IR calls for linked symbols).
    let ll_edges = parse_ll_call_edges(ll_text);
    for (caller, callees) in ll_edges {
        if !disasm.elf_symbols.contains(&caller) && !frame_sizes.contains_key(&caller) {
            continue;
        }
        for callee in callees {
            if disasm.elf_symbols.contains(&callee) || frame_sizes.contains_key(&callee) {
                add_edge(&mut graph, &caller, &callee, 0);
            }
        }
    }

    (graph, indirect_callers)
}

fn walk_worst_stack(
    node: &str,
    frame_sizes: &BTreeMap<String, u64>,
    graph: &EdgeGraph,
    excluded_callees: &BTreeSet<String>,
    visiting: &mut BTreeSet<String>,
    memo: &mut BTreeMap<String, (u64, Vec<String>)>,
    has_cycle: &mut bool,
) -> (u64, Vec<String>) {
    if !visiting.insert(node.to_string()) {
        *has_cycle = true;
        return (0, vec![format!("{node} [cycle]")]);
    }
    if excluded_callees.is_empty() {
        if let Some(cached) = memo.get(node) {
            visiting.remove(node);
            return cached.clone();
        }
    }

    let self_frame = frame_sizes.get(node).copied().unwrap_or(0);
    let self_label = format!("{node} ({self_frame}B)");

    let mut best_child_total = 0u64;
    let mut best_child_path: Vec<String> = Vec::new();
    let mut local_cycle = false;

    if let Some(edges) = graph.get(node) {
        for (callee, edge_cost) in edges {
            if excluded_callees.contains(callee) {
                continue;
            }
            let mut sub_cycle = false;
            let (child_total, child_path) = walk_worst_stack(
                callee,
                frame_sizes,
                graph,
                excluded_callees,
                visiting,
                memo,
                &mut sub_cycle,
            );
            if sub_cycle {
                local_cycle = true;
                *has_cycle = true;
            }
            let total_via_edge = edge_cost.saturating_add(child_total);
            if total_via_edge > best_child_total
                || (total_via_edge == best_child_total
                    && best_child_path.is_empty()
                    && !child_path.is_empty())
            {
                best_child_total = total_via_edge;
                let mut path = child_path;
                if *edge_cost > 0 && !path.is_empty() {
                    path[0] = format!("[+{}B call] {}", edge_cost, path[0]);
                }
                best_child_path = path;
            }
        }
    }

    let total = self_frame.saturating_add(best_child_total);
    let mut full_path = vec![self_label];
    full_path.extend(best_child_path);
    let res = (total, full_path);

    visiting.remove(node);
    if !local_cycle && excluded_callees.is_empty() {
        memo.insert(node.to_string(), res.clone());
    }
    res
}

type FnPtrSigKey = (Vec<MirType>, MirType, MirErrorSpec);

fn canonical_stack_fnptr_ty(ty: &MirType) -> MirType {
    match ty.host_value_type() {
        MirType::Ptr(inner) => MirType::Ptr(Box::new(canonical_stack_fnptr_ty(&inner))),
        MirType::Slice(inner) => MirType::Slice(Box::new(canonical_stack_fnptr_ty(&inner))),
        MirType::Array { elem, len } => MirType::Array {
            elem: Box::new(canonical_stack_fnptr_ty(&elem)),
            len,
        },
        MirType::FnPtr {
            params,
            ret,
            error_spec,
            ..
        } => MirType::FnPtr {
            is_noyield: false,
            params: params.iter().map(canonical_stack_fnptr_ty).collect(),
            ret: Box::new(canonical_stack_fnptr_ty(&ret)),
            error_spec: Box::new(match *error_spec {
                MirErrorSpec::Typed(e) => MirErrorSpec::Typed(canonical_stack_fnptr_ty(&e)),
                other => other,
            }),
        },
        other => other,
    }
}

fn fnptr_sig_key(params: &[MirType], ret: &MirType, error_spec: &MirErrorSpec) -> FnPtrSigKey {
    (
        params.iter().map(canonical_stack_fnptr_ty).collect(),
        canonical_stack_fnptr_ty(ret),
        match error_spec {
            MirErrorSpec::Typed(e) => MirErrorSpec::Typed(canonical_stack_fnptr_ty(e)),
            other => other.clone(),
        },
    )
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum MirNodeKey {
    Fn(String),
    Task(String),
    Isr(String),
    Boot,
}

struct MirIndirectScan<'a> {
    fn_sigs: HashMap<&'a str, (String, FnPtrSigKey)>,
    ext_sigs: HashMap<&'a str, (String, FnPtrSigKey)>,
    elf_or_frame_syms: &'a BTreeSet<String>,
    sig_to_plain_targets: HashMap<FnPtrSigKey, BTreeSet<String>>,
    sig_to_restricted_targets: HashMap<FnPtrSigKey, BTreeSet<String>>,
    all_addr_taken: BTreeSet<String>,
    direct_callees: BTreeMap<MirNodeKey, BTreeSet<String>>,
    direct_sig_calls: BTreeMap<MirNodeKey, HashSet<(FnPtrSigKey, bool)>>,
}

impl<'a> MirIndirectScan<'a> {
    fn record_fn_addr(&mut self, name: &str, expr_ty: &MirType) {
        let entry = self.fn_sigs.get(name).or_else(|| self.ext_sigs.get(name));
        let Some((elf_sym, sig)) = entry else {
            return;
        };
        if !self.elf_or_frame_syms.contains(elf_sym) {
            return;
        }
        self.all_addr_taken.insert(elf_sym.clone());
        self.sig_to_plain_targets
            .entry(sig.clone())
            .or_default()
            .insert(elf_sym.clone());
        if !matches!(
            expr_ty,
            MirType::FnPtr {
                is_noyield: false,
                ..
            }
        ) {
            self.sig_to_restricted_targets
                .entry(sig.clone())
                .or_default()
                .insert(elf_sym.clone());
        }
    }

    fn scan_prim_target(
        &mut self,
        t: &MirPrimTarget,
        caller: Option<&MirNodeKey>,
        vars: &[MirVar],
    ) {
        match t {
            MirPrimTarget::Global(_) => {}
            MirPrimTarget::PtrExpr(e) => self.scan_expr(e, caller, vars),
        }
    }

    fn scan_place(&mut self, p: &MirPlace, caller: Option<&MirNodeKey>, vars: &[MirVar]) {
        match p {
            MirPlace::Var(_) => {}
            MirPlace::Deref(e, _) => self.scan_expr(e, caller, vars),
            MirPlace::Field { base, .. } => self.scan_place(base, caller, vars),
            MirPlace::Index { base, index, .. } => {
                self.scan_place(base, caller, vars);
                self.scan_expr(index, caller, vars);
            }
        }
    }

    fn scan_fmt_chunks(
        &mut self,
        chunks: &[MirFmtChunk],
        caller: Option<&MirNodeKey>,
        vars: &[MirVar],
    ) {
        for ch in chunks {
            match ch {
                MirFmtChunk::Literal(_) => {}
                MirFmtChunk::Int { value, .. }
                | MirFmtChunk::Float { value, .. }
                | MirFmtChunk::Bool(value)
                | MirFmtChunk::PaddedBool { value, .. }
                | MirFmtChunk::Str(value)
                | MirFmtChunk::PaddedStr { cstr: value, .. }
                | MirFmtChunk::Bytes(value)
                | MirFmtChunk::PaddedBytes { slice: value, .. }
                | MirFmtChunk::EnumName { value, .. }
                | MirFmtChunk::PaddedEnumName { value, .. }
                | MirFmtChunk::DebugSlice { slice: value, .. } => {
                    self.scan_expr(value, caller, vars);
                }
            }
        }
    }

    fn scan_expr(&mut self, expr: &MirExpr, caller: Option<&MirNodeKey>, vars: &[MirVar]) {
        match &expr.kind {
            MirExprKind::ZeroValue
            | MirExprKind::IntLit(_)
            | MirExprKind::FloatLit(_)
            | MirExprKind::BoolLit(_)
            | MirExprKind::StringLit(_)
            | MirExprKind::TicksLit(_)
            | MirExprKind::EnumVariant { .. }
            | MirExprKind::ConstRef(_)
            | MirExprKind::GlobalPrimAddr(_)
            | MirExprKind::SubsystemAddr(_)
            | MirExprKind::MmioAddr(_)
            | MirExprKind::Var(_)
            | MirExprKind::ArrayLen(_)
            | MirExprKind::ResultIsErr(_)
            | MirExprKind::ResultErr(_)
            | MirExprKind::ResultVal(_)
            | MirExprKind::NowTicks
            | MirExprKind::SysQuery(_) => {}
            MirExprKind::FnAddr(name) => self.record_fn_addr(name, &expr.ty),
            MirExprKind::MakePrimView { ptr, .. } => self.scan_expr(ptr, caller, vars),
            MirExprKind::MmioRead { base, index, .. } => {
                self.scan_expr(base, caller, vars);
                if let Some(idx) = index {
                    self.scan_expr(idx, caller, vars);
                }
            }
            MirExprKind::Unary { expr: sub, .. }
            | MirExprKind::CStrToSlice(sub)
            | MirExprKind::Cast { expr: sub, .. }
            | MirExprKind::FieldAccess { base: sub, .. }
            | MirExprKind::SliceLen(sub)
            | MirExprKind::FromBytes { slice: sub, .. }
            | MirExprKind::LeaseGet { lease: sub, .. }
            | MirExprKind::LeaseAddr { lease: sub, .. } => self.scan_expr(sub, caller, vars),
            MirExprKind::AddrOfPlace(place)
            | MirExprKind::ArrayToSlice {
                array_place: place, ..
            } => self.scan_place(place, caller, vars),
            MirExprKind::Binary { lhs, rhs, .. }
            | MirExprKind::SeqEq { lhs, rhs, .. }
            | MirExprKind::Index {
                base: lhs,
                index: rhs,
                ..
            } => {
                self.scan_expr(lhs, caller, vars);
                self.scan_expr(rhs, caller, vars);
            }
            MirExprKind::SubSlice {
                base, start, end, ..
            } => {
                self.scan_expr(base, caller, vars);
                self.scan_expr(start, caller, vars);
                self.scan_expr(end, caller, vars);
            }
            MirExprKind::StructLit { fields, .. } => {
                for (_, fe, _) in fields {
                    self.scan_expr(fe, caller, vars);
                }
            }
            MirExprKind::ArrayLit { elems, .. } => {
                for el in elems {
                    self.scan_expr(el, caller, vars);
                }
            }
            MirExprKind::PrimRead(op) => match op.as_ref() {
                MirPrimReadOp::WatchRead { target, .. }
                | MirPrimReadOp::GuardedGet { target, .. }
                | MirPrimReadOp::WatchGen { target, .. }
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
                    self.scan_prim_target(target, caller, vars);
                }
                MirPrimReadOp::InboxTrySend { target, value, .. } => {
                    self.scan_prim_target(target, caller, vars);
                    self.scan_expr(value, caller, vars);
                }
                MirPrimReadOp::DmaSlotPark { target, lease, .. } => {
                    self.scan_prim_target(target, caller, vars);
                    self.scan_expr(lease, caller, vars);
                }
                MirPrimReadOp::ByteRingPush { target, src, .. } => {
                    self.scan_prim_target(target, caller, vars);
                    self.scan_expr(src, caller, vars);
                }
                MirPrimReadOp::ByteRingPushByte { target, byte, .. } => {
                    self.scan_prim_target(target, caller, vars);
                    self.scan_expr(byte, caller, vars);
                }
                MirPrimReadOp::ByteRingGrantWrite { target, len, .. } => {
                    self.scan_prim_target(target, caller, vars);
                    self.scan_expr(len, caller, vars);
                }
                MirPrimReadOp::ByteRingPop { target, dst, .. } => {
                    self.scan_prim_target(target, caller, vars);
                    self.scan_expr(dst, caller, vars);
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
                    self.scan_expr(view, caller, vars);
                }
                MirPrimReadOp::InboxViewTrySend { view, value, .. } => {
                    self.scan_expr(view, caller, vars);
                    self.scan_expr(value, caller, vars);
                }
                MirPrimReadOp::ByteRingViewPush { view, src, .. } => {
                    self.scan_expr(view, caller, vars);
                    self.scan_expr(src, caller, vars);
                }
                MirPrimReadOp::ByteRingViewPushByte { view, byte, .. } => {
                    self.scan_expr(view, caller, vars);
                    self.scan_expr(byte, caller, vars);
                }
                MirPrimReadOp::ByteRingViewGrantWrite { view, len } => {
                    self.scan_expr(view, caller, vars);
                    self.scan_expr(len, caller, vars);
                }
                MirPrimReadOp::ByteRingViewPop { view, dst } => {
                    self.scan_expr(view, caller, vars);
                    self.scan_expr(dst, caller, vars);
                }
            },
            MirExprKind::CallDirect {
                callee,
                is_extern_c,
                args,
            } => {
                if let Some(cnode) = caller {
                    if !*is_extern_c {
                        self.direct_callees
                            .entry(cnode.clone())
                            .or_default()
                            .insert(callee.clone());
                    } else if callee != "assert" {
                        if let Some((sig, is_restricted)) =
                            vars.iter().rev().find_map(|v| match &v.ty {
                                MirType::FnPtr {
                                    is_noyield,
                                    params,
                                    ret,
                                    error_spec,
                                } if v.name == *callee => {
                                    Some((fnptr_sig_key(params, ret, error_spec), *is_noyield))
                                }
                                _ => None,
                            })
                        {
                            self.direct_sig_calls
                                .entry(cnode.clone())
                                .or_default()
                                .insert((sig, is_restricted));
                        }
                    }
                }
                for a in args {
                    self.scan_expr(a, caller, vars);
                }
            }
            MirExprKind::RomCallU32 { addr, args, .. } => {
                self.scan_expr(addr, caller, vars);
                for a in args {
                    self.scan_expr(a, caller, vars);
                }
            }
        }
    }

    fn scan_stmt(&mut self, stmt: &MirStmt, caller: &MirNodeKey, vars: &[MirVar]) {
        match stmt {
            MirStmt::Assign { target, value, .. } => {
                self.scan_place(target, Some(caller), vars);
                self.scan_expr(value, Some(caller), vars);
            }
            MirStmt::Expr(e) => self.scan_expr(e, Some(caller), vars),
            MirStmt::PrimOp(op) => match op {
                MirPrimNonBlockingOp::WatchPublish { target, value, .. }
                | MirPrimNonBlockingOp::GuardedSet { target, value, .. }
                | MirPrimNonBlockingOp::InboxPost { target, value, .. } => {
                    self.scan_prim_target(target, Some(caller), vars);
                    self.scan_expr(value, Some(caller), vars);
                }
                MirPrimNonBlockingOp::SignalNotify { target, .. }
                | MirPrimNonBlockingOp::SignalClear { target }
                | MirPrimNonBlockingOp::WaitQueuePrepare { target }
                | MirPrimNonBlockingOp::WaitQueueCancel { target }
                | MirPrimNonBlockingOp::WaitQueueWakeAll { target, .. }
                | MirPrimNonBlockingOp::PingPongFlip { target, .. }
                | MirPrimNonBlockingOp::PingPongSetActive { target, .. }
                | MirPrimNonBlockingOp::ByteRingClear { target, .. } => {
                    self.scan_prim_target(target, Some(caller), vars);
                }
                MirPrimNonBlockingOp::PingPongCompleteHalf { target, half, .. } => {
                    self.scan_prim_target(target, Some(caller), vars);
                    self.scan_expr(half, Some(caller), vars);
                }
                MirPrimNonBlockingOp::LeaseRelease { value, .. } => {
                    self.scan_expr(value, Some(caller), vars);
                }
                MirPrimNonBlockingOp::LeaseNullify { target, .. } => {
                    self.scan_place(target, Some(caller), vars);
                }
                MirPrimNonBlockingOp::ByteRingConsume { target, len, .. }
                | MirPrimNonBlockingOp::ByteRingCommitWrite { target, len, .. } => {
                    self.scan_prim_target(target, Some(caller), vars);
                    self.scan_expr(len, Some(caller), vars);
                }
                MirPrimNonBlockingOp::InboxViewPost { view, value, .. } => {
                    self.scan_expr(view, Some(caller), vars);
                    self.scan_expr(value, Some(caller), vars);
                }
                MirPrimNonBlockingOp::ByteRingViewConsume { view, len }
                | MirPrimNonBlockingOp::ByteRingViewCommitWrite { view, len, .. } => {
                    self.scan_expr(view, Some(caller), vars);
                    self.scan_expr(len, Some(caller), vars);
                }
                MirPrimNonBlockingOp::ByteRingViewClear { view } => {
                    self.scan_expr(view, Some(caller), vars);
                }
            },
            MirStmt::Intrinsic(op) => match op {
                MirIntrinsicOp::UseSystick { hz } => self.scan_expr(hz, Some(caller), vars),
                MirIntrinsicOp::SetPriority { .. } => {}
                MirIntrinsicOp::SysOp { args, .. } => {
                    for a in args {
                        self.scan_expr(a, Some(caller), vars);
                    }
                }
            },
            MirStmt::EnterCritical
            | MirStmt::ExitCritical
            | MirStmt::EndInboxTurn
            | MirStmt::Break
            | MirStmt::Continue
            | MirStmt::SetIrqSelector { .. }
            | MirStmt::BoundedForLatch
            | MirStmt::ZeroTaskLocals => {}
            MirStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.scan_expr(cond, Some(caller), vars);
                for s in then_body {
                    self.scan_stmt(s, caller, vars);
                }
                for s in else_body {
                    self.scan_stmt(s, caller, vars);
                }
            }
            MirStmt::While { cond, body } => {
                self.scan_expr(cond, Some(caller), vars);
                for s in body {
                    self.scan_stmt(s, caller, vars);
                }
            }
            MirStmt::Switch {
                discr,
                arms,
                default,
            } => {
                self.scan_expr(discr, Some(caller), vars);
                for (_, body) in arms {
                    for s in body {
                        self.scan_stmt(s, caller, vars);
                    }
                }
                for s in default {
                    self.scan_stmt(s, caller, vars);
                }
            }
            MirStmt::Return(v) => {
                if let Some(e) = v {
                    self.scan_expr(e, Some(caller), vars);
                }
            }
            MirStmt::Error { reason, partial } => {
                if let Some(e) = reason {
                    self.scan_expr(e, Some(caller), vars);
                }
                if let Some(e) = partial {
                    self.scan_expr(e, Some(caller), vars);
                }
            }
            MirStmt::SpawnActivate { args, .. } => {
                for a in args {
                    self.scan_expr(a, Some(caller), vars);
                }
            }
            MirStmt::PrintFmt { chunks, .. } => {
                self.scan_fmt_chunks(chunks, Some(caller), vars);
            }
            MirStmt::FmtBuf { buf, chunks, .. } => {
                self.scan_expr(buf, Some(caller), vars);
                self.scan_fmt_chunks(chunks, Some(caller), vars);
            }
            MirStmt::FmtRing { ring, chunks, .. } => {
                self.scan_expr(ring, Some(caller), vars);
                self.scan_fmt_chunks(chunks, Some(caller), vars);
            }
            MirStmt::MmioWrite {
                base, index, value, ..
            } => {
                self.scan_expr(base, Some(caller), vars);
                if let Some(idx) = index {
                    self.scan_expr(idx, Some(caller), vars);
                }
                self.scan_expr(value, Some(caller), vars);
            }
        }
    }

    fn scan_terminator(&mut self, term: &MirTerminator, caller: &MirNodeKey, vars: &[MirVar]) {
        match term {
            MirTerminator::Goto(_) => {}
            MirTerminator::Branch { cond, .. } => self.scan_expr(cond, Some(caller), vars),
            MirTerminator::Switch { discr, .. } => self.scan_expr(discr, Some(caller), vars),
            MirTerminator::Return(v) => {
                if let Some(e) = v {
                    self.scan_expr(e, Some(caller), vars);
                }
            }
            MirTerminator::Error { reason, partial } => {
                if let Some(e) = reason {
                    self.scan_expr(e, Some(caller), vars);
                }
                if let Some(e) = partial {
                    self.scan_expr(e, Some(caller), vars);
                }
            }
            MirTerminator::Yield(yk) => match yk {
                MirYieldKind::EveryLatch { .. }
                | MirYieldKind::DelayPoll { .. }
                | MirYieldKind::ShareCheckpoint { .. } => {}
                MirYieldKind::Wait { arms, .. } => {
                    for arm in arms {
                        match &arm.op {
                            MirWaitArmOp::InboxRecv { target, .. }
                            | MirWaitArmOp::InboxSend { target, .. }
                            | MirWaitArmOp::SignalWait { target, .. }
                            | MirWaitArmOp::WaitQueueCommit { target, .. }
                            | MirWaitArmOp::WatchWaitChange { target, .. }
                            | MirWaitArmOp::PingPongWaitChunk { target, .. }
                            | MirWaitArmOp::PoolAlloc { target, .. }
                            | MirWaitArmOp::ByteRingWaitRead { target, .. }
                            | MirWaitArmOp::ByteRingWaitWrite { target, .. } => {
                                self.scan_prim_target(target, Some(caller), vars);
                            }
                            MirWaitArmOp::InboxViewRecv { view, .. }
                            | MirWaitArmOp::InboxViewSend { view, .. }
                            | MirWaitArmOp::ByteRingViewWaitRead { view, .. }
                            | MirWaitArmOp::ByteRingViewWaitWrite { view, .. } => {
                                self.scan_expr(view, Some(caller), vars);
                            }
                        }
                    }
                }
                MirYieldKind::CallYielding { callee, args, .. } => {
                    self.direct_callees
                        .entry(caller.clone())
                        .or_default()
                        .insert(callee.clone());
                    for a in args {
                        self.scan_expr(a, Some(caller), vars);
                    }
                }
                MirYieldKind::PollChild { .. } => {}
            },
        }
    }
}

fn collect_mir_indirect_targets_by_caller(
    mir: &MirModule,
    elf_or_frame_syms: &BTreeSet<String>,
) -> (BTreeMap<String, BTreeSet<String>>, BTreeSet<String>) {
    let mut fn_sigs = HashMap::new();
    for f in &mir.functions {
        if f.is_yielding {
            // Coroutine step functions are invoked via direct step calls, never LLVM indirect call instructions.
            continue;
        }
        let sym = format!("__syn_fn_{}", sanitize_ident(&f.name));
        let params: Vec<MirType> = f
            .params
            .iter()
            .map(|vid| f.vars[vid.0].ty.host_value_type())
            .collect();
        let sig = (params, f.ret_ty.host_value_type(), f.error_spec.clone());
        fn_sigs.insert(f.name.as_str(), (sym, sig));
    }
    let mut ext_sigs = HashMap::new();
    for ext in &mir.extern_fns {
        let sym = sanitize_ident(&ext.name);
        let params: Vec<MirType> = ext
            .params
            .iter()
            .map(|(_, ty)| ty.host_value_type())
            .collect();
        let sig = (params, ext.ret_ty.host_value_type(), MirErrorSpec::None);
        ext_sigs.insert(ext.name.as_str(), (sym, sig));
    }

    let mut scan = MirIndirectScan {
        fn_sigs,
        ext_sigs,
        elf_or_frame_syms,
        sig_to_plain_targets: HashMap::new(),
        sig_to_restricted_targets: HashMap::new(),
        all_addr_taken: BTreeSet::new(),
        direct_callees: BTreeMap::new(),
        direct_sig_calls: BTreeMap::new(),
    };

    for c in &mir.consts {
        scan.scan_expr(&c.init, None, &[]);
    }
    for p in &mir.primitives {
        if let Some(init) = &p.init {
            scan.scan_expr(init, None, &[]);
        }
    }
    for sp in &mir.spawned_tasks {
        for a in &sp.init_args {
            scan.scan_expr(a, None, &[]);
        }
    }
    for v in &mir.irq_vectors {
        for b in &v.bindings {
            for ba in &b.baked_args {
                if let MirBakedArg::Const(e) = ba {
                    scan.scan_expr(e, None, &[]);
                }
            }
            for sv in &b.state_vars {
                scan.scan_expr(&sv.init, None, &[]);
            }
        }
    }

    for f in &mir.functions {
        let node = MirNodeKey::Fn(f.name.clone());
        match &f.body {
            MirFnBody::Direct(stmts) => {
                for s in stmts {
                    scan.scan_stmt(s, &node, &f.vars);
                }
            }
            MirFnBody::Coroutine { blocks, .. } => {
                for b in blocks {
                    for s in &b.stmts {
                        scan.scan_stmt(s, &node, &f.vars);
                    }
                    scan.scan_terminator(&b.terminator, &node, &f.vars);
                }
            }
        }
    }
    for bp in &mir.task_blueprints {
        let node = MirNodeKey::Task(bp.name.clone());
        for b in &bp.blocks {
            for s in &b.stmts {
                scan.scan_stmt(s, &node, &bp.vars);
            }
            scan.scan_terminator(&b.terminator, &node, &bp.vars);
        }
    }
    for isr in &mir.isrs {
        let node = MirNodeKey::Isr(isr.name.clone());
        for sv in &isr.state_vars {
            scan.scan_expr(&sv.init, Some(&node), &isr.vars);
        }
        for s in &isr.body {
            scan.scan_stmt(s, &node, &isr.vars);
        }
    }
    for step in &mir.boot.steps {
        let node = MirNodeKey::Boot;
        match step {
            MirBootStep::Stmt(s) => scan.scan_stmt(s, &node, &mir.boot.vars),
            MirBootStep::RunYieldingFn { callee, args, .. } => {
                scan.direct_callees
                    .entry(node.clone())
                    .or_default()
                    .insert(callee.clone());
                for a in args {
                    scan.scan_expr(a, Some(&node), &mir.boot.vars);
                }
            }
            MirBootStep::SpawnActivate { args, .. } => {
                for a in args {
                    scan.scan_expr(a, Some(&node), &mir.boot.vars);
                }
            }
        }
    }

    // Fixed-point propagation of indirect FnPtr call signatures over direct MIR call edges
    // so `/bin/opt` cross-function inlining preserves every caller's possible indirect targets.
    let mut node_sigs: BTreeMap<MirNodeKey, HashSet<(FnPtrSigKey, bool)>> = scan.direct_sig_calls;
    let mut all_nodes: BTreeSet<MirNodeKey> = BTreeSet::new();
    all_nodes.extend(scan.direct_callees.keys().cloned());
    all_nodes.extend(node_sigs.keys().cloned());

    let mut changed = true;
    while changed {
        changed = false;
        for caller in &all_nodes {
            let Some(callees) = scan.direct_callees.get(caller) else {
                continue;
            };
            let mut inherited = Vec::new();
            for callee_name in callees {
                if let Some(csigs) = node_sigs.get(&MirNodeKey::Fn(callee_name.clone())) {
                    inherited.extend(csigs.iter().cloned());
                }
            }
            if !inherited.is_empty() {
                let entry = node_sigs.entry(caller.clone()).or_default();
                for sig in inherited {
                    if entry.insert(sig) {
                        changed = true;
                    }
                }
            }
        }
    }

    let targets_for_sigs = |sigs: &HashSet<(FnPtrSigKey, bool)>| -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for (sig, is_restricted) in sigs {
            let map = if *is_restricted {
                &scan.sig_to_restricted_targets
            } else {
                &scan.sig_to_plain_targets
            };
            if let Some(cands) = map.get(sig) {
                out.extend(cands.iter().cloned());
            }
        }
        out
    };

    let mut per_caller_targets: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for f in &mir.functions {
        let node = MirNodeKey::Fn(f.name.clone());
        let targets = node_sigs
            .get(&node)
            .map(targets_for_sigs)
            .unwrap_or_default();
        let sym = if f.is_yielding {
            format!("__syn_fn_step_{}", sanitize_ident(&f.name))
        } else {
            format!("__syn_fn_{}", sanitize_ident(&f.name))
        };
        per_caller_targets.insert(sym, targets);
    }
    for bp in &mir.task_blueprints {
        let node = MirNodeKey::Task(bp.name.clone());
        let targets = node_sigs
            .get(&node)
            .map(targets_for_sigs)
            .unwrap_or_default();
        per_caller_targets.insert(
            format!("__syn_task_step_{}", sanitize_ident(&bp.name)),
            targets,
        );
    }
    for isr in &mir.isrs {
        let node = MirNodeKey::Isr(isr.name.clone());
        let targets = node_sigs
            .get(&node)
            .map(targets_for_sigs)
            .unwrap_or_default();
        per_caller_targets.insert(format!("__syn_isr_{}", sanitize_ident(&isr.name)), targets);
    }
    for v in &mir.irq_vectors {
        let mut vec_targets = BTreeSet::new();
        for b in &v.bindings {
            if let Some(ts) = node_sigs
                .get(&MirNodeKey::Isr(b.isr_name.clone()))
                .map(targets_for_sigs)
            {
                vec_targets.extend(ts);
            }
        }
        per_caller_targets.insert(format!("__syn_vec_irq_{}", v.irq_num), vec_targets.clone());
        per_caller_targets.insert(format!("__syn_avr_vec_{}", v.irq_num), vec_targets);
    }
    let boot_targets = node_sigs
        .get(&MirNodeKey::Boot)
        .map(targets_for_sigs)
        .unwrap_or_default();
    for boot_sym in [
        "__syn_boot",
        "__syn_avr_entry",
        "Reset_Handler",
        "__syn_rv_reset",
    ] {
        per_caller_targets.insert(boot_sym.to_string(), boot_targets.clone());
    }

    // Map `sys.set_*` indirect dispatch wrappers (used only when dynamic `fn(...)` is passed).
    let hook_wrappers: [(&str, FnPtrSigKey); 7] = [
        (
            "__syn_on_putc",
            (vec![MirType::U8], MirType::Void, MirErrorSpec::None),
        ),
        (
            "__syn_now_ms",
            (Vec::new(), MirType::U32, MirErrorSpec::None),
        ),
        (
            "__syn_on_exit",
            (vec![MirType::I32], MirType::Void, MirErrorSpec::None),
        ),
        (
            "__syn_on_fault",
            (Vec::new(), MirType::Void, MirErrorSpec::None),
        ),
        (
            "__syn_on_idle",
            (
                vec![MirType::U32, MirType::U32],
                MirType::Void,
                MirErrorSpec::None,
            ),
        ),
        (
            "__syn_on_rv_irq_route",
            (
                vec![MirType::U32, MirType::U32],
                MirType::Void,
                MirErrorSpec::None,
            ),
        ),
        (
            "__syn_on_rv_irq_ack",
            (vec![MirType::U32], MirType::Void, MirErrorSpec::None),
        ),
    ];
    for (wsym, sig) in hook_wrappers {
        if let Some(cands) = scan.sig_to_restricted_targets.get(&sig) {
            per_caller_targets.insert(wsym.to_string(), cands.clone());
        }
    }

    (per_caller_targets, scan.all_addr_taken)
}

/// Compile `mir` using the resolved `TargetSpec` and prove worst-case hardware
/// single-stack + `.data`/`.bss` SRAM high-water mark against `spec.sram_budget_bytes`.
pub fn analyze_hardware_stack_high_water(
    mir: &MirModule,
    opts: &CodegenOptions,
) -> Result<HardwareStackProof, CodegenError> {
    let spec = opts.effective_spec()?;
    if spec.isa == CpuIsa::HostX86_64Linux {
        return Err(CodegenError::new(
            "stack-sizes",
            "hardware stack high-water proof only supports bare-metal embedded targets",
        ));
    }
    let sram_budget_bytes = u64::from(spec.sram_budget_bytes);

    let temp = TempProofDir::create("zev-stack-proof")?;
    let ll_path = temp.dir.join("firmware.ll");
    let opt_ll_path = temp.dir.join("firmware.opt.ll");
    let obj_path = temp.dir.join("firmware.o");
    let ld_path = temp.dir.join("firmware.ld");
    let elf_path = temp.dir.join("firmware.elf");

    let raw_ll = crate::emit_llvm_ir_with_options(mir, opts)?;
    fs::write(&ll_path, &raw_ll).map_err(|e| {
        CodegenError::new("io", format!("failed to write {}: {e}", ll_path.display()))
    })?;
    let ld_script = synthesize_target_linker_script(&spec);
    fs::write(&ld_path, &ld_script).map_err(|e| {
        CodegenError::new("io", format!("failed to write {}: {e}", ld_path.display()))
    })?;

    let used_ll_path =
        compile_ll_to_obj_with_stack_sizes(&ll_path, &opt_ll_path, &obj_path, opts, &spec)?;
    link_obj_to_elf(&obj_path, &ld_path, &elf_path)?;

    let (text_bytes, data_bytes, bss_bytes) = inspect_elf_text_data_bss(&elf_path)?;
    let static_sram_bytes = if spec.code_in_ram {
        text_bytes
            .saturating_add(data_bytes)
            .saturating_add(bss_bytes)
    } else {
        data_bytes.saturating_add(bss_bytes)
    };

    let disasm_text = run_objdump_disasm(&elf_path)?;
    let disasm = parse_objdump_disassembly(&disasm_text);
    let ll_text = fs::read_to_string(&used_ll_path).unwrap_or(raw_ll);
    let ll_indirect_callers = parse_ll_indirect_callers(&ll_text);

    let (mut frame_sizes, mut graph, mut indirect_callers) = if spec.isa.is_avr() {
        analyze_avr_disasm(&disasm, &ll_indirect_callers)
    } else {
        let readobj_raw = run_readobj_stack_sizes(&obj_path)?;
        let mut frames = parse_llvm_readobj_stack_sizes(&readobj_raw);
        if spec.isa == CpuIsa::Riscv32 {
            for (sym, insns) in &disasm.fn_insns {
                if sym.starts_with("__syn_rv_irq_tramp_") || sym == "__syn_rv_exc_tramp" {
                    let parsed_alloc = extract_rv_tramp_sp_alloc(insns).unwrap_or(64);
                    frames
                        .entry(sym.clone())
                        .and_modify(|v| {
                            if *v == 0 {
                                *v = parsed_alloc;
                            }
                        })
                        .or_insert(parsed_alloc);
                }
            }
        }
        let (g, disasm_ind) =
            build_arm_or_rv_call_graph(spec.isa, &disasm, &ll_text, &frames, &ll_indirect_callers);
        let ind_map = disasm_ind.into_iter().map(|c| (c, 0u64)).collect();
        (frames, g, ind_map)
    };

    // Add declared compile-time constant `_sb<N>` stack bounds for `@__syn_rom_call_*` wrappers.
    let known_syms: BTreeSet<String> = disasm
        .elf_symbols
        .iter()
        .cloned()
        .chain(frame_sizes.keys().cloned())
        .collect();
    for sym in known_syms {
        if let Some(sb) = parse_rom_call_stack_bound(&sym) {
            frame_sizes
                .entry(sym)
                .and_modify(|v| *v = v.saturating_add(sb))
                .or_insert(sb);
        }
    }

    // Merge indirect callers observed in `.ll` + disassembly, and wire conservative edges
    // to non-entry, non-coroutine user functions (`__syn_fn_*` excluding `__syn_fn_step_*`) + `extern "C"`.
    let step_edge_cost = if spec.isa.is_avr() { 2 } else { 0 };
    for caller in &ll_indirect_callers {
        if !is_exempt_indirect_wrapper(caller)
            && (disasm.elf_symbols.contains(caller) || frame_sizes.contains_key(caller))
        {
            indirect_callers
                .entry(caller.clone())
                .or_insert(step_edge_cost);
        }
    }
    if !indirect_callers.is_empty() {
        let mut known_elf_syms: BTreeSet<String> = disasm.elf_symbols.clone();
        known_elf_syms.extend(frame_sizes.keys().cloned());
        let (per_caller_targets, all_addr_taken) =
            collect_mir_indirect_targets_by_caller(mir, &known_elf_syms);

        let mut legacy_fallback_targets = BTreeSet::new();
        if all_addr_taken.is_empty() && mir.functions.is_empty() {
            for sym in disasm.elf_symbols.iter().chain(frame_sizes.keys()) {
                if sym.starts_with("__syn_fn_") && !sym.starts_with("__syn_fn_step_") {
                    legacy_fallback_targets.insert(sym.clone());
                }
            }
            for ext in &mir.extern_fns {
                let ext_sym = sanitize_ident(&ext.name);
                if disasm.elf_symbols.contains(&ext_sym) || frame_sizes.contains_key(&ext_sym) {
                    legacy_fallback_targets.insert(ext_sym);
                }
            }
        }

        for (caller, cost) in &indirect_callers {
            if is_exempt_indirect_wrapper(caller) {
                continue;
            }
            let targets: &BTreeSet<String> = match per_caller_targets.get(caller) {
                Some(specific) => specific,
                None if !all_addr_taken.is_empty() => &all_addr_taken,
                None => &legacy_fallback_targets,
            };
            for target in targets {
                add_edge(&mut graph, caller, target, *cost);
            }
        }
    }

    // Wire indirect task-step dispatch edges from `__syn_sched_loop` -> `__syn_task_step_<bp>`
    let mut seen_bps = BTreeSet::new();
    for sp in &mir.spawned_tasks {
        if seen_bps.insert(sp.blueprint_name.clone()) {
            let step_sym = format!("__syn_task_step_{}", sanitize_ident(&sp.blueprint_name));
            if disasm.elf_symbols.contains(&step_sym) || frame_sizes.contains_key(&step_sym) {
                add_edge(&mut graph, "__syn_sched_loop", &step_sym, step_edge_cost);
            }
        }
    }

    let mut has_cycle = false;
    let mut memo = BTreeMap::new();
    let mut visiting = BTreeSet::new();
    let empty_excl = BTreeSet::new();

    // 1. Boot phase worst-case stack (`__syn_boot` or fallback entry excluding `__syn_sched_loop`)
    let (boot_stack_bytes, boot_worst_path) =
        if disasm.elf_symbols.contains("__syn_boot") || frame_sizes.contains_key("__syn_boot") {
            walk_worst_stack(
                "__syn_boot",
                &frame_sizes,
                &graph,
                &empty_excl,
                &mut visiting,
                &mut memo,
                &mut has_cycle,
            )
        } else {
            let fallback_boot = [
                "Reset_Handler",
                "__syn_rv_reset",
                "__syn_avr_entry",
                "_start",
            ]
            .into_iter()
            .find(|s| disasm.elf_symbols.contains(*s) || frame_sizes.contains_key(*s));
            if let Some(root) = fallback_boot {
                let excl = BTreeSet::from(["__syn_sched_loop".to_string()]);
                walk_worst_stack(
                    root,
                    &frame_sizes,
                    &graph,
                    &excl,
                    &mut visiting,
                    &mut memo,
                    &mut has_cycle,
                )
            } else {
                (0, Vec::new())
            }
        };

    // 2. Scheduler + spawned task-step worst-case stack
    let (sched_task_stack_bytes, sched_task_worst_path) =
        if disasm.elf_symbols.contains("__syn_sched_loop")
            || frame_sizes.contains_key("__syn_sched_loop")
        {
            walk_worst_stack(
                "__syn_sched_loop",
                &frame_sizes,
                &graph,
                &empty_excl,
                &mut visiting,
                &mut memo,
                &mut has_cycle,
            )
        } else {
            (0, Vec::new())
        };

    // 3. Worst-case preempting ISR stack (including hardware interrupt frame overhead)
    let (worst_isr_stack_bytes, worst_isr_name, worst_isr_path, isr_hw_overhead_bytes) =
        match spec.isa {
            CpuIsa::ArmCortexM => {
                let (hw_frame, hw_label) = if spec.has_fpu {
                    (104u64, "<HW NVIC FPU frame (104B)>")
                } else {
                    (32u64, "<HW NVIC frame (32B)>")
                };
                let mut candidates = Vec::new();
                for st_sym in ["SysTick_Handler", "__syn_systick_isr"] {
                    if disasm.elf_symbols.contains(st_sym) || frame_sizes.contains_key(st_sym) {
                        candidates.push(st_sym.to_string());
                    }
                }
                for v in &mir.irq_vectors {
                    let sym = format!("__syn_vec_irq_{}", v.irq_num);
                    if disasm.elf_symbols.contains(&sym) || frame_sizes.contains_key(&sym) {
                        candidates.push(sym);
                    }
                }
                if mir.irq_vectors.is_empty() {
                    for isr in &mir.isrs {
                        let sym = format!("__syn_isr_{}", sanitize_ident(&isr.name));
                        if disasm.elf_symbols.contains(&sym) || frame_sizes.contains_key(&sym) {
                            candidates.push(sym);
                        }
                    }
                }
                pick_worst_isr(
                    &candidates,
                    hw_frame,
                    hw_label,
                    &frame_sizes,
                    &graph,
                    &mut visiting,
                    &mut memo,
                    &mut has_cycle,
                )
            }
            CpuIsa::Riscv32 => {
                let routes = assign_rv_irq_routes_for_spec(&mir.irq_vectors, &spec)?;
                if routes.is_empty()
                    && !disasm.elf_symbols.contains("__syn_systick_isr")
                    && !disasm.elf_symbols.contains("__syn_rv_exc_tramp")
                {
                    (0, "(none)".to_string(), Vec::new(), 0)
                } else {
                    let mut tramp_candidates = Vec::new();
                    let cpu_slots: BTreeSet<u8> = routes.iter().map(|r| r.cpu_slot).collect();
                    for c in cpu_slots {
                        let tramp = format!("__syn_rv_irq_tramp_{c}");
                        if disasm.elf_symbols.contains(&tramp) || frame_sizes.contains_key(&tramp) {
                            tramp_candidates.push(tramp);
                        }
                    }
                    if disasm.elf_symbols.contains("__syn_rv_exc_tramp") {
                        tramp_candidates.push("__syn_rv_exc_tramp".to_string());
                    }
                    if !tramp_candidates.is_empty() {
                        // Trap trampoline frame already carries the 64 B context save in `frame_sizes`
                        let (bytes, name, path, _) = pick_worst_isr(
                            &tramp_candidates,
                            0,
                            "",
                            &frame_sizes,
                            &graph,
                            &mut visiting,
                            &mut memo,
                            &mut has_cycle,
                        );
                        (bytes, name, path, 64)
                    } else {
                        let mut vec_candidates = Vec::new();
                        for v in &mir.irq_vectors {
                            let sym = format!("__syn_vec_irq_{}", v.irq_num);
                            if disasm.elf_symbols.contains(&sym) || frame_sizes.contains_key(&sym) {
                                vec_candidates.push(sym);
                            }
                        }
                        pick_worst_isr(
                            &vec_candidates,
                            64,
                            "<HW RV32 trap context (64B)>",
                            &frame_sizes,
                            &graph,
                            &mut visiting,
                            &mut memo,
                            &mut has_cycle,
                        )
                    }
                }
            }
            CpuIsa::Avr => {
                let hw_frame = 2u64;
                let mut candidates = Vec::new();
                for v in &mir.irq_vectors {
                    let avr_vec = format!("__syn_avr_vec_{}", v.irq_num);
                    let raw_vec = format!("__syn_vec_irq_{}", v.irq_num);
                    if disasm.elf_symbols.contains(&avr_vec) || frame_sizes.contains_key(&avr_vec) {
                        candidates.push(avr_vec);
                    } else if disasm.elf_symbols.contains(&raw_vec)
                        || frame_sizes.contains_key(&raw_vec)
                    {
                        candidates.push(raw_vec);
                    }
                }
                pick_worst_isr(
                    &candidates,
                    hw_frame,
                    "<HW AVR vector PC (2B)>",
                    &frame_sizes,
                    &graph,
                    &mut visiting,
                    &mut memo,
                    &mut has_cycle,
                )
            }
            CpuIsa::HostX86_64Linux => unreachable!(),
        };

    // Retain only non-dot ELF symbols in `frame_sizes` for clean reporting
    frame_sizes.retain(|k, _| !k.starts_with('.'));

    let peak_stack_bytes = boot_stack_bytes
        .max(sched_task_stack_bytes)
        .saturating_add(worst_isr_stack_bytes);
    let total_high_water_bytes = static_sram_bytes.saturating_add(peak_stack_bytes);
    let within_budget = !has_cycle && total_high_water_bytes <= sram_budget_bytes;

    Ok(HardwareStackProof {
        isa: spec.isa,
        chip_name: spec.chip_name,
        board_name: spec.board_name,
        code_in_ram: spec.code_in_ram,
        opt_level: spec.opt_level,
        text_bytes,
        data_bytes,
        bss_bytes,
        static_sram_bytes,
        sram_budget_bytes,
        boot_stack_bytes,
        boot_worst_path,
        sched_task_stack_bytes,
        sched_task_worst_path,
        worst_isr_stack_bytes,
        worst_isr_name,
        worst_isr_path,
        isr_hw_overhead_bytes,
        peak_stack_bytes,
        total_high_water_bytes,
        within_budget,
        frame_sizes,
        has_cycle,
    })
}

#[allow(clippy::too_many_arguments)]
fn pick_worst_isr(
    candidates: &[String],
    hw_overhead: u64,
    hw_label: &str,
    frame_sizes: &BTreeMap<String, u64>,
    graph: &EdgeGraph,
    visiting: &mut BTreeSet<String>,
    memo: &mut BTreeMap<String, (u64, Vec<String>)>,
    has_cycle: &mut bool,
) -> (u64, String, Vec<String>, u64) {
    let empty_excl = BTreeSet::new();
    let mut best_bytes = 0u64;
    let mut best_name = "(none)".to_string();
    let mut best_path = Vec::new();
    let mut any_found = false;

    for cand in candidates {
        let (sw_bytes, sw_path) = walk_worst_stack(
            cand,
            frame_sizes,
            graph,
            &empty_excl,
            visiting,
            memo,
            has_cycle,
        );
        let total = hw_overhead.saturating_add(sw_bytes);
        if !any_found || total > best_bytes {
            any_found = true;
            best_bytes = total;
            best_name = cand.clone();
            let mut path = Vec::new();
            if hw_overhead > 0 && !hw_label.is_empty() {
                path.push(hw_label.to_string());
            }
            path.extend(sw_path);
            best_path = path;
        }
    }

    if any_found {
        (best_bytes, best_name, best_path, hw_overhead)
    } else {
        (0, "(none)".to_string(), Vec::new(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llvm_readobj_stack_sizes_entries() {
        let sample = r#"
File: test.o
Format: elf32-littlearm
Arch: arm
AddressSize: 32bit
LoadName: <Not found>
StackSizes [
  Entry {
    Functions: [__syn_boot]
    Size: 0x20
  }
  Entry {
    Functions: [fn_a, fn_b]
    Size: 0x18
  }
  Entry {
    Functions: [leaf]
    Size: 0x0
  }
]
"#;
        let map = parse_llvm_readobj_stack_sizes(sample);
        assert_eq!(map.get("__syn_boot"), Some(&32));
        assert_eq!(map.get("fn_a"), Some(&24));
        assert_eq!(map.get("fn_b"), Some(&24));
        assert_eq!(map.get("leaf"), Some(&0));
    }

    #[test]
    fn parses_avr_prologue_frames_and_call_edges() {
        let sample_disasm = r#"
00000100 <__syn_avr_vec_11>:
     100: 0f 92        	push	r0
     102: 1f 92        	push	r1
     104: 2f 93        	push	r18
     106: 00 d0        	rcall	.+0
     108: 0e 94 00 01  	call	0x200
     10c: 2f 91        	pop	r18
     10e: 1f 90        	pop	r1
     110: 0f 90        	pop	r0
     112: 18 95        	reti

00000200 <__syn_fn_helper>:
     200: cf 93        	push	r28
     202: df 93        	push	r29
     204: cd b7        	in	r28, 0x3d
     206: de b7        	in	r29, 0x3e
     208: 64 97        	sbiw	r28, 0x14
     20a: 01 d0        	rcall	.+4
     20c: 08 95        	ret

00000210 <__syn_fn_leaf>:
     210: 08 95        	ret
"#;
        let disasm = parse_objdump_disassembly(sample_disasm);
        let (frames, graph, indirect) = analyze_avr_disasm(&disasm, &BTreeSet::new());
        assert!(indirect.is_empty());
        // __syn_avr_vec_11: 3 pushes (3 B) + rcall .+0 (2 B) = 5 B
        assert_eq!(frames.get("__syn_avr_vec_11"), Some(&5));
        // __syn_fn_helper: 2 pushes (2 B) + sbiw 0x14 (20 B) = 22 B
        assert_eq!(frames.get("__syn_fn_helper"), Some(&22));
        assert_eq!(frames.get("__syn_fn_leaf"), Some(&0));

        // 0x20a + 2 + 4 = 0x210 (__syn_fn_leaf)
        assert_eq!(
            graph.get("__syn_avr_vec_11"),
            Some(&vec![("__syn_fn_helper".to_string(), 2)])
        );
        assert_eq!(
            graph.get("__syn_fn_helper"),
            Some(&vec![("__syn_fn_leaf".to_string(), 2)])
        );

        let mut visiting = BTreeSet::new();
        let mut memo = BTreeMap::new();
        let mut has_cycle = false;
        let (sw_total, path) = walk_worst_stack(
            "__syn_avr_vec_11",
            &frames,
            &graph,
            &BTreeSet::new(),
            &mut visiting,
            &mut memo,
            &mut has_cycle,
        );
        assert!(!has_cycle);
        // 5 + 2 + 22 + 2 + 0 = 31 B (+ 2 B HW vector PC = 33 B)
        assert_eq!(sw_total, 31);
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn dag_walker_guards_recursive_cycles_safely() {
        let frames = BTreeMap::from([("a".to_string(), 16), ("b".to_string(), 24)]);
        let mut graph = EdgeGraph::new();
        add_edge(&mut graph, "a", "b", 0);
        add_edge(&mut graph, "b", "a", 0);

        let mut visiting = BTreeSet::new();
        let mut memo = BTreeMap::new();
        let mut has_cycle = false;
        let (total, path) = walk_worst_stack(
            "a",
            &frames,
            &graph,
            &BTreeSet::new(),
            &mut visiting,
            &mut memo,
            &mut has_cycle,
        );
        assert!(has_cycle);
        assert_eq!(total, 40);
        assert!(path.last().unwrap().contains("[cycle]"));
    }

    #[test]
    fn avr_z_register_outgoing_stack_args_and_icall_detection() {
        let sample_disasm = r#"
00000100 <__syn_fn_caller>:
     100: cf 93        	push	r28
     102: df 93        	push	r29
     104: ed b7        	in	r30, 0x3d
     106: fe b7        	in	r31, 0x3e
     108: 38 97        	sbiw	r30, 0x08
     10a: ef bf        	out	0x3d, r30
     10c: 0e 94 00 01  	call	0x200 <__syn_fn_big>
     110: ed b7        	in	r30, 0x3d
     112: 38 96        	adiw	r30, 0x08
     114: ef bf        	out	0x3d, r30
     116: ed b7        	in	r30, 0x3d
     118: e6 50        	subi	r30, 0x06
     11a: f0 40        	sbci	r31, 0x00
     11c: ef bf        	out	0x3d, r30
     11e: 09 95        	icall
     120: 08 95        	ret
     122: 0b d0        	rcall	.+22

0000013a <__syn_fn_leaf>:
     13a: 08 95        	ret

00000140 <__syn_fn_uncalled_sp>:
     140: ed b7        	in	r30, 0x3d
     142: 3c 97        	sbiw	r30, 0x0c
     144: ef bf        	out	0x3d, r30
     146: 08 95        	ret

00000200 <__syn_fn_big>:
     200: 08 95        	ret
"#;
        let disasm = parse_objdump_disassembly(sample_disasm);
        let (frames, graph, indirect) = analyze_avr_disasm(&disasm, &BTreeSet::new());

        assert_eq!(frames.get("__syn_fn_caller"), Some(&2));
        assert_eq!(frames.get("__syn_fn_uncalled_sp"), Some(&12));
        // `call 0x200` has 2B return addr + 8B outgoing stack args; `rcall` after `ret` resets back to 2B.
        assert_eq!(
            graph.get("__syn_fn_caller"),
            Some(&vec![
                ("__syn_fn_big".to_string(), 10),
                ("__syn_fn_leaf".to_string(), 2),
            ])
        );
        // `icall` after `subi 6`/`sbci 0` carries 2 + 6 = 8B cost.
        assert_eq!(indirect.get("__syn_fn_caller"), Some(&8));
    }

    #[test]
    fn indirect_call_detection_for_ll_thumb_and_riscv() {
        let ll = r#"
define void @__syn_fn_dispatch(ptr %cb) {
entry:
  call void %cb()
  ret void
}
define void @__syn_fn_asm_only() {
entry:
  call void asm sideeffect "nop", ""()
  ret void
}
define void @__syn_sched_loop(ptr %step) {
entry:
  call void %step(ptr null)
  ret void
}
define void @__syn_rom_call_u32_a2_i32_i32_sb32(i32 %fn_addr, i32 %a0, i32 %a1) {
entry:
  %fp = inttoptr i32 %fn_addr to ptr
  %r = call i32 %fp(i32 %a0, i32 %a1)
  ret void
}
define void @__syn_rom_noreturn_a2_i32_i32(i32 %fn_addr, i32 %a0, i32 %a1) {
entry:
  %fp = inttoptr i32 %fn_addr to ptr
  call void %fp(i32 %a0, i32 %a1)
  unreachable
}
"#;
        let ll_indirect = parse_ll_indirect_callers(ll);
        assert!(ll_indirect.contains("__syn_fn_dispatch"));
        assert!(!ll_indirect.contains("__syn_fn_asm_only"));
        assert!(!ll_indirect.contains("__syn_sched_loop"));
        assert!(!ll_indirect.contains("__syn_rom_call_u32_a2_i32_i32_sb32"));
        assert!(!ll_indirect.contains("__syn_rom_noreturn_a2_i32_i32"));
        assert_eq!(
            parse_rom_call_stack_bound("__syn_rom_call_u32_a2_i32_i32_sb32"),
            Some(32)
        );
        assert_eq!(
            parse_rom_call_stack_bound("__syn_rom_noreturn_a2_i32_i32"),
            None
        );

        let mut sb_frames = BTreeMap::from([("__syn_boot".to_string(), 8u64)]);
        if let Some(sb) = parse_rom_call_stack_bound("__syn_rom_call_u32_a2_i32_i32_sb32") {
            sb_frames.insert("__syn_rom_call_u32_a2_i32_i32_sb32".to_string(), sb);
        }
        let mut sb_graph = EdgeGraph::new();
        add_edge(
            &mut sb_graph,
            "__syn_boot",
            "__syn_rom_call_u32_a2_i32_i32_sb32",
            0,
        );
        let mut visiting = BTreeSet::new();
        let mut memo = BTreeMap::new();
        let mut has_cycle = false;
        let (depth, path) = walk_worst_stack(
            "__syn_boot",
            &sb_frames,
            &sb_graph,
            &BTreeSet::new(),
            &mut visiting,
            &mut memo,
            &mut has_cycle,
        );
        assert_eq!(depth, 40);
        assert!(path
            .iter()
            .any(|p| p.contains("__syn_rom_call_u32_a2_i32_i32_sb32 (32B)")));

        let thumb_disasm = r#"
00000100 <__syn_fn_dispatch>:
     100: 98 47        	blx	r3
     102: 70 47        	bx	lr
00000110 <__syn_fn_tail>:
     110: 18 47        	bx	r3
00000120 <__syn_sched_loop>:
     120: 98 47        	blx	r3
     122: 70 47        	bx	lr
"#;
        let mut ll_thumb = ll_indirect.clone();
        ll_thumb.insert("__syn_fn_tail".to_string());
        let thumb_parsed = parse_objdump_disassembly(thumb_disasm);
        let (_, thumb_indirect) = build_arm_or_rv_call_graph(
            CpuIsa::ArmCortexM,
            &thumb_parsed,
            "",
            &BTreeMap::new(),
            &ll_thumb,
        );
        assert!(thumb_indirect.contains("__syn_fn_dispatch"));
        assert!(thumb_indirect.contains("__syn_fn_tail"));
        assert!(!thumb_indirect.contains("__syn_sched_loop"));

        let rv_disasm = r#"
42000100 <__syn_fn_dispatch>:
42000100: 000500e7     	jalr	a0
42000104: 00008067     	ret
42000110 <__syn_fn_ret_only>:
42000110: 00008067     	jalr	ra
42000120 <__syn_fn_tail>:
42000120: 00028067     	jr	t0
"#;
        let rv_parsed = parse_objdump_disassembly(rv_disasm);
        let (_, rv_indirect) = build_arm_or_rv_call_graph(
            CpuIsa::Riscv32,
            &rv_parsed,
            "",
            &BTreeMap::new(),
            &ll_thumb,
        );
        assert!(rv_indirect.contains("__syn_fn_dispatch"));
        assert!(rv_indirect.contains("__syn_fn_tail"));
        assert!(!rv_indirect.contains("__syn_fn_ret_only"));
    }
}
