//! Hardware Single-Stack + `.data`/`.bss` High-Water Mark Proof (`stack_sizes`).
//!
//! Computes exact worst-case bare-metal SRAM consumption across:
//! - **ARM Cortex-M3 (`thumbv7m-none-eabi`, `65,536 B` SRAM)**:
//!   `/bin/opt` -> `/bin/llc --stack-size-section` -> `/bin/ld.lld` -> `/bin/llvm-size`
//!   + `/bin/llvm-readobj --stack-sizes` + `/bin/llvm-objdump -d` + `32 B` NVIC HW exception frame.
//! - **Espressif ESP32-C6 (`riscv32-unknown-none-elf`, `524,288 B` HP-SRAM)**:
//!   `/bin/opt` -> `/bin/llc --stack-size-section` -> `/bin/ld.lld` -> `/bin/llvm-size`
//!   + `/bin/llvm-readobj --stack-sizes` + `/bin/llvm-objdump -d` + `64 B` trap context frame.
//! - **Microchip ATmega328P (`avr`, `2,048 B` SRAM)**:
//!   `/bin/opt` -> `/bin/llc -mcpu=atmega328p` -> `/bin/ld.lld` -> `/bin/llvm-size`
//!   + `/bin/llvm-objdump -d` AVR prologue/call-edge analysis (`push` depth + `2 * rcall .+0`
//!   + `sbiw r28, K` / `subi+sbci` SP subtracts + `2 B` return-PC per call + `2 B` vector push).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use zev_mir::MirModule;

use crate::esp32c6::assign_esp32c6_irq_routes;
use crate::nvic::sanitize_ident;
use crate::toolchain::{
    CodegenError, CodegenOptions, TargetArch, inspect_elf_sizes, link_obj_to_elf,
    optimize_ll_with_opt, synthesize_linker_script,
};

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Hardware SRAM budget in bytes for supported bare-metal targets.
pub const fn target_sram_budget_bytes(target: TargetArch) -> Option<u64> {
    match target {
        TargetArch::Thumbv7mNoneEabi => Some(65_536),
        TargetArch::Riscv32Esp32C6 => Some(524_288),
        TargetArch::AvrAtmega328p => Some(2_048),
        TargetArch::AvrAtmega4809 => Some(6_144),
        TargetArch::HostX86_64Linux => None,
    }
}

/// Hardware single-stack + `.data`/`.bss` high-water proof for a compiled bare-metal ELF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareStackProof {
    pub target: TargetArch,
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
    /// Human-readable hardware stack + `.data`/`.bss` high-water proof table for `zevc ram --target`.
    pub fn format_table(&self) -> String {
        let (target_label, analyzer_label) = match self.target {
            TargetArch::Thumbv7mNoneEabi => (
                "ARM Cortex-M3 (thumbv7m-none-eabi)",
                "LLVM .stack_sizes + ELF call graph + 32B NVIC frame",
            ),
            TargetArch::Riscv32Esp32C6 => (
                "Espressif ESP32-C6 (riscv32-unknown-none-elf, rv32imac)",
                "LLVM .stack_sizes + ELF call graph + 64B trap context frame",
            ),
            TargetArch::AvrAtmega328p => (
                "Microchip ATmega328P AVR (avr / Arduino Uno R3)",
                "AVR ELF disassembly (push + sbiw/rcall.+0 + 2B return PC + 2B IRQ push)",
            ),
            TargetArch::AvrAtmega4809 => (
                "Microchip ATmega4809 AVR (avrxmega3 / Arduino Nano Every)",
                "AVR ELF disassembly (push + sbiw/rcall.+0 + 2B return PC + 2B IRQ push)",
            ),
            TargetArch::HostX86_64Linux => (
                "Host x86_64",
                "unsupported",
            ),
        };

        let mut out = String::new();
        out.push_str("\n================================================================================\n");
        let _ = writeln!(out, "  HARDWARE SINGLE-STACK + .BSS HIGH-WATER PROOF");
        let _ = writeln!(out, "  Target   : {target_label} [{}]", self.opt_level);
        let _ = writeln!(out, "  Analyzer : {analyzer_label}");
        out.push_str("================================================================================\n\n");

        out.push_str("1. Linked ELF Section Footprint (/bin/llvm-size)\n");
        out.push_str("--------------------------------------------------------------------------------\n");
        let _ = writeln!(out, "  {:<36} {:>10} B", "Flash Code (.text)", self.text_bytes);
        let _ = writeln!(out, "  {:<36} {:>10} B", "Initialized SRAM (.data)", self.data_bytes);
        let _ = writeln!(out, "  {:<36} {:>10} B", "Zero-Init SRAM (.bss)", self.bss_bytes);
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "Static SRAM (.data + .bss)", self.static_sram_bytes
        );
        out.push('\n');

        out.push_str("2. Worst-Case Hardware Call Stack Proof\n");
        out.push_str("--------------------------------------------------------------------------------\n");
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
        out.push_str("--------------------------------------------------------------------------------\n");
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "Static SRAM (.data + .bss)", self.static_sram_bytes
        );
        let _ = writeln!(
            out,
            "  {:<36} {:>10} B",
            "Peak Hardware Stack", self.peak_stack_bytes
        );
        out.push_str("  ------------------------------------------------------------------------------\n");
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
        let verdict = if self.within_budget {
            format!("PASS ({margin} B headroom remaining)")
        } else {
            format!("FAIL (exceeds SRAM budget by {} B!)", -margin)
        };
        let _ = writeln!(out, "  {:<36}   {verdict}", "Budget Status");
        out.push_str("================================================================================\n");
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

fn find_tool(preferred_abs: &'static str, fallback_name: &'static str) -> &'static str {
    if Path::new(preferred_abs).exists() {
        preferred_abs
    } else {
        fallback_name
    }
}

fn compile_ll_to_obj_with_stack_sizes(
    ll_path: &Path,
    opt_ll_path: &Path,
    obj_path: &Path,
    opts: &CodegenOptions,
) -> Result<PathBuf, CodegenError> {
    let llc_flag = opts.llc_opt_flag()?;
    let llc_input: PathBuf = if opts.is_opt_o0() {
        ll_path.to_path_buf()
    } else {
        optimize_ll_with_opt(ll_path, opt_ll_path, opts)?;
        opt_ll_path.to_path_buf()
    };

    let llc = find_tool("/bin/llc", "llc");
    let mut cmd = Command::new(llc);
    cmd.arg(format!("-mtriple={}", opts.target.triple()));
    match opts.target {
        TargetArch::Thumbv7mNoneEabi => {
            cmd.arg("-mcpu=cortex-m3")
                .arg("-function-sections")
                .arg("-data-sections")
                .arg("--stack-size-section");
        }
        TargetArch::Riscv32Esp32C6 => {
            cmd.arg("-mcpu=generic-rv32")
                .arg("-mattr=+m,+a,+c")
                .arg("-relocation-model=static")
                .arg("-function-sections")
                .arg("-data-sections")
                .arg("--stack-size-section");
        }
        TargetArch::AvrAtmega328p | TargetArch::AvrAtmega4809 => {
            cmd.arg(format!("-mcpu={}", opts.target.avr_mcpu()))
                .arg("-function-sections")
                .arg("-data-sections");
        }
        TargetArch::HostX86_64Linux => {}
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
    let llvm_size = find_tool("/bin/llvm-size", "llvm-size");
    let out = Command::new(llvm_size)
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
    let readobj = find_tool("/bin/llvm-readobj", "llvm-readobj");
    let out = Command::new(readobj)
        .arg("--stack-sizes")
        .arg(obj_path)
        .output()
        .map_err(|e| {
            CodegenError::new(
                "llvm-readobj",
                format!("failed to invoke `{readobj}`: {e}"),
            )
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
    let objdump = find_tool("/bin/llvm-objdump", "llvm-objdump");
    let out = Command::new(objdump)
        .arg("-d")
        .arg(elf_path)
        .output()
        .map_err(|e| {
            CodegenError::new(
                "llvm-objdump",
                format!("failed to invoke `{objdump}`: {e}"),
            )
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
            Some(pos) => (insn[..pos].to_ascii_lowercase(), insn[pos..].trim().to_string()),
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

fn parse_ll_call_edges(ll_text: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut cur_fn: Option<String> = None;

    for line in ll_text.lines() {
        let t = line.trim();
        if t.starts_with("define ") && t.ends_with('{') {
            if let Some(at) = t.find('@') {
                let after = &t[at + 1..];
                if let Some(paren) = after.find('(') {
                    let name = after[..paren].trim().trim_matches('"').to_string();
                    cur_fn = Some(name);
                }
            }
            continue;
        }
        if t == "}" {
            cur_fn = None;
            continue;
        }
        let Some(ref caller) = cur_fn else {
            continue;
        };
        if t.starts_with(';') || t.contains(" asm ") {
            continue;
        }
        let has_call = t.starts_with("call ")
            || t.starts_with("tail call ")
            || t.starts_with("musttail call ")
            || t.contains(" = call ")
            || t.contains(" = tail call ")
            || t.contains(" = musttail call ");
        if !has_call {
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

fn parse_imm_u64(tok: &str) -> Option<u64> {
    let s = tok.trim().trim_end_matches(',');
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

fn parse_signed_rel_offset(tok: &str) -> Option<i64> {
    let s = tok.trim().split_whitespace().next()?;
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

fn extract_esp32c6_tramp_sp_alloc(insns: &[(u64, String, String)]) -> Option<u64> {
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

fn analyze_avr_disasm(
    disasm: &DisasmData,
) -> (BTreeMap<String, u64>, EdgeGraph) {
    let mut frame_sizes = BTreeMap::new();
    let mut graph = EdgeGraph::new();

    for (fn_name, insns) in &disasm.fn_insns {
        let mut frame_bytes = 0u64;
        let mut in_prologue = true;
        let mut pending_subi_lo: Option<u8> = None;
        let mut pending_sbci_hi: Option<u8> = None;
        let mut saw_sp_read = false;

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
                            if let Some(v) = parse_imm_u64(k_tok).and_then(|x| u8::try_from(x).ok()) {
                                pending_subi_lo = Some(v);
                            }
                        }
                    }
                    "sbci" if saw_sp_read && ops.starts_with("r29,") => {
                        if let Some(k_tok) = ops.split(',').nth(1) {
                            if let Some(v) = parse_imm_u64(k_tok).and_then(|x| u8::try_from(x).ok()) {
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

            // 2. Call & tail-call edges (skip vector table `_start` jumps)
            if fn_name == "_start" {
                continue;
            }
            match mnem.as_str() {
                "call" => {
                    if let Some(target_addr) = parse_imm_u64(ops.split_whitespace().next().unwrap_or("")) {
                        if let Some((_, sym)) = disasm.addr_to_symbol.range(..=target_addr).next_back() {
                            if !sym.starts_with('.') {
                                add_edge(&mut graph, fn_name, sym, 2);
                            }
                        }
                    }
                }
                "rcall" => {
                    if let Some(off) = parse_signed_rel_offset(ops) {
                        if off != 0 {
                            let target_addr = ((*addr as i64) + 2 + off) as u64;
                            if let Some((_, sym)) = disasm.addr_to_symbol.range(..=target_addr).next_back() {
                                if !sym.starts_with('.') {
                                    add_edge(&mut graph, fn_name, sym, 2);
                                }
                            }
                        }
                    }
                }
                "jmp" => {
                    if let Some(target_addr) = parse_imm_u64(ops.split_whitespace().next().unwrap_or("")) {
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
        }

        frame_sizes.insert(fn_name.clone(), frame_bytes);
    }

    (frame_sizes, graph)
}

fn build_arm_or_rv_call_graph(
    target: TargetArch,
    disasm: &DisasmData,
    ll_text: &str,
    frame_sizes: &BTreeMap<String, u64>,
) -> EdgeGraph {
    let mut graph = EdgeGraph::new();

    for (fn_name, insns) in &disasm.fn_insns {
        if fn_name == "__syn_rv_vector_table" {
            continue;
        }
        for (_, mnem, ops) in insns {
            let is_call = match target {
                TargetArch::Thumbv7mNoneEabi => matches!(mnem.as_str(), "bl" | "bl.w" | "blx"),
                TargetArch::Riscv32Esp32C6 => {
                    matches!(mnem.as_str(), "jal" | "jalr" | "c.jal" | "c.jalr" | "call")
                }
                _ => false,
            };
            let is_tail_branch = match target {
                TargetArch::Thumbv7mNoneEabi => matches!(mnem.as_str(), "b" | "b.w" | "b.n"),
                TargetArch::Riscv32Esp32C6 => matches!(mnem.as_str(), "j" | "c.j" | "tail"),
                _ => false,
            };
            if is_call {
                if let Some(sym) = extract_angle_symbol(ops) {
                    add_edge(&mut graph, fn_name, sym, 0);
                }
            } else if is_tail_branch {
                if let Some(sym) = extract_angle_symbol(ops) {
                    if sym != fn_name && !sym.starts_with('.') && disasm.elf_symbols.contains(sym) {
                        add_edge(&mut graph, fn_name, sym, 0);
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

    graph
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

/// Compile `mir` for `opts.target` (`Thumbv7mNoneEabi`, `Riscv32Esp32C6`, `AvrAtmega328p`, or `AvrAtmega4809`)
/// and prove worst-case hardware single-stack + `.data`/`.bss` SRAM high-water mark against target limits.
pub fn analyze_hardware_stack_high_water(
    mir: &MirModule,
    opts: &CodegenOptions,
) -> Result<HardwareStackProof, CodegenError> {
    let sram_budget_bytes = target_sram_budget_bytes(opts.target).ok_or_else(|| {
        CodegenError::new(
            "stack-sizes",
            "hardware stack high-water proof only supports bare-metal targets (`cortex-m3`, `esp32c6`, `avr`, `nano-every`)",
        )
    })?;

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
    let ld_script = synthesize_linker_script(opts);
    fs::write(&ld_path, &ld_script).map_err(|e| {
        CodegenError::new("io", format!("failed to write {}: {e}", ld_path.display()))
    })?;

    let used_ll_path = compile_ll_to_obj_with_stack_sizes(&ll_path, &opt_ll_path, &obj_path, opts)?;
    link_obj_to_elf(&obj_path, &ld_path, &elf_path)?;

    let (text_bytes, data_bytes, bss_bytes) = inspect_elf_text_data_bss(&elf_path)?;
    let static_sram_bytes = data_bytes.saturating_add(bss_bytes);

    let disasm_text = run_objdump_disasm(&elf_path)?;
    let disasm = parse_objdump_disassembly(&disasm_text);
    let ll_text = fs::read_to_string(&used_ll_path).unwrap_or(raw_ll);

    let (mut frame_sizes, mut graph) = match opts.target {
        TargetArch::Thumbv7mNoneEabi | TargetArch::Riscv32Esp32C6 => {
            let readobj_raw = run_readobj_stack_sizes(&obj_path)?;
            let mut frames = parse_llvm_readobj_stack_sizes(&readobj_raw);
            if opts.target == TargetArch::Riscv32Esp32C6 {
                for (sym, insns) in &disasm.fn_insns {
                    if sym.starts_with("__syn_rv_irq_tramp_") || sym == "__syn_esp32c6_trap_entry" {
                        let parsed_alloc = extract_esp32c6_tramp_sp_alloc(insns).unwrap_or(64);
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
            let g = build_arm_or_rv_call_graph(opts.target, &disasm, &ll_text, &frames);
            (frames, g)
        }
        TargetArch::AvrAtmega328p | TargetArch::AvrAtmega4809 => analyze_avr_disasm(&disasm),
        TargetArch::HostX86_64Linux => unreachable!(),
    };

    // Wire indirect task-step dispatch edges from `__syn_sched_loop` -> `__syn_task_step_<bp>`
    let step_edge_cost = if opts.target.is_avr() {
        2
    } else {
        0
    };
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
    let (boot_stack_bytes, boot_worst_path) = if disasm.elf_symbols.contains("__syn_boot")
        || frame_sizes.contains_key("__syn_boot")
    {
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
            "__syn_esp32c6_reset",
            "__syn_avr_entry",
            "__syn_avr_mega0_entry",
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
        match opts.target {
            TargetArch::Thumbv7mNoneEabi => {
                let hw_frame = 32u64;
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
                    "<HW NVIC frame (32B)>",
                    &frame_sizes,
                    &graph,
                    &mut visiting,
                    &mut memo,
                    &mut has_cycle,
                )
            }
            TargetArch::Riscv32Esp32C6 => {
                let routes = assign_esp32c6_irq_routes(&mir.irq_vectors)?;
                if routes.is_empty()
                    && !disasm.elf_symbols.contains("__syn_systick_isr")
                    && !disasm.elf_symbols.contains("__syn_esp32c6_trap_entry")
                {
                    (0, "(none)".to_string(), Vec::new(), 0)
                } else {
                    let mut tramp_candidates = Vec::new();
                    let cpu_ints: BTreeSet<u8> = routes.iter().map(|r| r.cpu_int).collect();
                    for c in cpu_ints {
                        let tramp = format!("__syn_rv_irq_tramp_{c}");
                        if disasm.elf_symbols.contains(&tramp) || frame_sizes.contains_key(&tramp) {
                            tramp_candidates.push(tramp);
                        }
                    }
                    if disasm.elf_symbols.contains("__syn_esp32c6_trap_entry") {
                        tramp_candidates.push("__syn_esp32c6_trap_entry".to_string());
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
            TargetArch::AvrAtmega328p | TargetArch::AvrAtmega4809 => {
                let hw_frame = 2u64;
                let mut candidates = Vec::new();
                if disasm.elf_symbols.contains("__syn_avr_systick_isr")
                    || frame_sizes.contains_key("__syn_avr_systick_isr")
                {
                    candidates.push("__syn_avr_systick_isr".to_string());
                }
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
            TargetArch::HostX86_64Linux => unreachable!(),
        };

    // Retain only non-dot ELF symbols in `frame_sizes` for clean reporting
    frame_sizes.retain(|k, _| !k.starts_with('.'));

    let peak_stack_bytes = boot_stack_bytes
        .max(sched_task_stack_bytes)
        .saturating_add(worst_isr_stack_bytes);
    let total_high_water_bytes = static_sram_bytes.saturating_add(peak_stack_bytes);
    let within_budget = total_high_water_bytes <= sram_budget_bytes;

    Ok(HardwareStackProof {
        target: opts.target,
        opt_level: opts.opt_level.clone(),
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
        let (frames, graph) = analyze_avr_disasm(&disasm);
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
}
