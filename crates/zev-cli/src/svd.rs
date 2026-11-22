use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;

// ─────────────────────────────────────────────────────────────────────────────
// Public CLI Entry Points
// ─────────────────────────────────────────────────────────────────────────────

/// Execute `zevc import-svd <chip.svd> [-o <out.zev>]`.
pub fn run_import_svd(svd_path: &Path, output: Option<&Path>) -> ExitCode {
    let xml = match std::fs::read_to_string(svd_path) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("error: failed to read `{}`: {err}", svd_path.display());
            return ExitCode::FAILURE;
        }
    };

    let zev = match svd_to_zev(&xml) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("error: failed to parse SVD `{}`: {err}", svd_path.display());
            return ExitCode::FAILURE;
        }
    };

    match output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    if let Err(err) = std::fs::create_dir_all(parent) {
                        eprintln!(
                            "error: failed to create directory `{}`: {err}",
                            parent.display()
                        );
                        return ExitCode::FAILURE;
                    }
                }
            }
            if let Err(err) = std::fs::write(path, zev) {
                eprintln!("error: failed to write `{}`: {err}", path.display());
                return ExitCode::FAILURE;
            }
        }
        None => {
            print!("{zev}");
        }
    }

    ExitCode::SUCCESS
}

/// Parse a CMSIS-SVD XML string and emit a deterministic, Wave-1-valid `.zev` hardware map.
pub fn svd_to_zev(xml: &str) -> Result<String, String> {
    let root = parse_xml_document(xml)?;
    if root.name != "device" {
        return Err(format!(
            "expected root `<device>` element in SVD, found `<{}>`",
            root.name
        ));
    }

    let device_name = root
        .child_text("name")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown");

    let device_props = RegProps::default().merge(&root)?;

    let peripherals_el = root
        .child("peripherals")
        .ok_or_else(|| "missing `<peripherals>` element in SVD `<device>`".to_string())?;

    let mut raw_periphs: Vec<RawPeripheral<'_>> = Vec::new();
    let mut name_to_idx: BTreeMap<String, usize> = BTreeMap::new();

    for p_el in peripherals_el.children("peripheral") {
        let name = p_el
            .child_text("name")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "missing `<name>` in `<peripheral>`".to_string())?
            .to_string();

        let derived_from = p_el
            .attr("derivedFrom")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        let base_address = match p_el.child_text("baseAddress") {
            Some(raw) => Some(parse_svd_u32(raw, "<baseAddress>")?),
            None => None,
        };

        let props = device_props.merge(p_el)?;

        let mut interrupts = Vec::new();
        for irq_el in p_el.children("interrupt") {
            let Some(irq_raw_name) = irq_el.child_text("name") else {
                continue;
            };
            let Some(val_raw) = irq_el.child_text("value") else {
                continue;
            };
            // Cortex-M core exceptions can carry negative numbers; Zev `irq` tokens are `u32`.
            let trimmed_val = val_raw.trim();
            if trimmed_val.starts_with('-') {
                continue;
            }
            let irq_num = parse_svd_u32(trimmed_val, "<interrupt><value>")?;
            interrupts.push(RawInterrupt {
                name: irq_raw_name.to_string(),
                num: irq_num,
            });
        }

        let idx = raw_periphs.len();
        name_to_idx.entry(name.clone()).or_insert(idx);
        raw_periphs.push(RawPeripheral {
            name,
            derived_from,
            base_address,
            props,
            registers_el: p_el.child("registers"),
            interrupts,
        });
    }

    // Resolve `derivedFrom` chains cleanly and detect cycles or missing bases.
    let mut resolved_base_idx: Vec<Option<usize>> = Vec::with_capacity(raw_periphs.len());
    let mut resolved_addrs: Vec<u32> = Vec::with_capacity(raw_periphs.len());

    for i in 0..raw_periphs.len() {
        let layout_owner = resolve_layout_owner(i, &raw_periphs, &name_to_idx)?;
        let base_addr = resolve_base_address(i, &raw_periphs, &name_to_idx)?;
        resolved_base_idx.push(layout_owner);
        resolved_addrs.push(base_addr);
    }

    // Collect deduplicated interrupts across all peripherals.
    let mut irqs: Vec<(String, u32)> = Vec::new();
    let mut seen_irq_names: BTreeMap<String, u32> = BTreeMap::new();
    for p in &raw_periphs {
        for irq in &p.interrupts {
            let ident = format_irq_ident(&irq.name);
            match seen_irq_names.get(&ident).copied() {
                Some(existing) if existing == irq.num => {}
                Some(_) => {
                    let disambig = format!("{ident}_{}", irq.num);
                    if !seen_irq_names.contains_key(&disambig) {
                        seen_irq_names.insert(disambig.clone(), irq.num);
                        irqs.push((disambig, irq.num));
                    }
                }
                None => {
                    seen_irq_names.insert(ident.clone(), irq.num);
                    irqs.push((ident, irq.num));
                }
            }
        }
    }
    irqs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    // Reserve all top-level names to prevent layout/instance/irq/const collisions.
    let mut used_top_names: HashSet<String> = HashSet::new();
    for (irq_name, _) in &irqs {
        used_top_names.insert(irq_name.clone());
    }

    // Determine instance identifiers first so layouts never collide with instances.
    let mut inst_names: Vec<String> = Vec::with_capacity(raw_periphs.len());
    for p in &raw_periphs {
        let mut inst = sanitize_ident(&p.name);
        if used_top_names.contains(&inst) {
            inst = make_unique_name(&inst, &used_top_names);
        }
        used_top_names.insert(inst.clone());
        inst_names.push(inst);
    }

    // Build shared layouts for every peripheral that owns `<registers>`.
    let mut layouts_by_owner: BTreeMap<usize, BuiltLayout> = BTreeMap::new();
    let mut used_consts: HashSet<String> = used_top_names.clone();

    for (idx, p) in raw_periphs.iter().enumerate() {
        let Some(regs_el) = p.registers_el else {
            continue;
        };

        let mut layout_name = to_layout_name(&p.name);
        if used_top_names.contains(&layout_name) {
            layout_name = make_unique_name(&layout_name, &used_top_names);
        }
        used_top_names.insert(layout_name.clone());
        used_consts.insert(layout_name.clone());

        let mut candidates = Vec::new();
        let mut seq = 0usize;
        collect_registers_from_container(
            regs_el,
            0,
            "",
            "",
            &p.props,
            &p.name,
            true,
            &mut candidates,
            &mut seq,
        )?;

        let (regs, consts) = finalize_layout_registers(candidates, &mut used_consts);
        if !regs.is_empty() {
            layouts_by_owner.insert(
                idx,
                BuiltLayout {
                    layout_name,
                    regs,
                    consts,
                    instances: Vec::new(),
                },
            );
        }
    }

    // Attach all peripheral instances (including `derivedFrom` ones) to their shared layout.
    for (idx, owner_opt) in resolved_base_idx.iter().enumerate() {
        if let Some(owner_idx) = owner_opt {
            if let Some(layout) = layouts_by_owner.get_mut(owner_idx) {
                layout.instances.push(BuiltInstance {
                    name: inst_names[idx].clone(),
                    base_addr: resolved_addrs[idx],
                });
            }
        }
    }

    // Render deterministic `.zev` output.
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// Auto-generated from CMSIS-SVD ({device_name}) by `zevc import-svd`."
    );
    out.push('\n');

    if !irqs.is_empty() {
        out.push_str("// ── Hardware Interrupts ──────────────────────────────────────────────────────\n");
        for (name, num) in &irqs {
            let _ = writeln!(out, "pub irq {name} = {num};");
        }
        out.push('\n');
    }

    for layout in layouts_by_owner.values() {
        let _ = writeln!(out, "pub mmio {} {{", layout.layout_name);
        for reg in &layout.regs {
            match reg.array_len {
                Some(n) => {
                    let _ = writeln!(
                        out,
                        "    {} [{n}]u{} {} @ {};",
                        reg.access.as_str(),
                        reg.elem_bits,
                        reg.name,
                        format_hex_offset(reg.offset),
                    );
                }
                None => {
                    let _ = writeln!(
                        out,
                        "    {} u{} {} @ {};",
                        reg.access.as_str(),
                        reg.elem_bits,
                        reg.name,
                        format_hex_offset(reg.offset),
                    );
                }
            }
        }
        out.push_str("}\n\n");

        for inst in &layout.instances {
            let _ = writeln!(
                out,
                "pub mmio {} : {} @ {};",
                inst.name,
                layout.layout_name,
                format_hex_u32(inst.base_addr),
            );
        }
        if !layout.instances.is_empty() {
            out.push('\n');
        }

        for c in &layout.consts {
            let _ = writeln!(out, "pub const u32 {}_POS = {};", c.base_name, c.pos);
            let _ = writeln!(
                out,
                "pub const u32 {}_MASK = {};",
                c.base_name,
                format_hex_u32(c.mask)
            );
        }
        if !layout.consts.is_empty() {
            out.push('\n');
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// SVD Internal Representation & Resolution
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MmioAccess {
    Rw,
    Ro,
    Wo,
    W1c,
}

impl MmioAccess {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Rw => "@rw",
            Self::Ro => "@ro",
            Self::Wo => "@wo",
            Self::W1c => "@w1c",
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RegProps {
    size_bits: Option<u32>,
    access: Option<String>,
    modified_write_values: Option<String>,
}

impl RegProps {
    fn merge(&self, el: &XmlElement) -> Result<Self, String> {
        let size_bits = match el.child_text("size") {
            Some(s) => Some(parse_svd_u32(s, "<size>")?),
            None => self.size_bits,
        };
        let access = el
            .child_text("access")
            .map(|s| s.trim().to_string())
            .or_else(|| self.access.clone());
        let modified_write_values = el
            .child_text("modifiedWriteValues")
            .map(|s| s.trim().to_string())
            .or_else(|| self.modified_write_values.clone());

        Ok(Self {
            size_bits,
            access,
            modified_write_values,
        })
    }
}

fn resolve_access(props: &RegProps) -> MmioAccess {
    let access = props.access.as_deref().unwrap_or("read-write").trim();
    if access == "read-only" {
        MmioAccess::Ro
    } else if props
        .modified_write_values
        .as_deref()
        .map(str::trim)
        == Some("oneToClear")
    {
        MmioAccess::W1c
    } else if access == "write-only" || access == "writeOnce" {
        MmioAccess::Wo
    } else {
        MmioAccess::Rw
    }
}

struct RawInterrupt {
    name: String,
    num: u32,
}

struct RawPeripheral<'a> {
    name: String,
    derived_from: Option<String>,
    base_address: Option<u32>,
    props: RegProps,
    registers_el: Option<&'a XmlElement>,
    interrupts: Vec<RawInterrupt>,
}

#[derive(Clone)]
struct CandidateConst {
    base_name: String,
    pos: u32,
    mask: u32,
}

struct CandidateReg {
    name: String,
    offset: u32,
    elem_bits: u8,
    array_len: Option<usize>,
    access: MmioAccess,
    seq: usize,
    consts: Vec<CandidateConst>,
}

struct EmittedReg {
    name: String,
    offset: u32,
    elem_bits: u8,
    array_len: Option<usize>,
    access: MmioAccess,
}

struct BuiltInstance {
    name: String,
    base_addr: u32,
}

struct BuiltLayout {
    layout_name: String,
    regs: Vec<EmittedReg>,
    consts: Vec<CandidateConst>,
    instances: Vec<BuiltInstance>,
}

fn resolve_layout_owner(
    start: usize,
    periphs: &[RawPeripheral<'_>],
    by_name: &BTreeMap<String, usize>,
) -> Result<Option<usize>, String> {
    let mut cur = start;
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(cur) {
            return Err(format!(
                "cyclic `derivedFrom` chain starting at peripheral `{}`",
                periphs[start].name
            ));
        }
        if periphs[cur].registers_el.is_some() {
            return Ok(Some(cur));
        }
        let Some(ref target) = periphs[cur].derived_from else {
            return Ok(None);
        };
        cur = *by_name.get(target).ok_or_else(|| {
            format!(
                "peripheral `{}` has `derivedFrom=\"{target}\"`, which does not exist",
                periphs[cur].name
            )
        })?;
    }
}

fn resolve_base_address(
    start: usize,
    periphs: &[RawPeripheral<'_>],
    by_name: &BTreeMap<String, usize>,
) -> Result<u32, String> {
    let mut cur = start;
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(cur) {
            return Err(format!(
                "cyclic `derivedFrom` chain resolving `<baseAddress>` at `{}`",
                periphs[start].name
            ));
        }
        if let Some(addr) = periphs[cur].base_address {
            return Ok(addr);
        }
        let Some(ref target) = periphs[cur].derived_from else {
            return Err(format!(
                "missing `<baseAddress>` on peripheral `{}`",
                periphs[start].name
            ));
        };
        cur = *by_name.get(target).ok_or_else(|| {
            format!(
                "peripheral `{}` references unknown `derivedFrom=\"{target}\"`",
                periphs[cur].name
            )
        })?;
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_registers_from_container(
    container: &XmlElement,
    base_offset: u64,
    cluster_prefix: &str,
    const_cluster_prefix: &str,
    parent_props: &RegProps,
    periph_name: &str,
    emit_consts: bool,
    out: &mut Vec<CandidateReg>,
    seq: &mut usize,
) -> Result<(), String> {
    for child in &container.children {
        match child.name.as_str() {
            "register" => {
                if child.child("alternateRegister").is_some() {
                    continue;
                }
                let Some(raw_name) = child.child_text("name") else {
                    continue;
                };
                let Some(rel_raw) = child.child_text("addressOffset") else {
                    continue;
                };
                let rel_offset = parse_svd_u64(rel_raw, "<addressOffset>")?;
                let props = parent_props.merge(child)?;
                let size_bits = props.size_bits.unwrap_or(32);
                if !matches!(size_bits, 8 | 16 | 32) {
                    continue;
                }
                let elem_bytes = (size_bits / 8) as u64;
                let access = resolve_access(&props);
                let Some(abs_offset) = base_offset.checked_add(rel_offset) else {
                    continue;
                };

                if let Some(dim_raw) = child.child_text("dim") {
                    let dim = parse_svd_u64(dim_raw, "<dim>")? as usize;
                    if dim == 0 {
                        continue;
                    }
                    let dim_inc = match child.child_text("dimIncrement") {
                        Some(inc_raw) => parse_svd_u64(inc_raw, "<dimIncrement>")?,
                        None => elem_bytes,
                    };
                    let dim_index = child.child_text("dimIndex");
                    let base_reg = strip_dim_placeholder(raw_name);
                    let const_reg_name = join_prefix(const_cluster_prefix, &base_reg);
                    let reg_consts = if emit_consts {
                        collect_register_consts(child, periph_name, &const_reg_name)?
                    } else {
                        Vec::new()
                    };

                    if dim_inc == elem_bytes {
                        let Ok(offset_u32) = u32::try_from(abs_offset) else {
                            continue;
                        };
                        let joined = join_prefix(cluster_prefix, &base_reg);
                        let reg_ident = sanitize_reg_ident(&joined);
                        *seq += 1;
                        out.push(CandidateReg {
                            name: reg_ident,
                            offset: offset_u32,
                            elem_bits: size_bits as u8,
                            array_len: Some(dim),
                            access,
                            seq: *seq,
                            consts: reg_consts,
                        });
                    } else {
                        let indices = parse_dim_indices(dim, dim_index);
                        for (k, idx_str) in indices.iter().enumerate() {
                            let Some(elem_offset) =
                                (k as u64).checked_mul(dim_inc).and_then(|d| abs_offset.checked_add(d))
                            else {
                                continue;
                            };
                            let Ok(offset_u32) = u32::try_from(elem_offset) else {
                                continue;
                            };
                            let expanded = substitute_dim_name(raw_name, idx_str, true);
                            let joined = join_prefix(cluster_prefix, &expanded);
                            let reg_ident = sanitize_reg_ident(&joined);
                            *seq += 1;
                            out.push(CandidateReg {
                                name: reg_ident,
                                offset: offset_u32,
                                elem_bits: size_bits as u8,
                                array_len: None,
                                access,
                                seq: *seq,
                                consts: reg_consts.clone(),
                            });
                        }
                    }
                } else {
                    let Ok(offset_u32) = u32::try_from(abs_offset) else {
                        continue;
                    };
                    let joined = join_prefix(cluster_prefix, raw_name);
                    let reg_ident = sanitize_reg_ident(&joined);
                    let const_reg_name = join_prefix(const_cluster_prefix, raw_name);
                    let reg_consts = if emit_consts {
                        collect_register_consts(child, periph_name, &const_reg_name)?
                    } else {
                        Vec::new()
                    };
                    *seq += 1;
                    out.push(CandidateReg {
                        name: reg_ident,
                        offset: offset_u32,
                        elem_bits: size_bits as u8,
                        array_len: None,
                        access,
                        seq: *seq,
                        consts: reg_consts,
                    });
                }
            }
            "cluster" => {
                if child.child("alternateCluster").is_some() {
                    continue;
                }
                let raw_name = child.child_text("name").unwrap_or("");
                let Some(rel_raw) = child.child_text("addressOffset") else {
                    continue;
                };
                let rel_offset = parse_svd_u64(rel_raw, "<cluster><addressOffset>")?;
                let props = parent_props.merge(child)?;
                let Some(abs_offset) = base_offset.checked_add(rel_offset) else {
                    continue;
                };

                if let Some(dim_raw) = child.child_text("dim") {
                    let dim = parse_svd_u64(dim_raw, "<cluster><dim>")? as usize;
                    if dim == 0 {
                        continue;
                    }
                    let dim_inc = match child.child_text("dimIncrement") {
                        Some(inc_raw) => parse_svd_u64(inc_raw, "<cluster><dimIncrement>")?,
                        None => 0,
                    };
                    let indices = parse_dim_indices(dim, child.child_text("dimIndex"));
                    let base_cluster = strip_dim_placeholder(raw_name);
                    let next_const_prefix = join_prefix(const_cluster_prefix, &base_cluster);
                    for (k, idx_str) in indices.iter().enumerate() {
                        let Some(inst_offset) =
                            (k as u64).checked_mul(dim_inc).and_then(|d| abs_offset.checked_add(d))
                        else {
                            continue;
                        };
                        let expanded = substitute_dim_name(raw_name, idx_str, true);
                        let next_prefix = join_prefix(cluster_prefix, &expanded);
                        collect_registers_from_container(
                            child,
                            inst_offset,
                            &next_prefix,
                            &next_const_prefix,
                            &props,
                            periph_name,
                            emit_consts && k == 0,
                            out,
                            seq,
                        )?;
                    }
                } else {
                    let next_prefix = join_prefix(cluster_prefix, raw_name);
                    let next_const_prefix = join_prefix(const_cluster_prefix, raw_name);
                    collect_registers_from_container(
                        child,
                        abs_offset,
                        &next_prefix,
                        &next_const_prefix,
                        &props,
                        periph_name,
                        emit_consts,
                        out,
                        seq,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn collect_register_consts(
    reg_el: &XmlElement,
    periph_name: &str,
    reg_name: &str,
) -> Result<Vec<CandidateConst>, String> {
    let mut out = Vec::new();
    let Some(fields_el) = reg_el.child("fields") else {
        return Ok(out);
    };

    for field_el in fields_el.children("field") {
        let Some(raw_field_name) = field_el.child_text("name") else {
            continue;
        };
        let Some((base_lsb, width)) = parse_field_bit_range(field_el)? else {
            continue;
        };
        if width == 0 || width > 32 {
            continue;
        }

        if let Some(dim_raw) = field_el.child_text("dim") {
            let dim = parse_svd_u64(dim_raw, "<field><dim>")? as usize;
            if dim == 0 {
                continue;
            }
            let dim_inc = match field_el.child_text("dimIncrement") {
                Some(inc) => parse_svd_u64(inc, "<field><dimIncrement>")? as u32,
                None => width,
            };
            let indices = parse_dim_indices(dim, field_el.child_text("dimIndex"));
            for (k, idx_str) in indices.iter().enumerate() {
                let Some(lsb) = (k as u32)
                    .checked_mul(dim_inc)
                    .and_then(|d| base_lsb.checked_add(d))
                else {
                    continue;
                };
                if lsb + width > 32 {
                    continue;
                }
                let expanded_field = substitute_dim_name(raw_field_name, idx_str, true);
                if let Some(c) = build_const(periph_name, reg_name, &expanded_field, lsb, width) {
                    out.push(c);
                }
            }
        } else {
            if base_lsb + width > 32 {
                continue;
            }
            if let Some(c) = build_const(periph_name, reg_name, raw_field_name, base_lsb, width) {
                out.push(c);
            }
        }
    }

    Ok(out)
}

fn build_const(
    periph_name: &str,
    reg_name: &str,
    field_name: &str,
    lsb: u32,
    width: u32,
) -> Option<CandidateConst> {
    let p_clean = clean_upper_segment(periph_name);
    let r_clean = clean_upper_segment(reg_name);
    let f_clean = clean_upper_segment(field_name);
    if f_clean.is_empty() {
        return None;
    }

    let mut combined = String::new();
    if !p_clean.is_empty() {
        combined.push_str(&p_clean);
    }
    if !r_clean.is_empty() {
        if !combined.is_empty() {
            combined.push('_');
        }
        combined.push_str(&r_clean);
    }
    if !combined.is_empty() {
        combined.push('_');
    }
    combined.push_str(&f_clean);

    let base_name = sanitize_ident(&combined);
    let mask = if width >= 32 {
        0xFFFF_FFFFu32
    } else {
        (((1u64 << width) - 1) << lsb) as u32
    };

    Some(CandidateConst {
        base_name,
        pos: lsb,
        mask,
    })
}

fn parse_field_bit_range(field_el: &XmlElement) -> Result<Option<(u32, u32)>, String> {
    if let Some(offset_raw) = field_el.child_text("bitOffset") {
        let lsb = parse_svd_u32(offset_raw, "<bitOffset>")?;
        let width = match field_el.child_text("bitWidth") {
            Some(w_raw) => parse_svd_u32(w_raw, "<bitWidth>")?,
            None => 1,
        };
        return Ok(Some((lsb, width)));
    }

    if let (Some(lsb_raw), Some(msb_raw)) = (field_el.child_text("lsb"), field_el.child_text("msb"))
    {
        let lsb = parse_svd_u32(lsb_raw, "<lsb>")?;
        let msb = parse_svd_u32(msb_raw, "<msb>")?;
        if msb < lsb {
            return Ok(None);
        }
        return Ok(Some((lsb, msb - lsb + 1)));
    }

    if let Some(range_raw) = field_el.child_text("bitRange") {
        let inner = range_raw.trim().trim_start_matches('[').trim_end_matches(']');
        let Some((msb_str, lsb_str)) = inner.split_once(':') else {
            return Err(format!("malformed `<bitRange>` value `{range_raw}`"));
        };
        let msb = parse_svd_u32(msb_str, "<bitRange> msb")?;
        let lsb = parse_svd_u32(lsb_str, "<bitRange> lsb")?;
        if msb < lsb {
            return Ok(None);
        }
        return Ok(Some((lsb, msb - lsb + 1)));
    }

    Ok(None)
}

fn finalize_layout_registers(
    mut candidates: Vec<CandidateReg>,
    used_consts: &mut HashSet<String>,
) -> (Vec<EmittedReg>, Vec<CandidateConst>) {
    candidates.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.seq.cmp(&b.seq)));

    let mut occupied: Vec<(u32, u32)> = Vec::new();
    let mut used_reg_names: HashSet<String> = HashSet::new();
    let mut emitted_regs: Vec<EmittedReg> = Vec::new();
    let mut emitted_consts: Vec<CandidateConst> = Vec::new();

    for cand in candidates {
        let elem_bytes = (cand.elem_bits / 8) as u32;
        if elem_bytes == 0 || cand.offset % elem_bytes != 0 {
            continue;
        }
        let count = cand.array_len.unwrap_or(1) as u32;
        let Some(total_bytes) = elem_bytes.checked_mul(count) else {
            continue;
        };
        let Some(end) = cand.offset.checked_add(total_bytes) else {
            continue;
        };

        if occupied
            .iter()
            .any(|&(s, e)| cand.offset < e && end > s)
        {
            continue;
        }
        occupied.push((cand.offset, end));

        let name = if used_reg_names.contains(&cand.name) {
            make_unique_name(&cand.name, &used_reg_names)
        } else {
            cand.name
        };
        used_reg_names.insert(name.clone());

        emitted_regs.push(EmittedReg {
            name,
            offset: cand.offset,
            elem_bits: cand.elem_bits,
            array_len: cand.array_len,
            access: cand.access,
        });

        for c in cand.consts {
            let pos_sym = format!("{}_POS", c.base_name);
            let mask_sym = format!("{}_MASK", c.base_name);
            if used_consts.contains(&pos_sym) || used_consts.contains(&mask_sym) {
                continue;
            }
            used_consts.insert(pos_sym);
            used_consts.insert(mask_sym);
            emitted_consts.push(c);
        }
    }

    (emitted_regs, emitted_consts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Naming, Sanitization & Formatting Helpers
// ─────────────────────────────────────────────────────────────────────────────

const ZEV_KEYWORDS: &[&str] = &[
    // Control flow & core language keywords
    "in",
    "loop",
    "task",
    "fn",
    "error",
    "return",
    "default",
    "switch",
    "if",
    "else",
    "while",
    "for",
    "every",
    "defer",
    "noyield",
    "critical",
    "spawn",
    "continue",
    "break",
    "restart",
    "true",
    "false",
    "import",
    "as",
    "packed",
    "struct",
    "enum",
    "const",
    "extern",
    "isr",
    "catch",
    "or",
    // Foreign-idiom / Wave-1 keywords
    "let",
    "var",
    "mut",
    "async",
    "await",
    "select",
    "match",
    "try",
    "static",
    "pub",
    "mmio",
    "irq",
    "state",
    // Primitive & OS type names
    "u8",
    "u16",
    "u32",
    "u64",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "isize",
    "f32",
    "f64",
    "u16be",
    "u16le",
    "u32be",
    "u32le",
    "u64be",
    "u64le",
    "bool",
    "void",
    "signal",
    "watch",
    "inbox",
    "ping_pong",
    "lease",
];

fn is_zev_keyword(s: &str) -> bool {
    ZEV_KEYWORDS.contains(&s)
}

fn sanitize_chars(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
}

fn sanitize_ident(raw: &str) -> String {
    let mut out = sanitize_chars(raw);
    if out.is_empty() {
        out.push_str("_ident");
        return out;
    }
    if out.as_bytes()[0].is_ascii_digit() {
        out.insert(0, '_');
    }
    if is_zev_keyword(&out) {
        out.push('_');
    }
    out
}

fn sanitize_reg_ident(raw: &str) -> String {
    let lowered = sanitize_chars(raw).to_ascii_lowercase();
    sanitize_ident(&lowered)
}

fn clean_upper_segment(raw: &str) -> String {
    sanitize_chars(raw)
        .trim_matches('_')
        .to_ascii_uppercase()
}

fn format_irq_ident(raw: &str) -> String {
    let upper = clean_upper_segment(raw);
    if upper.is_empty() {
        return "IRQ_UNKNOWN".to_string();
    }
    if upper.starts_with("IRQ_") {
        sanitize_ident(&upper)
    } else {
        sanitize_ident(&format!("IRQ_{upper}"))
    }
}

fn to_layout_name(base_periph_name: &str) -> String {
    let mut pascal = String::new();
    for chunk in base_periph_name.split(|c: char| !c.is_ascii_alphanumeric()) {
        if chunk.is_empty() {
            continue;
        }
        let has_lower = chunk.bytes().any(|b| b.is_ascii_lowercase());
        let mut chars = chunk.chars();
        if let Some(first) = chars.next() {
            pascal.push(first.to_ascii_uppercase());
            if has_lower {
                for c in chars {
                    pascal.push(c);
                }
            } else {
                for c in chars {
                    pascal.push(c.to_ascii_lowercase());
                }
            }
        }
    }
    if pascal.is_empty() {
        pascal.push_str("Periph");
    }
    pascal.push_str("Regs");
    sanitize_ident(&pascal)
}

fn join_prefix(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else if name.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}_{name}")
    }
}

fn make_unique_name(base: &str, used: &HashSet<String>) -> String {
    let mut n = 2usize;
    loop {
        let candidate = format!("{base}_{n}");
        if !used.contains(&candidate) {
            return candidate;
        }
        n += 1;
    }
}

fn parse_dim_indices(dim: usize, dim_index: Option<&str>) -> Vec<String> {
    let mut out = Vec::with_capacity(dim);
    if let Some(raw) = dim_index.map(str::trim).filter(|s| !s.is_empty()) {
        if raw.contains(',') {
            for part in raw.split(',') {
                let p = part.trim();
                if !p.is_empty() {
                    out.push(p.to_string());
                }
            }
        } else if let Some((lhs, rhs)) = raw.split_once('-') {
            let l = lhs.trim();
            let r = rhs.trim();
            if let (Ok(start), Ok(end)) = (l.parse::<usize>(), r.parse::<usize>()) {
                if start <= end {
                    let take = dim.min(end - start + 1);
                    for i in 0..take {
                        out.push((start + i).to_string());
                    }
                }
            } else if l.len() == 1 && r.len() == 1 {
                let lb = l.as_bytes()[0];
                let rb = r.as_bytes()[0];
                if lb.is_ascii_alphabetic()
                    && rb.is_ascii_alphabetic()
                    && lb.is_ascii_uppercase() == rb.is_ascii_uppercase()
                    && lb <= rb
                {
                    for b in lb..=rb {
                        out.push((b as char).to_string());
                        if out.len() >= dim {
                            break;
                        }
                    }
                }
            }
        } else {
            out.push(raw.to_string());
        }
    }

    out.truncate(dim);
    while out.len() < dim {
        out.push(out.len().to_string());
    }
    out
}

fn strip_dim_placeholder(template: &str) -> String {
    let s = template.replace("[%s]", "").replace("%s", "");
    let mut collapsed = String::with_capacity(s.len());
    let mut prev_underscore = false;
    for ch in s.chars() {
        if ch == '_' {
            if !prev_underscore {
                collapsed.push('_');
            }
            prev_underscore = true;
        } else {
            collapsed.push(ch);
            prev_underscore = false;
        }
    }
    collapsed.trim_matches('_').to_string()
}

fn substitute_dim_name(template: &str, idx: &str, append_if_absent: bool) -> String {
    if let Some(pos) = template.find("[%s]") {
        let prefix = &template[..pos];
        let suffix = &template[pos + 4..];
        if prefix.ends_with('_') {
            format!("{prefix}{idx}{suffix}")
        } else {
            format!("{prefix}_{idx}{suffix}")
        }
    } else if template.contains("%s") {
        template.replace("%s", idx)
    } else if append_if_absent {
        format!("{template}_{idx}")
    } else {
        template.to_string()
    }
}

fn format_hex_offset(offset: u32) -> String {
    if offset <= 0xFFFF {
        format!("0x{offset:02X}")
    } else {
        format_hex_u32(offset)
    }
}

fn format_hex_u32(val: u32) -> String {
    format!("0x{:04X}_{:04X}", val >> 16, val & 0xFFFF)
}

fn parse_svd_u64(raw: &str, ctx: &str) -> Result<u64, String> {
    let cleaned: String = raw.trim().chars().filter(|&c| c != '_').collect();
    if cleaned.is_empty() {
        return Err(format!("empty integer in `{ctx}`"));
    }
    let (radix, digits) = if let Some(h) = cleaned
        .strip_prefix("0x")
        .or_else(|| cleaned.strip_prefix("0X"))
    {
        (16, h)
    } else if let Some(b) = cleaned
        .strip_prefix("0b")
        .or_else(|| cleaned.strip_prefix("0B"))
        .or_else(|| cleaned.strip_prefix('#'))
    {
        (2, b)
    } else {
        (10, cleaned.as_str())
    };
    u64::from_str_radix(digits, radix)
        .map_err(|e| format!("invalid integer `{raw}` in `{ctx}`: {e}"))
}

fn parse_svd_u32(raw: &str, ctx: &str) -> Result<u32, String> {
    let val = parse_svd_u64(raw, ctx)?;
    u32::try_from(val).map_err(|_| format!("integer `{raw}` in `{ctx}` exceeds 32-bit range"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Dependency-Free Pure Rust `std` XML Parser
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct XmlElement {
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<XmlElement>,
    text: String,
}

impl XmlElement {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn child(&self, name: &str) -> Option<&XmlElement> {
        self.children.iter().find(|c| c.name == name)
    }

    fn children<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a XmlElement> {
        self.children.iter().filter(move |c| c.name == name)
    }

    fn child_text(&self, name: &str) -> Option<&str> {
        self.child(name)
            .map(|c| c.text.trim())
            .filter(|s| !s.is_empty())
    }
}

struct XmlScanner<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> XmlScanner<'a> {
    fn new(input: &'a str) -> Self {
        let trimmed = input.strip_prefix('\u{FEFF}').unwrap_or(input);
        Self {
            src: trimmed,
            bytes: trimmed.as_bytes(),
            pos: 0,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn starts_with(&self, pat: &str) -> bool {
        self.rest().starts_with(pat)
    }

    fn skip_until(&mut self, needle: &str, what: &str) -> Result<&'a str, String> {
        let start = self.pos;
        if let Some(rel) = self.rest().find(needle) {
            let content = &self.src[start..start + rel];
            self.pos = start + rel + needle.len();
            Ok(content)
        } else {
            Err(format!("unterminated {what}"))
        }
    }

    fn skip_doctype(&mut self) -> Result<(), String> {
        self.pos += 9; // "<!DOCTYPE"
        let mut bracket_depth = 0usize;
        let mut quote: Option<u8> = None;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if let Some(q) = quote {
                if b == q {
                    quote = None;
                }
            } else {
                match b {
                    b'"' | b'\'' => quote = Some(b),
                    b'[' => bracket_depth += 1,
                    b']' => bracket_depth = bracket_depth.saturating_sub(1),
                    b'>' if bracket_depth == 0 => return Ok(()),
                    _ => {}
                }
            }
        }
        Err("unterminated `<!DOCTYPE ...>` declaration".to_string())
    }

    fn parse_name(&mut self, what: &str) -> Result<String, String> {
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b.is_ascii_whitespace() || matches!(b, b'>' | b'/' | b'=' | b'<' | b'"' | b'\'') {
                break;
            }
            self.pos += 1;
        }
        if self.pos == start {
            return Err(format!("expected {what}"));
        }
        Ok(self.src[start..self.pos].to_string())
    }

    fn parse_attr_value(&mut self) -> Result<String, String> {
        if self.is_eof() {
            return Err("unexpected EOF in XML attribute value".to_string());
        }
        let quote = self.bytes[self.pos];
        if quote != b'"' && quote != b'\'' {
            return Err("expected quoted XML attribute value".to_string());
        }
        self.pos += 1;
        let start = self.pos;
        while self.pos < self.bytes.len() {
            if self.bytes[self.pos] == quote {
                let raw = &self.src[start..self.pos];
                self.pos += 1;
                return unescape_xml(raw);
            }
            self.pos += 1;
        }
        Err("unterminated XML attribute value".to_string())
    }
}

fn parse_xml_document(input: &str) -> Result<XmlElement, String> {
    let mut sc = XmlScanner::new(input);
    let mut stack: Vec<XmlElement> = Vec::new();
    let mut root: Option<XmlElement> = None;

    while !sc.is_eof() {
        if sc.starts_with("<") {
            if sc.starts_with("<!--") {
                sc.pos += 4;
                sc.skip_until("-->", "XML comment")?;
            } else if sc.starts_with("<?") {
                sc.pos += 2;
                sc.skip_until("?>", "XML processing instruction")?;
            } else if sc.starts_with("<![CDATA[") {
                sc.pos += 9;
                let cdata = sc.skip_until("]]>", "CDATA section")?;
                let top = stack
                    .last_mut()
                    .ok_or_else(|| "CDATA section outside root element".to_string())?;
                top.text.push_str(cdata);
            } else if sc.starts_with("<!DOCTYPE") {
                sc.skip_doctype()?;
            } else if sc.starts_with("</") {
                sc.pos += 2;
                sc.skip_ws();
                let close_name = sc.parse_name("closing XML tag name")?;
                sc.skip_ws();
                if !sc.starts_with(">") {
                    return Err(format!("expected `>` after `</{close_name}`"));
                }
                sc.pos += 1;
                let elem = stack.pop().ok_or_else(|| {
                    format!("unexpected closing tag `</{close_name}>` with no matching open tag")
                })?;
                if elem.name != close_name {
                    return Err(format!(
                        "mismatched XML closing tag: expected `</{}>`, found `</{close_name}>`",
                        elem.name
                    ));
                }
                attach_finished(elem, &mut stack, &mut root)?;
            } else {
                // Opening or self-closing tag
                sc.pos += 1;
                let tag_name = sc.parse_name("XML tag name")?;
                let mut attrs = Vec::new();
                let self_closing;
                loop {
                    sc.skip_ws();
                    if sc.starts_with("/>") {
                        sc.pos += 2;
                        self_closing = true;
                        break;
                    }
                    if sc.starts_with(">") {
                        sc.pos += 1;
                        self_closing = false;
                        break;
                    }
                    if sc.is_eof() {
                        return Err(format!("unterminated opening tag `<{tag_name}>`"));
                    }
                    let attr_name = sc.parse_name("XML attribute name")?;
                    sc.skip_ws();
                    if !sc.starts_with("=") {
                        return Err(format!("expected `=` after attribute `{attr_name}`"));
                    }
                    sc.pos += 1;
                    sc.skip_ws();
                    let attr_val = sc.parse_attr_value()?;
                    attrs.push((attr_name, attr_val));
                }

                let elem = XmlElement {
                    name: tag_name,
                    attrs,
                    children: Vec::new(),
                    text: String::new(),
                };
                if self_closing {
                    attach_finished(elem, &mut stack, &mut root)?;
                } else {
                    if stack.is_empty() && root.is_some() {
                        return Err("multiple root XML elements are not allowed".to_string());
                    }
                    stack.push(elem);
                }
            }
        } else {
            let start = sc.pos;
            while sc.pos < sc.bytes.len() && sc.bytes[sc.pos] != b'<' {
                sc.pos += 1;
            }
            let raw_text = &sc.src[start..sc.pos];
            if let Some(top) = stack.last_mut() {
                if raw_text.contains('&') {
                    top.text.push_str(&unescape_xml(raw_text)?);
                } else {
                    top.text.push_str(raw_text);
                }
            } else if !raw_text.trim().is_empty() {
                return Err("unexpected character data outside root XML element".to_string());
            }
        }
    }

    if let Some(unclosed) = stack.pop() {
        return Err(format!("unclosed XML tag `<{}>`", unclosed.name));
    }

    root.ok_or_else(|| "empty XML document (missing root `<device>` element)".to_string())
}

fn attach_finished(
    elem: XmlElement,
    stack: &mut Vec<XmlElement>,
    root: &mut Option<XmlElement>,
) -> Result<(), String> {
    if let Some(parent) = stack.last_mut() {
        parent.children.push(elem);
        Ok(())
    } else if root.is_none() {
        *root = Some(elem);
        Ok(())
    } else {
        Err("multiple root XML elements are not allowed".to_string())
    }
}

fn unescape_xml(raw: &str) -> Result<String, String> {
    if !raw.contains('&') {
        return Ok(raw.to_string());
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after_amp = &rest[amp + 1..];
        let Some(semi) = after_amp.find(';') else {
            return Err("unterminated XML entity reference".to_string());
        };
        let entity = &after_amp[..semi];
        match entity {
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "amp" => out.push('&'),
            "apos" => out.push('\''),
            "quot" => out.push('"'),
            _ if entity.starts_with("#x") || entity.starts_with("#X") => {
                let code = u32::from_str_radix(&entity[2..], 16)
                    .map_err(|_| format!("invalid XML hex character reference `&{entity};`"))?;
                let ch = char::from_u32(code)
                    .ok_or_else(|| format!("invalid Unicode scalar `&{entity};`"))?;
                out.push(ch);
            }
            _ if entity.starts_with('#') => {
                let code = entity[1..]
                    .parse::<u32>()
                    .map_err(|_| format!("invalid XML decimal character reference `&{entity};`"))?;
                let ch = char::from_u32(code)
                    .ok_or_else(|| format!("invalid Unicode scalar `&{entity};`"))?;
                out.push(ch);
            }
            other => {
                return Err(format!("unknown XML entity `&{other};`"));
            }
        }
        rest = &after_amp[semi + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::TempWorkDir;

    #[test]
    fn converts_derived_peripherals_irqs_access_arrays_and_bitfields() {
        let svd = r#"<?xml version="1.0" encoding="utf-8"?>
<device schemaVersion="1.1">
  <name>STM32_MINI</name>
  <size>32</size>
  <access>read-write</access>
  <peripherals>
    <peripheral>
      <name>USART1</name>
      <baseAddress>0x40011000</baseAddress>
      <interrupt>
        <name>USART1</name>
        <value>37</value>
      </interrupt>
      <interrupt>
        <name>IRQ_SHARED</name>
        <value>40</value>
      </interrupt>
      <registers>
        <register>
          <name>CR1</name>
          <addressOffset>0x00</addressOffset>
          <fields>
            <field>
              <name>UE</name>
              <bitOffset>0</bitOffset>
              <bitWidth>1</bitWidth>
            </field>
            <field>
              <name>M</name>
              <lsb>12</lsb>
              <msb>13</msb>
            </field>
          </fields>
        </register>
        <register>
          <name>ISR</name>
          <addressOffset>0x1C</addressOffset>
          <access>read-only</access>
          <fields>
            <field>
              <name>RXNE</name>
              <bitRange>[5:5]</bitRange>
            </field>
          </fields>
        </register>
        <register>
          <name>ICR</name>
          <addressOffset>0x20</addressOffset>
          <access>write-only</access>
          <modifiedWriteValues>oneToClear</modifiedWriteValues>
          <fields>
            <field>
              <name>ORECF</name>
              <bitRange>[3:3]</bitRange>
            </field>
          </fields>
        </register>
        <register>
          <name>TDR</name>
          <addressOffset>0x28</addressOffset>
          <access>write-only</access>
        </register>
        <register>
          <dim>4</dim>
          <dimIncrement>4</dimIncrement>
          <dimIndex>0-3</dimIndex>
          <name>BRR[%s]</name>
          <addressOffset>0x30</addressOffset>
        </register>
        <register>
          <dim>2</dim>
          <dimIncrement>8</dimIncrement>
          <name>STRIDE</name>
          <addressOffset>0x40</addressOffset>
        </register>
      </registers>
    </peripheral>
    <peripheral derivedFrom="USART1">
      <name>USART2</name>
      <baseAddress>0x40004400</baseAddress>
      <interrupt>
        <name>USART2</name>
        <value>38</value>
      </interrupt>
      <interrupt>
        <name>IRQ_SHARED</name>
        <value>40</value>
      </interrupt>
    </peripheral>
  </peripherals>
</device>"#;

        let zev = svd_to_zev(svd).expect("valid SVD must convert cleanly");

        // Shared layout emitted ONCE, two instances emitted
        assert_eq!(
            zev.matches("pub mmio Usart1Regs {").count(),
            1,
            "expected exactly one shared Usart1Regs layout:\n{zev}"
        );
        assert!(
            zev.contains("pub mmio USART1 : Usart1Regs @ 0x4001_1000;"),
            "missing USART1 instance:\n{zev}"
        );
        assert!(
            zev.contains("pub mmio USART2 : Usart1Regs @ 0x4000_4400;"),
            "missing USART2 instance:\n{zev}"
        );

        // Interrupt tokens (no double IRQ_IRQ_, deduplicated shared IRQ)
        assert!(zev.contains("pub irq IRQ_USART1 = 37;"), "{zev}");
        assert!(zev.contains("pub irq IRQ_USART2 = 38;"), "{zev}");
        assert_eq!(zev.matches("pub irq IRQ_SHARED = 40;").count(), 1, "{zev}");
        assert!(!zev.contains("IRQ_IRQ_"), "{zev}");

        // Access qualifiers & sanitized `isr` keyword
        assert!(zev.contains("@rw u32 cr1 @ 0x00;"), "{zev}");
        assert!(zev.contains("@ro u32 isr_ @ 0x1C;"), "{zev}");
        assert!(zev.contains("@w1c u32 icr @ 0x20;"), "{zev}");
        assert!(zev.contains("@wo u32 tdr @ 0x28;"), "{zev}");

        // Contiguous dim array & strided expansion
        assert!(zev.contains("@rw [4]u32 brr @ 0x30;"), "{zev}");
        assert!(zev.contains("@rw u32 stride_0 @ 0x40;"), "{zev}");
        assert!(zev.contains("@rw u32 stride_1 @ 0x48;"), "{zev}");

        // Bitfields from bitOffset+bitWidth, lsb+msb, and bitRange[msb:lsb]
        assert!(zev.contains("pub const u32 USART1_CR1_UE_POS = 0;"), "{zev}");
        assert!(
            zev.contains("pub const u32 USART1_CR1_UE_MASK = 0x0000_0001;"),
            "{zev}"
        );
        assert!(zev.contains("pub const u32 USART1_CR1_M_POS = 12;"), "{zev}");
        assert!(
            zev.contains("pub const u32 USART1_CR1_M_MASK = 0x0000_3000;"),
            "{zev}"
        );
        assert!(zev.contains("pub const u32 USART1_ISR_RXNE_POS = 5;"), "{zev}");
        assert!(
            zev.contains("pub const u32 USART1_ISR_RXNE_MASK = 0x0000_0020;"),
            "{zev}"
        );
        assert!(zev.contains("pub const u32 USART1_ICR_ORECF_POS = 3;"), "{zev}");
        assert!(
            zev.contains("pub const u32 USART1_ICR_ORECF_MASK = 0x0000_0008;"),
            "{zev}"
        );
    }

    #[test]
    fn handles_nested_clusters_inheritance_sizes_alternate_and_overlaps() {
        let svd = r#"<?xml version="1.0" encoding="utf-8"?>
<device>
  <name>CLUSTER_DEV</name>
  <size>16</size>
  <access>read-only</access>
  <peripherals>
    <peripheral>
      <name>DMA1</name>
      <baseAddress>0x40020000</baseAddress>
      <registers>
        <register>
          <name>STATUS8</name>
          <addressOffset>0x00</addressOffset>
          <size>8</size>
        </register>
        <register>
          <name>ALT_STATUS8</name>
          <alternateRegister>STATUS8</alternateRegister>
          <addressOffset>0x00</addressOffset>
          <size>8</size>
          <access>write-only</access>
        </register>
        <register>
          <name>MISALIGNED32</name>
          <addressOffset>0x02</addressOffset>
          <size>32</size>
        </register>
        <register>
          <name>FULL32</name>
          <addressOffset>0x04</addressOffset>
          <size>32</size>
          <access>read-write</access>
        </register>
        <register>
          <name>OVERLAP16</name>
          <addressOffset>0x06</addressOffset>
          <size>16</size>
        </register>
        <cluster>
          <name>CH[%s]</name>
          <dim>2</dim>
          <dimIncrement>0x10</dimIncrement>
          <dimIndex>1,2</dimIndex>
          <addressOffset>0x10</addressOffset>
          <size>32</size>
          <access>read-write</access>
          <register>
            <name>CTRL</name>
            <addressOffset>0x00</addressOffset>
          </register>
          <cluster>
            <name>CFG</name>
            <addressOffset>0x04</addressOffset>
            <modifiedWriteValues>oneToClear</modifiedWriteValues>
            <register>
              <name>CLEAR</name>
              <addressOffset>0x00</addressOffset>
            </register>
            <register>
              <name>HALF</name>
              <addressOffset>0x04</addressOffset>
              <size>16</size>
              <modifiedWriteValues>modify</modifiedWriteValues>
            </register>
          </cluster>
        </cluster>
      </registers>
    </peripheral>
  </peripherals>
</device>"#;

        let zev = svd_to_zev(svd).expect("cluster SVD must convert cleanly");
        assert!(zev.contains("pub mmio Dma1Regs {"), "{zev}");
        assert!(zev.contains("pub mmio DMA1 : Dma1Regs @ 0x4002_0000;"), "{zev}");

        // Device-level 8/16/32 and read-only inheritance
        assert!(zev.contains("@ro u8 status8 @ 0x00;"), "{zev}");
        assert!(!zev.contains("alt_status8"), "{zev}");
        assert!(!zev.contains("misaligned32"), "{zev}");
        assert!(zev.contains("@rw u32 full32 @ 0x04;"), "{zev}");
        assert!(!zev.contains("overlap16"), "{zev}");

        // Nested cluster offsets & inherited access/modifiedWriteValues
        assert!(zev.contains("@rw u32 ch_1_ctrl @ 0x10;"), "{zev}");
        assert!(zev.contains("@w1c u32 ch_1_cfg_clear @ 0x14;"), "{zev}");
        assert!(zev.contains("@rw u16 ch_1_cfg_half @ 0x18;"), "{zev}");
        assert!(zev.contains("@rw u32 ch_2_ctrl @ 0x20;"), "{zev}");
        assert!(zev.contains("@w1c u32 ch_2_cfg_clear @ 0x24;"), "{zev}");
        assert!(zev.contains("@rw u16 ch_2_cfg_half @ 0x28;"), "{zev}");
    }

    #[test]
    fn sanitizes_all_zev_keywords_types_and_leading_digits() {
        let keywords = [
            "in", "loop", "task", "fn", "error", "return", "default", "switch", "if", "else",
            "while", "for", "every", "defer", "noyield", "critical", "spawn", "continue", "break",
            "restart", "true", "false", "import", "as", "packed", "struct", "enum", "const",
            "extern", "isr", "catch", "or", "let", "var", "mut", "async", "await", "select",
            "match", "try", "static", "pub", "mmio", "irq", "state", "u8", "u16", "u32", "u64",
            "usize", "i8", "i16", "i32", "i64", "isize", "f32", "f64", "u16be", "u16le", "u32be",
            "u32le", "u64be", "u64le", "bool", "void", "signal", "watch", "inbox", "ping_pong",
            "lease",
        ];

        let mut regs_xml = String::new();
        for (idx, kw) in keywords.iter().enumerate() {
            let off = idx * 4;
            let _ = writeln!(
                regs_xml,
                "<register><name>{kw}</name><addressOffset>0x{off:X}</addressOffset></register>"
            );
        }
        let digit_off = keywords.len() * 4;
        let _ = writeln!(
            regs_xml,
            "<register><name>0_cfg-reg</name><addressOffset>0x{digit_off:X}</addressOffset></register>"
        );

        let svd = format!(
            "<device><name>KW_TEST</name><peripherals><peripheral><name>TEST</name><baseAddress>0x40000000</baseAddress><registers>{regs_xml}</registers></peripheral></peripherals></device>"
        );
        let zev = svd_to_zev(&svd).expect("keyword SVD must convert cleanly");

        for (idx, kw) in keywords.iter().enumerate() {
            let off = format_hex_offset((idx * 4) as u32);
            let expected = format!("@rw u32 {kw}_ @ {off};");
            assert!(zev.contains(&expected), "missing `{expected}` in:\n{zev}");
        }
        let expected_digit = format!(
            "@rw u32 _0_cfg_reg @ {};",
            format_hex_offset(digit_off as u32)
        );
        assert!(zev.contains(&expected_digit), "{zev}");
    }

    #[test]
    fn handles_xml_entities_cdata_doctype_and_full_32bit_field_mask() {
        let svd = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE device [
  <!ELEMENT device ANY>
]>
<!-- Top-level comment -->
<device xmlns:xs="http://www.w3.org/2001/XMLSchema-instance">
  <name>ESP32_MINI</name>
  <description><![CDATA[Test <device> & "quoted" text]]></description>
  <peripherals>
    <peripheral>
      <name>SYSTIMER</name>
      <baseAddress>0x6000A000</baseAddress>
      <registers>
        <register>
          <name>LOAD</name>
          <description>Bits &lt;31:0&gt; &amp; &#x41; &#66;</description>
          <addressOffset>0x00</addressOffset>
          <fields>
            <field>
              <name>ALL</name>
              <bitOffset>0</bitOffset>
              <bitWidth>32</bitWidth>
            </field>
            <field>
              <dim>2</dim>
              <dimIncrement>4</dimIncrement>
              <dimIndex>A-B</dimIndex>
              <name>CH%s</name>
              <bitOffset>0</bitOffset>
              <bitWidth>4</bitWidth>
            </field>
          </fields>
        </register>
      </registers>
    </peripheral>
  </peripherals>
</device>"#;

        let zev = svd_to_zev(svd).expect("XML with entities/CDATA/DOCTYPE must convert");
        assert!(zev.contains("pub const u32 SYSTIMER_LOAD_ALL_POS = 0;"), "{zev}");
        assert!(
            zev.contains("pub const u32 SYSTIMER_LOAD_ALL_MASK = 0xFFFF_FFFF;"),
            "{zev}"
        );
        assert!(zev.contains("pub const u32 SYSTIMER_LOAD_CHA_POS = 0;"), "{zev}");
        assert!(
            zev.contains("pub const u32 SYSTIMER_LOAD_CHA_MASK = 0x0000_000F;"),
            "{zev}"
        );
        assert!(zev.contains("pub const u32 SYSTIMER_LOAD_CHB_POS = 4;"), "{zev}");
        assert!(
            zev.contains("pub const u32 SYSTIMER_LOAD_CHB_MASK = 0x0000_00F0;"),
            "{zev}"
        );
    }

    #[test]
    fn rejects_malformed_xml_and_cyclic_derived_from() {
        assert!(svd_to_zev("").is_err());
        assert!(svd_to_zev("<device><unclosed></device>").is_err());
        assert!(svd_to_zev("<other/>").is_err());
        assert!(svd_to_zev("<device><name>X</name></device>").is_err());

        let cyclic = r#"<device><name>CYC</name><peripherals>
          <peripheral derivedFrom="B"><name>A</name><baseAddress>0x1000</baseAddress></peripheral>
          <peripheral derivedFrom="A"><name>B</name><baseAddress>0x2000</baseAddress></peripheral>
        </peripherals></device>"#;
        assert!(svd_to_zev(cyclic).is_err());
    }

    #[test]
    fn run_import_svd_writes_output_file_and_handles_missing_input() {
        let temp = TempWorkDir::create("zevc-svd-cli").unwrap();
        let svd_path = temp.dir.join("mini.svd");
        let out_path = temp.dir.join("gen/mini.zev");

        std::fs::write(
            &svd_path,
            "<device><name>MINI</name><peripherals><peripheral><name>GPIOA</name><baseAddress>0x48000000</baseAddress><registers><register><name>ODR</name><addressOffset>0x14</addressOffset></register></registers></peripheral></peripherals></device>",
        )
        .unwrap();

        let code = run_import_svd(&svd_path, Some(&out_path));
        assert_eq!(code, ExitCode::SUCCESS);
        let generated = std::fs::read_to_string(&out_path).unwrap();
        assert!(generated.contains("pub mmio GpioaRegs {"));
        assert!(generated.contains("pub mmio GPIOA : GpioaRegs @ 0x4800_0000;"));
        assert!(generated.contains("@rw u32 odr @ 0x14;"));

        let missing_code = run_import_svd(&temp.dir.join("missing.svd"), Some(&out_path));
        assert_eq!(missing_code, ExitCode::FAILURE);
    }
}
