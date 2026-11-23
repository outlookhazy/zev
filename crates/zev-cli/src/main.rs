mod cli;
mod diag;
mod docs;
mod docs_errors;
mod runner;
mod svd;

use cli::{parse_args, Command, HELP_TEXT};
use diag::{
    format_check_human_ok, primary_doc_cmd, render_check_json_err,
    render_check_json_ok_with_tasks, JsonDiag, SourceFile, TaskRunSummary,
};
use runner::TempWorkDir;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use zev_ast::Program;
use zev_mir::MirModule;
use zev_sema::{analyze_and_monomorphize_named_programs, AnalyzeOptions, SemanticModel};

struct CheckedBundle {
    programs: Vec<Program>,
    sema: SemanticModel,
}

struct CheckFailure {
    human: String,
    json_diags: Vec<JsonDiag>,
}

#[cfg(test)]
fn load_and_check_detailed(files: &[PathBuf]) -> Result<CheckedBundle, CheckFailure> {
    load_and_check_detailed_with_includes(files, &[])
}

#[cfg(test)]
fn load_and_check_detailed_with_includes(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
) -> Result<CheckedBundle, CheckFailure> {
    load_and_check_detailed_with_includes_and_targets(files, include_dirs, None, None)
}

fn load_and_check_detailed_with_includes_and_targets(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    hal_dir: Option<&str>,
    board_module: Option<&str>,
) -> Result<CheckedBundle, CheckFailure> {
    let mut source_files: Vec<(String, PathBuf, SourceFile)> = Vec::with_capacity(files.len());
    let mut seen_canon: HashSet<PathBuf> = HashSet::with_capacity(files.len());

    for path in files {
        let canon_key = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone());
        if !seen_canon.insert(canon_key) {
            continue;
        }

        let display = path.display().to_string();
        let sf = SourceFile::load(path).map_err(|msg| CheckFailure {
            human: format!("error: {msg}"),
            json_diags: vec![JsonDiag {
                code: None,
                file: display.clone(),
                line: 1,
                col: 1,
                message: msg,
                help: None,
                doc_cmd: "zevc doc targets".to_string(),
            }],
        })?;

        source_files.push((display, path.clone(), sf));
    }

    let generic_roots: Vec<(&str, Option<&Path>)> = source_files
        .iter()
        .map(|(_, path, sf)| (sf.src.as_str(), Some(path.as_path())))
        .collect();
    let all_generics: BTreeSet<String> = zev_sema::collect_import_aware_generic_names_with_targets(
        &generic_roots,
        include_dirs,
        hal_dir,
        board_module,
    );

    let mut loaded: Vec<(String, SourceFile, Program)> = Vec::with_capacity(source_files.len());
    let mut by_name: HashMap<String, usize> = HashMap::with_capacity(source_files.len());

    for (display, _, sf) in source_files {
        match zev_parser::parse_str_named_with_generics(&sf.src, &display, Some(&all_generics)) {
            Ok(prog) => {
                let idx = loaded.len();
                by_name.insert(display.clone(), idx);
                loaded.push((display, sf, prog));
            }
            Err(pes) => {
                let mut human = String::new();
                let mut json_diags = Vec::with_capacity(pes.len());
                for pe in pes {
                    let (line, col) = sf.line_col(pe.span.start as usize);
                    let topic = pe.doc_topic.unwrap_or("syntax");
                    let doc_cmd = pe.doc_command();
                    human.push_str(&sf.format_rich_span_error(
                        pe.span,
                        pe.code,
                        &pe.message,
                        pe.hint.as_deref(),
                        Some(topic),
                    ));
                    json_diags.push(JsonDiag {
                        code: pe.code.map(str::to_string),
                        file: display.clone(),
                        line,
                        col,
                        message: pe.message,
                        help: pe.hint,
                        doc_cmd,
                    });
                }
                return Err(CheckFailure {
                    human: human.trim_end().to_string(),
                    json_diags,
                });
            }
        }
    }

    let named_refs: Vec<(&Program, Option<&str>)> = loaded
        .iter()
        .map(|(name, _, prog)| (prog, Some(name.as_str())))
        .collect();

    let mut opts = AnalyzeOptions::default();
    opts.include_dirs = include_dirs.to_vec();
    opts.hal_dir = hal_dir.map(str::to_string);
    opts.board_module = board_module.map(str::to_string);
    let out = analyze_and_monomorphize_named_programs(&named_refs, &opts);
    if !out.diagnostics.is_empty() {
        let mut rendered = String::new();
        let mut json_diags = Vec::with_capacity(out.diagnostics.len());
        let mut imported_sfs: HashMap<String, SourceFile> = HashMap::new();

        for d in &out.diagnostics {
            let code = d.code();
            let topic = d.doc_topic();
            let help_str = d.help_text();
            let doc_cmd = primary_doc_cmd(Some(code), Some(topic));

            let matched_sf: Option<&SourceFile> = if let Some(fname) = d.file.as_ref() {
                if let Some(&idx) = by_name.get(fname) {
                    Some(&loaded[idx].1)
                } else {
                    if !imported_sfs.contains_key(fname) {
                        if let Ok(sf) = SourceFile::load(Path::new(fname)) {
                            imported_sfs.insert(fname.clone(), sf);
                        }
                    }
                    imported_sfs.get(fname)
                }
            } else if loaded.len() == 1 {
                Some(&loaded[0].1)
            } else {
                None
            };

            if let Some(sf) = matched_sf {
                let (line, col) = sf.line_col(d.span.start as usize);
                rendered.push_str(&sf.format_rich_span_error(
                    d.span,
                    Some(code),
                    &d.message,
                    Some(help_str),
                    Some(topic),
                ));
                json_diags.push(JsonDiag {
                    code: Some(code.to_string()),
                    file: sf.path.display().to_string(),
                    line,
                    col,
                    message: d.message.clone(),
                    help: Some(help_str.to_string()),
                    doc_cmd,
                });
            } else {
                rendered.push_str(&d.to_string());
                rendered.push('\n');
                json_diags.push(JsonDiag {
                    code: Some(code.to_string()),
                    file: d.file.clone().unwrap_or_else(|| "<unknown>".to_string()),
                    line: d.span.line as usize,
                    col: d.span.col as usize,
                    message: d.message.clone(),
                    help: Some(help_str.to_string()),
                    doc_cmd,
                });
            }
        }

        return Err(CheckFailure {
            human: rendered.trim_end().to_string(),
            json_diags,
        });
    }

    Ok(CheckedBundle {
        programs: out.programs,
        sema: out.model,
    })
}

fn compile_to_mir_with_includes(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
) -> Result<(CheckedBundle, MirModule), String> {
    compile_to_mir_with_includes_and_targets(files, include_dirs, None, None)
}

fn compile_to_mir_with_includes_and_targets(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    hal_dir: Option<&str>,
    board_module: Option<&str>,
) -> Result<(CheckedBundle, MirModule), String> {
    let bundle = load_and_check_detailed_with_includes_and_targets(
        files,
        include_dirs,
        hal_dir,
        board_module,
    )
    .map_err(|f| f.human)?;
    let mir = zev_mir::lower_programs(&bundle.programs)
        .map_err(|e| format!("error: {e}\n  = docs: run 'zevc doc tasks'"))?;
    Ok((bundle, mir))
}

#[cfg(test)]
fn compile_to_mir(files: &[PathBuf]) -> Result<(CheckedBundle, MirModule), String> {
    compile_to_mir_with_includes(files, &[])
}

fn write_output(output: Option<&Path>, text: &str) -> Result<(), String> {
    match output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        format!("failed to create directory `{}`: {e}", parent.display())
                    })?;
                }
            }
            std::fs::write(path, text)
                .map_err(|e| format!("failed to write `{}`: {e}", path.display()))
        }
        None => {
            print!("{text}");
            Ok(())
        }
    }
}

fn default_output_with_ext(files: &[PathBuf], ext: &str) -> PathBuf {
    let stem = files
        .first()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("firmware");
    PathBuf::from(format!("{stem}.{ext}"))
}

fn resolve_llvm_options(
    target: Option<&str>,
) -> Result<zev_codegen_llvm::CodegenOptions, String> {
    match target {
        None | Some("cortex-m3") | Some("thumbv7m") | Some("thumbv7m-none-eabi") => {
            Ok(zev_codegen_llvm::CodegenOptions::default())
        }
        Some("esp32c6") | Some("riscv32") | Some("rv32imac") | Some("riscv32-esp32c6") => {
            Ok(zev_codegen_llvm::CodegenOptions::esp32c6())
        }
        Some("avr") | Some("atmega328p") | Some("uno") | Some("arduino-uno") => {
            Ok(zev_codegen_llvm::CodegenOptions::avr_uno())
        }
        Some("nano-every")
        | Some("arduino-nano-every")
        | Some("atmega4809")
        | Some("avr-mega0")
        | Some("avrxmega3") => Ok(zev_codegen_llvm::CodegenOptions::avr_nano_every()),
        Some("rp2040")
        | Some("pico")
        | Some("pico-w")
        | Some("cortex-m0plus")
        | Some("thumbv6m")
        | Some("thumbv6m-none-eabi") => Ok(zev_codegen_llvm::CodegenOptions::rp2040()),
        Some("host") | Some("x86_64") | Some("x86_64-unknown-linux-gnu") => {
            let mut opts = zev_codegen_llvm::CodegenOptions::default();
            opts.target = zev_codegen_llvm::TargetArch::HostX86_64Linux;
            Ok(opts)
        }
        Some(other) => Err(format!(
            "unsupported `--target {other}` (expected `cortex-m3`, `esp32c6`, `avr`, `nano-every`, `rp2040`, or `host`)"
        )),
    }
}

fn print_esp32c6_summary(res: &zev_codegen_llvm::Esp32C6ImageResult) {
    println!(
        "Built ESP32-C6 Direct-Boot image: {} ({} bytes, magic={:02x?})",
        res.bin_path.display(),
        res.bin_size,
        res.magic_header
    );
    print!("{}", res.elf.sizes.raw_output);
    println!(
        "Zero-heap audit: undefined_symbols={}, allocator_symbols={}",
        res.elf.audit.undefined_symbols.len(),
        res.elf.audit.heap_or_libc_symbols.len()
    );
}

fn print_avr_summary(res: &zev_codegen_llvm::AvrImageResult) {
    println!(
        "Built ATmega328P AVR image: {} ({} bytes flash / 32256 B, {} bytes static SRAM / 2048 B)",
        res.bin_path.display(),
        res.bin_size,
        res.sram_static_bytes
    );
    print!("{}", res.elf.sizes.raw_output);
    println!(
        "Zero-heap audit: undefined_symbols={}, allocator_symbols={}",
        res.elf.audit.undefined_symbols.len(),
        res.elf.audit.heap_or_libc_symbols.len()
    );
}

fn print_avr_nano_every_summary(res: &zev_codegen_llvm::AvrImageResult) {
    println!(
        "Built ATmega4809 AVR image: {} ({} bytes flash / 49152 B, {} bytes static SRAM / 6144 B)",
        res.bin_path.display(),
        res.bin_size,
        res.sram_static_bytes
    );
    print!("{}", res.elf.sizes.raw_output);
    println!(
        "Zero-heap audit: undefined_symbols={}, allocator_symbols={}",
        res.elf.audit.undefined_symbols.len(),
        res.elf.audit.heap_or_libc_symbols.len()
    );
}

fn print_rp2040_summary(res: &zev_codegen_llvm::Rp2040ImageResult) {
    println!(
        "Built RP2040 SRAM UF2 image: {} ({} bytes / {} block(s), raw bin={} bytes / 270336 B SRAM)",
        res.uf2_path.display(),
        res.uf2_size,
        res.uf2_blocks,
        res.bin_size
    );
    print!("{}", res.elf.sizes.raw_output);
    println!(
        "Zero-heap audit: undefined_symbols={}, allocator_symbols={}",
        res.elf.audit.undefined_symbols.len(),
        res.elf.audit.heap_or_libc_symbols.len()
    );
}

enum RamExecOutcome {
    Output { text: String, within_budget: bool },
    UsageErr(String),
    CompileErr(String),
}

fn discover_manifest_for_inputs(
    files: &[PathBuf],
) -> Result<Option<(PathBuf, zev_codegen_llvm::ProjectManifest)>, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("failed to get cwd: {e}"))?;
    if let Some(first) = files.first() {
        let abs = if first.is_absolute() {
            first.clone()
        } else {
            cwd.join(first)
        };
        if let Some(parent) = abs.parent().filter(|p| !p.as_os_str().is_empty()) {
            if let Some(found) = zev_codegen_llvm::ProjectManifest::discover(parent)
                .map_err(|e| e.to_string())?
            {
                return Ok(Some(found));
            }
        }
    }
    zev_codegen_llvm::ProjectManifest::discover(&cwd).map_err(|e| e.to_string())
}

fn resolve_target_spec_for_inputs(
    files: &[PathBuf],
    manifest_dir: Option<&Path>,
    chip: Option<&str>,
    board: Option<&str>,
) -> Result<zev_codegen_llvm::TargetSpec, String> {
    if board.is_some_and(|b| Path::new(b).extension().is_some_and(|e| e == "toml"))
        || chip.is_some_and(|c| Path::new(c).extension().is_some_and(|e| e == "toml"))
    {
        return zev_codegen_llvm::TargetSpec::resolve(chip, board).map_err(|e| e.to_string());
    }

    let mut custom_roots: Vec<PathBuf> = Vec::new();
    let mut push_root = |p: PathBuf| {
        if (p.join("chips").is_dir() || p.join("boards").is_dir()) && !custom_roots.contains(&p) {
            custom_roots.push(p);
        }
    };
    if let Some(mdir) = manifest_dir {
        push_root(mdir.to_path_buf());
    }
    if let Some(first) = files.first() {
        if let Ok(cwd) = std::env::current_dir() {
            let abs = if first.is_absolute() {
                first.clone()
            } else {
                cwd.join(first)
            };
            let mut cur = abs.parent();
            while let Some(d) = cur {
                push_root(d.to_path_buf());
                cur = d.parent();
            }
        }
    }

    for root in &custom_roots {
        if let Ok(spec) = zev_codegen_llvm::TargetSpec::load_from_root(root, chip, board) {
            return Ok(spec);
        }
        if let Some(bname) = board {
            let bpath = root.join("boards").join(format!("{bname}.toml"));
            if bpath.is_file() {
                let chip_arg = chip.map(|c| {
                    let cpath = root.join("chips").join(format!("{c}.toml"));
                    if cpath.is_file() {
                        cpath.to_string_lossy().into_owned()
                    } else {
                        c.to_string()
                    }
                });
                if let Ok(spec) = zev_codegen_llvm::TargetSpec::resolve(
                    chip_arg.as_deref(),
                    Some(bpath.to_string_lossy().as_ref()),
                ) {
                    return Ok(spec);
                }
            }
        }
    }

    zev_codegen_llvm::TargetSpec::resolve(chip, board).map_err(|e| e.to_string())
}

fn resolve_llvm_options_for_cli(
    files: &[PathBuf],
    target: Option<&str>,
    board: Option<&str>,
    chip: Option<&str>,
) -> Result<zev_codegen_llvm::CodegenOptions, String> {
    if target.is_some() && (board.is_some() || chip.is_some()) {
        return Err("cannot combine legacy `--target` with `--board` or `--chip`".to_string());
    }
    if board.is_some() || chip.is_some() {
        let manifest = discover_manifest_for_inputs(files)?;
        let mdir = manifest.as_ref().map(|(d, _)| d.as_path());
        let spec = resolve_target_spec_for_inputs(files, mdir, chip, board)?;
        Ok(zev_codegen_llvm::CodegenOptions::from_spec(spec))
    } else {
        resolve_llvm_options(target)
    }
}

struct ResolvedProjectBuild {
    files: Vec<PathBuf>,
    include_dirs: Vec<PathBuf>,
    spec: zev_codegen_llvm::TargetSpec,
    opts: zev_codegen_llvm::CodegenOptions,
    port: Option<String>,
}

fn resolve_project_build_inputs(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    board: Option<&str>,
    chip: Option<&str>,
    port: Option<&str>,
    opt_level: Option<&str>,
) -> Result<ResolvedProjectBuild, String> {
    let manifest = discover_manifest_for_inputs(files)?;
    let (manifest_dir, resolved_files, eff_board, eff_chip, eff_port, eff_opt) =
        match manifest {
            Some((mdir, m)) => {
                let rfiles = if files.is_empty() {
                    let Some(ref entry) = m.entry else {
                        return Err(format!(
                            "no input `.zev` files given and `{}` has no `entry` field",
                            mdir.join("zev.toml").display()
                        ));
                    };
                    let resolved = if entry.is_absolute() {
                        entry.clone()
                    } else {
                        mdir.join(entry)
                    };
                    vec![resolved]
                } else {
                    files.to_vec()
                };
                (
                    Some(mdir),
                    rfiles,
                    board.map(str::to_string).or(m.board),
                    chip.map(str::to_string).or(m.chip),
                    port.map(str::to_string).or(m.port),
                    opt_level.map(str::to_string).or(m.opt_level),
                )
            }
            None => {
                if files.is_empty() {
                    return Err(
                        "no input `.zev` files given and no `zev.toml` found in current or parent directories"
                            .to_string(),
                    );
                }
                (
                    None,
                    files.to_vec(),
                    board.map(str::to_string),
                    chip.map(str::to_string),
                    port.map(str::to_string),
                    opt_level.map(str::to_string),
                )
            }
        };

    if eff_board.is_none() && eff_chip.is_none() {
        return Err(
            "no target specified: pass `--board <name>`, `--chip <name>`, or set `board`/`chip` in `zev.toml`"
                .to_string(),
        );
    }

    let mut resolved_includes = include_dirs.to_vec();
    if let Some(ref mdir) = manifest_dir {
        if !resolved_includes.contains(mdir) {
            resolved_includes.push(mdir.clone());
        }
    }

    let spec = resolve_target_spec_for_inputs(
        &resolved_files,
        manifest_dir.as_deref(),
        eff_chip.as_deref(),
        eff_board.as_deref(),
    )?;
    if spec.isa == zev_codegen_llvm::CpuIsa::HostX86_64Linux {
        return Err("host x86_64 target is not valid for embedded firmware builds".to_string());
    }
    let mut opts = zev_codegen_llvm::CodegenOptions::from_spec(spec.clone());
    if let Some(lvl) = eff_opt {
        opts.opt_level = lvl;
    }
    if let Some(ref root) = spec.root_dir {
        if !resolved_includes.contains(root) {
            resolved_includes.push(root.clone());
        }
    }

    Ok(ResolvedProjectBuild {
        files: resolved_files,
        include_dirs: resolved_includes,
        spec,
        opts,
        port: eff_port,
    })
}

fn print_target_build_summary(
    spec: &zev_codegen_llvm::TargetSpec,
    res: &zev_codegen_llvm::TargetBuildArtifact,
) {
    let label = spec
        .board_name
        .as_deref()
        .map(|b| format!("{b} ({})", spec.chip_name))
        .unwrap_or_else(|| spec.chip_name.clone());
    match spec.image {
        zev_codegen_llvm::ImageFormat::Uf2 { .. } => {
            println!(
                "Built {label} UF2 image: {} ({} bytes / {} block(s), raw bin={} bytes / {} B SRAM)",
                res.image_path.display(),
                res.image_size,
                res.uf2_blocks,
                res.bin_size,
                spec.sram_budget_bytes
            );
        }
        zev_codegen_llvm::ImageFormat::RawBin => {
            println!(
                "Built {label} image: {} ({} bytes flash / {} B, {} bytes static SRAM / {} B)",
                res.image_path.display(),
                res.image_size,
                spec.effective_flash_size(),
                res.sram_static_bytes,
                spec.sram_budget_bytes
            );
        }
        zev_codegen_llvm::ImageFormat::ElfOnly => {
            println!(
                "Built {label} ELF: {} ({} bytes static SRAM / {} B)",
                res.image_path.display(),
                res.sram_static_bytes,
                spec.sram_budget_bytes
            );
        }
    }
    print!("{}", res.elf.sizes.raw_output);
    println!(
        "Zero-heap audit: undefined_symbols={}, allocator_symbols={}",
        res.elf.audit.undefined_symbols.len(),
        res.elf.audit.heap_or_libc_symbols.len()
    );
}

fn run_ram_analysis(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    llvm_opts: Option<zev_codegen_llvm::CodegenOptions>,
) -> RamExecOutcome {
    let mut eff_includes = include_dirs.to_vec();
    if let Some(root) = llvm_opts
        .as_ref()
        .and_then(|o| o.spec.as_ref())
        .and_then(|s| s.root_dir.as_ref())
    {
        if !eff_includes.contains(root) {
            eff_includes.push(root.clone());
        }
    }
    let hal_dir = llvm_opts.as_ref().and_then(|o| o.hal_dir.as_deref());
    let board_module = llvm_opts.as_ref().and_then(|o| o.board_module.as_deref());
    let (_, mir) = match compile_to_mir_with_includes_and_targets(
        files,
        &eff_includes,
        hal_dir,
        board_module,
    ) {
        Ok(res) => res,
        Err(err) => return RamExecOutcome::CompileErr(err),
    };
    let mut text = mir.ram_report.format_table();
    if let Some(opts) = llvm_opts {
        match zev_codegen_llvm::analyze_hardware_stack_high_water(&mir, &opts) {
            Ok(proof) => {
                text.push_str(&proof.format_table());
                RamExecOutcome::Output {
                    text,
                    within_budget: proof.within_budget,
                }
            }
            Err(err) => RamExecOutcome::CompileErr(format!("error: {err}")),
        }
    } else {
        RamExecOutcome::Output {
            text,
            within_budget: true,
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn execute_ram_command(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    target: Option<&str>,
    opt_level: Option<&str>,
) -> RamExecOutcome {
    execute_ram_command_with_spec(files, include_dirs, target, None, None, opt_level)
}

fn execute_ram_command_with_spec(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    target: Option<&str>,
    board: Option<&str>,
    chip: Option<&str>,
    opt_level: Option<&str>,
) -> RamExecOutcome {
    let llvm_opts = if target.is_none() && board.is_none() && chip.is_none() {
        None
    } else {
        match resolve_llvm_options_for_cli(files, target, board, chip) {
            Ok(mut o) => {
                if o.target == zev_codegen_llvm::TargetArch::HostX86_64Linux {
                    return RamExecOutcome::UsageErr(format!(
                        "unsupported target `{}` for `zevc ram` (expected embedded `--board`, `--chip`, or `--target <cortex-m3|esp32c6|avr|nano-every|rp2040>`)",
                        target.or(board).or(chip).unwrap_or("host")
                    ));
                }
                if let Some(lvl) = opt_level {
                    o.opt_level = lvl.to_string();
                }
                Some(o)
            }
            Err(err) => return RamExecOutcome::UsageErr(err),
        }
    };
    run_ram_analysis(files, include_dirs, llvm_opts)
}

fn main() -> ExitCode {
    let args = std::env::args().skip(1);
    let cmd = match parse_args(args) {
        Ok(cmd) => cmd,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::from(2);
        }
    };

    match cmd {
        Command::Help => {
            print!("{HELP_TEXT}");
            ExitCode::SUCCESS
        }
        Command::Version => {
            println!("zevc 0.1.0");
            ExitCode::SUCCESS
        }
        Command::Doc { query, list } => match docs::lookup_doc(query.as_deref(), list) {
            Ok(text) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        },
        Command::ImportSvd { svd_path, output } => svd::run_import_svd(&svd_path, output.as_deref()),
        Command::Check {
            files,
            include_dirs,
            board,
            chip,
            json,
        } => {
            let mut eff_includes = include_dirs.clone();
            let (hal_dir, board_module): (Option<String>, Option<String>) =
                if board.is_some() || chip.is_some() {
                    let manifest = match discover_manifest_for_inputs(&files) {
                        Ok(m) => m,
                        Err(err) => {
                            eprintln!("error: {err}");
                            return ExitCode::from(2);
                        }
                    };
                    let mdir = manifest.as_ref().map(|(d, _)| d.as_path());
                    let spec = match resolve_target_spec_for_inputs(
                        &files,
                        mdir,
                        chip.as_deref(),
                        board.as_deref(),
                    ) {
                        Ok(s) => s,
                        Err(err) => {
                            eprintln!("error: {err}");
                            return ExitCode::from(2);
                        }
                    };
                    if let Some(ref root) = spec.root_dir {
                        if !eff_includes.contains(root) {
                            eff_includes.push(root.clone());
                        }
                    }
                    (spec.hal_dir, spec.board_module)
                } else {
                    (None, None)
                };
            match load_and_check_detailed_with_includes_and_targets(
                &files,
                &eff_includes,
                hal_dir.as_deref(),
                board_module.as_deref(),
            ) {
            Ok(bundle) => {
                let verified_files = bundle.programs.len();
                let dual_vec: Vec<String> =
                    bundle.sema.dual_specialized_fns.iter().cloned().collect();
                if json {
                    let task_runs: Vec<TaskRunSummary> = zev_mir::lower_programs(&bundle.programs)
                        .map(|mir| {
                            mir.ram_report
                                .spawned_tasks
                                .iter()
                                .map(|t| TaskRunSummary {
                                    slot_id: t.slot_id,
                                    instance: t.instance_name.clone(),
                                    blueprint: t.blueprint_name.clone(),
                                    priority: t.priority.label().to_string(),
                                    max_run_kind: t.max_run_kind.as_str().to_string(),
                                    max_run_blocks: t.max_run_blocks,
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    println!(
                        "{}",
                        render_check_json_ok_with_tasks(
                            verified_files,
                            bundle.sema.spawned_tasks.len(),
                            bundle.sema.primitives.len(),
                            bundle.sema.tick_period_us,
                            &dual_vec,
                            &bundle.sema.monomorphized_instances,
                            &task_runs,
                        )
                    );
                } else {
                    println!(
                        "{}",
                        format_check_human_ok(
                            verified_files,
                            bundle.sema.spawned_tasks.len(),
                            bundle.sema.primitives.len(),
                            bundle.sema.tick_period_us,
                            &dual_vec,
                            &bundle.sema.monomorphized_instances,
                        )
                    );
                }
                ExitCode::SUCCESS
            }
            Err(fail) => {
                if json {
                    println!("{}", render_check_json_err(&fail.json_diags));
                } else {
                    eprintln!("{}", fail.human);
                }
                ExitCode::FAILURE
            }
            }
        }
        Command::Ram {
            files,
            include_dirs,
            target,
            board,
            chip,
            opt_level,
        } => match execute_ram_command_with_spec(
            &files,
            &include_dirs,
            target.as_deref(),
            board.as_deref(),
            chip.as_deref(),
            opt_level.as_deref(),
        ) {
            RamExecOutcome::Output {
                text,
                within_budget,
            } => {
                print!("{text}");
                if within_budget {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            RamExecOutcome::UsageErr(err) => {
                eprintln!("error: {err}");
                ExitCode::from(2)
            }
            RamExecOutcome::CompileErr(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        Command::Run {
            files,
            include_dirs,
            sanitize,
            max_ticks,
            run_args,
        } => {
            let mut opts = zev_codegen_llvm::CodegenOptions::host_linux();
            opts.opt_level = if sanitize {
                "-O1".to_string()
            } else {
                "-O2".to_string()
            };
            if let Some(ticks) = max_ticks {
                opts.max_ticks = ticks.min(i32::MAX as u64) as u32;
            }
            match compile_to_mir_with_includes_and_targets(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
            ) {
                Ok((_, mir)) => match zev_codegen_llvm::emit_llvm_ir_with_options(&mir, &opts) {
                    Ok(ll_src) => {
                        match runner::compile_and_run_host_llvm(&ll_src, sanitize, &run_args) {
                            Ok(0) => ExitCode::SUCCESS,
                            Ok(code) => ExitCode::from((code.clamp(1, 255)) as u8),
                            Err(err) => {
                                eprintln!("error: {err}");
                                ExitCode::FAILURE
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                },
                Err(err) => {
                    eprintln!("{err}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::EmitLlvm {
            files,
            include_dirs,
            target,
            board,
            chip,
            opt_level,
            output,
        } => {
            let mut opts = match resolve_llvm_options_for_cli(
                &files,
                target.as_deref(),
                board.as_deref(),
                chip.as_deref(),
            ) {
                Ok(o) => o,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::from(2);
                }
            };
            if let Some(ref lvl) = opt_level {
                opts.opt_level = lvl.clone();
            }
            let mut eff_includes = include_dirs.clone();
            if let Some(ref root) = opts.spec.as_ref().and_then(|s| s.root_dir.as_ref()) {
                if !eff_includes.contains(root) {
                    eff_includes.push((*root).clone());
                }
            }
            match compile_to_mir_with_includes_and_targets(
                &files,
                &eff_includes,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
            ) {
                Ok((_, mir)) => match zev_codegen_llvm::emit_llvm_ir_with_options(&mir, &opts) {
                    Ok(raw_ll) => {
                        let final_ll = match opt_level.as_deref() {
                            None | Some("-O0") => raw_ll,
                            Some(_) => {
                                let temp = match TempWorkDir::create("zevc-opt-ll") {
                                    Ok(t) => t,
                                    Err(err) => {
                                        eprintln!("error: {err}");
                                        return ExitCode::FAILURE;
                                    }
                                };
                                let in_ll = temp.dir.join("raw.ll");
                                let out_ll = temp.dir.join("opt.ll");
                                if let Err(e) = std::fs::write(&in_ll, &raw_ll) {
                                    eprintln!("error: failed to stage raw LLVM IR: {e}");
                                    return ExitCode::FAILURE;
                                }
                                if let Err(err) =
                                    zev_codegen_llvm::optimize_ll_with_opt(&in_ll, &out_ll, &opts)
                                {
                                    eprintln!("error: {err}");
                                    return ExitCode::FAILURE;
                                }
                                match std::fs::read_to_string(&out_ll) {
                                    Ok(s) => s,
                                    Err(e) => {
                                        eprintln!("error: failed to read optimized LLVM IR: {e}");
                                        return ExitCode::FAILURE;
                                    }
                                }
                            }
                        };
                        if let Err(err) = write_output(output.as_deref(), &final_ll) {
                            eprintln!("error: {err}");
                            return ExitCode::FAILURE;
                        }
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                },
                Err(err) => {
                    eprintln!("{err}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Build {
            files,
            include_dirs,
            board,
            chip,
            opt_level,
            emit_ll,
            output,
        } => {
            let built = match resolve_project_build_inputs(
                &files,
                &include_dirs,
                board.as_deref(),
                chip.as_deref(),
                None,
                opt_level.as_deref(),
            ) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::from(2);
                }
            };
            match compile_to_mir_with_includes_and_targets(
                &built.files,
                &built.include_dirs,
                built.spec.hal_dir.as_deref(),
                built.spec.board_module.as_deref(),
            ) {
                Ok((_, mir)) => {
                    let default_ext = match built.spec.image {
                        zev_codegen_llvm::ImageFormat::Uf2 { .. } => "uf2",
                        zev_codegen_llvm::ImageFormat::RawBin => "bin",
                        zev_codegen_llvm::ImageFormat::ElfOnly => "elf",
                    };
                    let out_path = output
                        .unwrap_or_else(|| default_output_with_ext(&built.files, default_ext));
                    match zev_codegen_llvm::build_target_image(&mir, &built.opts, &out_path) {
                        Ok(res) => {
                            if emit_ll {
                                let ll_out = out_path.with_extension("ll");
                                if res.elf.ll_path != ll_out {
                                    if let Err(e) = std::fs::copy(&res.elf.ll_path, &ll_out) {
                                        eprintln!(
                                            "error: failed to copy LLVM IR to `{}`: {e}",
                                            ll_out.display()
                                        );
                                        return ExitCode::FAILURE;
                                    }
                                }
                                println!("Emitted LLVM IR: {}", ll_out.display());
                            }
                            print_target_build_summary(&built.spec, &res);
                            ExitCode::SUCCESS
                        }
                        Err(err) => {
                            eprintln!("error: {err}");
                            ExitCode::FAILURE
                        }
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Flash {
            files,
            include_dirs,
            board,
            chip,
            port,
            opt_level,
            output,
        } => {
            let built = match resolve_project_build_inputs(
                &files,
                &include_dirs,
                board.as_deref(),
                chip.as_deref(),
                port.as_deref(),
                opt_level.as_deref(),
            ) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::from(2);
                }
            };
            match compile_to_mir_with_includes_and_targets(
                &built.files,
                &built.include_dirs,
                built.spec.hal_dir.as_deref(),
                built.spec.board_module.as_deref(),
            ) {
                Ok((_, mir)) => {
                    let ext = match built.spec.image {
                        zev_codegen_llvm::ImageFormat::Uf2 { .. } => "uf2",
                        zev_codegen_llvm::ImageFormat::RawBin => "bin",
                        zev_codegen_llvm::ImageFormat::ElfOnly => "elf",
                    };
                    let mut temp_holder: Option<TempWorkDir> = None;
                    let out_path = match output {
                        Some(p) => p,
                        None => {
                            let temp = match TempWorkDir::create("zevc-flash") {
                                Ok(t) => t,
                                Err(err) => {
                                    eprintln!("error: {err}");
                                    return ExitCode::FAILURE;
                                }
                            };
                            let p = temp.dir.join(format!("firmware.{ext}"));
                            temp_holder = Some(temp);
                            p
                        }
                    };
                    let res = match zev_codegen_llvm::build_target_image(&mir, &built.opts, &out_path)
                    {
                        Ok(r) => r,
                        Err(err) => {
                            eprintln!("error: {err}");
                            return ExitCode::FAILURE;
                        }
                    };
                    print_target_build_summary(&built.spec, &res);
                    let target_label = built
                        .spec
                        .board_name
                        .as_deref()
                        .unwrap_or(&built.spec.chip_name);
                    let port_display = built
                        .port
                        .as_deref()
                        .or(built.spec.default_port.as_deref())
                        .unwrap_or("<default>");
                    println!("Flashing {target_label} on `{port_display}`...");
                    match zev_codegen_llvm::flash_target_image(
                        &built.spec,
                        &res.image_path,
                        built.port.as_deref(),
                    ) {
                        Ok(flash_out) => {
                            print!("{flash_out}");
                            drop(temp_holder);
                            ExitCode::SUCCESS
                        }
                        Err(err) => {
                            eprintln!("error: {err}");
                            ExitCode::FAILURE
                        }
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::BuildElf {
            files,
            include_dirs,
            opt_level,
            output,
        } => {
            match compile_to_mir_with_includes(&files, &include_dirs) {
            Ok((_, mir)) => {
                let out_elf = output.unwrap_or_else(|| default_output_with_ext(&files, "elf"));
                let mut opts = zev_codegen_llvm::CodegenOptions::default();
                opts.opt_level = opt_level;
                match zev_codegen_llvm::build_elf(&mir, &out_elf, &opts) {
                    Ok(res) => {
                        println!(
                            "Built freestanding ARM Cortex-M3 ELF ({}): {}",
                            opts.opt_level,
                            res.elf_path.display()
                        );
                        print!("{}", res.sizes.raw_output);
                        println!(
                            "Zero-heap audit: undefined_symbols={}, allocator_symbols={}",
                            res.audit.undefined_symbols.len(),
                            res.audit.heap_or_libc_symbols.len()
                        );
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
            }
        }
        Command::RunQemu {
            files,
            include_dirs,
            max_ticks,
            opt_level,
        } => {
            let mut opts = zev_codegen_llvm::CodegenOptions::lm3s6965();
            opts.opt_level = opt_level;
            if let Some(ticks) = max_ticks {
                opts.max_ticks = ticks.min(u32::MAX as u64) as u32;
            }
            match compile_to_mir_with_includes_and_targets(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
            ) {
            Ok((_, mir)) => {
                let temp = match TempWorkDir::create("zevc-qemu") {
                    Ok(t) => t,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                let elf_path = temp.dir.join("firmware.elf");
                let qemu_opts = zev_codegen_llvm::QemuRunOptions::default();
                match zev_codegen_llvm::run_qemu_cortex_m(&mir, &elf_path, &opts, &qemu_opts) {
                    Ok(res) => {
                        if !res.exec.stdout.is_empty() {
                            print!("{}", res.exec.stdout);
                        }
                        if !res.exec.stderr.is_empty() {
                            eprint!("{}", res.exec.stderr);
                        }
                        if res.exec.exit_code == 0 {
                            ExitCode::SUCCESS
                        } else {
                            ExitCode::from((res.exec.exit_code.clamp(1, 255)) as u8)
                        }
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
            }
        }
        Command::BuildEsp32C6 {
            files,
            include_dirs,
            opt_level,
            output,
        } => {
            let mut opts = zev_codegen_llvm::CodegenOptions::esp32c6();
            opts.opt_level = opt_level;
            match compile_to_mir_with_includes_and_targets(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
            ) {
            Ok((_, mir)) => {
                let out_bin = output.unwrap_or_else(|| default_output_with_ext(&files, "bin"));
                let out_elf = out_bin.with_extension("elf");
                match zev_codegen_llvm::build_esp32c6_image_with_options(
                    &mir, &out_elf, &out_bin, &opts,
                ) {
                    Ok(res) => {
                        print_esp32c6_summary(&res);
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
            }
        }
        Command::FlashEsp32C6 {
            files,
            include_dirs,
            port,
            opt_level,
            output,
        } => {
            let mut opts = zev_codegen_llvm::CodegenOptions::esp32c6();
            opts.opt_level = opt_level;
            match compile_to_mir_with_includes_and_targets(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
            ) {
            Ok((_, mir)) => {
                let mut temp_holder: Option<TempWorkDir> = None;
                let out_bin = match output {
                    Some(p) => p,
                    None => {
                        let temp = match TempWorkDir::create("zevc-esp32c6") {
                            Ok(t) => t,
                            Err(err) => {
                                eprintln!("error: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let p = temp.dir.join("firmware.bin");
                        temp_holder = Some(temp);
                        p
                    }
                };
                let out_elf = out_bin.with_extension("elf");
                let res = match zev_codegen_llvm::build_esp32c6_image_with_options(
                    &mir, &out_elf, &out_bin, &opts,
                ) {
                    Ok(r) => r,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                print_esp32c6_summary(&res);
                println!("Flashing ESP32-C6 on `{port}` at offset 0x0...");
                match zev_codegen_llvm::flash_esp32c6_bin(&res.bin_path, &port) {
                    Ok(esptool_out) => {
                        print!("{esptool_out}");
                        drop(temp_holder);
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
            }
        }
        Command::BuildAvr {
            files,
            include_dirs,
            opt_level,
            output,
        } => match compile_to_mir_with_includes(&files, &include_dirs) {
            Ok((_, mir)) => {
                let out_bin = output.unwrap_or_else(|| default_output_with_ext(&files, "bin"));
                let out_elf = out_bin.with_extension("elf");
                let mut opts = zev_codegen_llvm::CodegenOptions::avr_uno();
                opts.opt_level = opt_level;
                match zev_codegen_llvm::build_avr_uno_image_with_options(
                    &mir, &out_elf, &out_bin, &opts,
                ) {
                    Ok(res) => {
                        print_avr_summary(&res);
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        Command::FlashAvr {
            files,
            include_dirs,
            port,
            opt_level,
            output,
        } => match compile_to_mir_with_includes(&files, &include_dirs) {
            Ok((_, mir)) => {
                let mut temp_holder: Option<TempWorkDir> = None;
                let out_bin = match output {
                    Some(p) => p,
                    None => {
                        let temp = match TempWorkDir::create("zevc-avr") {
                            Ok(t) => t,
                            Err(err) => {
                                eprintln!("error: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let p = temp.dir.join("firmware.bin");
                        temp_holder = Some(temp);
                        p
                    }
                };
                let out_elf = out_bin.with_extension("elf");
                let mut opts = zev_codegen_llvm::CodegenOptions::avr_uno();
                opts.opt_level = opt_level;
                let res = match zev_codegen_llvm::build_avr_uno_image_with_options(
                    &mir, &out_elf, &out_bin, &opts,
                ) {
                    Ok(r) => r,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                print_avr_summary(&res);
                println!("Flashing Arduino Uno (ATmega328P) on `{port}` via STK500v1...");
                match zev_codegen_llvm::flash_avr_uno_bin(&res.bin_path, &port) {
                    Ok(flash_out) => {
                        print!("{flash_out}");
                        drop(temp_holder);
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        Command::BuildAvrNanoEvery {
            files,
            include_dirs,
            opt_level,
            output,
        } => match compile_to_mir_with_includes(&files, &include_dirs) {
            Ok((_, mir)) => {
                let out_bin = output.unwrap_or_else(|| default_output_with_ext(&files, "bin"));
                let out_elf = out_bin.with_extension("elf");
                let mut opts = zev_codegen_llvm::CodegenOptions::avr_nano_every();
                opts.opt_level = opt_level;
                match zev_codegen_llvm::build_avr_nano_every_image_with_options(
                    &mir, &out_elf, &out_bin, &opts,
                ) {
                    Ok(res) => {
                        print_avr_nano_every_summary(&res);
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        Command::FlashAvrNanoEvery {
            files,
            include_dirs,
            port,
            opt_level,
            output,
        } => match compile_to_mir_with_includes(&files, &include_dirs) {
            Ok((_, mir)) => {
                let mut temp_holder: Option<TempWorkDir> = None;
                let out_bin = match output {
                    Some(p) => p,
                    None => {
                        let temp = match TempWorkDir::create("zevc-nano-every") {
                            Ok(t) => t,
                            Err(err) => {
                                eprintln!("error: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let p = temp.dir.join("firmware.bin");
                        temp_holder = Some(temp);
                        p
                    }
                };
                let out_elf = out_bin.with_extension("elf");
                let mut opts = zev_codegen_llvm::CodegenOptions::avr_nano_every();
                opts.opt_level = opt_level;
                let res = match zev_codegen_llvm::build_avr_nano_every_image_with_options(
                    &mir, &out_elf, &out_bin, &opts,
                ) {
                    Ok(r) => r,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                print_avr_nano_every_summary(&res);
                println!(
                    "Flashing Arduino Nano Every (ATmega4809) on `{port}` via MuxTO UPDI..."
                );
                match zev_codegen_llvm::flash_avr_nano_every_bin(&res.bin_path, &port) {
                    Ok(flash_out) => {
                        print!("{flash_out}");
                        drop(temp_holder);
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        Command::BuildRp2040 {
            files,
            include_dirs,
            opt_level,
            output,
        } => {
            let mut opts = zev_codegen_llvm::CodegenOptions::rp2040();
            opts.opt_level = opt_level;
            match compile_to_mir_with_includes_and_targets(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
            ) {
            Ok((_, mir)) => {
                let out_uf2 = output.unwrap_or_else(|| default_output_with_ext(&files, "uf2"));
                let out_elf = out_uf2.with_extension("elf");
                match zev_codegen_llvm::build_rp2040_image_with_options(
                    &mir, &out_elf, &out_uf2, &opts,
                ) {
                    Ok(res) => {
                        print_rp2040_summary(&res);
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
            }
        },
        Command::FlashRp2040 {
            files,
            include_dirs,
            port,
            opt_level,
            output,
        } => {
            let mut opts = zev_codegen_llvm::CodegenOptions::rp2040();
            opts.opt_level = opt_level;
            match compile_to_mir_with_includes_and_targets(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
            ) {
            Ok((_, mir)) => {
                let mut temp_holder: Option<TempWorkDir> = None;
                let out_uf2 = match output {
                    Some(p) => p,
                    None => {
                        let temp = match TempWorkDir::create("zevc-rp2040") {
                            Ok(t) => t,
                            Err(err) => {
                                eprintln!("error: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let p = temp.dir.join("firmware.uf2");
                        temp_holder = Some(temp);
                        p
                    }
                };
                let out_elf = out_uf2.with_extension("elf");
                let res = match zev_codegen_llvm::build_rp2040_image_with_options(
                    &mir, &out_elf, &out_uf2, &opts,
                ) {
                    Ok(r) => r,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                print_rp2040_summary(&res);
                println!("Flashing Raspberry Pi RP2040 (`{port}` -> `RPI-RP2`)...");
                match zev_codegen_llvm::flash_rp2040_uf2(&res.uf2_path, &port) {
                    Ok(flash_out) => {
                        print!("{flash_out}");
                        drop(temp_holder);
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::render_check_json_ok;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn checks_emits_and_runs_north_star_sim_demo_cortex_m_and_esp32c6() {
        let root = repo_root();
        let (_, mc_mir) = compile_to_mir(&[
            root.join("examples/motor_controller.zev"),
            root.join("examples/motor_support.zev"),
        ])
        .expect("motor_controller + motor_support must check & lower");
        assert!(mc_mir.ram_report.mcu32.union_bytes_saved > 0);
        assert_eq!(mc_mir.ram_report.mcu32.heap_bytes, 0);

        let mut mc_opts = zev_codegen_llvm::CodegenOptions::host_linux();
        mc_opts.max_ticks = 25;
        let mc_ll = zev_codegen_llvm::emit_llvm_ir_with_options(&mc_mir, &mc_opts).unwrap();
        let mc_exit = runner::compile_and_run_host_llvm(&mc_ll, true, &[]).unwrap();
        assert_eq!(mc_exit, 0);

        let (_, sim_mir) =
            compile_to_mir(&[root.join("examples/sim_demo.zev")]).expect("sim_demo must check");
        let sim_ll = zev_codegen_llvm::emit_llvm_ir_with_options(
            &sim_mir,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        let sim_exit = runner::compile_and_run_host_llvm(&sim_ll, true, &[]).unwrap();
        assert_eq!(sim_exit, 0);

        let (_, cm3_mir) = compile_to_mir(&[root.join("examples/cortex_m_demo.zev")])
            .expect("cortex_m_demo must check");
        let ll = zev_codegen_llvm::emit_llvm_ir(&cm3_mir).expect("emit_llvm_ir must succeed");
        assert!(ll.contains(".isr_vector"));
        let temp = TempWorkDir::create("zevc-cli-test-qemu").unwrap();
        let elf_path = temp.dir.join("cm3.elf");
        let qres = zev_codegen_llvm::run_qemu_cortex_m(
            &cm3_mir,
            &elf_path,
            &zev_codegen_llvm::CodegenOptions::default(),
            &zev_codegen_llvm::QemuRunOptions::default(),
        )
        .expect("QEMU execution must succeed");
        assert_eq!(qres.exec.exit_code, 0);
        assert!(qres.build.audit.is_zero_heap_freestanding());

        let (_, esp_mir) = compile_to_mir(&[root.join("examples/esp32c6_demo.zev")])
            .expect("esp32c6_demo must check");
        let esp_elf = temp.dir.join("esp32c6_demo.elf");
        let esp_bin = temp.dir.join("esp32c6_demo.bin");
        let esp_res = zev_codegen_llvm::build_esp32c6_image(&esp_mir, &esp_elf, &esp_bin)
            .expect("ESP32-C6 Direct-Boot build must succeed");
        assert_eq!(
            esp_res.magic_header,
            zev_codegen_llvm::ESP32C6_DIRECT_BOOT_MAGIC
        );
        assert!(esp_res.elf.audit.is_zero_heap_freestanding());
        assert!(esp_res.bin_size > 8);
    }

    fn check_temp_file(name: &str, src: &str) -> Result<CheckedBundle, CheckFailure> {
        let temp = TempWorkDir::create("zevc-cli-diag").unwrap();
        let p = temp.dir.join(name);
        std::fs::write(&p, src).unwrap();
        load_and_check_detailed(&[p])
    }

    #[test]
    fn parser_foreign_idiom_traps_and_json_output_work_end_to_end() {
        let cases = [
            ("p1.zev", "fn main() { let x = 1; }\n", "P0001"),
            ("p2.zev", "fn f(x: u16) -> u16 { return x; }\n", "P0002"),
            ("p3.zev", "async fn f() {}\n", "P0003"),
            ("p4.zev", "fn main() { select { } }\n", "P0004"),
            ("p5.zev", "fn main() { match x { } }\n", "P0005"),
            (
                "p6.zev",
                "enum E { A }\nu16 f() error E { error A; }\nfn main() { u16 v = f() catch |e| { return; }; }\n",
                "P0006",
            ),
            ("p7.zev", "!u16 f() { return 1; }\n", "P0007"),
            ("p8.zev", "fn main() { for (u32 i = 0; i < 4; i += 1) {} }\n", "P0008"),
            ("p9a.zev", "struct Reader<'a> { []u8 buf; }\n", "P0009"),
            ("p9b.zev", "fn step(&mut i32 acc) {}\n", "P0009"),
            ("p10.zev", "static u32 x = 0;\nfn main() {}\n", "P0010"),
            (
                "p11.zev",
                "task ctrl() { every 5ms { checkpoint(); } }\nfn main() { spawn ctrl() @normal; }\n",
                "P0011",
            ),
            ("p12.zev", "chan<u32, 4> q;\nfn main() {}\n", "P0012"),
            ("p14.zev", "fn main() { u8 buf[8]; }\n", "P0014"),
        ];

        for (fname, src, expected_code) in cases {
            let Err(fail) = check_temp_file(fname, src) else {
                panic!("expected `{fname}` to fail with `{expected_code}`");
            };
            assert!(
                fail.human.contains(&format!("error[{expected_code}]")),
                "missing error[{expected_code}] in:\n{}",
                fail.human
            );
            assert!(
                fail.human.contains(&format!("zevc doc {expected_code}")),
                "missing doc cmd in:\n{}",
                fail.human
            );
            let json = render_check_json_err(&fail.json_diags);
            assert!(json.contains(&format!("\"code\": \"{expected_code}\"")), "json={json}");
            assert!(
                json.contains(&format!("\"doc_cmd\": \"zevc doc {expected_code}\"")),
                "json={json}"
            );
            assert!(docs::lookup_doc(Some(expected_code), false).is_ok());
        }
    }

    #[test]
    fn adversarial_safety_suite_yield_chains_and_method_hints_work_end_to_end() {
        let Err(f_chain) = check_temp_file(
            "chain.zev",
            "fn f2() { checkpoint(); }\nfn f1() { f2(); }\nisr EXTI0 { f1(); }\nfn main() {}\n",
        ) else {
            panic!("expected 3-hop yield chain in ISR to fail");
        };
        assert!(f_chain.human.contains("error[E0005]"), "human=\n{}", f_chain.human);
        assert!(f_chain.human.contains("f1"), "human=\n{}", f_chain.human);
        assert!(f_chain.human.contains("f2"), "human=\n{}", f_chain.human);
        assert!(f_chain.human.contains("zevc doc E0005"), "human=\n{}", f_chain.human);

        let Err(f_meth) = check_temp_file(
            "bad_meth.zev",
            "enum Ev { Ping }\ninbox<Ev, 4> q;\nfn main() { q.push(Ping); }\n",
        ) else {
            panic!("expected invalid inbox method to fail");
        };
        assert!(f_meth.human.contains("error[E0020]"), "human=\n{}", f_meth.human);
        assert!(f_meth.human.contains("post"), "human=\n{}", f_meth.human);
        let json = render_check_json_err(&f_meth.json_diags);
        assert!(json.contains("\"code\": \"E0020\""), "json={json}");
        assert!(json.contains("\"doc_cmd\": \"zevc doc E0020\""), "json={json}");

        let ok = check_temp_file(
            "partial_ok.zev",
            "enum IoErr { Timeout }\nu32 read_chunk() error IoErr {\n    error Timeout, 12;\n}\nfn main() {\n    u32 n = read_chunk() catch (IoErr e, u32 partial) {\n        return;\n    };\n}\n",
        )
        .unwrap_or_else(|e| panic!("partial_ok.zev must pass:\n{}", e.human));
        let ok_dual: Vec<String> = ok.sema.dual_specialized_fns.iter().cloned().collect();
        let ok_json = render_check_json_ok(
            1,
            ok.sema.spawned_tasks.len(),
            ok.sema.primitives.len(),
            ok.sema.tick_period_us,
            &ok_dual,
            &ok.sema.monomorphized_instances,
        );
        assert!(ok_json.starts_with("{\"ok\": true"));
        assert!(ok_json.contains("\"dual_specialized_functions\": []"), "json={ok_json}");
        assert!(ok_json.contains("\"monomorphized_instances\": []"), "json={ok_json}");
    }

    #[test]
    fn check_json_and_human_report_dual_specialized_and_wave1_sema_codes() {
        let dual_src = "\
watch<u32> latest = 0;

u32 hash_chunk(u32 seed) {
    u32 acc = seed;
    for (u32 i in 0..4) {
        acc = (acc * 33) ^ i;
        checkpoint();
    }
    return acc;
}

task bg_hasher() {
    u32 round = 1;
    loop every 5ms {
        u32 h = hash_chunk(round);
        latest.publish(h);
        round += 1;
    }
}

fn main() {
    sys.use_systick(12_000_000);
    noyield {
        u32 boot_h = hash_chunk(42);
        latest.publish(boot_h);
    }
    spawn bg_hasher() @normal;
}
";
        let bundle = check_temp_file("dual.zev", dual_src)
            .unwrap_or_else(|e| panic!("dual.zev must check cleanly:\n{}", e.human));
        let dual_vec: Vec<String> = bundle.sema.dual_specialized_fns.iter().cloned().collect();
        assert_eq!(dual_vec, vec!["hash_chunk".to_string()]);

        let json = render_check_json_ok(
            1,
            bundle.sema.spawned_tasks.len(),
            bundle.sema.primitives.len(),
            bundle.sema.tick_period_us,
            &dual_vec,
            &bundle.sema.monomorphized_instances,
        );
        assert!(
            json.contains("\"dual_specialized_functions\": [\"hash_chunk\"]"),
            "json={json}"
        );

        let human = format_check_human_ok(
            1,
            bundle.sema.spawned_tasks.len(),
            bundle.sema.primitives.len(),
            bundle.sema.tick_period_us,
            &dual_vec,
            &bundle.sema.monomorphized_instances,
        );
        assert!(
            human.contains(", dual_specialized=[hash_chunk]"),
            "human={human}"
        );

        // Verify Wave 1 E0025, E0026, E0027 codes & doc pointers end-to-end
        let wave1_errs = [
            (
                "e25.zev",
                "fn main() { return; u32 dead = 1; }\n",
                "E0025",
            ),
            (
                "e26.zev",
                "task one_shot() { checkpoint(); }\nfn main() { spawn one_shot() @normal; }\n",
                "E0026",
            ),
            (
                "e27.zev",
                "task worker() { loop { checkpoint(); } }\nfn main() { for (u32 i in 0..2) { spawn worker() @normal; } }\n",
                "E0027",
            ),
        ];
        for (fname, src, code) in wave1_errs {
            let Err(fail) = check_temp_file(fname, src) else {
                panic!("expected `{fname}` to fail with `{code}`");
            };
            assert!(fail.human.contains(&format!("error[{code}]")), "human=\n{}", fail.human);
            assert!(fail.human.contains(&format!("zevc doc {code}")), "human=\n{}", fail.human);
            assert!(docs::lookup_doc(Some(code), false).is_ok());
        }
    }

    #[test]
    fn wave2_monomorphized_generics_check_json_ram_and_fmt_buf_fit_vs_catch() {
        let wave2_src = "\
struct FixedVec<T, usize N> {
    [N]T data;
    usize len;
}

T clamp<T>(T val, T lo, T hi) {
    if (val < lo) {
        return lo;
    }
    if (val > hi) {
        return hi;
    }
    return val;
}

fn check_dynamic_slice([]u8 dyn_tag) {
    // 1. Statically proven buffer fit (`worst = 4 + 5 = 9 <= 16`) requires NO `or`/`catch`!
    [16]u8 static_buf;
    u16 rpm = 2400;
    []u8 fit_line = fmt_buf(static_buf, f\"rpm={rpm}\");
    println(fit_line);

    // 2. Dynamic `[]u8` slice hole uses `catch (FmtErr err, []u8 partial)` cleanly!
    [10]u8 small_buf;
    []u8 trunc_line = fmt_buf(small_buf, f\"tag={dyn_tag}\") catch (FmtErr err, []u8 partial) {
        partial
    };
    println(trunc_line);
}

fn main() {
    FixedVec<u32, 8> vec = FixedVec<u32, 8> {
        data: [1, 2, 3, 4, 5, 6, 7, 8],
        len: 8
    };
    f32 ratio = clamp<f32>(1.75f32, 0.0f32, 1.0f32);
    [12]u8 long_tag = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76];
    check_dynamic_slice(long_tag[0..12]);
    println(f\"len={vec.len} ratio={ratio:.2}\");
    sys.exit(0);
}
";
        let temp = TempWorkDir::create("zevc-wave2").unwrap();
        let src_path = temp.dir.join("wave2.zev");
        std::fs::write(&src_path, wave2_src).unwrap();

        let (bundle, mir) = compile_to_mir(&[src_path])
            .unwrap_or_else(|e| panic!("wave2.zev must check & lower:\n{e}"));

        // 1. Verify `zevc check --json` outputs `monomorphized_instances`
        let dual_vec: Vec<String> = bundle.sema.dual_specialized_fns.iter().cloned().collect();
        let json = render_check_json_ok(
            1,
            bundle.sema.spawned_tasks.len(),
            bundle.sema.primitives.len(),
            bundle.sema.tick_period_us,
            &dual_vec,
            &bundle.sema.monomorphized_instances,
        );
        assert!(
            json.contains("{\"kind\": \"struct\", \"signature\": \"FixedVec<u32, 8>\", \"mangled_name\": \"FixedVec__u32__8\"}"),
            "json={json}"
        );
        assert!(
            json.contains("{\"kind\": \"fn\", \"signature\": \"clamp<f32>\", \"mangled_name\": \"clamp__f32\"}"),
            "json={json}"
        );

        let human = format_check_human_ok(
            1,
            bundle.sema.spawned_tasks.len(),
            bundle.sema.primitives.len(),
            bundle.sema.tick_period_us,
            &dual_vec,
            &bundle.sema.monomorphized_instances,
        );
        assert!(
            human.contains("monomorphized=[FixedVec<u32, 8>, clamp<f32>]"),
            "human={human}"
        );

        // 2. Verify `zevc ram` prints Section 7 (`Monomorphized Generic Instantiations`)
        let ram_table = mir.ram_report.format_table();
        assert!(
            ram_table.contains("7. Monomorphized Generic Instantiations"),
            "ram_table=\n{ram_table}"
        );
        assert!(ram_table.contains("FixedVec<u32, 8>"), "ram_table=\n{ram_table}");
        assert!(ram_table.contains("clamp<f32>"), "ram_table=\n{ram_table}");

        // 3. Verify emitted host LLVM IR runs cleanly under ASan/UBSan (`fit_line` + `trunc_line` + `println(f"...")`)
        let mut wave2_host_opts = zev_codegen_llvm::CodegenOptions::host_linux();
        wave2_host_opts.max_ticks = 10;
        let wave2_ll = zev_codegen_llvm::emit_llvm_ir_with_options(&mir, &wave2_host_opts).unwrap();
        let host_exit = runner::compile_and_run_host_llvm(&wave2_ll, true, &[]).unwrap();
        assert_eq!(host_exit, 0);

        // 4. Verify E0028..E0033 diagnostics and `zevc doc` explainers (45 total)
        let doc_list = docs::lookup_doc(None, true).unwrap();
        assert!(doc_list.contains("E0028") && doc_list.contains("E0033"));

        let wave2_errs = [
            (
                "e28.zev",
                "fn main() { [8]u8 dst; u32 rpm = 1800; []u8 s = fmt_buf(dst, f\"rpm={rpm}\"); }\n",
                "E0028",
            ),
            (
                "e29.zev",
                "struct FixedVec<T, usize N> { [N]T data; }\nfn main() { usize n = 4; FixedVec<u8, n> v; }\n",
                "E0029",
            ),
        ];
        for (fname, src, code) in wave2_errs {
            let Err(fail) = check_temp_file(fname, src) else {
                panic!("expected `{fname}` to fail with `{code}`");
            };
            assert!(fail.human.contains(&format!("error[{code}]")), "human=\n{}", fail.human);
            assert!(fail.human.contains(&format!("zevc doc {code}")), "human=\n{}", fail.human);
            assert!(docs::lookup_doc(Some(code), false).is_ok());
        }
    }

    #[test]
    fn multi_file_generic_disambiguation_and_duplicate_path_dedup() {
        let lib_src = "\
const u32 A = 2;
const u32 B = 5;
const u32 C = 9;
const u32 D = 3;

struct FixedVec<T, usize N> {
    [N]T data;
    usize len;
}

T scale_add<T, usize N>(T v) {
    return v;
}

bool check_window(bool lo_ok, bool hi_ok) {
    return lo_ok && hi_ok;
}
";
        let app_src = "\
fn main() {
    FixedVec<u32, 4> buf = FixedVec<u32, 4> { data: [1, 2, 3, 4], len: 4 };
    u32 x = scale_add<u32, 4>(buf.data[0]);
    bool ok = check_window(A < B, C > (D + 1));
    if (ok && x == 1) {
        return;
    }
}
";
        let temp = TempWorkDir::create("zevc-multifile-gen").unwrap();
        let lib_path = temp.dir.join("lib.zev");
        let app_path = temp.dir.join("app.zev");
        std::fs::write(&lib_path, lib_src).unwrap();
        std::fs::write(&app_path, app_src).unwrap();

        // Pass `app_path` twice to also verify canonical input path deduplication (no E0021!)
        let bundle = load_and_check_detailed(&[lib_path, app_path.clone(), app_path])
            .unwrap_or_else(|e| panic!("multi-file generic + comparison check failed:\n{}", e.human));
        assert_eq!(bundle.programs.len(), 2);

        let dual_vec: Vec<String> = bundle.sema.dual_specialized_fns.iter().cloned().collect();
        let json = render_check_json_ok(
            bundle.programs.len(),
            bundle.sema.spawned_tasks.len(),
            bundle.sema.primitives.len(),
            bundle.sema.tick_period_us,
            &dual_vec,
            &bundle.sema.monomorphized_instances,
        );
        assert!(json.contains("\"files\": 2"), "json={json}");
        assert!(
            json.contains("\"mangled_name\": \"FixedVec__u32__4\""),
            "json={json}"
        );
        assert!(
            json.contains("\"mangled_name\": \"scale_add__u32__4\""),
            "json={json}"
        );
    }

    #[test]
    fn cross_target_parity_const_switch_multistmt_catch_and_max_ticks() {
        let Err(f_print) = check_temp_file("bad_print.zev", "fn main() { print(123); }\n") else {
            panic!("print(123) must be rejected by zev-sema");
        };
        assert!(f_print.human.contains("error[E0019]"), "human=\n{}", f_print.human);

        let Err(f_sim) = check_temp_file("bad_sim.zev", "fn main() { sim.log(1, 2, 3); }\n") else {
            panic!("sim.log must be rejected by zev-sema");
        };
        assert!(f_sim.human.contains("error[E0020]"), "human=\n{}", f_sim.human);

        let parity_src = "\
const u8 CMD_HEAT = 7;
const u8 CMD_COOL = 9;

struct DecodedPacket {
    u8  cmd;
    u16 temp;
}

enum PacketErr { Corrupt }

DecodedPacket decode_bad() error PacketErr {
    error Corrupt;
}

task monitor() {
    loop every 5ms {
        println(\"tick\");
    }
}

fn main() {
    sys.use_systick(12_000_000);
    u32 errs = 0;
    DecodedPacket p = decode_bad() catch (PacketErr e) {
        errs += 1;
        DecodedPacket{ cmd: CMD_COOL, temp: 42 }
    };
    switch (p.cmd) {
        CMD_HEAT => { sys.log_kv(\"arm=\", 1); }
        CMD_COOL => { sys.log_kv(\"arm=\", 2); }
        _        => { sys.log_kv(\"arm=\", 0); }
    }
    u16 cur = 500;
    i32 diff = 300 - (cur as i32);
    sys.log_kv(\"diff=\", diff);
    sys.log_kv(\"temp=\", p.temp as i32);
    spawn monitor() @normal;
    set_priority(@low);
    loop {
        checkpoint();
    }
}
";
        let temp = TempWorkDir::create("zevc-parity").unwrap();
        let src_path = temp.dir.join("parity.zev");
        std::fs::write(&src_path, parity_src).unwrap();

        let (_, mir) = compile_to_mir(&[src_path]).expect("parity.zev must check & lower");

        // 1. Verify host LLVM simulation with --max-ticks 15
        let mut host_opts = zev_codegen_llvm::CodegenOptions::host_linux();
        host_opts.max_ticks = 15;
        let host_ll = zev_codegen_llvm::emit_llvm_ir_with_options(&mir, &host_opts).unwrap();
        let host_exit = runner::compile_and_run_host_llvm(&host_ll, true, &[]).unwrap();
        assert_eq!(host_exit, 0);

        // 2. Verify Cortex-M3 QEMU with --max-ticks 15
        let mut q_opts = zev_codegen_llvm::CodegenOptions::default();
        q_opts.max_ticks = 15;
        let elf_path = temp.dir.join("parity.elf");
        let qres = zev_codegen_llvm::run_qemu_cortex_m(
            &mir,
            &elf_path,
            &q_opts,
            &zev_codegen_llvm::QemuRunOptions::default(),
        )
        .expect("QEMU parity run must succeed");
        assert_eq!(qres.exec.exit_code, 0);
        assert!(qres.exec.stdout.contains("arm=2"), "out={}", qres.exec.stdout);
        assert!(qres.exec.stdout.contains("diff=-200"), "out={}", qres.exec.stdout);
        assert!(qres.exec.stdout.contains("temp=42"), "out={}", qres.exec.stdout);
        assert!(qres.exec.stdout.contains("tick"), "out={}", qres.exec.stdout);
    }

    #[test]
    fn arduino_uno_example_checks_and_avr_target_resolves() {
        for alias in ["avr", "atmega328p", "uno", "arduino-uno"] {
            let opts = resolve_llvm_options(Some(alias))
                .unwrap_or_else(|e| panic!("target {alias} failed: {e}"));
            assert_eq!(opts.target, zev_codegen_llvm::TargetArch::AvrAtmega328p);
        }

        let demo_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/arduino_uno_demo.zev");
        assert!(
            load_and_check_detailed(std::slice::from_ref(&demo_path)).is_ok(),
            "examples/arduino_uno_demo.zev must check cleanly"
        );
        let (_, mir) = compile_to_mir(&[demo_path]).expect("uno demo checks & lowers to MIR");
        assert_eq!(mir.ram_report.mcu32.heap_bytes, 0);
    }

    #[test]
    fn arduino_nano_every_example_checks_builds_and_ram_target_proves_stack() {
        for alias in [
            "nano-every",
            "arduino-nano-every",
            "atmega4809",
            "avr-mega0",
            "avrxmega3",
        ] {
            let opts = resolve_llvm_options(Some(alias))
                .unwrap_or_else(|e| panic!("target {alias} failed: {e}"));
            assert_eq!(opts.target, zev_codegen_llvm::TargetArch::AvrAtmega4809);
            assert_eq!(opts.flash_len, 49_152);
            assert_eq!(opts.ram_len, 6_144);
            assert_eq!(opts.opt_level, "-Os");
        }

        let root = repo_root();
        let nano_path = root.join("examples/arduino_nano_every_demo.zev");
        let (_, mir) = compile_to_mir(std::slice::from_ref(&nano_path))
            .expect("examples/arduino_nano_every_demo.zev must check & lower to MIR");
        assert_eq!(mir.ram_report.mcu32.heap_bytes, 0);

        let temp = TempWorkDir::create("zevc-cli-test-nano").unwrap();
        let nano_elf = temp.dir.join("arduino_nano_every_demo.elf");
        let nano_bin = temp.dir.join("arduino_nano_every_demo.bin");
        let nano_res = zev_codegen_llvm::build_avr_nano_every_image_with_options(
            &mir,
            &nano_elf,
            &nano_bin,
            &zev_codegen_llvm::CodegenOptions::avr_nano_every(),
        )
        .expect("ATmega4809 Nano Every image build must succeed");
        assert!(nano_res.elf.audit.is_zero_heap_freestanding());
        assert!(nano_res.bin_size > 160 && nano_res.bin_size <= 49_152);
        assert!(nano_res.sram_static_bytes < 5_632);

        match execute_ram_command(&[nano_path], &[], Some("nano-every"), None) {
            RamExecOutcome::Output {
                text,
                within_budget,
            } => {
                assert!(within_budget, "expected within_budget for nano-every:\n{text}");
                assert!(text.contains("STATIC ZERO-HEAP SRAM REPORT"));
                assert!(text.contains("HARDWARE SINGLE-STACK + .BSS HIGH-WATER PROOF"));
                assert!(text.contains("6144 B"), "missing 6144 B in:\n{text}");
                assert!(text.contains("PASS"));
            }
            _ => panic!("expected targeted ram output for nano-every"),
        }
    }

    #[test]
    fn pico_w_example_checks_and_rp2040_target_resolves() {
        for alias in [
            "rp2040",
            "pico",
            "pico-w",
            "cortex-m0plus",
            "thumbv6m",
            "thumbv6m-none-eabi",
        ] {
            let opts = resolve_llvm_options(Some(alias))
                .unwrap_or_else(|e| panic!("target {alias} failed: {e}"));
            assert_eq!(opts.target, zev_codegen_llvm::TargetArch::Thumbv6mRp2040);
        }

        let demo_path = repo_root().join("examples/pico_w_demo.zev");
        assert!(
            load_and_check_detailed(std::slice::from_ref(&demo_path)).is_ok(),
            "examples/pico_w_demo.zev must check cleanly"
        );
        let (_, mir) = compile_to_mir(&[demo_path]).expect("pico_w_demo checks & lowers to MIR");
        assert_eq!(mir.ram_report.mcu32.heap_bytes, 0);
    }

    #[test]
    fn ram_command_preserves_plain_output_and_prints_target_stack_proofs() {
        let root = repo_root();
        let cm3_path = root.join("examples/cortex_m_demo.zev");
        let esp_path = root.join("examples/esp32c6_demo.zev");
        let uno_path = root.join("examples/arduino_uno_demo.zev");
        let nano_path = root.join("examples/arduino_nano_every_demo.zev");
        let pico_path = root.join("examples/pico_w_demo.zev");

        // 1. Plain `zevc ram` (target = None) matches `mir.ram_report.format_table()` identically
        let (_, cm3_mir) = compile_to_mir(std::slice::from_ref(&cm3_path)).unwrap();
        match execute_ram_command(std::slice::from_ref(&cm3_path), &[], None, None) {
            RamExecOutcome::Output {
                text,
                within_budget,
            } => {
                assert!(within_budget);
                assert_eq!(text, cm3_mir.ram_report.format_table());
                assert!(!text.contains("HARDWARE SINGLE-STACK"));
            }
            _ => panic!("expected plain ram output"),
        }

        // 2. Targeted `zevc ram --target <cortex-m3|esp32c6|avr|nano-every|rp2040>` appends hardware proof and passes budget
        for (path, tgt, budget_str) in [
            (cm3_path, "cortex-m3", "65536 B"),
            (esp_path, "esp32c6", "524288 B"),
            (uno_path, "avr", "2048 B"),
            (nano_path, "nano-every", "6144 B"),
            (pico_path, "rp2040", "270336 B"),
        ] {
            match execute_ram_command(&[path], &[], Some(tgt), None) {
                RamExecOutcome::Output {
                    text,
                    within_budget,
                } => {
                    assert!(within_budget, "expected within_budget for {tgt}:\n{text}");
                    assert!(text.contains("STATIC ZERO-HEAP SRAM REPORT"));
                    assert!(text.contains("HARDWARE SINGLE-STACK + .BSS HIGH-WATER PROOF"));
                    assert!(text.contains(budget_str), "missing {budget_str} in:\n{text}");
                    assert!(text.contains("PASS"));
                }
                _ => panic!("expected targeted ram output for {tgt}"),
            }
        }

        // 3. `zevc ram --target host` is rejected as usage error
        assert!(matches!(
            execute_ram_command(&[root.join("examples/sim_demo.zev")], &[], Some("host"), None),
            RamExecOutcome::UsageErr(_)
        ));
    }

    #[test]
    fn board_flag_ram_and_zev_toml_project_resolution() {
        let root = repo_root();
        match execute_ram_command_with_spec(
            &[root.join("examples/pico_w_demo.zev")],
            &[],
            None,
            Some("rpi_pico_w"),
            None,
            None,
        ) {
            RamExecOutcome::Output {
                text,
                within_budget,
            } => {
                assert!(within_budget, "rpi_pico_w budget fail:\n{text}");
                assert!(text.contains("rp2040"));
                assert!(text.contains("270336 B"));
                assert!(text.contains("PASS"));
            }
            other => panic!(
                "expected RAM output for --board rpi_pico_w, got usage/compile error: {:?}",
                matches!(other, RamExecOutcome::Output { .. })
            ),
        }

        assert!(matches!(
            execute_ram_command_with_spec(
                &[root.join("examples/arduino_uno_demo.zev")],
                &[],
                Some("avr"),
                Some("arduino_uno"),
                None,
                None,
            ),
            RamExecOutcome::UsageErr(_)
        ));

        let temp = TempWorkDir::create("zevc-manifest-test").expect("temp dir");
        std::fs::create_dir_all(temp.dir.join("chips")).unwrap();
        std::fs::create_dir_all(temp.dir.join("boards")).unwrap();
        std::fs::create_dir_all(temp.dir.join("src")).unwrap();
        std::fs::write(
            temp.dir.join("chips/my_cm3.toml"),
            r#"[chip]
name = "my_cm3"
isa = "arm-cortex-m"
cpu = "cortex-m3"
triple = "thumbv7m-none-eabi"
datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
ptr_bits = 32
opt_level = "-O2"
has_thumb2 = true

[memory]
flash_base = 0x0000_0000
flash_size = 131072
ram_base = 0x2000_0000
ram_size = 32768
stack_reserve_bytes = 512

[irq]
model = "cortex-m-nvic"
max_irqs = 64
min_vector_irqs = 16

[image]
format = "elf"
"#,
        )
        .unwrap();
        std::fs::write(
            temp.dir.join("boards/my_board.toml"),
            r#"[board]
name = "my_board"
chip = "my_cm3"

[run]
default_port = "/dev/ttyUSB9"
"#,
        )
        .unwrap();
        std::fs::write(
            temp.dir.join("zev.toml"),
            r#"[project]
name = "custom_sensor"
entry = "src/app.zev"
board = "my_board"
port = "/dev/ttyACM7"
opt_level = "-Oz"
"#,
        )
        .unwrap();
        let entry_path = temp.dir.join("src/app.zev");
        std::fs::write(
            &entry_path,
            r#"
fn putc(u8 _c) {}
fn main() {
    sys.set_putc(putc);
    sys.exit(0);
}
"#,
        )
        .unwrap();

        let (mdir, manifest) =
            zev_codegen_llvm::ProjectManifest::discover(&temp.dir.join("src"))
                .expect("manifest parse")
                .expect("manifest found");
        assert_eq!(mdir, temp.dir);
        assert_eq!(manifest.entry.as_deref(), Some(Path::new("src/app.zev")));

        let resolved =
            resolve_project_build_inputs(&[entry_path], &[], None, None, None, None)
                .expect("resolve custom project");
        assert_eq!(resolved.spec.chip_name, "my_cm3");
        assert_eq!(resolved.spec.board_name.as_deref(), Some("my_board"));
        assert_eq!(resolved.spec.ram_size, 32768);
        assert_eq!(resolved.opts.opt_level, "-Oz");
        assert_eq!(resolved.port.as_deref(), Some("/dev/ttyACM7"));

        let (_, mir) =
            compile_to_mir_with_includes(&resolved.files, &resolved.include_dirs).unwrap();
        let out_elf = temp.dir.join("custom.elf");
        let built =
            zev_codegen_llvm::build_target_image(&mir, &resolved.opts, &out_elf).unwrap();
        assert!(built.image_path.is_file());
        assert_eq!(built.elf.audit.undefined_symbols.len(), 0);
    }

    #[test]
    fn host_fuzz_irqs_fires_attached_isr_on_crit_exit() {
        let temp = TempWorkDir::create("zev_fuzz_irqs_host").unwrap();
        let main_path = temp.dir.join("app.zev");
        std::fs::write(
            &main_path,
            r#"
watch<u32> ticks = 0;
irq IRQ_FUZZ = 3;

isr on_fuzz(*watch<u32> w) {
    w.publish(w.read() + 1);
}

task runner(*watch<u32> w) {
    for (u32 i in 0..40) {
        critical {
            sys.nop();
        }
        checkpoint();
    }
    u32 hits = w.read();
    println(f"fuzz_hits={hits}");
    assert(hits > 0);
    sys.exit(0);
}

fn main() {
    IRQ_FUZZ.attach(on_fuzz(&ticks));
    sys.fuzz_irqs(0xCAFE_BABEu32, 1u32 << 3);
    spawn runner(&ticks) @normal;
}
"#,
        )
        .unwrap();
        let (_, mir) = compile_to_mir_with_includes(&[main_path], &[]).unwrap();
        let ll = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .expect("emit host fuzz_irqs IR");
        let code = runner::compile_and_run_host_llvm(&ll, true, &[])
            .expect("host sanitize run should succeed");
        assert_eq!(code, 0);
    }

    #[test]
    fn host_fuzz_irqs_no_lost_wakeups_across_primitives() {
        let temp = TempWorkDir::create("zev_fuzz_lost_wakeups").unwrap();
        let main_path = temp.dir.join("app.zev");
        std::fs::write(
            &main_path,
            r#"
inbox<u32, 2> fq;
watch<u32> fq_sent = 0;

signal fsig;
watch<u32> sig_req = 0;
watch<u32> sig_ack = 0;

byte_ring<8> fring;
watch<u32> ring_sent = 0;

watch<u32> fw = 0;
watch<u32> w_req = 0;

pool<u32, 1> fpool;
inbox<Lease<u32>, 1> prelease_q;
signal pool_rel_sig;

watch<u32> done_mask = 0;

irq IRQ_INBOX = 1;
irq IRQ_SIG   = 2;
irq IRQ_RING  = 3;
irq IRQ_WATCH = 4;
irq IRQ_POOL  = 5;

isr on_fq(*inbox<u32, 2> q, *watch<u32> sent) {
    u32 s = sent.read();
    if (s < 35u32 && !q.is_full()) {
        q.try_post(s + 1u32, boost: @critical);
        sent.publish(s + 1u32);
    }
}

isr on_fsig(*signal s, *watch<u32> req, *watch<u32> ack) {
    u32 r = req.read();
    if (r > ack.read()) {
        ack.publish(r);
        s.notify(boost: @high);
    }
}

isr on_fring(*byte_ring<8> r, *watch<u32> sent) {
    u32 s = sent.read();
    if (s < 35u32 && r.free_space() > 0) {
        [1]u8 b = [(s & 0x7fu32) as u8];
        r.push(b[0..1]);
        sent.publish(s + 1u32);
    }
}

isr on_fwatch(*watch<u32> w) {
    w.publish(w.read() + 1u32);
}

isr on_fpool(*inbox<Lease<u32>, 1> rel_q, *signal done_s) {
    Lease<u32> l = rel_q.try_recv() catch { return; };
    l.release();
    done_s.notify(boost: @high);
}

u32 wait_watch_helper(*watch<u32> w) {
    return w.wait_change(timeout: 500ms) catch {
        return 999999u32;
    };
}

task t_inbox(*inbox<u32, 2> q, *watch<u32> dm) {
    for (u32 i in 0..35) {
        u32 v = q.recv();
        if (v != i + 1u32) {
            println(f"INBOX_FAIL i={i} v={v}");
            sys.exit(2);
        }
        critical { sys.nop(); }
    }
    critical { dm.publish(dm.read() | 1u32); }
    loop { delay(5000ms); }
}

task t_signal(*signal s, *watch<u32> req, *watch<u32> dm) {
    for (u32 i in 0..35) {
        req.publish(i + 1u32);
        critical { sys.nop(); }
        s.wait();
    }
    critical { dm.publish(dm.read() | 2u32); }
    loop { delay(5000ms); }
}

task t_ring(*byte_ring<8> r, *watch<u32> dm) {
    u32 got = 0u32;
    while (got < 35u32) {
        []u8 c = r.wait_read();
        for (usize j in 0..c.len) {
            u8 exp = (got & 0x7fu32) as u8;
            if (c[j] != exp) {
                println(f"RING_FAIL got={got} c={c[j]} exp={exp}");
                sys.exit(3);
            }
            got += 1u32;
        }
        r.consume(c.len);
        critical { sys.nop(); }
    }
    critical { dm.publish(dm.read() | 4u32); }
    loop { delay(5000ms); }
}

task t_watch(*watch<u32> w, *watch<u32> dm) {
    u32 prev = 0u32;
    for (u32 i in 0..20) {
        u32 v = 0u32;
        if ((i & 1u32) == 0u32) {
            v = wait_watch_helper(w);
        } else {
            v = w.wait_change(timeout: 500ms) catch { 999999u32 };
        }
        if (v == 999999u32 || v <= prev) {
            println(f"WATCH_FAIL i={i} v={v} prev={prev}");
            sys.exit(4);
        }
        prev = v;
        critical { sys.nop(); }
    }
    critical { dm.publish(dm.read() | 8u32); }
    loop { delay(5000ms); }
}

task t_pool(*pool<u32, 1> p, *inbox<Lease<u32>, 1> rel_q, *signal rel_s, *watch<u32> dm) {
    for (u32 i in 0..20) {
        Lease<u32> l1 = p.alloc();
        *u32 p1 = l1.get();
        *p1 = i;
        rel_q.send(l1);
        Lease<u32> l2 = p.alloc();
        *u32 p2 = l2.get();
        *p2 = i + 100u32;
        l2.release();
        rel_s.wait();
        critical { sys.nop(); }
    }
    critical { dm.publish(dm.read() | 16u32); }
    loop { delay(5000ms); }
}

task t_pacer(*watch<u32> dm) {
    while (dm.read() != 31u32) {
        delay(1ms);
        critical { sys.nop(); }
    }
    println("fuzz_no_lost_wakeups_ok");
    sys.exit(0);
}

fn main() {
    IRQ_INBOX.attach(on_fq(&fq, &fq_sent));
    IRQ_SIG.attach(on_fsig(&fsig, &sig_req, &sig_ack));
    IRQ_RING.attach(on_fring(&fring, &ring_sent));
    IRQ_WATCH.attach(on_fwatch(&fw));
    IRQ_POOL.attach(on_fpool(&prelease_q, &pool_rel_sig));
    sys.fuzz_irqs(0xDEAD_BEEFu32, (1u32 << 1) | (1u32 << 2) | (1u32 << 3) | (1u32 << 4) | (1u32 << 5));
    spawn t_inbox(&fq, &done_mask) @normal;
    spawn t_signal(&fsig, &sig_req, &done_mask) @normal;
    spawn t_ring(&fring, &done_mask) @normal;
    spawn t_watch(&fw, &done_mask) @normal;
    spawn t_pool(&fpool, &prelease_q, &pool_rel_sig, &done_mask) @normal;
    spawn t_pacer(&done_mask) @low;
}
"#,
        )
        .unwrap();
        let (_, mir) = compile_to_mir_with_includes(&[main_path], &[]).unwrap();
        let ll = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .expect("emit host lost-wakeup stress IR");
        let code = runner::compile_and_run_host_llvm(&ll, true, &[])
            .expect("host sanitize lost-wakeup stress should pass");
        assert_eq!(code, 0);
    }

    #[test]
    fn host_fuzz_irqs_std_sync_semaphore_flags_latch_and_barrier() {
        let temp = TempWorkDir::create("zev_std_sync_host").unwrap();
        let main_path = temp.dir.join("app.zev");
        std::fs::write(
            &main_path,
            r#"
import std.sync as sync;

sync.Semaphore sem;
watch<u32> sem_released = 0;

sync.EventFlags flags;
sync.Latch gate;
sync.Barrier<3> barrier;

watch<u32> barrier_trips = 0;
watch<u32> done_mask = 0;

irq IRQ_SEM = 1;
irq IRQ_FLAGS = 2;
irq IRQ_LATCH = 3;

isr on_sem(*sync.Semaphore s, *watch<u32> rel) {
    u32 r = rel.read();
    if (r < 25u32) {
        rel.publish(r + 1u32);
        s.release();
    }
}

isr on_flags(*sync.EventFlags f) {
    f.set(0x03u32);
}

isr on_latch(*sync.Latch g) {
    if (!g.is_ready()) {
        g.arrive();
    }
}

task t_sem_worker(*sync.Semaphore s, *watch<u32> dm) {
    for (u32 i in 0..25) {
        s.acquire();
        critical { sys.nop(); }
    }
    critical { dm.publish(dm.read() | 1u32); }
    loop { delay(5000ms); }
}

task t_flags_worker(*sync.EventFlags f, *sync.Latch g, *watch<u32> dm) {
    g.wait();
    assert(g.is_ready());
    u32 got = f.wait_all(0x03u32);
    assert((got & 0x03u32) == 0x03u32);
    f.clear(0x03u32);
    critical { dm.publish(dm.read() | 2u32); }
    loop { delay(5000ms); }
}

task t_barrier_peer(u32 id, *sync.Barrier<3> b, *sync.Latch g, *watch<u32> trips, *watch<u32> dm) {
    g.wait();
    for (u32 r in 0..5) {
        if (b.wait()) {
            critical { trips.publish(trips.read() + 1u32); }
        }
        checkpoint();
    }
    critical { dm.publish(dm.read() | (1u32 << (2u32 + id))); }
    loop { delay(5000ms); }
}

task t_pacer(*watch<u32> dm, *watch<u32> trips) {
    while (dm.read() != 31u32) {
        delay(1ms);
        critical { sys.nop(); }
    }
    assert(trips.read() == 5u32);
    println("std_sync_fuzz_ok");
    sys.exit(0);
}

fn main() {
    sem.init(0u32);
    IRQ_SEM.attach(on_sem(&sem, &sem_released));
    IRQ_FLAGS.attach(on_flags(&flags));
    IRQ_LATCH.attach(on_latch(&gate));
    sys.fuzz_irqs(0xBAD_C0FFEu32, (1u32 << 1) | (1u32 << 2) | (1u32 << 3));
    spawn t_sem_worker(&sem, &done_mask) @normal;
    spawn t_flags_worker(&flags, &gate, &done_mask) @normal;
    spawn t_barrier_peer(0u32, &barrier, &gate, &barrier_trips, &done_mask) @normal;
    spawn t_barrier_peer(1u32, &barrier, &gate, &barrier_trips, &done_mask) @normal;
    spawn t_barrier_peer(2u32, &barrier, &gate, &barrier_trips, &done_mask) @normal;
    spawn t_pacer(&done_mask, &barrier_trips) @low;
}
"#,
        )
        .unwrap();
        let (_, mir) = compile_to_mir_with_includes(&[main_path], &[]).unwrap();
        let ll = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .expect("emit host std.sync fuzz IR");
        let code = runner::compile_and_run_host_llvm(&ll, true, &[])
            .expect("host sanitize std.sync fuzz run should pass");
        assert_eq!(code, 0);
    }

    #[test]
    fn host_fuzz_irqs_wait_block_multi_arm_priority_and_deadline() {
        let temp = TempWorkDir::create("zev_wait_block_host").unwrap();
        let main_path = temp.dir.join("app.zev");
        std::fs::write(
            &main_path,
            r#"
signal hi_sig;
inbox<u32, 4> hi_q;
inbox<u32, 8> lo_q;
watch<u32> w_chg = 0;
byte_ring<16> rx_ring;
pool<u32, 1> work_pool;
dma_slot<u32> pool_sink;

watch<u32> hi_hits = 0;
watch<u32> sig_hits = 0;
watch<u32> rx_bytes = 0;
watch<u32> dl_beats = 0;
watch<u32> seq = 0;

irq IRQ_WAIT_FEED = 2;

isr on_feed(*inbox<u32, 8> q, *signal s, *watch<u32> seq_w) {
    u32 n = seq_w.read();
    if (n < 30u32 && !q.is_full()) {
        q.try_post(n + 1u32, boost: @high);
        if ((n & 3u32) == 0u32) {
            s.notify(boost: @critical);
        }
        seq_w.publish(n + 1u32);
    }
}

task multi_waiter() {
    u32 next_beat = sys.now_ms() + 10u32;
    loop {
        wait {
            hi_sig.wait() => {
                sig_hits.publish(sig_hits.read() + 1u32);
            }
            u32 h = hi_q.recv() => {
                hi_hits.publish(hi_hits.read() + h);
            }
            u32 lo = lo_q.recv() => {
                u32 _tmp = lo;
            }
            []u8 chunk = rx_ring.wait_read() => {
                rx_bytes.publish(rx_bytes.read() + (chunk.len as u32));
                rx_ring.consume(chunk.len);
            }
            u32 wv = w_chg.wait_change() => {
                u32 _wv = wv;
            }
            Lease<u32> l = work_pool.alloc() => {
                *u32 lp = l.get();
                *lp = 1u32;
                pool_sink.park(l) catch { sys.exit(1); };
            }
            deadline next_beat => {
                dl_beats.publish(dl_beats.read() + 1u32);
                next_beat += 10u32;
            }
        }
        if (dl_beats.read() >= 3u32 && sig_hits.read() > 0u32) {
            println(f"wait_fuzz_ok beats={dl_beats.read()} sigs={sig_hits.read()}");
            sys.exit(0);
        }
    }
}

task pacer() {
    loop {
        delay(1ms);
        critical { sys.nop(); }
    }
}

fn main() {
    IRQ_WAIT_FEED.attach(on_feed(&lo_q, &hi_sig, &seq));
    sys.fuzz_irqs(0x55AA_1234u32, 1u32 << 2);
    spawn multi_waiter() @normal;
    spawn pacer() @low;
    hi_q.try_post(42u32);
    [1]u8 pkt = [0x5Au8];
    rx_ring.push(pkt[0..1]);
}
"#,
        )
        .unwrap();
        let (_, mir) = compile_to_mir_with_includes(&[main_path], &[]).unwrap();
        let ll = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .expect("emit host wait_block IR");
        let code = runner::compile_and_run_host_llvm(&ll, true, &[])
            .expect("host sanitize wait_block run should pass");
        assert_eq!(code, 0);
    }

    #[test]
    fn host_runtime_panic_traps_and_int_min_div_safe() {
        let temp = TempWorkDir::create("zev_traps_host").unwrap();
        let safe_path = temp.dir.join("safe.zev");
        std::fs::write(
            &safe_path,
            r#"
watch<i32> wn = -2147483647 - 1;
watch<i32> wd = -1;
fn main() {
    i32 n = wn.read();
    i32 d = wd.read();
    assert((n / d) == n);
    assert((n % d) == 0);
    sys.exit(0);
}
"#,
        )
        .unwrap();
        let (_, mir_s) = compile_to_mir_with_includes(&[safe_path], &[]).unwrap();
        let ll_s = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir_s,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        let code_s = runner::compile_and_run_host_llvm(&ll_s, true, &[]).unwrap();
        assert_eq!(code_s, 0);

        for (name, src) in [
            (
                "idx.zev",
                "watch<usize> i = 2; fn main() { [2]u8 a = [1, 2]; u8 _v = a[i.read()]; }\n",
            ),
            (
                "slice.zev",
                "watch<usize> e = 5; fn main() { [2]u8 a = [1, 2]; []u8 _s = a[0..e.read()]; }\n",
            ),
            (
                "div.zev",
                "watch<i32> d = 0; fn main() { i32 _v = 9 / d.read(); }\n",
            ),
        ] {
            let p = temp.dir.join(name);
            std::fs::write(&p, src).unwrap();
            let (_, mir) = compile_to_mir_with_includes(&[p], &[]).unwrap();
            let ll = zev_codegen_llvm::emit_llvm_ir_with_options(
                &mir,
                &zev_codegen_llvm::CodegenOptions::host_linux(),
            )
            .unwrap();
            let code = runner::compile_and_run_host_llvm(&ll, true, &[]).unwrap();
            assert_eq!(code, 101, "{name} should exit 101");
        }
    }

    #[test]
    fn ram_and_check_json_classify_task_run_bounds() {
        let temp = TempWorkDir::create("zev_run_bounds").unwrap();
        let src_path = temp.dir.join("run_bounds.zev");
        std::fs::write(
            &src_path,
            r#"
watch<u32> w = 0;

task t_straight() {
    loop {
        delay(5ms);
    }
}

task t_for() {
    u32 acc = 0;
    loop {
        for (u32 i in 0..8) {
            acc += i;
        }
        w.publish(acc);
        delay(5ms);
    }
}

task t_spin() {
    loop {
        noyield {
            while (w.read() == 0) {}
        }
        delay(5ms);
    }
}

fn main() {
    spawn t_straight() @high;
    spawn t_for() @normal;
    spawn t_spin() @low;
}
"#,
        )
        .unwrap();
        let (bundle, mir) = compile_to_mir_with_includes(&[src_path], &[]).unwrap();
        let table = mir.ram_report.format_table();
        assert!(table.contains("max non-yield run: straight-line"), "table=\n{table}");
        assert!(table.contains("max non-yield run: bounded-for"), "table=\n{table}");
        assert!(table.contains("max non-yield run: noyield-spin"), "table=\n{table}");

        let task_runs: Vec<TaskRunSummary> = mir
            .ram_report
            .spawned_tasks
            .iter()
            .map(|t| TaskRunSummary {
                slot_id: t.slot_id,
                instance: t.instance_name.clone(),
                blueprint: t.blueprint_name.clone(),
                priority: t.priority.label().to_string(),
                max_run_kind: t.max_run_kind.as_str().to_string(),
                max_run_blocks: t.max_run_blocks,
            })
            .collect();
        let dual_vec: Vec<String> = bundle.sema.dual_specialized_fns.iter().cloned().collect();
        let json = render_check_json_ok_with_tasks(
            bundle.programs.len(),
            bundle.sema.spawned_tasks.len(),
            bundle.sema.primitives.len(),
            bundle.sema.tick_period_us,
            &dual_vec,
            &bundle.sema.monomorphized_instances,
            &task_runs,
        );
        assert!(json.contains("\"blueprint\": \"t_straight\""), "json={json}");
        assert!(json.contains("\"max_run_kind\": \"straight-line\""), "json={json}");
        assert!(json.contains("\"blueprint\": \"t_for\""), "json={json}");
        assert!(json.contains("\"max_run_kind\": \"bounded-for\""), "json={json}");
        assert!(json.contains("\"blueprint\": \"t_spin\""), "json={json}");
        assert!(json.contains("\"max_run_kind\": \"noyield-spin\""), "json={json}");
    }
}
