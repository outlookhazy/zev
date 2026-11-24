mod cli;
mod diag;
mod docs;
mod docs_errors;
mod runner;
mod script_runner;
mod serial;
mod svd;

use cli::{parse_args, Command, HELP_TEXT};
use diag::{
    format_check_human_ok, primary_doc_cmd, render_check_json_err, render_check_json_ok_with_tasks,
    JsonDiag, SourceFile, TaskRunSummary,
};
use runner::TempWorkDir;
use std::collections::{HashMap, HashSet};
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

#[cfg(test)]
fn load_and_check_detailed_with_includes_and_targets(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    hal_dir: Option<&str>,
    board_module: Option<&str>,
) -> Result<CheckedBundle, CheckFailure> {
    load_and_check_detailed_with_target(files, include_dirs, hal_dir, board_module, 64, "host")
}

fn load_and_check_detailed_with_target(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    hal_dir: Option<&str>,
    board_module: Option<&str>,
    target_ptr_bits: u32,
    target_name: &str,
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

    let mut loaded: Vec<(String, SourceFile, Program)> = Vec::with_capacity(source_files.len());
    let mut by_name: HashMap<String, usize> = HashMap::with_capacity(source_files.len());
    let mut parse_humans: Vec<String> = Vec::new();
    let mut parse_json_diags: Vec<JsonDiag> = Vec::new();

    for (display, _, sf) in source_files {
        let (prog, parse_errs) = zev_parser::parse_str_named_recovering(&sf.src, &display);
        if parse_errs.is_empty() {
            let idx = loaded.len();
            by_name.insert(display.clone(), idx);
            loaded.push((display, sf, prog));
        } else {
            for pe in parse_errs {
                let (line, col) = sf.line_col(pe.span.start as usize);
                let topic = pe.doc_topic.unwrap_or("syntax");
                let doc_cmd = pe.doc_command();
                let human = sf.format_rich_span_error(
                    pe.span,
                    pe.code,
                    &pe.message,
                    pe.hint.as_deref(),
                    Some(topic),
                );
                parse_humans.push(human.trim_end().to_string());
                parse_json_diags.push(JsonDiag {
                    code: pe.code.map(str::to_string),
                    file: display.clone(),
                    line,
                    col,
                    message: pe.message,
                    help: pe.hint.map(String::from),
                    doc_cmd,
                });
            }
        }
    }
    if !parse_json_diags.is_empty() {
        return Err(CheckFailure {
            human: parse_humans.join("\n\n"),
            json_diags: parse_json_diags,
        });
    }

    let named_refs: Vec<(&Program, Option<&str>)> = loaded
        .iter()
        .map(|(name, _, prog)| (prog, Some(name.as_str())))
        .collect();

    let opts = AnalyzeOptions {
        include_dirs: include_dirs.to_vec(),
        hal_dir: hal_dir.map(str::to_string),
        board_module: board_module.map(str::to_string),
        target_ptr_bits,
        target_name: target_name.to_string(),
        ..Default::default()
    };
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

#[cfg(test)]
fn compile_to_mir_with_includes(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
) -> Result<(CheckedBundle, MirModule), String> {
    compile_to_mir_with_includes_and_targets(files, include_dirs, None, None)
}

#[cfg(test)]
fn compile_to_mir_with_includes_and_targets(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    hal_dir: Option<&str>,
    board_module: Option<&str>,
) -> Result<(CheckedBundle, MirModule), String> {
    compile_to_mir_with_target(files, include_dirs, hal_dir, board_module, 64, "host")
}

fn compile_to_mir_with_target(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    hal_dir: Option<&str>,
    board_module: Option<&str>,
    target_ptr_bits: u32,
    target_name: &str,
) -> Result<(CheckedBundle, MirModule), String> {
    let bundle = load_and_check_detailed_with_target(
        files,
        include_dirs,
        hal_dir,
        board_module,
        target_ptr_bits,
        target_name,
    )
    .map_err(|f| f.human)?;
    let mir = zev_mir::lower_programs(&bundle.programs)
        .map_err(|e| format!("error: {e}\n  = docs: run 'zevc doc tasks'"))?;
    Ok((bundle, mir))
}

fn target_info_from_llvm_opts(opts: Option<&zev_codegen_llvm::CodegenOptions>) -> (u32, String) {
    let Some(o) = opts else {
        return (64, "host".to_string());
    };
    if let Some(ref spec) = o.spec {
        let name = spec
            .board_name
            .clone()
            .unwrap_or_else(|| spec.chip_name.clone());
        return (spec.ptr_bits, name);
    }
    match o.target {
        zev_codegen_llvm::TargetArch::HostX86_64Linux => (64, "host".to_string()),
        zev_codegen_llvm::TargetArch::Thumbv7mNoneEabi => (32, "cortex-m3".to_string()),
        zev_codegen_llvm::TargetArch::Thumbv6mRp2040 => (32, "rp2040".to_string()),
        zev_codegen_llvm::TargetArch::Riscv32Esp32C6 => (32, "esp32c6".to_string()),
        zev_codegen_llvm::TargetArch::AvrAtmega328p => (16, "atmega328p".to_string()),
        zev_codegen_llvm::TargetArch::AvrAtmega4809 => (16, "atmega4809".to_string()),
    }
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

fn resolve_llvm_options(target: Option<&str>) -> Result<zev_codegen_llvm::CodegenOptions, String> {
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
            Ok(zev_codegen_llvm::CodegenOptions {
                target: zev_codegen_llvm::TargetArch::HostX86_64Linux,
                ..Default::default()
            })
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

struct CheckExecOutcome {
    exit_code: ExitCode,
    stdout: Option<String>,
    stderr: Option<String>,
}

fn execute_check_command(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    board: Option<&str>,
    chip: Option<&str>,
    json: bool,
) -> CheckExecOutcome {
    let mut eff_includes = include_dirs.to_vec();
    let (hal_dir, board_module, target_ptr_bits, target_name): (
        Option<String>,
        Option<String>,
        u32,
        String,
    ) = if board.is_some() || chip.is_some() {
        let manifest = match discover_manifest_for_inputs(files) {
            Ok(m) => m,
            Err(err) => {
                return CheckExecOutcome {
                    exit_code: ExitCode::from(2),
                    stdout: None,
                    stderr: Some(format!("error: {err}")),
                };
            }
        };
        let mdir = manifest.as_ref().map(|(d, _)| d.as_path());
        let spec = match resolve_target_spec_for_inputs(files, mdir, chip, board) {
            Ok(s) => s,
            Err(err) => {
                return CheckExecOutcome {
                    exit_code: ExitCode::from(2),
                    stdout: None,
                    stderr: Some(format!("error: {err}")),
                };
            }
        };
        if let Some(ref root) = spec.root_dir {
            if !eff_includes.contains(root) {
                eff_includes.push(root.clone());
            }
        }
        let target_name = spec.board_name.unwrap_or(spec.chip_name);
        (spec.hal_dir, spec.board_module, spec.ptr_bits, target_name)
    } else {
        (None, None, 64, "host".to_string())
    };

    let bundle = match load_and_check_detailed_with_target(
        files,
        &eff_includes,
        hal_dir.as_deref(),
        board_module.as_deref(),
        target_ptr_bits,
        &target_name,
    ) {
        Ok(b) => b,
        Err(fail) => {
            return if json {
                CheckExecOutcome {
                    exit_code: ExitCode::FAILURE,
                    stdout: Some(render_check_json_err(&fail.json_diags)),
                    stderr: None,
                }
            } else {
                CheckExecOutcome {
                    exit_code: ExitCode::FAILURE,
                    stdout: None,
                    stderr: Some(fail.human),
                }
            };
        }
    };

    let mir = match zev_mir::lower_programs(&bundle.programs) {
        Ok(m) => m,
        Err(err) => {
            return if json {
                let diag = JsonDiag {
                    code: None,
                    file: files
                        .first()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    line: err.span.line as usize,
                    col: err.span.col as usize,
                    message: err.message.clone(),
                    help: Some("inspect lowering rules in `zevc doc tasks`".to_string()),
                    doc_cmd: "zevc doc tasks".to_string(),
                };
                CheckExecOutcome {
                    exit_code: ExitCode::FAILURE,
                    stdout: Some(render_check_json_err(&[diag])),
                    stderr: None,
                }
            } else {
                CheckExecOutcome {
                    exit_code: ExitCode::FAILURE,
                    stdout: None,
                    stderr: Some(format!("error: {err}\n  = docs: run 'zevc doc tasks'")),
                }
            };
        }
    };

    let verified_files = bundle.programs.len();
    let dual_vec: Vec<String> = bundle.sema.dual_specialized_fns.iter().cloned().collect();
    if json {
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
        CheckExecOutcome {
            exit_code: ExitCode::SUCCESS,
            stdout: Some(render_check_json_ok_with_tasks(
                verified_files,
                bundle.sema.spawned_tasks.len(),
                bundle.sema.primitives.len(),
                bundle.sema.tick_period_us,
                &dual_vec,
                &bundle.sema.monomorphized_instances,
                &task_runs,
            )),
            stderr: None,
        }
    } else {
        CheckExecOutcome {
            exit_code: ExitCode::SUCCESS,
            stdout: Some(format_check_human_ok(
                verified_files,
                bundle.sema.spawned_tasks.len(),
                bundle.sema.primitives.len(),
                bundle.sema.tick_period_us,
                &dual_vec,
                &bundle.sema.monomorphized_instances,
            )),
            stderr: None,
        }
    }
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
            if let Some(found) =
                zev_codegen_llvm::ProjectManifest::discover(parent).map_err(|e| e.to_string())?
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
    let (manifest_dir, resolved_files, eff_board, eff_chip, eff_port, eff_opt) = match manifest {
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

fn resolve_monitor_port(board: Option<&str>, port: Option<&str>) -> Result<String, String> {
    if let Some(p) = port {
        return Ok(p.to_string());
    }
    let manifest = discover_manifest_for_inputs(&[])?;
    let (mdir, mboard, mport) = match manifest {
        Some((dir, m)) => (Some(dir), m.board, m.port),
        None => (None, None, None),
    };
    if board.is_none() {
        if let Some(p) = mport {
            return Ok(p);
        }
    }
    let eff_board = board.map(str::to_string).or(mboard);
    let Some(bname) = eff_board else {
        return Err(
            "no serial port specified: pass `--port <dev>`, `--board <name>`, or set `port`/`board` in `zev.toml`"
                .to_string(),
        );
    };
    let spec = resolve_target_spec_for_inputs(&[], mdir.as_deref(), None, Some(&bname))?;
    spec.default_port.ok_or_else(|| {
        format!("board `{bname}` does not declare `[run].default_port`; pass `--port <dev>`")
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
    let (target_ptr_bits, target_name) = target_info_from_llvm_opts(llvm_opts.as_ref());
    let (_, mir) = match compile_to_mir_with_target(
        files,
        &eff_includes,
        hal_dir,
        board_module,
        target_ptr_bits,
        &target_name,
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
        Command::ImportSvd { svd_path, output } => {
            svd::run_import_svd(&svd_path, output.as_deref())
        }
        Command::Check {
            files,
            include_dirs,
            board,
            chip,
            json,
        } => {
            let outcome = execute_check_command(
                &files,
                &include_dirs,
                board.as_deref(),
                chip.as_deref(),
                json,
            );
            if let Some(out) = outcome.stdout {
                println!("{out}");
            }
            if let Some(err) = outcome.stderr {
                eprintln!("{err}");
            }
            outcome.exit_code
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
            script,
            cmds,
            script_timeout_ms,
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
            let scripted = script.is_some() || !cmds.is_empty();
            let script_cmds = if scripted {
                match script_runner::collect_script_commands(script.as_deref(), &cmds) {
                    Ok(c) => c,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                Vec::new()
            };
            let (target_ptr_bits, target_name) = target_info_from_llvm_opts(Some(&opts));
            match compile_to_mir_with_target(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
                target_ptr_bits,
                &target_name,
            ) {
                Ok((_, mir)) => match zev_codegen_llvm::emit_llvm_ir_with_options(&mir, &opts) {
                    Ok(ll_src) => {
                        if scripted {
                            let temp = match TempWorkDir::create("zevc-run-script") {
                                Ok(t) => t,
                                Err(err) => {
                                    eprintln!("error: {err}");
                                    return ExitCode::FAILURE;
                                }
                            };
                            let bin_path =
                                match runner::compile_host_llvm_binary(&temp, &ll_src, sanitize) {
                                    Ok(p) => p,
                                    Err(err) => {
                                        eprintln!("error: {err}");
                                        return ExitCode::FAILURE;
                                    }
                                };
                            let mut child =
                                match runner::spawn_host_llvm_binary(&bin_path, &run_args) {
                                    Ok(c) => c,
                                    Err(err) => {
                                        eprintln!("error: {err}");
                                        return ExitCode::FAILURE;
                                    }
                                };
                            let timeout = std::time::Duration::from_millis(
                                script_timeout_ms
                                    .unwrap_or(script_runner::DEFAULT_SCRIPT_TIMEOUT_MS),
                            );
                            match script_runner::run_script_on_child(
                                &mut child,
                                &script_cmds,
                                timeout,
                            ) {
                                Ok(()) => ExitCode::SUCCESS,
                                Err(err) => {
                                    eprintln!("error: {err}");
                                    ExitCode::FAILURE
                                }
                            }
                        } else {
                            match runner::compile_and_run_host_llvm(&ll_src, sanitize, &run_args) {
                                Ok(0) => ExitCode::SUCCESS,
                                Ok(code) => ExitCode::from((code.clamp(1, 255)) as u8),
                                Err(err) => {
                                    eprintln!("error: {err}");
                                    ExitCode::FAILURE
                                }
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
            if let Some(root) = opts.spec.as_ref().and_then(|s| s.root_dir.as_ref()) {
                if !eff_includes.contains(root) {
                    eff_includes.push(root.clone());
                }
            }
            let (target_ptr_bits, target_name) = target_info_from_llvm_opts(Some(&opts));
            match compile_to_mir_with_target(
                &files,
                &eff_includes,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
                target_ptr_bits,
                &target_name,
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
            let (target_ptr_bits, target_name) = target_info_from_llvm_opts(Some(&built.opts));
            match compile_to_mir_with_target(
                &built.files,
                &built.include_dirs,
                built.spec.hal_dir.as_deref(),
                built.spec.board_module.as_deref(),
                target_ptr_bits,
                &target_name,
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
            monitor,
            script,
            cmds,
            script_timeout_ms,
        } => {
            let scripted = script.is_some() || !cmds.is_empty();
            let script_cmds = if scripted {
                match script_runner::collect_script_commands(script.as_deref(), &cmds) {
                    Ok(c) => c,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                Vec::new()
            };
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
            let (target_ptr_bits, target_name) = target_info_from_llvm_opts(Some(&built.opts));
            match compile_to_mir_with_target(
                &built.files,
                &built.include_dirs,
                built.spec.hal_dir.as_deref(),
                built.spec.board_module.as_deref(),
                target_ptr_bits,
                &target_name,
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
                    let res =
                        match zev_codegen_llvm::build_target_image(&mir, &built.opts, &out_path) {
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
                            if !monitor && !scripted {
                                return ExitCode::SUCCESS;
                            }
                            let serial_dev = built
                                .port
                                .as_deref()
                                .filter(|p| !Path::new(p).is_dir())
                                .or(built.spec.default_port.as_deref());
                            let Some(dev_path) = serial_dev else {
                                eprintln!(
                                    "error: `{target_label}` does not declare a serial port for `--monitor`/`--script`; pass `--port <dev>`"
                                );
                                return ExitCode::from(2);
                            };
                            let sp = match serial::wait_for_serial_port(
                                dev_path,
                                std::time::Duration::from_secs(5),
                            ) {
                                Ok(s) => s,
                                Err(err) => {
                                    eprintln!("error: {err}");
                                    return ExitCode::FAILURE;
                                }
                            };
                            let post_res = if scripted {
                                let timeout = std::time::Duration::from_millis(
                                    script_timeout_ms
                                        .unwrap_or(script_runner::DEFAULT_SCRIPT_TIMEOUT_MS),
                                );
                                script_runner::run_script_on_serial(&sp, &script_cmds, timeout)
                            } else {
                                serial::run_interactive_monitor_on(dev_path, sp)
                            };
                            match post_res {
                                Ok(()) => ExitCode::SUCCESS,
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
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::Monitor {
            board,
            port,
            script,
            cmds,
            script_timeout_ms,
        } => {
            let scripted = script.is_some() || !cmds.is_empty();
            let script_cmds = if scripted {
                match script_runner::collect_script_commands(script.as_deref(), &cmds) {
                    Ok(c) => c,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                Vec::new()
            };
            let eff_port = match resolve_monitor_port(board.as_deref(), port.as_deref()) {
                Ok(p) => p,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::from(2);
                }
            };
            let res = if scripted {
                let timeout = std::time::Duration::from_millis(
                    script_timeout_ms.unwrap_or(script_runner::DEFAULT_SCRIPT_TIMEOUT_MS),
                );
                serial::open_serial_115200(&eff_port)
                    .and_then(|sp| script_runner::run_script_on_serial(&sp, &script_cmds, timeout))
            } else {
                serial::run_interactive_monitor(&eff_port)
            };
            match res {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::BuildElf {
            files,
            include_dirs,
            opt_level,
            output,
        } => match compile_to_mir_with_target(&files, &include_dirs, None, None, 32, "cortex-m3") {
            Ok((_, mir)) => {
                let out_elf = output.unwrap_or_else(|| default_output_with_ext(&files, "elf"));
                let opts = zev_codegen_llvm::CodegenOptions {
                    opt_level,
                    ..Default::default()
                };
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
        },
        Command::RunQemu {
            files,
            include_dirs,
            max_ticks,
            opt_level,
            script,
            cmds,
            script_timeout_ms,
        } => {
            let mut opts = zev_codegen_llvm::CodegenOptions::lm3s6965();
            opts.opt_level = opt_level;
            if let Some(ticks) = max_ticks {
                opts.max_ticks = ticks.min(u32::MAX as u64) as u32;
            }
            let scripted = script.is_some() || !cmds.is_empty();
            let script_cmds = if scripted {
                match script_runner::collect_script_commands(script.as_deref(), &cmds) {
                    Ok(c) => c,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                Vec::new()
            };
            let (target_ptr_bits, target_name) = target_info_from_llvm_opts(Some(&opts));
            match compile_to_mir_with_target(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
                target_ptr_bits,
                &target_name,
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
                    if scripted {
                        let build = match zev_codegen_llvm::build_elf(&mir, &elf_path, &opts) {
                            Ok(b) => b,
                            Err(err) => {
                                eprintln!("error: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let mut child = match zev_codegen_llvm::spawn_elf_in_qemu(
                            &build.elf_path,
                            &qemu_opts,
                            true,
                        ) {
                            Ok(c) => c,
                            Err(err) => {
                                eprintln!("error: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let timeout = std::time::Duration::from_millis(
                            script_timeout_ms.unwrap_or(script_runner::DEFAULT_SCRIPT_TIMEOUT_MS),
                        );
                        match script_runner::run_script_on_child(&mut child, &script_cmds, timeout)
                        {
                            Ok(()) => ExitCode::SUCCESS,
                            Err(err) => {
                                eprintln!("error: {err}");
                                ExitCode::FAILURE
                            }
                        }
                    } else if max_ticks.is_none() && serial::stdin_is_tty() {
                        let build = match zev_codegen_llvm::build_elf(&mir, &elf_path, &opts) {
                            Ok(b) => b,
                            Err(err) => {
                                eprintln!("error: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let _tty_guard = serial::InteractiveTtyGuard::enter(true);
                        let mut child = match zev_codegen_llvm::spawn_elf_in_qemu(
                            &build.elf_path,
                            &qemu_opts,
                            false,
                        ) {
                            Ok(c) => c,
                            Err(err) => {
                                eprintln!("error: {err}");
                                return ExitCode::FAILURE;
                            }
                        };
                        match child.wait() {
                            Ok(st) if st.success() => ExitCode::SUCCESS,
                            Ok(st) => ExitCode::from((st.code().unwrap_or(1).clamp(1, 255)) as u8),
                            Err(err) => {
                                eprintln!("error: {err}");
                                ExitCode::FAILURE
                            }
                        }
                    } else {
                        match zev_codegen_llvm::run_qemu_cortex_m(
                            &mir, &elf_path, &opts, &qemu_opts,
                        ) {
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
            let (target_ptr_bits, target_name) = target_info_from_llvm_opts(Some(&opts));
            match compile_to_mir_with_target(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
                target_ptr_bits,
                &target_name,
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
            let (target_ptr_bits, target_name) = target_info_from_llvm_opts(Some(&opts));
            match compile_to_mir_with_target(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
                target_ptr_bits,
                &target_name,
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
        } => {
            match compile_to_mir_with_target(&files, &include_dirs, None, None, 16, "atmega328p") {
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
            }
        }
        Command::FlashAvr {
            files,
            include_dirs,
            port,
            opt_level,
            output,
        } => {
            match compile_to_mir_with_target(&files, &include_dirs, None, None, 16, "atmega328p") {
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
            }
        }
        Command::BuildAvrNanoEvery {
            files,
            include_dirs,
            opt_level,
            output,
        } => {
            match compile_to_mir_with_target(&files, &include_dirs, None, None, 16, "atmega4809") {
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
            }
        }
        Command::FlashAvrNanoEvery {
            files,
            include_dirs,
            port,
            opt_level,
            output,
        } => {
            match compile_to_mir_with_target(&files, &include_dirs, None, None, 16, "atmega4809") {
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
            }
        }
        Command::BuildRp2040 {
            files,
            include_dirs,
            opt_level,
            output,
        } => {
            let mut opts = zev_codegen_llvm::CodegenOptions::rp2040();
            opts.opt_level = opt_level;
            let (target_ptr_bits, target_name) = target_info_from_llvm_opts(Some(&opts));
            match compile_to_mir_with_target(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
                target_ptr_bits,
                &target_name,
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
        }
        Command::FlashRp2040 {
            files,
            include_dirs,
            port,
            opt_level,
            output,
        } => {
            let mut opts = zev_codegen_llvm::CodegenOptions::rp2040();
            opts.opt_level = opt_level;
            let (target_ptr_bits, target_name) = target_info_from_llvm_opts(Some(&opts));
            match compile_to_mir_with_target(
                &files,
                &include_dirs,
                opts.hal_dir.as_deref(),
                opts.board_module.as_deref(),
                target_ptr_bits,
                &target_name,
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
                "enum E { A }\nfn f() -> u16 error E { error A; }\nfn main() { u16 v = f() catch |e| { return; }; }\n",
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
            assert!(
                json.contains(&format!("\"code\": \"{expected_code}\"")),
                "json={json}"
            );
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
        assert!(
            f_chain.human.contains("error[E0005]"),
            "human=\n{}",
            f_chain.human
        );
        assert!(f_chain.human.contains("f1"), "human=\n{}", f_chain.human);
        assert!(f_chain.human.contains("f2"), "human=\n{}", f_chain.human);
        assert!(
            f_chain.human.contains("zevc doc E0005"),
            "human=\n{}",
            f_chain.human
        );

        let Err(f_meth) = check_temp_file(
            "bad_meth.zev",
            "enum Ev { Ping }\ninbox<Ev, 4> q;\nfn main() { q.push(Ping); }\n",
        ) else {
            panic!("expected invalid inbox method to fail");
        };
        assert!(
            f_meth.human.contains("error[E0020]"),
            "human=\n{}",
            f_meth.human
        );
        assert!(f_meth.human.contains("post"), "human=\n{}", f_meth.human);
        let json = render_check_json_err(&f_meth.json_diags);
        assert!(json.contains("\"code\": \"E0020\""), "json={json}");
        assert!(
            json.contains("\"doc_cmd\": \"zevc doc E0020\""),
            "json={json}"
        );

        let ok = check_temp_file(
            "partial_ok.zev",
            "enum IoErr { Timeout }\nfn read_chunk() -> u32 error IoErr {\n    error Timeout, 12;\n}\nfn main() {\n    u32 n = read_chunk() catch (IoErr e, u32 partial) {\n        return;\n    };\n}\n",
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
        assert!(
            ok_json.contains("\"dual_specialized_functions\": []"),
            "json={ok_json}"
        );
        assert!(
            ok_json.contains("\"monomorphized_instances\": []"),
            "json={ok_json}"
        );
    }

    #[test]
    fn check_json_and_human_report_dual_specialized_and_wave1_sema_codes() {
        let dual_src = "\
watch<u32> latest = 0;

fn hash_chunk(u32 seed) -> u32 {
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
            assert!(
                fail.human.contains(&format!("error[{code}]")),
                "human=\n{}",
                fail.human
            );
            assert!(
                fail.human.contains(&format!("zevc doc {code}")),
                "human=\n{}",
                fail.human
            );
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

fn clamp<T>(T val, T lo, T hi) -> T {
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
        assert!(
            ram_table.contains("FixedVec<u32, 8>"),
            "ram_table=\n{ram_table}"
        );
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
            assert!(
                fail.human.contains(&format!("error[{code}]")),
                "human=\n{}",
                fail.human
            );
            assert!(
                fail.human.contains(&format!("zevc doc {code}")),
                "human=\n{}",
                fail.human
            );
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

fn scale_add<T, usize N>(T v) -> T {
    return v;
}

fn check_window(bool lo_ok, bool hi_ok) -> bool {
    return lo_ok && hi_ok;
}
";
        let app_src = "\
fn main() {
    FixedVec<u32, 4> buf = FixedVec<u32, 4> { data: [1, 2, 3, 4], len: 4 };
    u32 x = scale_add<u32, 4>(buf.data[0]);
    bool ok = check_window((A < B), (C > (D + 1)));
    if (ok && x == 1) {
        return;
    }
}
";
        let temp = TempWorkDir::create("zevc-multifile-gen").unwrap();
        let lib_path = temp.dir.join("lib.zev");
        let app_path = temp.dir.join("app.zev");
        let unparen_path = temp.dir.join("unparen.zev");
        std::fs::write(&lib_path, lib_src).unwrap();
        std::fs::write(&app_path, app_src).unwrap();
        std::fs::write(
            &unparen_path,
            "fn main() { bool ok = check_window(A < B, C > (D + 1)); }\n",
        )
        .unwrap();

        let Err(unparen_err) = load_and_check_detailed(&[lib_path.clone(), unparen_path]) else {
            panic!("expected unparenthesized check_window(A < B, C > (D + 1)) to fail sema");
        };
        assert!(
            unparen_err.human.contains("error[E0029]"),
            "human=\n{}",
            unparen_err.human
        );
        assert!(
            unparen_err.human.contains("parenthesize"),
            "expected parenthesization note, got:\n{}",
            unparen_err.human
        );

        // Pass `app_path` twice to also verify canonical input path deduplication (no E0021!)
        let bundle = load_and_check_detailed(&[lib_path, app_path.clone(), app_path])
            .unwrap_or_else(|e| {
                panic!("multi-file generic + comparison check failed:\n{}", e.human)
            });
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
        assert!(
            f_print.human.contains("error[E0019]"),
            "human=\n{}",
            f_print.human
        );

        let Err(f_sim) = check_temp_file("bad_sim.zev", "fn main() { sim.log(1, 2, 3); }\n") else {
            panic!("sim.log must be rejected by zev-sema");
        };
        assert!(
            f_sim.human.contains("error[E0020]"),
            "human=\n{}",
            f_sim.human
        );

        let parity_src = "\
const u8 CMD_HEAT = 7;
const u8 CMD_COOL = 9;

struct DecodedPacket {
    u8  cmd;
    u16 temp;
}

enum PacketErr { Corrupt }

fn decode_bad() -> DecodedPacket error PacketErr {
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
        let q_opts = zev_codegen_llvm::CodegenOptions {
            max_ticks: 15,
            ..Default::default()
        };
        let elf_path = temp.dir.join("parity.elf");
        let qres = zev_codegen_llvm::run_qemu_cortex_m(
            &mir,
            &elf_path,
            &q_opts,
            &zev_codegen_llvm::QemuRunOptions::default(),
        )
        .expect("QEMU parity run must succeed");
        assert_eq!(qres.exec.exit_code, 0);
        assert!(
            qres.exec.stdout.contains("arm=2"),
            "out={}",
            qres.exec.stdout
        );
        assert!(
            qres.exec.stdout.contains("diff=-200"),
            "out={}",
            qres.exec.stdout
        );
        assert!(
            qres.exec.stdout.contains("temp=42"),
            "out={}",
            qres.exec.stdout
        );
        assert!(
            qres.exec.stdout.contains("tick"),
            "out={}",
            qres.exec.stdout
        );
    }

    #[test]
    fn host_sanitize_verifies_destination_typed_integer_semantics_table() {
        let src = "\
fn main() {
    u8 a = 200;
    u8 b = 100;
    u16 sec = 100;
    u32 reg32 = 0x1234_56FFu32;
    u8 mask8 = 0x0Fu8;

    u16 w = a + b;
    u32 ms = sec * 1000;
    u32 bit = a << 8;
    u32 masked = reg32 & ~mask8;
    u8 s = a + b;
    u16 w_cast = (a + b) as u16;
    i16 neg_u8 = -a;

    assert(w == 300u16);
    assert(ms == 100_000u32);
    assert(bit == 51_200u32);
    assert(masked == 0x1234_56F0u32);
    assert(s == 44u8);
    assert(w_cast == 44u16);
    assert(neg_u8 == -200i16);
    assert((a + b) == 300u16);
    sys.exit(0);
}
";
        let temp = TempWorkDir::create("zevc-int-sem").unwrap();
        let src_path = temp.dir.join("int_sem.zev");
        std::fs::write(&src_path, src).unwrap();

        let (_, mir) = compile_to_mir(&[src_path]).expect("int_sem.zev must compile");
        let host_opts = zev_codegen_llvm::CodegenOptions::host_linux();
        let host_ll = zev_codegen_llvm::emit_llvm_ir_with_options(&mir, &host_opts).unwrap();
        let host_exit = runner::compile_and_run_host_llvm(&host_ll, true, &[]).unwrap();
        assert_eq!(host_exit, 0);
    }

    #[test]
    fn arduino_uno_example_checks_and_avr_target_resolves() {
        for alias in ["avr", "atmega328p", "uno", "arduino-uno"] {
            let opts = resolve_llvm_options(Some(alias))
                .unwrap_or_else(|e| panic!("target {alias} failed: {e}"));
            assert_eq!(opts.target, zev_codegen_llvm::TargetArch::AvrAtmega328p);
        }

        let demo_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/arduino_uno_demo.zev");
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
                assert!(
                    within_budget,
                    "expected within_budget for nano-every:\n{text}"
                );
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
                    assert!(
                        text.contains(budget_str),
                        "missing {budget_str} in:\n{text}"
                    );
                    assert!(text.contains("PASS"));
                }
                _ => panic!("expected targeted ram output for {tgt}"),
            }
        }

        // 3. `zevc ram --target host` is rejected as usage error
        assert!(matches!(
            execute_ram_command(
                &[root.join("examples/sim_demo.zev")],
                &[],
                Some("host"),
                None
            ),
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

        let (mdir, manifest) = zev_codegen_llvm::ProjectManifest::discover(&temp.dir.join("src"))
            .expect("manifest parse")
            .expect("manifest found");
        assert_eq!(mdir, temp.dir);
        assert_eq!(manifest.entry.as_deref(), Some(Path::new("src/app.zev")));

        let resolved = resolve_project_build_inputs(&[entry_path], &[], None, None, None, None)
            .expect("resolve custom project");
        assert_eq!(resolved.spec.chip_name, "my_cm3");
        assert_eq!(resolved.spec.board_name.as_deref(), Some("my_board"));
        assert_eq!(resolved.spec.ram_size, 32768);
        assert_eq!(resolved.opts.opt_level, "-Oz");
        assert_eq!(resolved.port.as_deref(), Some("/dev/ttyACM7"));

        let (_, mir) =
            compile_to_mir_with_includes(&resolved.files, &resolved.include_dirs).unwrap();
        let out_elf = temp.dir.join("custom.elf");
        let built = zev_codegen_llvm::build_target_image(&mir, &resolved.opts, &out_elf).unwrap();
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

guarded<u32> done_mask = 0;

irq IRQ_INBOX = 1;
irq IRQ_SIG   = 2;
irq IRQ_RING  = 3;
irq IRQ_WATCH = 4;
irq IRQ_POOL  = 5;

isr on_fq(*inbox<u32, 2> q, *watch<u32> sent) {
    u32 s = sent.read();
    if (s < 35u32 && !q.is_full()) {
        q.post(s + 1u32, boost: @critical);
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

fn wait_watch_helper(*watch<u32> w) -> u32 {
    return w.wait_change(timeout: 500ms) catch {
        return 999999u32;
    };
}

task t_inbox(*inbox<u32, 2> q, *guarded<u32> dm) {
    for (u32 i in 0..35) {
        u32 v = q.recv();
        if (v != i + 1u32) {
            println(f"INBOX_FAIL i={i} v={v}");
            sys.exit(2);
        }
        critical { sys.nop(); }
    }
    critical { dm.set(dm.get() | 1u32); }
    loop { delay(5000ms); }
}

task t_signal(*signal s, *watch<u32> req, *guarded<u32> dm) {
    for (u32 i in 0..35) {
        req.publish(i + 1u32);
        critical { sys.nop(); }
        s.wait();
    }
    critical { dm.set(dm.get() | 2u32); }
    loop { delay(5000ms); }
}

task t_ring(*byte_ring<8> r, *guarded<u32> dm) {
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
    critical { dm.set(dm.get() | 4u32); }
    loop { delay(5000ms); }
}

task t_watch(*watch<u32> w, *guarded<u32> dm) {
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
    critical { dm.set(dm.get() | 8u32); }
    loop { delay(5000ms); }
}

task t_pool(*pool<u32, 1> p, *inbox<Lease<u32>, 1> rel_q, *signal rel_s, *guarded<u32> dm) {
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
    critical { dm.set(dm.get() | 16u32); }
    loop { delay(5000ms); }
}

task t_pacer(*guarded<u32> dm) {
    loop {
        u32 cur = 0u32;
        critical { cur = dm.get(); }
        if (cur == 31u32) { break; }
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

guarded<u32> barrier_trips = 0;
guarded<u32> done_mask = 0;

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

task t_sem_worker(*sync.Semaphore s, *guarded<u32> dm) {
    for (u32 i in 0..25) {
        s.acquire();
        critical { sys.nop(); }
    }
    critical { dm.set(dm.get() | 1u32); }
    loop { delay(5000ms); }
}

task t_flags_worker(*sync.EventFlags f, *sync.Latch g, *guarded<u32> dm) {
    g.wait();
    assert(g.is_ready());
    u32 got = f.wait_all(0x03u32);
    assert((got & 0x03u32) == 0x03u32);
    f.clear(0x03u32);
    critical { dm.set(dm.get() | 2u32); }
    loop { delay(5000ms); }
}

task t_barrier_peer(u32 id, *sync.Barrier<3> b, *sync.Latch g, *guarded<u32> trips, *guarded<u32> dm) {
    g.wait();
    for (u32 r in 0..5) {
        if (b.wait()) {
            critical { trips.set(trips.get() + 1u32); }
        }
        checkpoint();
    }
    critical { dm.set(dm.get() | (1u32 << (2u32 + id))); }
    loop { delay(5000ms); }
}

task t_pacer(*guarded<u32> dm, *guarded<u32> trips) {
    loop {
        u32 cur = 0u32;
        critical { cur = dm.get(); }
        if (cur == 31u32) { break; }
        delay(1ms);
        critical { sys.nop(); }
    }
    u32 tr = 0u32;
    critical { tr = trips.get(); }
    assert(tr == 5u32);
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
        q.post(n + 1u32, boost: @high);
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
    hi_q.post(42u32);
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
        assert!(
            table.contains("max non-yield run: straight-line"),
            "table=\n{table}"
        );
        assert!(
            table.contains("max non-yield run: bounded-for"),
            "table=\n{table}"
        );
        assert!(
            table.contains("max non-yield run: noyield-spin"),
            "table=\n{table}"
        );

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
        assert!(
            json.contains("\"blueprint\": \"t_straight\""),
            "json={json}"
        );
        assert!(
            json.contains("\"max_run_kind\": \"straight-line\""),
            "json={json}"
        );
        assert!(json.contains("\"blueprint\": \"t_for\""), "json={json}");
        assert!(
            json.contains("\"max_run_kind\": \"bounded-for\""),
            "json={json}"
        );
        assert!(json.contains("\"blueprint\": \"t_spin\""), "json={json}");
        assert!(
            json.contains("\"max_run_kind\": \"noyield-spin\""),
            "json={json}"
        );
    }

    #[test]
    fn fn_pick_returns_fn_pointer_and_runs() {
        let src = "\
fn inc_one(u32 x) -> u32 {
    return x + 1u32;
}

fn pick() -> fn(u32) -> u32 {
    return inc_one;
}

fn main() {
    fn(u32) -> u32 cb = pick();
    assert(cb(41u32) == 42u32);
    assert(pick()(99u32) == 100u32);
    sys.exit(0);
}
";
        let temp = TempWorkDir::create("zevc-fn-pick-ptr").unwrap();
        let src_path = temp.dir.join("pick.zev");
        std::fs::write(&src_path, src).unwrap();

        let (_, mir) =
            compile_to_mir(&[src_path]).expect("fn pick() -> fn(u32) -> u32 must check & lower");
        let host_ll = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        let exit = runner::compile_and_run_host_llvm(&host_ll, true, &[]).unwrap();
        assert_eq!(exit, 0);
    }

    #[test]
    fn host_sanitize_verifies_runtime_loop_every_period_clamping() {
        let src = "\
watch<u32> cnt_zero = 0;
watch<u32> cnt_neg = 0;
watch<u32> cnt_u32_huge = 0;
watch<u32> cnt_u64_huge = 0;

task worker_zero(u8 p) {
    u32 n = 0;
    loop every p {
        n += 1;
        cnt_zero.publish(n);
    }
}

task worker_neg(i32 p) {
    u32 n = 0;
    loop every p {
        n += 1;
        cnt_neg.publish(n);
    }
}

task worker_u32_huge(u32 p) {
    u32 n = 0;
    loop every p {
        n += 1;
        cnt_u32_huge.publish(n);
    }
}

task worker_u64_huge(u64 p) {
    u32 n = 0;
    loop every p {
        n += 1;
        cnt_u64_huge.publish(n);
    }
}

task driver() {
    // At t=0 all @high workers executed once and armed their next deadline.
    assert(cnt_zero.read() == 1);
    assert(cnt_neg.read() == 1);
    assert(cnt_u32_huge.read() == 1);
    assert(cnt_u64_huge.read() == 1);

    // 0 and -5i32 clamp to 1 tick each; 3_000_000_000u32 and (1u64 << 40) clamp to 2_147_483_647 ticks.
    delay(3ms);
    assert(cnt_zero.read() == 4);
    assert(cnt_neg.read() == 4);
    assert(cnt_u32_huge.read() == 1);
    assert(cnt_u64_huge.read() == 1);

    // Advance just below i32::MAX (2_147_483_647): huge-period workers must still be parked.
    sys.tick_by(2147483640);
    delay(1ms); // now = 2_147_483_644
    assert(cnt_u32_huge.read() == 1);
    assert(cnt_u64_huge.read() == 1);

    // Step across 2_147_483_647 (clamped i32::MAX horizon): both huge-period workers wake exactly once.
    sys.tick_by(2);
    delay(1ms); // now = 2_147_483_647
    assert(cnt_u32_huge.read() == 2);
    assert(cnt_u64_huge.read() == 2);

    println(\"loop_every_clamp_ok\");
    sys.exit(0);
}

fn main() {
    u8 p0 = 0;
    i32 p_neg = -5;
    u32 p_u32 = 3000000000;
    u64 p_u64 = 1u64 << 40;
    spawn worker_zero(p0) @high;
    spawn worker_neg(p_neg) @high;
    spawn worker_u32_huge(p_u32) @high;
    spawn worker_u64_huge(p_u64) @high;
    spawn driver() @low;
}
";
        let temp = TempWorkDir::create("zevc-loop-every-clamp").unwrap();
        let src_path = temp.dir.join("clamp.zev");
        std::fs::write(&src_path, src).unwrap();

        let (_, mir) = compile_to_mir(&[src_path]).unwrap();
        let host_ll = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        assert!(host_ll.contains("icmp sle i32"));
        assert!(host_ll.contains("icmp ugt i32"));
        assert!(host_ll.contains("icmp ugt i64"));
        assert!(host_ll.contains("2147483647"));

        let exit = runner::compile_and_run_host_llvm(&host_ll, true, &[]).unwrap();
        assert_eq!(exit, 0);

        // Also verify custom sub-millisecond tick scaling (`sys.set_tick(500us)`) with signed negative and u64 huge periods.
        let src_scaled = "\
watch<u32> c_neg = 0;
watch<u32> c_huge = 0;

task w_neg(i64 p) {
    u32 n = 0;
    loop every p {
        n += 1;
        c_neg.publish(n);
    }
}

task w_huge(u64 p) {
    u32 n = 0;
    loop every p {
        n += 1;
        c_huge.publish(n);
    }
}

task driver() {
    delay(2ms); // 4 ticks at 500us
    assert(c_neg.read() == 5);
    assert(c_huge.read() == 1);
    sys.exit(0);
}

fn main() {
    sys.set_tick(500us);
    i64 pn = -5;
    u64 ph = 1u64 << 40;
    spawn w_neg(pn) @high;
    spawn w_huge(ph) @high;
    spawn driver() @low;
}
";
        let src_scaled_path = temp.dir.join("clamp_scaled.zev");
        std::fs::write(&src_scaled_path, src_scaled).unwrap();
        let (_, mir_scaled) = compile_to_mir(&[src_scaled_path]).unwrap();
        let ll_scaled = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir_scaled,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        let exit_scaled = runner::compile_and_run_host_llvm(&ll_scaled, true, &[]).unwrap();
        assert_eq!(exit_scaled, 0);
    }

    #[test]
    fn host_sanitize_verifies_scheduler_across_tick_wraparound() {
        let temp = TempWorkDir::create("zevc-tick-wraparound").unwrap();

        // 1. Timers (`delay`, `wait timeout`/`deadline`, `loop every` cadence & overrun resync)
        //    crossing both `i32::MAX` (`0x7FFF_FFFF -> 0x8000_0000`) and `u32::MAX` (`0xFFFF_FFFF -> 0`).
        let src_timers = "\
signal never_sig;
watch<bool> i32_delay_done = false;
watch<bool> i32_wait_done = false;
watch<bool> i32_every_done = false;

signal u32_delay_sig;
signal u32_wait_sig;
signal u32_every_sig;
watch<bool> u32_delay_done = false;
watch<bool> u32_wait_done = false;
watch<bool> u32_every_done = false;

task delay_worker() {
    u32 s1 = now_ticks();
    assert(s1 == 0x7FFF_FFF8u32);
    delay(15ms);
    assert(now_ticks() == 0x8000_0007u32);
    assert(now_ticks() - s1 == 15u32);
    i32_delay_done.publish(true);

    u32_delay_sig.wait();
    u32 s2 = now_ticks();
    assert(s2 == 0xFFFF_FFF0u32);
    delay(25ms);
    assert(now_ticks() == 0x0000_0009u32);
    assert(now_ticks() - s2 == 25u32);
    u32_delay_done.publish(true);

    never_sig.wait();
    loop { checkpoint(); }
}

task wait_worker() {
    u32 s1 = now_ticks();
    bool timed_out = false;
    wait {
        never_sig.wait() => { assert(false); }
        timeout 12ms => { timed_out = true; }
    }
    assert(timed_out);
    assert(now_ticks() == 0x8000_0004u32);
    assert(now_ticks() - s1 == 12u32);
    i32_wait_done.publish(true);

    u32_wait_sig.wait();
    u32 s2 = now_ticks();
    u32 dl2 = s2 + 20u32;
    bool dl_fired = false;
    wait {
        never_sig.wait() => { assert(false); }
        deadline dl2 => { dl_fired = true; }
    }
    assert(dl_fired);
    assert(now_ticks() == 0x0000_0004u32);
    assert(now_ticks() - s2 == 20u32);
    u32_wait_done.publish(true);

    never_sig.wait();
    loop { checkpoint(); }
}

task every_worker() {
    u32 step = 0u32;
    u32 prev = 0u32;
    loop every 5ms {
        step += 1u32;
        u32 t = now_ticks();
        if (step == 1u32) {
            assert(t == 0x7FFF_FFF8u32);
        } else if (step <= 4u32) {
            // Wakes cleanly across `0x7FFF_FFFF -> 0x8000_0000`: +5, +5, +5 => 0x8000_0007
            assert(t - prev == 5u32);
            if (step == 4u32) {
                // Both delay_worker and wait_worker finish by 0x8000_0007; stall 12ms -> 0x8000_0013.
                sys.tick_by(12u32);
            }
        } else if (step == 5u32) {
            // EveryLatch coalesces missed deadlines (0x8000_000C & 0x8000_0011 -> resync_dl = 0x8000_0011)
            // and runs one immediate recovery step at 0x8000_0013 without burst-catching up both skipped ticks.
            assert(t == 0x8000_0013u32);
        } else {
            // Next wake locks cleanly onto the original 5ms phase grid: 0x8000_0011 + 5 = 0x8000_0016.
            assert(t == 0x8000_0016u32);
            break;
        }
        prev = t;
    }
    i32_every_done.publish(true);

    u32_every_sig.wait();
    u32 step2 = 0u32;
    u32 prev2 = 0u32;
    loop every 10ms {
        step2 += 1u32;
        u32 t = now_ticks();
        if (step2 == 1u32) {
            assert(t == 0xFFFF_FFF0u32);
        } else if (step2 == 2u32) {
            assert(t == 0xFFFF_FFFAu32);
        } else if (step2 == 3u32) {
            // Crossed u32 wrap (`0xFFFF_FFFF -> 0`) on normal cadence:
            assert(t == 0x0000_0004u32);
            assert(t - prev2 == 10u32);
        } else if (step2 == 4u32) {
            assert(t == 0x0000_000Eu32);
            // Overrun stall 23ms -> 0x0000_0025; skips 0x0000_0018, resync_dl = 0x0000_0022.
            sys.tick_by(23u32);
        } else if (step2 == 5u32) {
            assert(t == 0x0000_0025u32);
        } else {
            assert(t == 0x0000_002Cu32);
            break;
        }
        prev2 = t;
    }
    u32_every_done.publish(true);
    never_sig.wait();
    loop { checkpoint(); }
}

task coordinator() {
    while (!(i32_delay_done.read() && i32_wait_done.read() && i32_every_done.read())) {
        delay(1ms);
    }
    // Jump near u32 wraparound (`0xFFFF_FFF0`).
    u32 jump = 0xFFFF_FFF0u32 - now_ticks();
    sys.tick_by(jump);
    assert(now_ticks() == 0xFFFF_FFF0u32);
    u32_delay_sig.notify();
    u32_wait_sig.notify();
    u32_every_sig.notify();

    while (!(u32_delay_done.read() && u32_wait_done.read() && u32_every_done.read())) {
        delay(1ms);
    }
    println(\"tick_wrap_timers_ok\");
    sys.exit(0);
}

fn main() {
    sys.tick_by(0x7FFF_FFF8u32);
    spawn delay_worker() @high;
    spawn wait_worker() @high;
    spawn every_worker() @normal;
    spawn coordinator() @low;
}
";
        let timers_path = temp.dir.join("wrap_timers.zev");
        std::fs::write(&timers_path, src_timers).unwrap();
        let (_, mir_timers) = compile_to_mir(&[timers_path]).unwrap();
        let ll_timers = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir_timers,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        assert_eq!(
            runner::compile_and_run_host_llvm(&ll_timers, true, &[]).unwrap(),
            0
        );

        // 2. Priority starvation preemption windows across `i32::MAX` and `u32::MAX` wraparound.
        let src_starve = "\
watch<u32> starver_hits = 0u32;

task starver() {
    // At t0 = 0x7FFF_FFFD, @normal starvation window is 15ms => due = 0x8000_000C.
    assert(now_ticks() == 0x8000_000Cu32);
    starver_hits.publish(1u32);
    loop { checkpoint(); }
}

task hog() {
    // @high hogs CPU via soft-yielding while loop until @normal `starver` hits its 15ms starvation window across `i32::MAX`.
    while (starver_hits.read() == 0u32) {
        sys.tick_by(1u32);
        checkpoint();
    }
    println(\"tick_wrap_starve_i32_ok\");
    sys.exit(0);
}

fn main() {
    sys.tick_by(0x7FFF_FFFDu32);
    spawn hog() @high;
    spawn starver() @normal;
}
";
        let starve_path = temp.dir.join("wrap_starve_i32.zev");
        std::fs::write(&starve_path, src_starve).unwrap();
        let (_, mir_starve) = compile_to_mir(&[starve_path]).unwrap();
        let ll_starve = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir_starve,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        assert_eq!(
            runner::compile_and_run_host_llvm(&ll_starve, true, &[]).unwrap(),
            0
        );

        // Also test starvation window crossing `u32::MAX -> 0` (`0xFFFF_FFFD -> 0x0000_0002`).
        let src_starve_u32 = "\
watch<u32> starver_hits = 0u32;

task starver() {
    assert(now_ticks() == 0x0000_000Cu32);
    starver_hits.publish(1u32);
    loop { checkpoint(); }
}

task hog() {
    while (starver_hits.read() == 0u32) {
        sys.tick_by(1u32);
        checkpoint();
    }
    println(\"tick_wrap_starve_u32_ok\");
    sys.exit(0);
}

fn main() {
    sys.tick_by(0xFFFF_FFFDu32);
    spawn hog() @high;
    spawn starver() @normal;
}
";
        let starve_u32_path = temp.dir.join("wrap_starve_u32.zev");
        std::fs::write(&starve_u32_path, src_starve_u32).unwrap();
        let (_, mir_starve_u32) = compile_to_mir(&[starve_u32_path]).unwrap();
        let ll_starve_u32 = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir_starve_u32,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        assert_eq!(
            runner::compile_and_run_host_llvm(&ll_starve_u32, true, &[]).unwrap(),
            0
        );

        // 3. `--max-ticks` wraparound: must run 16 ticks across `0x7FFF_FFFC -> 0x8000_000C` without stopping early.
        let src_max_ticks = "\
task pulse() {
    loop every 4ms {
        println(f\"pulse t={now_ticks():08x}\");
    }
}

fn main() {
    sys.tick_by(0x7FFF_FFFCu32);
    spawn pulse() @normal;
}
";
        let max_ticks_path = temp.dir.join("wrap_max_ticks.zev");
        std::fs::write(&max_ticks_path, src_max_ticks).unwrap();
        let (_, mir_mt) = compile_to_mir(&[max_ticks_path]).unwrap();
        let mut opts = zev_codegen_llvm::CodegenOptions::host_linux();
        opts.max_ticks = 16;
        let ll_mt = zev_codegen_llvm::emit_llvm_ir_with_options(&mir_mt, &opts).unwrap();
        let (exit_mt, stdout_mt) =
            runner::compile_and_run_host_llvm_output(&ll_mt, true, &[]).unwrap();
        assert_eq!(exit_mt, 0);
        assert!(stdout_mt.contains("pulse t=7ffffffc"), "stdout={stdout_mt}");
        assert!(stdout_mt.contains("pulse t=80000000"), "stdout={stdout_mt}");
        assert!(stdout_mt.contains("pulse t=80000008"), "stdout={stdout_mt}");
        assert!(
            !stdout_mt.contains("pulse t=8000000c"),
            "max_ticks=16 from 0x7ffffffc must stop before 0x8000000c, got stdout={stdout_mt}"
        );
    }

    #[test]
    fn plain_check_runs_mir_lowering_and_surfaces_mir_errors() {
        let temp = TempWorkDir::create("zevc-check-mir").unwrap();

        // 1. Valid program passes plain `zevc check` (through both sema and MIR lowering).
        let ok_path = temp.dir.join("ok.zev");
        std::fs::write(
            &ok_path,
            "task idle() { loop every 5ms { checkpoint(); } }\nfn main() { spawn idle() @normal; }\n",
        )
        .unwrap();
        let ok_res = execute_check_command(std::slice::from_ref(&ok_path), &[], None, None, false);
        assert_eq!(ok_res.exit_code, ExitCode::SUCCESS);
        let ok_out = ok_res.stdout.expect("stdout on ok check");
        assert!(ok_out.contains("OK (1 file(s) verified"), "stdout={ok_out}");

        // 2. Program that passes zev-sema but hits a MIR lowering error (`conditional .attach_shared`)
        //    must fail in BOTH plain and `--json` modes of `zevc check`!
        let conflict_irq = "\
irq IRQ_A = 3;
isr h1() {}
isr h2() {}
fn read_pin(bool sel) -> bool { return sel; }
fn main() {
    if (read_pin(true)) { IRQ_A.attach_shared(h1); }
    if (read_pin(false)) { IRQ_A.attach_shared(h2); }
}
";
        let err_path = temp.dir.join("conflict_irq.zev");
        std::fs::write(&err_path, conflict_irq).unwrap();

        let plain_res =
            execute_check_command(std::slice::from_ref(&err_path), &[], None, None, false);
        assert_eq!(plain_res.exit_code, ExitCode::FAILURE);
        let err_text = plain_res.stderr.expect("plain check MIR error on stderr");
        assert!(err_text.contains("MIR lowering error"), "stderr={err_text}");

        let json_res =
            execute_check_command(std::slice::from_ref(&err_path), &[], None, None, true);
        assert_eq!(json_res.exit_code, ExitCode::FAILURE);
        let json_out = json_res.stdout.expect("json check MIR error on stdout");
        assert!(json_out.contains("attach_shared"), "json={json_out}");
    }

    #[test]
    fn cli_check_reports_multiple_parse_errors_across_items() {
        let temp = TempWorkDir::create("zevc-multi-parse-err").unwrap();
        let path = temp.dir.join("multi_bad.zev");
        std::fs::write(
            &path,
            "\
fn bad1() {
    let x = 1;
}

task bad2() {
    every 5ms { checkpoint(); }
}

u32 bad3(u32 x) {
    return x;
}
",
        )
        .unwrap();

        let plain_res = execute_check_command(std::slice::from_ref(&path), &[], None, None, false);
        assert_eq!(plain_res.exit_code, ExitCode::FAILURE);
        let stderr = plain_res.stderr.expect("human parse errors");
        assert!(stderr.contains("P0001"), "stderr={stderr}");
        assert!(stderr.contains("P0011"), "stderr={stderr}");
        assert!(stderr.contains("P0018"), "stderr={stderr}");

        let json_res = execute_check_command(std::slice::from_ref(&path), &[], None, None, true);
        assert_eq!(json_res.exit_code, ExitCode::FAILURE);
        let json = json_res.stdout.expect("json parse errors");
        assert!(json.contains("\"code\": \"P0001\""), "json={json}");
        assert!(json.contains("\"code\": \"P0011\""), "json={json}");
        assert!(json.contains("\"code\": \"P0018\""), "json={json}");
    }

    #[test]
    fn review_fixes_a_b_c_d_work_end_to_end() {
        let temp = TempWorkDir::create("zevc-review-fixes").unwrap();

        // Fix A: destination-typed index, slice bounds, untyped array literal, and `loop every` widening.
        let fix_a_src = "\
fn idx([400]u32 t, u8 a, u8 b) -> u32 { return t[a + b]; }
fn set_idx(*[400]u32 t, u8 a, u8 b, u32 val) { (*t)[a + b] = val; }
fn sl([]u8 s, u8 a, u8 b) -> usize { []u8 v = s[0..a + b]; return v.len; }
fn sl_lo([]u8 s, u8 a, u8 b) -> usize { []u8 v = s[a + b..350]; return v.len; }
task per_check(u8 a, u8 b) {
    u32 iter = 0;
    loop every a + b {
        println(f\"tick_ms={sys.now_ms()}\");
        if (iter == 1) {
            sys.exit(0);
        }
        iter += 1;
    }
}
fn main() {
    [400]u32 t;
    t[300] = 777u32;
    u32 v1 = idx(t, 200u8, 100u8);
    set_idx(&t, 200u8, 100u8, 888u32);
    u32 v2 = t[300];
    [400]u8 buf;
    usize n1 = sl(buf, 200u8, 100u8);
    usize n2 = sl_lo(buf, 200u8, 100u8);
    i32 u_arr = [1, 300][1];
    println(f\"v1={v1} v2={v2} n1={n1} n2={n2} arr={u_arr}\");
    spawn per_check(200u8, 100u8) @normal;
}
";
        let fix_a_path = temp.dir.join("fix_a.zev");
        std::fs::write(&fix_a_path, fix_a_src).unwrap();
        let (_, fix_a_mir) = compile_to_mir(&[fix_a_path]).unwrap();
        let mut fix_a_opts = zev_codegen_llvm::CodegenOptions::host_linux();
        fix_a_opts.max_ticks = 350;
        let fix_a_ll =
            zev_codegen_llvm::emit_llvm_ir_with_options(&fix_a_mir, &fix_a_opts).unwrap();
        let (exit_a, out_a) =
            runner::compile_and_run_host_llvm_output(&fix_a_ll, true, &[]).unwrap();
        assert_eq!(exit_a, 0);
        assert!(
            out_a.contains("v1=777 v2=888 n1=300 n2=50 arr=300"),
            "Fix A runtime mismatch:\n{out_a}"
        );
        assert!(
            out_a.contains("tick_ms=300"),
            "Fix A loop every widening mismatch (expected tick_ms=300):\n{out_a}"
        );

        // Fix B: host accepts 100000 and 5_000_000_000 across locals, const, generic `<usize N>`
        // and runs on host; 32-bit board accepts 100000 & rejects 5_000_000_000; AVR accepts 65535
        // and rejects 70000.
        let fix_b_host = "\
const usize CAP = 100000;
const usize BIG = 5_000_000_000;
fn id_u<usize N>() -> usize { return N; }
fn main() {
    usize n = 100000;
    usize b = 5_000_000_000;
    println(f\"n={n} cap={CAP} g={id_u<100000>()} big={b} gbig={id_u<BIG>()}\");
    sys.exit(0);
}
";
        let fix_b_host_path = temp.dir.join("fix_b_host.zev");
        std::fs::write(&fix_b_host_path, fix_b_host).unwrap();
        let host_chk = execute_check_command(
            std::slice::from_ref(&fix_b_host_path),
            &[],
            None,
            None,
            false,
        );
        assert_eq!(
            host_chk.exit_code,
            ExitCode::SUCCESS,
            "{:?}",
            host_chk.stderr
        );
        let (_, mir_b) = compile_to_mir(&[fix_b_host_path]).unwrap();
        let ll_b = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir_b,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        let (exit_b, out_b) = runner::compile_and_run_host_llvm_output(&ll_b, true, &[]).unwrap();
        assert_eq!(exit_b, 0);
        assert!(
            out_b.contains("n=100000 cap=100000 g=100000 big=5000000000 gbig=5000000000"),
            "out_b={out_b}"
        );

        let fix_b_32_ok_path = temp.dir.join("fix_b_32_ok.zev");
        std::fs::write(
            &fix_b_32_ok_path,
            "const usize CAP = 100000;\nfn id_u<usize N>() -> usize { return N; }\nfn main() { usize n = 100000; usize _s = n + CAP + id_u<100000>(); }\n",
        )
        .unwrap();
        let ok_32 =
            execute_check_command(&[fix_b_32_ok_path], &[], Some("nucleo_f334r8"), None, false);
        assert_eq!(ok_32.exit_code, ExitCode::SUCCESS, "{:?}", ok_32.stderr);

        let fix_b_16_ok_path = temp.dir.join("fix_b_16_ok.zev");
        std::fs::write(
            &fix_b_16_ok_path,
            "const usize CAP = 65535;\nfn id_u<usize N>() -> usize { return N; }\nfn main() { usize n = 65535; usize _s = n + CAP + id_u<65535>(); }\n",
        )
        .unwrap();
        let ok_16 =
            execute_check_command(&[fix_b_16_ok_path], &[], Some("arduino_uno"), None, false);
        assert_eq!(ok_16.exit_code, ExitCode::SUCCESS, "{:?}", ok_16.stderr);

        for (board, bits_str, src) in [
            (
                "nucleo_f334r8",
                "32-bit",
                "fn main() { usize n = 5_000_000_000; }\n",
            ),
            (
                "nucleo_f334r8",
                "32-bit",
                "const usize CAP = 5_000_000_000;\nfn main() {}\n",
            ),
            (
                "nucleo_f334r8",
                "32-bit",
                "fn id_u<usize N>() -> usize { return N; }\nfn main() { usize _x = id_u<5_000_000_000>(); }\n",
            ),
            ("arduino_uno", "16-bit", "fn main() { usize n = 70000; }\n"),
            (
                "arduino_uno",
                "16-bit",
                "const usize CAP = 70000;\nfn main() {}\n",
            ),
            (
                "arduino_uno",
                "16-bit",
                "fn id_u<usize N>() -> usize { return N; }\nfn main() { usize _x = id_u<70000>(); }\n",
            ),
        ] {
            let p = temp.dir.join("fix_b_bad.zev");
            std::fs::write(&p, src).unwrap();
            let res = execute_check_command(&[p], &[], Some(board), None, false);
            assert_eq!(res.exit_code, ExitCode::FAILURE, "expected failure for {board}: {src}");
            let stderr = res.stderr.unwrap_or_default();
            assert!(
                stderr.contains(board) && stderr.contains(bits_str),
                "expected target `{board}` and `{bits_str}` in:\n{stderr}"
            );
        }

        // Fix C: bare const expression in generic call args explains parenthesization + rewrite.
        let fix_c_bad_path = temp.dir.join("fix_c_bad.zev");
        std::fs::write(
            &fix_c_bad_path,
            "const usize K = 3;\nfn lim<usize N>(u32 x) -> u32 { return x + (N as u32); }\nfn main() { u32 _y = lim<K + 1>(1); }\n",
        )
        .unwrap();
        let res_c = execute_check_command(&[fix_c_bad_path], &[], None, None, false);
        assert_eq!(res_c.exit_code, ExitCode::FAILURE);
        let stderr_c = res_c.stderr.unwrap_or_default();
        assert!(
            stderr_c.contains("error[E0029]")
                && stderr_c.contains("lim<(K + 1)>(1)")
                && stderr_c.contains("parenthes"),
            "Fix C diagnostic mismatch:\n{stderr_c}"
        );
        assert!(
            !stderr_c.contains("cannot be referenced without concrete call arguments"),
            "Fix C should not show bare-ref error:\n{stderr_c}"
        );

        let fix_c_ok_path = temp.dir.join("fix_c_ok.zev");
        std::fs::write(
            &fix_c_ok_path,
            "fn f(bool x, bool y) -> bool { return x && y; }\nfn main() { u32 a = 1; u32 b = 2; u32 c = 3; u32 d = 4; bool _r = f(a < b + 1, c > (d)); }\n",
        )
        .unwrap();
        let res_c_ok = execute_check_command(&[fix_c_ok_path], &[], None, None, false);
        assert_eq!(
            res_c_ok.exit_code,
            ExitCode::SUCCESS,
            "{:?}",
            res_c_ok.stderr
        );

        // Fix D: all-literal generic calls infer from destination, keep `i32` without destination,
        // and explain defaulted type + `pick<u32>(3, 4)` when still failing.
        let fix_d_ok_src = "\
fn pick<T>(T a, T b) -> T { return a; }
fn ret_u16() -> u16 { return pick(7, 8); }
fn takes_u8(u8 v) -> u8 { return v; }
fn main() {
    u32 b = pick(3, 4);
    u32 c = 0;
    c = pick(5, 6);
    u16 r = ret_u16();
    u8 u = takes_u8(pick(9, 10));
    println(f\"b={b} c={c} r={r} u={u} def={pick(-3, 4)}\");
    sys.exit(0);
}
";
        let fix_d_ok_path = temp.dir.join("fix_d_ok.zev");
        std::fs::write(&fix_d_ok_path, fix_d_ok_src).unwrap();
        let (_, mir_d) = compile_to_mir(&[fix_d_ok_path]).unwrap();
        let ll_d = zev_codegen_llvm::emit_llvm_ir_with_options(
            &mir_d,
            &zev_codegen_llvm::CodegenOptions::host_linux(),
        )
        .unwrap();
        let (exit_d, out_d) = runner::compile_and_run_host_llvm_output(&ll_d, true, &[]).unwrap();
        assert_eq!(exit_d, 0);
        assert!(
            out_d.contains("b=3 c=5 r=7 u=9 def=-3"),
            "Fix D runtime mismatch:\n{out_d}"
        );

        let fix_d_bad_path = temp.dir.join("fix_d_bad.zev");
        std::fs::write(
            &fix_d_bad_path,
            "fn pick<T>(T a, T b) -> T { return a; }\nfn main() { bool _eq = pick(3, 4) == 5u32; }\n",
        )
        .unwrap();
        let res_d_bad = execute_check_command(&[fix_d_bad_path], &[], None, None, false);
        assert_eq!(res_d_bad.exit_code, ExitCode::FAILURE);
        let stderr_d = res_d_bad.stderr.unwrap_or_default();
        assert!(
            stderr_d.contains("defaulted to 'i32'") && stderr_d.contains("pick<u32>(3, 4)"),
            "Fix D defaulted diagnostic mismatch:\n{stderr_d}"
        );
    }

    #[test]
    fn item2_host_and_qemu_run_same_script_and_report_ok_err_timeout() {
        let temp = TempWorkDir::create("zevc-item2-script").unwrap();
        let src = "\
import board;
import std.io as io;

fn slice_eq([]u8 a, []u8 b) -> bool {
    if (a.len != b.len) {
        return false;
    }
    for (usize i in 0..a.len) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

task echo_loop(io.DuplexStream port) {
    [64]u8 line;
    usize len = 0;
    loop {
        []u8 chunk = port.rx.wait_read();
        usize n = chunk.len;
        usize idx = 0;
        while (idx < n) {
            u8 b = chunk[idx];
            idx += 1;
            if (b == 13u8) {
                continue;
            }
            if (b == 10u8) {
                []u8 cmd = line[0..len];
                if (slice_eq(cmd, \"ping\")) {
                    io.stream_write(&port, \"pong\\nok\\n\");
                } else if (slice_eq(cmd, \"fail\")) {
                    io.stream_write(&port, \"err BadCmd\\n\");
                } else if (slice_eq(cmd, \"hang\")) {
                } else {
                    io.stream_write(&port, \"err Unknown\\n\");
                }
                len = 0;
            } else if (len < line.len) {
                line[len] = b;
                len += 1;
            }
        }
        port.rx.consume(n);
    }
}

fn main() {
    board.init();
    io.DuplexStream port = board.console_open();
    spawn echo_loop(port) @normal;
}
";
        let app_path = temp.dir.join("console_app.zev");
        std::fs::write(&app_path, src).unwrap();

        let ok_script = temp.dir.join("ok.script");
        std::fs::write(&ok_script, "# comment\n\nping\nping\n").unwrap();
        let ok_cmds = script_runner::collect_script_commands(Some(&ok_script), &[]).unwrap();
        assert_eq!(ok_cmds, vec!["ping".to_string(), "ping".to_string()]);

        let timeout = std::time::Duration::from_millis(3_000);
        let short_timeout = std::time::Duration::from_millis(300);

        // 1. Host simulator (`zevc run --sanitize` equivalent)
        let host_opts = zev_codegen_llvm::CodegenOptions::host_linux();
        let (h_bits, h_name) = target_info_from_llvm_opts(Some(&host_opts));
        let (_, host_mir) = compile_to_mir_with_target(
            std::slice::from_ref(&app_path),
            &[],
            host_opts.hal_dir.as_deref(),
            host_opts.board_module.as_deref(),
            h_bits,
            &h_name,
        )
        .unwrap();
        let host_ll = zev_codegen_llvm::emit_llvm_ir_with_options(&host_mir, &host_opts).unwrap();
        let host_bin = runner::compile_host_llvm_binary(&temp, &host_ll, true).unwrap();

        let mut host_transcript = Vec::new();
        let mut child_ok = runner::spawn_host_llvm_binary(&host_bin, &[]).unwrap();
        script_runner::run_script_on_child_to(
            &mut child_ok,
            &ok_cmds,
            timeout,
            &mut host_transcript,
        )
        .expect("host scripted ok run should succeed");
        let host_out = String::from_utf8_lossy(&host_transcript);
        assert!(host_out.contains("pong\nok\n"), "host_out={host_out:?}");

        let mut host_err_transcript = Vec::new();
        let mut child_err = runner::spawn_host_llvm_binary(&host_bin, &[]).unwrap();
        let err_res = script_runner::run_script_on_child_to(
            &mut child_err,
            &["fail".to_string()],
            timeout,
            &mut host_err_transcript,
        )
        .unwrap_err();
        assert!(err_res.contains("err BadCmd"), "err_res={err_res}");

        let mut host_to_transcript = Vec::new();
        let mut child_to = runner::spawn_host_llvm_binary(&host_bin, &[]).unwrap();
        let to_res = script_runner::run_script_on_child_to(
            &mut child_to,
            &["hang".to_string()],
            short_timeout,
            &mut host_to_transcript,
        )
        .unwrap_err();
        assert!(to_res.contains("timed out"), "to_res={to_res}");

        // 2. QEMU (`zevc run-qemu` equivalent with exact same script)
        let qemu_opts_cg = zev_codegen_llvm::CodegenOptions::lm3s6965();
        let (q_bits, q_name) = target_info_from_llvm_opts(Some(&qemu_opts_cg));
        let (_, qemu_mir) = compile_to_mir_with_target(
            std::slice::from_ref(&app_path),
            &[],
            qemu_opts_cg.hal_dir.as_deref(),
            qemu_opts_cg.board_module.as_deref(),
            q_bits,
            &q_name,
        )
        .unwrap();
        let elf_path = temp.dir.join("console_qemu.elf");
        let qemu_build = zev_codegen_llvm::build_elf(&qemu_mir, &elf_path, &qemu_opts_cg).unwrap();
        let q_run_opts = zev_codegen_llvm::QemuRunOptions::default();

        let mut qemu_transcript = Vec::new();
        let mut q_ok =
            zev_codegen_llvm::spawn_elf_in_qemu(&qemu_build.elf_path, &q_run_opts, true).unwrap();
        script_runner::run_script_on_child_to(&mut q_ok, &ok_cmds, timeout, &mut qemu_transcript)
            .expect("qemu scripted ok run should succeed");
        let qemu_out = String::from_utf8_lossy(&qemu_transcript);
        assert!(qemu_out.contains("pong\nok\n"), "qemu_out={qemu_out:?}");

        let mut qemu_err_transcript = Vec::new();
        let mut q_err =
            zev_codegen_llvm::spawn_elf_in_qemu(&qemu_build.elf_path, &q_run_opts, true).unwrap();
        let q_err_res = script_runner::run_script_on_child_to(
            &mut q_err,
            &["fail".to_string()],
            timeout,
            &mut qemu_err_transcript,
        )
        .unwrap_err();
        assert!(q_err_res.contains("err BadCmd"), "q_err_res={q_err_res}");

        let mut qemu_to_transcript = Vec::new();
        let mut q_to =
            zev_codegen_llvm::spawn_elf_in_qemu(&qemu_build.elf_path, &q_run_opts, true).unwrap();
        let q_to_res = script_runner::run_script_on_child_to(
            &mut q_to,
            &["hang".to_string()],
            short_timeout,
            &mut qemu_to_transcript,
        )
        .unwrap_err();
        assert!(q_to_res.contains("timed out"), "q_to_res={q_to_res}");
    }
}
