mod cli;
mod diag;
mod docs;
mod docs_errors;
mod runner;
mod svd;

use cli::{parse_args, Command, HELP_TEXT};
use diag::{
    format_check_human_ok, primary_doc_cmd, render_check_json_err, render_check_json_ok, JsonDiag,
    SourceFile,
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

fn load_and_check_detailed_with_includes(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
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
    let all_generics: BTreeSet<String> =
        zev_sema::collect_import_aware_generic_names(&generic_roots, include_dirs);

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
    let bundle = load_and_check_detailed_with_includes(files, include_dirs).map_err(|f| f.human)?;
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
        Some("host") | Some("x86_64") | Some("x86_64-unknown-linux-gnu") => {
            let mut opts = zev_codegen_llvm::CodegenOptions::default();
            opts.target = zev_codegen_llvm::TargetArch::HostX86_64Linux;
            Ok(opts)
        }
        Some(other) => Err(format!(
            "unsupported `--target {other}` (expected `cortex-m3`, `esp32c6`, `avr`, or `host`)"
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

enum RamExecOutcome {
    Output { text: String, within_budget: bool },
    UsageErr(String),
    CompileErr(String),
}

fn execute_ram_command(
    files: &[PathBuf],
    include_dirs: &[PathBuf],
    target: Option<&str>,
    opt_level: Option<&str>,
) -> RamExecOutcome {
    let llvm_opts = match target {
        None => None,
        Some(t) => match resolve_llvm_options(Some(t)) {
            Ok(mut o) => {
                if o.target == zev_codegen_llvm::TargetArch::HostX86_64Linux {
                    return RamExecOutcome::UsageErr(format!(
                        "unsupported `--target {t}` for `zevc ram` (expected `cortex-m3`, `esp32c6`, or `avr`)"
                    ));
                }
                if let Some(lvl) = opt_level {
                    o.opt_level = lvl.to_string();
                }
                Some(o)
            }
            Err(err) => return RamExecOutcome::UsageErr(err),
        },
    };
    let (_, mir) = match compile_to_mir_with_includes(files, include_dirs) {
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
            json,
        } => {
            match load_and_check_detailed_with_includes(&files, &include_dirs) {
            Ok(bundle) => {
                let verified_files = bundle.programs.len();
                let dual_vec: Vec<String> =
                    bundle.sema.dual_specialized_fns.iter().cloned().collect();
                if json {
                    println!(
                        "{}",
                        render_check_json_ok(
                            verified_files,
                            bundle.sema.spawned_tasks.len(),
                            bundle.sema.primitives.len(),
                            bundle.sema.tick_period_us,
                            &dual_vec,
                            &bundle.sema.monomorphized_instances,
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
            opt_level,
        } => match execute_ram_command(
            &files,
            &include_dirs,
            target.as_deref(),
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
        Command::EmitC {
            files,
            include_dirs,
            output,
        } => {
            match compile_to_mir_with_includes(&files, &include_dirs) {
            Ok((_, mir)) => {
                let c_src = zev_codegen_c::emit_c(&mir);
                if let Err(err) = write_output(output.as_deref(), &c_src) {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
            }
        }
        Command::Run {
            files,
            include_dirs,
            sanitize,
            max_ticks,
            run_args,
        } => {
            match compile_to_mir_with_includes(&files, &include_dirs) {
            Ok((_, mir)) => {
                let c_src = zev_codegen_c::emit_c(&mir);
                match runner::compile_and_run_c99(&c_src, sanitize, max_ticks, &run_args) {
                    Ok(0) => ExitCode::SUCCESS,
                    Ok(code) => ExitCode::from((code.clamp(1, 255)) as u8),
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
        Command::EmitLlvm {
            files,
            include_dirs,
            target,
            opt_level,
            output,
        } => {
            let mut opts = match resolve_llvm_options(target.as_deref()) {
                Ok(o) => o,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::from(2);
                }
            };
            if let Some(ref lvl) = opt_level {
                opts.opt_level = lvl.clone();
            }
            match compile_to_mir_with_includes(&files, &include_dirs) {
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
            match compile_to_mir_with_includes(&files, &include_dirs) {
            Ok((_, mir)) => {
                let temp = match TempWorkDir::create("zevc-qemu") {
                    Ok(t) => t,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                let elf_path = temp.dir.join("firmware.elf");
                let mut opts = zev_codegen_llvm::CodegenOptions::default();
                opts.opt_level = opt_level;
                if let Some(ticks) = max_ticks {
                    opts.max_ticks = ticks.min(u32::MAX as u64) as u32;
                }
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
            match compile_to_mir_with_includes(&files, &include_dirs) {
            Ok((_, mir)) => {
                let out_bin = output.unwrap_or_else(|| default_output_with_ext(&files, "bin"));
                let out_elf = out_bin.with_extension("elf");
                let mut opts = zev_codegen_llvm::CodegenOptions::esp32c6();
                opts.opt_level = opt_level;
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
            match compile_to_mir_with_includes(&files, &include_dirs) {
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
                let mut opts = zev_codegen_llvm::CodegenOptions::esp32c6();
                opts.opt_level = opt_level;
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let mc_c = zev_codegen_c::emit_c(&mc_mir);
        let mc_exit = runner::compile_and_run_c99(&mc_c, true, Some(25), &[]).unwrap();
        assert_eq!(mc_exit, 0);

        let (_, sim_mir) =
            compile_to_mir(&[root.join("examples/sim_demo.zev")]).expect("sim_demo must check");
        let sim_c = zev_codegen_c::emit_c(&sim_mir);
        let sim_exit = runner::compile_and_run_c99(&sim_c, true, None, &[]).unwrap();
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
    u8 static_buf[16];
    u16 rpm = 2400;
    []u8 fit_line = fmt_buf(static_buf, f\"rpm={rpm}\");
    println(fit_line);

    // 2. Dynamic `[]u8` slice hole uses `catch (FmtErr err, []u8 partial)` cleanly!
    u8 small_buf[10];
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
    u8 long_tag[12] = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76];
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

        // 3. Verify emitted C99 runs cleanly under ASan/UBSan (`fit_line` + `trunc_line` + `println(f"...")`)
        let c_src = zev_codegen_c::emit_c(&mir);
        let c_exit = runner::compile_and_run_c99(&c_src, true, Some(10), &[]).unwrap();
        assert_eq!(c_exit, 0);

        // 4. Verify E0028..E0033 diagnostics and `zevc doc` explainers (45 total)
        let doc_list = docs::lookup_doc(None, true).unwrap();
        assert!(doc_list.contains("E0028") && doc_list.contains("E0033"));

        let wave2_errs = [
            (
                "e28.zev",
                "fn main() { u8 dst[8]; u32 rpm = 1800; []u8 s = fmt_buf(dst, f\"rpm={rpm}\"); }\n",
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

        // 1. Verify host C99 simulation with --max-ticks 15
        let c_src = zev_codegen_c::emit_c(&mir);
        let c_exit = runner::compile_and_run_c99(&c_src, true, Some(15), &[]).unwrap();
        assert_eq!(c_exit, 0);

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
    fn ram_command_preserves_plain_output_and_prints_target_stack_proofs() {
        let root = repo_root();
        let cm3_path = root.join("examples/cortex_m_demo.zev");
        let esp_path = root.join("examples/esp32c6_demo.zev");
        let uno_path = root.join("examples/arduino_uno_demo.zev");

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

        // 2. Targeted `zevc ram --target <cortex-m3|esp32c6|avr>` appends hardware proof and passes budget
        for (path, tgt, budget_str) in [
            (cm3_path, "cortex-m3", "65536 B"),
            (esp_path, "esp32c6", "524288 B"),
            (uno_path, "avr", "2048 B"),
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
}
