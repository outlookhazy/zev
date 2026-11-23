use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Check {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        board: Option<String>,
        chip: Option<String>,
        json: bool,
    },
    Ram {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        target: Option<String>,
        board: Option<String>,
        chip: Option<String>,
        opt_level: Option<String>,
    },
    Run {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        sanitize: bool,
        max_ticks: Option<u64>,
        run_args: Vec<String>,
    },
    EmitLlvm {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        target: Option<String>,
        board: Option<String>,
        chip: Option<String>,
        opt_level: Option<String>,
        output: Option<PathBuf>,
    },
    Build {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        board: Option<String>,
        chip: Option<String>,
        opt_level: Option<String>,
        emit_ll: bool,
        output: Option<PathBuf>,
    },
    Flash {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        board: Option<String>,
        chip: Option<String>,
        port: Option<String>,
        opt_level: Option<String>,
        output: Option<PathBuf>,
    },
    BuildElf {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        opt_level: String,
        output: Option<PathBuf>,
    },
    RunQemu {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        max_ticks: Option<u64>,
        opt_level: String,
    },
    BuildEsp32C6 {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        opt_level: String,
        output: Option<PathBuf>,
    },
    FlashEsp32C6 {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        port: String,
        opt_level: String,
        output: Option<PathBuf>,
    },
    BuildAvr {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        opt_level: String,
        output: Option<PathBuf>,
    },
    FlashAvr {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        port: String,
        opt_level: String,
        output: Option<PathBuf>,
    },
    BuildAvrNanoEvery {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        opt_level: String,
        output: Option<PathBuf>,
    },
    FlashAvrNanoEvery {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        port: String,
        opt_level: String,
        output: Option<PathBuf>,
    },
    BuildRp2040 {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        opt_level: String,
        output: Option<PathBuf>,
    },
    FlashRp2040 {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        port: String,
        opt_level: String,
        output: Option<PathBuf>,
    },
    ImportSvd {
        svd_path: PathBuf,
        output: Option<PathBuf>,
    },
    Doc {
        query: Option<String>,
        list: bool,
    },
    Help,
    Version,
}

pub const HELP_TEXT: &str = "\
zevc 0.1.0 — Zev Bare-Metal Cooperative Language-OS Compiler (LLVM/ARM/RISC-V/AVR/RP2040)

USAGE:
    zevc <COMMAND> [OPTIONS] [FILES...]

COMMANDS:
    doc [topic | code | --list]             Built-in Zev language primer, 8 topic guides & error explainers
    import-svd <chip.svd> [-o <out.zev>]    Convert a CMSIS-SVD XML description into Zev mmio/irq declarations
    check [--board <b> | --chip <c>] [--json] [-I <dir>] <files...>
                                            Parse and statically verify .zev source files (with target-bound `hal.*`)
    build [--board <b>] [--chip <c>] [--emit-ll] [OPT] [-o <out>] [files...]
                                            Build ELF/BIN/UF2/LLVM IR from TOML specs (`zev.toml` aware)
    flash [--board <b>] [--chip <c>] [--port <p>] [OPT] [-o <out>] [files...]
                                            Build and flash firmware using board TOML `flash_cmd` (`zev.toml` aware)
    ram [--board <b> | --chip <c> | --target <t>] [OPT] [-I <dir>] <files...>
                                            Print static SRAM budget (+ optional HW stack proof)
    run [-I <dir>] <files...> [OPTIONS]     Compile host LLVM 23 IR via clang and run deterministic host simulation
    emit-llvm [--board <b> | --chip <c> | --target <t>] [OPT] <files...> [-o <out.ll>]
                                            Emit LLVM 23 IR (raw by default, or optimized via opt w/ -O1..-Oz)
    build-elf / run-qemu / build-esp32c6 / flash-esp32c6 / build-avr / flash-avr /
    build-nano-every / flash-nano-every / build-rp2040 / flash-rp2040
                                            Legacy target-specific aliases for `build --board` / `flash --board`
    help                                    Show this help message

MODULE / INCLUDE OPTIONS (check, ram, run, emit-llvm, build, flash, build-*, flash-*, run-qemu):
    -I <dir> | --include <dir>              Add a module search root directory for `import` resolution (repeatable)

CHECK & TARGET OPTIONS:
    --json                                  Emit structured machine-readable JSON diagnostics (`check`)
    --board <name|path.toml>                Select board spec from `boards/<name>.toml` (or `.toml` path)
    --chip <name|path.toml>                 Select/override chip spec from `chips/<name>.toml` (or `.toml` path)
    --emit-ll                               Emit `.ll` LLVM IR instead of linking (`build` only)
    -p <dev> | --port <dev>                 Serial/flash port override (`flash` / `flash-*`, else board default)

LLVM OPTIMIZATION & TARGET OPTIONS (ram, emit-llvm, build, flash, build-*, flash-*, run-qemu):
    -O0 | -O1 | -O2 | -O3 | -Os | -Oz       LLVM 23 opt + llc optimization level (default from chip TOML or -O2)
    --opt-level <O0|O1|O2|O3|Os|Oz>         Explicit long-form optimization level selector
    --target <cortex-m3|esp32c6|avr|nano-every|rp2040|host> Legacy target alias (`ram`: MCUs; `emit-llvm`: +host)

RUN & RUN-QEMU OPTIONS:
    --sanitize                              Compile host simulation binary with -fsanitize=address,undefined (`run` only)
    --max-ticks <N>                         Bound simulation / QEMU execution to N millisecond ticks
    -- <args...>                            Forward trailing arguments to compiled host binary (`run` only)
";

pub fn normalize_opt_level(raw: &str) -> Result<String, String> {
    match raw {
        "-O0" | "O0" => Ok("-O0".to_string()),
        "-O1" | "O1" => Ok("-O1".to_string()),
        "-O2" | "O2" => Ok("-O2".to_string()),
        "-O3" | "O3" => Ok("-O3".to_string()),
        "-Os" | "Os" => Ok("-Os".to_string()),
        "-Oz" | "Oz" => Ok("-Oz".to_string()),
        _ => Err(format!(
            "invalid optimization level `{raw}` (expected `-O0`, `-O1`, `-O2`, `-O3`, `-Os`, or `-Oz`)"
        )),
    }
}

fn is_short_opt_flag(arg: &str) -> bool {
    arg == "-O" || arg.starts_with("-O")
}

pub fn parse_args<I, S>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let raw: Vec<String> = args.into_iter().map(Into::into).collect();
    if raw.is_empty() {
        return Ok(Command::Help);
    }

    match raw[0].as_str() {
        "-h" | "--help" | "help" => Ok(Command::Help),
        "-V" | "--version" | "version" => Ok(Command::Version),
        "doc" | "docs" | "explain" => parse_doc(&raw[1..]),
        "import-svd" => parse_import_svd(&raw[1..]),
        "check" => parse_check(&raw[1..]),
        "ram" => parse_ram(&raw[1..]),
        "build" => parse_build(&raw[1..]),
        "flash" => parse_flash(&raw[1..]),
        "emit-llvm" => parse_emit_llvm(&raw[1..]),
        "build-elf" => match parse_files_with_output_and_opt(&raw[1..], "build-elf")? {
            Some((files, include_dirs, opt_level, output)) => Ok(Command::BuildElf {
                files,
                include_dirs,
                opt_level,
                output,
            }),
            None => Ok(Command::Help),
        },
        "run-qemu" => parse_run_qemu(&raw[1..]),
        "build-esp32c6" => match parse_files_with_output_and_opt(&raw[1..], "build-esp32c6")? {
            Some((files, include_dirs, opt_level, output)) => Ok(Command::BuildEsp32C6 {
                files,
                include_dirs,
                opt_level,
                output,
            }),
            None => Ok(Command::Help),
        },
        "flash-esp32c6" => match parse_flash_serial(&raw[1..], "flash-esp32c6", "-O2")? {
            Some((files, include_dirs, port, opt_level, output)) => Ok(Command::FlashEsp32C6 {
                files,
                include_dirs,
                port,
                opt_level,
                output,
            }),
            None => Ok(Command::Help),
        },
        "build-avr" | "build-uno" => match parse_files_with_output_and_opt_default(&raw[1..], &raw[0], "-Os")? {
            Some((files, include_dirs, opt_level, output)) => Ok(Command::BuildAvr {
                files,
                include_dirs,
                opt_level,
                output,
            }),
            None => Ok(Command::Help),
        },
        "flash-avr" | "flash-uno" => match parse_flash_serial(&raw[1..], &raw[0], "-Os")? {
            Some((files, include_dirs, port, opt_level, output)) => Ok(Command::FlashAvr {
                files,
                include_dirs,
                port,
                opt_level,
                output,
            }),
            None => Ok(Command::Help),
        },
        "build-nano-every" | "build-atmega4809" => {
            match parse_files_with_output_and_opt_default(&raw[1..], &raw[0], "-Os")? {
                Some((files, include_dirs, opt_level, output)) => Ok(Command::BuildAvrNanoEvery {
                    files,
                    include_dirs,
                    opt_level,
                    output,
                }),
                None => Ok(Command::Help),
            }
        }
        "flash-nano-every" | "flash-atmega4809" => {
            match parse_flash_serial(&raw[1..], &raw[0], "-Os")? {
                Some((files, include_dirs, port, opt_level, output)) => {
                    Ok(Command::FlashAvrNanoEvery {
                        files,
                        include_dirs,
                        port,
                        opt_level,
                        output,
                    })
                }
                None => Ok(Command::Help),
            }
        }
        "build-rp2040" | "build-pico" => {
            match parse_files_with_output_and_opt_default(&raw[1..], &raw[0], "-O2")? {
                Some((files, include_dirs, opt_level, output)) => Ok(Command::BuildRp2040 {
                    files,
                    include_dirs,
                    opt_level,
                    output,
                }),
                None => Ok(Command::Help),
            }
        }
        "flash-rp2040" | "flash-pico" => match parse_flash_serial(&raw[1..], &raw[0], "-O2")? {
            Some((files, include_dirs, port, opt_level, output)) => Ok(Command::FlashRp2040 {
                files,
                include_dirs,
                port,
                opt_level,
                output,
            }),
            None => Ok(Command::Help),
        },
        "run" => parse_run(&raw[1..]),
        other => Err(format!(
            "unknown subcommand `{other}` (expected `doc`, `import-svd`, `check`, `ram`, `run`, `emit-llvm`, `build`, `flash`, `build-elf`, `run-qemu`, `build-esp32c6`, `flash-esp32c6`, `build-avr`, `flash-avr`, `build-nano-every`, `flash-nano-every`, `build-rp2040`, or `flash-rp2040`; try `zevc --help`)"
        )),
    }
}

fn try_parse_include_arg(
    args: &[String],
    i: &mut usize,
    include_dirs: &mut Vec<PathBuf>,
    subcmd: &str,
) -> Result<bool, String> {
    let arg = &args[*i];
    if arg == "-I" || arg == "--include" {
        *i += 1;
        if *i >= args.len() {
            return Err(format!(
                "`{arg}` requires an include directory path for `zevc {subcmd}`"
            ));
        }
        if args[*i].is_empty() {
            return Err(format!(
                "`{arg}` requires a non-empty include directory path for `zevc {subcmd}`"
            ));
        }
        include_dirs.push(PathBuf::from(&args[*i]));
        return Ok(true);
    }
    if let Some(dir) = arg.strip_prefix("--include=") {
        if dir.is_empty() {
            return Err(format!(
                "`--include=` requires a non-empty include directory path for `zevc {subcmd}`"
            ));
        }
        include_dirs.push(PathBuf::from(dir));
        return Ok(true);
    }
    if let Some(dir) = arg.strip_prefix("-I=") {
        if dir.is_empty() {
            return Err(format!(
                "`-I=` requires a non-empty include directory path for `zevc {subcmd}`"
            ));
        }
        include_dirs.push(PathBuf::from(dir));
        return Ok(true);
    }
    if let Some(dir) = arg.strip_prefix("-I") {
        include_dirs.push(PathBuf::from(dir));
        return Ok(true);
    }
    Ok(false)
}

fn parse_import_svd(args: &[String]) -> Result<Command, String> {
    let mut svd_path: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() || args[i].is_empty() {
                    return Err("`-o` requires an output `.zev` path for `zevc import-svd`".to_string());
                }
                output = Some(PathBuf::from(&args[i]));
            }
            _ if arg.starts_with("-o=") => {
                let val = &arg[3..];
                if val.is_empty() {
                    return Err("`-o=` requires a non-empty output `.zev` path for `zevc import-svd`".to_string());
                }
                output = Some(PathBuf::from(val));
            }
            _ if arg.starts_with("--output=") => {
                let val = &arg[9..];
                if val.is_empty() {
                    return Err("`--output=` requires a non-empty output `.zev` path for `zevc import-svd`".to_string());
                }
                output = Some(PathBuf::from(val));
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc import-svd`"));
            }
            _ => {
                if svd_path.is_some() {
                    return Err("`zevc import-svd` accepts exactly one `.svd` input file".to_string());
                }
                svd_path = Some(PathBuf::from(arg));
            }
        }
        i += 1;
    }
    let Some(svd_path) = svd_path else {
        return Err("`zevc import-svd` requires an input `.svd` file path".to_string());
    };
    Ok(Command::ImportSvd { svd_path, output })
}

fn parse_doc(args: &[String]) -> Result<Command, String> {
    let mut query: Option<String> = None;
    let mut list = false;
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-l" | "--list" => list = true,
            _ if arg.starts_with('-') => {
                return Err(format!(
                    "unexpected option `{arg}` for `zevc doc` (expected `zevc doc`, `zevc doc --list`, or `zevc doc <topic|code>`)"
                ));
            }
            _ => {
                if query.is_some() {
                    return Err(
                        "`zevc doc` accepts at most one topic or error code query".to_string(),
                    );
                }
                query = Some(arg.clone());
            }
        }
    }
    Ok(Command::Doc { query, list })
}

fn parse_check(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut board: Option<String> = None;
    let mut chip: Option<String> = None;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "check")?
            || try_parse_board_or_chip_arg(args, &mut i, &mut board, &mut chip, "check")?
        {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--json" => json = true,
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc check`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if files.is_empty() {
        return Err("`zevc check` requires at least one `.zev` input file".to_string());
    }
    Ok(Command::Check {
        files,
        include_dirs,
        board,
        chip,
        json,
    })
}

fn try_parse_board_or_chip_arg(
    args: &[String],
    i: &mut usize,
    board: &mut Option<String>,
    chip: &mut Option<String>,
    subcmd: &str,
) -> Result<bool, String> {
    let arg = &args[*i];
    if arg == "--board" {
        *i += 1;
        if *i >= args.len() || args[*i].is_empty() {
            return Err(format!("`--board` requires a board name or `.toml` path for `zevc {subcmd}`"));
        }
        *board = Some(args[*i].clone());
        return Ok(true);
    }
    if let Some(val) = arg.strip_prefix("--board=") {
        if val.is_empty() {
            return Err(format!("`--board=` requires a board name or `.toml` path for `zevc {subcmd}`"));
        }
        *board = Some(val.to_string());
        return Ok(true);
    }
    if arg == "--chip" {
        *i += 1;
        if *i >= args.len() || args[*i].is_empty() {
            return Err(format!("`--chip` requires a chip name or `.toml` path for `zevc {subcmd}`"));
        }
        *chip = Some(args[*i].clone());
        return Ok(true);
    }
    if let Some(val) = arg.strip_prefix("--chip=") {
        if val.is_empty() {
            return Err(format!("`--chip=` requires a chip name or `.toml` path for `zevc {subcmd}`"));
        }
        *chip = Some(val.to_string());
        return Ok(true);
    }
    Ok(false)
}

fn parse_ram(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut target: Option<String> = None;
    let mut board: Option<String> = None;
    let mut chip: Option<String> = None;
    let mut opt_level: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "ram")?
            || try_parse_board_or_chip_arg(args, &mut i, &mut board, &mut chip, "ram")?
        {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--target" => {
                i += 1;
                if i >= args.len() {
                    return Err(
                        "`--target` requires a target name (`cortex-m3`, `esp32c6`, `avr`, `nano-every`, or `rp2040`)"
                            .to_string(),
                    );
                }
                target = Some(args[i].clone());
            }
            "--opt-level" => {
                i += 1;
                if i >= args.len() {
                    return Err(
                        "`--opt-level` requires a level (`O0`, `O1`, `O2`, `O3`, `Os`, or `Oz`) for `zevc ram`"
                            .to_string(),
                    );
                }
                opt_level = Some(normalize_opt_level(&args[i])?);
            }
            _ if arg.starts_with("--target=") => {
                target = Some(arg["--target=".len()..].to_string());
            }
            _ if arg.starts_with("--opt-level=") => {
                opt_level = Some(normalize_opt_level(&arg["--opt-level=".len()..])?);
            }
            _ if is_short_opt_flag(arg) => {
                opt_level = Some(normalize_opt_level(arg)?);
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc ram`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if files.is_empty() && board.is_none() && chip.is_none() {
        return Err("`zevc ram` requires at least one `.zev` input file".to_string());
    }
    Ok(Command::Ram {
        files,
        include_dirs,
        target,
        board,
        chip,
        opt_level,
    })
}

fn parse_files_with_output_and_opt(
    args: &[String],
    subcmd: &str,
) -> Result<Option<(Vec<PathBuf>, Vec<PathBuf>, String, Option<PathBuf>)>, String> {
    parse_files_with_output_and_opt_default(args, subcmd, "-O2")
}

fn parse_files_with_output_and_opt_default(
    args: &[String],
    subcmd: &str,
    default_opt: &str,
) -> Result<Option<(Vec<PathBuf>, Vec<PathBuf>, String, Option<PathBuf>)>, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut opt_level = default_opt.to_string();
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, subcmd)? {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("`-o` requires an output file path for `zevc {subcmd}`"));
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "--opt-level" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!(
                        "`--opt-level` requires a level (`O0`, `O1`, `O2`, `O3`, `Os`, or `Oz`) for `zevc {subcmd}`"
                    ));
                }
                opt_level = normalize_opt_level(&args[i])?;
            }
            _ if arg.starts_with("-o=") => {
                output = Some(PathBuf::from(&arg[3..]));
            }
            _ if arg.starts_with("--output=") => {
                output = Some(PathBuf::from(&arg[9..]));
            }
            _ if arg.starts_with("--opt-level=") => {
                opt_level = normalize_opt_level(&arg["--opt-level=".len()..])?;
            }
            _ if is_short_opt_flag(arg) => {
                opt_level = normalize_opt_level(arg)?;
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc {subcmd}`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if files.is_empty() {
        return Err(format!(
            "`zevc {subcmd}` requires at least one `.zev` input file"
        ));
    }
    Ok(Some((files, include_dirs, opt_level, output)))
}

fn parse_emit_llvm(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut target: Option<String> = None;
    let mut board: Option<String> = None;
    let mut chip: Option<String> = None;
    let mut opt_level: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "emit-llvm")?
            || try_parse_board_or_chip_arg(args, &mut i, &mut board, &mut chip, "emit-llvm")?
        {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err("`-o` requires an output file path for `zevc emit-llvm`".to_string());
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "--target" => {
                i += 1;
                if i >= args.len() {
                    return Err("`--target` requires a target name (`cortex-m3`, `esp32c6`, `avr`, `nano-every`, `rp2040`, or `host`)".to_string());
                }
                target = Some(args[i].clone());
            }
            "--opt-level" => {
                i += 1;
                if i >= args.len() {
                    return Err(
                        "`--opt-level` requires a level (`O0`, `O1`, `O2`, `O3`, `Os`, or `Oz`) for `zevc emit-llvm`"
                            .to_string(),
                    );
                }
                opt_level = Some(normalize_opt_level(&args[i])?);
            }
            _ if arg.starts_with("-o=") => {
                output = Some(PathBuf::from(&arg[3..]));
            }
            _ if arg.starts_with("--output=") => {
                output = Some(PathBuf::from(&arg[9..]));
            }
            _ if arg.starts_with("--target=") => {
                target = Some(arg["--target=".len()..].to_string());
            }
            _ if arg.starts_with("--opt-level=") => {
                opt_level = Some(normalize_opt_level(&arg["--opt-level=".len()..])?);
            }
            _ if is_short_opt_flag(arg) => {
                opt_level = Some(normalize_opt_level(arg)?);
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc emit-llvm`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if files.is_empty() && board.is_none() && chip.is_none() {
        return Err("`zevc emit-llvm` requires at least one `.zev` input file".to_string());
    }
    Ok(Command::EmitLlvm {
        files,
        include_dirs,
        target,
        board,
        chip,
        opt_level,
        output,
    })
}

fn parse_build(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut board: Option<String> = None;
    let mut chip: Option<String> = None;
    let mut opt_level: Option<String> = None;
    let mut emit_ll = false;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "build")?
            || try_parse_board_or_chip_arg(args, &mut i, &mut board, &mut chip, "build")?
        {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--emit-ll" => emit_ll = true,
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() || args[i].is_empty() {
                    return Err("`-o` requires an output path for `zevc build`".to_string());
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "--opt-level" => {
                i += 1;
                if i >= args.len() {
                    return Err("`--opt-level` requires `O0..Oz` for `zevc build`".to_string());
                }
                opt_level = Some(normalize_opt_level(&args[i])?);
            }
            _ if arg.starts_with("-o=") => {
                output = Some(PathBuf::from(&arg[3..]));
            }
            _ if arg.starts_with("--output=") => {
                output = Some(PathBuf::from(&arg[9..]));
            }
            _ if arg.starts_with("--opt-level=") => {
                opt_level = Some(normalize_opt_level(&arg["--opt-level=".len()..])?);
            }
            _ if is_short_opt_flag(arg) => {
                opt_level = Some(normalize_opt_level(arg)?);
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc build`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    Ok(Command::Build {
        files,
        include_dirs,
        board,
        chip,
        opt_level,
        emit_ll,
        output,
    })
}

fn parse_flash(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut board: Option<String> = None;
    let mut chip: Option<String> = None;
    let mut port: Option<String> = None;
    let mut opt_level: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "flash")?
            || try_parse_board_or_chip_arg(args, &mut i, &mut board, &mut chip, "flash")?
        {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-p" | "--port" => {
                i += 1;
                if i >= args.len() || args[i].is_empty() {
                    return Err("`--port` requires a serial device or mount path for `zevc flash`".to_string());
                }
                port = Some(args[i].clone());
            }
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() || args[i].is_empty() {
                    return Err("`-o` requires an output path for `zevc flash`".to_string());
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "--opt-level" => {
                i += 1;
                if i >= args.len() {
                    return Err("`--opt-level` requires `O0..Oz` for `zevc flash`".to_string());
                }
                opt_level = Some(normalize_opt_level(&args[i])?);
            }
            _ if arg.starts_with("--port=") => {
                port = Some(arg["--port=".len()..].to_string());
            }
            _ if arg.starts_with("-o=") => {
                output = Some(PathBuf::from(&arg[3..]));
            }
            _ if arg.starts_with("--output=") => {
                output = Some(PathBuf::from(&arg[9..]));
            }
            _ if arg.starts_with("--opt-level=") => {
                opt_level = Some(normalize_opt_level(&arg["--opt-level=".len()..])?);
            }
            _ if is_short_opt_flag(arg) => {
                opt_level = Some(normalize_opt_level(arg)?);
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc flash`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    Ok(Command::Flash {
        files,
        include_dirs,
        board,
        chip,
        port,
        opt_level,
        output,
    })
}

fn parse_run_qemu(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut max_ticks: Option<u64> = None;
    let mut opt_level = "-O2".to_string();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "run-qemu")? {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--max-ticks" => {
                i += 1;
                if i >= args.len() {
                    return Err(
                        "`--max-ticks` requires a positive integer for `zevc run-qemu`".to_string(),
                    );
                }
                let n = args[i]
                    .parse::<u64>()
                    .map_err(|_| format!("invalid `--max-ticks` value `{}`", args[i]))?;
                max_ticks = Some(n);
            }
            "--opt-level" => {
                i += 1;
                if i >= args.len() {
                    return Err(
                        "`--opt-level` requires a level (`O0`, `O1`, `O2`, `O3`, `Os`, or `Oz`) for `zevc run-qemu`"
                            .to_string(),
                    );
                }
                opt_level = normalize_opt_level(&args[i])?;
            }
            _ if arg.starts_with("--max-ticks=") => {
                let val = &arg["--max-ticks=".len()..];
                let n = val
                    .parse::<u64>()
                    .map_err(|_| format!("invalid `--max-ticks` value `{val}`"))?;
                max_ticks = Some(n);
            }
            _ if arg.starts_with("--opt-level=") => {
                opt_level = normalize_opt_level(&arg["--opt-level=".len()..])?;
            }
            _ if is_short_opt_flag(arg) => {
                opt_level = normalize_opt_level(arg)?;
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc run-qemu`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if files.is_empty() {
        return Err("`zevc run-qemu` requires at least one `.zev` input file".to_string());
    }
    Ok(Command::RunQemu {
        files,
        include_dirs,
        max_ticks,
        opt_level,
    })
}

type FlashArgs = (Vec<PathBuf>, Vec<PathBuf>, String, String, Option<PathBuf>);

fn parse_flash_serial(
    args: &[String],
    subcmd: &str,
    default_opt: &str,
) -> Result<Option<FlashArgs>, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut port = "/dev/ttyACM0".to_string();
    let mut opt_level = default_opt.to_string();
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, subcmd)? {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "-p" | "--port" => {
                i += 1;
                if i >= args.len() {
                    return Err("`--port` requires a serial device path (e.g. `/dev/ttyACM0`)".to_string());
                }
                port = args[i].clone();
            }
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!("`-o` requires an output `.bin` path for `zevc {subcmd}`"));
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "--opt-level" => {
                i += 1;
                if i >= args.len() {
                    return Err(format!(
                        "`--opt-level` requires a level (`O0`, `O1`, `O2`, `O3`, `Os`, or `Oz`) for `zevc {subcmd}`"
                    ));
                }
                opt_level = normalize_opt_level(&args[i])?;
            }
            _ if arg.starts_with("--port=") => {
                port = arg["--port=".len()..].to_string();
            }
            _ if arg.starts_with("-o=") => {
                output = Some(PathBuf::from(&arg[3..]));
            }
            _ if arg.starts_with("--output=") => {
                output = Some(PathBuf::from(&arg[9..]));
            }
            _ if arg.starts_with("--opt-level=") => {
                opt_level = normalize_opt_level(&arg["--opt-level=".len()..])?;
            }
            _ if is_short_opt_flag(arg) => {
                opt_level = normalize_opt_level(arg)?;
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc {subcmd}`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if files.is_empty() {
        return Err(format!(
            "`zevc {subcmd}` requires at least one `.zev` input file"
        ));
    }
    Ok(Some((files, include_dirs, port, opt_level, output)))
}

fn parse_run(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut sanitize = false;
    let mut max_ticks = None;
    let mut run_args = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--" {
            run_args.extend_from_slice(&args[i + 1..]);
            break;
        }
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "run")? {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--sanitize" => sanitize = true,
            "--max-ticks" => {
                i += 1;
                if i >= args.len() {
                    return Err("`--max-ticks` requires a positive integer".to_string());
                }
                let n = args[i]
                    .parse::<u64>()
                    .map_err(|_| format!("invalid `--max-ticks` value `{}`", args[i]))?;
                max_ticks = Some(n);
            }
            _ if arg.starts_with("--max-ticks=") => {
                let val = &arg["--max-ticks=".len()..];
                let n = val
                    .parse::<u64>()
                    .map_err(|_| format!("invalid `--max-ticks` value `{val}`"))?;
                max_ticks = Some(n);
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc run`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if files.is_empty() {
        return Err("`zevc run` requires at least one `.zev` input file".to_string());
    }
    Ok(Command::Run {
        files,
        include_dirs,
        sanitize,
        max_ticks,
        run_args,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_doc_and_check_json() {
        assert_eq!(
            parse_args(["doc"]).unwrap(),
            Command::Doc {
                query: None,
                list: false
            }
        );
        assert_eq!(
            parse_args(["doc", "--list"]).unwrap(),
            Command::Doc {
                query: None,
                list: true
            }
        );
        assert_eq!(
            parse_args(["doc", "runtime"]).unwrap(),
            Command::Doc {
                query: Some("runtime".to_string()),
                list: false
            }
        );
        assert_eq!(
            parse_args(["check", "--json", "a.zev", "b.zev"]).unwrap(),
            Command::Check {
                files: vec![PathBuf::from("a.zev"), PathBuf::from("b.zev")],
                include_dirs: vec![],
                board: None,
                chip: None,
                json: true,
            }
        );
        assert_eq!(
            parse_args(["check", "--board", "nucleo_f334r8", "a.zev"]).unwrap(),
            Command::Check {
                files: vec![PathBuf::from("a.zev")],
                include_dirs: vec![],
                board: Some("nucleo_f334r8".to_string()),
                chip: None,
                json: false,
            }
        );
        assert_eq!(
            parse_args(["check", "--chip=stm32f334r8", "a.zev"]).unwrap(),
            Command::Check {
                files: vec![PathBuf::from("a.zev")],
                include_dirs: vec![],
                board: None,
                chip: Some("stm32f334r8".to_string()),
                json: false,
            }
        );
    }

    #[test]
    fn parses_ram_multiple_files() {
        let cmd =
            parse_args(["ram", "examples/motor_controller.zev", "examples/motor_support.zev"])
                .unwrap();
        assert_eq!(
            cmd,
            Command::Ram {
                files: vec![
                    PathBuf::from("examples/motor_controller.zev"),
                    PathBuf::from("examples/motor_support.zev"),
                ],
                include_dirs: vec![],
                target: None,
                board: None,
                chip: None,
                opt_level: None,
            }
        );

        assert_eq!(
            parse_args([
                "ram",
                "--target",
                "cortex-m3",
                "-O2",
                "examples/cortex_m_demo.zev",
            ])
            .unwrap(),
            Command::Ram {
                files: vec![PathBuf::from("examples/cortex_m_demo.zev")],
                include_dirs: vec![],
                target: Some("cortex-m3".to_string()),
                board: None,
                chip: None,
                opt_level: Some("-O2".to_string()),
            }
        );

        assert_eq!(
            parse_args([
                "ram",
                "--target=esp32c6",
                "--opt-level=Oz",
                "-I",
                "libs",
                "examples/esp32c6_demo.zev",
            ])
            .unwrap(),
            Command::Ram {
                files: vec![PathBuf::from("examples/esp32c6_demo.zev")],
                include_dirs: vec![PathBuf::from("libs")],
                target: Some("esp32c6".to_string()),
                board: None,
                chip: None,
                opt_level: Some("-Oz".to_string()),
            }
        );

        assert_eq!(
            parse_args(["ram", "--target", "avr", "-Os", "examples/arduino_uno_demo.zev"]).unwrap(),
            Command::Ram {
                files: vec![PathBuf::from("examples/arduino_uno_demo.zev")],
                include_dirs: vec![],
                target: Some("avr".to_string()),
                board: None,
                chip: None,
                opt_level: Some("-Os".to_string()),
            }
        );

        assert_eq!(
            parse_args([
                "ram",
                "--target",
                "nano-every",
                "-Os",
                "examples/arduino_nano_every_demo.zev"
            ])
            .unwrap(),
            Command::Ram {
                files: vec![PathBuf::from("examples/arduino_nano_every_demo.zev")],
                include_dirs: vec![],
                target: Some("nano-every".to_string()),
                board: None,
                chip: None,
                opt_level: Some("-Os".to_string()),
            }
        );

        assert!(parse_args(["ram", "--target"]).is_err());
        assert!(parse_args(["ram", "--opt-level"]).is_err());
        assert!(parse_args(["ram", "-O4", "a.zev"]).is_err());
    }

    #[test]
    fn parses_llvm_and_elf_and_esp32c6() {
        let cmd_ll_raw =
            parse_args(["emit-llvm", "examples/esp32c6_demo.zev", "-o", "out.ll"]).unwrap();
        assert_eq!(
            cmd_ll_raw,
            Command::EmitLlvm {
                files: vec![PathBuf::from("examples/esp32c6_demo.zev")],
                include_dirs: vec![],
                target: None,
                board: None,
                chip: None,
                opt_level: None,
                output: Some(PathBuf::from("out.ll")),
            }
        );

        let cmd_ll_opt = parse_args([
            "emit-llvm",
            "--target",
            "esp32c6",
            "-Oz",
            "examples/esp32c6_demo.zev",
            "-o",
            "out.ll",
        ])
        .unwrap();
        assert_eq!(
            cmd_ll_opt,
            Command::EmitLlvm {
                files: vec![PathBuf::from("examples/esp32c6_demo.zev")],
                include_dirs: vec![],
                target: Some("esp32c6".to_string()),
                board: None,
                chip: None,
                opt_level: Some("-Oz".to_string()),
                output: Some(PathBuf::from("out.ll")),
            }
        );

        let cmd_elf =
            parse_args(["build-elf", "-Os", "-o", "fw.elf", "examples/cortex_m_demo.zev"]).unwrap();
        assert_eq!(
            cmd_elf,
            Command::BuildElf {
                files: vec![PathBuf::from("examples/cortex_m_demo.zev")],
                include_dirs: vec![],
                opt_level: "-Os".to_string(),
                output: Some(PathBuf::from("fw.elf")),
            }
        );

        let cmd_qemu = parse_args([
            "run-qemu",
            "--max-ticks",
            "50",
            "--opt-level",
            "Oz",
            "examples/cortex_m_demo.zev",
        ])
        .unwrap();
        assert_eq!(
            cmd_qemu,
            Command::RunQemu {
                files: vec![PathBuf::from("examples/cortex_m_demo.zev")],
                include_dirs: vec![],
                max_ticks: Some(50),
                opt_level: "-Oz".to_string(),
            }
        );

        let cmd_esp_build = parse_args([
            "build-esp32c6",
            "--opt-level=O2",
            "examples/esp32c6_demo.zev",
            "-o",
            "/tmp/esp32c6_demo.bin",
        ])
        .unwrap();
        assert_eq!(
            cmd_esp_build,
            Command::BuildEsp32C6 {
                files: vec![PathBuf::from("examples/esp32c6_demo.zev")],
                include_dirs: vec![],
                opt_level: "-O2".to_string(),
                output: Some(PathBuf::from("/tmp/esp32c6_demo.bin")),
            }
        );

        let cmd_esp_flash = parse_args([
            "flash-esp32c6",
            "-Oz",
            "examples/esp32c6_demo.zev",
            "--port",
            "/dev/ttyACM0",
        ])
        .unwrap();
        assert_eq!(
            cmd_esp_flash,
            Command::FlashEsp32C6 {
                files: vec![PathBuf::from("examples/esp32c6_demo.zev")],
                include_dirs: vec![],
                port: "/dev/ttyACM0".to_string(),
                opt_level: "-Oz".to_string(),
                output: None,
            }
        );
    }

    #[test]
    fn parses_all_opt_levels_and_rejects_invalid_opt_flags() {
        for (flag, norm) in [
            ("-O0", "-O0"),
            ("-O1", "-O1"),
            ("-O2", "-O2"),
            ("-O3", "-O3"),
            ("-Os", "-Os"),
            ("-Oz", "-Oz"),
        ] {
            let cmd = parse_args(["build-elf", flag, "a.zev"]).unwrap();
            assert_eq!(
                cmd,
                Command::BuildElf {
                    files: vec![PathBuf::from("a.zev")],
                    include_dirs: vec![],
                    opt_level: norm.to_string(),
                    output: None,
                }
            );
        }

        for bad in ["-O", "-O4", "-Ofast", "-Og"] {
            assert!(parse_args(["build-elf", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["run-qemu", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["emit-llvm", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["build-esp32c6", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["flash-esp32c6", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["build-avr", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["build-uno", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["flash-avr", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["flash-uno", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["build-nano-every", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["build-atmega4809", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["flash-nano-every", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["flash-atmega4809", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["build-rp2040", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["build-pico", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["flash-rp2040", bad, "a.zev"]).is_err(), "expected err on {bad}");
            assert!(parse_args(["flash-pico", bad, "a.zev"]).is_err(), "expected err on {bad}");
        }
        assert!(parse_args(["build-elf", "--opt-level", "O9", "a.zev"]).is_err());
        assert!(parse_args(["build-elf", "--opt-level=fast", "a.zev"]).is_err());
        assert!(parse_args(["build-elf", "--opt-level"]).is_err());
    }

    #[test]
    fn parses_run_with_options() {
        let cmd = parse_args([
            "run",
            "--sanitize",
            "--max-ticks",
            "500",
            "examples/sim_demo.zev",
            "--",
            "--trace",
        ])
        .unwrap();
        assert_eq!(
            cmd,
            Command::Run {
                files: vec![PathBuf::from("examples/sim_demo.zev")],
                include_dirs: vec![],
                sanitize: true,
                max_ticks: Some(500),
                run_args: vec!["--trace".to_string()],
            }
        );
    }

    #[test]
    fn parses_import_svd_subcommand() {
        assert_eq!(
            parse_args(["import-svd", "stm32g4.svd"]).unwrap(),
            Command::ImportSvd {
                svd_path: PathBuf::from("stm32g4.svd"),
                output: None,
            }
        );
        assert_eq!(
            parse_args(["import-svd", "stm32g4.svd", "-o", "chip/stm32g4.zev"]).unwrap(),
            Command::ImportSvd {
                svd_path: PathBuf::from("stm32g4.svd"),
                output: Some(PathBuf::from("chip/stm32g4.zev")),
            }
        );
        assert_eq!(
            parse_args(["import-svd", "--output=out.zev", "chip.svd"]).unwrap(),
            Command::ImportSvd {
                svd_path: PathBuf::from("chip.svd"),
                output: Some(PathBuf::from("out.zev")),
            }
        );
        assert_eq!(
            parse_args(["import-svd", "-o=out.zev", "chip.svd"]).unwrap(),
            Command::ImportSvd {
                svd_path: PathBuf::from("chip.svd"),
                output: Some(PathBuf::from("out.zev")),
            }
        );
        assert_eq!(parse_args(["import-svd", "--help"]).unwrap(), Command::Help);

        assert!(parse_args(["import-svd"]).is_err());
        assert!(parse_args(["import-svd", "a.svd", "b.svd"]).is_err());
        assert!(parse_args(["import-svd", "a.svd", "-o"]).is_err());
        assert!(parse_args(["import-svd", "a.svd", "-o="]).is_err());
        assert!(parse_args(["import-svd", "a.svd", "--output="]).is_err());
        assert!(parse_args(["import-svd", "a.svd", "--unknown"]).is_err());
    }

    #[test]
    fn parses_include_dirs_across_all_file_commands() {
        let expected_dirs = vec![
            PathBuf::from("libs"),
            PathBuf::from("chips"),
            PathBuf::from("bsp"),
            PathBuf::from("hal"),
            PathBuf::from("extra"),
        ];
        let include_flags = [
            "-I",
            "libs",
            "--include",
            "chips",
            "--include=bsp",
            "-I=hal",
            "-Iextra",
        ];

        let mut check_args = vec!["check", "--json"];
        check_args.extend_from_slice(&include_flags);
        check_args.push("app.zev");
        assert_eq!(
            parse_args(check_args).unwrap(),
            Command::Check {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: expected_dirs.clone(),
                board: None,
                chip: None,
                json: true,
            }
        );

        let mut ram_args = vec!["ram"];
        ram_args.extend_from_slice(&include_flags);
        ram_args.push("app.zev");
        assert_eq!(
            parse_args(ram_args).unwrap(),
            Command::Ram {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: expected_dirs.clone(),
                target: None,
                board: None,
                chip: None,
                opt_level: None,
            }
        );

        let mut run_args = vec!["run", "--sanitize"];
        run_args.extend_from_slice(&include_flags);
        run_args.extend_from_slice(&["app.zev", "--", "-I", "user_arg"]);
        assert_eq!(
            parse_args(run_args).unwrap(),
            Command::Run {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: expected_dirs.clone(),
                sanitize: true,
                max_ticks: None,
                run_args: vec!["-I".to_string(), "user_arg".to_string()],
            }
        );

        let mut llvm_args = vec!["emit-llvm"];
        llvm_args.extend_from_slice(&include_flags);
        llvm_args.push("app.zev");
        assert_eq!(
            parse_args(llvm_args).unwrap(),
            Command::EmitLlvm {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: expected_dirs.clone(),
                target: None,
                board: None,
                chip: None,
                opt_level: None,
                output: None,
            }
        );

        let mut elf_args = vec!["build-elf"];
        elf_args.extend_from_slice(&include_flags);
        elf_args.push("app.zev");
        assert_eq!(
            parse_args(elf_args).unwrap(),
            Command::BuildElf {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: expected_dirs.clone(),
                opt_level: "-O2".to_string(),
                output: None,
            }
        );

        let mut qemu_args = vec!["run-qemu"];
        qemu_args.extend_from_slice(&include_flags);
        qemu_args.push("app.zev");
        assert_eq!(
            parse_args(qemu_args).unwrap(),
            Command::RunQemu {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: expected_dirs.clone(),
                max_ticks: None,
                opt_level: "-O2".to_string(),
            }
        );

        let mut esp_build_args = vec!["build-esp32c6"];
        esp_build_args.extend_from_slice(&include_flags);
        esp_build_args.push("app.zev");
        assert_eq!(
            parse_args(esp_build_args).unwrap(),
            Command::BuildEsp32C6 {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: expected_dirs.clone(),
                opt_level: "-O2".to_string(),
                output: None,
            }
        );

        let mut esp_flash_args = vec!["flash-esp32c6"];
        esp_flash_args.extend_from_slice(&include_flags);
        esp_flash_args.push("app.zev");
        assert_eq!(
            parse_args(esp_flash_args).unwrap(),
            Command::FlashEsp32C6 {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: expected_dirs.clone(),
                port: "/dev/ttyACM0".to_string(),
                opt_level: "-O2".to_string(),
                output: None,
            }
        );

        for alias in ["build-avr", "build-uno"] {
            let mut avr_build_args = vec![alias, "-Os", "-o", "uno.bin"];
            avr_build_args.extend_from_slice(&include_flags);
            avr_build_args.push("app.zev");
            assert_eq!(
                parse_args(avr_build_args).unwrap(),
                Command::BuildAvr {
                    files: vec![PathBuf::from("app.zev")],
                    include_dirs: expected_dirs.clone(),
                    opt_level: "-Os".to_string(),
                    output: Some(PathBuf::from("uno.bin")),
                }
            );
        }

        for alias in ["flash-avr", "flash-uno"] {
            let mut avr_flash_args = vec![alias, "--port", "/dev/ttyACM0", "-Oz", "-o", "uno.bin"];
            avr_flash_args.extend_from_slice(&include_flags);
            avr_flash_args.push("app.zev");
            assert_eq!(
                parse_args(avr_flash_args).unwrap(),
                Command::FlashAvr {
                    files: vec![PathBuf::from("app.zev")],
                    include_dirs: expected_dirs.clone(),
                    port: "/dev/ttyACM0".to_string(),
                    opt_level: "-Oz".to_string(),
                    output: Some(PathBuf::from("uno.bin")),
                }
            );
        }

        for alias in ["build-nano-every", "build-atmega4809"] {
            let mut nano_build_args = vec![alias, "-o", "nano.bin"];
            nano_build_args.extend_from_slice(&include_flags);
            nano_build_args.push("app.zev");
            assert_eq!(
                parse_args(nano_build_args).unwrap(),
                Command::BuildAvrNanoEvery {
                    files: vec![PathBuf::from("app.zev")],
                    include_dirs: expected_dirs.clone(),
                    opt_level: "-Os".to_string(),
                    output: Some(PathBuf::from("nano.bin")),
                }
            );
        }

        for alias in ["flash-nano-every", "flash-atmega4809"] {
            let mut nano_flash_args = vec![alias, "--port", "/dev/ttyACM0", "-Oz", "-o", "nano.bin"];
            nano_flash_args.extend_from_slice(&include_flags);
            nano_flash_args.push("app.zev");
            assert_eq!(
                parse_args(nano_flash_args).unwrap(),
                Command::FlashAvrNanoEvery {
                    files: vec![PathBuf::from("app.zev")],
                    include_dirs: expected_dirs.clone(),
                    port: "/dev/ttyACM0".to_string(),
                    opt_level: "-Oz".to_string(),
                    output: Some(PathBuf::from("nano.bin")),
                }
            );
        }

        for alias in ["build-rp2040", "build-pico"] {
            let mut pico_build_args = vec![alias, "-O2", "-o", "pico.uf2"];
            pico_build_args.extend_from_slice(&include_flags);
            pico_build_args.push("app.zev");
            assert_eq!(
                parse_args(pico_build_args).unwrap(),
                Command::BuildRp2040 {
                    files: vec![PathBuf::from("app.zev")],
                    include_dirs: expected_dirs.clone(),
                    opt_level: "-O2".to_string(),
                    output: Some(PathBuf::from("pico.uf2")),
                }
            );
        }

        for alias in ["flash-rp2040", "flash-pico"] {
            let mut pico_flash_args = vec![alias, "--port", "/dev/ttyACM2", "-Oz", "-o", "pico.uf2"];
            pico_flash_args.extend_from_slice(&include_flags);
            pico_flash_args.push("app.zev");
            assert_eq!(
                parse_args(pico_flash_args).unwrap(),
                Command::FlashRp2040 {
                    files: vec![PathBuf::from("app.zev")],
                    include_dirs: expected_dirs.clone(),
                    port: "/dev/ttyACM2".to_string(),
                    opt_level: "-Oz".to_string(),
                    output: Some(PathBuf::from("pico.uf2")),
                }
            );
        }

        assert_eq!(
            parse_args(["emit-llvm", "--target", "avr", "app.zev"]).unwrap(),
            Command::EmitLlvm {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: Vec::new(),
                target: Some("avr".to_string()),
                board: None,
                chip: None,
                opt_level: None,
                output: None,
            }
        );

        assert_eq!(
            parse_args(["emit-llvm", "--target", "nano-every", "app.zev"]).unwrap(),
            Command::EmitLlvm {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: Vec::new(),
                target: Some("nano-every".to_string()),
                board: None,
                chip: None,
                opt_level: None,
                output: None,
            }
        );

        assert_eq!(
            parse_args(["emit-llvm", "--target", "rp2040", "app.zev"]).unwrap(),
            Command::EmitLlvm {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: Vec::new(),
                target: Some("rp2040".to_string()),
                board: None,
                chip: None,
                opt_level: None,
                output: None,
            }
        );

        assert_eq!(
            parse_args(["ram", "--target", "rp2040", "-O2", "examples/pico_w_demo.zev"]).unwrap(),
            Command::Ram {
                files: vec![PathBuf::from("examples/pico_w_demo.zev")],
                include_dirs: Vec::new(),
                target: Some("rp2040".to_string()),
                board: None,
                chip: None,
                opt_level: Some("-O2".to_string()),
            }
        );

        assert!(parse_args(["check", "-I"]).is_err());
        assert!(parse_args(["check", "--include"]).is_err());
        assert!(parse_args(["check", "-I="]).is_err());
        assert!(parse_args(["check", "--include="]).is_err());
        assert!(parse_args(["check", "-I", "", "app.zev"]).is_err());
    }

    #[test]
    fn parses_unified_build_and_flash_and_board_chip_flags() {
        assert_eq!(
            parse_args([
                "build",
                "--board",
                "rpi_pico_w",
                "--emit-ll",
                "-Oz",
                "-I",
                "bsp",
                "examples/pico_w_demo.zev",
                "-o",
                "pico.uf2",
            ])
            .unwrap(),
            Command::Build {
                files: vec![PathBuf::from("examples/pico_w_demo.zev")],
                include_dirs: vec![PathBuf::from("bsp")],
                board: Some("rpi_pico_w".to_string()),
                chip: None,
                opt_level: Some("-Oz".to_string()),
                emit_ll: true,
                output: Some(PathBuf::from("pico.uf2")),
            }
        );

        assert_eq!(
            parse_args([
                "build",
                "--chip=esp32c6",
                "--opt-level=O2",
            ])
            .unwrap(),
            Command::Build {
                files: Vec::new(),
                include_dirs: Vec::new(),
                board: None,
                chip: Some("esp32c6".to_string()),
                opt_level: Some("-O2".to_string()),
                emit_ll: false,
                output: None,
            }
        );

        assert_eq!(
            parse_args(["build"]).unwrap(),
            Command::Build {
                files: Vec::new(),
                include_dirs: Vec::new(),
                board: None,
                chip: None,
                opt_level: None,
                emit_ll: false,
                output: None,
            }
        );

        assert_eq!(
            parse_args([
                "flash",
                "--board=arduino_uno",
                "--port",
                "/dev/ttyACM0",
                "examples/arduino_uno_demo.zev",
            ])
            .unwrap(),
            Command::Flash {
                files: vec![PathBuf::from("examples/arduino_uno_demo.zev")],
                include_dirs: Vec::new(),
                board: Some("arduino_uno".to_string()),
                chip: None,
                port: Some("/dev/ttyACM0".to_string()),
                opt_level: None,
                output: None,
            }
        );

        assert_eq!(
            parse_args(["ram", "--board", "rpi_pico_w", "examples/pico_w_demo.zev"]).unwrap(),
            Command::Ram {
                files: vec![PathBuf::from("examples/pico_w_demo.zev")],
                include_dirs: Vec::new(),
                target: None,
                board: Some("rpi_pico_w".to_string()),
                chip: None,
                opt_level: None,
            }
        );

        assert!(parse_args(["build", "--board"]).is_err());
        assert!(parse_args(["build", "--chip="]).is_err());
        assert!(parse_args(["flash", "--port"]).is_err());
    }

    #[test]
    fn rejects_missing_files() {
        assert!(parse_args(["check"]).is_err());
        assert!(parse_args(["check", "--json"]).is_err());
        assert!(parse_args(["ram"]).is_err());
        assert!(parse_args(["emit-c"]).is_err());
        assert!(parse_args(["emit-llvm", "-o", "foo.ll"]).is_err());
        assert!(parse_args(["build-elf", "-o", "foo.elf"]).is_err());
        assert!(parse_args(["run-qemu"]).is_err());
        assert!(parse_args(["run-qemu", "--max-ticks", "50"]).is_err());
        assert!(parse_args(["build-esp32c6", "-o", "fw.bin"]).is_err());
        assert!(parse_args(["flash-esp32c6", "--port", "/dev/ttyACM0"]).is_err());
        assert!(parse_args(["build-avr", "-o", "uno.bin"]).is_err());
        assert!(parse_args(["build-uno", "-o", "uno.bin"]).is_err());
        assert!(parse_args(["flash-avr", "--port", "/dev/ttyACM0"]).is_err());
        assert!(parse_args(["flash-uno", "--port", "/dev/ttyACM0"]).is_err());
        assert!(parse_args(["build-nano-every", "-o", "nano.bin"]).is_err());
        assert!(parse_args(["build-atmega4809", "-o", "nano.bin"]).is_err());
        assert!(parse_args(["flash-nano-every", "--port", "/dev/ttyACM0"]).is_err());
        assert!(parse_args(["flash-atmega4809", "--port", "/dev/ttyACM0"]).is_err());
        assert!(parse_args(["build-rp2040", "-o", "pico.uf2"]).is_err());
        assert!(parse_args(["build-pico", "-o", "pico.uf2"]).is_err());
        assert!(parse_args(["flash-rp2040", "--port", "/dev/ttyACM2"]).is_err());
        assert!(parse_args(["flash-pico", "--port", "/dev/ttyACM2"]).is_err());
        assert!(parse_args(["run"]).is_err());
    }
}
