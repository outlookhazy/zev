use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Check {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        json: bool,
    },
    Ram {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        target: Option<String>,
        opt_level: Option<String>,
    },
    EmitC {
        files: Vec<PathBuf>,
        include_dirs: Vec<PathBuf>,
        output: Option<PathBuf>,
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
zevc 0.1.0 — Zev Bare-Metal Cooperative Language-OS Compiler (C99 + LLVM/ARM + ESP32-C6 RISC-V + AVR)

USAGE:
    zevc <COMMAND> [OPTIONS] [FILES...]

COMMANDS:
    doc [topic | code | --list]             Built-in Zev language primer, 8 topic guides & error explainers
    import-svd <chip.svd> [-o <out.zev>]    Convert a CMSIS-SVD XML description into Zev mmio/irq declarations
    check [--json] [-I <dir>] <files...>    Parse and statically verify .zev source files
    ram [--target <t>] [OPT] [-I <dir>] <files...> Print static SRAM budget (+ optional HW stack proof)
    emit-c [-I <dir>] <files...> [-o <out>] Emit freestanding C99 source + synthesized cooperative kernel
    run [-I <dir>] <files...> [OPTIONS]     Compile emitted C99 via clang and run deterministic host simulation
    emit-llvm [OPT] <files...> [-o <out.ll>] Emit LLVM 23 IR (raw by default, or optimized via opt w/ -O1..-Oz)
    build-elf [OPT] <files...> [-o <elf>]   Synthesize vector table + linker script and link Cortex-M3 .elf
    run-qemu [OPT] <files...>               Build bare-metal ARM Cortex-M3 .elf and run in qemu-system-arm
    build-esp32c6 [OPT] <files...> [-o <bin>] Synthesize Direct-Boot header + RV32IMAC .elf/.bin for ESP32-C6
    flash-esp32c6 [OPT] <files...> [--port] Build Direct-Boot .bin and flash ESP32-C6 via esptool (default /dev/ttyACM0)
    build-avr [OPT] <files...> [-o <bin>]   Synthesize vectors + ATmega328P AVR .elf/.bin for Arduino Uno R3 (default -Os)
    flash-avr [OPT] <files...> [--port]     Build AVR .bin and flash Arduino Uno R3 via STK500v1 Optiboot (aliases: build-uno/flash-uno)
    build-nano-every [OPT] <files...> [-o]  Synthesize 40-slot vectors + ATmega4809 .elf/.bin for Arduino Nano Every (alias: build-atmega4809)
    flash-nano-every [OPT] <files...> [-p]  Build ATmega4809 .bin and flash Arduino Nano Every via MuxTO UPDI (alias: flash-atmega4809)
    help                                    Show this help message

MODULE / INCLUDE OPTIONS (check, ram, emit-c, run, emit-llvm, build-elf, run-qemu, build-esp32c6, flash-esp32c6, build-avr, flash-avr, build-nano-every, flash-nano-every):
    -I <dir> | --include <dir>              Add a module search root directory for `import` resolution (repeatable)

CHECK OPTIONS:
    --json                                  Emit structured machine-readable JSON diagnostics

LLVM OPTIMIZATION & TARGET OPTIONS (ram, emit-llvm, build-elf, run-qemu, build-esp32c6, flash-esp32c6, build-avr, flash-avr, build-nano-every, flash-nano-every):
    -O0 | -O1 | -O2 | -O3 | -Os | -Oz       LLVM 23 opt + llc optimization level (default -O2; -Os on AVR; raw on emit-llvm)
    --opt-level <O0|O1|O2|O3|Os|Oz>         Explicit long-form optimization level selector
    --target <cortex-m3|esp32c6|avr|nano-every|host> Target selector (`ram`: cortex-m3/esp32c6/avr/nano-every; `emit-llvm`: +host)

RUN & RUN-QEMU OPTIONS:
    --sanitize                              Compile C99 host binary with -fsanitize=address,undefined (`run` only)
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
        "emit-c" => match parse_files_with_output(&raw[1..], "emit-c")? {
            Some((files, include_dirs, output)) => Ok(Command::EmitC {
                files,
                include_dirs,
                output,
            }),
            None => Ok(Command::Help),
        },
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
        "run" => parse_run(&raw[1..]),
        other => Err(format!(
            "unknown subcommand `{other}` (expected `doc`, `import-svd`, `check`, `ram`, `emit-c`, `run`, `emit-llvm`, `build-elf`, `run-qemu`, `build-esp32c6`, `flash-esp32c6`, `build-avr`, `flash-avr`, `build-nano-every`, or `flash-nano-every`; try `zevc --help`)"
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
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "check")? {
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
        json,
    })
}

fn parse_ram(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
    let mut target: Option<String> = None;
    let mut opt_level: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "ram")? {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--target" => {
                i += 1;
                if i >= args.len() {
                    return Err(
                        "`--target` requires a target name (`cortex-m3`, `esp32c6`, `avr`, or `nano-every`)"
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
    if files.is_empty() {
        return Err("`zevc ram` requires at least one `.zev` input file".to_string());
    }
    Ok(Command::Ram {
        files,
        include_dirs,
        target,
        opt_level,
    })
}

fn parse_files_with_output(
    args: &[String],
    subcmd: &str,
) -> Result<Option<(Vec<PathBuf>, Vec<PathBuf>, Option<PathBuf>)>, String> {
    let mut files = Vec::new();
    let mut include_dirs = Vec::new();
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
            _ if arg.starts_with("-o=") => {
                output = Some(PathBuf::from(&arg[3..]));
            }
            _ if arg.starts_with("--output=") => {
                output = Some(PathBuf::from(&arg[9..]));
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
    Ok(Some((files, include_dirs, output)))
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
    let mut opt_level: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if try_parse_include_arg(args, &mut i, &mut include_dirs, "emit-llvm")? {
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
                    return Err("`--target` requires a target name (`cortex-m3`, `esp32c6`, `avr`, `nano-every`, or `host`)".to_string());
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
    if files.is_empty() {
        return Err("`zevc emit-llvm` requires at least one `.zev` input file".to_string());
    }
    Ok(Command::EmitLlvm {
        files,
        include_dirs,
        target,
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
                json: true,
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
                opt_level: Some("-Oz".to_string()),
            }
        );

        assert_eq!(
            parse_args(["ram", "--target", "avr", "-Os", "examples/arduino_uno_demo.zev"]).unwrap(),
            Command::Ram {
                files: vec![PathBuf::from("examples/arduino_uno_demo.zev")],
                include_dirs: vec![],
                target: Some("avr".to_string()),
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
                opt_level: Some("-Os".to_string()),
            }
        );

        assert!(parse_args(["ram", "--target"]).is_err());
        assert!(parse_args(["ram", "--opt-level"]).is_err());
        assert!(parse_args(["ram", "-O4", "a.zev"]).is_err());
    }

    #[test]
    fn parses_emit_c_and_llvm_and_elf_and_esp32c6() {
        let cmd_c = parse_args(["emit-c", "a.zev", "b.zev", "-o", "out.c"]).unwrap();
        assert_eq!(
            cmd_c,
            Command::EmitC {
                files: vec![PathBuf::from("a.zev"), PathBuf::from("b.zev")],
                include_dirs: vec![],
                output: Some(PathBuf::from("out.c")),
            }
        );

        let cmd_ll_raw =
            parse_args(["emit-llvm", "examples/esp32c6_demo.zev", "-o", "out.ll"]).unwrap();
        assert_eq!(
            cmd_ll_raw,
            Command::EmitLlvm {
                files: vec![PathBuf::from("examples/esp32c6_demo.zev")],
                include_dirs: vec![],
                target: None,
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
                opt_level: None,
            }
        );

        let mut emit_c_args = vec!["emit-c"];
        emit_c_args.extend_from_slice(&include_flags);
        emit_c_args.extend_from_slice(&["app.zev", "-o", "app.c"]);
        assert_eq!(
            parse_args(emit_c_args).unwrap(),
            Command::EmitC {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: expected_dirs.clone(),
                output: Some(PathBuf::from("app.c")),
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

        assert_eq!(
            parse_args(["emit-llvm", "--target", "avr", "app.zev"]).unwrap(),
            Command::EmitLlvm {
                files: vec![PathBuf::from("app.zev")],
                include_dirs: Vec::new(),
                target: Some("avr".to_string()),
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
                opt_level: None,
                output: None,
            }
        );

        assert!(parse_args(["check", "-I"]).is_err());
        assert!(parse_args(["check", "--include"]).is_err());
        assert!(parse_args(["check", "-I="]).is_err());
        assert!(parse_args(["check", "--include="]).is_err());
        assert!(parse_args(["check", "-I", "", "app.zev"]).is_err());
    }

    #[test]
    fn rejects_missing_files() {
        assert!(parse_args(["check"]).is_err());
        assert!(parse_args(["check", "--json"]).is_err());
        assert!(parse_args(["ram"]).is_err());
        assert!(parse_args(["emit-c", "-o", "foo.c"]).is_err());
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
        assert!(parse_args(["run"]).is_err());
    }
}
