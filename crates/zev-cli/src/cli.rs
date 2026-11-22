use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Check {
        files: Vec<PathBuf>,
        json: bool,
    },
    Ram {
        files: Vec<PathBuf>,
    },
    EmitC {
        files: Vec<PathBuf>,
        output: Option<PathBuf>,
    },
    Run {
        files: Vec<PathBuf>,
        sanitize: bool,
        max_ticks: Option<u64>,
        run_args: Vec<String>,
    },
    EmitLlvm {
        files: Vec<PathBuf>,
        target: Option<String>,
        opt_level: Option<String>,
        output: Option<PathBuf>,
    },
    BuildElf {
        files: Vec<PathBuf>,
        opt_level: String,
        output: Option<PathBuf>,
    },
    RunQemu {
        files: Vec<PathBuf>,
        max_ticks: Option<u64>,
        opt_level: String,
    },
    BuildEsp32C6 {
        files: Vec<PathBuf>,
        opt_level: String,
        output: Option<PathBuf>,
    },
    FlashEsp32C6 {
        files: Vec<PathBuf>,
        port: String,
        opt_level: String,
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
zevc 0.1.0 — Zev Bare-Metal Cooperative Language-OS Compiler (C99 + LLVM/ARM + ESP32-C6 RISC-V)

USAGE:
    zevc <COMMAND> [OPTIONS] [FILES...]

COMMANDS:
    doc [topic | code | --list]             Built-in Zev language primer, 8 topic guides & error explainers
    check [--json] <files...>               Parse and statically verify .zev source files
    ram <files...>                          Print static zero-heap SRAM budget & coroutine frame report
    emit-c <files...> [-o <out.c>]          Emit freestanding C99 source + synthesized cooperative kernel
    run <files...> [OPTIONS]                Compile emitted C99 via clang and run deterministic host simulation
    emit-llvm [OPT] <files...> [-o <out.ll>] Emit LLVM 23 IR (raw by default, or optimized via opt w/ -O1..-Oz)
    build-elf [OPT] <files...> [-o <elf>]   Synthesize vector table + linker script and link Cortex-M3 .elf
    run-qemu [OPT] <files...>               Build bare-metal ARM Cortex-M3 .elf and run in qemu-system-arm
    build-esp32c6 [OPT] <files...> [-o <bin>] Synthesize Direct-Boot header + RV32IMAC .elf/.bin for ESP32-C6
    flash-esp32c6 [OPT] <files...> [--port] Build Direct-Boot .bin and flash ESP32-C6 via esptool (default /dev/ttyACM0)
    help                                    Show this help message

CHECK OPTIONS:
    --json                                  Emit structured machine-readable JSON diagnostics

LLVM OPTIMIZATION & TARGET OPTIONS (emit-llvm, build-elf, run-qemu, build-esp32c6, flash-esp32c6):
    -O0 | -O1 | -O2 | -O3 | -Os | -Oz       LLVM 23 opt + llc optimization level (default -O2; raw on emit-llvm)
    --opt-level <O0|O1|O2|O3|Os|Oz>         Explicit long-form optimization level selector
    --target <cortex-m3|esp32c6|host>       Target triple selector (`emit-llvm` only; default `cortex-m3`)

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
        "check" => parse_check(&raw[1..]),
        "ram" => match parse_file_list(&raw[1..], "ram")? {
            Some(files) => Ok(Command::Ram { files }),
            None => Ok(Command::Help),
        },
        "emit-c" => match parse_files_with_output(&raw[1..], "emit-c")? {
            Some((files, output)) => Ok(Command::EmitC { files, output }),
            None => Ok(Command::Help),
        },
        "emit-llvm" => parse_emit_llvm(&raw[1..]),
        "build-elf" => match parse_files_with_output_and_opt(&raw[1..], "build-elf")? {
            Some((files, opt_level, output)) => Ok(Command::BuildElf {
                files,
                opt_level,
                output,
            }),
            None => Ok(Command::Help),
        },
        "run-qemu" => parse_run_qemu(&raw[1..]),
        "build-esp32c6" => match parse_files_with_output_and_opt(&raw[1..], "build-esp32c6")? {
            Some((files, opt_level, output)) => Ok(Command::BuildEsp32C6 {
                files,
                opt_level,
                output,
            }),
            None => Ok(Command::Help),
        },
        "flash-esp32c6" => parse_flash_esp32c6(&raw[1..]),
        "run" => parse_run(&raw[1..]),
        other => Err(format!(
            "unknown subcommand `{other}` (expected `doc`, `check`, `ram`, `emit-c`, `run`, `emit-llvm`, `build-elf`, `run-qemu`, `build-esp32c6`, or `flash-esp32c6`; try `zevc --help`)"
        )),
    }
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
    let mut json = false;
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--json" => json = true,
            _ if arg.starts_with('-') => {
                return Err(format!("unexpected option `{arg}` for `zevc check`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
    }
    if files.is_empty() {
        return Err("`zevc check` requires at least one `.zev` input file".to_string());
    }
    Ok(Command::Check { files, json })
}

fn parse_file_list(args: &[String], subcmd: &str) -> Result<Option<Vec<PathBuf>>, String> {
    let mut files = Vec::new();
    for arg in args {
        if arg == "-h" || arg == "--help" {
            return Ok(None);
        }
        if arg.starts_with('-') {
            return Err(format!("unexpected option `{arg}` for `zevc {subcmd}`"));
        }
        files.push(PathBuf::from(arg));
    }
    if files.is_empty() {
        return Err(format!(
            "`zevc {subcmd}` requires at least one `.zev` input file"
        ));
    }
    Ok(Some(files))
}

fn parse_files_with_output(
    args: &[String],
    subcmd: &str,
) -> Result<Option<(Vec<PathBuf>, Option<PathBuf>)>, String> {
    let mut files = Vec::new();
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
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
    Ok(Some((files, output)))
}

fn parse_files_with_output_and_opt(
    args: &[String],
    subcmd: &str,
) -> Result<Option<(Vec<PathBuf>, String, Option<PathBuf>)>, String> {
    let mut files = Vec::new();
    let mut opt_level = "-O2".to_string();
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
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
    Ok(Some((files, opt_level, output)))
}

fn parse_emit_llvm(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut target: Option<String> = None;
    let mut opt_level: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
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
                    return Err("`--target` requires a target name (`cortex-m3`, `esp32c6`, or `host`)".to_string());
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
        target,
        opt_level,
        output,
    })
}

fn parse_run_qemu(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut max_ticks: Option<u64> = None;
    let mut opt_level = "-O2".to_string();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
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
        max_ticks,
        opt_level,
    })
}

fn parse_flash_esp32c6(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut port = "/dev/ttyACM0".to_string();
    let mut opt_level = "-O2".to_string();
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
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
                    return Err("`-o` requires an output `.bin` path for `zevc flash-esp32c6`".to_string());
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "--opt-level" => {
                i += 1;
                if i >= args.len() {
                    return Err(
                        "`--opt-level` requires a level (`O0`, `O1`, `O2`, `O3`, `Os`, or `Oz`) for `zevc flash-esp32c6`"
                            .to_string(),
                    );
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
                return Err(format!("unexpected option `{arg}` for `zevc flash-esp32c6`"));
            }
            _ => files.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if files.is_empty() {
        return Err("`zevc flash-esp32c6` requires at least one `.zev` input file".to_string());
    }
    Ok(Command::FlashEsp32C6 {
        files,
        port,
        opt_level,
        output,
    })
}

fn parse_run(args: &[String]) -> Result<Command, String> {
    let mut files = Vec::new();
    let mut sanitize = false;
    let mut max_ticks = None;
    let mut run_args = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "--" => {
                run_args.extend_from_slice(&args[i + 1..]);
                break;
            }
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
                ]
            }
        );
    }

    #[test]
    fn parses_emit_c_and_llvm_and_elf_and_esp32c6() {
        let cmd_c = parse_args(["emit-c", "a.zev", "b.zev", "-o", "out.c"]).unwrap();
        assert_eq!(
            cmd_c,
            Command::EmitC {
                files: vec![PathBuf::from("a.zev"), PathBuf::from("b.zev")],
                output: Some(PathBuf::from("out.c")),
            }
        );

        let cmd_ll_raw =
            parse_args(["emit-llvm", "examples/esp32c6_demo.zev", "-o", "out.ll"]).unwrap();
        assert_eq!(
            cmd_ll_raw,
            Command::EmitLlvm {
                files: vec![PathBuf::from("examples/esp32c6_demo.zev")],
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
                sanitize: true,
                max_ticks: Some(500),
                run_args: vec!["--trace".to_string()],
            }
        );
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
        assert!(parse_args(["run"]).is_err());
    }
}
