use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::Child;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::serial::SerialPort;

pub const DEFAULT_SCRIPT_TIMEOUT_MS: u64 = 5_000;

/// Load scripted command lines from an optional script file (skipping empty/`#` lines)
/// followed by inline `--cmd` strings.
pub fn collect_script_commands(
    script_path: Option<&Path>,
    inline_cmds: &[String],
) -> Result<Vec<String>, String> {
    let mut commands = Vec::new();
    if let Some(path) = script_path {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read script file `{}`: {e}", path.display()))?;
        for raw in text.lines() {
            let trimmed = raw.trim_end_matches('\r').trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            commands.push(raw.trim_end_matches('\r').to_string());
        }
    }
    for cmd in inline_cmds {
        let trimmed = cmd.trim_end_matches('\r');
        if !trimmed.is_empty() {
            commands.push(trimmed.to_string());
        }
    }
    Ok(commands)
}

trait ConsoleTransport {
    fn write_cmd_line(&mut self, line: &str) -> Result<(), String>;
    fn read_chunk(&mut self, timeout: Duration) -> Result<Option<Vec<u8>>, String>;
}

enum PipeEvent {
    Bytes(Vec<u8>),
    Eof,
    Err(String),
}

struct ChildPipeTransport {
    stdin: Option<std::process::ChildStdin>,
    rx: Receiver<PipeEvent>,
}

impl ChildPipeTransport {
    fn from_child(child: &mut Child) -> Result<Self, String> {
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "child process stdin pipe was not captured".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "child process stdout pipe was not captured".to_string())?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                match reader.fill_buf() {
                    Ok([]) => {
                        let _ = tx.send(PipeEvent::Eof);
                        break;
                    }
                    Ok(avail) => {
                        let chunk = avail.to_vec();
                        let len = chunk.len();
                        reader.consume(len);
                        if tx.send(PipeEvent::Bytes(chunk)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(PipeEvent::Err(format!("stdout read failed: {e}")));
                        break;
                    }
                }
            }
        });
        Ok(Self {
            stdin: Some(stdin),
            rx,
        })
    }
}

impl ConsoleTransport for ChildPipeTransport {
    fn write_cmd_line(&mut self, line: &str) -> Result<(), String> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| "child stdin closed".to_string())?;
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .map_err(|e| format!("failed to write command `{line}` to target: {e}"))
    }

    fn read_chunk(&mut self, timeout: Duration) -> Result<Option<Vec<u8>>, String> {
        match self.rx.recv_timeout(timeout) {
            Ok(PipeEvent::Bytes(bytes)) => Ok(Some(bytes)),
            Ok(PipeEvent::Eof) => Err("target exited before completing script command".to_string()),
            Ok(PipeEvent::Err(e)) => Err(e),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => {
                Err("target output closed before completing script command".to_string())
            }
        }
    }
}

struct SerialTransport<'a> {
    port: &'a SerialPort,
}

impl ConsoleTransport for SerialTransport<'_> {
    fn write_cmd_line(&mut self, line: &str) -> Result<(), String> {
        self.port.write_all_bytes(line.as_bytes())?;
        self.port.write_all_bytes(b"\n")
    }

    fn read_chunk(&mut self, timeout: Duration) -> Result<Option<Vec<u8>>, String> {
        let ms = timeout.as_millis().clamp(1, i32::MAX as u128) as i32;
        let mut buf = [0u8; 512];
        match self.port.poll_read(&mut buf, ms)? {
            Some(n) => Ok(Some(buf[..n].to_vec())),
            None => Ok(None),
        }
    }
}

fn normalize_status_line(raw: &str) -> &str {
    let mut s = raw.trim_end_matches('\r');
    while let Some(rest) = s.strip_prefix("> ") {
        s = rest;
    }
    s.trim()
}

fn run_lockstep_to<T: ConsoleTransport, W: Write>(
    transport: &mut T,
    commands: &[String],
    timeout_per_cmd: Duration,
    transcript: &mut W,
) -> Result<(), String> {
    let mut line_buf: Vec<u8> = Vec::new();

    for (idx, cmd) in commands.iter().enumerate() {
        transport.write_cmd_line(cmd)?;
        let deadline = Instant::now() + timeout_per_cmd;
        let mut finished_cmd = false;

        while !finished_cmd {
            let now = Instant::now();
            if now >= deadline {
                return Err(format!(
                    "script timed out after {} ms waiting for `ok` or `err` response to command #{}: `{cmd}`",
                    timeout_per_cmd.as_millis(),
                    idx + 1
                ));
            }
            let remaining = deadline - now;
            let wait = remaining.min(Duration::from_millis(100));
            let Some(chunk) = transport.read_chunk(wait)? else {
                continue;
            };
            transcript
                .write_all(&chunk)
                .and_then(|()| transcript.flush())
                .map_err(|e| format!("failed to write script transcript to stdout: {e}"))?;

            line_buf.extend_from_slice(&chunk);
            while let Some(nl_pos) = line_buf.iter().position(|&b| b == b'\n') {
                let raw_line = String::from_utf8_lossy(&line_buf[..nl_pos]).into_owned();
                line_buf.drain(..=nl_pos);
                let norm = normalize_status_line(&raw_line);
                if norm == "ok" {
                    finished_cmd = true;
                    break;
                }
                if norm == "err" || norm.starts_with("err ") {
                    return Err(format!("script command `{cmd}` returned `{norm}`"));
                }
            }
        }
    }
    Ok(())
}

pub fn run_script_on_child_to<W: Write>(
    child: &mut Child,
    commands: &[String],
    timeout_per_cmd: Duration,
    transcript: &mut W,
) -> Result<(), String> {
    let mut transport = ChildPipeTransport::from_child(child)?;
    let res = run_lockstep_to(&mut transport, commands, timeout_per_cmd, transcript);
    let _ = child.kill();
    let _ = child.wait();
    res
}

pub fn run_script_on_child(
    child: &mut Child,
    commands: &[String],
    timeout_per_cmd: Duration,
) -> Result<(), String> {
    let mut stdout = io::stdout().lock();
    run_script_on_child_to(child, commands, timeout_per_cmd, &mut stdout)
}

fn run_script_on_serial_to<W: Write>(
    port: &SerialPort,
    commands: &[String],
    timeout_per_cmd: Duration,
    transcript: &mut W,
) -> Result<(), String> {
    let mut transport = SerialTransport { port };
    run_lockstep_to(&mut transport, commands, timeout_per_cmd, transcript)
}

pub fn run_script_on_serial(
    port: &SerialPort,
    commands: &[String],
    timeout_per_cmd: Duration,
) -> Result<(), String> {
    let mut stdout = io::stdout().lock();
    run_script_on_serial_to(port, commands, timeout_per_cmd, &mut stdout)
}
