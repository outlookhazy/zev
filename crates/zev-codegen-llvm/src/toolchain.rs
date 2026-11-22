//! Toolchain driver for LLVM 23 (`llc`), LLD (`ld.lld`), `llvm-size`, ELF32 symbol auditing,
//! and bare-metal ARM Cortex-M3 execution in `qemu-system-arm`.

use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Target architecture for LLVM 23 code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    /// Bare-metal ARM Cortex-M3 (`thumbv7m-none-eabi`, Stellaris `lm3s6965evb` / STM32 compatible).
    Thumbv7mNoneEabi,
    /// Bare-metal Espressif ESP32-C6 RISC-V (`riscv32-unknown-none-elf`, `rv32imac` Direct-Boot).
    Riscv32Esp32C6,
    /// Bare-metal Microchip ATmega328P AVR (`avr`, Arduino Uno R3 Optiboot 16 MHz).
    AvrAtmega328p,
    /// Bare-metal Microchip ATmega4809 AVR (`avr` / `avrxmega3`, Arduino Nano Every 16/20 MHz).
    AvrAtmega4809,
    /// Host 64-bit Linux (`x86_64-unknown-linux-gnu`) for host `.ll` inspection/testing.
    HostX86_64Linux,
}

impl TargetArch {
    pub const fn is_avr(self) -> bool {
        matches!(self, Self::AvrAtmega328p | Self::AvrAtmega4809)
    }

    pub const fn avr_mcpu(self) -> &'static str {
        match self {
            Self::AvrAtmega328p => "atmega328p",
            Self::AvrAtmega4809 => "atmega4809",
            Self::Thumbv7mNoneEabi | Self::Riscv32Esp32C6 | Self::HostX86_64Linux => "",
        }
    }

    pub const fn triple(self) -> &'static str {
        match self {
            Self::Thumbv7mNoneEabi => "thumbv7m-none-eabi",
            Self::Riscv32Esp32C6 => "riscv32-unknown-none-elf",
            Self::AvrAtmega328p | Self::AvrAtmega4809 => "avr",
            Self::HostX86_64Linux => "x86_64-unknown-linux-gnu",
        }
    }

    pub const fn datalayout(self) -> &'static str {
        match self {
            Self::Thumbv7mNoneEabi => "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64",
            Self::Riscv32Esp32C6 => "e-m:e-p:32:32-i64:64-n32-S128",
            Self::AvrAtmega328p | Self::AvrAtmega4809 => {
                "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8:16-a:8"
            }
            Self::HostX86_64Linux => {
                "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
            }
        }
    }

    pub const fn ptr_bits(self) -> u32 {
        match self {
            Self::Thumbv7mNoneEabi | Self::Riscv32Esp32C6 => 32,
            Self::AvrAtmega328p | Self::AvrAtmega4809 => 16,
            Self::HostX86_64Linux => 64,
        }
    }

    pub const fn usize_llvm_ty(self) -> &'static str {
        match self {
            Self::Thumbv7mNoneEabi | Self::Riscv32Esp32C6 => "i32",
            Self::AvrAtmega328p | Self::AvrAtmega4809 => "i16",
            Self::HostX86_64Linux => "i64",
        }
    }

    pub const fn fn_ptr_llvm_ty(self) -> &'static str {
        match self {
            Self::AvrAtmega328p | Self::AvrAtmega4809 => "ptr addrspace(1)",
            Self::Thumbv7mNoneEabi | Self::Riscv32Esp32C6 | Self::HostX86_64Linux => "ptr",
        }
    }
}

/// Configuration for LLVM 23 IR generation and bare-metal Cortex-M3 `.elf` linking.
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    pub target: TargetArch,
    /// Flash base address (`0x0000_0000` on `lm3s6965evb`).
    pub flash_origin: u32,
    /// Flash size in bytes (`256 * 1024` on `lm3s6965evb`).
    pub flash_len: u32,
    /// SRAM base address (`0x2000_0000` on Cortex-M).
    pub ram_origin: u32,
    /// SRAM size in bytes (`64 * 1024` on `lm3s6965evb` -> initial SP = `0x2001_0000`).
    pub ram_len: u32,
    /// Optimization flag passed to `llc` (`"-O2"` by default).
    pub opt_level: String,
    /// Optional deterministic tick cap (`0` = unlimited, `> 0` = auto-exit at tick cap if reached).
    pub max_ticks: u32,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            target: TargetArch::Thumbv7mNoneEabi,
            flash_origin: 0x0000_0000,
            flash_len: 256 * 1024,
            ram_origin: 0x2000_0000,
            ram_len: 64 * 1024,
            opt_level: "-O2".to_string(),
            max_ticks: 0,
        }
    }
}

impl CodegenOptions {
    /// Default options for bare-metal Espressif ESP32-C6 (`rv32imac` Direct-Boot):
    /// - `FLASH`: `0x4200_0000` (4 MB unified MMU cache mapped from flash offset `0x0`)
    /// - `RAM`:   `0x4080_0000` (448 KB HP-SRAM -> initial `sp = 0x4087_0000`, below ROM top SRAM)
    pub fn esp32c6() -> Self {
        Self {
            target: TargetArch::Riscv32Esp32C6,
            flash_origin: 0x4200_0000,
            flash_len: 4 * 1024 * 1024,
            ram_origin: 0x4080_0000,
            ram_len: 448 * 1024,
            opt_level: "-O2".to_string(),
            max_ticks: 0,
        }
    }

    /// Default options for bare-metal Microchip ATmega328P (`avr`, Arduino Uno R3 Optiboot):
    /// - `FLASH`: `0x0000_0000` (`32,256 B` usable below 512 B Optiboot)
    /// - `RAM`:   `0x0000_0100` (`2,048 B` SRAM `0x0100..=0x08FF`, initial `SP = 0x08FF`)
    pub fn avr_uno() -> Self {
        Self {
            target: TargetArch::AvrAtmega328p,
            flash_origin: 0x0000_0000,
            flash_len: 32_256,
            ram_origin: 0x0000_0100,
            ram_len: 2_048,
            opt_level: "-Os".to_string(),
            max_ticks: 0,
        }
    }

    /// Default options for bare-metal Microchip ATmega4809 (`avrxmega3`, Arduino Nano Every UPDI):
    /// - `FLASH`: `0x0000_0000` (`49,152 B` = 48 KB full Flash via SAMD11 `MuxTO` UPDI)
    /// - `RAM`:   `0x0000_2800` (`6,144 B` = 6 KB SRAM `0x2800..=0x3FFF`, initial `SP = 0x3FFF`)
    pub fn avr_nano_every() -> Self {
        Self {
            target: TargetArch::AvrAtmega4809,
            flash_origin: 0x0000_0000,
            flash_len: 49_152,
            ram_origin: 0x0000_2800,
            ram_len: 6_144,
            opt_level: "-Os".to_string(),
            max_ticks: 0,
        }
    }

    /// Initial stack pointer (`ram_origin + ram_len`, or `- 1` on AVR post-decrement push).
    pub const fn initial_sp(&self) -> u32 {
        let top = self.ram_origin.wrapping_add(self.ram_len);
        if self.target.is_avr() {
            top.wrapping_sub(1)
        } else {
            top
        }
    }

    /// Returns `true` when middle-end `opt` passes should be skipped (`-O0`).
    pub fn is_opt_o0(&self) -> bool {
        matches!(self.opt_level.trim(), "-O0" | "O0")
    }

    /// Function-level LLVM 23 size-optimization attributes emitted for `-Os` / `-Oz`
    /// (omitted on hardware-timed `.iram1` functions).
    pub fn fn_size_attrs(&self) -> &'static str {
        match self.opt_level.trim() {
            "-Os" | "Os" => " optsize",
            "-Oz" | "Oz" => " minsize optsize",
            _ => "",
        }
    }

    /// LLVM 23 `opt -passes=...` pipeline corresponding to `self.opt_level`.
    /// Note: LLVM 23 uses `default<O2>` combined with `optsize` / `minsize` function attributes
    /// for `-Os` / `-Oz`.
    pub fn opt_passes_pipeline(&self) -> Result<&'static str, CodegenError> {
        match self.opt_level.trim() {
            "-O0" | "O0" => Ok("default<O0>"),
            "-O1" | "O1" => Ok("default<O1>"),
            "-O2" | "O2" | "-Os" | "Os" | "-Oz" | "Oz" => Ok("default<O2>"),
            "-O3" | "O3" => Ok("default<O3>"),
            other => Err(CodegenError::new(
                "opt",
                format!(
                    "unsupported optimization level `{other}` (expected -O0, -O1, -O2, -O3, -Os, or -Oz)"
                ),
            )),
        }
    }

    /// Code-generator optimization flag passed to `/bin/llc`.
    pub fn llc_opt_flag(&self) -> Result<&'static str, CodegenError> {
        match self.opt_level.trim() {
            "-O0" | "O0" => Ok("-O0"),
            "-O1" | "O1" => Ok("-O1"),
            "-O2" | "O2" | "-Os" | "Os" | "-Oz" | "Oz" => Ok("-O2"),
            "-O3" | "O3" => Ok("-O3"),
            other => Err(CodegenError::new(
                "llc",
                format!(
                    "unsupported optimization level `{other}` (expected -O0, -O1, -O2, -O3, -Os, or -Oz)"
                ),
            )),
        }
    }
}

/// Parsed output of `/bin/llvm-size <firmware.elf>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfSectionSizes {
    pub text: u64,
    pub data: u64,
    pub bss: u64,
    pub dec: u64,
    pub hex: String,
    pub raw_output: String,
}

impl fmt::Display for ElfSectionSizes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "text={}B data={}B bss={}B total={}B (0x{})",
            self.text, self.data, self.bss, self.dec, self.hex
        )
    }
}

/// ELF32 symbol table audit verifying freestanding zero-heap, zero-libc compliance.
#[derive(Debug, Clone, Default)]
pub struct ElfSymbolAudit {
    pub defined_symbols: Vec<String>,
    pub undefined_symbols: Vec<String>,
    pub heap_or_libc_symbols: Vec<String>,
}

impl ElfSymbolAudit {
    pub fn is_zero_heap_freestanding(&self) -> bool {
        self.undefined_symbols.is_empty() && self.heap_or_libc_symbols.is_empty()
    }
}

/// Artifacts produced by `build_elf` / `compile_ll_to_elf`.
#[derive(Debug, Clone)]
pub struct ElfBuildResult {
    pub ll_path: PathBuf,
    pub obj_path: PathBuf,
    pub ld_path: PathBuf,
    pub elf_path: PathBuf,
    pub sizes: ElfSectionSizes,
    pub audit: ElfSymbolAudit,
}

/// Artifacts produced by `build_esp32c6_image`.
#[derive(Debug, Clone)]
pub struct Esp32C6ImageResult {
    pub elf: ElfBuildResult,
    pub bin_path: PathBuf,
    pub bin_size: u64,
    pub magic_header: [u8; 8],
}

/// Artifacts produced by `build_avr_uno_image`.
#[derive(Debug, Clone)]
pub struct AvrImageResult {
    pub elf: ElfBuildResult,
    pub bin_path: PathBuf,
    pub bin_size: u64,
    pub sram_static_bytes: u64,
}

/// Options for executing a Cortex-M3 `.elf` inside `qemu-system-arm`.
#[derive(Debug, Clone)]
pub struct QemuRunOptions {
    pub machine: String,
    pub cpu: String,
    pub timeout: Duration,
}

impl Default for QemuRunOptions {
    fn default() -> Self {
        Self {
            machine: "lm3s6965evb".to_string(),
            cpu: "cortex-m3".to_string(),
            timeout: Duration::from_millis(5_000),
        }
    }
}

/// Result of running a bare-metal `.elf` in `qemu-system-arm`.
#[derive(Debug, Clone)]
pub struct QemuExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub elapsed: Duration,
}

/// Complete build + QEMU execution report.
#[derive(Debug, Clone)]
pub struct QemuRunResult {
    pub build: ElfBuildResult,
    pub exec: QemuExecOutput,
}

/// Error returned by LLVM IR emission, `llc` compilation, `ld.lld` linking, or QEMU execution.
#[derive(Debug, Clone)]
pub struct CodegenError {
    pub stage: &'static str,
    pub message: String,
}

impl CodegenError {
    pub fn new(stage: &'static str, message: impl Into<String>) -> Self {
        Self {
            stage,
            message: message.into(),
        }
    }
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[zev-codegen-llvm:{}] {}", self.stage, self.message)
    }
}

impl std::error::Error for CodegenError {}

/// Auto-generate the freestanding linker script (`FLASH` + `RAM`, zero libc) for Cortex-M3, ESP32-C6, or AVR.
pub fn synthesize_linker_script(opts: &CodegenOptions) -> String {
    if opts.target.is_avr() {
        let ram_vma = 0x0080_0000u32.wrapping_add(opts.ram_origin);
        return format!(
            r#"/* Auto-synthesized by zev-codegen-llvm for AVR ({mcpu}) */
MEMORY
{{
  FLASH (rx)  : ORIGIN = 0x{flash_orig:08X}, LENGTH = {flash_len}
  RAM   (rw!x): ORIGIN = 0x{ram_vma:08X}, LENGTH = {ram_len}
}}

ENTRY(_start)

SECTIONS
{{
  .text 0x{flash_orig:08X} :
  {{
    KEEP(*(.vectors))
    KEEP(*(.text.startup))
    *(.text .text.*)
    . = ALIGN(2);
    __sidata = .;
  }} > FLASH

  .data 0x{ram_vma:08X} : AT(__sidata)
  {{
    __sdata = .;
    *(.rodata .rodata.*)
    *(.data .data.*)
    . = ALIGN(2);
    __edata = .;
  }} > RAM

  .bss (NOLOAD) :
  {{
    __sbss = .;
    *(.bss .bss.* COMMON)
    __ebss = .;
  }} > RAM

  /DISCARD/ :
  {{
    *(.comment)
    *(.note*)
    *(.eh_frame*)
  }}
}}
"#,
            flash_orig = opts.flash_origin,
            flash_len = opts.flash_len,
            ram_vma = ram_vma,
            ram_len = opts.ram_len,
            mcpu = opts.target.avr_mcpu(),
        );
    }

    if opts.target == TargetArch::Riscv32Esp32C6 {
        return format!(
            r#"/* Auto-synthesized by zev-codegen-llvm for ESP32-C6 (rv32imac Direct-Boot) */
MEMORY
{{
  FLASH (rx)  : ORIGIN = 0x{flash_orig:08X}, LENGTH = {flash_len}
  RAM   (rwx) : ORIGIN = 0x{ram_orig:08X}, LENGTH = {ram_len}
}}

ENTRY(_start)

SECTIONS
{{
  .text ORIGIN(FLASH) :
  {{
    /* ESP32-C6 ROM Direct-Boot magic header (8 bytes at flash 0x0 -> jumps to 0x42000008) */
    LONG(0xAEDB041D);
    LONG(0xAEDB041D);
    KEEP(*(.entry))
    *(.text .text.*)
    *(.rodata .rodata.* .srodata .srodata.*)
    . = ALIGN(4);
    __sidata = .;
  }} > FLASH

  .data : AT(__sidata)
  {{
    . = ALIGN(4);
    __sdata = .;
    KEEP(*(.iram1 .iram1.*))
    *(.data .data.* .sdata .sdata.*)
    . = ALIGN(4);
    __edata = .;
  }} > RAM

  .bss (NOLOAD) :
  {{
    . = ALIGN(4);
    __sbss = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(4);
    __ebss = .;
  }} > RAM

  /DISCARD/ :
  {{
    *(.eh_frame*)
    *(.riscv.attributes)
    *(.comment)
    *(.note*)
  }}
}}
"#,
            flash_orig = opts.flash_origin,
            flash_len = opts.flash_len,
            ram_orig = opts.ram_origin,
            ram_len = opts.ram_len,
        );
    }

    format!(
        r#"/* Auto-synthesized by zev-codegen-llvm — zero user .ld / zero CRT glue */
MEMORY
{{
  FLASH (rx)  : ORIGIN = 0x{flash_orig:08X}, LENGTH = {flash_len}
  RAM   (rwx) : ORIGIN = 0x{ram_orig:08X}, LENGTH = {ram_len}
}}

ENTRY(Reset_Handler)

SECTIONS
{{
  .text :
  {{
    KEEP(*(.isr_vector))
    *(.text .text.*)
    *(.rodata .rodata.*)
    . = ALIGN(4);
    __sidata = .;
  }} > FLASH

  .data : AT(__sidata)
  {{
    . = ALIGN(4);
    __sdata = .;
    *(.data .data.*)
    . = ALIGN(4);
    __edata = .;
  }} > RAM

  .bss (NOLOAD) :
  {{
    . = ALIGN(4);
    __sbss = .;
    *(.bss .bss.* COMMON)
    . = ALIGN(4);
    __ebss = .;
  }} > RAM

  /DISCARD/ :
  {{
    *(.ARM.exidx*)
    *(.ARM.extab*)
    *(.note.GNU-stack)
  }}
}}
"#,
        flash_orig = opts.flash_origin,
        flash_len = opts.flash_len,
        ram_orig = opts.ram_origin,
        ram_len = opts.ram_len,
    )
}

pub(crate) fn find_tool(preferred_abs: &'static str, fallback_name: &'static str) -> &'static str {
    if Path::new(preferred_abs).exists() {
        preferred_abs
    } else {
        fallback_name
    }
}

/// Optimize textual LLVM 23 `.ll` IR via `/bin/opt` (`default<O1|O2|O3>`).
pub fn optimize_ll_with_opt(
    in_ll: &Path,
    out_ll: &Path,
    opts: &CodegenOptions,
) -> Result<(), CodegenError> {
    let pipeline = opts.opt_passes_pipeline()?;
    if let Some(parent) = out_ll.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                CodegenError::new("io", format!("failed to create {}: {e}", parent.display()))
            })?;
        }
    }

    let opt = find_tool("/bin/opt", "opt");
    let mut cmd = Command::new(opt);
    cmd.arg(format!("-mtriple={}", opts.target.triple()));
    match opts.target {
        TargetArch::Thumbv7mNoneEabi => {
            cmd.arg("-mcpu=cortex-m3");
        }
        TargetArch::Riscv32Esp32C6 => {
            cmd.arg("-mcpu=generic-rv32").arg("-mattr=+m,+a,+c");
        }
        TargetArch::AvrAtmega328p | TargetArch::AvrAtmega4809 => {
            cmd.arg(format!("-mcpu={}", opts.target.avr_mcpu()));
        }
        TargetArch::HostX86_64Linux => {}
    }
    cmd.arg(format!("-passes={pipeline}"))
        .arg("-S")
        .arg(in_ll)
        .arg("-o")
        .arg(out_ll)
        .stdin(Stdio::null());

    let out = cmd
        .output()
        .map_err(|e| CodegenError::new("opt", format!("failed to invoke `{opt}`: {e}")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "opt",
            format!("opt exited with {}: {}", out.status, stderr.trim()),
        ));
    }
    Ok(())
}

/// Compile textual LLVM 23 `.ll` IR to an ELF object file `.o` via `/bin/opt` + `/bin/llc`.
pub fn compile_ll_to_obj(
    ll_path: &Path,
    obj_path: &Path,
    opts: &CodegenOptions,
) -> Result<(), CodegenError> {
    let llc_flag = opts.llc_opt_flag()?;
    let opt_ll_path = obj_path.with_extension("opt.ll");
    let llc_input: &Path = if opts.is_opt_o0() {
        ll_path
    } else {
        optimize_ll_with_opt(ll_path, &opt_ll_path, opts)?;
        &opt_ll_path
    };

    let llc = find_tool("/bin/llc", "llc");
    let mut cmd = Command::new(llc);
    cmd.arg(format!("-mtriple={}", opts.target.triple()));
    match opts.target {
        TargetArch::Thumbv7mNoneEabi => {
            cmd.arg("-mcpu=cortex-m3")
                .arg("-function-sections")
                .arg("-data-sections");
        }
        TargetArch::Riscv32Esp32C6 => {
            cmd.arg("-mcpu=generic-rv32")
                .arg("-mattr=+m,+a,+c")
                .arg("-relocation-model=static")
                .arg("-function-sections")
                .arg("-data-sections");
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
        .arg(llc_input)
        .arg("-o")
        .arg(obj_path)
        .stdin(Stdio::null());

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
    Ok(())
}

/// Link a compiled `.o` object into a freestanding Cortex-M3 `.elf` via `/bin/ld.lld`.
pub fn link_obj_to_elf(
    obj_path: &Path,
    ld_path: &Path,
    elf_path: &Path,
) -> Result<(), CodegenError> {
    let lld = find_tool("/bin/ld.lld", "ld.lld");
    let out = Command::new(lld)
        .arg("-T")
        .arg(ld_path)
        .arg("--gc-sections")
        .arg(obj_path)
        .arg("-o")
        .arg(elf_path)
        .output()
        .map_err(|e| CodegenError::new("ld.lld", format!("failed to invoke `{lld}`: {e}")))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "ld.lld",
            format!("ld.lld exited with {}: {}", out.status, stderr.trim()),
        ));
    }
    Ok(())
}

/// Inspect `.text`, `.data`, and `.bss` sizes of a linked `.elf` using `/bin/llvm-size`.
pub fn inspect_elf_sizes(elf_path: &Path) -> Result<ElfSectionSizes, CodegenError> {
    let llvm_size = find_tool("/bin/llvm-size", "llvm-size");
    let out = Command::new(llvm_size)
        .arg(elf_path)
        .output()
        .map_err(|e| {
            CodegenError::new("llvm-size", format!("failed to invoke `{llvm_size}`: {e}"))
        })?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "llvm-size",
            format!("llvm-size exited with {}: {}", out.status, stderr.trim()),
        ));
    }

    let raw = String::from_utf8_lossy(&out.stdout).to_string();
    let data_line = raw
        .lines()
        .skip(1)
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| {
            CodegenError::new("llvm-size", format!("unexpected llvm-size output: {raw}"))
        })?;
    let cols: Vec<&str> = data_line.split_whitespace().collect();
    if cols.len() < 5 {
        return Err(CodegenError::new(
            "llvm-size",
            format!("could not parse llvm-size columns from: {data_line}"),
        ));
    }
    let parse_u64 = |s: &str| {
        s.parse::<u64>()
            .map_err(|e| CodegenError::new("llvm-size", format!("bad integer `{s}`: {e}")))
    };

    Ok(ElfSectionSizes {
        text: parse_u64(cols[0])?,
        data: parse_u64(cols[1])?,
        bss: parse_u64(cols[2])?,
        dec: parse_u64(cols[3])?,
        hex: cols[4].to_string(),
        raw_output: raw,
    })
}

/// Parse the ELF32 symbol table directly to verify zero undefined symbols and zero heap/libc calls.
pub fn inspect_elf_symbols(elf_path: &Path) -> Result<ElfSymbolAudit, CodegenError> {
    let bytes = fs::read(elf_path)
        .map_err(|e| CodegenError::new("elf-audit", format!("failed to read ELF: {e}")))?;
    if bytes.len() < 52 || &bytes[0..4] != b"\x7FELF" || bytes[4] != 1 {
        return Err(CodegenError::new(
            "elf-audit",
            "expected 32-bit little-endian ARM ELF binary",
        ));
    }

    let rd16 = |off: usize| -> u16 { u16::from_le_bytes([bytes[off], bytes[off + 1]]) };
    let rd32 = |off: usize| -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };

    let e_shoff = rd32(32) as usize;
    let e_shentsize = rd16(46) as usize;
    let e_shnum = rd16(48) as usize;

    let mut audit = ElfSymbolAudit::default();
    let forbidden = [
        "malloc", "calloc", "realloc", "free", "_sbrk", "sbrk", "__libc_start_main",
    ];

    for i in 0..e_shnum {
        let sh = e_shoff + i * e_shentsize;
        if sh + e_shentsize > bytes.len() {
            break;
        }
        let sh_type = rd32(sh + 4);
        // SHT_SYMTAB == 2
        if sh_type != 2 {
            continue;
        }
        let sym_off = rd32(sh + 16) as usize;
        let sym_size = rd32(sh + 20) as usize;
        let str_idx = rd32(sh + 24) as usize;
        let entsize = rd32(sh + 36) as usize;
        if entsize == 0 || str_idx >= e_shnum {
            continue;
        }
        let str_sh = e_shoff + str_idx * e_shentsize;
        let str_off = rd32(str_sh + 16) as usize;
        let str_size = rd32(str_sh + 20) as usize;
        if str_off + str_size > bytes.len() {
            continue;
        }
        let strtab = &bytes[str_off..str_off + str_size];

        let count = sym_size / entsize;
        // Skip entry 0 (STN_UNDEF null symbol)
        for s in 1..count {
            let so = sym_off + s * entsize;
            if so + 16 > bytes.len() {
                break;
            }
            let st_name = rd32(so) as usize;
            let st_info = bytes[so + 12];
            let st_shndx = rd16(so + 14);
            let st_type = st_info & 0x0f;
            // Skip STT_FILE (4) and STT_SECTION (3) and ARM mapping symbols ($t, $d)
            if st_type == 3 || st_type == 4 || st_name >= strtab.len() {
                continue;
            }
            let end = strtab[st_name..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| st_name + p)
                .unwrap_or(strtab.len());
            let name = String::from_utf8_lossy(&strtab[st_name..end]).to_string();
            if name.is_empty() || name.starts_with('$') {
                continue;
            }
            if forbidden.iter().any(|&bad| name == bad) {
                audit.heap_or_libc_symbols.push(name.clone());
            }
            if st_shndx == 0 {
                audit.undefined_symbols.push(name);
            } else {
                audit.defined_symbols.push(name);
            }
        }
    }

    Ok(audit)
}

/// Compile textual LLVM 23 `.ll` IR all the way to a freestanding `.elf` binary,
/// emitting `.ll` and `.ld` alongside `out_elf` and auditing `llvm-size` + ELF32 symbols.
pub fn compile_ll_to_elf(
    ll_ir: &str,
    out_elf: &Path,
    opts: &CodegenOptions,
) -> Result<ElfBuildResult, CodegenError> {
    if !matches!(
        opts.target,
        TargetArch::Thumbv7mNoneEabi
            | TargetArch::Riscv32Esp32C6
            | TargetArch::AvrAtmega328p
            | TargetArch::AvrAtmega4809
    ) {
        return Err(CodegenError::new(
            "build-elf",
            "bare-metal .elf linking requires Thumbv7mNoneEabi, Riscv32Esp32C6, AvrAtmega328p, or AvrAtmega4809",
        ));
    }
    if let Some(parent) = out_elf.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                CodegenError::new("io", format!("failed to create {}: {e}", parent.display()))
            })?;
        }
    }

    let ll_path = out_elf.with_extension("ll");
    let obj_path = out_elf.with_extension("o");
    let ld_path = out_elf.with_extension("ld");

    fs::write(&ll_path, ll_ir).map_err(|e| {
        CodegenError::new("io", format!("failed to write {}: {e}", ll_path.display()))
    })?;
    let ld_script = synthesize_linker_script(opts);
    fs::write(&ld_path, &ld_script).map_err(|e| {
        CodegenError::new("io", format!("failed to write {}: {e}", ld_path.display()))
    })?;

    compile_ll_to_obj(&ll_path, &obj_path, opts)?;
    link_obj_to_elf(&obj_path, &ld_path, out_elf)?;
    let sizes = inspect_elf_sizes(out_elf)?;
    let audit = inspect_elf_symbols(out_elf)?;

    if !audit.is_zero_heap_freestanding() {
        return Err(CodegenError::new(
            "elf-audit",
            format!(
                "ELF failed freestanding zero-heap audit: undefined={:?}, forbidden={:?}",
                audit.undefined_symbols, audit.heap_or_libc_symbols
            ),
        ));
    }

    Ok(ElfBuildResult {
        ll_path,
        obj_path,
        ld_path,
        elf_path: out_elf.to_path_buf(),
        sizes,
        audit,
    })
}

/// Run a freestanding Cortex-M3 `.elf` in `/bin/qemu-system-arm` (`lm3s6965evb`), capturing
/// UART0 telemetry from stdout and exiting cleanly via ARM semihosting (`bkpt #0xab`).
pub fn run_elf_in_qemu(
    elf_path: &Path,
    qemu_opts: &QemuRunOptions,
) -> Result<QemuExecOutput, CodegenError> {
    let qemu = find_tool("/bin/qemu-system-arm", "qemu-system-arm");
    let start = Instant::now();

    let mut child = Command::new(qemu)
        .arg("-M")
        .arg(&qemu_opts.machine)
        .arg("-cpu")
        .arg(&qemu_opts.cpu)
        .arg("-display")
        .arg("none")
        .arg("-monitor")
        .arg("none")
        .arg("-serial")
        .arg("stdio")
        .arg("-semihosting-config")
        .arg("enable=on,target=native")
        .arg("-kernel")
        .arg(elf_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| CodegenError::new("qemu", format!("failed to spawn `{qemu}`: {e}")))?;

    let mut stdout_pipe = child.stdout.take().unwrap();
    let mut stderr_pipe = child.stderr.take().unwrap();

    let out_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    });
    let err_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    });

    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {
                if start.elapsed() >= qemu_opts.timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let partial_out = out_handle.join().unwrap_or_default();
                    let partial_err = err_handle.join().unwrap_or_default();
                    return Err(CodegenError::new(
                        "qemu",
                        format!(
                            "QEMU timed out after {:?} (stdout={:?}, stderr={:?})",
                            qemu_opts.timeout, partial_out, partial_err
                        ),
                    ));
                }
                thread::sleep(Duration::from_millis(5));
            }
            Err(e) => {
                let _ = child.kill();
                return Err(CodegenError::new("qemu", format!("try_wait failed: {e}")));
            }
        }
    };

    let elapsed = start.elapsed();
    let stdout = out_handle.join().unwrap_or_default();
    let stderr = err_handle.join().unwrap_or_default();
    let exit_code = status.code().unwrap_or(-1);

    Ok(QemuExecOutput {
        stdout,
        stderr,
        exit_code,
        elapsed,
    })
}

/// Verify textual `.ll` IR using LLVM 23 `/bin/opt -passes=verify -disable-output`.
pub fn verify_ll_with_opt(ll_path: &Path) -> Result<(), CodegenError> {
    let opt = find_tool("/bin/opt", "opt");
    let out = Command::new(opt)
        .arg("-passes=verify")
        .arg("-disable-output")
        .arg(ll_path)
        .output()
        .map_err(|e| CodegenError::new("opt", format!("failed to invoke `{opt}`: {e}")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "opt",
            format!("opt verifier failed: {}", stderr.trim()),
        ));
    }
    Ok(())
}

/// Convert a linked ESP32-C6 `.elf` into a raw Direct-Boot `.bin` image via `/bin/llvm-objcopy -O binary`,
/// verifying the 8-byte `0xAEDB041D` x2 Direct-Boot magic header at offset `0x0`.
pub fn objcopy_elf_to_esp32c6_bin(
    elf_path: &Path,
    out_bin: &Path,
) -> Result<(u64, [u8; 8]), CodegenError> {
    if let Some(parent) = out_bin.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                CodegenError::new("io", format!("failed to create {}: {e}", parent.display()))
            })?;
        }
    }
    let objcopy = find_tool("/bin/llvm-objcopy", "llvm-objcopy");
    let out = Command::new(objcopy)
        .arg("-O")
        .arg("binary")
        .arg(elf_path)
        .arg(out_bin)
        .output()
        .map_err(|e| {
            CodegenError::new("llvm-objcopy", format!("failed to invoke `{objcopy}`: {e}"))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "llvm-objcopy",
            format!("llvm-objcopy exited with {}: {}", out.status, stderr.trim()),
        ));
    }

    let bytes = fs::read(out_bin).map_err(|e| {
        CodegenError::new("io", format!("failed to read {}: {e}", out_bin.display()))
    })?;
    if bytes.len() < 8 {
        return Err(CodegenError::new(
            "esp32c6-image",
            format!("generated .bin is too short ({} bytes)", bytes.len()),
        ));
    }
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&bytes[0..8]);
    if magic != crate::esp32c6::ESP32C6_DIRECT_BOOT_MAGIC {
        return Err(CodegenError::new(
            "esp32c6-image",
            format!(
                "invalid Direct-Boot header at offset 0: got {:02x?}, expected {:02x?}",
                magic,
                crate::esp32c6::ESP32C6_DIRECT_BOOT_MAGIC
            ),
        ));
    }
    Ok((bytes.len() as u64, magic))
}

/// Flash a Direct-Boot `.bin` image onto a physical ESP32-C6 at flash offset `0x0` using `esptool` v5.
pub fn flash_esp32c6_bin(bin_path: &Path, port: &str) -> Result<String, CodegenError> {
    let esptool = find_tool("/bin/esptool", "esptool");
    let out = Command::new(esptool)
        .arg("--chip")
        .arg("esp32c6")
        .arg("--port")
        .arg(port)
        .arg("--after")
        .arg("hard-reset")
        .arg("write-flash")
        .arg("0x0")
        .arg(bin_path)
        .output()
        .map_err(|e| CodegenError::new("esptool", format!("failed to invoke `{esptool}`: {e}")))?;

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        return Err(CodegenError::new(
            "esptool",
            format!("esptool exited with {}:\n{}", out.status, combined.trim()),
        ));
    }
    Ok(combined)
}

pub(crate) fn read_elf32_sym_addr(elf_path: &Path, target_name: &str) -> Option<u32> {
    let bytes = fs::read(elf_path).ok()?;
    if bytes.len() < 52 || &bytes[0..4] != b"\x7FELF" || bytes[4] != 1 {
        return None;
    }
    let rd16 = |off: usize| -> u16 { u16::from_le_bytes([bytes[off], bytes[off + 1]]) };
    let rd32 = |off: usize| -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let e_shoff = rd32(32) as usize;
    let e_shentsize = rd16(46) as usize;
    let e_shnum = rd16(48) as usize;

    for i in 0..e_shnum {
        let sh = e_shoff + i * e_shentsize;
        if sh + e_shentsize > bytes.len() {
            break;
        }
        if rd32(sh + 4) != 2 {
            continue;
        }
        let sym_off = rd32(sh + 16) as usize;
        let sym_size = rd32(sh + 20) as usize;
        let str_idx = rd32(sh + 24) as usize;
        let entsize = rd32(sh + 36) as usize;
        if entsize == 0 || str_idx >= e_shnum {
            continue;
        }
        let str_sh = e_shoff + str_idx * e_shentsize;
        let str_off = rd32(str_sh + 16) as usize;
        let str_size = rd32(str_sh + 20) as usize;
        if str_off + str_size > bytes.len() {
            continue;
        }
        let strtab = &bytes[str_off..str_off + str_size];
        let count = sym_size / entsize;
        for s in 1..count {
            let so = sym_off + s * entsize;
            if so + 16 > bytes.len() {
                break;
            }
            let st_name = rd32(so) as usize;
            if st_name >= strtab.len() {
                continue;
            }
            let end = strtab[st_name..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| st_name + p)
                .unwrap_or(strtab.len());
            if &strtab[st_name..end] == target_name.as_bytes() {
                return Some(rd32(so + 4));
            }
        }
    }
    None
}

/// Convert a linked ATmega328P `.elf` into a raw Flash `.bin` image via `/bin/llvm-objcopy -O binary`,
/// verifying reset `jmp` vector at `0x0000`, `<= 32,256 B` Optiboot Flash budget, and `< 1,792 B`
/// static SRAM budget (reserving `>= 256 B` hardware call stack in the 2 KB SRAM).
pub fn objcopy_elf_to_avr_bin(
    elf_path: &Path,
    out_bin: &Path,
) -> Result<(u64, u64), CodegenError> {
    if let Some(parent) = out_bin.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                CodegenError::new("io", format!("failed to create {}: {e}", parent.display()))
            })?;
        }
    }
    let objcopy = find_tool("/bin/llvm-objcopy", "llvm-objcopy");
    let out = Command::new(objcopy)
        .arg("-O")
        .arg("binary")
        .arg(elf_path)
        .arg(out_bin)
        .output()
        .map_err(|e| {
            CodegenError::new("llvm-objcopy", format!("failed to invoke `{objcopy}`: {e}"))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(CodegenError::new(
            "llvm-objcopy",
            format!("llvm-objcopy exited with {}: {}", out.status, stderr.trim()),
        ));
    }

    let bytes = fs::read(out_bin).map_err(|e| {
        CodegenError::new("io", format!("failed to read {}: {e}", out_bin.display()))
    })?;
    if bytes.len() < 4 || bytes.len() > 32_256 {
        return Err(CodegenError::new(
            "avr-image",
            format!(
                "generated AVR .bin size {} B is out of ATmega328P Optiboot bounds (4..=32256 B)",
                bytes.len()
            ),
        ));
    }
    let word0 = u16::from_le_bytes([bytes[0], bytes[1]]);
    if (word0 & 0xFE0E) != 0x940C {
        return Err(CodegenError::new(
            "avr-image",
            format!("expected AVR `jmp` instruction at 0x0000, got word 0x{word0:04X}"),
        ));
    }

    let sram_static_bytes = match (
        read_elf32_sym_addr(elf_path, "__sdata"),
        read_elf32_sym_addr(elf_path, "__ebss"),
    ) {
        (Some(sdata), Some(ebss)) if ebss >= sdata => u64::from(ebss - sdata),
        _ => {
            let sizes = inspect_elf_sizes(elf_path)?;
            sizes.data + sizes.bss
        }
    };
    if sram_static_bytes >= 1_792 {
        return Err(CodegenError::new(
            "avr-image",
            format!(
                "static SRAM footprint {sram_static_bytes} B leaves fewer than 256 B of stack in 2,048 B ATmega328P SRAM"
            ),
        ));
    }

    Ok((bytes.len() as u64, sram_static_bytes))
}

const AVR_STK500V1_FLASH_SCRIPT: &str = r#"
import sys, time, serial

port, bin_path = sys.argv[1], sys.argv[2]
data = open(bin_path, "rb").read()
if len(data) < 4 or len(data) > 32256:
    raise SystemExit(f"AVR image size {len(data)} B out of Optiboot bounds (4..=32256 B)")

page_size = 128
padded_len = ((len(data) + page_size - 1) // page_size) * page_size
padded = data + b"\xFF" * (padded_len - len(data))
pages = padded_len // page_size

def read_exact(ser, n, timeout=0.6):
    buf = bytearray()
    deadline = time.monotonic() + timeout
    while len(buf) < n and time.monotonic() < deadline:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
    return bytes(buf)

def cmd(ser, payload, resp_len, what):
    ser.reset_input_buffer()
    ser.write(payload)
    ser.flush()
    resp = read_exact(ser, resp_len)
    if len(resp) != resp_len or resp[0] != 0x14 or resp[-1] != 0x10:
        raise RuntimeError(f"STK500v1 {what} failed: got {resp.hex(' ').upper()}")
    return resp

ser = serial.Serial(port, 115200, timeout=0.1)
try:
    ser.dtr = False
    time.sleep(0.05)
    ser.dtr = True
    time.sleep(0.25)
    ser.reset_input_buffer()

    synced = False
    for _ in range(8):
        ser.reset_input_buffer()
        ser.write(b"\x30\x20")
        ser.flush()
        r = read_exact(ser, 2, 0.25)
        if r == b"\x14\x10":
            synced = True
            break
        time.sleep(0.05)
    if not synced:
        raise RuntimeError("STK_GET_SYNC failed after 8 attempts")

    maj = cmd(ser, b"\x41\x81\x20", 3, "GET_PARAMETER SW_MAJOR")[1]
    minr = cmd(ser, b"\x41\x82\x20", 3, "GET_PARAMETER SW_MINOR")[1]
    sign = cmd(ser, b"\x75\x20", 5, "STK_READ_SIGN")[1:4]
    if sign != b"\x1e\x95\x0f":
        raise RuntimeError(f"Expected ATmega328P signature 1E 95 0F, got {sign.hex(' ').upper()}")

    print(f"Optiboot {maj}.{minr} synced on {port}: signature={sign.hex(' ').upper()} (ATmega328P)")
    cmd(ser, b"\x50\x20", 2, "STK_ENTER_PROGMODE")

    for p in range(pages):
        byte_addr = p * page_size
        word_addr = byte_addr // 2
        addr_cmd = bytes([0x55, word_addr & 0xFF, (word_addr >> 8) & 0xFF, 0x20])
        cmd(ser, addr_cmd, 2, f"STK_LOAD_ADDRESS(write page {p})")
        chunk = padded[byte_addr : byte_addr + page_size]
        prog_cmd = bytes([0x64, 0x00, page_size, 0x46]) + chunk + b"\x20"
        cmd(ser, prog_cmd, 2, f"STK_PROG_PAGE(page {p})")

    for p in range(pages):
        byte_addr = p * page_size
        word_addr = byte_addr // 2
        addr_cmd = bytes([0x55, word_addr & 0xFF, (word_addr >> 8) & 0xFF, 0x20])
        cmd(ser, addr_cmd, 2, f"STK_LOAD_ADDRESS(read page {p})")
        read_cmd = bytes([0x74, 0x00, page_size, 0x46, 0x20])
        resp = cmd(ser, read_cmd, 2 + page_size, f"STK_READ_PAGE(page {p})")
        got = resp[1:-1]
        expected = padded[byte_addr : byte_addr + page_size]
        if got != expected:
            raise RuntimeError(f"Flash verify mismatch in page {p} @ 0x{byte_addr:04X}")

    cmd(ser, b"\x51\x20", 2, "STK_LEAVE_PROGMODE")
    print(f"Flashed and verified {pages} page(s) ({len(data)} bytes) at 0x0000..0x{padded_len:04X}")
finally:
    ser.close()
"#;

/// Flash a raw AVR Flash `.bin` image onto an Arduino Uno R3 (`ATmega328P` / Optiboot) at `0x0000`
/// using STK500v1 over `pyserial` (`115200` baud) with full page readback verification.
pub fn flash_avr_uno_bin(bin_path: &Path, port: &str) -> Result<String, CodegenError> {
    let out = Command::new("python3")
        .arg("-c")
        .arg(AVR_STK500V1_FLASH_SCRIPT)
        .arg(port)
        .arg(bin_path)
        .output()
        .map_err(|e| CodegenError::new("avr-flash", format!("failed to invoke `python3`: {e}")))?;

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        return Err(CodegenError::new(
            "avr-flash",
            format!("AVR STK500v1 flasher exited with {}:\n{}", out.status, combined.trim()),
        ));
    }
    Ok(combined)
}

pub use crate::avr_mega0::{flash_avr_nano_every_bin, objcopy_elf_to_avr_nano_every_bin};
