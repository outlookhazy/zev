//! Toolchain driver for LLVM 23 (`llc`), LLD (`ld.lld`), `llvm-size`, ELF32 symbol auditing,
//! and bare-metal ARM Cortex-M3 execution in `qemu-system-arm`.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use crate::target_spec::{discover_spec_roots, CpuIsa, ImageFormat, IrqModel, TargetSpec};

/// Compatibility target alias mapping legacy hardcoded enum variants onto declarative `chips/` + `boards/` TOMLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    Thumbv7mNoneEabi,
    Thumbv6mRp2040,
    Riscv32Esp32C6,
    AvrAtmega328p,
    AvrAtmega4809,
    HostX86_64Linux,
}

impl TargetArch {
    pub const fn is_avr(self) -> bool {
        matches!(self, Self::AvrAtmega328p | Self::AvrAtmega4809)
    }

    pub const fn is_arm_eabi(self) -> bool {
        matches!(self, Self::Thumbv7mNoneEabi | Self::Thumbv6mRp2040)
    }

    pub const fn avr_mcpu(self) -> &'static str {
        match self {
            Self::AvrAtmega328p => "atmega328p",
            Self::AvrAtmega4809 => "atmega4809",
            Self::Thumbv7mNoneEabi
            | Self::Thumbv6mRp2040
            | Self::Riscv32Esp32C6
            | Self::HostX86_64Linux => "",
        }
    }

    pub const fn triple(self) -> &'static str {
        match self {
            Self::Thumbv7mNoneEabi => "thumbv7m-none-eabi",
            Self::Thumbv6mRp2040 => "thumbv6m-none-eabi",
            Self::Riscv32Esp32C6 => "riscv32-unknown-none-elf",
            Self::AvrAtmega328p | Self::AvrAtmega4809 => "avr",
            Self::HostX86_64Linux => "x86_64-unknown-linux-gnu",
        }
    }

    pub const fn datalayout(self) -> &'static str {
        match self {
            Self::Thumbv7mNoneEabi | Self::Thumbv6mRp2040 => {
                "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
            }
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
            Self::Thumbv7mNoneEabi | Self::Thumbv6mRp2040 | Self::Riscv32Esp32C6 => 32,
            Self::AvrAtmega328p | Self::AvrAtmega4809 => 16,
            Self::HostX86_64Linux => 64,
        }
    }

    pub const fn usize_llvm_ty(self) -> &'static str {
        match self {
            Self::Thumbv7mNoneEabi | Self::Thumbv6mRp2040 | Self::Riscv32Esp32C6 => "i32",
            Self::AvrAtmega328p | Self::AvrAtmega4809 => "i16",
            Self::HostX86_64Linux => "i64",
        }
    }

    pub const fn fn_ptr_llvm_ty(self) -> &'static str {
        match self {
            Self::AvrAtmega328p | Self::AvrAtmega4809 => "ptr addrspace(1)",
            Self::Thumbv7mNoneEabi
            | Self::Thumbv6mRp2040
            | Self::Riscv32Esp32C6
            | Self::HostX86_64Linux => "ptr",
        }
    }

    pub const fn fn_addrspace_attr(self) -> &'static str {
        if self.is_avr() {
            " addrspace(1)"
        } else {
            ""
        }
    }

    pub const fn call_kw(self) -> &'static str {
        if self.is_avr() {
            "call addrspace(1)"
        } else {
            "call"
        }
    }
}

/// Configuration for LLVM 23 IR generation and bare-metal linking.
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    pub target: TargetArch,
    /// Optional declarative [`TargetSpec`] loaded from `chips/*.toml` + `boards/*.toml`.
    pub spec: Option<TargetSpec>,
    /// Target-bound HAL directory override (e.g. `"hal/stm32f3"`), defaulting to `"hal/sim"`.
    pub hal_dir: Option<String>,
    /// Target-bound board module override (e.g. `"bsp/nucleo_f334r8"`), defaulting to `"bsp/sim"`.
    pub board_module: Option<String>,
    /// Flash base address override.
    pub flash_origin: u32,
    /// Flash size in bytes override.
    pub flash_len: u32,
    /// SRAM base address override.
    pub ram_origin: u32,
    /// SRAM size in bytes override.
    pub ram_len: u32,
    /// Optimization flag passed to `opt`/`llc` (`"-O2"` by default; `"-Os"` on AVR).
    pub opt_level: String,
    /// Optional deterministic tick cap (`0` = unlimited, `> 0` = auto-exit at tick cap if reached).
    pub max_ticks: u32,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            target: TargetArch::Thumbv7mNoneEabi,
            spec: None,
            hal_dir: None,
            board_module: None,
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
    /// Construct host x86_64 Linux simulation options matching [`TargetSpec::host_linux`].
    pub fn host_linux() -> Self {
        Self::from_spec(TargetSpec::host_linux())
    }

    /// Construct codegen options directly from any declarative [`TargetSpec`].
    pub fn from_spec(spec: TargetSpec) -> Self {
        let target = match spec.isa {
            CpuIsa::ArmCortexM => {
                if spec.code_in_ram {
                    TargetArch::Thumbv6mRp2040
                } else {
                    TargetArch::Thumbv7mNoneEabi
                }
            }
            CpuIsa::Avr => {
                if !spec.rodata_in_ram {
                    TargetArch::AvrAtmega4809
                } else {
                    TargetArch::AvrAtmega328p
                }
            }
            CpuIsa::Riscv32 => TargetArch::Riscv32Esp32C6,
            CpuIsa::HostX86_64Linux => TargetArch::HostX86_64Linux,
        };
        Self {
            target,
            hal_dir: spec.hal_dir.clone(),
            board_module: spec.board_module.clone(),
            flash_origin: spec.flash_base,
            flash_len: spec.effective_flash_size(),
            ram_origin: spec.ram_base,
            ram_len: spec.ram_size,
            opt_level: spec.opt_level.clone(),
            max_ticks: 0,
            spec: Some(spec),
        }
    }

    /// Resolve the active [`TargetSpec`], falling back to the built-in TOML mapped by `self.target`.
    pub fn effective_spec(&self) -> Result<TargetSpec, CodegenError> {
        let mut s = if let Some(spec) = &self.spec {
            spec.clone()
        } else {
            let mut resolved = match self.target {
                TargetArch::HostX86_64Linux => TargetSpec::host_linux(),
                TargetArch::Thumbv7mNoneEabi => TargetSpec::resolve(None, Some("lm3s6965_evb"))?,
                TargetArch::Thumbv6mRp2040 => TargetSpec::resolve(None, Some("rpi_pico_w"))?,
                TargetArch::Riscv32Esp32C6 => TargetSpec::resolve(None, Some("esp32c6_pico"))?,
                TargetArch::AvrAtmega328p => TargetSpec::resolve(None, Some("arduino_uno"))?,
                TargetArch::AvrAtmega4809 => TargetSpec::resolve(None, Some("arduino_nano_every"))?,
            };
            let def = Self::default();
            if self.target == TargetArch::Thumbv7mNoneEabi
                && (self.flash_origin != def.flash_origin
                    || self.flash_len != def.flash_len
                    || self.ram_origin != def.ram_origin
                    || self.ram_len != def.ram_len)
            {
                resolved.flash_base = self.flash_origin;
                resolved.flash_size = self.flash_len;
                resolved.usable_flash_size = None;
                resolved.ram_base = self.ram_origin;
                resolved.ram_size = self.ram_len;
                resolved.sram_budget_bytes = self.ram_len;
            }
            resolved
        };
        s.opt_level = self.opt_level.clone();
        Ok(s)
    }

    pub fn esp32c6() -> Self {
        match TargetSpec::resolve(None, Some("esp32c6_pico")) {
            Ok(spec) => Self::from_spec(spec),
            Err(_) => Self {
                target: TargetArch::Riscv32Esp32C6,
                spec: None,
                hal_dir: None,
                board_module: None,
                flash_origin: 0x4200_0000,
                flash_len: 4 * 1024 * 1024,
                ram_origin: 0x4080_0000,
                ram_len: 448 * 1024,
                opt_level: "-O2".to_string(),
                max_ticks: 0,
            },
        }
    }

    pub fn lm3s6965() -> Self {
        match TargetSpec::resolve(None, Some("lm3s6965_evb")) {
            Ok(spec) => Self::from_spec(spec),
            Err(_) => Self::default(),
        }
    }

    pub fn avr_uno() -> Self {
        match TargetSpec::resolve(None, Some("arduino_uno")) {
            Ok(spec) => Self::from_spec(spec),
            Err(_) => Self {
                target: TargetArch::AvrAtmega328p,
                spec: None,
                hal_dir: None,
                board_module: None,
                flash_origin: 0x0000_0000,
                flash_len: 32_256,
                ram_origin: 0x0000_0100,
                ram_len: 2_048,
                opt_level: "-Os".to_string(),
                max_ticks: 0,
            },
        }
    }

    pub fn avr_nano_every() -> Self {
        match TargetSpec::resolve(None, Some("arduino_nano_every")) {
            Ok(spec) => {
                let mut opts = Self::from_spec(spec);
                opts.target = TargetArch::AvrAtmega4809;
                opts
            }
            Err(_) => Self {
                target: TargetArch::AvrAtmega4809,
                spec: None,
                hal_dir: None,
                board_module: None,
                flash_origin: 0x0000_0000,
                flash_len: 49_152,
                ram_origin: 0x0000_2800,
                ram_len: 6_144,
                opt_level: "-Os".to_string(),
                max_ticks: 0,
            },
        }
    }

    pub fn rp2040() -> Self {
        match TargetSpec::resolve(None, Some("rpi_pico_w")) {
            Ok(spec) => Self::from_spec(spec),
            Err(_) => Self {
                target: TargetArch::Thumbv6mRp2040,
                spec: None,
                hal_dir: None,
                board_module: None,
                flash_origin: 0x2000_0000,
                flash_len: 264 * 1024,
                ram_origin: 0x2000_0000,
                ram_len: 264 * 1024,
                opt_level: "-O2".to_string(),
                max_ticks: 0,
            },
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

/// Build output returned by [`build_target_image`].
#[derive(Debug, Clone)]
pub struct TargetBuildArtifact {
    pub elf: ElfBuildResult,
    pub image_path: PathBuf,
    pub bin_path: Option<PathBuf>,
    pub uf2_path: Option<PathBuf>,
    pub bin_size: u64,
    pub image_size: u64,
    pub uf2_blocks: u32,
    pub sram_static_bytes: u64,
}

/// Artifacts produced by `build_esp32c6_image` (legacy wrapper).
#[derive(Debug, Clone)]
pub struct Esp32C6ImageResult {
    pub elf: ElfBuildResult,
    pub bin_path: PathBuf,
    pub bin_size: u64,
    pub magic_header: [u8; 8],
}

/// Artifacts produced by `build_avr_uno_image` / `build_avr_nano_every_image` (legacy wrapper).
#[derive(Debug, Clone)]
pub struct AvrImageResult {
    pub elf: ElfBuildResult,
    pub bin_path: PathBuf,
    pub bin_size: u64,
    pub sram_static_bytes: u64,
}

/// Artifacts produced by `build_rp2040_image` (legacy wrapper).
#[derive(Debug, Clone)]
pub struct Rp2040ImageResult {
    pub elf: ElfBuildResult,
    pub bin_path: PathBuf,
    pub uf2_path: PathBuf,
    pub bin_size: u64,
    pub uf2_size: u64,
    pub uf2_blocks: u32,
}

/// Options for executing a Cortex-M `.elf` inside `qemu-system-arm`.
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

impl QemuRunOptions {
    pub fn from_spec(spec: &TargetSpec) -> Result<Self, CodegenError> {
        let machine = spec.qemu_machine.clone().ok_or_else(|| {
            CodegenError::new(
                "qemu",
                format!(
                    "target '{}' does not declare `[run].qemu_machine`",
                    spec.board_name.as_deref().unwrap_or(&spec.chip_name)
                ),
            )
        })?;
        Ok(Self {
            machine,
            cpu: spec.cpu.clone(),
            timeout: Duration::from_millis(5_000),
        })
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

/// Synthesize a declarative LLD linker script purely from [`TargetSpec`].
pub fn synthesize_target_linker_script(spec: &TargetSpec) -> String {
    if spec.isa.is_avr() {
        let data_ram_offset = match spec.irq {
            IrqModel::AvrVectored {
                data_ram_offset, ..
            } => data_ram_offset,
            _ => 0x0080_0000,
        };
        let ram_vma = data_ram_offset.wrapping_add(spec.ram_base);
        let flash_orig = spec.flash_base;
        let flash_len = spec.effective_flash_size();
        let ram_len = spec.ram_size;
        let rodata_in_text = if spec.rodata_in_ram {
            ""
        } else {
            "\n    *(.rodata .rodata.*)"
        };
        let rodata_in_data = if spec.rodata_in_ram {
            "    *(.rodata .rodata.*)\n"
        } else {
            ""
        };
        return format!(
            r#"/* Auto-synthesized by zev-codegen-llvm for AVR ({cpu}) */
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
    *(.text .text.*){rodata_in_text}
    . = ALIGN(2);
    __sidata = .;
  }} > FLASH

  .data 0x{ram_vma:08X} : AT(__sidata)
  {{
    __sdata = .;
{rodata_in_data}    *(.data .data.*)
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
            cpu = spec.cpu
        );
    }

    if spec.code_in_ram {
        let ram_orig = spec.ram_base;
        let ram_len = spec.ram_size;
        let iram_keep = if spec.iram_in_data {
            "    KEEP(*(.iram1 .iram1.*))\n"
        } else {
            ""
        };
        return format!(
            r#"/* Auto-synthesized by zev-codegen-llvm for RAM-resident {cpu} */
MEMORY
{{
  SRAM (rwx) : ORIGIN = 0x{ram_orig:08X}, LENGTH = {ram_len}
}}

ENTRY(_start)

SECTIONS
{{
  .entry ORIGIN(SRAM) :
  {{
    KEEP(*(.entry))
  }} > SRAM

  .isr_vector ALIGN(256) :
  {{
    KEEP(*(.isr_vector))
  }} > SRAM

  .text : ALIGN(4)
  {{
    *(.text .text.*)
    *(.rodata .rodata.* .srodata .srodata.*)
    . = ALIGN(4);
  }} > SRAM

  .data : ALIGN(4)
  {{
    __sidata = .;
    __sdata = .;
{iram_keep}    *(.data .data.* .sdata .sdata.*)
    . = ALIGN(4);
    __edata = .;
  }} > SRAM

  .bss (NOLOAD) : ALIGN(4)
  {{
    __sbss = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(4);
    __ebss = .;
  }} > SRAM

  /DISCARD/ :
  {{
    *(.ARM.exidx*)
    *(.ARM.extab*)
    *(.ARM.attributes*)
    *(.note.GNU-stack)
    *(.eh_frame*)
    *(.riscv.attributes)
    *(.comment)
    *(.note*)
  }}
}}
"#,
            cpu = spec.cpu
        );
    }

    let flash_orig = spec.flash_base;
    let flash_len = spec.effective_flash_size();
    let ram_orig = spec.ram_base;
    let ram_len = spec.ram_size;
    let entry_sym = if spec.isa == CpuIsa::ArmCortexM && spec.header_magic_words.is_empty() {
        "Reset_Handler"
    } else {
        "_start"
    };
    let mut magic_ir = String::new();
    for w in &spec.header_magic_words {
        magic_ir.push_str(&format!("    LONG(0x{w:08X});\n"));
    }
    let iram_keep = if spec.iram_in_data {
        "    KEEP(*(.iram1 .iram1.*))\n"
    } else {
        ""
    };
    format!(
        r#"/* Auto-synthesized by zev-codegen-llvm for {cpu} */
MEMORY
{{
  FLASH (rx)  : ORIGIN = 0x{flash_orig:08X}, LENGTH = {flash_len}
  RAM   (rwx) : ORIGIN = 0x{ram_orig:08X}, LENGTH = {ram_len}
}}

ENTRY({entry_sym})

SECTIONS
{{
  .text ORIGIN(FLASH) :
  {{
{magic_ir}    KEEP(*(.entry))
    KEEP(*(.isr_vector))
    *(.text .text.*)
    *(.rodata .rodata.* .srodata .srodata.*)
    . = ALIGN(4);
    __sidata = .;
  }} > FLASH

  .data : AT(__sidata)
  {{
    . = ALIGN(4);
    __sdata = .;
{iram_keep}    *(.data .data.* .sdata .sdata.*)
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
    *(.ARM.exidx*)
    *(.ARM.extab*)
    *(.ARM.attributes*)
    *(.note.GNU-stack)
    *(.eh_frame*)
    *(.riscv.attributes)
    *(.comment)
    *(.note*)
  }}
}}
"#,
        cpu = spec.cpu
    )
}

/// Legacy compatibility linker script helper delegating to [`synthesize_target_linker_script`].
pub fn synthesize_linker_script(opts: &CodegenOptions) -> String {
    match opts.effective_spec() {
        Ok(spec) => synthesize_target_linker_script(&spec),
        Err(_) => synthesize_target_linker_script(&TargetSpec::host_linux()),
    }
}

/// Minimum supported upstream LLVM / LLD major version for `zev-codegen-llvm`.
pub const MIN_LLVM_MAJOR: u32 = 23;

fn parse_leading_major(s: &str) -> Option<u32> {
    let trimmed = s.trim_start();
    let digits_len = trimmed.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits_len == 0 {
        return None;
    }
    let after = &trimmed[digits_len..];
    if !after.is_empty()
        && !after.starts_with('.')
        && !after.starts_with(' ')
        && !after.starts_with('-')
    {
        return None;
    }
    trimmed[..digits_len].parse::<u32>().ok()
}

/// Extract `(major_version, summary_line)` from `opt`/`llc`/`ld.lld`/`llvm-*`/`clang --version` output.
pub fn parse_llvm_major_version(version_output: &str) -> Option<(u32, String)> {
    for raw_line in version_output.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if lower.contains("apple clang") || lower.contains("apple llvm") {
            continue;
        }
        for marker in ["llvm version ", "clang version "] {
            if let Some(idx) = lower.find(marker) {
                if let Some(maj) = parse_leading_major(&line[idx + marker.len()..]) {
                    return Some((maj, line.to_string()));
                }
            }
        }
        for (idx, _) in lower.match_indices("lld ") {
            if idx == 0 || lower.as_bytes()[idx - 1].is_ascii_whitespace() {
                if let Some(maj) = parse_leading_major(&line[idx + 4..]) {
                    return Some((maj, line.to_string()));
                }
            }
        }
    }
    None
}

fn probe_llvm_version(bin: &Path) -> Option<(u32, String)> {
    type CacheKey = (PathBuf, Option<SystemTime>, Option<u64>);
    type VersionCache = Mutex<HashMap<CacheKey, Option<(u32, String)>>>;
    static VERSION_CACHE: OnceLock<VersionCache> = OnceLock::new();

    let meta = fs::metadata(bin).ok();
    let key: CacheKey = (
        bin.to_path_buf(),
        meta.as_ref().and_then(|m| m.modified().ok()),
        meta.as_ref().map(|m| m.len()),
    );

    let cache = VERSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
        .cloned()
    {
        return cached;
    }

    let parsed = Command::new(bin)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .ok()
        .and_then(|out| {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            parse_llvm_major_version(&stdout).or_else(|| parse_llvm_major_version(&stderr))
        });

    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, parsed.clone());
    parsed
}

fn is_llvm_backend_tool(name: &str) -> bool {
    matches!(
        name,
        "opt"
            | "llc"
            | "ld.lld"
            | "llvm-size"
            | "llvm-objcopy"
            | "llvm-objdump"
            | "llvm-readobj"
            | "llvm-nm"
    )
}

fn is_llvm_tool(name: &str) -> bool {
    name == "clang" || is_llvm_backend_tool(name)
}

fn exe_candidate(dir: &Path, base: &str) -> Option<PathBuf> {
    let p = dir.join(base);
    if p.is_file() {
        return Some(p);
    }
    #[cfg(windows)]
    {
        let with_exe = dir.join(format!("{base}.exe"));
        if with_exe.is_file() {
            return Some(with_exe);
        }
    }
    None
}

fn find_tool_in_dirs(
    preferred_abs: &str,
    name: &str,
    env_prefix: Option<&Path>,
    path_dirs: &[PathBuf],
    extra_sys_dirs: &[PathBuf],
    versioned_dirs: fn(u32) -> Vec<PathBuf>,
) -> String {
    if let Some(p) = env_prefix {
        for dir in [p.join("bin"), p.to_path_buf()] {
            if let Some(found) = exe_candidate(&dir, name) {
                return found.to_string_lossy().into_owned();
            }
            if is_llvm_tool(name) {
                for ver in (MIN_LLVM_MAJOR..=30).rev() {
                    let vname = format!("{name}-{ver}");
                    if let Some(found) = exe_candidate(&dir, &vname) {
                        return found.to_string_lossy().into_owned();
                    }
                }
            }
        }
    }

    if !is_llvm_tool(name) {
        for dir in path_dirs {
            if let Some(found) = exe_candidate(dir, name) {
                return found.to_string_lossy().into_owned();
            }
        }
        if !preferred_abs.is_empty() && Path::new(preferred_abs).is_file() {
            return preferred_abs.to_string();
        }
        return name.to_string();
    }

    let mut older_fallback: Option<PathBuf> = None;
    let mut consider = |cand: PathBuf| -> Option<String> {
        match probe_llvm_version(&cand) {
            Some((maj, _)) if maj >= MIN_LLVM_MAJOR => Some(cand.to_string_lossy().into_owned()),
            Some(_) => {
                if older_fallback.is_none() {
                    older_fallback = Some(cand);
                }
                None
            }
            None if name == "clang" => Some(cand.to_string_lossy().into_owned()),
            None => {
                if older_fallback.is_none() {
                    older_fallback = Some(cand);
                }
                None
            }
        }
    };

    for dir in path_dirs {
        if let Some(found) = exe_candidate(dir, name) {
            if let Some(ok) = consider(found) {
                return ok;
            }
        }
    }

    if !preferred_abs.is_empty() && Path::new(preferred_abs).is_file() {
        if let Some(ok) = consider(PathBuf::from(preferred_abs)) {
            return ok;
        }
    }

    for ver in (MIN_LLVM_MAJOR..=30).rev() {
        let vname = format!("{name}-{ver}");
        for dir in path_dirs.iter().chain(extra_sys_dirs.iter()) {
            if let Some(found) = exe_candidate(dir, &vname) {
                if let Some(ok) = consider(found) {
                    return ok;
                }
            }
        }
        for prefix_dir in versioned_dirs(ver) {
            for cand_name in [name, vname.as_str()] {
                if let Some(found) = exe_candidate(&prefix_dir, cand_name) {
                    if let Some(ok) = consider(found) {
                        return ok;
                    }
                }
            }
        }
    }

    for brew_dir in extra_sys_dirs {
        if let Some(found) = exe_candidate(brew_dir, name) {
            if let Some(ok) = consider(found) {
                return ok;
            }
        }
    }

    if let Some(old) = older_fallback {
        return old.to_string_lossy().into_owned();
    }

    for ver in (14..MIN_LLVM_MAJOR).rev() {
        let vname = format!("{name}-{ver}");
        for dir in path_dirs.iter().chain(extra_sys_dirs.iter()) {
            if let Some(found) = exe_candidate(dir, &vname) {
                return found.to_string_lossy().into_owned();
            }
        }
        for llvm_dir in versioned_dirs(ver) {
            if let Some(found) = exe_candidate(&llvm_dir, name) {
                return found.to_string_lossy().into_owned();
            }
        }
    }

    name.to_string()
}

/// Per-version LLVM install prefixes probed on the host (Debian/Ubuntu, Arch-style, Homebrew).
fn system_versioned_llvm_dirs(ver: u32) -> Vec<PathBuf> {
    vec![
        PathBuf::from(format!("/usr/lib/llvm-{ver}/bin")),
        PathBuf::from(format!("/usr/lib/llvm{ver}/bin")),
        PathBuf::from(format!("/opt/homebrew/opt/llvm@{ver}/bin")),
    ]
}

/// Locate an external LLVM/QEMU executable portably, preferring LLVM >= [`MIN_LLVM_MAJOR`] (`23`):
/// `ZEV_LLVM_PREFIX` -> `PATH` (>= 23) -> `preferred_abs` (>= 23) -> versioned `23+` -> fallback.
pub fn find_tool(preferred_abs: &str, name: &str) -> String {
    let prefix = std::env::var_os("ZEV_LLVM_PREFIX").map(PathBuf::from);
    let path_dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|v| std::env::split_paths(&v).collect())
        .unwrap_or_default();
    let sys_dirs = [
        PathBuf::from("/usr/bin"),
        PathBuf::from("/bin"),
        PathBuf::from("/opt/homebrew/opt/llvm/bin"),
    ];
    find_tool_in_dirs(
        preferred_abs,
        name,
        prefix.as_deref(),
        &path_dirs,
        &sys_dirs,
        system_versioned_llvm_dirs,
    )
}

/// Verify that `tool_path` is at least LLVM [`MIN_LLVM_MAJOR`] (`23`), returning a clear
/// [`CodegenError`] when an older LLVM/LLD toolchain is detected.
pub fn check_llvm_tool_version(tool_path: &str, tool_name: &str) -> Result<(), CodegenError> {
    if !is_llvm_backend_tool(tool_name) {
        return Ok(());
    }
    if let Some((major, ver_line)) = probe_llvm_version(Path::new(tool_path)) {
        if major < MIN_LLVM_MAJOR {
            return Err(CodegenError::new(
                "toolchain",
                format!(
                    "LLVM {MIN_LLVM_MAJOR} or newer is required for `{tool_name}`, but `{tool_path}` reports `{ver_line}`; install LLVM {MIN_LLVM_MAJOR}+ (e.g. from https://apt.llvm.org) or set ZEV_LLVM_PREFIX to an LLVM {MIN_LLVM_MAJOR}+ installation"
                ),
            ));
        }
    }
    Ok(())
}

/// Resolve an LLVM backend tool and verify that its version is >= [`MIN_LLVM_MAJOR`] (`23`).
pub fn require_llvm_tool(preferred_abs: &str, name: &str) -> Result<String, CodegenError> {
    let resolved = find_tool(preferred_abs, name);
    check_llvm_tool_version(&resolved, name)?;
    Ok(resolved)
}

/// Optimize textual LLVM `.ll` IR via `opt` (`default<O1|O2|O3>`).
pub fn optimize_ll_with_opt(
    in_ll: &Path,
    out_ll: &Path,
    opts: &CodegenOptions,
) -> Result<(), CodegenError> {
    let spec = opts.effective_spec()?;
    let pipeline = opts.opt_passes_pipeline()?;
    if let Some(parent) = out_ll.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                CodegenError::new("io", format!("failed to create {}: {e}", parent.display()))
            })?;
        }
    }

    let opt = require_llvm_tool("/bin/opt", "opt")?;
    let mut cmd = Command::new(&opt);
    cmd.arg(format!("-mtriple={}", spec.triple));
    if spec.isa != CpuIsa::HostX86_64Linux && !spec.cpu.is_empty() {
        cmd.arg(format!("-mcpu={}", spec.cpu));
    }
    if !spec.features.is_empty() {
        cmd.arg(format!("-mattr={}", spec.features));
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

/// Compile textual LLVM `.ll` IR to an ELF object file `.o` via `opt` + `llc`.
pub fn compile_ll_to_obj(
    ll_path: &Path,
    obj_path: &Path,
    opts: &CodegenOptions,
) -> Result<(), CodegenError> {
    let spec = opts.effective_spec()?;
    let llc_flag = opts.llc_opt_flag()?;
    let opt_ll_path = obj_path.with_extension("opt.ll");
    let llc_input: &Path = if opts.is_opt_o0() {
        ll_path
    } else {
        optimize_ll_with_opt(ll_path, &opt_ll_path, opts)?;
        &opt_ll_path
    };

    let llc = require_llvm_tool("/bin/llc", "llc")?;
    let mut cmd = Command::new(&llc);
    cmd.arg(format!("-mtriple={}", spec.triple));
    if spec.isa != CpuIsa::HostX86_64Linux {
        if !spec.cpu.is_empty() {
            cmd.arg(format!("-mcpu={}", spec.cpu));
        }
        if !spec.features.is_empty() {
            cmd.arg(format!("-mattr={}", spec.features));
        }
        if spec.isa == CpuIsa::Riscv32 {
            cmd.arg("-relocation-model=static");
        }
        cmd.arg("-function-sections").arg("-data-sections");
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

/// Link a compiled `.o` object into a freestanding `.elf` via `ld.lld`.
pub fn link_obj_to_elf(
    obj_path: &Path,
    ld_path: &Path,
    elf_path: &Path,
) -> Result<(), CodegenError> {
    let lld = require_llvm_tool("/bin/ld.lld", "ld.lld")?;
    let out = Command::new(&lld)
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

/// Inspect `.text`, `.data`, and `.bss` sizes of a linked `.elf` using `llvm-size`.
pub fn inspect_elf_sizes(elf_path: &Path) -> Result<ElfSectionSizes, CodegenError> {
    let llvm_size = require_llvm_tool("/bin/llvm-size", "llvm-size")?;
    let out = Command::new(&llvm_size)
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
        "malloc",
        "calloc",
        "realloc",
        "free",
        "_sbrk",
        "sbrk",
        "__libc_start_main",
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
    let spec = opts.effective_spec()?;
    if spec.isa == CpuIsa::HostX86_64Linux {
        return Err(CodegenError::new(
            "build-elf",
            "bare-metal .elf linking requires a freestanding chip/board target (not host x86_64)",
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
    let ld_script = synthesize_target_linker_script(&spec);
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

    let mut child = Command::new(&qemu)
        .arg("-M")
        .arg(&qemu_opts.machine)
        .arg("-cpu")
        .arg(&qemu_opts.cpu)
        .arg("-display")
        .arg("none")
        .arg("-monitor")
        .arg("none")
        .arg("-chardev")
        .arg("stdio,id=con0,mux=on,signal=off")
        .arg("-serial")
        .arg("chardev:con0")
        .arg("-semihosting-config")
        .arg("enable=on,target=native,chardev=con0")
        .arg("-kernel")
        .arg(elf_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| CodegenError::new("qemu", format!("failed to spawn `{qemu}`: {e}")))?;

    let stdout_opt = child.stdout.take();
    let stderr_opt = child.stderr.take();

    let out_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut stdout_pipe) = stdout_opt {
            let _ = stdout_pipe.read_to_end(&mut buf);
        }
        String::from_utf8_lossy(&buf).into_owned()
    });
    let err_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut stderr_pipe) = stderr_opt {
            let _ = stderr_pipe.read_to_end(&mut buf);
        }
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

/// Spawn `/bin/qemu-system-arm` with multiplexed UART0 + semihosting on stdio for
/// interactive console I/O (`piped_stdio == false`) or lockstep scripted I/O (`piped_stdio == true`).
pub fn spawn_elf_in_qemu(
    elf_path: &Path,
    qemu_opts: &QemuRunOptions,
    piped_stdio: bool,
) -> Result<std::process::Child, CodegenError> {
    let qemu = find_tool("/bin/qemu-system-arm", "qemu-system-arm");
    let mut cmd = Command::new(&qemu);
    cmd.arg("-M")
        .arg(&qemu_opts.machine)
        .arg("-cpu")
        .arg(&qemu_opts.cpu)
        .arg("-display")
        .arg("none")
        .arg("-monitor")
        .arg("none")
        .arg("-chardev")
        .arg("stdio,id=con0,mux=on,signal=off")
        .arg("-serial")
        .arg("chardev:con0")
        .arg("-semihosting-config")
        .arg("enable=on,target=native,chardev=con0")
        .arg("-kernel")
        .arg(elf_path);
    if piped_stdio {
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
    } else {
        cmd.stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
    }
    cmd.spawn()
        .map_err(|e| CodegenError::new("qemu", format!("failed to spawn `{qemu}`: {e}")))
}

/// Verify textual `.ll` IR using LLVM 23 `/bin/opt -passes=verify -disable-output`.
pub fn verify_ll_with_opt(ll_path: &Path) -> Result<(), CodegenError> {
    let opt = require_llvm_tool("/bin/opt", "opt")?;
    let out = Command::new(&opt)
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

/// Pack a contiguous binary payload into standard 512-byte UF2 blocks.
pub fn pack_uf2(bin: &[u8], base_addr: u32, family_id: u32) -> Vec<u8> {
    const UF2_MAGIC_START0: u32 = 0x0A32_4655;
    const UF2_MAGIC_START1: u32 = 0x9E5D_5157;
    const UF2_MAGIC_END: u32 = 0x0AB1_6F30;
    const UF2_FLAG_FAMILY_ID_PRESENT: u32 = 0x0000_2000;

    let chunks: Vec<&[u8]> = if bin.is_empty() {
        vec![&[][..]]
    } else {
        bin.chunks(256).collect()
    };
    let num_blocks = chunks.len() as u32;
    let mut out = Vec::with_capacity(chunks.len() * 512);
    for (idx, chunk) in chunks.iter().enumerate() {
        let mut blk = [0u8; 512];
        blk[0..4].copy_from_slice(&UF2_MAGIC_START0.to_le_bytes());
        blk[4..8].copy_from_slice(&UF2_MAGIC_START1.to_le_bytes());
        blk[8..12].copy_from_slice(&UF2_FLAG_FAMILY_ID_PRESENT.to_le_bytes());
        blk[12..16].copy_from_slice(&(base_addr + (idx as u32) * 256).to_le_bytes());
        blk[16..20].copy_from_slice(&256u32.to_le_bytes());
        blk[20..24].copy_from_slice(&(idx as u32).to_le_bytes());
        blk[24..28].copy_from_slice(&num_blocks.to_le_bytes());
        blk[28..32].copy_from_slice(&family_id.to_le_bytes());
        blk[32..32 + chunk.len()].copy_from_slice(chunk);
        blk[508..512].copy_from_slice(&UF2_MAGIC_END.to_le_bytes());
        out.extend_from_slice(&blk);
    }
    out
}

/// Convert a linked `.elf` into a raw binary via `/bin/llvm-objcopy -O binary`.
pub fn objcopy_elf_to_bin(elf_path: &Path, out_bin: &Path) -> Result<Vec<u8>, CodegenError> {
    if let Some(parent) = out_bin.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                CodegenError::new("io", format!("failed to create {}: {e}", parent.display()))
            })?;
        }
    }
    let objcopy = require_llvm_tool("/bin/llvm-objcopy", "llvm-objcopy")?;
    let out = Command::new(&objcopy)
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
    fs::read(out_bin)
        .map_err(|e| CodegenError::new("io", format!("failed to read {}: {e}", out_bin.display())))
}

fn measure_static_sram(
    elf_path: &Path,
    spec: &TargetSpec,
    bin_len: usize,
) -> Result<u64, CodegenError> {
    if spec.code_in_ram {
        let sizes = inspect_elf_sizes(elf_path)?;
        let padded_bin_len = (bin_len as u64).next_multiple_of(256);
        return Ok(padded_bin_len.saturating_add(sizes.bss));
    }
    match (
        read_elf32_sym_addr(elf_path, "__sdata"),
        read_elf32_sym_addr(elf_path, "__ebss"),
    ) {
        (Some(sdata), Some(ebss)) if ebss >= sdata => Ok(u64::from(ebss - sdata)),
        _ => {
            let sizes = inspect_elf_sizes(elf_path)?;
            Ok(sizes.data.saturating_add(sizes.bss))
        }
    }
}

fn validate_bin_against_spec(
    spec: &TargetSpec,
    elf_path: &Path,
    bytes: &[u8],
) -> Result<u64, CodegenError> {
    let target_label = spec.board_name.as_deref().unwrap_or(&spec.chip_name);
    if bytes.is_empty() {
        return Err(CodegenError::new(
            "target-image",
            format!("generated binary for `{target_label}` is empty"),
        ));
    }
    if !spec.code_in_ram && (bytes.len() as u64) > u64::from(spec.effective_flash_size()) {
        return Err(CodegenError::new(
            "target-image",
            format!(
                "`{target_label}` image size {} B exceeds usable flash {} B",
                bytes.len(),
                spec.effective_flash_size()
            ),
        ));
    }
    let magic = spec.header_magic_bytes();
    if !magic.is_empty() && (bytes.len() < magic.len() || bytes[..magic.len()] != magic[..]) {
        let got_len = bytes.len().min(magic.len());
        return Err(CodegenError::new(
            "target-image",
            format!(
                "`{target_label}` image header mismatch at offset 0: got {:02x?}, expected {:02x?}",
                &bytes[..got_len],
                magic
            ),
        ));
    }
    if spec.reset_first_word_mask != 0 {
        if bytes.len() < 2 {
            return Err(CodegenError::new(
                "target-image",
                format!("`{target_label}` image shorter than 2 bytes for reset vector check"),
            ));
        }
        let word0 = u16::from_le_bytes([bytes[0], bytes[1]]);
        if (word0 & spec.reset_first_word_mask) != spec.reset_first_word_value {
            return Err(CodegenError::new(
                "target-image",
                format!(
                    "`{target_label}` reset word0 0x{word0:04X} failed mask 0x{:04X} == 0x{:04X}",
                    spec.reset_first_word_mask, spec.reset_first_word_value
                ),
            ));
        }
    }
    let static_sram = measure_static_sram(elf_path, spec, bytes.len())?;
    let max_static = u64::from(
        spec.sram_budget_bytes
            .saturating_sub(spec.stack_reserve_bytes),
    );
    if spec.stack_reserve_bytes > 0 && static_sram >= max_static && !spec.code_in_ram {
        return Err(CodegenError::new(
            "target-image",
            format!(
                "`{target_label}` static SRAM footprint {static_sram} B leaves fewer than {} B stack in {} B SRAM",
                spec.stack_reserve_bytes, spec.sram_budget_bytes
            ),
        ));
    }
    if spec.code_in_ram && spec.stack_reserve_bytes > 0 && static_sram > max_static {
        return Err(CodegenError::new(
            "target-image",
            format!(
                "`{target_label}` RAM image footprint ({static_sram} B) exceeds RAM minus stack reserve ({max_static} B)"
            ),
        ));
    }
    Ok(static_sram)
}

/// Build an ELF + target deliverable image (`ElfOnly`, `RawBin`, or `Uf2`) driven by `TargetSpec`.
pub fn build_target_image(
    module: &zev_mir::MirModule,
    opts: &CodegenOptions,
    out_path: &Path,
) -> Result<TargetBuildArtifact, CodegenError> {
    let spec = opts.effective_spec()?;
    let ll = crate::emit_llvm_ir_with_options(module, opts)?;
    match spec.image {
        ImageFormat::ElfOnly => {
            let elf = compile_ll_to_elf(&ll, out_path, opts)?;
            let image_size = fs::metadata(&elf.elf_path).map(|m| m.len()).unwrap_or(0);
            let sram_static_bytes = measure_static_sram(&elf.elf_path, &spec, 0)?;
            Ok(TargetBuildArtifact {
                image_path: elf.elf_path.clone(),
                elf,
                bin_path: None,
                uf2_path: None,
                bin_size: 0,
                image_size,
                uf2_blocks: 0,
                sram_static_bytes,
            })
        }
        ImageFormat::RawBin => {
            let elf_path = out_path.with_extension("elf");
            let elf = compile_ll_to_elf(&ll, &elf_path, opts)?;
            let bytes = objcopy_elf_to_bin(&elf.elf_path, out_path)?;
            let sram_static_bytes = validate_bin_against_spec(&spec, &elf.elf_path, &bytes)?;
            let bin_size = bytes.len() as u64;
            Ok(TargetBuildArtifact {
                elf,
                image_path: out_path.to_path_buf(),
                bin_path: Some(out_path.to_path_buf()),
                uf2_path: None,
                bin_size,
                image_size: bin_size,
                uf2_blocks: 0,
                sram_static_bytes,
            })
        }
        ImageFormat::Uf2 {
            family_id,
            target_base,
        } => {
            let elf_path = out_path.with_extension("elf");
            let bin_path = out_path.with_extension("bin");
            let elf = compile_ll_to_elf(&ll, &elf_path, opts)?;
            let bytes = objcopy_elf_to_bin(&elf.elf_path, &bin_path)?;
            let sram_static_bytes = validate_bin_against_spec(&spec, &elf.elf_path, &bytes)?;
            let uf2 = pack_uf2(&bytes, target_base, family_id);
            let uf2_blocks = (uf2.len() / 512) as u32;
            fs::write(out_path, &uf2).map_err(|e| {
                CodegenError::new("io", format!("failed to write {}: {e}", out_path.display()))
            })?;
            Ok(TargetBuildArtifact {
                elf,
                image_path: out_path.to_path_buf(),
                bin_path: Some(bin_path),
                uf2_path: Some(out_path.to_path_buf()),
                bin_size: bytes.len() as u64,
                image_size: uf2.len() as u64,
                uf2_blocks,
                sram_static_bytes,
            })
        }
    }
}

fn resolve_flash_arg(spec: &TargetSpec, arg: &str) -> String {
    let p = Path::new(arg);
    if !p.is_absolute()
        && (arg.ends_with(".py") || arg.ends_with(".sh") || arg.contains('/'))
        && !p.exists()
    {
        if let Some(root) = &spec.root_dir {
            let cand = root.join(p);
            if cand.is_file() {
                return cand.to_string_lossy().into_owned();
            }
        }
        for root in discover_spec_roots() {
            let cand = root.join(p);
            if cand.is_file() {
                return cand.to_string_lossy().into_owned();
            }
        }
    }
    arg.to_string()
}

/// Flash a compiled target image using `spec.flash_cmd` (`{image}`, `{elf}`, `{port}`).
pub fn flash_target_image(
    spec: &TargetSpec,
    image_path: &Path,
    port: Option<&str>,
) -> Result<String, CodegenError> {
    let label = spec.board_name.as_deref().unwrap_or(&spec.chip_name);
    if spec.flash_cmd.is_empty() {
        return Err(CodegenError::new(
            "flash",
            format!("target `{label}` does not declare `[run].flash_cmd` in its board TOML"),
        ));
    }
    let eff_port = port
        .or(spec.default_port.as_deref())
        .unwrap_or("/dev/ttyACM0");
    let image_str = image_path.to_string_lossy();
    let elf_path = image_path.with_extension("elf");
    let elf_str = elf_path.to_string_lossy();

    let mut expanded = Vec::with_capacity(spec.flash_cmd.len());
    for tok in &spec.flash_cmd {
        let sub = tok
            .replace("{image}", &image_str)
            .replace("{elf}", &elf_str)
            .replace("{port}", eff_port);
        expanded.push(resolve_flash_arg(spec, &sub));
    }

    let raw_prog = &expanded[0];
    let prog = if !raw_prog.contains('/') && !raw_prog.contains('\\') {
        find_tool("", raw_prog)
    } else {
        raw_prog.clone()
    };

    let out = Command::new(&prog)
        .args(&expanded[1..])
        .output()
        .map_err(|e| CodegenError::new("flash", format!("failed to invoke `{prog}`: {e}")))?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        return Err(CodegenError::new(
            "flash",
            format!(
                "flash command `{}` exited with {}:\n{}",
                expanded.join(" "),
                out.status,
                combined.trim()
            ),
        ));
    }
    Ok(combined)
}

/// Legacy compatibility helper delegating via `esp32c6_pico` TOML spec.
pub fn objcopy_elf_to_esp32c6_bin(
    elf_path: &Path,
    out_bin: &Path,
) -> Result<(u64, [u8; 8]), CodegenError> {
    let spec = TargetSpec::resolve(None, Some("esp32c6_pico"))?;
    let bytes = objcopy_elf_to_bin(elf_path, out_bin)?;
    let _ = validate_bin_against_spec(&spec, elf_path, &bytes)?;
    let mut magic = [0u8; 8];
    if bytes.len() >= 8 {
        magic.copy_from_slice(&bytes[..8]);
    }
    Ok((bytes.len() as u64, magic))
}

/// Legacy compatibility flasher delegating via `esp32c6_pico` TOML spec.
pub fn flash_esp32c6_bin(bin_path: &Path, port: &str) -> Result<String, CodegenError> {
    let spec = TargetSpec::resolve(None, Some("esp32c6_pico"))?;
    flash_target_image(&spec, bin_path, Some(port))
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
/// Legacy compatibility helper delegating via `arduino_uno` TOML spec.
pub fn objcopy_elf_to_avr_bin(elf_path: &Path, out_bin: &Path) -> Result<(u64, u64), CodegenError> {
    let spec = TargetSpec::resolve(None, Some("arduino_uno"))?;
    let bytes = objcopy_elf_to_bin(elf_path, out_bin)?;
    let sram_static_bytes = validate_bin_against_spec(&spec, elf_path, &bytes)?;
    Ok((bytes.len() as u64, sram_static_bytes))
}

/// Legacy compatibility helper delegating via `arduino_nano_every` TOML spec.
pub fn objcopy_elf_to_avr_nano_every_bin(
    elf_path: &Path,
    out_bin: &Path,
) -> Result<(u64, u64), CodegenError> {
    let spec = TargetSpec::resolve(None, Some("arduino_nano_every"))?;
    let bytes = objcopy_elf_to_bin(elf_path, out_bin)?;
    let sram_static_bytes = validate_bin_against_spec(&spec, elf_path, &bytes)?;
    Ok((bytes.len() as u64, sram_static_bytes))
}

/// Legacy compatibility UF2 packer delegating to [`pack_uf2`].
pub fn pack_rp2040_uf2(bin: &[u8], base_addr: u32) -> Vec<u8> {
    let family_id = TargetSpec::resolve(None, Some("rpi_pico_w"))
        .ok()
        .and_then(|s| match s.image {
            ImageFormat::Uf2 { family_id, .. } => Some(family_id),
            _ => None,
        })
        .unwrap_or(crate::RP2040_UF2_FAMILY_ID);
    pack_uf2(bin, base_addr, family_id)
}

/// Legacy compatibility helper delegating via `rpi_pico_w` TOML spec.
pub fn objcopy_elf_to_rp2040_uf2(
    elf_path: &Path,
    out_uf2: &Path,
) -> Result<(PathBuf, u64, u64, u32), CodegenError> {
    let spec = TargetSpec::resolve(None, Some("rpi_pico_w"))?;
    let (family_id, target_base) = match spec.image {
        ImageFormat::Uf2 {
            family_id,
            target_base,
        } => (family_id, target_base),
        _ => {
            return Err(CodegenError::new(
                "target-image",
                "rpi_pico_w board TOML is not configured for UF2 image output",
            ));
        }
    };
    let out_bin = out_uf2.with_extension("bin");
    let bytes = objcopy_elf_to_bin(elf_path, &out_bin)?;
    let _ = validate_bin_against_spec(&spec, elf_path, &bytes)?;
    let uf2 = pack_uf2(&bytes, target_base, family_id);
    let blocks = (uf2.len() / 512) as u32;
    fs::write(out_uf2, &uf2).map_err(|e| {
        CodegenError::new("io", format!("failed to write {}: {e}", out_uf2.display()))
    })?;
    Ok((out_bin, bytes.len() as u64, uf2.len() as u64, blocks))
}

/// Legacy compatibility flasher delegating via `rpi_pico_w` TOML spec.
pub fn flash_rp2040_uf2(uf2_path: &Path, port: &str) -> Result<String, CodegenError> {
    let spec = TargetSpec::resolve(None, Some("rpi_pico_w"))?;
    flash_target_image(&spec, uf2_path, Some(port))
}

/// Legacy compatibility flasher delegating via `arduino_uno` TOML spec.
pub fn flash_avr_uno_bin(bin_path: &Path, port: &str) -> Result<String, CodegenError> {
    let spec = TargetSpec::resolve(None, Some("arduino_uno"))?;
    flash_target_image(&spec, bin_path, Some(port))
}

/// Legacy compatibility flasher delegating via `arduino_nano_every` TOML spec.
pub fn flash_avr_nano_every_bin(bin_path: &Path, port: &str) -> Result<String, CodegenError> {
    let spec = TargetSpec::resolve(None, Some("arduino_nano_every"))?;
    flash_target_image(&spec, bin_path, Some(port))
}

#[cfg(test)]
mod tests {
    use super::{
        check_llvm_tool_version, find_tool_in_dirs, parse_llvm_major_version, MIN_LLVM_MAJOR,
    };

    #[test]
    fn parses_llvm_lld_and_clang_version_strings() {
        assert_eq!(
            parse_llvm_major_version("LLVM (http://llvm.org/):\n  LLVM version 23.1.5\n"),
            Some((23, "LLVM version 23.1.5".to_string()))
        );
        assert_eq!(
            parse_llvm_major_version("Ubuntu LLVM version 18.1.3\n  Optimized build."),
            Some((18, "Ubuntu LLVM version 18.1.3".to_string()))
        );
        assert_eq!(
            parse_llvm_major_version("LLD 23.1.5 (compatible with GNU linkers)"),
            Some((23, "LLD 23.1.5 (compatible with GNU linkers)".to_string()))
        );
        assert_eq!(
            parse_llvm_major_version("Ubuntu LLD 18.1.3 (compatible with GNU linkers)"),
            Some((
                18,
                "Ubuntu LLD 18.1.3 (compatible with GNU linkers)".to_string()
            ))
        );
        assert_eq!(
            parse_llvm_major_version("clang version 23.1.5 (Fedora 23.1.5-2.fc45)"),
            Some((
                23,
                "clang version 23.1.5 (Fedora 23.1.5-2.fc45)".to_string()
            ))
        );
        // Apple Xcode clang versions must not be confused with upstream LLVM versions.
        assert_eq!(
            parse_llvm_major_version("Apple clang version 16.0.0 (clang-1600.0.26.6)"),
            None
        );
        assert_eq!(MIN_LLVM_MAJOR, 23);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_old_llvm_toolchain_and_prefers_llvm_23_candidate() {
        use std::os::unix::fs::PermissionsExt;

        let base =
            std::env::temp_dir().join(format!("zev-toolchain-ver-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir_old = base.join("old_bin");
        let dir_new = base.join("new_bin");
        std::fs::create_dir_all(&dir_old).expect("create old_bin");
        std::fs::create_dir_all(&dir_new).expect("create new_bin");

        let write_mock = |path: &std::path::Path, banner: &str| {
            let script = format!("#!/bin/sh\necho \"{banner}\"\n");
            std::fs::write(path, script).expect("write mock tool");
            let mut perms = std::fs::metadata(path)
                .expect("stat mock tool")
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(path, perms).expect("chmod mock tool");
        };

        let old_llc = dir_old.join("llc");
        let old_lld = dir_old.join("ld.lld");
        let new_llc_ver = dir_new.join("llc-23");
        write_mock(&old_llc, "Ubuntu LLVM version 18.1.3");
        write_mock(&old_lld, "Ubuntu LLD 18.1.3 (compatible with GNU linkers)");
        write_mock(&new_llc_ver, "LLVM version 23.1.5");

        let err_llc = check_llvm_tool_version(old_llc.to_str().expect("utf8"), "llc")
            .expect_err("LLVM 18 llc must be rejected");
        assert!(err_llc.message.contains("LLVM 23 or newer is required"));
        assert!(err_llc.message.contains("18.1.3"));
        assert!(err_llc.message.contains("ZEV_LLVM_PREFIX"));

        let err_lld = check_llvm_tool_version(old_lld.to_str().expect("utf8"), "ld.lld")
            .expect_err("LLD 18 must be rejected");
        assert!(err_lld.message.contains("LLVM 23 or newer is required"));
        assert!(err_lld.message.contains("18.1.3"));

        check_llvm_tool_version(new_llc_ver.to_str().expect("utf8"), "llc")
            .expect("LLVM 23 llc-23 must be accepted");

        // When PATH has LLVM 18 `llc` before `llc-23`, discovery skips 18 and selects `llc-23`.
        // Hermetic: no real system LLVM prefixes, so a host-installed LLVM 23 can't win.
        let no_sys = |_: u32| Vec::new();
        let chosen = find_tool_in_dirs(
            "",
            "llc",
            None,
            &[dir_old.clone(), dir_new.clone()],
            &[],
            no_sys,
        );
        assert_eq!(chosen, new_llc_ver.to_string_lossy());

        // When only LLVM 18 exists, discovery returns the old path so `require_llvm_tool`
        // reports a clear version mismatch rather than "file not found".
        let fallback = find_tool_in_dirs("", "llc", None, &[dir_old], &[], no_sys);
        assert_eq!(fallback, old_llc.to_string_lossy());

        let _ = std::fs::remove_dir_all(&base);
    }
}
