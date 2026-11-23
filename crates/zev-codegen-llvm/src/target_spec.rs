//! Declarative TOML Chip & Board Target Specifications + `zev.toml` Project Manifest Loader.
//!
//! Zero external crates (`std`-only). All target memory maps, LLVM triples/CPUs, vector table
//! geometries, binary image formats, and flasher command templates live in `chips/*.toml` and
//! `boards/*.toml` rather than being hardcoded in the compiler.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::toolchain::CodegenError;

/// Generic CPU instruction set architecture handled by the 3 freestanding compiler backends
/// (`arm_cortex_m`, `avr`, `riscv32`) plus host IR inspection (`HostX86_64Linux`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuIsa {
    ArmCortexM,
    Avr,
    Riscv32,
    HostX86_64Linux,
}

impl CpuIsa {
    pub fn parse(s: &str) -> Result<Self, CodegenError> {
        match s.trim() {
            "arm-cortex-m" | "cortex-m" | "arm" => Ok(Self::ArmCortexM),
            "avr" => Ok(Self::Avr),
            "riscv32" | "rv32" => Ok(Self::Riscv32),
            "host-x86_64" | "host-x86_64-linux" | "x86_64" => Ok(Self::HostX86_64Linux),
            other => Err(CodegenError::new(
                "toml",
                format!("unknown `[chip].isa` '{other}'; expected `arm-cortex-m`, `avr`, `riscv32`, or `host-x86_64`"),
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArmCortexM => "arm-cortex-m",
            Self::Avr => "avr",
            Self::Riscv32 => "riscv32",
            Self::HostX86_64Linux => "host-x86_64",
        }
    }

    pub const fn is_avr(self) -> bool {
        matches!(self, Self::Avr)
    }

    pub const fn is_arm(self) -> bool {
        matches!(self, Self::ArmCortexM)
    }

    pub const fn is_riscv(self) -> bool {
        matches!(self, Self::Riscv32)
    }

    pub const fn ptr_bits(self) -> u32 {
        match self {
            Self::ArmCortexM | Self::Riscv32 => 32,
            Self::Avr => 16,
            Self::HostX86_64Linux => 64,
        }
    }

    pub const fn usize_llvm_ty(self) -> &'static str {
        match self {
            Self::ArmCortexM | Self::Riscv32 => "i32",
            Self::Avr => "i16",
            Self::HostX86_64Linux => "i64",
        }
    }

    pub const fn fn_ptr_llvm_ty(self) -> &'static str {
        match self {
            Self::Avr => "ptr addrspace(1)",
            Self::ArmCortexM | Self::Riscv32 | Self::HostX86_64Linux => "ptr",
        }
    }

    pub const fn fn_addrspace_attr(self) -> &'static str {
        match self {
            Self::Avr => " addrspace(1)",
            Self::ArmCortexM | Self::Riscv32 | Self::HostX86_64Linux => "",
        }
    }

    pub const fn call_kw(self) -> &'static str {
        match self {
            Self::Avr => "call addrspace(1)",
            Self::ArmCortexM | Self::Riscv32 | Self::HostX86_64Linux => "call",
        }
    }
}

/// Hardware image output format produced from the linked `.elf`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageFormat {
    ElfOnly,
    RawBin,
    Uf2 { family_id: u32, target_base: u32 },
}

/// Architectural interrupt vector & controller model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrqModel {
    CortexMNvic {
        max_irqs: u32,
        min_vector_irqs: u32,
    },
    AvrVectored {
        vector_slots: u32,
        slot_words: u8,
        data_ram_offset: u32,
    },
    RiscvVectored {
        vector_slots: u32,
        max_irq_source: u32,
        reserved_slots: Vec<u32>,
    },
    None,
}

/// Resolved target hardware + toolchain configuration merged from `chips/<chip>.toml` and optional
/// `boards/<board>.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetSpec {
    pub chip_name: String,
    pub board_name: Option<String>,
    pub isa: CpuIsa,
    pub cpu: String,
    pub features: String,
    pub triple: String,
    pub datalayout: String,
    pub ptr_bits: u32,
    pub opt_level: String,
    pub code_in_ram: bool,
    pub rodata_in_ram: bool,
    pub has_thumb2: bool,
    pub has_vtor: bool,
    pub has_fpu: bool,
    pub iram_in_data: bool,
    pub allow_sleep_deep: bool,
    pub semihosting: bool,
    pub hal_dir: Option<String>,
    pub board_module: Option<String>,
    pub flash_base: u32,
    pub flash_size: u32,
    pub usable_flash_size: Option<u32>,
    pub ram_base: u32,
    pub ram_size: u32,
    pub stack_reserve_bytes: u32,
    pub sram_budget_bytes: u32,
    pub irq: IrqModel,
    pub image: ImageFormat,
    pub header_magic_words: Vec<u32>,
    pub reset_first_word_mask: u16,
    pub reset_first_word_value: u16,
    pub qemu_machine: Option<String>,
    pub flash_cmd: Vec<String>,
    pub default_port: Option<String>,
    pub root_dir: Option<PathBuf>,
}

impl TargetSpec {
    /// Host 64-bit Linux synthetic target used for host IR tests.
    pub fn host_linux() -> Self {
        Self {
            chip_name: "host-x86_64".to_string(),
            board_name: None,
            isa: CpuIsa::HostX86_64Linux,
            cpu: "x86-64".to_string(),
            features: String::new(),
            triple: "x86_64-unknown-linux-gnu".to_string(),
            datalayout:
                "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
                    .to_string(),
            ptr_bits: 64,
            opt_level: "-O2".to_string(),
            code_in_ram: false,
            rodata_in_ram: false,
            has_thumb2: false,
            has_vtor: false,
            has_fpu: false,
            iram_in_data: false,
            allow_sleep_deep: false,
            semihosting: false,
            hal_dir: None,
            board_module: None,
            flash_base: 0,
            flash_size: 4 * 1024 * 1024,
            usable_flash_size: None,
            ram_base: 0,
            ram_size: 4 * 1024 * 1024,
            stack_reserve_bytes: 4096,
            sram_budget_bytes: 4 * 1024 * 1024,
            irq: IrqModel::None,
            image: ImageFormat::ElfOnly,
            header_magic_words: Vec::new(),
            reset_first_word_mask: 0,
            reset_first_word_value: 0,
            qemu_machine: None,
            flash_cmd: Vec::new(),
            default_port: None,
            root_dir: None,
        }
    }

    pub const fn initial_sp(&self) -> u32 {
        let top = self.ram_base.wrapping_add(self.ram_size);
        if self.isa.is_avr() {
            top.wrapping_sub(1)
        } else {
            top
        }
    }

    pub fn effective_flash_size(&self) -> u32 {
        self.usable_flash_size.unwrap_or(self.flash_size)
    }

    pub fn header_magic_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.header_magic_words.len() * 4);
        for w in &self.header_magic_words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    /// Resolve a chip/board spec using standard workspace/environment discovery roots.
    pub fn resolve(chip: Option<&str>, board: Option<&str>) -> Result<Self, CodegenError> {
        // Explicit `.toml` path support on --chip / --board for arbitrary external directories.
        if let Some(b_arg) = board {
            let bp = Path::new(b_arg);
            if bp.extension().is_some_and(|e| e == "toml") && bp.is_file() {
                let board_src = fs::read_to_string(bp).map_err(|e| {
                    CodegenError::new("toml", format!("failed to read {}: {e}", bp.display()))
                })?;
                let bdoc = parse_toml(&board_src)?;
                let chip_name = bdoc.req_str("board", "chip")?;
                let chip_path = resolve_chip_file_from_hint(bp.parent(), &chip_name, chip)?;
                let chip_src = fs::read_to_string(&chip_path).map_err(|e| {
                    CodegenError::new(
                        "toml",
                        format!("failed to read {}: {e}", chip_path.display()),
                    )
                })?;
                let mut spec = Self::from_tomls(&chip_src, Some(&board_src))?;
                spec.root_dir = bp.parent().and_then(|p| p.parent()).map(PathBuf::from);
                return Ok(spec);
            }
        }
        if let Some(c_arg) = chip {
            let cp = Path::new(c_arg);
            if cp.extension().is_some_and(|e| e == "toml") && cp.is_file() {
                let chip_src = fs::read_to_string(cp).map_err(|e| {
                    CodegenError::new("toml", format!("failed to read {}: {e}", cp.display()))
                })?;
                let mut spec = Self::from_tomls(&chip_src, None)?;
                spec.root_dir = cp.parent().and_then(|p| p.parent()).map(PathBuf::from);
                return Ok(spec);
            }
        }

        let roots = discover_spec_roots();
        for root in &roots {
            if let Ok(spec) = Self::load_from_root(root, chip, board) {
                return Ok(spec);
            }
        }

        let tried: Vec<String> = roots.iter().map(|p| p.display().to_string()).collect();
        Err(CodegenError::new(
            "target",
            format!(
                "failed to resolve target (chip={:?}, board={:?}); searched `chips/` & `boards/` in [{}]",
                chip,
                board,
                tried.join(", ")
            ),
        ))
    }

    /// Load a `TargetSpec` from an explicit root containing `chips/<chip>.toml` and `boards/<board>.toml`.
    pub fn load_from_root(
        root: &Path,
        chip: Option<&str>,
        board: Option<&str>,
    ) -> Result<Self, CodegenError> {
        let board_src = if let Some(bname) = board {
            let bpath = root.join("boards").join(format!("{bname}.toml"));
            if !bpath.is_file() {
                return Err(CodegenError::new(
                    "target",
                    format!("board specification not found: {}", bpath.display()),
                ));
            }
            Some(fs::read_to_string(&bpath).map_err(|e| {
                CodegenError::new("toml", format!("failed to read {}: {e}", bpath.display()))
            })?)
        } else {
            None
        };

        let chip_name = if let Some(cname) = chip {
            cname.to_string()
        } else if let Some(ref bsrc) = board_src {
            let bdoc = parse_toml(bsrc)?;
            bdoc.req_str("board", "chip")?
        } else {
            return Err(CodegenError::new(
                "target",
                "neither `--board` nor `--chip` was specified and no default target was given",
            ));
        };

        let cpath = root.join("chips").join(format!("{chip_name}.toml"));
        if !cpath.is_file() {
            return Err(CodegenError::new(
                "target",
                format!("chip specification not found: {}", cpath.display()),
            ));
        }
        let chip_src = fs::read_to_string(&cpath).map_err(|e| {
            CodegenError::new("toml", format!("failed to read {}: {e}", cpath.display()))
        })?;

        let mut spec = Self::from_tomls(&chip_src, board_src.as_deref())?;
        spec.root_dir = Some(root.to_path_buf());
        Ok(spec)
    }

    /// Parse and merge raw chip TOML + optional board TOML into a `TargetSpec`.
    pub fn from_tomls(chip_toml: &str, board_toml: Option<&str>) -> Result<Self, CodegenError> {
        let cdoc = parse_toml(chip_toml)?;
        let chip_name = cdoc.req_str("chip", "name")?;
        let isa = CpuIsa::parse(&cdoc.req_str("chip", "isa")?)?;
        let cpu = cdoc.req_str("chip", "cpu")?;
        let features = cdoc.opt_str("chip", "features")?.unwrap_or_default();
        let triple = cdoc.req_str("chip", "triple")?;
        let datalayout = cdoc.req_str("chip", "datalayout")?;
        let ptr_bits = cdoc
            .opt_u32("chip", "ptr_bits")?
            .unwrap_or(isa.ptr_bits());
        let mut opt_level = cdoc
            .opt_str("chip", "opt_level")?
            .unwrap_or_else(|| "-O2".to_string());
        let code_in_ram = cdoc.opt_bool("chip", "code_in_ram")?.unwrap_or(false);
        let rodata_in_ram = cdoc.opt_bool("chip", "rodata_in_ram")?.unwrap_or(false);
        let has_thumb2 = cdoc.opt_bool("chip", "has_thumb2")?.unwrap_or(false);
        let has_vtor = cdoc.opt_bool("chip", "has_vtor")?.unwrap_or(false);
        let has_fpu = cdoc.opt_bool("chip", "has_fpu")?.unwrap_or(false);
        let iram_in_data = cdoc.opt_bool("chip", "iram_in_data")?.unwrap_or(false);
        let allow_sleep_deep = cdoc.opt_bool("chip", "allow_sleep_deep")?.unwrap_or(false);
        let mut semihosting = cdoc.opt_bool("chip", "semihosting")?.unwrap_or(false);
        let mut hal_dir = cdoc.opt_str("chip", "hal_dir")?;

        let flash_base = cdoc.req_u32("memory", "flash_base")?;
        let flash_size = cdoc.req_u32("memory", "flash_size")?;
        let mut usable_flash_size = cdoc.opt_u32("memory", "usable_flash_size")?;
        let ram_base = cdoc.req_u32("memory", "ram_base")?;
        let ram_size = cdoc.req_u32("memory", "ram_size")?;
        let mut stack_reserve_bytes = cdoc
            .opt_u32("memory", "stack_reserve_bytes")?
            .unwrap_or(256);
        let sram_budget_bytes = cdoc
            .opt_u32("memory", "sram_budget_bytes")?
            .unwrap_or(ram_size);

        let irq_model_name = cdoc
            .opt_str("irq", "model")?
            .unwrap_or_else(|| "none".to_string());
        let irq = match irq_model_name.as_str() {
            "cortex-m-nvic" => IrqModel::CortexMNvic {
                max_irqs: cdoc.opt_u32("irq", "max_irqs")?.unwrap_or(32),
                min_vector_irqs: cdoc.opt_u32("irq", "min_vector_irqs")?.unwrap_or(32),
            },
            "avr-vectored" => IrqModel::AvrVectored {
                vector_slots: cdoc.req_u32("irq", "vector_slots")?,
                slot_words: u8::try_from(cdoc.opt_u32("irq", "slot_words")?.unwrap_or(2))
                    .map_err(|_| CodegenError::new("toml", "irq.slot_words out of u8 range"))?,
                data_ram_offset: cdoc
                    .opt_u32("irq", "data_ram_offset")?
                    .unwrap_or(0x0080_0000),
            },
            "riscv-vectored" => IrqModel::RiscvVectored {
                vector_slots: cdoc.opt_u32("irq", "vector_slots")?.unwrap_or(32),
                max_irq_source: cdoc.req_u32("irq", "max_irq_source")?,
                reserved_slots: cdoc.opt_u32_array("irq", "reserved_slots")?.unwrap_or_default(),
            },
            "none" => IrqModel::None,
            other => {
                return Err(CodegenError::new(
                    "toml",
                    format!("unknown `[irq].model` '{other}'"),
                ));
            }
        };

        let fmt_name = cdoc
            .opt_str("image", "format")?
            .unwrap_or_else(|| "elf".to_string());
        let image = match fmt_name.as_str() {
            "elf" => ImageFormat::ElfOnly,
            "raw-bin" => ImageFormat::RawBin,
            "uf2" => ImageFormat::Uf2 {
                family_id: cdoc.req_u32("image", "uf2_family_id")?,
                target_base: cdoc
                    .opt_u32("image", "uf2_target_base")?
                    .unwrap_or(ram_base),
            },
            other => {
                return Err(CodegenError::new(
                    "toml",
                    format!("unknown `[image].format` '{other}'; expected `elf`, `raw-bin`, or `uf2`"),
                ));
            }
        };
        let header_magic_words = cdoc
            .opt_u32_array("image", "header_magic_words")?
            .unwrap_or_default();
        let reset_first_word_mask =
            u16::try_from(cdoc.opt_u32("image", "reset_first_word_mask")?.unwrap_or(0))
                .map_err(|_| CodegenError::new("toml", "image.reset_first_word_mask > u16"))?;
        let reset_first_word_value =
            u16::try_from(cdoc.opt_u32("image", "reset_first_word_value")?.unwrap_or(0))
                .map_err(|_| CodegenError::new("toml", "image.reset_first_word_value > u16"))?;

        let mut board_name = None;
        let mut board_module = None;
        let mut qemu_machine = cdoc.opt_str("run", "qemu_machine")?;
        let mut flash_cmd = cdoc.opt_str_array("run", "flash_cmd")?.unwrap_or_default();
        let mut default_port = cdoc.opt_str("run", "default_port")?;

        if let Some(btoml) = board_toml {
            let bdoc = parse_toml(btoml)?;
            board_name = Some(bdoc.req_str("board", "name")?);
            let b_chip = bdoc.req_str("board", "chip")?;
            if b_chip != chip_name {
                return Err(CodegenError::new(
                    "toml",
                    format!(
                        "board '{}' expects chip '{}', but loaded chip '{}'",
                        board_name.as_deref().unwrap_or("?"),
                        b_chip,
                        chip_name
                    ),
                ));
            }
            if let Some(o) = bdoc.opt_str("board", "opt_level")? {
                opt_level = o;
            }
            if let Some(hd) = bdoc.opt_str("board", "hal_dir")? {
                hal_dir = Some(hd);
            }
            if let Some(bm) = bdoc.opt_str("board", "board_module")? {
                board_module = Some(bm);
            } else if let Some(ref bn) = board_name {
                board_module = Some(format!("bsp/{bn}"));
            }
            if let Some(uf) = bdoc.opt_u32("memory", "usable_flash_size")? {
                usable_flash_size = Some(uf);
            }
            if let Some(sr) = bdoc.opt_u32("memory", "stack_reserve_bytes")? {
                stack_reserve_bytes = sr;
            }
            if let Some(qm) = bdoc.opt_str("run", "qemu_machine")? {
                qemu_machine = Some(qm);
            }
            if let Some(sh) = bdoc.opt_bool("run", "semihosting")? {
                semihosting = sh;
            }
            if let Some(fc) = bdoc.opt_str_array("run", "flash_cmd")? {
                flash_cmd = fc;
            }
            if let Some(dp) = bdoc.opt_str("run", "default_port")? {
                default_port = Some(dp);
            }
        }

        Ok(Self {
            chip_name,
            board_name,
            isa,
            cpu,
            features,
            triple,
            datalayout,
            ptr_bits,
            opt_level,
            code_in_ram,
            rodata_in_ram,
            has_thumb2,
            has_vtor,
            has_fpu,
            iram_in_data,
            allow_sleep_deep,
            semihosting,
            hal_dir,
            board_module,
            flash_base,
            flash_size,
            usable_flash_size,
            ram_base,
            ram_size,
            stack_reserve_bytes,
            sram_budget_bytes,
            irq,
            image,
            header_magic_words,
            reset_first_word_mask,
            reset_first_word_value,
            qemu_machine,
            flash_cmd,
            default_port,
            root_dir: None,
        })
    }
}

/// Optional project-level `zev.toml` manifest holding project defaults (no CLI flags needed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectManifest {
    pub name: Option<String>,
    pub entry: Option<PathBuf>,
    pub chip: Option<String>,
    pub board: Option<String>,
    pub port: Option<String>,
    pub opt_level: Option<String>,
}

impl ProjectManifest {
    pub fn parse(src: &str) -> Result<Self, CodegenError> {
        let doc = parse_toml(src)?;
        Ok(Self {
            name: doc.opt_str("project", "name")?,
            entry: doc.opt_str("project", "entry")?.map(PathBuf::from),
            chip: doc.opt_str("project", "chip")?,
            board: doc.opt_str("project", "board")?,
            port: doc.opt_str("project", "port")?,
            opt_level: doc.opt_str("project", "opt_level")?,
        })
    }

    /// Walk `start_dir` and ancestors looking for `zev.toml`.
    pub fn discover(start_dir: &Path) -> Result<Option<(PathBuf, Self)>, CodegenError> {
        let mut cur = if start_dir.is_file() {
            start_dir.parent().map(Path::to_path_buf)
        } else {
            Some(start_dir.to_path_buf())
        };
        while let Some(dir) = cur {
            let candidate = dir.join("zev.toml");
            if candidate.is_file() {
                let src = fs::read_to_string(&candidate).map_err(|e| {
                    CodegenError::new(
                        "toml",
                        format!("failed to read {}: {e}", candidate.display()),
                    )
                })?;
                return Ok(Some((dir, Self::parse(&src)?)));
            }
            cur = dir.parent().map(Path::to_path_buf);
        }
        Ok(None)
    }
}

fn resolve_chip_file_from_hint(
    board_dir: Option<&Path>,
    chip_name: &str,
    chip_override: Option<&str>,
) -> Result<PathBuf, CodegenError> {
    if let Some(ov) = chip_override {
        let p = PathBuf::from(ov);
        if p.is_file() {
            return Ok(p);
        }
    }
    let cname = chip_override.unwrap_or(chip_name);
    if let Some(bdir) = board_dir {
        if let Some(root) = bdir.parent() {
            let cand = root.join("chips").join(format!("{cname}.toml"));
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    for root in discover_spec_roots() {
        let cand = root.join("chips").join(format!("{cname}.toml"));
        if cand.is_file() {
            return Ok(cand);
        }
    }
    Err(CodegenError::new(
        "target",
        format!("could not locate chip spec '{cname}.toml'"),
    ))
}

/// Ordered candidate roots containing `chips/` and/or `boards/`.
pub fn discover_spec_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut seen = BTreeSet::new();
    let mut push = |p: PathBuf| {
        if (p.join("chips").is_dir() || p.join("boards").is_dir()) && seen.insert(p.clone()) {
            roots.push(p);
        }
    };

    if let Ok(env_root) = std::env::var("ZEV_HOME") {
        push(PathBuf::from(env_root));
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut cur: Option<&Path> = Some(&cwd);
        while let Some(d) = cur {
            push(d.to_path_buf());
            cur = d.parent();
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut cur = exe.parent();
        while let Some(d) = cur {
            push(d.to_path_buf());
            cur = d.parent();
        }
    }
    // Compile-time crate root fallback (`<repo>/crates/zev-codegen-llvm` -> `<repo>`).
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut cur: Option<&Path> = Some(manifest_dir);
    while let Some(d) = cur {
        push(d.to_path_buf());
        cur = d.parent();
    }
    roots
}

// ── Minimal Pure-Std TOML Subset Parser ──────────────────────────────────────

#[derive(Debug, Clone)]
enum TomlVal {
    Str(String),
    Int(i128),
    Bool(bool),
    Array(Vec<TomlVal>),
}

impl fmt::Display for TomlVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Str(s) => write!(f, "\"{s}\""),
            Self::Int(i) => write!(f, "{i}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Array(items) => {
                let parts: Vec<String> = items.iter().map(|i| i.to_string()).collect();
                write!(f, "[{}]", parts.join(", "))
            }
        }
    }
}

#[derive(Debug, Default)]
struct TomlDoc {
    sections: BTreeMap<String, BTreeMap<String, TomlVal>>,
}

impl TomlDoc {
    fn get(&self, sec: &str, key: &str) -> Option<&TomlVal> {
        self.sections.get(sec).and_then(|m| m.get(key))
    }

    fn req_str(&self, sec: &str, key: &str) -> Result<String, CodegenError> {
        match self.get(sec, key) {
            Some(TomlVal::Str(s)) => Ok(s.clone()),
            Some(other) => Err(CodegenError::new(
                "toml",
                format!("expected string for `[{sec}].{key}`, found {other}"),
            )),
            None => Err(CodegenError::new(
                "toml",
                format!("missing required field `[{sec}].{key}`"),
            )),
        }
    }

    fn opt_str(&self, sec: &str, key: &str) -> Result<Option<String>, CodegenError> {
        match self.get(sec, key) {
            None => Ok(None),
            Some(TomlVal::Str(s)) => Ok(Some(s.clone())),
            Some(other) => Err(CodegenError::new(
                "toml",
                format!("expected string for `[{sec}].{key}`, found {other}"),
            )),
        }
    }

    fn req_u32(&self, sec: &str, key: &str) -> Result<u32, CodegenError> {
        self.opt_u32(sec, key)?.ok_or_else(|| {
            CodegenError::new("toml", format!("missing required integer `[{sec}].{key}`"))
        })
    }

    fn opt_u32(&self, sec: &str, key: &str) -> Result<Option<u32>, CodegenError> {
        match self.get(sec, key) {
            None => Ok(None),
            Some(TomlVal::Int(v)) => u32::try_from(*v).map(Some).map_err(|_| {
                CodegenError::new(
                    "toml",
                    format!("integer `[{sec}].{key}` ({v}) out of u32 range"),
                )
            }),
            Some(other) => Err(CodegenError::new(
                "toml",
                format!("expected integer for `[{sec}].{key}`, found {other}"),
            )),
        }
    }

    fn opt_bool(&self, sec: &str, key: &str) -> Result<Option<bool>, CodegenError> {
        match self.get(sec, key) {
            None => Ok(None),
            Some(TomlVal::Bool(b)) => Ok(Some(*b)),
            Some(other) => Err(CodegenError::new(
                "toml",
                format!("expected boolean for `[{sec}].{key}`, found {other}"),
            )),
        }
    }

    fn opt_str_array(&self, sec: &str, key: &str) -> Result<Option<Vec<String>>, CodegenError> {
        match self.get(sec, key) {
            None => Ok(None),
            Some(TomlVal::Array(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        TomlVal::Str(s) => out.push(s.clone()),
                        other => {
                            return Err(CodegenError::new(
                                "toml",
                                format!("expected string element in `[{sec}].{key}`, found {other}"),
                            ));
                        }
                    }
                }
                Ok(Some(out))
            }
            Some(other) => Err(CodegenError::new(
                "toml",
                format!("expected array for `[{sec}].{key}`, found {other}"),
            )),
        }
    }

    fn opt_u32_array(&self, sec: &str, key: &str) -> Result<Option<Vec<u32>>, CodegenError> {
        match self.get(sec, key) {
            None => Ok(None),
            Some(TomlVal::Array(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        TomlVal::Int(v) => out.push(u32::try_from(*v).map_err(|_| {
                            CodegenError::new(
                                "toml",
                                format!("integer in `[{sec}].{key}` ({v}) out of u32 range"),
                            )
                        })?),
                        other => {
                            return Err(CodegenError::new(
                                "toml",
                                format!("expected integer element in `[{sec}].{key}`, found {other}"),
                            ));
                        }
                    }
                }
                Ok(Some(out))
            }
            Some(other) => Err(CodegenError::new(
                "toml",
                format!("expected array for `[{sec}].{key}`, found {other}"),
            )),
        }
    }
}

fn strip_toml_comment(line: &str) -> String {
    let mut out = String::new();
    let mut in_str = false;
    let mut esc = false;
    for ch in line.chars() {
        if esc {
            out.push(ch);
            esc = false;
            continue;
        }
        if in_str {
            if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
            out.push(ch);
            continue;
        }
        if ch == '#' {
            break;
        }
        if ch == '"' {
            in_str = true;
        }
        out.push(ch);
    }
    out
}

fn parse_toml(src: &str) -> Result<TomlDoc, CodegenError> {
    let mut doc = TomlDoc::default();
    let mut cur_section = String::new();
    let raw_lines: Vec<String> = src.lines().map(strip_toml_comment).collect();
    let mut idx = 0usize;
    while idx < raw_lines.len() {
        let mut trimmed = raw_lines[idx].trim().to_string();
        idx += 1;
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('[') && !trimmed.contains('=') {
            if !trimmed.ends_with(']') {
                return Err(CodegenError::new(
                    "toml",
                    format!("unterminated section header on line {idx}: `{trimmed}`"),
                ));
            }
            cur_section = trimmed[1..trimmed.len() - 1].trim().to_string();
            continue;
        }
        let Some(eq_pos) = trimmed.find('=') else {
            return Err(CodegenError::new(
                "toml",
                format!("expected `key = value` on line {idx}: `{trimmed}`"),
            ));
        };
        let key = trimmed[..eq_pos].trim().to_string();
        let mut val_str = trimmed[eq_pos + 1..].trim().to_string();
        // Join multi-line arrays `[` ... `]`
        while val_str.starts_with('[') && !has_balanced_brackets(&val_str) {
            if idx >= raw_lines.len() {
                return Err(CodegenError::new(
                    "toml",
                    format!("unterminated array value for key `{key}`"),
                ));
            }
            val_str.push(' ');
            val_str.push_str(raw_lines[idx].trim());
            idx += 1;
            trimmed.clear();
        }
        let val = parse_toml_val(val_str.trim(), idx)?;
        doc.sections
            .entry(cur_section.clone())
            .or_default()
            .insert(key, val);
    }
    Ok(doc)
}

fn has_balanced_brackets(s: &str) -> bool {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for ch in s.chars() {
        if esc {
            esc = false;
            continue;
        }
        if in_str {
            if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '[' => depth += 1,
            ']' => depth -= 1,
            _ => {}
        }
    }
    depth <= 0
}

fn parse_toml_val(s: &str, line_no: usize) -> Result<TomlVal, CodegenError> {
    if s.starts_with('"') {
        let (parsed, rest) = parse_quoted_str(s, line_no)?;
        if !rest.trim().is_empty() {
            return Err(CodegenError::new(
                "toml",
                format!("unexpected trailing tokens on line {line_no}: `{rest}`"),
            ));
        }
        return Ok(TomlVal::Str(parsed));
    }
    if s.starts_with('[') {
        if !s.ends_with(']') {
            return Err(CodegenError::new(
                "toml",
                format!("unterminated array on line {line_no}: `{s}`"),
            ));
        }
        let inner = s[1..s.len() - 1].trim();
        let mut items = Vec::new();
        let mut pos = 0usize;
        let bytes = inner.as_bytes();
        while pos < bytes.len() {
            while pos < bytes.len() && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b',') {
                pos += 1;
            }
            if pos >= bytes.len() {
                break;
            }
            if bytes[pos] == b'"' {
                let (st, rest) = parse_quoted_str(&inner[pos..], line_no)?;
                items.push(TomlVal::Str(st));
                pos = inner.len() - rest.len();
            } else {
                let start = pos;
                while pos < bytes.len() && bytes[pos] != b',' && bytes[pos] != b']' {
                    pos += 1;
                }
                let tok = inner[start..pos].trim();
                items.push(parse_toml_val(tok, line_no)?);
            }
        }
        return Ok(TomlVal::Array(items));
    }
    if s == "true" {
        return Ok(TomlVal::Bool(true));
    }
    if s == "false" {
        return Ok(TomlVal::Bool(false));
    }
    let clean: String = s.chars().filter(|&c| c != '_').collect();
    if let Some(hex) = clean.strip_prefix("0x").or_else(|| clean.strip_prefix("0X")) {
        let v = i128::from_str_radix(hex, 16).map_err(|e| {
            CodegenError::new("toml", format!("invalid hex integer `{s}` on line {line_no}: {e}"))
        })?;
        return Ok(TomlVal::Int(v));
    }
    if let Some(bin) = clean.strip_prefix("0b").or_else(|| clean.strip_prefix("0B")) {
        let v = i128::from_str_radix(bin, 2).map_err(|e| {
            CodegenError::new(
                "toml",
                format!("invalid binary integer `{s}` on line {line_no}: {e}"),
            )
        })?;
        return Ok(TomlVal::Int(v));
    }
    let v = clean.parse::<i128>().map_err(|e| {
        CodegenError::new("toml", format!("invalid value `{s}` on line {line_no}: {e}"))
    })?;
    Ok(TomlVal::Int(v))
}

fn parse_quoted_str(s: &str, line_no: usize) -> Result<(String, &str), CodegenError> {
    let mut chars = s.char_indices();
    let Some((_, '"')) = chars.next() else {
        return Err(CodegenError::new("toml", format!("expected `\"` on line {line_no}")));
    };
    let mut out = String::new();
    let mut esc = false;
    for (idx, ch) in chars {
        if esc {
            match ch {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                other => out.push(other),
            }
            esc = false;
        } else if ch == '\\' {
            esc = true;
        } else if ch == '"' {
            return Ok((out, &s[idx + 1..]));
        } else {
            out.push(ch);
        }
    }
    Err(CodegenError::new(
        "toml",
        format!("unterminated string literal on line {line_no}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_project_manifest_and_custom_toml_spec() {
        let toml = r#"
# Example zev.toml
[project]
name = "motor_node"
entry = "src/main.zev"
board = "custom_board_demo"
port = "/dev/ttyACM9"
"#;
        let mf = ProjectManifest::parse(toml).expect("parse zev.toml");
        assert_eq!(mf.name.as_deref(), Some("motor_node"));
        assert_eq!(mf.entry.as_deref(), Some(Path::new("src/main.zev")));
        assert_eq!(mf.board.as_deref(), Some("custom_board_demo"));
        assert_eq!(mf.port.as_deref(), Some("/dev/ttyACM9"));

        let chip_toml = r#"
[chip]
name = "generic_cm"
isa = "arm-cortex-m"
cpu = "cortex-m4"
triple = "thumbv7em-none-eabi"
datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
has_thumb2 = true
has_vtor = true

[memory]
flash_base = 0x0800_0000
flash_size = 524288
ram_base = 0x2000_0000
ram_size = 131072

[irq]
model = "cortex-m-nvic"
max_irqs = 82

[image]
format = "elf"
"#;
        let board_toml = r#"
[board]
name = "generic_eval"
chip = "generic_cm"

[run]
qemu_machine = "netduinoplus2"
semihosting = true
"#;
        let spec = TargetSpec::from_tomls(chip_toml, Some(board_toml)).expect("merge custom tomls");
        assert_eq!(spec.chip_name, "generic_cm");
        assert_eq!(spec.board_name.as_deref(), Some("generic_eval"));
        assert_eq!(spec.isa, CpuIsa::ArmCortexM);
        assert_eq!(spec.effective_flash_size(), 524_288);
        assert_eq!(spec.ram_size, 131_072);
        assert!(spec.semihosting);
    }
}
