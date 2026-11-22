//! Recursive `.zev` module resolution, `pub` visibility enforcement, cycle/diamond handling,
//! `MmioDeclKind::Inline` desugaring, and canonical symbol linking (`<mod>__<name>`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use zev_ast::{
    ArrayLen, Block, CallArg, CatchClause, CatchParam, Expr, ExprKind, FieldInit, FnDecl,
    FormatPart, GenericArg, GenericParamKind, Ident, IsrDecl, Item, ItemKind, MmioDecl,
    MmioDeclKind, OsPrimitiveType, Param, Program, SpawnStmt, Span, Stmt, StmtKind, SwitchArm,
    SwitchPat, TaskDecl, TypeExpr, TypeKind,
};

use crate::diag::{Diagnostic, DiagnosticKind};

/// Canonical identity of a resolved non-root module.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ModuleKey {
    File(PathBuf),
    Embedded(String),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ExportNs {
    Type,
    Value,
}

#[derive(Clone, Debug)]
struct ExportedSymbol {
    canonical_name: String,
    is_pub: bool,
    is_extern_c: bool,
}

#[derive(Clone, Debug, Default)]
struct ModuleExports {
    types: BTreeMap<String, ExportedSymbol>,
    values: BTreeMap<String, ExportedSymbol>,
}

/// Resolution of a single `import` statement inside a file.
#[derive(Clone, Debug)]
enum ResolvedImport {
    Module {
        key: ModuleKey,
        dot_path: String,
    },
    LegacyChip,
}

#[derive(Clone, Debug)]
struct LoadedModule {
    dot_path: String,
    prefix: String,
    file_display: String,
    program: Program,
    exports: ModuleExports,
    /// `alias -> resolved import target` for imports declared in this module.
    imports: BTreeMap<String, ResolvedImport>,
    /// Legacy fallback aliases kept as `ItemKind::Import`.
    legacy_aliases: BTreeSet<String>,
    /// `extern "C"` prototypes/functions declared directly inside this module.
    declared_extern_c: BTreeSet<String>,
}

/// Result of recursively resolving, linking, and canonicalizing root + imported `.zev` programs.
#[derive(Clone, Debug)]
pub struct LinkedModulesOutput {
    /// Linked `(Program, Option<file_path>)` sequence (imported modules first, then root programs).
    pub programs: Vec<(Program, Option<String>)>,
    /// Module dot-paths (`"chip.stm32g4"`, `"drivers.uart"`) that resolved to real `.zev` modules.
    pub loaded_module_paths: BTreeSet<String>,
    /// Human-readable display names for canonicalized module symbols (`mod__Foo` -> `mod.Foo`).
    pub display_names: BTreeMap<String, String>,
    /// Per-program allowed `extern "C"` names declared in that program's scope.
    pub program_extern_c_scopes: Vec<BTreeSet<String>>,
    /// Exact `(program_index, span.start, span.end)` sites rewritten from `alias.pub_extern_c`.
    pub rewritten_pub_extern_sites: BTreeSet<(usize, u32, u32)>,
    /// Non-root `extern "C"` names (used by `checker` to prevent un-imported private leaks).
    pub non_root_extern_c_names: BTreeSet<String>,
    /// Diagnostics emitted during module discovery, parsing, cycle detection, or `pub` enforcement.
    pub diagnostics: Vec<Diagnostic>,
}

/// Embedded fallback source lookup for built-in library modules when no disk override exists.
///
/// NOTE: `chip.*` targets intentionally do NOT register stubs here yet so pre-Step-2 tests keep
/// using the legacy `NamespaceKind::Chip` fallback unless an actual `chip/*.zev` file exists on disk
/// or in `include_dirs`.
pub fn embedded_module_source(_dot_path: &str) -> Option<&'static str> {
    None
}

/// Recursively resolve all `import` declarations across `programs` + `include_dirs`, enforce
/// `pub` visibility and acyclic module graphs (`E0030`), canonicalize non-root module symbols to
/// `<mod>__<name>` (keeping `extern "C"` names unmangled), and desugar `MmioDeclKind::Inline`.
pub fn load_and_link_modules(
    programs: &[(&Program, Option<&str>)],
    include_dirs: &[PathBuf],
) -> LinkedModulesOutput {
    let mut linker = ModuleLinker::new(programs, include_dirs);
    linker.run(programs)
}

struct ModuleLinker {
    include_dirs: Vec<PathBuf>,
    entry_dirs: Vec<PathBuf>,
    used_prefixes: BTreeSet<String>,
    visiting: Vec<(ModuleKey, String)>,
    loaded: BTreeMap<ModuleKey, LoadedModule>,
    load_order: Vec<ModuleKey>,
    loaded_module_paths: BTreeSet<String>,
    display_names: BTreeMap<String, String>,
    diagnostics: Vec<Diagnostic>,
}

impl ModuleLinker {
    fn new(programs: &[(&Program, Option<&str>)], include_dirs: &[PathBuf]) -> Self {
        let mut entry_dirs = Vec::new();
        for (_, fname) in programs {
            if let Some(dir) = fname.and_then(real_parent_dir) {
                if !entry_dirs.contains(&dir) {
                    entry_dirs.push(dir);
                }
            }
        }
        Self {
            include_dirs: include_dirs.to_vec(),
            entry_dirs,
            used_prefixes: BTreeSet::new(),
            visiting: Vec::new(),
            loaded: BTreeMap::new(),
            load_order: Vec::new(),
            loaded_module_paths: BTreeSet::new(),
            display_names: BTreeMap::new(),
            diagnostics: Vec::new(),
        }
    }

    fn emit(&mut self, mut diag: Diagnostic, file: Option<&str>) {
        if diag.file.is_none() {
            diag.file = file.map(str::to_owned);
        }
        self.diagnostics.push(diag);
    }

    fn alloc_prefix(&mut self, segments: &[Ident]) -> String {
        let base = segments
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join("_");
        let base = if base.is_empty() {
            "mod".to_owned()
        } else {
            base
        };
        if self.used_prefixes.insert(base.clone()) {
            return base;
        }
        let mut idx = 1usize;
        loop {
            let candidate = format!("{base}_{idx}");
            if self.used_prefixes.insert(candidate.clone()) {
                return candidate;
            }
            idx += 1;
        }
    }

    fn run(&mut self, root_programs: &[(&Program, Option<&str>)]) -> LinkedModulesOutput {
        // 1. Resolve imports recursively for all root programs, loading dependencies bottom-up.
        let mut root_imports: Vec<(BTreeMap<String, ResolvedImport>, BTreeSet<String>)> =
            Vec::with_capacity(root_programs.len());
        for (prog, fname) in root_programs {
            let importer_path = fname.and_then(real_file_path);
            let res = self.resolve_imports_in_program(
                prog,
                importer_path.as_deref(),
                *fname,
            );
            root_imports.push(res);
        }

        // 2. Build module-local `type`/`value` symbol tables for each loaded module so references
        //    within a module rewrite cleanly before external imports rewrite across boundaries.
        let order = self.load_order.clone();
        let mut out_programs: Vec<(Program, Option<String>)> =
            Vec::with_capacity(order.len() + root_programs.len());
        let mut program_extern_c_scopes: Vec<BTreeSet<String>> =
            Vec::with_capacity(order.len() + root_programs.len());
        let mut rewritten_pub_extern_sites: BTreeSet<(usize, u32, u32)> = BTreeSet::new();
        let mut non_root_extern_c_names: BTreeSet<String> = BTreeSet::new();

        // Record all non-root extern "C" names globally across loaded modules for isolation checks.
        for key in &order {
            if let Some(m) = self.loaded.get(key) {
                non_root_extern_c_names.extend(m.declared_extern_c.iter().cloned());
            }
        }

        // Snapshot immutable export/prefix tables for rewrite lookups while mutating diagnostics.
        let exports_by_key: BTreeMap<ModuleKey, (String, ModuleExports)> = self
            .loaded
            .iter()
            .map(|(k, v)| (k.clone(), (v.dot_path.clone(), v.exports.clone())))
            .collect();

        // Track `extern "C"` prototypes across all modules to deduplicate identical declarations
        // and reject conflicting signatures with `E0030`.
        let mut seen_extern_c: BTreeMap<String, (FnDecl, Option<String>)> = BTreeMap::new();

        for (prog_idx, key) in order.iter().enumerate() {
            let m = self.loaded.get(key).expect("loaded module in order").clone();
            let mut diags = Vec::new();
            let rewritten = rewrite_program(
                &m.program,
                Some(&m.prefix),
                Some(&m.dot_path),
                &m.exports,
                &m.imports,
                &m.legacy_aliases,
                &exports_by_key,
                &mut self.display_names,
                prog_idx,
                m.file_display.as_str(),
                true,
                &mut seen_extern_c,
                &mut rewritten_pub_extern_sites,
                &mut diags,
            );
            for mut d in diags {
                if d.file.is_none() {
                    d.file = Some(m.file_display.clone());
                }
                self.diagnostics.push(d);
            }
            out_programs.push((rewritten, Some(m.file_display)));
            program_extern_c_scopes.push(m.declared_extern_c);
        }

        // 3. Desugar `Inline` mmio + rewrite imported module references in all root programs.
        for (offset, ((prog, fname), (imports, legacy_aliases))) in root_programs
            .iter()
            .zip(root_imports.into_iter())
            .enumerate()
        {
            let prog_idx = order.len() + offset;
            let desugared = desugar_inline_mmios(prog);
            let root_externs = collect_declared_extern_c(&desugared);
            let empty_exports = ModuleExports::default();
            let file_label = fname.unwrap_or("<input>");
            let mut diags = Vec::new();
            let rewritten = rewrite_program(
                &desugared,
                None,
                None,
                &empty_exports,
                &imports,
                &legacy_aliases,
                &exports_by_key,
                &mut self.display_names,
                prog_idx,
                file_label,
                false,
                &mut seen_extern_c,
                &mut rewritten_pub_extern_sites,
                &mut diags,
            );
            for mut d in diags {
                if d.file.is_none() {
                    d.file = fname.map(str::to_owned);
                }
                self.diagnostics.push(d);
            }
            out_programs.push((rewritten, fname.map(str::to_owned)));
            program_extern_c_scopes.push(root_externs);
        }

        LinkedModulesOutput {
            programs: out_programs,
            loaded_module_paths: self.loaded_module_paths.clone(),
            display_names: self.display_names.clone(),
            program_extern_c_scopes,
            rewritten_pub_extern_sites,
            non_root_extern_c_names,
            diagnostics: std::mem::take(&mut self.diagnostics),
        }
    }

    fn resolve_imports_in_program(
        &mut self,
        prog: &Program,
        importer_file: Option<&Path>,
        importer_label: Option<&str>,
    ) -> (BTreeMap<String, ResolvedImport>, BTreeSet<String>) {
        let mut imports = BTreeMap::new();
        let mut legacy_aliases = BTreeSet::new();

        for item in &prog.items {
            let ItemKind::Import(imp) = &item.kind else {
                continue;
            };
            let dot_path = imp
                .path
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
                .join(".");
            let alias = imp
                .alias
                .as_ref()
                .map(|a| a.name.clone())
                .or_else(|| imp.path.last().map(|l| l.name.clone()))
                .unwrap_or_else(|| dot_path.clone());

            if matches!(alias.as_str(), "sys" | "sim") {
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::ModuleError,
                        imp.span,
                        format!("import alias `{alias}` collides with built-in namespace `{alias}`"),
                    ),
                    importer_label,
                );
                continue;
            }
            if imports.contains_key(&alias) {
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::ModuleError,
                        imp.span,
                        format!("duplicate module import alias `{alias}`"),
                    ),
                    importer_label,
                );
                continue;
            }

            match self.locate_module(&imp.path, &dot_path, importer_file, imp.span, importer_label) {
                LocateResult::FoundFile(path) => {
                    let canon = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                    let key = ModuleKey::File(canon.clone());
                    self.loaded_module_paths.insert(dot_path.clone());
                    if self.ensure_loaded(
                        key.clone(),
                        &imp.path,
                        &dot_path,
                        Some(&path),
                        None,
                        imp.span,
                        importer_label,
                    ) {
                        imports.insert(alias, ResolvedImport::Module { key, dot_path });
                    }
                }
                LocateResult::FoundEmbedded(src) => {
                    let key = ModuleKey::Embedded(dot_path.clone());
                    self.loaded_module_paths.insert(dot_path.clone());
                    if self.ensure_loaded(
                        key.clone(),
                        &imp.path,
                        &dot_path,
                        None,
                        Some(src),
                        imp.span,
                        importer_label,
                    ) {
                        imports.insert(alias, ResolvedImport::Module { key, dot_path });
                    }
                }
                LocateResult::NotFound => {
                    // Legacy pre-Step-2 fallback: unresolved `chip.*` uses `NamespaceKind::Chip`.
                    if imp.path.first().is_some_and(|p| p.name == "chip") {
                        legacy_aliases.insert(alias.clone());
                        imports.insert(alias, ResolvedImport::LegacyChip);
                    } else {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::ModuleError,
                                imp.span,
                                format!("cannot resolve imported module `{dot_path}`"),
                            )
                            .with_note(
                                "checked importing directory, workspace ancestors, include directories, and embedded modules",
                            ),
                            importer_label,
                        );
                    }
                }
                LocateResult::Error => {}
            }
        }

        (imports, legacy_aliases)
    }

    fn ensure_loaded(
        &mut self,
        key: ModuleKey,
        segments: &[Ident],
        dot_path: &str,
        disk_path: Option<&Path>,
        embedded_src: Option<&'static str>,
        import_span: Span,
        importer_label: Option<&str>,
    ) -> bool {
        if let Some(pos) = self.visiting.iter().position(|(k, _)| k == &key) {
            let mut chain: Vec<&str> = self.visiting[pos..].iter().map(|(_, p)| p.as_str()).collect();
            chain.push(dot_path);
            self.emit(
                Diagnostic::new(
                    DiagnosticKind::ModuleError,
                    import_span,
                    format!("circular module import detected: {}", chain.join(" -> ")),
                ),
                importer_label,
            );
            return false;
        }
        if self.loaded.contains_key(&key) {
            // Diamond import already resolved/parsed.
            return true;
        }

        let (file_display, source) = match (disk_path, embedded_src) {
            (Some(p), _) => {
                let disp = p.to_string_lossy().into_owned();
                match std::fs::read_to_string(p) {
                    Ok(s) => (disp, s),
                    Err(e) => {
                        self.emit(
                            Diagnostic::new(
                                DiagnosticKind::ModuleError,
                                import_span,
                                format!("failed to read module `{dot_path}` at `{disp}`: {e}"),
                            ),
                            importer_label,
                        );
                        return false;
                    }
                }
            }
            (None, Some(s)) => (format!("<embedded:{dot_path}>"), s.to_owned()),
            (None, None) => return false,
        };

        let parsed = match zev_parser::parse_str_named(&source, &file_display) {
            Ok(p) => p,
            Err(pe) => {
                let mut d = Diagnostic::new(
                    DiagnosticKind::ModuleError,
                    pe.span,
                    format!("syntax error while parsing module `{dot_path}`: {}", pe.message),
                )
                .with_file(Some(file_display));
                if let Some(hint) = pe.hint {
                    d = d.with_note(hint);
                }
                self.diagnostics.push(d);
                return false;
            }
        };

        let desugared = desugar_inline_mmios(&parsed);
        let prefix = self.alloc_prefix(segments);
        let exports = collect_module_exports(&desugared, &prefix);
        let declared_extern_c = collect_declared_extern_c(&desugared);

        self.visiting.push((key.clone(), dot_path.to_owned()));
        let (imports, legacy_aliases) =
            self.resolve_imports_in_program(&desugared, disk_path, Some(&file_display));
        self.visiting.pop();

        self.loaded.insert(
            key.clone(),
            LoadedModule {
                dot_path: dot_path.to_owned(),
                prefix,
                file_display,
                program: desugared,
                exports,
                imports,
                legacy_aliases,
                declared_extern_c,
            },
        );
        self.load_order.push(key);
        true
    }

    fn locate_module(
        &mut self,
        segments: &[Ident],
        dot_path: &str,
        importer_file: Option<&Path>,
        import_span: Span,
        importer_label: Option<&str>,
    ) -> LocateResult {
        if segments.is_empty() {
            return LocateResult::NotFound;
        }
        let mut rel_file = PathBuf::new();
        for seg in segments {
            rel_file.push(&seg.name);
        }
        let mut rel_mod = rel_file.clone();
        rel_file.set_extension("zev");
        rel_mod.push("mod.zev");

        let candidate_dirs = self.search_dirs(importer_file);
        for dir in candidate_dirs {
            let file_cand = dir.join(&rel_file);
            let mod_cand = dir.join(&rel_mod);
            let file_exists = file_cand.is_file();
            let mod_exists = mod_cand.is_file();
            if file_exists && mod_exists {
                self.emit(
                    Diagnostic::new(
                        DiagnosticKind::ModuleError,
                        import_span,
                        format!(
                            "ambiguous module `{dot_path}`: both `{}` and `{}` exist",
                            file_cand.display(),
                            mod_cand.display()
                        ),
                    ),
                    importer_label,
                );
                return LocateResult::Error;
            }
            if file_exists {
                return LocateResult::FoundFile(file_cand);
            }
            if mod_exists {
                return LocateResult::FoundFile(mod_cand);
            }
        }

        if let Some(src) = embedded_module_source(dot_path) {
            return LocateResult::FoundEmbedded(src);
        }
        LocateResult::NotFound
    }

    fn search_dirs(&self, importer_file: Option<&Path>) -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        let mut push_unique = |p: PathBuf| {
            let norm = normalize_abs(&p);
            if !dirs.contains(&norm) {
                dirs.push(norm);
            }
        };

        // (1a) Importer's directory and root entry directories.
        let importer_dir = importer_file.and_then(|p| p.parent().map(normalize_abs));
        if let Some(ref d) = importer_dir {
            push_unique(d.clone());
        }
        for d in &self.entry_dirs {
            push_unique(d.clone());
        }

        // (1b) Workspace ancestors of importer/entry dirs (or cwd if in-memory `<input>`).
        let mut seeds = Vec::new();
        if let Some(d) = importer_dir {
            seeds.push(d);
        }
        for d in &self.entry_dirs {
            seeds.push(d.clone());
        }
        if seeds.is_empty() {
            if let Ok(cwd) = std::env::current_dir() {
                seeds.push(cwd);
            }
        }
        for seed in seeds {
            for anc in workspace_ancestors(&seed) {
                push_unique(anc);
            }
        }

        // (2) Explicit include directories (`-I` / `AnalyzeOptions.include_dirs`).
        for inc in &self.include_dirs {
            push_unique(inc.clone());
        }

        dirs
    }
}

enum LocateResult {
    FoundFile(PathBuf),
    FoundEmbedded(&'static str),
    NotFound,
    Error,
}

fn real_file_path(label: &str) -> Option<PathBuf> {
    if label.is_empty() || (label.starts_with('<') && label.ends_with('>')) {
        return None;
    }
    Some(normalize_abs(Path::new(label)))
}

fn real_parent_dir(label: &str) -> Option<PathBuf> {
    real_file_path(label).and_then(|p| p.parent().map(Path::to_path_buf))
}

fn normalize_abs(p: &Path) -> PathBuf {
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else if let Ok(cwd) = std::env::current_dir() {
        cwd.join(p)
    } else {
        p.to_path_buf()
    };
    let mut out = PathBuf::new();
    for comp in abs.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                let _ = out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn workspace_ancestors(start: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut cur = start;
    // Walk up at most 6 parents (never searching filesystem `/` itself), stopping after hitting
    // a workspace marker (`Zev.toml`, `Cargo.toml`, or `.git`).
    for _ in 0..6 {
        let Some(parent) = cur.parent() else {
            break;
        };
        if parent.as_os_str() == "/" || parent.as_os_str().is_empty() {
            break;
        }
        out.push(parent.to_path_buf());
        if parent.join("Zev.toml").exists()
            || parent.join("Cargo.toml").exists()
            || parent.join(".git").exists()
        {
            break;
        }
        cur = parent;
    }
    out
}

fn desugar_inline_mmios(prog: &Program) -> Program {
    let mut items = Vec::with_capacity(prog.items.len());
    for item in &prog.items {
        if let ItemKind::Mmio(mmio) = &item.kind {
            if let MmioDeclKind::Inline { base_addr, regs } = &mmio.kind {
                let layout_name = format!("{}__Layout", mmio.name.name);
                let layout_ident = Ident::new(layout_name, mmio.name.span);
                let layout_decl = MmioDecl {
                    name: layout_ident.clone(),
                    kind: MmioDeclKind::Layout { regs: regs.clone() },
                    span: mmio.span,
                };
                let inst_decl = MmioDecl {
                    name: mmio.name.clone(),
                    kind: MmioDeclKind::Instance {
                        layout: TypeExpr::new(TypeKind::Named(layout_ident), mmio.name.span),
                        base_addr: base_addr.clone(),
                    },
                    span: mmio.span,
                };
                items.push(Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::Mmio(layout_decl),
                    item.span,
                ));
                items.push(Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::Mmio(inst_decl),
                    item.span,
                ));
                continue;
            }
        }
        items.push(item.clone());
    }
    Program::new(items, prog.span)
}

fn collect_declared_extern_c(prog: &Program) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for item in &prog.items {
        if let ItemKind::Fn(f) = &item.kind {
            if f.is_extern_c {
                set.insert(f.name.name.clone());
            }
        }
    }
    set
}

fn collect_module_exports(prog: &Program, prefix: &str) -> ModuleExports {
    let mut exports = ModuleExports::default();
    let mut insert_type = |name: &str, canonical: String, is_pub: bool| {
        exports
            .types
            .entry(name.to_owned())
            .or_insert(ExportedSymbol {
                canonical_name: canonical,
                is_pub,
                is_extern_c: false,
            });
    };
    let mut insert_val = |name: &str, canonical: String, is_pub: bool, is_extern_c: bool| {
        exports
            .values
            .entry(name.to_owned())
            .or_insert(ExportedSymbol {
                canonical_name: canonical,
                is_pub,
                is_extern_c,
            });
    };

    for item in &prog.items {
        match &item.kind {
            ItemKind::Struct(st) => {
                insert_type(
                    &st.name.name,
                    format!("{prefix}__{}", st.name.name),
                    item.is_pub,
                );
            }
            ItemKind::Enum(en) => {
                insert_type(
                    &en.name.name,
                    format!("{prefix}__{}", en.name.name),
                    item.is_pub,
                );
            }
            ItemKind::Mmio(mm) => match &mm.kind {
                MmioDeclKind::Layout { .. } => {
                    insert_type(
                        &mm.name.name,
                        format!("{prefix}__{}", mm.name.name),
                        item.is_pub,
                    );
                }
                MmioDeclKind::Instance { .. } => {
                    insert_val(
                        &mm.name.name,
                        format!("{prefix}__{}", mm.name.name),
                        item.is_pub,
                        false,
                    );
                }
                MmioDeclKind::Inline { .. } => {
                    insert_type(
                        &format!("{}__Layout", mm.name.name),
                        format!("{prefix}__{}__Layout", mm.name.name),
                        item.is_pub,
                    );
                    insert_val(
                        &mm.name.name,
                        format!("{prefix}__{}", mm.name.name),
                        item.is_pub,
                        false,
                    );
                }
            },
            ItemKind::Irq(iq) => {
                insert_val(
                    &iq.name.name,
                    format!("{prefix}__{}", iq.name.name),
                    item.is_pub,
                    false,
                );
            }
            ItemKind::Const(c) => {
                insert_val(
                    &c.name.name,
                    format!("{prefix}__{}", c.name.name),
                    item.is_pub,
                    false,
                );
            }
            ItemKind::GlobalVar(g) => {
                insert_val(
                    &g.name.name,
                    format!("{prefix}__{}", g.name.name),
                    item.is_pub,
                    false,
                );
            }
            ItemKind::OsPrimitive(p) => {
                insert_val(
                    &p.name.name,
                    format!("{prefix}__{}", p.name.name),
                    item.is_pub,
                    false,
                );
            }
            ItemKind::Fn(f) => {
                // `extern "C"` symbols stay unmangled so C linkage matches exact C ABI names.
                let canonical = if f.is_extern_c {
                    f.name.name.clone()
                } else {
                    format!("{prefix}__{}", f.name.name)
                };
                insert_val(&f.name.name, canonical, item.is_pub, f.is_extern_c);
            }
            ItemKind::Task(t) => {
                insert_val(
                    &t.name.name,
                    format!("{prefix}__{}", t.name.name),
                    item.is_pub,
                    false,
                );
            }
            ItemKind::Isr(i) => {
                insert_val(
                    &i.vector.name,
                    format!("{prefix}__{}", i.vector.name),
                    item.is_pub,
                    false,
                );
            }
            ItemKind::Import(_) => {}
        }
    }

    exports
}

struct AstRewriter<'a> {
    prefix: Option<&'a str>,
    module_dot_path: Option<&'a str>,
    self_exports: &'a ModuleExports,
    imports: &'a BTreeMap<String, ResolvedImport>,
    legacy_aliases: &'a BTreeSet<String>,
    exports_by_key: &'a BTreeMap<ModuleKey, (String, ModuleExports)>,
    display_names: &'a mut BTreeMap<String, String>,
    prog_idx: usize,
    file_label: &'a str,
    rewritten_pub_extern_sites: &'a mut BTreeSet<(usize, u32, u32)>,
    diagnostics: &'a mut Vec<Diagnostic>,
    local_scopes: Vec<BTreeSet<String>>,
    generic_type_params: BTreeSet<String>,
}

impl<'a> AstRewriter<'a> {
    fn emit(&mut self, span: Span, msg: impl Into<String>) {
        self.diagnostics.push(
            Diagnostic::new(DiagnosticKind::ModuleError, span, msg.into())
                .with_file(Some(self.file_label.to_owned())),
        );
    }

    fn record_display_name(&mut self, canonical: &str, readable: String) {
        self.display_names
            .entry(canonical.to_owned())
            .or_insert(readable);
    }

    fn self_type_canonical(&self, name: &str) -> Option<&str> {
        self.self_exports
            .types
            .get(name)
            .map(|e| e.canonical_name.as_str())
    }

    fn self_value_canonical(&self, name: &str) -> Option<&str> {
        self.self_exports
            .values
            .get(name)
            .map(|e| e.canonical_name.as_str())
    }

    fn push_scope(&mut self) {
        self.local_scopes.push(BTreeSet::new());
    }

    fn pop_scope(&mut self) {
        let _ = self.local_scopes.pop();
    }

    fn define_local(&mut self, name: &str) {
        if let Some(top) = self.local_scopes.last_mut() {
            top.insert(name.to_owned());
        }
    }

    fn in_local_scope(&self, name: &str) -> bool {
        self.local_scopes.iter().rev().any(|s| s.contains(name))
    }

    fn lookup_imported_member(
        &mut self,
        alias: &str,
        member: &str,
        preferred_ns: Option<ExportNs>,
        span: Span,
    ) -> Option<ExportedSymbol> {
        let Some(res) = self.imports.get(alias) else {
            return None;
        };
        let ResolvedImport::Module { key, dot_path } = res else {
            return None;
        };
        let Some((_, exports)) = self.exports_by_key.get(key) else {
            return None;
        };

        let found = match preferred_ns {
            Some(ExportNs::Type) => exports.types.get(member),
            Some(ExportNs::Value) => exports.values.get(member),
            None => exports
                .values
                .get(member)
                .or_else(|| exports.types.get(member)),
        };

        let Some(sym) = found else {
            self.emit(
                span,
                format!("module `{dot_path}` (imported as `{alias}`) has no member `{member}`"),
            );
            return None;
        };
        if !sym.is_pub {
            self.emit(
                span,
                format!(
                    "`{alias}.{member}` in module `{dot_path}` is private; mark it `pub` to export it"
                ),
            );
            return None;
        }
        let readable = format!("{dot_path}.{member}");
        let canonical = sym.canonical_name.clone();
        let out = sym.clone();
        self.record_display_name(&canonical, readable);
        Some(out)
    }

    fn is_resolved_module_alias(&self, alias: &str) -> bool {
        matches!(self.imports.get(alias), Some(ResolvedImport::Module { .. }))
    }

    fn is_legacy_chip_alias(&self, alias: &str) -> bool {
        self.legacy_aliases.contains(alias)
    }

    fn canonicalize_top_name(&mut self, orig: &str) -> String {
        if let (Some(prefix), Some(dot)) = (self.prefix, self.module_dot_path) {
            let can = format!("{prefix}__{orig}");
            self.record_display_name(&can, format!("{dot}.{orig}"));
            can
        } else {
            orig.to_owned()
        }
    }

    fn rewrite_type(&mut self, ty: &TypeExpr) -> TypeExpr {
        let span = ty.span;
        let kind = match &ty.kind {
            TypeKind::Prim(p) => TypeKind::Prim(*p),
            TypeKind::Named(id) => {
                if !self.generic_type_params.contains(&id.name) {
                    if let Some(can) = self.self_type_canonical(&id.name) {
                        TypeKind::Named(Ident::new(can.to_owned(), id.span))
                    } else {
                        TypeKind::Named(id.clone())
                    }
                } else {
                    TypeKind::Named(id.clone())
                }
            }
            TypeKind::Path(segs) => {
                if let Some(first) = segs.first() {
                    if self.is_resolved_module_alias(&first.name) {
                        if segs.len() != 2 {
                            self.emit(
                                span,
                                format!(
                                    "expected `<alias>.<Type>` after module alias `{}`, found `{}`",
                                    first.name,
                                    segs.iter()
                                        .map(|s| s.name.as_str())
                                        .collect::<Vec<_>>()
                                        .join(".")
                                ),
                            );
                            TypeKind::Path(segs.clone())
                        } else if let Some(sym) = self.lookup_imported_member(
                            &first.name,
                            &segs[1].name,
                            Some(ExportNs::Type),
                            span,
                        ) {
                            TypeKind::Named(Ident::new(sym.canonical_name, span))
                        } else {
                            TypeKind::Path(segs.clone())
                        }
                    } else if self.is_legacy_chip_alias(&first.name) {
                        TypeKind::Path(segs.clone())
                    } else {
                        TypeKind::Path(segs.clone())
                    }
                } else {
                    TypeKind::Path(segs.clone())
                }
            }
            TypeKind::Generic { base, args } => {
                let new_args = args.iter().map(|a| self.rewrite_generic_arg(a)).collect();
                let new_base = if let Some(can) = self.self_type_canonical(&base.name) {
                    Ident::new(can.to_owned(), base.span)
                } else {
                    base.clone()
                };
                TypeKind::Generic {
                    base: new_base,
                    args: new_args,
                }
            }
            TypeKind::GenericPath { path, args } => {
                let new_args = args.iter().map(|a| self.rewrite_generic_arg(a)).collect();
                if let Some(first) = path.first() {
                    if self.is_resolved_module_alias(&first.name) {
                        if path.len() != 2 {
                            self.emit(
                                span,
                                format!("expected `<alias>.<GenericType><...>` after `{}`", first.name),
                            );
                            TypeKind::GenericPath {
                                path: path.clone(),
                                args: new_args,
                            }
                        } else if let Some(sym) = self.lookup_imported_member(
                            &first.name,
                            &path[1].name,
                            Some(ExportNs::Type),
                            span,
                        ) {
                            TypeKind::Generic {
                                base: Ident::new(sym.canonical_name, first.span.merge(path[1].span)),
                                args: new_args,
                            }
                        } else {
                            TypeKind::GenericPath {
                                path: path.clone(),
                                args: new_args,
                            }
                        }
                    } else {
                        TypeKind::GenericPath {
                            path: path.clone(),
                            args: new_args,
                        }
                    }
                } else {
                    TypeKind::GenericPath {
                        path: path.clone(),
                        args: new_args,
                    }
                }
            }
            TypeKind::Slice(inner) => TypeKind::Slice(Box::new(self.rewrite_type(inner))),
            TypeKind::Array { elem, len } => {
                let new_elem = Box::new(self.rewrite_type(elem));
                let new_len = match len {
                    ArrayLen::Fixed(n) => ArrayLen::Fixed(*n),
                    ArrayLen::Expr(e) => ArrayLen::Expr(Box::new(self.rewrite_expr(e))),
                };
                TypeKind::Array {
                    elem: new_elem,
                    len: new_len,
                }
            }
            TypeKind::Ptr(inner) => TypeKind::Ptr(Box::new(self.rewrite_type(inner))),
            TypeKind::OsPrimitive(op) => TypeKind::OsPrimitive(self.rewrite_os_primitive(op)),
            TypeKind::FnPtr { params, ret } => TypeKind::FnPtr {
                params: params.iter().map(|p| self.rewrite_type(p)).collect(),
                ret: Box::new(self.rewrite_type(ret)),
            },
        };
        TypeExpr::new(kind, span)
    }

    fn rewrite_os_primitive(&mut self, op: &OsPrimitiveType) -> OsPrimitiveType {
        match op {
            OsPrimitiveType::Signal => OsPrimitiveType::Signal,
            OsPrimitiveType::Watch { elem } => OsPrimitiveType::Watch {
                elem: Box::new(self.rewrite_type(elem)),
            },
            OsPrimitiveType::Inbox { elem, cap } => OsPrimitiveType::Inbox {
                elem: Box::new(self.rewrite_type(elem)),
                cap: *cap,
            },
            OsPrimitiveType::PingPong { elem, cap } => OsPrimitiveType::PingPong {
                elem: Box::new(self.rewrite_type(elem)),
                cap: *cap,
            },
            OsPrimitiveType::Lease { elem, cap } => OsPrimitiveType::Lease {
                elem: Box::new(self.rewrite_type(elem)),
                cap: *cap,
            },
        }
    }

    fn rewrite_generic_arg(&mut self, arg: &GenericArg) -> GenericArg {
        match arg {
            GenericArg::Type(t) => GenericArg::Type(self.rewrite_type(t)),
            GenericArg::Const(e) => GenericArg::Const(self.rewrite_expr(e)),
        }
    }

    fn rewrite_params(&mut self, params: &[Param]) -> Vec<Param> {
        params
            .iter()
            .map(|p| Param {
                ty: self.rewrite_type(&p.ty),
                name: p.name.clone(),
                span: p.span,
            })
            .collect()
    }

    fn rewrite_block(&mut self, block: &Block) -> Block {
        self.push_scope();
        let stmts = block.stmts.iter().map(|s| self.rewrite_stmt(s)).collect();
        self.pop_scope();
        Block::new(stmts, block.span)
    }

    fn rewrite_stmt(&mut self, stmt: &Stmt) -> Stmt {
        let span = stmt.span;
        let kind = match &stmt.kind {
            StmtKind::VarDecl { ty, name, init } => {
                let new_ty = self.rewrite_type(ty);
                let new_init = init.as_ref().map(|e| self.rewrite_expr(e));
                self.define_local(&name.name);
                StmtKind::VarDecl {
                    ty: new_ty,
                    name: name.clone(),
                    init: new_init,
                }
            }
            StmtKind::StateVar { ty, name, init } => {
                let new_ty = self.rewrite_type(ty);
                let new_init = init.as_ref().map(|e| self.rewrite_expr(e));
                self.define_local(&name.name);
                StmtKind::StateVar {
                    ty: new_ty,
                    name: name.clone(),
                    init: new_init,
                }
            }
            StmtKind::Assign { target, op, value } => StmtKind::Assign {
                target: self.rewrite_expr(target),
                op: *op,
                value: self.rewrite_expr(value),
            },
            StmtKind::Return(opt) => StmtKind::Return(opt.as_ref().map(|e| self.rewrite_expr(e))),
            StmtKind::Error { reason, partial } => StmtKind::Error {
                reason: reason.as_ref().map(|e| self.rewrite_expr(e)),
                partial: partial.as_ref().map(|e| self.rewrite_expr(e)),
            },
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => StmtKind::If {
                cond: self.rewrite_expr(cond),
                then_block: self.rewrite_block(then_block),
                else_block: else_block.as_ref().map(|b| self.rewrite_block(b)),
            },
            StmtKind::While { cond, body } => StmtKind::While {
                cond: self.rewrite_expr(cond),
                body: self.rewrite_block(body),
            },
            StmtKind::Loop { body } => StmtKind::Loop {
                body: self.rewrite_block(body),
            },
            StmtKind::LoopEvery { period, body } => StmtKind::LoopEvery {
                period: Box::new(self.rewrite_expr(period)),
                body: self.rewrite_block(body),
            },
            StmtKind::ForRange {
                var_ty,
                var_name,
                start,
                end,
                inclusive,
                body,
            } => {
                let new_ty = self.rewrite_type(var_ty);
                let new_start = Box::new(self.rewrite_expr(start));
                let new_end = Box::new(self.rewrite_expr(end));
                self.push_scope();
                self.define_local(&var_name.name);
                let new_body = self.rewrite_block(body);
                self.pop_scope();
                StmtKind::ForRange {
                    var_ty: new_ty,
                    var_name: var_name.clone(),
                    start: new_start,
                    end: new_end,
                    inclusive: *inclusive,
                    body: new_body,
                }
            }
            StmtKind::ForEach {
                elem_ty,
                elem_name,
                iter,
                body,
            } => {
                let new_ty = self.rewrite_type(elem_ty);
                let new_iter = Box::new(self.rewrite_expr(iter));
                self.push_scope();
                self.define_local(&elem_name.name);
                let new_body = self.rewrite_block(body);
                self.pop_scope();
                StmtKind::ForEach {
                    elem_ty: new_ty,
                    elem_name: elem_name.clone(),
                    iter: new_iter,
                    body: new_body,
                }
            }
            StmtKind::Defer(inner) => StmtKind::Defer(Box::new(self.rewrite_stmt(inner))),
            StmtKind::NoYield { body } => StmtKind::NoYield {
                body: self.rewrite_block(body),
            },
            StmtKind::Critical { body } => StmtKind::Critical {
                body: self.rewrite_block(body),
            },
            StmtKind::Switch { scrutinee, arms } => {
                let new_scrut = self.rewrite_expr(scrutinee);
                let new_arms = arms
                    .iter()
                    .map(|arm| SwitchArm {
                        pat: self.rewrite_switch_pat(&arm.pat),
                        body: self.rewrite_block(&arm.body),
                        span: arm.span,
                    })
                    .collect();
                StmtKind::Switch {
                    scrutinee: new_scrut,
                    arms: new_arms,
                }
            }
            StmtKind::Spawn(sp) => {
                let new_args = self.rewrite_call_args(&sp.args);
                let (new_module, new_task) = if let Some(mod_id) = &sp.module {
                    if self.is_resolved_module_alias(&mod_id.name) {
                        if let Some(sym) = self.lookup_imported_member(
                            &mod_id.name,
                            &sp.task.name,
                            Some(ExportNs::Value),
                            sp.span,
                        ) {
                            (None, Ident::new(sym.canonical_name, sp.task.span))
                        } else {
                            (Some(mod_id.clone()), sp.task.clone())
                        }
                    } else {
                        self.emit(
                            mod_id.span,
                            format!("unknown module alias `{}` in `spawn`", mod_id.name),
                        );
                        (Some(mod_id.clone()), sp.task.clone())
                    }
                } else if let Some(can) = self.self_value_canonical(&sp.task.name) {
                    (None, Ident::new(can.to_owned(), sp.task.span))
                } else {
                    (None, sp.task.clone())
                };
                StmtKind::Spawn(SpawnStmt {
                    module: new_module,
                    task: new_task,
                    args: new_args,
                    priority: sp.priority,
                    core: sp.core,
                    span: sp.span,
                })
            }
            StmtKind::Continue => StmtKind::Continue,
            StmtKind::Break => StmtKind::Break,
            StmtKind::Restart => StmtKind::Restart,
            StmtKind::Checkpoint => StmtKind::Checkpoint,
            StmtKind::Block(b) => StmtKind::Block(self.rewrite_block(b)),
            StmtKind::Expr(e) => StmtKind::Expr(self.rewrite_expr(e)),
        };
        Stmt::with_attrs(stmt.attrs.clone(), kind, span)
    }

    fn rewrite_switch_pat(&mut self, pat: &SwitchPat) -> SwitchPat {
        match pat {
            SwitchPat::Ident { prefix, name, span } => {
                if let Some(pref) = prefix {
                    if let Some(can) = self.self_type_canonical(&pref.name) {
                        SwitchPat::Ident {
                            prefix: Some(Ident::new(can.to_owned(), pref.span)),
                            name: name.clone(),
                            span: *span,
                        }
                    } else {
                        SwitchPat::Ident {
                            prefix: Some(pref.clone()),
                            name: name.clone(),
                            span: *span,
                        }
                    }
                } else if let Some(can) = self.self_value_canonical(&name.name) {
                    SwitchPat::Ident {
                        prefix: None,
                        name: Ident::new(can.to_owned(), name.span),
                        span: *span,
                    }
                } else {
                    pat.clone()
                }
            }
            other => other.clone(),
        }
    }

    fn rewrite_call_args(&mut self, args: &[CallArg]) -> Vec<CallArg> {
        args.iter()
            .map(|a| CallArg {
                name: a.name.clone(),
                value: self.rewrite_expr(&a.value),
                span: a.span,
            })
            .collect()
    }

    fn rewrite_expr(&mut self, expr: &Expr) -> Expr {
        let span = expr.span;
        let kind = match &expr.kind {
            ExprKind::Literal(l) => ExprKind::Literal(l.clone()),
            ExprKind::Priority(p) => ExprKind::Priority(*p),
            ExprKind::Ident(id) => {
                if self.in_local_scope(&id.name) {
                    ExprKind::Ident(id.clone())
                } else if self.is_resolved_module_alias(&id.name) {
                    self.emit(
                        id.span,
                        format!(
                            "module alias `{}` cannot be used as a standalone value; access a `pub` member with `{}.<name>`",
                            id.name, id.name
                        ),
                    );
                    ExprKind::Ident(id.clone())
                } else if let Some(can) = self.self_value_canonical(&id.name) {
                    ExprKind::Ident(Ident::new(can.to_owned(), id.span))
                } else if let Some(can) = self.self_type_canonical(&id.name) {
                    ExprKind::Ident(Ident::new(can.to_owned(), id.span))
                } else {
                    ExprKind::Ident(id.clone())
                }
            }
            ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
                op: *op,
                expr: Box::new(self.rewrite_expr(inner)),
            },
            ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
                op: *op,
                lhs: Box::new(self.rewrite_expr(lhs)),
                rhs: Box::new(self.rewrite_expr(rhs)),
            },
            ExprKind::Cast { expr: inner, ty } => ExprKind::Cast {
                expr: Box::new(self.rewrite_expr(inner)),
                ty: self.rewrite_type(ty),
            },
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(alias) = &target.kind {
                    if !self.in_local_scope(&alias.name) && self.is_resolved_module_alias(&alias.name)
                    {
                        if let Some(sym) =
                            self.lookup_imported_member(&alias.name, &field.name, None, span)
                        {
                            if sym.is_extern_c {
                                self.rewritten_pub_extern_sites.insert((
                                    self.prog_idx,
                                    span.start,
                                    span.end,
                                ));
                            }
                            return Expr::new(
                                ExprKind::Ident(Ident::new(sym.canonical_name, span)),
                                span,
                            );
                        }
                        return Expr::new(
                            ExprKind::Field {
                                target: target.clone(),
                                field: field.clone(),
                            },
                            span,
                        );
                    }
                }
                ExprKind::Field {
                    target: Box::new(self.rewrite_expr(target)),
                    field: field.clone(),
                }
            }
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                let new_generic_args = generic_args
                    .iter()
                    .map(|g| self.rewrite_generic_arg(g))
                    .collect();
                let new_args = self.rewrite_call_args(args);
                let new_callee = Box::new(self.rewrite_expr(callee));
                ExprKind::Call {
                    callee: new_callee,
                    generic_args: new_generic_args,
                    args: new_args,
                }
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let new_args = self.rewrite_call_args(args);
                if let ExprKind::Ident(alias) = &receiver.kind {
                    if !self.in_local_scope(&alias.name) && self.is_resolved_module_alias(&alias.name)
                    {
                        let callee_span = receiver.span.merge(method.span);
                        if let Some(sym) = self.lookup_imported_member(
                            &alias.name,
                            &method.name,
                            Some(ExportNs::Value),
                            callee_span,
                        ) {
                            if sym.is_extern_c {
                                self.rewritten_pub_extern_sites.insert((
                                    self.prog_idx,
                                    callee_span.start,
                                    callee_span.end,
                                ));
                            }
                            return Expr::new(
                                ExprKind::Call {
                                    callee: Box::new(Expr::new(
                                        ExprKind::Ident(Ident::new(sym.canonical_name, callee_span)),
                                        callee_span,
                                    )),
                                    generic_args: Vec::new(),
                                    args: new_args,
                                },
                                span,
                            );
                        }
                        return Expr::new(
                            ExprKind::MethodCall {
                                receiver: receiver.clone(),
                                method: method.clone(),
                                args: new_args,
                            },
                            span,
                        );
                    }
                }
                ExprKind::MethodCall {
                    receiver: Box::new(self.rewrite_expr(receiver)),
                    method: method.clone(),
                    args: new_args,
                }
            }
            ExprKind::Index { target, index } => ExprKind::Index {
                target: Box::new(self.rewrite_expr(target)),
                index: Box::new(self.rewrite_expr(index)),
            },
            ExprKind::Slice { target, start, end } => ExprKind::Slice {
                target: Box::new(self.rewrite_expr(target)),
                start: start.as_ref().map(|e| Box::new(self.rewrite_expr(e))),
                end: end.as_ref().map(|e| Box::new(self.rewrite_expr(e))),
            },
            ExprKind::StructLit {
                module,
                name,
                generic_args,
                fields,
            } => {
                let new_generic_args = generic_args
                    .iter()
                    .map(|g| self.rewrite_generic_arg(g))
                    .collect();
                let new_fields = fields
                    .iter()
                    .map(|f| FieldInit {
                        name: f.name.clone(),
                        value: self.rewrite_expr(&f.value),
                        span: f.span,
                    })
                    .collect();
                let (new_module, new_name) = if let Some(mod_id) = module {
                    if self.is_resolved_module_alias(&mod_id.name) {
                        if let Some(sym) = self.lookup_imported_member(
                            &mod_id.name,
                            &name.name,
                            Some(ExportNs::Type),
                            mod_id.span.merge(name.span),
                        ) {
                            (
                                None,
                                Ident::new(sym.canonical_name, mod_id.span.merge(name.span)),
                            )
                        } else {
                            (Some(mod_id.clone()), name.clone())
                        }
                    } else if self.is_legacy_chip_alias(&mod_id.name) {
                        (Some(mod_id.clone()), name.clone())
                    } else {
                        self.emit(
                            mod_id.span,
                            format!("unknown module alias `{}` in struct literal", mod_id.name),
                        );
                        (Some(mod_id.clone()), name.clone())
                    }
                } else if let Some(can) = self.self_type_canonical(&name.name) {
                    (None, Ident::new(can.to_owned(), name.span))
                } else {
                    (None, name.clone())
                };
                ExprKind::StructLit {
                    module: new_module,
                    name: new_name,
                    generic_args: new_generic_args,
                    fields: new_fields,
                }
            }
            ExprKind::ArrayLit(elems) => {
                ExprKind::ArrayLit(elems.iter().map(|e| self.rewrite_expr(e)).collect())
            }
            ExprKind::FormatString(parts) => {
                let new_parts = parts
                    .iter()
                    .map(|p| match p {
                        FormatPart::Literal(s) => FormatPart::Literal(s.clone()),
                        FormatPart::Hole { expr, spec, span } => FormatPart::Hole {
                            expr: Box::new(self.rewrite_expr(expr)),
                            spec: *spec,
                            span: *span,
                        },
                    })
                    .collect();
                ExprKind::FormatString(new_parts)
            }
            ExprKind::Or { expr, fallback } => ExprKind::Or {
                expr: Box::new(self.rewrite_expr(expr)),
                fallback: Box::new(self.rewrite_expr(fallback)),
            },
            ExprKind::Catch { expr, clause } => {
                let new_expr = Box::new(self.rewrite_expr(expr));
                let new_err_param = clause.err_param.as_ref().map(|p| CatchParam {
                    ty: self.rewrite_type(&p.ty),
                    name: p.name.clone(),
                    span: p.span,
                });
                let new_partial_param = clause.partial_param.as_ref().map(|p| CatchParam {
                    ty: self.rewrite_type(&p.ty),
                    name: p.name.clone(),
                    span: p.span,
                });
                self.push_scope();
                if let Some(p) = &new_err_param {
                    self.define_local(&p.name.name);
                }
                if let Some(p) = &new_partial_param {
                    self.define_local(&p.name.name);
                }
                let new_body = self.rewrite_block(&clause.body);
                self.pop_scope();
                ExprKind::Catch {
                    expr: new_expr,
                    clause: CatchClause {
                        err_param: new_err_param,
                        partial_param: new_partial_param,
                        body: new_body,
                        span: clause.span,
                    },
                }
            }
        };
        Expr::new(kind, span)
    }
}

#[allow(clippy::too_many_arguments)]
fn rewrite_program(
    prog: &Program,
    prefix: Option<&str>,
    module_dot_path: Option<&str>,
    self_exports: &ModuleExports,
    imports: &BTreeMap<String, ResolvedImport>,
    legacy_aliases: &BTreeSet<String>,
    exports_by_key: &BTreeMap<ModuleKey, (String, ModuleExports)>,
    display_names: &mut BTreeMap<String, String>,
    prog_idx: usize,
    file_label: &str,
    is_non_root: bool,
    seen_extern_c: &mut BTreeMap<String, (FnDecl, Option<String>)>,
    rewritten_pub_extern_sites: &mut BTreeSet<(usize, u32, u32)>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Program {
    let mut rw = AstRewriter {
        prefix,
        module_dot_path,
        self_exports,
        imports,
        legacy_aliases,
        exports_by_key,
        display_names,
        prog_idx,
        file_label,
        rewritten_pub_extern_sites,
        diagnostics,
        local_scopes: Vec::new(),
        generic_type_params: BTreeSet::new(),
    };

    let mut out_items = Vec::with_capacity(prog.items.len());
    for item in &prog.items {
        let new_item = match &item.kind {
            ItemKind::Import(imp) => {
                // Keep only legacy unresolved `chip.*` imports so `zev-sema` fallback and `zev-mir`
                // `import_aliases` stay intact without polluting namespaces with resolved `.zev` modules.
                let alias = imp
                    .alias
                    .as_ref()
                    .map(|a| a.name.as_str())
                    .or_else(|| imp.path.last().map(|l| l.name.as_str()))
                    .unwrap_or("");
                if !rw.is_legacy_chip_alias(alias) {
                    continue;
                }
                item.clone()
            }
            ItemKind::Struct(st) => {
                rw.generic_type_params = st
                    .generic_params
                    .iter()
                    .filter(|g| g.kind == GenericParamKind::Type)
                    .map(|g| g.name.name.clone())
                    .collect();
                let mut fields = Vec::with_capacity(st.fields.len());
                for f in &st.fields {
                    fields.push(zev_ast::StructField {
                        ty: rw.rewrite_type(&f.ty),
                        name: f.name.clone(),
                        span: f.span,
                    });
                }
                rw.generic_type_params.clear();
                let new_name = rw.canonicalize_top_name(&st.name.name);
                Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::Struct(zev_ast::StructDecl {
                        is_packed: st.is_packed,
                        name: Ident::new(new_name, st.name.span),
                        generic_params: st.generic_params.clone(),
                        fields,
                        span: st.span,
                    }),
                    item.span,
                )
            }
            ItemKind::Enum(en) => {
                let new_name = rw.canonicalize_top_name(&en.name.name);
                Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::Enum(zev_ast::EnumDecl {
                        name: Ident::new(new_name, en.name.span),
                        variants: en.variants.clone(),
                        span: en.span,
                    }),
                    item.span,
                )
            }
            ItemKind::Mmio(mm) => {
                let new_kind = match &mm.kind {
                    MmioDeclKind::Layout { regs } => {
                        let new_regs = regs
                            .iter()
                            .map(|r| zev_ast::MmioRegDecl {
                                access: r.access,
                                ty: rw.rewrite_type(&r.ty),
                                name: r.name.clone(),
                                offset: rw.rewrite_expr(&r.offset),
                                span: r.span,
                            })
                            .collect();
                        MmioDeclKind::Layout { regs: new_regs }
                    }
                    MmioDeclKind::Instance { layout, base_addr } => MmioDeclKind::Instance {
                        layout: rw.rewrite_type(layout),
                        base_addr: rw.rewrite_expr(base_addr),
                    },
                    MmioDeclKind::Inline { base_addr, regs } => MmioDeclKind::Inline {
                        base_addr: rw.rewrite_expr(base_addr),
                        regs: regs
                            .iter()
                            .map(|r| zev_ast::MmioRegDecl {
                                access: r.access,
                                ty: rw.rewrite_type(&r.ty),
                                name: r.name.clone(),
                                offset: rw.rewrite_expr(&r.offset),
                                span: r.span,
                            })
                            .collect(),
                    },
                };
                let new_name = rw.canonicalize_top_name(&mm.name.name);
                Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::Mmio(MmioDecl {
                        name: Ident::new(new_name, mm.name.span),
                        kind: new_kind,
                        span: mm.span,
                    }),
                    item.span,
                )
            }
            ItemKind::Irq(iq) => {
                let new_name = rw.canonicalize_top_name(&iq.name.name);
                Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::Irq(zev_ast::IrqDecl {
                        name: Ident::new(new_name, iq.name.span),
                        number: rw.rewrite_expr(&iq.number),
                        span: iq.span,
                    }),
                    item.span,
                )
            }
            ItemKind::Const(c) => {
                let new_name = rw.canonicalize_top_name(&c.name.name);
                Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::Const(zev_ast::ConstDecl {
                        ty: rw.rewrite_type(&c.ty),
                        name: Ident::new(new_name, c.name.span),
                        value: rw.rewrite_expr(&c.value),
                        span: c.span,
                    }),
                    item.span,
                )
            }
            ItemKind::GlobalVar(g) => {
                let new_name = rw.canonicalize_top_name(&g.name.name);
                Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::GlobalVar(zev_ast::GlobalVarDecl {
                        ty: rw.rewrite_type(&g.ty),
                        name: Ident::new(new_name, g.name.span),
                        init: g.init.as_ref().map(|e| rw.rewrite_expr(e)),
                        span: g.span,
                    }),
                    item.span,
                )
            }
            ItemKind::OsPrimitive(p) => {
                let new_name = rw.canonicalize_top_name(&p.name.name);
                let new_ty = rw.rewrite_os_primitive(&p.ty);
                Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::OsPrimitive(zev_ast::OsPrimitiveDecl {
                        attrs: p.attrs.clone(),
                        ty: new_ty,
                        name: Ident::new(new_name, p.name.span),
                        init: p.init.as_ref().map(|e| rw.rewrite_expr(e)),
                        span: p.span,
                    }),
                    item.span,
                )
            }
            ItemKind::Fn(f) => {
                rw.generic_type_params = f
                    .generic_params
                    .iter()
                    .filter(|g| g.kind == GenericParamKind::Type)
                    .map(|g| g.name.name.clone())
                    .collect();
                let new_params = rw.rewrite_params(&f.params);
                let new_ret = rw.rewrite_type(&f.ret_type);
                let new_err = f.error_spec.as_ref().map(|es| match es {
                    zev_ast::ErrorSpec::Bare(sp) => zev_ast::ErrorSpec::Bare(*sp),
                    zev_ast::ErrorSpec::Typed(t) => zev_ast::ErrorSpec::Typed(rw.rewrite_type(t)),
                });
                rw.push_scope();
                for gp in &f.generic_params {
                    if gp.kind != GenericParamKind::Type {
                        rw.define_local(&gp.name.name);
                    }
                }
                for p in &new_params {
                    rw.define_local(&p.name.name);
                }
                let new_body = f.body.as_ref().map(|b| rw.rewrite_block(b));
                rw.pop_scope();
                rw.generic_type_params.clear();

                // Keep `extern "C"` C symbol names unmangled across all modules.
                let new_name = if f.is_extern_c {
                    f.name.name.clone()
                } else {
                    rw.canonicalize_top_name(&f.name.name)
                };

                let new_decl = FnDecl {
                    attrs: f.attrs.clone(),
                    is_extern_c: f.is_extern_c,
                    name: Ident::new(new_name.clone(), f.name.span),
                    generic_params: f.generic_params.clone(),
                    params: new_params,
                    is_variadic: f.is_variadic,
                    ret_type: new_ret,
                    error_spec: new_err,
                    body: new_body,
                    span: f.span,
                };

                if new_decl.is_extern_c && (is_non_root || !seen_extern_c.is_empty()) {
                    if let Some((prev_decl, _)) = seen_extern_c.get(&new_name) {
                        if !extern_signatures_match(prev_decl, &new_decl) {
                            rw.emit(
                                new_decl.name.span,
                                format!(
                                    "conflicting `extern \"C\"` signature for `{new_name}` across modules"
                                ),
                            );
                        } else if new_decl.body.is_none() && prev_decl.body.is_none() {
                            // Deduplicate identical `extern "C"` declarations across linked files.
                            continue;
                        }
                    } else if is_non_root {
                        seen_extern_c.insert(
                            new_name,
                            (new_decl.clone(), Some(rw.file_label.to_owned())),
                        );
                    }
                } else if new_decl.is_extern_c {
                    // Only register cross-module `extern "C"` tracking when non-root modules exist;
                    // root-only files rely on normal `Analyzer` duplicate checks if repeated in root.
                    let _ = is_non_root;
                }

                Item::with_pub(item.attrs.clone(), item.is_pub, ItemKind::Fn(new_decl), item.span)
            }
            ItemKind::Task(t) => {
                let new_params = rw.rewrite_params(&t.params);
                rw.push_scope();
                for p in &new_params {
                    rw.define_local(&p.name.name);
                }
                let new_body = rw.rewrite_block(&t.body);
                rw.pop_scope();
                let new_name = rw.canonicalize_top_name(&t.name.name);
                Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::Task(TaskDecl {
                        attrs: t.attrs.clone(),
                        name: Ident::new(new_name, t.name.span),
                        params: new_params,
                        body: new_body,
                        span: t.span,
                    }),
                    item.span,
                )
            }
            ItemKind::Isr(i) => {
                let new_params = rw.rewrite_params(&i.params);
                rw.push_scope();
                for p in &new_params {
                    rw.define_local(&p.name.name);
                }
                let new_body = rw.rewrite_block(&i.body);
                rw.pop_scope();
                let new_name = rw.canonicalize_top_name(&i.vector.name);
                Item::with_pub(
                    item.attrs.clone(),
                    item.is_pub,
                    ItemKind::Isr(IsrDecl {
                        attrs: i.attrs.clone(),
                        vector: Ident::new(new_name, i.vector.span),
                        params: new_params,
                        body: new_body,
                        span: i.span,
                    }),
                    item.span,
                )
            }
        };
        out_items.push(new_item);
    }

    Program::new(out_items, prog.span)
}

fn extern_signatures_match(a: &FnDecl, b: &FnDecl) -> bool {
    if a.is_variadic != b.is_variadic || a.params.len() != b.params.len() {
        return false;
    }
    if !types_structurally_equal(&a.ret_type, &b.ret_type) {
        return false;
    }
    a.params
        .iter()
        .zip(b.params.iter())
        .all(|(pa, pb)| types_structurally_equal(&pa.ty, &pb.ty))
}

fn types_structurally_equal(a: &TypeExpr, b: &TypeExpr) -> bool {
    match (&a.kind, &b.kind) {
        (TypeKind::Prim(x), TypeKind::Prim(y)) => x == y,
        (TypeKind::Named(x), TypeKind::Named(y)) => x.name == y.name,
        (TypeKind::Path(x), TypeKind::Path(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(l, r)| l.name == r.name)
        }
        (TypeKind::Slice(x), TypeKind::Slice(y)) | (TypeKind::Ptr(x), TypeKind::Ptr(y)) => {
            types_structurally_equal(x, y)
        }
        (
            TypeKind::Array { elem: ex, len: lx },
            TypeKind::Array { elem: ey, len: ly },
        ) => lx == ly && types_structurally_equal(ex, ey),
        (
            TypeKind::FnPtr { params: px, ret: rx },
            TypeKind::FnPtr { params: py, ret: ry },
        ) => {
            px.len() == py.len()
                && px
                    .iter()
                    .zip(py.iter())
                    .all(|(l, r)| types_structurally_equal(l, r))
                && types_structurally_equal(rx, ry)
        }
        _ => a.kind == b.kind,
    }
}
