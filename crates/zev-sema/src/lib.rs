//! `zev-sema` — Static Verifier, Monomorphizer & Semantic Analyzer for Zev (`zevc`).
//!
//! Monomorphizes `<T>` and `<T, usize N>` generics and validates one or more parsed [`zev_ast::Program`]
//! ASTs against Zev's safety and zero-heap scheduling invariants, producing concrete
//! [`Program`]s and a [`SemanticModel`]:
//! 1. **Compile-Time `<T>` & `<T, usize N>` Monomorphization** (`<orig>__<arg1>__<arg2>`, `@monomorphized`, `[N]T` bounds)
//! 2. **Whole-Program 3-Tier Call-Graph Yield Inference** (`NonYielding`, `SoftYielding` dual-spec, `HardBlocking`, `extern "C"`)
//! 3. **`noyield`, `critical`, `isr`, `defer`, and C-Callback Safety**
//! 4. **Task & ISR Isolation** (rejects mutable raw globals; enforces communication via `signal`, `inbox`, `watch`, `byte_ring`, `wait_queue`, `ping_pong`, `pool`/`Lease`/`dma_slot`)
//! 5. **Scope-Depth, Per-Arm `wait` Slice Lease & Linear `Lease<T>` Ownership Tracking (`E0008`–`E0011`, `E0034`)**
//! 6. **Zero-Heap `f"..."` Format Verification & `fmt_buf` Worst-Case Buffer Fit Proof (`E0028`)**
//! 7. **Boot-Tree `spawn` Unrolling, Immortal Tasks & Dual-Exit `return`/`error`/`catch`/`or` Verification**

pub mod call_graph;
pub mod checker;
pub mod diag;
pub mod model;
pub mod modules;
pub mod monomorphize;
pub mod provenance;
pub mod suggest;
pub mod types;

pub use call_graph::{CallContext, CallEdge, CallGraph, CallableKind, ExternCallbackSite};
pub use checker::{AnalysisOutput, AnalyzeOptions};
pub use diag::{Diagnostic, DiagnosticKind};
pub use model::{
    ConstInfo, FnEffect, FunctionInfo, IrqBindingArg, IrqBindingInfo, IrqInfo, IsrInfo,
    IsrStateInfo, MmioInstanceInfo, MmioLayoutInfo, MmioRegInfo, MonomorphizedInstance,
    MonomorphizedKind, ParamInfo, PrimitiveInfo, PriorityBoostSite, SemanticModel,
    SpawnedTaskInstance, SubsystemInfo, TaskInfo, DEFAULT_TICK_PERIOD_US,
};
pub use modules::{
    load_and_link_modules, load_and_link_modules_with_hal, load_and_link_modules_with_targets,
    LinkedModulesOutput,
};
pub use monomorphize::{
    mangle_type_expr, monomorphize_named_programs, monomorphize_programs, MonomorphizeOutput,
    MAX_GENERIC_DEPTH, MAX_MONOMORPHIZED_INSTANCES,
};
pub use provenance::{LeaseEnv, LeaseToken, Provenance};
pub use types::{
    EnumInfo, FieldInfo, FnPtrKind, NamespaceKind, PrimitiveKind, ResolvedErrorSpec,
    ResolvedOsPrimitive, StructInfo, Ty,
};

use zev_ast::Program;

/// Combined output of monomorphizing and semantically verifying one or more [`Program`]s.
#[derive(Clone, Debug)]
pub struct MonomorphizedAnalysisOutput {
    /// Concrete monomorphized [`Program`] ASTs ready for `zev-mir` lowering.
    pub programs: Vec<Program>,
    /// Validated whole-program [`SemanticModel`] (including `monomorphized_instances`).
    pub model: SemanticModel,
    /// Any monomorphization (`E0029`) or semantic (`E0001`..`E0028`) diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

impl MonomorphizedAnalysisOutput {
    pub fn is_ok(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn into_result(self) -> Result<(Vec<Program>, SemanticModel), Vec<Diagnostic>> {
        if self.diagnostics.is_empty() {
            Ok((self.programs, self.model))
        } else {
            Err(self.diagnostics)
        }
    }
}

/// Analyze a single parsed [`Program`] using default options (`1ms` = `1000us` tick period).
pub fn analyze_program(program: &Program) -> Result<SemanticModel, Vec<Diagnostic>> {
    analyze_programs(std::slice::from_ref(program))
}

/// Analyze one or more parsed [`Program`] compilation units together as a whole program.
pub fn analyze_programs(programs: &[Program]) -> Result<SemanticModel, Vec<Diagnostic>> {
    analyze_with_options(programs, &AnalyzeOptions::default()).into_result()
}

/// Alias for [`analyze_programs`] for concise multi-file invocation.
pub fn analyze(programs: &[Program]) -> Result<SemanticModel, Vec<Diagnostic>> {
    analyze_programs(programs)
}

/// Monomorphize and analyze named `(Program, filename)` pairs, returning concrete [`Program`]s,
/// the [`SemanticModel`], and all [`Diagnostic`]s with attached file paths.
pub fn analyze_and_monomorphize_named_programs<'a>(
    programs: &[(&'a Program, Option<&'a str>)],
    opts: &AnalyzeOptions,
) -> MonomorphizedAnalysisOutput {
    let linked = modules::load_and_link_modules_with_targets(
        programs,
        &opts.include_dirs,
        opts.hal_dir.as_deref(),
        opts.board_module.as_deref(),
    );
    if !linked.diagnostics.is_empty() {
        let mut diagnostics = linked.diagnostics;
        if programs.len() == 1 {
            if let Some(fname) = programs[0].1 {
                for d in &mut diagnostics {
                    if d.file.is_none() {
                        d.file = Some(fname.to_string());
                    }
                }
            }
        }
        let mut empty_model = checker::Analyzer::new(opts).run(&[]).model;
        empty_model.display_names = linked.display_names;
        return MonomorphizedAnalysisOutput {
            programs: linked.programs.into_iter().map(|(p, _)| p).collect(),
            model: empty_model,
            diagnostics,
        };
    }

    let linked_pairs: Vec<(&Program, Option<&str>)> = linked
        .programs
        .iter()
        .map(|(p, f)| (p, f.as_deref()))
        .collect();

    let mono_out = monomorphize::monomorphize_named_programs_with_target(
        &linked_pairs,
        linked.non_root_program_count,
        opts.target_ptr_bits,
        &opts.target_name,
    );
    if !mono_out.diagnostics.is_empty() {
        let mut diagnostics = mono_out.diagnostics;
        if programs.len() == 1 {
            if let Some(fname) = programs[0].1 {
                for d in &mut diagnostics {
                    if d.file.is_none() {
                        d.file = Some(fname.to_string());
                    }
                }
            }
        }
        let mut empty_model = checker::Analyzer::new(opts).run(&[]).model;
        empty_model.monomorphized_instances = mono_out.instances;
        empty_model.display_names = linked.display_names;
        return MonomorphizedAnalysisOutput {
            programs: mono_out.programs,
            model: empty_model,
            diagnostics,
        };
    }

    let mono_pairs: Vec<(&Program, Option<&str>)> = mono_out
        .programs
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let fname = linked.programs.get(idx).and_then(|(_, f)| f.as_deref());
            (p, fname)
        })
        .collect();

    let mut sema_out = checker::Analyzer::new(opts)
        .with_linked_context(&linked)
        .with_defaulted_generic_calls(mono_out.defaulted_calls)
        .run(&mono_pairs);

    // Ensure `model.monomorphized_instances` includes all deduplicated & sorted instances.
    if !mono_out.instances.is_empty() {
        let mut merged = mono_out.instances;
        merged.extend(sema_out.model.monomorphized_instances);
        merged.sort();
        merged.dedup();
        sema_out.model.monomorphized_instances = merged;
    }
    sema_out.model.display_names = linked.display_names;

    let mut diagnostics = mono_out.diagnostics;
    diagnostics.extend(sema_out.diagnostics);

    if programs.len() == 1 {
        if let Some(fname) = programs[0].1 {
            for d in &mut diagnostics {
                if d.file.is_none() {
                    d.file = Some(fname.to_string());
                }
            }
        }
    }

    MonomorphizedAnalysisOutput {
        programs: mono_out.programs,
        model: sema_out.model,
        diagnostics,
    }
}

/// Monomorphize and analyze one or more unnamed [`Program`]s with custom [`AnalyzeOptions`].
pub fn analyze_and_monomorphize_programs(
    programs: &[Program],
    opts: &AnalyzeOptions,
) -> MonomorphizedAnalysisOutput {
    let pairs: Vec<(&Program, Option<&str>)> = programs.iter().map(|p| (p, None)).collect();
    analyze_and_monomorphize_named_programs(&pairs, opts)
}

/// Analyze named `(Program, filename)` pairs (including automatic monomorphization) so diagnostics carry source file paths.
pub fn analyze_named_programs<'a>(
    programs: &[(&'a Program, Option<&'a str>)],
    opts: &AnalyzeOptions,
) -> AnalysisOutput {
    let res = analyze_and_monomorphize_named_programs(programs, opts);
    AnalysisOutput {
        model: res.model,
        diagnostics: res.diagnostics,
    }
}

/// Analyze one or more parsed [`Program`]s with custom [`AnalyzeOptions`], returning both
/// the [`SemanticModel`] and any [`Diagnostic`]s.
pub fn analyze_with_options(programs: &[Program], opts: &AnalyzeOptions) -> AnalysisOutput {
    let pairs: Vec<(&Program, Option<&str>)> = programs.iter().map(|p| (p, None)).collect();
    analyze_named_programs(&pairs, opts)
}
