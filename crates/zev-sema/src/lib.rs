//! `zev-sema` — Static Verifier & Semantic Analyzer for the Zev Language-OS.
//!
//! Validates one or more parsed [`zev_ast::Program`] ASTs against all v0 Language-OS
//! safety and zero-heap scheduling invariants and produces a rich [`SemanticModel`]:
//! 1. **Whole-program Call-Graph Yield Inference** (`can_yield`, `extern "C"` implicit `noyield`)
//! 2. **`noyield`, `critical`, `isr`, and C-Callback Safety** (zero stackless yields across atomic/C frames; zero unbounded loops in `critical`)
//! 3. **Task & ISR Isolation** (rejects mutable raw globals; enforces communication via `signal`, `chan`, `watch`, `inbox`, `ping_pong`, `lease`)
//! 4. **Scope-Depth & DMA Lease Origin Tracking** (zero `'a` lifetimes; tracks slices, pointers, sub-slices, and wrapper structs; rejects outer-scope escapes, returning `&local`, and stale reads past subsequent `wait_chunk()`)
//! 5. **No Recursion Across Yielding Paths** (rejects cycles in the call graph containing any yielding function)
//! 6. **Time Resolution & Dual-Exit `return`/`error`/`catch`/`or` Verification**

pub mod call_graph;
pub mod checker;
pub mod diag;
pub mod model;
pub mod provenance;
pub mod suggest;
pub mod types;

pub use call_graph::{CallContext, CallEdge, CallGraph, CallableKind, ExternCallbackSite};
pub use checker::{AnalysisOutput, AnalyzeOptions};
pub use diag::{Diagnostic, DiagnosticKind};
pub use model::{
    ConstInfo, DEFAULT_TICK_PERIOD_US, FunctionInfo, IsrInfo, ParamInfo, PrimitiveInfo,
    PriorityBoostSite, SemanticModel, SpawnedTaskInstance, TaskInfo,
};
pub use provenance::{LeaseEnv, LeaseToken, Provenance};
pub use types::{
    EnumInfo, FieldInfo, NamespaceKind, PrimitiveKind, ResolvedErrorSpec, ResolvedOsPrimitive,
    StructInfo, Ty,
};

use zev_ast::Program;

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

/// Analyze named `(Program, filename)` pairs so diagnostics carry source file paths.
pub fn analyze_named_programs<'a>(
    programs: &[(&'a Program, Option<&'a str>)],
    opts: &AnalyzeOptions,
) -> AnalysisOutput {
    let mut out = checker::Analyzer::new(opts).run(programs);
    // Attach filename if a single file was analyzed or leave item-level spans intact.
    if programs.len() == 1 {
        if let Some(fname) = programs[0].1 {
            for d in &mut out.diagnostics {
                if d.file.is_none() {
                    d.file = Some(fname.to_string());
                }
            }
        }
    }
    out
}

/// Analyze one or more parsed [`Program`]s with custom [`AnalyzeOptions`], returning both
/// the [`SemanticModel`] and any [`Diagnostic`]s.
pub fn analyze_with_options(programs: &[Program], opts: &AnalyzeOptions) -> AnalysisOutput {
    let pairs: Vec<(&Program, Option<&str>)> = programs.iter().map(|p| (p, None)).collect();
    checker::Analyzer::new(opts).run(&pairs)
}
