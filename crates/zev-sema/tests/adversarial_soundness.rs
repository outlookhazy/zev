use std::collections::BTreeMap;
use zev_sema::{
    AnalyzeOptions, Diagnostic, DiagnosticKind, SemanticModel, analyze_and_monomorphize_programs,
    analyze_program,
};

fn check_src(src: &str) -> Result<SemanticModel, Vec<Diagnostic>> {
    let prog = zev_parser::parse_str(src)
        .unwrap_or_else(|e| panic!("test source failed to parse:\n{:?}\n---\n{}", e, src));
    analyze_program(&prog)
}

fn check_mono_src(src: &str) -> Result<SemanticModel, Vec<Diagnostic>> {
    let generics = zev_parser::collect_declared_generic_names(src);
    let prog = zev_parser::parse_str_named_with_generics(src, "<adversarial>", Some(&generics))
        .unwrap_or_else(|e| panic!("test source failed to parse:\n{:?}\n---\n{}", e, src));
    analyze_and_monomorphize_programs(&[prog], &AnalyzeOptions::default())
        .into_result()
        .map(|(_, model)| model)
}

fn expect_ok(src: &str) {
    if let Err(diags) = check_src(src) {
        let msgs: Vec<String> = diags.iter().map(ToString::to_string).collect();
        panic!("expected clean check, got:\n{}\n---\n{}", msgs.join("\n"), src);
    }
}

fn expect_mono_ok(src: &str) {
    if let Err(diags) = check_mono_src(src) {
        let msgs: Vec<String> = diags.iter().map(ToString::to_string).collect();
        panic!("expected clean mono check, got:\n{}\n---\n{}", msgs.join("\n"), src);
    }
}

fn expect_err(src: &str, expected: DiagnosticKind) {
    match check_src(src) {
        Ok(_) => panic!(
            "SOUNDNESS HOLE: expected {:?} ({}), but program passed:\n{}",
            expected,
            expected.code(),
            src
        ),
        Err(diags) => {
            assert!(
                diags.iter().any(|d| d.kind == expected),
                "expected {:?} ({}), got:\n{}\n---\n{}",
                expected,
                expected.code(),
                diags
                    .iter()
                    .map(|d| format!("[{}] {:?}: {}", d.kind.code(), d.kind, d))
                    .collect::<Vec<_>>()
                    .join("\n"),
                src
            );
        }
    }
}

fn expect_mono_err(src: &str, expected: DiagnosticKind) {
    match check_mono_src(src) {
        Ok(_) => panic!(
            "SOUNDNESS HOLE: expected {:?} ({}), but monomorphized program passed:\n{}",
            expected,
            expected.code(),
            src
        ),
        Err(diags) => {
            assert!(
                diags.iter().any(|d| d.kind == expected),
                "expected {:?} ({}), got:\n{}\n---\n{}",
                expected,
                expected.code(),
                diags
                    .iter()
                    .map(|d| format!("[{}] {:?}: {}", d.kind.code(), d.kind, d))
                    .collect::<Vec<_>>()
                    .join("\n"),
                src
            );
        }
    }
}

/// Expands every adversarial negative case into its own standalone `#[test]` function (so a
/// failure in one case never masks subsequent cases) while recording `COVERED_NEGATIVE_KINDS`
/// so a meta-test can verify complete `E0001`..`E0034` coverage.
macro_rules! adversarial_negative_suite {
    (
        $(
            $test_name:ident ( $runner:ident, $kind:expr ) => $src:expr ;
        )+
    ) => {
        $(
            #[test]
            fn $test_name() {
                $runner($src, $kind);
            }
        )+

        const COVERED_NEGATIVE_KINDS: &[DiagnosticKind] = &[
            $( $kind ),+
        ];
    };
}

adversarial_negative_suite! {
    // ── E0001: IllegalMutableGlobal ──────────────────────────────────────────
    e0001_mutable_scalar_global_rejected(expect_err, DiagnosticKind::IllegalMutableGlobal) => r#"
u32 bad_counter = 0u32;
fn main() { bad_counter += 1u32; }
"#;

    e0001_mutable_array_global_rejected(expect_err, DiagnosticKind::IllegalMutableGlobal) => r#"
[4]u8 bad_buf = [0u8, 0u8, 0u8, 0u8];
fn main() { bad_buf[0] = 1u8; }
"#;

    // ── E0002: GlobalIsolationViolation ──────────────────────────────────────
    e0002_mutable_struct_global_mutated_in_helper(expect_err, DiagnosticKind::GlobalIsolationViolation) => r#"
struct Pair { u32 a; u32 b; }
Pair bad_pair = Pair { a: 1u32, b: 2u32 };
fn bump() { bad_pair.a += 1u32; }
fn main() { bump(); }
"#;

    e0002_mutable_global_accessed_in_isr_via_ptr(expect_err, DiagnosticKind::GlobalIsolationViolation) => r#"
irq IRQ_0 = 0;
u32 shared_val = 0u32;
isr my_isr() {
    *u32 p = &shared_val;
    *p = 7u32;
}
fn main() { IRQ_0.attach(my_isr()); }
"#;

    e0002_isr_referencing_unpassed_global_subsystem(expect_err, DiagnosticKind::GlobalIsolationViolation) => r#"
irq IRQ_0 = 0;
struct Port { byte_ring<64> rx; byte_ring<64> tx; }
Port p0;
isr bad_isr() {
    bool _ok = p0.rx.push_byte(1u8);
}
fn main() { IRQ_0.attach(bad_isr()); }
"#;

    e0002_byte_ring_multi_producer_isrs_rejected(expect_err, DiagnosticKind::GlobalIsolationViolation) => r#"
irq IRQ_0 = 0;
irq IRQ_1 = 1;
byte_ring<64> ring;
isr p1(*byte_ring tx) { bool _ok = tx.push_byte(1u8); }
isr p2(*byte_ring tx) { bool _ok = tx.push_byte(2u8); }
fn main() {
    IRQ_0.attach(p1(&ring));
    IRQ_1.attach(p2(&ring));
}
"#;

    e0002_byte_ring_multi_consumer_tasks_rejected(expect_err, DiagnosticKind::GlobalIsolationViolation) => r#"
byte_ring<64> ring;
task c1(*byte_ring rx) {
    loop { []u8 _a = rx.wait_read(); }
}
task c2(*byte_ring rx) {
    loop { []u8 _b = rx.wait_read(); }
}
fn main() {
    spawn c1(&ring) @normal;
    spawn c2(&ring) @normal;
}
"#;

    // ── E0003: YieldInNoYield ────────────────────────────────────────────────
    e0003_three_hop_transitive_hard_yield_in_noyield(expect_err, DiagnosticKind::YieldInNoYield) => r#"
fn leaf() { delay(5ms); }
fn mid() { leaf(); }
fn top() { mid(); }
task t() {
    loop {
        noyield { top(); }
        delay(10ms);
    }
}
fn main() { spawn t() @normal; }
"#;

    e0003_soft_checkpoint_inside_defer_rejected(expect_err, DiagnosticKind::YieldInNoYield) => r#"
fn soft() { checkpoint(); }
task t() {
    loop {
        {
            defer { soft(); }
        }
        delay(1ms);
    }
}
fn main() { spawn t() @normal; }
"#;

    // ── E0004: YieldInCritical ───────────────────────────────────────────────
    e0004_soft_checkpoint_in_critical_rejected(expect_err, DiagnosticKind::YieldInCritical) => r#"
fn soft() { checkpoint(); }
task t() {
    loop {
        critical { soft(); }
        delay(1ms);
    }
}
fn main() { spawn t() @normal; }
"#;

    e0004_hard_delay_in_critical_rejected(expect_err, DiagnosticKind::YieldInCritical) => r#"
task t() {
    loop {
        critical { delay(1ms); }
    }
}
fn main() { spawn t() @normal; }
"#;

    // ── E0005: YieldInIsr ────────────────────────────────────────────────────
    e0005_soft_checkpoint_in_isr_rejected(expect_err, DiagnosticKind::YieldInIsr) => r#"
irq IRQ_0 = 0;
fn soft() { checkpoint(); }
isr bad_isr() { soft(); }
fn main() { IRQ_0.attach(bad_isr()); }
"#;

    e0005_hard_delay_in_isr_rejected(expect_err, DiagnosticKind::YieldInIsr) => r#"
irq IRQ_0 = 0;
isr bad_isr() { delay(1ms); }
fn main() { IRQ_0.attach(bad_isr()); }
"#;

    // ── E0006: YieldingCallbackToExternC ─────────────────────────────────────
    e0006_yielding_callback_passed_to_extern_c(expect_err, DiagnosticKind::YieldingCallbackToExternC) => r#"
extern "C" fn register_cb(fn(u8) cb);
fn my_cb(u8 b) { checkpoint(); }
fn main() { register_cb(my_cb); }
"#;

    e0006_yielding_callback_passed_to_sys_set_putc(expect_err, DiagnosticKind::YieldingCallbackToExternC) => r#"
fn slow_putc(u8 b) { delay(1ms); }
fn main() { sys.set_putc(slow_putc); }
"#;

    // ── E0007: UnboundedLoopInCritical ───────────────────────────────────────
    e0007_unbounded_loop_in_critical_rejected(expect_err, DiagnosticKind::UnboundedLoopInCritical) => r#"
fn main() {
    critical {
        loop { sys.nop(); }
    }
}
"#;

    e0007_nested_branch_unbounded_loop_in_critical_rejected(expect_err, DiagnosticKind::UnboundedLoopInCritical) => r#"
fn spin_critical(bool c) {
    critical {
        if (c) {
            loop { sys.nop(); }
        }
    }
}
fn main() { spin_critical(true); }
"#;

    // ── E0008: ScopeDepthEscape ──────────────────────────────────────────────
    e0008_dma_slice_escaping_to_outer_scope(expect_err, DiagnosticKind::ScopeDepthEscape) => r#"
ping_pong<u8, 8> rx;
task t() {
    []u8 outer = [];
    {
        []u8 cur = rx.wait_chunk();
        outer = cur;
    }
    loop {
        u8 _x = outer[0];
        delay(5ms);
    }
}
fn main() { spawn t() @normal; }
"#;

    e0008_inner_array_coerced_to_outer_slice(expect_err, DiagnosticKind::ScopeDepthEscape) => r#"
fn bad_scope_assign() {
    []u8 outer = [];
    {
        [4]u8 inner_arr = [1u8, 2u8, 3u8, 4u8];
        outer = inner_arr;
    }
    []u8 _x = outer;
}
fn main() { bad_scope_assign(); }
"#;

    // ── E0009: ReturnLocalReference ──────────────────────────────────────────
    e0009_return_local_subslice(expect_err, DiagnosticKind::ReturnLocalReference) => r#"
[]u8 bad_ret() {
    [4]u8 local_arr = [1u8, 2u8, 3u8, 4u8];
    return local_arr[0..2];
}
fn main() { []u8 _s = bad_ret(); }
"#;

    e0009_return_struct_wrapping_local_array_slice(expect_err, DiagnosticKind::ReturnLocalReference) => r#"
struct SliceWrap { []u8 s; }
SliceWrap bad_ret_struct() {
    [4]u8 arr = [1u8, 2u8, 3u8, 4u8];
    return SliceWrap { s: arr };
}
fn main() { SliceWrap _w = bad_ret_struct(); }
"#;

    e0009_return_local_array_directly_as_slice(expect_err, DiagnosticKind::ReturnLocalReference) => r#"
[]u8 bad_direct() {
    [4]u8 arr = [1u8, 2u8, 3u8, 4u8];
    return arr;
}
fn main() { []u8 _s = bad_direct(); }
"#;

    e0009_return_local_slice_bound_via_array_coercion(expect_err, DiagnosticKind::ReturnLocalReference) => r#"
[]u8 bad_via_local() {
    [4]u8 arr = [1u8, 2u8, 3u8, 4u8];
    []u8 s = arr;
    return s;
}
fn main() { []u8 _s = bad_via_local(); }
"#;

    e0009_return_local_array_through_identity_helper(expect_err, DiagnosticKind::ReturnLocalReference) => r#"
[]u8 id([]u8 x) { return x; }
[]u8 bad_via_call() {
    [4]u8 local_arr = [1u8, 2u8, 3u8, 4u8];
    return id(local_arr);
}
fn main() { []u8 _s = bad_via_call(); }
"#;

    // ── E0010: SpawnNonStaticReference ───────────────────────────────────────
    e0010_spawn_with_local_array_coerced_to_slice(expect_err, DiagnosticKind::SpawnNonStaticReference) => r#"
task worker([]u8 msg) {
    loop {
        u8 _b = msg[0];
        delay(10ms);
    }
}
fn main() {
    [4]u8 local_arr = [1u8, 2u8, 3u8, 4u8];
    spawn worker(local_arr) @normal;
}
"#;

    e0010_spawn_with_pointer_to_main_local(expect_err, DiagnosticKind::SpawnNonStaticReference) => r#"
task worker(*u32 p) {
    loop {
        u32 _v = *p;
        delay(10ms);
    }
}
fn main() {
    u32 local_val = 42u32;
    spawn worker(&local_val) @normal;
}
"#;

    // ── E0011: StaleDmaLease ─────────────────────────────────────────────────
    e0011_stale_ping_pong_slice_after_second_wait_chunk(expect_err, DiagnosticKind::StaleDmaLease) => r#"
ping_pong<u8, 8> rx;
task t() {
    loop {
        []u8 a = rx.wait_chunk();
        []u8 b = rx.wait_chunk();
        u8 _x = a[0] + b[0];
        delay(1ms);
    }
}
fn main() { spawn t() @normal; }
"#;

    e0011_stale_byte_ring_slice_after_advance(expect_err, DiagnosticKind::StaleDmaLease) => r#"
byte_ring<16> rx;
task t() {
    loop {
        []u8 chunk = rx.wait_read();
        rx.consume(chunk.len);
        u8 _bad = chunk[0];
        delay(1ms);
    }
}
fn main() { spawn t() @normal; }
"#;

    // ── E0012: YieldingRecursion ─────────────────────────────────────────────
    e0012_mutual_yielding_recursion_rejected(expect_err, DiagnosticKind::YieldingRecursion) => r#"
fn a(u32 n) {
    if (n > 0u32) { b(n - 1u32); }
}
fn b(u32 n) {
    checkpoint();
    if (n > 0u32) { a(n - 1u32); }
}
task w() {
    a(3u32);
    loop { delay(10ms); }
}
fn main() { spawn w() @normal; }
"#;

    e0012_direct_self_yielding_recursion_rejected(expect_err, DiagnosticKind::YieldingRecursion) => r#"
fn self_yield(u32 n) {
    delay(1ms);
    if (n > 0u32) { self_yield(n - 1u32); }
}
task w() {
    self_yield(2u32);
    loop { delay(10ms); }
}
fn main() { spawn w() @normal; }
"#;

    // ── E0013: InvalidDurationResolution ─────────────────────────────────────
    e0013_subtick_duration_truncating_to_zero_rejected(expect_err, DiagnosticKind::InvalidDurationResolution) => r#"
task w() {
    loop { delay(500us); }
}
fn main() { spawn w() @normal; }
"#;

    e0013_non_integer_tick_multiple_duration_rejected(expect_err, DiagnosticKind::InvalidDurationResolution) => r#"
task w() {
    loop { delay(1500us); }
}
fn main() { spawn w() @normal; }
"#;

    e0013_delay_past_i32_tick_deadline_horizon_rejected(expect_err, DiagnosticKind::InvalidDurationResolution) => r#"
task w() {
    loop { delay(2147484s); }
}
fn main() { spawn w() @normal; }
"#;

    e0013_wait_timeout_past_u32_tick_range_rejected(expect_err, DiagnosticKind::InvalidDurationResolution) => r#"
signal s;
task w() {
    loop {
        s.wait(timeout: 5000000s) catch { continue; };
    }
}
fn main() { spawn w() @normal; }
"#;

    e0013_const_loop_every_period_past_deadline_horizon_rejected(expect_err, DiagnosticKind::InvalidDurationResolution) => r#"
const u32 PERIOD = 3000000000;
task w() {
    loop every PERIOD { checkpoint(); }
}
fn main() { spawn w() @normal; }
"#;

    // ── E0014: ReturnMismatch ────────────────────────────────────────────────
    e0014_missing_return_on_else_branch_rejected(expect_err, DiagnosticKind::ReturnMismatch) => r#"
u32 missing_else(bool c) {
    if (c) {
        return 1u32;
    }
}
fn main() { u32 _x = missing_else(false); }
"#;

    e0014_wrong_return_value_type_rejected(expect_err, DiagnosticKind::ReturnMismatch) => r#"
u32 wrong_ty() {
    return true;
}
fn main() { u32 _x = wrong_ty(); }
"#;

    // ── E0015: ErrorSpecMismatch ─────────────────────────────────────────────
    e0015_error_stmt_in_infallible_fn_rejected(expect_err, DiagnosticKind::ErrorSpecMismatch) => r#"
fn infallible() {
    error;
}
fn main() { infallible(); }
"#;

    e0015_raising_wrong_error_enum_type_rejected(expect_err, DiagnosticKind::ErrorSpecMismatch) => r#"
enum ErrA { A }
enum ErrB { B }
fn mismatch() error ErrA {
    error B;
}
fn main() { mismatch() catch (ErrA _) {}; }
"#;

    // ── E0016: UnhandledFallibleExpr ─────────────────────────────────────────
    e0016_unhandled_dynamic_slice_fmt_buf_rejected(expect_err, DiagnosticKind::UnhandledFallibleExpr) => r#"
fn dyn_slice([]u8 dst, u8 x) {
    []u8 _s = fmt_buf(dst, f"v={x}");
}
fn main() {
    [8]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    dyn_slice(buf, 1u8);
}
"#;

    e0016_unhandled_typed_error_call_rejected(expect_err, DiagnosticKind::UnhandledFallibleExpr) => r#"
enum BusErr { Nack }
u8 read_reg() error BusErr { error Nack; }
fn main() {
    u8 _v = read_reg();
}
"#;

    // ── E0017: InvalidCatchSignature ─────────────────────────────────────────
    e0017_value_catch_without_fallback_or_diverge_rejected(expect_err, DiagnosticKind::InvalidCatchSignature) => r#"
enum E { Fail }
u32 fallible() error E { error Fail; }
fn bad() {
    u32 _x = fallible() catch (E _) {
        sys.nop();
    };
}
fn main() { bad(); }
"#;

    e0017_catch_wrong_parameter_type_rejected(expect_err, DiagnosticKind::InvalidCatchSignature) => r#"
enum E { Fail }
u32 fallible() error E { error Fail; }
fn bad() {
    u32 _x = fallible() catch (u32 _wrong) {
        return;
    };
}
fn main() { bad(); }
"#;

    // ── E0018: InvalidOrFallback ─────────────────────────────────────────────
    e0018_mismatched_or_fallback_type_rejected(expect_err, DiagnosticKind::InvalidOrFallback) => r#"
enum E { Fail }
u32 fallible() error E { error Fail; }
fn main() {
    u32 _x = fallible() or false;
}
"#;

    e0018_or_fallback_on_infallible_expr_rejected(expect_err, DiagnosticKind::InvalidOrFallback) => r#"
u32 pure_val() { return 7u32; }
fn main() {
    u32 _x = pure_val() or 0u32;
}
"#;

    // ── E0019: TypeMismatch ──────────────────────────────────────────────────
    e0019_incompatible_pointer_assignment_rejected(expect_err, DiagnosticKind::TypeMismatch) => r#"
fn main() {
    u32 x = 1u32;
    *u8 _p = &x;
}
"#;

    e0019_non_boolean_if_condition_rejected(expect_err, DiagnosticKind::TypeMismatch) => r#"
fn main() {
    if (5u32) {
        sys.nop();
    }
}
"#;

    // ── E0020: UndefinedSymbol ───────────────────────────────────────────────
    e0020_inner_block_local_used_after_scope_exit(expect_err, DiagnosticKind::UndefinedSymbol) => r#"
fn main() {
    {
        u32 inner_only = 9u32;
    }
    u32 _leak = inner_only;
}
"#;

    e0020_nonexistent_struct_method_rejected(expect_err, DiagnosticKind::UndefinedSymbol) => r#"
struct Cfg { u32 hz; }
fn main() {
    Cfg c = Cfg { hz: 100u32 };
    c.no_such_method();
}
"#;

    // ── E0021: DuplicateDefinition ───────────────────────────────────────────
    e0021_duplicate_top_level_fn_and_task_rejected(expect_err, DiagnosticKind::DuplicateDefinition) => r#"
fn worker() {}
task worker() { loop { delay(1ms); } }
fn main() { worker(); }
"#;

    e0021_duplicate_local_variable_in_same_scope_rejected(expect_err, DiagnosticKind::DuplicateDefinition) => r#"
fn main() {
    u32 dup = 1u32;
    u32 dup = 2u32;
}
"#;

    // ── E0022: ImmutableAssignment ───────────────────────────────────────────
    e0022_mutating_const_scalar_rejected(expect_err, DiagnosticKind::ImmutableAssignment) => r#"
const u32 LIMIT = 10u32;
fn main() { LIMIT = 20u32; }
"#;

    e0022_mutating_const_array_element_rejected(expect_err, DiagnosticKind::ImmutableAssignment) => r#"
const [2]u8 LUT = [1u8, 2u8];
fn main() { LUT[0] = 9u8; }
"#;

    e0022_mutating_const_struct_field_rejected(expect_err, DiagnosticKind::ImmutableAssignment) => r#"
struct Pair { u32 a; u32 b; }
const Pair P = Pair { a: 1u32, b: 2u32 };
fn main() { P.a = 9u32; }
"#;

    // ── E0023: InvalidControlFlow ────────────────────────────────────────────
    e0023_return_inside_defer_rejected(expect_err, DiagnosticKind::InvalidControlFlow) => r#"
fn bad_defer() {
    defer { return; }
}
fn main() { bad_defer(); }
"#;

    e0023_restart_outside_task_rejected(expect_err, DiagnosticKind::InvalidControlFlow) => r#"
fn not_a_task() {
    restart;
}
fn main() { not_a_task(); }
"#;

    // ── E0024: InvalidOperation ──────────────────────────────────────────────
    e0024_pointer_inside_packed_struct_rejected(expect_err, DiagnosticKind::InvalidOperation) => r#"
packed struct BadWirePtr {
    u8  op;
    *u8 ptr;
}
fn main() {}
"#;

    e0024_slice_inside_packed_struct_rejected(expect_err, DiagnosticKind::InvalidOperation) => r#"
packed struct BadWireSlice {
    u8   op;
    []u8 data;
}
fn main() {}
"#;

    // ── E0025: UnreachableCode ───────────────────────────────────────────────
    e0025_statement_after_restart_rejected(expect_err, DiagnosticKind::UnreachableCode) => r#"
task t() {
    restart;
    delay(1ms);
}
fn main() { spawn t() @normal; }
"#;

    e0025_statement_after_return_rejected(expect_err, DiagnosticKind::UnreachableCode) => r#"
u32 dead_tail() {
    return 1u32;
    u32 x = 2u32;
    return x;
}
fn main() { u32 _v = dead_tail(); }
"#;

    // ── E0026: NonDivergingTask ──────────────────────────────────────────────
    e0026_return_inside_task_rejected(expect_err, DiagnosticKind::NonDivergingTask) => r#"
task bad_ret() {
    return;
}
fn main() { spawn bad_ret() @normal; }
"#;

    e0026_task_falling_off_end_rejected(expect_err, DiagnosticKind::NonDivergingTask) => r#"
task bad_fall(bool c) {
    if (c) {
        restart;
    }
}
fn main() { spawn bad_fall(true) @normal; }
"#;

    // ── E0027: InvalidSpawnContext ───────────────────────────────────────────
    e0027_boot_spawning_helper_yielding_via_delay_rejected(expect_err, DiagnosticKind::InvalidSpawnContext) => r#"
task w() {
    loop { delay(10ms); }
}
fn bad_boot() {
    spawn w() @normal;
    delay(20ms);
}
fn main() { bad_boot(); }
"#;

    e0027_spawn_inside_loop_rejected(expect_err, DiagnosticKind::InvalidSpawnContext) => r#"
task w() { loop { delay(10ms); } }
fn main() {
    for (u32 i in 0..2) {
        spawn w() @normal;
    }
}
"#;

    // ── E0028: FormatBufferTooSmall ──────────────────────────────────────────
    e0028_fmt_buf_u8_one_byte_too_small(expect_err, DiagnosticKind::FormatBufferTooSmall) => r#"
fn bad_u8(u8 x) {
    [4]u8 buf = [0u8, 0u8, 0u8, 0u8];
    []u8 _s = fmt_buf(buf, f"v={x}");
}
fn main() { bad_u8(255u8); }
"#;

    e0028_fmt_buf_i32_min_sign_overflow(expect_err, DiagnosticKind::FormatBufferTooSmall) => r#"
fn bad_i32(i32 x) {
    [10]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 _s = fmt_buf(buf, f"{x}");
}
fn main() { bad_i32(-2147483648); }
"#;

    e0028_fmt_buf_padded_hex_overflow(expect_err, DiagnosticKind::FormatBufferTooSmall) => r#"
fn bad_hex(u32 x) {
    [7]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 _s = fmt_buf(buf, f"{x:08x}");
}
fn main() { bad_hex(0x1234u32); }
"#;

    e0028_fmt_buf_self_aliasing_rejected(expect_err, DiagnosticKind::FormatBufferTooSmall) => r#"
fn self_alias() {
    [16]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 _s = fmt_buf(buf, f"x={buf}");
}
fn main() { self_alias(); }
"#;

    e0028_fmt_buf_subslice_capacity_overflow(expect_err, DiagnosticKind::FormatBufferTooSmall) => r#"
fn bad_subslice(u8 x) {
    [8]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 _s = fmt_buf(buf[0..4], f"v={x}");
}
fn main() { bad_subslice(255u8); }
"#;

    e0028_fmt_buf_enum_longest_variant_overflow(expect_err, DiagnosticKind::FormatBufferTooSmall) => r#"
enum State { Ok, Disconnected }
fn fmt_enum_too_small(State s) {
    [11]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 _out = fmt_buf(buf, f"{s}");
}
fn main() { fmt_enum_too_small(State.Ok); }
"#;

    // ── E0029: GenericInstantiationError ─────────────────────────────────────
    e0029_generic_wrong_type_arg_arity_rejected(expect_mono_err, DiagnosticKind::GenericInstantiationError) => r#"
T id_val<T>(T x) { return x; }
fn main() {
    u32 _v = id_val<u32, u8>(1u32);
}
"#;

    e0029_generic_non_const_usize_arg_rejected(expect_mono_err, DiagnosticKind::GenericInstantiationError) => r#"
struct Fixed<usize N> { [N]u8 data; }
fn bad(usize n) {
    Fixed<n> _f;
}
fn main() { bad(4usize); }
"#;

    // ── E0030: ModuleError ───────────────────────────────────────────────────
    e0030_missing_imported_module_rejected(expect_mono_err, DiagnosticKind::ModuleError) => r#"
import nonexistent.ghost_mod;
fn main() {}
"#;

    e0030_duplicate_module_import_alias_rejected(expect_mono_err, DiagnosticKind::ModuleError) => r#"
import std.math as m;
import std.str as m;
fn main() {}
"#;

    // ── E0031: MmioViolation ─────────────────────────────────────────────────
    e0031_write_to_ro_mmio_register_rejected(expect_err, DiagnosticKind::MmioViolation) => r#"
mmio Dev {
    @ro u32 status @ 0x00;
    @wo u32 cmd    @ 0x04;
}
mmio DEV : Dev @ 0x4000_0000;
fn main() { DEV.status = 1u32; }
"#;

    e0031_read_from_wo_mmio_register_rejected(expect_err, DiagnosticKind::MmioViolation) => r#"
mmio Dev {
    @ro u32 status @ 0x00;
    @wo u32 cmd    @ 0x04;
}
mmio DEV : Dev @ 0x4000_0000;
fn main() { u32 _x = DEV.cmd; }
"#;

    e0031_address_of_mmio_register_rejected(expect_err, DiagnosticKind::MmioViolation) => r#"
mmio Dev { @rw u32 cr @ 0x00; }
mmio DEV : Dev @ 0x4000_0000;
fn main() { *u32 _p = &DEV.cr; }
"#;

    e0031_by_value_mmio_parameter_rejected(expect_err, DiagnosticKind::MmioViolation) => r#"
mmio Dev { @rw u32 cr @ 0x00; }
mmio DEV : Dev @ 0x4000_0000;
fn take(Dev d) { d.cr = 1u32; }
fn main() { take(DEV); }
"#;

    // ── E0032: IrqAttachmentError ────────────────────────────────────────────
    e0032_irq_attaching_boot_helper_yielding_rejected(expect_err, DiagnosticKind::IrqAttachmentError) => r#"
irq IRQ_0 = 0;
isr h() {}
fn soft_step() { checkpoint(); }
fn bad_irq_boot() {
    IRQ_0.attach(h());
    soft_step();
}
fn main() { bad_irq_boot(); }
"#;

    e0032_duplicate_exclusive_irq_attach_rejected(expect_err, DiagnosticKind::IrqAttachmentError) => r#"
irq IRQ_0 = 0;
isr h1() {}
isr h2() {}
fn main() {
    IRQ_0.attach(h1());
    IRQ_0.attach(h2());
}
"#;

    // ── E0033: InvalidIsrState ───────────────────────────────────────────────
    e0033_state_var_outside_isr_rejected(expect_err, DiagnosticKind::InvalidIsrState) => r#"
fn not_an_isr() {
    state u32 count = 0u32;
    count += 1u32;
}
fn main() { not_an_isr(); }
"#;

    e0033_direct_isr_state_ptr_escape_via_inbox(expect_err, DiagnosticKind::InvalidIsrState) => r#"
irq IRQ_0 = 0;
inbox<*u32, 4> q;
isr leak_isr(*inbox<*u32, 4> qq) {
    state u32 x = 0u32;
    qq.post(&x);
}
fn main() { IRQ_0.attach(leak_isr(&q)); }
"#;

    e0033_indirect_local_ptr_isr_state_escape(expect_err, DiagnosticKind::InvalidIsrState) => r#"
irq IRQ_0 = 0;
inbox<*u32, 4> q;
isr indirect_leak_isr(*inbox<*u32, 4> qq) {
    state u32 x = 0u32;
    *u32 p = &x;
    qq.post(p);
}
fn main() { IRQ_0.attach(indirect_leak_isr(&q)); }
"#;

    e0033_transitive_helper_isr_state_escape(expect_err, DiagnosticKind::InvalidIsrState) => r#"
irq IRQ_0 = 0;
inbox<*u32, 4> q;
fn sneak_post(*u32 p, *inbox<*u32, 4> qq) {
    qq.post(p);
}
isr bad_isr(*inbox<*u32, 4> qq) {
    state u32 count = 0u32;
    sneak_post(&count, qq);
}
fn main() { IRQ_0.attach(bad_isr(&q)); }
"#;

    // ── E0034: LinearLeaseViolation ──────────────────────────────────────────
    e0034_leaked_lease_at_scope_exit(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
task t() {
    loop {
        Lease<[4]u8> _l = p.alloc();
        delay(5ms);
    }
}
fn main() { spawn t() @normal; }
"#;

    e0034_leaked_lease_on_early_return(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
fn helper(bool bail) {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    if (bail) {
        return;
    }
    l.release();
}
fn main() { helper(true); }
"#;

    e0034_leaked_lease_on_error_return(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
enum E { Fail }
pool<[4]u8, 2> p;
fn helper(bool bail) error E {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    if (bail) {
        error Fail;
    }
    l.release();
}
fn main() { helper(true) catch (E _) {}; }
"#;

    e0034_leaked_lease_on_loop_break(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
fn helper() {
    for (u32 i in 0..4) {
        Lease<[4]u8> l = p.try_alloc() catch { break; };
        if (i == 2u32) {
            break;
        }
        l.release();
    }
}
fn main() { helper(); }
"#;

    e0034_leaked_lease_on_loop_continue(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
fn helper() {
    for (u32 i in 0..4) {
        Lease<[4]u8> l = p.try_alloc() catch { continue; };
        if (i == 1u32) {
            continue;
        }
        l.release();
    }
}
fn main() { helper(); }
"#;

    e0034_asymmetric_if_without_else_consuming_lease(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
fn helper(bool cond) {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    if (cond) {
        l.release();
    }
}
fn main() { helper(true); }
"#;

    e0034_asymmetric_switch_arm_leaking_lease(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
fn helper(u8 sel) {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    switch (sel) {
        0u8 => { l.release(); }
        _ => {}
    }
}
fn main() { helper(1u8); }
"#;

    e0034_double_release_rejected(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
fn helper() {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    l.release();
    l.release();
}
fn main() { helper(); }
"#;

    e0034_use_after_move_into_helper_rejected(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
fn sink(Lease<[4]u8> l) { l.release(); }
fn helper() {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    sink(l);
    l.release();
}
fn main() { helper(); }
"#;

    e0034_use_after_inbox_send_rejected(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
inbox<Lease<[4]u8>, 2> q;
task t() {
    loop {
        Lease<[4]u8> l = p.alloc();
        q.send(l);
        u8 _x = l.data()[0];
        delay(5ms);
    }
}
fn main() { spawn t() @normal; }
"#;

    e0034_overwriting_live_lease_without_release_rejected(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
fn helper() {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    l = p.try_alloc() catch { l.release(); return; };
    l.release();
}
fn main() { helper(); }
"#;

    e0034_generic_double_move_rejected(expect_mono_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
fn consume_any<T>(Lease<T> l) {
    l.release();
}
fn helper() {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    consume_any<[4]u8>(l);
    consume_any<[4]u8>(l);
}
fn main() { helper(); }
"#;

    e0034_discarded_unused_lease_expression_rejected(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
task t() {
    loop {
        p.alloc();
        delay(5ms);
    }
}
fn main() { spawn t() @normal; }
"#;

    e0034_struct_embedding_lease_rejected_or_unconsumed(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
pool<[4]u8, 2> p;
struct Wrapper { Lease<[4]u8> buf; }
fn leak_wrapper() {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    Wrapper w = Wrapper { buf: l };
    Wrapper _unused = w;
}
fn main() { leak_wrapper(); }
"#;

    e0034_leaking_lease_inside_catch_block_rejected(expect_err, DiagnosticKind::LinearLeaseViolation) => r#"
enum E { Fail }
pool<[4]u8, 2> p;
u32 fallible() error E { error Fail; }
fn bad_catch() {
    u32 _v = fallible() catch (E _) {
        Lease<[4]u8> _l = p.try_alloc() catch { return; };
        return;
    };
}
fn main() { bad_catch(); }
"#;
}

// ── `E0013` Deadline-Horizon Boundary (largest accepted durations) ──────────

#[test]
fn e0013_durations_at_the_i32_tick_deadline_horizon_accepted() {
    expect_ok(
        r#"
const u32 PERIOD = 2147483647;
task w() {
    loop { delay(2147483647ms); }
}
task p() {
    loop every PERIOD { checkpoint(); }
}
fn main() { spawn w() @normal; spawn p() @normal; }
"#,
    );
}

// ── Multi-File `E0030` Visibility & Forged Canonical Name Tests ─────────────

#[test]
fn e0030_forged_private_canonical_symbol_across_modules_rejected() {
    let base = std::env::temp_dir().join(format!("zev-adv-e0030-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("create temp module dir");
    std::fs::write(
        base.join("vault.zev"),
        "u32 hidden_key() { return 7u32; }\npub u32 public_api() { return hidden_key(); }\n",
    )
    .expect("write vault.zev");

    let main_src = "import vault;\nfn main() { u32 _x = vault__hidden_key(); }\n";
    let prog = zev_parser::parse_str(main_src).expect("parse main");
    let mut opts = AnalyzeOptions::default();
    opts.include_dirs.push(base.clone());
    let res = analyze_and_monomorphize_programs(&[prog], &opts).into_result();
    let _ = std::fs::remove_dir_all(&base);

    let diags = res.expect_err("forged canonical private symbol must fail with E0030");
    assert!(
        diags.iter().any(|d| d.kind == DiagnosticKind::ModuleError),
        "expected E0030 ModuleError, got: {diags:?}"
    );
}

// ── Positive Soundness Boundary Controls (Separate `#[test]` per Control) ───

#[test]
fn positive_soft_checkpoint_allowed_inside_noyield() {
    expect_ok(
        r#"
fn soft() { checkpoint(); }
task t() {
    loop {
        noyield { soft(); }
        soft();
        delay(1ms);
    }
}
fn main() { spawn t() @normal; }
"#,
    );
}

#[test]
fn positive_symmetric_if_else_generic_lease_consumption() {
    expect_mono_ok(
        r#"
pool<[4]u8, 2> p;
fn consume_any<T>(Lease<T> l) {
    l.release();
}
fn helper(bool cond) {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    if (cond) {
        consume_any<[4]u8>(l);
    } else {
        l.release();
    }
}
fn main() { helper(true); }
"#,
    );
}

#[test]
fn positive_defer_lease_release_on_early_return() {
    expect_ok(
        r#"
pool<[4]u8, 2> p;
fn safe_defer(bool early) {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    defer { l.release(); }
    if (early) {
        return;
    }
}
fn main() { safe_defer(true); }
"#,
    );
}

#[test]
fn positive_fmt_buf_exact_fit_u8_5_bytes() {
    expect_ok(
        r#"
fn ok_u8(u8 x) {
    [5]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 _s = fmt_buf(buf, f"v={x}");
}
fn main() { ok_u8(255u8); }
"#,
    );
}

#[test]
fn positive_fmt_buf_exact_fit_i32_11_bytes() {
    expect_ok(
        r#"
fn ok_i32(i32 x) {
    [11]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 _s = fmt_buf(buf, f"{x}");
}
fn main() { ok_i32(-2147483648); }
"#,
    );
}

#[test]
fn positive_fmt_buf_exact_fit_enum_12_bytes() {
    expect_ok(
        r#"
enum State { Ok, Disconnected }
fn fmt_enum_ok(State s) {
    [12]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 _out = fmt_buf(buf, f"{s}");
}
fn main() { fmt_enum_ok(State.Ok); }
"#,
    );
}

#[test]
fn positive_fmt_buf_dynamic_slice_with_catch() {
    expect_ok(
        r#"
fn dyn_slice_caught([]u8 dst, u8 x) {
    []u8 _s = fmt_buf(dst, f"v={x}") catch { return; };
}
fn main() {
    [8]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    dyn_slice_caught(buf, 1u8);
}
"#,
    );
}

#[test]
fn positive_const_and_caller_ptr_array_to_slice_return() {
    expect_ok(
        r#"
const [4]u8 LUT = [1u8, 2u8, 3u8, 4u8];
struct S { [4]u8 arr; }
[]u8 lut_view() { return LUT; }
[]u8 view(*S p) { return p.arr; }
fn main() {
    []u8 _a = lut_view();
    S inst = S { arr: [5u8, 6u8, 7u8, 8u8] };
    []u8 _b = view(&inst);
}
"#,
    );
}

// ── Completeness Guard Over All 34 `E0001`..`E0034` Codes ───────────────────

#[test]
fn meta_all_34_e_codes_have_at_least_two_independent_adversarial_tests() {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for kind in COVERED_NEGATIVE_KINDS {
        *counts.entry(kind.code()).or_insert(0) += 1;
    }

    let mut under_covered = Vec::new();
    for kind in DiagnosticKind::ALL {
        let n = counts.get(kind.code()).copied().unwrap_or(0);
        if n < 2 {
            under_covered.push(format!("{} ({:?}): {} test(s)", kind.code(), kind, n));
        }
    }

    assert!(
        under_covered.is_empty(),
        "every DiagnosticKind in E0001..E0034 must have >= 2 standalone adversarial #[test]s; under-covered:\n{}",
        under_covered.join("\n")
    );
}

#[test]
fn wait_block_arm_validation_scoping_e0011_e0034_and_divergence() {
    // 1. E0011 stale byte_ring lease inside wait arm
    expect_err(
        r#"
byte_ring<16> rx;
task t() {
    loop {
        wait {
            []u8 buf = rx.wait_read() => {
                rx.consume(buf.len);
                u8 _stale = buf[0];
            }
            timeout 10ms => {}
        }
    }
}
fn main() { spawn t(); }
"#,
        DiagnosticKind::StaleDmaLease,
    );

    // 2. E0011 stale ping_pong DMA slice inside wait arm
    expect_err(
        r#"
@dma_safe ping_pong<u8, 8> pp;
task t() {
    loop {
        wait {
            []u8 c0 = pp.wait_chunk() => {
                []u8 c1 = pp.wait_chunk();
                u8 _stale = c0[0];
            }
        }
    }
}
fn main() { spawn t(); }
"#,
        DiagnosticKind::StaleDmaLease,
    );

    // 3. E0034 unconsumed linear Lease<T> inside wait arm
    expect_err(
        r#"
pool<u32, 2> p;
task t() {
    loop {
        wait {
            Lease<u32> l = p.alloc() => {
                *u32 ptr = l.get();
                *ptr = 7u32;
            }
        }
    }
}
fn main() { spawn t(); }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 4. Reject helper fn calls, per-arm timeout:, empty wait {}, void binding, critical/isr/noyield
    expect_err(
        r#"
signal s;
fn helper() { s.wait(); }
task t() { loop { wait { helper() => {} } } }
fn main() { spawn t(); }
"#,
        DiagnosticKind::InvalidOperation,
    );
    expect_err(
        r#"
inbox<u32, 4> q;
task t() { loop { wait { u32 x = q.recv(timeout: 5ms) => {} } } }
fn main() { spawn t(); }
"#,
        DiagnosticKind::InvalidOperation,
    );
    expect_err(
        r#"
task t() { loop { wait {} } }
fn main() { spawn t(); }
"#,
        DiagnosticKind::InvalidOperation,
    );
    expect_err(
        r#"
signal s;
task t() { loop { wait { u32 x = s.wait() => {} } } }
fn main() { spawn t(); }
"#,
        DiagnosticKind::TypeMismatch,
    );
    expect_err(
        r#"
signal s;
task t() { loop { critical { wait { s.wait() => {} } } } }
fn main() { spawn t(); }
"#,
        DiagnosticKind::YieldInCritical,
    );
    expect_err(
        r#"
signal s;
task t() { loop { noyield { wait { s.wait() => {} } } } }
fn main() { spawn t(); }
"#,
        DiagnosticKind::YieldInNoYield,
    );
    expect_err(
        r#"
signal s;
irq Q = 1;
isr h() { wait { s.wait() => {} } }
fn main() { Q.attach(h); }
"#,
        DiagnosticKind::YieldInIsr,
    );
}
