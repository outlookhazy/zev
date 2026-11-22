use std::path::Path;

use zev_ast::Priority;
use zev_sema::{
    AnalyzeOptions, Diagnostic, DiagnosticKind, FnEffect, PrimitiveKind, SemanticModel, analyze,
    analyze_named_programs, analyze_program,
};

fn load_example_prog(path: &Path) -> zev_ast::Program {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    // Normalize pre-Stage-4 `every <dur>` lines in `examples/*.zev` without modifying files outside `zev-sema`.
    let normalized = raw
        .replace("\n    every ", "\n    loop every ")
        .replace("\n        every ", "\n        loop every ");
    zev_parser::parse_str(&normalized)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path.display(), e))
}

fn check_src(src: &str) -> Result<SemanticModel, Vec<Diagnostic>> {
    let prog = zev_parser::parse_str(src)
        .unwrap_or_else(|e| panic!("test source failed to parse:\n{}\n---\n{}", e, src));
    analyze_program(&prog)
}

fn expect_ok(src: &str) -> SemanticModel {
    match check_src(src) {
        Ok(model) => model,
        Err(diags) => {
            let formatted: Vec<String> = diags.iter().map(ToString::to_string).collect();
            panic!(
                "expected clean semantic check, got {} diagnostic(s):\n{}\n---\n{}",
                diags.len(),
                formatted.join("\n"),
                src
            );
        }
    }
}

fn expect_err_kind(src: &str, expected: DiagnosticKind) -> Vec<Diagnostic> {
    match check_src(src) {
        Ok(_) => panic!(
            "expected error {:?}, but program passed semantic check:\n{}",
            expected, src
        ),
        Err(diags) => {
            assert!(
                diags.iter().any(|d| d.kind == expected),
                "expected diagnostic kind {:?}, got:\n{}",
                expected,
                diags
                    .iter()
                    .map(|d| format!("{:?}: {}", d.kind, d))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            for d in &diags {
                assert!(d.span.line >= 1, "diagnostic must have 1-indexed line >= 1");
                assert!(d.span.col >= 1, "diagnostic must have 1-indexed col >= 1");
            }
            diags
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. North Star Multi-File & Simulation Demo Verification
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn check_motor_controller_and_support_multifile() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mc_path = root.join("examples/motor_controller.zev");
    let ms_path = root.join("examples/motor_support.zev");

    let p_mc = load_example_prog(&mc_path);
    let p_ms = load_example_prog(&ms_path);

    let out = analyze_named_programs(
        &[
            (&p_mc, Some("motor_controller.zev")),
            (&p_ms, Some("motor_support.zev")),
        ],
        &AnalyzeOptions::default(),
    );
    let model = out.into_result().unwrap_or_else(|diags| {
        panic!(
            "motor_controller.zev + motor_support.zev failed sema:\n{}",
            diags
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        );
    });

    // Tick & clock configuration
    assert_eq!(model.tick_period_us, 1_000);
    assert_eq!(model.systick_hz, Some(170_000_000));
    assert_eq!(model.micros_to_ticks(1_000), 1);
    assert_eq!(model.micros_to_ticks(5_000_000), 5_000);

    // Yield inference across functions and tasks
    assert!(!model.can_yield("setup_clocks"));
    assert!(!model.can_yield("configure_circular_uart_dma"));
    assert!(!model.can_yield("crc16"));
    assert!(!model.can_yield("parse_setpoint"));
    assert!(!model.can_yield("pi_step"));

    assert!(model.can_yield("run_power_on_self_test"));
    assert!(model.can_yield("adc_read_rpm"));
    assert!(model.can_yield("save_fault_record"));
    assert!(model.can_yield("scrub_flash_crc"));
    assert!(model.can_yield("main"));
    assert!(model.can_yield("motor_control"));
    assert!(model.can_yield("fieldbus_listener"));
    assert!(model.main_can_yield);

    // Disjoint yielding child functions in `main`
    let main_yield_callees = model.yielding_callees("main");
    assert!(main_yield_callees.contains(&"run_power_on_self_test".to_string()));
    assert!(main_yield_callees.contains(&"save_fault_record".to_string()));
    assert!(main_yield_callees.contains(&"scrub_flash_crc".to_string()));

    // Spawned task instances & priorities
    assert_eq!(model.spawned_tasks.len(), 4);
    assert_eq!(model.spawned_tasks[0].task_name, "motor_control");
    assert_eq!(model.spawned_tasks[0].priority, Priority::Critical);
    assert_eq!(model.spawned_tasks[1].task_name, "motor_control");
    assert_eq!(model.spawned_tasks[1].priority, Priority::Critical);
    assert_eq!(model.spawned_tasks[2].task_name, "fieldbus_listener");
    assert_eq!(model.spawned_tasks[2].priority, Priority::Normal);
    assert_eq!(model.spawned_tasks[3].task_name, "fieldbus_listener");
    assert_eq!(model.spawned_tasks[3].priority, Priority::Normal);
    assert_eq!(model.main_supervisor_priority, Some(Priority::Low));

    // Primitives & DMA wake-lock clamping
    assert!(model.has_dma_sleep_locks());
    let u1 = &model.primitives["uart1_dma"];
    assert_eq!(u1.kind, PrimitiveKind::PingPong);
    assert!(u1.dma_safe);
    assert!(u1.clamps_sleep_depth);
    assert_eq!(u1.capacity, Some(64));

    // ISR boost metadata
    let exti0 = &model.isrs["EXTI0"];
    assert_eq!(exti0.priority_boosts.len(), 1);
    assert_eq!(exti0.priority_boosts[0].primitive_name, "sup_inbox");
    assert_eq!(exti0.priority_boosts[0].boost_priority, Priority::Critical);
}

#[test]
fn check_sim_demo_clean() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sim_path = root.join("examples/sim_demo.zev");
    let prog = load_example_prog(&sim_path);
    let model = analyze(&[prog]).unwrap_or_else(|diags| {
        panic!(
            "sim_demo.zev failed sema:\n{}",
            diags
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        );
    });

    assert_eq!(model.spawned_tasks.len(), 5);
    assert!(model.functions["syn_sim_log"].is_extern_c);
    assert!(!model.can_yield("syn_sim_log"));
    assert!(model.can_yield("sim_stimulus"));
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Yield Inference, `noyield`, `critical`, `isr`, and `extern "C"` Callbacks
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn transitive_yield_inference_and_manual_yield_suppression() {
    let model = expect_ok(
        r#"
        signal done_sig;

        fn leaf_wait() {
            done_sig.wait();
        }

        fn mid_step() {
            leaf_wait();
        }

        fn top_caller() {
            mid_step();
        }

        fn manual_spi_tx() {
            u32 i = 0;
            @manual_yield loop {
                if (i >= 8) {
                    break;
                }
                i += 1;
            }
        }

        fn auto_yield_bg() {
            u32 i = 0;
            loop {
                if (i >= 100) {
                    break;
                }
                i += 1;
            }
        }

        fn main() {
            top_caller();
            manual_spi_tx();
            auto_yield_bg();
        }
        "#,
    );

    assert!(model.can_yield("leaf_wait"));
    assert!(model.can_yield("mid_step"));
    assert!(model.can_yield("top_caller"));
    assert!(!model.can_yield("manual_spi_tx"));
    assert!(model.can_yield("auto_yield_bg"));
}

#[test]
fn reject_yielding_call_in_noyield_critical_and_isr() {
    // Transitive yield inside `noyield`
    expect_err_kind(
        r#"
        signal s;
        fn wait_helper() { s.wait(); }
        fn caller() { wait_helper(); }
        fn main() {
            noyield {
                caller();
            }
        }
        "#,
        DiagnosticKind::YieldInNoYield,
    );

    // Direct yield inside `critical`
    expect_err_kind(
        r#"
        fn main() {
            critical {
                checkpoint();
            }
        }
        "#,
        DiagnosticKind::YieldInCritical,
    );

    // Transitive yield inside `critical`
    expect_err_kind(
        r#"
        fn slow_op() { delay(2ms); }
        fn main() {
            critical {
                slow_op();
            }
        }
        "#,
        DiagnosticKind::YieldInCritical,
    );

    // Direct & transitive yield inside `isr`
    expect_err_kind(
        r#"
        inbox<u32, 4> q;
        isr TIM2 {
            u32 v = q.recv();
        }
        fn main() {}
        "#,
        DiagnosticKind::YieldInIsr,
    );

    expect_err_kind(
        r#"
        fn flash_sync() { checkpoint(); }
        isr EXTI1 {
            flash_sync();
        }
        fn main() {}
        "#,
        DiagnosticKind::YieldInIsr,
    );
}

#[test]
fn reject_auto_yield_loop_in_noyield_and_unbounded_loop_in_critical() {
    // Auto-yield `loop` inside `noyield`
    expect_err_kind(
        r#"
        fn main() {
            noyield {
                u32 x = 0;
                loop {
                    x += 1;
                    if (x > 10) { break; }
                }
            }
        }
        "#,
        DiagnosticKind::YieldInNoYield,
    );

    // Unbounded `loop` inside `critical` (even with `break` or `@manual_yield`)
    expect_err_kind(
        r#"
        fn main() {
            critical {
                u32 x = 0;
                @manual_yield loop {
                    x += 1;
                    if (x > 4) { break; }
                }
            }
        }
        "#,
        DiagnosticKind::UnboundedLoopInCritical,
    );

    // `loop` inside `isr`
    expect_err_kind(
        r#"
        isr SPI1 {
            loop {
                break;
            }
        }
        fn main() {}
        "#,
        DiagnosticKind::YieldInIsr,
    );
}

#[test]
fn extern_c_callback_safety_direct_and_struct_wrapped() {
    // Non-yielding callback to extern "C" is allowed!
    expect_ok(
        r#"
        extern "C" fn register_blk_hook(fn(u32) -> i32 cb);

        i32 fast_hook(u32 sector) {
            return 0;
        }

        fn main() {
            register_blk_hook(fast_hook);
        }
        "#,
    );

    // Direct/transitive yielding callback passed to extern "C" is rejected!
    expect_err_kind(
        r#"
        extern "C" fn register_blk_hook(fn(u32) -> i32 cb);
        signal flash_ready;

        fn wait_flash() {
            flash_ready.wait();
        }

        i32 bad_yielding_hook(u32 sector) {
            wait_flash();
            return 0;
        }

        fn main() {
            register_blk_hook(bad_yielding_hook);
        }
        "#,
        DiagnosticKind::YieldingCallbackToExternC,
    );

    // Yielding callback wrapped inside a struct passed by pointer to extern "C" is ALSO rejected!
    expect_err_kind(
        r#"
        struct BlockOps {
            fn(u32) -> i32 read_block;
        }
        extern "C" fn mount_fs(BlockOps *ops);

        i32 bad_read(u32 lba) {
            checkpoint();
            return 0;
        }

        fn main() {
            BlockOps ops = BlockOps { read_block: bad_read };
            mount_fs(&ops);
        }
        "#,
        DiagnosticKind::YieldingCallbackToExternC,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Task & ISR Isolation (Zero Mutable Raw Globals)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn reject_mutable_raw_global_and_isolation_violation_in_isr_and_task() {
    let diags = expect_err_kind(
        r#"
        u32 shared_counter = 0;

        isr EXTI0 {
            shared_counter += 1;
        }

        task worker() {
            loop every 1ms {
                u32 snap = shared_counter;
            }
        }

        fn main() {
            spawn worker() @normal;
        }
        "#,
        DiagnosticKind::IllegalMutableGlobal,
    );

    // Both declaration error AND task/ISR isolation violations are reported!
    let isolation_count = diags
        .iter()
        .filter(|d| d.kind == DiagnosticKind::GlobalIsolationViolation)
        .count();
    assert_eq!(isolation_count, 2, "expected isolation errors in both EXTI0 and worker");
}

#[test]
fn allow_immutable_const_in_isr_and_task_but_reject_mutation() {
    expect_ok(
        r#"
        const u32 MAX_LIMIT = 1000;
        watch<u32> out_val = 0;

        isr EXTI0 {
            out_val.publish(MAX_LIMIT);
        }

        fn main() {}
        "#,
    );

    expect_err_kind(
        r#"
        const u32 MAX_LIMIT = 1000;
        fn main() {
            MAX_LIMIT = 2000;
        }
        "#,
        DiagnosticKind::ImmutableAssignment,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Scope-Depth & DMA Lease Origin Tracking (Zero `'a` Lifetimes)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allow_local_struct_wrapping_and_downward_calls_for_dma_slices() {
    expect_ok(
        r#"
        struct ByteReader {
            []u8  buf;
            usize pos;
        }

        @dma_safe ping_pong<u8, 64> rx_dma;
        @dma_safe lease<u8, 32>      spi_lease;

        u8 peek_first(ByteReader *r) {
            if (r.buf.len == 0) {
                return 0;
            }
            return r.buf[0];
        }

        ByteReader make_sub_reader([]u8 chunk) {
            return ByteReader { buf: chunk[0..4], pos: 0 };
        }

        task consumer() {
            loop {
                []u8 half = rx_dma.wait_chunk();
                ByteReader reader = ByteReader { buf: half, pos: 0 };
                u8 b0 = peek_first(&reader);

                {
                    ByteReader sub = make_sub_reader(half);
                    u8 b1 = peek_first(&sub);
                }

                // Reassigning `half` to the next chunk refreshes `half`'s lease cleanly!
                half = rx_dma.wait_chunk();
                u8 b2 = half[0];

                []u8 spi_buf = spi_lease.wait_chunk();
                u8 s0 = spi_buf[0];
            }
        }

        fn main() {
            spawn consumer() @normal @core(0);
        }
        "#,
    );
}

#[test]
fn reject_escaping_dma_slice_and_wrapper_struct_to_outer_scope() {
    // Escaping `half` directly to outer-scope variable
    expect_err_kind(
        r#"
        @dma_safe ping_pong<u8, 64> rx_dma;

        task bad_escape() {
            []u8 leaked;
            loop {
                []u8 half = rx_dma.wait_chunk();
                leaked = half;
            }
        }

        fn main() {
            spawn bad_escape() @normal;
        }
        "#,
        DiagnosticKind::ScopeDepthEscape,
    );

    // Escaping `ByteReader { buf: half, pos: 0 }` wrapper struct to outer-scope variable
    expect_err_kind(
        r#"
        struct ByteReader {
            []u8  buf;
            usize pos;
        }

        @dma_safe ping_pong<u8, 64> rx_dma;

        task bad_wrapper_escape() {
            ByteReader outer_reader;
            loop {
                []u8 half = rx_dma.wait_chunk();
                ByteReader local_reader = ByteReader { buf: half, pos: 0 };
                outer_reader = local_reader;
            }
        }

        fn main() {
            spawn bad_wrapper_escape() @normal;
        }
        "#,
        DiagnosticKind::ScopeDepthEscape,
    );

    // Escaping via field assignment `outer_reader.buf = half;`
    expect_err_kind(
        r#"
        struct ByteReader {
            []u8  buf;
            usize pos;
        }

        @dma_safe ping_pong<u8, 64> rx_dma;

        task bad_field_escape() {
            ByteReader outer_reader;
            loop {
                []u8 half = rx_dma.wait_chunk();
                outer_reader.buf = half;
            }
        }

        fn main() {
            spawn bad_field_escape() @normal;
        }
        "#,
        DiagnosticKind::ScopeDepthEscape,
    );
}

#[test]
fn reject_returning_local_pointer_slice_or_leased_wrapper() {
    // Returning `&local` upward
    expect_err_kind(
        r#"
        *i32 bad_ret_ptr() {
            i32 local_val = 42;
            return &local_val;
        }
        fn main() {}
        "#,
        DiagnosticKind::ReturnLocalReference,
    );

    // Returning slice of local fixed array upward
    expect_err_kind(
        r#"
        []u8 bad_ret_slice() {
            u8 tmp[8] = [1, 2, 3, 4, 5, 6, 7, 8];
            return tmp[0..4];
        }
        fn main() {}
        "#,
        DiagnosticKind::ReturnLocalReference,
    );

    // Returning `ByteReader` wrapping `wait_chunk()` upward
    expect_err_kind(
        r#"
        struct ByteReader {
            []u8  buf;
            usize pos;
        }
        @dma_safe ping_pong<u8, 64> rx_dma;

        ByteReader bad_ret_lease() {
            []u8 half = rx_dma.wait_chunk();
            return ByteReader { buf: half, pos: 0 };
        }
        fn main() {}
        "#,
        DiagnosticKind::ReturnLocalReference,
    );

    // Spawning a task with `&local` stack variable
    expect_err_kind(
        r#"
        task worker(i32 *p) {
            loop every 1ms {
                *p += 1;
            }
        }
        fn main() {
            i32 stack_local = 0;
            spawn worker(&stack_local) @normal;
        }
        "#,
        DiagnosticKind::SpawnNonStaticReference,
    );
}

#[test]
fn reject_reading_stale_dma_slice_or_wrapper_after_second_wait_chunk() {
    // Reading `half` after second `wait_chunk()` in same scope
    expect_err_kind(
        r#"
        @dma_safe ping_pong<u8, 64> rx_dma;

        task bad_double_wait() {
            loop {
                []u8 half1 = rx_dma.wait_chunk();
                []u8 half2 = rx_dma.wait_chunk();
                u8 clash = half1[0];
            }
        }

        fn main() {
            spawn bad_double_wait() @normal;
        }
        "#,
        DiagnosticKind::StaleDmaLease,
    );

    // Reading wrapper struct `ByteReader` after second `wait_chunk()` (even via helper fn!)
    expect_err_kind(
        r#"
        struct ByteReader {
            []u8  buf;
            usize pos;
        }
        @dma_safe ping_pong<u8, 64> rx_dma;

        ByteReader wrap([]u8 b) {
            return ByteReader { buf: b, pos: 0 };
        }

        task bad_wrapper_stale() {
            loop {
                []u8 half = rx_dma.wait_chunk();
                ByteReader r = wrap(half);
                []u8 next = rx_dma.wait_chunk();
                u8 stale = r.buf[0];
            }
        }

        fn main() {
            spawn bad_wrapper_stale() @normal;
        }
        "#,
        DiagnosticKind::StaleDmaLease,
    );

    // Conditional branch `wait_chunk()` invalidating outer lease
    expect_err_kind(
        r#"
        @dma_safe lease<u8, 32> dma_lease;

        fn bad_branch_invalidation(bool flag) {
            []u8 buf = dma_lease.wait_chunk();
            if (flag) {
                []u8 buf2 = dma_lease.wait_chunk();
            }
            u8 x = buf[0];
        }

        fn main() {
            bad_branch_invalidation(true);
        }
        "#,
        DiagnosticKind::StaleDmaLease,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. No Recursion Across Yielding Paths
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn reject_yielding_recursion_cycle_and_allow_nonyielding_recursion() {
    // Pure stack recursion without yields is allowed
    let model = expect_ok(
        r#"
        u32 fact(u32 n) {
            if (n <= 1) {
                return 1;
            }
            return n * fact(n - 1);
        }
        fn main() {
            u32 x = fact(5);
        }
        "#,
    );
    assert!(!model.can_yield("fact"));

    // Mutual recursion `a -> b -> a` where `a` yields is rejected!
    let diags = expect_err_kind(
        r#"
        fn a(u32 n) {
            checkpoint();
            if (n > 0) {
                b(n - 1);
            }
        }

        fn b(u32 n) {
            a(n);
        }

        fn main() {
            a(3);
        }
        "#,
        DiagnosticKind::YieldingRecursion,
    );
    assert!(diags[0].message.contains("a -> b -> a"));

    // Direct self-recursion on yielding function is rejected!
    expect_err_kind(
        r#"
        fn retry_forever() {
            delay(5ms);
            retry_forever();
        }
        fn main() {
            retry_forever();
        }
        "#,
        DiagnosticKind::YieldingRecursion,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Time Resolution & Dual-Exit `return` / `error` / `catch` / `or`
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn duration_tick_resolution_validation() {
    // Default tick = 1ms (1000us) -> 500us truncates to 0 and MUST fail!
    expect_err_kind(
        r#"
        task fast_pwm() {
            loop every 500us {
                u32 x = 1;
            }
        }
        fn main() {
            spawn fast_pwm() @critical;
        }
        "#,
        DiagnosticKind::InvalidDurationResolution,
    );

    // Non-multiple 1500us on 1ms tick MUST also fail!
    expect_err_kind(
        r#"
        fn main() {
            delay(1500us);
        }
        "#,
        DiagnosticKind::InvalidDurationResolution,
    );

    // Configuring a 500us tick explicitly allows 500us and 1500us!
    let model = expect_ok(
        r#"
        task fast_pwm() {
            loop every 500us {
                u32 x = 1;
            }
        }
        fn main() {
            sys.use_systick(170_000_000, 500us);
            delay(1500us);
            spawn fast_pwm() @critical;
        }
        "#,
    );
    assert_eq!(model.tick_period_us, 500);
    assert_eq!(model.micros_to_ticks(1500), 3);
}

#[test]
fn dual_exit_error_partial_progress_catch_and_or_validation() {
    // Valid 0-arg `error;`, 1-arg `error Reason;`, and 2-arg `error Reason, partial;` + `catch (Err e, Val partial)`!
    expect_ok(
        r#"
        enum IoErr { Timeout, BusFault }

        fn probe_device() error {
            error;
        }

        usize read_packet([]u8 dst, bool fail_halfway) error IoErr {
            if (dst.len == 0) {
                error BusFault;
            }
            if (fail_halfway) {
                usize partial_bytes = 3;
                error Timeout, partial_bytes;
            }
            return dst.len;
        }

        fn main() {
            probe_device() catch {
                // handled bare error
            };

            u8 buf[8] = [0, 0, 0, 0, 0, 0, 0, 0];
            usize saved_partial = 0;
            usize n1 = read_packet(buf[0..8], true) catch (IoErr err, usize got) {
                saved_partial = got;
                got;
            };
            usize n2 = read_packet(buf[0..8], false) or 0;
        }
        "#,
    );

    // Unhandled fallible call must be rejected
    expect_err_kind(
        r#"
        enum Err { Bad }
        u16 fallible() error Err {
            error Bad;
        }
        fn main() {
            u16 x = fallible();
        }
        "#,
        DiagnosticKind::UnhandledFallibleExpr,
    );

    // `error;` (0 args) inside typed `error Err` fn must be rejected
    expect_err_kind(
        r#"
        enum Err { Bad }
        u16 fallible() error Err {
            error;
        }
        fn main() {}
        "#,
        DiagnosticKind::ErrorSpecMismatch,
    );

    // 2-param `catch (Err e, u16 p)` on untyped bare `error` fn must be rejected
    expect_err_kind(
        r#"
        u16 bare_fail() error {
            error;
        }
        fn main() {
            u16 v = bare_fail() catch (u32 e, u16 p) {
                return;
            };
        }
        "#,
        DiagnosticKind::InvalidCatchSignature,
    );

    // Missing return in non-void function must be rejected
    expect_err_kind(
        r#"
        u16 forgot_return(bool flag) {
            if (flag) {
                return 1;
            }
        }
        fn main() {}
        "#,
        DiagnosticKind::ReturnMismatch,
    );
}

#[test]
fn qa_adversarial_checklist_suite() {
    // (a) Raw mutable global accessed by ISR / task
    expect_err_kind(
        r#"
        u32 shared_counter = 0;
        isr EXTI0 { shared_counter += 1; }
        fn main() {}
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // (b) Hard-blocking call inside noyield / any yielding call inside critical
    expect_err_kind(
        r#"
        fn do_hard_yield() { delay(1ms); }
        fn main() {
            noyield { do_hard_yield(); }
        }
        "#,
        DiagnosticKind::YieldInNoYield,
    );
    expect_err_kind(
        r#"
        fn do_yield() { checkpoint(); }
        fn main() {
            critical { do_yield(); }
        }
        "#,
        DiagnosticKind::YieldInCritical,
    );

    // (c) Unbounded loop inside critical
    expect_err_kind(
        r#"
        fn main() {
            critical {
                loop { }
            }
        }
        "#,
        DiagnosticKind::UnboundedLoopInCritical,
    );

    // (d) Escaping DMA slice & wrapper struct `ByteReader{ buf: half }`, plus reading `half` after 2nd `wait_chunk()`
    let dma_diags = expect_err_kind(
        r#"
        @dma_safe ping_pong<u8, 64> rx_dma;
        struct ByteReader { []u8 buf; }
        task bad_dma() {
            loop {
                []u8 escaped = rx_dma.wait_chunk();
                {
                    []u8 half = rx_dma.wait_chunk();
                    ByteReader r = ByteReader{ buf: half };
                    escaped = r.buf;
                    []u8 half2 = rx_dma.wait_chunk();
                    u8 stale = half[0] + half2[0];
                }
            }
        }
        fn main() { spawn bad_dma() @normal; }
        "#,
        DiagnosticKind::ScopeDepthEscape,
    );
    assert!(
        dma_diags.iter().any(|d| d.kind == DiagnosticKind::StaleDmaLease),
        "expected StaleDmaLease alongside ScopeDepthEscape"
    );

    // (e) Recursive yielding cycle (`fn a()` -> `b()` -> `a()`)
    expect_err_kind(
        r#"
        fn a() { checkpoint(); b(); }
        fn b() { a(); }
        fn main() { a(); }
        "#,
        DiagnosticKind::YieldingRecursion,
    );

    // (f) Sub-tick duration `500us` on `1ms` tick
    expect_err_kind(
        r#"
        task fast() { loop every 500us { } }
        fn main() { sys.use_systick(170_000_000); spawn fast() @high; }
        "#,
        DiagnosticKind::InvalidDurationResolution,
    );

    // (g) Passing yielding function as callback to `extern "C"`
    expect_err_kind(
        r#"
        extern "C" fn register_irq_cb(fn() cb);
        fn bad_cb() { checkpoint(); }
        fn main() { register_irq_cb(bad_cb); }
        "#,
        DiagnosticKind::YieldingCallbackToExternC,
    );

    // (h) Valid partial-progress `error Reason, partial_val;` + `catch (Err e, Val partial)`
    expect_ok(
        r#"
        enum IoErr { ShortRead }
        usize read_some([]u8 b) error IoErr {
            error ShortRead, 2;
        }
        fn main() {
            u8 buf[8] = [0, 1, 2, 3, 4, 5, 6, 7];
            usize n = read_some(buf[0..4]) catch (IoErr e, usize partial) {
                return;
            };
        }
        "#,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Agent-Guided Diagnostics, Transitive Yield Chains, Method Hints & `zevc doc`
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn check_cortex_m_and_esp32c6_examples_clean() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for rel in ["examples/cortex_m_demo.zev", "examples/esp32c6_demo.zev"] {
        let path = root.join(rel);
        let prog = load_example_prog(&path);
        analyze(&[prog]).unwrap_or_else(|diags| {
            panic!(
                "{rel} failed sema:\n{}",
                diags
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        });
    }
}

#[test]
fn doc_topic_and_doc_command_cover_all_e0001_through_e0027() {
    assert_eq!(DiagnosticKind::ALL.len(), 27);

    let expected_topics = [
        ("E0001", "primitives"),
        ("E0002", "primitives"),
        ("E0003", "yielding"),
        ("E0004", "yielding"),
        ("E0005", "yielding"),
        ("E0006", "yielding"),
        ("E0007", "yielding"),
        ("E0008", "dma"),
        ("E0009", "dma"),
        ("E0010", "dma"),
        ("E0011", "dma"),
        ("E0012", "yielding"),
        ("E0013", "tasks"),
        ("E0014", "errors"),
        ("E0015", "errors"),
        ("E0016", "errors"),
        ("E0017", "errors"),
        ("E0018", "errors"),
        ("E0019", "syntax"),
        ("E0020", "syntax"),
        ("E0021", "syntax"),
        ("E0022", "syntax"),
        ("E0023", "tasks"),
        ("E0024", "syntax"),
        ("E0025", "syntax"),
        ("E0026", "tasks"),
        ("E0027", "tasks"),
    ];

    for (kind, (code, topic)) in DiagnosticKind::ALL.iter().zip(expected_topics.iter()) {
        assert_eq!(kind.code(), *code);
        assert_eq!(kind.doc_topic(), *topic);
        let expected_cmd = format!("run 'zevc doc {}' or 'zevc doc {}'", code, topic);
        assert_eq!(kind.doc_command(), expected_cmd);
        assert!(!kind.doc_hint().is_empty());

        let sample = Diagnostic::new(*kind, zev_ast::Span::new(0, 4, 3, 5), "sample error")
            .with_file(Some("demo.zev".to_string()))
            .with_note(kind.doc_hint());
        let rendered = sample.to_string();
        assert!(
            rendered.contains(&format!("demo.zev:3:5: error[{}]: sample error", code)),
            "missing header in:\n{rendered}"
        );
        assert!(
            rendered.contains(&format!("= docs: {}", expected_cmd)),
            "missing = docs pointer in:\n{rendered}"
        );
    }
}

#[test]
fn multi_hop_shortest_transitive_yield_chains() {
    // 3-hop ISR -> f1() -> f2() -> checkpoint() [yields here]
    // Also includes a longer 4-hop path `f1 -> slow1 -> slow2 -> delay` to prove BFS picks shortest!
    let isr_diags = expect_err_kind(
        r#"
        fn f2() {
            checkpoint();
        }
        fn slow2() {
            delay(5ms);
        }
        fn slow1() {
            slow2();
        }
        fn f1() {
            slow1();
            f2();
        }
        isr EXTI0 {
            f1();
        }
        fn main() {}
        "#,
        DiagnosticKind::YieldInIsr,
    );
    let isr_note = isr_diags[0].note.as_deref().unwrap();
    assert!(
        isr_note.contains("yield chain: isr EXTI0 -> f1() -> f2() -> checkpoint() [yields here]"),
        "unexpected ISR yield chain note: {isr_note}"
    );
    assert!(isr_diags[0].to_string().contains("zevc doc E0005"));

    // Transitive yield chain inside `noyield`
    let ny_diags = expect_err_kind(
        r#"
        @dma_safe ping_pong<u8, 64> rx;
        fn read_half() {
            []u8 h = rx.wait_chunk();
        }
        fn step_mid() {
            read_half();
        }
        fn main() {
            noyield {
                step_mid();
            }
        }
        "#,
        DiagnosticKind::YieldInNoYield,
    );
    let ny_note = ny_diags[0].note.as_deref().unwrap();
    assert!(
        ny_note.contains("step_mid() -> read_half() -> rx.wait_chunk() [yields here]"),
        "unexpected noyield chain note: {ny_note}"
    );

    // Transitive yield chain inside `critical`
    let crit_diags = expect_err_kind(
        r#"
        inbox<u32, 4> q;
        fn pull_ev() {
            u32 x = q.recv();
        }
        fn handle_ev() {
            pull_ev();
        }
        fn main() {
            critical {
                handle_ev();
            }
        }
        "#,
        DiagnosticKind::YieldInCritical,
    );
    let crit_note = crit_diags[0].note.as_deref().unwrap();
    assert!(
        crit_note.contains("handle_ev() -> pull_ev() -> q.recv() [yields here]"),
        "unexpected critical chain note: {crit_note}"
    );

    // Transitive yield chain on `extern "C"` callback
    let cb_diags = expect_err_kind(
        r#"
        extern "C" fn reg_hook(fn() cb);
        signal done;
        fn inner_block() {
            done.wait();
        }
        fn outer_cb() {
            inner_block();
        }
        fn main() {
            reg_hook(outer_cb);
        }
        "#,
        DiagnosticKind::YieldingCallbackToExternC,
    );
    let cb_note = cb_diags[0].note.as_deref().unwrap();
    assert!(
        cb_note.contains("extern \"C\" reg_hook(...) callback -> outer_cb() -> inner_block() -> done.wait() [yields here]"),
        "unexpected extern C callback chain note: {cb_note}"
    );
}

#[test]
fn primitive_method_suggestions_for_foreign_idioms() {
    // 1. `inbox.enqueue` -> suggests `.post(...)` / `.send(...)`
    let d_inbox = expect_err_kind(
        r#"
        enum Ev { Ping }
        inbox<Ev, 8> q;
        fn main() {
            q.enqueue(Ping);
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_inbox[0].message.contains(".post("), "got: {}", d_inbox[0]);
    assert!(d_inbox[0].note.as_deref().unwrap().contains("valid methods on `inbox<Ev, 8>`"));

    // 2. `watch.get` & `watch.set` -> suggests `.read()` and `.publish(...)`
    let d_watch_get = expect_err_kind(
        r#"
        watch<u16> rpm = 0;
        fn main() {
            u16 x = rpm.get();
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_watch_get[0].message.contains(".read()"), "got: {}", d_watch_get[0]);

    let d_watch_set = expect_err_kind(
        r#"
        watch<u16> rpm = 0;
        fn main() {
            rpm.set(1200);
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_watch_set[0].message.contains(".publish("), "got: {}", d_watch_set[0]);

    // 3. `ping_pong.recv` -> suggests `.wait_chunk()`
    let d_pp = expect_err_kind(
        r#"
        @dma_safe ping_pong<u8, 64> rx;
        task listener() {
            loop {
                []u8 chunk = rx.recv();
            }
        }
        fn main() { spawn listener() @normal; }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_pp[0].message.contains(".wait_chunk()"), "got: {}", d_pp[0]);

    // 4. `signal.set` -> suggests `.notify()`
    let d_sig = expect_err_kind(
        r#"
        signal irq_done;
        isr EXTI0 {
            irq_done.set();
        }
        fn main() {}
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_sig[0].message.contains(".notify()"), "got: {}", d_sig[0]);

    // 5. `inbox.push` -> suggests `.post(...)` / `.send(...)` on unified `inbox`
    let d_push = expect_err_kind(
        r#"
        inbox<u32, 4> c;
        fn main() {
            c.push(42);
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_push[0].message.contains(".send("), "got: {}", d_push[0]);
}

#[test]
fn fuzzy_identifier_type_field_and_fixit_notes() {
    // Foreign C type `uint16_t` -> `u16`
    let d_ty = expect_err_kind(
        r#"
        fn main() {
            uint16_t x = 10;
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_ty[0].note.as_deref().unwrap().contains("`u16`"));

    // Identifier typo (`motor1_rmp` -> `motor1_rpm`)
    let d_id = expect_err_kind(
        r#"
        watch<u16> motor1_rpm = 0;
        fn main() {
            motor1_rmp.publish(100);
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_id[0].note.as_deref().unwrap().contains("`motor1_rpm`"));

    // Struct field typo (`setpiont` -> `setpoint`) & slice `.length` -> `.len`
    let d_field = expect_err_kind(
        r#"
        packed struct CmdFrame {
            u16be setpoint;
        }
        fn check_slice([]u8 buf, CmdFrame f) {
            usize n = buf.length;
            u16 s = f.setpiont;
        }
        fn main() {}
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_field.iter().any(|d| d.note.as_deref().unwrap().contains(".len")));
    assert!(d_field.iter().any(|d| d.note.as_deref().unwrap().contains("CmdFrame.setpoint")));

    // Raw global actionable `watch<u32>` / `inbox<u32, 8>` note
    let d_glob = expect_err_kind(
        r#"
        u32 fault_flags = 0;
        isr EXTI0 {
            fault_flags = 1;
        }
        fn main() {}
        "#,
        DiagnosticKind::IllegalMutableGlobal,
    );
    assert!(d_glob[0].note.as_deref().unwrap().contains("watch<u32> fault_flags"));
    assert!(d_glob[0].note.as_deref().unwrap().contains("inbox<u32, 8> fault_flags"));
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Built-In `print`/`println`/`assert`, Strict `sys.*`/`sim.*`, and Value `catch`
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn validate_print_println_and_assert_signatures() {
    // Valid string & *u8 arguments for print/println and bool for assert
    expect_ok(
        r#"
        fn emit_banner(*u8 prefix) {
            print(prefix);
            println("boot ok");
            assert(1 + 1 == 2);
        }
        fn main() {
            emit_banner("[ZEV] ");
        }
        "#,
    );

    // Non-string arguments to `print` / `println` -> E0019 suggesting `sys.log_kv`!
    let d_print = expect_err_kind(
        r#"
        fn main() {
            print(123);
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );
    assert!(d_print[0].code() == "E0019");
    assert!(
        d_print[0].note.as_deref().unwrap().contains("sys.log_kv"),
        "expected sys.log_kv hint, got: {:?}",
        d_print[0].note
    );

    let d_println = expect_err_kind(
        r#"
        fn main() {
            u32 rpm = 456;
            println(rpm);
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );
    assert!(d_println[0].code() == "E0019");
    assert!(d_println[0].note.as_deref().unwrap().contains("sys.log_kv"));

    // Wrong arity on `println`
    expect_err_kind(
        r#"
        fn main() {
            println("a=", "b");
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );

    // Non-bool argument to `assert` -> E0019
    expect_err_kind(
        r#"
        fn main() {
            assert(42);
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );
}

#[test]
fn strict_sys_and_sim_namespace_validation() {
    // Valid portable `sys.*` intrinsics
    expect_ok(
        r#"
        fn main() {
            sys.use_systick(12_000_000);
            u32 ms = sys.now_ms();
            u32 tk = sys.now_ticks();
            sys.pend_irq(5);
            sys.log("ready");
            sys.log_kv("delta=", -5);
            sys.log_hex("ticks=", ms + tk);
            sys.exit(0);
        }
        "#,
    );

    // Reject unknown `sim.*` methods with E0020 and note listing valid `sys.*` methods!
    for bad_sim in [
        "fn main() { sim.enable_trace(true); }",
        "fn main() { sim.log(1, 2, 3); }",
        "fn main() { u32 t = sim.now_ms(); }",
    ] {
        let diags = expect_err_kind(bad_sim, DiagnosticKind::UndefinedSymbol);
        assert_eq!(diags[0].code(), "E0020");
        assert!(
            diags[0]
                .note
                .as_deref()
                .unwrap()
                .contains("sys.log_kv(*u8 label, i32 val)"),
            "expected valid sys.* methods note, got: {:?}",
            diags[0].note
        );
    }

    // Reject unknown `sys.*` method with E0020 + fuzzy hint / valid table
    let d_sys_unknown = expect_err_kind(
        r#"
        fn main() {
            sys.enable_trace(true);
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert_eq!(d_sys_unknown[0].code(), "E0020");
    assert!(d_sys_unknown[0]
        .note
        .as_deref()
        .unwrap()
        .contains("sys.use_systick(u32 hz)"));

    // Reject wrong argument types on `sys.log` and `sys.log_kv` with E0019
    expect_err_kind(
        r#"
        fn main() {
            sys.log(123);
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );
    expect_err_kind(
        r#"
        fn main() {
            sys.log_kv(123, 456);
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );
}

#[test]
fn validate_value_producing_catch_blocks_e0017() {
    // Valid multi-statement value-recovering `catch` block!
    expect_ok(
        r#"
        enum PacketErr { BadHeader }
        struct DecodedPacket {
            u8  cmd;
            i16 temp;
        }

        DecodedPacket decode(u8 raw) error PacketErr {
            if (raw == 0) {
                error BadHeader;
            }
            return DecodedPacket { cmd: raw, temp: 25 };
        }

        fn main() {
            u32 count = 0;
            DecodedPacket p1 = decode(0) catch (PacketErr e) {
                count += 1;
                DecodedPacket { cmd: 9, temp: 42 }
            };
            DecodedPacket p2 = decode(0) catch (PacketErr e) {
                count += 1;
                DecodedPacket { cmd: 9, temp: 42 };
            };
            sys.log_kv("cmd=", p1.cmd + p2.cmd);
        }
        "#,
    );

    // Reject non-diverging `catch` on non-void `T` without a trailing fallback expression -> E0017!
    let d_missing_val = expect_err_kind(
        r#"
        enum PacketErr { BadHeader }
        u32 decode_u32(u8 raw) error PacketErr {
            error BadHeader;
        }
        fn main() {
            u32 count = 0;
            u32 val = decode_u32(0) catch (PacketErr e) {
                count += 1;
            };
        }
        "#,
        DiagnosticKind::InvalidCatchSignature,
    );
    assert_eq!(d_missing_val[0].code(), "E0017");

    // Reject non-diverging `catch` on non-void `T` whose trailing expression has wrong type -> E0017!
    let d_wrong_ty = expect_err_kind(
        r#"
        enum PacketErr { BadHeader }
        struct DecodedPacket {
            u8  cmd;
            i16 temp;
        }
        DecodedPacket decode(u8 raw) error PacketErr {
            error BadHeader;
        }
        fn main() {
            DecodedPacket p = decode(0) catch (PacketErr e) {
                true;
            };
        }
        "#,
        DiagnosticKind::InvalidCatchSignature,
    );
    assert_eq!(d_wrong_ty[0].code(), "E0017");

    // Reject non-diverging `catch` ending with a `void` call on a non-void `T` -> E0017!
    let d_void_tail = expect_err_kind(
        r#"
        enum PacketErr { BadHeader }
        u32 decode_u32(u8 raw) error PacketErr {
            error BadHeader;
        }
        fn main() {
            u32 val = decode_u32(0) catch (PacketErr e) {
                println("failed");
            };
        }
        "#,
        DiagnosticKind::InvalidCatchSignature,
    );
    assert_eq!(d_void_tail[0].code(), "E0017");
}



// ─────────────────────────────────────────────────────────────────────────────
// 9. Section 6.2 Wave 1 — Stage 2 Semantic Invariants
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unreachable_code_e0025_all_diverging_forms_and_single_warning_per_tail() {
    // 1. Multiple statements after `return;` emit E0025 ONLY ONCE per unreachable tail.
    let diags = expect_err_kind(
        r#"
        u32 compute() {
            return 7;
            u32 dead1 = 1;
            u32 dead2 = dead1 + 2;
        }
        fn main() {
            u32 x = compute();
        }
        "#,
        DiagnosticKind::UnreachableCode,
    );
    let e0025_count = diags
        .iter()
        .filter(|d| d.kind == DiagnosticKind::UnreachableCode)
        .count();
    assert_eq!(e0025_count, 1, "expected exactly 1 E0025 for dead tail, got: {diags:?}");
    assert_eq!(diags[0].code(), "E0025");

    // 2. Unreachable after `error`, `break`, `continue`, `restart`, and `sys.exit(...)`.
    expect_err_kind(
        r#"
        fn fail_fn() error {
            error;
            println("dead");
        }
        fn main() {}
        "#,
        DiagnosticKind::UnreachableCode,
    );
    expect_err_kind(
        r#"
        fn main() {
            loop {
                break;
                println("dead after break");
            }
        }
        "#,
        DiagnosticKind::UnreachableCode,
    );
    expect_err_kind(
        r#"
        fn main() {
            for (usize i in 0..4) {
                continue;
                println("dead after continue");
            }
        }
        "#,
        DiagnosticKind::UnreachableCode,
    );
    expect_err_kind(
        r#"
        task actor() {
            restart;
            println("dead after restart");
        }
        fn main() { spawn actor(); }
        "#,
        DiagnosticKind::UnreachableCode,
    );
    expect_err_kind(
        r#"
        fn main() {
            sys.exit(0);
            println("dead after sys.exit");
        }
        "#,
        DiagnosticKind::UnreachableCode,
    );

    // 3. Unreachable after break-less `loop`, `loop every`, `while (true)`, all-diverging `if/else`, and exhaustive `switch`.
    for snippet in [
        r#"
        fn main() {
            loop { checkpoint(); }
            println("dead");
        }
        "#,
        r#"
        fn main() {
            loop every 10ms { }
            println("dead");
        }
        "#,
        r#"
        fn main() {
            while (true) { checkpoint(); }
            println("dead");
        }
        "#,
        r#"
        fn check_branch(bool c) {
            if (c) {
                return;
            } else {
                return;
            }
            println("dead");
        }
        fn main() { check_branch(true); }
        "#,
        r#"
        enum Mode { A, B }
        fn check_switch(Mode m) {
            switch (m) {
                A => { return; }
                B => { return; }
            }
            println("dead");
        }
        fn main() { check_switch(A); }
        "#,
    ] {
        expect_err_kind(snippet, DiagnosticKind::UnreachableCode);
    }

    // 4. Loops WITH `break` do NOT diverge and allow subsequent statements cleanly!
    expect_ok(
        r#"
        fn main() {
            loop {
                break;
            }
            loop every 10ms {
                break;
            }
            while (true) {
                break;
            }
            println("reachable");
        }
        "#,
    );
}

#[test]
fn immortal_tasks_e0026_restart_and_sys_exit() {
    // 1. Finite task falling off closing `}` -> E0026 (`NonDivergingTask`)
    let d_falloff = expect_err_kind(
        r#"
        watch<u32> w = 0;
        task one_shot() {
            w.publish(42);
        }
        fn main() {
            spawn one_shot();
        }
        "#,
        DiagnosticKind::NonDivergingTask,
    );
    assert_eq!(d_falloff[0].code(), "E0026");

    // 2. Empty `return;` inside a `task` -> E0026 (`NonDivergingTask`)
    let d_ret = expect_err_kind(
        r#"
        task early_exit(bool stop) {
            loop {
                if (stop) {
                    return;
                }
                checkpoint();
            }
        }
        fn main() {
            spawn early_exit(false);
        }
        "#,
        DiagnosticKind::NonDivergingTask,
    );
    assert_eq!(d_ret.len(), 1, "expected single E0026 on `return;`, got: {d_ret:?}");

    // 3. Valid immortal tasks: `restart;` recovery, `sys.exit(0)` sim driver, `loop` / `loop every`,
    //    plus `fn main()` exiting `}` cleanly after `spawn`s!
    let model = expect_ok(
        r#"
        enum Fault { ResetReq, Ok }
        inbox<Fault, 4> fault_q;

        task fault_tolerant_worker() {
            u32 state = 1;
            loop {
                Fault f = fault_q.recv();
                if (f == ResetReq) {
                    restart;
                }
                state += 1;
            }
        }

        task self_reset_actor() {
            delay(5ms);
            restart;
        }

        task sim_harness() {
            delay(10ms);
            sys.exit(0);
        }

        fn main() {
            spawn fault_tolerant_worker() @high;
            spawn self_reset_actor() @normal;
            spawn sim_harness() @low;
        }
        "#,
    );
    assert_eq!(model.spawned_tasks.len(), 3);
    assert_eq!(model.fn_effect("self_reset_actor"), FnEffect::HardBlocking);

    // 4. `restart;` outside a `task` (`fn` or `isr`) -> E0023 (`InvalidControlFlow`)
    expect_err_kind(
        r#"
        fn helper() {
            restart;
        }
        fn main() {}
        "#,
        DiagnosticKind::InvalidControlFlow,
    );
    expect_err_kind(
        r#"
        isr EXTI0 {
            restart;
        }
        fn main() {}
        "#,
        DiagnosticKind::InvalidControlFlow,
    );

    // 5. `restart;` inside `noyield` (`E0003`) or `critical` (`E0004`) inside a `task`
    expect_err_kind(
        r#"
        task bad_ny() {
            noyield {
                restart;
            }
        }
        fn main() { spawn bad_ny(); }
        "#,
        DiagnosticKind::YieldInNoYield,
    );
    expect_err_kind(
        r#"
        task bad_crit() {
            critical {
                restart;
            }
        }
        fn main() { spawn bad_crit(); }
        "#,
        DiagnosticKind::YieldInCritical,
    );
}

#[test]
fn boot_tree_spawn_provenance_unrolling_conditional_and_e0027_e0010() {
    // 1. Valid boot-tree unrolling: 2x helper call (`setup_bus(&m1, 1); setup_bus(&m2, 2);`)
    //    + conditional `if` boot branch (`usb_start()`) + `const` slice parameter!
    let model = expect_ok(
        r#"
        const u8 INIT_SEQ[4] = [1, 2, 3, 4];
        watch<u16> m1 = 0;
        watch<u16> m2 = 0;
        signal usb_irq_sig;

        task bus_worker(watch<u16> *rpm, []u8 init_bytes, u8 bus_id) {
            loop every 2ms {
                u16 cur = rpm.read() + init_bytes[0] + bus_id;
                rpm.publish(cur);
            }
        }

        task usb_worker() {
            loop {
                usb_irq_sig.wait();
            }
        }

        fn setup_bus(watch<u16> *rpm, []u8 seq, u8 id) {
            spawn bus_worker(rpm, seq, id) @high;
        }

        fn usb_start() {
            spawn usb_worker() @critical @core(1);
        }

        fn main() {
            bool enable_usb = true;
            setup_bus(&m1, INIT_SEQ[0..4], 1);
            setup_bus(&m2, INIT_SEQ[0..4], 2);
            if (enable_usb) {
                usb_start();
            }
        }
        "#,
    );

    assert!(model.functions["setup_bus"].spawns_in_boot);
    assert!(model.functions["usb_start"].spawns_in_boot);
    assert!(model.functions["main"].spawns_in_boot);

    assert_eq!(model.tasks["bus_worker"].instance_count, 2);
    assert_eq!(model.tasks["usb_worker"].instance_count, 1);
    assert_eq!(model.spawned_tasks.len(), 3);

    let s0 = &model.spawned_tasks[0];
    assert_eq!(s0.instance_id, 0);
    assert_eq!(s0.task_name, "bus_worker");
    assert_eq!(s0.spawned_in, "setup_bus");
    assert_eq!(s0.provenance_path, vec!["main", "setup_bus", "bus_worker"]);
    assert_eq!(s0.format_provenance(), "main -> setup_bus -> bus_worker");
    assert!(!s0.conditional);
    assert_eq!(s0.endpoint_bindings[0].as_deref(), Some("m1"));

    let s1 = &model.spawned_tasks[1];
    assert_eq!(s1.instance_id, 1);
    assert_eq!(s1.task_name, "bus_worker");
    assert_eq!(s1.provenance_path, vec!["main", "setup_bus", "bus_worker"]);
    assert!(!s1.conditional);
    assert_eq!(s1.endpoint_bindings[0].as_deref(), Some("m2"));

    let s2 = &model.spawned_tasks[2];
    assert_eq!(s2.instance_id, 2);
    assert_eq!(s2.task_name, "usb_worker");
    assert_eq!(s2.provenance_path, vec!["main", "usb_start", "usb_worker"]);
    assert!(s2.conditional);
    assert_eq!(s2.core, Some(1));
    assert_eq!(s2.priority, Priority::Critical);

    // 2. Passing `&local` from `main()` into a boot helper that forwards it to `spawn` -> E0010 (`SpawnNonStaticReference`)!
    expect_err_kind(
        r#"
        task ptr_task(u16 *p) {
            loop every 1ms {
                *p += 1;
            }
        }
        fn boot_wire(u16 *p) {
            spawn ptr_task(p) @normal;
        }
        fn main() {
            u16 stack_local = 99;
            boot_wire(&stack_local);
        }
        "#,
        DiagnosticKind::SpawnNonStaticReference,
    );

    // 3. `E0027` (`InvalidSpawnContext`) across `task`, `isr`, loops, recursive cycles, and post-supervisor `main()`!
    for bad_spawn in [
        // Direct spawn inside `task`
        r#"
        task child() { loop { checkpoint(); } }
        task parent() {
            spawn child();
            loop { checkpoint(); }
        }
        fn main() { spawn parent(); }
        "#,
        // Transitive boot-helper call inside `task`
        r#"
        task child() { loop { checkpoint(); } }
        fn wire_child() { spawn child(); }
        task parent() {
            loop {
                wire_child();
                checkpoint();
            }
        }
        fn main() { spawn parent(); }
        "#,
        // Direct & transitive spawn inside `isr`
        r#"
        task child() { loop { checkpoint(); } }
        fn wire_child() { spawn child(); }
        isr TIM2 {
            wire_child();
        }
        fn main() {}
        "#,
        // Transitive spawn inside bounded `for` loop in `main()`
        r#"
        task child() { loop { checkpoint(); } }
        fn wire_child() { spawn child(); }
        fn main() {
            for (usize i in 0..2) {
                wire_child();
            }
        }
        "#,
        // Recursive cycle across boot-spawning functions
        r#"
        task child() { loop { checkpoint(); } }
        fn boot_a(u8 n) {
            spawn child();
            if (n > 0) { boot_b(n - 1); }
        }
        fn boot_b(u8 n) {
            boot_a(n);
        }
        fn main() { boot_a(2); }
        "#,
        // Spawn AFTER `main()`'s supervisor loop (even when supervisor loop has `break`!)
        r#"
        task child() { loop { checkpoint(); } }
        fn wire_child() { spawn child(); }
        fn main() {
            loop {
                break;
            }
            wire_child();
        }
        "#,
    ] {
        let diags = expect_err_kind(bad_spawn, DiagnosticKind::InvalidSpawnContext);
        assert!(diags.iter().any(|d| d.code() == "E0027"));
    }
}

#[test]
fn loop_every_runtime_period_and_unified_inbox_methods() {
    // Valid `loop every` with both `DurationLit` and runtime integer expression, plus unified `inbox` methods!
    let model = expect_ok(
        r#"
        inbox<u32, 8> cmd_q;
        watch<u32> rate_ms = 5;

        task producer() {
            u32 seq = 0;
            loop every rate_ms.read() {
                bool ok = cmd_q.try_send(seq);
                if (!ok) {
                    cmd_q.send(seq);
                    cmd_q.send(seq, timeout: 10ms) catch {
                        cmd_q.post(seq);
                    };
                }
                seq += 1;
            }
        }

        task consumer() {
            loop every 2ms {
                if (!cmd_q.is_empty() || cmd_q.is_full() || cmd_q.len() > 0) {
                    u32 v1 = cmd_q.try_recv() or 0;
                    u32 v2 = cmd_q.recv();
                    sys.log_kv("sum=", v1 + v2);
                }
            }
        }

        fn main() {
            spawn producer() @normal;
            spawn consumer() @high;
        }
        "#,
    );
    assert_eq!(model.fn_effect("producer"), FnEffect::HardBlocking);
    assert_eq!(model.fn_effect("consumer"), FnEffect::HardBlocking);
    assert_eq!(model.primitives["cmd_q"].kind, PrimitiveKind::Inbox);

    // Reject `loop every 0` -> E0013 (`InvalidDurationResolution`)
    expect_err_kind(
        r#"
        task bad_zero() {
            loop every 0 {
                checkpoint();
            }
        }
        fn main() { spawn bad_zero(); }
        "#,
        DiagnosticKind::InvalidDurationResolution,
    );

    // Reject non-duration / non-integer period -> E0019 (`TypeMismatch`)
    expect_err_kind(
        r#"
        task bad_ty() {
            loop every true {
                checkpoint();
            }
        }
        fn main() { spawn bad_ty(); }
        "#,
        DiagnosticKind::TypeMismatch,
    );

    // Reject `inbox.send` inside `isr` (`E0005`) and `noyield` (`E0003`)
    expect_err_kind(
        r#"
        inbox<u32, 4> q;
        isr EXTI0 {
            q.send(1);
        }
        fn main() {}
        "#,
        DiagnosticKind::YieldInIsr,
    );
    expect_err_kind(
        r#"
        inbox<u32, 4> q;
        fn main() {
            noyield {
                q.send(1);
            }
        }
        "#,
        DiagnosticKind::YieldInNoYield,
    );
}

#[test]
fn three_tier_fn_effects_noyield_specialization_and_dual_specialized_fns() {
    // 1. Dual-specialization: `sha_round` & `sha_block` are `SoftYielding`, called BOTH in normal
    //    coroutine task AND inside `noyield { ... }`! Meanwhile `fast_only_soft` is ONLY called in `noyield`.
    let model = expect_ok(
        r#"
        u32 pure_mix(u32 a, u32 b) {
            return a ^ b;
        }

        u32 sha_round(u32 acc) {
            checkpoint();
            return pure_mix(acc, 0x5A5A);
        }

        u32 sha_block(u32 seed) {
            u32 cur = seed;
            u32 iter = 0;
            loop {
                if (iter >= 4) {
                    break;
                }
                cur = sha_round(cur);
                iter += 1;
            }
            return cur;
        }

        u32 fast_only_soft(u32 x) {
            checkpoint();
            return x + 1;
        }

        u32 atomic_wrapper(u32 x) {
            u32 out = 0;
            noyield {
                out = fast_only_soft(x) + sha_block(x);
            }
            return out;
        }

        task bg_hasher() {
            u32 h = 1;
            loop {
                h = sha_block(h);
                delay(5ms);
            }
        }

        fn main() {
            u32 v = atomic_wrapper(10);
            sys.log_kv("v=", v);
            spawn bg_hasher() @low;
        }
        "#,
    );

    assert_eq!(model.fn_effect("pure_mix"), FnEffect::NonYielding);
    assert_eq!(model.fn_effect("sha_round"), FnEffect::SoftYielding);
    assert_eq!(model.fn_effect("sha_block"), FnEffect::SoftYielding);
    assert_eq!(model.fn_effect("fast_only_soft"), FnEffect::SoftYielding);
    // `atomic_wrapper` invokes soft-yielding fns strictly inside `noyield`, so `atomic_wrapper` itself never suspends!
    assert_eq!(model.fn_effect("atomic_wrapper"), FnEffect::NonYielding);
    assert!(!model.can_yield("atomic_wrapper"));
    assert_eq!(model.fn_effect("bg_hasher"), FnEffect::HardBlocking);

    // `sha_block` and `sha_round` are reached in BOTH coroutine and `noyield` modes -> dual-specialized!
    assert!(model.soft_yielding_in_coro.contains("sha_block"));
    assert!(model.soft_yielding_in_coro.contains("sha_round"));
    assert!(model.soft_yielding_in_noyield.contains("sha_block"));
    assert!(model.soft_yielding_in_noyield.contains("sha_round"));
    assert!(model.dual_specialized_fns.contains("sha_block"));
    assert!(model.dual_specialized_fns.contains("sha_round"));

    // `fast_only_soft` is ONLY reached inside `noyield` -> NOT in `soft_yielding_in_coro` and NOT dual-specialized!
    assert!(model.soft_yielding_in_noyield.contains("fast_only_soft"));
    assert!(!model.soft_yielding_in_coro.contains("fast_only_soft"));
    assert!(!model.dual_specialized_fns.contains("fast_only_soft"));

    // 2. `SoftYielding` functions remain FORBIDDEN inside `critical` (`E0004`), `isr` (`E0005`), and `extern "C"` callbacks (`E0006`)!
    expect_err_kind(
        r#"
        fn soft_fn() { checkpoint(); }
        fn main() {
            critical {
                soft_fn();
            }
        }
        "#,
        DiagnosticKind::YieldInCritical,
    );
    expect_err_kind(
        r#"
        fn soft_fn() { checkpoint(); }
        isr EXTI0 {
            soft_fn();
        }
        fn main() {}
        "#,
        DiagnosticKind::YieldInIsr,
    );
    expect_err_kind(
        r#"
        extern "C" fn reg_cb(fn() cb);
        fn soft_fn() { checkpoint(); }
        fn main() {
            reg_cb(soft_fn);
        }
        "#,
        DiagnosticKind::YieldingCallbackToExternC,
    );

    // 3. Non-breaking infinite `loop` inside an `fn` is `HardBlocking` and rejected inside `noyield` (`E0003`)!
    expect_err_kind(
        r#"
        fn spin_forever() {
            loop {
                checkpoint();
            }
        }
        fn main() {
            noyield {
                spin_forever();
            }
        }
        "#,
        DiagnosticKind::YieldInNoYield,
    );
}

#[test]
fn defer_and_bounded_for_in_verification() {
    // 1. Valid `defer` scope-exit cleanup + bounded `for (T i in a..b)` / `for (T i in a..=b)` / `for (T x in slice)`
    //    (including inside `noyield`, `critical`, and `isr`!)
    let model = expect_ok(
        r#"
        import chip.stm32g4 as hw;

        enum SpiErr { TransferFailed }
        @dma_safe ping_pong<u8, 32> rx_dma;
        watch<u32> last_sum = 0;

        u32 tx_frame([]u8 payload, bool fail) error SpiErr {
            hw.GPIOA.clear_pin(PIN_0);
            defer hw.GPIOA.set_pin(PIN_0);

            u32 acc = 0;
            for (usize i in 0..payload.len) {
                acc += payload[i];
            }
            for (u8 bit in 0..=7) {
                acc ^= bit;
            }
            if (fail) {
                error TransferFailed;
            }
            return acc;
        }

        isr DMA1_CH5 {
            u8 lut[4] = [1, 2, 3, 4];
            u32 s = 0;
            for (u8 v in lut) {
                s += v;
            }
            last_sum.publish(s);
            rx_dma.flip();
        }

        task dma_reader() {
            loop {
                []u8 half = rx_dma.wait_chunk();
                defer {
                    last_sum.publish(99);
                }
                u32 total = 0;
                noyield {
                    for (u8 b in half) {
                        total += b;
                    }
                }
                u32 res = tx_frame(half, false) or 0;
                last_sum.publish(total + res);
            }
        }

        fn main() {
            critical {
                for (u32 ch in 0..4) {
                    hw.TIM1.set_compare(ch, 0);
                }
            }
            spawn dma_reader() @normal;
        }
        "#,
    );
    assert_eq!(model.fn_effect("tx_frame"), FnEffect::NonYielding);
    assert!(!model.can_yield("tx_frame"));

    // 2. Yielding inside `defer` (both `SoftYielding` and `HardBlocking`) -> E0003 (`YieldInNoYield`)
    expect_err_kind(
        r#"
        fn soft_op() { checkpoint(); }
        fn main() {
            defer soft_op();
        }
        "#,
        DiagnosticKind::YieldInNoYield,
    );
    expect_err_kind(
        r#"
        fn main() {
            defer delay(1ms);
        }
        "#,
        DiagnosticKind::YieldInNoYield,
    );

    // 3. Control-flow jumps (`return`, `error`, `break`, `continue`, `restart`, `spawn`) and nested `defer` inside `defer` -> E0023
    for bad_defer in [
        r#"
        fn main() {
            defer { return; }
        }
        "#,
        r#"
        fn f() error {
            defer { error; }
        }
        fn main() {}
        "#,
        r#"
        fn main() {
            for (usize i in 0..4) {
                defer { break; }
            }
        }
        "#,
        r#"
        fn main() {
            for (usize i in 0..4) {
                defer { continue; }
            }
        }
        "#,
        r#"
        task t() {
            defer { restart; }
            loop { checkpoint(); }
        }
        fn main() { spawn t(); }
        "#,
        r#"
        task t() { loop { checkpoint(); } }
        fn main() {
            defer spawn t();
        }
        "#,
        r#"
        fn main() {
            defer {
                defer println("nested");
            }
        }
        "#,
    ] {
        expect_err_kind(bad_defer, DiagnosticKind::InvalidControlFlow);
    }

    // 4. Iterating a DMA slice `for (u8 b in half)` across a `wait_chunk()` inside the loop body -> E0011 (`StaleDmaLease`)!
    expect_err_kind(
        r#"
        @dma_safe ping_pong<u8, 32> rx_dma;
        task bad_iter() {
            loop {
                []u8 half = rx_dma.wait_chunk();
                for (u8 b in half) {
                    u8 cur = b;
                    []u8 next = rx_dma.wait_chunk();
                    sys.log_kv("v=", cur + next[0]);
                }
            }
        }
        fn main() { spawn bad_iter(); }
        "#,
        DiagnosticKind::StaleDmaLease,
    );

    // 5. Invalid `for` range / `for..in` element types -> E0019 (`TypeMismatch`)
    expect_err_kind(
        r#"
        fn main() {
            for (bool b in 0..2) {}
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );
    expect_err_kind(
        r#"
        fn main() {
            u32 not_a_slice = 10;
            for (u8 x in not_a_slice) {}
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );
}
