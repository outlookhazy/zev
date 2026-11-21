use std::path::Path;

use zev_ast::Priority;
use zev_sema::{
    AnalyzeOptions, Diagnostic, DiagnosticKind, PrimitiveKind, SemanticModel, analyze,
    analyze_named_programs, analyze_program,
};

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

    let p_mc = zev_parser::parse_file(&mc_path).expect("parse motor_controller.zev");
    let p_ms = zev_parser::parse_file(&ms_path).expect("parse motor_support.zev");

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
    let prog = zev_parser::parse_file(&sim_path).expect("parse sim_demo.zev");
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
            every 1ms {
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
            every 1ms {
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
            every 500us {
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
            every 500us {
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

    // (b) Yielding call inside noyield / critical
    expect_err_kind(
        r#"
        fn do_yield() { checkpoint(); }
        fn main() {
            noyield { do_yield(); }
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
            []u8 escaped = rx_dma.wait_chunk();
            {
                []u8 half = rx_dma.wait_chunk();
                ByteReader r = ByteReader{ buf: half };
                escaped = r.buf;
                []u8 half2 = rx_dma.wait_chunk();
                u8 stale = half[0] + half2[0];
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
        task fast() { every 500us { } }
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
        let prog = zev_parser::parse_file(&path).unwrap_or_else(|e| panic!("parse {rel}: {e}"));
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
fn doc_topic_and_doc_command_cover_all_e0001_through_e0024() {
    assert_eq!(DiagnosticKind::ALL.len(), 24);

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
    // 1. `inbox.send` -> suggests `.post(...)`
    let d_inbox = expect_err_kind(
        r#"
        enum Ev { Ping }
        inbox<Ev, 8> q;
        fn main() {
            q.send(Ping);
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
            []u8 chunk = rx.recv();
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

    // 5. `chan.push` -> suggests `.send(...)` / `.try_send(...)`
    let d_chan = expect_err_kind(
        r#"
        chan<u32, 4> c;
        fn main() {
            c.push(42);
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_chan[0].message.contains(".send("), "got: {}", d_chan[0]);
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



