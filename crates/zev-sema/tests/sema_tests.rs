use std::path::Path;

use zev_ast::{ArrayLen, AttrKind, ItemKind, Priority, TypeKind};
use zev_sema::{
    AnalyzeOptions, Diagnostic, DiagnosticKind, FnEffect, MAX_MONOMORPHIZED_INSTANCES,
    MonomorphizedInstance, MonomorphizedKind, PrimitiveKind, SemanticModel, analyze,
    analyze_and_monomorphize_named_programs, analyze_and_monomorphize_programs,
    analyze_named_programs, analyze_program, mangle_type_expr, monomorphize_named_programs,
    monomorphize_programs,
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

    // ISR boost metadata + Boot-Tree `irq` attachments
    let oc = &model.isrs["overcurrent_isr"];
    assert_eq!(oc.priority_boosts.len(), 1);
    assert_eq!(oc.priority_boosts[0].primitive_name, "param:sup");
    assert_eq!(oc.priority_boosts[0].boost_priority, Priority::Critical);
    assert_eq!(model.irq_bindings.len(), 3);
    assert_eq!(
        model.irq_bindings[0].args.first(),
        Some(&zev_sema::IrqBindingArg::PrimAddr("sup_inbox".to_string()))
    );
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
    assert!(model.functions.values().all(|f| !f.is_extern_c));
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
fn doc_topic_and_doc_command_cover_all_e0001_through_e0033() {
    assert_eq!(DiagnosticKind::ALL.len(), 33);

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
        ("E0028", "runtime"),
        ("E0029", "syntax"),
        ("E0030", "runtime"),
        ("E0031", "runtime"),
        ("E0032", "runtime"),
        ("E0033", "yielding"),
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
            hw.GPIOA.brr = 1;
            defer hw.GPIOA.bsrr = 1;

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
                    hw.TIM1.ccr[ch as usize] = 0;
                }
            }
            hw.IRQ_DMA1_CH5.attach(DMA1_CH5);
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

// ─────────────────────────────────────────────────────────────────────────────
// 10. Section 6.2 Wave 2 — IEEE-754 Floats, Monomorphization & Zero-Heap `f"..."`
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn wave2_floats_f32_f64_literals_coercion_ops_casts_and_errors() {
    // 1. Valid `f32` and `f64` literals (suffixed, unsuffixed, scientific), contextual coercion,
    //    arithmetic (`+ - * / %`), unary `-`, compound assign (`+= -= *= /= %=`), comparisons,
    //    casts (`int <-> float`, `f32 <-> f64`), OS primitives (`watch<f32>`, `inbox<f64, 4>`),
    //    and dual-exit `or` / `catch`!
    let model = expect_ok(
        r#"
        const f32 KP = 0.75;
        const f64 KI = 1.25e-2;

        struct PidState {
            f32 integral;
            f64 accum64;
        }

        watch<f32> duty_out = 0.0;
        inbox<f64, 4> sample_q;

        enum SensorErr { Disconnected }

        f32 read_adc_volts(u16 raw, bool ok) error SensorErr {
            if (!ok) {
                error Disconnected;
            }
            return ((f32)raw) * 3.3 / 4095.0;
        }

        f32 step_pid(PidState *st, f32 target, f32 measured) {
            f32 err = target - measured;
            st.integral += err * 0.01;
            st.integral -= 0.0;
            st.integral *= 0.99;
            st.integral /= 1.0;
            st.integral %= 10.0f32;

            f64 wide = (f64)err + KI * 2.5e1;
            st.accum64 = wide % 1.5;

            if (err >= -0.05 && err <= 0.05 && wide != 0.0 && !(err == 99.0)) {
                return -0.0;
            }
            return KP * err + st.integral + (f32)st.accum64;
        }

        task control_loop() {
            PidState st = PidState { integral: 0.0, accum64: 0.0 };
            f32 weights[3] = [0.25, 0.5, 0.25];
            loop every 2ms {
                f32 v1 = read_adc_volts(2048, true) or 0.0;
                f32 v2 = read_adc_volts(0, false) catch (SensorErr e) {
                    1.65
                };
                f32 cmd = step_pid(&st, v1 + v2, weights[0]);
                i32 pwm_ticks = (i32)(cmd * 1000.0);
                if (pwm_ticks > 0) {
                    duty_out.publish(cmd);
                    sample_q.post(cmd as f64);
                }
            }
        }

        fn main() {
            spawn control_loop() @high;
        }
        "#,
    );
    assert!(model.primitives.contains_key("duty_out"));
    assert!(model.primitives.contains_key("sample_q"));

    // 2. Foreign C float type names (`float` -> `f32`, `double` -> `f64`) produce actionable notes!
    let d_float = expect_err_kind("fn main() { float x = 1.0; }", DiagnosticKind::UndefinedSymbol);
    assert!(d_float[0].note.as_deref().unwrap().contains("`f32`"));
    let d_double =
        expect_err_kind("fn main() { double x = 1.0; }", DiagnosticKind::UndefinedSymbol);
    assert!(d_double[0].note.as_deref().unwrap().contains("`f64`"));

    // 3. Reject implicit mixed `f32` + `f64`, implicit `f64 -> f32` narrowing, implicit `int -> float`,
    //    and bitwise ops on floats!
    for bad_float in [
        "fn main() { f32 a = 1.0f32; f64 b = 2.0f64; f64 c = a + b; }",
        "fn main() { f64 b = 2.0f64; f32 a = b; }",
        "fn main() { i32 n = 5; f32 a = n; }",
        "fn main() { f32 a = 1.5f32; i32 n = a; }",
        "fn main() { f32 a = 1.0f32 & 2.0f32; }",
        "fn main() { f64 a = 1.0 << 2.0; }",
        "fn main() { f32 a = ~1.0f32; }",
    ] {
        expect_err_kind(bad_float, DiagnosticKind::TypeMismatch);
    }
}

#[test]
fn wave2_monomorphize_generic_structs_fns_inference_mangling_and_attrs() {
    let lib_src = r#"
        const usize DEFAULT_CAP = 8;

        struct Pair<A, B> {
            A first;
            B second;
        }

        struct FixedVec<T, usize N> {
            [N]T data;
            [N + 1]u8 tags;
            usize len;
        }

        T clamp<T>(T val, T lo, T hi) {
            if (val < lo) {
                return lo;
            }
            if (val > hi) {
                return hi;
            }
            return val;
        }

        bool vec_push<T, usize N>(FixedVec<T, N> *v, T item) {
            if (v.len >= N) {
                return false;
            }
            v.data[v.len] = item;
            v.tags[v.len] = 1;
            v.len += 1;
            return true;
        }
    "#;

    let app_src = r#"
        fn main() {
            // Explicit generic struct with module `const usize`, pointer type `*u8`, and nested generic struct!
            FixedVec<u32, DEFAULT_CAP> nums = FixedVec<u32, DEFAULT_CAP> {
                data: [0, 0, 0, 0, 0, 0, 0, 0],
                tags: [0, 0, 0, 0, 0, 0, 0, 0, 0],
                len: 0,
            };
            // Call-site inference deducing BOTH `T = u32` and `usize N = 8` through `*FixedVec<u32, 8>`!
            bool ok1 = vec_push(&nums, 42);
            // Turbofish `::<...>` explicit call!
            u32 c_u32 = clamp::<u32>(nums.data[0], 10, 100);
            // Explicit `<f32>` AND inferred `clamp(x, -1.0, 1.0)` with contextual unsuffixed float literals!
            f32 duty = 1.5f32;
            f32 c1 = clamp<f32>(duty, 0.0, 1.0);
            f32 c2 = clamp(duty, -1.0, 1.0);

            // Struct literal inference + pointer mangling `ptr_u8`!
            u8 b = 7;
            Pair<*u8, f32> p = Pair { first: &b, second: c1 + c2 };
            assert(ok1 && c_u32 >= 10 && *p.first == 7);
        }
    "#;

    let lib_prog = zev_parser::parse_str(lib_src).unwrap();
    let app_prog = zev_parser::parse_str(app_src).unwrap();

    // 1. Direct `monomorphize_programs` AST contract check (`@backend` contract):
    let mono_progs = monomorphize_programs(&[lib_prog.clone(), app_prog.clone()])
        .expect("monomorphize_programs should succeed");
    assert_eq!(mono_progs.len(), 2);

    // Every generic template declaration (`!generic_params.is_empty()`) MUST be stripped!
    for p in &mono_progs {
        for item in &p.items {
            match &item.kind {
                ItemKind::Struct(st) => {
                    assert!(
                        st.generic_params.is_empty(),
                        "template struct `{}` was not stripped",
                        st.name.name
                    );
                    for sf in &st.fields {
                        if let TypeKind::Array { len, .. } = &sf.ty.kind {
                            assert!(
                                matches!(len, ArrayLen::Fixed(_)),
                                "field `{}.{}` still has un-evaluated ArrayLen::Expr",
                                st.name.name,
                                sf.name.name
                            );
                        }
                    }
                }
                ItemKind::Fn(f) => assert!(
                    f.generic_params.is_empty(),
                    "template fn `{}` was not stripped",
                    f.name.name
                ),
                _ => {}
            }
        }
    }

    // Verify `FixedVec__u32__8` has concrete `[8]u32`, `[9]u8`, and `@monomorphized("FixedVec<u32, 8>")`!
    let fv_item = mono_progs[0]
        .items
        .iter()
        .find(|it| matches!(&it.kind, ItemKind::Struct(st) if st.name.name == "FixedVec__u32__8"))
        .expect("expected synthesized FixedVec__u32__8 struct");
    assert!(fv_item.attrs.iter().any(|a| matches!(
        &a.kind,
        AttrKind::Custom { name, arg: Some(sig) }
            if name == "monomorphized" && sig == "FixedVec<u32, 8>"
    )));
    if let ItemKind::Struct(fv_st) = &fv_item.kind {
        assert!(matches!(
            fv_st.fields[0].ty.kind,
            TypeKind::Array {
                len: ArrayLen::Fixed(8),
                ..
            }
        ));
        assert!(matches!(
            fv_st.fields[1].ty.kind,
            TypeKind::Array {
                len: ArrayLen::Fixed(9),
                ..
            }
        ));
    }

    // Verify `Pair___Pu8__f32` injective mangling (`*u8` -> `_Pu8`)!
    assert!(mono_progs[0]
        .items
        .iter()
        .any(|it| matches!(&it.kind, ItemKind::Struct(st) if st.name.name == "Pair___Pu8__f32")));

    // 2. Combined `analyze_and_monomorphize_named_programs` & `analyze_and_monomorphize_programs` check:
    let (mono2, model2) = analyze_and_monomorphize_programs(
        &[lib_prog.clone(), app_prog.clone()],
        &AnalyzeOptions::default(),
    )
    .into_result()
    .expect("analyze_and_monomorphize_programs should succeed");
    assert_eq!(mono2.len(), 2);

    let named_out = analyze_and_monomorphize_named_programs(
        &[
            (&lib_prog, Some("lib.zev")),
            (&app_prog, Some("app.zev")),
        ],
        &AnalyzeOptions::default(),
    );
    let (_, model) = named_out
        .into_result()
        .expect("named monomorphized sema should pass");
    assert_eq!(model.monomorphized_instances, model2.monomorphized_instances);

    let expected_instances = vec![
        MonomorphizedInstance {
            kind: MonomorphizedKind::Struct,
            signature: "FixedVec<u32, 8>".to_string(),
            mangled_name: "FixedVec__u32__8".to_string(),
        },
        MonomorphizedInstance {
            kind: MonomorphizedKind::Struct,
            signature: "Pair<*u8, f32>".to_string(),
            mangled_name: "Pair___Pu8__f32".to_string(),
        },
        MonomorphizedInstance {
            kind: MonomorphizedKind::Fn,
            signature: "clamp<f32>".to_string(),
            mangled_name: "clamp__f32".to_string(),
        },
        MonomorphizedInstance {
            kind: MonomorphizedKind::Fn,
            signature: "clamp<u32>".to_string(),
            mangled_name: "clamp__u32".to_string(),
        },
        MonomorphizedInstance {
            kind: MonomorphizedKind::Fn,
            signature: "vec_push<u32, 8>".to_string(),
            mangled_name: "vec_push__u32__8".to_string(),
        },
    ];
    assert_eq!(model.monomorphized_instances, expected_instances);

    // Direct `mangle_type_expr` helper check for `*T`, `[]T`, and `[N]T`.
    let parsed_slice_arr = zev_parser::parse_str("fn f([4]*[]u16 x) {} fn main() {}").unwrap();
    if let ItemKind::Fn(f) = &parsed_slice_arr.items[0].kind {
        let mono_fn_prog = monomorphize_named_programs(&[(&parsed_slice_arr, None)]).programs;
        if let ItemKind::Fn(mf) = &mono_fn_prog[0].items[0].kind {
            assert_eq!(mangle_type_expr(&mf.params[0].ty), "_A4__P_Su16");
        }
        let _ = f;
    }
}

#[test]
fn wave2_monomorphize_sema_invariants_yield_cloning_and_dma_provenance() {
    // 1. Dual-specialized `SoftYielding` generic function + valid local DMA `SliceView<u8>` wrapper!
    let model = expect_ok(
        r#"
        @dma_safe ping_pong<u8, 32> rx_dma;
        watch<f32> filtered_out = 0.0;

        struct SliceView<T> {
            []T buf;
            usize cursor;
        }

        SliceView<T> make_view<T>([]T s) {
            return SliceView<T> { buf: s, cursor: 0 };
        }

        T smooth_step<T>(T cur, T target) {
            checkpoint();
            if (cur < target) {
                return target;
            }
            return cur;
        }

        task worker() {
            f32 acc = 0.0;
            loop {
                []u8 half = rx_dma.wait_chunk();
                SliceView<u8> v = make_view(half);
                acc = smooth_step(acc, (f32)v.buf[0]);
                noyield {
                    f32 fast = smooth_step(acc, 10.0f32);
                    filtered_out.publish(fast);
                }
            }
        }

        fn main() {
            spawn worker() @normal;
        }
        "#,
    );
    assert_eq!(model.fn_effect("smooth_step__f32"), FnEffect::SoftYielding);
    assert!(model.dual_specialized_fns.contains("smooth_step__f32"));

    // 2. Escaping DMA slice wrapped inside monomorphized `SliceView<u8>` to outer scope -> E0008!
    expect_err_kind(
        r#"
        @dma_safe ping_pong<u8, 32> rx_dma;
        struct SliceView<T> { []T buf; }
        task bad() {
            loop {
                SliceView<u8> outer = SliceView<u8> { buf: rx_dma.wait_chunk() };
                {
                    []u8 inner_half = rx_dma.wait_chunk();
                    outer = SliceView<u8> { buf: inner_half };
                }
            }
        }
        fn main() { spawn bad() @normal; }
        "#,
        DiagnosticKind::ScopeDepthEscape,
    );

    // 3. Reading monomorphized `SliceView<u8>` after a second `wait_chunk()` -> E0011 (`StaleDmaLease`)!
    expect_err_kind(
        r#"
        @dma_safe ping_pong<u8, 32> rx_dma;
        struct SliceView<T> { []T buf; }
        SliceView<T> wrap<T>([]T s) { return SliceView<T> { buf: s }; }
        task bad() {
            loop {
                SliceView<u8> v = wrap(rx_dma.wait_chunk());
                []u8 next = rx_dma.wait_chunk();
                u8 stale = v.buf[0] + next[0];
                sys.log_kv("s=", stale);
            }
        }
        fn main() { spawn bad() @normal; }
        "#,
        DiagnosticKind::StaleDmaLease,
    );

    // 4. Returning `&local` from a monomorphized generic function -> E0009 (`ReturnLocalReference`)!
    expect_err_kind(
        r#"
        *T dangle<T>(T v) {
            T local_copy = v;
            return &local_copy;
        }
        fn main() {
            *u32 p = dangle(7u32);
        }
        "#,
        DiagnosticKind::ReturnLocalReference,
    );

    // 5. Hard-blocking generic function called inside `noyield` -> E0003 (`YieldInNoYield`)!
    expect_err_kind(
        r#"
        T nap_and_echo<T>(T x) {
            delay(2ms);
            return x;
        }
        fn main() {
            noyield {
                u32 v = nap_and_echo(5u32);
            }
        }
        "#,
        DiagnosticKind::YieldInNoYield,
    );
}

#[test]
fn wave2_generic_instantiation_errors_e0029() {
    for (label, bad_src) in [
        (
            "wrong struct generic arity",
            r#"
            struct FixedVec<T, usize N> { [N]T data; }
            fn main() { FixedVec<u32> v; }
            "#,
        ),
        (
            "wrong fn generic arity",
            r#"
            T id<T>(T x) { return x; }
            fn main() { u32 y = id<u32, u16>(1); }
            "#,
        ),
        (
            "generic args on non-generic struct",
            r#"
            struct Plain { u32 x; }
            fn main() { Plain<u32> p; }
            "#,
        ),
        (
            "generic args on non-generic fn",
            r#"
            u32 plain_fn(u32 x) { return x; }
            fn main() { u32 y = plain_fn<u32>(1); }
            "#,
        ),
        (
            "const integer in type generic slot",
            r#"
            struct FixedVec<T, usize N> { [N]T data; }
            fn main() { FixedVec<16, u32> v; }
            "#,
        ),
        (
            "module const identifier in type generic slot",
            r#"
            const usize CAP = 8;
            struct FixedVec<T, usize N> { [N]T data; }
            fn main() { FixedVec<CAP, 4> v; }
            "#,
        ),
        (
            "type in const usize generic slot",
            r#"
            struct FixedVec<T, usize N> { [N]T data; }
            fn main() { FixedVec<u32, f32> v; }
            "#,
        ),
        (
            "uninferable type parameter",
            r#"
            T make_zero<T>() { return 0 as T; }
            fn main() { u32 x = make_zero(); }
            "#,
        ),
        (
            "uninferable const usize parameter",
            r#"
            usize get_cap<usize N>() { return N; }
            fn main() { usize c = get_cap(); }
            "#,
        ),
        (
            "conflicting inferred float types (f32 vs f64)",
            r#"
            T pick<T>(T a, T b) { return a; }
            fn main() { f32 x = pick(1.0f32, 2.0f64); }
            "#,
        ),
        (
            "non-constant runtime variable in const usize slot",
            r#"
            struct FixedVec<T, usize N> { [N]T data; }
            fn main() {
                usize runtime_n = 4;
                FixedVec<u8, runtime_n> v;
            }
            "#,
        ),
        (
            "infinite type-growing generic fn instantiation (> 32 depth)",
            r#"
            fn grow_ptr<T>(T x) {
                grow_ptr<*T>(&x);
            }
            fn main() {
                u32 seed = 1;
                grow_ptr(seed);
            }
            "#,
        ),
        (
            "infinite const-growing generic fn instantiation (> 32 depth)",
            r#"
            fn grow_n<usize N>() {
                grow_n<N + 1>();
            }
            fn main() {
                grow_n<0>();
            }
            "#,
        ),
        (
            "infinite type-growing generic struct instantiation (> 32 depth)",
            r#"
            struct BadNode<T> {
                *BadNode<*T> next;
            }
            fn main() {
                BadNode<u8> root;
            }
            "#,
        ),
    ] {
        let diags = expect_err_kind(bad_src, DiagnosticKind::GenericInstantiationError);
        assert!(
            diags
                .iter()
                .any(|d| d.kind == DiagnosticKind::GenericInstantiationError && d.code() == "E0029"),
            "case `{label}` did not emit E0029: {diags:?}"
        );
    }
}

#[test]
fn wave2_zero_heap_format_strings_fmt_buf_and_e0028() {
    // 1. Valid `print(f"...")`, `println(f"...")`, and `fmt_buf(dst, f"...")` across all format
    //    specifiers + exact static-fit boundaries!
    expect_ok(
        r#"
        enum Fault { Ok, Overcurrent }

        struct Telemetry {
            u8 id;
            bool ok;
        }

        struct EmptyMarker {}

        fn format_all([]u8 dyn_tag, *[9]u8 ptr_buf9) {
            u8 b = 255;
            i8 sb = -12;
            u16 rpm = 4200;
            u16be wire_v = 0x1234;
            i32 temp_mc = -250;
            u64 ticks64 = 999999;
            f32 volts = 3.30f32;
            f64 precise = -12.3456;
            Fault f = Overcurrent;
            Telemetry pkt = Telemetry { id: 7, ok: true };
            EmptyMarker em = EmptyMarker {};
            u8 arr3[3] = [1, 2, 3];

            *u8 hdr = "ok";
            print(f"[hdr={hdr}] b={b} sb={sb} wire=0x{wire_v:04X} bits={b:08b} ");
            println(f"t={temp_mc} tk={ticks64:x} v={volts} p0={volts:.0} p3={precise:.3} sci={volts:e} sci4={precise:.4e} f={f} pkt={pkt:?} em={em:?} arr={arr3:?}");

            // Exact worst-case static fit proofs (NO `or` / `catch` required when `worst <= cap`!):
            // (a) `"rpm="` (4) + `u16` (5) = 9 bytes -> fits `[9]u8` and `*[9]u8`!
            u8 buf9[9];
            []u8 s1 = fmt_buf(buf9, f"rpm={rpm}");
            []u8 s2 = fmt_buf(ptr_buf9, f"rpm={rpm}");
            []u8 s3 = fmt_buf(buf9[..], f"rpm={rpm}");

            // (b) Constant sub-slice `big[2..16]` has static capacity 14 -> fits `f32` Default (`14` bytes)!
            u8 big[20];
            []u8 s4 = fmt_buf(big[2..16], f"{volts}");

            // (c) Float `:.0` (`11`), `:.3` (`15`), `f32` `:e` (`11`), `f64` `:e` (`12`)!
            u8 b_f0[11];
            u8 b_f3[15];
            u8 b_se32[11];
            u8 b_se64[12];
            []u8 sf0 = fmt_buf(b_f0, f"{volts:.0}");
            []u8 sf3 = fmt_buf(b_f3, f"{precise:.3}");
            []u8 sse32 = fmt_buf(b_se32, f"{volts:e}");
            []u8 sse64 = fmt_buf(b_se64, f"{precise:e}");

            // (d) Enum `Fault` max variant `"Overcurrent"` = 11 bytes!
            u8 b_enum[11];
            []u8 se = fmt_buf(b_enum, f"{f}");

            // (e) Struct `{pkt:?}`: `"Telemetry { id: "` (16) + `u8` (3) + `", ok: "` (6) + `bool` (5) + `" }"` (2) = 32 bytes!
            u8 b_pkt[32];
            []u8 spkt = fmt_buf(b_pkt, f"{pkt:?}");

            // (f) Dynamic `[]u8` hole or dynamic `[]u8` destination handled via `catch (FmtErr err, []u8 clamped)` or `or`!
            u8 small_dyn[8];
            []u8 dyn_dst = small_dyn[0..8];
            []u8 s_dyn1 = fmt_buf(small_dyn, f"tag={dyn_tag}") catch (FmtErr err, []u8 clamped) {
                if (err == Truncated) {
                    println("clamped!");
                }
                clamped
            };
            []u8 s_dyn2 = fmt_buf(dyn_dst, f"rpm={rpm}") or s1;

            println(s1);
            println(s2);
            println(s3);
            println(s4);
            println(sf0);
            println(sf3);
            println(sse32);
            println(sse64);
            println(se);
            println(spkt);
            println(s_dyn1);
            println(s_dyn2);
        }

        fn main() {
            u8 p9[9];
            u8 tag[2] = [65, 66];
            format_all(tag[0..2], &p9);
        }
        "#,
    );

    // 2. Exact `cap - 1` overflow triggers `E0028` (`FormatBufferTooSmall`) with worst-case vs capacity in message!
    let d_rpm = expect_err_kind(
        r#"
        fn main() {
            u16 rpm = 4200;
            u8 buf[8];
            []u8 s = fmt_buf(buf, f"rpm={rpm}");
        }
        "#,
        DiagnosticKind::FormatBufferTooSmall,
    );
    assert_eq!(d_rpm[0].code(), "E0028");
    assert!(
        d_rpm[0].message.contains("9 bytes") && d_rpm[0].message.contains("8 bytes"),
        "expected 9 vs 8 bytes in E0028 message, got: {}",
        d_rpm[0]
    );

    for (label, overflow_src) in [
        (
            "pointer-to-array *[8]u8 overflow",
            r#"
            fn helper(*[8]u8 dst, u16 rpm) {
                []u8 s = fmt_buf(dst, f"rpm={rpm}");
            }
            fn main() {}
            "#,
        ),
        (
            "full slice buf[..] overflow",
            r#"
            fn main() {
                u8 buf[8];
                u16 rpm = 1;
                []u8 s = fmt_buf(buf[..], f"rpm={rpm}");
            }
            "#,
        ),
        (
            "const sub-slice buf[2..9] (cap 7) overflow",
            r#"
            fn main() {
                u8 buf[16];
                u16 rpm = 1;
                []u8 s = fmt_buf(buf[2..9], f"rpm={rpm}");
            }
            "#,
        ),
        (
            "f32 Default (14 bytes) in [13]u8",
            r#"
            fn main() {
                f32 v = 1.5f32;
                u8 buf[13];
                []u8 s = fmt_buf(buf, f"{v}");
            }
            "#,
        ),
        (
            "f64 Sci (12 bytes) in [11]u8",
            r#"
            fn main() {
                f64 v = 1.5;
                u8 buf[11];
                []u8 s = fmt_buf(buf, f"{v:e}");
            }
            "#,
        ),
        (
            "struct Debug (32 bytes) in [31]u8",
            r#"
            struct Telemetry { u8 id; bool ok; }
            fn main() {
                Telemetry t = Telemetry { id: 1, ok: false };
                u8 buf[31];
                []u8 s = fmt_buf(buf, f"{t:?}");
            }
            "#,
        ),
    ] {
        let diags = expect_err_kind(overflow_src, DiagnosticKind::FormatBufferTooSmall);
        assert!(
            diags.iter().any(|d| d.code() == "E0028"),
            "case `{label}` did not emit E0028: {diags:?}"
        );
    }

    // 3. Unhandled `fmt_buf` into runtime `[]u8` slice or with dynamic `[]u8` hole -> E0016 (`UnhandledFallibleExpr`)!
    expect_err_kind(
        r#"
        fn bad([]u8 dyn_dst, u16 rpm) {
            []u8 s = fmt_buf(dyn_dst, f"rpm={rpm}");
        }
        fn main() {}
        "#,
        DiagnosticKind::UnhandledFallibleExpr,
    );
    expect_err_kind(
        r#"
        fn bad([]u8 dyn_hole) {
            u8 buf[16];
            []u8 s = fmt_buf(buf, f"msg={dyn_hole}");
        }
        fn main() {}
        "#,
        DiagnosticKind::UnhandledFallibleExpr,
    );

    // 4. Returning slice backed by local `[N]u8` stack buffer from `fmt_buf` -> E0009 (`ReturnLocalReference`)!
    expect_err_kind(
        r#"
        []u8 bad_return(u16 rpm) {
            u8 local_buf[16];
            return fmt_buf(local_buf, f"rpm={rpm}");
        }
        fn main() {}
        "#,
        DiagnosticKind::ReturnLocalReference,
    );

    // 5. Standalone `f"..."` outside sink -> E0024 (`InvalidOperation`), and invalid `FormatSpec` -> E0019 (`TypeMismatch`)!
    expect_err_kind(
        r#"
        fn main() {
            u32 x = 5;
            *u8 s = f"x={x}";
        }
        "#,
        DiagnosticKind::InvalidOperation,
    );
    for bad_spec in [
        r#"fn main() { f32 v = 1.5f32; println(f"{v:x}"); }"#,
        r#"fn main() { u32 n = 10; println(f"{n:.2}"); }"#,
        r#"fn main() { u32 n = 10; println(f"{n:e}"); }"#,
        r#"fn main() { bool b = true; println(f"{b:04x}"); }"#,
        r#"struct P { u8 x; } fn main() { P p = P { x: 1 }; println(f"{p}"); }"#,
    ] {
        expect_err_kind(bad_spec, DiagnosticKind::TypeMismatch);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage R2-A Review Regression Tests (H1..H5, H9, M3, M5, M7..M11, M14, Low)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn r2a_h1_static_dst_capacity_soundness_and_m5_fmt_buf_self_aliasing() {
    // 1. H1: Subslice of a dynamic `[]u8` (`tiny[0..10]`) has unknown runtime backing capacity
    //    and MUST NOT be treated as proven static capacity 10!
    expect_err_kind(
        r#"
        fn bad_dyn_subslice([]u8 tiny) {
            []u8 s = fmt_buf(tiny[0..10], f"hello");
            println(s);
        }
        fn main() {}
        "#,
        DiagnosticKind::UnhandledFallibleExpr,
    );

    // 2. H1: Chained subslices (`big[0..2][0..10]`) clamp structurally to capacity 2, not `big`'s 16!
    let d_chained = expect_err_kind(
        r#"
        fn main() {
            u8 big[16];
            []u8 s = fmt_buf(big[0..2][0..10], f"hello");
            println(s);
        }
        "#,
        DiagnosticKind::FormatBufferTooSmall,
    );
    assert!(d_chained[0].message.contains("(5 bytes)"));
    assert!(d_chained[0].message.contains("(2 bytes)"));

    // 3. H1: Multidimensional `[10][2]u8 mat; mat[0][0..10]` resolves `mat[0]` structurally as `[2]u8` (cap 2), not outer len 10!
    let d_mat = expect_err_kind(
        r#"
        fn main() {
            [10][2]u8 mat;
            []u8 s = fmt_buf(mat[0][0..10], f"hello");
            println(s);
        }
        "#,
        DiagnosticKind::FormatBufferTooSmall,
    );
    assert!(d_mat[0].message.contains("(2 bytes)"));

    // 4. M5-defense: Self-aliasing `fmt_buf(dst, f"...{same_buf}...")` rejected with `E0024` (`InvalidOperation`)!
    for bad_alias in [
        r#"
        fn main() {
            u8 buf[16];
            []u8 s = fmt_buf(buf, f"hi");
            s = fmt_buf(buf, f"x{s}") or s;
        }
        "#,
        r#"
        fn main() {
            u8 buf[16];
            []u8 s = fmt_buf(buf, f"x{buf}") or buf[0..1];
            println(s);
        }
        "#,
        r#"
        fn helper([]u8 dst) {
            []u8 sub = dst[0..4];
            []u8 out = fmt_buf(dst, f"x{sub}") or dst;
            println(out);
        }
        fn main() {}
        "#,
    ] {
        let diags = expect_err_kind(bad_alias, DiagnosticKind::InvalidOperation);
        assert!(diags.iter().any(|d| d.code() == "E0024"));
    }

    // 5. Formatting scalar `s.len` or formatting `s` into a distinct buffer `buf2` succeeds cleanly!
    expect_ok(
        r#"
        fn main() {
            u8 buf[32];
            u8 buf2[32];
            []u8 s = fmt_buf(buf, f"hi");
            []u8 s_len = fmt_buf(buf, f"len={s.len}");
            []u8 s2 = fmt_buf(buf2, f"x{s}") or buf2[0..1];
            println(s_len);
            println(s2);
        }
        "#,
    );
}

#[test]
fn r2a_h2_m3_h3_h9_low_format_debug_ptrs_slices_enums_hex_escapes_and_precision() {
    // 1. H2 + H3 + H9 + M3 valid cases:
    //    - `{np:?}` on self-referential `*Node` and `{cstr:?}` on `*u8` bounded at 18 bytes (`0x` + 16 hex)
    //    - `{node:?}` formats `*Node next` field as 18 bytes (`"Node { v: "` 10 + 10 + `", next: "` 8 + 18 + `" }"` 2 = 48)
    //    - `enum Op { A = 1, B = 2 }` worst-case width is `max(1, "Op".len() + 3) = 5` (`"Op(?)"`)
    //    - `f"\xFF"` and `f"{\"\xAA\"}"` count decoded raw bytes (`1`, not `2`)
    //    - Fixed `[2]Mini` debug array allowed and exact-bounded (`"[Mini { x: 255 }, Mini { x: 255 }]"` = 34 bytes)
    //    - Float precision up to `.9` / `.9e` accepted
    expect_ok(
        r#"
        enum Op { A = 1, B = 2 }
        struct Node { u32 v; *Node next; }
        struct Mini { u8 x; }

        fn main() {
            Node n = Node { v: 7, next: 0 as *Node };
            *Node np = &n;
            *u8 cstr = "hi";

            u8 b_ptr[18];
            []u8 sp1 = fmt_buf(b_ptr, f"{np:?}");
            []u8 sp2 = fmt_buf(b_ptr, f"{cstr:?}");

            u8 b_node[48];
            []u8 sn = fmt_buf(b_node, f"{n:?}");

            Op op = Op.A;
            u8 b_op[5];
            []u8 sop = fmt_buf(b_op, f"{op}");

            u8 b_raw1[1];
            []u8 s_hex1 = fmt_buf(b_raw1, f"\xFF");
            []u8 s_hex2 = fmt_buf(b_raw1, f"{"\xAA"}");

            Mini m1 = Mini { x: 1 };
            Mini m2 = Mini { x: 2 };
            [2]Mini arr_m = [m1, m2];
            u8 b_arr_m[34];
            []u8 sam = fmt_buf(b_arr_m, f"{arr_m:?}");

            f64 fv = 1.25;
            u8 b_p9[21];
            []u8 sp9 = fmt_buf(b_p9, f"{fv:.9}");
            u8 b_e9[19];
            []u8 se9 = fmt_buf(b_e9, f"{fv:.9e}");

            println(sp1);
            println(sp2);
            println(sn);
            println(sop);
            println(s_hex1);
            println(s_hex2);
            println(sam);
            println(sp9);
            println(se9);
        }
        "#,
    );

    // 2. Tight `cap - 1` checks for H2 (`17 < 18`, `47 < 48`), H3 (`4 < 5`), and H9 (`1 < 2`)!
    for tight_fail in [
        r#"
        struct Node { u32 v; *Node next; }
        fn main() {
            Node n = Node { v: 1, next: 0 as *Node };
            *Node np = &n;
            u8 b[17];
            []u8 s = fmt_buf(b, f"{np:?}");
        }
        "#,
        r#"
        struct Node { u32 v; *Node next; }
        fn main() {
            Node n = Node { v: 1, next: 0 as *Node };
            u8 b[47];
            []u8 s = fmt_buf(b, f"{n:?}");
        }
        "#,
        r#"
        enum Op { A = 1, B = 2 }
        fn main() {
            Op op = Op.A;
            u8 b[4];
            []u8 s = fmt_buf(b, f"{op}");
        }
        "#,
        r#"
        enum Op { A = 1, B = 2 }
        struct Wrap { Op op; }
        fn main() {
            Wrap w = Wrap { op: Op.A };
            u8 b[17]; // "Wrap { op: " (11) + "Op(?)" (5) + " }" (2) = 18 > 17
            []u8 s = fmt_buf(b, f"{w:?}");
        }
        "#,
        r#"
        fn main() {
            u8 b[1];
            []u8 s = fmt_buf(b, f"\xFF\xAA");
        }
        "#,
    ] {
        expect_err_kind(tight_fail, DiagnosticKind::FormatBufferTooSmall);
    }

    // 3. M3: Dynamic `[]Struct`, `[][]u8`, and `[][4]u8` debug slices rejected with `E0019` (`TypeMismatch`)!
    for bad_dyn_debug in [
        r#"
        struct Mini { u8 x; }
        fn f([]Mini xs) { println(f"{xs:?}"); }
        fn main() {}
        "#,
        r#"
        fn f([][]u8 xs) { println(f"{xs:?}"); }
        fn main() {}
        "#,
        r#"
        fn f([][4]u8 xs) { println(f"{xs:?}"); }
        fn main() {}
        "#,
    ] {
        expect_err_kind(bad_dyn_debug, DiagnosticKind::TypeMismatch);
    }

    // 4. Low: Float precision `> 9` (`:.12` and `:.12e`) rejected with `0..=9` message!
    for bad_prec in [
        r#"fn main() { f32 v = 1.0f32; println(f"{v:.12}"); }"#,
        r#"fn main() { f64 v = 1.0; println(f"{v:.12e}"); }"#,
    ] {
        let diags = expect_err_kind(bad_prec, DiagnosticKind::TypeMismatch);
        assert!(diags.iter().any(|d| d.message.contains("0..=9")));
    }
}

#[test]
fn r2a_h4_h5_injective_mangling_and_monomorphization_cap() {
    assert_eq!(MAX_MONOMORPHIZED_INSTANCES, 1024);

    // 1. H4: Former collision pairs now produce distinct injective mangled names and compile cleanly!
    let src = r#"
        struct x { u8 a; }
        struct y_z { u8 a; }
        struct x_y { u8 a; }
        struct z { u8 a; }
        struct ptr_u8 { u8 a; }
        struct Inner<T> { T v; }
        struct Inner_u8 { u8 v; }
        struct GInner_L_u8_R { u8 v; }
        struct Outer<T> { T item; }

        usize pair_tag<A, B>(A a, B b) {
            return 1;
        }

        usize kind_tag<T>(T v) {
            return 2;
        }

        fn main() {
            x vx = x { a: 1 };
            y_z vyz = y_z { a: 2 };
            x_y vxy = x_y { a: 3 };
            z vz = z { a: 4 };
            usize t1 = pair_tag<x, y_z>(vx, vyz);
            usize t2 = pair_tag<x_y, z>(vxy, vz);

            u8 byte = 7;
            ptr_u8 fake_ptr = ptr_u8 { a: 8 };
            usize k1 = kind_tag<*u8>(&byte);
            usize k2 = kind_tag<ptr_u8>(fake_ptr);

            Outer<Inner<u8>> o1 = Outer<Inner<u8>> { item: Inner<u8> { v: 9 } };
            Outer<Inner_u8> o2 = Outer<Inner_u8> { item: Inner_u8 { v: 10 } };
            Outer<GInner_L_u8_R> o3 = Outer<GInner_L_u8_R> { item: GInner_L_u8_R { v: 11 } };
            sys.log_kv("ok=", (i32)(t1 + t2 + k1 + k2 + (usize)o1.item.v + (usize)o2.item.v + (usize)o3.item.v));
        }
    "#;
    let model = expect_ok(src);
    let mangled: Vec<&str> = model
        .monomorphized_instances
        .iter()
        .map(|m| m.mangled_name.as_str())
        .collect();
    assert!(mangled.contains(&"pair_tag__x__y_0z"), "got: {mangled:?}");
    assert!(mangled.contains(&"pair_tag__x_0y__z"), "got: {mangled:?}");
    assert!(mangled.contains(&"kind_tag___Pu8"), "got: {mangled:?}");
    assert!(mangled.contains(&"kind_tag__ptr_0u8"), "got: {mangled:?}");
    assert!(
        mangled.contains(&"Outer___GInner_L_u8_R_"),
        "got: {mangled:?}"
    );
    assert!(mangled.contains(&"Outer__Inner_0u8"), "got: {mangled:?}");
    assert!(
        mangled.contains(&"Outer__GInner_0L_0u8_0R"),
        "got: {mangled:?}"
    );

    // 2. H5: Exponential 2-branch generic fan-out (`2^11 - 1 > 1024` well below depth 32)
    //    hits `MAX_MONOMORPHIZED_INSTANCES = 1024` and aborts immediately with `E0029`!
    let bomb_src = r#"
        fn bomb<T>(T x) {
            [1]T arr = [x];
            bomb<*T>(&x);
            bomb<[1]T>(arr);
        }
        fn main() {
            u8 seed = 0;
            bomb<u8>(seed);
        }
    "#;
    let bomb_diags = expect_err_kind(bomb_src, DiagnosticKind::GenericInstantiationError);
    assert!(
        bomb_diags
            .iter()
            .any(|d| d.code() == "E0029" && d.message.contains("1024")),
        "expected E0029 cap diagnostic, got: {bomb_diags:?}"
    );
}

#[test]
fn r2a_m7_m8_m9_m10_m11_m14_and_instantiation_notes() {
    // 1. M7 + M8 + M9 + M10 valid end-to-end check:
    //    - M7: `Buf<(usize)((u8)300)>` truncates `300` to `44`; `Buf<(usize)(~((u8)0))>` evaluates to `255`
    //    - M8: `first_elem([1000, 2000])` infers `T = i32` (`first_elem__i32__2`) instead of overflowing `u8`
    //    - M9: `const usize TOTAL = HDR + PAYLOAD;` populates `ConstInfo.int_value == Some(6)` for `buf[0..TOTAL]`
    //    - M10: `switch (x) { N => ... }` substitutes unshadowed `N`, while local `usize N = 99` shadows `N`
    let src = r#"
        const usize HDR = 2;
        const usize PAYLOAD = 4;
        const usize TOTAL = HDR + PAYLOAD;
        const i8 SIGNED_WRAP = (i8)255;
        const usize NINE = (usize)(SIGNED_WRAP + 10);

        struct Buf<usize N> {
            [N]u8 data;
        }

        T first_elem<T, usize N>([N]T arr) {
            return arr[0];
        }

        usize switch_and_shadow<usize N>(usize x) {
            switch (x) {
                N => {
                    usize N = 99;
                    return N;
                }
                _ => {
                    return N;
                }
            }
        }

        fn main() {
            Buf<(usize)((u8)300)> b44;
            Buf<(usize)(~((u8)0))> b255;
            i32 v = first_elem([1000, 2000]);
            usize r = switch_and_shadow<7>(7);

            u8 raw[16];
            []u8 fit = fmt_buf(raw[0..TOTAL], f"abcdef");
            u8 raw9[16];
            []u8 fit9 = fmt_buf(raw9[0..NINE], f"123456789");
            println(fit);
            println(fit9);
            sys.log_kv("v=", v + (i32)r + (i32)b44.data[0] + (i32)b255.data[0]);
        }
    "#;
    let prog = zev_parser::parse_str(src).unwrap();
    let (mono_progs, model) =
        analyze_and_monomorphize_programs(&[prog], &AnalyzeOptions::default())
            .into_result()
            .expect("M7/M8/M9/M10 program should succeed");
    assert_eq!(model.consts["TOTAL"].int_value, Some(6));
    assert_eq!(model.consts["NINE"].int_value, Some(9));
    assert!(
        model
            .monomorphized_instances
            .iter()
            .any(|m| m.mangled_name == "Buf__44"),
        "expected Buf__44, got: {:?}",
        model.monomorphized_instances
    );
    assert!(
        model
            .monomorphized_instances
            .iter()
            .any(|m| m.mangled_name == "Buf__255"),
        "expected Buf__255, got: {:?}",
        model.monomorphized_instances
    );
    assert!(
        model
            .monomorphized_instances
            .iter()
            .any(|m| m.mangled_name == "first_elem__i32__2"),
        "expected first_elem__i32__2, got: {:?}",
        model.monomorphized_instances
    );
    let _ = mono_progs;

    // Also verify `raw[0..TOTAL]` rejects 7 bytes (`7 > 6`) with `E0028`!
    expect_err_kind(
        r#"
        const usize HDR = 2;
        const usize PAYLOAD = 4;
        const usize TOTAL = HDR + PAYLOAD;
        fn main() {
            u8 raw[16];
            []u8 bad = fmt_buf(raw[0..TOTAL], f"abcdefg");
        }
        "#,
        DiagnosticKind::FormatBufferTooSmall,
    );

    // 2. M11: Unsuffixed float literal out of finite `f32` range (`1e40`, `-1e40`) rejected on `f32` coercion,
    //    while `f64` and explicit `(f32)1e40` cast pass!
    expect_ok(
        r#"
        fn main() {
            f64 big64 = 1e40;
            f32 inf32 = (f32)1e40;
            if (big64 > 0.0 && inf32 > 0.0f32) {}
        }
        "#,
    );
    for bad_f32 in [
        r#"fn main() { f32 x = 1e40; }"#,
        r#"fn main() { f32 x = -1e40; }"#,
        r#"fn take(f32 x) {} fn main() { take(1e40); }"#,
        r#"watch<f32> w = 1e40; fn main() {}"#,
        r#"f32 ret_bad() { return 1e40; } fn main() {}"#,
    ] {
        let diags = expect_err_kind(bad_f32, DiagnosticKind::TypeMismatch);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("out of finite 'f32' range")),
            "got: {diags:?}"
        );
    }

    // 3. M14: Multi-file `E0029` carries exact `file` (`Some("app.zev")`) with zero duplicate parameter-type errors!
    let lib_p = zev_parser::parse_str("struct FixedVec<T, usize N> { [N]T data; }").unwrap();
    let app_p = zev_parser::parse_str("fn use_vec(FixedVec<u32> bad_param) {} fn main() {}").unwrap();
    let mf_out = analyze_and_monomorphize_named_programs(
        &[(&lib_p, Some("lib.zev")), (&app_p, Some("app.zev"))],
        &AnalyzeOptions::default(),
    );
    assert_eq!(
        mf_out.diagnostics.len(),
        1,
        "expected exactly 1 non-duplicated E0029 diagnostic, got: {:?}",
        mf_out.diagnostics
    );
    assert_eq!(mf_out.diagnostics[0].code(), "E0029");
    assert_eq!(mf_out.diagnostics[0].file.as_deref(), Some("app.zev"));

    // 4. Low: Multi-file recursive value struct preserves `file == Some("types.zev")`!
    let rec_types = zev_parser::parse_str("struct Cycle { Cycle self_val; }").unwrap();
    let rec_main = zev_parser::parse_str("fn main() {}").unwrap();
    let rec_out = analyze_named_programs(
        &[(&rec_types, Some("types.zev")), (&rec_main, Some("main.zev"))],
        &AnalyzeOptions::default(),
    );
    assert!(
        rec_out
            .diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::InvalidOperation
                && d.file.as_deref() == Some("types.zev")),
        "expected file 'types.zev' on recursive struct error, got: {:?}",
        rec_out.diagnostics
    );

    // 5. Low: Error inside a monomorphized `fn` body carries `instantiated from` call-site note!
    let inst_note_diags = expect_err_kind(
        r#"
        T bad_generic<T>(T x) {
            bool flag = x;
            if (flag) {}
            return x;
        }
        fn main() {
            u32 y = bad_generic((u32)42);
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );
    assert!(
        inst_note_diags.iter().any(|d| d
            .note
            .as_deref()
            .is_some_and(|n| n.contains("instantiated from") && n.contains("bad_generic<u32>"))),
        "expected `instantiated from` note on monomorphized body diagnostic, got: {inst_note_diags:?}"
    );
}

#[test]
fn r4_struct_fnptr_field_call_and_ptr_receiver_validation() {
    // 1. Valid `bf.cb(32)`, `(bf.cb)(32)`, and `pbf.cb(32)` through generic `BoxFn<fn(u32) -> u32>`!
    expect_ok(
        r#"
        struct BoxFn<F> {
            F cb;
        }

        u32 add10(u32 x) {
            return x + 10;
        }

        fn main() {
            BoxFn<fn(u32) -> u32> bf = BoxFn<fn(u32) -> u32> { cb: add10 };
            *BoxFn<fn(u32) -> u32> pbf = &bf;
            u32 r1 = bf.cb(32);
            u32 r2 = (bf.cb)(32);
            u32 r3 = pbf.cb(32);
            sys.log_kv("sum=", (i32)(r1 + r2 + r3));
        }
        "#,
    );

    // 2. Arity mismatch and argument type mismatch on `bf.cb(...)` report `E0019` (`TypeMismatch`)!
    for bad_call in [
        r#"
        struct BoxFn<F> { F cb; }
        u32 add10(u32 x) { return x + 10; }
        fn main() {
            BoxFn<fn(u32) -> u32> bf = BoxFn<fn(u32) -> u32> { cb: add10 };
            u32 r = bf.cb(1, 2);
        }
        "#,
        r#"
        struct BoxFn<F> { F cb; }
        u32 add10(u32 x) { return x + 10; }
        fn main() {
            BoxFn<fn(u32) -> u32> bf = BoxFn<fn(u32) -> u32> { cb: add10 };
            u32 r = bf.cb(true);
        }
        "#,
    ] {
        expect_err_kind(bad_call, DiagnosticKind::TypeMismatch);
    }

    // 3. Calling a non-`FnPtr` field or non-existent method on a struct still reports `E0020` (`UndefinedSymbol`)!
    expect_err_kind(
        r#"
        struct Plain { u32 val; }
        fn main() {
            Plain p = Plain { val: 5 };
            p.val(1);
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
}

struct TempWorkspace {
    root: std::path::PathBuf,
}

impl TempWorkspace {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "zev_sema_mod_{}_{}_{}",
            tag,
            std::process::id(),
            id
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    fn write(&self, rel: &str, content: &str) -> std::path::PathBuf {
        let p = self.root.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&p, content).unwrap();
        p
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[test]
fn wave1_module_resolution_diamond_pub_extern_c_and_generics() {
    use zev_sema::IrqBindingArg;

    let ws = TempWorkspace::new("mod_ok");
    // Shared leaf module (`common/types.zev`) imported via `mod.zev` / sibling / include dir.
    ws.write(
        "libs/common/mod.zev",
        r#"
        pub const u32 DEFAULT_BAUD = 115200;
        pub enum Mode { Idle = 0, Active = 1 }
        pub struct Box<T, usize N> { [N]T data; }
        pub T first_item<T, usize N>(Box<T, N> b) { return b.data[0]; }
        pub extern "C" fn hal_tick();
        "#,
    );
    ws.write(
        "app/drivers/uart.zev",
        r#"
        import common as cmn;

        pub extern "C" fn hal_tick();
        extern "C" fn private_c_hook();

        pub mmio UartRegs {
            @rw u32 cr1 @ 0x00;
            @ro u32 isr @ 0x1C;
            @w1c u32 icr @ 0x20;
            @rw [2]u32 brr @ 0x24;
        }

        pub mmio USART1 : UartRegs @ 0x4001_1000;
        pub irq IRQ_UART1 = 37;

        pub isr rx_isr(*UartRegs regs, *watch<u32> out, u32 shift) {
            state u32 hits = 0;
            hits += 1;
            u32 s = regs.isr;
            regs.icr = s;
            out.publish((s >> shift) + hits);
        }

        pub fn configure(*UartRegs regs, u32 baud) {
            private_c_hook();
            cmn.hal_tick();
            regs.cr1 |= 1;
            regs.brr[0] = baud;
        }

        pub task uart_worker(*watch<u32> w) {
            loop {
                u32 v = w.wait_change();
                sys.log_kv("v=", (i32)v);
            }
        }
        "#,
    );
    let main_path = ws.write(
        "app/sub/main.zev",
        r#"
        import common as cmn;
        import drivers.uart as drv;

        pub watch<u32> rx_watch = 0;

        fn main() {
            cmn.Box<u32, 2> b = cmn.Box<u32, 2> { data: [10, 20] };
            u32 first = cmn.first_item<u32, 2>(b);
            drv.configure(drv.USART1, cmn.DEFAULT_BAUD + first);
            drv.IRQ_UART1.attach(drv.rx_isr(drv.USART1, &rx_watch, 1));
            cmn.hal_tick();
            spawn drv.uart_worker(&rx_watch) @high;
        }
        "#,
    );
    // Write fake `Zev.toml` workspace root at `ws.root/app` AND include `ws.root/libs` via `include_dirs`.
    ws.write("app/Zev.toml", "[package]\nname = \"demo\"\n");

    let main_src = std::fs::read_to_string(&main_path).unwrap();
    let main_prog = zev_parser::parse_str_named(&main_src, main_path.to_str().unwrap()).unwrap();
    let mut opts = AnalyzeOptions::default();
    opts.include_dirs.push(ws.root.join("libs"));

    let out = analyze_and_monomorphize_named_programs(
        &[(&main_prog, main_path.to_str())],
        &opts,
    );
    assert!(
        out.diagnostics.is_empty(),
        "unexpected module diagnostics: {:?}",
        out.diagnostics
    );

    // 1. Canonicalized symbols and `display_names`.
    assert!(out.model.mmio_layouts.contains_key("drivers_uart__UartRegs"));
    assert!(out.model.mmio_instances.contains_key("drivers_uart__USART1"));
    assert!(out.model.irqs.contains_key("drivers_uart__IRQ_UART1"));
    assert_eq!(
        out.model
            .display_names
            .get("drivers_uart__USART1")
            .map(String::as_str),
        Some("drivers.uart.USART1")
    );
    // 2. `extern "C"` keeps C symbol unmangled (`hal_tick`, `private_c_hook`).
    assert!(out.model.functions.contains_key("hal_tick"));
    assert!(out.model.functions.contains_key("private_c_hook"));
    // 3. Cross-module IRQ binding & ISR state metadata recorded cleanly.
    assert_eq!(out.model.irq_bindings.len(), 1);
    let binding = &out.model.irq_bindings[0];
    assert_eq!(binding.irq_num, 37);
    assert_eq!(binding.isr_name, "drivers_uart__rx_isr");
    assert_eq!(
        binding.args,
        vec![
            IrqBindingArg::MmioAddr("drivers_uart__USART1".to_string()),
            IrqBindingArg::PrimAddr("rx_watch".to_string()),
            IrqBindingArg::ConstInt {
                ty: zev_sema::Ty::Prim(zev_ast::PrimType::U32),
                value: 1,
            },
        ]
    );
    assert_eq!(out.model.isr_states.len(), 1);
    assert_eq!(out.model.isr_states[0].name, "hits");
    assert_eq!(out.model.spawned_tasks.len(), 1);
    assert_eq!(out.model.spawned_tasks[0].task_name, "drivers_uart__uart_worker");
}

#[test]
fn wave1_module_errors_cycle_private_extern_leak_and_chip_override() {
    // 1. Circular import (`a -> b -> a`) -> `E0030` with file path preserved.
    {
        let ws = TempWorkspace::new("mod_cycle");
        let a_path = ws.write("a.zev", "import b;\nfn main() {}\n");
        ws.write("b.zev", "import a;\npub const u32 X = 1;\n");
        let a_prog = zev_parser::parse_file(&a_path).unwrap();
        let out = analyze_and_monomorphize_named_programs(
            &[(&a_prog, a_path.to_str())],
            &AnalyzeOptions::default(),
        );
        assert!(
            out.diagnostics
                .iter()
                .any(|d| d.kind == DiagnosticKind::ModuleError && d.message.contains("circular")),
            "expected E0030 cycle error, got: {:?}",
            out.diagnostics
        );
    }

    // 2. Private cross-module symbol + bare un-imported `extern "C"` leak -> `E0030`.
    {
        let ws = TempWorkspace::new("mod_priv");
        ws.write(
            "secret.zev",
            r#"
            struct Hidden { u32 v; }
            extern "C" fn secret_c();
            fn hidden_fn() { secret_c(); }
            pub fn ok() {}
            "#,
        );
        for (idx, main_src) in [
            "import secret as s;\nfn main() { s.Hidden h = s.Hidden { v: 1 }; }\n",
            "import secret as s;\nfn main() { s.hidden_fn(); }\n",
            "import secret as s;\nfn main() { s.secret_c(); }\n",
            "import secret as s;\nfn main() { s.ok(); secret_c(); }\n",
            "import does_not_exist as m;\nfn main() {}\n",
        ]
        .iter()
        .enumerate()
        {
            let main_path = ws.write(&format!("main_{idx}.zev"), main_src);
            let main_prog = zev_parser::parse_file(&main_path).unwrap();
            let out = analyze_and_monomorphize_named_programs(
                &[(&main_prog, main_path.to_str())],
                &AnalyzeOptions::default(),
            );
            assert!(
                out.diagnostics
                    .iter()
                    .any(|d| d.kind == DiagnosticKind::ModuleError),
                "case #{idx} expected E0030 ModuleError, got: {:?}",
                out.diagnostics
            );
        }
    }

    // 3. Real `.zev` `chip.*` module on disk takes priority over legacy `NamespaceKind::Chip` fallback!
    {
        let ws = TempWorkspace::new("chip_override");
        ws.write(
            "chip/myboard.zev",
            r#"
            pub mmio Gpio @ 0x4800_0000 {
                @rw u32 odr @ 0x14;
            }
            pub irq IRQ_EXTI0 = 6;
            "#,
        );
        let main_path = ws.write(
            "main.zev",
            r#"
            import chip.myboard as hw;
            isr exti0() {}
            fn main() {
                hw.Gpio.odr |= 1;
                hw.IRQ_EXTI0.attach(exti0);
            }
            "#,
        );
        let main_prog = zev_parser::parse_file(&main_path).unwrap();
        let out = analyze_and_monomorphize_named_programs(
            &[(&main_prog, main_path.to_str())],
            &AnalyzeOptions::default(),
        );
        assert!(
            out.diagnostics.is_empty(),
            "expected disk chip/*.zev override to succeed, got: {:?}",
            out.diagnostics
        );
        assert_eq!(out.model.irq_bindings.len(), 1);
    }

    // 7. Unknown `chip.<foo>` without a `.zev` file emits E0030 (no LegacyChip fallback).
    expect_err_kind(
        r#"
        import chip.nonexistent_silicon as hw;
        fn main() {}
        "#,
        DiagnosticKind::ModuleError,
    );
}

#[test]
fn wave1_mmio_validation_and_e0031_enforcement() {
    // 1. Valid `mmio` layouts, instances, inline desugaring, `&PERIPH`, `PERIPH -> *Layout`,
    //    `(*ptr).reg`, array registers, `@rw`/`@ro`/`@wo`/`@w1c`.
    let m = expect_ok(
        r#"
        const u32 RCC_BASE = 0x4002_1000;
        const u32 CR1_OFF = 0x00;

        mmio UartRegs {
            @rw u32 cr1 @ CR1_OFF;
            @ro u16 sr @ 0x04;
            @wo u8 dr @ 0x06;
            @w1c u32 icr @ 0x08;
            @rw [2]u32 brr @ 0x0C;
            @wo u32 tx[2] @ 0x14;
        }

        mmio USART1 : UartRegs @ 0x4001_1000;
        mmio Rcc @ RCC_BASE {
            @rw u32 cr @ 0x00;
        }

        void step_uart(*UartRegs r) {
            r.cr1 |= 1;
            (*r).cr1 = 2;
            u16 st = r.sr;
            u32 flags = r.icr;
            r.icr = flags;
            r.dr = 0x41;
            r.brr[0] = 9600;
            r.brr[1] |= 0x10;
            r.tx[1] = 0x55;
            sys.log_kv("st=", (i32)st);
        }

        fn main() {
            *UartRegs p1 = USART1;
            *UartRegs p2 = &USART1;
            step_uart(USART1);
            step_uart(p1);
            step_uart(p2);
            Rcc.cr |= 1;
        }
        "#,
    );
    assert!(m.mmio_layouts.contains_key("UartRegs"));
    assert_eq!(m.mmio_layouts["UartRegs"].byte_size, 0x1C);
    assert!(m.mmio_layouts.contains_key("Rcc__Layout"));
    assert_eq!(m.mmio_instances["Rcc"].base_addr, 0x4002_1000);

    // 2. Reject invalid register widths, misalignment, overlap, duplicate reg names, bad bases,
    //    by-value `mmio` in locals/params/returns/fields/casts, `&regs.reg`, and access qualifier violations (`E0031`).
    for bad_mmio in [
        // Unsupported register types (`u64`, `i32`, `f32`, `*u32`, struct)
        "mmio Bad { @rw u64 r @ 0x00; } fn main() {}",
        "mmio Bad { @rw i32 r @ 0x00; } fn main() {}",
        "mmio Bad { @rw *u32 r @ 0x00; } fn main() {}",
        // Unaligned `u16` and `u32`
        "mmio Bad { @rw u16 r @ 0x01; } fn main() {}",
        "mmio Bad { @rw u32 r @ 0x02; } fn main() {}",
        // Overlapping registers
        "mmio Bad { @rw [2]u32 a @ 0x00; @rw u32 b @ 0x04; } fn main() {}",
        // Duplicate register name
        "mmio Bad { @rw u32 cr @ 0x00; @rw u32 cr @ 0x04; } fn main() {}",
        // Out-of-range base address
        "mmio Regs { @rw u32 a @ 0x00; } mmio P : Regs @ 0x1_0000_0000; fn main() {}",
        // By-value `mmio` local, parameter, return, struct field, array, cast
        "mmio Regs { @rw u32 a @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { Regs r; }",
        "mmio Regs { @rw u32 a @ 0x00; } void take(Regs r) {} fn main() {}",
        "mmio Regs { @rw u32 a @ 0x00; } Regs make() { while (true) {} } fn main() {}",
        "mmio Regs { @rw u32 a @ 0x00; } struct Holder { Regs r; } fn main() {}",
        "mmio Regs { @rw u32 a @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { P as Regs; }",
        // Taking `&regs.icr` or `&regs.arr[0]`
        "mmio Regs { @w1c u32 icr @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { *u32 p = &P.icr; }",
        "mmio Regs { @rw [2]u32 brr @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { *u32 p = &P.brr[0]; }",
        // `@ro` write / RMW forbidden
        "mmio Regs { @ro u32 sr @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { P.sr = 1; }",
        "mmio Regs { @ro u32 sr @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { P.sr |= 1; }",
        // `@wo` read & RMW forbidden
        "mmio Regs { @wo u32 dr @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { u32 x = P.dr; }",
        "mmio Regs { @wo u32 dr @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { P.dr |= 1; }",
        "mmio Regs { @wo [2]u32 dr @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { P.dr[0] |= 1; }",
        // `@w1c` compound RMW forbidden (`|=`, `&=`, `^=`)
        "mmio Regs { @w1c u32 icr @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { P.icr |= 1; }",
        "mmio Regs { @w1c u32 icr @ 0x00; } mmio P : Regs @ 0x4000_0000; fn main() { P.icr &= ~1; }",
    ] {
        expect_err_kind(bad_mmio, DiagnosticKind::MmioViolation);
    }
}

#[test]
fn wave1_irq_boot_tree_attachment_shared_and_e0032_errors() {
    // 1. Valid boot-tree `attach` / `attach_shared` / 0-arg `attach`, boot-helper forwarding,
    //    mutually exclusive `if`/`else`/`switch` branches, and `sys.pend_irq(IRQ)`.
    let m = expect_ok(
        r#"
        mmio UartRegs {
            @rw u32 cr1 @ 0x00;
            @w1c u32 icr @ 0x04;
        }
        mmio USART1 : UartRegs @ 0x4001_1000;
        mmio USART2 : UartRegs @ 0x4000_4400;

        irq IRQ_MODE = 10;
        irq IRQ_SHARED = 11;
        irq IRQ_SIMPLE = 12;

        watch<u32> evt_watch = 0;

        enum BootCfg { Fast, Safe }

        isr fast_isr(*UartRegs regs, u32 tag) {
            regs.icr = tag;
        }
        isr safe_isr(*UartRegs regs, *watch<u32> out) {
            regs.icr = 1;
            out.publish(2);
        }
        isr shared_a(bool flag) {
            if (flag) { sys.log("a"); }
        }
        isr shared_b(u32 id) {
            sys.log_kv("b=", (i32)id);
        }
        isr simple_isr {}

        void bind_uart_irq(*UartRegs regs, u32 tag) {
            IRQ_MODE.attach(fast_isr(regs, tag));
        }

        fn main() {
            BootCfg cfg = BootCfg.Fast;
            switch (cfg) {
                Fast => { bind_uart_irq(USART1, 0xA); }
                Safe => { IRQ_MODE.attach(safe_isr(USART2, &evt_watch)); }
            }
            IRQ_SHARED.attach_shared(shared_a(true));
            IRQ_SHARED.attach_shared(shared_b(2));
            IRQ_SIMPLE.attach(simple_isr);
            sys.pend_irq(IRQ_SIMPLE);
            sys.pend_irq(99);
        }
        "#,
    );
    assert_eq!(m.irqs.len(), 3);
    assert_eq!(m.irq_bindings.len(), 5);
    assert!(m.irq_bindings[0].conditional_branch.is_some());
    assert!(m.irq_bindings[1].conditional_branch.is_some());
    assert_ne!(
        m.irq_bindings[0].conditional_branch,
        m.irq_bindings[1].conditional_branch
    );
    assert!(m.irq_bindings[2].is_shared);
    assert!(m.irq_bindings[3].is_shared);

    // 2. Reject invalid `attach` contexts, duplicate exclusive bindings (including aliased IRQ numbers),
    //    mixing `attach` + `attach_shared`, post-`if` conflict, and non-static/non-const handler args (`E0032`).
    for bad_irq in [
        // Inside `task`
        "irq Q = 1; isr h {} task t() { Q.attach(h); loop { sys.delay(1ms); } } fn main() { spawn t(); }",
        // Inside `isr`
        "irq Q = 1; irq Q2 = 2; isr h {} isr bad { Q2.attach(h); } fn main() { Q.attach(bad); }",
        // Inside `loop` / `while`
        "irq Q = 1; isr h {} fn main() { for (u32 i in 0..2) { Q.attach(h); } }",
        // After supervisor loop in `main()`
        "irq Q = 1; isr h {} fn main() { loop { sys.delay(1ms); } Q.attach(h); }",
        // Duplicate exclusive attach in sequence
        "irq Q = 5; isr h1 {} isr h2 {} fn main() { Q.attach(h1); Q.attach(h2); }",
        // Two `irq` tokens sharing same vector number!
        "irq Q1 = 5; irq Q2 = 5; isr h1 {} isr h2 {} fn main() { Q1.attach(h1); Q2.attach(h2); }",
        // Mixing exclusive `attach` and `attach_shared` even in mutually exclusive branches!
        "irq Q = 5; isr h1 {} isr h2 {} fn main() { bool c = true; if (c) { Q.attach(h1); } else { Q.attach_shared(h2); } }",
        // Non-exclusive fallthrough before second `attach`
        "irq Q = 5; isr h1 {} isr h2 {} fn main() { bool c = true; if (c) { Q.attach(h1); } Q.attach(h2); }",
        // Recursive boot-attaching cycle
        "irq Q = 5; isr h {} fn rec(u32 n) { if (n > 0) { rec(n - 1); } else { Q.attach(h); } } fn main() { rec(1); }",
        // Non-const runtime arg directly in `attach`
        "irq Q = 5; isr h(u32 x) {} fn main() { u32 x = 1; x += 1; Q.attach(h(x)); }",
        // Local stack address passed to `attach`
        "irq Q = 5; isr h(*u32 p) {} fn main() { u32 local = 1; Q.attach(h(&local)); }",
        // Non-const runtime arg forwarded through boot helper!
        "irq Q = 5; isr h(u32 x) {} void setup(u32 x) { Q.attach(h(x)); } u32 dyn_val() { return 7; } fn main() { setup(dyn_val()); }",
        // Helper forwarding param with no boot callers
        "irq Q = 5; isr h(u32 x) {} void dead_setup(u32 x) { Q.attach(h(x)); } fn main() {}",
    ] {
        expect_err_kind(bad_irq, DiagnosticKind::IrqAttachmentError);
    }
}

#[test]
fn wave1_isr_state_and_e0033_escape_checks() {
    // 1. Valid ISR `state` variables (scalar, fixed array, const struct init, local helper use).
    let m = expect_ok(
        r#"
        struct Pair { u32 a; u32 b; }
        const u32 ZERO = 0;

        void bump(*u32 p) {
            *p += 1;
        }

        irq IRQ_TIMER = 15;
        isr tim_isr() {
            state u32 count = ZERO;
            state [4]u32 ring = [0, 0, 0, 0];
            state Pair pair = Pair { a: 1, b: 2 };
            bump(&count);
            ring[count & 3] = pair.a;
        }

        fn main() {
            IRQ_TIMER.attach(tim_isr);
        }
        "#,
    );
    assert_eq!(m.isr_states.len(), 3);

    // 2. Reject invalid `state` declarations and escaping `&state_var` (`E0033`).
    for bad_state in [
        // Outside `isr`: inside `fn` and inside `task`
        "fn helper() { state u32 x = 0; } fn main() { helper(); }",
        "task worker() { state u32 x = 0; loop { sys.delay(1ms); } } fn main() { spawn worker(); }",
        // Missing initializer
        "isr bad { state u32 x; } fn main() {}",
        // Non-constant initializer
        "isr bad(u32 seed) { state u32 x = seed + 1; } fn main() {}",
        // Reference-bearing type (`*u32`, `[]u8`, struct with `*u8`)
        "isr bad { state u32 n = 0; state *u32 p = 0 as *u32; } fn main() {}",
        "struct RefWrap { *u8 p; } isr bad { state RefWrap w = RefWrap { p: 0 as *u8 }; } fn main() {}",
        // Duplicate `state` name within same ISR (even in nested block)
        "isr bad { state u32 x = 0; if (true) { state u32 x = 1; } } fn main() {}",
        // Escaping `&state_var` via outer local variable
        "isr bad { *u32 outer = 0 as *u32; if (true) { state u32 inner = 1; outer = &inner; } } fn main() {}",
        // Escaping `&state_var` through ISR pointer parameter (`**u32`)
        "isr bad(**u32 slot) { state u32 x = 0; *slot = &x; } fn main() {}",
        // Escaping `&state_var` through OS primitive `watch.publish` / `inbox.post`
        "watch<*u32> leak_w; isr bad { state u32 x = 0; leak_w.publish(&x); } fn main() {}",
        "inbox<*u32, 4> leak_q; isr bad { state u32 x = 0; leak_q.post(&x); } fn main() {}",
    ] {
        expect_err_kind(bad_state, DiagnosticKind::InvalidIsrState);
    }
}

#[test]
fn wave1_audit_frontend_regression_tests() {
    // #1: Interprocedural ISR `state` pointer escape through helper fn(s) -> E0033.
    expect_err_kind(
        r#"
        watch<*u32> leak_w;
        void forward(*u32 p) { leak_w.publish(p); }
        void bounce(*u32 p) { forward(p); }
        isr bad() {
            state u32 x = 0;
            bounce(&x);
        }
        fn main() {}
        "#,
        DiagnosticKind::InvalidIsrState,
    );

    // #3: Boot-tree IRQ early-return helper call propagation & diverging branch union -> E0032.
    expect_err_kind(
        r#"
        irq Q0 = 5;
        isr h0 {}
        isr h1 {}
        void helper() {
            if (true) {
                Q0.attach(h0);
                return;
            } else {
                return;
            }
        }
        fn main() {
            helper();
            Q0.attach(h1);
        }
        "#,
        DiagnosticKind::IrqAttachmentError,
    );
    expect_err_kind(
        r#"
        irq Q0 = 5;
        irq Q1 = 6;
        isr h0 {}
        isr h1 {}
        void helper() {
            if (true) {
                Q0.attach(h0);
                return;
            }
            Q1.attach(h1);
        }
        fn main() {
            helper();
            Q0.attach(h1);
        }
        "#,
        DiagnosticKind::IrqAttachmentError,
    );

    // #4: `arr.ptr` carries the exact same local/ISR-state/DMA provenance as `&arr[0]`.
    expect_err_kind(
        r#"
        watch<*u32> leak_w;
        isr bad() {
            state [4]u32 buf = [0, 0, 0, 0];
            leak_w.publish(buf.ptr);
        }
        fn main() {}
        "#,
        DiagnosticKind::InvalidIsrState,
    );
    expect_err_kind(
        r#"
        *u32 dangle() {
            [4]u32 local = [1, 2, 3, 4];
            return local.ptr;
        }
        fn main() { *u32 p = dangle(); }
        "#,
        DiagnosticKind::ReturnLocalReference,
    );

    // #7: Local variables/parameters shadowing top-level `const` must not trigger false const checks
    // or slip runtime shadowed variables into compile-time `state` / `irq` const positions.
    expect_err_kind(
        r#"
        const u32 K = 7;
        isr bad(u32 K) {
            state u32 s = K;
        }
        fn main() {}
        "#,
        DiagnosticKind::InvalidIsrState,
    );
    expect_err_kind(
        r#"
        const u32 C = 5;
        irq Q = 9;
        isr h(u32 val) {}
        fn main() {
            u32 C = 1;
            C += 1;
            Q.attach(h(C));
        }
        "#,
        DiagnosticKind::IrqAttachmentError,
    );

    // #10: MMIO register array OOB + unindexed `.ptr` / slice / rvalue / whole-array write -> E0031.
    for bad_mmio_arr in [
        "mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000; u32 f() { return DEV.r[4]; } fn main() {}",
        "mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000; const i32 N = -1; u32 f() { return DEV.r[N]; } fn main() {}",
        "mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000; fn f() { DEV.r[4] = 1; } fn main() {}",
        "mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000; fn f() { *u32 p = DEV.r.ptr; } fn main() {}",
        "mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000; fn f() { []u32 s = DEV.r[0..2]; } fn main() {}",
        "mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000; fn f() { [4]u32 c = DEV.r; } fn main() {}",
        "mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000; fn f() { [4]u32 c = [1,2,3,4]; DEV.r = c; } fn main() {}",
    ] {
        expect_err_kind(bad_mmio_arr, DiagnosticKind::MmioViolation);
    }
    // Shadowed local `IDX` over out-of-bounds global `const IDX = 99` must succeed!
    expect_ok(
        r#"
        mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000;
        const usize IDX = 99;
        u32 read_at(usize IDX) { return DEV.r[IDX]; }
        fn main() { u32 v = read_at(1); }
        "#,
    );

    // #13: Typed `error` spec cannot contain by-value MMIO layout -> E0031.
    expect_err_kind(
        r#"
        mmio R { @rw u32 cr @ 0x0; }
        fn bad_fn() error R {}
        fn main() {}
        "#,
        DiagnosticKind::MmioViolation,
    );

    // #2 + #8 + #9 + #11: Multi-file module checks (generic `usize` scope in struct, qualified enum/const
    // switch patterns, import-aware generic names, and anti-forgery of private canonical `<mod>__<sym>`).
    let tmp = std::env::temp_dir().join(format!("zev_wave1_audit_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("create temp dir");

    let mod_src = r#"
pub struct Ring<T, usize N> {
    [N]T data;
    usize head;
}
pub enum Kind { Alpha, Beta }
pub const u32 MODE_A = 1;
pub bool take2(bool a, bool b) { return a && b; }
const u32 N = 999;
pub const u32 A = 10;
pub const u32 C = 20;
u32 secret_helper() { return 7; }
"#;
    std::fs::write(tmp.join("sub.zev"), mod_src).expect("write sub.zev");

    // Valid use: `Ring<u8, 4>` inside `sub.zev` where `pub struct Ring<T, usize N>` sees `N`,
    // `sub.Kind.Alpha` & `sub.MODE_A` in `switch`, and `sub.take2(sub.A < b, sub.C > (d + 1))`.
    let main_ok = r#"
import sub;
fn main() {
    sub.Ring<u8, 4> r = sub.Ring<u8, 4> { data: [1, 2, 3, 4], head: 0 };
    sub.Kind k = sub.Kind.Alpha;
    u32 x = 1;
    switch (k) {
        sub.Kind.Alpha => { x = 2; }
        default => { x = 3; }
    }
    switch (x) {
        sub.MODE_A => { x = 10; }
        default => {}
    }
    u32 b = 20;
    u32 d = 10;
    bool ok = sub.take2(sub.A < b, sub.C > (d + 1));
    if (ok && r.head == 0) { x = 11; }
}
"#;
    let main_path = tmp.join("main.zev");
    std::fs::write(&main_path, main_ok).expect("write main.zev");
    let gens = zev_sema::collect_import_aware_generic_names(
        &[(main_ok, Some(main_path.as_path()))],
        &[],
    );
    let parsed_ok = zev_parser::parse_str_named_with_generics(
        main_ok,
        main_path.to_str().unwrap(),
        Some(&gens),
    )
    .expect("parse main_ok");
    let out_ok = analyze_and_monomorphize_named_programs(
        &[(&parsed_ok, main_path.to_str())],
        &AnalyzeOptions::default(),
    );
    assert!(
        out_ok.diagnostics.is_empty(),
        "expected clean multi-module build, got:\n{}",
        out_ok
            .diagnostics
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );

    // #11: Direct forged reference to canonical name `sub__secret_helper` must fail with E0030.
    let main_forge = r#"
import sub;
fn main() {
    u32 v = sub__secret_helper();
}
"#;
    let forge_path = tmp.join("forge.zev");
    std::fs::write(&forge_path, main_forge).expect("write forge.zev");
    let parsed_forge =
        zev_parser::parse_str_named(main_forge, forge_path.to_str().unwrap()).expect("parse forge");
    let out_forge = analyze_and_monomorphize_named_programs(
        &[(&parsed_forge, forge_path.to_str())],
        &AnalyzeOptions::default(),
    );
    assert!(
        out_forge
            .diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::ModuleError),
        "expected E0030 rejecting forged private canonical symbol, got: {:?}",
        out_forge.diagnostics
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn byte_ring_methods_and_compile_time_lease_invalidation_e0011() {
    // 1. Multiple `read_slice()` calls and local `ByteReader` structs stay valid BEFORE `consume()`,
    //    and `grant_write()` lease stays independent from `read_slice()` lease!
    let good = r#"
struct ViewWrap { []u8 s; }
byte_ring<64> rx;

fn check_ok() {
    []u8 a = rx.read_slice();
    []u8 b = rx.read_slice();
    ViewWrap w = ViewWrap { s: a };
    []u8 wr = rx.grant_write(4) catch { return; };
    wr[0] = b[0] + w.s[0];
    rx.commit_write(1);
    rx.consume(b.len);
}

fn main() {
    check_ok();
}
"#;
    let prog_good = zev_parser::parse_str(good).expect("parse good byte_ring");
    analyze(&[prog_good]).expect("valid byte_ring usage before consume/commit must pass");

    // 2. Reading a `read_slice()` after `consume()`, a `wait_read()` slice after `pop()`,
    //    or a `grant_write()` slice after `commit_write()` / second `grant_write()` -> E0011 (`StaleDmaLease`)!
    let bad = r#"
struct ViewWrap { []u8 s; }
byte_ring<64> rx;

task reader(*byte_ring<64> ring) {
    loop {
        []u8 chunk = ring.wait_read();
        ViewWrap w = ViewWrap { s: chunk };
        u8 tmp[4];
        usize n = ring.pop(tmp);
        u8 stale_after_pop = w.s[0];
    }
}

fn bad_seqs() {
    []u8 s1 = rx.read_slice();
    rx.consume(1);
    u8 stale1 = s1[0];

    []u8 g1 = rx.grant_write(4) catch { return; };
    rx.commit_write(2);
    u8 stale_g1 = g1[0];

    []u8 g2 = rx.grant_write(4) catch { return; };
    []u8 g3 = rx.grant_write(2) catch { return; };
    g2[0] = 1;
}

fn main() {
    spawn reader(&rx) @normal;
    bad_seqs();
}
"#;
    let prog_bad = zev_parser::parse_str(bad).expect("parse bad byte_ring");
    let diags = analyze(&[prog_bad]).expect_err("stale byte_ring slices must be rejected");
    let stale_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.kind == DiagnosticKind::StaleDmaLease)
        .collect();
    assert_eq!(
        stale_diags.len(),
        4,
        "expected 4 E0011 StaleDmaLease errors (after pop, consume, commit_write, grant_write), got:\n{}",
        diags
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn verifies_unsized_views_and_stale_lease_tracking() {
    let good = r#"
byte_ring<64> r1;
byte_ring<128> r2;
inbox<u32, 4> q1;
inbox<u32, 8> q2;

fn push_msg(*byte_ring rx, []u8 msg) {
    usize n = rx.push(msg);
    bool b = rx.push_byte(65, boost: @high);
    usize c = rx.cap();
}

fn read_msg(*byte_ring rx) {
    []u8 s = rx.read_slice();
    if (s.len > 0) {
        u8 first = s[0];
        rx.consume(1);
    }
}

fn send_and_cap(*inbox<u32> q, u32 v) {
    usize total = q.cap();
    usize cur = q.len();
    bool e = q.is_empty();
    bool f = q.is_full();
    bool ok = q.try_send(v);
    q.post(v, boost: @normal);
}

task consumer(*inbox<u32> q, *byte_ring rx) {
    loop {
        u32 v = q.recv();
        []u8 slice = rx.wait_read();
        rx.consume(slice.len);
        q.send(v + 1);
    }
}

fn main() {
    push_msg(&r1, "hi");
    push_msg(&r2, "yo");
    read_msg(&r1);
    send_and_cap(&q1, 1);
    send_and_cap(&q2, 2);
    spawn consumer(&q1, &r1) @normal;
    spawn consumer(&q2, &r2) @normal;
}
"#;
    let prog = zev_parser::parse_str(good).expect("parse good unsized views");
    let model = analyze(&[prog]).expect("good unsized views should pass");
    assert_eq!(model.spawned_tasks.len(), 2);

    // Reject bare unsized `byte_ring` and `inbox<T>` in top-level, struct, param, return, and local.
    for bad in [
        "byte_ring r; fn main() {}",
        "inbox<u8> q; fn main() {}",
        "struct Bad { byte_ring r; } fn main() {}",
        "struct Bad { inbox<u8> q; } fn main() {}",
        "fn f(byte_ring r) {} fn main() {}",
        "inbox<u8> f() { fn_unreachable(); } fn fn_unreachable() {}",
        "fn main() { byte_ring r; }",
        "fn main() { inbox<u8> q; }",
    ] {
        let diags = expect_err_kind(bad, DiagnosticKind::InvalidOperation);
        assert!(!diags.is_empty());
    }

    // Reject mismatched element type on `*inbox<T, N>` -> `*inbox<U>`.
    let mismatch = r#"
inbox<u16, 4> q16;
fn take(*inbox<u32> q) {}
fn main() { take(&q16); }
"#;
    expect_err_kind(mismatch, DiagnosticKind::TypeMismatch);

    // Enforce E0011 StaleDmaLease on `*byte_ring` views (`read_slice`, `wait_read`, `grant_write`).
    let stale_view = r#"
byte_ring<64> ring;
task t(*byte_ring rx) {
    loop {
        []u8 w = rx.wait_read();
        rx.consume(1);
        u8 stale_w = w[0];

        []u8 s = rx.read_slice();
        rx.consume(s.len);
        u8 stale_s = s[0];

        []u8 g = rx.grant_write(4) catch { continue; };
        rx.commit_write(2);
        u8 stale_g = g[0];
    }
}
fn main() { spawn t(&ring) @normal; }
"#;
    let p_stale = zev_parser::parse_str(stale_view).expect("parse stale view");
    let diags = analyze(&[p_stale]).expect_err("stale view leases must fail");
    let stale_count = diags
        .iter()
        .filter(|d| d.kind == DiagnosticKind::StaleDmaLease)
        .count();
    assert_eq!(stale_count, 3, "expected 3 StaleDmaLease errors on *byte_ring");
}

#[test]
fn verifies_subsystem_structs_rules_and_isr_isolation() {
    let good = r#"
irq IRQ_UART = 17;

struct Channel {
    byte_ring<64> rx;
    inbox<u8, 8> tx;
}

struct DualUart {
    Channel ch0;
    Channel ch1;
    signal ready;
}

DualUart uarts;

isr uart_rx_isr(*DualUart d) {
    d.ch0.rx.push_byte(42, boost: @high);
    d.ch1.tx.post(7);
    d.ready.notify();
}

fn check_independent_fields(*DualUart d) {
    []u8 s0 = d.ch0.rx.read_slice();
    d.ch1.rx.consume(1); // different field must NOT invalidate ch0.rx
    if (s0.len > 0) {
        u8 b = s0[0];
    }
    d.ch0.rx.consume(1);
}

task reader(*byte_ring rx) {
    loop {
        []u8 chunk = rx.wait_read();
        rx.consume(chunk.len);
    }
}

fn boot_setup(*DualUart d) {
    IRQ_UART.attach(uart_rx_isr(&uarts));
    spawn reader(&d.ch0.rx) @high;
    spawn reader(&d.ch1.rx) @normal;
}

fn main() {
    check_independent_fields(&uarts);
    boot_setup(&uarts);
}
"#;
    let prog = zev_parser::parse_str(good).expect("parse good subsystem");
    let model = analyze(&[prog]).expect("good subsystem must pass");
    assert!(model.structs["Channel"].is_subsystem);
    assert!(model.structs["DualUart"].is_subsystem);
    assert_eq!(model.subsystems["uarts"].struct_name, "DualUart");
    assert_eq!(model.irq_bindings.len(), 1);
    assert!(matches!(
        &model.irq_bindings[0].args[0],
        zev_sema::IrqBindingArg::SubsystemPath { root, root_struct, field_chain }
            if root == "uarts" && root_struct == "DualUart" && field_chain.is_empty()
    ));
    assert_eq!(
        model.spawned_tasks[0].endpoint_bindings,
        vec![Some("uarts.ch0.rx".to_string())]
    );
    assert_eq!(
        model.spawned_tasks[1].endpoint_bindings,
        vec![Some("uarts.ch1.rx".to_string())]
    );

    // Reject rule (a): mixed raw scalar/array fields and packed subsystem structs.
    for bad in [
        "struct Mixed { byte_ring<16> rx; u32 baud; } fn main() {}",
        "struct NestedMixed { Channel c; u8 tag; } struct Channel { signal s; } fn main() {}",
        "packed struct PackedSub { byte_ring<16> rx; } fn main() {}",
    ] {
        expect_err_kind(bad, DiagnosticKind::InvalidOperation);
    }

    // Reject rule (b): by-value locals, params, returns, struct literals, from_bytes, and init exprs.
    let sub_def = "struct S { byte_ring<16> rx; }";
    for bad in [
        format!("{sub_def} fn f(S x) {{}} fn main() {{}}"),
        format!("{sub_def} S f() {{}} fn main() {{}}"),
        format!("{sub_def} fn main() {{ S local_s; }}"),
        format!("{sub_def} byte_ring<16> tmp; fn main() {{ S s = S {{ rx: tmp }}; }}"),
        format!("{sub_def} fn main() {{ [4]u8 b = [0, 0, 0, 0]; S s = S.from_bytes(b[0..4]) catch {{ return; }}; }}"),
        format!("{sub_def} S s1 = 0; fn main() {{}}"),
    ] {
        expect_err_kind(&bad, DiagnosticKind::InvalidOperation);
    }

    // Reject rule (d): direct global subsystem instance access inside ISR -> E0002.
    let bad_isr = r#"
irq IRQ0 = 5;
struct Sub { byte_ring<16> rx; }
Sub inst;
isr bad_handler() {
    inst.rx.push_byte(1);
}
fn main() {
    IRQ0.attach(bad_handler());
}
"#;
    expect_err_kind(bad_isr, DiagnosticKind::GlobalIsolationViolation);
}

#[test]
fn verifies_fmt_ring_typing_boost_and_e0028_overflow() {
    let good = r#"
byte_ring<32> tx32;

fn log_view(*byte_ring tx, u32 code) {
    usize n = fmt_ring(tx, f"err={code}", boost: @high) catch (FmtErr e, usize w) {
        return;
    };
}

fn main() {
    u32 x = 42;
    usize a = fmt_ring(&tx32, f"val={x}") or 0;
    usize b = fmt_ring(&tx32, "raw") or 0;
    log_view(&tx32, 7);
}
"#;
    let prog = zev_parser::parse_str(good).expect("parse good fmt_ring");
    let _ = analyze(&[prog]).expect("good fmt_ring should pass");

    // E0028 when static minimum formatted bytes exceed sized `*byte_ring<N>`.
    let bad_overflow = r#"
byte_ring<4> tiny;
fn main() {
    u32 x = 1;
    usize w = fmt_ring(&tiny, f"val={x}") or 0;
}
"#;
    expect_err_kind(bad_overflow, DiagnosticKind::FormatBufferTooSmall);

    // Must handle fallible result via `or`/`catch`.
    let bad_unhandled = r#"
byte_ring<32> tx;
fn main() {
    usize w = fmt_ring(&tx, f"hi");
}
"#;
    expect_err_kind(bad_unhandled, DiagnosticKind::UnhandledFallibleExpr);
}



