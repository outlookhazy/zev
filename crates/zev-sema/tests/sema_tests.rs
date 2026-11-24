use std::path::Path;

use zev_ast::{ArrayLen, AttrKind, ItemKind, Priority, TypeKind};
use zev_sema::{
    analyze, analyze_and_monomorphize_named_programs, analyze_and_monomorphize_programs,
    analyze_named_programs, analyze_program, mangle_type_expr, monomorphize_named_programs,
    monomorphize_programs, AnalyzeOptions, Diagnostic, DiagnosticKind, FnEffect,
    MonomorphizedInstance, MonomorphizedKind, PrimitiveKind, SemanticModel,
    MAX_MONOMORPHIZED_INSTANCES,
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

        fn fast_hook(u32 sector) -> i32 {
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

        fn bad_yielding_hook(u32 sector) -> i32 {
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
        extern "C" fn mount_fs(*BlockOps ops);

        fn bad_read(u32 lba) -> i32 {
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
    assert_eq!(
        isolation_count, 2,
        "expected isolation errors in both EXTI0 and worker"
    );
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
        @dma_safe pool<[32]u8, 4>   spi_pool;

        fn peek_first(*ByteReader r) -> u8 {
            if (r.buf.len == 0) {
                return 0;
            }
            return r.buf[0];
        }

        fn make_sub_reader([]u8 chunk) -> ByteReader {
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

                Lease<[32]u8> spi_l = spi_pool.alloc();
                defer spi_l.release();
                *[32]u8 spi_buf = spi_l.get();
                u8 s0 = (*spi_buf)[0];
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
        fn bad_ret_ptr() -> *i32 {
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
        fn bad_ret_slice() -> []u8 {
            [8]u8 tmp = [1, 2, 3, 4, 5, 6, 7, 8];
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

        fn bad_ret_lease() -> ByteReader {
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
        task worker(*i32 p) {
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

        fn wrap([]u8 b) -> ByteReader {
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
        @dma_safe ping_pong<u8, 32> dma_lease;

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
        fn fact(u32 n) -> u32 {
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

        fn read_packet([]u8 dst, bool fail_halfway) -> usize error IoErr {
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

            [8]u8 buf = [0, 0, 0, 0, 0, 0, 0, 0];
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
        fn fallible() -> u16 error Err {
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
        fn fallible() -> u16 error Err {
            error;
        }
        fn main() {}
        "#,
        DiagnosticKind::ErrorSpecMismatch,
    );

    // 2-param `catch (Err e, u16 p)` on untyped bare `error` fn must be rejected
    expect_err_kind(
        r#"
        fn bare_fail() -> u16 error {
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
        fn forgot_return(bool flag) -> u16 {
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
        dma_diags
            .iter()
            .any(|d| d.kind == DiagnosticKind::StaleDmaLease),
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
        fn read_some([]u8 b) -> usize error IoErr {
            error ShortRead, 2;
        }
        fn main() {
            [8]u8 buf = [0, 1, 2, 3, 4, 5, 6, 7];
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
fn doc_topic_and_doc_command_cover_all_e0001_through_e0034() {
    assert_eq!(DiagnosticKind::ALL.len(), 34);

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
        ("E0034", "dma"),
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
    assert!(d_inbox[0]
        .note
        .as_deref()
        .unwrap()
        .contains("valid methods on `inbox<Ev, 8>`"));

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
    assert!(
        d_watch_get[0].message.contains(".read()"),
        "got: {}",
        d_watch_get[0]
    );

    let d_watch_set = expect_err_kind(
        r#"
        watch<u16> rpm = 0;
        fn main() {
            rpm.set(1200);
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(
        d_watch_set[0].message.contains(".publish("),
        "got: {}",
        d_watch_set[0]
    );

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
    assert!(
        d_pp[0].message.contains(".wait_chunk()"),
        "got: {}",
        d_pp[0]
    );

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
    assert!(d_field
        .iter()
        .any(|d| d.note.as_deref().unwrap().contains(".len")));
    assert!(d_field
        .iter()
        .any(|d| d.note.as_deref().unwrap().contains("CmdFrame.setpoint")));

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
    assert!(d_glob[0]
        .note
        .as_deref()
        .unwrap()
        .contains("watch<u32> fault_flags"));
    assert!(d_glob[0]
        .note
        .as_deref()
        .unwrap()
        .contains("inbox<u32, 8> fault_flags"));
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

        fn decode(u8 raw) -> DecodedPacket error PacketErr {
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
        fn decode_u32(u8 raw) -> u32 error PacketErr {
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
        fn decode(u8 raw) -> DecodedPacket error PacketErr {
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
        fn decode_u32(u8 raw) -> u32 error PacketErr {
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
        fn compute() -> u32 {
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
    assert_eq!(
        e0025_count, 1,
        "expected exactly 1 E0025 for dead tail, got: {diags:?}"
    );
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
    assert_eq!(
        d_ret.len(),
        1,
        "expected single E0026 on `return;`, got: {d_ret:?}"
    );

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
        const [4]u8 INIT_SEQ = [1, 2, 3, 4];
        watch<u16> m1 = 0;
        watch<u16> m2 = 0;
        signal usb_irq_sig;

        task bus_worker(*watch<u16> rpm, []u8 init_bytes, u8 bus_id) {
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

        fn setup_bus(*watch<u16> rpm, []u8 seq, u8 id) {
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
        task ptr_task(*u16 p) {
            loop every 1ms {
                *p += 1;
            }
        }
        fn boot_wire(*u16 p) {
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
                    sys.log_kv("sum=", (v1 + v2) as i32);
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

    // Reject `loop every 0`, negative constants/literals, and > MAX_DURATION_TICKS constants -> E0013
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
    expect_err_kind(
        r#"
        task bad_neg_lit() {
            loop every -5 {
                checkpoint();
            }
        }
        fn main() { spawn bad_neg_lit(); }
        "#,
        DiagnosticKind::InvalidDurationResolution,
    );
    expect_err_kind(
        r#"
        const i32 NEG_PERIOD = -10;
        task bad_neg_const() {
            loop every NEG_PERIOD {
                checkpoint();
            }
        }
        fn main() { spawn bad_neg_const(); }
        "#,
        DiagnosticKind::InvalidDurationResolution,
    );
    expect_err_kind(
        r#"
        const u64 HUGE_PERIOD = 3000000000;
        task bad_huge_const() {
            loop every HUGE_PERIOD {
                checkpoint();
            }
        }
        fn main() { spawn bad_huge_const(); }
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
        fn pure_mix(u32 a, u32 b) -> u32 {
            return a ^ b;
        }

        fn sha_round(u32 acc) -> u32 {
            checkpoint();
            return pure_mix(acc, 0x5A5A);
        }

        fn sha_block(u32 seed) -> u32 {
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

        fn fast_only_soft(u32 x) -> u32 {
            checkpoint();
            return x + 1;
        }

        fn atomic_wrapper(u32 x) -> u32 {
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
            sys.log_kv("v=", v as i32);
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
        watch<u32> isr_sum = 0;
        watch<u32> last_sum = 0;

        fn tx_frame([]u8 payload, bool fail) -> u32 error SpiErr {
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
            [4]u8 lut = [1, 2, 3, 4];
            u32 s = 0;
            for (u8 v in lut) {
                s += v;
            }
            isr_sum.publish(s);
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

        fn read_adc_volts(u16 raw, bool ok) -> f32 error SensorErr {
            if (!ok) {
                error Disconnected;
            }
            return (raw as f32) * 3.3 / 4095.0;
        }

        fn step_pid(*PidState st, f32 target, f32 measured) -> f32 {
            f32 err = target - measured;
            st.integral += err * 0.01;
            st.integral -= 0.0;
            st.integral *= 0.99;
            st.integral /= 1.0;
            st.integral %= 10.0f32;

            f64 wide = (err as f64) + KI * 2.5e1;
            st.accum64 = wide % 1.5;

            if (err >= -0.05 && err <= 0.05 && wide != 0.0 && !(err == 99.0)) {
                return -0.0;
            }
            return KP * err + st.integral + (st.accum64 as f32);
        }

        task control_loop() {
            PidState st = PidState { integral: 0.0, accum64: 0.0 };
            [3]f32 weights = [0.25, 0.5, 0.25];
            loop every 2ms {
                f32 v1 = read_adc_volts(2048, true) or 0.0;
                f32 v2 = read_adc_volts(0, false) catch (SensorErr e) {
                    1.65
                };
                f32 cmd = step_pid(&st, v1 + v2, weights[0]);
                i32 pwm_ticks = (cmd * 1000.0) as i32;
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
    let d_float = expect_err_kind(
        "fn main() { float x = 1.0; }",
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(d_float[0].note.as_deref().unwrap().contains("`f32`"));
    let d_double = expect_err_kind(
        "fn main() { double x = 1.0; }",
        DiagnosticKind::UndefinedSymbol,
    );
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

        fn clamp<T>(T val, T lo, T hi) -> T {
            if (val < lo) {
                return lo;
            }
            if (val > hi) {
                return hi;
            }
            return val;
        }

        fn vec_push<T, usize N>(*FixedVec<T, N> v, T item) -> bool {
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
            // Explicit `<...>` call!
            u32 c_u32 = clamp<u32>(nums.data[0], 10, 100);
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
        &[(&lib_prog, Some("lib.zev")), (&app_prog, Some("app.zev"))],
        &AnalyzeOptions::default(),
    );
    let (_, model) = named_out
        .into_result()
        .expect("named monomorphized sema should pass");
    assert_eq!(
        model.monomorphized_instances,
        model2.monomorphized_instances
    );

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

        fn make_view<T>([]T s) -> SliceView<T> {
            return SliceView<T> { buf: s, cursor: 0 };
        }

        fn smooth_step<T>(T cur, T target) -> T {
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
                acc = smooth_step(acc, v.buf[0] as f32);
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
        fn wrap<T>([]T s) -> SliceView<T> { return SliceView<T> { buf: s }; }
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
        fn dangle<T>(T v) -> *T {
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
        fn nap_and_echo<T>(T x) -> T {
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
            fn id<T>(T x) -> T { return x; }
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
            fn plain_fn(u32 x) -> u32 { return x; }
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
            fn make_zero<T>() -> T { return 0 as T; }
            fn main() { u32 x = make_zero(); }
            "#,
        ),
        (
            "uninferable const usize parameter",
            r#"
            fn get_cap<usize N>() -> usize { return N; }
            fn main() { usize c = get_cap(); }
            "#,
        ),
        (
            "conflicting inferred float types (f32 vs f64)",
            r#"
            fn pick<T>(T a, T b) -> T { return a; }
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
                grow_n<(N + 1)>();
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
fn local_bindings_shadow_same_named_generic_fn_templates() {
    // Locals, parameters, `for` bindings, and `fn(...)` pointer locals shadow a generic `fn`
    // template of the same name instead of tripping `E0029` in the monomorphizer.
    let model = expect_ok(
        r#"
        fn lim<usize N>(u32 x) -> u32 { return x + (N as u32); }
        fn inc(u32 x) -> u32 { return x + 1u32; }
        fn use_param(u32 lim) -> u32 { return lim * 2u32; }
        fn via_ptr() -> u32 {
            fn(u32) -> u32 lim = inc;
            return lim(3u32);
        }
        fn main() {
            u32 a = lim<4>(1u32);
            u32 lim = 7u32;
            u32 y = lim + 1u32;
            bool below = lim < 9u32;
            for (u32 lim in 0..3) { y += lim; }
            y += use_param(a) + via_ptr();
        }
        "#,
    );
    assert!(
        model
            .monomorphized_instances
            .iter()
            .any(|i| i.signature == "lim<4>"),
        "template call before the shadowing local must still instantiate: {:?}",
        model.monomorphized_instances
    );

    // A bare reference that really does resolve to the template is still rejected.
    expect_err_kind(
        r#"
        fn lim<usize N>(u32 x) -> u32 { return x + (N as u32); }
        fn main() { u32 y = lim + 1u32; }
        "#,
        DiagnosticKind::GenericInstantiationError,
    );
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
            [3]u8 arr3 = [1, 2, 3];

            *u8 hdr = "ok";
            print(f"[hdr={hdr}] b={b} sb={sb} wire=0x{wire_v:04X} bits={b:08b} ");
            println(f"t={temp_mc} tk={ticks64:x} v={volts} p0={volts:.0} p3={precise:.3} sci={volts:e} sci4={precise:.4e} f={f} pkt={pkt:?} em={em:?} arr={arr3:?}");

            // Exact worst-case static fit proofs (NO `or` / `catch` required when `worst <= cap`!):
            // (a) `"rpm="` (4) + `u16` (5) = 9 bytes -> fits `[9]u8` and `*[9]u8`!
            [9]u8 buf9;
            []u8 s1 = fmt_buf(buf9, f"rpm={rpm}");
            []u8 s2 = fmt_buf(ptr_buf9, f"rpm={rpm}");
            []u8 s3 = fmt_buf(buf9[..], f"rpm={rpm}");

            // (b) Constant sub-slice `big[2..16]` has static capacity 14 -> fits `f32` Default (`14` bytes)!
            [20]u8 big;
            []u8 s4 = fmt_buf(big[2..16], f"{volts}");

            // (c) Float `:.0` (`11`), `:.3` (`15`), `f32` `:e` (`11`), `f64` `:e` (`12`)!
            [11]u8 b_f0;
            [15]u8 b_f3;
            [11]u8 b_se32;
            [12]u8 b_se64;
            []u8 sf0 = fmt_buf(b_f0, f"{volts:.0}");
            []u8 sf3 = fmt_buf(b_f3, f"{precise:.3}");
            []u8 sse32 = fmt_buf(b_se32, f"{volts:e}");
            []u8 sse64 = fmt_buf(b_se64, f"{precise:e}");

            // (d) Enum `Fault` max variant `"Overcurrent"` = 11 bytes!
            [11]u8 b_enum;
            []u8 se = fmt_buf(b_enum, f"{f}");

            // (e) Struct `{pkt:?}`: `"Telemetry { id: "` (16) + `u8` (3) + `", ok: "` (6) + `bool` (5) + `" }"` (2) = 32 bytes!
            [32]u8 b_pkt;
            []u8 spkt = fmt_buf(b_pkt, f"{pkt:?}");

            // (f) Dynamic `[]u8` hole or dynamic `[]u8` destination handled via `catch (FmtErr err, []u8 clamped)` or `or`!
            [8]u8 small_dyn;
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
            [9]u8 p9;
            [2]u8 tag = [65, 66];
            format_all(tag[0..2], &p9);
        }
        "#,
    );

    // 2. Exact `cap - 1` overflow triggers `E0028` (`FormatBufferTooSmall`) with worst-case vs capacity in message!
    let d_rpm = expect_err_kind(
        r#"
        fn main() {
            u16 rpm = 4200;
            [8]u8 buf;
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
                [8]u8 buf;
                u16 rpm = 1;
                []u8 s = fmt_buf(buf[..], f"rpm={rpm}");
            }
            "#,
        ),
        (
            "const sub-slice buf[2..9] (cap 7) overflow",
            r#"
            fn main() {
                [16]u8 buf;
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
                [13]u8 buf;
                []u8 s = fmt_buf(buf, f"{v}");
            }
            "#,
        ),
        (
            "f64 Sci (12 bytes) in [11]u8",
            r#"
            fn main() {
                f64 v = 1.5;
                [11]u8 buf;
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
                [31]u8 buf;
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
            [16]u8 buf;
            []u8 s = fmt_buf(buf, f"msg={dyn_hole}");
        }
        fn main() {}
        "#,
        DiagnosticKind::UnhandledFallibleExpr,
    );

    // 4. Returning slice backed by local `[N]u8` stack buffer from `fmt_buf` -> E0009 (`ReturnLocalReference`)!
    expect_err_kind(
        r#"
        fn bad_return(u16 rpm) -> []u8 {
            [16]u8 local_buf;
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
            [16]u8 big;
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
            [16]u8 buf;
            []u8 s = fmt_buf(buf, f"hi");
            s = fmt_buf(buf, f"x{s}") or s;
        }
        "#,
        r#"
        fn main() {
            [16]u8 buf;
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
            [32]u8 buf;
            [32]u8 buf2;
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

            [18]u8 b_ptr;
            []u8 sp1 = fmt_buf(b_ptr, f"{np:?}");
            []u8 sp2 = fmt_buf(b_ptr, f"{cstr:?}");

            [48]u8 b_node;
            []u8 sn = fmt_buf(b_node, f"{n:?}");

            Op op = Op.A;
            [5]u8 b_op;
            []u8 sop = fmt_buf(b_op, f"{op}");

            [1]u8 b_raw1;
            []u8 s_hex1 = fmt_buf(b_raw1, f"\xFF");
            []u8 s_hex2 = fmt_buf(b_raw1, f"{"\xAA"}");

            Mini m1 = Mini { x: 1 };
            Mini m2 = Mini { x: 2 };
            [2]Mini arr_m = [m1, m2];
            [34]u8 b_arr_m;
            []u8 sam = fmt_buf(b_arr_m, f"{arr_m:?}");

            f64 fv = 1.25;
            [21]u8 b_p9;
            []u8 sp9 = fmt_buf(b_p9, f"{fv:.9}");
            [19]u8 b_e9;
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
            [17]u8 b;
            []u8 s = fmt_buf(b, f"{np:?}");
        }
        "#,
        r#"
        struct Node { u32 v; *Node next; }
        fn main() {
            Node n = Node { v: 1, next: 0 as *Node };
            [47]u8 b;
            []u8 s = fmt_buf(b, f"{n:?}");
        }
        "#,
        r#"
        enum Op { A = 1, B = 2 }
        fn main() {
            Op op = Op.A;
            [4]u8 b;
            []u8 s = fmt_buf(b, f"{op}");
        }
        "#,
        r#"
        enum Op { A = 1, B = 2 }
        struct Wrap { Op op; }
        fn main() {
            Wrap w = Wrap { op: Op.A };
            [17]u8 b; // "Wrap { op: " (11) + "Op(?)" (5) + " }" (2) = 18 > 17
            []u8 s = fmt_buf(b, f"{w:?}");
        }
        "#,
        r#"
        fn main() {
            [1]u8 b;
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

        fn pair_tag<A, B>(A a, B b) -> usize {
            return 1;
        }

        fn kind_tag<T>(T v) -> usize {
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
            sys.log_kv("ok=", (t1 + t2 + k1 + k2 + (o1.item.v as usize) + (o2.item.v as usize) + (o3.item.v as usize)) as i32);
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
        const i8 SIGNED_WRAP = 255 as i8;
        const usize NINE = (SIGNED_WRAP + 10) as usize;

        struct Buf<usize N> {
            [N]u8 data;
        }

        fn first_elem<T, usize N>([N]T arr) -> T {
            return arr[0];
        }

        fn switch_and_shadow<usize N>(usize x) -> usize {
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
            Buf<((300 as u8) as usize)> b44;
            Buf<((~(0 as u8)) as usize)> b255;
            i32 v = first_elem([1000, 2000]);
            usize r = switch_and_shadow<7>(7);

            [16]u8 raw;
            []u8 fit = fmt_buf(raw[0..TOTAL], f"abcdef");
            [16]u8 raw9;
            []u8 fit9 = fmt_buf(raw9[0..NINE], f"123456789");
            println(fit);
            println(fit9);
            sys.log_kv("v=", v + (r as i32) + (b44.data[0] as i32) + (b255.data[0] as i32));
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
            [16]u8 raw;
            []u8 bad = fmt_buf(raw[0..TOTAL], f"abcdefg");
        }
        "#,
        DiagnosticKind::FormatBufferTooSmall,
    );

    // 2. M11: Unsuffixed float literal out of finite `f32` range (`1e40`, `-1e40`) rejected on `f32` coercion,
    //    while `f64` and explicit `1e40 as f32` cast pass!
    expect_ok(
        r#"
        fn main() {
            f64 big64 = 1e40;
            f32 inf32 = 1e40 as f32;
            if (big64 > 0.0 && inf32 > 0.0f32) {}
        }
        "#,
    );
    for bad_f32 in [
        r#"fn main() { f32 x = 1e40; }"#,
        r#"fn main() { f32 x = -1e40; }"#,
        r#"fn take(f32 x) {} fn main() { take(1e40); }"#,
        r#"watch<f32> w = 1e40; fn main() {}"#,
        r#"fn ret_bad() -> f32 { return 1e40; } fn main() {}"#,
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
    let app_p =
        zev_parser::parse_str("fn use_vec(FixedVec<u32> bad_param) {} fn main() {}").unwrap();
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
        &[
            (&rec_types, Some("types.zev")),
            (&rec_main, Some("main.zev")),
        ],
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
        fn bad_generic<T>(T x) -> T {
            bool flag = x;
            if (flag) {}
            return x;
        }
        fn main() {
            u32 y = bad_generic(42 as u32);
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

        fn add10(u32 x) -> u32 {
            return x + 10;
        }

        fn main() {
            BoxFn<fn(u32) -> u32> bf = BoxFn<fn(u32) -> u32> { cb: add10 };
            *BoxFn<fn(u32) -> u32> pbf = &bf;
            u32 r1 = bf.cb(32);
            u32 r2 = (bf.cb)(32);
            u32 r3 = pbf.cb(32);
            sys.log_kv("sum=", (r1 + r2 + r3) as i32);
        }
        "#,
    );

    // 2. Arity mismatch and argument type mismatch on `bf.cb(...)` report `E0019` (`TypeMismatch`)!
    for bad_call in [
        r#"
        struct BoxFn<F> { F cb; }
        fn add10(u32 x) -> u32 { return x + 10; }
        fn main() {
            BoxFn<fn(u32) -> u32> bf = BoxFn<fn(u32) -> u32> { cb: add10 };
            u32 r = bf.cb(1, 2);
        }
        "#,
        r#"
        struct BoxFn<F> { F cb; }
        fn add10(u32 x) -> u32 { return x + 10; }
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
        pub fn first_item<T, usize N>(Box<T, N> b) -> T { return b.data[0]; }
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
                sys.log_kv("v=", v as i32);
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

    let out = analyze_and_monomorphize_named_programs(&[(&main_prog, main_path.to_str())], &opts);
    assert!(
        out.diagnostics.is_empty(),
        "unexpected module diagnostics: {:?}",
        out.diagnostics
    );

    // 1. Canonicalized symbols and `display_names`.
    assert!(out
        .model
        .mmio_layouts
        .contains_key("drivers_uart__UartRegs"));
    assert!(out
        .model
        .mmio_instances
        .contains_key("drivers_uart__USART1"));
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
    assert_eq!(
        out.model.spawned_tasks[0].task_name,
        "drivers_uart__uart_worker"
    );
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
            @wo [2]u32 tx @ 0x14;
        }

        mmio USART1 : UartRegs @ 0x4001_1000;
        mmio Rcc @ RCC_BASE {
            @rw u32 cr @ 0x00;
        }

        fn step_uart(*UartRegs r) {
            r.cr1 |= 1;
            (*r).cr1 = 2;
            u16 st = r.sr;
            u32 flags = r.icr;
            r.icr = flags;
            r.dr = 0x41;
            r.brr[0] = 9600;
            r.brr[1] |= 0x10;
            r.tx[1] = 0x55;
            sys.log_kv("st=", st as i32);
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
        "mmio Regs { @rw u32 a @ 0x00; } fn take(Regs r) {} fn main() {}",
        "mmio Regs { @rw u32 a @ 0x00; } fn make() -> Regs { while (true) {} } fn main() {}",
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
            sys.log_kv("b=", id as i32);
        }
        isr simple_isr {}

        fn bind_uart_irq(*UartRegs regs, u32 tag) {
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
        "irq Q = 5; isr h(u32 x) {} fn setup(u32 x) { Q.attach(h(x)); } fn dyn_val() -> u32 { return 7; } fn main() { setup(dyn_val()); }",
        // Helper forwarding param with no boot callers
        "irq Q = 5; isr h(u32 x) {} fn dead_setup(u32 x) { Q.attach(h(x)); } fn main() {}",
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

        fn bump(*u32 p) {
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
        fn forward(*u32 p) { leak_w.publish(p); }
        fn bounce(*u32 p) { forward(p); }
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
        fn helper() {
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
        fn helper() {
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
        fn dangle() -> *u32 {
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
        "mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000; fn f() -> u32 { return DEV.r[4]; } fn main() {}",
        "mmio R { @rw [4]u32 r @ 0x0; } mmio DEV : R @ 0x40000000; const i32 N = -1; fn f() -> u32 { return DEV.r[N]; } fn main() {}",
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
        fn read_at(usize IDX) -> u32 { return DEV.r[IDX]; }
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
pub fn take2(bool a, bool b) -> bool { return a && b; }
const u32 N = 999;
pub const u32 A = 10;
pub const u32 C = 20;
fn secret_helper() -> u32 { return 7; }
"#;
    std::fs::write(tmp.join("sub.zev"), mod_src).expect("write sub.zev");

    // Valid use: `Ring<u8, 4>` inside `sub.zev` where `pub struct Ring<T, usize N>` sees `N`,
    // `sub.Kind.Alpha` & `sub.MODE_A` in `switch`, and parenthesized `sub.take2((sub.A < b), (sub.C > (d + 1)))`.
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
    bool ok = sub.take2((sub.A < b), (sub.C > (d + 1)));
    if (ok && r.head == 0) { x = 11; }
}
"#;
    let main_path = tmp.join("main.zev");
    std::fs::write(&main_path, main_ok).expect("write main.zev");
    let parsed_ok =
        zev_parser::parse_str_named(main_ok, main_path.to_str().unwrap()).expect("parse main_ok");
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
        [4]u8 tmp;
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
        "fn f() -> inbox<u8> { fn_unreachable(); } fn fn_unreachable() {}",
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
    assert_eq!(
        stale_count, 3,
        "expected 3 StaleDmaLease errors on *byte_ring"
    );
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
        format!("{sub_def} fn f() -> S {{}} fn main() {{}}"),
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

#[test]
fn verifies_pool_dma_slot_linear_lease_methods_and_view_bundle_generics() {
    let good = r#"
struct Frame {
    [16]u8 data;
    u16 len;
}

struct SpiBus {
    @dma_safe pool<Frame, 8> tx_pool;
    dma_slot<Frame> tx_slot;
    inbox<Lease<Frame>, 4> done_q;
}

struct DuplexStream {
    *byte_ring rx;
    *byte_ring tx;
}

struct StatusCell<T> {
    *watch<T> state;
}

fn publish_cell<T>(*StatusCell<T> cell, T val) {
    cell.state.publish(val);
}

fn read_cell<T>(*StatusCell<T> cell) -> T {
    return cell.state.read();
}

fn alloc_from<T>(*pool<T, 8> p) -> Lease<T> {
    return p.alloc();
}

fn release_with_defer(*pool<Frame, 8> p, *dma_slot<Frame> slot, *inbox<Lease<Frame>> q) error {
    Lease<Frame> l1 = p.try_alloc() catch {
        error;
    };
    defer l1.release();
    *Frame fp = l1.get();
    fp.len = 4;
    usize a = l1.addr();
    let_noop(a);

    Lease<Frame> l2 = p.alloc(timeout: 5ms) catch {
        return;
    };
    slot.park(l2) catch {
        return;
    };
    Lease<Frame> l3 = slot.take() catch {
        return;
    };
    q.post(l3);
    Lease<Frame> l4 = q.recv();
    l4.release(boost: @high);
    // Reassigning after move is permitted:
    l4 = p.alloc();
    if (l4.addr() != 0) {
        l4.release();
    } else {
        slot.park(l4) catch {};
    }
}

fn let_noop(usize x) {}

irq IRQ_SPI = 12;
SpiBus bus;
@dma_safe pool<Frame, 4> aux_pool;
watch<u32> g_state = 0;
byte_ring<16> g_rx;
byte_ring<16> g_tx;

isr spi_isr(*SpiBus b, *pool<Frame, 4> aux) {
    bool did = b.tx_slot.release(boost: @critical);
    if (did) {
        Lease<Frame> extra = aux.try_alloc() catch { return; };
        extra.release(boost: @high);
    }
}

task worker(*pool<Frame, 8> p, *dma_slot<Frame> slot, *inbox<Lease<Frame>> q) {
    StatusCell<u32> cell = StatusCell { state: &g_state };
    DuplexStream ds = DuplexStream { rx: &g_rx, tx: &g_tx };
    loop {
        publish_cell(&cell, 77);
        u32 cur = read_cell(&cell);
        []u8 rbuf = ds.rx.read_slice();
        if (rbuf.len > 0) {
            usize written = ds.tx.push(rbuf);
            ds.rx.consume(rbuf.len);
        }
        Lease<Frame> f = alloc_from(p);
        usize av = p.avail();
        usize used = p.in_use();
        usize cap = p.cap();
        bool ex = p.is_exhausted();
        bool busy = slot.is_busy();
        bool emp = slot.is_empty();
        usize saddr = slot.addr();
        if (cur == 0 && av == 0 && used == 0 && cap == 0 && ex && busy && emp && saddr == 0) {
            f.release();
            continue;
        }
        release_with_defer(p, slot, q) catch {};
        f.release();
    }
}

fn main() {
    IRQ_SPI.attach(spi_isr(&bus, &aux_pool));
    spawn worker(&bus.tx_pool, &bus.tx_slot, &bus.done_q) @normal;
}
"#;
    let model = expect_ok(good);
    assert!(model.has_dma_sleep_locks());
    assert!(model.primitives["aux_pool"].clamps_sleep_depth);
    assert!(model.structs["SpiBus"].is_subsystem);
    assert!(model.structs["SpiBus"].field("tx_pool").unwrap().dma_safe);
}

#[test]
fn rejects_e0034_linear_lease_violations_and_e0011_stale_get() {
    // 1. Unconsumed lease at scope / fn exit -> E0034
    expect_err_kind(
        r#"
        pool<u8, 4> p;
        fn main() {
            Lease<u8> l = p.alloc();
        }
        "#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 2. Unused expression statement -> E0034
    expect_err_kind(
        r#"
        pool<u8, 4> p;
        fn main() {
            p.alloc();
        }
        "#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 3. Use after move (.release / slot.park catch / inbox.post / double fn arg) -> E0034
    for bad in [
        "pool<u8, 4> p; fn main() { Lease<u8> l = p.alloc(); l.release(); usize a = l.addr(); }",
        "pool<u8, 4> p; dma_slot<u8> s; fn main() { Lease<u8> l = p.alloc(); s.park(l) catch {}; l.release(); }",
        "pool<u8, 4> p; inbox<Lease<u8>, 2> q; fn main() { Lease<u8> l = p.alloc(); q.post(l); *u8 ptr = l.get(); }",
        "fn two(Lease<u8> a, Lease<u8> b) { a.release(); b.release(); } pool<u8, 4> p; fn main() { Lease<u8> l = p.alloc(); two(l, l); }",
    ] {
        expect_err_kind(bad, DiagnosticKind::LinearLeaseViolation);
    }

    // 4. Overwriting Live or DeferredRelease variable -> E0034
    expect_err_kind(
        r#"
        pool<u8, 4> p;
        fn main() {
            Lease<u8> l = p.alloc();
            l = p.alloc();
            l.release();
        }
        "#,
        DiagnosticKind::LinearLeaseViolation,
    );
    expect_err_kind(
        r#"
        pool<u8, 4> p;
        fn main() {
            Lease<u8> l = p.alloc();
            defer l.release();
            l = p.alloc();
        }
        "#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 5. Double consume after defer -> E0034
    expect_err_kind(
        r#"
        pool<u8, 4> p;
        fn main() {
            Lease<u8> l = p.alloc();
            defer l.release();
            l.release();
        }
        "#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 6. Branch mismatch across if/else and switch -> E0034
    expect_err_kind(
        r#"
        pool<u8, 4> p;
        fn f(bool c) {
            Lease<u8> l = p.alloc();
            if (c) { l.release(); }
        }
        fn main() {}
        "#,
        DiagnosticKind::LinearLeaseViolation,
    );
    expect_err_kind(
        r#"
        pool<u8, 4> p;
        fn f(u8 k) {
            Lease<u8> l = p.alloc();
            switch (k) {
                0 => { l.release(); }
                _ => {}
            }
        }
        fn main() {}
        "#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 7. Moving outer live lease inside loop, or leaking on break/continue/return/error/sys.exit -> E0034
    for bad in [
        "pool<u8, 4> p; fn main() { Lease<u8> l = p.alloc(); for (usize i in 0..2) { l.release(); } }",
        "pool<u8, 4> p; fn main() { for (usize i in 0..2) { Lease<u8> l = p.alloc(); if (i == 1) { break; } l.release(); } }",
        "pool<u8, 4> p; fn f(bool b) error { Lease<u8> l = p.alloc(); if (b) { error; } l.release(); } fn main() {}",
        "pool<u8, 4> p; fn main() { Lease<u8> l = p.alloc(); sys.exit(1); }",
    ] {
        expect_err_kind(bad, DiagnosticKind::LinearLeaseViolation);
    }

    // 8. Disallowed pool bounds, element types, and Lease<T> placement -> E0034
    for bad in [
        "pool<u8, 0> p; fn main() {}",
        "pool<u8, 33> p; fn main() {}",
        "pool<*u8, 4> p; fn main() {}",
        "pool<watch<u8>, 4> p; fn main() {}",
        "dma_slot<[]u8> s; fn main() {}",
        "struct Bad { Lease<u8> l; } fn main() {}",
        "watch<Lease<u8>> w; fn main() {}",
        "ping_pong<Lease<u8>, 4> pp; fn main() {}",
        "pool<Lease<u8>, 4> p; fn main() {}",
        "task bad_task(Lease<u8> l) { loop { l.release(); } } fn main() {}",
        "irq IRQ0 = 1; isr bad_isr() { state u8 dummy = 0; } fn f(*Lease<u8> p) {} fn main() {}",
        "fn main() { [2]Lease<u8> arr; }",
        "fn main() { []Lease<u8> sl; }",
        "pool<u8, 4> p; fn main() { Lease<u8> l = p.alloc(); *Lease<u8> ptr = &l; l.release(); }",
        "enum E { Fail } pool<u8, 4> p; fn f() -> Lease<u8> error E { Lease<u8> l = p.alloc(); error E.Fail, l; } fn main() {}",
    ] {
        expect_err_kind(bad, DiagnosticKind::LinearLeaseViolation);
    }

    // 9. E0011 (StaleDmaLease) when dereferencing/reading `*T` from `Lease.get()` after `.release()` or move
    expect_err_kind(
        r#"
        pool<u8, 4> p;
        fn main() {
            Lease<u8> l = p.alloc();
            *u8 ptr = l.get();
            l.release();
            u8 v = *ptr;
        }
        "#,
        DiagnosticKind::StaleDmaLease,
    );

    // 10. E0009 (ReturnLocalReference) when returning `l.get()` pointer from a function
    expect_err_kind(
        r#"
        pool<u8, 4> p;
        fn bad_ret() -> *u8 {
            Lease<u8> l = p.alloc();
            defer l.release();
            return l.get();
        }
        fn main() {}
        "#,
        DiagnosticKind::ReturnLocalReference,
    );
}

#[test]
fn sys_tick_and_set_putc_hooks_validation() {
    // 1. Valid `sys.tick()`, `sys.tick_ms()`, and `sys.set_putc(fn(u8))` across ISR / fn / task
    let model = expect_ok(
        r#"
        irq IRQ_SYSTICK = 36;
        mmio TcbRegs { @w1c u8 intflags @ 0x06; }
        mmio TCB3 : TcbRegs @ 0x0AB0;

        fn uart_putc(u8 c) {
            usize _unused = c as usize;
        }

        isr systick_isr(*TcbRegs tcb) {
            tcb.intflags = 1;
            sys.tick();
        }

        task ticker_worker() {
            loop {
                sys.tick_ms();
                checkpoint();
            }
        }

        fn main() {
            sys.set_putc(uart_putc);
            IRQ_SYSTICK.attach(systick_isr(TCB3));
            spawn ticker_worker() @normal;
        }
        "#,
    );
    assert_eq!(model.fn_effect("uart_putc"), FnEffect::NonYielding);

    // 2. Arity errors on `sys.tick(1)` and `sys.tick_ms(1)`
    expect_err_kind(
        r#"
        fn main() {
            sys.tick(1);
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );
    expect_err_kind(
        r#"
        fn main() {
            sys.tick_ms(1);
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );

    // 3. `sys.set_putc` wrong callback signature -> TypeMismatch
    for bad_sig in [
        "fn bad(u16 c) {} fn main() { sys.set_putc(bad); }",
        "fn bad(u8 c) -> u8 { return c; } fn main() { sys.set_putc(bad); }",
        "fn bad() {} fn main() { sys.set_putc(bad); }",
        "fn main() { sys.set_putc(); }",
    ] {
        expect_err_kind(bad_sig, DiagnosticKind::TypeMismatch);
    }

    // 4. `sys.set_putc` fallible (`error`) callback rejected -> TypeMismatch
    expect_err_kind(
        r#"
        fn fallible_putc(u8 c) error {
            if (c == 0) { error; }
        }
        fn main() {
            sys.set_putc(fallible_putc);
        }
        "#,
        DiagnosticKind::TypeMismatch,
    );

    // 5. `sys.set_putc` yielding callback rejected -> YieldingCallbackToExternC
    expect_err_kind(
        r#"
        fn yielding_putc(u8 c) {
            usize _u = c as usize;
            checkpoint();
        }
        fn main() {
            sys.set_putc(yielding_putc);
        }
        "#,
        DiagnosticKind::YieldingCallbackToExternC,
    );

    // 6. `sys.set_putc` inside `isr` rejected -> InvalidControlFlow
    expect_err_kind(
        r#"
        irq IRQ0 = 1;
        fn uart_putc(u8 c) { usize _u = c as usize; }
        isr bad_isr() {
            sys.set_putc(uart_putc);
        }
        fn main() {
            IRQ0.attach(bad_isr);
        }
        "#,
        DiagnosticKind::InvalidControlFlow,
    );
}

#[test]
fn regression_s_h1_place_addr_provenance_preserving_scalar_reads() {
    // 1. Positive: scalar rvalue read `u8 x = sl[0]` does NOT carry ring lease past consume(),
    //    and `&p.hdr` / `&(*p)` on caller parameter `*Packet` can return cleanly.
    expect_ok(
        r#"
        byte_ring<16> rx;

        struct Packet {
            u32 hdr;
            [4]u8 payload;
        }

        fn get_hdr(*Packet p) -> *u32 {
            return &p.hdr;
        }

        fn id_pkt(*Packet p) -> *Packet {
            return &(*p);
        }

        fn read_once() -> u8 {
            []u8 sl = rx.read_slice();
            u8 first = sl[0];
            rx.consume(1);
            return first + 1;
        }

        fn main() {
            Packet pkt = Packet { hdr: 42, payload: [1, 2, 3, 4] };
            *Packet p = id_pkt(&pkt);
            *u32 h = get_hdr(p);
            u8 b = read_once();
            sys.log_kv("h=", (*h) as i32);
            sys.log_kv("b=", b);
        }
        "#,
    );

    // 2. `&slice[i]` from ring.read_slice() read after consume() -> E0011 (`StaleDmaLease`)
    expect_err_kind(
        r#"
        byte_ring<16> rx;
        fn main() {
            []u8 sl = rx.read_slice();
            *u8 p = &sl[0];
            rx.consume(1);
            u8 bad = *p;
            sys.log_kv("bad=", bad);
        }
        "#,
        DiagnosticKind::StaleDmaLease,
    );

    // 3. `&grant[0]` from grant_write() read after commit_write() -> E0011 (`StaleDmaLease`)
    expect_err_kind(
        r#"
        byte_ring<16> tx;
        fn main() {
            []u8 g = tx.grant_write(4) catch { return; };
            *u8 p = &g[0];
            tx.commit_write(1);
            *p = 7;
        }
        "#,
        DiagnosticKind::StaleDmaLease,
    );

    // 4. `&l.get().hdr` escaped / read after park -> E0011 (`StaleDmaLease`) or E0009
    expect_err_kind(
        r#"
        struct Packet { u32 hdr; }
        pool<Packet, 2> pkt_pool;
        fn main() {
            Lease<Packet> l = pkt_pool.alloc() catch { return; };
            *u32 hp = &l.get().hdr;
            *u32 hp2 = &l.get().hdr;
            u32 stale = *hp;
            sys.log_kv("s=", stale);
            sys.log_kv("s2=", *hp2);
            l.release();
        }
        "#,
        DiagnosticKind::StaleDmaLease,
    );

    // 5. Escaping local `&p.field`, `&(*p)`, `&sl[0]`, `sl.ptr` -> E0009 (`DanglingReference`)
    for bad in [
        r#"
        struct Packet { u32 hdr; }
        fn esc_field() -> *u32 {
            Packet pkt = Packet { hdr: 1 };
            *Packet p = &pkt;
            return &p.hdr;
        }
        fn main() {}
        "#,
        r#"
        fn esc_deref() -> *u32 {
            u32 x = 5;
            *u32 p = &x;
            return &(*p);
        }
        fn main() {}
        "#,
        r#"
        byte_ring<16> rx;
        fn esc_slice_elem() -> *u8 {
            []u8 sl = rx.read_slice();
            return &sl[0];
        }
        fn main() {}
        "#,
        r#"
        byte_ring<16> rx;
        fn esc_slice_ptr() -> *u8 {
            []u8 sl = rx.read_slice();
            return sl.ptr;
        }
        fn main() {}
        "#,
    ] {
        expect_err_kind(bad, DiagnosticKind::ReturnLocalReference);
    }

    // 6. `&MMIO.reg`, `&MMIO.arr[i]`, `MMIO.arr.ptr`, `MMIO.arr[0..2]` -> E0031 (`MmioViolation`)
    for bad_mmio in [
        "mmio R { @rw u32 cr @ 0; } mmio D : R @ 0x4000_0000; fn main() { *u32 p = &D.cr; }",
        "mmio R { @rw [4]u32 arr @ 0; } mmio D : R @ 0x4000_0000; fn main() { *u32 p = &D.arr[0]; }",
        "mmio R { @rw [4]u32 arr @ 0; } mmio D : R @ 0x4000_0000; fn main() { *u32 p = D.arr.ptr; }",
        "mmio R { @rw [4]u32 arr @ 0; } mmio D : R @ 0x4000_0000; fn main() { []u32 s = D.arr[0..2]; }",
    ] {
        expect_err_kind(bad_mmio, DiagnosticKind::MmioViolation);
    }
}

#[test]
fn regression_s_h2_short_circuit_and_or_lease_control_flow() {
    // 1. Moving linear `Lease<T>` only on RHS of `&&` or `||` -> E0034 (`LinearLeaseViolation`)
    for op in ["&&", "||"] {
        let src = format!(
            r#"
            struct Pkt {{ u32 id; }}
            pool<Pkt, 2> p;
            fn take_it(Lease<Pkt> l) -> bool {{
                l.release();
                return true;
            }}
            fn run(bool cond) {{
                Lease<Pkt> l = p.alloc() catch {{ return; }};
                if (cond {op} take_it(l)) {{
                    sys.log("yes");
                }}
            }}
            fn main() {{ run(false); }}
            "#
        );
        expect_err_kind(&src, DiagnosticKind::LinearLeaseViolation);
    }

    // 2. Invalidating byte_ring lease on RHS of `&&` / `||` makes earlier slice stale -> E0011 (`StaleDmaLease`)
    for op in ["&&", "||"] {
        let src = format!(
            r#"
            byte_ring<16> rx;
            fn check(bool cond) {{
                []u8 sl = rx.read_slice();
                if (cond {op} ((rx.try_pop_byte() catch {{ 0 }}) > 0)) {{
                    sys.log("hit");
                }}
                u8 b = sl[0];
                sys.log_kv("b=", b);
            }}
            fn main() {{ check(false); }}
            "#
        );
        expect_err_kind(&src, DiagnosticKind::StaleDmaLease);
    }
}

#[test]
fn regression_s_h3_catch_mmio_and_linear_lease_bindings() {
    // 1. By-value MMIO in catch error param -> E0031 (`MmioViolation`)
    expect_err_kind(
        r#"
        mmio Regs { @rw u32 cr @ 0; }
        fn fall() -> u32 error u32 { error 1; }
        fn main() {
            u32 r = fall() catch (Regs e) { return; };
        }
        "#,
        DiagnosticKind::MmioViolation,
    );

    // 2. Linear `Lease<T>` in catch partial parameter -> E0034 (`LinearLeaseViolation`), NO spurious unused-lease cascade!
    let diags = expect_err_kind(
        r#"
        struct Pkt { u32 x; }
        pool<Pkt, 2> p;
        enum AllocErr { Empty }
        fn try_get() -> Lease<Pkt> error AllocErr {
            Lease<Pkt> l = p.alloc(timeout: 1ms) catch { error Empty; };
            return l;
        }
        fn main() {
            Lease<Pkt> out = try_get() catch (AllocErr e, Lease<Pkt> partial) {
                return;
            };
            out.release();
        }
        "#,
        DiagnosticKind::LinearLeaseViolation,
    );
    assert_eq!(
        diags.len(),
        1,
        "expected single clean E0034 without double-free/leak cascade, got {diags:?}"
    );
}

#[test]
fn regression_s_h4_s_h5_mmio_and_os_primitive_void_coercion_and_casts() {
    // 1. Positive allowed casts
    expect_ok(
        r#"
        mmio Regs { @rw u32 cr @ 0; }
        mmio DEV : Regs @ 0x4000_0000;
        signal sig;
        byte_ring<16> br;
        inbox<u32, 4> q;

        fn main() {
            *Regs p1 = DEV as *Regs;
            *Regs p2 = p1 as *Regs;
            usize addr = DEV as usize;
            *signal sp = &sig as *signal;
            *byte_ring<16> bp1 = &br as *byte_ring<16>;
            *byte_ring bp2 = &br as *byte_ring;
            *inbox<u32> qp = &q as *inbox<u32>;
            sys.log_kv("a=", addr as i32);
            p2.cr = 1;
            sp.notify();
            bp1.push_byte(1);
            usize _cap = bp2.cap() + qp.cap();
        }
        "#,
    );

    // 2. Forbidden MMIO coercion to `*void` & casts -> E0031 (`MmioViolation`)
    for bad_mmio in [
        "mmio A { @rw u32 r @ 0; } mmio D : A @ 0x4000_0000; fn take(*void v) {} fn main() { take(D as *A); }",
        "mmio A { @rw u32 r @ 0; } mmio D : A @ 0x4000_0000; fn take(*void v) {} fn main() { take(&D); }",
        "mmio A { @rw u32 r @ 0; } mmio D : A @ 0x4000_0000; fn main() { *void v = (D as *A) as *void; }",
        "mmio A { @rw u32 r @ 0; } mmio D : A @ 0x4000_0000; fn main() { *u32 p = (D as *A) as *u32; }",
        "mmio A { @rw u32 r @ 0; } mmio B { @rw u32 r @ 0; } mmio D : A @ 0x4000_0000; fn main() { *B p = (D as *A) as *B; }",
        "mmio A { @rw u32 r @ 0; } fn main() { *A p = 0x4000_0000 as *A; }",
        "mmio A { @rw u32 r @ 0; } fn main() { *void vp = 0 as *void; *A p = vp as *A; }",
        "mmio A { @rw u32 r @ 0; } mmio D : A @ 0x4000_0000; fn main() { *A p = D as *A; usize u = p as usize; }",
        "mmio A { @rw u32 r @ 0; } mmio B { @rw u32 r @ 0; } mmio D : A @ 0x4000_0000; fn main() { *B p = D as *B; }",
    ] {
        expect_err_kind(bad_mmio, DiagnosticKind::MmioViolation);
    }

    // 3. Forbidden `*OsPrimitive` coercion to `*void` & raw casts -> E0019 (`TypeMismatch`)
    for bad_prim in [
        "signal s; fn take(*void v) {} fn main() { take(&s); }",
        "byte_ring<8> r; fn take(*void v) {} fn main() { take(&r); }",
        "signal s; fn main() { *void v = &s as *void; }",
        "signal s; fn main() { *u8 p = &s as *u8; }",
        "signal s; fn main() { usize u = &s as usize; }",
        "fn main() { *void v = 0 as *void; *byte_ring r = v as *byte_ring; }",
        "fn main() { *signal s = 0x2000 as *signal; }",
        "byte_ring<8> r; fn main() { *byte_ring<16> p = &r as *byte_ring<16>; }",
        "fn bad(*byte_ring r) { *byte_ring<8> p = r as *byte_ring<8>; }",
    ] {
        expect_err_kind(bad_prim, DiagnosticKind::TypeMismatch);
    }
}

#[test]
fn regression_s_h6_byval_deref_vs_projection() {
    // 1. Positive: projected `(*p).field`, `(*p).method()`, `&(*p).field` MUST remain valid!
    expect_ok(
        r#"
        mmio Regs { @rw u32 cr @ 0; }
        mmio DEV : Regs @ 0x4000_0000;

        struct UartSub {
            byte_ring<16> tx;
        }
        UartSub uart0;

        struct Pkt {
            u32 hdr;
        }

        fn work(*Regs r, *UartSub s, *byte_ring<16> br, *Pkt pkt) {
            (*r).cr = 1;
            u32 v = (*r).cr;
            (*s).tx.push_byte(1);
            usize n = (*br).len();
            *u32 hp = &(*pkt).hdr;
            sys.log_kv("v=", (v + *hp + (n as u32)) as i32);
        }

        fn main() {
            Pkt pkt = Pkt { hdr: 9 };
            work(DEV as *Regs, &uart0, &uart0.tx, &pkt);
        }
        "#,
    );

    // 2. Negative: by-value deref read and write on MMIO layout -> E0031 (`MmioViolation`)
    expect_err_kind(
        r#"
        mmio Regs { @rw u32 cr @ 0; }
        fn f(*Regs a) { *a; }
        fn main() {}
        "#,
        DiagnosticKind::MmioViolation,
    );
    expect_err_kind(
        r#"
        mmio Regs { @rw u32 cr @ 0; }
        fn f(*Regs a, *Regs b) { *a = *b; }
        fn main() {}
        "#,
        DiagnosticKind::MmioViolation,
    );

    // 3. Negative: by-value deref read and write on subsystem struct & OS primitive -> E0024 (`InvalidOperation`)
    for bad in [
        "struct Sub { signal s; } fn f(*Sub a) { *a; } fn main() {}",
        "struct Sub { signal s; } fn f(*Sub a, *Sub b) { *a = *b; } fn main() {}",
        "fn f(*byte_ring<8> a) { *a; } fn main() {}",
        "fn f(*byte_ring<8> a, *byte_ring<8> b) { *a = *b; } fn main() {}",
    ] {
        expect_err_kind(bad, DiagnosticKind::InvalidOperation);
    }
}

#[test]
fn regression_s_h7_fmt_ring_invalidation_and_byte_ring_spsc() {
    // 1. S-H7a: `fmt_ring(&tx, ...)` invalidates any outstanding `tx.grant_write(...)` slice -> E0011!
    expect_err_kind(
        r#"
        byte_ring<32> tx;
        fn main() {
            []u8 g = tx.grant_write(8) catch { return; };
            usize _n = fmt_ring(&tx, f"hi") catch { 0 };
            g[0] = 65;
        }
        "#,
        DiagnosticKind::StaleDmaLease,
    );

    // 2. S-H7b Positive: Single producer task + single consumer ISR via view bundle `DuplexStream`,
    //    sequential pre-spawn selftest in `main()`, plus multi-actor pure inspection ops (`.len()` / `.is_empty()`).
    expect_ok(
        r#"
        irq IRQ_TX = 10;
        byte_ring<64> rx_ring;
        byte_ring<64> tx_ring;

        struct DuplexStream {
            *byte_ring rx;
            *byte_ring tx;
        }

        fn boot_check(*byte_ring r) {
            r.push_byte(0xAA);
            u8 _v = r.try_pop_byte() catch { 0 };
        }

        fn write_helper(DuplexStream s) {
            usize _n = fmt_ring(s.tx, f"ok") catch { 0 };
        }

        task app_task(DuplexStream io_stream) {
            loop {
                u8 b = io_stream.rx.try_pop_byte() catch {
                    delay(1ms);
                    continue;
                };
                write_helper(io_stream);
                usize _free = io_stream.tx.free_space();
                sys.log_kv("b=", b);
            }
        }

        isr tx_isr(*byte_ring r) {
            if (!r.is_empty()) {
                u8 _b = r.try_pop_byte() catch { return; };
            }
            usize _cap = r.cap();
        }

        fn main() {
            // Sequential pre-spawn boot check on both producer & consumer operations:
            boot_check(&tx_ring);
            rx_ring.push_byte(1);

            DuplexStream s = DuplexStream { rx: &rx_ring, tx: &tx_ring };
            IRQ_TX.attach(tx_isr(&tx_ring));
            spawn app_task(s) @normal;

            // Main post-boot acts only as single producer for `rx_ring` and inspects `tx_ring.len()`:
            loop {
                rx_ring.push_byte(2);
                usize _n = tx_ring.len();
                delay(5ms);
            }
        }
        "#,
    );

    // 3. S-H7b Negative: two producers -> E0002 (`GlobalIsolationViolation`)
    expect_err_kind(
        r#"
        byte_ring<16> r;
        task p1() { loop { r.push_byte(1); delay(1ms); } }
        task p2() { loop { r.push_byte(2); delay(1ms); } }
        fn main() {
            spawn p1();
            spawn p2();
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // 4. S-H7b Negative: producer ISR + post-spawn `main()` producer -> E0002
    expect_err_kind(
        r#"
        irq IRQ0 = 3;
        byte_ring<16> r;
        isr rx_isr(*byte_ring p) { p.push_byte(1); }
        fn main() {
            IRQ0.attach(rx_isr(&r));
            r.push_byte(2);
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // 5. S-H7b Negative: consumer task + post-spawn `main()` consumer -> E0002
    expect_err_kind(
        r#"
        byte_ring<16> r;
        task c1() { loop { []u8 s = r.wait_read(); r.consume(s.len); } }
        fn main() {
            spawn c1();
            u8 _b = r.try_pop_byte() catch { 0 };
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // 6. S-H7b Negative: transitive helper + `DuplexStream` aliasing two writers -> E0002
    expect_err_kind(
        r#"
        byte_ring<16> tx;
        struct DuplexStream { *byte_ring tx; }
        fn helper_write(DuplexStream s) { s.tx.push_byte(1); }
        task w1(DuplexStream s) { loop { helper_write(s); delay(1ms); } }
        task w2(*byte_ring r) { loop { usize _n = fmt_ring(r, f"x") catch { 0 }; delay(1ms); } }
        fn main() {
            DuplexStream s = DuplexStream { tx: &tx };
            spawn w1(s);
            spawn w2(&tx);
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // 7. S-H7b Negative: `clear()` counts as both producer & consumer -> conflicts with writer task -> E0002
    expect_err_kind(
        r#"
        byte_ring<16> r;
        task writer() { loop { r.push_byte(1); delay(1ms); } }
        task cleaner() { loop { r.clear(); delay(10ms); } }
        fn main() {
            spawn writer();
            spawn cleaner();
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );
}

#[test]
fn regression_s_m1_s_m2_s_l2_module_enums_pub_visibility_and_extern_c() {
    // 1. S-M1 + S-L2 positive: private and public enum variants in different modules do NOT collide,
    //    and identical `extern "C"` prototypes across multiple roots + modules deduplicate cleanly!
    {
        let ws = TempWorkspace::new("s_m1_l2_ok");
        ws.write("Zev.toml", "[package]\nname = \"app\"\n");
        ws.write(
            "mod_a.zev",
            r#"
            pub extern "C" fn hal_init(u32 v) -> u32;
            enum PrivA { Shared, OnlyA }
            pub enum PubA { Common, ExtraA }
            pub fn pick_a() -> u32 {
                PrivA x = Shared;
                PubA y = Common;
                return (x as u32) + (y as u32);
            }
            "#,
        );
        ws.write(
            "mod_b.zev",
            r#"
            pub extern "C" fn hal_init(u32 v) -> u32;
            pub enum PubB { Shared, Common, ExtraB }
            pub fn pick_b() -> u32 {
                PubB x = Shared;
                return x as u32;
            }
            "#,
        );
        let root1 = ws.write(
            "main.zev",
            r#"
            import mod_a as ma;
            import mod_b as mb;
            extern "C" fn hal_init(u32 v) -> u32;
            enum RootMode { Shared = 1 }
            fn main() {
                RootMode m = Shared;
                u32 x = hal_init((m as u32) + ma.pick_a() + mb.pick_b());
                sys.log_kv("x=", x as i32);
            }
            "#,
        );
        let root2 = ws.write(
            "extra.zev",
            r#"
            pub extern "C" fn hal_init(u32 v) -> u32;
            "#,
        );
        let p1 = zev_parser::parse_str_named(
            &std::fs::read_to_string(&root1).unwrap(),
            root1.to_str().unwrap(),
        )
        .unwrap();
        let p2 = zev_parser::parse_str_named(
            &std::fs::read_to_string(&root2).unwrap(),
            root2.to_str().unwrap(),
        )
        .unwrap();
        let out = analyze_and_monomorphize_named_programs(
            &[(&p1, root1.to_str()), (&p2, root2.to_str())],
            &AnalyzeOptions::default(),
        );
        assert!(
            out.diagnostics.is_empty(),
            "unexpected diagnostics in S-M1/S-L2 positive test: {:?}",
            out.diagnostics
        );
    }

    // 2. S-M2: Reject private local struct/enum/mmio exposed in `pub` items -> E0030 (`ModuleError`)
    for priv_pub_mod in [
        "struct Priv { u32 x; } pub struct Wrap { Priv p; }",
        "struct Priv { u32 x; } pub fn f(Priv p) {}",
        "struct Priv { u32 x; } pub fn f() -> Priv { return Priv { x: 1 }; }",
        "enum PrivE { E0 } pub fn f() error PrivE { error E0; }",
        "struct Priv { u32 x; } pub extern \"C\" fn c_f(Priv p);",
        "enum PrivE { E0 } pub const PrivE C = PrivE.E0;",
        "struct Priv { u32 x; } pub inbox<Priv, 4> q;",
        "struct Priv { u32 x; } pub task t(Priv p) {}",
        "struct Priv { u32 x; } pub isr my_isr(*Priv p) {}",
        "mmio PrivRegs { @rw u32 cr @ 0; } pub mmio DEV : PrivRegs @ 0x4000_0000;",
    ] {
        let ws = TempWorkspace::new("s_m2_err");
        ws.write("Zev.toml", "[package]\nname = \"app\"\n");
        ws.write("badmod.zev", priv_pub_mod);
        let main_path = ws.write("main.zev", "import badmod as b;\nfn main() {}\n");
        let prog = zev_parser::parse_str_named(
            &std::fs::read_to_string(&main_path).unwrap(),
            main_path.to_str().unwrap(),
        )
        .unwrap();
        let out = analyze_and_monomorphize_named_programs(
            &[(&prog, main_path.to_str())],
            &AnalyzeOptions::default(),
        );
        assert!(
            out.diagnostics
                .iter()
                .any(|d| d.kind == DiagnosticKind::ModuleError),
            "expected E0030 ModuleError for `{priv_pub_mod}`, got {:?}",
            out.diagnostics
        );
    }

    // 3. S-L2 negative: same-file duplicate `extern "C"` -> E0021, cross-file mismatch -> E0030
    expect_err_kind(
        r#"
        extern "C" fn c_dup();
        extern "C" fn c_dup();
        fn main() {}
        "#,
        DiagnosticKind::DuplicateDefinition,
    );

    {
        let ws = TempWorkspace::new("s_l2_conflict");
        ws.write("Zev.toml", "[package]\nname = \"app\"\n");
        ws.write("other.zev", "pub extern \"C\" fn mismatch(u32 a);\n");
        let main_path = ws.write(
            "main.zev",
            "import other as o;\nextern \"C\" fn mismatch(i32 a);\nfn main() {}\n",
        );
        let prog = zev_parser::parse_str_named(
            &std::fs::read_to_string(&main_path).unwrap(),
            main_path.to_str().unwrap(),
        )
        .unwrap();
        let out = analyze_and_monomorphize_named_programs(
            &[(&prog, main_path.to_str())],
            &AnalyzeOptions::default(),
        );
        assert!(
            out.diagnostics
                .iter()
                .any(|d| d.kind == DiagnosticKind::ModuleError),
            "expected E0030 cross-file extern C conflict, got {:?}",
            out.diagnostics
        );
    }
}

#[test]
fn regression_s_m5a_mmio_u32_overflow() {
    // 1. Register end offset overflowing u32 -> E0031 (`MmioViolation`)
    expect_err_kind(
        r#"
        mmio TooHigh {
            @rw u32 r @ 0xFFFF_FFFC;
        }
        fn main() {}
        "#,
        DiagnosticKind::MmioViolation,
    );

    // 2. Register array size overflowing u32 -> E0031 (`MmioViolation`)
    expect_err_kind(
        r#"
        mmio HugeArr {
            @rw [0x4000_0001]u32 r @ 0x0;
        }
        fn main() {}
        "#,
        DiagnosticKind::MmioViolation,
    );

    // 3. MMIO instance base + layout byte_size overflowing u32 -> E0031 (`MmioViolation`)
    expect_err_kind(
        r#"
        mmio Block16 {
            @rw [4]u32 regs @ 0x0;
        }
        mmio BAD_INST : Block16 @ 0xFFFF_FFF8;
        fn main() {}
        "#,
        DiagnosticKind::MmioViolation,
    );

    // 4. Exact boundary fitting inside 32-bit address space is accepted cleanly
    expect_ok(
        r#"
        mmio Reg4 {
            @rw u32 reg @ 0x0;
        }
        mmio GOOD_INST : Reg4 @ 0xFFFF_FFF0;
        fn main() {}
        "#,
    );
}

#[test]
fn regression_l_h1_shift_preserves_lhs_integer_type() {
    expect_ok(
        r#"
        fn main() {
            u32 a = 0x8000_0000u32 >> 4i32;
            u32 b = 0x8000_0000u32 >> 4u64;
            u8 c = 1u8 << 3u32;
            u16 d = 0x8000u16 >> 1usize;
            i8 e = (-8i8) >> 1u32;
            sys.log_kv("a=", (a + b + (c as u32) + (d as u32) + (e as u32)) as i32);
        }
        "#,
    );
}

#[test]
fn explicit_sys_hooks_and_rom_call_validation() {
    // 1. Positive usage across all explicit `sys.*` OS hooks + compile-time bounded ROM calls
    let model = expect_ok(
        r#"
        const usize ROM_LOOKUP_STACK = 32;
        irq IRQ_TIMER = 1;

        fn uart_putc(u8 c) { usize _u = c as usize; }
        fn read_ms() -> u32 { return 42; }
        fn on_exit(i32 code) { i32 _c = code; }
        fn on_fault() {}
        fn on_idle(u32 ms, u32 depth) { u32 _x = ms + depth; }
        fn on_route(u32 src, u32 slot) { u32 _x = src + slot; }
        fn on_ack(u32 slot) { u32 _s = slot; }

        isr tick_isr() {
            sys.tick_by(10);
        }

        fn lookup_rom(usize tbl, u32 code) -> u32 {
            u32 r = sys.rom_call_u32(0x18usize, ROM_LOOKUP_STACK, tbl, code);
            sys.rom_call_void(0x20usize, 16, r);
            return r;
        }

        fn reboot_bootrom() {
            sys.rom_call_noreturn(0x100usize, 0u32, 0u32);
        }

        fn diverging_fn() -> u32 {
            sys.rom_call_noreturn(0x100usize);
        }

        task worker() {
            u32 x = lookup_rom(0x14usize, 0x4255u32);
            if (x == 999) {
                u32 _d = diverging_fn();
                sys.rom_call_noreturn(0x200usize);
            }
            loop {
                checkpoint();
            }
        }

        fn main() {
            sys.set_putc(uart_putc);
            sys.set_clock_ms(read_ms);
            sys.set_exit(on_exit);
            sys.set_fault_handler(on_fault);
            sys.set_idle(on_idle);
            sys.set_irq_router(on_route, on_ack);
            if (false) {
                reboot_bootrom();
            }
            IRQ_TIMER.attach(tick_isr);
            spawn worker() @normal;
        }
        "#,
    );
    assert_eq!(model.fn_effect("read_ms"), FnEffect::NonYielding);
    assert_eq!(model.fn_effect("on_ack"), FnEffect::NonYielding);

    // 2. Wrong signatures and arities -> TypeMismatch
    for bad in [
        "fn main() { sys.tick_by(); }",
        "fn main() { sys.tick_by(true); }",
        "fn bad() -> i32 { return 0; } fn main() { sys.set_clock_ms(bad); }",
        "fn bad(u32 x) {} fn main() { sys.set_exit(bad); }",
        "fn bad(u8 x) {} fn main() { sys.set_fault_handler(bad); }",
        "fn bad(u32 x) {} fn main() { sys.set_idle(bad); }",
        "fn route(u32 a, u32 b) {} fn bad_ack() {} fn main() { sys.set_irq_router(route, bad_ack); }",
        "fn bad_route(u32 a) {} fn ack(u32 s) {} fn main() { sys.set_irq_router(bad_route, ack); }",
        "fn main() { sys.set_irq_router(ack); }",
        "fn main() { u32 x = sys.rom_call_u32(0x10usize); }",
        "fn main() { sys.rom_call_void(0x10usize, 16, 1, 2, 3, 4, 5); }",
        "fn main() { sys.rom_call_noreturn(0x10usize, 1, 2, 3, 4, 5); }",
    ] {
        expect_err_kind(bad, DiagnosticKind::TypeMismatch);
    }

    // 3. Fallible (`error`) callback rejected on all new hooks -> TypeMismatch
    for bad_fallible in [
        "fn f() -> u32 error { error; } fn main() { sys.set_clock_ms(f); }",
        "fn f(i32 c) error { error; } fn main() { sys.set_exit(f); }",
        "fn f() error { error; } fn main() { sys.set_fault_handler(f); }",
        "fn f(u32 a, u32 b) error { error; } fn main() { sys.set_idle(f); }",
        "fn r(u32 a, u32 b) error { error; } fn ack(u32 s) {} fn main() { sys.set_irq_router(r, ack); }",
        "fn r(u32 a, u32 b) {} fn ack(u32 s) error { error; } fn main() { sys.set_irq_router(r, ack); }",
    ] {
        expect_err_kind(bad_fallible, DiagnosticKind::TypeMismatch);
    }

    // 4. Yielding callbacks rejected on all new hooks -> YieldingCallbackToExternC
    for bad_yielding in [
        "fn f() -> u32 { checkpoint(); return 1; } fn main() { sys.set_clock_ms(f); }",
        "fn f(i32 c) { i32 _c = c; checkpoint(); } fn main() { sys.set_exit(f); }",
        "fn f() { checkpoint(); } fn main() { sys.set_fault_handler(f); }",
        "fn f(u32 a, u32 b) { u32 _x = a + b; checkpoint(); } fn main() { sys.set_idle(f); }",
        "fn r(u32 a, u32 b) { u32 _x = a + b; checkpoint(); } fn ack(u32 s) {} fn main() { sys.set_irq_router(r, ack); }",
        "fn r(u32 a, u32 b) {} fn ack(u32 s) { u32 _s = s; checkpoint(); } fn main() { sys.set_irq_router(r, ack); }",
    ] {
        expect_err_kind(bad_yielding, DiagnosticKind::YieldingCallbackToExternC);
    }

    // 5. Calling any `sys.set_*` inside an `isr` rejected -> InvalidControlFlow
    for bad_in_isr in [
        "irq IRQ0 = 1; fn f() -> u32 { return 0; } isr h() { sys.set_clock_ms(f); } fn main() { IRQ0.attach(h); }",
        "irq IRQ0 = 1; fn f(i32 c) {} isr h() { sys.set_exit(f); } fn main() { IRQ0.attach(h); }",
        "irq IRQ0 = 1; fn f() {} isr h() { sys.set_fault_handler(f); } fn main() { IRQ0.attach(h); }",
        "irq IRQ0 = 1; fn f(u32 a, u32 b) {} isr h() { sys.set_idle(f); } fn main() { IRQ0.attach(h); }",
        "irq IRQ0 = 1; fn r(u32 a, u32 b) {} fn ack(u32 s) {} isr h() { sys.set_irq_router(r, ack); } fn main() { IRQ0.attach(h); }",
    ] {
        expect_err_kind(bad_in_isr, DiagnosticKind::InvalidControlFlow);
    }

    // 6. Non-constant or negative `stack_bound` rejected -> InvalidOperation
    for bad_bound in [
        "fn call_it(usize dyn_sb) { u32 _x = sys.rom_call_u32(0x10usize, dyn_sb); } fn main() { call_it(32); }",
        "fn main() { sys.rom_call_void(0x10usize, (-4i32) as usize); }",
        "fn main() { usize sb = 16; sys.rom_call_void(0x10usize, sb); }",
    ] {
        expect_err_kind(bad_bound, DiagnosticKind::InvalidOperation);
    }
}

#[test]
fn sys_nop_verifies_and_removed_timing_sleep_intrinsics_rejected() {
    let ok = r#"
irq IRQ_T = 1;
isr t_isr() {
    sys.nop();
}
fn main() {
    IRQ_T.attach(t_isr);
    assert(sys.sleep_depth() == 2);
    sys.nop();
}
"#;
    let prog = zev_parser::parse_str(ok).unwrap();
    zev_sema::analyze_program(&prog).expect("sys.nop() should verify in tasks and ISRs");

    expect_err_kind("fn main() { sys.nop(1); }", DiagnosticKind::TypeMismatch);
    expect_err_kind(
        "fn main() { sys.lock_sleep(); }",
        DiagnosticKind::UndefinedSymbol,
    );
    expect_err_kind(
        "fn main() { sys.unlock_sleep(); }",
        DiagnosticKind::UndefinedSymbol,
    );
    expect_err_kind(
        "fn main() { sys.delay_us(5u32); }",
        DiagnosticKind::UndefinedSymbol,
    );
    expect_err_kind(
        "fn f(u32 us) { u32 _x = us; } fn main() { sys.set_delay_us(f); }",
        DiagnosticKind::UndefinedSymbol,
    );
}

#[test]
fn std_math_str_and_ping_pong_circular_dma_sema() {
    let ok_src = r#"
import std.math as math;
import std.str as str;

@dma_safe ping_pong<u8, 16> pp;
irq IRQ_DMA = 15;

isr dma_isr(*ping_pong<u8, 16> rx) {
    rx.complete_half(0u8);
    rx.complete_half(1u8);
}

task consumer(*ping_pong<u8, 16> rx) {
    [16]u8 local;
    loop {
        []u8 chunk = rx.wait_chunk();
        usize n = math.min<usize>(chunk.len, 16);
        for (usize i in 0..n) {
            local[i] = chunk[i];
        }
        []u8 t = str.trim_ascii(local[0..n]);
        u32 _v = str.parse_u32(t) or 0u32;
    }
}

fn main() {
    i32 c = math.clamp<i32>(-5, 0, 10);
    u32 a = math.abs_i32(-42);
    u8 cs = str.xor_checksum("PMTK605");
    assert(c == 0 && a == 42u32 && cs == 0x31u8);
    usize dma_ptr = pp.base_addr();
    assert(dma_ptr != 0usize);
    pp.start();
    IRQ_DMA.attach(dma_isr(&pp));
    spawn consumer(&pp);
    loop {
        delay(10ms);
    }
}
"#;
    let model = expect_ok(ok_src);
    assert!(model.functions.contains_key("std_math__clamp__i32"));
    assert!(model.functions.contains_key("std_math__abs_i32"));
    assert!(model.functions.contains_key("std_str__xor_checksum"));

    expect_err_kind(
        r#"
@dma_safe ping_pong<u8, 16> pp;
irq IRQ_DMA = 15;
isr dma_isr(*ping_pong<u8, 16> rx) { rx.complete_half(true); }
fn main() { IRQ_DMA.attach(dma_isr(&pp)); }
"#,
        DiagnosticKind::TypeMismatch,
    );
    expect_err_kind(
        r#"
@dma_safe ping_pong<u8, 16> pp;
fn main() { usize _x = pp.base_addr(1); }
"#,
        DiagnosticKind::TypeMismatch,
    );
}

#[test]
fn hal_dir_target_binding_sim_vs_stm32f3_sema() {
    let src = r#"
import hal.gpio as gpio;

fn main() {
    gpio.Pin p = gpio.make_pin(0u8, 5u8, false);
    p.set_mode(gpio.Mode.Output, gpio.Pull.None);
    p.write(true);
    bool _v = p.read();
}
"#;
    let prog = zev_parser::parse_str(src).expect("parse hal test");

    let sim_out =
        zev_sema::analyze_with_options(std::slice::from_ref(&prog), &AnalyzeOptions::default());
    assert!(
        sim_out.diagnostics.is_empty(),
        "hal/sim diagnostics: {:?}",
        sim_out.diagnostics
    );
    assert!(sim_out.model.functions.contains_key("hal_gpio__make_pin"));
    assert!(sim_out
        .model
        .functions
        .contains_key("hal_gpio__Pin__set_mode"));
    assert!(sim_out.model.functions.contains_key("hal_gpio__Pin__write"));
    assert!(sim_out.model.functions.contains_key("hal_gpio__Pin__read"));
    assert!(!sim_out
        .model
        .mmio_instances
        .contains_key("chip_stm32f334__GPIOA"));

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f3_opts = AnalyzeOptions {
        include_dirs: vec![root],
        hal_dir: Some("hal/stm32f3".to_string()),
        ..AnalyzeOptions::default()
    };
    let f3_out = zev_sema::analyze_with_options(std::slice::from_ref(&prog), &f3_opts);
    assert!(
        f3_out.diagnostics.is_empty(),
        "hal/stm32f3 diagnostics: {:?}",
        f3_out.diagnostics
    );
    assert!(f3_out.model.functions.contains_key("hal_gpio__make_pin"));
    assert!(f3_out
        .model
        .functions
        .contains_key("hal_gpio__Pin__set_mode"));
    assert!(f3_out.model.functions.contains_key("hal_gpio__Pin__write"));
    assert!(f3_out.model.functions.contains_key("hal_gpio__Pin__read"));
    assert!(f3_out
        .model
        .mmio_instances
        .contains_key("chip_stm32f334__GPIOA"));
}

#[test]
fn grouped_imports_pub_import_reexports_and_ambiguity() {
    let base = std::env::temp_dir().join(format!("zev_pub_reexport_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(base.join("pkg")).unwrap();

    std::fs::write(
        base.join("pkg/a.zev"),
        r#"
pub struct ItemA { u8 v; }
pub fn helper_a() -> u8 { return 1u8; }
"#,
    )
    .unwrap();
    std::fs::write(
        base.join("pkg/b.zev"),
        r#"
pub struct ItemB { u16 v; }
pub fn helper_b() -> u16 { return 2u16; }
"#,
    )
    .unwrap();
    std::fs::write(
        base.join("pkg/dup.zev"),
        r#"
pub fn helper_a() -> u8 { return 9u8; }
"#,
    )
    .unwrap();
    std::fs::write(
        base.join("pkg/facade.zev"),
        r#"
pub import pkg.{a, b};
"#,
    )
    .unwrap();
    std::fs::write(
        base.join("pkg/ambig.zev"),
        r#"
pub import pkg.{a, dup};
"#,
    )
    .unwrap();

    // Clean diamond + grouped + pub import re-export
    let ok_src = r#"
import pkg.facade as p;
import pkg.a as a_direct;

fn main() {
    p.ItemA a1 = p.ItemA { v: p.helper_a() };
    a_direct.ItemA a2 = a1;
    p.ItemB b1 = p.ItemB { v: p.helper_b() };
    u32 _sum = (a2.v as u32) + (b1.v as u32);
}
"#;
    let ok_prog = zev_parser::parse_str(ok_src).expect("parse ok_prog");
    let opts = AnalyzeOptions {
        include_dirs: vec![base.clone()],
        ..AnalyzeOptions::default()
    };
    let ok_out = zev_sema::analyze_with_options(&[ok_prog], &opts);
    assert!(
        ok_out.diagnostics.is_empty(),
        "diags: {:?}",
        ok_out.diagnostics
    );

    // Ambiguous pub import must report ModuleError
    let ambig_src = r#"
import pkg.ambig as bad;

fn main() {
    u8 _x = bad.helper_a();
}
"#;
    let ambig_prog = zev_parser::parse_str(ambig_src).expect("parse ambig_prog");
    let ambig_out = zev_sema::analyze_with_options(&[ambig_prog], &opts);
    assert!(
        ambig_out
            .diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::ModuleError),
        "expected ambiguous re-export ModuleError, got: {:?}",
        ambig_out.diagnostics
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn same_module_impl_ownership_visibility_and_generic_specialization() {
    let base = std::env::temp_dir().join(format!("zev_impl_rules_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(base.join("modlib")).unwrap();

    std::fs::write(
        base.join("modlib/inner.zev"),
        r#"
pub struct Widget { u8 id; }

impl Widget {
    pub fn make(u8 id) -> Widget {
        return Widget { id: id };
    }

    pub fn id(self) -> u8 {
        return self.id;
    }

    fn secret(self) { }
}
"#,
    )
    .unwrap();
    std::fs::write(
        base.join("modlib/facade.zev"),
        r#"
pub import modlib.inner;
"#,
    )
    .unwrap();
    let opts = AnalyzeOptions {
        include_dirs: vec![base.clone()],
        ..AnalyzeOptions::default()
    };

    // 1a. Qualified foreign-type `impl m.Widget` is rejected at parse time
    let qualified_foreign_impl_src = r#"
import modlib.facade as m;

impl m.Widget {
    pub fn ext(self) -> u8 { return self.id; }
}

fn main() {}
"#;
    let pe = zev_parser::parse_str(qualified_foreign_impl_src)
        .expect_err("qualified `impl m.Widget` must be rejected by parser");
    assert!(
        pe.message.contains("foreign module type"),
        "unexpected parse error: {pe}"
    );

    // 1b. Unqualified `impl Widget` for a struct not defined in this module must fail ModuleError
    let foreign_impl_src = r#"
import modlib.facade as m;

impl Widget {
    pub fn ext(self) -> u8 { return 0u8; }
}

fn main() {}
"#;
    let p1 = zev_parser::parse_str(foreign_impl_src).unwrap();
    let o1 = zev_sema::analyze_with_options(&[p1], &opts);
    assert!(
        o1.diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::ModuleError),
        "expected foreign-type impl rejection, got: {:?}",
        o1.diagnostics
    );

    // 2. Private method cannot be called from another module
    let priv_call_src = r#"
import modlib.inner as m;

fn main() {
    m.Widget w = m.Widget.make(1u8);
    w.secret();
}
"#;
    let p2 = zev_parser::parse_str(priv_call_src).unwrap();
    let o2 = zev_sema::analyze_with_options(&[p2], &opts);
    assert!(
        o2.diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::ModuleError),
        "expected private method ModuleError, got: {:?}",
        o2.diagnostics
    );

    // 3. `self` allowed on const, `*self` rejected on const, generic + specialized impl allowed
    let good_impl_src = r#"
struct Pair<T, usize N> {
    [N]T  vals;
    usize len;
}

impl<T, usize N> Pair<T, N> {
    pub fn push(*self, T v) {
        if (self.len < N) {
            self.vals[self.len] = v;
            self.len += 1;
        }
    }
}

impl<usize N> Pair<f32, N> {
    pub fn sum(*self) -> f32 {
        f32 s = 0.0f32;
        for (usize i in 0..self.len) {
            s += self.vals[i];
        }
        return s;
    }
}

struct Tag { u8 code; }
impl Tag {
    pub fn get(self) -> u8 { return self.code; }
    pub fn bump(*self) { self.code += 1u8; }
}

const Tag C_TAG = Tag { code: 42u8 };

fn main() {
    u8 _c = C_TAG.get();
    Pair<f32, 3> p = Pair<f32, 3> { vals: [0.0f32, 0.0f32, 0.0f32], len: 0 };
    p.push(1.0f32);
    p.push(2.0f32);
    f32 _s = p.sum();
}
"#;
    expect_ok(good_impl_src);

    let mut_self_on_const = r#"
struct Tag { u8 code; }
impl Tag {
    pub fn bump(*self) { self.code += 1u8; }
}
const Tag C_TAG = Tag { code: 42u8 };
fn main() {
    C_TAG.bump();
}
"#;
    expect_err_kind(mut_self_on_const, DiagnosticKind::ImmutableAssignment);

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn const_aggregate_validation_and_mutability_guards() {
    // Non-constant struct/array initializers rejected
    let non_const_arr = r#"
fn runtime_val() -> u16 { return 7u16; }
const [2]u16 BAD = [1u16, runtime_val()];
fn main() {}
"#;
    let p_bad = zev_parser::parse_str(non_const_arr).unwrap();
    let o_bad = zev_sema::analyze_program(&p_bad);
    assert!(o_bad.is_err(), "expected non-const array init error");

    // `&CONST_STRUCT.field` and `&CONST_ARR[0]` rejected as ImmutableAssignment
    let addr_field = r#"
struct Pin { u8 port; }
const Pin P = Pin { port: 1u8 };
fn main() {
    *u8 _ptr = &P.port;
}
"#;
    expect_err_kind(addr_field, DiagnosticKind::ImmutableAssignment);

    let addr_elem = r#"
const [2]u16 LUT = [10u16, 20u16];
fn main() {
    *u16 _ptr = &LUT[0];
}
"#;
    expect_err_kind(addr_elem, DiagnosticKind::ImmutableAssignment);

    // Read-only indexing, slicing, and passing const arrays allowed
    let good_ro = r#"
struct Pin { u8 port; }
const Pin P = Pin { port: 1u8 };
const [3]u16 LUT = [10u16, 20u16, 30u16];
fn take([]u16 s) -> u16 { return s[0]; }
fn main() {
    u8 _p = P.port;
    u16 _x = LUT[1];
    []u16 _s = LUT[0..2];
    u16 _y = take(LUT);
}
"#;
    expect_ok(good_ro);
}

#[test]
fn board_module_and_hal_target_binding_across_four_targets() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let demo_path = root.join("examples/portable_controller_demo.zev");
    let demo_src = std::fs::read_to_string(&demo_path).expect("read portable_controller_demo.zev");
    let demo_label = demo_path.display().to_string();

    let targets: [(&str, Option<&str>, Option<&str>); 5] = [
        ("host_sim", None, None),
        (
            "lm3s6965_evb",
            Some("hal/lm3s6965"),
            Some("bsp/lm3s6965_evb"),
        ),
        (
            "nucleo_f334r8",
            Some("hal/stm32f3"),
            Some("bsp/nucleo_f334r8"),
        ),
        (
            "esp32c6_pico",
            Some("hal/esp32c6"),
            Some("bsp/esp32c6_pico"),
        ),
        (
            "rpi_pico_w",
            Some("hal/rp2040"),
            Some("bsp/rpi_pico_w_board"),
        ),
    ];

    for (label, hal_dir, board_module) in targets {
        let prog = zev_parser::parse_str_named(&demo_src, &demo_label)
            .unwrap_or_else(|e| panic!("parse failed for {label}: {e:#?}"));
        let aopts = AnalyzeOptions {
            include_dirs: vec![root.clone()],
            hal_dir: hal_dir.map(str::to_string),
            board_module: board_module.map(str::to_string),
            ..AnalyzeOptions::default()
        };
        let out = zev_sema::analyze_and_monomorphize_named_programs(
            &[(&prog, Some(demo_label.as_str()))],
            &aopts,
        );
        assert!(
            out.diagnostics.is_empty(),
            "sema failed for {label}: {:?}",
            out.diagnostics
        );
    }

    // Diamond-import deduplication: `import board;` and `import bsp.nucleo_f334r8 as bsp;`
    let diamond_src = r#"
import board;
import bsp.nucleo_f334r8 as bsp;

fn main() {
    board.init();
    bsp.set_led(true);
    assert(board.BOARD_NAME.len == bsp.BOARD_NAME.len);
}
"#;
    let diamond_prog =
        zev_parser::parse_str_named(diamond_src, "diamond.zev").expect("parse diamond import");
    let diamond_opts = AnalyzeOptions {
        include_dirs: vec![root],
        hal_dir: Some("hal/stm32f3".to_string()),
        board_module: Some("bsp/nucleo_f334r8".to_string()),
        ..AnalyzeOptions::default()
    };
    let diamond_out =
        zev_sema::analyze_and_monomorphize_named_programs(&[(&diamond_prog, None)], &diamond_opts);
    assert!(
        diamond_out.diagnostics.is_empty(),
        "diamond deduplication failed: {:?}",
        diamond_out.diagnostics
    );

    // Embedded fallback resolution with no workspace include_dirs
    let emb_src = r#"
import board;

fn main() {
    board.init();
    board.set_led(true);
    bool _b = board.read_button();
}
"#;
    for (label, hal_dir, board_module) in [
        ("emb_sim", None, None),
        ("emb_esp32c6", Some("hal/esp32c6"), Some("bsp/esp32c6_pico")),
        (
            "emb_rp2040",
            Some("hal/rp2040"),
            Some("bsp/rpi_pico_w_board"),
        ),
    ] {
        let p = zev_parser::parse_str_named(emb_src, "embedded.zev")
            .unwrap_or_else(|e| panic!("parse embedded {label}: {e:#?}"));
        let o = AnalyzeOptions {
            include_dirs: Vec::new(),
            hal_dir: hal_dir.map(str::to_string),
            board_module: board_module.map(str::to_string),
            ..AnalyzeOptions::default()
        };
        let res = zev_sema::analyze_and_monomorphize_named_programs(&[(&p, None)], &o);
        assert!(
            res.diagnostics.is_empty(),
            "embedded fallback {label} failed: {:?}",
            res.diagnostics
        );
    }
}

#[test]
fn enforces_32_task_static_scheduler_limit_e0027() {
    fn gen_spawns(count: usize, with_main_supervisor: bool) -> String {
        let mut s = String::from(
            "signal sig;\ntask worker(*signal p) { loop { p.wait(); } }\nfn main() {\n",
        );
        for _ in 0..count {
            s.push_str("    spawn worker(&sig);\n");
        }
        if with_main_supervisor {
            s.push_str("    loop {\n        delay(10ms);\n    }\n");
        }
        s.push_str("}\n");
        s
    }

    // 32 spawned workers (no main supervisor): 32 slots -> valid
    let ok32 = check_src(&gen_spawns(32, false));
    assert!(ok32.is_ok(), "32 tasks should fit: {:?}", ok32.err());

    // 31 spawned workers + main supervisor loop: 32 slots -> valid
    let ok31_sup = check_src(&gen_spawns(31, true));
    assert!(
        ok31_sup.is_ok(),
        "31 tasks + main supervisor should fit: {:?}",
        ok31_sup.err()
    );

    // 33 spawned workers -> E0027
    let bad33 = check_src(&gen_spawns(33, false)).unwrap_err();
    assert!(
        bad33
            .iter()
            .any(|d| d.kind == DiagnosticKind::InvalidSpawnContext && d.message.contains("32")),
        "expected E0027 on 33 tasks: {:?}",
        bad33
    );

    // 32 spawned workers + main supervisor loop = 33 -> E0027
    let bad32_sup = check_src(&gen_spawns(32, true)).unwrap_err();
    assert!(
        bad32_sup
            .iter()
            .any(|d| d.kind == DiagnosticKind::InvalidSpawnContext && d.message.contains("32")),
        "expected E0027 on 32 tasks + main supervisor: {:?}",
        bad32_sup
    );
}

#[test]
fn wait_queue_safety_and_std_sync_sema() {
    let ok = r#"
import std.sync as sync;

wait_queue raw_wq;
sync.Semaphore sem;
sync.EventFlags flags;
sync.Latch gate;
sync.Barrier<3> barrier;

irq IRQ_W = 4;

isr w_isr(*wait_queue w, *sync.Semaphore s) {
    w.wake_one(boost: @high);
    w.wake_all();
    s.release();
}

task waiter(*wait_queue w, *sync.Semaphore s, *sync.EventFlags f, *sync.Latch g, *sync.Barrier<3> b) {
    loop {
        critical {
            w.prepare_wait();
        }
        w.cancel_wait();
        critical {
            w.prepare_wait();
        }
        w.commit_wait(timeout: 20ms) catch {};
        s.acquire();
        u32 _bits = f.wait_any(0x1u32);
        g.wait();
        bool _leader = b.wait();
    }
}

fn main() {
    sem.init(1u32);
    IRQ_W.attach(w_isr(&raw_wq, &sem));
    spawn waiter(&raw_wq, &sem, &flags, &gate, &barrier) @normal;
    spawn waiter(&raw_wq, &sem, &flags, &gate, &barrier) @normal;
    spawn waiter(&raw_wq, &sem, &flags, &gate, &barrier) @normal;
}
"#;
    let _ = expect_ok(ok);

    // prepare_wait outside critical -> InvalidOperation (E0024)
    expect_err_kind(
        "wait_queue wq; task t() { loop { wq.prepare_wait(); wq.commit_wait(); } } fn main() { spawn t(); }",
        DiagnosticKind::InvalidOperation,
    );
    // prepare_wait / cancel_wait / commit_wait inside ISR -> YieldInIsr (E0005)
    expect_err_kind(
        "wait_queue wq; irq Q = 1; isr h() { critical { wq.prepare_wait(); } } fn main() { Q.attach(h); }",
        DiagnosticKind::YieldInIsr,
    );
    expect_err_kind(
        "wait_queue wq; irq Q = 1; isr h() { wq.cancel_wait(); } fn main() { Q.attach(h); }",
        DiagnosticKind::YieldInIsr,
    );
    expect_err_kind(
        "wait_queue wq; irq Q = 1; isr h() { wq.commit_wait(); } fn main() { Q.attach(h); }",
        DiagnosticKind::YieldInIsr,
    );
    // commit_wait inside critical -> YieldInCritical (E0004), inside noyield -> YieldInNoYield (E0003)
    expect_err_kind(
        "wait_queue wq; task t() { loop { critical { wq.prepare_wait(); wq.commit_wait(); } } } fn main() { spawn t(); }",
        DiagnosticKind::YieldInCritical,
    );
    expect_err_kind(
        "wait_queue wq; task t() { loop { noyield { wq.commit_wait(); } } } fn main() { spawn t(); }",
        DiagnosticKind::YieldInNoYield,
    );
}

#[test]
fn while_cond_soft_yielding_vs_noyield_wrapper() {
    let in_isr = r#"
watch<u32> reg = 0;
irq Q = 1;
fn poll_reg(*watch<u32> w) {
    while (w.read() == 0) {}
}
isr h() {
    poll_reg(&reg);
}
fn main() {
    Q.attach(h);
}
"#;
    expect_err_kind(in_isr, DiagnosticKind::YieldInIsr);

    let in_critical = r#"
watch<u32> reg = 0;
fn poll_reg(*watch<u32> w) {
    while (w.read() == 0) {}
}
task t() {
    loop {
        critical {
            poll_reg(&reg);
        }
        sys.delay(1ms);
    }
}
fn main() {
    spawn t();
}
"#;
    expect_err_kind(in_critical, DiagnosticKind::YieldInCritical);

    let soft_ok = r#"
watch<u32> reg = 0;
fn poll_reg(*watch<u32> w) {
    while (w.read() == 0) {}
}
task t() {
    loop {
        poll_reg(&reg);
        sys.delay(1ms);
    }
}
fn main() {
    spawn t();
}
"#;
    let soft_model = expect_ok(soft_ok);
    assert_eq!(soft_model.fn_effect("poll_reg"), FnEffect::SoftYielding);

    let ok = r#"
watch<u32> reg = 0;
irq Q = 1;
fn poll_reg(*watch<u32> w) {
    noyield {
        while (w.read() == 0) {}
    }
}
isr h() {
    poll_reg(&reg);
}
task t() {
    loop {
        critical {
            poll_reg(&reg);
        }
        sys.delay(1ms);
    }
}
fn main() {
    Q.attach(h);
    spawn t();
}
"#;
    let model = expect_ok(ok);
    assert_eq!(model.fn_effect("poll_reg"), FnEffect::NonYielding);
}

#[test]
fn const_shift_range_validation() {
    for bad in [
        "fn main() { u32 x = 1u32 << 32; }",
        "fn main() { u8 x = 0x80u8 >> 8; }",
        "fn main() { u32 x = 1 << 32; }",
        "const u16 BAD = 1 << 16; fn main() {}",
        "fn main() { u32 x = 1; x <<= 32; }",
        "fn main() { i32 x = -8; x >>= -1; }",
        "fn main() { u64 x = 1 << 64; }",
    ] {
        expect_err_kind(bad, DiagnosticKind::InvalidOperation);
    }

    // Valid constant shifts across widths + local variable shadowing a top-level const
    let valid = r#"
const u32 N = 32;
const u64 WIDE = 1 << 40;

fn main() {
    u8 a = 1 << 7;
    u32 b = 1 << 31;
    u64 c = WIDE | (1u64 << 63);
    u32 N = 4;
    u32 shadowed_ok = 1u32 << N;
    u32 shadowed_untyped = 1 << N;
    u32 _sum = (a as u32) + b + (c as u32) + shadowed_ok + shadowed_untyped;
}
"#;
    expect_ok(valid);
}

#[test]
fn integer_conversion_portability_ranges_and_widening_sema() {
    // 1. Positive: lossless widening, wire-endian <-> native, `usize`/`isize` portable subsets,
    //    `i8::MIN`, and destination-widened shifts.
    expect_ok(
        r#"
        packed struct Pkt { u16be val; }
        fn main() {
            u8 a = 200;
            u8 b = 100;
            u16 w = a + b;
            u32 bit = a << 8;
            u32 mask_hi = 0x1234_56FFu32 & ~a;
            i8 mn = -128;
            i8 mn_s = -128i8;
            i16 neg_w = -a;
            u64 wide_u = (0x10usize) as u64;
            usize p_u = a;
            p_u = w;
            u64 from_usize = p_u;
            i64 from_isize = (-32768i16);
            isize p_i = -32768;
            i64 p_i64 = p_i;
            p_i = a;
            u16 native = 12;
            Pkt pkt = Pkt { val: native };
            u16 back = pkt.val;
            u32 _keep = w + bit + mask_hi + from_usize as u32 + (mn as u32) + (mn_s as u32) + (neg_w as u32) + (wide_u as u32) + (from_isize as u32) + (p_i64 as u32) + back;
        }
        "#,
    );

    // 2. Negative: narrowing, signed <-> unsigned, non-portable `usize`/`isize`, out-of-range literals,
    //    mixed-signedness comparisons/binary ops, and non-widened `as` shift overflow (`E0024`).
    for bad in [
        "fn main() { u8 x = 300; }",
        "fn main() { u8 x = -1; }",
        "fn main() { i8 x = 128; }",
        "fn main() { i8 x = -129; }",
        "fn main() { u8 x = 300u8; }",
        "fn main() { i8 x = -129i8; }",
        "fn main() { u16 a = 1; u8 b = a; }",
        "fn main() { i32 a = -1; u32 b = a; }",
        "fn main() { u32 a = 1; i32 b = a; }",
        "fn main() { i16 a = 1; i32 b = 2; u32 c = a + b; }",
        "fn main() { u32 a = 1; i32 b = 2; bool c = a < b; }",
        "fn main() { u32 a = 1; i32 b = 2; u32 c = a + b; }",
        "fn main() { u32 a = 1; usize b = a; }",
        "fn main() { usize a = 1; u32 b = a; }",
        "fn main() { usize a = 1; i64 b = a; }",
        "fn main() { u16 a = 1; isize b = a; }",
        "fn main() { i32 a = 1; isize b = a; }",
        "fn main() { isize a = 1; usize b = a; }",
        "packed struct P { u16be x; } fn main() { P p = P { x: 1 }; u16le q = 2; bool _b = p.x == q; }",
        "fn main() { u8 a = 200; u32 w = 0; w = (a + 300) as u32; }",
    ] {
        expect_err_kind(bad, DiagnosticKind::TypeMismatch);
    }

    for bad_shift in [
        "fn main() { u8 a = 200; u32 x = (a << 8) as u32; }",
        "fn main() { u8 a = 200; u8 x = a << 8; }",
    ] {
        expect_err_kind(bad_shift, DiagnosticKind::InvalidOperation);
    }

    // 3. Target pointer-width range checks on `usize`/`isize` literals and constant expressions
    //    (locals, `const` declarations, and `<usize N>` generic arguments), while typed conversions
    //    remain target-portable (`u8`/`u16` -> `usize` -> `u64`, checked above).
    let check_with_target = |src: &str, ptr_bits: u32, target_name: &str| {
        let prog = zev_parser::parse_str(src)
            .unwrap_or_else(|e| panic!("test source failed to parse:\n{e}\n---\n{src}"));
        let opts = AnalyzeOptions {
            target_ptr_bits: ptr_bits,
            target_name: target_name.to_string(),
            ..Default::default()
        };
        analyze_and_monomorphize_named_programs(&[(&prog, None)], &opts)
    };

    // Host (default 64-bit): accepts 100_000 and 5_000_000_000 across locals, consts, and generics.
    for ok_host in [
        "const usize CAP = 100000; fn take<usize N>() -> usize { return N; } fn main() { usize n = 100000; usize c = CAP; usize g = take<100000>(); usize _sum = n + c + g; }",
        "const usize BIG = 5_000_000_000; const isize S = -5_000_000_000; fn take<usize N>() -> usize { return N; } fn main() { usize n = 5_000_000_000; usize g = take<5_000_000_000>(); usize gc = take<BIG>(); isize sn = S; usize _u = n + g + gc; isize _si = sn; }",
    ] {
        let out = check_with_target(ok_host, 64, "host");
        assert!(
            out.diagnostics.is_empty(),
            "expected 64-bit host program to succeed, got:\n{:?}",
            out.diagnostics
        );
    }

    // 32-bit board: accepts 100_000 across locals, consts, generics; rejects 5_000_000_000 naming target & 32-bit.
    let ok_32 = "const usize CAP = 100000; fn take<usize N>() -> usize { return N; } fn main() { usize n = 100000; usize g = take<CAP>(); usize _x = n + g; }";
    let out_32_ok = check_with_target(ok_32, 32, "nucleo_f334r8");
    assert!(
        out_32_ok.diagnostics.is_empty(),
        "expected 32-bit board to accept 100000, got:\n{:?}",
        out_32_ok.diagnostics
    );
    for bad_32 in [
        "fn main() { usize n = 5_000_000_000; }",
        "const usize CAP = 5_000_000_000; fn main() {}",
        "fn main() { isize s = -2_147_483_649; }",
        "fn take<usize N>() -> usize { return N; } fn main() { usize g = take<5_000_000_000>(); }",
        "const usize CAP = 5_000_000_000; fn take<usize N>() -> usize { return N; } fn main() { usize g = take<CAP>(); }",
    ] {
        let out = check_with_target(bad_32, 32, "nucleo_f334r8");
        assert!(
            !out.diagnostics.is_empty(),
            "expected 32-bit target error for:\n{bad_32}"
        );
        let msg = out.diagnostics[0].to_string();
        assert!(
            msg.contains("32-bit") && msg.contains("nucleo_f334r8"),
            "error must name target & 32-bit pointer width:\n{msg}"
        );
    }

    // AVR board (16-bit): accepts 65535; rejects 70000, 65536, 32768, -32769 naming target & 16-bit.
    let ok_16 = "const usize CAP = 65535; const isize S = -32768; fn take<usize N>() -> usize { return N; } fn main() { usize n = 65535; usize g = take<CAP>(); isize s = S; usize _x = n + g; isize _y = s; }";
    let out_16_ok = check_with_target(ok_16, 16, "arduino_uno");
    assert!(
        out_16_ok.diagnostics.is_empty(),
        "expected 16-bit AVR board to accept 65535, got:\n{:?}",
        out_16_ok.diagnostics
    );
    for bad_16 in [
        "fn main() { usize a = 65536; }",
        "fn main() { usize a = 70000; }",
        "fn main() { isize a = 32768; }",
        "fn main() { isize a = -32769; }",
        "const usize CAP = 70000; fn main() {}",
        "const isize S = 32768; fn main() {}",
        "fn take<usize N>() -> usize { return N; } fn main() { usize g = take<70000>(); }",
        "const usize CAP = 70000; fn take<usize N>() -> usize { return N; } fn main() { usize g = take<CAP>(); }",
    ] {
        let out = check_with_target(bad_16, 16, "arduino_uno");
        assert!(
            !out.diagnostics.is_empty(),
            "expected 16-bit AVR error for:\n{bad_16}"
        );
        let msg = out.diagnostics[0].to_string();
        assert!(
            msg.contains("16-bit") && msg.contains("arduino_uno"),
            "error must name target & 16-bit pointer width:\n{msg}"
        );
    }
}

#[test]
fn watch_single_writer_and_guarded_critical_cell_semantics() {
    // 1. Positive: boot-phase `main()` initialization + 1 writer task + multiple readers,
    //    plus unrelated `irq.attach` before boot-phase `publish()`.
    let ok_watch = expect_ok(
        r#"
        watch<u32> telemetry = 0;
        signal ping;
        irq IRQ_TEST = 1;

        isr ping_isr(*signal s) {
            s.notify();
        }

        fn boot_init_helper() {
            telemetry.publish(1);
        }

        task writer() {
            u32 seq = 10;
            loop every 5ms {
                seq += 1;
                telemetry.publish(seq);
            }
        }

        task reader_a(*watch<u32> w) {
            loop {
                u32 _v = w.wait_change();
            }
        }

        task reader_b(*watch<u32> w) {
            loop every 10ms {
                u32 _v = w.read();
            }
        }

        fn main() {
            telemetry.publish(0);
            boot_init_helper();
            IRQ_TEST.attach(ping_isr(&ping));
            telemetry.publish(2); // Still before `spawn`, and no ISR publishes to `telemetry`
            spawn writer() @high;
            spawn reader_a(&telemetry) @normal;
            spawn reader_b(&telemetry) @low;
        }
        "#,
    );
    assert!(ok_watch.primitives.contains_key("telemetry"));

    // 2. Negative (`E0002`): two tasks publishing to the same `watch<T>`.
    expect_err_kind(
        r#"
        watch<u32> w = 0;
        task t1(*watch<u32> pw) { loop { pw.publish(1); delay(5ms); } }
        task t2(*watch<u32> pw) { loop { pw.publish(2); delay(5ms); } }
        fn main() {
            spawn t1(&w) @normal;
            spawn t2(&w) @normal;
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // 3. Negative (`E0002`): two spawned instances of the same task writing one `watch<T>`.
    expect_err_kind(
        r#"
        watch<u32> w = 0;
        task t() { loop { w.publish(1); delay(5ms); } }
        fn main() {
            spawn t() @high;
            spawn t() @normal;
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // 4. Negative (`E0002`): task + ISR publishing to the same `watch<T>`.
    expect_err_kind(
        r#"
        watch<u32> w = 0;
        irq IRQ_T = 2;
        isr w_isr(*watch<u32> pw) { pw.publish(1); }
        task t(*watch<u32> pw) { loop { pw.publish(2); delay(5ms); } }
        fn main() {
            IRQ_T.attach(w_isr(&w));
            spawn t(&w) @normal;
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // 5. Negative (`E0002`): `main()` post-spawn write + task write.
    expect_err_kind(
        r#"
        watch<u32> w = 0;
        task t(*watch<u32> pw) { loop { pw.publish(2); delay(5ms); } }
        fn main() {
            spawn t(&w) @normal;
            loop {
                w.publish(3);
                delay(10ms);
            }
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // 6. Negative (`E0002`): `main()` write after `irq.attach` + same ISR publishing.
    expect_err_kind(
        r#"
        watch<u32> w = 0;
        irq IRQ_T = 2;
        isr w_isr(*watch<u32> pw) { pw.publish(1); }
        fn main() {
            IRQ_T.attach(w_isr(&w));
            w.publish(2);
        }
        "#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // 7. Negative (`E0024`): `guarded<T>` `.get()`/`.set()` outside `critical { ... }`,
    //    including inside helpers called from `critical`, ISRs, and inside `wait { ... }`.
    for bad_guarded in [
        r#"
        guarded<u32> g = 0;
        fn main() { u32 _x = g.get(); }
        "#,
        r#"
        guarded<u32> g = 0;
        fn main() { g.set(1); }
        "#,
        r#"
        guarded<u32> g = 0;
        task t() { loop { g.set(g.get() + 1); delay(5ms); } }
        fn main() { spawn t() @normal; }
        "#,
        r#"
        guarded<u32> g = 0;
        irq IRQ_G = 1;
        isr g_isr() { g.set(1); }
        fn main() { IRQ_G.attach(g_isr); }
        "#,
        r#"
        guarded<u32> g = 0;
        fn raw_set(u32 v) { g.set(v); }
        fn main() { critical { raw_set(1); } }
        "#,
        r#"
        guarded<u32> g = 0;
        task t() {
            loop {
                wait {
                    u32 v = g.get() => { delay(1ms); }
                    timeout 5ms => {}
                }
            }
        }
        fn main() { spawn t() @normal; }
        "#,
    ] {
        expect_err_kind(bad_guarded, DiagnosticKind::InvalidOperation);
    }

    // 8. Negative (`E0033`): escaping ISR `state` pointer via `guarded<*u32>.set(&x)` in `critical`.
    expect_err_kind(
        r#"
        guarded<*u32> leak_g;
        irq IRQ_G = 1;
        isr bad() {
            state u32 x = 0;
            critical { leak_g.set(&x); }
        }
        fn main() { IRQ_G.attach(bad); }
        "#,
        DiagnosticKind::InvalidIsrState,
    );

    // 9. Positive: multi-writer `guarded<T>` accessed safely inside `critical { ... }`
    //    across multiple tasks and an ISR.
    let ok_guarded = expect_ok(
        r#"
        guarded<u32> shared_acc = 10;
        wait_queue shared_wq;
        irq IRQ_G = 2;

        isr bump_isr() {
            critical { shared_acc.set(shared_acc.get() + 1); }
            shared_wq.wake_all();
        }

        task worker_a() {
            loop every 5ms {
                critical { shared_acc.set(shared_acc.get() + 2); }
                shared_wq.wake_one();
            }
        }

        task worker_b() {
            loop {
                bool ok = false;
                critical {
                    if (shared_acc.get() >= 20) {
                        shared_acc.set(0);
                        ok = true;
                    } else {
                        shared_wq.prepare_wait();
                    }
                }
                if (!ok) {
                    shared_wq.commit_wait();
                }
            }
        }

        fn main() {
            IRQ_G.attach(bump_isr);
            spawn worker_a() @high;
            spawn worker_b() @normal;
        }
        "#,
    );
    assert!(ok_guarded.primitives.contains_key("shared_acc"));
}

#[test]
fn syntactic_generics_comparison_confusion_advises_parentheses() {
    for bad in [
        r#"
        fn check_pair(bool a, bool b) {}
        fn main() {
            u32 a = 1;
            u32 b = 2;
            u32 c = 3;
            u32 d = 4;
            check_pair(a < b, c > (d));
        }
        "#,
        r#"
        struct Pair { u32 a; }
        fn check_pair(bool a, bool b) {}
        fn main() {
            Pair s = Pair { a: 1 };
            u32 b = 2;
            u32 c = 3;
            u32 d = 4;
            check_pair(s.a < b, c > (d));
        }
        "#,
        r#"
        const u32 A = 1;
        fn check_pair(bool a, bool b) {}
        fn main() {
            u32 b = 2;
            u32 c = 3;
            u32 d = 4;
            check_pair(A < b, c > (d));
        }
        "#,
    ] {
        let diags = expect_err_kind(bad, DiagnosticKind::GenericInstantiationError);
        assert!(
            diags.iter().any(|d| d
                .note
                .as_deref()
                .is_some_and(|n| n.contains("parenthesize") && n.contains("(a < b)"))),
            "expected parenthesization note for:\n{bad}\ngot:\n{diags:?}"
        );
    }

    expect_ok(
        r#"
        fn check_pair(bool a, bool b) {}
        fn main() {
            u32 a = 1;
            u32 b = 2;
            u32 c = 3;
            u32 d = 4;
            check_pair((a < b), (c > (d)));
        }
        "#,
    );
}

#[test]
fn retired_inbox_try_post_is_rejected_with_post_suggestion() {
    let diags = expect_err_kind(
        r#"
        inbox<u32, 4> q;
        fn main() {
            q.try_post(1u32);
        }
        "#,
        DiagnosticKind::UndefinedSymbol,
    );
    assert!(
        diags
            .iter()
            .any(|d| d.message.contains("try_post")
                && d.message.contains(".post(")
                && d.message.contains(".try_send(")),
        "expected try_post removal diagnostic recommending .post(...) or .try_send(...), got: {diags:?}"
    );
}

#[test]
fn bare_const_expr_generic_args_and_defaulted_literal_calls_diagnostics() {
    // 1. Bare const expr in generic call args (`lim<K + 1>(1)`) advises parenthesizing to `lim<(K + 1)>(1)`,
    //    and non-generic comparisons (`f(a < b + 1, c > (d))`) still succeed.
    let bad_c = "\
const usize K = 3;
fn lim<usize N>(u32 x) -> u32 { return x + (N as u32); }
fn main() { u32 _y = lim<K + 1>(1); }
";
    let p_c = zev_parser::parse_str(bad_c).unwrap();
    let out_c = analyze_and_monomorphize_programs(&[p_c], &AnalyzeOptions::default());
    assert!(
        out_c.diagnostics.iter().any(|d| {
            d.kind == DiagnosticKind::GenericInstantiationError
                && d.message.contains("lim<(K + 1)>(1)")
                && d.message.contains("parenthes")
        }),
        "expected Fix C diagnostic, got: {:?}",
        out_c.diagnostics
    );
    assert!(
        out_c.diagnostics.iter().all(|d| !d
            .message
            .contains("cannot be referenced without concrete call arguments")),
        "unexpected bare-ref cascade: {:?}",
        out_c.diagnostics
    );

    // Multi-arg syntax recovery error case `lim<K + 1>(1, 2)` / `lim<K + 1, K + 2>(1)` also shows rewrite.
    let (_p_multi, errs_multi) = zev_parser::parse_str_named_recovering(
        "fn lim<usize A, usize B>(u32 x) -> u32 { return x; }\nfn main() { lim<1 + 2, 3 + 4>(5); }\n",
        "c_multi.zev",
    );
    assert!(
        errs_multi
            .iter()
            .any(|e| e.message.contains("lim<(1 + 2), (3 + 4)>(5)")),
        "expected multi-arg Fix C rewrite message, got: {errs_multi:?}"
    );

    // 2. Fix D: wrapper generic struct destination infers inner T from all-literal call,
    //    and literal overflow inside defaulted generic call suggests explicit type arguments.
    let box_ok = "\
struct Box<T> { T v; }
fn make_box<T>(T a, T b) -> Box<T> { return Box { v: a }; }
fn main() { Box<u32> _bx = make_box(3, 4); }
";
    let p_box = zev_parser::parse_str(box_ok).unwrap();
    let out_box = analyze_and_monomorphize_programs(&[p_box], &AnalyzeOptions::default());
    assert!(out_box.diagnostics.is_empty(), "{:?}", out_box.diagnostics);

    let ovf = "\
fn pick<T>(T a, T b) -> T { return a; }
fn main() { pick(5_000_000_000, 4); }
";
    let p_ovf = zev_parser::parse_str(ovf).unwrap();
    let out_ovf = analyze_and_monomorphize_programs(&[p_ovf], &AnalyzeOptions::default());
    assert!(
        out_ovf.diagnostics.iter().any(|d| {
            d.kind == DiagnosticKind::TypeMismatch
                && d.message.contains("defaulted to 'i32'")
                && d.message.contains("pick<i64>(5000000000, 4)")
        }),
        "expected Fix D overflow suggestion, got: {:?}",
        out_ovf.diagnostics
    );
}

#[test]
fn item1_function_pointer_soundness_rejects_blocking_and_fallible_routes_soft_to_noyield() {
    // 1. Hard-blocking targets rejected across locals, struct fields, arguments, returns, arrays, and generics,
    //    naming the shortest blocking chain and stating "not yet supported through function pointers".
    let blocking_cases = [
        (
            "local",
            r#"
            fn leaf_delay() { delay(10ms); }
            fn slow([]u8 a) -> bool { leaf_delay(); return true; }
            task sh() {
                fn([]u8) -> bool h = slow;
                loop { bool _ok = h("x"); delay(1ms); }
            }
            fn main() { spawn sh() @normal; }
            "#,
            "slow() -> leaf_delay() -> delay()",
        ),
        (
            "struct_field",
            r#"
            fn wait_tick() { delay(5ms); }
            struct Handler { fn([]u8) -> bool run; }
            fn main() {
                Handler _h = Handler { run: wait_tick_wrap };
            }
            fn wait_tick_wrap([]u8 _b) -> bool { wait_tick(); return true; }
            "#,
            "wait_tick_wrap() -> wait_tick() -> delay()",
        ),
        (
            "arg",
            r#"
            fn slow([]u8 a) -> bool { delay(10ms); return true; }
            fn reg(fn([]u8) -> bool _cb) {}
            fn main() { reg(slow); }
            "#,
            "slow() -> delay()",
        ),
        (
            "return",
            r#"
            fn slow([]u8 a) -> bool { delay(10ms); return true; }
            fn get_cb() -> fn([]u8) -> bool { return slow; }
            fn main() {}
            "#,
            "slow() -> delay()",
        ),
        (
            "generic",
            r#"
            struct Box<T> { T cb; }
            fn slow_u8(u8 _x) -> bool { delay(3ms); return true; }
            fn take<T>(fn(T) -> bool _cb) {}
            fn via_generic<T>(T _v) {
                fn(u8) -> bool _h = slow_u8;
            }
            fn main() {
                Box<fn(u8) -> bool> _b = Box<fn(u8) -> bool> { cb: slow_u8 };
                take<u8>(slow_u8);
                via_generic<u8>(1u8);
            }
            "#,
            "delay()",
        ),
    ];

    for (label, src, expected_chain) in blocking_cases {
        let p = zev_parser::parse_str(src).unwrap();
        let out = analyze_and_monomorphize_programs(&[p], &AnalyzeOptions::default());
        assert!(
            out.diagnostics.iter().any(|d| {
                d.kind == DiagnosticKind::YieldInNoYield
                    && d.message.contains("not yet supported through function pointers")
                    && d.note.iter().any(|n| n.contains(expected_chain))
            }),
            "case '{label}' expected blocking fn-ptr error containing chain '{expected_chain}', got: {:?}",
            out.diagnostics
        );
    }

    // 2. Fallible (`error` and `error E`) targets rejected across locals, struct fields, arguments, returns,
    //    casts to `*void`, and generic instantiations.
    let fallible_cases = [
        (
            "local_typed",
            r#"
            enum Err { Bad }
            fn f([]u8 _a) -> bool error Err { return true; }
            fn main() { fn([]u8) -> bool _h = f; }
            "#,
        ),
        (
            "local_untyped",
            r#"
            fn f() error { error; }
            fn main() { fn() _h = f; }
            "#,
        ),
        (
            "struct_field",
            r#"
            enum Err { Bad }
            struct Cmd { fn([]u8) -> bool run; }
            fn f([]u8 _a) -> bool error Err { return true; }
            fn main() { Cmd _c = Cmd { run: f }; }
            "#,
        ),
        (
            "argument",
            r#"
            enum Err { Bad }
            fn f(u8 _x) -> u8 error Err { return 1u8; }
            fn reg(fn(u8) -> u8 _cb) {}
            fn main() { reg(f); }
            "#,
        ),
        (
            "return_val",
            r#"
            enum Err { Bad }
            fn f(u8 _x) -> u8 error Err { return 1u8; }
            fn get_f() -> fn(u8) -> u8 { return f; }
            fn main() {}
            "#,
        ),
        (
            "array_elem",
            r#"
            enum Err { Bad }
            fn f() error Err { return; }
            fn ok() {}
            fn main() { [2]fn() _a = [ok, f]; }
            "#,
        ),
        (
            "generic_arg",
            r#"
            enum Err { Bad }
            fn f_u8(u8 x) -> u8 error Err { return x; }
            fn apply<T>(fn(T) -> T cb, T v) -> T { return cb(v); }
            fn main() { u8 _v = apply<u8>(f_u8, 1u8); }
            "#,
        ),
        (
            "generic_local",
            r#"
            enum Err { Bad }
            fn f_u8(u8 x) -> u8 error Err { return x; }
            fn wrap<T>(T _v) { fn(u8) -> u8 _h = f_u8; }
            fn main() { wrap<u8>(1u8); }
            "#,
        ),
        (
            "cast_void",
            r#"
            enum Err { Bad }
            fn f() error Err { return; }
            fn main() { *void _p = f as *void; }
            "#,
        ),
    ];

    for (label, src) in fallible_cases {
        let p = zev_parser::parse_str(src).unwrap();
        let out = analyze_and_monomorphize_programs(&[p], &AnalyzeOptions::default());
        assert!(
            !out.diagnostics.is_empty(),
            "case '{label}' must reject fallible function assigned/passed to infallible pointer type"
        );
    }

    // 3. Soft-yielding (`while` loop and `checkpoint()`) targets are accepted cleanly and routed to `noyield` clone.
    let soft_src = r#"
        fn soft_while(u32 n) -> u32 {
            u32 s = 0u32;
            u32 i = 0u32;
            while (i < n) {
                s += i;
                i += 1u32;
            }
            return s;
        }
        struct Ops { fn(u32) -> u32 work; }
        task sh() {
            fn(u32) -> u32 h = soft_while;
            Ops op = Ops { work: soft_while };
            loop {
                u32 _a = h(4u32);
                u32 _b = (op.work)(4u32);
                delay(1ms);
            }
        }
        fn main() { spawn sh() @normal; }
    "#;
    let p_soft = zev_parser::parse_str(soft_src).unwrap();
    let (model, diags) = analyze_and_monomorphize_programs(&[p_soft], &AnalyzeOptions::default())
        .into_result()
        .map(|m| (m, Vec::<zev_sema::Diagnostic>::new()))
        .unwrap_or_else(|d| panic!("soft-yielding fn pointer must pass sema cleanly: {d:?}"));
    assert!(diags.is_empty());
    assert!(
        model.1.soft_yielding_in_noyield.contains("soft_while"),
        "soft_while should be scheduled for __noyield clone generation: {:?}",
        model.1.soft_yielding_in_noyield
    );
}

#[test]
fn item2_missing_console_board_reports_no_console_input() {
    let tmp = std::env::temp_dir().join(format!(
        "zev_item2_no_console_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(tmp.join("bsp")).unwrap();
    std::fs::write(tmp.join("bsp/no_console.zev"), "pub fn init() {}\n").unwrap();
    let app_path = tmp.join("app.zev");
    let app_src = "import board;\nfn main() { board.init(); auto _c = board.console_open(); }\n";
    let prog = zev_parser::parse_str_named(app_src, app_path.to_str().unwrap()).unwrap();
    let aopts = AnalyzeOptions {
        include_dirs: vec![tmp.clone()],
        board_module: Some("bsp/no_console".to_string()),
        ..AnalyzeOptions::default()
    };
    let out =
        zev_sema::analyze_and_monomorphize_named_programs(&[(&prog, app_path.to_str())], &aopts);
    let _ = std::fs::remove_dir_all(&tmp);
    let rendered = out
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        rendered.contains("does not provide console input"),
        "expected missing console diagnostic, got:\n{rendered}"
    );
}

#[test]
fn item3_const_fnptr_rodata_table_and_fallible_fnptr_semantics() {
    let ok_src = r#"
enum Err { Bad, Denied }
struct Ctx { u32 acc; }
struct Cmd {
    []u8 name;
    fn(*Ctx, u32) -> u32 error Err run;
}

const fn(u32) -> u32 FWD_ADD = forward_add;
const fn(u32) -> u32 FWD_ADD_ADDR = &forward_add;
const fn(*Ctx, u32) -> u32 error Err DIRECT_CONST = cmd_ok;
const [2]Cmd COMMANDS = [
    Cmd { name: "ok", run: cmd_ok },
    Cmd { name: "soft", run: cmd_soft },
];

fn forward_add(u32 x) -> u32 { return x + 1u32; }

fn cmd_ok(*Ctx c, u32 v) -> u32 error Err {
    if (v == 0u32) { error Err.Bad; }
    c.acc += v;
    return c.acc;
}

fn cmd_soft(*Ctx c, u32 v) -> u32 error Err {
    u32 i = 0u32;
    while (i < v) {
        c.acc += 1u32;
        i += 1u32;
    }
    if (v == 99u32) { error Err.Denied; }
    return c.acc;
}

fn bare_err_fn(u8 x) error {
    if (x == 0u8) { error; }
}

fn propagate(*Ctx c, usize i, u32 v) -> u32 error Err {
    return COMMANDS[i].run(c, v) catch (Err e) { error e; };
}

fn main() {
    Ctx c = Ctx { acc: 0u32 };
    u32 _a = FWD_ADD(2u32);
    u32 _b = FWD_ADD_ADDR(3u32);
    u32 _d = DIRECT_CONST(&c, 5u32) or 0u32;
    u32 _e = COMMANDS[0].run(&c, 4u32) catch (Err err) { 0u32 };
    u32 _p = propagate(&c, 1, 3u32) or 1u32;
    fn(u8) error bfn = bare_err_fn;
    bfn(0u8) catch { };
}
"#;
    let p_ok = zev_parser::parse_str(ok_src).unwrap();
    let (model, diags) = analyze_and_monomorphize_programs(&[p_ok], &AnalyzeOptions::default())
        .into_result()
        .map(|m| (m, Vec::<zev_sema::Diagnostic>::new()))
        .unwrap_or_else(|d| panic!("item3 valid program must pass sema: {d:#?}"));
    assert!(diags.is_empty());
    assert!(model.1.soft_yielding_in_noyield.contains("cmd_soft"));

    // Unhandled fallible indirect calls (E0016)
    for (label, bad_src) in [
        (
            "var_unhandled",
            r#"
enum Err { Bad }
fn f() error Err { error Err.Bad; }
fn main() { fn() error Err fp = f; fp(); }
"#,
        ),
        (
            "table_unhandled",
            r#"
enum Err { Bad }
struct Cmd { fn() error Err run; }
const [1]Cmd TBL = [ Cmd { run: f } ];
fn f() error Err { error Err.Bad; }
fn main() { TBL[0].run(); }
"#,
        ),
    ] {
        let p = zev_parser::parse_str(bad_src).unwrap();
        let out = analyze_and_monomorphize_programs(&[p], &AnalyzeOptions::default());
        assert!(
            out.diagnostics
                .iter()
                .any(|d| matches!(d.kind, DiagnosticKind::UnhandledFallibleExpr)),
            "{label}: expected E0016 UnhandledFallibleExpr, got {:?}",
            out.diagnostics
        );
    }

    // Mismatched error_spec on const / var / void error (E0019)
    for (label, bad_src) in [
        (
            "const_mismatch_typed_vs_bare",
            r#"
enum Err { Bad }
fn f() error Err { return; }
const fn() error C = f;
fn main() { C() catch { }; }
"#,
        ),
        (
            "var_mismatch_err_enum",
            r#"
enum E1 { A }
enum E2 { B }
fn f() error E1 { error E1.A; }
fn main() { fn() error E2 fp = f; fp() catch (E2 _e) { }; }
"#,
        ),
        (
            "void_typed_error_spec",
            r#"
fn take(fn(u8) error void _cb) {}
fn main() { }
"#,
        ),
    ] {
        let p = zev_parser::parse_str(bad_src).unwrap();
        let out = analyze_and_monomorphize_programs(&[p], &AnalyzeOptions::default());
        assert!(
            out.diagnostics
                .iter()
                .any(|d| matches!(d.kind, DiagnosticKind::ErrorSpecMismatch)),
            "{label}: expected E0019 ErrorSpecMismatch, got {:?}",
            out.diagnostics
        );
    }
}

#[test]
fn pr19_rejects_unsupported_aggregate_equality_and_invalid_operator_type_pairs() {
    let cases = [
        (
            "struct_eq",
            "Point",
            "field",
            "struct Point { u32 x; u32 y; }\nfn main() { Point a = Point { x: 1, y: 2 }; Point b = Point { x: 1, y: 2 }; bool e = a == b; }",
        ),
        (
            "struct_slice_ne",
            "Item",
            "field",
            "struct Item { u8 id; }\nfn cmp([]Item a, []Item b) -> bool { return a != b; }\nfn main() {}",
        ),
        (
            "struct_array_eq",
            "Item",
            "field",
            "struct Item { u8 id; }\nfn main() { [2]Item a = [Item { id: 1 }, Item { id: 2 }]; bool e = a == a; }",
        ),
        (
            "float_slice_eq",
            "[]f32",
            "loop",
            "fn cmp([]f32 a, []f32 b) -> bool { return a == b; }\nfn main() {}",
        ),
        (
            "float_array_eq",
            "[2]f64",
            "loop",
            "fn main() { [2]f64 a = [1.0, 2.0]; bool e = a == a; }",
        ),
        (
            "nested_array_eq",
            "[2][2]u8",
            "loop",
            "fn main() { [2][2]u8 a = [[1u8, 2u8], [3u8, 4u8]]; bool e = a == a; }",
        ),
        (
            "ptr_slice_eq",
            "[]*u8",
            "loop",
            "fn cmp([]*u8 a, []*u8 b) -> bool { return a == b; }\nfn main() {}",
        ),
        (
            "enum_slice_eq",
            "[]Err",
            "loop",
            "enum Err { A }\nfn cmp([]Err a, []Err b) -> bool { return a == b; }\nfn main() {}",
        ),
        (
            "diff_len_array_eq",
            "[2]u8",
            "[3]u8",
            "fn main() { [2]u8 a = [1, 2]; [3]u8 b = [1, 2, 3]; bool e = a == b; }",
        ),
        (
            "diff_elem_slice_eq",
            "[]u8",
            "[]u16",
            "fn cmp([]u8 a, []u16 b) -> bool { return a == b; }\nfn main() {}",
        ),
        (
            "void_eq",
            "void",
            "compare",
            "fn v() {}\nfn main() { bool e = v() == v(); }",
        ),
        (
            "byte_ring_view_eq",
            "*byte_ring",
            "compare",
            "fn cmp(*byte_ring a, *byte_ring b) -> bool { return a == b; }\nfn main() {}",
        ),
        (
            "signal_val_eq",
            "signal",
            "compare",
            "signal s;\nfn main() { bool e = s == s; }",
        ),
        (
            "deref_void_rvalue",
            "*void",
            "dereference",
            "fn f(*void vp) { *vp; }\nfn main() {}",
        ),
        (
            "deref_void_lvalue",
            "*void",
            "dereference",
            "fn f(*void a, *void b) { *a = *b; }\nfn main() {}",
        ),
        (
            "switch_string",
            "switch",
            "str",
            "fn main() { switch (\"abc\") { _ => {} } }",
        ),
        (
            "switch_slice",
            "switch",
            "[]u8",
            "fn f([]u8 s) { switch (s) { _ => {} } }\nfn main() {}",
        ),
        (
            "switch_struct",
            "switch",
            "Point",
            "struct Point { u32 x; }\nfn main() { Point p = Point { x: 1 }; switch (p) { _ => {} } }",
        ),
        (
            "cast_struct_self",
            "Point",
            "cast",
            "struct Point { u32 x; }\nfn main() { Point p = Point { x: 1 }; Point q = p as Point; }",
        ),
        (
            "cast_slice_self",
            "[]u8",
            "cast",
            "fn f([]u8 s) { []u8 x = s as []u8; }\nfn main() {}",
        ),
        (
            "addr_of_rvalue",
            "&",
            "temporary",
            "fn main() { *i32 p = &5; }",
        ),
    ];

    for (label, needle1, needle2, src) in cases {
        let p = zev_parser::parse_str(src).unwrap_or_else(|e| panic!("{label} parse failed: {e}"));
        let out = analyze_and_monomorphize_programs(&[p], &AnalyzeOptions::default());
        assert!(
            !out.diagnostics.is_empty(),
            "{label}: expected Sema error, got OK"
        );
        let joined = out
            .diagnostics
            .iter()
            .map(|d| format!("{} {:?}", d.message, d.note))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains(needle1) && joined.contains(needle2),
            "{label}: expected diagnostics mentioning {needle1:?} and {needle2:?}, got:\n{joined}"
        );
    }
}

#[test]
fn item4_user_sink_print_diagnostics_and_effect_inheritance() {
    // 1. Standalone f"..." error suggests println(sink, f"...").
    let standalone_src = r#"
fn main() {
    u32 x = 1;
    []u8 bad = f"{x}";
}
"#;
    let p_standalone = zev_parser::parse_str(standalone_src).unwrap();
    let out_standalone =
        analyze_and_monomorphize_programs(&[p_standalone], &AnalyzeOptions::default());
    let msg_standalone = out_standalone
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        msg_standalone.contains("println(sink, f\"...\")"),
        "expected println(sink, f\"...\") suggestion, got:\n{msg_standalone}"
    );

    let cases: [(&str, &str, &str); 7] = [
        (
            "missing_write_method",
            "does not implement `fn write(*self, []u8)`",
            "struct EmptySink { u32 x; }\nfn main() { EmptySink s = EmptySink { x: 0 }; println(s, \"hi\"); }",
        ),
        (
            "wrong_write_signature",
            "must have signature `fn write(*self, []u8)`",
            "struct BadSig { u32 x; }\nimpl BadSig { pub fn write(*self, u32 b) {} }\nfn main() { BadSig s = BadSig { x: 0 }; println(s, \"hi\"); }",
        ),
        (
            "const_value_sink",
            "immutable `const`",
            "struct CLog { u32 n; }\nimpl CLog { pub fn write(*self, []u8 b) {} }\nconst CLog CS = CLog { n: 0 };\nfn main() { println(CS, \"hi\"); }",
        ),
        (
            "unhandled_fallible_write",
            "catch (IOErr err)",
            "pub enum IOErr { Full }\nstruct ErrSink { u32 n; }\nimpl ErrSink { pub fn write(*self, []u8 b) error IOErr { error Full; } }\nfn main() { ErrSink s = ErrSink { n: 0 }; println(s, f\"v={1}\"); }",
        ),
        (
            "yielding_write_in_critical",
            "critical",
            "struct SlowSink { u32 n; }\nimpl SlowSink { pub fn write(*self, []u8 b) { delay(1ms); } }\nfn main() { SlowSink s = SlowSink { n: 0 }; critical { println(s, \"hi\"); } }",
        ),
        (
            "yielding_write_in_noyield",
            "noyield",
            "struct SlowSink { u32 n; }\nimpl SlowSink { pub fn write(*self, []u8 b) { delay(1ms); } }\nfn main() { SlowSink s = SlowSink { n: 0 }; noyield { print(s, \"hi\"); } }",
        ),
        (
            "non_struct_sink",
            "must be a struct or `*Struct`",
            "fn main() { u32 x = 5; println(x, \"hi\"); }",
        ),
    ];

    for (label, needle, src) in cases {
        let p = zev_parser::parse_str(src).unwrap_or_else(|e| panic!("{label} parse: {e}"));
        let out = analyze_and_monomorphize_programs(&[p], &AnalyzeOptions::default());
        assert!(!out.diagnostics.is_empty(), "{label}: expected Sema error");
        let joined = out
            .diagnostics
            .iter()
            .map(|d| format!("{} {:?}", d.message, d.note))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains(needle),
            "{label}: expected {needle:?} in diagnostics:\n{joined}"
        );
    }

    // Check cross-module private `write` rejection & endpoint provenance through sink.
    let tmp = std::env::temp_dir().join(format!(
        "zev_item4_mod_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(
        tmp.join("priv_sink.zev"),
        "pub struct PrivSink { u32 x; }\nimpl PrivSink { fn write(*self, []u8 b) { self.x += b.len as u32; } }\n",
    )
    .unwrap();
    let app_path = tmp.join("app.zev");
    let app_src = "import priv_sink;\nfn main() { priv_sink.PrivSink s = priv_sink.PrivSink { x: 0 }; println(s, \"hi\"); }\n";
    let p_mod = zev_parser::parse_str_named(app_src, app_path.to_str().unwrap()).unwrap();
    let out_mod = analyze_and_monomorphize_named_programs(
        &[(&p_mod, app_path.to_str())],
        &AnalyzeOptions {
            include_dirs: vec![tmp.clone()],
            ..AnalyzeOptions::default()
        },
    );
    let _ = std::fs::remove_dir_all(&tmp);
    let mod_msgs = out_mod
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        mod_msgs.contains("private"),
        "expected private write error, got:\n{mod_msgs}"
    );

    // Check endpoint provenance through user sink `write`: two tasks printing to sinks wrapping the same `byte_ring`.
    let ring_src = r#"
byte_ring<16> R;
struct RingSink { *byte_ring rx; }
impl RingSink { pub fn write(*self, []u8 b) { usize _n = self.rx.push(b); } }
task t1(*byte_ring r) { RingSink s = RingSink { rx: r }; loop { println(s, "a"); delay(1ms); } }
task t2(*byte_ring r) { RingSink s = RingSink { rx: r }; loop { println(s, "b"); delay(1ms); } }
fn main() { spawn t1(&R) @normal; spawn t2(&R) @low; }
"#;
    let p_ring = zev_parser::parse_str(ring_src).unwrap();
    let out_ring = analyze_and_monomorphize_programs(&[p_ring], &AnalyzeOptions::default());
    let ring_msgs = out_ring
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !out_ring.diagnostics.is_empty(),
        "expected E0004 single-producer error via user sink `println(s, ...)`, got OK:\n{ring_msgs}"
    );
}

#[test]
fn item5_width_and_alignment_sema_static_fit_and_type_checks() {
    // 1. E0028 static overflow when padded width exceeds destination buffer
    let e_u8_too_small = analyze_and_monomorphize_programs(
        &[zev_parser::parse_str(
            "fn main() { u8 x = 3; [4]u8 buf; []u8 _out = fmt_buf(buf, f\"{x:>5}\") or buf[0..0]; }",
        )
        .unwrap()],
        &AnalyzeOptions::default(),
    );
    assert!(
        e_u8_too_small
            .diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::FormatBufferTooSmall),
        "expected E0028 for u8:>5 into [4]u8, got {:?}",
        e_u8_too_small.diagnostics
    );

    let e_slice_min_too_small = analyze_and_monomorphize_programs(
        &[zev_parser::parse_str(
            "fn f([]u8 s) -> usize { [4]u8 buf; []u8 r = fmt_buf(buf, f\"{s:<5}\") or buf[0..0]; return r.len; } fn main() {}",
        )
        .unwrap()],
        &AnalyzeOptions::default(),
    );
    assert!(
        e_slice_min_too_small
            .diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::FormatBufferTooSmall),
        "expected E0028 for []u8:<5 into [4]u8, got {:?}",
        e_slice_min_too_small.diagnostics
    );

    let e_ring_too_small = analyze_and_monomorphize_programs(
        &[zev_parser::parse_str(
            "byte_ring<4> R;\nfn main() { []u8 s = \"a\"; usize _n = fmt_ring(&R, f\"{s:<5}\") or 0; }",
        )
        .unwrap()],
        &AnalyzeOptions::default(),
    );
    assert!(
        e_ring_too_small
            .diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::FormatBufferTooSmall),
        "expected E0028 for []u8:<5 into byte_ring<4>, got {:?}",
        e_ring_too_small.diagnostics
    );

    // 2. Proven-fit without catch/or: u8:>5 fits in [5]u8; [4]u8:<12 fits in [12]u8
    let ok_fit = analyze_and_monomorphize_programs(
        &[zev_parser::parse_str(
            r#"
fn main() {
    u8 x = 3;
    [5]u8 b1;
    []u8 _o1 = fmt_buf(b1, f"{x:>5}");
    [4]u8 arr = ['a', 'b', 'c', 'd'];
    [12]u8 b2;
    []u8 _o2 = fmt_buf(b2, f"{arr:<12}");
}
"#,
        )
        .unwrap()],
        &AnalyzeOptions::default(),
    );
    assert!(
        ok_fit.diagnostics.is_empty(),
        "expected proven-fit padded fmt_buf without catch/or, got {:?}",
        ok_fit.diagnostics
    );

    // 3. Zero-padded alignment `{s:0>5}` and `{s:0<5}` are accepted for text,
    //    while numeric `{s:05}`, `{s:>4x}`, `{true:>5}`, `{1.5f32:>5}` are rejected.
    let ok_text_zero_fill = analyze_and_monomorphize_programs(
        &[
            zev_parser::parse_str("fn main() { []u8 s = \"hi\"; println(f\"{s:0>5}|{s:0<5}\"); }")
                .unwrap(),
        ],
        &AnalyzeOptions::default(),
    );
    assert!(
        ok_text_zero_fill.diagnostics.is_empty(),
        "expected {{s:0>5}} and {{s:0<5}} to succeed, got {:?}",
        ok_text_zero_fill.diagnostics
    );

    for (label, src) in [
        (
            "text_bare_05",
            "fn main() { []u8 s = \"hi\"; println(f\"{s:05}\"); }",
        ),
        (
            "text_hex_pad",
            "fn main() { []u8 s = \"hi\"; println(f\"{s:>4x}\"); }",
        ),
        (
            "bool_width",
            "fn main() { bool b = true; println(f\"{b:>5}\"); }",
        ),
        (
            "float_width",
            "fn main() { f32 f = 1.5f32; println(f\"{f:>5}\"); }",
        ),
    ] {
        let res = analyze_and_monomorphize_programs(
            &[zev_parser::parse_str(src).unwrap()],
            &AnalyzeOptions::default(),
        );
        assert!(
            res.diagnostics
                .iter()
                .any(|d| d.kind == DiagnosticKind::TypeMismatch),
            "{label}: expected TypeMismatch, got {:?}",
            res.diagnostics
        );
    }
}

#[test]
fn pr22_e0008_e0009_e0010_implicit_array_to_slice_escape_checks() {
    for (label, src) in [
        (
            "assign_outer",
            "fn main() { []u8 s = \"x\"; { [3]u8 b = [1, 2, 3]; s = b; } }",
        ),
        (
            "struct_lit",
            "struct W { []u8 s; } fn main() { W w = W { s: \"x\" }; { [3]u8 b = [1, 2, 3]; w = W { s: b }; } }",
        ),
        (
            "struct_field_assign",
            "struct W { []u8 s; } fn main() { W w = W { s: \"x\" }; { [3]u8 b = [1, 2, 3]; w.s = b; } }",
        ),
        (
            "array_of_slices",
            "fn main() { [2][]u8 p = [\"a\", \"b\"]; { [3]u8 b = [1, 2, 3]; p = [b, \"ok\"]; } }",
        ),
        (
            "index_assign",
            "fn main() { [2][]u8 p = [\"a\", \"b\"]; { [3]u8 b = [1, 2, 3]; p[0] = b; } }",
        ),
        (
            "or_fallback",
            "enum E { Fail } fn f() -> []u8 error E { error E.Fail; } fn main() { []u8 s = \"x\"; { [3]u8 b = [1, 2, 3]; s = f() or b; } }",
        ),
        (
            "catch_fallback",
            "enum E { Fail } fn f() -> []u8 error E { error E.Fail; } fn main() { []u8 s = \"x\"; { [3]u8 b = [1, 2, 3]; s = f() catch (E e) { b }; } }",
        ),
    ] {
        let res = analyze_and_monomorphize_programs(
            &[zev_parser::parse_str(src).unwrap()],
            &AnalyzeOptions::default(),
        );
        assert!(
            res.diagnostics
                .iter()
                .any(|d| d.kind == DiagnosticKind::ScopeDepthEscape),
            "{label}: expected E0008 ScopeDepthEscape, got {:?}",
            res.diagnostics
        );
    }

    for (label, src) in [
        ("return_direct", "fn bad() -> []u8 { [3]u8 b = [1, 2, 3]; return b; } fn main() { []u8 _s = bad(); }"),
        ("return_struct", "struct W { []u8 s; } fn bad() -> W { [3]u8 b = [1, 2, 3]; return W { s: b }; } fn main() { W _w = bad(); }"),
        ("return_array_of_slices", "fn bad() -> [2][]u8 { [3]u8 b = [1, 2, 3]; return [b, \"ok\"]; } fn main() { [2][]u8 _p = bad(); }"),
    ] {
        let res = analyze_and_monomorphize_programs(
            &[zev_parser::parse_str(src).unwrap()],
            &AnalyzeOptions::default(),
        );
        assert!(
            res.diagnostics
                .iter()
                .any(|d| d.kind == DiagnosticKind::ReturnLocalReference),
            "{label}: expected E0009 ReturnLocalReference, got {:?}",
            res.diagnostics
        );
    }

    let spawn_res = analyze_and_monomorphize_programs(
        &[zev_parser::parse_str(
            "task worker([]u8 s) { loop { delay(10ms); } } fn main() { [3]u8 buf = [1, 2, 3]; spawn worker(buf); }",
        )
        .unwrap()],
        &AnalyzeOptions::default(),
    );
    assert!(
        spawn_res
            .diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::SpawnNonStaticReference),
        "expected E0010 SpawnNonStaticReference, got {:?}",
        spawn_res.diagnostics
    );
}

#[test]
fn pr22_fnptr_invariance_rejects_mismatched_signatures() {
    for (label, src) in [
        (
            "widen_param_and_ret",
            "fn f(u8 x) -> u8 { return x; } fn main() { fn(u32) -> u32 cb = f; u32 r = cb(1); }",
        ),
        (
            "widen_ret_only",
            "fn f(u32 x) -> u8 { return (x & 0xff) as u8; } fn main() { fn(u32) -> u32 cb = f; u32 r = cb(1); }",
        ),
        (
            "array_vs_slice_param",
            "fn f([]u8 s) -> usize { return s.len; } fn main() { fn([3]u8) -> usize cb = f; [3]u8 a = [1, 2, 3]; usize r = cb(a); }",
        ),
    ] {
        let res = analyze_and_monomorphize_programs(
            &[zev_parser::parse_str(src).unwrap()],
            &AnalyzeOptions::default(),
        );
        assert!(
            res.diagnostics
                .iter()
                .any(|d| d.kind == DiagnosticKind::TypeMismatch)
                && !res
                    .diagnostics
                    .iter()
                    .any(|d| d.kind == DiagnosticKind::UndefinedSymbol),
            "{label}: expected only TypeMismatch on mismatched fn ptr signature, got {:?}",
            res.diagnostics
        );
    }

    let exact = analyze_and_monomorphize_programs(
        &[zev_parser::parse_str(
            "fn f(u8 x) -> u16 { return x; } fn main() { fn(u8) -> u16 cb = f; u16 r = cb(7); }",
        )
        .unwrap()],
        &AnalyzeOptions::default(),
    );
    assert!(
        exact.diagnostics.is_empty(),
        "exact fn ptr match should pass, got {:?}",
        exact.diagnostics
    );
}
