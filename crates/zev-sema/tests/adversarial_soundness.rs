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

// ── 1. Mutable Global, Task/ISR Isolation & SPSC (`E0001`, `E0002`, `E0033`) ─

#[test]
fn adversarial_mutable_globals_and_alias_access_rejected() {
    // Mutable scalar
    expect_err(
        r#"
u32 bad_counter = 0u32;
fn main() { bad_counter += 1u32; }
"#,
        DiagnosticKind::IllegalMutableGlobal,
    );

    // Mutable array global
    expect_err(
        r#"
[4]u8 bad_buf = [0u8, 0u8, 0u8, 0u8];
fn main() { bad_buf[0] = 1u8; }
"#,
        DiagnosticKind::IllegalMutableGlobal,
    );

    // Mutable struct global read/written inside helper
    expect_err(
        r#"
struct Pair { u32 a; u32 b; }
Pair bad_pair = Pair { a: 1u32, b: 2u32 };
fn bump() { bad_pair.a += 1u32; }
fn main() { bump(); }
"#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // Mutable global accessed from ISR via pointer indirection
    expect_err(
        r#"
irq IRQ_0 = 0;
u32 shared_val = 0u32;
isr my_isr() {
    *u32 p = &shared_val;
    *p = 7u32;
}
fn main() {
    IRQ_0.attach(my_isr());
}
"#,
        DiagnosticKind::GlobalIsolationViolation,
    );
}

#[test]
fn adversarial_isr_subsystem_isolation_and_spsc_violations() {
    // ISR referencing unpassed global subsystem struct instance -> `E0002`
    expect_err(
        r#"
irq IRQ_0 = 0;
struct Port { byte_ring<64> rx; byte_ring<64> tx; }
Port p0;
isr bad_isr() {
    bool _ok = p0.rx.push_byte(1u8);
}
fn main() {
    IRQ_0.attach(bad_isr());
}
"#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // Multi-producer violation on `byte_ring` (two ISRs pushing to same `tx` ring)
    expect_err(
        r#"
irq IRQ_0 = 0;
irq IRQ_1 = 1;
byte_ring<64> ring;
isr p1(*byte_ring tx) { bool _ok = tx.push_byte(1u8); }
isr p2(*byte_ring tx) { bool _ok = tx.push_byte(2u8); }
fn main() {
    IRQ_0.attach(p1(&ring));
    IRQ_1.attach(p2(&ring));
}
"#,
        DiagnosticKind::GlobalIsolationViolation,
    );

    // Multi-consumer violation on `byte_ring` (two tasks popping from same `rx` ring)
    expect_err(
        r#"
byte_ring<64> ring;
task c1(*byte_ring rx) {
    loop {
        []u8 _a = rx.wait_read();
    }
}
task c2(*byte_ring rx) {
    loop {
        []u8 _b = rx.wait_read();
    }
}
fn main() {
    spawn c1(&ring) @normal;
    spawn c2(&ring) @normal;
}
"#,
        DiagnosticKind::GlobalIsolationViolation,
    );
}

#[test]
fn adversarial_isr_state_escape_and_invalid_placement() {
    // `state` variable declared outside `isr`
    expect_err(
        r#"
fn not_an_isr() {
    state u32 count = 0u32;
    count += 1u32;
}
fn main() { not_an_isr(); }
"#,
        DiagnosticKind::InvalidIsrState,
    );

    // Passing `&state_var` pointer into `inbox.post` or escaping helper
    expect_err(
        r#"
irq IRQ_0 = 0;
inbox<*u32, 4> q;
isr leak_isr(*inbox<*u32, 4> qq) {
    state u32 x = 0u32;
    qq.post(&x);
}
fn main() {
    IRQ_0.attach(leak_isr(&q));
}
"#,
        DiagnosticKind::InvalidIsrState,
    );

    // Storing `&state_var` in another variable then escaping via `inbox.post`
    expect_err(
        r#"
irq IRQ_0 = 0;
inbox<*u32, 4> q;
isr indirect_leak_isr(*inbox<*u32, 4> qq) {
    state u32 x = 0u32;
    *u32 p = &x;
    qq.post(p);
}
fn main() {
    IRQ_0.attach(indirect_leak_isr(&q));
}
"#,
        DiagnosticKind::InvalidIsrState,
    );
}

// ── 2. Yielding, Atomicity, Boot-Tree & Callbacks (`E0003`..`E0007`, `E0012`, `E0027`, `E0032`) ─

#[test]
fn adversarial_noyield_critical_isr_defer_and_callback_yields() {
    // 3-hop transitive hard yield inside `noyield`
    expect_err(
        r#"
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
"#,
        DiagnosticKind::YieldInNoYield,
    );

    // SoftYielding (`checkpoint`) allowed inside `noyield`, but rejected in `critical`, `isr`, and `defer`
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

    expect_err(
        r#"
fn soft() { checkpoint(); }
task t() {
    loop {
        critical { soft(); }
        delay(1ms);
    }
}
fn main() { spawn t() @normal; }
"#,
        DiagnosticKind::YieldInCritical,
    );

    expect_err(
        r#"
irq IRQ_0 = 0;
fn soft() { checkpoint(); }
isr bad_isr() { soft(); }
fn main() { IRQ_0.attach(bad_isr()); }
"#,
        DiagnosticKind::YieldInIsr,
    );

    expect_err(
        r#"
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
"#,
        DiagnosticKind::YieldInNoYield,
    );

    // Yielding callback passed to `extern "C"` or `sys.set_putc`
    expect_err(
        r#"
extern "C" fn register_cb(fn(u8) cb);
fn my_cb(u8 b) { checkpoint(); }
fn main() { register_cb(my_cb); }
"#,
        DiagnosticKind::YieldingCallbackToExternC,
    );

    expect_err(
        r#"
fn slow_putc(u8 b) { delay(1ms); }
fn main() { sys.set_putc(slow_putc); }
"#,
        DiagnosticKind::YieldingCallbackToExternC,
    );

    // Unbounded `loop` in `critical`
    expect_err(
        r#"
fn main() {
    critical {
        loop { sys.nop(); }
    }
}
"#,
        DiagnosticKind::UnboundedLoopInCritical,
    );
}

#[test]
fn adversarial_boot_helpers_and_recursion_cycles() {
    // Boot-spawning helper yielding via `delay`
    expect_err(
        r#"
task w() {
    loop { delay(10ms); }
}
fn bad_boot() {
    spawn w() @normal;
    delay(20ms);
}
fn main() { bad_boot(); }
"#,
        DiagnosticKind::InvalidSpawnContext,
    );

    // IRQ-attaching helper yielding via `checkpoint`
    expect_err(
        r#"
irq IRQ_0 = 0;
isr h() {}
fn soft_step() { checkpoint(); }
fn bad_irq_boot() {
    IRQ_0.attach(h());
    soft_step();
}
fn main() { bad_irq_boot(); }
"#,
        DiagnosticKind::IrqAttachmentError,
    );

    // Mutual yielding recursion rejected (`E0012`), non-yielding recursion accepted
    expect_err(
        r#"
fn a(u32 n) {
    if (n > 0u32) {
        b(n - 1u32);
    }
}
fn b(u32 n) {
    checkpoint();
    if (n > 0u32) {
        a(n - 1u32);
    }
}
task w() {
    a(3u32);
    loop { delay(10ms); }
}
fn main() { spawn w() @normal; }
"#,
        DiagnosticKind::YieldingRecursion,
    );

    // Spawn inside loop / task
    expect_err(
        r#"
task w() { loop { delay(10ms); } }
fn main() {
    for (u32 i in 0..2) {
        spawn w() @normal;
    }
}
"#,
        DiagnosticKind::InvalidSpawnContext,
    );

    // Duplicate exclusive `irq.attach`
    expect_err(
        r#"
irq IRQ_0 = 0;
isr h1() {}
isr h2() {}
fn main() {
    IRQ_0.attach(h1());
    IRQ_0.attach(h2());
}
"#,
        DiagnosticKind::IrqAttachmentError,
    );
}

// ── 3. Linear `Lease<T>` & DMA Provenance Rules (`E0008`, `E0009`, `E0011`, `E0034`) ─

#[test]
fn adversarial_linear_lease_leak_double_use_branch_mismatch_and_loops() {
    // 1. Leaked Lease at scope exit
    expect_err(
        r#"
pool<[4]u8, 2> p;
task t() {
    loop {
        Lease<[4]u8> l = p.alloc();
        delay(5ms);
    }
}
fn main() { spawn t() @normal; }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 2. Leaked Lease on early return
    expect_err(
        r#"
pool<[4]u8, 2> p;
fn helper(bool bail) {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    if (bail) {
        return;
    }
    l.release();
}
fn main() { helper(true); }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 3. Leaked Lease on `error`
    expect_err(
        r#"
enum E { Fail }
pool<[4]u8, 2> p;
fn helper(bool bail) error E {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    if (bail) {
        error Fail;
    }
    l.release();
}
fn main() {
    helper(true) catch (E _) {};
}
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 4. Leaked Lease on `break` inside loop
    expect_err(
        r#"
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
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 5. Leaked Lease on `continue` inside loop
    expect_err(
        r#"
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
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 6. Asymmetric `if` without `else` consuming Lease
    expect_err(
        r#"
pool<[4]u8, 2> p;
fn helper(bool cond) {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    if (cond) {
        l.release();
    }
}
fn main() { helper(true); }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 7. Asymmetric `switch` arm forgetting to consume Lease
    expect_err(
        r#"
pool<[4]u8, 2> p;
fn helper(u8 sel) {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    switch (sel) {
        0u8 => { l.release(); }
        _ => {}
    }
}
fn main() { helper(1u8); }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 8. Double-consume (`release` twice)
    expect_err(
        r#"
pool<[4]u8, 2> p;
fn helper() {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    l.release();
    l.release();
}
fn main() { helper(); }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 9. Use after moving into helper
    expect_err(
        r#"
pool<[4]u8, 2> p;
fn sink(Lease<[4]u8> l) { l.release(); }
fn helper() {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    sink(l);
    l.release();
}
fn main() { helper(); }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 10. Use after `inbox.send(l)`
    expect_err(
        r#"
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
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 10b. Overwriting live `Lease<T>` variable without consuming previous value
    expect_err(
        r#"
pool<[4]u8, 2> p;
fn helper() {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    l = p.try_alloc() catch { l.release(); return; };
    l.release();
}
fn main() { helper(); }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // 11. Symmetric consumption across `if` / `else` and generic consumer accepted cleanly
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

    // 12. Generic double-move rejected
    expect_mono_err(
        r#"
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
"#,
        DiagnosticKind::LinearLeaseViolation,
    );
}

#[test]
fn adversarial_ping_pong_dma_slice_escape_and_staleness() {
    // Stale slice read after second `wait_chunk`
    expect_err(
        r#"
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
"#,
        DiagnosticKind::StaleDmaLease,
    );

    // Escaping DMA slice into outer-scope slice variable
    expect_err(
        r#"
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
"#,
        DiagnosticKind::ScopeDepthEscape,
    );

    // Returning local buffer slice
    expect_err(
        r#"
[]u8 bad_ret() {
    [4]u8 local_arr = [1u8, 2u8, 3u8, 4u8];
    return local_arr[0..2];
}
fn main() {
    []u8 _s = bad_ret();
}
"#,
        DiagnosticKind::ReturnLocalReference,
    );
}

// ── 4. `fmt_buf` Worst-Case Bounds Proof (`E0028`) ───────────────────────────

#[test]
fn adversarial_fmt_buf_exact_fit_vs_one_byte_overflow() {
    // `u8` max decimal digits = 3 ("255"), so "v=" (2) + 3 = 5 exact-fit passes, 4 overflows!
    expect_ok(
        r#"
fn ok_u8(u8 x) {
    [5]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 s = fmt_buf(buf, f"v={x}");
}
fn main() { ok_u8(255u8); }
"#,
    );

    expect_err(
        r#"
fn bad_u8(u8 x) {
    [4]u8 buf = [0u8, 0u8, 0u8, 0u8];
    []u8 s = fmt_buf(buf, f"v={x}");
}
fn main() { bad_u8(255u8); }
"#,
        DiagnosticKind::FormatBufferTooSmall,
    );

    // `i32` worst case = 11 bytes ("-2147483648"): 11 fits, 10 overflows
    expect_ok(
        r#"
fn ok_i32(i32 x) {
    [11]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 s = fmt_buf(buf, f"{x}");
}
fn main() { ok_i32(-2147483648); }
"#,
    );

    expect_err(
        r#"
fn bad_i32(i32 x) {
    [10]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 s = fmt_buf(buf, f"{x}");
}
fn main() { bad_i32(-2147483648); }
"#,
        DiagnosticKind::FormatBufferTooSmall,
    );

    // `:08x` max = 8 bytes: 8 fits, 7 overflows
    expect_err(
        r#"
fn bad_hex(u32 x) {
    [7]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 s = fmt_buf(buf, f"{x:08x}");
}
fn main() { bad_hex(0x1234u32); }
"#,
        DiagnosticKind::FormatBufferTooSmall,
    );

    // Self-aliasing `fmt_buf(buf, f"{buf}")` or slice without known capacity without `catch`
    expect_err(
        r#"
fn self_alias() {
    [16]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 s = fmt_buf(buf, f"x={buf}");
}
fn main() { self_alias(); }
"#,
        DiagnosticKind::FormatBufferTooSmall,
    );

    // Unhandled dynamic-slice `fmt_buf` is rejected as `E0016` (must handle runtime `FmtErr` with `catch`/`or`),
    // while subslice of fixed capacity `[4]u8` that overflows worst-case fails `E0028`
    expect_err(
        r#"
fn dyn_slice([]u8 dst, u8 x) {
    []u8 s = fmt_buf(dst, f"v={x}");
}
fn main() {
    [8]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    dyn_slice(buf, 1u8);
}
"#,
        DiagnosticKind::UnhandledFallibleExpr,
    );

    expect_ok(
        r#"
fn dyn_slice_caught([]u8 dst, u8 x) {
    []u8 s = fmt_buf(dst, f"v={x}") catch { return; };
}
fn main() {
    [8]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    dyn_slice_caught(buf, 1u8);
}
"#,
    );

    expect_err(
        r#"
fn bad_subslice(u8 x) {
    [8]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 s = fmt_buf(buf[0..4], f"v={x}");
}
fn main() { bad_subslice(255u8); }
"#,
        DiagnosticKind::FormatBufferTooSmall,
    );
}

// ── 5. MMIO Access, Immortal Tasks, Unreachable Code & Consts (`E0022`, `E0025`, `E0026`, `E0031`) ─

#[test]
fn adversarial_mmio_immortal_task_unreachable_and_const_checks() {
    // Write to `@ro` register
    expect_err(
        r#"
mmio Dev {
    @ro u32 status @ 0x00;
    @wo u32 cmd    @ 0x04;
}
mmio DEV : Dev @ 0x4000_0000;
fn main() { DEV.status = 1u32; }
"#,
        DiagnosticKind::MmioViolation,
    );

    // Read of `@wo` register
    expect_err(
        r#"
mmio Dev {
    @ro u32 status @ 0x00;
    @wo u32 cmd    @ 0x04;
}
mmio DEV : Dev @ 0x4000_0000;
fn main() { u32 _x = DEV.cmd; }
"#,
        DiagnosticKind::MmioViolation,
    );

    // Raw address-of MMIO register
    expect_err(
        r#"
mmio Dev { @rw u32 cr @ 0x00; }
mmio DEV : Dev @ 0x4000_0000;
fn main() { *u32 _p = &DEV.cr; }
"#,
        DiagnosticKind::MmioViolation,
    );

    // By-value MMIO passing
    expect_err(
        r#"
mmio Dev { @rw u32 cr @ 0x00; }
mmio DEV : Dev @ 0x4000_0000;
fn take(Dev d) { d.cr = 1u32; }
fn main() { take(DEV); }
"#,
        DiagnosticKind::MmioViolation,
    );

    // Task exiting via `return;` and via falling off closing `}`
    expect_err(
        r#"
task bad_ret() {
    return;
}
fn main() { spawn bad_ret() @normal; }
"#,
        DiagnosticKind::NonDivergingTask,
    );

    expect_err(
        r#"
task bad_fall(bool c) {
    if (c) {
        restart;
    }
}
fn main() { spawn bad_fall(true) @normal; }
"#,
        DiagnosticKind::NonDivergingTask,
    );

    // Statement after diverging `restart;` -> `E0025`
    expect_err(
        r#"
task t() {
    restart;
    delay(1ms);
}
fn main() { spawn t() @normal; }
"#,
        DiagnosticKind::UnreachableCode,
    );

    // Mutating `const` scalar
    expect_err(
        r#"
const u32 LIMIT = 10u32;
fn main() { LIMIT = 20u32; }
"#,
        DiagnosticKind::ImmutableAssignment,
    );
}

#[test]
fn adversarial_tricky_lease_temporaries_structs_and_transitive_escapes() {
    // Discarding a `Lease<T>` returned as a standalone expression statement (`p.alloc();` / `q.recv();`) -> `E0034`
    expect_err(
        r#"
pool<[4]u8, 2> p;
task t() {
    loop {
        p.alloc();
        delay(5ms);
    }
}
fn main() { spawn t() @normal; }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // Copying or leaking a struct containing a `Lease<T>` field
    expect_err(
        r#"
pool<[4]u8, 2> p;
struct Wrapper { Lease<[4]u8> buf; }
fn leak_wrapper() {
    Lease<[4]u8> l = p.try_alloc() catch { return; };
    Wrapper w = Wrapper { buf: l };
    Wrapper _unused = w;
}
fn main() { leak_wrapper(); }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // Leaking a `Lease<T>` inside a `catch` block before returning
    expect_err(
        r#"
enum E { Fail }
pool<[4]u8, 2> p;
u32 fallible() error E { error Fail; }
fn bad_catch() {
    u32 _v = fallible() catch (E _) {
        Lease<[4]u8> l = p.try_alloc() catch { return; };
        return;
    };
}
fn main() { bad_catch(); }
"#,
        DiagnosticKind::LinearLeaseViolation,
    );

    // Consuming a `Lease<T>` via `defer { l.release(); }` on early return succeeds cleanly
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

    // Returning a struct wrapping a slice to a callee-local array -> `E0009`
    expect_err(
        r#"
struct SliceWrap { []u8 s; }
SliceWrap bad_ret_struct() {
    [4]u8 arr = [1u8, 2u8, 3u8, 4u8];
    return SliceWrap { s: arr };
}
fn main() {
    SliceWrap _w = bad_ret_struct();
}
"#,
        DiagnosticKind::ReturnLocalReference,
    );

    // Transitive escape of ISR `&state` through a helper into `inbox.post` -> `E0033`
    expect_err(
        r#"
irq IRQ_0 = 0;
inbox<*u32, 4> q;
fn sneak_post(*u32 p, *inbox<*u32, 4> qq) {
    qq.post(p);
}
isr bad_isr(*inbox<*u32, 4> qq) {
    state u32 count = 0u32;
    sneak_post(&count, qq);
}
fn main() {
    IRQ_0.attach(bad_isr(&q));
}
"#,
        DiagnosticKind::InvalidIsrState,
    );

    // `fmt_buf` enum formatting worst-case: longest variant `Disconnected` = 12 bytes
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

    expect_err(
        r#"
enum State { Ok, Disconnected }
fn fmt_enum_too_small(State s) {
    [11]u8 buf = [0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    []u8 _out = fmt_buf(buf, f"{s}");
}
fn main() { fmt_enum_too_small(State.Ok); }
"#,
        DiagnosticKind::FormatBufferTooSmall,
    );
}

#[test]
fn adversarial_array_to_slice_coercion_provenance() {
    // 1. Direct return of local array coerced to `[]u8`
    expect_err(
        r#"
[]u8 bad_direct() {
    [4]u8 arr = [1u8, 2u8, 3u8, 4u8];
    return arr;
}
fn main() { []u8 _s = bad_direct(); }
"#,
        DiagnosticKind::ReturnLocalReference,
    );

    // 2. Returning local slice variable initialized via implicit array-to-slice coercion
    expect_err(
        r#"
[]u8 bad_via_local() {
    [4]u8 arr = [1u8, 2u8, 3u8, 4u8];
    []u8 s = arr;
    return s;
}
fn main() { []u8 _s = bad_via_local(); }
"#,
        DiagnosticKind::ReturnLocalReference,
    );

    // 3. Escaping inner-block array into outer slice variable
    expect_err(
        r#"
fn bad_scope_assign() {
    []u8 outer = [];
    {
        [4]u8 inner_arr = [1u8, 2u8, 3u8, 4u8];
        outer = inner_arr;
    }
    []u8 _x = outer;
}
fn main() { bad_scope_assign(); }
"#,
        DiagnosticKind::ScopeDepthEscape,
    );

    // 4. Passing local array coerced to `[]u8` into `spawn`
    expect_err(
        r#"
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
"#,
        DiagnosticKind::SpawnNonStaticReference,
    );

    // 5. Passing local array coerced to `[]u8` through identity helper and returning it
    expect_err(
        r#"
[]u8 id([]u8 x) { return x; }
[]u8 bad_via_call() {
    [4]u8 local_arr = [1u8, 2u8, 3u8, 4u8];
    return id(local_arr);
}
fn main() { []u8 _s = bad_via_call(); }
"#,
        DiagnosticKind::ReturnLocalReference,
    );

    // 6. Positive cases: returning `const` array as `[]u8`, and returning `p.arr` from caller `*S p`
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
