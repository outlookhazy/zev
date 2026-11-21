//! Lexer and recursive-descent parser for the Zev Language-OS (`.zev`).
//!
//! Public entry points:
//! - [`parse_str`] — parse an in-memory `.zev` source string into a [`zev_ast::Program`].
//! - [`parse_str_named`] — parse an in-memory string with a filename attached to diagnostics.
//! - [`parse_file`] — read and parse a `.zev` file on disk, attaching its path to diagnostics.
//! - [`lex`] — tokenize `.zev` source into `Vec<Token>`.

mod error;
mod lexer;
mod parser;
mod token;

use std::path::Path;

pub use error::ParseError;
pub use token::{Token, TokenKind};
pub use zev_ast;
pub use zev_ast::Program;

/// Tokenize an in-memory `.zev` source string.
pub fn lex(source: &str) -> Result<Vec<Token>, ParseError> {
    lexer::lex(source, None)
}

/// Parse an in-memory `.zev` source string into a [`Program`].
pub fn parse_str(source: &str) -> Result<Program, ParseError> {
    parse_str_named(source, "<input>")
}

/// Parse an in-memory `.zev` source string with an explicit logical filename for error diagnostics.
pub fn parse_str_named(source: &str, filename: &str) -> Result<Program, ParseError> {
    let tokens = lexer::lex(source, Some(filename))?;
    let mut parser = parser::Parser::new(source, Some(filename), tokens);
    parser.parse_program()
}

/// Read and parse a `.zev` source file from disk.
pub fn parse_file(path: impl AsRef<Path>) -> Result<Program, ParseError> {
    let path_ref = path.as_ref();
    let display = path_ref.display().to_string();
    let source = std::fs::read_to_string(path_ref).map_err(|io_err| ParseError {
        message: format!("failed to read `{display}`: {io_err}"),
        span: zev_ast::Span::dummy(),
        file: Some(display.clone()),
        source_line: None,
        hint: None,
        code: None,
        doc_topic: Some("targets"),
    })?;
    parse_str_named(&source, &display)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use zev_ast::*;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn parses_north_star_motor_controller() {
        let path = repo_root().join("examples/motor_controller.zev");
        let program = parse_file(&path).unwrap_or_else(|e| panic!("failed to parse:\n{e}"));

        // 1 import + 1 packed struct + 2 enums + 5 OS primitives + 3 ISRs + 1 helper fn + 2 tasks + 1 main fn = 16 items
        assert_eq!(program.items.len(), 16, "unexpected item count");

        match &program.items[0].kind {
            ItemKind::Import(imp) => {
                let segs: Vec<&str> = imp.path.iter().map(|i| i.as_str()).collect();
                assert_eq!(segs, vec!["chip", "stm32g4"]);
                assert_eq!(imp.alias.as_ref().unwrap().as_str(), "hw");
            }
            other => panic!("expected Import, got {other:?}"),
        }

        match &program.items[1].kind {
            ItemKind::Struct(st) => {
                assert!(st.is_packed);
                assert_eq!(st.name.as_str(), "CmdFrame");
                assert_eq!(st.fields.len(), 3);
                assert_eq!(st.fields[0].ty.kind, TypeKind::Prim(PrimType::U8));
                assert_eq!(st.fields[1].ty.kind, TypeKind::Prim(PrimType::U16Be));
                assert_eq!(st.fields[2].ty.kind, TypeKind::Prim(PrimType::U16Be));
            }
            other => panic!("expected Struct, got {other:?}"),
        }

        match &program.items[2].kind {
            ItemKind::Enum(en) => {
                assert_eq!(en.name.as_str(), "ParseErr");
                assert_eq!(en.variants.len(), 2);
            }
            other => panic!("expected Enum, got {other:?}"),
        }

        match &program.items[4].kind {
            ItemKind::OsPrimitive(os) => {
                assert_eq!(os.name.as_str(), "uart1_dma");
                assert_eq!(os.attrs.len(), 1);
                assert_eq!(os.attrs[0].kind, AttrKind::DmaSafe);
                match &os.ty {
                    OsPrimitiveType::PingPong { elem, cap } => {
                        assert_eq!(elem.kind, TypeKind::Prim(PrimType::U8));
                        assert_eq!(*cap, 64);
                    }
                    other => panic!("expected PingPong, got {other:?}"),
                }
            }
            other => panic!("expected OsPrimitive, got {other:?}"),
        }

        match &program.items[6].kind {
            ItemKind::OsPrimitive(os) => {
                assert_eq!(os.name.as_str(), "motor1_rpm");
                assert!(os.init.is_some());
            }
            other => panic!("expected Watch OsPrimitive, got {other:?}"),
        }

        match &program.items[9].kind {
            ItemKind::Isr(isr) => {
                assert_eq!(isr.vector.as_str(), "EXTI0");
                assert_eq!(isr.body.stmts.len(), 2);
                match &isr.body.stmts[1].kind {
                    StmtKind::Expr(Expr {
                        kind: ExprKind::MethodCall { method, args, .. },
                        ..
                    }) => {
                        assert_eq!(method.as_str(), "post");
                        assert_eq!(args.len(), 2);
                        assert_eq!(args[1].name.as_ref().unwrap().as_str(), "boost");
                        assert_eq!(args[1].value.kind, ExprKind::Priority(Priority::Critical));
                    }
                    other => panic!("expected MethodCall, got {other:?}"),
                }
            }
            other => panic!("expected Isr, got {other:?}"),
        }

        match &program.items[12].kind {
            ItemKind::Fn(f) => {
                assert_eq!(f.name.as_str(), "parse_setpoint");
                assert_eq!(f.ret_type.kind, TypeKind::Prim(PrimType::U16));
                assert_eq!(f.params.len(), 1);
                assert!(matches!(f.params[0].ty.kind, TypeKind::Slice(_)));
                assert!(matches!(f.error_spec, Some(ErrorSpec::Typed(_))));
            }
            other => panic!("expected Fn, got {other:?}"),
        }

        match &program.items[13].kind {
            ItemKind::Task(t) => {
                assert_eq!(t.name.as_str(), "motor_control");
                assert_eq!(t.params.len(), 2);
                match &t.params[0].ty.kind {
                    TypeKind::Ptr(inner) => {
                        assert!(matches!(
                            inner.kind,
                            TypeKind::OsPrimitive(OsPrimitiveType::Watch { .. })
                        ));
                    }
                    other => panic!("expected Ptr(Watch), got {other:?}"),
                }
            }
            other => panic!("expected Task, got {other:?}"),
        }

        match &program.items[15].kind {
            ItemKind::Fn(f) => {
                assert_eq!(f.name.as_str(), "main");
                assert!(f.ret_type.is_void());
                assert!(f.body.is_some());
            }
            other => panic!("expected main Fn, got {other:?}"),
        }
    }

    #[test]
    fn parses_all_workspace_examples() {
        for rel in [
            "examples/motor_controller.zev",
            "examples/motor_support.zev",
            "examples/sim_demo.zev",
            "examples/cortex_m_demo.zev",
            "examples/esp32c6_demo.zev",
        ] {
            let path = repo_root().join(rel);
            if path.exists() {
                let prog = parse_file(&path)
                    .unwrap_or_else(|e| panic!("failed to parse {rel}:\n{e}"));
                assert!(!prog.items.is_empty());
            }
        }
    }

    #[test]
    fn parses_all_dual_exit_error_and_catch_variants() {
        let src = r#"
            enum ReadErr { Timeout, Disconnected }

            void simple_check(u8 x) error {
                if (x == 0) {
                    error;
                }
                return;
            }

            usize read_packet([]u8 dst) error ReadErr {
                usize n = 3;
                if (dst.len < 4) {
                    error Timeout, n;
                }
                return n;
            }

            fn caller([]u8 buf) {
                simple_check(1) catch {
                    return;
                };

                usize got = read_packet(buf) catch (ReadErr err, usize partial) {
                    if (err == Timeout) {
                        return;
                    }
                    return;
                };

                usize fallback_val = read_packet(buf) or 0;
            }
        "#;

        let prog = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(prog.items.len(), 4);

        match &prog.items[1].kind {
            ItemKind::Fn(f) => {
                assert!(matches!(f.error_spec, Some(ErrorSpec::Bare(_))));
            }
            other => panic!("unexpected {other:?}"),
        }

        match &prog.items[2].kind {
            ItemKind::Fn(f) => {
                let if_stmt = &f.body.as_ref().unwrap().stmts[1];
                match &if_stmt.kind {
                    StmtKind::If { then_block, .. } => match &then_block.stmts[0].kind {
                        StmtKind::Error { reason, partial } => {
                            assert!(reason.is_some());
                            assert!(partial.is_some());
                        }
                        other => panic!("expected Error stmt, got {other:?}"),
                    },
                    other => panic!("expected If, got {other:?}"),
                }
            }
            other => panic!("unexpected {other:?}"),
        }

        match &prog.items[3].kind {
            ItemKind::Fn(f) => {
                let stmts = &f.body.as_ref().unwrap().stmts;
                match &stmts[1].kind {
                    StmtKind::VarDecl {
                        init: Some(Expr {
                            kind: ExprKind::Catch { clause, .. },
                            ..
                        }),
                        ..
                    } => {
                        assert_eq!(clause.err_param.as_ref().unwrap().name.as_str(), "err");
                        assert_eq!(
                            clause.partial_param.as_ref().unwrap().name.as_str(),
                            "partial"
                        );
                    }
                    other => panic!("expected Catch VarDecl, got {other:?}"),
                }
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_spawn_core_pinning_extern_c_callbacks_and_structs() {
        let src = r#"
            const u32 MAX_CHAN = 16;
            signal irq_ready;
            chan<u32, 16> work_q;
            lease<u8, 128> spi_buf;
            u32 illegal_mutable_global = 42;

            struct ByteReader {
                []u8 buf;
                usize pos;
            }

            extern "C" fn c_register_hook(fn(u32) -> i32 cb);
            extern "C" i32 c_legacy_hook(void (*cb)(u8));

            @manual_yield
            task worker(u8 id) {
                u8 local_arr[4] = [1, 2, 3, 4];
                []u8 sub = local_arr[1..3];
                ByteReader r = ByteReader { buf: sub, pos: 0 };
                @manual_yield
                loop {
                    if (r.pos >= r.buf.len) {
                        break;
                    }
                    r.pos += 1;
                    checkpoint();
                }
            }

            fn main() {
                spawn worker(0) @high @core(0);
                spawn worker(1) @core(1) @low;
            }
        "#;

        let prog = parse_str(src).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(prog.items.len(), 10);

        assert!(matches!(prog.items[4].kind, ItemKind::GlobalVar(_)));

        // Function-pointer parameter types `fn(u32) -> i32` and `void (*cb)(u8)` remain valid!
        for idx in [6, 7] {
            match &prog.items[idx].kind {
                ItemKind::Fn(f) => {
                    assert!(f.is_extern_c);
                    assert!(f.body.is_none());
                    assert!(matches!(f.params[0].ty.kind, TypeKind::FnPtr { .. }));
                }
                other => panic!("expected extern Fn, got {other:?}"),
            }
        }

        match &prog.items[9].kind {
            ItemKind::Fn(f) => {
                let stmts = &f.body.as_ref().unwrap().stmts;
                match &stmts[0].kind {
                    StmtKind::Spawn(sp) => {
                        assert_eq!(sp.priority, Some(Priority::High));
                        assert_eq!(sp.core, Some(0));
                    }
                    other => panic!("expected Spawn, got {other:?}"),
                }
                match &stmts[1].kind {
                    StmtKind::Spawn(sp) => {
                        assert_eq!(sp.priority, Some(Priority::Low));
                        assert_eq!(sp.core, Some(1));
                    }
                    other => panic!("expected Spawn, got {other:?}"),
                }
            }
            other => panic!("expected Fn, got {other:?}"),
        }
    }

    #[test]
    fn formats_clear_line_col_diagnostics_and_doc_pointers_on_syntax_errors() {
        let bad_src = "fn broken() {\n    every 10 {\n        checkpoint();\n    }\n}\n";
        let err = parse_str_named(bad_src, "bad_timer.zev").unwrap_err();
        assert_eq!(err.span.line, 2);
        assert_eq!(err.span.col, 11);
        assert_eq!(err.code, None);
        assert_eq!(err.doc_topic, Some("tasks"));
        let formatted = err.to_string();
        assert!(formatted.contains("bad_timer.zev:2:11: error:"));
        assert!(formatted.contains("every 10 {"));
        assert!(formatted.contains("^^"));
        assert!(formatted.contains("= docs: run 'zevc doc tasks'"));

        // Verify fallback `zevc doc syntax` when neither code nor doc_topic is set
        let bare = ParseError::new("custom syntax error", Span::new(0, 2, 1, 1), "??\n", Some("x.zev"));
        let bare_fmt = bare.to_string();
        assert!(bare_fmt.contains("x.zev:1:1: error: custom syntax error"));
        assert!(bare_fmt.contains("= docs: run 'zevc doc syntax'"));

        // Verify multi-byte UTF-8 box-drawing chars on the error line never panic in Display!
        let utf8_bad = "/* ── header ── */ fn f() { u32 x = @dma_safe; }\n";
        let utf8_err = parse_str_named(utf8_bad, "utf8_err.zev").unwrap_err();
        let rendered = utf8_err.to_string();
        assert!(rendered.contains("utf8_err.zev:1:"));
        assert!(rendered.contains("= docs: run 'zevc doc "));
    }

    #[test]
    fn parses_casts_shifts_switch_single_stmt_and_while_loops() {
        let src = r#"
            packed struct WirePacket {
                u16le header;
                u32be payload_be;
                u32le crc_le;
            }

            extern "C" {
                fn uart_putc(u8 ch);
                i32 printf(*const u8 fmt, ...);
            }

            u32 bare_metal_ops([]u8 data) {
                u32 acc = 0;
                usize i = 0;
                while (i < data.len) {
                    if (i == 2) {
                        i++;
                        continue;
                    }
                    acc = (acc << 4) ^ (data[i] as u32) ^ (u32)0xA5;
                    i++;
                }

                []u8 head = data[..2];
                []u8 tail = data[1..];
                u32 mask = ~0xFF_u32;
                *(0x4000_C000 as *u32) = 'A';

                switch (acc & 3) {
                    0 => acc += 10,
                    1 => acc += 20,
                    default => acc += (head.len + tail.len) as u32,
                }

                return acc & mask;
            }
        "#;

        let prog = parse_str(src).unwrap_or_else(|e| panic!("parse failed:\n{e}"));
        assert_eq!(prog.items.len(), 4);

        match &prog.items[0].kind {
            ItemKind::Struct(st) => {
                assert!(st.is_packed);
                assert_eq!(st.fields[0].ty.kind, TypeKind::Prim(PrimType::U16Le));
                assert_eq!(st.fields[1].ty.kind, TypeKind::Prim(PrimType::U32Be));
                assert_eq!(st.fields[2].ty.kind, TypeKind::Prim(PrimType::U32Le));
            }
            other => panic!("expected Struct, got {other:?}"),
        }

        match &prog.items[2].kind {
            ItemKind::Fn(f) => {
                assert!(f.is_extern_c);
                assert!(f.is_variadic);
                assert_eq!(f.name.as_str(), "printf");
            }
            other => panic!("expected variadic extern Fn, got {other:?}"),
        }
    }

    #[test]
    fn parses_all_qa_adversarial_programs_into_clean_asts_for_sema() {
        let cases = [
            r#"
                u32 shared_counter = 0;
                isr EXTI0 { shared_counter += 1; }
                fn main() {}
            "#,
            r#"
                fn do_yield() { checkpoint(); }
                fn main() {
                    noyield { do_yield(); }
                    critical { do_yield(); }
                }
            "#,
            r#"
                fn main() {
                    critical {
                        loop { }
                    }
                }
            "#,
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
            r#"
                fn a() { checkpoint(); b(); }
                fn b() { a(); }
                fn main() { a(); }
            "#,
            r#"
                task fast() { every 500us { } }
                fn main() { sys.use_systick(170_000_000); spawn fast() @high; }
            "#,
            r#"
                extern "C" fn register_irq_cb(fn() cb);
                fn bad_cb() { checkpoint(); }
                fn main() { register_irq_cb(bad_cb); }
            "#,
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
        ];

        for (idx, src) in cases.iter().enumerate() {
            parse_str(src).unwrap_or_else(|e| panic!("QA case #{idx} failed to parse:\n{e}"));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Agent-First Foreign-Idiom Traps (`P0001`..`P0010`)
    // ─────────────────────────────────────────────────────────────────────────

    fn assert_trap(
        src: &str,
        expected_code: &'static str,
        expected_topic: &'static str,
        must_contain_in_help: &str,
    ) {
        let err = match parse_str_named(src, "trap.zev") {
            Ok(_) => panic!("expected trap {expected_code} on:\n{src}"),
            Err(e) => e,
        };
        assert_eq!(
            err.code,
            Some(expected_code),
            "wrong code for `{src}`:\n{err}"
        );
        assert_eq!(
            err.doc_topic,
            Some(expected_topic),
            "wrong doc_topic for `{src}`:\n{err}"
        );
        let hint = err
            .hint
            .as_deref()
            .unwrap_or_else(|| panic!("missing hint on {expected_code}:\n{err}"));
        assert!(
            hint.contains(must_contain_in_help),
            "expected `{must_contain_in_help}` in hint `{hint}` for:\n{src}"
        );
        let rendered = err.to_string();
        assert!(
            rendered.contains(&format!("error[{expected_code}]:")),
            "missing `error[{expected_code}]:` header in:\n{rendered}"
        );
        assert!(
            rendered.contains(&format!(
                "= docs: run 'zevc doc {expected_code}' or 'zevc doc {expected_topic}'"
            )),
            "missing docs pointer in:\n{rendered}"
        );
    }

    #[test]
    fn trap_p0001_let_var_mut_bindings() {
        assert_trap(
            "fn main() { let x = 42; }",
            "P0001",
            "syntax",
            "u16 x = 42;",
        );
        assert_trap(
            "fn main() { let mut rpm: u16 = 1800; }",
            "P0001",
            "syntax",
            "u16 x = 42;",
        );
        assert_trap(
            "fn main() { var x: u16 = 42; }",
            "P0001",
            "syntax",
            "u16 x = 42;",
        );
        assert_trap(
            "fn main() { mut x = 1; }",
            "P0001",
            "syntax",
            "u16 x = 42;",
        );
        assert_trap("let GLOBAL = 1;", "P0001", "syntax", "const");
    }

    #[test]
    fn trap_p0002_colon_params_and_fn_arrow_return() {
        assert_trap(
            "fn f(x: u16) -> u16 { return x; }",
            "P0002",
            "syntax",
            "u16 x",
        );
        assert_trap(
            "fn clamp(u16 x) -> u16 { return x; }",
            "P0002",
            "syntax",
            "u16 clamp",
        );
        assert_trap(
            "struct S { field: u32 }",
            "P0002",
            "syntax",
            "u32 field",
        );
        assert_trap(
            "const LIMIT: u32 = 10;",
            "P0002",
            "syntax",
            "u32 LIMIT",
        );
    }

    #[test]
    fn trap_p0003_async_and_await() {
        assert_trap(
            "async fn read_sensor() { checkpoint(); }",
            "P0003",
            "yielding",
            "noyield",
        );
        assert_trap(
            "fn main() { u16 v = adc_read_rpm(1).await; }",
            "P0003",
            "yielding",
            ".await",
        );
        assert_trap(
            "fn main() { await checkpoint(); }",
            "P0003",
            "yielding",
            "async",
        );
    }

    #[test]
    fn trap_p0004_select_keyword() {
        assert_trap(
            "fn main() { select { a => {} } }",
            "P0004",
            "primitives",
            "inbox",
        );
    }

    #[test]
    fn trap_p0005_match_keyword() {
        assert_trap(
            "fn main() { match ev { IdleCheck => {} } }",
            "P0005",
            "syntax",
            "switch (expr)",
        );
    }

    #[test]
    fn trap_p0006_zig_catch_pipe() {
        assert_trap(
            "fn main() { u16 v = parse_setpoint(half) catch |err| { return; }; }",
            "P0006",
            "errors",
            "catch (ParseErr err)",
        );
    }

    #[test]
    fn trap_p0007_bang_return_try_and_return_error() {
        assert_trap(
            "!u16 read_reg() { return 0; }",
            "P0007",
            "errors",
            "error ParseErr",
        );
        assert_trap(
            "ParseErr!u16 read_reg() { return 0; }",
            "P0007",
            "errors",
            "error ParseErr",
        );
        assert_trap(
            "fn main() { u16 x = try read_reg(); }",
            "P0007",
            "errors",
            "catch",
        );
        assert_trap(
            "u16 read_reg() error IoErr { return error.Timeout; }",
            "P0007",
            "errors",
            "error BadCrc;",
        );
        assert_trap(
            "u16 read_reg() error IoErr { return Err(Timeout); }",
            "P0007",
            "errors",
            "error BadCrc;",
        );
        assert_trap(
            "fn main() { u16 x = read_reg()?; }",
            "P0007",
            "errors",
            "or fallback",
        );
    }

    #[test]
    fn trap_p0008_for_loop() {
        assert_trap(
            "fn main() { for (u32 i = 0; i < 4; i += 1) { } }",
            "P0008",
            "syntax",
            "while",
        );
        assert_trap(
            "fn main() { for x in buf { } }",
            "P0008",
            "syntax",
            "every",
        );
    }

    #[test]
    fn trap_p0009_lifetimes_and_mut_borrows() {
        assert_trap(
            "struct ByteReader<'a> { []u8 buf; }",
            "P0009",
            "dma",
            "[]u8",
        );
        assert_trap(
            "fn read(&'a []u8 buf) { }",
            "P0009",
            "dma",
            "'a",
        );
        assert_trap(
            "fn step(acc: &mut i32) { *acc += 1; }",
            "P0009",
            "dma",
            "&x",
        );
        assert_trap(
            "fn step(&mut i32 acc) { *acc += 1; }",
            "P0009",
            "dma",
            "*T",
        );
        assert_trap(
            "fn main() { pi_step(100, 90, &mut integral); }",
            "P0009",
            "dma",
            "&x",
        );
    }

    #[test]
    fn trap_p0010_static_and_pub_modifiers() {
        assert_trap(
            "static u32 counter = 0;",
            "P0010",
            "primitives",
            "watch<T>",
        );
        assert_trap(
            "fn main() { static i32 integral = 0; }",
            "P0010",
            "primitives",
            "const",
        );
        assert_trap(
            "pub fn main() { }",
            "P0010",
            "primitives",
            "pub",
        );
        assert_trap(
            "struct Frame { pub u8 op; }",
            "P0010",
            "primitives",
            "pub",
        );
    }

    #[test]
    fn parses_value_recovering_catch_blocks_with_and_without_trailing_semicolon() {
        let src = r#"
            enum PacketErr { BadLen, BadCrc }

            struct DecodedPacket {
                u8 cmd;
                i16 temp;
            }

            DecodedPacket decode([]u8 raw) error PacketErr {
                error BadLen;
            }

            u16 read_partial([]u8 raw) error PacketErr {
                error BadCrc, 4;
            }

            fn main([]u8 bad) {
                u32 count = 0;

                // 1. Multi-statement catch WITHOUT trailing `;` on final StructLit
                DecodedPacket p1 = decode(bad) catch (PacketErr e) {
                    count += 1;
                    DecodedPacket{ cmd: 9, temp: 42 }
                };

                // 2. Multi-statement catch WITH trailing `;` on final StructLit
                DecodedPacket p2 = decode(bad) catch (PacketErr e) {
                    count += 1;
                    DecodedPacket{ cmd: 9, temp: 42 };
                };

                // 3. Single-expression catch WITHOUT `;`
                u16 v1 = read_partial(bad) catch (PacketErr e, u16 partial) {
                    partial + 100
                };

                // 4. Bare single-expression catch WITHOUT `;` and WITH `;`
                u16 v2 = read_partial(bad) catch { 0 };
                u16 v3 = read_partial(bad) catch { 0; };
            }
        "#;

        let prog = parse_str(src).unwrap_or_else(|e| panic!("parse failed:\n{e}"));
        let main_fn = match &prog.items[4].kind {
            ItemKind::Fn(f) => f,
            other => panic!("expected main Fn, got {other:?}"),
        };
        let stmts = &main_fn.body.as_ref().unwrap().stmts;
        assert_eq!(stmts.len(), 6);

        // Verify both `p1` (no `;`) and `p2` (with `;`) produce `[Assign, Expr(StructLit)]`
        for idx in [1, 2] {
            match &stmts[idx].kind {
                StmtKind::VarDecl {
                    init:
                        Some(Expr {
                            kind: ExprKind::Catch { clause, .. },
                            ..
                        }),
                    ..
                } => {
                    assert_eq!(clause.body.stmts.len(), 2);
                    assert!(matches!(clause.body.stmts[0].kind, StmtKind::Assign { .. }));
                    match &clause.body.stmts[1].kind {
                        StmtKind::Expr(Expr {
                            kind: ExprKind::StructLit { name, fields },
                            ..
                        }) => {
                            assert_eq!(name.as_str(), "DecodedPacket");
                            assert_eq!(fields.len(), 2);
                        }
                        other => panic!("expected trailing StmtKind::Expr(StructLit), got {other:?}"),
                    }
                }
                other => panic!("expected Catch VarDecl at {idx}, got {other:?}"),
            }
        }

        // Verify `v1`, `v2`, `v3` single-expression catch bodies all parse into `[StmtKind::Expr]`
        for idx in [3, 4, 5] {
            match &stmts[idx].kind {
                StmtKind::VarDecl {
                    init:
                        Some(Expr {
                            kind: ExprKind::Catch { clause, .. },
                            ..
                        }),
                    ..
                } => {
                    assert_eq!(clause.body.stmts.len(), 1);
                    assert!(matches!(clause.body.stmts[0].kind, StmtKind::Expr(_)));
                }
                other => panic!("expected Catch VarDecl at {idx}, got {other:?}"),
            }
        }

        // Non-final expression without `;` inside `catch` must still fail cleanly
        assert!(parse_str("fn main() { u16 x = foo() catch { bar() 42 }; }").is_err());
        // Missing `;` at the end of a regular `fn` block must still fail cleanly
        assert!(parse_str("u16 f() { 42 }").is_err());
    }
}
