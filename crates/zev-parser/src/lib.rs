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
pub use token::{LexedFormatPart, Token, TokenKind};
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

/// Parse an in-memory `.zev` source string while recovering at item boundaries to collect multiple
/// syntax errors in a single pass.
pub fn parse_str_recovering(source: &str) -> (Program, Vec<ParseError>) {
    parse_str_named_recovering(source, "<input>")
}

/// Parse an in-memory `.zev` source string with an explicit filename while recovering at item
/// boundaries to collect multiple syntax errors in a single pass.
pub fn parse_str_named_recovering(source: &str, filename: &str) -> (Program, Vec<ParseError>) {
    match lexer::lex(source, Some(filename)) {
        Ok(tokens) => {
            let mut parser = parser::Parser::new(source, Some(filename), tokens);
            parser.parse_program_recovering()
        }
        Err(err) => (Program::new(Vec::new(), err.span), vec![err]),
    }
}

/// Read and parse a `.zev` source file from disk.
pub fn parse_file(path: impl AsRef<Path>) -> Result<Program, ParseError> {
    let path_ref = path.as_ref();
    let display = path_ref.display().to_string();
    let source = std::fs::read_to_string(path_ref).map_err(|io_err| ParseError {
        message: format!("failed to read `{display}`: {io_err}"),
        span: zev_ast::Span::dummy(),
        file: Some(display.clone().into_boxed_str()),
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

    /// Parse a workspace `.zev` example file, transparently normalizing pre-migration
    /// bare `\n    every ` loops if `examples/*.zev` have not yet been migrated in a later stage.
    fn parse_workspace_example(path: &Path) -> Program {
        let display = path.display().to_string();
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read `{display}`: {e}"));
        match parse_str_named(&raw, &display) {
            Ok(prog) => prog,
            Err(e) if e.code == Some("P0011") => {
                let migrated = raw.replace("\n    every ", "\n    loop every ");
                parse_str_named(&migrated, &display)
                    .unwrap_or_else(|e2| panic!("failed to parse migrated `{display}`:\n{e2}"))
            }
            Err(e) => panic!("failed to parse `{display}`:\n{e}"),
        }
    }

    #[test]
    fn parses_north_star_motor_controller() {
        let path = repo_root().join("examples/motor_controller.zev");
        let program = parse_workspace_example(&path);

        // 2 imports + 1 packed struct + 2 enums + 6 OS primitives + 2 ISRs + 1 helper fn + 2 tasks + 1 main fn = 17 items
        assert_eq!(program.items.len(), 17, "unexpected item count");

        match &program.items[0].kind {
            ItemKind::Import(imp) => {
                let segs: Vec<&str> = imp.path.iter().map(|i| i.as_str()).collect();
                assert_eq!(segs, vec!["chip", "stm32g4"]);
                assert_eq!(imp.alias.as_ref().unwrap().as_str(), "hw");
            }
            other => panic!("expected Import, got {other:?}"),
        }

        match &program.items[1].kind {
            ItemKind::Import(imp) => {
                let segs: Vec<&str> = imp.path.iter().map(|i| i.as_str()).collect();
                assert_eq!(segs, vec!["bsp", "motor_drive"]);
                assert_eq!(imp.alias.as_ref().unwrap().as_str(), "bsp");
            }
            other => panic!("expected Import, got {other:?}"),
        }

        match &program.items[2].kind {
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

        match &program.items[3].kind {
            ItemKind::Enum(en) => {
                assert_eq!(en.name.as_str(), "ParseErr");
                assert_eq!(en.variants.len(), 2);
            }
            other => panic!("expected Enum, got {other:?}"),
        }

        match &program.items[5].kind {
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

        match &program.items[7].kind {
            ItemKind::OsPrimitive(os) => {
                assert_eq!(os.name.as_str(), "motor1_rpm");
                assert!(matches!(os.ty, OsPrimitiveType::Watch { .. }));
                assert!(os.init.is_some());
            }
            other => panic!("expected Watch OsPrimitive, got {other:?}"),
        }

        match &program.items[9].kind {
            ItemKind::OsPrimitive(os) => {
                assert_eq!(os.name.as_str(), "fault_latched");
                assert!(matches!(os.ty, OsPrimitiveType::Watch { .. }));
                assert!(os.init.is_some());
            }
            other => panic!("expected Watch OsPrimitive, got {other:?}"),
        }

        match &program.items[11].kind {
            ItemKind::Isr(isr) => {
                assert_eq!(isr.vector.as_str(), "overcurrent_isr");
                assert_eq!(isr.params.len(), 2);
                assert_eq!(isr.body.stmts.len(), 4);
                match &isr.body.stmts[3].kind {
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

        match &program.items[13].kind {
            ItemKind::Fn(f) => {
                assert_eq!(f.name.as_str(), "parse_setpoint");
                assert_eq!(f.ret_type.kind, TypeKind::Prim(PrimType::U16));
                assert_eq!(f.params.len(), 1);
                assert!(matches!(f.params[0].ty.kind, TypeKind::Slice(_)));
                assert!(matches!(f.error_spec, Some(ErrorSpec::Typed(_))));
            }
            other => panic!("expected Fn, got {other:?}"),
        }

        match &program.items[14].kind {
            ItemKind::Task(t) => {
                assert_eq!(t.name.as_str(), "motor_control");
                assert_eq!(t.params.len(), 3);
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

        match &program.items[16].kind {
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
                let prog = parse_workspace_example(&path);
                assert!(!prog.items.is_empty());
            }
        }
    }

    #[test]
    fn parses_all_dual_exit_error_and_catch_variants() {
        let src = r#"
            enum ReadErr { Timeout, Disconnected }

            fn simple_check(u8 x) error {
                if (x == 0) {
                    error;
                }
                return;
            }

            fn read_packet([]u8 dst) -> usize error ReadErr {
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
                        init:
                            Some(Expr {
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
            inbox<u32, 16> work_q;
            @dma_safe pool<u8, 8> spi_buf;
            u32 illegal_mutable_global = 42;

            struct ByteReader {
                []u8 buf;
                usize pos;
            }

            extern "C" fn c_register_hook(fn(u32) -> i32 cb);
            extern "C" fn c_legacy_hook(fn(u8) cb) -> i32;

            @manual_yield
            task worker(u8 id) {
                [4]u8 local_arr = [1, 2, 3, 4];
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

        // Canonical function-pointer parameter types `fn(u32) -> i32` and `fn(u8)` parse cleanly!
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
        let bad_src = "fn broken() {\n    loop every {\n        checkpoint();\n    }\n}\n";
        let err = parse_str_named(bad_src, "bad_timer.zev").unwrap_err();
        assert_eq!(err.span.line, 2);
        assert_eq!(err.span.col, 16);
        assert_eq!(err.code, None);
        assert_eq!(err.doc_topic, Some("tasks"));
        let formatted = err.to_string();
        assert!(formatted.contains("bad_timer.zev:2:16: error:"));
        assert!(formatted.contains("loop every {"));
        assert!(formatted.contains("^"));
        assert!(formatted.contains("= docs: run 'zevc doc tasks'"));

        // Verify fallback `zevc doc syntax` when neither code nor doc_topic is set
        let bare = ParseError::new(
            "custom syntax error",
            Span::new(0, 2, 1, 1),
            "??\n",
            Some("x.zev"),
        );
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
                fn printf(*const u8 fmt, ...) -> i32;
            }

            fn bare_metal_ops([]u8 data) -> u32 {
                u32 acc = 0;
                usize i = 0;
                while (i < data.len) {
                    if (i == 2) {
                        i++;
                        continue;
                    }
                    acc = (acc << 4) ^ (data[i] as u32) ^ 0xA5u32;
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
                task fast() { loop every 500us { } }
                fn main() { sys.use_systick(170_000_000); spawn fast() @high; }
            "#,
            r#"
                extern "C" fn register_irq_cb(fn() cb);
                fn bad_cb() { checkpoint(); }
                fn main() { register_irq_cb(bad_cb); }
            "#,
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
        assert_trap("fn main() { mut x = 1; }", "P0001", "syntax", "u16 x = 42;");
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
            "task t() -> u16 { loop { delay(1ms); } }",
            "P0002",
            "tasks",
            "remove `-> ...`",
        );
        assert_trap("isr h() -> u16 { }", "P0002", "tasks", "remove `-> ...`");
        assert_trap("struct S { field: u32 }", "P0002", "syntax", "u32 field");
        assert_trap("const LIMIT: u32 = 10;", "P0002", "syntax", "u32 LIMIT");
        assert_trap(
            "fn read_button() bool { return true; }",
            "P0002",
            "syntax",
            "fn read_button(...) -> bool",
        );
        assert_trap(
            "fn view() []u8 { return buf[0..1]; }",
            "P0002",
            "syntax",
            "fn view(...) -> []u8",
        );
        assert_trap(
            "fn make_s() MyType { return MyType {}; }",
            "P0002",
            "syntax",
            "fn make_s(...) -> MyType",
        );
        assert_trap(
            "fn noop() void { return; }",
            "P0002",
            "syntax",
            "remove `void`",
        );
        assert_trap(
            "fn f() !u16 { return 0; }",
            "P0007",
            "errors",
            "error ParseErr",
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
            "wait {",
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
            "fn read_reg() -> u16 error IoErr { return error.Timeout; }",
            "P0007",
            "errors",
            "error BadCrc;",
        );
        assert_trap(
            "fn read_reg() -> u16 error IoErr { return Err(Timeout); }",
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
            "for (usize i in 0..n)",
        );
        assert_trap(
            "fn main() { for (;;) { } }",
            "P0008",
            "syntax",
            "for (usize i in 0..n)",
        );
        assert_trap(
            "fn main() { for x in buf { } }",
            "P0008",
            "syntax",
            "for (u8 b in slice)",
        );
        assert_trap(
            "fn main() { for (i in 0..8) { } }",
            "P0008",
            "syntax",
            "for (usize i in 0..n)",
        );
    }

    #[test]
    fn trap_p0011_bare_every() {
        assert_trap(
            "task ctrl() { every 20ms { checkpoint(); } }",
            "P0011",
            "tasks",
            "loop every 20ms",
        );
        assert_trap(
            "task ctrl(u32 p) { every p { checkpoint(); } }",
            "P0011",
            "tasks",
            "loop every p",
        );
    }

    #[test]
    fn trap_p0012_retired_chan() {
        assert_trap(
            "chan<u32, 16> work_q;",
            "P0012",
            "primitives",
            "inbox<T, N>",
        );
        assert_trap(
            "task worker(chan<u8, 8> *q) { loop { checkpoint(); } }",
            "P0012",
            "primitives",
            "inbox<T, N>",
        );
    }

    #[test]
    fn trap_p0013_retired_lease() {
        assert_trap("lease<u8, 128> spi_buf;", "P0013", "dma", "@dma_safe pool");
        assert_trap(
            "task worker(lease<u8, 8> *l) { loop { checkpoint(); } }",
            "P0013",
            "dma",
            "dma_slot",
        );
        assert_trap("fn f() { lease<u8, 8> x; }", "P0013", "dma", "Lease<");
        assert_trap(
            "fn f(*lease<u8, 8> l) -> u8 { return 0; }",
            "P0013",
            "dma",
            "Lease<",
        );
    }

    #[test]
    fn trap_p0014_c_style_pointers_arrays_and_casts() {
        assert_trap("fn main() { u8 buf[16]; }", "P0014", "syntax", "[16]u8 buf");
        assert_trap("fn main() { u8 *p = 0; }", "P0014", "syntax", "*u8");
        assert_trap("fn f(watch<u16> *w) {}", "P0014", "syntax", "*watch<u16>");
        assert_trap("struct S { u8 data[]; }", "P0014", "syntax", "[]u8 data");
        assert_trap("fn f(u8[] s) {}", "P0014", "syntax", "[]u8");
        assert_trap("fn f(u8[4] a) {}", "P0014", "syntax", "[4]u8");
        assert_trap(
            "extern \"C\" fn reg_cb(void (*cb)(u8)) -> i32;",
            "P0014",
            "syntax",
            "fn(...) cb",
        );
        assert_trap(
            "struct Foo { u32 x; }\nfn f(*void vp) { *Foo p = (Foo*)vp; }",
            "P0014",
            "syntax",
            "*Foo",
        );

        // Ensure indexing, deref-mul, and plain multiplication never collide with P0014!
        let ok_src = r#"
            fn main(*i32 a) {
                [4]u8 buf = [1, 2, 3, 4];
                buf[0] = 9;
                undecl_arr[0] = 1;
                i32 prod = (*a) * 2;
                i32 x = buf[0] as i32;
                x * prod;
            }
        "#;
        parse_str(ok_src).unwrap_or_else(|e| panic!("unexpected parse failure:\n{e}"));
    }

    #[test]
    fn trap_p0015_p0016_p0017_retired_syntax() {
        let assert_trap = |src: &str, code: &str, want_help_substr: &str| {
            let err = parse_str(src).expect_err(&format!("expected {code} for:\n{src}"));
            let msg = format!("{err:?}");
            assert!(
                msg.contains(code),
                "expected code {code} for:\n{src}\ngot:\n{msg}"
            );
            assert!(
                msg.contains(want_help_substr),
                "expected help containing {want_help_substr:?} for:\n{src}\ngot:\n{msg}"
            );
        };

        // P0015: C-style `->` member access
        assert_trap(
            "struct Foo { u32 x; }\nfn f(*Foo p) -> u32 { return p->x; }",
            "P0015",
            "use `.`",
        );

        // P0016: `::` module/enum/path/import/spawn/switch separator and `::<...>` turbofish
        assert_trap("import hal::gpio;", "P0016", "use `.`");
        assert_trap(
            "fn f() { f32 x = clamp::<f32>(1.0, 0.0, 2.0); }",
            "P0016",
            "clamp<f32>",
        );
        assert_trap(
            "fn f() { f32 x = mod.clamp::<f32>(1.0, 0.0, 2.0); }",
            "P0016",
            "clamp<f32>",
        );
        assert_trap(
            "fn f() { FixedVec<u32, 4> x = FixedVec::<u32, 4> { data: [1, 2, 3, 4], len: 4 }; }",
            "P0016",
            "FixedVec<u32, 4>",
        );
        assert_trap(
            "enum Mode { Idle, Active }\nfn f() { Mode m = Mode::Idle; }",
            "P0016",
            "use `.`",
        );
        assert_trap("impl hw::Gpio { fn set() {} }", "P0016", "use `.`");
        assert_trap(
            "fn f(Mode m) { switch (m) { Mode::Idle => {} _ => {} } }",
            "P0016",
            "use `.`",
        );
        assert_trap("fn main() { spawn hw::worker(); }", "P0016", "use `.`");

        // P0017: C-style casts (excluding `(Foo*)p`, which keeps P0014)
        assert_trap("fn f(u32 v) { u16 x = (u16)v; }", "P0017", "as");
        assert_trap("fn f() { f32 x = (f32)10; }", "P0017", "as");
        assert_trap("fn f(u32 v) { i32 x = (i32)-1; }", "P0017", "as");
        assert_trap(
            "struct Foo { u32 x; }\nfn f(*void p) { *Foo f = (*Foo)p; }",
            "P0017",
            "as",
        );
        assert_trap(
            "struct Foo { u32 x; }\nfn f(*void p) { *Foo f = (*Foo)p; }",
            "P0017",
            "P0017",
        );
        // `(Foo*)p` continues to trigger P0014
        assert_trap(
            "struct Foo { u32 x; }\nfn f(*void p) { *Foo f = (Foo*)p; }",
            "P0014",
            "*Foo",
        );
        // Valid calls on grouped fields still parse
        parse_str("fn main(*Box bf) { (bf.cb)(32); }")
            .unwrap_or_else(|e| panic!("unexpected parse failure for indirect call:\n{e}"));
    }

    #[test]
    fn trap_p0018_and_fn_returning_fn_pointer() {
        let assert_p0018 = |src: &str, want_hint: &str| {
            let err = parse_str(src).expect_err(&format!("expected P0018 for:\n{src}"));
            let msg = format!("{err:?}");
            assert!(
                msg.contains("P0018"),
                "expected P0018 for:\n{src}\ngot:\n{msg}"
            );
            assert!(
                msg.contains(want_hint),
                "expected hint {want_hint:?} for:\n{src}\ngot:\n{msg}"
            );
        };

        assert_p0018("u16 crc16([]u8 buf) { return 0; }", "fn crc16(...) -> u16");
        assert_p0018("void setup_clocks() {}", "fn setup_clocks(...)");
        assert_p0018(
            "fn u16 crc16([]u8 buf) { return 0; }",
            "fn crc16(...) -> u16",
        );
        assert_p0018(
            "struct Foo { u32 x; }\nimpl Foo { u32 count(*self) { return self.x; } }",
            "fn count(...) -> u32",
        );
        assert_p0018(
            "extern \"C\" i32 c_add(i32 a, i32 b);",
            "fn c_add(...) -> i32",
        );
        assert_p0018("extern \"C\" { void c_init(); }", "fn c_init(...)");

        // Function returning a function pointer type `fn pick() -> fn(u32) -> u32`
        let prog = parse_str(
            "fn inc(u32 x) -> u32 { return x + 1u32; }\nfn pick() -> fn(u32) -> u32 { return inc; }\n",
        )
        .expect("fn returning fn pointer must parse");
        match &prog.items[1].kind {
            ItemKind::Fn(f) => assert!(matches!(
                &f.ret_type.kind,
                TypeKind::FnPtr { params, .. } if params.len() == 1
            )),
            other => panic!("expected Fn, got {other:?}"),
        }
    }

    #[test]
    fn parses_pool_dma_slot_and_linear_lease_types() {
        let src = r#"
struct DmaSubsys {
    @dma_safe pool<[16]u8, 4> rx_pool;
    dma_slot<[16]u8> active_rx;
    inbox<Lease<[16]u8>, 4> done_q;
}
pub @dma_safe pool<u32, 8> g_pool;

fn process(*pool<[16]u8, 4> p, *dma_slot<[16]u8> slot, Lease<[16]u8> buf) -> Lease<[16]u8> error {
    Lease<[16]u8> next = p.alloc();
    return next;
}
"#;
        let prog = parse_str(src).expect("pool/dma_slot/Lease program should parse");
        let ItemKind::Struct(st) = &prog.items[0].kind else {
            panic!("expected struct");
        };
        assert_eq!(st.fields[0].attrs.len(), 1);
        assert_eq!(st.fields[0].attrs[0].kind, AttrKind::DmaSafe);
        assert!(matches!(
            st.fields[0].ty.kind,
            TypeKind::OsPrimitive(OsPrimitiveType::Pool { cap: 4, .. })
        ));
        assert!(matches!(
            st.fields[1].ty.kind,
            TypeKind::OsPrimitive(OsPrimitiveType::DmaSlot { .. })
        ));
        let ItemKind::OsPrimitive(pool_decl) = &prog.items[1].kind else {
            panic!("expected top-level os primitive");
        };
        assert_eq!(pool_decl.attrs.len(), 1);
        assert_eq!(pool_decl.attrs[0].kind, AttrKind::DmaSafe);
        assert!(matches!(pool_decl.ty, OsPrimitiveType::Pool { cap: 8, .. }));

        let ItemKind::Fn(f) = &prog.items[2].kind else {
            panic!("expected fn");
        };
        assert!(matches!(
            f.params[0].ty.kind,
            TypeKind::Ptr(ref inner) if matches!(inner.kind, TypeKind::OsPrimitive(OsPrimitiveType::Pool { cap: 4, .. }))
        ));
        assert!(matches!(
            f.params[1].ty.kind,
            TypeKind::Ptr(ref inner) if matches!(inner.kind, TypeKind::OsPrimitive(OsPrimitiveType::DmaSlot { .. }))
        ));
        assert!(matches!(f.params[2].ty.kind, TypeKind::Lease(_)));
        assert!(matches!(f.error_spec, Some(ErrorSpec::Bare(_))));
        assert!(matches!(f.ret_type.kind, TypeKind::Lease(_)));
    }

    #[test]
    fn parses_wave1_restart_defer_for_in_and_loop_every() {
        let src = r#"
            inbox<u8, 8> cmd_q;

            fn spi_cs_low() {}
            fn spi_cs_high() {}

            fn sum_packet([]u8 buf, u32 limit) -> u32 {
                spi_cs_low();
                defer spi_cs_high();
                defer {
                    u8 chan = 1;
                    u32 in = chan as u32;
                }

                u32 acc = 0;
                for (usize i in 0..8) {
                    acc += i as u32;
                }
                for (u32 j in 0..=limit) {
                    acc += j;
                }
                for (usize k in 0..buf.len) {
                    acc += buf[k] as u32;
                }
                for (u8 b in buf) {
                    acc += b as u32;
                }
                for (u8 sub_b in buf[0..2]) {
                    acc += sub_b as u32;
                }
                return acc;
            }

            task fault_tolerant_worker(u32 period_ms) {
                u32 errs = 0;
                loop every 20ms {
                    if (errs > 3) {
                        restart;
                    }
                    switch (errs) {
                        9 => restart,
                        _ => errs += 1,
                    }
                }
            }

            task dynamic_rate_worker(u32 period_ticks) {
                loop every period_ticks {
                    checkpoint();
                }
                loop every (period_ticks * 2) {
                    checkpoint();
                }
            }
        "#;

        let prog = parse_str(src).unwrap_or_else(|e| panic!("Wave 1 parse failed:\n{e}"));
        assert_eq!(prog.items.len(), 6);

        // Inspect `sum_packet`: 1 call + 2 defers + 1 var_decl + 3 ForRange + 2 ForEach + 1 return = 10 stmts
        let sum_fn = match &prog.items[3].kind {
            ItemKind::Fn(f) => f,
            other => panic!("expected Fn, got {other:?}"),
        };
        let stmts = &sum_fn.body.as_ref().unwrap().stmts;
        assert_eq!(stmts.len(), 10);

        match &stmts[1].kind {
            StmtKind::Defer(inner) => assert!(matches!(inner.kind, StmtKind::Expr(_))),
            other => panic!("expected single-stmt Defer, got {other:?}"),
        }
        match &stmts[2].kind {
            StmtKind::Defer(inner) => assert!(matches!(inner.kind, StmtKind::Block(_))),
            other => panic!("expected block Defer, got {other:?}"),
        }
        match &stmts[4].kind {
            StmtKind::ForRange {
                var_name,
                inclusive,
                ..
            } => {
                assert_eq!(var_name.as_str(), "i");
                assert!(!*inclusive);
            }
            other => panic!("expected exclusive ForRange, got {other:?}"),
        }
        match &stmts[5].kind {
            StmtKind::ForRange {
                var_name,
                inclusive,
                ..
            } => {
                assert_eq!(var_name.as_str(), "j");
                assert!(*inclusive);
            }
            other => panic!("expected inclusive ForRange, got {other:?}"),
        }
        match &stmts[6].kind {
            StmtKind::ForRange { end, inclusive, .. } => {
                assert!(!*inclusive);
                assert!(matches!(end.kind, ExprKind::Field { .. }));
            }
            other => panic!("expected `0..buf.len` ForRange, got {other:?}"),
        }
        assert!(matches!(stmts[7].kind, StmtKind::ForEach { .. }));
        assert!(matches!(stmts[8].kind, StmtKind::ForEach { .. }));

        // Inspect `fault_tolerant_worker`: `LoopEvery` containing `StmtKind::Restart`
        let worker1 = match &prog.items[4].kind {
            ItemKind::Task(t) => t,
            other => panic!("expected Task, got {other:?}"),
        };
        match &worker1.body.stmts[1].kind {
            StmtKind::LoopEvery { period, body } => {
                assert!(matches!(
                    period.kind,
                    ExprKind::Literal(Literal::Duration(_))
                ));
                match &body.stmts[0].kind {
                    StmtKind::If { then_block, .. } => {
                        assert!(matches!(then_block.stmts[0].kind, StmtKind::Restart));
                    }
                    other => panic!("expected If, got {other:?}"),
                }
            }
            other => panic!("expected LoopEvery, got {other:?}"),
        }

        // Inspect `dynamic_rate_worker`: `loop every period_ticks` & `loop every (period_ticks * 2)`
        let worker2 = match &prog.items[5].kind {
            ItemKind::Task(t) => t,
            other => panic!("expected Task, got {other:?}"),
        };
        match &worker2.body.stmts[0].kind {
            StmtKind::LoopEvery { period, .. } => {
                assert!(matches!(period.kind, ExprKind::Ident(_)));
            }
            other => panic!("expected identifier period LoopEvery, got {other:?}"),
        }
        match &worker2.body.stmts[1].kind {
            StmtKind::LoopEvery { period, .. } => {
                assert!(matches!(period.kind, ExprKind::Binary { .. }));
            }
            other => panic!("expected binary period LoopEvery, got {other:?}"),
        }
    }

    #[test]
    fn trap_p0009_lifetimes_and_mut_borrows() {
        assert_trap(
            "struct ByteReader<'a> { []u8 buf; }",
            "P0009",
            "dma",
            "[]u8",
        );
        assert_trap("fn read(&'a []u8 buf) { }", "P0009", "dma", "'a");
        assert_trap(
            "fn step(acc: &mut i32) { *acc += 1; }",
            "P0009",
            "dma",
            "&x",
        );
        assert_trap("fn step(&mut i32 acc) { *acc += 1; }", "P0009", "dma", "*T");
        assert_trap(
            "fn main() { pi_step(100, 90, &mut integral); }",
            "P0009",
            "dma",
            "&x",
        );
    }

    #[test]
    fn trap_p0010_static_and_pub_modifiers() {
        assert_trap("static u32 counter = 0;", "P0010", "primitives", "watch<T>");
        assert_trap(
            "fn main() { static i32 integral = 0; }",
            "P0010",
            "primitives",
            "state Type x = init;",
        );
        assert_trap(
            "struct Frame { static u8 op; }",
            "P0010",
            "primitives",
            "const",
        );

        let pub_prog = parse_str("pub fn main() {}").expect("top-level pub fn should parse");
        assert!(pub_prog.items[0].is_pub);

        let grouped_prog = parse_str(
            r#"
import std.{str, fmt as f, bus,};
pub import chip.stm32g4 as hw;
pub import chip.stm32f3.{rcc, tim};
"#,
        )
        .expect("grouped and pub import declarations should parse");
        assert_eq!(grouped_prog.items.len(), 6);
        assert!(!grouped_prog.items[0].is_pub);
        assert!(grouped_prog.items[3].is_pub);
        assert!(grouped_prog.items[4].is_pub);
        assert!(grouped_prog.items[5].is_pub);

        for bad_pub in [
            "fn main() { pub u32 x = 1; }",
            "struct Frame { pub u8 op; }",
            "pub pub fn foo() {}",
        ] {
            let err = parse_str(bad_pub).expect_err(&format!("expected error on `{bad_pub}`"));
            assert!(
                err.code.is_none() && err.message.contains("`pub`"),
                "expected clean non-P0010 syntax error on `{bad_pub}`, got: {err}"
            );
        }
    }

    #[test]
    fn parses_value_recovering_catch_blocks_with_and_without_trailing_semicolon() {
        let src = r#"
            enum PacketErr { BadLen, BadCrc }

            struct DecodedPacket {
                u8 cmd;
                i16 temp;
            }

            fn decode([]u8 raw) -> DecodedPacket error PacketErr {
                error BadLen;
            }

            fn read_partial([]u8 raw) -> u16 error PacketErr {
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
                            kind: ExprKind::StructLit { name, fields, .. },
                            ..
                        }) => {
                            assert_eq!(name.as_str(), "DecodedPacket");
                            assert_eq!(fields.len(), 2);
                        }
                        other => {
                            panic!("expected trailing StmtKind::Expr(StructLit), got {other:?}")
                        }
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
        assert!(parse_str("fn f() -> u16 { 42 }").is_err());
    }

    #[test]
    fn wave2_parses_floats_vs_ranges_and_rejects_malformed_floats() {
        let src = r#"
            fn float_and_range_ops() {
                f32 a = 0.4;
                f32 b = 3.141_592f32;
                f64 c = 2.5e-3;
                f64 d = 1_000e+2f64;
                f32 e = 42f32;
                f32 neg = -0.5;
                f32 casted = (10 as f32) + (a as f32);

                [8]u8 arr = [0, 1, 2, 3, 4, 5, 6, 7];
                []u8 sl = arr[0..4];
                u32 acc = 0;
                for (usize i in 0..4) {
                    acc += sl[i] as u32;
                }
                for (usize j in 0..=4) {
                    acc += j as u32;
                }
            }
        "#;

        let prog = parse_str(src).unwrap_or_else(|e| panic!("float/range parse failed:\n{e}"));
        let f = match &prog.items[0].kind {
            ItemKind::Fn(f) => f,
            other => panic!("expected Fn, got {other:?}"),
        };
        let stmts = &f.body.as_ref().unwrap().stmts;

        // `0.4` -> unsuffixed Float
        match &stmts[0].kind {
            StmtKind::VarDecl {
                ty,
                init:
                    Some(Expr {
                        kind: ExprKind::Literal(Literal::Float { bits, suffix }),
                        ..
                    }),
                ..
            } => {
                assert_eq!(ty.kind, TypeKind::Prim(PrimType::F32));
                assert_eq!(f64::from_bits(*bits), 0.4);
                assert_eq!(*suffix, None);
            }
            other => panic!("expected f32 VarDecl, got {other:?}"),
        }

        // `3.141_592f32` -> f32-suffixed Float
        match &stmts[1].kind {
            StmtKind::VarDecl {
                init:
                    Some(Expr {
                        kind: ExprKind::Literal(Literal::Float { bits, suffix }),
                        ..
                    }),
                ..
            } => {
                #[allow(clippy::approx_constant)]
                let expected_f32 = 3.141_592_f32 as f64;
                assert_eq!(f64::from_bits(*bits), expected_f32);
                assert_eq!(*suffix, Some(PrimType::F32));
            }
            other => panic!("expected suffixed f32 VarDecl, got {other:?}"),
        }

        // `42f32` -> `<int>f32` Float
        match &stmts[4].kind {
            StmtKind::VarDecl {
                init:
                    Some(Expr {
                        kind: ExprKind::Literal(Literal::Float { bits, suffix }),
                        ..
                    }),
                ..
            } => {
                assert_eq!(f64::from_bits(*bits), 42.0);
                assert_eq!(*suffix, Some(PrimType::F32));
            }
            other => panic!("expected 42f32 Float VarDecl, got {other:?}"),
        }

        // Malformed float literals must fail cleanly in lexer (no panic)
        for bad in [
            "fn main() { f32 x = 1.; }",
            "fn main() { f32 x = 1.f32; }",
            "fn main() { f32 x = 1.5e; }",
            "fn main() { f32 x = 1.5e+; }",
            "fn main() { f32 x = 1.2.3; }",
            "fn main() { f32 x = 1.5f16; }",
        ] {
            assert!(
                parse_str(bad).is_err(),
                "expected malformed float error on `{bad}`"
            );
        }
    }

    #[test]
    fn wave2_parses_bounded_generics_nested_shr_calls_and_comparisons() {
        let src = r#"
            struct FixedVec<T, usize N> {
                [N]T data;
                usize len;
            }

            struct Boxed<T> {
                T inner;
            }

            fn clamp<T>(T val, T lo, T hi) -> T {
                T v = val;
                if (v < lo) {
                    return lo;
                }
                if (v > hi) {
                    return hi;
                }
                return v;
            }

            fn reset_vec<T, usize N>(*FixedVec<T, N> vec) {
                vec.len = 0;
            }

            fn main() {
                [16]u8 raw;
                [16]u8 raw2;
                FixedVec<u32, 4> v = FixedVec<u32, 4> { data: [1, 2, 3, 4], len: 4 };
                FixedVec<FixedVec<u8, 4>, 2> nested;
                Boxed<Boxed<u32>> dbl = Boxed<Boxed<u32>> {
                    inner: Boxed<u32> { inner: 9 }
                };

                f32 c1 = clamp<f32>(1.5, 0.0, 1.0);
                f32 c2 = clamp<f32>(-0.5, 0.0, 1.0);
                reset_vec<u32, 4>(&v);

                // Ensure `<`, `>`, `&&`, and `>>` comparisons NEVER collide with generic lookahead!
                u32 x = 3;
                u32 y = 5;
                u32 z = 2;
                if (x < y && (z > 0)) {
                    x += 1;
                }
                if (x < y >> (1)) {
                    y += 1;
                }
            }
        "#;

        let prog = parse_str(src).unwrap_or_else(|e| panic!("Wave 2 generics parse failed:\n{e}"));
        assert_eq!(prog.items.len(), 5);

        // 1. `struct FixedVec<T, usize N> { [N]T data; usize len; }`
        match &prog.items[0].kind {
            ItemKind::Struct(st) => {
                assert_eq!(st.name.as_str(), "FixedVec");
                assert_eq!(st.generic_params.len(), 2);
                assert_eq!(st.generic_params[0].name.as_str(), "T");
                assert_eq!(st.generic_params[0].kind, GenericParamKind::Type);
                assert_eq!(st.generic_params[1].name.as_str(), "N");
                assert_eq!(st.generic_params[1].kind, GenericParamKind::ConstUsize);
                match &st.fields[0].ty.kind {
                    TypeKind::Array { elem, len } => {
                        assert!(matches!(elem.kind, TypeKind::Named(_)));
                        assert!(matches!(len, ArrayLen::Expr(_)));
                    }
                    other => panic!("expected [N]T Array, got {other:?}"),
                }
            }
            other => panic!("expected Struct, got {other:?}"),
        }

        // 2. `T clamp<T>(T val, T lo, T hi)` with `T v = val;` local VarDecl
        match &prog.items[2].kind {
            ItemKind::Fn(f) => {
                assert_eq!(f.name.as_str(), "clamp");
                assert_eq!(f.generic_params.len(), 1);
                assert_eq!(f.generic_params[0].kind, GenericParamKind::Type);
                assert!(matches!(
                    f.body.as_ref().unwrap().stmts[0].kind,
                    StmtKind::VarDecl { .. }
                ));
            }
            other => panic!("expected generic Fn clamp, got {other:?}"),
        }

        // 3. Verify `main` statements: `[16]u8`, `FixedVec<u32, 4> { ... }`, `Boxed<Boxed<u32>>`, `clamp<f32>`, `x < y && (z > 0)`
        let main_fn = match &prog.items[4].kind {
            ItemKind::Fn(f) => f,
            other => panic!("expected main Fn, got {other:?}"),
        };
        let stmts = &main_fn.body.as_ref().unwrap().stmts;

        // Both `[16]u8 raw;` and `[16]u8 raw2;` use `ArrayLen::Fixed(16)`
        for idx in [0, 1] {
            match &stmts[idx].kind {
                StmtKind::VarDecl { ty, .. } => match &ty.kind {
                    TypeKind::Array { len, .. } => assert_eq!(*len, ArrayLen::Fixed(16)),
                    other => panic!("expected Fixed(16) Array, got {other:?}"),
                },
                other => panic!("expected VarDecl, got {other:?}"),
            }
        }

        // `Boxed<Boxed<u32>>` (`>>` split cleanly!)
        match &stmts[4].kind {
            StmtKind::VarDecl {
                ty,
                init: Some(init),
                ..
            } => {
                match &ty.kind {
                    TypeKind::Generic { base, args } => {
                        assert_eq!(base.as_str(), "Boxed");
                        assert_eq!(args.len(), 1);
                        assert!(matches!(
                            &args[0],
                            GenericArg::Type(TypeExpr {
                                kind: TypeKind::Generic { .. },
                                ..
                            })
                        ));
                    }
                    other => panic!("expected nested Generic type, got {other:?}"),
                }
                assert!(matches!(
                    init.kind,
                    ExprKind::StructLit {
                        ref generic_args,
                        ..
                    } if generic_args.len() == 1
                ));
            }
            other => panic!("expected Boxed<Boxed<u32>> VarDecl, got {other:?}"),
        }

        // `clamp<f32>(...)` calls
        for idx in [5, 6] {
            match &stmts[idx].kind {
                StmtKind::VarDecl {
                    init:
                        Some(Expr {
                            kind:
                                ExprKind::Call {
                                    generic_args, args, ..
                                },
                            ..
                        }),
                    ..
                } => {
                    assert_eq!(generic_args.len(), 1);
                    assert_eq!(args.len(), 3);
                }
                other => panic!("expected generic Call VarDecl at {idx}, got {other:?}"),
            }
        }

        // Empty `<>` and trait bounds `<T: Copy>` must fail cleanly
        assert!(parse_str("struct Bad<> { u8 x; }").is_err());
        assert!(parse_str("fn bad<T: Copy>(T x) {}").is_err());
    }

    #[test]
    fn wave2_parses_zero_heap_format_strings_and_preserves_subexpr_spans() {
        let src = concat!(
            "fn log_it(f32 t, u32 v, CmdFrame pkt) {\n",
            "    println(f\"temp={t:.2} hex={v:04x} HEX={v:X} bin={v:08b} sci={t:e} sci2={t:.3e} pkt={pkt:?} {{brace}}\");\n",
            "    println(f\"nested={CmdFrame { op: 1 }:?} path={hw.REG:02x}\");\n",
            "}\n",
        );

        let prog = parse_str(src).unwrap_or_else(|e| panic!("f-string parse failed:\n{e}"));
        let f = match &prog.items[0].kind {
            ItemKind::Fn(f) => f,
            other => panic!("expected Fn, got {other:?}"),
        };
        let stmts = &f.body.as_ref().unwrap().stmts;

        // First `println(f"...")`
        match &stmts[0].kind {
            StmtKind::Expr(Expr {
                kind: ExprKind::Call { args, .. },
                ..
            }) => match &args[0].value.kind {
                ExprKind::FormatString(parts) => {
                    // Expected parts:
                    // 0: "temp=", 1: {t:.2}, 2: " hex=", 3: {v:04x}, 4: " HEX=", 5: {v:X},
                    // 6: " bin=", 7: {v:08b}, 8: " sci=", 9: {t:e}, 10: " sci2=", 11: {t:.3e},
                    // 12: " pkt=", 13: {pkt:?}, 14: " {brace}"
                    assert_eq!(parts.len(), 15);
                    assert_eq!(parts[0], FormatPart::Literal("temp=".to_owned()));
                    match &parts[1] {
                        FormatPart::Hole { expr, spec, .. } => {
                            assert_eq!(*spec, FormatSpec::Fixed { precision: 2 });
                            // Verify exact line & col span inside line 2 (`    println(f"temp={t:.2}...`):
                            // `t` is at line 2, column 21!
                            assert_eq!(expr.span.line, 2);
                            assert_eq!(expr.span.col, 21);
                            assert_eq!(&src[expr.span.start as usize..expr.span.end as usize], "t");
                        }
                        other => panic!("expected Hole, got {other:?}"),
                    }
                    assert!(matches!(
                        parts[3],
                        FormatPart::Hole {
                            spec: FormatSpec::Hex {
                                upper: false,
                                width: Some(4)
                            },
                            ..
                        }
                    ));
                    assert!(matches!(
                        parts[5],
                        FormatPart::Hole {
                            spec: FormatSpec::Hex {
                                upper: true,
                                width: None
                            },
                            ..
                        }
                    ));
                    assert!(matches!(
                        parts[7],
                        FormatPart::Hole {
                            spec: FormatSpec::Bin { width: Some(8) },
                            ..
                        }
                    ));
                    assert!(matches!(
                        parts[9],
                        FormatPart::Hole {
                            spec: FormatSpec::Sci { precision: None },
                            ..
                        }
                    ));
                    assert!(matches!(
                        parts[11],
                        FormatPart::Hole {
                            spec: FormatSpec::Sci { precision: Some(3) },
                            ..
                        }
                    ));
                    assert!(matches!(
                        parts[13],
                        FormatPart::Hole {
                            spec: FormatSpec::Debug,
                            ..
                        }
                    ));
                    assert_eq!(parts[14], FormatPart::Literal(" {brace}".to_owned()));
                }
                other => panic!("expected FormatString arg, got {other:?}"),
            },
            other => panic!("expected Call stmt, got {other:?}"),
        }

        // Second `println(f"nested={CmdFrame { op: 1 }:?} path={hw.CONST:02x}")`
        match &stmts[1].kind {
            StmtKind::Expr(Expr {
                kind: ExprKind::Call { args, .. },
                ..
            }) => match &args[0].value.kind {
                ExprKind::FormatString(parts) => {
                    assert_eq!(parts.len(), 4);
                    match &parts[1] {
                        FormatPart::Hole { expr, spec, .. } => {
                            assert_eq!(*spec, FormatSpec::Debug);
                            assert!(matches!(expr.kind, ExprKind::StructLit { .. }));
                        }
                        other => panic!("expected nested StructLit Hole, got {other:?}"),
                    }
                }
                other => panic!("expected FormatString, got {other:?}"),
            },
            other => panic!("expected Call stmt, got {other:?}"),
        }

        // Malformed f-strings must produce clean errors (zero panics)
        for bad_fstr in [
            "fn main() { println(f\"empty={}\"); }",
            "fn main() { println(f\"unmatched }\"); }",
            "fn main() { println(f\"badspec={x:zzz}\"); }",
            "fn main() { println(f\"empty_spec={x:}\"); }",
            "fn main() { println(f\"unclosed {x\"); }",
            "fn main() { println(f\"extra={x y}\"); }",
        ] {
            assert!(
                parse_str(bad_fstr).is_err(),
                "expected f-string error on `{bad_fstr}`"
            );
        }
    }

    #[test]
    fn review_r1_raw_hex_byte_escapes_preserve_exact_bytes_and_utf8() {
        let src = r#"
fn main() {
    println("\x00\x7F5µs\x80\xAA\xFF");
    println(f"\xFF\xFE dt=5µs \x80");
    u8 ch = '\xFF';
}
"#;
        let prog = parse_str(src).expect("parse raw byte escapes");
        let stmts = match &prog.items[0].kind {
            ItemKind::Fn(f) => &f.body.as_ref().unwrap().stmts,
            other => panic!("expected Fn, got {other:?}"),
        };

        // 1. Standard string literal `"\x00\x7F5µs\x80\xAA\xFF"`
        match &stmts[0].kind {
            StmtKind::Expr(Expr {
                kind: ExprKind::Call { args, .. },
                ..
            }) => match &args[0].value.kind {
                ExprKind::Literal(Literal::String(s)) => {
                    let expected = vec![0x00, 0x7F, b'5', 0xC2, 0xB5, b's', 0x80, 0xAA, 0xFF];
                    assert_eq!(decoded_string_bytes(s), expected);
                    assert_eq!(decoded_string_byte_len(s), 9);
                }
                other => panic!("expected String literal, got {other:?}"),
            },
            other => panic!("expected Call, got {other:?}"),
        }

        // 2. Format string literal `f"\xFF\xFE dt=5µs \x80"`
        match &stmts[1].kind {
            StmtKind::Expr(Expr {
                kind: ExprKind::Call { args, .. },
                ..
            }) => match &args[0].value.kind {
                ExprKind::FormatString(parts) => match &parts[0] {
                    FormatPart::Literal(s) => {
                        let expected = vec![
                            0xFF, 0xFE, b' ', b'd', b't', b'=', b'5', 0xC2, 0xB5, b's', b' ', 0x80,
                        ];
                        assert_eq!(decoded_string_bytes(s), expected);
                        assert_eq!(decoded_string_byte_len(s), expected.len());
                    }
                    other => panic!("expected FormatPart::Literal, got {other:?}"),
                },
                other => panic!("expected FormatString, got {other:?}"),
            },
            other => panic!("expected Call, got {other:?}"),
        }

        // 3. Char literal `'\xFF'` still produces integer 255
        match &stmts[2].kind {
            StmtKind::VarDecl {
                init:
                    Some(Expr {
                        kind:
                            ExprKind::Literal(Literal::Int {
                                value: 255,
                                suffix: None,
                            }),
                        ..
                    }),
                ..
            } => {}
            other => panic!("expected `u8 ch = 255`, got {other:?}"),
        }
    }

    #[test]
    fn review_r1_generic_calls_never_hijack_multi_arg_relational_comparisons() {
        let app_src = r#"
import math.dsp as dsp;

const u32 GLOBAL_LO = 3;

fn check_window(bool lo_ok, bool hi_ok) {
    assert(lo_ok && hi_ok);
}

fn main() {
    u32 a = 3;
    u32 b = 10;
    u32 c = 8;
    u32 d = 4;
    Pair s = Pair { a: 3, b: 10 };

    // 1. Unambiguous comparisons (after `>` is NOT `(` or allowed `{`, or parenthesized, or binary rhs):
    check_window(a < b, c > d + 1);
    check_window((GLOBAL_LO < b), (c > (d + 1)));
    check_window(s.a < b, c > d + 1);
    [2]bool flags = [(a < 10), (c > (d + 1))];
    [2]bool flags2 = [a < 10, c > d + 1];

    // 2. Real generic calls (`clamp<f32>`, `dsp.clamp<f32>`, parenthesized const expr):
    f32 x1 = clamp<f32>(1.5f32, 0.0f32, 2.0f32);
    f32 x2 = dsp.clamp<f32>(x1, 0.0f32, 1.0f32);

    // 3. Control-flow headers disallow `{` struct literals, so `if x < y { ... }` never hijacks `{`:
    if a < b {
        a += 1;
    }

    // 4. `f(a < b, c > (d + 1))` syntactically commits as a generic call `a<b, c>(d + 1)`
    //    (rejected in sema with the parenthesization help):
    check_window(a < b, c > (d + 1));
    println(f"{flags[0]} {flags2[0]} {x2}");
}
"#;
        let prog = parse_str_named(app_src, "app.zev").expect("pure syntactic parse app.zev");

        let main_fn = prog
            .items
            .iter()
            .find_map(|it| match &it.kind {
                ItemKind::Fn(f) if f.name.name == "main" => Some(f),
                _ => None,
            })
            .unwrap();
        let stmts = &main_fn.body.as_ref().unwrap().stmts;

        // Verify comparisons in `check_window(...)` remain 2-arg calls:
        for idx in [5usize, 6, 7] {
            match &stmts[idx].kind {
                StmtKind::Expr(Expr {
                    kind:
                        ExprKind::Call {
                            args, generic_args, ..
                        },
                    ..
                }) => {
                    assert!(generic_args.is_empty());
                    assert_eq!(args.len(), 2, "stmt {idx} should have 2 comparison args");
                }
                other => panic!("expected 2-arg Call at stmt {idx}, got {other:?}"),
            }
        }

        // Verify `[ (a < 10), (c > (d + 1)) ]` and `[a < 10, c > d + 1]` parsed as 2-element `ArrayLit`
        for idx in [8usize, 9] {
            match &stmts[idx].kind {
                StmtKind::VarDecl {
                    init:
                        Some(Expr {
                            kind: ExprKind::ArrayLit(elems),
                            ..
                        }),
                    ..
                } => assert_eq!(elems.len(), 2, "stmt {idx} should have 2 array elements"),
                other => panic!("expected 2-element ArrayLit at stmt {idx}, got {other:?}"),
            }
        }

        // Verify `clamp<f32>` and `dsp.clamp<f32>` parsed with 1 generic arg & 3 call args
        for idx in [10usize, 11] {
            match &stmts[idx].kind {
                StmtKind::VarDecl {
                    init:
                        Some(Expr {
                            kind:
                                ExprKind::Call {
                                    generic_args, args, ..
                                },
                            ..
                        }),
                    ..
                } => {
                    assert_eq!(generic_args.len(), 1);
                    assert_eq!(args.len(), 3);
                }
                other => panic!("expected generic Call at stmt {idx}, got {other:?}"),
            }
        }

        // Verify `if a < b { ... }` kept `a < b` as Binary comparison
        assert!(matches!(
            &stmts[12].kind,
            StmtKind::If {
                cond: Expr {
                    kind: ExprKind::Binary {
                        op: BinaryOp::Lt,
                        ..
                    },
                    ..
                },
                ..
            }
        ));

        // Verify `check_window(a < b, c > (d + 1))` syntactically parsed its inner arg as `a<b, c>(d + 1)`
        match &stmts[13].kind {
            StmtKind::Expr(Expr {
                kind: ExprKind::Call { args, .. },
                ..
            }) => {
                assert_eq!(args.len(), 1);
                assert!(matches!(
                    &args[0].value.kind,
                    ExprKind::Call { generic_args, .. } if generic_args.len() == 2
                ));
            }
            other => panic!("expected Call wrapping syntactic a<b, c>(d + 1), got {other:?}"),
        }
    }

    #[test]
    fn review_r1_f32_direct_parse_fstring_comments_pad_specs_shl_generics_and_local_const() {
        // 1. M11: Direct `f32` parsing avoids `str -> f64 -> f32` double-rounding at binary64/binary32 midpoints!
        // `1 + 2^-24 + 2^-54` = `1.000000059604644775390625 + ~5.55e-17`:
        // - `parse::<f64>()` rounds to even `1 + 2^-24`, and `(v64 as f32)` rounds to even `1.0f32` (`0x3f800000`).
        // - Direct `parse::<f32>()` sees the exact decimal > midpoint and rounds up to `0x3f800001`!
        let f32_src =
            "fn main() { f32 x = 1.000000059604644830901776231257827021181583404541015625f32; }";
        let f32_prog = parse_str(f32_src).expect("parse midpoint f32");
        if let ItemKind::Fn(f) = &f32_prog.items[0].kind {
            if let StmtKind::VarDecl {
                init:
                    Some(Expr {
                        kind: ExprKind::Literal(Literal::Float { bits, suffix }),
                        ..
                    }),
                ..
            } = &f.body.as_ref().unwrap().stmts[0].kind
            {
                assert_eq!(*suffix, Some(PrimType::F32));
                assert_eq!((f64::from_bits(*bits) as f32).to_bits(), 0x3f80_0001);
            } else {
                panic!("expected Float literal");
            }
        }

        // 2. M12 + Low #1: Comments (`/* a:b */`, `/* } */`) inside f-string holes + `<<`/`|`/`&`/`^` in generic const args!
        let clean_src = r#"
struct FixedVec<T, usize N> { [N]T data; }
struct Outer<T> { T inner; }

fn main() {
    FixedVec<u8, 1 << 4> v1;
    Outer<FixedVec<u8, (1 << 3) | 2>> v2;
    u32 x = 10;
    u32 y = 0xab;
    println(f"val={x /* a:b */} hex={y /* } */ :04x}");
}
"#;
        let clean_prog = parse_str(clean_src).expect("parse << generic args and f-string comments");
        assert_eq!(clean_prog.items.len(), 3);

        // 3. M12: Escaped `\"` inside `{...}` hole produces targeted hint instead of bogus unterminated hole!
        let esc_err = parse_str("fn main() { println(f\"hdr={\\\"ok\\\"}\"); }")
            .expect_err("expected targeted error on escaped quote inside hole");
        assert!(
            esc_err.message.contains("unescaped")
                && esc_err
                    .hint
                    .as_deref()
                    .unwrap_or("")
                    .contains("hdr={\"ok\"}"),
            "unexpected error: {esc_err}"
        );

        // 4. M13: Reject `:4x`, `:8b`, `:0x`, `:0X`, `:0b`, `:00x` with actionable zero-pad / prefix hints!
        let e_4x = parse_str("fn main() { u16 v = 1; println(f\"{v:4x}\"); }").unwrap_err();
        assert!(
            e_4x.hint.as_deref().unwrap_or("").contains(":04x"),
            "{e_4x}"
        );
        let e_8b = parse_str("fn main() { u8 v = 1; println(f\"{v:8b}\"); }").unwrap_err();
        assert!(
            e_8b.hint.as_deref().unwrap_or("").contains(":08b"),
            "{e_8b}"
        );
        let e_0x = parse_str("fn main() { u16 v = 1; println(f\"{v:0x}\"); }").unwrap_err();
        assert!(
            e_0x.hint.as_deref().unwrap_or("").contains("0x{val:x}")
                && e_0x.hint.as_deref().unwrap_or("").contains(":02x"),
            "{e_0x}"
        );
        let e_0b = parse_str("fn main() { u8 v = 1; println(f\"{v:0b}\"); }").unwrap_err();
        assert!(
            e_0b.hint.as_deref().unwrap_or("").contains("0b{val:b}"),
            "{e_0b}"
        );

        // 5. Low #2: Reject block-local `const Type name = ...;` with clear top-level guidance!
        let e_local_const =
            parse_str("fn main() { const usize CAP = 8; }").expect_err("reject block-local const");
        assert!(
            e_local_const.message.contains("block-local `const`")
                && e_local_const
                    .hint
                    .as_deref()
                    .unwrap_or("")
                    .contains("module top level"),
            "{e_local_const}"
        );

        // 6. Low #3: Malformed floats (`0x1.5`, `1._5`, `1.<2.0`, `1.==1.0`, `.5`) emit float diagnostics,
        //    while hex/dec ranges (`0x00..0x10`, `0..4`) still parse cleanly!
        parse_str("fn main() { for (u8 i in 0x00..0x10) {} }").expect("hex range must still parse");
        for bad_num in [
            "fn main() { f64 x = 0x1.5; }",
            "fn main() { f64 x = 1._5; }",
            "fn main() { bool b = 1.<2.0; }",
            "fn main() { bool b = 1.==1.0; }",
            "fn main() { f64 x = .5; }",
        ] {
            let err = parse_str(bad_num).expect_err(&format!("expected error on `{bad_num}`"));
            assert!(
                err.message.contains("float literal"),
                "expected float literal diagnostic for `{bad_num}`, got: {err}"
            );
        }
    }

    #[test]
    fn parses_pub_visibility_across_all_top_level_items_and_attrs() {
        let src = r#"
pub struct Vec2 { u32 x; u32 y; }
@packed pub packed struct Frame { u8 cmd; u16 crc; }
pub enum Mode { Idle = 0, Run = 1 }
pub const u32 BAUD = 115200;
pub fn helper(u32 a) {}
pub fn add(u32 a, u32 b) -> u32 { return a + b; }
pub extern "C" fn c_hook(u32 v);
extern "C" {
    pub fn ext_pub(u32 x);
    fn ext_priv(u32 y) -> u32;
}
@high pub task worker(u32 id) {}
pub @normal isr exti0(u32 pin) {}
pub mmio UartRegs { @rw u32 cr1 @ 0x00; }
pub irq IRQ_UART = 37;
pub watch<u32> w_val = 0;
pub inbox<u8, 4> rx_q;
pub signal ready;
pub ping_pong<u8, 8> pp;
pub @dma_safe pool<u8, 16> pkt_pool;
"#;
        let prog = parse_str(src).expect("all top-level pub declarations must parse");
        assert_eq!(prog.items.len(), 18);
        for (idx, item) in prog.items.iter().enumerate() {
            if idx == 8 {
                assert!(
                    !item.is_pub,
                    "extern block item `ext_priv` should default to private"
                );
            } else {
                assert!(
                    item.is_pub,
                    "item #{idx} should be marked `pub`: {:?}",
                    item.kind
                );
            }
        }
    }

    #[test]
    fn parses_mmio_irq_parameterized_isr_state_and_qualified_constructs() {
        let src = r#"
import drivers.uart as drv;
import board.gpio as gpio;

const u32 RCC_BASE = 0x4002_1000;

pub mmio UartRegs {
    @rw u32 cr1 @ 0x00;
    @ro u32 isr @ 0x1C;
    @w1c u32 icr @ 0x20;
    @rw [4]u32 brr @ 0x24;
    @wo u8 dr @ 0x28;
}

pub mmio USART1 : UartRegs @ 0x4001_1000;
mmio USART2 : gpio.UartRegs @ 0x4000_4400;
mmio Rcc @ RCC_BASE {
    @rw u32 cr @ 0x00;
}

pub irq IRQ_USART1 = 37;

isr no_parens {
    state u32 count = 0;
    state [4]u32 ring = [0, 0, 0, 0];
    state drv.RingCfg cfg = drv.RingCfg { cap: 4 };
    state u32 uninit;
    count += 1;
}

isr empty_parens() {
    u32 irq = 3;
    u32 state = 1;
    [2]u32 arr = [0, 0];
    state += irq;
    state = 2;
    arr[0] = state;
}

isr with_params(*UartRegs regs, u32 mask) {
    state u32 ticks = 0;
    ticks += mask;
    regs.icr = ticks;
}

fn mmio(u32 a) {}

fn main() {
    mmio(1);
    u32 state = 0;
    [2]u32 state_arr = [0, 0];
    state_arr[0] = 1;
    let_free: {
        drv.Config c = drv.Config { baud: 115200, .stop_bits = 1 };
        drv.Box<u32, 4> b1 = drv.Box<u32, 4> { val: 7 };
        drv.Box<u32, 4> b2 = drv.Box<u32, (2 + 2)> { val: 8 };
        if (b1.val < b2.val) {
            spawn drv.worker(c.baud) @high;
        }
    }
}
"#;
        // Replace label if unsupported (`{ ... }` block directly works).
        let src = src.replace("let_free: ", "");
        let prog = parse_str(&src).expect("new syntax constructs must parse cleanly");

        // Check mmio forms
        let mmio_decls: Vec<&MmioDecl> = prog
            .items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Mmio(m) => Some(m),
                _ => None,
            })
            .collect();
        assert_eq!(mmio_decls.len(), 4);
        assert!(matches!(&mmio_decls[0].kind, MmioDeclKind::Layout { regs } if regs.len() == 5));
        assert!(matches!(&mmio_decls[1].kind, MmioDeclKind::Instance { .. }));
        assert!(matches!(
            &mmio_decls[2].kind,
            MmioDeclKind::Instance { layout, .. } if matches!(&layout.kind, TypeKind::Path(p) if p.len() == 2)
        ));
        assert!(matches!(
            &mmio_decls[3].kind,
            MmioDeclKind::Inline { regs, .. } if regs.len() == 1
        ));

        // Check irq
        let irqs: Vec<&IrqDecl> = prog
            .items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Irq(q) => Some(q),
                _ => None,
            })
            .collect();
        assert_eq!(irqs.len(), 1);
        assert_eq!(irqs[0].name.name, "IRQ_USART1");

        // Check ISRs
        let isrs: Vec<&IsrDecl> = prog
            .items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Isr(r) => Some(r),
                _ => None,
            })
            .collect();
        assert_eq!(isrs.len(), 3);
        assert!(isrs[0].params.is_empty());
        assert!(isrs[1].params.is_empty());
        assert_eq!(isrs[2].params.len(), 2);

        // Check contextual `state` declarations vs assignments/arrays
        let state_var_count = isrs[0]
            .body
            .stmts
            .iter()
            .filter(|s| matches!(s.kind, StmtKind::StateVar { .. }))
            .count();
        assert_eq!(state_var_count, 4);

        // Verify contextual `state += irq;` did NOT become StateVar
        assert!(isrs[1]
            .body
            .stmts
            .iter()
            .all(|s| !matches!(s.kind, StmtKind::StateVar { .. })));

        // Check qualified StructLit and Spawn in main
        let main_fn = prog
            .items
            .iter()
            .find_map(|i| match &i.kind {
                ItemKind::Fn(f) if f.name.name == "main" => Some(f),
                _ => None,
            })
            .unwrap();
        let body = main_fn.body.as_ref().unwrap();
        // Verify `state[0] = 1;` also parses normally inside main
        parse_str("fn test_state_idx() { [2]u32 state = [0, 0]; state[0] = 1; }")
            .expect("state[0] = 1 must parse as indexing, not StateVar");

        let blk = match &body.stmts[4].kind {
            StmtKind::Block(b) => b,
            other => panic!("expected block, got {other:?}"),
        };
        assert!(matches!(
            &blk.stmts[0].kind,
            StmtKind::VarDecl { init: Some(Expr { kind: ExprKind::StructLit { module: Some(m), name, generic_args, .. }, .. }), .. }
                if m.name == "drv" && name.name == "Config" && generic_args.is_empty()
        ));
        assert!(matches!(
            &blk.stmts[1].kind,
            StmtKind::VarDecl { init: Some(Expr { kind: ExprKind::StructLit { module: Some(m), name, generic_args, .. }, .. }), .. }
                if m.name == "drv" && name.name == "Box" && generic_args.len() == 2
        ));
        assert!(matches!(
            &blk.stmts[2].kind,
            StmtKind::VarDecl { init: Some(Expr { kind: ExprKind::StructLit { module: Some(m), name, generic_args, .. }, .. }), .. }
                if m.name == "drv" && name.name == "Box" && generic_args.len() == 2
        ));
        if let StmtKind::If { then_block, .. } = &blk.stmts[3].kind {
            assert!(matches!(
                &then_block.stmts[0].kind,
                StmtKind::Spawn(SpawnStmt { module: Some(m), task, priority: Some(Priority::High), .. })
                    if m.name == "drv" && task.name == "worker"
            ));
        } else {
            panic!("expected if stmt");
        }
    }

    #[test]
    fn rejects_malformed_mmio_declarations() {
        for bad in [
            "mmio Bad { u32 cr1 @ 0x00; }",
            "mmio Bad { @dma_safe u32 cr1 @ 0x00; }",
            "mmio Bad { @rw u32 cr1; }",
            "mmio Bad { @ro u32 isr @ 0x04 }",
        ] {
            parse_str(bad).expect_err(&format!("expected parse error on `{bad}`"));
        }
    }

    #[test]
    fn parses_qualified_switch_patterns_and_module_member_comparisons() {
        let src = r#"
import mod_consts as m;
import sw_mod as sm;

fn take2(bool x, bool y) {}

fn main() {
    u32 b = 2;
    u32 d = 3;
    take2((m.A < b), (m.C > (d + 1)));
    take2(m.A < b, m.C > d + 1);

    u32 k = 0;
    switch (k) {
        sm.Kind.Alpha => {}
        sm.MODE_A => {}
        default => {}
    }
}
"#;
        let prog = parse_str(src).expect("qualified switch & module comparison must parse");
        let main_fn = prog
            .items
            .iter()
            .find_map(|i| match &i.kind {
                ItemKind::Fn(f) if f.name.name == "main" => Some(f),
                _ => None,
            })
            .unwrap();
        let stmts = &main_fn.body.as_ref().unwrap().stmts;
        for idx in [2usize, 3] {
            assert!(matches!(
                &stmts[idx].kind,
                StmtKind::Expr(Expr { kind: ExprKind::Call { args, .. }, .. }) if args.len() == 2
            ));
        }
        if let StmtKind::Switch { arms, .. } = &stmts[5].kind {
            assert!(matches!(
                &arms[0].pat,
                SwitchPat::Qualified { path, .. }
                    if path.iter().map(|i| i.as_str()).collect::<Vec<_>>() == vec!["sm", "Kind", "Alpha"]
            ));
            assert!(matches!(
                &arms[1].pat,
                SwitchPat::Ident { prefix: Some(p), name, .. } if p.as_str() == "sm" && name.as_str() == "MODE_A"
            ));
        } else {
            panic!("expected Switch stmt");
        }
    }

    #[test]
    fn parses_byte_ring_primitive_and_ptr() {
        let src = r#"
byte_ring<64> usb_rx;

isr usb_isr(*byte_ring<64> rx) {
    rx.push_byte(65, boost: @high);
}

task worker(*byte_ring<64> rx) {
    loop {
        []u8 chunk = rx.wait_read();
        rx.consume(chunk.len);
    }
}
"#;
        let prog = parse_str(src).expect("byte_ring<64> and *byte_ring<64> must parse");
        match &prog.items[0].kind {
            ItemKind::OsPrimitive(os) => {
                assert_eq!(os.name.as_str(), "usb_rx");
                assert_eq!(os.ty, OsPrimitiveType::ByteRing { cap: 64 });
            }
            other => panic!("expected OsPrimitive(ByteRing), got {other:?}"),
        }
        match &prog.items[1].kind {
            ItemKind::Isr(isr) => {
                assert!(matches!(
                    &isr.params[0].ty.kind,
                    TypeKind::Ptr(inner)
                        if matches!(inner.kind, TypeKind::OsPrimitive(OsPrimitiveType::ByteRing { cap: 64 }))
                ));
            }
            other => panic!("expected Isr, got {other:?}"),
        }
    }

    #[test]
    fn parses_unsized_primitive_views_and_subsystem_decl() {
        let src = r#"
struct UartSubsys {
    byte_ring<64> rx;
    inbox<u8, 8> tx;
}

UartSubsys uart0;

fn take_views(*byte_ring r, *inbox<u8> q, *inbox<watch<u8>> nested) {}

fn main() {
    byte_ring bare_r;
    inbox<u8> bare_q;
}
"#;
        let prog = parse_str(src).expect("unsized primitive views & subsystem decl must parse");
        assert!(matches!(
            &prog.items[1].kind,
            ItemKind::GlobalVar(g) if g.name.as_str() == "uart0" && g.init.is_none()
        ));
        if let ItemKind::Fn(f) = &prog.items[2].kind {
            assert!(matches!(
                &f.params[0].ty.kind,
                TypeKind::Ptr(inner) if matches!(inner.kind, TypeKind::OsPrimitive(OsPrimitiveType::ByteRingUnsized))
            ));
            assert!(matches!(
                &f.params[1].ty.kind,
                TypeKind::Ptr(inner) if matches!(inner.kind, TypeKind::OsPrimitive(OsPrimitiveType::InboxUnsized { .. }))
            ));
            assert!(matches!(
                &f.params[2].ty.kind,
                TypeKind::Ptr(inner) if matches!(
                    &inner.kind,
                    TypeKind::OsPrimitive(OsPrimitiveType::InboxUnsized { elem })
                        if matches!(elem.kind, TypeKind::OsPrimitive(OsPrimitiveType::Watch { .. }))
                )
            ));
        } else {
            panic!("expected Fn take_views");
        }
        if let ItemKind::Fn(m) = &prog.items[3].kind {
            let stmts = &m.body.as_ref().unwrap().stmts;
            assert!(matches!(
                &stmts[0].kind,
                StmtKind::VarDecl { ty, init: None, .. }
                    if matches!(ty.kind, TypeKind::OsPrimitive(OsPrimitiveType::ByteRingUnsized))
            ));
            assert!(matches!(
                &stmts[1].kind,
                StmtKind::VarDecl { ty, init: None, .. }
                    if matches!(ty.kind, TypeKind::OsPrimitive(OsPrimitiveType::InboxUnsized { .. }))
            ));
        } else {
            panic!("expected Fn main");
        }
    }

    #[test]
    fn regression_s_m6_scoped_generics_and_non_usize_top_level_values() {
        let app_src = r#"
import math.mod as m;

const u32 HI = 10;
const usize CAP = 4;

struct FixedVec<T, usize N> {
    [N]T items;
}

struct Holder {
    u32 clamp;
}

fn grow_n<usize N>(FixedVec<u32, N> v) -> FixedVec<u32, N + 1> {
    return FixedVec<u32, (N + 1)> { items: [0] };
}

fn check_pair(bool a, bool b) {
    assert(a && b);
}

fn use_n<usize N>(u32 cmp, FixedVec<u32, N> v) {
    // Generic type/fn instantiations with usize CAP, (N + 1), and local Name<T> var decls:
    FixedVec<u32, CAP> fixed = FixedVec<u32, CAP> { items: [1, 2, 3, 4] };
    FixedVec<u32, N + 1> grown = grow_n<N>(v);
    u32 c = m.clamp<u32>(cmp, 0, HI);
    m.Pair<u8, u16> mod_var;

    // Parenthesized comparisons remain two arguments; unparenthesized binary `+` inside `<...>`
    // in expression position restores as comparisons!
    Holder b = Holder { clamp: 1 };
    check_pair((b.clamp < 2), (HI > (0)));
    check_pair(b.clamp < 2 + 1, HI > (0));
    if cmp < HI {
        c = 1;
    }
    println(f"{fixed.items[0]} {grown.items[0]} {c} {mod_var}");
}
"#;
        let prog = parse_str_named(app_src, "app.zev")
            .expect("parse app with purely syntactic generics and declarations");
        let use_n_fn = prog
            .items
            .iter()
            .find_map(|it| match &it.kind {
                ItemKind::Fn(f) if f.name.name == "use_n" => Some(f),
                _ => None,
            })
            .unwrap();
        let stmts = &use_n_fn.body.as_ref().unwrap().stmts;

        // stmt 0: FixedVec<u32, CAP> local type + struct lit
        assert!(matches!(
            &stmts[0].kind,
            StmtKind::VarDecl { ty, .. } if matches!(&ty.kind, TypeKind::Generic { args, .. } if args.len() == 2)
        ));
        // stmt 1: grow_n<N>(v) generic call
        assert!(matches!(
            &stmts[1].kind,
            StmtKind::VarDecl {
                init: Some(Expr { kind: ExprKind::Call { generic_args, args, .. }, .. }),
                ..
            } if generic_args.len() == 1 && args.len() == 1
        ));
        // stmt 2: m.clamp<u32>(cmp, 0, HI) generic call
        assert!(matches!(
            &stmts[2].kind,
            StmtKind::VarDecl {
                init: Some(Expr { kind: ExprKind::Call { generic_args, args, .. }, .. }),
                ..
            } if generic_args.len() == 1 && args.len() == 3
        ));
        // stmt 3: m.Pair<u8, u16> mod_var; declaration at statement start
        assert!(matches!(
            &stmts[3].kind,
            StmtKind::VarDecl {
                name,
                ty: TypeExpr { kind: TypeKind::GenericPath { args, .. }, .. },
                ..
            } if name.name == "mod_var" && args.len() == 2
        ));
        // stmts 5..=6: both `check_pair(...)` calls stay 2-arg relational comparisons
        for idx in [5usize, 6] {
            match &stmts[idx].kind {
                StmtKind::Expr(Expr {
                    kind:
                        ExprKind::Call {
                            args, generic_args, ..
                        },
                    ..
                }) => {
                    assert!(generic_args.is_empty(), "stmt {idx} hijacked generic args!");
                    assert_eq!(args.len(), 2, "stmt {idx} expected 2 comparison args");
                }
                other => panic!("expected 2-arg Call at stmt {idx}, got {other:?}"),
            }
        }
        // stmt 7: `if cmp < HI { ... }` keeps comparison
        assert!(matches!(
            &stmts[7].kind,
            StmtKind::If {
                cond: Expr {
                    kind: ExprKind::Binary {
                        op: BinaryOp::Lt,
                        ..
                    },
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn regression_s_l1_c_style_struct_generic_path_ptr_casts() {
        let src = r#"
import lib.mod_m as mod_m;

struct MyStruct {
    u32 x;
}

struct Box<T, usize N> {
    [N]T arr;
}

fn test(*void vp, *i32 a) {
    *MyStruct p1 = vp as *MyStruct;
    *mod_m.MyStruct p3 = vp as *mod_m.MyStruct;
    *Box<u32, 4> p5 = vp as *Box<u32, 4>;

    i32 s1 = (*a) - 1;
    i32 s2 = (*a) * 2;
    i32 s3 = (*a) + 3;
}
"#;
        let prog = parse_str(src).expect("prefix struct/generic/path ptr `as` casts must parse");
        let f = match &prog.items[3].kind {
            ItemKind::Fn(f) => f,
            other => panic!("expected Fn, got {other:?}"),
        };
        let stmts = &f.body.as_ref().unwrap().stmts;
        for (idx, stmt) in stmts.iter().take(3).enumerate() {
            assert!(
                matches!(
                    &stmt.kind,
                    StmtKind::VarDecl {
                        init: Some(Expr {
                            kind: ExprKind::Cast { .. },
                            ..
                        }),
                        ..
                    }
                ),
                "stmt {idx} must parse as Cast"
            );
        }
        for c_cast_stmt in [
            "struct MyStruct { u32 x; }\nfn test(*void vp) { *MyStruct p1 = (*MyStruct)vp; }",
            "import lib.mod_m as mod_m;\nfn test(*void vp) { *mod_m.MyStruct p3 = (*mod_m.MyStruct)vp; }",
            "struct Box<T, usize N> { [N]T arr; }\nfn test(*void vp) { *Box<u32, 4> p5 = (*Box<u32, 4>)vp; }",
        ] {
            let err = parse_str(c_cast_stmt).expect_err("retired C-style pointer cast must fail with P0017");
            let msg = format!("{err:?}");
            assert!(msg.contains("P0017"), "expected P0017, got:\n{msg}");
        }
        for (idx, expected_op) in [
            (3usize, BinaryOp::Sub),
            (4, BinaryOp::Mul),
            (5, BinaryOp::Add),
        ] {
            assert!(
                matches!(
                    &stmts[idx].kind,
                    StmtKind::VarDecl {
                        init: Some(Expr {
                            kind: ExprKind::Binary {
                                op,
                                lhs,
                                ..
                            },
                            ..
                        }),
                        ..
                    } if *op == expected_op && matches!(lhs.kind, ExprKind::Unary { op: UnaryOp::Deref, .. })
                ),
                "stmt {idx} must remain binary deref expression"
            );
        }
    }

    #[test]
    fn parses_impl_blocks_methods_and_rejects_invalid_impl_modifiers() {
        let src = r#"
struct Pin { u8 port; u8 pin; }
struct FixedVec<T, usize N> { [N]T data; usize len; }

impl Pin {
    pub fn make(u8 port, u8 pin) -> Pin {
        return Pin { port: port, pin: pin };
    }

    pub fn port_id(self) -> u8 {
        return self.port;
    }

    fn bump(*self, u8 delta) {
        self.pin += delta;
    }
}

impl<T, usize N> FixedVec<T, N> {
    pub fn push(*self, T item) {
        self.data[0] = item;
    }
}

impl<usize N> FixedVec<f32, N> {
    pub fn avg(*self) -> f32 {
        return self.data[0];
    }
}
"#;
        let prog = parse_str(src)
            .expect("valid concrete, generic, and specialized impl blocks should parse");
        assert_eq!(prog.items.len(), 5);
        let imp_pin = match &prog.items[2].kind {
            ItemKind::Impl(d) => d,
            other => panic!("expected Impl, got {other:?}"),
        };
        assert_eq!(imp_pin.methods.len(), 3);
        assert_eq!(imp_pin.methods[0].receiver, None);
        assert!(imp_pin.methods[0].is_pub);
        assert_eq!(imp_pin.methods[1].receiver, Some(SelfReceiver::Value));
        assert_eq!(imp_pin.methods[2].receiver, Some(SelfReceiver::Ptr));
        assert!(!imp_pin.methods[2].is_pub);

        let imp_vec = match &prog.items[3].kind {
            ItemKind::Impl(d) => d,
            other => panic!("expected generic Impl, got {other:?}"),
        };
        assert_eq!(imp_vec.generic_params.len(), 2);
        assert_eq!(imp_vec.target_generic_args.len(), 2);

        let imp_spec = match &prog.items[4].kind {
            ItemKind::Impl(d) => d,
            other => panic!("expected specialized Impl, got {other:?}"),
        };
        assert_eq!(imp_spec.generic_params.len(), 1);
        assert_eq!(imp_spec.target_generic_args.len(), 2);

        for bad in [
            "struct S { u8 x; }\npub impl S { pub fn f(self) {} }",
            "struct S { u8 x; }\nimpl S { fn f<T>(self, T a) {} }",
            "struct S { u8 x; }\nimpl S { extern \"C\" fn c_hook(); }",
        ] {
            parse_str(bad).expect_err(&format!("expected parse failure for:\n{bad}"));
        }
    }

    #[test]
    fn parses_contextual_wait_blocks_and_preserves_wait_identifier() {
        let src = r#"
task sup() {
    u32 wait_cnt = 0;
    u32 wait = 1;
    wait {
        u32 c = q.recv() => { wait_cnt += c; }
        sig.wait() => { wait_cnt += wait; }
        timeout 50ms => { wait_cnt += 1; }
    }
    u32 dl = sys.now_ms() + 20u32;
    wait {
        []u8 chunk = rx.wait_read() => rx.consume(chunk.len);
        deadline dl => { dl += 20u32; }
    }
}
"#;
        let prog = parse_str(src).expect("wait { ... } should parse cleanly");
        let td = match &prog.items[0].kind {
            ItemKind::Task(t) => t,
            other => panic!("expected Task, got {other:?}"),
        };
        match &td.body.stmts[2].kind {
            StmtKind::Wait { arms, timer } => {
                assert_eq!(arms.len(), 2);
                assert!(arms[0].bind.is_some());
                assert!(arms[1].bind.is_none());
                let t = timer.as_ref().expect("timeout arm expected");
                assert_eq!(t.kind, WaitTimerKind::Timeout);
            }
            other => panic!("expected Wait stmt, got {other:?}"),
        }
        match &td.body.stmts[4].kind {
            StmtKind::Wait { arms, timer } => {
                assert_eq!(arms.len(), 1);
                let t = timer.as_ref().expect("deadline arm expected");
                assert_eq!(t.kind, WaitTimerKind::Deadline);
            }
            other => panic!("expected second Wait stmt, got {other:?}"),
        }
        parse_str("fn main() { wait { a.wait() => {} timeout 5ms => {} deadline 10u32 => {} } }")
            .expect_err("duplicate timer arm in wait block must fail parse");
    }

    #[test]
    fn preserves_integer_literal_width_suffixes_and_rejects_c_suffixes() {
        let prog = parse_str(
            r#"
fn main() {
    u8 a = 10u8;
    i32 b = -5i32;
    usize c = 42usize;
    u32 d = 7;
}
"#,
        )
        .expect("valid width suffixes parse cleanly");
        let ItemKind::Fn(f) = &prog.items[0].kind else {
            panic!("expected fn");
        };
        let body = f.body.as_ref().expect("fn body");
        assert!(matches!(
            body.stmts[0].kind,
            StmtKind::VarDecl {
                init: Some(Expr {
                    kind: ExprKind::Literal(Literal::Int {
                        value: 10,
                        suffix: Some(PrimType::U8)
                    }),
                    ..
                }),
                ..
            }
        ));
        assert!(matches!(
            body.stmts[1].kind,
            StmtKind::VarDecl {
                init: Some(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        expr: ref inner,
                    },
                    ..
                }),
                ..
            } if matches!(
                inner.kind,
                ExprKind::Literal(Literal::Int { value: 5, suffix: Some(PrimType::I32) })
            )
        ));
        assert!(matches!(
            body.stmts[2].kind,
            StmtKind::VarDecl {
                init: Some(Expr {
                    kind: ExprKind::Literal(Literal::Int {
                        value: 42,
                        suffix: Some(PrimType::Usize)
                    }),
                    ..
                }),
                ..
            }
        ));
        assert!(matches!(
            body.stmts[3].kind,
            StmtKind::VarDecl {
                init: Some(Expr {
                    kind: ExprKind::Literal(Literal::Int {
                        value: 7,
                        suffix: None
                    }),
                    ..
                }),
                ..
            }
        ));
        for bad in [
            "fn main() { u32 x = 10u; }",
            "fn main() { u32 x = 10UL; }",
            "fn main() { i32 x = 10L; }",
        ] {
            parse_str(bad).expect_err("C integer suffix must be rejected");
        }
    }

    #[test]
    fn parser_depth_limit_and_item_recovery() {
        let handle = std::thread::Builder::new()
            .name("parser-depth-test".to_string())
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                // 40 nested parens and 40 nested `if` blocks exceed the old 24 cap and now succeed.
                let open_parens = "(".repeat(40);
                let close_parens = ")".repeat(40);
                let src_parens =
                    format!("fn f() -> u32 {{ return {open_parens}1u32{close_parens}; }}\n");
                parse_str(&src_parens).expect("40 nested parens must parse");

                let mut src_ifs = String::from("fn f(bool c) {\n");
                for _ in 0..40 {
                    src_ifs.push_str("if (c) {\n");
                }
                src_ifs.push_str("u32 _x = 1;\n");
                for _ in 0..40 {
                    src_ifs.push_str("}\n");
                }
                src_ifs.push_str("}\n");
                parse_str(&src_ifs).expect("40 nested if blocks must parse");

                // 300-deep inputs fail gracefully on a 2 MiB debug thread stack without overflowing.
                let deep_open = "(".repeat(300);
                let deep_close = ")".repeat(300);
                let err_p = parse_str(&format!(
                    "fn f() -> u32 {{ return {deep_open}1{deep_close}; }}"
                ))
                .expect_err("300 nested parens must error cleanly");
                assert!(err_p.message.contains("depth"), "{err_p}");

                let deep_blocks_open = "{ ".repeat(300);
                let deep_blocks_close = "} ".repeat(300);
                let err_b = parse_str(&format!(
                    "fn f() {{ {deep_blocks_open}u32 x = 1;{deep_blocks_close} }}"
                ))
                .expect_err("300 nested blocks must error cleanly");
                assert!(err_b.message.contains("depth"), "{err_b}");
            })
            .expect("spawn depth test thread");
        handle
            .join()
            .expect("depth test thread must not overflow stack");

        // Item-boundary error recovery collects errors from multiple broken top-level items while keeping valid ones.
        let src_multi = "\
fn broken_one() {
    let x = 1;
}

fn valid_mid(u32 a) -> u32 {
    return a + 1u32;
}

task broken_two() {
    every 5ms { checkpoint(); }
}

u32 broken_three(u32 v) {
    return v;
}
";
        let (prog, errs) = parse_str_recovering(src_multi);
        assert_eq!(
            errs.len(),
            3,
            "expected 3 recovered parse errors, got {errs:#?}"
        );
        assert_eq!(errs[0].code, Some("P0001"));
        assert_eq!(errs[1].code, Some("P0011"));
        assert_eq!(errs[2].code, Some("P0018"));
        assert_eq!(
            prog.items.len(),
            1,
            "expected 1 surviving valid item, got {:?}",
            prog.items
        );
    }

    #[test]
    fn item3_parses_fallible_fn_ptr_types_in_declarators_and_signatures() {
        let src = r#"
enum Err { Bad }
struct Cmd {
    fn(u8) error run_bare;
    fn(u8) -> u32 error Err run_typed;
}
const fn(u8) error BARE_CONST = a;
const fn(u8) -> u16 error Err TYPED_CONST = b;
fn take(fn(fn(u8) error) cb) -> fn() -> fn(u8) error Err {
    [2]fn(u8) error tbl = [a, a];
    fn(u8) error fp = a;
    fn(u8) error Err fp_e = b;
    return c;
}
"#;
        let prog = parse_str(src).expect("fallible fn-ptr syntax must parse");
        match &prog.items[1].kind {
            ItemKind::Struct(st) => {
                assert!(matches!(
                    &st.fields[0].ty.kind,
                    TypeKind::FnPtr { error_spec: Some(es), .. } if matches!(**es, ErrorSpec::Bare(_))
                ));
                assert!(matches!(
                    &st.fields[1].ty.kind,
                    TypeKind::FnPtr { error_spec: Some(es), .. } if matches!(**es, ErrorSpec::Typed(_))
                ));
            }
            other => panic!("expected Struct, got {other:?}"),
        }
        match &prog.items[2].kind {
            ItemKind::Const(c) => {
                assert_eq!(c.name.name, "BARE_CONST");
                assert!(matches!(
                    &c.ty.kind,
                    TypeKind::FnPtr { error_spec: Some(es), .. } if matches!(**es, ErrorSpec::Bare(_))
                ));
            }
            other => panic!("expected Const, got {other:?}"),
        }
        match &prog.items[3].kind {
            ItemKind::Const(c) => {
                assert_eq!(c.name.name, "TYPED_CONST");
                assert!(matches!(
                    &c.ty.kind,
                    TypeKind::FnPtr { error_spec: Some(es), .. } if matches!(**es, ErrorSpec::Typed(_))
                ));
            }
            other => panic!("expected Const, got {other:?}"),
        }
        match &prog.items[4].kind {
            ItemKind::Fn(f) => {
                assert!(matches!(
                    &f.params[0].ty.kind,
                    TypeKind::FnPtr { params, error_spec: None, .. }
                        if matches!(
                            &params[0].kind,
                            TypeKind::FnPtr { error_spec: Some(es), .. } if matches!(**es, ErrorSpec::Bare(_))
                        )
                ));
                assert!(matches!(
                    &f.ret_type.kind,
                    TypeKind::FnPtr { ret, error_spec: None, .. }
                        if matches!(
                            &ret.kind,
                            TypeKind::FnPtr { error_spec: Some(es), .. } if matches!(**es, ErrorSpec::Typed(_))
                        )
                ));
            }
            other => panic!("expected Fn, got {other:?}"),
        }
    }

    #[test]
    fn parses_parenthesized_fn_ptr_types_to_disambiguate_error_binding() {
        let src = r#"
enum Err { Bad }
enum Err2 { Worse }
fn pick_outer(bool ok) -> (fn(u8) -> u8) error Err {
    (fn(u8) -> u8) cb = inc;
    return cb;
}
fn pick_both(bool ok) -> (fn(u8) -> u8 error Err) error Err2 {
    (fn(u8) -> u8 error Err) cb = inc_err;
    return cb;
}
fn pick_bare() -> (fn(u8) -> u8) error {
    return inc;
}
"#;
        let prog = parse_str(src).expect("parenthesized fn-ptr types must parse cleanly");
        match &prog.items[2].kind {
            ItemKind::Fn(f) => {
                assert!(matches!(
                    &f.ret_type.kind,
                    TypeKind::FnPtr {
                        error_spec: None,
                        ..
                    }
                ));
                assert!(matches!(
                    &f.error_spec,
                    Some(ErrorSpec::Typed(ty)) if matches!(&ty.kind, TypeKind::Named(id) if id.name == "Err")
                ));
                assert!(matches!(
                    &f.body.as_ref().unwrap().stmts[0].kind,
                    StmtKind::VarDecl { ty, name, .. }
                        if name.name == "cb"
                            && matches!(&ty.kind, TypeKind::FnPtr { error_spec: None, .. })
                ));
            }
            other => panic!("expected Fn pick_outer, got {other:?}"),
        }
        match &prog.items[3].kind {
            ItemKind::Fn(f) => {
                assert!(matches!(
                    &f.ret_type.kind,
                    TypeKind::FnPtr { error_spec: Some(es), .. }
                        if matches!(&**es, ErrorSpec::Typed(ty) if matches!(&ty.kind, TypeKind::Named(id) if id.name == "Err"))
                ));
                assert!(matches!(
                    &f.error_spec,
                    Some(ErrorSpec::Typed(ty)) if matches!(&ty.kind, TypeKind::Named(id) if id.name == "Err2")
                ));
            }
            other => panic!("expected Fn pick_both, got {other:?}"),
        }
        match &prog.items[4].kind {
            ItemKind::Fn(f) => {
                assert!(matches!(
                    &f.ret_type.kind,
                    TypeKind::FnPtr {
                        error_spec: None,
                        ..
                    }
                ));
                assert!(matches!(&f.error_spec, Some(ErrorSpec::Bare(_))));
            }
            other => panic!("expected Fn pick_bare, got {other:?}"),
        }
    }
}
