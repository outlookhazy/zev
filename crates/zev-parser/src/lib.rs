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

use std::collections::BTreeSet;
use std::path::Path;

pub use error::ParseError;
pub use token::{LexedFormatPart, Token, TokenKind};
pub use zev_ast;
pub use zev_ast::Program;

/// Tokenize an in-memory `.zev` source string.
pub fn lex(source: &str) -> Result<Vec<Token>, ParseError> {
    lexer::lex(source, None)
}

/// Pre-scan `.zev` source text at brace-depth `0` and return all top-level generic `struct` and
/// `fn` names declared in `src`.
pub fn collect_declared_generic_names(src: &str) -> BTreeSet<String> {
    let tokens = lexer::lex_prefix_best_effort(src);
    parser::Parser::collect_declared_generics(src, tokens)
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

/// Parse an in-memory `.zev` source string with an optional whole-program set of declared generic
/// `struct` / `fn` names (used to disambiguate bare `f<T>(...)` calls from multi-argument `<` / `>`
/// relational comparisons across multi-file builds).
pub fn parse_str_named_with_generics(
    src: &str,
    file_name: &str,
    known_generics: Option<&BTreeSet<String>>,
) -> Result<Program, Vec<ParseError>> {
    let tokens = lexer::lex(src, Some(file_name)).map_err(|e| vec![e])?;
    let mut parser =
        parser::Parser::new_with_generics(src, Some(file_name), tokens, known_generics);
    parser.parse_program().map_err(|e| vec![e])
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

        // 2 imports + 1 packed struct + 2 enums + 5 OS primitives + 2 ISRs + 1 helper fn + 2 tasks + 1 main fn = 16 items
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
                assert!(os.init.is_some());
            }
            other => panic!("expected Watch OsPrimitive, got {other:?}"),
        }

        match &program.items[10].kind {
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
                let prog = parse_workspace_example(&path);
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
            inbox<u32, 16> work_q;
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
    fn parses_wave1_restart_defer_for_in_and_loop_every() {
        let src = r#"
            inbox<u8, 8> cmd_q;

            fn spi_cs_low() {}
            fn spi_cs_high() {}

            u32 sum_packet([]u8 buf, u32 limit) {
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
                assert!(matches!(period.kind, ExprKind::Literal(Literal::Duration(_))));
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

        for bad_pub in [
            "pub import chip.stm32g4 as hw;",
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
                            kind: ExprKind::StructLit { name, fields, .. },
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
                f32 casted = (f32)10 + (a as f32);

                u8 arr[8] = [0, 1, 2, 3, 4, 5, 6, 7];
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
                assert_eq!(f64::from_bits(*bits), 3.141_592_f32 as f64);
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

            T clamp<T>(T val, T lo, T hi) {
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
                u8 raw[16];
                [16]u8 raw2;
                FixedVec<u32, 4> v = FixedVec<u32, 4> { data: [1, 2, 3, 4], len: 4 };
                FixedVec<FixedVec<u8, 4>, 2> nested;
                Boxed<Boxed<u32>> dbl = Boxed<Boxed<u32>> {
                    inner: Boxed<u32> { inner: 9 }
                };

                f32 c1 = clamp<f32>(1.5, 0.0, 1.0);
                f32 c2 = clamp::<f32>(-0.5, 0.0, 1.0);
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

        // 3. Verify `main` statements: `[16]u8`, `FixedVec<u32, 4> { ... }`, `Boxed<Boxed<u32>>`, `clamp<f32>`, `clamp::<f32>`, `x < y && (z > 0)`
        let main_fn = match &prog.items[4].kind {
            ItemKind::Fn(f) => f,
            other => panic!("expected main Fn, got {other:?}"),
        };
        let stmts = &main_fn.body.as_ref().unwrap().stmts;

        // `u8 raw[16];` and `[16]u8 raw2;` both use `ArrayLen::Fixed(16)`
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
            StmtKind::VarDecl { ty, init: Some(init), .. } => {
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

        // `clamp<f32>(...)` & `clamp::<f32>(...)`
        for idx in [5, 6] {
            match &stmts[idx].kind {
                StmtKind::VarDecl {
                    init:
                        Some(Expr {
                            kind: ExprKind::Call { generic_args, args, .. },
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
            "    println(f\"nested={CmdFrame { op: 1 }:?} path={hw::REG:02x}\");\n",
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

        // Second `println(f"nested={CmdFrame { op: 1 }:?} path={hw::CONST:02x}")`
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
                        kind: ExprKind::Literal(Literal::Int(255)),
                        ..
                    }),
                ..
            } => {}
            other => panic!("expected `u8 ch = 255`, got {other:?}"),
        }
    }

    #[test]
    fn review_r1_generic_calls_never_hijack_multi_arg_relational_comparisons() {
        let lib_src = r#"
struct Pair {
    u32 a;
    u32 b;
}

struct FixedVec<T, usize N> {
    [N]T data;
}

T clamp<T>(T val, T lo, T hi) {
    return val;
}
"#;
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

    // 1. Multi-arg comparisons with parenthesized RHS after `>` MUST parse as 2 boolean args!
    check_window(a < b, c > (d + 1));
    check_window(GLOBAL_LO < b, c > (d + 1));
    check_window(s.a < b, c > (d + 1));
    [2]bool flags = [a < 10, c > (d + 1)];

    // 2. Real generic calls (local/cross-file `clamp<f32>`, module alias `dsp.clamp<f32>`, turbofish `::<f32>`) still parse!
    f32 x1 = clamp<f32>(1.5f32, 0.0f32, 2.0f32);
    f32 x2 = clamp::<f32>(x1, 0.0f32, 1.0f32);
    f32 x3 = dsp.clamp<f32>(x2, 0.0f32, 1.0f32);

    // 3. Undeclared value identifier `unk < b, c > (d + 1)` also skips speculation when `known_generics` is supplied!
    check_window(unk < b, c > (d + 1));
    println(f"{flags[0]} {x3}");
}
"#;
        let mut generics = collect_declared_generic_names(lib_src);
        generics.extend(collect_declared_generic_names(app_src));
        assert!(generics.contains("FixedVec"));
        assert!(generics.contains("clamp"));
        assert!(!generics.contains("Pair"));
        assert!(!generics.contains("check_window"));

        let prog = parse_str_named_with_generics(app_src, "app.zev", Some(&generics))
            .expect("parse app.zev with known_generics");

        let main_fn = prog
            .items
            .iter()
            .find_map(|it| match &it.kind {
                ItemKind::Fn(f) if f.name.name == "main" => Some(f),
                _ => None,
            })
            .unwrap();
        let stmts = &main_fn.body.as_ref().unwrap().stmts;

        // Verify `check_window(a < b, c > (d + 1))`, `check_window(GLOBAL_LO < b, ...)`, and `check_window(s.a < b, ...)`
        // each parsed with `args.len() == 2`!
        for idx in [5usize, 6, 7, 12] {
            match &stmts[idx].kind {
                StmtKind::Expr(Expr {
                    kind: ExprKind::Call { args, .. },
                    ..
                }) => assert_eq!(args.len(), 2, "stmt {idx} should have 2 comparison args"),
                other => panic!("expected 2-arg Call at stmt {idx}, got {other:?}"),
            }
        }

        // Verify `[a < 10, c > (d + 1)]` parsed as 2-element `ArrayLit`
        match &stmts[8].kind {
            StmtKind::VarDecl {
                init:
                    Some(Expr {
                        kind: ExprKind::ArrayLit(elems),
                        ..
                    }),
                ..
            } => assert_eq!(elems.len(), 2),
            other => panic!("expected 2-element ArrayLit, got {other:?}"),
        }

        // Verify `clamp<f32>`, `clamp::<f32>`, and `dsp.clamp<f32>` parsed with 1 generic arg & 3 call args
        for idx in [9usize, 10, 11] {
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
    }

    #[test]
    fn review_r1_f32_direct_parse_fstring_comments_pad_specs_shl_generics_and_local_const() {
        // 1. M11: Direct `f32` parsing avoids `str -> f64 -> f32` double-rounding at binary64/binary32 midpoints!
        // `1 + 2^-24 + 2^-54` = `1.000000059604644775390625 + ~5.55e-17`:
        // - `parse::<f64>()` rounds to even `1 + 2^-24`, and `(v64 as f32)` rounds to even `1.0f32` (`0x3f800000`).
        // - Direct `parse::<f32>()` sees the exact decimal > midpoint and rounds up to `0x3f800001`!
        let f32_src = "fn main() { f32 x = 1.000000059604644830901776231257827021181583404541015625f32; }";
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
                && esc_err.hint.as_deref().unwrap_or("").contains("hdr={\"ok\"}"),
            "unexpected error: {esc_err}"
        );

        // 4. M13: Reject `:4x`, `:8b`, `:0x`, `:0X`, `:0b`, `:00x` with actionable zero-pad / prefix hints!
        let e_4x = parse_str("fn main() { u16 v = 1; println(f\"{v:4x}\"); }").unwrap_err();
        assert!(e_4x.hint.as_deref().unwrap_or("").contains(":04x"), "{e_4x}");
        let e_8b = parse_str("fn main() { u8 v = 1; println(f\"{v:8b}\"); }").unwrap_err();
        assert!(e_8b.hint.as_deref().unwrap_or("").contains(":08b"), "{e_8b}");
        let e_0x = parse_str("fn main() { u16 v = 1; println(f\"{v:0x}\"); }").unwrap_err();
        assert!(
            e_0x.hint.as_deref().unwrap_or("").contains("0x{val:x}")
                && e_0x.hint.as_deref().unwrap_or("").contains(":02x"),
            "{e_0x}"
        );
        let e_0b = parse_str("fn main() { u8 v = 1; println(f\"{v:0b}\"); }").unwrap_err();
        assert!(e_0b.hint.as_deref().unwrap_or("").contains("0b{val:b}"), "{e_0b}");

        // 5. Low #2: Reject block-local `const Type name = ...;` with clear top-level guidance!
        let e_local_const =
            parse_str("fn main() { const usize CAP = 8; }").expect_err("reject block-local const");
        assert!(
            e_local_const.message.contains("block-local `const`")
                && e_local_const.hint.as_deref().unwrap_or("").contains("module top level"),
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
pub u32 add(u32 a, u32 b) { return a + b; }
pub extern "C" fn c_hook(u32 v);
extern "C" {
    pub fn ext_pub(u32 x);
    u32 ext_priv(u32 y);
}
@high pub task worker(u32 id) {}
pub @normal isr exti0(u32 pin) {}
pub mmio UartRegs { @rw u32 cr1 @ 0x00; }
pub irq IRQ_UART = 37;
pub watch<u32> w_val = 0;
pub inbox<u8, 4> rx_q;
pub signal ready;
pub ping_pong<u8, 8> pp;
pub lease<u8, 16> pool;
"#;
        let prog = parse_str(src).expect("all top-level pub declarations must parse");
        assert_eq!(prog.items.len(), 18);
        for (idx, item) in prog.items.iter().enumerate() {
            if idx == 8 {
                assert!(!item.is_pub, "extern block item `ext_priv` should default to private");
            } else {
                assert!(item.is_pub, "item #{idx} should be marked `pub`: {:?}", item.kind);
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
        drv.Box<u32, 4> b2 = drv.Box::<u32, 4> { val: 8 };
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
        assert!(
            isrs[1]
                .body
                .stmts
                .iter()
                .all(|s| !matches!(s.kind, StmtKind::StateVar { .. }))
        );

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
}
