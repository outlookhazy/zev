//! Freestanding C99 Emitter + Synthesized Cooperative Kernel (`zev-codegen-c`).
//!
//! Lowers `zev_mir::MirModule` into clean, human-readable ISO C99 code with:
//! - Ranked cooperative scheduler (`Overdue -> Work (MORE) -> Poll (IDLE) -> Sleep`)
//! - Starvation bounds (`1 / 5 / 15 / 50 ms`) and `should_share()` demand-driven chunking
//! - ISR priority elevation (`boost: @critical`) and automatic end-of-turn restoration
//! - Zero-copy `@dma_safe ping_pong<T, N>` & `lease<T, N>` clamping sleep depth `DEEP -> LIGHT`
//! - Wire-endian (`u16be`, `u32le`, ...) & `packed struct` `from_bytes` + dual-exit `{ is_err, err, val }`
//! - Host deterministic simulation harness (`syn_sim_*`) for native Clang + ASan/UBSan verification.

pub mod emit;
pub mod runtime;

pub use emit::{emit_c, emit_c_with_options, CodegenOptions};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;
    use zev_ast::*;
    use zev_mir::lower_program;

    fn id(name: &str) -> Ident {
        Ident::dummy(name)
    }

    fn ty_prim(p: PrimType) -> TypeExpr {
        TypeExpr::new(TypeKind::Prim(p), Span::dummy())
    }

    fn ty_named(name: &str) -> TypeExpr {
        TypeExpr::new(TypeKind::Named(id(name)), Span::dummy())
    }

    fn expr_int(v: u64) -> Expr {
        Expr::new(ExprKind::Literal(Literal::Int(v)), Span::dummy())
    }

    fn expr_id(name: &str) -> Expr {
        Expr::new(ExprKind::Ident(id(name)), Span::dummy())
    }

    fn dur_ms(ms: u64) -> DurationLit {
        DurationLit {
            value: ms,
            unit: DurationUnit::Ms,
            span: Span::dummy(),
        }
    }

    /// Compile `c_src` + `harness_c` with `/bin/clang -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined`,
    /// verify object file has zero `malloc`/`calloc`/`realloc` symbols, and run the binary!
    fn compile_and_run_clang_sanitized(test_name: &str, c_src: &str, extra_driver_c: Option<&str>) -> String {
        let dir = format!("/tmp/zev-c-tests-{}/{test_name}", std::process::id());
        fs::create_dir_all(&dir).expect("mkdir test dir");

        let full_c = match extra_driver_c {
            Some(drv) => format!("#define SYN_NO_MAIN 1\n{c_src}\n{drv}\n"),
            None => c_src.to_string(),
        };

        let c_path = format!("{dir}/gen.c");
        let obj_path = format!("{dir}/gen.o");
        let bin_path = format!("{dir}/gen_bin");
        fs::write(&c_path, &full_c).expect("write gen.c");

        // 1. Compile object file (without ASan instrumentation) to verify ZERO heap allocator symbols!
        let obj_out = Command::new("/bin/clang")
            .args([
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-O2",
                "-c",
                &c_path,
                "-o",
                &obj_path,
            ])
            .output()
            .expect("invoke clang -c");
        if !obj_out.status.success() {
            panic!(
                "clang -c failed:\n{}\n--- C SOURCE ---\n{}",
                String::from_utf8_lossy(&obj_out.stderr),
                full_c
            );
        }

        let nm_out = Command::new("nm")
            .arg(&obj_path)
            .output()
            .expect("invoke nm");
        let nm_txt = String::from_utf8_lossy(&nm_out.stdout);
        for forbidden in ["malloc", "calloc", "realloc"] {
            assert!(
                !nm_txt.contains(forbidden),
                "Generated object must have ZERO `{forbidden}` symbols, found in nm:\n{nm_txt}"
            );
        }

        // 2. Compile & link with AddressSanitizer + UndefinedBehaviorSanitizer and execute!
        let bin_out = Command::new("/bin/clang")
            .args([
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-g",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                &c_path,
                "-o",
                &bin_path,
            ])
            .output()
            .expect("invoke clang sanitizer build");
        if !bin_out.status.success() {
            panic!(
                "clang ASan+UBSan failed:\n{}\n--- C SOURCE ---\n{}",
                String::from_utf8_lossy(&bin_out.stderr),
                full_c
            );
        }

        let run_out = Command::new(&bin_path)
            .env("ASAN_OPTIONS", "detect_stack_use_after_return=1")
            .env("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
            .output()
            .expect("execute compiled simulation binary");
        if !run_out.status.success() {
            panic!(
                "simulation binary failed (status {:?}):\nstdout:\n{}\nstderr:\n{}",
                run_out.status,
                String::from_utf8_lossy(&run_out.stdout),
                String::from_utf8_lossy(&run_out.stderr)
            );
        }
        let stdout = String::from_utf8_lossy(&run_out.stdout).to_string();
        let _ = fs::remove_dir_all(&dir);
        stdout
    }

    #[test]
    fn end_to_end_motor_controller_sim_boost_dma_endian_and_errors() {
        let sp = Span::dummy();
        let mut items = Vec::new();

        // Typed MMIO blocks for `GPIOA` and `TIM1`
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Mmio(MmioDecl {
                name: id("GPIOA"),
                kind: MmioDeclKind::Inline {
                    base_addr: expr_int(0x4800_0000),
                    regs: vec![MmioRegDecl {
                        access: MmioAccess::Wo,
                        ty: ty_prim(PrimType::U32),
                        name: id("brr"),
                        offset: expr_int(0x28),
                        span: sp,
                    }],
                },
                span: sp,
            }),
            span: sp,
        });
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Mmio(MmioDecl {
                name: id("TIM1"),
                kind: MmioDeclKind::Inline {
                    base_addr: expr_int(0x4001_2C00),
                    regs: vec![MmioRegDecl {
                        access: MmioAccess::Rw,
                        ty: TypeExpr::new(
                            TypeKind::Array {
                                elem: Box::new(ty_prim(PrimType::U32)),
                                len: ArrayLen::Fixed(4),
                            },
                            sp,
                        ),
                        name: id("ccr"),
                        offset: expr_int(0x34),
                        span: sp,
                    }],
                },
                span: sp,
            }),
            span: sp,
        });

        // packed struct CmdFrame { u8 op; u16be setpoint; u16be crc; }
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Struct(StructDecl {
                is_packed: true,
                name: id("CmdFrame"),
                generic_params: vec![],
                fields: vec![
                    StructField {
                        attrs: vec![],
                        ty: ty_prim(PrimType::U8),
                        name: id("op"),
                        span: sp,
                    },
                    StructField {
                        attrs: vec![],
                        ty: ty_prim(PrimType::U16Be),
                        name: id("setpoint"),
                        span: sp,
                    },
                    StructField {
                        attrs: vec![],
                        ty: ty_prim(PrimType::U16Be),
                        name: id("crc"),
                        span: sp,
                    },
                ],
                span: sp,
            }),
            span: sp,
        });

        // enum ParseErr { TooShort, BadCrc }
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Enum(EnumDecl {
                name: id("ParseErr"),
                variants: vec![
                    EnumVariant {
                        name: id("TooShort"),
                        discriminant: None,
                        span: sp,
                    },
                    EnumVariant {
                        name: id("BadCrc"),
                        discriminant: None,
                        span: sp,
                    },
                ],
                span: sp,
            }),
            span: sp,
        });

        // enum SupEvent { Overcurrent, BadPacket, IdleCheck }
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Enum(EnumDecl {
                name: id("SupEvent"),
                variants: vec![
                    EnumVariant {
                        name: id("Overcurrent"),
                        discriminant: None,
                        span: sp,
                    },
                    EnumVariant {
                        name: id("BadPacket"),
                        discriminant: None,
                        span: sp,
                    },
                    EnumVariant {
                        name: id("IdleCheck"),
                        discriminant: None,
                        span: sp,
                    },
                ],
                span: sp,
            }),
            span: sp,
        });

        // @dma_safe ping_pong<u8, 64> uart1_dma;
        items.push(Item {
            attrs: vec![Attr {
                kind: AttrKind::DmaSafe,
                span: sp,
            }],
            is_pub: false,
            kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                attrs: vec![],
                ty: OsPrimitiveType::PingPong {
                    elem: Box::new(ty_prim(PrimType::U8)),
                    cap: 64,
                },
                name: id("uart1_dma"),
                init: None,
                span: sp,
            }),
            span: sp,
        });

        // watch<u16> motor1_rpm = 0; watch<u32> fault_saved = 0; watch<u32> scrub_runs = 0;
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                attrs: vec![],
                ty: OsPrimitiveType::Watch {
                    elem: Box::new(ty_prim(PrimType::U16)),
                },
                name: id("motor1_rpm"),
                init: Some(expr_int(0)),
                span: sp,
            }),
            span: sp,
        });
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                attrs: vec![],
                ty: OsPrimitiveType::Watch {
                    elem: Box::new(ty_prim(PrimType::U32)),
                },
                name: id("fault_saved_prio"),
                init: Some(expr_int(99)),
                span: sp,
            }),
            span: sp,
        });
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                attrs: vec![],
                ty: OsPrimitiveType::Watch {
                    elem: Box::new(ty_prim(PrimType::U32)),
                },
                name: id("scrub_runs"),
                init: Some(expr_int(0)),
                span: sp,
            }),
            span: sp,
        });

        // inbox<SupEvent, 8> sup_inbox;
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                attrs: vec![],
                ty: OsPrimitiveType::Inbox {
                    elem: Box::new(ty_named("SupEvent")),
                    cap: 8,
                },
                name: id("sup_inbox"),
                init: None,
                span: sp,
            }),
            span: sp,
        });

        // isr EXTI0 { GPIOA.brr = 1; sup_inbox.post(Overcurrent, boost: @critical); }
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Isr(IsrDecl {
                attrs: vec![],
                vector: id("EXTI0"),
                params: vec![],
                body: Block::new(
                    vec![
                        Stmt::new(
                            StmtKind::Assign {
                                target: Expr::new(
                                    ExprKind::Field {
                                        target: Box::new(expr_id("GPIOA")),
                                        field: id("brr"),
                                    },
                                    sp,
                                ),
                                op: AssignOp::Assign,
                                value: expr_int(1),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Expr(Expr::new(
                                ExprKind::MethodCall {
                                    receiver: Box::new(expr_id("sup_inbox")),
                                    method: id("post"),
                                    args: vec![
                                        CallArg::positional(expr_id("Overcurrent")),
                                        CallArg::named(
                                            id("boost"),
                                            Expr::new(ExprKind::Priority(Priority::Critical), sp),
                                            sp,
                                        ),
                                    ],
                                },
                                sp,
                            )),
                            sp,
                        ),
                    ],
                    sp,
                ),
                span: sp,
            }),
            span: sp,
        });

        // isr DMA1_CH5 { uart1_dma.flip(); }
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Isr(IsrDecl {
                attrs: vec![],
                vector: id("DMA1_CH5"),
                params: vec![],
                body: Block::new(
                    vec![Stmt::new(
                        StmtKind::Expr(Expr::new(
                            ExprKind::MethodCall {
                                receiver: Box::new(expr_id("uart1_dma")),
                                method: id("flip"),
                                args: vec![],
                            },
                            sp,
                        )),
                        sp,
                    )],
                    sp,
                ),
                span: sp,
            }),
            span: sp,
        });

        // fn crc16([]u8 buf) -> u16: simple sum ((u16)buf[0] + (u16)buf[1] + (u16)buf[2])
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("crc16"),
                generic_params: vec![],
                params: vec![Param {
                    ty: TypeExpr::new(TypeKind::Slice(Box::new(ty_prim(PrimType::U8))), sp),
                    name: id("buf"),
                    span: sp,
                }],
                is_variadic: false,
                ret_type: ty_prim(PrimType::U16),
                error_spec: None,
                body: Some(Block::new(
                    vec![Stmt::new(
                        StmtKind::Return(Some(Expr::new(
                            ExprKind::Binary {
                                op: BinaryOp::Add,
                                lhs: Box::new(Expr::new(
                                    ExprKind::Cast {
                                        expr: Box::new(Expr::new(
                                            ExprKind::Index {
                                                target: Box::new(expr_id("buf")),
                                                index: Box::new(expr_int(0)),
                                            },
                                            sp,
                                        )),
                                        ty: ty_prim(PrimType::U16),
                                    },
                                    sp,
                                )),
                                rhs: Box::new(Expr::new(
                                    ExprKind::Binary {
                                        op: BinaryOp::Add,
                                        lhs: Box::new(Expr::new(
                                            ExprKind::Cast {
                                                expr: Box::new(Expr::new(
                                                    ExprKind::Index {
                                                        target: Box::new(expr_id("buf")),
                                                        index: Box::new(expr_int(1)),
                                                    },
                                                    sp,
                                                )),
                                                ty: ty_prim(PrimType::U16),
                                            },
                                            sp,
                                        )),
                                        rhs: Box::new(Expr::new(
                                            ExprKind::Cast {
                                                expr: Box::new(Expr::new(
                                                    ExprKind::Index {
                                                        target: Box::new(expr_id("buf")),
                                                        index: Box::new(expr_int(2)),
                                                    },
                                                    sp,
                                                )),
                                                ty: ty_prim(PrimType::U16),
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                )),
                            },
                            sp,
                        ))),
                        sp,
                    )],
                    sp,
                )),
                span: sp,
            }),
            span: sp,
        });

        // u16 parse_setpoint([]u8 buf) error ParseErr
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("parse_setpoint"),
                generic_params: vec![],
                params: vec![Param {
                    ty: TypeExpr::new(TypeKind::Slice(Box::new(ty_prim(PrimType::U8))), sp),
                    name: id("buf"),
                    span: sp,
                }],
                is_variadic: false,
                ret_type: ty_prim(PrimType::U16),
                error_spec: Some(ErrorSpec::Typed(ty_named("ParseErr"))),
                body: Some(Block::new(
                    vec![
                        Stmt::new(
                            StmtKind::If {
                                cond: Expr::new(
                                    ExprKind::Binary {
                                        op: BinaryOp::Lt,
                                        lhs: Box::new(Expr::new(
                                            ExprKind::Field {
                                                target: Box::new(expr_id("buf")),
                                                field: id("len"),
                                            },
                                            sp,
                                        )),
                                        rhs: Box::new(expr_int(5)),
                                    },
                                    sp,
                                ),
                                then_block: Block::new(
                                    vec![Stmt::new(
                                        StmtKind::Error {
                                            reason: Some(expr_id("TooShort")),
                                            partial: None,
                                        },
                                        sp,
                                    )],
                                    sp,
                                ),
                                else_block: None,
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::VarDecl {
                                ty: ty_named("CmdFrame"),
                                name: id("frame"),
                                init: Some(Expr::new(
                                    ExprKind::Catch {
                                        expr: Box::new(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("CmdFrame")),
                                                method: id("from_bytes"),
                                                args: vec![CallArg::positional(Expr::new(
                                                    ExprKind::Slice {
                                                        target: Box::new(expr_id("buf")),
                                                        start: Some(Box::new(expr_int(0))),
                                                        end: Some(Box::new(expr_int(5))),
                                                    },
                                                    sp,
                                                ))],
                                            },
                                            sp,
                                        )),
                                        clause: CatchClause {
                                            err_param: None,
                                            partial_param: None,
                                            body: Block::new(
                                                vec![Stmt::new(
                                                    StmtKind::Error {
                                                        reason: Some(expr_id("TooShort")),
                                                        partial: None,
                                                    },
                                                    sp,
                                                )],
                                                sp,
                                            ),
                                            span: sp,
                                        },
                                    },
                                    sp,
                                )),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::If {
                                cond: Expr::new(
                                    ExprKind::Binary {
                                        op: BinaryOp::Ne,
                                        lhs: Box::new(Expr::new(
                                            ExprKind::Call {
                                                callee: Box::new(expr_id("crc16")),
                                                generic_args: vec![],
                                                args: vec![CallArg::positional(Expr::new(
                                                    ExprKind::Slice {
                                                        target: Box::new(expr_id("buf")),
                                                        start: Some(Box::new(expr_int(0))),
                                                        end: Some(Box::new(expr_int(3))),
                                                    },
                                                    sp,
                                                ))],
                                            },
                                            sp,
                                        )),
                                        rhs: Box::new(Expr::new(
                                            ExprKind::Field {
                                                target: Box::new(expr_id("frame")),
                                                field: id("crc"),
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                                then_block: Block::new(
                                    vec![Stmt::new(
                                        StmtKind::Error {
                                            reason: Some(expr_id("BadCrc")),
                                            partial: None,
                                        },
                                        sp,
                                    )],
                                    sp,
                                ),
                                else_block: None,
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Return(Some(Expr::new(
                                ExprKind::Field {
                                    target: Box::new(expr_id("frame")),
                                    field: id("setpoint"),
                                },
                                sp,
                            ))),
                            sp,
                        ),
                    ],
                    sp,
                )),
                span: sp,
            }),
            span: sp,
        });

        // Yielding helper: save_fault_record(u32 bad_packets) — yields 1ms and records active priority!
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: true,
                name: id("c_query_running_prio"),
                generic_params: vec![],
                params: vec![],
                is_variadic: false,
                ret_type: ty_prim(PrimType::U32),
                error_spec: None,
                body: None,
                span: sp,
            }),
            span: sp,
        });
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("save_fault_record"),
                generic_params: vec![],
                params: vec![Param {
                    ty: ty_prim(PrimType::U32),
                    name: id("bad_cnt"),
                    span: sp,
                }],
                is_variadic: false,
                ret_type: ty_prim(PrimType::Void),
                error_spec: None,
                body: Some(Block::new(
                    vec![
                        Stmt::new(
                            StmtKind::VarDecl {
                                ty: ty_prim(PrimType::U32),
                                name: id("saved"),
                                init: Some(expr_id("bad_cnt")),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Expr(Expr::new(
                                ExprKind::Call {
                                    callee: Box::new(expr_id("delay")),
                                    generic_args: vec![],
                                    args: vec![CallArg::positional(Expr::new(
                                        ExprKind::Literal(Literal::Duration(dur_ms(1))),
                                        sp,
                                    ))],
                                },
                                sp,
                            )),
                            sp,
                        ),
                        // Even AFTER yielding 1ms inside `save_fault_record`, priority MUST still be @critical (0)!
                        Stmt::new(
                            StmtKind::Expr(Expr::new(
                                ExprKind::MethodCall {
                                    receiver: Box::new(expr_id("fault_saved_prio")),
                                    method: id("publish"),
                                    args: vec![CallArg::positional(Expr::new(
                                        ExprKind::Binary {
                                            op: BinaryOp::Add,
                                            lhs: Box::new(Expr::new(
                                                ExprKind::Call {
                                                    callee: Box::new(expr_id(
                                                        "c_query_running_prio",
                                                    )),
                                                    generic_args: vec![],
                                                    args: vec![],
                                                },
                                                sp,
                                            )),
                                            rhs: Box::new(Expr::new(
                                                ExprKind::Binary {
                                                    op: BinaryOp::Mul,
                                                    lhs: Box::new(expr_id("saved")),
                                                    rhs: Box::new(expr_int(100)),
                                                },
                                                sp,
                                            )),
                                        },
                                        sp,
                                    ))],
                                },
                                sp,
                            )),
                            sp,
                        ),
                    ],
                    sp,
                )),
                span: sp,
            }),
            span: sp,
        });

        // Disjoint yielding sibling: scrub_flash_crc()
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("scrub_flash_crc"),
                generic_params: vec![],
                params: vec![],
                is_variadic: false,
                ret_type: ty_prim(PrimType::Void),
                error_spec: None,
                body: Some(Block::new(
                    vec![
                        Stmt::new(
                            StmtKind::VarDecl {
                                ty: ty_prim(PrimType::U32),
                                name: id("sector"),
                                init: Some(expr_int(0)),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Loop {
                                body: Block::new(
                                    vec![
                                        Stmt::new(
                                            StmtKind::Assign {
                                                target: expr_id("sector"),
                                                op: AssignOp::AddAssign,
                                                value: expr_int(1),
                                            },
                                            sp,
                                        ),
                                        Stmt::new(
                                            StmtKind::If {
                                                cond: Expr::new(
                                                    ExprKind::Binary {
                                                        op: BinaryOp::Ge,
                                                        lhs: Box::new(expr_id("sector")),
                                                        rhs: Box::new(expr_int(4)),
                                                    },
                                                    sp,
                                                ),
                                                then_block: Block::new(
                                                    vec![Stmt::new(StmtKind::Break, sp)],
                                                    sp,
                                                ),
                                                else_block: None,
                                            },
                                            sp,
                                        ),
                                    ],
                                    sp,
                                ),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Expr(Expr::new(
                                ExprKind::MethodCall {
                                    receiver: Box::new(expr_id("scrub_runs")),
                                    method: id("publish"),
                                    args: vec![CallArg::positional(expr_id("sector"))],
                                },
                                sp,
                            )),
                            sp,
                        ),
                    ],
                    sp,
                )),
                span: sp,
            }),
            span: sp,
        });

        // task motor_control(watch<u16> *target_rpm, u8 pwm_ch)
        let watch_u16_ptr = TypeExpr::new(
            TypeKind::Ptr(Box::new(TypeExpr::new(
                TypeKind::OsPrimitive(OsPrimitiveType::Watch {
                    elem: Box::new(ty_prim(PrimType::U16)),
                }),
                sp,
            ))),
            sp,
        );
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Task(TaskDecl {
                attrs: vec![],
                name: id("motor_control"),
                params: vec![
                    Param {
                        ty: watch_u16_ptr.clone(),
                        name: id("target_rpm"),
                        span: sp,
                    },
                    Param {
                        ty: ty_prim(PrimType::U8),
                        name: id("pwm_ch"),
                        span: sp,
                    },
                ],
                body: Block::new(
                    vec![
                        Stmt::new(
                            StmtKind::VarDecl {
                                ty: ty_prim(PrimType::I32),
                                name: id("integral"),
                                init: Some(expr_int(0)),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::LoopEvery {
                                period: Box::new(Expr::new(
                                    ExprKind::Literal(Literal::Duration(dur_ms(1))),
                                    sp,
                                )),
                                body: Block::new(
                                    vec![
                                        Stmt::new(
                                            StmtKind::VarDecl {
                                                ty: ty_prim(PrimType::U16),
                                                name: id("target"),
                                                init: Some(Expr::new(
                                                    ExprKind::MethodCall {
                                                        receiver: Box::new(expr_id("target_rpm")),
                                                        method: id("read"),
                                                        args: vec![],
                                                    },
                                                    sp,
                                                )),
                                            },
                                            sp,
                                        ),
                                        Stmt::new(
                                            StmtKind::NoYield {
                                                body: Block::new(
                                                    vec![
                                                        Stmt::new(
                                                            StmtKind::Assign {
                                                                target: expr_id("integral"),
                                                                op: AssignOp::AddAssign,
                                                                value: Expr::new(
                                                                    ExprKind::Cast {
                                                                        expr: Box::new(expr_id(
                                                                            "target",
                                                                        )),
                                                                        ty: ty_prim(PrimType::I32),
                                                                    },
                                                                    sp,
                                                                ),
                                                            },
                                                            sp,
                                                        ),
                                                        Stmt::new(
                                                            StmtKind::Assign {
                                                                target: Expr::new(
                                                                    ExprKind::Index {
                                                                        target: Box::new(Expr::new(
                                                                            ExprKind::Field {
                                                                                target: Box::new(
                                                                                    expr_id("TIM1"),
                                                                                ),
                                                                                field: id("ccr"),
                                                                            },
                                                                            sp,
                                                                        )),
                                                                        index: Box::new(Expr::new(
                                                                            ExprKind::Cast {
                                                                                expr: Box::new(
                                                                                    expr_id(
                                                                                        "pwm_ch",
                                                                                    ),
                                                                                ),
                                                                                ty: ty_prim(
                                                                                    PrimType::Usize,
                                                                                ),
                                                                            },
                                                                            sp,
                                                                        )),
                                                                    },
                                                                    sp,
                                                                ),
                                                                op: AssignOp::Assign,
                                                                value: Expr::new(
                                                                    ExprKind::Cast {
                                                                        expr: Box::new(expr_id(
                                                                            "target",
                                                                        )),
                                                                        ty: ty_prim(PrimType::U32),
                                                                    },
                                                                    sp,
                                                                ),
                                                            },
                                                            sp,
                                                        ),
                                                    ],
                                                    sp,
                                                ),
                                            },
                                            sp,
                                        ),
                                    ],
                                    sp,
                                ),
                            },
                            sp,
                        ),
                    ],
                    sp,
                ),
                span: sp,
            }),
            span: sp,
        });

        // task fieldbus_listener(ping_pong<u8, 64> *rx_dma, watch<u16> *out_rpm)
        let pp_ptr = TypeExpr::new(
            TypeKind::Ptr(Box::new(TypeExpr::new(
                TypeKind::OsPrimitive(OsPrimitiveType::PingPong {
                    elem: Box::new(ty_prim(PrimType::U8)),
                    cap: 64,
                }),
                sp,
            ))),
            sp,
        );
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Task(TaskDecl {
                attrs: vec![],
                name: id("fieldbus_listener"),
                params: vec![
                    Param {
                        ty: pp_ptr,
                        name: id("rx_dma"),
                        span: sp,
                    },
                    Param {
                        ty: watch_u16_ptr,
                        name: id("out_rpm"),
                        span: sp,
                    },
                ],
                body: Block::new(
                    vec![Stmt::new(
                        StmtKind::Loop {
                            body: Block::new(
                                vec![
                                    Stmt::new(
                                        StmtKind::VarDecl {
                                            ty: TypeExpr::new(
                                                TypeKind::Slice(Box::new(ty_prim(PrimType::U8))),
                                                sp,
                                            ),
                                            name: id("half"),
                                            init: Some(Expr::new(
                                                ExprKind::MethodCall {
                                                    receiver: Box::new(expr_id("rx_dma")),
                                                    method: id("wait_chunk"),
                                                    args: vec![],
                                                },
                                                sp,
                                            )),
                                        },
                                        sp,
                                    ),
                                    Stmt::new(
                                        StmtKind::VarDecl {
                                            ty: ty_prim(PrimType::U16),
                                            name: id("rpm"),
                                            init: Some(Expr::new(
                                                ExprKind::Catch {
                                                    expr: Box::new(Expr::new(
                                                        ExprKind::Call {
                                                            callee: Box::new(expr_id(
                                                                "parse_setpoint",
                                                            )),
                                                            generic_args: vec![],
                                                            args: vec![CallArg::positional(
                                                                expr_id("half"),
                                                            )],
                                                        },
                                                        sp,
                                                    )),
                                                    clause: CatchClause {
                                                        err_param: Some(CatchParam {
                                                            ty: ty_named("ParseErr"),
                                                            name: id("err"),
                                                            span: sp,
                                                        }),
                                                        partial_param: None,
                                                        body: Block::new(
                                                            vec![
                                                                Stmt::new(
                                                                    StmtKind::Expr(Expr::new(
                                                                        ExprKind::MethodCall {
                                                                            receiver: Box::new(
                                                                                expr_id(
                                                                                    "sup_inbox",
                                                                                ),
                                                                            ),
                                                                            method: id("post"),
                                                                            args: vec![
                                                                                CallArg::positional(
                                                                                    expr_id(
                                                                                        "BadPacket",
                                                                                    ),
                                                                                ),
                                                                            ],
                                                                        },
                                                                        sp,
                                                                    )),
                                                                    sp,
                                                                ),
                                                                Stmt::new(StmtKind::Continue, sp),
                                                            ],
                                                            sp,
                                                        ),
                                                        span: sp,
                                                    },
                                                },
                                                sp,
                                            )),
                                        },
                                        sp,
                                    ),
                                    Stmt::new(
                                        StmtKind::Expr(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("out_rpm")),
                                                method: id("publish"),
                                                args: vec![CallArg::positional(expr_id("rpm"))],
                                            },
                                            sp,
                                        )),
                                        sp,
                                    ),
                                ],
                                sp,
                            ),
                        },
                        sp,
                    )],
                    sp,
                ),
                span: sp,
            }),
            span: sp,
        });

        // fn main()
        let addr_of = |name: &str| {
            CallArg::positional(Expr::new(
                ExprKind::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(expr_id(name)),
                },
                sp,
            ))
        };
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("main"),
                generic_params: vec![],
                params: vec![],
                is_variadic: false,
                ret_type: ty_prim(PrimType::Void),
                error_spec: None,
                body: Some(Block::new(
                    vec![
                        Stmt::new(
                            StmtKind::Expr(Expr::new(
                                ExprKind::MethodCall {
                                    receiver: Box::new(expr_id("sys")),
                                    method: id("use_systick"),
                                    args: vec![CallArg::positional(expr_int(170_000_000))],
                                },
                                sp,
                            )),
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Spawn(SpawnStmt {
                                module: None,
                                task: id("motor_control"),
                                args: vec![addr_of("motor1_rpm"), CallArg::positional(expr_int(1))],
                                priority: Some(Priority::Critical),
                                core: None,
                                span: sp,
                            }),
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Spawn(SpawnStmt {
                                module: None,
                                task: id("fieldbus_listener"),
                                args: vec![addr_of("uart1_dma"), addr_of("motor1_rpm")],
                                priority: Some(Priority::Normal),
                                core: None,
                                span: sp,
                            }),
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Expr(Expr::new(
                                ExprKind::Call {
                                    callee: Box::new(expr_id("set_priority")),
                                    generic_args: vec![],
                                    args: vec![CallArg::positional(Expr::new(
                                        ExprKind::Priority(Priority::Low),
                                        sp,
                                    ))],
                                },
                                sp,
                            )),
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::VarDecl {
                                ty: ty_prim(PrimType::U32),
                                name: id("bad_packets"),
                                init: Some(expr_int(0)),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Loop {
                                body: Block::new(
                                    vec![
                                        Stmt::new(
                                            StmtKind::VarDecl {
                                                ty: ty_named("SupEvent"),
                                                name: id("ev"),
                                                init: Some(Expr::new(
                                                    ExprKind::Or {
                                                        expr: Box::new(Expr::new(
                                                            ExprKind::MethodCall {
                                                                receiver: Box::new(expr_id(
                                                                    "sup_inbox",
                                                                )),
                                                                method: id("recv"),
                                                                args: vec![CallArg::named(
                                                                    id("timeout"),
                                                                    Expr::new(
                                                                        ExprKind::Literal(
                                                                            Literal::Duration(
                                                                                dur_ms(10),
                                                                            ),
                                                                        ),
                                                                        sp,
                                                                    ),
                                                                    sp,
                                                                )],
                                                            },
                                                            sp,
                                                        )),
                                                        fallback: Box::new(expr_id("IdleCheck")),
                                                    },
                                                    sp,
                                                )),
                                            },
                                            sp,
                                        ),
                                        Stmt::new(
                                            StmtKind::Switch {
                                                scrutinee: expr_id("ev"),
                                                arms: vec![
                                                    SwitchArm {
                                                        pat: SwitchPat::Ident {
                                                            prefix: None,
                                                            name: id("Overcurrent"),
                                                            span: sp,
                                                        },
                                                        body: Block::new(
                                                            vec![
                                                                Stmt::new(
                                                                    StmtKind::Expr(Expr::new(
                                                                        ExprKind::MethodCall {
                                                                            receiver: Box::new(
                                                                                expr_id(
                                                                                    "motor1_rpm",
                                                                                ),
                                                                            ),
                                                                            method: id("publish"),
                                                                            args: vec![
                                                                                CallArg::positional(
                                                                                    expr_int(0),
                                                                                ),
                                                                            ],
                                                                        },
                                                                        sp,
                                                                    )),
                                                                    sp,
                                                                ),
                                                                Stmt::new(
                                                                    StmtKind::Expr(Expr::new(
                                                                        ExprKind::Call {
                                                                            callee: Box::new(
                                                                                expr_id(
                                                                                    "save_fault_record",
                                                                                ),
                                                                            ),
                                                                            generic_args: vec![],
                                                                            args: vec![
                                                                                CallArg::positional(
                                                                                    expr_id(
                                                                                        "bad_packets",
                                                                                    ),
                                                                                ),
                                                                            ],
                                                                        },
                                                                        sp,
                                                                    )),
                                                                    sp,
                                                                ),
                                                            ],
                                                            sp,
                                                        ),
                                                        span: sp,
                                                    },
                                                    SwitchArm {
                                                        pat: SwitchPat::Ident {
                                                            prefix: None,
                                                            name: id("BadPacket"),
                                                            span: sp,
                                                        },
                                                        body: Block::new(
                                                            vec![Stmt::new(
                                                                StmtKind::Assign {
                                                                    target: expr_id("bad_packets"),
                                                                    op: AssignOp::AddAssign,
                                                                    value: expr_int(1),
                                                                },
                                                                sp,
                                                            )],
                                                            sp,
                                                        ),
                                                        span: sp,
                                                    },
                                                    SwitchArm {
                                                        pat: SwitchPat::Ident {
                                                            prefix: None,
                                                            name: id("IdleCheck"),
                                                            span: sp,
                                                        },
                                                        body: Block::new(
                                                            vec![Stmt::new(
                                                                StmtKind::Expr(Expr::new(
                                                                    ExprKind::Call {
                                                                        callee: Box::new(expr_id(
                                                                            "scrub_flash_crc",
                                                                        )),
                                                                        generic_args: vec![],
                                                                        args: vec![],
                                                                    },
                                                                    sp,
                                                                )),
                                                                sp,
                                                            )],
                                                            sp,
                                                        ),
                                                        span: sp,
                                                    },
                                                ],
                                            },
                                            sp,
                                        ),
                                    ],
                                    sp,
                                ),
                            },
                            sp,
                        ),
                    ],
                    sp,
                )),
                span: sp,
            }),
            span: sp,
        });

        let prog = Program::new(items, sp);
        let mir = lower_program(&prog).expect("MIR lowering");
        let c_code = emit_c(&mir);

        let driver = r#"
uint32_t c_query_running_prio(void) {
    return g_syn_sched.running ? (uint32_t)g_syn_sched.running->priority : 99U;
}

int main(void) {
    syn_boot_init();

    /* 1. Before DMA is active, allowed sleep depth MUST be SYN_SLEEP_DEEP (2). */
    if (syn_sim_allowed_sleep_depth() != (uint32_t)SYN_SLEEP_DEEP) {
        fprintf(stderr, "Expected DEEP sleep before DMA active!\n");
        return 1;
    }

    /* 2. Push a VALID big-endian CmdFrame: op=1, setpoint=0x05DC (1500), crc=1+5+0xDC = 0x00E2 */
    const uint8_t valid_pkt[5] = { 0x01, 0x05, 0xDC, 0x00, 0xE2 };
    syn_sim_push_dma_bytes(&uart1_dma, valid_pkt, 5);

    /* While DMA is active, allowed sleep depth MUST clamp to SYN_SLEEP_LIGHT (1)! */
    if (syn_sim_allowed_sleep_depth() != (uint32_t)SYN_SLEEP_LIGHT) {
        fprintf(stderr, "Expected LIGHT sleep while ping_pong DMA active!\n");
        return 2;
    }

    syn_sim_fire_isr_DMA1_CH5();
    syn_sim_advance_ms(2);

    if (syn_watch_u16_read(&motor1_rpm) != 1500U) {
        fprintf(stderr, "Expected motor1_rpm == 1500, got %u\n", syn_watch_u16_read(&motor1_rpm));
        return 3;
    }
    uint32_t ccr1 = 0U;
    memcpy(&ccr1, (const void *)(__zev_mmio_buf_TIM1 + 0x34U + sizeof(uint32_t) * 1U), sizeof(uint32_t));
    if (ccr1 != 1500U) {
        fprintf(stderr, "Expected TIM1 CCR1 == 1500, got %u\n", (unsigned)ccr1);
        return 4;
    }

    /* 3. Push a CORRUPT CRC packet -> fieldbus_listener catches ParseErr and posts BadPacket! */
    const uint8_t bad_pkt[5] = { 0x01, 0x09, 0xC4, 0xFF, 0xFF };
    syn_sim_push_dma_bytes(&uart1_dma, bad_pkt, 5);
    syn_sim_fire_isr_DMA1_CH5();
    syn_sim_advance_ms(1);

    /* 4. Fire EXTI0 Overcurrent ISR -> boosts `main` supervisor from @low (3) to @critical (0)! */
    syn_sim_fire_isr_EXTI0();
    /* Immediately after EXTI0 ISR, main (slot 2) MUST be boosted to SYN_PRIO_CRITICAL (0)! */
    if (g_syn_tasks[2].priority != SYN_PRIO_CRITICAL) {
        fprintf(stderr, "Expected main task boosted to CRITICAL (0), got %u\n", g_syn_tasks[2].priority);
        return 5;
    }

    /* Advance 2ms so `save_fault_record` finishes its 1ms flash delay and turn completes */
    syn_sim_advance_ms(2);

    /* Inside `save_fault_record` (after its internal 1ms yield!), priority was 0, and bad_packets was 1 -> 1*100 + 0 = 100! */
    if (syn_watch_u32_read(&fault_saved_prio) != 100U) {
        fprintf(stderr, "Expected fault_saved_prio == 100 (bad_cnt=1, prio=0), got %u\n",
                syn_watch_u32_read(&fault_saved_prio));
        return 6;
    }
    if (syn_watch_u16_read(&motor1_rpm) != 0U) {
        fprintf(stderr, "Expected Overcurrent to zero motor1_rpm!\n");
        return 7;
    }
    /* After turn yielded back to sup_inbox.recv, main's priority MUST be restored to @low (3)! */
    if (g_syn_tasks[2].priority != SYN_PRIO_LOW) {
        fprintf(stderr, "Expected main priority restored to LOW (3), got %u\n", g_syn_tasks[2].priority);
        return 8;
    }

    /* 5. Stop DMA -> allowed sleep depth restores to SYN_SLEEP_DEEP (2), advance 12ms to trigger 10ms timeout `or IdleCheck` -> `scrub_flash_crc()`! */
    syn_sim_set_pp_dma_active(&uart1_dma, false);
    if (syn_sim_allowed_sleep_depth() != (uint32_t)SYN_SLEEP_DEEP) {
        fprintf(stderr, "Expected DEEP sleep restored after DMA deactivated!\n");
        return 9;
    }
    syn_sim_advance_ms(12);
    if (syn_watch_u32_read(&scrub_runs) != 4U) {
        fprintf(stderr, "Expected IdleCheck fallback to run scrub_flash_crc (4 sectors), got %u\n",
                syn_watch_u32_read(&scrub_runs));
        return 10;
    }

    printf("ALL_MOTOR_SIM_CHECKS_PASSED\n");
    return 0;
}
"#;

        let stdout = compile_and_run_clang_sanitized("motor_full_sim", &c_code, Some(driver));
        assert!(stdout.contains("ALL_MOTOR_SIM_CHECKS_PASSED"), "stdout: {stdout}");
    }

    #[test]
    fn partial_progress_dual_exit_error_and_catch() {
        let sp = Span::dummy();
        let items = vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Enum(EnumDecl {
                    name: id("IoErr"),
                    variants: vec![EnumVariant {
                        name: id("Timeout"),
                        discriminant: None,
                        span: sp,
                    }],
                    span: sp,
                }),
                span: sp,
            },
            // usize read_stream() error IoErr { error Timeout, 42; }
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("read_stream"),
                    generic_params: vec![],
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Usize),
                    error_spec: Some(ErrorSpec::Typed(ty_named("IoErr"))),
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Error {
                                reason: Some(expr_id("Timeout")),
                                partial: Some(expr_int(42)),
                            },
                            sp,
                        )],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                    attrs: vec![],
                    ty: OsPrimitiveType::Watch {
                        elem: Box::new(ty_prim(PrimType::Usize)),
                    },
                    name: id("captured_partial"),
                    init: Some(expr_int(0)),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("main"),
                    generic_params: vec![],
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::VarDecl {
                                ty: ty_prim(PrimType::Usize),
                                name: id("n"),
                                init: Some(Expr::new(
                                    ExprKind::Catch {
                                        expr: Box::new(Expr::new(
                                            ExprKind::Call {
                                                callee: Box::new(expr_id("read_stream")),
                                                generic_args: vec![],
                                                args: vec![],
                                            },
                                            sp,
                                        )),
                                        clause: CatchClause {
                                            err_param: Some(CatchParam {
                                                ty: ty_named("IoErr"),
                                                name: id("e"),
                                                span: sp,
                                            }),
                                            partial_param: Some(CatchParam {
                                                ty: ty_prim(PrimType::Usize),
                                                name: id("partial"),
                                                span: sp,
                                            }),
                                            body: Block::new(
                                                vec![
                                                    Stmt::new(
                                                        StmtKind::Expr(Expr::new(
                                                            ExprKind::MethodCall {
                                                                receiver: Box::new(expr_id(
                                                                    "captured_partial",
                                                                )),
                                                                method: id("publish"),
                                                                args: vec![CallArg::positional(
                                                                    expr_id("partial"),
                                                                )],
                                                            },
                                                            sp,
                                                        )),
                                                        sp,
                                                    ),
                                                    Stmt::new(StmtKind::Return(None), sp),
                                                ],
                                                sp,
                                            ),
                                            span: sp,
                                        },
                                    },
                                    sp,
                                )),
                            },
                            sp,
                        )],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
        ];

        let prog = Program::new(items, sp);
        let mir = lower_program(&prog).expect("MIR lowering");
        let c_code = emit_c(&mir);

        let driver = r#"
int main(void) {
    syn_boot_init();
    if (syn_watch_usize_read(&captured_partial) != 42U) {
        fprintf(stderr, "Expected partial progress 42, got %zu\n", syn_watch_usize_read(&captured_partial));
        return 1;
    }
    printf("PARTIAL_PROGRESS_OK\n");
    return 0;
}
"#;
        let stdout = compile_and_run_clang_sanitized("partial_err", &c_code, Some(driver));
        assert!(stdout.contains("PARTIAL_PROGRESS_OK"));
    }

    fn parse_examples_compat(paths: &[&str]) -> Vec<(Program, String)> {
        let mut raws = Vec::with_capacity(paths.len());
        for &path in paths {
            let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            raws.push((path, raw));
        }
        let roots: Vec<(&str, Option<&std::path::Path>)> = raws
            .iter()
            .map(|(p, r)| (r.as_str(), Some(std::path::Path::new(*p))))
            .collect();
        let all_generics = zev_sema::collect_import_aware_generic_names(&roots, &[]);
        raws.into_iter()
            .map(|(path, raw)| {
                let prog = match zev_parser::parse_str_named_with_generics(
                    &raw,
                    path,
                    Some(&all_generics),
                ) {
                    Ok(prog) => prog,
                    Err(_) => {
                        let migrated = raw.replace("    every ", "    loop every ");
                        zev_parser::parse_str_named_with_generics(
                            &migrated,
                            path,
                            Some(&all_generics),
                        )
                        .unwrap_or_else(|e| panic!("parse {path}: {e:?}"))
                    }
                };
                (prog, path.to_string())
            })
            .collect()
    }

    #[test]
    fn parse_lower_and_compile_motor_controller_plus_support() {
        let progs = parse_examples_compat(&[
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/motor_controller.zev"),
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/motor_support.zev"),
        ]);
        let named: Vec<(&Program, Option<&str>)> = progs
            .iter()
            .map(|(p, path)| (p, Some(path.as_str())))
            .collect();
        let analyzed = zev_sema::analyze_and_monomorphize_named_programs(
            &named,
            &zev_sema::AnalyzeOptions::default(),
        );
        assert!(
            analyzed.diagnostics.is_empty(),
            "sema motor_controller + motor_support: {:?}",
            analyzed.diagnostics
        );

        let mir = zev_mir::lower_programs(&analyzed.programs)
            .expect("lower motor_controller + motor_support");

        // Verify North Star static RAM invariants
        assert_eq!(mir.spawned_tasks.len(), 5);
        assert!(mir.ram_report.mcu32.union_bytes_saved > 0);
        assert_eq!(mir.ram_report.mcu32.heap_bytes, 0);

        let mc_bp = mir
            .task_blueprints
            .iter()
            .find(|b| b.name == "motor_control")
            .unwrap();
        assert!(mc_bp.vars.iter().find(|v| v.name == "integral").unwrap().hoisted);
        assert!(!mc_bp.vars.iter().find(|v| v.name == "duty").unwrap().hoisted);

        let main_bp = mir
            .task_blueprints
            .iter()
            .find(|b| b.name == "main")
            .unwrap();
        assert!(main_bp.vars.iter().find(|v| v.name == "bad_packets").unwrap().hoisted);

        let mut opts = CodegenOptions::default();
        opts.default_sim_max_ticks = 15;
        let c_code = emit_c_with_options(&mir, &opts);
        let _ = compile_and_run_clang_sanitized("motor_north_star", &c_code, None);
    }

    #[test]
    fn parse_lower_compile_and_run_sim_demo() {
        let progs = parse_examples_compat(&[concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/sim_demo.zev"
        )]);
        let named: Vec<(&Program, Option<&str>)> = progs
            .iter()
            .map(|(p, path)| (p, Some(path.as_str())))
            .collect();
        let analyzed = zev_sema::analyze_and_monomorphize_named_programs(
            &named,
            &zev_sema::AnalyzeOptions::default(),
        );
        assert!(
            analyzed.diagnostics.is_empty(),
            "sema sim_demo.zev: {:?}",
            analyzed.diagnostics
        );
        let mir = zev_mir::lower_programs(&analyzed.programs).expect("lower sim_demo.zev");
        assert!(mir.ram_report.mcu32.union_bytes_saved > 0);

        let c_code = emit_c(&mir);
        let stdout = compile_and_run_clang_sanitized("sim_demo_e2e", &c_code, None);

        // Milestone 1: Active @dma_safe ping_pong clamps sleep depth to Light (1)
        assert!(stdout.contains("milestone=1 a=1 b=0"), "stdout:\n{stdout}");
        // Milestone 2: u16be decoded RPMs -> 2400 & 1800
        assert!(stdout.contains("milestone=2 a=2400 b=1800"), "stdout:\n{stdout}");
        // Milestone 4 (tag=4): EXTI0 Overcurrent zeroed both motor RPMs
        assert!(stdout.contains("milestone=4 a=0 b=0"), "stdout:\n{stdout}");
        // Milestone 5 (tag=5): Released DMA wake-lock restores sleep depth to Deep (2)
        assert!(stdout.contains("milestone=5 a=2 "), "stdout:\n{stdout}");
    }

    #[test]
    fn signal_inbox_lease_little_endian_and_freestanding_mode() {
        let src = r#"
packed struct LeHeader {
    u8    magic;
    u16le length;
    u32le seq_num;
}

signal ready_sig;
signal slot_ready_sig;
inbox<u32, 4> work_inbox;
@dma_safe pool<[16]u8, 2> spi_pool;
dma_slot<[16]u8> spi_slot;
watch<u32> result_watch = 0;

task producer() {
    work_inbox.send(77);
    ready_sig.notify(boost: @high);
    loop {
        delay(1s);
    }
}

task consumer() {
    loop {
        ready_sig.wait();
        u32 v = work_inbox.recv();
        slot_ready_sig.wait();
        Lease<[16]u8> l = spi_slot.take() catch {
            restart;
        };
        [16]u8 frame = *l.get();
        LeHeader hdr = LeHeader.from_bytes(frame[0..7]) catch {
            l.release();
            restart;
        };
        l.release();
        u32 total = v + (hdr.length as u32) + hdr.seq_num;
        result_watch.publish(total);
    }
}

fn main() {
    spawn producer() @normal;
    spawn consumer() @low;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse signal/inbox/pool test");
        let mir = zev_mir::lower_program(&prog).expect("lower signal/inbox/pool");
        let c_code = emit_c(&mir);

        // Verify freestanding mode (-DSYN_FREESTANDING) compiles with zero warnings!
        let fs_dir = format!("/tmp/zev-c-tests-{}/freestanding_check", std::process::id());
        fs::create_dir_all(&fs_dir).unwrap();
        let fs_c = format!("{fs_dir}/fs.c");
        let fs_o = format!("{fs_dir}/fs.o");
        fs::write(&fs_c, &c_code).unwrap();
        let fs_out = Command::new("/bin/clang")
            .args([
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-DSYN_FREESTANDING",
                "-c",
                &fs_c,
                "-o",
                &fs_o,
            ])
            .output()
            .unwrap();
        assert!(
            fs_out.status.success(),
            "SYN_FREESTANDING compile failed:\n{}",
            String::from_utf8_lossy(&fs_out.stderr)
        );

        let lease_res_ty = "syn_res_lease_arr_16_u8_err_bool_t";
        let driver = format!(r#"
int main(void) {{
    syn_boot_init();
    syn_sim_pump_ready();

    /* Complete an allocated DMA pool buffer with little-endian magic=0xA5, length=100 (0x0064 -> 64 00), seq_num=1000 (0x000003E8 -> E8 03 00 00) */
    const uint8_t wire[7] = {{ 0xA5, 0x64, 0x00, 0xE8, 0x03, 0x00, 0x00 }};
    {lease_res_ty} ar = syn_pool_arr_16_u8_2_try_alloc(&spi_pool);
    if (ar.is_err) {{ return 1; }}
    if (syn_sim_allowed_sleep_depth() != (uint32_t)SYN_SLEEP_LIGHT) {{
        return 2;
    }}
    memcpy(ar.val.data->data, wire, 7U);
    if (syn_dma_slot_arr_16_u8_park(&spi_slot, ar.val).is_err) {{
        return 3;
    }}
    syn_signal_notify(&slot_ready_sig, SYN_NO_BOOST);

    syn_sim_pump_ready();

    /* 77 + 100 + 1000 = 1177, and releasing the lease restores DEEP sleep! */
    if (syn_watch_u32_read(&result_watch) != 1177U) {{
        fprintf(stderr, "Expected 1177, got %u\n", syn_watch_u32_read(&result_watch));
        return 4;
    }}
    if (syn_sim_allowed_sleep_depth() != (uint32_t)SYN_SLEEP_DEEP) {{
        return 5;
    }}
    printf("PRIMS_AND_LE_OK\n");
    return 0;
}}
"#);
        let stdout = compile_and_run_clang_sanitized("prims_and_le", &c_code, Some(&driver));
        assert!(stdout.contains("PRIMS_AND_LE_OK"));
    }

    #[test]
    fn starvation_bound_and_should_share_cooperative_chunking() {
        let src = r#"
extern "C" fn c_tick_inside_loop();

watch<u32> bg_iters = 0;
watch<u32> low_ran = 0;

task heavy_normal() {
    loop {
        u32 cur = bg_iters.read();
        bg_iters.publish(cur + 1);
        c_tick_inside_loop();
        if (cur + 1 >= 200) {
            return;
        }
    }
}

task low_peer() {
    low_ran.publish(1);
}

fn main() {
    spawn heavy_normal() @normal;
    spawn low_peer() @low;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse starvation test");
        let mir = zev_mir::lower_program(&prog).expect("lower starvation test");
        let c_code = emit_c(&mir);

        let driver = r#"
/* Simulate 10ms passing per heavy loop iteration -> after 5 iterations (50ms == @low starvation window!),
 * `should_share()` MUST yield MORE so `@low` runs in OVERDUE rank before `heavy_normal` reaches 200! */
static uint32_t low_ran_at_bg_iter = 0U;

void c_tick_inside_loop(void) {
    g_syn_tick_ms += 10U;
}

int main(void) {
    syn_boot_init();
    while (g_syn_tasks[0].state != SYN_TASK_DEAD) {
        bool alive = false;
        syn_sched_step(&alive);
        if (syn_watch_u32_read(&low_ran) == 1U && low_ran_at_bg_iter == 0U) {
            low_ran_at_bg_iter = syn_watch_u32_read(&bg_iters);
        }
    }
    if (low_ran_at_bg_iter == 0U || low_ran_at_bg_iter >= 20U) {
        fprintf(stderr, "Expected @low task to run via 50ms starvation window around iter 5, ran at %u\n",
                low_ran_at_bg_iter);
        return 1;
    }
    printf("STARVATION_SHARE_OK iter=%u\n", low_ran_at_bg_iter);
    return 0;
}
"#;
        let stdout = compile_and_run_clang_sanitized("starvation_share", &c_code, Some(driver));
        assert!(stdout.contains("STARVATION_SHARE_OK"));
    }

    #[test]
    fn const_switch_multistmt_catch_signed_sub_and_io_parity() {
        let src = r#"
const u8 CMD_READ = 1;
const u8 CMD_SET  = 2;

struct DecodedPacket {
    u8  cmd;
    i32 temp;
}

enum PacketErr { BadFrame }

watch<u32> err_count = 0;

DecodedPacket decode(u8 raw) error PacketErr {
    switch (raw) {
        CMD_READ => {
            return DecodedPacket{ cmd: CMD_READ, temp: 25 };
        }
        CMD_SET => {
            u16 cur = 500;
            i32 delta = 300 - (cur as i32);
            return DecodedPacket{ cmd: CMD_SET, temp: delta };
        }
        _ => {
            error BadFrame, DecodedPacket{ cmd: 0, temp: 4 };
        }
    }
}

task worker() {
    u32 local_errs = 0;
    DecodedPacket p1 = decode(2) catch (PacketErr e) {
        sys.exit(2);
    };
    sys.log_kv("p1_temp=", p1.temp);

    DecodedPacket p2 = decode(99) catch (PacketErr e, DecodedPacket partial) {
        local_errs += 1;
        err_count.publish(local_errs);
        DecodedPacket{ cmd: 9, temp: partial.temp + 100 }
    };
    DecodedPacket p3 = decode(98) catch (PacketErr e, DecodedPacket partial) {
        local_errs += 1;
        DecodedPacket{ cmd: 7, temp: partial.temp + 50 };
    };
    sys.log_kv("p2_cmd=", p2.cmd as i32);
    sys.log_kv("p2_temp=", p2.temp);
    sys.log_kv("p3_temp=", p3.temp);
    sys.log_kv("errs=", err_count.read() as i32);
    println("hello");
    sys.log("syslog_ok");
    sys.log_kv("k=", -5);
    sys.log_hex("hex=", 0x1A2B);
    print("uart0_ok\n");
    print("usart1_ok\n");
    sys.log_kv("now=", sys.now_ms() as i32);
    sys.exit(0);
}

fn main() {
    spawn worker() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse combined fix test");
        let mir = zev_mir::lower_program(&prog).expect("lower combined fix test");
        let c_code = emit_c(&mir);

        let stdout = compile_and_run_clang_sanitized("fixes_combined", &c_code, None);
        assert!(stdout.contains("p1_temp=-200\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("p2_cmd=9\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("p2_temp=104\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("p3_temp=54\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("errs=1\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("hello\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("syslog_ok\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("k=-5\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("hex=0x00001A2B\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("uart0_ok\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("usart1_ok\n"), "stdout:\n{stdout}");
    }

    #[test]
    fn busy_checkpoint_loop_advances_ticks_and_terminates_at_max_ticks() {
        let src = r#"
fn main() {
    loop {
        checkpoint();
    }
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse busy checkpoint loop");
        let mir = zev_mir::lower_program(&prog).expect("lower busy checkpoint loop");
        let mut opts = CodegenOptions::default();
        opts.default_sim_max_ticks = 10;
        let c_code = emit_c_with_options(&mir, &opts);

        // Must terminate promptly (in <10ms) once 10 simulated ticks advance across busy checkpoints!
        let _stdout = compile_and_run_clang_sanitized("busy_checkpoint_max_ticks", &c_code, None);
    }

    fn assert_freestanding_compiles(test_name: &str, c_src: &str) {
        let fs_dir = format!("/tmp/zev-c-tests-{}/{test_name}_fs", std::process::id());
        fs::create_dir_all(&fs_dir).unwrap();
        let fs_c = format!("{fs_dir}/fs.c");
        let fs_o = format!("{fs_dir}/fs.o");
        fs::write(&fs_c, c_src).unwrap();
        let fs_out = Command::new("/bin/clang")
            .args([
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-DSYN_FREESTANDING",
                "-c",
                &fs_c,
                "-o",
                &fs_o,
            ])
            .output()
            .expect("invoke clang -DSYN_FREESTANDING -c");
        assert!(
            fs_out.status.success(),
            "SYN_FREESTANDING compile failed for {test_name}:\n{}\n--- C SOURCE ---\n{}",
            String::from_utf8_lossy(&fs_out.stderr),
            c_src
        );
        let _ = fs::remove_dir_all(&fs_dir);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Section 6.2 Wave 1 — Live Clang ASan/UBSan + Freestanding Execution Suite
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn wave1_boot_tree_spawn_conditional_activation_and_ram_provenance() {
        let src = r#"
watch<u32> w_a = 0;
watch<u32> w_b = 0;
watch<u32> w_diag = 0;
watch<u32> w_opt = 0;

task worker(u32 val, *watch<u32> out) {
    loop every 5ms {
        out.publish(val);
    }
}

task diag_task(u32 tag) {
    loop every 5ms {
        w_diag.publish(tag);
    }
}

task opt_task(u32 tag) {
    loop every 5ms {
        w_opt.publish(tag);
    }
}

fn init_workers(*watch<u32> a, *watch<u32> b) {
    spawn worker(101, a) @high;
    spawn worker(202, b) @high;
}

fn init_optional(bool early_stop) {
    spawn opt_task(77) @normal;
    if (early_stop) {
        return;
    }
    spawn opt_task(88) @normal;
}

fn setup_system(bool en_diag, bool en_aux) {
    init_workers(&w_a, &w_b);
    if (en_diag) {
        spawn diag_task(111) @low;
    }
    if (en_aux) {
        spawn diag_task(222) @low;
    }
    init_optional(true);
}

fn main() {
    setup_system(true, false);
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse wave1 boot tree");
        let mir = zev_mir::lower_program(&prog).expect("lower wave1 boot tree");

        assert_eq!(mir.spawned_tasks.len(), 6);
        assert_eq!(
            mir.spawned_tasks[0].provenance_path,
            vec!["main", "setup_system", "init_workers", "worker"]
        );
        assert!(!mir.spawned_tasks[0].conditional);
        assert!(mir.spawned_tasks[2].conditional);
        assert!(mir.spawned_tasks[3].conditional);

        let table = mir.ram_report.format_table();
        assert!(
            table.contains("boot provenance  : main -> setup_system -> init_workers -> worker"),
            "table:\n{table}"
        );
        assert!(table.contains("[conditional]"), "table:\n{table}");

        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave1_boot_tree", &c_code);

        let driver = r#"
int main(void) {
    syn_boot_init();
    syn_sim_advance_ms(6);

    if (syn_watch_u32_read(&w_a) != 101U || syn_watch_u32_read(&w_b) != 202U) {
        fprintf(stderr, "Expected w_a=101, w_b=202, got %u, %u\n",
                syn_watch_u32_read(&w_a), syn_watch_u32_read(&w_b));
        return 1;
    }
    if (syn_watch_u32_read(&w_diag) != 111U) {
        fprintf(stderr, "Expected w_diag=111, got %u\n", syn_watch_u32_read(&w_diag));
        return 2;
    }
    if (syn_watch_u32_read(&w_opt) != 77U) {
        fprintf(stderr, "Expected w_opt=77 (early return skipped 88), got %u\n", syn_watch_u32_read(&w_opt));
        return 3;
    }
    /* Slot 3 (en_aux == false) and Slot 5 (post early-return) MUST stay SYN_TASK_INACTIVE! */
    if (g_syn_tasks[3].state != SYN_TASK_INACTIVE) {
        fprintf(stderr, "Expected untaken conditional slot 3 to stay INACTIVE, got %u\n", g_syn_tasks[3].state);
        return 4;
    }
    if (g_syn_tasks[5].state != SYN_TASK_INACTIVE) {
        fprintf(stderr, "Expected post-return slot 5 to stay INACTIVE, got %u\n", g_syn_tasks[5].state);
        return 5;
    }
    printf("WAVE1_BOOT_TREE_OK\n");
    return 0;
}
"#;
        let stdout = compile_and_run_clang_sanitized("wave1_boot_tree", &c_code, Some(driver));
        assert!(stdout.contains("WAVE1_BOOT_TREE_OK"));
    }

    #[test]
    fn wave1_loop_every_runtime_period_and_overrun_skip_resync() {
        let src = r#"
extern "C" fn c_maybe_stall_in_body(u32 iter);

watch<u32> inbody_runs = 0;
watch<u32> sleep_runs = 0;

u32 compute_period_ms(u32 mult) {
    return mult * 5;
}

task inbody_periodic() {
    u32 p = compute_period_ms(2);
    loop every p {
        u32 next = inbody_runs.read() + 1;
        inbody_runs.publish(next);
        c_maybe_stall_in_body(next);
    }
}

task sleep_periodic() {
    loop every 10ms {
        sleep_runs.publish(sleep_runs.read() + 1);
    }
}

fn main() {
    spawn inbody_periodic() @high;
    spawn sleep_periodic() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse wave1 loop every");
        let mir = zev_mir::lower_program(&prog).expect("lower wave1 loop every");
        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave1_loop_every", &c_code);

        let driver = r#"
void c_maybe_stall_in_body(u32_dummy) { (void)u32_dummy; }
"#;
        let _ = driver;
        let real_driver = r#"
void c_maybe_stall_in_body(uint32_t iter) {
    if (iter == 1U) {
        /* Stall simulated clock by 50ms on a 10ms loop during Iteration 1! */
        g_syn_tick_ms += 50U;
    }
}

int main(void) {
    syn_boot_init();
    /* Pump at t=0 -> Iteration 1 of `inbody_periodic` stalls clock to t=50ms!
     * Overrun skip/resync MUST run 1 resynced iteration (total 2, NOT 6!) for BOTH
     * `inbody_periodic` (stalled in body) AND `sleep_periodic` (stalled while sleeping or starting)! */
    syn_sim_pump_ready();

    uint32_t ib = syn_watch_u32_read(&inbody_runs);
    if (ib != 2U) {
        fprintf(stderr, "Expected inbody_runs == 2 after 50ms stall (1 initial + 1 resynced, NO burst!), got %u\n", ib);
        return 1;
    }

    /* Advance 5ms (t=55ms): next grid tick is t=60ms, so NEITHER task should fire yet! */
    syn_sim_advance_ms(5);
    if (syn_watch_u32_read(&inbody_runs) != 2U) {
        fprintf(stderr, "Expected inbody_runs still 2 at t=55ms, got %u\n", syn_watch_u32_read(&inbody_runs));
        return 2;
    }
    uint32_t sb_before_stall = syn_watch_u32_read(&sleep_runs);

    /* Now stall clock by 50ms while BOTH tasks are sleeping (t=55ms -> t=105ms) and pump! */
    g_syn_tick_ms += 50U;
    syn_sim_pump_ready();

    /* Each 10ms periodic task MUST execute EXACTLY 1 resynced iteration on wakeup, NOT 5! */
    if (syn_watch_u32_read(&inbody_runs) != 3U) {
        fprintf(stderr, "Expected inbody_runs == 3 after 50ms sleep stall, got %u\n",
                syn_watch_u32_read(&inbody_runs));
        return 3;
    }
    if (syn_watch_u32_read(&sleep_runs) != sb_before_stall + 1U) {
        fprintf(stderr, "Expected sleep_runs to advance by EXACTLY 1 after 50ms sleep stall (was %u, got %u)\n",
                sb_before_stall, syn_watch_u32_read(&sleep_runs));
        return 4;
    }

    printf("WAVE1_LOOP_EVERY_OVERRUN_OK\n");
    return 0;
}
"#;
        let stdout =
            compile_and_run_clang_sanitized("wave1_loop_every", &c_code, Some(real_driver));
        assert!(stdout.contains("WAVE1_LOOP_EVERY_OVERRUN_OK"));
    }

    #[test]
    fn wave1_restart_preserves_task_params_unwinds_defers_and_restores_boost() {
        let src = r#"
enum ActorCmd { AddWork, TriggerRestart }

inbox<ActorCmd, 4> actor_q;
watch<u32> actor_state = 0;
watch<u32> actor_defers = 0;

task supervisor_actor(u32 base_id, *watch<u32> out_state) {
    u32 local_acc = 10;
    defer {
        actor_defers.publish(actor_defers.read() + 1);
    }
    out_state.publish(base_id + local_acc);
    loop {
        ActorCmd cmd = actor_q.recv();
        switch (cmd) {
            AddWork => {
                local_acc += 25;
                out_state.publish(base_id + local_acc);
            }
            TriggerRestart => {
                local_acc += 900;
                restart;
            }
        }
    }
}

fn main() {
    spawn supervisor_actor(500, &actor_state) @low;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse wave1 restart");
        let mir = zev_mir::lower_program(&prog).expect("lower wave1 restart");
        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave1_restart", &c_code);

        let driver = r#"
int main(void) {
    syn_boot_init();
    syn_sim_pump_ready();

    /* Initial boot entry: base_id (500) + local_acc (10) == 510 */
    if (syn_watch_u32_read(&actor_state) != 510U) {
        fprintf(stderr, "Expected initial 510, got %u\n", syn_watch_u32_read(&actor_state));
        return 1;
    }

    /* Post AddWork -> 535 */
    syn_inbox_ActorCmd_4_post(&actor_q, syn_ActorCmd_AddWork, SYN_NO_BOOST);
    syn_sim_pump_ready();
    if (syn_watch_u32_read(&actor_state) != 535U) {
        fprintf(stderr, "Expected 535 after AddWork, got %u\n", syn_watch_u32_read(&actor_state));
        return 2;
    }

    /* Post TriggerRestart WITH @critical boost!
     * `restart;` MUST:
     * 1. Run task-level `defer` (actor_defers -> 1),
     * 2. Restore boosted priority back to @low (SYN_PRIO_LOW = 3),
     * 3. Re-enter BlockId(0) keeping `base_id == 500` intact and resetting `local_acc = 10` -> 510! */
    syn_inbox_ActorCmd_4_post(&actor_q, syn_ActorCmd_TriggerRestart, SYN_PRIO_CRITICAL);
    syn_sim_pump_ready();

    if (syn_watch_u32_read(&actor_defers) != 1U) {
        fprintf(stderr, "Expected actor_defers == 1 on restart, got %u\n", syn_watch_u32_read(&actor_defers));
        return 3;
    }
    if (g_syn_tasks[0].priority != SYN_PRIO_LOW) {
        fprintf(stderr, "Expected priority restored to SYN_PRIO_LOW (3) across restart, got %u\n",
                g_syn_tasks[0].priority);
        return 4;
    }
    if (syn_watch_u32_read(&actor_state) != 510U) {
        fprintf(stderr, "Expected restarted state 510 (base_id=500 preserved, local_acc=10 reset), got %u\n",
                syn_watch_u32_read(&actor_state));
        return 5;
    }

    printf("WAVE1_RESTART_OK\n");
    return 0;
}
"#;
        let stdout = compile_and_run_clang_sanitized("wave1_restart", &c_code, Some(driver));
        assert!(stdout.contains("WAVE1_RESTART_OK"));
    }

    #[test]
    fn wave1_lifo_defer_after_return_eval_and_error_and_loop_unwinding() {
        let src = r#"
enum OpErr { Failed }

watch<u32> trace_log = 0;
watch<u32> direct_ok_out = 0;
watch<u32> direct_err_partial = 0;
watch<u32> coro_ok_out = 0;
watch<u32> coro_err_partial = 0;

fn append_trace(u32 digit) {
    trace_log.publish(trace_log.read() * 10 + digit);
}

u32 direct_op(bool should_fail) error OpErr {
    u32 ret_val = 42;
    defer {
        append_trace(1);
    }
    {
        defer {
            append_trace(2);
        }
    }
    for (u8 i in 0..3) {
        defer {
            append_trace(3);
        }
        if (i == 1) {
            continue;
        }
        if (i == 2) {
            break;
        }
    }
    defer {
        ret_val = 999;
        append_trace(4);
    }
    if (should_fail) {
        error Failed, ret_val + 5;
    }
    return ret_val + 8;
}

u32 coro_op(bool should_fail) error OpErr {
    u32 ret_val = 100;
    defer {
        append_trace(7);
    }
    delay(1ms);
    defer {
        ret_val = 888;
        append_trace(8);
    }
    if (should_fail) {
        error Failed, ret_val + 11;
    }
    return ret_val + 22;
}

task tester() {
    trace_log.publish(0);
    u32 ok1 = direct_op(false) catch { 0 };
    direct_ok_out.publish(ok1);
    u32 t1 = trace_log.read();

    trace_log.publish(0);
    u32 err1 = direct_op(true) catch (OpErr e, u32 part) {
        direct_err_partial.publish(part);
        0
    };
    u32 t2 = trace_log.read();

    trace_log.publish(0);
    u32 ok2 = coro_op(false) catch { 0 };
    coro_ok_out.publish(ok2);
    u32 t3 = trace_log.read();

    trace_log.publish(0);
    u32 err2 = coro_op(true) catch (OpErr e, u32 part) {
        coro_err_partial.publish(part);
        0
    };
    u32 t4 = trace_log.read();

    assert(t1 == 233341);
    assert(t2 == 233341);
    assert(t3 == 87);
    assert(t4 == 87);
    assert(err1 == 0 && err2 == 0);
    sys.exit(0);
    loop {
        delay(1s);
    }
}

fn main() {
    spawn tester() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse wave1 lifo defer");
        let mir = zev_mir::lower_program(&prog).expect("lower wave1 lifo defer");
        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave1_defer", &c_code);

        let driver = r#"
int main(void) {
    syn_boot_init();
    syn_sim_advance_ms(5);

    /* `ret_val` was 42 when `return ret_val + 8;` / `error Failed, ret_val + 5;` evaluated,
     * BEFORE `defer { ret_val = 999; append_trace(4); }` mutated `ret_val`! */
    if (syn_watch_u32_read(&direct_ok_out) != 50U) {
        fprintf(stderr, "Expected direct_ok_out == 50, got %u\n", syn_watch_u32_read(&direct_ok_out));
        return 1;
    }
    if (syn_watch_u32_read(&direct_err_partial) != 47U) {
        fprintf(stderr, "Expected direct_err_partial == 47, got %u\n", syn_watch_u32_read(&direct_err_partial));
        return 2;
    }
    if (syn_watch_u32_read(&coro_ok_out) != 122U) {
        fprintf(stderr, "Expected coro_ok_out == 122, got %u\n", syn_watch_u32_read(&coro_ok_out));
        return 3;
    }
    if (syn_watch_u32_read(&coro_err_partial) != 111U) {
        fprintf(stderr, "Expected coro_err_partial == 111, got %u\n", syn_watch_u32_read(&coro_err_partial));
        return 4;
    }
    if (!g_syn_sim_stopped || g_syn_sim_exit_code != 0) {
        fprintf(stderr, "Expected sys.exit(0) to stop sim cleanly!\n");
        return 5;
    }
    printf("WAVE1_LIFO_DEFER_OK\n");
    return 0;
}
"#;
        let stdout = compile_and_run_clang_sanitized("wave1_defer", &c_code, Some(driver));
        assert!(stdout.contains("WAVE1_LIFO_DEFER_OK"));
    }

    #[test]
    fn wave1_bounded_for_range_inclusive_max_and_foreach_slice_array_with_continue() {
        let src = r#"
watch<u32> direct_res = 0;
watch<u32> coro_res = 0;
watch<u32> max_u8_count = 0;
watch<i32> max_i8_sum = 0;

u32 sum_direct([]u8 sl) {
    u32 total = 0;
    for (u8 i in 0..=4) {
        if (i == 2) {
            continue;
        }
        total += (i as u32);
    }
    for (u8 b in sl) {
        if (b == 99) {
            continue;
        }
        total += (b as u32);
    }
    [3]u16 fixed = [100, 200, 300];
    for (u16 x in fixed) {
        if (x == 200) {
            continue;
        }
        total += (x as u32);
    }
    return total;
}

task range_task() {
    [4]u8 bytes = [10, 99, 20, 30];
    direct_res.publish(sum_direct(bytes[0..4]));

    u32 c_total = 0;
    for (u8 i in 0..=4) {
        checkpoint();
        if (i == 2) {
            continue;
        }
        c_total += (i as u32);
    }
    []u8 sl = bytes[0..4];
    for (u8 b in sl) {
        checkpoint();
        if (b == 99) {
            continue;
        }
        c_total += (b as u32);
    }
    [3]u16 fixed = [100, 200, 300];
    for (u16 x in fixed) {
        checkpoint();
        if (x == 200) {
            continue;
        }
        c_total += (x as u32);
    }
    coro_res.publish(c_total);

    /* Adversarial overflow safety: `0..=255` on `u8` and `125..=127` on `i8`! */
    u32 u8_iters = 0;
    for (u8 k in 0..=255) {
        u8_iters += 1;
        if (k == 254) {
            continue;
        }
    }
    max_u8_count.publish(u8_iters);

    i32 i8_acc = 0;
    for (i8 s in 125..=127) {
        i8_acc += (s as i32);
    }
    max_i8_sum.publish(i8_acc);

    sys.exit(0);
    loop {
        delay(1s);
    }
}

fn main() {
    spawn range_task() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse wave1 bounded for");
        let mir = zev_mir::lower_program(&prog).expect("lower wave1 bounded for");
        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave1_for_loops", &c_code);

        let driver = r#"
int main(void) {
    syn_boot_init();
    syn_sim_advance_ms(5);

    /* (0+1+3+4)=8 + (10+20+30)=60 + (100+300)=400 == 468 */
    if (syn_watch_u32_read(&direct_res) != 468U) {
        fprintf(stderr, "Expected direct_res == 468, got %u\n", syn_watch_u32_read(&direct_res));
        return 1;
    }
    if (syn_watch_u32_read(&coro_res) != 468U) {
        fprintf(stderr, "Expected coro_res == 468, got %u\n", syn_watch_u32_read(&coro_res));
        return 2;
    }
    if (syn_watch_u32_read(&max_u8_count) != 256U) {
        fprintf(stderr, "Expected 0..=255 u8 loop to run 256 iterations without wrap, got %u\n",
                syn_watch_u32_read(&max_u8_count));
        return 3;
    }
    if (syn_watch_i32_read(&max_i8_sum) != 378) {
        fprintf(stderr, "Expected 125..=127 i8 sum == 378 without UBSan overflow, got %d\n",
                (int)syn_watch_i32_read(&max_i8_sum));
        return 4;
    }
    printf("WAVE1_FOR_LOOPS_OK\n");
    return 0;
}
"#;
        let stdout = compile_and_run_clang_sanitized("wave1_for_loops", &c_code, Some(driver));
        assert!(stdout.contains("WAVE1_FOR_LOOPS_OK"));
    }

    #[test]
    fn wave1_unified_inbox_backpressure_send_timeout_try_ops_and_queries() {
        let src = r#"
inbox<u32, 2> pipe;
watch<u32> prod_stage = 0;
watch<u32> send_timeout_fired = 0;
watch<u32> cons_sum = 0;

task producer() {
    assert(pipe.is_empty());
    assert(!pipe.is_full());
    assert(pipe.len() == 0);
    u32 empty_probe = pipe.try_recv() or 777;
    assert(empty_probe == 777);

    bool ok1 = pipe.try_send(10);
    assert(ok1);
    pipe.send(20);
    assert(pipe.is_full());
    assert(pipe.len() == 2);
    bool ok_full = pipe.try_send(999);
    assert(!ok_full);

    prod_stage.publish(1);

    /* Queue is full (cap=2): `.send(99, timeout: 3ms)` MUST time out while consumer sleeps 10ms! */
    pipe.send(99, timeout: 3ms) catch {
        send_timeout_fired.publish(1);
    };
    prod_stage.publish(2);

    /* Now blocking `.send(30)` MUST suspend until consumer `.recv()` frees a slot! */
    pipe.send(30);
    prod_stage.publish(3);

    loop {
        delay(1s);
    }
}

task consumer() {
    delay(10ms);
    u32 a = pipe.recv();
    u32 b = pipe.recv();
    u32 c = pipe.recv();
    u32 d = pipe.try_recv() or 5;
    cons_sum.publish(a * 1000 + b * 10 + c + d);
    sys.exit(0);
    loop {
        delay(1s);
    }
}

fn main() {
    spawn producer() @high;
    spawn consumer() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse wave1 unified inbox");
        let mir = zev_mir::lower_program(&prog).expect("lower wave1 unified inbox");
        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave1_unified_inbox", &c_code);

        let driver = r#"
int main(void) {
    syn_boot_init();
    syn_sim_pump_ready();

    /* At t=0ms: producer filled cap=2 (10, 20) and is waiting on 3ms send timeout */
    if (syn_watch_u32_read(&prod_stage) != 1U) {
        fprintf(stderr, "Expected prod_stage == 1 at t=0, got %u\n", syn_watch_u32_read(&prod_stage));
        return 1;
    }

    /* Advance 4ms (t=4ms): send(99, timeout: 3ms) MUST have timed out, and producer MUST now be blocked on send(30)! */
    syn_sim_advance_ms(4);
    if (syn_watch_u32_read(&send_timeout_fired) != 1U || syn_watch_u32_read(&prod_stage) != 2U) {
        fprintf(stderr, "Expected send_timeout_fired=1 and prod_stage=2 at t=4ms, got %u, %u\n",
                syn_watch_u32_read(&send_timeout_fired), syn_watch_u32_read(&prod_stage));
        return 2;
    }

    /* Advance to t=12ms: consumer wakes at t=10ms, pops 10 -> wakes producer -> producer pushes 30 (prod_stage=3),
     * consumer drains 20, 30, and try_recv fallback 5 -> 10*1000 + 20*10 + 30 + 5 = 10235! */
    syn_sim_advance_ms(8);
    if (syn_watch_u32_read(&prod_stage) != 3U) {
        fprintf(stderr, "Expected producer unblocked (prod_stage=3), got %u\n", syn_watch_u32_read(&prod_stage));
        return 3;
    }
    if (syn_watch_u32_read(&cons_sum) != 10235U) {
        fprintf(stderr, "Expected cons_sum == 10235, got %u\n", syn_watch_u32_read(&cons_sum));
        return 4;
    }

    printf("WAVE1_UNIFIED_INBOX_OK\n");
    return 0;
}
"#;
        let stdout =
            compile_and_run_clang_sanitized("wave1_unified_inbox", &c_code, Some(driver));
        assert!(stdout.contains("WAVE1_UNIFIED_INBOX_OK"));
    }

    #[test]
    fn wave1_soft_yielding_dual_specialization_task_and_noyield_parity() {
        let src = r#"
watch<u32> coro_crc = 0;
watch<u32> noyield_crc = 0;

u32 mix_byte(u32 acc, u8 b) {
    checkpoint();
    u32 v = acc ^ (b as u32);
    u8 step = 0;
    while (step < 4) {
        switch (step) {
            3 => {
                v = (v * 3) + 1;
                break;
            }
            _ => {
                v = v + 7;
            }
        }
        step += 1;
    }
    return v;
}

u32 compute_crc_soft([]u8 buf) {
    u32 acc = 0x1234;
    for (u8 b in buf) {
        checkpoint();
        acc = mix_byte(acc, b);
    }
    return acc;
}

task verifier() {
    [5]u8 frame = [0x11, 0x22, 0x33, 0x44, 0x55];
    u32 c1 = compute_crc_soft(frame[0..5]);
    u32 c2 = 0;
    noyield {
        c2 = compute_crc_soft(frame[0..5]);
    }
    coro_crc.publish(c1);
    noyield_crc.publish(c2);
    sys.exit(0);
    loop {
        delay(1s);
    }
}

fn main() {
    spawn verifier() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse wave1 soft specialization");
        let mir = zev_mir::lower_program(&prog).expect("lower wave1 soft specialization");

        let dual_names: Vec<&str> = mir
            .ram_report
            .dual_specialized_fns
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        assert!(dual_names.contains(&"compute_crc_soft"), "dual_names: {dual_names:?}");
        assert!(dual_names.contains(&"mix_byte"), "dual_names: {dual_names:?}");

        let table = mir.ram_report.format_table();
        assert!(
            table.contains("Dual-Specialized Functions")
                && table.contains("compute_crc_soft__noyield")
                && table.contains("mix_byte__noyield"),
            "table:\n{table}"
        );

        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave1_soft_spec", &c_code);

        let driver = r#"
int main(void) {
    syn_boot_init();
    syn_sim_advance_ms(5);

    uint32_t c1 = syn_watch_u32_read(&coro_crc);
    uint32_t c2 = syn_watch_u32_read(&noyield_crc);
    if (c1 == 0U || c1 != c2) {
        fprintf(stderr, "Expected non-zero identical CRC from coroutine (%u) and __noyield clone (%u)!\n", c1, c2);
        return 1;
    }
    printf("WAVE1_SOFT_SPECIALIZATION_OK crc=%u\n", c1);
    return 0;
}
"#;
        let stdout = compile_and_run_clang_sanitized("wave1_soft_spec", &c_code, Some(driver));
        assert!(stdout.contains("WAVE1_SOFT_SPECIALIZATION_OK"));
    }

    #[test]
    fn wave2_f32_f64_arithmetic_remainder_casts_and_yield_hoisting() {
        let src = r#"
watch<f32> pid_out_f32 = 0.0f32;
watch<f64> rem_out_f64 = 0.0;
watch<i32> cast_out_i32 = 0;

f32 clamp_f32(f32 x, f32 lo, f32 hi) {
    if (x < lo) { return lo; }
    if (x > hi) { return hi; }
    return x;
}

task pid_worker() {
    f32 integral = 1.25f32;
    f64 accum64 = 10.5;
    u32 step = 0;

    loop every 1ms {
        f32 err = (f32)(step + 1) * 1.5f32;
        integral += err * 0.5f32;
        integral = clamp_f32(integral, -100.0f32, 100.0f32);

        f32 mod32 = integral % 1.0f32;
        f64 wide = (f64)integral + accum64;
        wide %= 3.25;
        f64 neg_rem = (-7.5) % 2.0;

        delay(1ms);

        pid_out_f32.publish(integral + mod32);
        rem_out_f64.publish(wide + neg_rem);
        cast_out_i32.publish((i32)(integral * 4.0f32));

        step += 1;
        if (step >= 2) {
            sys.exit(0);
        }
    }
}

fn main() {
    spawn pid_worker() @critical;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse wave2 floats");
        let mir = zev_mir::lower_program(&prog).expect("lower wave2 floats");

        // Verify `integral` (f32) and `wide` (f64) are hoisted across `delay(1ms)`!
        let bp = mir
            .task_blueprints
            .iter()
            .find(|t| t.name == "pid_worker")
            .expect("pid_worker blueprint");
        assert!(
            bp.vars
                .iter()
                .any(|v| v.name == "integral" && v.ty == zev_mir::MirType::F32 && v.hoisted),
            "f32 `integral` must be hoisted across yield"
        );
        assert!(
            bp.vars
                .iter()
                .any(|v| v.name == "wide" && v.ty == zev_mir::MirType::F64 && v.hoisted),
            "f64 `wide` must be hoisted across yield"
        );

        let c_code = emit_c(&mir);
        assert!(
            !c_code.contains("<math.h>"),
            "Freestanding C99 float support must not depend on <math.h> / -lm"
        );
        assert_freestanding_compiles("wave2_floats", &c_code);

        let driver = r#"
int main(void) {
    syn_boot_init();
    syn_sim_advance_ms(10);

    float pid = syn_watch_f32_read(&pid_out_f32);
    double rem = syn_watch_f64_read(&rem_out_f64);
    int32_t ci = syn_watch_i32_read(&cast_out_i32);

    /* step 0: err=1.5 -> integral=1.25+0.75=2.0
       step 1: err=3.0 -> integral=2.0+1.5=3.5, mod32=3.5%1.0=0.5 -> pid=4.0f
       wide = (3.5 + 10.5) % 3.25 = 14.0 % 3.25 = 1.0; neg_rem = (-7.5) % 2.0 = -1.5 -> rem = -0.5
       cast_out_i32 = (int32_t)(3.5f * 4.0f) = 14 */
    if (pid < 3.999f || pid > 4.001f) {
        fprintf(stderr, "Expected pid == 4.0, got %.6f\n", (double)pid);
        return 1;
    }
    if (rem < -0.5001 || rem > -0.4999) {
        fprintf(stderr, "Expected rem == -0.5, got %.6f\n", rem);
        return 2;
    }
    if (ci != 14) {
        fprintf(stderr, "Expected cast_out_i32 == 14, got %d\n", ci);
        return 3;
    }
    printf("WAVE2_FLOATS_OK pid=%.2f rem=%.2f ci=%d\n", (double)pid, rem, ci);
    return 0;
}
"#;
        let stdout = compile_and_run_clang_sanitized("wave2_floats", &c_code, Some(driver));
        assert!(stdout.contains("WAVE2_FLOATS_OK"), "stdout: {stdout}");
    }

    #[test]
    fn wave2_zero_heap_format_strings_arduino_floats_sci_debug_and_enums() {
        // 1. Verify `:e` formatter is omitted when `:e` is NOT used, and unformatted enums omit lookup tables!
        let src_no_sci = r#"
enum Mode { Idle, Armed, Fault }
enum UnformattedSecret { HiddenA, HiddenB }

fn main() {
    Mode m = Mode::Armed;
    f32 v = 3.14159f32;
    println(f"mode={m} v={v}");
}
"#;
        let prog_no_sci = zev_parser::parse_str(src_no_sci).expect("parse no_sci");
        let mir_no_sci = zev_mir::lower_program(&prog_no_sci).expect("lower no_sci");
        assert_eq!(mir_no_sci.formatted_enums, vec!["Mode".to_string()]);
        let c_no_sci = emit_c(&mir_no_sci);
        assert!(
            !c_no_sci.contains("syn_fmt_float_sci"),
            "syn_fmt_float_sci must be omitted when `:e` is unused"
        );
        assert!(c_no_sci.contains("syn_enum_name_Mode"));
        assert!(
            !c_no_sci.contains("syn_enum_name_UnformattedSecret"),
            "Unformatted enum must NOT emit `.rodata` name table"
        );
        assert_freestanding_compiles("wave2_fmt_no_sci", &c_no_sci);

        // 2. Full live execution of fixed floats, Arduino edge cases, `:e`, hex/bin/pad, `{pkt:?}`, `{arr:?}`, and enum names!
        let src = r#"
enum Mode { Idle, Armed, Fault }

struct Vec2 {
    f32 x;
    f32 y;
}

struct TelemetryPkt {
    u16 id;
    Mode mode;
    Vec2 pos;
}

fn main() {
    f64 huge = 1.0e200;
    f64 pos_inf = huge * huge;
    f64 neg_inf = -pos_inf;
    f64 nan_val = pos_inf - pos_inf;
    f64 big_ovf = 1.0e12;
    f64 neg_ovf = -1.0e12;
    f64 neg_zero = -0.0;

    println(f"FIXED:{3.14159}:{2.5:.0}:{-2.5:.0}:{1.996:.2}:{neg_zero}");
    println(f"EDGE:{nan_val}:{pos_inf}:{neg_inf}:{big_ovf}:{neg_ovf}");
    println(f"SCI:{1234.5:e}:{-0.0045:.3e}:{9.999:.2e}:{0.0:e}");

    u16 reg = 0x00AB;
    i8 neg_byte = -1;
    u8 bits = 0b0010_1101;
    bool ok = true;
    println(f"INT:0x{reg:04x}:0x{reg:04X}:0x{neg_byte:02x}:0b{bits:08b}:{ok}:{!ok}");

    TelemetryPkt pkt = TelemetryPkt {
        id: 7,
        mode: Mode::Armed,
        pos: Vec2 { x: 1.5f32, y: -2.25f32 },
    };
    [3]i32 samples = [-10, 0, 25];
    println(f"DBG:{pkt:?} arr={samples:?}");
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse wave2 fmt");
        let mir = zev_mir::lower_program(&prog).expect("lower wave2 fmt");
        let c_code = emit_c(&mir);
        assert!(c_code.contains("syn_fmt_float_sci"));
        assert_freestanding_compiles("wave2_fmt_full", &c_code);

        let stdout = compile_and_run_clang_sanitized("wave2_fmt_full", &c_code, None);
        assert!(
            stdout.contains("FIXED:3.14:3:-3:2.00:0.00\n"),
            "Unexpected FIXED line in stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("EDGE:nan:inf:-inf:ovf:-ovf\n"),
            "Unexpected EDGE line in stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("SCI:1.23e+03:-4.500e-03:1.00e+01:0.00e+00\n"),
            "Unexpected SCI line in stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("INT:0x00ab:0x00AB:0xff:0b00101101:true:false\n"),
            "Unexpected INT line in stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("DBG:TelemetryPkt { id: 7, mode: Armed, pos: Vec2 { x: 1.50, y: -2.25 } } arr=[-10, 0, 25]\n"),
            "Unexpected DBG line in stdout:\n{stdout}"
        );
    }

    #[test]
    fn wave2_fmt_buf_exact_fit_yield_hoisting_and_clamped_truncation() {
        let src = r#"
enum Mode { Idle, Active }

task logger() {
    [32]u8 ok_buf;
    Mode m = Mode::Active;
    f32 v = 12.34f32;
    []u8 msg = fmt_buf(ok_buf, f"m={m} v={v:.1}") catch {
        sys.exit(2);
    };

    // Hold `msg` ([]u8 slice into `ok_buf`) live across a yield!
    // Liveness alias tracking MUST hoist `ok_buf` into `_f` so ASan sees zero stack-use-after-return!
    delay(1ms);
    println(msg);

    // Overflow test into a 6-byte dynamic subslice: must trigger FmtErr::Truncated AND clamp partial slice!
    [16]u8 small_backing;
    []u8 small = small_backing[0..6];
    []u8 res = fmt_buf(small, f"voltage={v:.2}") catch (FmtErr err, []u8 clamped) {
        println(f"ERR:{err}:{clamped}:{clamped.len}");
        sys.exit(0);
    };
    println(res);
    sys.exit(3);
    loop {
        delay(1s);
    }
}

fn main() {
    spawn logger() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse fmt_buf test");
        let mir = zev_mir::lower_program(&prog).expect("lower fmt_buf test");

        let bp = mir
            .task_blueprints
            .iter()
            .find(|t| t.name == "logger")
            .expect("logger blueprint");
        assert!(
            bp.vars.iter().any(|v| v.name == "ok_buf" && v.hoisted),
            "`ok_buf` backing `msg` slice across `delay(1ms)` MUST be hoisted into task frame!"
        );

        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave2_fmt_buf", &c_code);

        let stdout = compile_and_run_clang_sanitized("wave2_fmt_buf", &c_code, None);
        assert!(
            stdout.contains("m=Active v=12.3\n"),
            "Expected exact-fit post-yield slice output, got:\n{stdout}"
        );
        assert!(
            stdout.contains("ERR:Truncated:voltag:6\n"),
            "Expected clamped 6-byte Truncated partial slice, got:\n{stdout}"
        );
    }

    #[test]
    fn wave2_monomorphized_metadata_ram_report_and_unmonomorphized_skip() {
        let sp = Span::dummy();
        let mono_attr = |sig: &str| Attr {
            kind: AttrKind::Custom {
                name: "monomorphized".to_string(),
                arg: Some(sig.to_string()),
            },
            span: sp,
        };

        let mut prog = zev_parser::parse_str(
            r#"
// Unmonomorphized templates (defensively skipped if passed directly to zev-mir)
struct FixedVec<T, usize N> {
    [N]T data;
    usize len;
}

T clamp<T>(T x, T lo, T hi) {
    if (x < lo) { return lo; }
    if (x > hi) { return hi; }
    return x;
}

struct FixedVec__f32_4 {
    [4]f32 data;
    usize len;
}

f32 clamp__f32(f32 x, f32 lo, f32 hi) {
    if (x < lo) { return lo; }
    if (x > hi) { return hi; }
    return x;
}

fn main() {
    FixedVec__f32_4 v = FixedVec__f32_4 {
        data: [1.0f32, 2.0f32, 3.0f32, 4.0f32],
        len: 4,
    };
    f32 c = clamp__f32(v.data[3] * 3.0f32, 0.0f32, 9.5f32);
    println(f"MONO:len={v.len} c={c:.2}");
}
"#,
        )
        .expect("parse mono test");

        // Attach `@monomorphized(...)` metadata attributes just as `zev-sema` monomorphizer stamps them
        for item in &mut prog.items {
            match &item.kind {
                ItemKind::Struct(sd) if sd.name.name == "FixedVec__f32_4" => {
                    item.attrs.push(mono_attr("FixedVec<f32, 4>"));
                }
                ItemKind::Fn(fd) if fd.name.name == "clamp__f32" => {
                    item.attrs.push(mono_attr("clamp<f32>"));
                }
                _ => {}
            }
        }

        let mir = zev_mir::lower_program(&prog).expect("lower mono test");
        assert_eq!(mir.monomorphized_instances.len(), 2);
        assert_eq!(mir.ram_report.monomorphized_instances.len(), 2);

        let st_inst = mir
            .ram_report
            .monomorphized_instances
            .iter()
            .find(|m| m.mangled_name == "FixedVec__f32_4")
            .expect("FixedVec__f32_4 metadata");
        assert_eq!(st_inst.signature, "FixedVec<f32, 4>");
        assert_eq!(st_inst.size_32, 20); // 4*4 + 4
        assert_eq!(st_inst.size_64, 24); // 4*4 + 8

        let table = mir.ram_report.format_table();
        assert!(
            table.contains("7. Monomorphized Generic Instantiations")
                && table.contains("FixedVec<f32, 4>")
                && table.contains("clamp<f32>"),
            "Expected Section 7 in RamReport table:\n{table}"
        );

        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave2_mono", &c_code);
        let stdout = compile_and_run_clang_sanitized("wave2_mono", &c_code, None);
        assert!(stdout.contains("MONO:len=4 c=9.50\n"), "stdout:\n{stdout}");
    }

    #[test]
    fn wave2_bare_fmt_buf_proven_fit_without_or_or_catch() {
        let src = r#"
usize slice_len_helper([]u8 s) {
    return s.len;
}

fn direct_bare_check(u32 x) {
    [64]u8 b1;
    []u8 s1 = fmt_buf(b1, f"x={x}");
    println(s1);
    println(fmt_buf(b1, f"inline={x:04x}"));
    usize n = slice_len_helper(fmt_buf(b1, f"arg={x}"));
    usize m = fmt_buf(b1, f"len={x}").len;
    println(f"lens={n}:{m}");
}

task bare_task() {
    [64]u8 tbuf;
    f32 v = 7.25f32;
    []u8 s = fmt_buf(tbuf, f"task_v={v:.2}");
    delay(1ms);
    println(s);

    [64]u8 tbuf2;
    s = fmt_buf(tbuf2, f"reassigned={v:.1}");
    println(s);

    [4]u8 tiny;
    [8]u8 fb = [79, 86, 70, 33, 0, 0, 0, 0];
    []u8 with_or = fmt_buf(tiny[0..4], f"overflow={v:.2}") or fb[0..4];
    println(with_or);

    sys.exit(0);
    loop {
        delay(1s);
    }
}

fn main() {
    direct_bare_check(42);
    spawn bare_task() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse bare fmt_buf");
        let mir = zev_mir::lower_program(&prog).expect("lower bare fmt_buf");

        let bp = mir
            .task_blueprints
            .iter()
            .find(|t| t.name == "bare_task")
            .expect("bare_task blueprint");
        assert!(
            bp.vars.iter().any(|v| v.name == "tbuf" && v.hoisted),
            "`tbuf` backing bare `fmt_buf` slice across `delay(1ms)` MUST be hoisted into `_f`!"
        );

        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave2_bare_fmt_buf", &c_code);
        let stdout = compile_and_run_clang_sanitized("wave2_bare_fmt_buf", &c_code, None);
        assert!(stdout.contains("x=42\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("inline=002a\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("lens=6:6\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("task_v=7.25\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("reassigned=7.3\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("OVF!\n"), "stdout:\n{stdout}");
    }

    #[test]
    fn wave2_topological_struct_and_array_dependency_ordering() {
        // Intentionally declare outer monomorphized/nested structs BEFORE inner structs AND
        // declare `LateSample` AFTER functions/structs that embed `[3]LateSample` and `[]LateSample`!
        let src = r#"
struct FixedVec__FixedVec__u8_2_2 {
    [2]FixedVec__u8_2 data;
    usize len;
}

struct OuterHolder {
    [2]MiddleNode nodes;
    LateSample direct_sample;
    OuterHolder *self_next;
}

struct MiddleNode {
    LateSample sample;
}

struct FixedVec__u8_2 {
    [2]u8 data;
    usize len;
}

struct LateSample {
    u16 code;
    f32 val;
}

u16 sum_late_slice([]LateSample sl) {
    u16 acc = 0;
    for (usize i in 0..sl.len) {
        acc += sl[i].code;
    }
    return acc;
}

fn main() {
    FixedVec__u8_2 r0 = FixedVec__u8_2 { data: [3, 4], len: 2 };
    FixedVec__u8_2 r1 = FixedVec__u8_2 { data: [5, 6], len: 2 };
    FixedVec__FixedVec__u8_2_2 mat = FixedVec__FixedVec__u8_2_2 {
        data: [r0, r1],
        len: 2,
    };

    LateSample s0 = LateSample { code: 10, val: 1.25f32 };
    LateSample s1 = LateSample { code: 20, val: 2.50f32 };
    LateSample s2 = LateSample { code: 30, val: 3.75f32 };
    [3]LateSample samples = [s0, s1, s2];
    []LateSample sub = samples[0..3][1..3];

    OuterHolder h0;
    OuterHolder h = OuterHolder {
        nodes: [MiddleNode { sample: s0 }, MiddleNode { sample: s2 }],
        direct_sample: s1,
        self_next: &h0,
    };

    u32 mat_sum = (u32)mat.data[0].data[1] + (u32)mat.data[1].data[0];
    u16 sub_sum = sum_late_slice(sub);
    println(f"TOPO:mat={mat_sum} sub={sub_sum} h={h.nodes[1].sample.val:.2}");
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse topo struct+array test");
        let mir = zev_mir::lower_program(&prog).expect("lower topo struct+array test");
        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave2_topo_structs_arrays", &c_code);
        let stdout = compile_and_run_clang_sanitized("wave2_topo_structs_arrays", &c_code, None);
        assert!(
            stdout.contains("TOPO:mat=9 sub=50 h=3.75\n"),
            "stdout:\n{stdout}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stage R2-B Cross-Review Regression Suite (H11, H12, H2, M3, H7, H9, M4, M5, M6, M15)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn wave2_r2b_h11_coroutine_buffer_hoisting_across_yields_asan() {
        let src = r#"
struct MsgView {
    []u8 body;
}

fn log_after_delay([]u8 msg) {
    delay(2ms);
    println(f"CHILD:{msg}");
}

task h11_worker() {
    u32 tag = 123;

    // 1. Slice passed as argument to yielding child function `log_after_delay(s)`
    [16]u8 buf1;
    []u8 s = fmt_buf(buf1, f"hello={tag}");
    log_after_delay(s);

    // 2. Backing array wrapped inside a struct literal `MsgView` live across `delay(2ms)`
    [16]u8 buf2;
    MsgView mv = MsgView {
        body: fmt_buf(buf2, f"wrap={tag + 333}"),
    };
    delay(2ms);
    println(f"VIEW:{mv.body}");

    // 3. Pointer taken to slice element (`&sl3[2]`) live across `delay(2ms)`
    [8]u8 buf3 = [10, 20, 30, 40, 50, 60, 70, 80];
    []u8 sl3 = buf3[1..6];
    u8 *elem_ptr = &sl3[2];
    delay(2ms);
    println(f"DEREF:{*elem_ptr}");

    sys.exit(0);
    loop {
        delay(1s);
    }
}

fn main() {
    spawn h11_worker() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse H11 test");
        let mir = zev_mir::lower_program(&prog).expect("lower H11 test");

        let bp = mir
            .task_blueprints
            .iter()
            .find(|t| t.name == "h11_worker")
            .expect("h11_worker blueprint");
        for bname in ["buf1", "buf2", "buf3"] {
            assert!(
                bp.vars.iter().any(|v| v.name == bname && v.hoisted),
                "H11 regression: `{bname}` MUST be hoisted into `_f` across yield points!"
            );
        }

        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave2_r2b_h11", &c_code);
        let stdout = compile_and_run_clang_sanitized("wave2_r2b_h11", &c_code, None);
        assert!(stdout.contains("CHILD:hello=123\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("VIEW:wrap=456\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("DEREF:40\n"), "stdout:\n{stdout}");
    }

    #[test]
    fn wave2_r2b_h12_fallible_or_chains_catch_and_from_bytes_fmt_buf() {
        let src = r#"
enum ChainErr {
    FirstErr = 1,
    SecondErr = 2,
}

packed struct WireHdr {
    u16be magic;
    u16le ver;
}

watch<u32> side_effects = 0;

[]u8 fallback_slice([]u8 fb) {
    side_effects.publish(side_effects.read() + 1);
    return fb;
}

u32 direct_step(u32 which) error ChainErr {
    if (which == 1) {
        error FirstErr, 101;
    }
    if (which == 2) {
        error SecondErr, 202;
    }
    return 777;
}

u32 coro_step(u32 which) error ChainErr {
    delay(1ms);
    if (which == 1) {
        error FirstErr, 111;
    }
    if (which == 2) {
        error SecondErr, 222;
    }
    return 888;
}

fn run_direct_h12() {
    [4]u8 s1;
    [4]u8 s2;
    [8]u8 ok_buf;
    [4]u8 fb = [70, 65, 76, 76];

    // Nested fallible `(a or b) or c` where BOTH `a` and `b` fail -> reaches `c` ("FALL")
    []u8 r1 = (fmt_buf(s1[0..4], f"overflow_one") or fmt_buf(s2[0..4], f"overflow_two")) or fallback_slice(fb[0..4]);
    println(f"D_OR1:{r1}:{side_effects.read()}");

    // Nested fallible `(a or b) or c` where `a` fails and `b` succeeds -> short-circuits `c`!
    []u8 r2 = (fmt_buf(s1[0..4], f"overflow_one") or fmt_buf(ok_buf[0..8], f"MID_OK")) or fallback_slice(fb[0..4]);
    println(f"D_OR2:{r2}:{side_effects.read()}");

    // `(a or b) catch (e, partial)` where BOTH fail -> catches `SecondErr` + `202`!
    u32 c1 = (direct_step(1) or direct_step(2)) catch (ChainErr e, u32 part) {
        println(f"D_CATCH:{e}:{part}");
        part + 1000
    };
    println(f"D_CVAL:{c1}");

    // `Struct.from_bytes(fmt_buf(...))` unwraps `.val` cleanly!
    [4]u8 hbuf;
    WireHdr hdr = WireHdr.from_bytes(fmt_buf(hbuf, f"\x12\x34\x05\x00")) or WireHdr { magic: 0, ver: 0 };
    println(f"D_HDR:0x{hdr.magic:04x}:{hdr.ver}");
}

task h12_task() {
    [4]u8 s1;
    [4]u8 s2;
    [4]u8 fb = [84, 70, 66, 33];
    []u8 tr = (fmt_buf(s1[0..4], f"overflow_one") or fmt_buf(s2[0..4], f"overflow_two")) or fb[0..4];
    println(f"T_OR:{tr}");

    u32 tc = (coro_step(1) or coro_step(2)) catch (ChainErr e, u32 part) {
        println(f"T_CATCH:{e}:{part}");
        part + 2000
    };
    println(f"T_CVAL:{tc}");

    sys.exit(0);
    loop {
        delay(1s);
    }
}

fn main() {
    run_direct_h12();
    spawn h12_task() @normal;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse H12 test");
        let mir = zev_mir::lower_program(&prog).expect("lower H12 test");
        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave2_r2b_h12", &c_code);
        let stdout = compile_and_run_clang_sanitized("wave2_r2b_h12", &c_code, None);
        assert!(stdout.contains("D_OR1:FALL:1\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("D_OR2:MID_OK:1\n"), "stdout:\n{stdout}");
        assert!(
            stdout.contains("D_CATCH:SecondErr:202\n") && stdout.contains("D_CVAL:1202\n"),
            "stdout:\n{stdout}"
        );
        assert!(stdout.contains("D_HDR:0x1234:5\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("T_OR:TFB!\n"), "stdout:\n{stdout}");
        assert!(
            stdout.contains("T_CATCH:SecondErr:222\n") && stdout.contains("T_CVAL:2222\n"),
            "stdout:\n{stdout}"
        );
    }

    #[test]
    fn wave2_r2b_h2_m3_pointer_hex_self_ref_struct_fixed_arrays_and_enum_slices() {
        let src = r#"
enum NavMode {
    Idle = 0,
    Cruise = 1,
    Rtl = 2,
}

struct Node {
    u32 val;
    Node *next;
}

struct SamplePt {
    u16 k;
    f32 v;
}

struct Batch {
    [2]SamplePt items;
}

fn main() {
    // 1. Self-referential struct with null & non-null pointers (zero stack overflow, zero null-deref!)
    Node n0 = Node { val: 7, next: 0 as *Node };
    Node n1 = Node { val: 9, next: &n0 };
    Node *null_np = 0 as *Node;
    println(f"N0:{n0:?}");
    println(f"N1:{n1:?}");
    println(f"NP:{null_np:?}");

    // 2. DebugSlice of pointers `[2]*u32` -> uniform `0x<lowercase hex>`
    u32 target = 42;
    [2]*u32 ptrs = [0 as *u32, &target];
    println(f"PTRS:{ptrs:?}");

    // 3. Enum ONLY formatted via `[3]NavMode` `{modes:?}` -> emits `syn_enum_name_NavMode`!
    [3]NavMode modes = [NavMode::Idle, NavMode::Cruise, NavMode::Rtl];
    println(f"MODES:{modes:?}");

    // 4. Fixed-size array of structs & 2D fixed array unrolled under `{:?}`!
    Batch b = Batch {
        items: [
            SamplePt { k: 1, v: 2.5f32 },
            SamplePt { k: 3, v: -1.25f32 },
        ],
    };
    [2][2]u16 grid = [[10, 20], [30, 40]];
    println(f"BATCH:{b:?}");
    println(f"GRID:{grid:?}");
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse H2/M3 test");
        let mir = zev_mir::lower_program(&prog).expect("lower H2/M3 test");
        let c_code = emit_c(&mir);
        assert!(c_code.contains("syn_enum_name_NavMode"));
        assert_freestanding_compiles("wave2_r2b_h2_m3", &c_code);

        let stdout = compile_and_run_clang_sanitized("wave2_r2b_h2_m3", &c_code, None);
        assert!(
            stdout.contains("N0:Node { val: 7, next: 0x0 }\n"),
            "stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("N1:Node { val: 9, next: 0x"),
            "stdout:\n{stdout}"
        );
        assert!(stdout.contains("NP:0x0\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("PTRS:[0x0, 0x"), "stdout:\n{stdout}");
        assert!(
            stdout.contains("MODES:[Idle, Cruise, Rtl]\n"),
            "stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("BATCH:Batch { items: [SamplePt { k: 1, v: 2.50 }, SamplePt { k: 3, v: -1.25 }] }\n"),
            "stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("GRID:[[10, 20], [30, 40]]\n"),
            "stdout:\n{stdout}"
        );
    }

    #[test]
    fn wave2_r2b_h7_h9_m4_m5_m6_m15_saturating_casts_raw_bytes_sci_overlap_and_fnptr() {
        let src = r#"
const f64 CF_A = 1.5;
const f64 CF_B = 2.5;
const bool C_LT = CF_A < CF_B;
const bool C_EQ = CF_A == CF_B;
const i32 C_I32 = 7.9 as i32;
const u32 C_U32_HI = 3000000000.0 as u32;

watch<i32> w_cast = (9.75 as i32);
watch<u32> g_hook_hits = 0;

extern "C" fn c_register_and_fire(fn() -> void cb);

struct BoxFn {
    fn(u32) -> u32 cb;
}

fn on_c_hook() {
    g_hook_hits.publish(g_hook_hits.read() + 10);
}

u32 triple_plus_one(u32 x) {
    return x * 3 + 1;
}

u32 apply_cb(fn(u32) -> u32 f, u32 v) {
    return f(v);
}

u32 sum_nested_slices([][]u32 rows) {
    return rows[0][1] + rows[1][0];
}

fn main() {
    f64 rt0 = sys.now_ms() as f64;
    f32 rt0_f32 = sys.now_ms() as f32;

    // H6 + H7 + H8: Saturating float-to-integer casts (runtime + compile-time consts, 0 UBSan!)
    u32 u32_rt = (3000000000.0 + rt0) as u32;
    u64 u64_rt = (10000000000000000000.0 + rt0) as u64;
    u32 u32_ct = (3000000000.0) as u32;
    i32 sat_hi = (1.0e20 + rt0) as i32;
    i32 sat_lo = (-1.0e20 + rt0) as i32;
    i32 sat_ct = (1.0e20) as i32;
    u8 sat_u8 = (300.0f32 + rt0_f32) as u8;
    i8 sat_i8 = (-200.0f32 + rt0_f32) as i8;
    u32 neg_u32 = (-1.5f32 + rt0_f32) as u32;
    f64 nan_v = rt0 / rt0;
    i32 nan_i32 = nan_v as i32;
    u64 nan_u64 = nan_v as u64;

    println(f"H6:{u32_rt}:{u64_rt}:{u32_ct}");
    println(f"H7:{sat_hi}:{sat_lo}:{sat_ct}:{sat_u8}:{sat_i8}:{neg_u32}:{nan_i32}:{nan_u64}");
    println(f"H8:{C_LT}:{C_EQ}:{C_I32}:{C_U32_HI}:{w_cast.read()}");

    // H9: Raw byte escapes `\x80..\xFF` + octal boundary safety (`\xFFabc`)
    [2]u8 raw_buf;
    []u8 raw_sl = fmt_buf(raw_buf, f"\xFF\xAA");
    [4]u8 cat_buf;
    []u8 cat_sl = fmt_buf(cat_buf, f"\xFFabc");
    println(f"H9:{raw_sl.len}:0x{raw_sl[0]:02x}:0x{raw_sl[1]:02x}:{cat_sl.len}:0x{cat_sl[0]:02x}:{cat_sl[1]}");

    // M4: Nested slices `[][]u32`
    [2]u32 r0 = [11, 22];
    [2]u32 r1 = [33, 44];
    [2][]u32 row_arr = [r0[0..2], r1[0..2]];
    [][]u32 mat = row_arr[0..2];
    println(f"M4:{sum_nested_slices(mat)}");

    // M5: Overlapping `fmt_buf` (`dst < src` and `dst > src` via `memmove`, 0 ASan `memcpy-param-overlap`!)
    [8]u8 ov1 = [65, 66, 67, 68, 69, 0, 0, 0];
    []u8 ov1_in = ov1[1..5];
    []u8 ov1_out = fmt_buf(ov1, f"{ov1_in}");
    [8]u8 ov2 = [65, 66, 67, 68, 0, 0, 0, 0];
    []u8 ov2_in = ov2[0..4];
    []u8 ov2_out = fmt_buf(ov2, f"x{ov2_in}");
    println(f"M5:{ov1_out}:{ov2_out}");

    // M6 + L-Fmt: Drift-free subnormal/tiny scientific formatting & negative-zero rounding suppression
    f64 d1 = 9.95e-306;
    f64 d2 = 1.25e-307;
    f64 nz_sci = -0.0;
    f64 tiny_neg = -0.0001;
    f64 subhalf_neg = -0.4;
    println(f"M6:{d1:.1e}:{d2:.1e}:{nz_sci:e}:{tiny_neg:.2}:{subhalf_neg:.0}");

    // M15: `FnPtr` in `extern "C"`, struct fields (`BoxFn`), and direct indirect calls!
    c_register_and_fire(on_c_hook);
    BoxFn bf = BoxFn { cb: triple_plus_one };
    u32 r_param = apply_cb(bf.cb, 10);
    u32 r_field = bf.cb(20);
    println(f"M15:{g_hook_hits.read()}:{r_param}:{r_field}");
}
"#;
        let driver_c = r#"
void c_register_and_fire(void (*cb)(void)) {
    if (cb != NULL) {
        cb();
    }
}

int main(void) {
    syn_boot_init();
    return g_syn_sim_exit_code;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse H7/H9/M4/M5/M6/M15 test");
        let mir = zev_mir::lower_program(&prog).expect("lower H7/H9/M4/M5/M6/M15 test");
        let c_code = emit_c(&mir);
        assert_freestanding_compiles("wave2_r2b_combined", &c_code);

        let stdout =
            compile_and_run_clang_sanitized("wave2_r2b_combined", &c_code, Some(driver_c));
        assert!(
            stdout.contains("H6:3000000000:10000000000000000000:3000000000\n"),
            "stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("H7:2147483647:-2147483648:2147483647:255:-128:0:0:0\n"),
            "stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("H8:true:false:7:3000000000:9\n"),
            "stdout:\n{stdout}"
        );
        assert!(
            stdout.contains("H9:2:0xff:0xaa:4:0xff:97\n"),
            "stdout:\n{stdout}"
        );
        assert!(stdout.contains("M4:55\n"), "stdout:\n{stdout}");
        assert!(stdout.contains("M5:BCDE:xxBCD\n"), "stdout:\n{stdout}");
        assert!(
            stdout.contains("M6:1.0e-305:1.3e-307:0.00e+00:0.00:0\n"),
            "stdout:\n{stdout}"
        );
        assert!(stdout.contains("M15:10:31:61\n"), "stdout:\n{stdout}");
    }

    fn w1_ptr(inner: TypeExpr) -> TypeExpr {
        TypeExpr::new(TypeKind::Ptr(Box::new(inner)), Span::dummy())
    }

    fn w1_arr(elem: TypeExpr, len: usize) -> TypeExpr {
        TypeExpr::new(
            TypeKind::Array {
                elem: Box::new(elem),
                len: ArrayLen::Fixed(len),
            },
            Span::dummy(),
        )
    }

    fn w1_field(target: Expr, field: &str) -> Expr {
        Expr::new(
            ExprKind::Field {
                target: Box::new(target),
                field: id(field),
            },
            Span::dummy(),
        )
    }

    fn w1_idx(target: Expr, index: Expr) -> Expr {
        Expr::new(
            ExprKind::Index {
                target: Box::new(target),
                index: Box::new(index),
            },
            Span::dummy(),
        )
    }

    fn w1_call(callee: &str, args: Vec<Expr>) -> Expr {
        Expr::new(
            ExprKind::Call {
                callee: Box::new(expr_id(callee)),
                generic_args: vec![],
                args: args.into_iter().map(CallArg::positional).collect(),
            },
            Span::dummy(),
        )
    }

    fn w1_method(receiver: Expr, method: &str, args: Vec<Expr>) -> Expr {
        Expr::new(
            ExprKind::MethodCall {
                receiver: Box::new(receiver),
                method: id(method),
                args: args.into_iter().map(CallArg::positional).collect(),
            },
            Span::dummy(),
        )
    }

    fn w1_addrof(inner: Expr) -> Expr {
        Expr::new(
            ExprKind::Unary {
                op: UnaryOp::AddrOf,
                expr: Box::new(inner),
            },
            Span::dummy(),
        )
    }

    fn w1_bin(op: BinaryOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(
            ExprKind::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
            Span::dummy(),
        )
    }

    fn w1_cast(inner: Expr, ty: TypeExpr) -> Expr {
        Expr::new(
            ExprKind::Cast {
                expr: Box::new(inner),
                ty,
            },
            Span::dummy(),
        )
    }

    fn w1_print_line(parts: Vec<FormatPart>) -> Stmt {
        let sp = Span::dummy();
        Stmt::new(
            StmtKind::Expr(w1_call(
                "println",
                vec![Expr::new(ExprKind::FormatString(parts), sp)],
            )),
            sp,
        )
    }

    #[test]
    fn wave1_c99_mmio_widths_arrays_clamp_inline_and_rmw_single_eval() {
        let sp = Span::dummy();
        let items = vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Mmio(MmioDecl {
                    name: id("UartRegs"),
                    kind: MmioDeclKind::Layout {
                        regs: vec![
                            MmioRegDecl {
                                access: MmioAccess::Rw,
                                ty: ty_prim(PrimType::U8),
                                name: id("ctrl"),
                                offset: expr_int(0x00),
                                span: sp,
                            },
                            MmioRegDecl {
                                access: MmioAccess::Rw,
                                ty: ty_prim(PrimType::U16),
                                name: id("sr"),
                                offset: expr_int(0x02),
                                span: sp,
                            },
                            MmioRegDecl {
                                access: MmioAccess::Rw,
                                ty: ty_prim(PrimType::U32),
                                name: id("cr"),
                                offset: expr_int(0x04),
                                span: sp,
                            },
                            MmioRegDecl {
                                access: MmioAccess::Rw,
                                ty: w1_arr(ty_prim(PrimType::U16), 4),
                                name: id("arr"),
                                offset: expr_int(0x08),
                                span: sp,
                            },
                            MmioRegDecl {
                                access: MmioAccess::W1c,
                                ty: ty_prim(PrimType::U32),
                                name: id("icr"),
                                offset: expr_int(0x10),
                                span: sp,
                            },
                        ],
                    },
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Mmio(MmioDecl {
                    name: id("UART1"),
                    kind: MmioDeclKind::Instance {
                        layout: ty_named("UartRegs"),
                        base_addr: expr_int(0x4001_1000),
                    },
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Mmio(MmioDecl {
                    name: id("SysCfg"),
                    kind: MmioDeclKind::Inline {
                        base_addr: expr_int(0x4000_0000),
                        regs: vec![MmioRegDecl {
                            access: MmioAccess::Rw,
                            ty: ty_prim(PrimType::U32),
                            name: id("mode"),
                            offset: expr_int(0x00),
                            span: sp,
                        }],
                    },
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("bump_idx"),
                    generic_params: vec![],
                    params: vec![Param {
                        ty: w1_ptr(ty_prim(PrimType::U32)),
                        name: id("calls"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Usize),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::Assign {
                                    target: Expr::new(
                                        ExprKind::Unary {
                                            op: UnaryOp::Deref,
                                            expr: Box::new(expr_id("calls")),
                                        },
                                        sp,
                                    ),
                                    op: AssignOp::AddAssign,
                                    value: expr_int(1),
                                },
                                sp,
                            ),
                            Stmt::new(StmtKind::Return(Some(expr_int(2))), sp),
                        ],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("exercise_mmio"),
                    generic_params: vec![],
                    params: vec![Param {
                        ty: w1_ptr(ty_named("UartRegs")),
                        name: id("r"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::U32),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("r"), "ctrl"),
                                    op: AssignOp::Assign,
                                    value: expr_int(0x5A),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("r"), "ctrl"),
                                    op: AssignOp::BitOrAssign,
                                    value: expr_int(0x05),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("r"), "sr"),
                                    op: AssignOp::Assign,
                                    value: expr_int(0x1234),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("r"), "sr"),
                                    op: AssignOp::AddAssign,
                                    value: expr_int(0x0010),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("r"), "cr"),
                                    op: AssignOp::Assign,
                                    value: expr_int(0xDEAD_BEE0),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("r"), "cr"),
                                    op: AssignOp::BitOrAssign,
                                    value: expr_int(0x0F),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("SysCfg"), "mode"),
                                    op: AssignOp::Assign,
                                    value: expr_int(42),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("r"), "icr"),
                                    op: AssignOp::Assign,
                                    value: expr_int(0xABCD),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("calls"),
                                    init: Some(expr_int(0)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_idx(w1_field(expr_id("r"), "arr"), expr_int(0)),
                                    op: AssignOp::Assign,
                                    value: expr_int(100),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_idx(w1_field(expr_id("r"), "arr"), expr_int(2)),
                                    op: AssignOp::Assign,
                                    value: expr_int(0x10),
                                },
                                sp,
                            ),
                            // Side-effecting index in compound RMW must run ONCE!
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_idx(
                                        w1_field(expr_id("r"), "arr"),
                                        w1_call("bump_idx", vec![w1_addrof(expr_id("calls"))]),
                                    ),
                                    op: AssignOp::BitOrAssign,
                                    value: expr_int(0x05),
                                },
                                sp,
                            ),
                            // Out-of-bounds runtime array index clamps safely to index 0 under host sim ASan!
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::Usize),
                                    name: id("oob"),
                                    init: Some(expr_int(99)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_idx(w1_field(expr_id("r"), "arr"), expr_id("oob")),
                                    op: AssignOp::AddAssign,
                                    value: expr_int(7),
                                },
                                sp,
                            ),
                            w1_print_line(vec![
                                FormatPart::Literal("MMIO:".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_field(expr_id("r"), "ctrl")),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_field(expr_id("r"), "sr")),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":0x".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_field(expr_id("r"), "cr")),
                                    spec: FormatSpec::Hex {
                                        upper: true,
                                        width: Some(8),
                                    },
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_idx(
                                        w1_field(expr_id("r"), "arr"),
                                        expr_int(0),
                                    )),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_idx(
                                        w1_field(expr_id("r"), "arr"),
                                        expr_int(2),
                                    )),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_field(expr_id("SysCfg"), "mode")),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_field(expr_id("r"), "icr")),
                                    spec: FormatSpec::Hex {
                                        upper: false,
                                        width: Some(4),
                                    },
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(expr_id("calls")),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                            ]),
                            Stmt::new(StmtKind::Return(Some(expr_id("calls"))), sp),
                        ],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("main"),
                    generic_params: vec![],
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Expr(w1_call(
                                "exercise_mmio",
                                vec![w1_addrof(expr_id("UART1"))],
                            )),
                            sp,
                        )],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
        ];

        let mir = lower_program(&Program::new(items, sp)).expect("lower C99 MMIO test");
        let c_code = emit_c(&mir);
        assert!(
            c_code.contains("#define ZEV_MMIO_UART1 ((volatile uint8_t *)(uintptr_t)0x40011000UL)"),
            "missing freestanding macro:\n{c_code}"
        );
        assert!(
            c_code.contains("static SYN_UNUSED SYN_DMA_ALIGN volatile uint8_t __zev_mmio_buf_UART1[20];"),
            "missing sim buffer:\n{c_code}"
        );

        assert_freestanding_compiles("wave1_mmio", &c_code);
        let stdout = compile_and_run_clang_sanitized("wave1_mmio", &c_code, None);
        // ctrl=0x5F=95, sr=0x1244=4676, cr=0xDEADBEEF, arr[0]=107, arr[2]=0x15=21, mode=42, icr=abcd, calls=1
        assert!(
            stdout.contains("MMIO:95:4676:0xDEADBEEF:107:21:42:abcd:1\n"),
            "stdout:\n{stdout}"
        );
    }

    #[test]
    fn wave1_c99_parameterized_isrs_isolated_state_shared_and_conditional_boot() {
        let sp = Span::dummy();
        let watch_u32_item = |name: &str| Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                attrs: vec![],
                ty: OsPrimitiveType::Watch {
                    elem: Box::new(ty_prim(PrimType::U32)),
                },
                name: id(name),
                init: Some(expr_int(0)),
                span: sp,
            }),
            span: sp,
        };

        let items = vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Struct(StructDecl {
                    is_packed: false,
                    name: id("AccState"),
                    generic_params: vec![],
                    fields: vec![
                        StructField {
                            attrs: vec![],
                            ty: ty_prim(PrimType::U32),
                            name: id("total"),
                            span: sp,
                        },
                        StructField {
                            attrs: vec![],
                            ty: ty_prim(PrimType::U16),
                            name: id("hits"),
                            span: sp,
                        },
                    ],
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Mmio(MmioDecl {
                    name: id("PortRegs"),
                    kind: MmioDeclKind::Layout {
                        regs: vec![MmioRegDecl {
                            access: MmioAccess::Rw,
                            ty: ty_prim(PrimType::U32),
                            name: id("dr"),
                            offset: expr_int(0x00),
                            span: sp,
                        }],
                    },
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Mmio(MmioDecl {
                    name: id("PORTA"),
                    kind: MmioDeclKind::Instance {
                        layout: ty_named("PortRegs"),
                        base_addr: expr_int(0x4800_0000),
                    },
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Mmio(MmioDecl {
                    name: id("PORTB"),
                    kind: MmioDeclKind::Instance {
                        layout: ty_named("PortRegs"),
                        base_addr: expr_int(0x4800_0400),
                    },
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Irq(IrqDecl {
                    name: id("IRQ_PORTA"),
                    number: expr_int(10),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Irq(IrqDecl {
                    name: id("IRQ_PORTB"),
                    number: expr_int(11),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Irq(IrqDecl {
                    name: id("IRQ_SHARED"),
                    number: expr_int(20),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Irq(IrqDecl {
                    name: id("IRQ_COND_TRUE"),
                    number: expr_int(30),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Irq(IrqDecl {
                    name: id("IRQ_COND_UNSET"),
                    number: expr_int(31),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Irq(IrqDecl {
                    name: id("IRQ_SW"),
                    number: expr_int(32),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                    attrs: vec![],
                    ty: OsPrimitiveType::Inbox {
                        elem: Box::new(ty_prim(PrimType::U32)),
                        cap: 8,
                    },
                    name: id("out_q"),
                    init: None,
                    span: sp,
                }),
                span: sp,
            },
            watch_u32_item("res_a"),
            watch_u32_item("res_b"),
            watch_u32_item("res_mode"),
            watch_u32_item("res_sw"),
            // isr port_isr(*PortRegs p, *inbox<u32, 8> q, u32 tag)
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Isr(IsrDecl {
                    attrs: vec![],
                    vector: id("port_isr"),
                    params: vec![
                        Param {
                            ty: w1_ptr(ty_named("PortRegs")),
                            name: id("p"),
                            span: sp,
                        },
                        Param {
                            ty: w1_ptr(TypeExpr::new(
                                TypeKind::OsPrimitive(OsPrimitiveType::Inbox {
                                    elem: Box::new(ty_prim(PrimType::U32)),
                                    cap: 8,
                                }),
                                sp,
                            )),
                            name: id("q"),
                            span: sp,
                        },
                        Param {
                            ty: ty_prim(PrimType::U32),
                            name: id("tag"),
                            span: sp,
                        },
                    ],
                    body: Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::StateVar {
                                    ty: ty_named("AccState"),
                                    name: id("acc"),
                                    init: Some(Expr::new(
                                        ExprKind::StructLit {
                                            module: None,
                                            name: id("AccState"),
                                            generic_args: vec![],
                                            fields: vec![
                                                FieldInit {
                                                    name: id("total"),
                                                    value: expr_int(100),
                                                    span: sp,
                                                },
                                                FieldInit {
                                                    name: id("hits"),
                                                    value: expr_int(0),
                                                    span: sp,
                                                },
                                            ],
                                        },
                                        sp,
                                    )),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::StateVar {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("plain"),
                                    init: Some(expr_int(1)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: w1_ptr(ty_prim(PrimType::U32)),
                                    name: id("pp"),
                                    init: Some(w1_addrof(expr_id("plain"))),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: Expr::new(
                                        ExprKind::Unary {
                                            op: UnaryOp::Deref,
                                            expr: Box::new(expr_id("pp")),
                                        },
                                        sp,
                                    ),
                                    op: AssignOp::AddAssign,
                                    value: expr_int(1),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("acc"), "hits"),
                                    op: AssignOp::AddAssign,
                                    value: expr_int(1),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("acc"), "total"),
                                    op: AssignOp::AddAssign,
                                    value: expr_id("tag"),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: w1_field(expr_id("p"), "dr"),
                                    op: AssignOp::Assign,
                                    value: w1_field(expr_id("acc"), "total"),
                                },
                                sp,
                            ),
                            // q.post(tag * 1000 + (acc.hits as u32) * 10 + plain)
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("q"),
                                    "post",
                                    vec![w1_bin(
                                        BinaryOp::Add,
                                        w1_bin(BinaryOp::Mul, expr_id("tag"), expr_int(1000)),
                                        w1_bin(
                                            BinaryOp::Add,
                                            w1_bin(
                                                BinaryOp::Mul,
                                                w1_cast(
                                                    w1_field(expr_id("acc"), "hits"),
                                                    ty_prim(PrimType::U32),
                                                ),
                                                expr_int(10),
                                            ),
                                            expr_id("plain"),
                                        ),
                                    )],
                                )),
                                sp,
                            ),
                        ],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            // isr shared_a(u32 v) / shared_b(u32 v) / mode_isr(u32 v) / sw_isr(u32 v)
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Isr(IsrDecl {
                    attrs: vec![],
                    vector: id("shared_a"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("v"),
                        span: sp,
                    }],
                    body: Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::StateVar {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("s"),
                                    init: Some(expr_int(0)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: expr_id("s"),
                                    op: AssignOp::AddAssign,
                                    value: expr_id("v"),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("res_a"),
                                    "set",
                                    vec![expr_id("s")],
                                )),
                                sp,
                            ),
                        ],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Isr(IsrDecl {
                    attrs: vec![],
                    vector: id("shared_b"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("v"),
                        span: sp,
                    }],
                    body: Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::StateVar {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("s"),
                                    init: Some(expr_int(10)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: expr_id("s"),
                                    op: AssignOp::AddAssign,
                                    value: expr_id("v"),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("res_b"),
                                    "set",
                                    vec![expr_id("s")],
                                )),
                                sp,
                            ),
                        ],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Isr(IsrDecl {
                    attrs: vec![],
                    vector: id("mode_isr"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("v"),
                        span: sp,
                    }],
                    body: Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::StateVar {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("s"),
                                    init: Some(expr_int(0)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: expr_id("s"),
                                    op: AssignOp::AddAssign,
                                    value: expr_id("v"),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("res_mode"),
                                    "set",
                                    vec![expr_id("s")],
                                )),
                                sp,
                            ),
                        ],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Isr(IsrDecl {
                    attrs: vec![],
                    vector: id("sw_isr"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("v"),
                        span: sp,
                    }],
                    body: Block::new(
                        vec![Stmt::new(
                            StmtKind::Expr(w1_method(
                                expr_id("res_sw"),
                                "set",
                                vec![expr_id("v")],
                            )),
                            sp,
                        )],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            // fn bind_port(*PortRegs p, irq vec, u32 tag) { vec.attach(port_isr(p, &out_q, tag)); }
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("bind_port"),
                    generic_params: vec![],
                    params: vec![
                        Param {
                            ty: w1_ptr(ty_named("PortRegs")),
                            name: id("p"),
                            span: sp,
                        },
                        Param {
                            ty: ty_named("irq"),
                            name: id("vec"),
                            span: sp,
                        },
                        Param {
                            ty: ty_prim(PrimType::U32),
                            name: id("tag"),
                            span: sp,
                        },
                    ],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Expr(w1_method(
                                expr_id("vec"),
                                "attach",
                                vec![w1_call(
                                    "port_isr",
                                    vec![expr_id("p"), w1_addrof(expr_id("out_q")), expr_id("tag")],
                                )],
                            )),
                            sp,
                        )],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            // fn main()
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("main"),
                    generic_params: vec![],
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::Expr(w1_call(
                                    "bind_port",
                                    vec![w1_addrof(expr_id("PORTA")), expr_id("IRQ_PORTA"), expr_int(5)],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_call(
                                    "bind_port",
                                    vec![expr_id("PORTB"), expr_id("IRQ_PORTB"), expr_int(9)],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("IRQ_SHARED"),
                                    "attach_shared",
                                    vec![w1_call("shared_a", vec![expr_int(3)])],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("IRQ_SHARED"),
                                    "attach_shared",
                                    vec![w1_call("shared_b", vec![expr_int(4)])],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::If {
                                    cond: Expr::new(ExprKind::Literal(Literal::Bool(true)), sp),
                                    then_block: Block::new(
                                        vec![Stmt::new(
                                            StmtKind::Expr(w1_method(
                                                expr_id("IRQ_COND_TRUE"),
                                                "attach",
                                                vec![w1_call("mode_isr", vec![expr_int(77)])],
                                            )),
                                            sp,
                                        )],
                                        sp,
                                    ),
                                    else_block: Some(Block::new(
                                        vec![Stmt::new(
                                            StmtKind::Expr(w1_method(
                                                expr_id("IRQ_COND_TRUE"),
                                                "attach",
                                                vec![w1_call("mode_isr", vec![expr_int(88)])],
                                            )),
                                            sp,
                                        )],
                                        sp,
                                    )),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::If {
                                    cond: Expr::new(ExprKind::Literal(Literal::Bool(false)), sp),
                                    then_block: Block::new(
                                        vec![Stmt::new(
                                            StmtKind::Expr(w1_method(
                                                expr_id("IRQ_COND_UNSET"),
                                                "attach",
                                                vec![w1_call("mode_isr", vec![expr_int(99)])],
                                            )),
                                            sp,
                                        )],
                                        sp,
                                    ),
                                    else_block: None,
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("sel"),
                                    init: Some(expr_int(2)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Switch {
                                    scrutinee: expr_id("sel"),
                                    arms: vec![
                                        SwitchArm {
                                            pat: SwitchPat::Literal(Literal::Int(1), sp),
                                            body: Block::new(
                                                vec![Stmt::new(
                                                    StmtKind::Expr(w1_method(
                                                        expr_id("IRQ_SW"),
                                                        "attach",
                                                        vec![w1_call("sw_isr", vec![expr_int(111)])],
                                                    )),
                                                    sp,
                                                )],
                                                sp,
                                            ),
                                            span: sp,
                                        },
                                        SwitchArm {
                                            pat: SwitchPat::Literal(Literal::Int(2), sp),
                                            body: Block::new(
                                                vec![Stmt::new(
                                                    StmtKind::Expr(w1_method(
                                                        expr_id("IRQ_SW"),
                                                        "attach",
                                                        vec![w1_call("sw_isr", vec![expr_int(222)])],
                                                    )),
                                                    sp,
                                                )],
                                                sp,
                                            ),
                                            span: sp,
                                        },
                                    ],
                                },
                                sp,
                            ),
                            // Pend IRQs!
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("sys"),
                                    "pend_irq",
                                    vec![expr_id("IRQ_PORTA")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("sys"),
                                    "pend_irq",
                                    vec![expr_id("IRQ_PORTA")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("sys"),
                                    "pend_irq",
                                    vec![expr_id("IRQ_PORTB")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("sys"),
                                    "pend_irq",
                                    vec![expr_id("IRQ_SHARED")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("sys"),
                                    "pend_irq",
                                    vec![expr_id("IRQ_SHARED")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("sys"),
                                    "pend_irq",
                                    vec![expr_id("IRQ_COND_TRUE")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("sys"),
                                    "pend_irq",
                                    vec![expr_id("IRQ_COND_UNSET")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(w1_method(
                                    expr_id("sys"),
                                    "pend_irq",
                                    vec![expr_id("IRQ_SW")],
                                )),
                                sp,
                            ),
                            // Drain 3 messages from out_q and print state!
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("m0"),
                                    init: Some(Expr::new(
                                        ExprKind::Or {
                                            expr: Box::new(w1_method(
                                                expr_id("out_q"),
                                                "try_recv",
                                                vec![],
                                            )),
                                            fallback: Box::new(expr_int(0)),
                                        },
                                        sp,
                                    )),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("m1"),
                                    init: Some(Expr::new(
                                        ExprKind::Or {
                                            expr: Box::new(w1_method(
                                                expr_id("out_q"),
                                                "try_recv",
                                                vec![],
                                            )),
                                            fallback: Box::new(expr_int(0)),
                                        },
                                        sp,
                                    )),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("m2"),
                                    init: Some(Expr::new(
                                        ExprKind::Or {
                                            expr: Box::new(w1_method(
                                                expr_id("out_q"),
                                                "try_recv",
                                                vec![],
                                            )),
                                            fallback: Box::new(expr_int(0)),
                                        },
                                        sp,
                                    )),
                                },
                                sp,
                            ),
                            w1_print_line(vec![
                                FormatPart::Literal("ISR:".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_field(expr_id("PORTA"), "dr")),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_field(expr_id("PORTB"), "dr")),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(expr_id("m0")),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(expr_id("m1")),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(expr_id("m2")),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_method(expr_id("res_a"), "read", vec![])),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_method(expr_id("res_b"), "read", vec![])),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_method(expr_id("res_mode"), "read", vec![])),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                                FormatPart::Literal(":".to_string()),
                                FormatPart::Hole {
                                    expr: Box::new(w1_method(expr_id("res_sw"), "read", vec![])),
                                    spec: FormatSpec::Default,
                                    span: sp,
                                },
                            ]),
                        ],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
        ];

        let mir = lower_program(&Program::new(items, sp)).expect("lower C99 ISR test");
        assert!(mir.ram_report.mcu32.isr_state_bytes > 0);
        assert!(mir.ram_report.format_table().contains("ISR Static State"));

        let c_code = emit_c(&mir);
        assert!(c_code.contains("static SYN_UNUSED void syn_isr_port_isr("));
        assert!(c_code.contains("static SYN_UNUSED void __zev_irq_vec_20(void)"));
        assert!(c_code.contains("static SYN_UNUSED uint8_t __zev_irq_sel_30 = 0U;"));

        assert_freestanding_compiles("wave1_isr_boot_attach", &c_code);
        let stdout = compile_and_run_clang_sanitized("wave1_isr_boot_attach", &c_code, None);
        // PORTA.dr=110 (100+5+5), PORTB.dr=109 (100+9),
        // m0=5012 (5*1000 + 1*10 + 2), m1=5023 (5*1000 + 2*10 + 3), m2=9012 (9*1000 + 1*10 + 2),
        // res_a=6 (3+3), res_b=18 (10+4+4), res_mode=77, res_sw=222
        assert!(
            stdout.contains("ISR:110:109:5012:5023:9012:6:18:77:222\n"),
            "stdout:\n{stdout}"
        );
    }

    #[test]
    fn byte_ring_spsc_wrap_bipbuffer_grant_wait_read_and_boost_asan() {
        let src = r#"
byte_ring<16> stream;
signal done_sig;

task consumer(*byte_ring<16> rx, *signal done) {
    u32 total = 0;
    loop {
        []u8 chunk = rx.wait_read(timeout: 20ms) catch {
            println("TIMEOUT");
            if (total == 36) {
                done.notify();
            }
            continue;
        };
        usize n = chunk.len;
        if (n > 0) {
            u32 prio = sys.current_prio();
            u8 first = chunk[0];
            println(f"CHUNK len={n} first={first} prio={prio}");
            total += n as u32;
            rx.consume(n);
        }
    }
}

fn main() {
    byte_ring<16> *r = &stream;
    // 1. Wrap-around push/pop/try_pop_byte
    usize w1 = r.push("0123456789AB");
    assert(w1 == 12);
    assert(r.len() == 12);
    assert(r.free_space() == 4);

    u8 tmp[6];
    usize p1 = r.pop(tmp);
    assert(p1 == 6);
    assert(tmp[0] == '0');
    assert(tmp[5] == '5');

    // 2. BipBuffer contiguous grant_write across wrap-around:
    //    back gap is 4 bytes (12..16), front space is 6 bytes (0..6) -> grant_write(5) wraps to index 0!
    []u8 g = r.grant_write(5) catch {
        println("ERR grant5");
        return;
    };
    g[0] = 'X';
    g[1] = 'Y';
    g[2] = 'Z';
    r.commit_write(3);
    assert(r.len() == 9);
    assert(r.free_space() == 3);

    // 3. Drain first segment (6789AB) -> frees front segment automatically -> FIFO order intact!
    u8 out_a[6];
    usize pa = r.pop(out_a);
    assert(pa == 6);
    assert(out_a[0] == '6');
    assert(out_a[5] == 'B');

    []u8 rs = r.read_slice();
    assert(rs.len == 3);
    assert(rs[0] == 'X');
    assert(rs[2] == 'Z');
    r.consume(1);

    u8 popped = r.try_pop_byte() or 0;
    assert(popped == 'Y');
    u8 popped2 = r.try_pop_byte() or 0;
    assert(popped2 == 'Z');
    assert(r.is_empty());

    // 4. Push exact 16 bytes + overflow rejection + clear()
    usize wfull = r.push("ABCDEFGHIJKLMNOP");
    assert(wfull == 16);
    assert(r.is_full());
    assert(!r.push_byte('Q'));
    r.clear();
    assert(r.is_empty());
    assert(r.free_space() == 16);

    spawn consumer(&stream, &done_sig) @low;

    // 5. Exercise wait_read + boost: @critical + consume across multiple chunks (= 36 bytes)
    usize s1 = stream.push("AAAAAAAAAA", boost: @critical);
    assert(s1 == 10);
    delay(5ms);
    usize s2 = stream.push("BBBBBBBBBBBB");
    assert(s2 == 12);
    delay(5ms);
    []u8 gw = stream.grant_write(6) catch {
        println("ERR gw");
        return;
    };
    for (usize i in 0..6) {
        gw[i] = 'C';
    }
    stream.commit_write(6);
    delay(5ms);
    usize s3 = stream.push("DDDDDDDD", boost: @high);
    assert(s3 == 8);
    done_sig.wait();
    println("[PASS] BYTE_RING C99 OK");
    sys.exit(0);
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse byte_ring C99 test");
        zev_sema::analyze_program(&prog).expect("sema byte_ring C99 test");
        let mir = lower_program(&prog).expect("lower byte_ring C99 test");
        assert!(mir.ram_report.format_table().contains("byte_ring<16>"));

        let c_code = emit_c(&mir);
        assert_freestanding_compiles("byte_ring_spsc_asan", &c_code);
        let stdout = compile_and_run_clang_sanitized("byte_ring_spsc_asan", &c_code, None);
        assert!(
            stdout.contains("CHUNK len=10 first=65 prio=0"),
            "expected critical boost on first chunk:\n{stdout}"
        );
        assert!(
            stdout.contains("CHUNK len=8 first=68 prio=1"),
            "expected high boost on D chunk:\n{stdout}"
        );
        assert!(
            stdout.contains("TIMEOUT") && stdout.contains("[PASS] BYTE_RING C99 OK"),
            "expected timeout and PASS marker:\n{stdout}"
        );
    }

    fn run_zev_c_sim_and_freestanding(
        test_name: &str,
        src: &str,
    ) -> (zev_mir::MirModule, String, String) {
        let prog = zev_parser::parse_str(src).unwrap_or_else(|e| panic!("parse {test_name}: {e:?}"));
        let analyzed = zev_sema::analyze_and_monomorphize_named_programs(
            &[(&prog, None)],
            &zev_sema::AnalyzeOptions::default(),
        );
        assert!(
            analyzed.diagnostics.is_empty(),
            "sema {test_name}: {:?}",
            analyzed.diagnostics
        );
        let mir = zev_mir::lower_programs(&analyzed.programs)
            .unwrap_or_else(|e| panic!("lower {test_name}: {e:?}"));
        let c_code = emit_c(&mir);
        assert_freestanding_compiles(test_name, &c_code);
        let stdout = compile_and_run_clang_sanitized(test_name, &c_code, None);
        (mir, c_code, stdout)
    }

    #[test]
    fn unsized_views_byte_ring_and_inbox_multi_cap_asan() {
        let src = r#"
enum PacketTag { Ping = 1, Pong = 2 }

struct ViewBundle {
    *byte_ring ring;
    *inbox<u32> q;
}

byte_ring<12> ring_a;
byte_ring<24> ring_b;
inbox<u32, 3> q_a;
inbox<u32, 6> q_b;

usize exercise_ring(*byte_ring r, u8 seed) {
    usize cap = r.cap();
    assert(r.is_empty());
    assert(!r.is_full());
    assert(r.free_space() == cap);
    assert(r.push("ABC", boost: @high) == 3);
    assert(r.push_byte(seed));
    assert(r.len() == 4);

    []u8 chunk = r.read_slice();
    assert(chunk.len == 4);
    assert(chunk[0] == 'A');
    assert(chunk[3] == seed);
    r.consume(2);
    assert(r.len() == 2);

    u8 popped[4];
    usize npop = r.pop(popped);
    assert(npop == 2);
    assert(popped[0] == 'C');
    assert(popped[1] == seed);
    u8 pop_sentinel = r.try_pop_byte() catch { 0xEE };
    assert(pop_sentinel == 0xEE);

    []u8 dma = r.grant_write(3) catch { return 0; };
    dma[0] = 'X';
    dma[1] = 'Y';
    dma[2] = 'Z';
    r.commit_write(2, boost: @low);
    assert(r.len() == 2);
    u8 b0 = r.try_pop_byte() catch { return 0; };
    assert(b0 == 'X');
    r.clear();
    assert(r.is_empty());
    return cap;
}

usize exercise_inbox(*inbox<u32> q) {
    usize cap = q.cap();
    assert(q.is_empty());
    assert(!q.is_full());
    q.post(10);
    assert(q.try_send(20, boost: @high));
    assert(q.try_send(30));
    assert(q.len() == 3);
    if (cap == 3) {
        assert(q.is_full());
        assert(!q.try_send(40));
    } else {
        assert(!q.is_full());
    }
    u32 v1 = q.try_recv() catch { return 0; };
    u32 v2 = q.try_recv() catch { return 0; };
    u32 v3 = q.try_recv() catch { return 0; };
    assert(v1 == 10 && v2 == 20 && v3 == 30);
    u32 empty_sentinel = q.try_recv() catch { 9999 };
    assert(empty_sentinel == 9999);
    return cap;
}

task consumer_task(ViewBundle vb) {
    set_priority(@low);
    u8 i = 0;
    while (i < 2) {
        i += 1;
        []u8 chunk = vb.ring.wait_read(timeout: 20ms) catch {
            println("RING_TIMEOUT");
            sys.exit(0);
        };
        println(f"VIEW_RX len={chunk.len} b0={chunk[0]} prio={sys.current_prio()}");
        vb.ring.consume(chunk.len);
        u32 msg = vb.q.recv(timeout: 20ms) catch {
            println("INBOX_TIMEOUT");
            sys.exit(0);
        };
        println(f"VIEW_MSG={msg} prio={sys.current_prio()}");
    }
    sys.exit(0);
}

task producer_task(*byte_ring r, *inbox<u32> q) {
    sys.delay(2ms);
    r.push("HI", boost: @critical);
    q.send(77, timeout: 10ms) catch { loop { sys.delay(10ms); } };
    sys.delay(3ms);
    r.push("OK", boost: @high);
    q.send(88, timeout: 10ms) catch { loop { sys.delay(10ms); } };
    loop { sys.delay(10ms); }
}

fn main() {
    assert(exercise_ring(&ring_a, 99) == 12);
    assert(exercise_ring(&ring_b, 42) == 24);
    assert(exercise_inbox(&q_a) == 3);
    assert(exercise_inbox(&q_b) == 6);
    spawn consumer_task(ViewBundle{ ring: &ring_a, q: &q_a }) @low;
    spawn producer_task(&ring_a, &q_a) @normal;
}
"#;
        let (_mir, _c, stdout) =
            run_zev_c_sim_and_freestanding("unsized_views_byte_ring_inbox_asan", src);
        assert!(
            stdout.contains("VIEW_RX len=2 b0=72 prio=0"),
            "expected critical boost on first wait_read:\n{stdout}"
        );
        assert!(
            stdout.contains("VIEW_MSG=77"),
            "expected inbox recv 77:\n{stdout}"
        );
        assert!(
            stdout.contains("VIEW_RX len=2 b0=79 prio=1"),
            "expected high boost on second wait_read:\n{stdout}"
        );
        assert!(
            stdout.contains("VIEW_MSG=88"),
            "expected inbox recv 88:\n{stdout}"
        );
    }

    #[test]
    fn subsystem_structs_nested_isr_and_view_coercion_asan() {
        let src = r#"
irq UartIrq = 0;

struct Channel {
    byte_ring<16> tx;
    inbox<u16, 4> q;
    signal ready;
}

struct UartSubsys {
    byte_ring<12> rx;
    Channel ch0;
}

UartSubsys uart0;
UartSubsys uart1;

isr on_rx(*byte_ring r, *inbox<u16> q) {
    r.push("AB", boost: @high);
    q.post(1234, boost: @high);
}

fn check_channel(*Channel ch, u16 tag) {
    assert(ch.tx.cap() == 16);
    assert(ch.q.cap() == 4);
    usize n = fmt_ring(&ch.tx, f"tag={tag}") catch { 0 };
    assert(n > 0);
}

task subsys_worker(*UartSubsys drv) {
    set_priority(@low);
    check_channel(&drv.ch0, 7);
    []u8 s = drv.rx.wait_read(timeout: 20ms) catch {
        println("SUBSYS_RX_TO");
        sys.exit(0);
    };
    println(f"SUBSYS_RX len={s.len} b0={s[0]} prio={sys.current_prio()}");
    drv.rx.consume(s.len);
    u16 code = drv.ch0.q.recv(timeout: 20ms) catch {
        println("SUBSYS_Q_TO");
        sys.exit(0);
    };
    println(f"SUBSYS_Q code={code}");
    []u8 txs = drv.ch0.tx.read_slice();
    println(f"SUBSYS_TX len={txs.len} b0={txs[0]}");
    drv.ch0.tx.consume(txs.len);
    sys.exit(0);
}

task irq_driver() {
    sys.delay(2ms);
    sys.pend_irq(0);
    loop { sys.delay(10ms); }
}

fn main() {
    check_channel(&uart1.ch0, 9);
    assert(uart1.ch0.tx.len() == 5);
    UartIrq.attach(on_rx(&uart0.rx, &uart0.ch0.q));
    spawn subsys_worker(&uart0) @low;
    spawn irq_driver() @normal;
}
"#;
        let (mir, _c, stdout) = run_zev_c_sim_and_freestanding("subsys_nested_isr_asan", src);
        let ram_tbl = mir.ram_report.format_table();
        assert!(
            ram_tbl.contains("uart0") && ram_tbl.contains("uart1"),
            "expected subsystem globals in RAM report:\n{ram_tbl}"
        );
        assert!(
            stdout.contains("SUBSYS_RX len=2 b0=65 prio=1"),
            "expected boosted SUBSYS_RX:\n{stdout}"
        );
        assert!(
            stdout.contains("SUBSYS_Q code=1234"),
            "expected SUBSYS_Q code=1234:\n{stdout}"
        );
        assert!(
            stdout.contains("SUBSYS_TX len=5 b0=116"),
            "expected SUBSYS_TX len=5 ('tag=7'):\n{stdout}"
        );
    }

    #[test]
    fn fmt_ring_wrap_single_wake_boost_and_truncation_asan() {
        let src = r#"
byte_ring<16> tx;
byte_ring<16> tx2;
signal sync;

usize write_fmt_view(*byte_ring r, u16 v) error FmtErr {
    return fmt_ring(r, f"V={v:04X}", boost: @high) catch (FmtErr e, usize pw) {
        error e, pw;
    };
}

task reader() {
    set_priority(@low);
    // 1. Wait on empty `tx` while `writer` calls multi-chunk `fmt_ring(..., boost: @high)`:
    //    must wake ONCE at end with all 8 bytes ("X=2A Y=7") and boosted prio=1.
    sync.notify();
    []u8 cw = tx.wait_read(timeout: 20ms) catch {
        println("FAIL_WAIT1");
        sys.exit(0);
    };
    u32 prio = sys.current_prio();
    usize cw_len = cw.len;
    u8 first_b = cw[0];
    assert(cw_len == 8);
    tx.consume(cw_len);
    println(f"FMT_WAKE len={cw_len} b0={first_b} prio={prio}");

    // 2. Wrap-around test on `tx2`: push 11 bytes then pop 10 -> head=10, tail=11, count=1 ('A').
    //    fmt_ring("V=1A2B", 6 bytes) wraps 5 bytes (11..16) + 1 byte (0..1)!
    assert(tx2.push("0123456789A") == 11);
    u8 trash[10];
    assert(tx2.pop(trash) == 10);
    usize w = write_fmt_view(&tx2, 0x1A2B) catch { 0 };
    assert(w == 6);

    []u8 c1 = tx2.read_slice();
    usize c1_len = c1.len;
    assert(c1_len == 6);
    tx2.consume(c1_len);
    []u8 c2 = tx2.read_slice();
    assert(c2.len == 1);
    u8 last_b = c2[0];
    tx2.consume(c2.len);
    println(f"FMT_WRAP c1len={c1_len} last={last_b}");

    // 3. Overflow test on `tx2`: prefill 12 bytes -> 4 free bytes left, then format 8 bytes -> Truncated with partial == 4
    assert(tx2.push("123456789012") == 12);
    usize partial_w = fmt_ring(&tx2, f"ABCD1234") catch (FmtErr e, usize pw) {
        println(f"FMT_TRUNC partial={pw} total_len={tx2.len()}");
        sys.exit(0);
    };
    println(f"UNEXPECTED_OK={partial_w}");
    sys.exit(0);
}

task writer() {
    sync.wait();
    sys.delay(2ms);
    u8 x = 0x2A;
    u8 y = 7;
    usize w = fmt_ring(&tx, f"X={x:02X} Y={y}", boost: @high) catch { 0 };
    assert(w == 8);
    loop { sys.delay(10ms); }
}

fn main() {
    spawn reader() @low;
    spawn writer() @normal;
}
"#;
        let (_mir, _c, stdout) =
            run_zev_c_sim_and_freestanding("fmt_ring_wrap_trunc_asan", src);
        assert!(
            stdout.contains("FMT_WAKE len=8 b0=88 prio=1"),
            "expected single-wake multi-chunk FMT_WAKE with @high boost:\n{stdout}"
        );
        assert!(
            stdout.contains("FMT_WRAP c1len=6 last=66"),
            "expected wrapped FMT_WRAP ('B'=66):\n{stdout}"
        );
        assert!(
            stdout.contains("FMT_TRUNC partial=4 total_len=16"),
            "expected Truncated partial=4 filling remaining ring space:\n{stdout}"
        );
    }

    #[test]
    fn pool_lease_dma_slot_defer_recycle_sleep_clamp_and_isr_release_asan() {
        let src = r#"
struct Frame {
    u32 id;
    [20]u8 payload;
}

struct SubBus {
    @dma_safe pool<u8, 2> dma_p;
    pool<u8, 2> cpu_p;
    dma_slot<u8> slot;
}

SubBus bus;
@dma_safe pool<Frame, 2> dma_pool;
dma_slot<Frame> tx_slot;
inbox<Lease<Frame>, 1> pkt_q;
signal done_sig;
watch<u32> phase = 0;

irq IRQ_TX = 3;

isr tx_done_isr(*dma_slot<Frame> slot, *pool<Frame, 2> p, *watch<u32> out_avail) {
    bool released = slot.release(boost: @high);
    assert(released);
    out_avail.publish(p.avail() as u32);
}

fn fill_and_send(*pool<Frame, 2> p, *inbox<Lease<Frame>> q, u32 id_val) {
    Lease<Frame> l = p.try_alloc() catch { return; };
    *Frame fp = l.get();
    fp.id = id_val;
    bool ok = q.try_send(l);
    assert(!ok);
}

u32 defer_early(bool early) {
    Lease<Frame> l = dma_pool.try_alloc() catch { return 0; };
    defer l.release();
    *Frame fp = l.get();
    fp.id = 42;
    if early {
        return fp.id + 1;
    }
    return fp.id;
}

u32 alloc_timeout_exhausted() {
    Lease<Frame> l = dma_pool.alloc(timeout: 1ms) catch {
        return 99;
    };
    l.release();
    return 0;
}

task waiter_task() {
    phase.publish(1);
    // All 2 slots are occupied -> this blocks until ISR releases tx_slot!
    Lease<Frame> l = dma_pool.alloc();
    assert(sys.current_prio() == 1);
    *Frame fp = l.get();
    u32 got = fp.id;
    l.release();
    println(f"ALLOC_WOKE id={got} depth={sys.sleep_depth()}");
    done_sig.notify();

    // Wait then drain two queued leases sent via blocking `pkt_q.send(...)`
    sys.delay(2ms);
    Lease<Frame> r1 = pkt_q.recv();
    assert(r1.get().id == 111);
    r1.release();
    Lease<Frame> r2 = pkt_q.recv();
    assert(r2.get().id == 222);
    r2.release();
    done_sig.notify();
    loop { sys.delay(50ms); }
}

task orchestrator() {
    // 1. Check initial Deep sleep & pool queries
    assert(dma_pool.cap() == 2);
    assert(dma_pool.avail() == 2);
    assert(dma_pool.in_use() == 0);
    assert(!dma_pool.is_exhausted());
    assert(sys.sleep_depth() == 2);

    // 2. @dma_safe cache-line 32B alignment & separation
    Lease<Frame> a0 = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    Lease<Frame> a1 = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    usize addr0 = a0.addr();
    usize addr1 = a1.addr();
    usize diff = 0;
    if (addr1 > addr0) { diff = addr1 - addr0; } else { diff = addr0 - addr1; }
    assert(addr0 % 32 == 0);
    assert(addr1 % 32 == 0);
    assert(diff >= 32);
    assert(dma_pool.is_exhausted());
    assert(dma_pool.in_use() == 2);
    assert(sys.sleep_depth() == 1);
    a0.release();
    a1.release();
    assert(dma_pool.avail() == 2);
    assert(sys.sleep_depth() == 2);

    // 3. Subsystem pools: non-@dma_safe stays DEEP (2), @dma_safe clamps LIGHT (1)
    Lease<u8> cu = bus.cpu_p.try_alloc() catch { assert(false); sys.exit(1); };
    assert(sys.sleep_depth() == 2);
    cu.release();
    Lease<u8> du = bus.dma_p.try_alloc() catch { assert(false); sys.exit(1); };
    assert(sys.sleep_depth() == 1);
    bus.slot.park(du) catch { assert(false); sys.exit(1); };
    assert(bus.slot.addr() != 0);
    assert(bus.slot.release());
    assert(bus.dma_p.avail() == 2);
    assert(sys.sleep_depth() == 2);

    // 4. defer cleanup + returning before release
    u32 dval = defer_early(true);
    assert(dval == 43);
    assert(dma_pool.avail() == 2);
    assert(sys.sleep_depth() == 2);

    // 5. Full inbox & busy dma_slot auto-recycle (0 leaks!)
    Lease<Frame> q0 = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    q0.get().id = 100;
    pkt_q.post(q0);
    // Fill remaining slot and try sending into full *inbox<Lease<Frame>> via try_send/post/timeout send
    fill_and_send(&dma_pool, &pkt_q, 999);
    assert(dma_pool.avail() == 1);
    Lease<Frame> q_drop = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    pkt_q.post(q_drop);
    assert(dma_pool.avail() == 1);
    Lease<Frame> q_to = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    bool timed_out = false;
    pkt_q.send(q_to, timeout: 1ms) catch {
        timed_out = true;
    };
    assert(timed_out);
    assert(dma_pool.avail() == 1);

    // Busy dma_slot.park auto-releases incoming lease and preserves first parked lease!
    Lease<Frame> p0 = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    p0.get().id = 77;
    tx_slot.park(p0) catch { assert(false); };
    assert(tx_slot.is_busy());
    assert(!tx_slot.is_empty());
    assert(dma_pool.is_exhausted());

    // Drain queued pkt_q item, allocate p1 to verify dma_slot.park fail auto-recycled without leak
    Lease<Frame> rx_q = pkt_q.try_recv() catch { assert(false); sys.exit(1); };
    assert(rx_q.get().id == 100);
    rx_q.release();
    assert(dma_pool.avail() == 1);
    Lease<Frame> p_dup = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    bool slot_busy_err = false;
    tx_slot.park(p_dup) catch {
        slot_busy_err = true;
    };
    assert(slot_busy_err);
    assert(dma_pool.avail() == 1);

    // Re-occupy slot 1 and let waiter_task block on exhausted dma_pool.alloc(), then trigger IRQ!
    Lease<Frame> hold1 = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    assert(dma_pool.is_exhausted());
    assert(alloc_timeout_exhausted() == 99);
    // Wait until waiter_task enters Poll on dma_pool.alloc()
    while phase.read() == 0 {
        sys.delay(1ms);
    }
    sys.delay(1ms);
    sys.pend_irq(IRQ_TX);
    done_sig.wait();
    // Note hold1 is still held until now -> depth 1, then release -> depth 2!
    assert(sys.sleep_depth() == 1);
    hold1.release();
    assert(dma_pool.avail() == 2);
    assert(sys.sleep_depth() == 2);

    // Blocking inbox<Lease<Frame>, 1>.send(lease) when full must NOT leak or double-release on poll retry!
    Lease<Frame> q1 = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    Lease<Frame> q2 = dma_pool.try_alloc() catch { assert(false); sys.exit(1); };
    q1.get().id = 111;
    q2.get().id = 222;
    pkt_q.send(q1);
    pkt_q.send(q2);
    done_sig.wait();
    assert(dma_pool.avail() == 2);
    assert(sys.sleep_depth() == 2);
    println("POOL_LEASE_E2E_OK");
    sys.exit(0);
}

fn main() {
    IRQ_TX.attach(tx_done_isr(&tx_slot, &dma_pool, &phase));
    spawn waiter_task() @normal;
    spawn orchestrator() @high;
}
"#;
        let (_mir, c_code, stdout) =
            run_zev_c_sim_and_freestanding("pool_lease_dma_slot_e2e_asan", src);
        assert!(stdout.contains("ALLOC_WOKE"), "stdout:\n{stdout}");
        assert!(stdout.contains("POOL_LEASE_E2E_OK"), "stdout:\n{stdout}");
        assert!(c_code.contains("__syn_pool_release"));
    }

    #[test]
    fn view_bundle_structs_status_cell_and_duplex_stream_asan() {
        let src = r#"
struct StatusCell<T> {
    *watch<T> state;
}

struct DuplexStream {
    *byte_ring rx;
    *byte_ring tx;
}

watch<u32> dev_status = 10;
byte_ring<16> uart_rx;
byte_ring<16> uart_tx;
signal synced;

fn update_status(StatusCell<u32> cell, u32 next_v) {
    assert(cell.state.read() == 10);
    cell.state.publish(next_v);
}

fn send_frame(DuplexStream io) {
    assert(io.tx.push("PING") == 4);
}

task reader(DuplexStream io, StatusCell<u32> cell) {
    []u8 chunk = io.rx.wait_read();
    u8 b0 = chunk[0];
    io.rx.consume(chunk.len);
    assert(cell.state.read() == 77);
    println(f"BUNDLE_OK b0={b0} st={cell.state.read()}");
    synced.notify();
    loop { sys.delay(50ms); }
}

task driver() {
    StatusCell<u32> c = StatusCell<u32> { state: &dev_status };
    DuplexStream s = DuplexStream { rx: &uart_rx, tx: &uart_tx };
    update_status(c, 77);
    send_frame(s);
    assert(uart_tx.len() == 4);
    assert(uart_rx.push("ZEV!") == 4);
    synced.wait();
    sys.exit(0);
}

fn main() {
    DuplexStream s = DuplexStream { rx: &uart_rx, tx: &uart_tx };
    StatusCell<u32> c = StatusCell<u32> { state: &dev_status };
    spawn reader(s, c) @normal;
    spawn driver() @low;
}
"#;
        let (_mir, _c, stdout) = run_zev_c_sim_and_freestanding("view_bundle_asan", src);
        assert!(
            stdout.contains("BUNDLE_OK b0=90 st=77"),
            "stdout:\n{stdout}"
        );
    }

    #[test]
    fn b_h2_subsystem_ping_pong_bits_signal_init_and_dma_safe_asan() {
        let src = r#"
struct InnerPool {
    pool<u8, 2> np;
}
struct DmaSub {
    InnerPool inner;
    pool<u8, 2> op;
    ping_pong<u8, 8> pp;
}
struct SigOnlySub {
    signal s;
}

ping_pong<u8, 8> top_pp;
@dma_safe DmaSub dsub;
DmaSub psub;
SigOnlySub sigsub;

task driver() {
    set_priority(@low);
    assert(sys.sleep_depth() == 2);

    // 1. Subsystem ping_pong bits must be distinct from top_pp and from each other!
    top_pp.start();
    dsub.pp.start();
    psub.pp.start();
    assert(sys.sleep_depth() == 1);
    top_pp.stop();
    assert(sys.sleep_depth() == 1);
    dsub.pp.stop();
    assert(sys.sleep_depth() == 1);
    psub.pp.stop();
    assert(sys.sleep_depth() == 2);

    // 2. SigOnlySub.s must initialize boost_prio to SYN_NO_BOOST (0xFF), NOT 0!
    sigsub.s.notify();
    sigsub.s.wait();
    assert(sys.current_prio() == 3);

    // 3. @dma_safe DmaSub dsub vs non-dma DmaSub psub sharing syn_struct_DmaSub:
    //    NO ASan global-buffer-overflow, dsub clamps sleep depth -> 1, psub keeps -> 2!
    Lease<u8> p0 = psub.op.try_alloc() catch { assert(false); sys.exit(1); };
    Lease<u8> p1 = psub.inner.np.try_alloc() catch { assert(false); sys.exit(1); };
    assert(p0.addr() % 32 == 0);
    assert(p1.addr() % 32 == 0);
    assert(sys.sleep_depth() == 2);
    p0.release();
    p1.release();

    Lease<u8> d0 = dsub.op.try_alloc() catch { assert(false); sys.exit(1); };
    assert(sys.sleep_depth() == 1);
    Lease<u8> d1 = dsub.inner.np.try_alloc() catch { assert(false); sys.exit(1); };
    assert(d0.addr() % 32 == 0);
    assert(d1.addr() % 32 == 0);
    assert(sys.sleep_depth() == 1);
    d0.release();
    assert(sys.sleep_depth() == 1);
    d1.release();
    assert(sys.sleep_depth() == 2);

    println("BH2_ALL_OK");
    sys.exit(0);
}

fn main() {
    spawn driver() @low;
}
"#;
        let (_mir, c_code, stdout) = run_zev_c_sim_and_freestanding("b_h2_subsys_dma_asan", src);
        assert!(stdout.contains("BH2_ALL_OK"), "stdout:\n{stdout}");
        assert!(c_code.contains("(1U << 0U)") && c_code.contains("(1U << 1U)") && c_code.contains("(1U << 2U)"));
        assert!(c_code.contains("SYN_NO_BOOST"));
        assert!(c_code.contains("static SYN_UNUSED SYN_DMA_ALIGN syn_struct_DmaSub __zev_subsys_dsub"));
    }

    #[test]
    fn b_h3_wait_read_helper_fn_and_main_tail_asan() {
        let src = r#"
byte_ring<16> r1;
byte_ring<16> r2;

usize read_once(*byte_ring r) {
    []u8 ch = r.wait_read();
    usize n = ch.len;
    assert(n > 0);
    r.consume(n);
    return n;
}

task sender() {
    sys.delay(1ms);
    assert(r1.push("ABCD") == 4);
    sys.delay(1ms);
    assert(r2.push("WXYZ!") == 5);
    loop { sys.delay(20ms); }
}

task helper_caller() {
    usize n = read_once(&r1);
    assert(n == 4);
    println(f"BH3_HELPER={n}");
    loop { sys.delay(20ms); }
}

fn main() {
    spawn sender() @normal;
    spawn helper_caller() @normal;
    []u8 mch = r2.wait_read();
    usize mn = mch.len;
    r2.consume(mn);
    assert(mn == 5);
    println(f"BH3_MAIN={mn}");
    sys.exit(0);
}
"#;
        let (_mir, _c, stdout) = run_zev_c_sim_and_freestanding("b_h3_wait_read_asan", src);
        assert!(stdout.contains("BH3_HELPER=4"), "stdout:\n{stdout}");
        assert!(stdout.contains("BH3_MAIN=5"), "stdout:\n{stdout}");
    }

    #[test]
    fn b_h4_boost_restored_across_signal_pool_loops_and_stolen_wakeups_asan() {
        let src = r#"
signal sig_a;
signal sig_b;
pool<u8, 1> p1;
inbox<u32, 4> q_stolen;
inbox<u32, 4> q_turn;
signal step_sig;
signal gate_sig;

fn delay_preserving_helper() {
    assert(sys.current_prio() == 0);
    sys.delay(1ms);
    assert(sys.current_prio() == 0);
}

task victim_sig_and_pool() {
    set_priority(@low);
    // 1. Back-to-back signal.wait(): first boosted @critical, second unboosted -> restored to @low (3)
    sig_a.wait();
    assert(sys.current_prio() == 0);
    sig_a.wait();
    assert(sys.current_prio() == 3);

    // 2. Back-to-back pool.alloc(): first boosted @critical, second unboosted -> restored to @low (3)
    Lease<u8> l1 = p1.alloc();
    assert(sys.current_prio() == 0);
    l1.release();
    sig_a.wait();
    Lease<u8> l2 = p1.alloc();
    assert(sys.current_prio() == 3);
    l2.release();

    // 3. Signal wakeup with boost followed by top-level non-receiving loop -> MUST restore to @low (3)
    sig_b.wait();
    assert(sys.current_prio() == 0);
    step_sig.notify();
    loop {
        assert(sys.current_prio() == 3);
        println("BH4_PRELOOP_RESTORED");
        break;
    }

    // 4. Signal wakeup followed by `loop every` -> MUST restore to @low (3)
    sig_b.wait();
    assert(sys.current_prio() == 0);
    loop every 2ms {
        assert(sys.current_prio() == 3);
        println("BH4_EVERY_RESTORED");
        break;
    }
    step_sig.notify();

    // 5. Active event turn via q_turn.recv() preserves boost across helper + delay, then restores on next iter
    for (u32 iter in 0..2) {
        u32 v = q_turn.recv();
        if (v == 1) {
            assert(sys.current_prio() == 0);
            delay_preserving_helper();
            assert(sys.current_prio() == 0);
        } else {
            assert(sys.current_prio() == 3);
        }
    }
    println("BH4_TURN_OK");
    step_sig.notify();
    loop { sys.delay(20ms); }
}

task victim_stolen_timeout() {
    set_priority(@low);
    gate_sig.wait();
    // Block on q_stolen with timeout; `thief` will steal the item after waking us with boost!
    // Timeout MUST restore prio back to @low (3).
    u32 r = q_stolen.recv(timeout: 3ms) or 999;
    assert(r == 999);
    assert(sys.current_prio() == 3);
    println("BH4_STOLEN_RESTORED");
    step_sig.notify();
    loop { sys.delay(20ms); }
}

task thief_and_driver() {
    // Hold p1 before sleeping so victim blocks cleanly on p1.alloc() after step 1
    Lease<u8> hold = p1.try_alloc() catch { assert(false); sys.exit(1); };
    sys.delay(1ms);
    // Drive step 1
    sig_a.notify(boost: @critical);
    sys.delay(1ms);
    sig_a.notify();
    sys.delay(1ms);

    // Drive step 2
    hold.release(boost: @critical);
    sys.delay(1ms);
    Lease<u8> hold2 = p1.try_alloc() catch { assert(false); sys.exit(1); };
    sig_a.notify();
    sys.delay(1ms);
    hold2.release();
    sys.delay(1ms);

    // Drive step 3
    sig_b.notify(boost: @critical);
    step_sig.wait();

    // Drive step 4
    sig_b.notify(boost: @critical);
    step_sig.wait();

    // Drive stolen wakeup: let victim arm q_stolen.recv(timeout: 3ms), post with @critical boost, then steal!
    gate_sig.notify();
    sys.delay(1ms);
    q_stolen.post(77, boost: @critical);
    u32 stolen = q_stolen.try_recv() catch { 0 };
    assert(stolen == 77);
    step_sig.wait();

    // Drive step 5
    q_turn.post(1, boost: @critical);
    sys.delay(2ms);
    q_turn.post(2);
    step_sig.wait();
    println("BH4_ALL_OK");
    sys.exit(0);
}

fn main() {
    spawn victim_sig_and_pool() @low;
    spawn victim_stolen_timeout() @low;
    spawn thief_and_driver() @normal;
}
"#;
        let (_mir, _c, stdout) = run_zev_c_sim_and_freestanding("b_h4_boost_asan", src);
        assert!(stdout.contains("BH4_PRELOOP_RESTORED"), "stdout:\n{stdout}");
        assert!(stdout.contains("BH4_EVERY_RESTORED"), "stdout:\n{stdout}");
        assert!(stdout.contains("BH4_STOLEN_RESTORED"), "stdout:\n{stdout}");
        assert!(stdout.contains("BH4_TURN_OK"), "stdout:\n{stdout}");
        assert!(stdout.contains("BH4_ALL_OK"), "stdout:\n{stdout}");
    }

    #[test]
    fn b_m1_pool_multi_waiter_priority_and_timeout_safety_asan() {
        let src = r#"
pool<u32, 1> shared_pool;
signal start_sig;
watch<u32> wake_order = 0;

task low_timeout_waiter() {
    set_priority(@low);
    // 1. Block on exhausted pool with short timeout while high_waiter is also blocked!
    bool timed_out = false;
    Lease<u32> l = shared_pool.alloc(timeout: 2ms) catch {
        timed_out = true;
        // Dummy unreachable path won't run if we return or exit, so handle directly:
        println("BM1_LOW_TO");
        loop { sys.delay(50ms); }
    };
    assert(!timed_out);
    l.release();
    loop { sys.delay(50ms); }
}

task low_perm_waiter() {
    set_priority(@low);
    start_sig.wait();
    // Blocks BEFORE high_perm_waiter; must wake SECOND!
    Lease<u32> l = shared_pool.alloc();
    u32 ord = wake_order.read() + 1;
    wake_order.publish(ord);
    assert(ord == 2);
    l.release();
    println(f"BM1_LOW_PERM={ord}");
    loop { sys.delay(50ms); }
}

task high_waiter() {
    set_priority(@high);
    // 1. Blocked alongside low_timeout_waiter; low_timeout_waiter times out at 2ms,
    //    cancel_wait MUST NOT clobber waiting state for high_waiter!
    Lease<u32> l1 = shared_pool.alloc();
    assert(sys.current_prio() == 0);
    println("BM1_HIGH_WOKE_1");
    l1.release();

    // 2. Delay past t=5 when pool_holder re-acquires h2 and wakes low_perm_waiter first
    sys.delay(2ms);
    Lease<u32> l2 = shared_pool.alloc();
    u32 ord = wake_order.read() + 1;
    wake_order.publish(ord);
    assert(ord == 1);
    println(f"BM1_HIGH_PERM={ord}");
    l2.release();
    sys.delay(2ms);
    assert(wake_order.read() == 2);
    println("BM1_ALL_OK");
    sys.exit(0);
}

task pool_holder() {
    Lease<u32> h1 = shared_pool.try_alloc() catch { assert(false); sys.exit(1); };
    // Let low_timeout_waiter time out at 2ms, then release at 4ms with @critical boost!
    sys.delay(4ms);
    h1.release(boost: @critical);

    // Wait 1ms for high_waiter to consume & release l1, then grab h2 and wake low_perm_waiter first
    sys.delay(1ms);
    Lease<u32> h2 = shared_pool.try_alloc() catch { assert(false); sys.exit(1); };
    start_sig.notify();
    // Both low_perm_waiter (at t=5) and high_waiter (at t=6) block; release at t=9 -> high_waiter wakes first!
    sys.delay(4ms);
    h2.release();
    loop { sys.delay(50ms); }
}

fn main() {
    spawn pool_holder() @critical;
    spawn low_timeout_waiter() @low;
    spawn high_waiter() @high;
    spawn low_perm_waiter() @low;
}
"#;
        let (_mir, _c, stdout) = run_zev_c_sim_and_freestanding("b_m1_pool_multi_waiter_asan", src);
        assert!(stdout.contains("BM1_LOW_TO"), "stdout:\n{stdout}");
        assert!(stdout.contains("BM1_HIGH_WOKE_1"), "stdout:\n{stdout}");
        assert!(stdout.contains("BM1_HIGH_PERM=1"), "stdout:\n{stdout}");
        assert!(stdout.contains("BM1_LOW_PERM=2"), "stdout:\n{stdout}");
        assert!(stdout.contains("BM1_ALL_OK"), "stdout:\n{stdout}");
    }

    #[test]
    fn b_m2_and_l_h1_sticky_truncation_and_mixed_sign_shr_asan() {
        let src = r#"
const u32 C_RSHIFT = (0x80000000 as u32) >> (4 as i32);
const i32 C_ARITH = (-16 as i32) >> (2 as u32);

byte_ring<16> ring;
irq IRQ_DRAIN = 4;
watch<u32> drain_stage = 0;
watch<u32> verified_len = 99;
watch<u8> verified_b8 = 0;
watch<u8> verified_b11 = 0;

u32 drain_mid_chunk() {
    sys.pend_irq(IRQ_DRAIN);
    return 1;
}

isr drain_isr(*byte_ring r) {
    if (drain_stage.read() == 0) {
        r.consume(4);
        drain_stage.publish(1);
    } else {
        u32 cur_len = r.len() as u32;
        verified_len.publish(cur_len);
        u8 out_b[12];
        assert(r.pop(out_b) == 12);
        // Expected contents: "456789AA1234" (NOT "456789AA12341ZZZZ")
        assert(out_b[6] == 65 && out_b[8] == 49 && out_b[11] == 52);
        verified_b8.publish(out_b[8]);
        verified_b11.publish(out_b[11]);
    }
}

fn main() {
    IRQ_DRAIN.attach(drain_isr(&ring));

    // L-H1 / B-M2: mixed-sign shifts (const + runtime)
    i32 rt_sh = 4;
    u32 rt_u = 0x80000000 as u32;
    u32 rt_res = rt_u >> rt_sh;
    u8 sh8 = 8;
    u32 shl_res = 1 << sh8;
    assert(C_RSHIFT == 0x08000000);
    assert(C_ARITH == -4);
    assert(rt_res == 0x08000000);
    assert(shl_res == 256);

    // B-M2: Multi-chunk fmt_ring where chunk 1 ("AA") fits, chunk 2 ("123456") truncates (fills remaining 4B),
    //       chunk 3 runs `drain_mid_chunk()` which drains 4B in an ISR, and chunk 4 ("ZZZZ") MUST NOT
    //       be written after dropped bytes because `s->truncated` is sticky!
    assert(ring.push("0123456789") == 10); // 6 bytes free
    bool got_err = false;
    usize pw_out = 0;
    usize w = fmt_ring(&ring, f"AA{123456}{drain_mid_chunk()}ZZZZ") catch (FmtErr e, usize pw) {
        got_err = true;
        pw_out = pw;
        0
    };
    assert(got_err);
    assert(pw_out == 6);
    assert(w == 0);
    // Remaining in ring: 10 + 6 - 4(ISR drained) == 12 bytes ("456789AA1234", NOT "456789AA12341ZZZZ"!)
    sys.pend_irq(IRQ_DRAIN);
    assert(verified_len.read() == 12);
    assert(verified_b8.read() == 49 && verified_b11.read() == 52);
    println("BM2_LH1_ALL_OK");
    sys.exit(0);
}
"#;
        let (_mir, _c, stdout) = run_zev_c_sim_and_freestanding("b_m2_l_h1_asan", src);
        assert!(stdout.contains("BM2_LH1_ALL_OK"), "stdout:\n{stdout}");
    }

    #[test]
    fn b_m4_freestanding_mmio_clamp_boot_delay_wfi_and_dma_align() {
        let src = r#"
mmio Regs @ 0x40020000 {
    @rw [4]u32 arr @ 0x00;
}

usize make_oob_idx(usize base) {
    return base + 90;
}

fn boot_wait() {
    sys.delay(2ms);
}

fn main() {
    boot_wait();
    usize idx = make_oob_idx(9);
    Regs.arr[idx] = 0xCAFEBABE;
    u32 v = Regs.arr[idx];
    assert(v == 0xCAFEBABE);
    assert(Regs.arr[0] == 0xCAFEBABE);
    println("BM4_OK");
    sys.exit(0);
}
"#;
        let (_mir, c_code, stdout) = run_zev_c_sim_and_freestanding("b_m4_freestanding_asan", src);
        assert!(stdout.contains("BM4_OK"), "stdout:\n{stdout}");
        assert!(c_code.contains("static SYN_UNUSED SYN_DMA_ALIGN volatile uint8_t __zev_mmio_buf_Regs[16];"));
        assert!(c_code.contains("#if defined(SYN_FREESTANDING)"));
        assert!(c_code.contains("__asm__ volatile (\"wfi\" ::: \"memory\")"));
        assert!(
            !c_code.contains("syn_mmio_read_arr_u8(volatile uint8_t *base, uint32_t offset, size_t idx, size_t arr_len) {\n    size_t safe_idx = idx;\n#if !defined(SYN_FREESTANDING)"),
            "MMIO array clamping must be unconditional in freestanding mode"
        );
    }

    #[test]
    fn b_l2_uniform_coro_fn_param_boot_and_task_asan() {
        let src = r#"
watch<u32> out_sum = 0;

u32 step_add(u32 mode, u32 delta) {
    sys.delay(1ms);
    return mode * 10 + delta;
}

task worker() {
    u32 b = step_add(7, 3);
    out_sum.publish(out_sum.read() + b);
    assert(out_sum.read() == 145);
    println(f"BL2_SUM={out_sum.read()}");
    sys.exit(0);
}

fn main() {
    u32 a = step_add(7, 2);
    out_sum.publish(a);
    spawn worker() @normal;
}
"#;
        let (mir, c_code, stdout) = run_zev_c_sim_and_freestanding("b_l2_uniform_coro_asan", src);
        assert!(stdout.contains("BL2_SUM=145"), "stdout:\n{stdout}");
        let sf = mir.functions.iter().find(|f| f.name == "step_add").unwrap();
        assert!(!sf.vars[0].hoisted, "uniform param `mode` must not be hoisted");
        assert!(sf.vars[1].hoisted, "non-uniform param `delta` must stay hoisted");
        assert!(
            !c_code.contains(".v0_mode ="),
            "folded param `.v0_mode` must not appear in compound literal initializers:\n{c_code}"
        );
    }

    #[test]
    fn string_lit_coerces_to_u8_slice_in_locals_assigns_structs_returns_and_calls() {
        let src = r#"
struct Desc {
    []u8 bytes;
}

[]u8 ret_lit_slice() {
    return "RET\x00!";
}

u8 check_slice([]u8 s, usize expected_len, usize idx, u8 expected_byte) {
    assert(s.len == expected_len);
    return s[idx];
}

fn main() {
    // Embedded NUL byte descriptor (like USB device descriptors in `bsp/rpi_pico_w.zev`)
    []u8 dev_desc = "\x12\x01\x00\x02\x8A\x2E";
    assert(dev_desc.len == 6usize);
    assert(dev_desc[0] == 0x12u8 && dev_desc[2] == 0x00u8 && dev_desc[5] == 0x2Eu8);

    dev_desc = "\x04\x03\x09\x04";
    assert(dev_desc.len == 4usize && dev_desc[3] == 0x04u8);

    []u8 r = ret_lit_slice();
    assert(r.len == 5usize && r[3] == 0x00u8 && r[4] == '!');

    Desc d = Desc { bytes: "AB\x00CD" };
    assert(d.bytes.len == 5usize && d.bytes[2] == 0x00u8 && d.bytes[4] == 'D');

    u8 tail = check_slice("ZEV2040", 7usize, 6usize, '0');
    assert(tail == '0');
    println(f"SLICE_LIT_OK tail={tail}");
    sys.exit(0);
}
"#;
        let (_mir, c_code, stdout) = run_zev_c_sim_and_freestanding("slice_lit_coercion_asan", src);
        assert!(stdout.contains("SLICE_LIT_OK tail=48"), "stdout:\n{stdout}");
        assert!(
            c_code.contains("syn_slice_u8"),
            "expected syn_slice_u8 compound literal in C99:\n{c_code}"
        );
    }

    #[test]
    fn sys_nop_delay_us_and_sleep_lock_runtime_asan() {
        let src = r#"
watch<u32> delay_hits = 0;

fn host_us_delay(u32 us) {
    delay_hits.publish(delay_hits.read() + us);
    sys.nop();
}

fn main() {
    sys.nop();
    assert(sys.sleep_depth() == 2);
    sys.lock_sleep();
    assert(sys.sleep_depth() == 1);
    sys.lock_sleep();
    assert(sys.sleep_depth() == 1);
    sys.unlock_sleep();
    assert(sys.sleep_depth() == 1);
    sys.unlock_sleep();
    assert(sys.sleep_depth() == 2);

    // Extra unlock saturates safely at 0.
    sys.unlock_sleep();
    assert(sys.sleep_depth() == 2);

    sys.set_delay_us(host_us_delay);
    sys.delay_us(42);
    assert(delay_hits.read() == 42);
    println(f"SYS_TIMING_LOCK_OK depth={sys.sleep_depth()} hits={delay_hits.read()}");
    sys.exit(0);
}
"#;
        let (_mir, _c_code, stdout) =
            run_zev_c_sim_and_freestanding("sys_nop_delay_us_sleep_lock_asan", src);
        assert!(
            stdout.contains("SYS_TIMING_LOCK_OK depth=2 hits=42"),
            "stdout:\n{stdout}"
        );
    }
}
