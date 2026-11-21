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
        String::from_utf8_lossy(&run_out.stdout).to_string()
    }

    #[test]
    fn end_to_end_motor_controller_sim_boost_dma_endian_and_errors() {
        let sp = Span::dummy();
        let mut items = Vec::new();

        // packed struct CmdFrame { u8 op; u16be setpoint; u16be crc; }
        items.push(Item {
            attrs: vec![],
            kind: ItemKind::Struct(StructDecl {
                is_packed: true,
                name: id("CmdFrame"),
                fields: vec![
                    StructField {
                        ty: ty_prim(PrimType::U8),
                        name: id("op"),
                        span: sp,
                    },
                    StructField {
                        ty: ty_prim(PrimType::U16Be),
                        name: id("setpoint"),
                        span: sp,
                    },
                    StructField {
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

        // isr EXTI0 { hw.GPIOA.clear_pin(MOTOR_EN); sup_inbox.post(Overcurrent, boost: @critical); }
        items.push(Item {
            attrs: vec![],
            kind: ItemKind::Isr(IsrDecl {
                attrs: vec![],
                vector: id("EXTI0"),
                body: Block::new(
                    vec![
                        Stmt::new(
                            StmtKind::Expr(Expr::new(
                                ExprKind::MethodCall {
                                    receiver: Box::new(Expr::new(
                                        ExprKind::Field {
                                            target: Box::new(expr_id("hw")),
                                            field: id("GPIOA"),
                                        },
                                        sp,
                                    )),
                                    method: id("clear_pin"),
                                    args: vec![CallArg::positional(expr_id("MOTOR_EN"))],
                                },
                                sp,
                            )),
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
            kind: ItemKind::Isr(IsrDecl {
                attrs: vec![],
                vector: id("DMA1_CH5"),
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
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("crc16"),
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
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("parse_setpoint"),
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
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: true,
                name: id("c_query_running_prio"),
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
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("save_fault_record"),
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
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("scrub_flash_crc"),
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
                            StmtKind::Every {
                                interval: dur_ms(1),
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
                                                            StmtKind::Expr(Expr::new(
                                                                ExprKind::MethodCall {
                                                                    receiver: Box::new(Expr::new(
                                                                        ExprKind::Field {
                                                                            target: Box::new(
                                                                                expr_id("hw"),
                                                                            ),
                                                                            field: id("TIM1"),
                                                                        },
                                                                        sp,
                                                                    )),
                                                                    method: id("set_compare"),
                                                                    args: vec![
                                                                        CallArg::positional(
                                                                            expr_id("pwm_ch"),
                                                                        ),
                                                                        CallArg::positional(
                                                                            expr_id("target"),
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
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("main"),
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
    if (g_syn_hw.tim_compare[1] != 1500U) {
        fprintf(stderr, "Expected TIM1 CCR1 == 1500, got %u\n", g_syn_hw.tim_compare[1]);
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
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("read_stream"),
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
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("main"),
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

    #[test]
    fn parse_lower_and_compile_motor_controller_plus_support() {
        let mc_prog =
            zev_parser::parse_file("/home/cgalant/Projects/Prototype/examples/motor_controller.zev")
                .expect("parse motor_controller.zev");
        let sup_prog =
            zev_parser::parse_file("/home/cgalant/Projects/Prototype/examples/motor_support.zev")
                .expect("parse motor_support.zev");

        let mir = zev_mir::lower_programs(&[mc_prog, sup_prog])
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
        let demo_prog =
            zev_parser::parse_file("/home/cgalant/Projects/Prototype/examples/sim_demo.zev")
                .expect("parse sim_demo.zev");
        let mir = zev_mir::lower_program(&demo_prog).expect("lower sim_demo.zev");
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
    fn signal_chan_lease_little_endian_and_freestanding_mode() {
        let src = r#"
packed struct LeHeader {
    u8    magic;
    u16le length;
    u32le seq_num;
}

signal ready_sig;
chan<u32, 4> work_chan;
@dma_safe lease<u8, 16> spi_lease;
watch<u32> result_watch = 0;

task producer() {
    work_chan.send(77);
    ready_sig.notify(boost: @high);
}

task consumer() {
    ready_sig.wait();
    u32 v = work_chan.recv();
    []u8 buf = spi_lease.wait_chunk();
    LeHeader hdr = LeHeader.from_bytes(buf[0..7]) catch {
        return;
    };
    u32 total = v + (hdr.length as u32) + hdr.seq_num;
    result_watch.publish(total);
}

fn main() {
    spawn producer() @normal;
    spawn consumer() @low;
}
"#;
        let prog = zev_parser::parse_str(src).expect("parse signal/chan/lease test");
        let mir = zev_mir::lower_program(&prog).expect("lower signal/chan/lease");
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

        let driver = r#"
int main(void) {
    syn_boot_init();
    syn_sim_pump_ready();

    /* Complete the SPI lease buffer with little-endian magic=0xA5, length=100 (0x0064 -> 64 00), seq_num=1000 (0x000003E8 -> E8 03 00 00) */
    const uint8_t wire[7] = { 0xA5, 0x64, 0x00, 0xE8, 0x03, 0x00, 0x00 };
    syn_lease_u8_16_set_active(&spi_lease, true);
    if (syn_sim_allowed_sleep_depth() != (uint32_t)SYN_SLEEP_LIGHT) {
        return 1;
    }
    memcpy(spi_lease.buf, wire, 7U);
    spi_lease.len = 7U;
    syn_lease_u8_16_complete(&spi_lease);

    syn_sim_pump_ready();
    syn_lease_u8_16_set_active(&spi_lease, false);

    /* 77 + 100 + 1000 = 1177 */
    if (syn_watch_u32_read(&result_watch) != 1177U) {
        fprintf(stderr, "Expected 1177, got %u\n", syn_watch_u32_read(&result_watch));
        return 2;
    }
    printf("PRIMS_AND_LE_OK\n");
    return 0;
}
"#;
        let stdout = compile_and_run_clang_sanitized("prims_and_le", &c_code, Some(driver));
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
import chip.stm32g4 as hw;

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
    hw.UART0.puts("uart0_ok\n");
    hw.USART1.puts("usart1_ok\n");
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
}
