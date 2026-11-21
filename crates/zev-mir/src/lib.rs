//! Coroutine State-Machine & Kernel MIR (`zev-mir`) for the Zev Language-OS.
//!
//! Lowers `zev_ast::Program` ASTs into:
//! - Explicit basic-block coroutine state machines (`MirTaskBlueprint`, `MirFunction`)
//! - Exact backward-liveness variable hoisting (`hoisted` frame fields vs stack-local variables)
//! - Disjoint child coroutine call `union`-aliasing
//! - Sequential `main()` boot sequence + static `spawn` task slot table + trailing supervisor task
//! - Zero-heap static SRAM budget (`RamReport`) for both 32-bit MCU and 64-bit host ABIs.

pub mod ir;
pub mod liveness;
pub mod lower;
pub mod ram;

pub use ir::*;
pub use lower::{lower_program, lower_programs, MirLowerError};
pub use ram::{
    compute_ram_report, populate_module_ram_report, ArchRamTotals, ChildSlotRamInfo,
    FrameLayoutReport, PrimitiveRam, RamReport, SizeAlign, SpawnedTaskRam, TargetAbi, VarRamInfo,
};

#[cfg(test)]
mod tests {
    use super::*;
    use zev_ast::*;

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

    #[test]
    fn hoisting_union_aliasing_and_ram_report_north_star() {
        // Construct AST mirroring `examples/motor_controller.zev` + yielding helpers:
        // - enum SupEvent { Overcurrent, BadPacket, IdleCheck }
        // - @dma_safe ping_pong<u8, 64> uart1_dma, uart2_dma;
        // - watch<u16> motor1_rpm = 0, motor2_rpm = 0;
        // - inbox<SupEvent, 8> sup_inbox;
        // - yielding fn adc_read_rpm(u8 ch) -> u16 { delay(1ms); return 1200; }
        // - pure fn pi_step(u16 target, u16 actual, *i32 integral) -> u16 { return target; }
        // - yielding fn save_fault_record(u32 bad_cnt) { u32 addr = 0x08010000; u32 crc = bad_cnt; delay(2ms); u32 done = addr + crc; }
        // - yielding fn scrub_flash_crc() { u32 page = 0; u32 acc = 0xFFFF; loop { acc += page; page += 1; if (page >= 16) { break; } } }
        // - task motor_control(*watch<u16> target_rpm, u8 pwm_ch) { i32 integral = 0; every 1ms { u16 target = target_rpm.read(); u16 actual = adc_read_rpm(pwm_ch); noyield { u16 duty = pi_step(target, actual, &integral); hw.TIM1.set_compare(pwm_ch, duty); } } }
        // - task fieldbus_listener(*ping_pong<u8, 64> rx_dma, *watch<u16> out_rpm) { loop { []u8 half = rx_dma.wait_chunk(); out_rpm.publish(1000); } }
        // - fn main() { spawn motor_control x2, spawn fieldbus_listener x2, set_priority(@low), u32 bad_packets = 0, loop { SupEvent ev = sup_inbox.recv(timeout: 5s) or IdleCheck; switch (ev) { ... } } }

        let sp = Span::dummy();
        let mut items = Vec::new();

        items.push(Item {
            attrs: vec![],
            kind: ItemKind::Import(ImportDecl {
                path: vec![id("chip"), id("stm32g4")],
                alias: Some(id("hw")),
                span: sp,
            }),
            span: sp,
        });

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

        for dma_name in ["uart1_dma", "uart2_dma"] {
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
                    name: id(dma_name),
                    init: None,
                    span: sp,
                }),
                span: sp,
            });
        }

        for wname in ["motor1_rpm", "motor2_rpm"] {
            items.push(Item {
                attrs: vec![],
                kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                    attrs: vec![],
                    ty: OsPrimitiveType::Watch {
                        elem: Box::new(ty_prim(PrimType::U16)),
                    },
                    name: id(wname),
                    init: Some(expr_int(0)),
                    span: sp,
                }),
                span: sp,
            });
        }

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

        // fn adc_read_rpm(u8 ch) -> u16
        items.push(Item {
            attrs: vec![],
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("adc_read_rpm"),
                params: vec![Param {
                    ty: ty_prim(PrimType::U8),
                    name: id("ch"),
                    span: sp,
                }],
                is_variadic: false,
                ret_type: ty_prim(PrimType::U16),
                error_spec: None,
                body: Some(Block::new(
                    vec![
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
                        Stmt::new(StmtKind::Return(Some(expr_int(1200))), sp),
                    ],
                    sp,
                )),
                span: sp,
            }),
            span: sp,
        });

        // fn pi_step(u16 target, u16 actual, *i32 integral) -> u16
        items.push(Item {
            attrs: vec![],
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false,
                name: id("pi_step"),
                params: vec![
                    Param {
                        ty: ty_prim(PrimType::U16),
                        name: id("target"),
                        span: sp,
                    },
                    Param {
                        ty: ty_prim(PrimType::U16),
                        name: id("actual"),
                        span: sp,
                    },
                    Param {
                        ty: TypeExpr::new(TypeKind::Ptr(Box::new(ty_prim(PrimType::I32))), sp),
                        name: id("integral"),
                        span: sp,
                    },
                ],
                is_variadic: false,
                ret_type: ty_prim(PrimType::U16),
                error_spec: None,
                body: Some(Block::new(
                    vec![Stmt::new(StmtKind::Return(Some(expr_id("target"))), sp)],
                    sp,
                )),
                span: sp,
            }),
            span: sp,
        });

        // fn save_fault_record(u32 bad_cnt) — yielding with live locals across delay(2ms)
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
                                name: id("flash_addr"),
                                init: Some(expr_int(0x0801_0000)),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::VarDecl {
                                ty: ty_prim(PrimType::U32),
                                name: id("checksum"),
                                init: Some(expr_id("bad_cnt")),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Expr(Expr::new(
                                ExprKind::Call {
                                    callee: Box::new(expr_id("delay")),
                                    args: vec![CallArg::positional(Expr::new(
                                        ExprKind::Literal(Literal::Duration(dur_ms(2))),
                                        sp,
                                    ))],
                                },
                                sp,
                            )),
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Assign {
                                target: expr_id("checksum"),
                                op: AssignOp::AddAssign,
                                value: expr_id("flash_addr"),
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

        // fn scrub_flash_crc() — yielding via unbounded loop with live locals
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
                                name: id("page"),
                                init: Some(expr_int(0)),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::VarDecl {
                                ty: ty_prim(PrimType::U32),
                                name: id("crc_acc"),
                                init: Some(expr_int(0xFFFF)),
                            },
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Loop {
                                body: Block::new(
                                    vec![
                                        Stmt::new(
                                            StmtKind::Assign {
                                                target: expr_id("crc_acc"),
                                                op: AssignOp::AddAssign,
                                                value: expr_id("page"),
                                            },
                                            sp,
                                        ),
                                        Stmt::new(
                                            StmtKind::Assign {
                                                target: expr_id("page"),
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
                                                        lhs: Box::new(expr_id("page")),
                                                        rhs: Box::new(expr_int(16)),
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
                    ],
                    sp,
                )),
                span: sp,
            }),
            span: sp,
        });

        // task motor_control(*watch<u16> target_rpm, u8 pwm_ch)
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
                                            StmtKind::VarDecl {
                                                ty: ty_prim(PrimType::U16),
                                                name: id("actual"),
                                                init: Some(Expr::new(
                                                    ExprKind::Call {
                                                        callee: Box::new(expr_id("adc_read_rpm")),
                                                        args: vec![CallArg::positional(expr_id(
                                                            "pwm_ch",
                                                        ))],
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
                                                            StmtKind::VarDecl {
                                                                ty: ty_prim(PrimType::U16),
                                                                name: id("duty"),
                                                                init: Some(Expr::new(
                                                                    ExprKind::Call {
                                                                        callee: Box::new(expr_id(
                                                                            "pi_step",
                                                                        )),
                                                                        args: vec![
                                                                            CallArg::positional(
                                                                                expr_id("target"),
                                                                            ),
                                                                            CallArg::positional(
                                                                                expr_id("actual"),
                                                                            ),
                                                                            CallArg::positional(
                                                                                Expr::new(
                                                                                    ExprKind::Unary {
                                                                                        op: UnaryOp::AddrOf,
                                                                                        expr: Box::new(expr_id("integral")),
                                                                                    },
                                                                                    sp,
                                                                                ),
                                                                            ),
                                                                        ],
                                                                    },
                                                                    sp,
                                                                )),
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
                                                                            expr_id("duty"),
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

        // task fieldbus_listener(*ping_pong<u8, 64> rx_dma, *watch<u16> out_rpm)
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
                                        StmtKind::Expr(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("out_rpm")),
                                                method: id("publish"),
                                                args: vec![CallArg::positional(expr_int(1000))],
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
                                task: id("motor_control"),
                                args: vec![addr_of("motor2_rpm"), CallArg::positional(expr_int(2))],
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
                            StmtKind::Spawn(SpawnStmt {
                                task: id("fieldbus_listener"),
                                args: vec![addr_of("uart2_dma"), addr_of("motor2_rpm")],
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
                                                                                DurationLit {
                                                                                    value: 5,
                                                                                    unit: DurationUnit::S,
                                                                                    span: sp,
                                                                                },
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
                                                            vec![Stmt::new(
                                                                StmtKind::Expr(Expr::new(
                                                                    ExprKind::Call {
                                                                        callee: Box::new(expr_id(
                                                                            "save_fault_record",
                                                                        )),
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
                                                            )],
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
        let mir = lower_program(&prog).expect("MIR lowering must succeed");

        // 1. Check 5 distinct static task slots (motor_control x2, fieldbus_listener x2, main supervisor x1)
        assert_eq!(mir.spawned_tasks.len(), 5);
        assert_eq!(mir.spawned_tasks[0].instance_name, "motor_control_0");
        assert_eq!(mir.spawned_tasks[1].instance_name, "motor_control_1");
        assert_eq!(mir.spawned_tasks[2].instance_name, "fieldbus_listener_0");
        assert_eq!(mir.spawned_tasks[3].instance_name, "fieldbus_listener_1");
        assert_eq!(mir.spawned_tasks[4].instance_name, "main");
        assert_eq!(mir.spawned_tasks[4].priority, MirPriority::Low);

        // 2. Check motor_control variable hoisting: `integral` and `target` hoisted, `duty` and `actual` on stack!
        let mc = mir
            .task_blueprints
            .iter()
            .find(|b| b.name == "motor_control")
            .unwrap();
        fn var_by_name<'a>(bp: &'a MirTaskBlueprint, n: &str) -> &'a MirVar {
            bp.vars.iter().find(|v| v.name == n).unwrap()
        }
        assert!(var_by_name(mc, "integral").hoisted, "`integral` must be hoisted");
        assert!(var_by_name(mc, "target").hoisted, "`target` crosses adc_read_rpm yield");
        assert!(!var_by_name(mc, "actual").hoisted, "`actual` must stay on stack");
        assert!(!var_by_name(mc, "duty").hoisted, "`duty` inside noyield must stay on stack");
        assert!(var_by_name(mc, "duty").in_noyield_scope);

        // 3. Check fieldbus_listener: `half` stays on stack!
        let fb = mir
            .task_blueprints
            .iter()
            .find(|b| b.name == "fieldbus_listener")
            .unwrap();
        assert!(!var_by_name(fb, "half").hoisted, "`half` must stay on stack");

        // 4. Check main supervisor: `bad_packets` hoisted, `ev` on stack, child frames union-aliased!
        let main_bp = mir
            .task_blueprints
            .iter()
            .find(|b| b.name == "main")
            .unwrap();
        assert!(
            var_by_name(main_bp, "bad_packets").hoisted,
            "`bad_packets` must be hoisted"
        );
        assert!(!var_by_name(main_bp, "ev").hoisted, "`ev` must stay on stack");
        assert_eq!(main_bp.child_slots.len(), 2);
        assert_eq!(main_bp.child_slots[0].union_group, 0);
        assert_eq!(main_bp.child_slots[1].union_group, 0);

        // 5. Check static RAM report: union_bytes_saved > 0 and heap_bytes == 0!
        assert!(
            mir.ram_report.mcu32.union_bytes_saved > 0,
            "expected union RAM savings > 0, got {}",
            mir.ram_report.mcu32.union_bytes_saved
        );
        assert_eq!(mir.ram_report.mcu32.heap_bytes, 0);
        assert_eq!(mir.ram_report.host64.heap_bytes, 0);

        let table = mir.ram_report.format_table();
        assert!(table.contains("motor_control_0"));
        assert!(table.contains("motor_control_1"));
        assert!(table.contains("save_fault_record"));
        assert!(table.contains("scrub_flash_crc"));
    }

    #[test]
    fn rejects_subtick_duration_in_mir() {
        let sp = Span::dummy();
        let prog = Program::new(
            vec![Item {
                attrs: vec![],
                kind: ItemKind::Task(TaskDecl {
                    attrs: vec![],
                    name: id("fast"),
                    params: vec![],
                    body: Block::new(
                        vec![Stmt::new(
                            StmtKind::Every {
                                interval: DurationLit {
                                    value: 500,
                                    unit: DurationUnit::Us,
                                    span: sp,
                                },
                                body: Block::new(vec![], sp),
                            },
                            sp,
                        )],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            }],
            sp,
        );
        let err = lower_program(&prog).unwrap_err();
        assert!(err.message.contains("500us"));
    }

    #[test]
    fn const_switch_arms_signed_binary_and_multistmt_catch_in_mir() {
        let sp = Span::dummy();
        let prog = Program::new(
            vec![
                Item {
                    attrs: vec![],
                    kind: ItemKind::Const(ConstDecl {
                        ty: ty_prim(PrimType::U8),
                        name: id("CMD_A"),
                        value: expr_int(0x11),
                        span: sp,
                    }),
                    span: sp,
                },
                Item {
                    attrs: vec![],
                    kind: ItemKind::Const(ConstDecl {
                        ty: ty_prim(PrimType::U8),
                        name: id("CMD_B"),
                        value: Expr::new(
                            ExprKind::Binary {
                                op: BinaryOp::Add,
                                lhs: Box::new(expr_id("CMD_A")),
                                rhs: Box::new(expr_int(5)),
                            },
                            sp,
                        ),
                        span: sp,
                    }),
                    span: sp,
                },
                Item {
                    attrs: vec![],
                    kind: ItemKind::Enum(EnumDecl {
                        name: id("PktErr"),
                        variants: vec![EnumVariant {
                            name: id("Bad"),
                            discriminant: None,
                            span: sp,
                        }],
                        span: sp,
                    }),
                    span: sp,
                },
                Item {
                    attrs: vec![],
                    kind: ItemKind::Fn(FnDecl {
                        attrs: vec![],
                        is_extern_c: false,
                        name: id("decode"),
                        params: vec![Param {
                            ty: ty_prim(PrimType::U8),
                            name: id("cmd"),
                            span: sp,
                        }],
                        is_variadic: false,
                        ret_type: ty_prim(PrimType::I32),
                        error_spec: Some(ErrorSpec::Typed(ty_named("PktErr"))),
                        body: Some(Block::new(
                            vec![Stmt::new(
                                StmtKind::Switch {
                                    scrutinee: expr_id("cmd"),
                                    arms: vec![
                                        SwitchArm {
                                            pat: SwitchPat::Ident {
                                                prefix: None,
                                                name: id("CMD_A"),
                                                span: sp,
                                            },
                                            body: Block::new(
                                                vec![Stmt::new(
                                                    StmtKind::Return(Some(expr_int(10))),
                                                    sp,
                                                )],
                                                sp,
                                            ),
                                            span: sp,
                                        },
                                        SwitchArm {
                                            pat: SwitchPat::Ident {
                                                prefix: None,
                                                name: id("CMD_B"),
                                                span: sp,
                                            },
                                            body: Block::new(
                                                vec![Stmt::new(
                                                    StmtKind::Return(Some(Expr::new(
                                                        ExprKind::Binary {
                                                            op: BinaryOp::Sub,
                                                            lhs: Box::new(expr_int(300)),
                                                            rhs: Box::new(Expr::new(
                                                                ExprKind::Cast {
                                                                    expr: Box::new(expr_int(500)),
                                                                    ty: ty_prim(PrimType::I32),
                                                                },
                                                                sp,
                                                            )),
                                                        },
                                                        sp,
                                                    ))),
                                                    sp,
                                                )],
                                                sp,
                                            ),
                                            span: sp,
                                        },
                                        SwitchArm {
                                            pat: SwitchPat::Wildcard(sp),
                                            body: Block::new(
                                                vec![Stmt::new(
                                                    StmtKind::Error {
                                                        reason: Some(expr_id("Bad")),
                                                        partial: Some(expr_int(7)),
                                                    },
                                                    sp,
                                                )],
                                                sp,
                                            ),
                                            span: sp,
                                        },
                                    ],
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
                    kind: ItemKind::Fn(FnDecl {
                        attrs: vec![],
                        is_extern_c: false,
                        name: id("caller"),
                        params: vec![],
                        is_variadic: false,
                        ret_type: ty_prim(PrimType::I32),
                        error_spec: None,
                        body: Some(Block::new(
                            vec![
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_prim(PrimType::I32),
                                        name: id("cnt"),
                                        init: Some(expr_int(0)),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_prim(PrimType::I32),
                                        name: id("v"),
                                        init: Some(Expr::new(
                                            ExprKind::Catch {
                                                expr: Box::new(Expr::new(
                                                    ExprKind::Call {
                                                        callee: Box::new(expr_id("decode")),
                                                        args: vec![CallArg::positional(expr_int(
                                                            99,
                                                        ))],
                                                    },
                                                    sp,
                                                )),
                                                clause: CatchClause {
                                                    err_param: Some(CatchParam {
                                                        ty: ty_named("PktErr"),
                                                        name: id("e"),
                                                        span: sp,
                                                    }),
                                                    partial_param: Some(CatchParam {
                                                        ty: ty_prim(PrimType::I32),
                                                        name: id("part"),
                                                        span: sp,
                                                    }),
                                                    body: Block::new(
                                                        vec![
                                                            Stmt::new(
                                                                StmtKind::Assign {
                                                                    target: expr_id("cnt"),
                                                                    op: AssignOp::AddAssign,
                                                                    value: expr_int(1),
                                                                },
                                                                sp,
                                                            ),
                                                            Stmt::new(
                                                                StmtKind::Expr(Expr::new(
                                                                    ExprKind::Binary {
                                                                        op: BinaryOp::Add,
                                                                        lhs: Box::new(expr_id(
                                                                            "part",
                                                                        )),
                                                                        rhs: Box::new(expr_int(
                                                                            100,
                                                                        )),
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
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                                Stmt::new(StmtKind::Return(Some(expr_id("v"))), sp),
                            ],
                            sp,
                        )),
                        span: sp,
                    }),
                    span: sp,
                },
            ],
            sp,
        );

        let mir = lower_program(&prog).expect("MIR lowering");
        let dec = mir.functions.iter().find(|f| f.name == "decode").unwrap();
        if let MirFnBody::Direct(stmts) = &dec.body {
            if let MirStmt::Switch { arms, .. } = &stmts[0] {
                assert_eq!(arms.len(), 2, "CMD_A and CMD_B must both be switch arms");
                assert!(matches!(arms[0].0, MirSwitchPattern::Int(0x11)));
                assert!(matches!(arms[1].0, MirSwitchPattern::Int(0x16)));
                if let MirStmt::Return(Some(ret_e)) = &arms[1].1[0] {
                    assert_eq!(ret_e.ty, MirType::I32, "`300 - (500 as i32)` must have MirType::I32");
                } else {
                    panic!("expected Return in CMD_B arm");
                }
            } else {
                panic!("expected MirStmt::Switch");
            }
        } else {
            panic!("expected Direct fn");
        }
    }
}
