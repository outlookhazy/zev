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
pub mod opt;
pub mod ram;

pub use ir::*;
pub use lower::{lower_program, lower_programs, MirLowerError};
pub use ram::{
    collect_dma_layout_structs, compute_ram_report, populate_module_ram_report, ArchRamTotals,
    ChildSlotRamInfo, DualSpecializedFnRam, FrameLayoutReport, PrimitiveRam, RamReport, SizeAlign,
    SpawnedTaskRam, TargetAbi, TaskRunKind, VarRamInfo,
};
pub use zev_ast::{decoded_string_byte_len, decoded_string_bytes};

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

        for dma_name in ["uart1_dma", "uart2_dma"] {
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
                is_pub: false,
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

        // fn adc_read_rpm(u8 ch) -> u16
        items.push(Item {
            attrs: vec![],
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false, generic_params: vec![],
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
                                ExprKind::Call { generic_args: vec![],
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
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false, generic_params: vec![],
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
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false, generic_params: vec![],
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
                                ExprKind::Call { generic_args: vec![],
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
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false, generic_params: vec![],
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
                                            StmtKind::VarDecl {
                                                ty: ty_prim(PrimType::U16),
                                                name: id("actual"),
                                                init: Some(Expr::new(
                                                    ExprKind::Call { generic_args: vec![],
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
                                                                    ExprKind::Call { generic_args: vec![],
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
                                                                            "duty",
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
            is_pub: false,
            kind: ItemKind::Fn(FnDecl {
                attrs: vec![],
                is_extern_c: false, generic_params: vec![],
                name: id("main"),
                params: vec![],
                is_variadic: false,
                ret_type: ty_prim(PrimType::Void),
                error_spec: None,
                body: Some(Block::new(
                    vec![
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
                            StmtKind::Spawn(SpawnStmt {
                                module: None,
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
                                ExprKind::Call { generic_args: vec![],
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
                                                                    ExprKind::Call { generic_args: vec![],
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
                                                                    ExprKind::Call { generic_args: vec![],
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
                is_pub: false,
                kind: ItemKind::Task(TaskDecl {
                    attrs: vec![],
                    name: id("fast"),
                    params: vec![],
                    body: Block::new(
                        vec![Stmt::new(
                            StmtKind::LoopEvery {
                                period: Box::new(Expr::new(
                                    ExprKind::Literal(Literal::Duration(DurationLit {
                                        value: 500,
                                        unit: DurationUnit::Us,
                                        span: sp,
                                    })),
                                    sp,
                                )),
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
                    is_pub: false,
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
                    is_pub: false,
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
                    is_pub: false,
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
                    is_pub: false,
                    kind: ItemKind::Fn(FnDecl {
                        attrs: vec![],
                        is_extern_c: false, generic_params: vec![],
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
                    is_pub: false,
                    kind: ItemKind::Fn(FnDecl {
                        attrs: vec![],
                        is_extern_c: false, generic_params: vec![],
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
                                                    ExprKind::Call { generic_args: vec![],
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

    #[test]
    fn wave1_boot_tree_unrolling_provenance_and_conditional_spawn_in_mir() {
        let sp = Span::dummy();
        let watch_u32_ty = TypeExpr::new(
            TypeKind::OsPrimitive(OsPrimitiveType::Watch {
                elem: Box::new(ty_prim(PrimType::U32)),
            }),
            sp,
        );
        let watch_u32_ptr = TypeExpr::new(TypeKind::Ptr(Box::new(watch_u32_ty)), sp);
        let addr_of = |n: &str| {
            Expr::new(
                ExprKind::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(expr_id(n)),
                },
                sp,
            )
        };
        let dur5 = Box::new(Expr::new(ExprKind::Literal(Literal::Duration(dur_ms(5))), sp));

        let items = vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                    attrs: vec![],
                    ty: OsPrimitiveType::Watch {
                        elem: Box::new(ty_prim(PrimType::U32)),
                    },
                    name: id("w0"),
                    init: Some(expr_int(0)),
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
                        elem: Box::new(ty_prim(PrimType::U32)),
                    },
                    name: id("w1"),
                    init: Some(expr_int(0)),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Task(TaskDecl {
                    attrs: vec![],
                    name: id("worker"),
                    params: vec![
                        Param {
                            ty: ty_prim(PrimType::U32),
                            name: id("id"),
                            span: sp,
                        },
                        Param {
                            ty: watch_u32_ptr.clone(),
                            name: id("out"),
                            span: sp,
                        },
                    ],
                    body: Block::new(
                        vec![Stmt::new(
                            StmtKind::LoopEvery {
                                period: dur5.clone(),
                                body: Block::new(
                                    vec![Stmt::new(
                                        StmtKind::Expr(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("out")),
                                                method: id("publish"),
                                                args: vec![CallArg::positional(expr_id("id"))],
                                            },
                                            sp,
                                        )),
                                        sp,
                                    )],
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
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Task(TaskDecl {
                    attrs: vec![],
                    name: id("diag_task"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("tag"),
                        span: sp,
                    }],
                    body: Block::new(
                        vec![Stmt::new(
                            StmtKind::LoopEvery {
                                period: dur5,
                                body: Block::new(vec![], sp),
                            },
                            sp,
                        )],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false, generic_params: vec![],
                    name: id("init_pair"),
                    params: vec![
                        Param {
                            ty: watch_u32_ptr.clone(),
                            name: id("a"),
                            span: sp,
                        },
                        Param {
                            ty: watch_u32_ptr,
                            name: id("b"),
                            span: sp,
                        },
                    ],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::Spawn(SpawnStmt {
                                    module: None,
                                    task: id("worker"),
                                    args: vec![
                                        CallArg::positional(expr_int(10)),
                                        CallArg::positional(expr_id("a")),
                                    ],
                                    priority: Some(Priority::High),
                                    core: None,
                                    span: sp,
                                }),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Spawn(SpawnStmt {
                                    module: None,
                                    task: id("worker"),
                                    args: vec![
                                        CallArg::positional(expr_int(20)),
                                        CallArg::positional(expr_id("b")),
                                    ],
                                    priority: Some(Priority::High),
                                    core: None,
                                    span: sp,
                                }),
                                sp,
                            ),
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
                    is_extern_c: false, generic_params: vec![],
                    name: id("setup_bus"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::Bool),
                        name: id("enable_diag"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::Expr(Expr::new(
                                    ExprKind::Call { generic_args: vec![],
                                        callee: Box::new(expr_id("init_pair")),
                                        args: vec![
                                            CallArg::positional(addr_of("w0")),
                                            CallArg::positional(addr_of("w1")),
                                        ],
                                    },
                                    sp,
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::If {
                                    cond: expr_id("enable_diag"),
                                    then_block: Block::new(
                                        vec![Stmt::new(
                                            StmtKind::Spawn(SpawnStmt {
                                                module: None,
                                                task: id("diag_task"),
                                                args: vec![CallArg::positional(expr_int(99))],
                                                priority: Some(Priority::Low),
                                                core: None,
                                                span: sp,
                                            }),
                                            sp,
                                        )],
                                        sp,
                                    ),
                                    else_block: None,
                                },
                                sp,
                            ),
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
                    is_extern_c: false, generic_params: vec![],
                    name: id("main"),
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Expr(Expr::new(
                                ExprKind::Call { generic_args: vec![],
                                    callee: Box::new(expr_id("setup_bus")),
                                    args: vec![CallArg::positional(Expr::new(
                                        ExprKind::Literal(Literal::Bool(true)),
                                        sp,
                                    ))],
                                },
                                sp,
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

        let prog = Program::new(items, sp);
        let mir = lower_program(&prog).expect("lower boot-tree test");

        assert_eq!(mir.spawned_tasks.len(), 3);
        assert_eq!(
            mir.spawned_tasks[0].provenance_path,
            vec!["main", "setup_bus", "init_pair", "worker"]
        );
        assert!(!mir.spawned_tasks[0].conditional);
        assert_eq!(
            mir.spawned_tasks[1].provenance_path,
            vec!["main", "setup_bus", "init_pair", "worker"]
        );
        assert!(!mir.spawned_tasks[1].conditional);
        assert_eq!(
            mir.spawned_tasks[2].provenance_path,
            vec!["main", "setup_bus", "diag_task"]
        );
        assert!(mir.spawned_tasks[2].conditional);

        let has_nested_cond_spawn = mir.boot.steps.iter().any(|step| match step {
            MirBootStep::Stmt(MirStmt::If { then_body, .. }) => then_body
                .iter()
                .any(|s| matches!(s, MirStmt::SpawnActivate { slot_id: 2, .. })),
            _ => false,
        });
        assert!(
            has_nested_cond_spawn,
            "expected conditional MirStmt::SpawnActivate inside boot If"
        );

        let table = mir.ram_report.format_table();
        assert!(
            table.contains("boot provenance  : main -> setup_bus -> init_pair -> worker"),
            "missing provenance in table:\n{table}"
        );
        assert!(
            table.contains("[conditional]"),
            "missing [conditional] marker in table:\n{table}"
        );
    }

    #[test]
    fn wave1_soft_yielding_dual_specialization_and_ram_report_in_mir() {
        let sp = Span::dummy();
        let dur5 = Box::new(Expr::new(ExprKind::Literal(Literal::Duration(dur_ms(5))), sp));
        let items = vec![
            // u32 leaf_soft(u32 x) { checkpoint(); return x + 1; }
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false, generic_params: vec![],
                    name: id("leaf_soft"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("x"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::U32),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(StmtKind::Checkpoint, sp),
                            Stmt::new(
                                StmtKind::Return(Some(Expr::new(
                                    ExprKind::Binary {
                                        op: BinaryOp::Add,
                                        lhs: Box::new(expr_id("x")),
                                        rhs: Box::new(expr_int(1)),
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
            },
            // u32 shared_soft(u32 n) { u32 acc = 0; for (u32 i in 0..=n) { acc += leaf_soft(i); } return acc; }
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false, generic_params: vec![],
                    name: id("shared_soft"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("n"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::U32),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("acc"),
                                    init: Some(expr_int(0)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::ForRange {
                                    var_ty: ty_prim(PrimType::U32),
                                    var_name: id("i"),
                                    start: Box::new(expr_int(0)),
                                    end: Box::new(expr_id("n")),
                                    inclusive: true,
                                    body: Block::new(
                                        vec![Stmt::new(
                                            StmtKind::Assign {
                                                target: expr_id("acc"),
                                                op: AssignOp::AddAssign,
                                                value: Expr::new(
                                                    ExprKind::Call { generic_args: vec![],
                                                        callee: Box::new(expr_id("leaf_soft")),
                                                        args: vec![CallArg::positional(expr_id(
                                                            "i",
                                                        ))],
                                                    },
                                                    sp,
                                                ),
                                            },
                                            sp,
                                        )],
                                        sp,
                                    ),
                                },
                                sp,
                            ),
                            Stmt::new(StmtKind::Return(Some(expr_id("acc"))), sp),
                        ],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            // u32 noyield_only_soft(u32 n) { checkpoint(); return n; }
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false, generic_params: vec![],
                    name: id("noyield_only_soft"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("n"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::U32),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(StmtKind::Checkpoint, sp),
                            Stmt::new(StmtKind::Return(Some(expr_id("n"))), sp),
                        ],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            // task runner()
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Task(TaskDecl {
                    attrs: vec![],
                    name: id("runner"),
                    params: vec![],
                    body: Block::new(
                        vec![Stmt::new(
                            StmtKind::LoopEvery {
                                period: dur5,
                                body: Block::new(
                                    vec![
                                        Stmt::new(
                                            StmtKind::VarDecl {
                                                ty: ty_prim(PrimType::U32),
                                                name: id("a"),
                                                init: Some(Expr::new(
                                                    ExprKind::Call { generic_args: vec![],
                                                        callee: Box::new(expr_id("shared_soft")),
                                                        args: vec![CallArg::positional(expr_int(
                                                            3,
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
                                                    vec![Stmt::new(
                                                        StmtKind::VarDecl {
                                                            ty: ty_prim(PrimType::U32),
                                                            name: id("b"),
                                                            init: Some(Expr::new(
                                                                ExprKind::Binary {
                                                                    op: BinaryOp::Add,
                                                                    lhs: Box::new(Expr::new(
                                                                        ExprKind::Call { generic_args: vec![],
                                                                            callee: Box::new(
                                                                                expr_id(
                                                                                    "shared_soft",
                                                                                ),
                                                                            ),
                                                                            args: vec![
                                                                                CallArg::positional(
                                                                                    expr_id("a"),
                                                                                ),
                                                                            ],
                                                                        },
                                                                        sp,
                                                                    )),
                                                                    rhs: Box::new(Expr::new(
                                                                        ExprKind::Call { generic_args: vec![],
                                                                            callee: Box::new(
                                                                                expr_id(
                                                                                    "noyield_only_soft",
                                                                                ),
                                                                            ),
                                                                            args: vec![
                                                                                CallArg::positional(
                                                                                    expr_int(2),
                                                                                ),
                                                                            ],
                                                                        },
                                                                        sp,
                                                                    )),
                                                                },
                                                                sp,
                                                            )),
                                                        },
                                                        sp,
                                                    )],
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
                        )],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false, generic_params: vec![],
                    name: id("main"),
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Spawn(SpawnStmt {
                                module: None,
                                task: id("runner"),
                                args: vec![],
                                priority: Some(Priority::Normal),
                                core: None,
                                span: sp,
                            }),
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
        let mir = lower_program(&prog).expect("lower soft-yielding specialization test");

        assert!(mir.dual_specialized_fns.contains(&"shared_soft".to_string()));
        assert!(mir.dual_specialized_fns.contains(&"leaf_soft".to_string()));
        assert!(!mir.dual_specialized_fns.contains(&"noyield_only_soft".to_string()));

        let shared_coro = mir.functions.iter().find(|f| f.name == "shared_soft").unwrap();
        assert!(shared_coro.is_yielding);
        let shared_ny = mir
            .functions
            .iter()
            .find(|f| f.name == "shared_soft__noyield")
            .unwrap();
        assert!(!shared_ny.is_yielding);

        assert!(mir.functions.iter().all(|f| f.name != "noyield_only_soft"));
        let ny_only = mir
            .functions
            .iter()
            .find(|f| f.name == "noyield_only_soft__noyield")
            .unwrap();
        assert!(!ny_only.is_yielding);

        let table = mir.ram_report.format_table();
        assert!(
            table.contains("Dual-Specialized Functions"),
            "expected Dual-Specialized Functions section in RamReport:\n{table}"
        );
        assert!(table.contains("shared_soft__noyield"));
        assert!(table.contains("leaf_soft__noyield"));
    }

    #[test]
    fn wave1_unified_inbox_restart_defer_and_for_loops_in_mir() {
        let sp = Span::dummy();
        let items = vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                    attrs: vec![],
                    ty: OsPrimitiveType::Signal,
                    name: id("gate"),
                    init: None,
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
                        cap: 4,
                    },
                    name: id("q"),
                    init: None,
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Task(TaskDecl {
                    attrs: vec![],
                    name: id("actor"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("seed"),
                        span: sp,
                    }],
                    body: Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("local_cnt"),
                                    init: Some(expr_id("seed")),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Defer(Box::new(Stmt::new(
                                    StmtKind::Assign {
                                        target: expr_id("local_cnt"),
                                        op: AssignOp::AddAssign,
                                        value: expr_int(1),
                                    },
                                    sp,
                                ))),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::LoopEvery {
                                    period: Box::new(Expr::new(
                                        ExprKind::Binary {
                                            op: BinaryOp::Add,
                                            lhs: Box::new(expr_id("seed")),
                                            rhs: Box::new(expr_int(5)),
                                        },
                                        sp,
                                    )),
                                    body: Block::new(
                                        vec![
                                            Stmt::new(
                                                StmtKind::Expr(Expr::new(
                                                    ExprKind::MethodCall {
                                                        receiver: Box::new(expr_id("q")),
                                                        method: id("send"),
                                                        args: vec![CallArg::positional(expr_id(
                                                            "local_cnt",
                                                        ))],
                                                    },
                                                    sp,
                                                )),
                                                sp,
                                            ),
                                            Stmt::new(
                                                StmtKind::If {
                                                    cond: Expr::new(
                                                        ExprKind::MethodCall {
                                                            receiver: Box::new(expr_id("gate")),
                                                            method: id("try_wait"),
                                                            args: vec![],
                                                        },
                                                        sp,
                                                    ),
                                                    then_block: Block::new(
                                                        vec![Stmt::new(StmtKind::Restart, sp)],
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
                    ),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false, generic_params: vec![],
                    name: id("main"),
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Spawn(SpawnStmt {
                                module: None,
                                task: id("actor"),
                                args: vec![CallArg::positional(expr_int(10))],
                                priority: Some(Priority::Normal),
                                core: None,
                                span: sp,
                            }),
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
        let mir = lower_program(&prog).expect("lower unified inbox + restart test");

        let bp = mir.task_blueprints.iter().find(|b| b.name == "actor").unwrap();
        assert!(
            bp.blocks[0].is_resume_entry,
            "restart target BlockId(0) must have is_resume_entry = true"
        );

        let has_restart_checkpoint = bp.blocks.iter().any(|b| {
            matches!(
                b.terminator,
                MirTerminator::Yield(MirYieldKind::ShareCheckpoint {
                    resume_bb: BlockId(0)
                })
            ) && b.stmts.iter().any(|s| matches!(s, MirStmt::EndInboxTurn))
        });
        assert!(
            has_restart_checkpoint,
            "restart must emit EndInboxTurn + ShareCheckpoint(BlockId(0))"
        );

        let has_inbox_send = bp.blocks.iter().any(|b| {
            matches!(
                &b.terminator,
                MirTerminator::Yield(MirYieldKind::Wait { arms, .. })
                    if arms.iter().any(|a| matches!(a.op, MirWaitArmOp::InboxSend { .. }))
            )
        });
        assert!(has_inbox_send, "expected MirYieldKind::Wait(InboxSend)");

        let has_every_latch_vars_hoisted = bp.blocks.iter().any(|b| match &b.terminator {
            MirTerminator::Yield(MirYieldKind::EveryLatch {
                next_deadline_var,
                period_var,
                ..
            }) => bp.vars[next_deadline_var.0].hoisted && bp.vars[period_var.0].hoisted,
            _ => false,
        });
        assert!(
            has_every_latch_vars_hoisted,
            "EveryLatch next_deadline_var and period_var must both be hoisted"
        );
    }

    #[test]
    fn wave2_floats_generics_metadata_and_zero_heap_format_in_mir() {
        let sp = Span::dummy();
        let mono_attr = |sig: &str| Attr {
            kind: AttrKind::Custom {
                name: "monomorphized".to_string(),
                arg: Some(sig.to_string()),
            },
            span: sp,
        };

        let items = vec![
            // Unmonomorphized template struct -> must be defensively skipped!
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Struct(StructDecl {
                    is_packed: false,
                    name: id("FixedVec"),
                    generic_params: vec![GenericParam {
                        name: id("T"),
                        kind: GenericParamKind::Type,
                        span: sp,
                    }],
                    fields: vec![],
                    span: sp,
                }),
                span: sp,
            },
            // Monomorphized struct instance `@monomorphized("FixedVec<f32, 4>")`
            Item {
                attrs: vec![mono_attr("FixedVec<f32, 4>")],
                is_pub: false,
                kind: ItemKind::Struct(StructDecl {
                    is_packed: false,
                    name: id("FixedVec__f32_4"),
                    generic_params: vec![],
                    fields: vec![
                        StructField {
                            attrs: vec![],
                            ty: TypeExpr::new(
                                TypeKind::Array {
                                    elem: Box::new(ty_prim(PrimType::F32)),
                                    len: ArrayLen::Fixed(4),
                                },
                                sp,
                            ),
                            name: id("buf"),
                            span: sp,
                        },
                        StructField {
                            attrs: vec![],
                            ty: ty_prim(PrimType::Usize),
                            name: id("len"),
                            span: sp,
                        },
                    ],
                    span: sp,
                }),
                span: sp,
            },
            // Struct for `{pkt:?}` compile-time debug expansion
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Struct(StructDecl {
                    is_packed: false,
                    name: id("SensorPkt"),
                    generic_params: vec![],
                    fields: vec![
                        StructField {
                            attrs: vec![],
                            ty: ty_prim(PrimType::U8),
                            name: id("id"),
                            span: sp,
                        },
                        StructField {
                            attrs: vec![],
                            ty: ty_prim(PrimType::F32),
                            name: id("temp"),
                            span: sp,
                        },
                    ],
                    span: sp,
                }),
                span: sp,
            },
            // Two enums: `BusErr` (formatted) and `UnusedEnum` (NOT formatted -> must NOT be in `formatted_enums`)
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Enum(EnumDecl {
                    name: id("BusErr"),
                    variants: vec![EnumVariant {
                        name: id("BadCrc"),
                        discriminant: None,
                        span: sp,
                    }],
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Enum(EnumDecl {
                    name: id("UnusedEnum"),
                    variants: vec![EnumVariant {
                        name: id("Silent"),
                        discriminant: None,
                        span: sp,
                    }],
                    span: sp,
                }),
                span: sp,
            },
            // Monomorphized fn `@monomorphized("clamp<f32>")`
            Item {
                attrs: vec![mono_attr("clamp<f32>")],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("clamp__f32"),
                    generic_params: vec![],
                    params: vec![Param {
                        ty: ty_prim(PrimType::F32),
                        name: id("v"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::F32),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(StmtKind::Return(Some(expr_id("v"))), sp)],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
            // Task holding hoisted `f32` + `f64` across `delay(1ms)` and formatting `{pkt:?}`, `{err}`, `fmt_buf`
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Task(TaskDecl {
                    attrs: vec![],
                    name: id("telem"),
                    params: vec![],
                    body: Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::F32),
                                    name: id("acc32"),
                                    init: Some(Expr::new(
                                        ExprKind::Literal(Literal::float_from_f64(
                                            1.25,
                                            Some(PrimType::F32),
                                        )),
                                        sp,
                                    )),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::F64),
                                    name: id("acc64"),
                                    init: Some(Expr::new(
                                        ExprKind::Literal(Literal::float_from_f64(2.5, None)),
                                        sp,
                                    )),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::LoopEvery {
                                    period: Box::new(Expr::new(
                                        ExprKind::Literal(Literal::Duration(dur_ms(5))),
                                        sp,
                                    )),
                                    body: Block::new(
                                        vec![
                                            Stmt::new(
                                                StmtKind::Expr(Expr::new(
                                                    ExprKind::Call {
                                                        callee: Box::new(expr_id("delay")),
                                                        generic_args: vec![],
                                                        args: vec![CallArg::positional(Expr::new(
                                                            ExprKind::Literal(Literal::Duration(
                                                                dur_ms(1),
                                                            )),
                                                            sp,
                                                        ))],
                                                    },
                                                    sp,
                                                )),
                                                sp,
                                            ),
                                            Stmt::new(
                                                StmtKind::Assign {
                                                    target: expr_id("acc32"),
                                                    op: AssignOp::AddAssign,
                                                    value: Expr::new(
                                                        ExprKind::Call {
                                                            callee: Box::new(expr_id("clamp__f32")),
                                                            generic_args: vec![],
                                                            args: vec![CallArg::positional(
                                                                Expr::new(
                                                                    ExprKind::Cast {
                                                                        expr: Box::new(expr_id(
                                                                            "acc64",
                                                                        )),
                                                                        ty: ty_prim(PrimType::F32),
                                                                    },
                                                                    sp,
                                                                ),
                                                            )],
                                                        },
                                                        sp,
                                                    ),
                                                },
                                                sp,
                                            ),
                                            Stmt::new(
                                                StmtKind::VarDecl {
                                                    ty: ty_named("SensorPkt"),
                                                    name: id("pkt"),
                                                    init: Some(Expr::new(
                                                        ExprKind::StructLit {
                                                            module: None,
                                                            name: id("SensorPkt"),
                                                            generic_args: vec![],
                                                            fields: vec![
                                                                FieldInit {
                                                                    name: id("id"),
                                                                    value: expr_int(7),
                                                                    span: sp,
                                                                },
                                                                FieldInit {
                                                                    name: id("temp"),
                                                                    value: expr_id("acc32"),
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
                                                StmtKind::Expr(Expr::new(
                                                    ExprKind::Call {
                                                        callee: Box::new(expr_id("println")),
                                                        generic_args: vec![],
                                                        args: vec![CallArg::positional(Expr::new(
                                                            ExprKind::FormatString(vec![
                                                                FormatPart::Literal(
                                                                    "err=".to_string(),
                                                                ),
                                                                FormatPart::Hole {
                                                                    expr: Box::new(expr_id(
                                                                        "BadCrc",
                                                                    )),
                                                                    spec: FormatSpec::Default,
                                                                    span: sp,
                                                                },
                                                                FormatPart::Literal(
                                                                    " pkt=".to_string(),
                                                                ),
                                                                FormatPart::Hole {
                                                                    expr: Box::new(expr_id("pkt")),
                                                                    spec: FormatSpec::Debug,
                                                                    span: sp,
                                                                },
                                                            ]),
                                                            sp,
                                                        ))],
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
                            StmtKind::Spawn(SpawnStmt {
                                module: None,
                                task: id("telem"),
                                args: vec![],
                                priority: Some(Priority::Normal),
                                core: None,
                                span: sp,
                            }),
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
        let mir = lower_program(&prog).expect("lower wave2 mir test");

        // 1. Unmonomorphized template `FixedVec` skipped; concrete `FixedVec__f32_4` kept!
        assert!(mir.structs.iter().all(|s| s.name != "FixedVec"));
        assert!(mir.structs.iter().any(|s| s.name == "FixedVec__f32_4"));

        // 2. Pre-registered `FmtErr { Truncated = 0 }`
        assert!(mir.enums.iter().any(|e| e.name == "FmtErr"));

        // 3. `formatted_enums` contains ONLY `BusErr`, NOT `UnusedEnum` or `FmtErr`!
        assert_eq!(mir.formatted_enums, vec!["BusErr".to_string()]);

        // 4. `acc32` (4 B) and `acc64` (8 B) hoisted into `telem` coroutine frame
        let bp = mir.task_blueprints.iter().find(|b| b.name == "telem").unwrap();
        let acc32 = bp.vars.iter().find(|v| v.name == "acc32").unwrap();
        let acc64 = bp.vars.iter().find(|v| v.name == "acc64").unwrap();
        assert!(acc32.hoisted && acc32.ty == MirType::F32);
        assert!(acc64.hoisted && acc64.ty == MirType::F64);

        // 5. `{pkt:?}` expanded into literal `"SensorPkt { id: "` + field chunks + `" }"`!
        let has_expanded_debug = bp.blocks.iter().any(|blk| {
            blk.stmts.iter().any(|s| match s {
                MirStmt::PrintFmt { chunks, newline: true } => chunks.iter().any(|c| {
                    matches!(c, MirFmtChunk::Literal(lit) if lit.contains("SensorPkt { id: "))
                }),
                _ => false,
            })
        });
        assert!(has_expanded_debug, "expected compile-time struct debug expansion");

        // 6. Monomorphized instances populated & rendered in Section 7 of `RamReport`
        assert_eq!(mir.monomorphized_instances.len(), 2);
        let st_inst = mir
            .monomorphized_instances
            .iter()
            .find(|m| m.mangled_name == "FixedVec__f32_4")
            .unwrap();
        assert_eq!(st_inst.size_32, 20); // 4 * 4B f32 + 4B usize
        assert_eq!(st_inst.size_64, 24); // 4 * 4B f32 + 8B usize
        let table = mir.ram_report.format_table();
        assert!(
            table.contains("7. Monomorphized Generic Instantiations")
                && table.contains("FixedVec<f32, 4>")
                && table.contains("clamp<f32>"),
            "table:\n{table}"
        );
    }

    #[test]
    fn wave2_r2b_self_ref_debug_and_enum_slice_scanning_in_mir() {
        let sp = Span::dummy();
        let items = vec![
            // enum NavMode { Idle = 0, Cruise = 1 } (ONLY formatted via `[2]NavMode` `{modes:?}`!)
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Enum(EnumDecl {
                    name: id("NavMode"),
                    variants: vec![
                        EnumVariant {
                            name: id("Idle"),
                            discriminant: Some(0),
                            span: sp,
                        },
                        EnumVariant {
                            name: id("Cruise"),
                            discriminant: Some(1),
                            span: sp,
                        },
                    ],
                    span: sp,
                }),
                span: sp,
            },
            // Self-referential struct Node { u32 val; Node *next; }
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Struct(StructDecl {
                    is_packed: false,
                    name: id("Node"),
                    generic_params: vec![],
                    fields: vec![
                        StructField {
                            attrs: vec![],
                            ty: ty_prim(PrimType::U32),
                            name: id("val"),
                            span: sp,
                        },
                        StructField {
                            attrs: vec![],
                            ty: TypeExpr::new(
                                TypeKind::Ptr(Box::new(TypeExpr::new(
                                    TypeKind::Named(id("Node")),
                                    sp,
                                ))),
                                sp,
                            ),
                            name: id("next"),
                            span: sp,
                        },
                    ],
                    span: sp,
                }),
                span: sp,
            },
            // fn main() { Node n; [2]NavMode modes; println(f"{n:?} {modes:?}"); }
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
                                StmtKind::VarDecl {
                                    ty: TypeExpr::new(TypeKind::Named(id("Node")), sp),
                                    name: id("n"),
                                    init: None,
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: TypeExpr::new(
                                        TypeKind::Array {
                                            elem: Box::new(TypeExpr::new(
                                                TypeKind::Named(id("NavMode")),
                                                sp,
                                            )),
                                            len: ArrayLen::Fixed(2),
                                        },
                                        sp,
                                    ),
                                    name: id("modes"),
                                    init: None,
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(Expr::new(
                                    ExprKind::Call {
                                        callee: Box::new(expr_id("println")),
                                        generic_args: vec![],
                                        args: vec![CallArg::positional(Expr::new(
                                            ExprKind::FormatString(vec![
                                                FormatPart::Hole {
                                                    expr: Box::new(expr_id("n")),
                                                    spec: FormatSpec::Debug,
                                                    span: sp,
                                                },
                                                FormatPart::Literal(" ".to_string()),
                                                FormatPart::Hole {
                                                    expr: Box::new(expr_id("modes")),
                                                    spec: FormatSpec::Debug,
                                                    span: sp,
                                                },
                                            ]),
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
            },
        ];

        let prog = Program::new(items, sp);
        let mir = lower_program(&prog).expect("lower self-ref Node + [2]NavMode debug");

        // M3: `NavMode` formatted only via `[2]NavMode` `DebugSlice` MUST be recorded in `formatted_enums`!
        assert_eq!(mir.formatted_enums, vec!["NavMode".to_string()]);

        // H2: `Node { val: ..., next: 0x... }` formats `next` as `"0x"` + hex integer without auto-dereferencing `Node *`!
        let has_ptr_hex = mir.boot.steps.iter().any(|st| match st {
            MirBootStep::Stmt(MirStmt::PrintFmt { chunks, .. }) => chunks.iter().any(|c| {
                matches!(c, MirFmtChunk::Literal(s) if s.contains("next: 0x"))
            }),
            _ => false,
        });
        assert!(has_ptr_hex, "expected `next: 0x` hex pointer formatting on self-referential Node");
    }

    fn ty_ptr(inner: TypeExpr) -> TypeExpr {
        TypeExpr::new(TypeKind::Ptr(Box::new(inner)), Span::default())
    }

    fn ty_arr(elem: TypeExpr, len: usize) -> TypeExpr {
        TypeExpr::new(
            TypeKind::Array {
                elem: Box::new(elem),
                len: ArrayLen::Fixed(len),
            },
            Span::default(),
        )
    }

    fn expr_field(target: Expr, field: &str) -> Expr {
        let sp = Span::default();
        Expr::new(
            ExprKind::Field {
                target: Box::new(target),
                field: id(field),
            },
            sp,
        )
    }

    fn expr_idx(target: Expr, index: Expr) -> Expr {
        let sp = Span::default();
        Expr::new(
            ExprKind::Index {
                target: Box::new(target),
                index: Box::new(index),
            },
            sp,
        )
    }

    fn expr_call(callee: &str, args: Vec<Expr>) -> Expr {
        let sp = Span::default();
        Expr::new(
            ExprKind::Call {
                callee: Box::new(expr_id(callee)),
                generic_args: vec![],
                args: args.into_iter().map(CallArg::positional).collect(),
            },
            sp,
        )
    }

    fn expr_method(receiver: Expr, method: &str, args: Vec<Expr>) -> Expr {
        let sp = Span::default();
        Expr::new(
            ExprKind::MethodCall {
                receiver: Box::new(receiver),
                method: id(method),
                args: args.into_iter().map(CallArg::positional).collect(),
            },
            sp,
        )
    }

    fn expr_addrof(inner: Expr) -> Expr {
        Expr::new(
            ExprKind::Unary {
                op: UnaryOp::AddrOf,
                expr: Box::new(inner),
            },
            Span::default(),
        )
    }

    #[test]
    fn wave1_mmio_layouts_instances_rw_rmw_and_single_eval() {
        let sp = Span::default();
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
                                access: MmioAccess::Ro,
                                ty: ty_prim(PrimType::U16),
                                name: id("sr"),
                                offset: expr_int(0x02),
                                span: sp,
                            },
                            MmioRegDecl {
                                access: MmioAccess::Wo,
                                ty: ty_prim(PrimType::U32),
                                name: id("tx"),
                                offset: expr_int(0x04),
                                span: sp,
                            },
                            MmioRegDecl {
                                access: MmioAccess::W1c,
                                ty: ty_prim(PrimType::U32),
                                name: id("icr"),
                                offset: expr_int(0x08),
                                span: sp,
                            },
                            MmioRegDecl {
                                access: MmioAccess::Rw,
                                ty: ty_arr(ty_prim(PrimType::U16), 4),
                                name: id("arr"),
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
                            name: id("cfg"),
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
                    name: id("next_slot"),
                    generic_params: vec![],
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Usize),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(StmtKind::Return(Some(expr_int(2))), sp)],
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
                    name: id("probe"),
                    generic_params: vec![],
                    params: vec![Param {
                        ty: ty_ptr(ty_named("UartRegs")),
                        name: id("regs"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::U32),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            // regs.arr[next_slot()] |= 0x80;
                            Stmt::new(
                                StmtKind::Assign {
                                    target: expr_idx(
                                        expr_field(expr_id("regs"), "arr"),
                                        expr_call("next_slot", vec![]),
                                    ),
                                    op: AssignOp::BitOrAssign,
                                    value: expr_int(0x80),
                                },
                                sp,
                            ),
                            // regs.icr = 1;
                            Stmt::new(
                                StmtKind::Assign {
                                    target: expr_field(expr_id("regs"), "icr"),
                                    op: AssignOp::Assign,
                                    value: expr_int(1),
                                },
                                sp,
                            ),
                            // return (SysCfg.cfg + (regs.sr as u32));
                            Stmt::new(
                                StmtKind::Return(Some(Expr::new(
                                    ExprKind::Binary {
                                        op: BinaryOp::Add,
                                        lhs: Box::new(expr_field(expr_id("SysCfg"), "cfg")),
                                        rhs: Box::new(Expr::new(
                                            ExprKind::Cast {
                                                expr: Box::new(expr_field(expr_id("regs"), "sr")),
                                                ty: ty_prim(PrimType::U32),
                                            },
                                            sp,
                                        )),
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
                            StmtKind::Expr(expr_call("probe", vec![expr_addrof(expr_id("UART1"))])),
                            sp,
                        )],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
        ];

        let mir = lower_program(&Program::new(items, sp)).expect("lower MMIO test");
        let uart_layout = mir
            .mmio_layouts
            .iter()
            .find(|l| l.name == "UartRegs")
            .expect("UartRegs layout");
        assert_eq!(uart_layout.byte_size, 0x10 + 4 * 2);

        let uart1 = mir
            .mmio_instances
            .iter()
            .find(|i| i.name == "UART1")
            .expect("UART1 instance");
        assert_eq!(uart1.base_addr, 0x4001_1000);
        assert_eq!(uart1.byte_size, 24);

        let syscfg = mir
            .mmio_instances
            .iter()
            .find(|i| i.name == "SysCfg")
            .expect("SysCfg inline instance");
        assert_eq!(syscfg.base_addr, 0x4000_0000);
        assert_eq!(syscfg.byte_size, 4);

        let probe = mir
            .functions
            .iter()
            .find(|f| f.name == "probe")
            .expect("probe fn");
        assert_eq!(
            probe.vars[probe.params[0].0].ty,
            MirType::MmioPtr("UartRegs".to_string())
        );
        let MirFnBody::Direct(stmts) = &probe.body else {
            panic!("expected Direct fn body for probe");
        };
        // Compound RMW on `regs.arr[next_slot()]` must emit base & index CompilerTemp assigns before MmioWrite(Binary(BitOr, MmioRead, 0x80))!
        let next_slot_calls = stmts
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    MirStmt::Assign {
                        value: MirExpr {
                            kind: MirExprKind::CallDirect { callee, .. },
                            ..
                        },
                        ..
                    } if callee == "next_slot"
                )
            })
            .count();
        assert_eq!(next_slot_calls, 1, "index side-effect must be evaluated strictly once");
        assert!(matches!(
            stmts.iter().find(|s| matches!(s, MirStmt::MmioWrite { elem_bits: 16, .. })),
            Some(MirStmt::MmioWrite {
                offset: 0x10,
                elem_bits: 16,
                array_len: Some(4),
                value: MirExpr {
                    kind: MirExprKind::Binary { op: MirBinOp::BitOr, .. },
                    ..
                },
                ..
            })
        ));
    }

    #[test]
    fn wave1_parameterized_isr_boot_attach_shared_conditional_and_ram_report() {
        let sp = Span::default();
        let items = vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Mmio(MmioDecl {
                    name: id("UartRegs"),
                    kind: MmioDeclKind::Layout {
                        regs: vec![MmioRegDecl {
                            access: MmioAccess::Wo,
                            ty: ty_prim(PrimType::U32),
                            name: id("tx"),
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
                kind: ItemKind::Irq(IrqDecl {
                    name: id("UART1_IRQ"),
                    number: expr_int(37),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Irq(IrqDecl {
                    name: id("DMA_IRQ"),
                    number: expr_int(12),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Irq(IrqDecl {
                    name: id("MODE_IRQ"),
                    number: expr_int(42),
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
                        elem: Box::new(ty_prim(PrimType::U8)),
                        cap: 8,
                    },
                    name: id("rx_q"),
                    init: None,
                    span: sp,
                }),
                span: sp,
            },
            // isr uart_isr(*UartRegs regs, *inbox<u8, 8> q, u8 ch) { state u32 count = 0; state u8 tag = 7; count += 1; regs.tx = (ch as u32); let p = &count; }
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Isr(IsrDecl {
                    attrs: vec![],
                    vector: id("uart_isr"),
                    params: vec![
                        Param {
                            ty: ty_ptr(ty_named("UartRegs")),
                            name: id("regs"),
                            span: sp,
                        },
                        Param {
                            ty: ty_ptr(TypeExpr::new(
                                TypeKind::OsPrimitive(OsPrimitiveType::Inbox {
                                    elem: Box::new(ty_prim(PrimType::U8)),
                                    cap: 8,
                                }),
                                sp,
                            )),
                            name: id("q"),
                            span: sp,
                        },
                        Param {
                            ty: ty_prim(PrimType::U8),
                            name: id("ch"),
                            span: sp,
                        },
                    ],
                    body: Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::StateVar {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("count"),
                                    init: Some(expr_int(0)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::StateVar {
                                    ty: ty_prim(PrimType::U8),
                                    name: id("tag"),
                                    init: Some(expr_int(7)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: expr_id("count"),
                                    op: AssignOp::AddAssign,
                                    value: expr_int(1),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: expr_field(expr_id("regs"), "tx"),
                                    op: AssignOp::Assign,
                                    value: Expr::new(
                                        ExprKind::Cast {
                                            expr: Box::new(expr_id("ch")),
                                            ty: ty_prim(PrimType::U32),
                                        },
                                        sp,
                                    ),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_ptr(ty_prim(PrimType::U32)),
                                    name: id("p"),
                                    init: Some(expr_addrof(expr_id("count"))),
                                },
                                sp,
                            ),
                        ],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            // isr shared_isr(u32 tag_id) { state u16 hits = 0; hits += 1; }
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Isr(IsrDecl {
                    attrs: vec![],
                    vector: id("shared_isr"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("tag_id"),
                        span: sp,
                    }],
                    body: Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::StateVar {
                                    ty: ty_prim(PrimType::U16),
                                    name: id("hits"),
                                    init: Some(expr_int(0)),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Assign {
                                    target: expr_id("hits"),
                                    op: AssignOp::AddAssign,
                                    value: expr_int(1),
                                },
                                sp,
                            ),
                        ],
                        sp,
                    ),
                    span: sp,
                }),
                span: sp,
            },
            // fn setup_uart(*UartRegs r, irq vec, u8 ch) { vec.attach(uart_isr(r, &rx_q, ch)); }
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    name: id("setup_uart"),
                    generic_params: vec![],
                    params: vec![
                        Param {
                            ty: ty_ptr(ty_named("UartRegs")),
                            name: id("r"),
                            span: sp,
                        },
                        Param {
                            ty: ty_named("irq"),
                            name: id("vec"),
                            span: sp,
                        },
                        Param {
                            ty: ty_prim(PrimType::U8),
                            name: id("ch"),
                            span: sp,
                        },
                    ],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Expr(expr_method(
                                expr_id("vec"),
                                "attach",
                                vec![expr_call(
                                    "uart_isr",
                                    vec![expr_id("r"), expr_addrof(expr_id("rx_q")), expr_id("ch")],
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
                                StmtKind::Expr(expr_call(
                                    "setup_uart",
                                    vec![expr_addrof(expr_id("UART1")), expr_id("UART1_IRQ"), expr_int(3)],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(expr_method(
                                    expr_id("DMA_IRQ"),
                                    "attach_shared",
                                    vec![expr_call("shared_isr", vec![expr_int(1)])],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(expr_method(
                                    expr_id("DMA_IRQ"),
                                    "attach_shared",
                                    vec![expr_call("shared_isr", vec![expr_int(2)])],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::If {
                                    cond: Expr::new(ExprKind::Literal(Literal::Bool(true)), sp),
                                    then_block: Block::new(
                                        vec![Stmt::new(
                                            StmtKind::Expr(expr_method(
                                                expr_id("MODE_IRQ"),
                                                "attach",
                                                vec![expr_call("shared_isr", vec![expr_int(10)])],
                                            )),
                                            sp,
                                        )],
                                        sp,
                                    ),
                                    else_block: Some(Block::new(
                                        vec![Stmt::new(
                                            StmtKind::Expr(expr_method(
                                                expr_id("MODE_IRQ"),
                                                "attach",
                                                vec![expr_call("shared_isr", vec![expr_int(20)])],
                                            )),
                                            sp,
                                        )],
                                        sp,
                                    )),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(expr_method(
                                    expr_id("sys"),
                                    "pend_irq",
                                    vec![expr_id("UART1_IRQ")],
                                )),
                                sp,
                            ),
                        ],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
        ];

        let mir = lower_program(&Program::new(items, sp)).expect("lower ISR + boot attach test");

        let uart_isr = mir.isrs.iter().find(|i| i.name == "uart_isr").unwrap();
        assert_eq!(uart_isr.params.len(), 3);
        assert_eq!(uart_isr.state_vars.len(), 2);
        let count_vid = uart_isr.state_vars[0].var_id;
        assert_eq!(uart_isr.vars[count_vid.0].role, VarRole::IsrState);
        assert!(uart_isr.vars[count_vid.0].addr_taken);

        assert_eq!(mir.irq_vectors.len(), 3);
        let dma_vec = mir.irq_vectors.iter().find(|v| v.irq_num == 12).unwrap();
        assert!(dma_vec.is_shared);
        assert_eq!(dma_vec.selector_symbol, None);
        assert_eq!(dma_vec.bindings.len(), 2);
        assert_ne!(
            dma_vec.bindings[0].state_vars[0].storage_symbol,
            dma_vec.bindings[1].state_vars[0].storage_symbol
        );

        let uart_vec = mir.irq_vectors.iter().find(|v| v.irq_num == 37).unwrap();
        assert!(!uart_vec.is_shared);
        assert_eq!(uart_vec.bindings.len(), 1);
        assert!(matches!(
            &uart_vec.bindings[0].baked_args[..],
            [
                MirBakedArg::MmioAddr(u),
                MirBakedArg::PrimAddr(q),
                MirBakedArg::Const(MirExpr { kind: MirExprKind::IntLit(3), .. }),
            ] if u == "UART1" && q == "rx_q"
        ));
        assert_eq!(uart_vec.bindings[0].state_vars[0].storage_symbol, "__zev_isr_state_0_count");

        let mode_vec = mir.irq_vectors.iter().find(|v| v.irq_num == 42).unwrap();
        assert_eq!(mode_vec.selector_symbol.as_deref(), Some("__zev_irq_sel_42"));
        assert_eq!(
            mode_vec.bindings[0].cond_selector,
            Some(("__zev_irq_sel_42".to_string(), 1))
        );
        assert_eq!(
            mode_vec.bindings[1].cond_selector,
            Some(("__zev_irq_sel_42".to_string(), 2))
        );

        assert!(mir.boot.steps.iter().any(|st| matches!(
            st,
            MirBootStep::Stmt(MirStmt::Intrinsic(MirIntrinsicOp::SysOp { method, args }))
            if method == "pend_irq" && matches!(&args[..], [MirExpr { kind: MirExprKind::IntLit(37), .. }])
        )));

        // uart_isr state: u32(4) + u8(1) = 5; 4x shared_isr instances (4 * u16 = 8); 1x selector = 1; total = 14 B!
        assert_eq!(mir.ram_report.mcu32.isr_state_bytes, 14);
        assert!(mir.ram_report.mcu32.total_sram_bytes >= 14);
        assert!(mir.ram_report.format_table().contains("ISR Static State"));
    }

    #[test]
    fn wave1_rejects_invalid_mmio_and_irq_attach_patterns() {
        let sp = Span::default();
        let mmio_items = vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Mmio(MmioDecl {
                    name: id("Regs"),
                    kind: MmioDeclKind::Layout {
                        regs: vec![
                            MmioRegDecl {
                                access: MmioAccess::Ro,
                                ty: ty_prim(PrimType::U32),
                                name: id("ro"),
                                offset: expr_int(0),
                                span: sp,
                            },
                            MmioRegDecl {
                                access: MmioAccess::Wo,
                                ty: ty_prim(PrimType::U32),
                                name: id("wo"),
                                offset: expr_int(4),
                                span: sp,
                            },
                            MmioRegDecl {
                                access: MmioAccess::W1c,
                                ty: ty_prim(PrimType::U32),
                                name: id("w1c"),
                                offset: expr_int(8),
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
                    name: id("R0"),
                    kind: MmioDeclKind::Instance {
                        layout: ty_named("Regs"),
                        base_addr: expr_int(0x4000_0000),
                    },
                    span: sp,
                }),
                span: sp,
            },
        ];

        let mk_fn_prog = |stmt: Stmt| {
            let mut items = mmio_items.clone();
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
                    body: Some(Block::new(vec![stmt], sp)),
                    span: sp,
                }),
                span: sp,
            });
            Program::new(items, sp)
        };

        // 1. Write to @ro
        assert!(lower_program(&mk_fn_prog(Stmt::new(
            StmtKind::Assign {
                target: expr_field(expr_id("R0"), "ro"),
                op: AssignOp::Assign,
                value: expr_int(1),
            },
            sp,
        )))
        .is_err());

        // 2. Read from @wo
        assert!(lower_program(&mk_fn_prog(Stmt::new(
            StmtKind::VarDecl {
                ty: ty_prim(PrimType::U32),
                name: id("v"),
                init: Some(expr_field(expr_id("R0"), "wo")),
            },
            sp,
        )))
        .is_err());

        // 3. Compound |= on @w1c
        assert!(lower_program(&mk_fn_prog(Stmt::new(
            StmtKind::Assign {
                target: expr_field(expr_id("R0"), "w1c"),
                op: AssignOp::BitOrAssign,
                value: expr_int(1),
            },
            sp,
        )))
        .is_err());

        // 4. StateVar outside ISR
        assert!(lower_program(&mk_fn_prog(Stmt::new(
            StmtKind::StateVar {
                ty: ty_prim(PrimType::U32),
                name: id("bad"),
                init: Some(expr_int(0)),
            },
            sp,
        )))
        .is_err());

        // 5. Duplicate exclusive .attach and .attach inside loop
        let irq_base = vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Irq(IrqDecl {
                    name: id("IRQ0"),
                    number: expr_int(0),
                    span: sp,
                }),
                span: sp,
            },
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Isr(IsrDecl {
                    attrs: vec![],
                    vector: id("h"),
                    params: vec![],
                    body: Block::new(vec![], sp),
                    span: sp,
                }),
                span: sp,
            },
        ];
        let mut dup_items = irq_base.clone();
        dup_items.push(Item {
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
                            StmtKind::Expr(expr_method(
                                expr_id("IRQ0"),
                                "attach",
                                vec![expr_call("h", vec![])],
                            )),
                            sp,
                        ),
                        Stmt::new(
                            StmtKind::Expr(expr_method(
                                expr_id("IRQ0"),
                                "attach",
                                vec![expr_call("h", vec![])],
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
        assert!(lower_program(&Program::new(dup_items, sp)).is_err());

        let mut loop_items = irq_base;
        loop_items.push(Item {
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
                        StmtKind::While {
                            cond: Expr::new(ExprKind::Literal(Literal::Bool(true)), sp),
                            body: Block::new(
                                vec![Stmt::new(
                                    StmtKind::Expr(expr_method(
                                        expr_id("IRQ0"),
                                        "attach",
                                        vec![expr_call("h", vec![])],
                                    )),
                                    sp,
                                )],
                                sp,
                            ),
                        },
                        sp,
                    )],
                    sp,
                )),
                span: sp,
            }),
            span: sp,
        });
        assert!(lower_program(&Program::new(loop_items, sp)).is_err());
    }

    #[test]
    fn wave1_audit_deref_mmio_pointer_and_array_ptr_lowering() {
        let sp = Span::dummy();
        let arr_ty = TypeExpr::new(
            TypeKind::Array {
                elem: Box::new(ty_prim(PrimType::U32)),
                len: ArrayLen::Fixed(4),
            },
            sp,
        );
        let ptr_regs_ty = TypeExpr::new(TypeKind::Ptr(Box::new(ty_named("Regs"))), sp);
        let ptr_u32_ty = TypeExpr::new(TypeKind::Ptr(Box::new(ty_prim(PrimType::U32))), sp);
        let deref_ptr = Expr::new(
            ExprKind::Unary {
                op: UnaryOp::Deref,
                expr: Box::new(expr_id("ptr")),
            },
            sp,
        );
        let deref_arr_i = Expr::new(
            ExprKind::Index {
                target: Box::new(expr_field(deref_ptr.clone(), "arr")),
                index: Box::new(expr_id("i")),
            },
            sp,
        );
        let prog = Program::new(
            vec![
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Mmio(MmioDecl {
                        name: id("Regs"),
                        kind: MmioDeclKind::Layout {
                            regs: vec![
                                MmioRegDecl {
                                    access: MmioAccess::Rw,
                                    ty: ty_prim(PrimType::U32),
                                    name: id("cr"),
                                    offset: expr_int(0),
                                    span: sp,
                                },
                                MmioRegDecl {
                                    access: MmioAccess::Rw,
                                    ty: arr_ty.clone(),
                                    name: id("arr"),
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
                        name: id("DEV"),
                        kind: MmioDeclKind::Instance {
                            layout: ty_named("Regs"),
                            base_addr: expr_int(0x4000_0000),
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
                        name: id("poke"),
                        generic_params: vec![],
                        params: vec![
                            Param {
                                ty: ptr_regs_ty,
                                name: id("ptr"),
                                span: sp,
                            },
                            Param {
                                ty: ty_prim(PrimType::Usize),
                                name: id("i"),
                                span: sp,
                            },
                        ],
                        is_variadic: false,
                        ret_type: ty_prim(PrimType::U32),
                        error_spec: None,
                        body: Some(Block::new(
                            vec![
                                Stmt::new(
                                    StmtKind::Assign {
                                        target: expr_field(deref_ptr.clone(), "cr"),
                                        op: AssignOp::Assign,
                                        value: expr_int(7),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::Assign {
                                        target: deref_arr_i.clone(),
                                        op: AssignOp::Assign,
                                        value: expr_field(deref_ptr.clone(), "cr"),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_prim(PrimType::Usize),
                                        name: id("n"),
                                        init: Some(expr_field(
                                            expr_field(deref_ptr.clone(), "arr"),
                                            "len",
                                        )),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: arr_ty,
                                        name: id("local"),
                                        init: Some(Expr::new(
                                            ExprKind::ArrayLit(vec![
                                                expr_int(10),
                                                expr_int(20),
                                                expr_int(30),
                                                expr_int(40),
                                            ]),
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ptr_u32_ty,
                                        name: id("p"),
                                        init: Some(expr_field(expr_id("local"), "ptr")),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::Return(Some(Expr::new(
                                        ExprKind::Binary {
                                            op: BinaryOp::Add,
                                            lhs: Box::new(deref_arr_i),
                                            rhs: Box::new(Expr::new(
                                                ExprKind::Cast {
                                                    expr: Box::new(expr_id("n")),
                                                    ty: ty_prim(PrimType::U32),
                                                },
                                                sp,
                                            )),
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
                                    ty: ty_prim(PrimType::U32),
                                    name: id("v"),
                                    init: Some(expr_call(
                                        "poke",
                                        vec![
                                            Expr::new(
                                                ExprKind::Unary {
                                                    op: UnaryOp::AddrOf,
                                                    expr: Box::new(expr_id("DEV")),
                                                },
                                                sp,
                                            ),
                                            expr_int(2),
                                        ],
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
            ],
            sp,
        );
        let mir = lower_program(&prog).expect(
            "MIR lowering must handle (*ptr).reg, (*ptr).arr[i], (*ptr).arr.len, and arr.ptr",
        );
        let poke_fn = mir.functions.iter().find(|f| f.name == "poke").expect("poke fn");
        let MirFnBody::Direct(stmts) = &poke_fn.body else {
            panic!("expected Direct body");
        };
        assert!(
            stmts
                .iter()
                .any(|s| matches!(s, MirStmt::MmioWrite { offset: 0x00, .. })),
            "expected MmioWrite to (*ptr).cr at offset 0x00: {stmts:?}"
        );
        assert!(
            stmts
                .iter()
                .any(|s| matches!(s, MirStmt::MmioWrite { offset: 0x10, index: Some(_), .. })),
            "expected indexed MmioWrite to (*ptr).arr[i] at offset 0x10: {stmts:?}"
        );
        assert!(
            stmts.iter().any(|s| matches!(
                s,
                MirStmt::Assign { value, .. }
                if matches!(&value.kind, MirExprKind::AddrOfPlace(_))
            )),
            "expected `local.ptr` to lower to AddrOfPlace: {stmts:?}"
        );
    }

    #[test]
    fn pool_dma_slot_and_linear_lease_lowering_and_ram_layout() {
        let sp = Span::default();
        let ty_u8 = || ty_prim(PrimType::U8);
        let ty_lease_u8 = || TypeExpr::new(TypeKind::Lease(Box::new(ty_u8())), sp);
        let dma_attr = Attr {
            kind: AttrKind::DmaSafe,
            span: sp,
        };
        let prog = Program::new(
            vec![
                // @dma_safe pool<u8, 4> pkt_pool;
                Item {
                    attrs: vec![dma_attr.clone()],
                    is_pub: false,
                    kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                        attrs: vec![],
                        ty: OsPrimitiveType::Pool {
                            elem: Box::new(ty_u8()),
                            cap: 4,
                        },
                        name: id("pkt_pool"),
                        init: None,
                        span: sp,
                    }),
                    span: sp,
                },
                // dma_slot<u8> tx_slot;
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                        attrs: vec![],
                        ty: OsPrimitiveType::DmaSlot {
                            elem: Box::new(ty_u8()),
                        },
                        name: id("tx_slot"),
                        init: None,
                        span: sp,
                    }),
                    span: sp,
                },
                // struct BusSub { @dma_safe pool<u8, 4> p; pool<u8, 4> cpu_p; }
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Struct(StructDecl {
                        is_packed: false,
                        generic_params: vec![],
                        name: id("BusSub"),
                        fields: vec![
                            StructField {
                                attrs: vec![dma_attr],
                                ty: TypeExpr::new(
                                    TypeKind::OsPrimitive(OsPrimitiveType::Pool {
                                        elem: Box::new(ty_u8()),
                                        cap: 4,
                                    }),
                                    sp,
                                ),
                                name: id("p"),
                                span: sp,
                            },
                            StructField {
                                attrs: vec![],
                                ty: TypeExpr::new(
                                    TypeKind::OsPrimitive(OsPrimitiveType::Pool {
                                        elem: Box::new(ty_u8()),
                                        cap: 4,
                                    }),
                                    sp,
                                ),
                                name: id("cpu_p"),
                                span: sp,
                            },
                        ],
                        span: sp,
                    }),
                    span: sp,
                },
                // BusSub bus;
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::GlobalVar(GlobalVarDecl {
                        ty: TypeExpr::new(TypeKind::Named(id("BusSub")), sp),
                        name: id("bus"),
                        init: None,
                        span: sp,
                    }),
                    span: sp,
                },
                // fn id_lease(Lease<u8> x) -> Lease<u8> { return x; }
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Fn(FnDecl {
                        attrs: vec![],
                        is_extern_c: false,
                        name: id("id_lease"),
                        generic_params: vec![],
                        params: vec![Param {
                            ty: ty_lease_u8(),
                            name: id("x"),
                            span: sp,
                        }],
                        is_variadic: false,
                        ret_type: ty_lease_u8(),
                        error_spec: None,
                        body: Some(Block::new(
                            vec![Stmt::new(StmtKind::Return(Some(expr_id("x"))), sp)],
                            sp,
                        )),
                        span: sp,
                    }),
                    span: sp,
                },
                // task Worker() {
                //   Lease<u8> a = pkt_pool.alloc();
                //   *u8 p0 = a.get();
                //   Lease<u8> b = id_lease(a);
                //   tx_slot.park(b) catch { return; };
                //   Lease<u8> c = tx_slot.take() catch { return; };
                //   c.release();
                // }
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Task(TaskDecl {
                        attrs: vec![],
                        name: id("Worker"),
                        params: vec![],
                        body: Block::new(
                            vec![
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_lease_u8(),
                                        name: id("a"),
                                        init: Some(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("pkt_pool")),
                                                method: id("alloc"),
                                                args: vec![],
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: TypeExpr::new(TypeKind::Ptr(Box::new(ty_u8())), sp),
                                        name: id("p0"),
                                        init: Some(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("a")),
                                                method: id("get"),
                                                args: vec![],
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_lease_u8(),
                                        name: id("b"),
                                        init: Some(expr_call("id_lease", vec![expr_id("a")])),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::Expr(Expr::new(
                                        ExprKind::Catch {
                                            expr: Box::new(Expr::new(
                                                ExprKind::MethodCall {
                                                    receiver: Box::new(expr_id("tx_slot")),
                                                    method: id("park"),
                                                    args: vec![CallArg::positional(expr_id("b"))],
                                                },
                                                sp,
                                            )),
                                            clause: CatchClause {
                                                err_param: None,
                                                partial_param: None,
                                                body: Block::new(
                                                    vec![Stmt::new(StmtKind::Return(None), sp)],
                                                    sp,
                                                ),
                                                span: sp,
                                            },
                                        },
                                        sp,
                                    )),
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_lease_u8(),
                                        name: id("c"),
                                        init: Some(Expr::new(
                                            ExprKind::Catch {
                                                expr: Box::new(Expr::new(
                                                    ExprKind::MethodCall {
                                                        receiver: Box::new(expr_id("tx_slot")),
                                                        method: id("take"),
                                                        args: vec![],
                                                    },
                                                    sp,
                                                )),
                                                clause: CatchClause {
                                                    err_param: None,
                                                    partial_param: None,
                                                    body: Block::new(
                                                        vec![Stmt::new(StmtKind::Return(None), sp)],
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
                                            receiver: Box::new(expr_id("c")),
                                            method: id("release"),
                                            args: vec![],
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
                                StmtKind::Spawn(SpawnStmt {
                                    module: None,
                                    task: id("Worker"),
                                    args: vec![],
                                    priority: None,
                                    core: None,
                                    span: sp,
                                }),
                                sp,
                            )],
                            sp,
                        )),
                        span: sp,
                    }),
                    span: sp,
                },
            ],
            sp,
        );

        let mir = lower_program(&prog).expect("pool/Lease/dma_slot lowering");
        let pool_ram = mir
            .ram_report
            .primitives
            .iter()
            .find(|p| p.name == "pkt_pool")
            .expect("pkt_pool ram");
        assert!(pool_ram.dma_safe);
        assert!(pool_ram.holds_sleep_lock);
        assert_eq!(pool_ram.bytes_32, 32 + 4 * 32);
        assert_eq!(pool_ram.bytes_64, 32 + 4 * 32);

        let bus_ram = mir
            .ram_report
            .primitives
            .iter()
            .find(|s| s.name == "bus")
            .expect("bus subsystem ram");
        assert!(bus_ram.holds_sleep_lock);
        // BusSub has @dma_safe pool<u8, 4> (160B, align 32) + non-dma pool<u8, 4> (20B/28B).
        assert_eq!(bus_ram.bytes_32, 192);
        assert_eq!(bus_ram.bytes_64, 192);

        let id_fn = mir.functions.iter().find(|f| f.name == "id_lease").unwrap();
        let MirFnBody::Direct(id_stmts) = &id_fn.body else {
            panic!("expected direct fn body");
        };
        assert!(
            id_stmts.iter().any(|s| matches!(
                s,
                MirStmt::PrimOp(MirPrimNonBlockingOp::LeaseNullify { .. })
            )),
            "returning Lease parameter must emit LeaseNullify: {id_stmts:?}"
        );

        let worker = &mir.task_blueprints[0];
        let mut saw_alloc = false;
        let mut saw_get = false;
        let mut saw_park = false;
        let mut saw_take = false;
        let mut saw_release = false;
        let mut nullify_count = 0usize;
        for blk in &worker.blocks {
            if matches!(
                &blk.terminator,
                MirTerminator::Yield(MirYieldKind::Wait { arms, .. })
                    if arms.iter().any(|a| matches!(a.op, MirWaitArmOp::PoolAlloc { .. }))
            ) {
                saw_alloc = true;
            }
            for st in &blk.stmts {
                match st {
                    MirStmt::PrimOp(MirPrimNonBlockingOp::LeaseNullify { .. }) => {
                        nullify_count += 1;
                    }
                    MirStmt::PrimOp(MirPrimNonBlockingOp::LeaseRelease { .. }) => {
                        saw_release = true;
                    }
                    MirStmt::Assign { value, .. } => match &value.kind {
                        MirExprKind::LeaseGet { .. } => saw_get = true,
                        MirExprKind::PrimRead(ro) => match ro.as_ref() {
                            MirPrimReadOp::DmaSlotPark { .. } => saw_park = true,
                            MirPrimReadOp::DmaSlotTake { .. } => saw_take = true,
                            _ => {}
                        },
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        assert!(saw_alloc && saw_get && saw_park && saw_take && saw_release);
        assert!(nullify_count >= 3, "expected nullify on moves into call, park, release: {nullify_count}");
    }

    #[test]
    fn b_m3_pool_contended_via_dma_slot_release_disables_irq_elision() {
        let sp = Span::dummy();
        let ty_u16 = || ty_prim(PrimType::U16);
        let ty_u32 = || ty_prim(PrimType::U32);
        let prog = Program::new(
            vec![
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                        attrs: vec![],
                        ty: OsPrimitiveType::Pool {
                            elem: Box::new(ty_u32()),
                            cap: 2,
                        },
                        name: id("dma_pool"),
                        init: None,
                        span: sp,
                    }),
                    span: sp,
                },
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                        attrs: vec![],
                        ty: OsPrimitiveType::Pool {
                            elem: Box::new(ty_u16()),
                            cap: 2,
                        },
                        name: id("task_pool"),
                        init: None,
                        span: sp,
                    }),
                    span: sp,
                },
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                        attrs: vec![],
                        ty: OsPrimitiveType::DmaSlot {
                            elem: Box::new(ty_u32()),
                        },
                        name: id("tx_slot"),
                        init: None,
                        span: sp,
                    }),
                    span: sp,
                },
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Irq(IrqDecl {
                        name: id("IRQ_DMA"),
                        number: expr_int(2),
                        span: sp,
                    }),
                    span: sp,
                },
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Isr(IsrDecl {
                        attrs: vec![],
                        vector: id("dma_done_isr"),
                        params: vec![Param {
                            ty: TypeExpr::new(
                                TypeKind::Ptr(Box::new(TypeExpr::new(
                                    TypeKind::OsPrimitive(OsPrimitiveType::DmaSlot {
                                        elem: Box::new(ty_u32()),
                                    }),
                                    sp,
                                ))),
                                sp,
                            ),
                            name: id("slot"),
                            span: sp,
                        }],
                        body: Block::new(
                            vec![Stmt::new(
                                StmtKind::Expr(Expr::new(
                                    ExprKind::MethodCall {
                                        receiver: Box::new(expr_id("slot")),
                                        method: id("release"),
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
                },
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Task(TaskDecl {
                        attrs: vec![],
                        name: id("worker"),
                        params: vec![],
                        body: Block::new(
                            vec![
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: TypeExpr::new(TypeKind::Lease(Box::new(ty_u32())), sp),
                                        name: id("l"),
                                        init: Some(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("dma_pool")),
                                                method: id("alloc"),
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
                                            receiver: Box::new(expr_id("tx_slot")),
                                            method: id("park"),
                                            args: vec![CallArg::positional(expr_id("l"))],
                                        },
                                        sp,
                                    )),
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: TypeExpr::new(TypeKind::Lease(Box::new(ty_u16())), sp),
                                        name: id("tl"),
                                        init: Some(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("task_pool")),
                                                method: id("alloc"),
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
                                            receiver: Box::new(expr_id("tl")),
                                            method: id("release"),
                                            args: vec![],
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
                },
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Fn(FnDecl {
                        attrs: vec![],
                        is_extern_c: false,
                        generic_params: vec![],
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
                                            receiver: Box::new(expr_id("IRQ_DMA")),
                                            method: id("attach"),
                                            args: vec![CallArg::positional(Expr::new(
                                                ExprKind::Call {
                                                    generic_args: vec![],
                                                    callee: Box::new(expr_id("dma_done_isr")),
                                                    args: vec![CallArg::positional(Expr::new(
                                                        ExprKind::Unary {
                                                            op: UnaryOp::AddrOf,
                                                            expr: Box::new(expr_id("tx_slot")),
                                                        },
                                                        sp,
                                                    ))],
                                                },
                                                sp,
                                            ))],
                                        },
                                        sp,
                                    )),
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::Spawn(SpawnStmt {
                                        module: None,
                                        task: id("worker"),
                                        args: vec![],
                                        priority: Some(Priority::Normal),
                                        core: None,
                                        span: sp,
                                    }),
                                    sp,
                                ),
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

        let mir = lower_program(&prog).expect("lower B-M3");
        let dma_prim = mir.primitives.iter().find(|p| p.name == "dma_pool").unwrap();
        let task_prim = mir.primitives.iter().find(|p| p.name == "task_pool").unwrap();
        assert!(
            !dma_prim.irq_mask_elided,
            "dma_pool contended via dma_slot.release must NOT mark irq_mask_elided"
        );
        assert!(
            task_prim.irq_mask_elided,
            "task_pool not touched by ISR must mark irq_mask_elided"
        );
        let dma_ram = mir
            .ram_report
            .primitives
            .iter()
            .find(|p| p.name == "dma_pool")
            .unwrap();
        assert!(!dma_ram.irq_mask_elided);
    }

    #[test]
    fn b_h3_l_h1_and_b_h2_lowering_regressions() {
        let sp = Span::dummy();
        let ty_u8 = || ty_prim(PrimType::U8);
        let ty_u32 = || ty_prim(PrimType::U32);
        let ty_i32 = || ty_prim(PrimType::I32);
        let ty_usize = || ty_prim(PrimType::Usize);
        let dma_attr = Attr {
            kind: AttrKind::DmaSafe,
            span: sp,
        };
        let prog = Program::new(
            vec![
                // struct Inner { pool<u8, 2> np; }
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Struct(StructDecl {
                        is_packed: false,
                        generic_params: vec![],
                        name: id("Inner"),
                        fields: vec![StructField {
                            attrs: vec![],
                            ty: TypeExpr::new(
                                TypeKind::OsPrimitive(OsPrimitiveType::Pool {
                                    elem: Box::new(ty_u8()),
                                    cap: 2,
                                }),
                                sp,
                            ),
                            name: id("np"),
                            span: sp,
                        }],
                        span: sp,
                    }),
                    span: sp,
                },
                // struct Outer { Inner inner; pool<u8, 2> op; }
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Struct(StructDecl {
                        is_packed: false,
                        generic_params: vec![],
                        name: id("Outer"),
                        fields: vec![
                            StructField {
                                attrs: vec![],
                                ty: ty_named("Inner"),
                                name: id("inner"),
                                span: sp,
                            },
                            StructField {
                                attrs: vec![],
                                ty: TypeExpr::new(
                                    TypeKind::OsPrimitive(OsPrimitiveType::Pool {
                                        elem: Box::new(ty_u8()),
                                        cap: 2,
                                    }),
                                    sp,
                                ),
                                name: id("op"),
                                span: sp,
                            },
                        ],
                        span: sp,
                    }),
                    span: sp,
                },
                // @dma_safe Outer d_outer;
                Item {
                    attrs: vec![dma_attr],
                    is_pub: false,
                    kind: ItemKind::GlobalVar(GlobalVarDecl {
                        ty: ty_named("Outer"),
                        name: id("d_outer"),
                        init: None,
                        span: sp,
                    }),
                    span: sp,
                },
                // Outer p_outer;
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::GlobalVar(GlobalVarDecl {
                        ty: ty_named("Outer"),
                        name: id("p_outer"),
                        init: None,
                        span: sp,
                    }),
                    span: sp,
                },
                // byte_ring<16> rx;
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::OsPrimitive(OsPrimitiveDecl {
                        attrs: vec![],
                        ty: OsPrimitiveType::ByteRing { cap: 16 },
                        name: id("rx"),
                        init: None,
                        span: sp,
                    }),
                    span: sp,
                },
                // fn wait_bytes() -> usize { []u8 ch = rx.wait_read(); return ch.len; }
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Fn(FnDecl {
                        attrs: vec![],
                        is_extern_c: false,
                        generic_params: vec![],
                        name: id("wait_bytes"),
                        params: vec![],
                        is_variadic: false,
                        ret_type: ty_usize(),
                        error_spec: None,
                        body: Some(Block::new(
                            vec![
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: TypeExpr::new(
                                            TypeKind::Slice(Box::new(ty_u8())),
                                            sp,
                                        ),
                                        name: id("ch"),
                                        init: Some(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("rx")),
                                                method: id("wait_read"),
                                                args: vec![],
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::Return(Some(Expr::new(
                                        ExprKind::Field {
                                            target: Box::new(expr_id("ch")),
                                            field: id("len"),
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
                },
                // task helper_user() {
                //   usize got = wait_bytes();
                //   u32 raw = 0x80000000 as u32;
                //   i32 sh = 4 as i32;
                //   u32 shifted = raw >> sh;
                // }
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Task(TaskDecl {
                        attrs: vec![],
                        name: id("helper_user"),
                        params: vec![],
                        body: Block::new(
                            vec![
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_usize(),
                                        name: id("got"),
                                        init: Some(Expr::new(
                                            ExprKind::Call {
                                                generic_args: vec![],
                                                callee: Box::new(expr_id("wait_bytes")),
                                                args: vec![],
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_u32(),
                                        name: id("raw"),
                                        init: Some(Expr::new(
                                            ExprKind::Cast {
                                                expr: Box::new(expr_int(0x8000_0000)),
                                                ty: ty_u32(),
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_i32(),
                                        name: id("sh"),
                                        init: Some(Expr::new(
                                            ExprKind::Cast {
                                                expr: Box::new(expr_int(4)),
                                                ty: ty_i32(),
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: ty_u32(),
                                        name: id("shifted"),
                                        init: Some(Expr::new(
                                            ExprKind::Binary {
                                                op: BinaryOp::Shr,
                                                lhs: Box::new(expr_id("raw")),
                                                rhs: Box::new(expr_id("sh")),
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
                            ],
                            sp,
                        ),
                        span: sp,
                    }),
                    span: sp,
                },
                // fn main() { spawn helper_user(); []u8 tail_ch = rx.wait_read(); }
                Item {
                    attrs: vec![],
                    is_pub: false,
                    kind: ItemKind::Fn(FnDecl {
                        attrs: vec![],
                        is_extern_c: false,
                        generic_params: vec![],
                        name: id("main"),
                        params: vec![],
                        is_variadic: false,
                        ret_type: ty_prim(PrimType::Void),
                        error_spec: None,
                        body: Some(Block::new(
                            vec![
                                Stmt::new(
                                    StmtKind::Spawn(SpawnStmt {
                                        module: None,
                                        task: id("helper_user"),
                                        args: vec![],
                                        priority: Some(Priority::Normal),
                                        core: None,
                                        span: sp,
                                    }),
                                    sp,
                                ),
                                Stmt::new(
                                    StmtKind::VarDecl {
                                        ty: TypeExpr::new(
                                            TypeKind::Slice(Box::new(ty_u8())),
                                            sp,
                                        ),
                                        name: id("tail_ch"),
                                        init: Some(Expr::new(
                                            ExprKind::MethodCall {
                                                receiver: Box::new(expr_id("rx")),
                                                method: id("wait_read"),
                                                args: vec![],
                                            },
                                            sp,
                                        )),
                                    },
                                    sp,
                                ),
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

        let mir = lower_program(&prog).expect("lower B-H3/L-H1/B-H2");

        let wb = mir.functions.iter().find(|f| f.name == "wait_bytes").unwrap();
        assert!(wb.is_yielding);
        assert!(mir.task_blueprints.iter().any(|b| b.is_main_supervisor));

        let dma_structs = collect_dma_layout_structs(&mir);
        assert!(dma_structs.contains("Outer"));
        assert!(dma_structs.contains("Inner"));
        let d_ram = mir
            .ram_report
            .primitives
            .iter()
            .find(|p| p.name == "d_outer")
            .unwrap();
        let p_ram = mir
            .ram_report
            .primitives
            .iter()
            .find(|p| p.name == "p_outer")
            .unwrap();
        assert!(d_ram.holds_sleep_lock);
        assert!(!p_ram.holds_sleep_lock);
        assert_eq!(d_ram.bytes_32, p_ram.bytes_32);
        // Two DMA pools of `u8 x 2`: each is 32B hdr + 2*32B slots = 96B; total = 192B!
        assert_eq!(d_ram.bytes_32, 192);

        let hu = mir
            .task_blueprints
            .iter()
            .find(|b| b.name == "helper_user")
            .unwrap();
        let mut saw_unsigned_shr = false;
        for blk in &hu.blocks {
            for st in &blk.stmts {
                if let MirStmt::Assign { value, .. } = st {
                    if let MirExprKind::Binary {
                        op: MirBinOp::Shr,
                        lhs,
                        ..
                    } = &value.kind
                    {
                        assert_eq!(lhs.ty, MirType::U32, "Shr lhs must stay U32: {value:?}");
                        assert_eq!(value.ty, MirType::U32);
                        saw_unsigned_shr = true;
                    }
                }
            }
        }
        assert!(saw_unsigned_shr);
    }

    #[test]
    fn sys_isr_callbacks_seed_isr_queue_and_rom_call_lowers_with_const_bound() {
        let sp = Span::new(0, 0, 1, 1);
        let items = vec![
            Item {
                attrs: vec![],
                is_pub: false,
                kind: ItemKind::Const(ConstDecl {
                    ty: ty_prim(PrimType::Usize),
                    name: id("ROM_STACK"),
                    value: expr_int(48),
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
                        elem: Box::new(ty_prim(PrimType::U32)),
                    },
                    name: id("w_route"),
                    init: Some(expr_int(0)),
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
                        elem: Box::new(ty_prim(PrimType::U32)),
                    },
                    name: id("w_ack"),
                    init: Some(expr_int(0)),
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
                        elem: Box::new(ty_prim(PrimType::U32)),
                    },
                    name: id("w_fault"),
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
                    generic_params: vec![],
                    name: id("on_route"),
                    params: vec![
                        Param {
                            ty: ty_prim(PrimType::U32),
                            name: id("a"),
                            span: sp,
                        },
                        Param {
                            ty: ty_prim(PrimType::U32),
                            name: id("b"),
                            span: sp,
                        },
                    ],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Expr(expr_method(
                                expr_id("w_route"),
                                "publish",
                                vec![expr_id("a")],
                            )),
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
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    generic_params: vec![],
                    name: id("on_ack"),
                    params: vec![Param {
                        ty: ty_prim(PrimType::U32),
                        name: id("s"),
                        span: sp,
                    }],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Expr(expr_method(
                                expr_id("w_ack"),
                                "publish",
                                vec![expr_id("s")],
                            )),
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
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    generic_params: vec![],
                    name: id("on_fault"),
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![Stmt::new(
                            StmtKind::Expr(expr_method(
                                expr_id("w_fault"),
                                "publish",
                                vec![expr_int(1)],
                            )),
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
                kind: ItemKind::Fn(FnDecl {
                    attrs: vec![],
                    is_extern_c: false,
                    generic_params: vec![],
                    name: id("main"),
                    params: vec![],
                    is_variadic: false,
                    ret_type: ty_prim(PrimType::Void),
                    error_spec: None,
                    body: Some(Block::new(
                        vec![
                            Stmt::new(
                                StmtKind::Expr(expr_method(
                                    expr_id("sys"),
                                    "set_irq_router",
                                    vec![expr_id("on_route"), expr_id("on_ack")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(expr_method(
                                    expr_id("sys"),
                                    "set_fault_handler",
                                    vec![expr_id("on_fault")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::VarDecl {
                                    ty: ty_prim(PrimType::U32),
                                    name: id("v"),
                                    init: Some(expr_method(
                                        expr_id("sys"),
                                        "rom_call_u32",
                                        vec![expr_int(0x18), expr_id("ROM_STACK"), expr_int(1)],
                                    )),
                                },
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(expr_method(
                                    expr_id("sys"),
                                    "rom_call_void",
                                    vec![expr_int(0x20), expr_int(16), expr_id("v")],
                                )),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::If {
                                    cond: Expr::new(ExprKind::Literal(Literal::Bool(false)), sp),
                                    then_block: Block::new(
                                        vec![Stmt::new(
                                            StmtKind::Expr(expr_method(
                                                expr_id("sys"),
                                                "rom_call_noreturn",
                                                vec![expr_int(0x30), expr_id("v")],
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
                                StmtKind::Expr(expr_method(expr_id("w_route"), "read", vec![])),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(expr_method(expr_id("w_ack"), "read", vec![])),
                                sp,
                            ),
                            Stmt::new(
                                StmtKind::Expr(expr_method(expr_id("w_fault"), "read", vec![])),
                                sp,
                            ),
                        ],
                        sp,
                    )),
                    span: sp,
                }),
                span: sp,
            },
        ];

        let mir = lower_program(&Program::new(items, sp)).expect("lower ok");
        let vis = crate::opt::analyze_visibility(&mir);
        assert!(vis.isr_contended_globals.contains("w_ack"));
        assert!(vis.isr_contended_globals.contains("w_fault"));
        assert!(!vis.isr_contended_globals.contains("w_route"));

        let mut saw_u32_bound = false;
        let mut saw_void_bound = false;
        let mut saw_noreturn = false;
        for step in &mir.boot.steps {
            if let MirBootStep::Stmt(s) = step {
                match s {
                    MirStmt::Assign {
                        value:
                            MirExpr {
                                kind: MirExprKind::RomCallU32 { stack_bound: 48, .. },
                                ..
                            },
                        ..
                    } => saw_u32_bound = true,
                    MirStmt::Intrinsic(MirIntrinsicOp::SysOp { method, args })
                        if method == "rom_call_void" =>
                    {
                        if let Some(MirExpr {
                            kind: MirExprKind::IntLit(16),
                            ..
                        }) = args.get(1)
                        {
                            saw_void_bound = true;
                        }
                    }
                    MirStmt::If { then_body, .. } => {
                        for inner in then_body {
                            if matches!(
                                inner,
                                MirStmt::Intrinsic(MirIntrinsicOp::SysOp { method, .. })
                                    if method == "rom_call_noreturn"
                            ) {
                                saw_noreturn = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        assert!(saw_u32_bound && saw_void_bound && saw_noreturn);
    }
}
