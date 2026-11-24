use super::*;

pub(super) fn target_int_signed_and_bits(
    target_ty: &MirType,
    ptr_bits: u32,
) -> Option<(bool, u32)> {
    match target_ty.host_value_type() {
        MirType::U8 => Some((false, 8)),
        MirType::I8 => Some((true, 8)),
        MirType::U16 => Some((false, 16)),
        MirType::I16 => Some((true, 16)),
        MirType::U32 => Some((false, 32)),
        MirType::I32 | MirType::Enum(_) => Some((true, 32)),
        MirType::U64 => Some((false, 64)),
        MirType::I64 => Some((true, 64)),
        MirType::Usize | MirType::Ptr(_) | MirType::FnPtr { .. } => Some((false, ptr_bits)),
        MirType::Isize => Some((true, ptr_bits)),
        _ => None,
    }
}

pub(super) fn saturating_float_to_int_i128(f: f64, target_ty: &MirType, ptr_bits: u32) -> i128 {
    match target_int_signed_and_bits(target_ty, ptr_bits) {
        Some((false, 8)) => (f as u8) as i128,
        Some((true, 8)) => (f as i8) as i128,
        Some((false, 16)) => (f as u16) as i128,
        Some((true, 16)) => (f as i16) as i128,
        Some((false, 32)) => (f as u32) as i128,
        Some((true, 32)) => (f as i32) as i128,
        Some((false, 64)) => (f as u64) as i128,
        Some((true, 64)) => (f as i64) as i128,
        _ => (f as i32) as i128,
    }
}

pub(super) fn mask_int_to_target_i128(v: i128, target_ty: &MirType, ptr_bits: u32) -> i128 {
    match target_int_signed_and_bits(target_ty, ptr_bits) {
        Some((false, 8)) => (v as u8) as i128,
        Some((true, 8)) => (v as i8) as i128,
        Some((false, 16)) => (v as u16) as i128,
        Some((true, 16)) => (v as i16) as i128,
        Some((false, 32)) => (v as u32) as i128,
        Some((true, 32)) => (v as i32) as i128,
        Some((false, 64)) => (v as u64) as i128,
        Some((true, 64)) => (v as i64) as i128,
        _ => v,
    }
}

pub(super) fn llvm_int_bits(ty: &str) -> u32 {
    match ty {
        "i1" => 1,
        "i8" => 8,
        "i16" => 16,
        "i32" => 32,
        "i64" => 64,
        _ => 32,
    }
}

impl<'ctx, 'm> FnIrBuilder<'ctx, 'm> {
    pub(super) fn emit_mmio_elem_ptr(
        &mut self,
        base: &MirExpr,
        offset: u32,
        elem_bits: u8,
        index: Option<&MirExpr>,
        array_len: Option<usize>,
    ) -> Result<(String, &'static str, usize), CodegenError> {
        let bv = self.emit_expr(base)?;
        let base_ptr = self.coerce(bv, "ptr", false).repr;
        let (lty, stride, align) = match elem_bits {
            8 => ("i8", 1u32, 1usize),
            16 => ("i16", 2u32, 2usize),
            _ => ("i32", 4u32, 4usize),
        };

        let is_host64 = self.ctx.spec.isa == CpuIsa::HostX86_64Linux;
        let idx_ty = if is_host64 { "i64" } else { "i32" };
        let ptr_tmp = if let Some(idx_expr) = index {
            let iv = self.emit_expr(idx_expr)?;
            let idx_val = self.coerce(iv, idx_ty, !idx_expr.ty.is_unsigned()).repr;
            let eff_idx = if let Some(alen) = array_len {
                if alen == 0 {
                    self.emit_bounds_check("false", 3);
                } else {
                    let inb = self.next_tmp();
                    self.emit_line(format!("{inb} = icmp ult {idx_ty} {idx_val}, {alen}"));
                    self.emit_bounds_check(&inb, 3);
                }
                idx_val
            } else {
                idx_val
            };

            let scaled = if stride == 1 {
                eff_idx
            } else {
                let s = self.next_tmp();
                self.emit_line(format!("{s} = mul {idx_ty} {eff_idx}, {stride}"));
                s
            };
            let total_off = if offset == 0 {
                scaled
            } else {
                let sum = self.next_tmp();
                let off_lit = if is_host64 {
                    i64::from(offset)
                } else {
                    i64::from(offset as i32)
                };
                self.emit_line(format!("{sum} = add {idx_ty} {off_lit}, {scaled}"));
                sum
            };
            let p = self.next_tmp();
            self.emit_line(format!(
                "{p} = getelementptr i8, ptr {base_ptr}, {idx_ty} {total_off}"
            ));
            p
        } else {
            let p = self.next_tmp();
            let off_lit = if is_host64 {
                i64::from(offset)
            } else {
                i64::from(offset as i32)
            };
            self.emit_line(format!(
                "{p} = getelementptr i8, ptr {base_ptr}, {idx_ty} {off_lit}"
            ));
            p
        };

        Ok((ptr_tmp, lty, align))
    }

    pub(super) fn emit_place_ptr(
        &mut self,
        place: &MirPlace,
    ) -> Result<(String, MirType, bool), CodegenError> {
        match place {
            MirPlace::Var(vid) => {
                let p = self.var_ptr(*vid);
                let ty = self.vars[vid.0].ty.clone();
                Ok((p, ty, false))
            }
            MirPlace::Deref(ptr_expr, pointee_ty) => {
                let pv = self.emit_expr(ptr_expr)?;
                let pc = self.coerce(pv, "ptr", false);
                Ok((pc.repr, pointee_ty.clone(), false))
            }
            MirPlace::Field {
                base,
                struct_name,
                packed,
                field,
                field_ty,
            } => {
                let (bptr, _, bpacked) = self.emit_place_ptr(base)?;
                let si = &self.ctx.structs[struct_name];
                let (fidx, _) = si.field_indices[field];
                let sty = si.llvm_name.clone();
                let gp = self.next_tmp();
                self.emit_line(format!(
                    "{gp} = getelementptr inbounds {sty}, ptr {bptr}, i32 0, i32 {fidx}"
                ));
                Ok((gp, field_ty.clone(), *packed || bpacked))
            }
            MirPlace::Index {
                base,
                index,
                elem_ty,
                is_slice,
            } => {
                let (bptr, bty, bpacked) = self.emit_place_ptr(base)?;
                let iv = self.emit_expr(index)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let ic = self.coerce(iv, &usize_ty, !index.ty.is_unsigned());
                let ety = self.ctx.llvm_ty(elem_ty);
                if *is_slice {
                    let sl_ty = format!("{{ ptr, {usize_ty} }}");
                    let sl = self.next_tmp();
                    let data_p = self.next_tmp();
                    let sl_len = self.next_tmp();
                    let inb = self.next_tmp();
                    let ep = self.next_tmp();
                    self.emit_line(format!("{sl} = load {sl_ty}, ptr {bptr}, align 4"));
                    self.emit_line(format!("{data_p} = extractvalue {sl_ty} {sl}, 0"));
                    self.emit_line(format!("{sl_len} = extractvalue {sl_ty} {sl}, 1"));
                    self.emit_line(format!("{inb} = icmp ult {usize_ty} {}, {sl_len}", ic.repr));
                    self.emit_bounds_check(&inb, 1);
                    self.emit_line(format!(
                        "{ep} = getelementptr {ety}, ptr {data_p}, {usize_ty} {}",
                        ic.repr
                    ));
                    Ok((ep, elem_ty.clone(), bpacked))
                } else {
                    if let MirType::Array { len, .. } = &bty {
                        if *len == 0 {
                            self.emit_bounds_check("false", 1);
                        } else {
                            let inb = self.next_tmp();
                            self.emit_line(format!(
                                "{inb} = icmp ult {usize_ty} {}, {len}",
                                ic.repr
                            ));
                            self.emit_bounds_check(&inb, 1);
                        }
                    }
                    let aty = self.ctx.llvm_ty(&bty);
                    let ep = self.next_tmp();
                    self.emit_line(format!(
                        "{ep} = getelementptr inbounds {aty}, ptr {bptr}, {usize_ty} 0, {usize_ty} {}",
                        ic.repr
                    ));
                    Ok((ep, elem_ty.clone(), bpacked))
                }
            }
        }
    }

    pub(super) fn try_emit_rodata_const_ptr(
        &mut self,
        expr: &MirExpr,
    ) -> Result<Option<(String, MirType, bool)>, CodegenError> {
        match &expr.kind {
            MirExprKind::ConstRef(name) => {
                if let Some((sym, ty)) = self.ctx.rodata_consts.get(name).cloned() {
                    Ok(Some((sym, ty, false)))
                } else {
                    Ok(None)
                }
            }
            MirExprKind::FieldAccess {
                base,
                struct_name,
                packed,
                is_ptr: false,
                field,
                raw_field_ty,
            } => {
                let Some((bp, _, _)) = self.try_emit_rodata_const_ptr(base)? else {
                    return Ok(None);
                };
                let si = self.ctx.structs[struct_name].clone();
                let (fidx, _) = si.field_indices[field];
                let sty = si.llvm_name;
                let gp = self.next_tmp();
                self.emit_line(format!(
                    "{gp} = getelementptr inbounds {sty}, ptr {bp}, i32 0, i32 {fidx}"
                ));
                Ok(Some((gp, raw_field_ty.clone(), *packed)))
            }
            MirExprKind::Index {
                base,
                index,
                raw_elem_ty,
                is_slice: false,
            } => {
                let Some((bp, bty, packed)) = self.try_emit_rodata_const_ptr(base)? else {
                    return Ok(None);
                };
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let iv = self.emit_expr(index)?;
                let ic = self.coerce(iv, &usize_ty, !index.ty.is_unsigned());
                let blty = self.ctx.llvm_ty(&bty);
                let ep = self.next_tmp();
                self.emit_line(format!(
                    "{ep} = getelementptr inbounds {blty}, ptr {bp}, {usize_ty} 0, {usize_ty} {}",
                    ic.repr
                ));
                let elem_mir = match &bty {
                    MirType::Array { elem, .. } => elem.as_ref().clone(),
                    _ => raw_elem_ty.clone(),
                };
                Ok(Some((ep, elem_mir, packed)))
            }
            _ => Ok(None),
        }
    }

    pub(super) fn load_from_place_ptr(
        &mut self,
        ptr: &str,
        raw_ty: &MirType,
        _packed: bool,
    ) -> SsaVal {
        let lty = self.ctx.llvm_ty(raw_ty);
        // `align 1` keeps unaligned `packed struct` / wire-cast pointer loads fault-safe while
        // `llc` folds known-aligned stack/frame accesses to natural load instructions.
        let raw = self.next_tmp();
        self.emit_line(format!("{raw} = load {lty}, ptr {ptr}, align 1"));
        if let MirType::EndianInt {
            bits,
            big_endian: true,
            ..
        } = raw_ty
        {
            let swapped = self.next_tmp();
            self.emit_line(format!(
                "{swapped} = call i{bits} @llvm.bswap.i{bits}(i{bits} {raw})"
            ));
            SsaVal::new(lty, swapped)
        } else {
            SsaVal::new(lty, raw)
        }
    }

    pub(super) fn store_to_place_ptr(
        &mut self,
        ptr: &str,
        raw_ty: &MirType,
        _packed: bool,
        host_val: SsaVal,
    ) {
        let lty = self.ctx.llvm_ty(raw_ty);
        let coerced = self.coerce(host_val, &lty, !raw_ty.is_unsigned());
        let wire_repr = if let MirType::EndianInt {
            bits,
            big_endian: true,
            ..
        } = raw_ty
        {
            let sw = self.next_tmp();
            self.emit_line(format!(
                "{sw} = call i{bits} @llvm.bswap.i{bits}(i{bits} {})",
                coerced.repr
            ));
            sw
        } else {
            coerced.repr
        };
        self.emit_line(format!("store {lty} {wire_repr}, ptr {ptr}, align 1"));
    }

    pub(super) fn pattern_int(&self, pat: &MirSwitchPattern) -> i64 {
        match pat {
            MirSwitchPattern::EnumVariant { enum_name, variant } => self
                .ctx
                .enum_variants
                .get(&(enum_name.clone(), variant.clone()))
                .or_else(|| self.ctx.bare_variants.get(variant))
                .copied()
                .unwrap_or(0),
            MirSwitchPattern::Int(v) => *v as i64,
            MirSwitchPattern::Bool(b) => i64::from(*b),
        }
    }

    // ── Pure / Direct Expressions ────────────────────────────────────────────

    pub(super) fn emit_expr(&mut self, expr: &MirExpr) -> Result<SsaVal, CodegenError> {
        let dst_lty = self.ctx.llvm_ty(&expr.ty.host_value_type());
        match &expr.kind {
            MirExprKind::IntLit(v) => {
                if dst_lty == "ptr" || dst_lty == "ptr addrspace(1)" {
                    if *v == 0 {
                        Ok(SsaVal::new(dst_lty, "null"))
                    } else if self.ctx.spec.isa.ptr_bits() == 16 {
                        let t = self.next_tmp();
                        let v16 = *v as i16;
                        self.emit_line(format!("{t} = inttoptr i16 {v16} to {dst_lty}"));
                        Ok(SsaVal::new(dst_lty, t))
                    } else {
                        let t = self.next_tmp();
                        let v32 = *v as i32;
                        self.emit_line(format!("{t} = inttoptr i32 {v32} to {dst_lty}"));
                        Ok(SsaVal::new(dst_lty, t))
                    }
                } else if dst_lty == "void" {
                    Ok(SsaVal::new("void", "0"))
                } else if dst_lty == "float" {
                    let fb = ((*v as f32) as f64).to_bits();
                    Ok(SsaVal::new("float", format!("0x{fb:016X}")))
                } else if dst_lty == "double" {
                    let db = (*v as f64).to_bits();
                    Ok(SsaVal::new("double", format!("0x{db:016X}")))
                } else if !dst_lty.starts_with('i') {
                    Ok(SsaVal::new(dst_lty, "zeroinitializer"))
                } else {
                    let lit = format_checked_llvm_int_const(&dst_lty, *v)?;
                    Ok(SsaVal::new(dst_lty, lit))
                }
            }
            MirExprKind::FloatLit(bits) => {
                let target_fty = if dst_lty == "float" || expr.ty == MirType::F32 {
                    "float"
                } else {
                    "double"
                };
                let norm_bits = if target_fty == "float" {
                    let f = f64::from_bits(*bits) as f32;
                    if f.is_nan() {
                        0x7FF8_0000_0000_0000u64
                    } else {
                        (f as f64).to_bits()
                    }
                } else {
                    *bits
                };
                Ok(SsaVal::new(target_fty, format!("0x{norm_bits:016X}")))
            }
            MirExprKind::BoolLit(b) => Ok(SsaVal::new("i8", if *b { "1" } else { "0" })),
            MirExprKind::TicksLit(t) => {
                let lit = format_checked_llvm_int_const(&dst_lty, i128::from(*t))?;
                Ok(SsaVal::new(dst_lty, lit))
            }
            MirExprKind::StringLit(s) => {
                let idx = self.ctx.intern_string(s);
                let slen = zev_ast::decoded_string_byte_len(s);
                if matches!(expr.ty, MirType::Slice(_)) {
                    let usize_ty = self.ctx.spec.isa.usize_llvm_ty();
                    let sl_ty = format!("{{ ptr, {usize_ty} }}");
                    let r0 = self.next_tmp();
                    let r1 = self.next_tmp();
                    self.emit_line(format!(
                        "{r0} = insertvalue {sl_ty} poison, ptr @.str.{idx}, 0"
                    ));
                    self.emit_line(format!(
                        "{r1} = insertvalue {sl_ty} {r0}, {usize_ty} {slen}, 1"
                    ));
                    Ok(SsaVal::new(sl_ty, r1))
                } else {
                    Ok(SsaVal::new("ptr", format!("@.str.{idx}")))
                }
            }
            MirExprKind::EnumVariant { enum_name, variant } => {
                let disc = self
                    .ctx
                    .enum_variants
                    .get(&(enum_name.clone(), variant.clone()))
                    .or_else(|| self.ctx.bare_variants.get(variant))
                    .copied()
                    .unwrap_or(0);
                Ok(SsaVal::new("i32", format!("{disc}")))
            }
            MirExprKind::ConstRef(name) => {
                if let Some((sym, mir_ty)) = self.ctx.rodata_consts.get(name).cloned() {
                    Ok(self.load_from_place_ptr(&sym, &mir_ty, false))
                } else if let Some(fname) = self.ctx.fnptr_consts.get(name) {
                    let fn_ptr_ty = self.ctx.spec.isa.fn_ptr_llvm_ty();
                    let sym = self
                        .ctx
                        .fn_sigs
                        .get(fname)
                        .map(|sig| sig.symbol.clone())
                        .unwrap_or_else(|| format!("@{}", sanitize_ident(fname)));
                    Ok(SsaVal::new(fn_ptr_ty, sym))
                } else if expr.ty.is_float() || dst_lty == "float" || dst_lty == "double" {
                    let raw = self
                        .ctx
                        .float_consts
                        .get(name)
                        .copied()
                        .or_else(|| {
                            self.ctx
                                .consts
                                .get(name)
                                .copied()
                                .map(|v| (v as f64).to_bits())
                        })
                        .unwrap_or(0);
                    let target_fty = if dst_lty == "float" || expr.ty == MirType::F32 {
                        "float"
                    } else {
                        "double"
                    };
                    let norm = if target_fty == "float" {
                        let f = f64::from_bits(raw) as f32;
                        if f.is_nan() {
                            0x7FF8_0000_0000_0000u64
                        } else {
                            (f as f64).to_bits()
                        }
                    } else {
                        raw
                    };
                    Ok(SsaVal::new(target_fty, format!("0x{norm:016X}")))
                } else {
                    let v = self.ctx.consts.get(name).copied().unwrap_or(1);
                    Ok(SsaVal::new(dst_lty, format!("{v}")))
                }
            }
            MirExprKind::GlobalPrimAddr(name) => Ok(SsaVal::new(
                "ptr",
                format!("@__syn_prim_{}", sanitize_ident(name)),
            )),
            MirExprKind::SubsystemAddr(name) => Ok(SsaVal::new(
                "ptr",
                format!("@__zev_subsys_{}", sanitize_ident(name)),
            )),
            MirExprKind::MakePrimView { ptr, cap, kind } => {
                let pv = self.emit_expr(ptr)?;
                let pc = self.coerce(pv, "ptr", false);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let n = (*cap).max(1);
                match kind {
                    MirPrimViewKind::ByteRing => {
                        let vty = format!("{{ ptr, {usize_ty} }}");
                        let v0 = self.next_tmp();
                        let v1 = self.next_tmp();
                        self.emit_line(format!(
                            "{v0} = insertvalue {vty} poison, ptr {}, 0",
                            pc.repr
                        ));
                        self.emit_line(format!("{v1} = insertvalue {vty} {v0}, {usize_ty} {n}, 1"));
                        Ok(SsaVal::new(vty, v1))
                    }
                    MirPrimViewKind::Inbox { elem } => {
                        let vty = format!("{{ ptr, ptr, {usize_ty} }}");
                        let pty = format!(
                            "%prim.{}",
                            sanitize_ident(
                                &MirPrimKind::Inbox {
                                    elem: elem.clone(),
                                    cap: n,
                                }
                                .mangle_name()
                            )
                        );
                        let items_base = self.next_tmp();
                        self.emit_line(format!(
                            "{items_base} = getelementptr inbounds {pty}, ptr {}, i32 0, i32 6, i32 0",
                            pc.repr
                        ));
                        let v0 = self.next_tmp();
                        let v1 = self.next_tmp();
                        let v2 = self.next_tmp();
                        self.emit_line(format!(
                            "{v0} = insertvalue {vty} poison, ptr {}, 0",
                            pc.repr
                        ));
                        self.emit_line(format!(
                            "{v1} = insertvalue {vty} {v0}, ptr {items_base}, 1"
                        ));
                        self.emit_line(format!("{v2} = insertvalue {vty} {v1}, {usize_ty} {n}, 2"));
                        Ok(SsaVal::new(vty, v2))
                    }
                }
            }
            MirExprKind::MmioAddr(inst) => {
                if !self.ctx.mmio_instances.contains_key(inst) {
                    return Err(CodegenError::new(
                        "mir_codegen",
                        format!("unknown MMIO peripheral instance `{inst}`"),
                    ));
                }
                if self.ctx.spec.isa == CpuIsa::HostX86_64Linux {
                    Ok(SsaVal::new(
                        "ptr",
                        format!("@__zev_mmio_buf_{}", sanitize_ident(inst)),
                    ))
                } else {
                    let addr = self.ctx.mmio_instances[inst].base_addr as i32;
                    Ok(SsaVal::new("ptr", format!("inttoptr (i32 {addr} to ptr)")))
                }
            }
            MirExprKind::MmioRead {
                base,
                offset,
                elem_bits,
                index,
                array_len,
            } => {
                let (p, lty, align) = self.emit_mmio_elem_ptr(
                    base,
                    *offset,
                    *elem_bits,
                    index.as_deref(),
                    *array_len,
                )?;
                let val = self.next_tmp();
                self.emit_line(format!(
                    "{val} = load volatile {lty}, ptr {p}, align {align}"
                ));
                Ok(self.coerce(SsaVal::new(lty, val), &dst_lty, !expr.ty.is_unsigned()))
            }
            MirExprKind::FnAddr(fname) => {
                let fn_ptr_ty = self.ctx.spec.isa.fn_ptr_llvm_ty();
                if let Some(sig) = self.ctx.fn_sigs.get(fname) {
                    Ok(SsaVal::new(fn_ptr_ty, sig.symbol.clone()))
                } else {
                    Ok(SsaVal::new(
                        fn_ptr_ty,
                        format!("@{}", sanitize_ident(fname)),
                    ))
                }
            }
            MirExprKind::Var(vid) => {
                let p = self.var_ptr(*vid);
                let vty = self.vars[vid.0].ty.clone();
                Ok(self.load_from_place_ptr(&p, &vty, false))
            }
            MirExprKind::AddrOfPlace(place) => {
                let (p, _, _) = self.emit_place_ptr(place)?;
                Ok(SsaVal::new("ptr", p))
            }
            MirExprKind::Unary { op, expr: sub } => {
                let sv = self.emit_expr(sub)?;
                match op {
                    MirUnaryOp::Neg => {
                        let sc = self.coerce(sv, &dst_lty, !sub.ty.is_unsigned());
                        let t = self.next_tmp();
                        if dst_lty == "float" || dst_lty == "double" {
                            self.emit_line(format!("{t} = fneg {dst_lty} {}", sc.repr));
                        } else {
                            self.emit_line(format!("{t} = sub {dst_lty} 0, {}", sc.repr));
                        }
                        Ok(SsaVal::new(dst_lty, t))
                    }
                    MirUnaryOp::Not => {
                        let i1v = self.coerce_to_i1(sv);
                        let inv = self.next_tmp();
                        let b8 = self.next_tmp();
                        self.emit_line(format!("{inv} = xor i1 {i1v}, true"));
                        self.emit_line(format!("{b8} = zext i1 {inv} to i8"));
                        Ok(SsaVal::new("i8", b8))
                    }
                    MirUnaryOp::BitNot => {
                        let sc = self.coerce(sv, &dst_lty, !sub.ty.is_unsigned());
                        let t = self.next_tmp();
                        self.emit_line(format!("{t} = xor {dst_lty} {}, -1", sc.repr));
                        Ok(SsaVal::new(dst_lty, t))
                    }
                    MirUnaryOp::Deref => {
                        let pc = self.coerce(sv, "ptr", false);
                        Ok(self.load_from_place_ptr(&pc.repr, &expr.ty, false))
                    }
                }
            }
            MirExprKind::Binary { op, lhs, rhs } => self.emit_binary(*op, lhs, rhs, &expr.ty),
            MirExprKind::SeqEq {
                eq,
                lhs,
                rhs,
                elem_ty,
            } => self.emit_seq_eq(*eq, lhs, rhs, elem_ty),
            MirExprKind::CStrToSlice(sub) => {
                let sv = self.emit_expr(sub)?;
                let p = self.coerce_str_ptr(sv);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let chk_lbl = self.next_label("cstr_chk");
                let hdr_lbl = self.next_label("cstr_hdr");
                let step_lbl = self.next_label("cstr_step");
                let done_lbl = self.next_label("cstr_done");
                self.emit_term(format!("br label %{chk_lbl}"));
                self.start_block(&chk_lbl);
                let is_null = self.next_tmp();
                self.emit_line(format!("{is_null} = icmp eq ptr {}, null", p.repr));
                self.emit_term(format!(
                    "br i1 {is_null}, label %{done_lbl}, label %{hdr_lbl}"
                ));
                self.start_block(&hdr_lbl);
                let idx = self.next_tmp();
                let next_idx = self.next_tmp();
                self.emit_line(format!(
                    "{idx} = phi {usize_ty} [ 0, %{chk_lbl} ], [ {next_idx}, %{step_lbl} ]"
                ));
                let cp = self.next_tmp();
                let ch = self.next_tmp();
                let is_z = self.next_tmp();
                self.emit_line(format!(
                    "{cp} = getelementptr inbounds i8, ptr {}, {usize_ty} {idx}",
                    p.repr
                ));
                // `load volatile` keeps LLVM LoopIdiomRecognize from lowering the NUL scan to `@strlen` in `-nostdlib` builds.
                self.emit_line(format!("{ch} = load volatile i8, ptr {cp}, align 1"));
                self.emit_line(format!("{is_z} = icmp eq i8 {ch}, 0"));
                self.emit_term(format!(
                    "br i1 {is_z}, label %{done_lbl}, label %{step_lbl}"
                ));
                self.start_block(&step_lbl);
                self.emit_line(format!("{next_idx} = add {usize_ty} {idx}, 1"));
                self.emit_term(format!("br label %{hdr_lbl}"));
                self.start_block(&done_lbl);
                let final_len = self.next_tmp();
                self.emit_line(format!(
                    "{final_len} = phi {usize_ty} [ 0, %{chk_lbl} ], [ {idx}, %{hdr_lbl} ]"
                ));
                let s0 = self.next_tmp();
                let s1 = self.next_tmp();
                self.emit_line(format!(
                    "{s0} = insertvalue {sl_ty} poison, ptr {}, 0",
                    p.repr
                ));
                self.emit_line(format!(
                    "{s1} = insertvalue {sl_ty} {s0}, {usize_ty} {final_len}, 1"
                ));
                Ok(SsaVal::new(sl_ty, s1))
            }
            MirExprKind::Cast {
                expr: sub,
                target_ty,
            } => {
                let sv = self.emit_expr(sub)?;
                if *target_ty == MirType::Bool {
                    let i1v = self.coerce_to_i1(sv);
                    let b8 = self.next_tmp();
                    self.emit_line(format!("{b8} = zext i1 {i1v} to i8"));
                    return Ok(SsaVal::new("i8", b8));
                }
                let t_lty = self.ctx.llvm_ty(&target_ty.host_value_type());
                let signed = if sub.ty.is_float() {
                    !target_ty.is_unsigned()
                } else {
                    !sub.ty.is_unsigned()
                };
                Ok(self.coerce(sv, &t_lty, signed))
            }
            MirExprKind::StructLit {
                struct_name,
                fields,
                ..
            } => {
                let si = self.ctx.structs[struct_name].clone();
                let sty = si.llvm_name.clone();
                let mut cur = SsaVal::new(&sty, "zeroinitializer");
                for (fname, fexpr, fty) in fields {
                    let (fidx, _) = si.field_indices[fname];
                    let raw_v = self.emit_expr(fexpr)?;
                    let flty = self.ctx.llvm_ty(fty);
                    let mut coerced = self.coerce(raw_v, &flty, !fexpr.ty.is_unsigned());
                    if let MirType::EndianInt {
                        bits,
                        big_endian: true,
                        ..
                    } = fty
                    {
                        let sw = self.next_tmp();
                        self.emit_line(format!(
                            "{sw} = call i{bits} @llvm.bswap.i{bits}(i{bits} {})",
                            coerced.repr
                        ));
                        coerced = SsaVal::new(flty, sw);
                    }
                    let nxt = self.next_tmp();
                    self.emit_line(format!(
                        "{nxt} = insertvalue {sty} {}, {} {}, {fidx}",
                        cur.repr, coerced.ty, coerced.repr
                    ));
                    cur = SsaVal::new(&sty, nxt);
                }
                Ok(cur)
            }
            MirExprKind::ArrayLit { elems, elem_ty } => {
                let n = elems.len().max(1);
                let ety = self.ctx.llvm_ty(elem_ty);
                let aty = format!("[{n} x {ety}]");
                let mut cur = SsaVal::new(&aty, "zeroinitializer");
                for (i, e) in elems.iter().enumerate() {
                    let ev = self.emit_expr(e)?;
                    let mut ec = self.coerce(ev, &ety, !e.ty.is_unsigned());
                    if let MirType::EndianInt {
                        bits,
                        big_endian: true,
                        ..
                    } = elem_ty
                    {
                        let sw = self.next_tmp();
                        self.emit_line(format!(
                            "{sw} = call i{bits} @llvm.bswap.i{bits}(i{bits} {})",
                            ec.repr
                        ));
                        ec = SsaVal::new(ety.clone(), sw);
                    }
                    let nxt = self.next_tmp();
                    self.emit_line(format!(
                        "{nxt} = insertvalue {aty} {}, {ety} {}, {i}",
                        cur.repr, ec.repr
                    ));
                    cur = SsaVal::new(&aty, nxt);
                }
                Ok(cur)
            }
            MirExprKind::FieldAccess {
                base,
                struct_name,
                packed,
                is_ptr,
                field,
                raw_field_ty,
            } => {
                let si = self.ctx.structs[struct_name].clone();
                let (fidx, _) = si.field_indices[field];
                let sty = si.llvm_name.clone();
                if *is_ptr {
                    let bv = self.emit_expr(base)?;
                    let bp = self.coerce(bv, "ptr", false);
                    let gp = self.next_tmp();
                    self.emit_line(format!(
                        "{gp} = getelementptr inbounds {sty}, ptr {}, i32 0, i32 {fidx}",
                        bp.repr
                    ));
                    Ok(self.load_from_place_ptr(&gp, raw_field_ty, *packed))
                } else if let Some((bp, _, _)) = self.try_emit_rodata_const_ptr(base)? {
                    let gp = self.next_tmp();
                    self.emit_line(format!(
                        "{gp} = getelementptr inbounds {sty}, ptr {bp}, i32 0, i32 {fidx}"
                    ));
                    Ok(self.load_from_place_ptr(&gp, raw_field_ty, *packed))
                } else {
                    let bv = self.emit_expr(base)?;
                    let ext = self.next_tmp();
                    let flty = self.ctx.llvm_ty(raw_field_ty);
                    self.emit_line(format!("{ext} = extractvalue {sty} {}, {fidx}", bv.repr));
                    if let MirType::EndianInt {
                        bits,
                        big_endian: true,
                        ..
                    } = raw_field_ty
                    {
                        let sw = self.next_tmp();
                        self.emit_line(format!(
                            "{sw} = call i{bits} @llvm.bswap.i{bits}(i{bits} {ext})"
                        ));
                        Ok(SsaVal::new(flty, sw))
                    } else {
                        Ok(SsaVal::new(flty, ext))
                    }
                }
            }
            MirExprKind::SliceLen(sl_expr) => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(sl_expr)?;
                let len = self.next_tmp();
                self.emit_line(format!("{len} = extractvalue {sl_ty} {}, 1", sv.repr));
                Ok(SsaVal::new(usize_ty, len))
            }
            MirExprKind::ArrayLen(n) => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(SsaVal::new(usize_ty, format!("{n}")))
            }
            MirExprKind::Index {
                base,
                index,
                raw_elem_ty,
                is_slice,
            } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let iv = self.emit_expr(index)?;
                let ic = self.coerce(iv, &usize_ty, !index.ty.is_unsigned());
                let ety = self.ctx.llvm_ty(raw_elem_ty);
                if *is_slice {
                    let sl_ty = format!("{{ ptr, {usize_ty} }}");
                    let bv = self.emit_expr(base)?;
                    let dp = self.next_tmp();
                    let sl_len = self.next_tmp();
                    let inb = self.next_tmp();
                    let ep = self.next_tmp();
                    self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", bv.repr));
                    self.emit_line(format!("{sl_len} = extractvalue {sl_ty} {}, 1", bv.repr));
                    self.emit_line(format!("{inb} = icmp ult {usize_ty} {}, {sl_len}", ic.repr));
                    self.emit_bounds_check(&inb, 1);
                    self.emit_line(format!(
                        "{ep} = getelementptr {ety}, ptr {dp}, {usize_ty} {}",
                        ic.repr
                    ));
                    Ok(self.load_from_place_ptr(&ep, raw_elem_ty, false))
                } else if let Some((bp, bty, packed)) = self.try_emit_rodata_const_ptr(base)? {
                    if let MirType::Array { len, .. } = &bty {
                        if *len == 0 {
                            self.emit_bounds_check("false", 1);
                        } else {
                            let inb = self.next_tmp();
                            self.emit_line(format!(
                                "{inb} = icmp ult {usize_ty} {}, {len}",
                                ic.repr
                            ));
                            self.emit_bounds_check(&inb, 1);
                        }
                    }
                    let blty = self.ctx.llvm_ty(&bty);
                    let ep = self.next_tmp();
                    self.emit_line(format!(
                        "{ep} = getelementptr inbounds {blty}, ptr {bp}, {usize_ty} 0, {usize_ty} {}",
                        ic.repr
                    ));
                    Ok(self.load_from_place_ptr(&ep, raw_elem_ty, packed))
                } else {
                    if let MirType::Array { len, .. } = &base.ty {
                        if *len == 0 {
                            self.emit_bounds_check("false", 1);
                        } else {
                            let inb = self.next_tmp();
                            self.emit_line(format!(
                                "{inb} = icmp ult {usize_ty} {}, {len}",
                                ic.repr
                            ));
                            self.emit_bounds_check(&inb, 1);
                        }
                    }
                    let bv = self.emit_expr(base)?;
                    let (tmp, al) = self.emit_temp_alloca(&bv.ty);
                    let ep = self.next_tmp();
                    self.emit_line(format!(
                        "store {} {}, ptr {tmp}, align {al}",
                        bv.ty, bv.repr
                    ));
                    self.emit_line(format!(
                        "{ep} = getelementptr inbounds {}, ptr {tmp}, {usize_ty} 0, {usize_ty} {}",
                        bv.ty, ic.repr
                    ));
                    Ok(self.load_from_place_ptr(&ep, raw_elem_ty, false))
                }
            }
            MirExprKind::SubSlice {
                base,
                start,
                end,
                elem_ty,
                base_is_array,
            } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let ety = self.ctx.llvm_ty(elem_ty);
                let st_v = self.emit_expr(start)?;
                let en_v = self.emit_expr(end)?;
                let st_c = self.coerce(st_v, &usize_ty, !start.ty.is_unsigned());
                let en_c = self.coerce(en_v, &usize_ty, !end.ty.is_unsigned());

                let (base_ptr, base_len) = if *base_is_array
                    || matches!(base.ty, MirType::Array { .. })
                {
                    let arr_len = match &base.ty {
                        MirType::Array { len, .. } => *len,
                        _ => 0,
                    };
                    let p = if let MirExprKind::Var(vid) = &base.kind {
                        let ap = self.var_ptr(*vid);
                        let aty = self.ctx.llvm_ty(&base.ty);
                        let gp = self.next_tmp();
                        self.emit_line(format!(
                                "{gp} = getelementptr inbounds {aty}, ptr {ap}, {usize_ty} 0, {usize_ty} 0"
                            ));
                        gp
                    } else if let Some((ap, aty_mir, _)) = self.try_emit_rodata_const_ptr(base)? {
                        let aty = self.ctx.llvm_ty(&aty_mir);
                        let gp = self.next_tmp();
                        self.emit_line(format!(
                                "{gp} = getelementptr inbounds {aty}, ptr {ap}, {usize_ty} 0, {usize_ty} 0"
                            ));
                        gp
                    } else {
                        let bv = self.emit_expr(base)?;
                        let (tmp, al) = self.emit_temp_alloca(&bv.ty);
                        self.emit_line(format!(
                            "store {} {}, ptr {tmp}, align {al}",
                            bv.ty, bv.repr
                        ));
                        tmp
                    };
                    (p, format!("{arr_len}"))
                } else {
                    let bv = self.emit_expr(base)?;
                    let p = self.next_tmp();
                    let bl = self.next_tmp();
                    self.emit_line(format!("{p} = extractvalue {sl_ty} {}, 0", bv.repr));
                    self.emit_line(format!("{bl} = extractvalue {sl_ty} {}, 1", bv.repr));
                    (p, bl)
                };

                // Enforce `start <= end && end <= base_len` via uniform `@__syn_panic(i8 2)` trap.
                let ok_ord = self.next_tmp();
                let ok_end = self.next_tmp();
                let ok_all = self.next_tmp();
                self.emit_line(format!(
                    "{ok_ord} = icmp ule {usize_ty} {}, {}",
                    st_c.repr, en_c.repr
                ));
                self.emit_line(format!(
                    "{ok_end} = icmp ule {usize_ty} {}, {base_len}",
                    en_c.repr
                ));
                self.emit_line(format!("{ok_all} = and i1 {ok_ord}, {ok_end}"));
                self.emit_bounds_check(&ok_all, 2);

                let is_null = self.next_tmp();
                let gep_p = self.next_tmp();
                let sub_p = self.next_tmp();
                let sub_len = self.next_tmp();
                let r0 = self.next_tmp();
                let r1 = self.next_tmp();
                self.emit_line(format!("{is_null} = icmp eq ptr {base_ptr}, null"));
                self.emit_line(format!(
                    "{gep_p} = getelementptr {ety}, ptr {base_ptr}, {usize_ty} {}",
                    st_c.repr
                ));
                self.emit_line(format!(
                    "{sub_p} = select i1 {is_null}, ptr null, ptr {gep_p}"
                ));
                self.emit_line(format!(
                    "{sub_len} = sub {usize_ty} {}, {}",
                    en_c.repr, st_c.repr
                ));
                self.emit_line(format!("{r0} = insertvalue {sl_ty} poison, ptr {sub_p}, 0"));
                self.emit_line(format!(
                    "{r1} = insertvalue {sl_ty} {r0}, {usize_ty} {sub_len}, 1"
                ));
                Ok(SsaVal::new(sl_ty, r1))
            }
            MirExprKind::ArrayToSlice {
                array_place, len, ..
            } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let (ap, aty_mir, _) = self.emit_place_ptr(array_place)?;
                let aty = self.ctx.llvm_ty(&aty_mir);
                let ep = self.next_tmp();
                let r0 = self.next_tmp();
                let r1 = self.next_tmp();
                self.emit_line(format!(
                    "{ep} = getelementptr inbounds {aty}, ptr {ap}, {usize_ty} 0, {usize_ty} 0"
                ));
                self.emit_line(format!("{r0} = insertvalue {sl_ty} poison, ptr {ep}, 0"));
                self.emit_line(format!(
                    "{r1} = insertvalue {sl_ty} {r0}, {usize_ty} {len}, 1"
                ));
                Ok(SsaVal::new(sl_ty, r1))
            }
            MirExprKind::FromBytes { struct_name, slice } => {
                let call_kw = self.call_kw();
                let sv = self.emit_expr(slice)?;
                let sname = sanitize_ident(struct_name);
                let sty = format!("%struct.{sname}");
                let rty = format!("{{ {sty}, i32 }}");
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {rty} @__syn_from_bytes_{sname}({} {})",
                    sv.ty, sv.repr
                ));
                Ok(SsaVal::new(rty, res))
            }
            MirExprKind::PrimRead(prop) => self.emit_prim_read(prop),
            MirExprKind::CallDirect {
                callee,
                is_extern_c,
                args,
            } => self.emit_direct_call(callee, *is_extern_c, args, &expr.ty),
            MirExprKind::ResultIsErr(vid) => {
                let p = self.var_ptr(*vid);
                let rty = self.ctx.llvm_ty(&self.vars[vid.0].ty);
                let rv = self.next_tmp();
                let tag = self.next_tmp();
                let ne = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{rv} = load {rty}, ptr {p}, align 4"));
                self.emit_line(format!("{tag} = extractvalue {rty} {rv}, 1"));
                self.emit_line(format!("{ne} = icmp ne i32 {tag}, 0"));
                self.emit_line(format!("{b8} = zext i1 {ne} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirExprKind::ResultErr(vid) => {
                let p = self.var_ptr(*vid);
                let rty = self.ctx.llvm_ty(&self.vars[vid.0].ty);
                let rv = self.next_tmp();
                let tag = self.next_tmp();
                let raw_e = self.next_tmp();
                self.emit_line(format!("{rv} = load {rty}, ptr {p}, align 4"));
                self.emit_line(format!("{tag} = extractvalue {rty} {rv}, 1"));
                self.emit_line(format!("{raw_e} = sub i32 {tag}, 1"));
                Ok(self.coerce(SsaVal::new("i32", raw_e), &dst_lty, true))
            }
            MirExprKind::ResultVal(vid) => {
                let p = self.var_ptr(*vid);
                let vmir = self.vars[vid.0].ty.clone();
                let rty = self.ctx.llvm_ty(&vmir);
                let (ok_lty, ok_signed) = match &vmir {
                    MirType::Result { ok, .. } => {
                        if ok.is_void() {
                            ("i32".to_string(), false)
                        } else {
                            (self.ctx.llvm_ty(ok), !ok.is_unsigned())
                        }
                    }
                    _ => (dst_lty.clone(), !expr.ty.is_unsigned()),
                };
                let rv = self.next_tmp();
                let val = self.next_tmp();
                self.emit_line(format!("{rv} = load {rty}, ptr {p}, align 4"));
                self.emit_line(format!("{val} = extractvalue {rty} {rv}, 0"));
                Ok(self.coerce(SsaVal::new(ok_lty, val), &dst_lty, ok_signed))
            }
            MirExprKind::NowTicks => {
                let call_kw = self.call_kw();
                let t = self.next_tmp();
                self.emit_line(format!("{t} = {call_kw} i32 @__syn_now_ms()"));
                Ok(self.coerce(SsaVal::new("i32", t), &dst_lty, false))
            }
            MirExprKind::SysQuery(q) => {
                let call_kw = self.call_kw();
                let v = match q {
                    MirSysQuery::CurrentPrio => {
                        let tmp = self.next_tmp();
                        self.emit_line(format!("{tmp} = {call_kw} i32 @__syn_current_prio()"));
                        tmp
                    }
                    MirSysQuery::SleepDepth => {
                        let m = self.next_tmp();
                        let dl = self.next_tmp();
                        let any = self.next_tmp();
                        let locked = self.next_tmp();
                        let out_v = self.next_tmp();
                        let pm = if self.ctx.spec.isa.is_avr() {
                            let pm = self.next_tmp();
                            self.emit_line(format!("{pm} = {call_kw} i32 @__syn_crit_enter()"));
                            Some(pm)
                        } else {
                            None
                        };
                        self.emit_line(format!(
                            "{m} = load volatile i32, ptr @__syn_sleep_lock_mask, align 4"
                        ));
                        self.emit_line(format!(
                            "{dl} = load volatile i32, ptr @__syn_dma_locks, align 4"
                        ));
                        if let Some(pm) = pm {
                            self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                        }
                        self.emit_line(format!("{any} = or i32 {m}, {dl}"));
                        self.emit_line(format!("{locked} = icmp ne i32 {any}, 0"));
                        self.emit_line(format!("{out_v} = select i1 {locked}, i32 1, i32 2"));
                        out_v
                    }
                    MirSysQuery::WfiCount => self.emit_avr_safe_i32_load("@__syn_wfi_count", true),
                };
                Ok(self.coerce(SsaVal::new("i32", v), &dst_lty, false))
            }
            MirExprKind::LeaseGet { lease, .. } => {
                let lv = self.emit_expr(lease)?;
                let p = self.next_tmp();
                self.emit_line(format!(
                    "{p} = extractvalue {{ ptr, ptr, i8 }} {}, 0",
                    lv.repr
                ));
                Ok(SsaVal::new("ptr", p))
            }
            MirExprKind::LeaseAddr { lease, .. } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let lv = self.emit_expr(lease)?;
                let p = self.next_tmp();
                let a = self.next_tmp();
                self.emit_line(format!(
                    "{p} = extractvalue {{ ptr, ptr, i8 }} {}, 0",
                    lv.repr
                ));
                self.emit_line(format!("{a} = ptrtoint ptr {p} to {usize_ty}"));
                Ok(SsaVal::new(usize_ty, a))
            }
            MirExprKind::RomCallU32 {
                addr,
                stack_bound,
                args,
            } => {
                let rv = self
                    .emit_rom_call_invocation(RomWrapperKind::U32, addr, *stack_bound, args)?
                    .ok_or_else(|| {
                        CodegenError::new("llvm-codegen", "rom_call_u32 wrapper returned void")
                    })?;
                Ok(self.coerce(rv, &dst_lty, false))
            }
        }
    }

    pub(super) fn emit_prim_read(&mut self, prop: &MirPrimReadOp) -> Result<SsaVal, CodegenError> {
        let call_kw = self.call_kw();
        match prop {
            MirPrimReadOp::WatchRead { target, elem_ty } => {
                let pk = MirPrimKind::Watch {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let wptr = self.resolve_prim_ptr(target)?;
                let ety = self.ctx.llvm_ty(elem_ty);
                let v = self.next_tmp();
                self.emit_line(format!(
                    "{v} = {call_kw} {ety} @__syn_watch_read_{id}(ptr {wptr})"
                ));
                Ok(SsaVal::new(ety, v))
            }
            MirPrimReadOp::GuardedGet { target, elem_ty } => {
                let pk = MirPrimKind::Guarded {
                    elem: Box::new(elem_ty.clone()),
                };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let gptr = self.resolve_prim_ptr(target)?;
                let ety = self.ctx.llvm_ty(elem_ty);
                let contended = match target {
                    MirPrimTarget::Global(name) => {
                        self.ctx.vis.isr_contended_globals.contains(name)
                    }
                    MirPrimTarget::PtrExpr(_) => self.ctx.vis.is_prim_family_contended(&pk),
                };
                let need_mask = contended && !self.in_isr && self.crit_depth == 0;
                let pm = if need_mask {
                    let pm = self.next_tmp();
                    self.emit_line(format!("{pm} = {call_kw} i32 @__syn_crit_enter()"));
                    Some(pm)
                } else {
                    None
                };
                let vp = self.next_tmp();
                self.emit_line(format!(
                    "{vp} = getelementptr inbounds {pty}, ptr {gptr}, i32 0, i32 0"
                ));
                let v = self.next_tmp();
                self.emit_line(format!("{v} = load volatile {ety}, ptr {vp}, align 1"));
                if let Some(pm) = pm {
                    self.emit_line(format!("{call_kw} void @__syn_crit_exit(i32 {pm})"));
                }
                if let MirType::EndianInt {
                    bits,
                    big_endian: true,
                    ..
                } = elem_ty
                {
                    let sw = self.next_tmp();
                    self.emit_line(format!(
                        "{sw} = call i{bits} @llvm.bswap.i{bits}(i{bits} {v})"
                    ));
                    Ok(SsaVal::new(ety, sw))
                } else {
                    Ok(SsaVal::new(ety, v))
                }
            }
            MirPrimReadOp::WatchGen { target, elem_ty } => {
                let pk = MirPrimKind::Watch {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let wptr = self.resolve_prim_ptr(target)?;
                let g = self.next_tmp();
                self.emit_line(format!(
                    "{g} = {call_kw} i32 @__syn_watch_gen_{id}(ptr {wptr})"
                ));
                Ok(SsaVal::new("i32", g))
            }
            MirPrimReadOp::InboxTryRecv {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                self.emit_try_recv_expr(target, &pk, elem_ty)
            }
            MirPrimReadOp::InboxTrySend {
                target,
                elem_ty,
                cap,
                value,
                boost,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let qptr = self.resolve_prim_ptr(target)?;
                let ety = self.ctx.llvm_ty(elem_ty);
                let vv = self.emit_expr(value)?;
                let vc = self.coerce(vv, &ety, !value.ty.is_unsigned());
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_q_push_{id}(ptr {qptr}, {ety} {}, i32 {b})",
                    vc.repr
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::InboxLen {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let qptr = self.resolve_prim_ptr(target)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", cnt), &usize_ty, false))
            }
            MirPrimReadOp::InboxIsEmpty {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let qptr = self.resolve_prim_ptr(target)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let emp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{emp} = icmp eq i32 {cnt}, 0"));
                self.emit_line(format!("{b8} = zext i1 {emp} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::InboxIsFull {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Inbox {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let qptr = self.resolve_prim_ptr(target)?;
                let c = (*cap).max(1) as i32;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {qptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let fl = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{fl} = icmp sge i32 {cnt}, {c}"));
                self.emit_line(format!("{b8} = zext i1 {fl} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::SignalIsSet { target } => {
                let sptr = self.resolve_prim_ptr(target)?;
                let fp = self.next_tmp();
                let f = self.next_tmp();
                self.emit_line(format!(
                    "{fp} = getelementptr inbounds %prim.signal, ptr {sptr}, i32 0, i32 0"
                ));
                self.emit_line(format!("{f} = load volatile i8, ptr {fp}, align 1"));
                Ok(SsaVal::new("i8", f))
            }
            MirPrimReadOp::SignalTryWait { target } => {
                let id = sanitize_ident(&MirPrimKind::Signal.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_sig_poll_{id}(ptr {sptr}, ptr {task_arg})"
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::WaitQueueWakeOne { target, boost } => {
                let wptr = self.resolve_prim_ptr(target)?;
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_wake_one_mask(ptr {wptr}, i32 {b})"
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::PingPongDmaBuf {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let pp_ty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let pptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let n = (*cap).max(1);
                let wh_p = self.next_tmp();
                let wh = self.next_tmp();
                let is_b0 = self.next_tmp();
                let b0 = self.next_tmp();
                let b1 = self.next_tmp();
                let buf_p = self.next_tmp();
                let s0 = self.next_tmp();
                let s1 = self.next_tmp();
                self.emit_line(format!(
                    "{wh_p} = getelementptr inbounds {pp_ty}, ptr {pptr}, i32 0, i32 0"
                ));
                self.emit_line(format!("{wh} = load i8, ptr {wh_p}, align 1"));
                self.emit_line(format!("{is_b0} = icmp eq i8 {wh}, 0"));
                self.emit_line(format!(
                    "{b0} = getelementptr inbounds {pp_ty}, ptr {pptr}, i32 0, i32 5, i32 0"
                ));
                self.emit_line(format!(
                    "{b1} = getelementptr inbounds {pp_ty}, ptr {pptr}, i32 0, i32 6, i32 0"
                ));
                self.emit_line(format!("{buf_p} = select i1 {is_b0}, ptr {b0}, ptr {b1}"));
                self.emit_line(format!("{s0} = insertvalue {sl_ty} poison, ptr {buf_p}, 0"));
                self.emit_line(format!(
                    "{s1} = insertvalue {sl_ty} {s0}, {usize_ty} {n}, 1"
                ));
                Ok(SsaVal::new(sl_ty, s1))
            }
            MirPrimReadOp::PingPongBaseAddr {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::PingPong {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let pp_ty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let pptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let b0 = self.next_tmp();
                let addr = self.next_tmp();
                self.emit_line(format!(
                    "{b0} = getelementptr inbounds {pp_ty}, ptr {pptr}, i32 0, i32 5, i32 0"
                ));
                self.emit_line(format!("{addr} = ptrtoint ptr {b0} to {usize_ty}"));
                Ok(SsaVal::new(usize_ty, addr))
            }
            MirPrimReadOp::PoolTryAlloc {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let rty = "{ { ptr, ptr, i8 }, i32 }".to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {rty} @__syn_pool_try_alloc_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new(rty, res))
            }
            MirPrimReadOp::PoolAvail {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {usize_ty} @__syn_pool_avail_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new(usize_ty, res))
            }
            MirPrimReadOp::PoolInUse {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {usize_ty} @__syn_pool_in_use_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new(usize_ty, res))
            }
            MirPrimReadOp::PoolIsExhausted {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} i8 @__syn_pool_is_exhausted_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new("i8", res))
            }
            MirPrimReadOp::PoolCap {
                target,
                elem_ty,
                cap,
            } => {
                let pk = MirPrimKind::Pool {
                    elem: Box::new(elem_ty.clone()),
                    cap: *cap,
                };
                let id = sanitize_ident(&pk.mangle_name());
                let pptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {usize_ty} @__syn_pool_cap_{id}(ptr {pptr})"
                ));
                Ok(SsaVal::new(usize_ty, res))
            }
            MirPrimReadOp::DmaSlotPark {
                target,
                elem_ty,
                lease,
            } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let lv = self.emit_expr(lease)?;
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {{ i32, i32 }} @__syn_slot_park_{id}(ptr {sptr}, {{ ptr, ptr, i8 }} {})",
                    lv.repr
                ));
                Ok(SsaVal::new("{ i32, i32 }", res))
            }
            MirPrimReadOp::DmaSlotTake { target, elem_ty } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let rty = "{ { ptr, ptr, i8 }, i32 }".to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {rty} @__syn_slot_take_{id}(ptr {sptr})"
                ));
                Ok(SsaVal::new(rty, res))
            }
            MirPrimReadOp::DmaSlotRelease {
                target,
                elem_ty,
                boost,
            } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let b = boost.map_or(-1i8, |p| p.as_u8() as i8);
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} i8 @__syn_slot_release_{id}(ptr {sptr}, i8 {b})"
                ));
                Ok(SsaVal::new("i8", res))
            }
            MirPrimReadOp::DmaSlotIsBusy { target, elem_ty } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} i8 @__syn_slot_is_busy_{id}(ptr {sptr})"
                ));
                Ok(SsaVal::new("i8", res))
            }
            MirPrimReadOp::DmaSlotIsEmpty { target, elem_ty } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} i8 @__syn_slot_is_empty_{id}(ptr {sptr})"
                ));
                Ok(SsaVal::new("i8", res))
            }
            MirPrimReadOp::DmaSlotAddr { target, elem_ty } => {
                let pk = MirPrimKind::DmaSlot {
                    elem: Box::new(elem_ty.clone()),
                };
                let id = sanitize_ident(&pk.mangle_name());
                let sptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res = self.next_tmp();
                self.emit_line(format!(
                    "{res} = {call_kw} {usize_ty} @__syn_slot_addr_{id}(ptr {sptr})"
                ));
                Ok(SsaVal::new(usize_ty, res))
            }
            MirPrimReadOp::ByteRingPush {
                target,
                cap,
                src,
                boost,
            } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(src)?;
                let sc = self.coerce(sv, &sl_ty, false);
                let sp = self.next_tmp();
                let sl = self.next_tmp();
                self.emit_line(format!("{sp} = extractvalue {sl_ty} {}, 0", sc.repr));
                self.emit_line(format!("{sl} = extractvalue {sl_ty} {}, 1", sc.repr));
                let sl32 = self.coerce(SsaVal::new(usize_ty.clone(), sl), "i32", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let w32 = self.next_tmp();
                self.emit_line(format!(
                    "{w32} = {call_kw} i32 @__syn_ring_push_{id}(ptr {rptr}, ptr {sp}, i32 {}, i32 {b})",
                    sl32.repr
                ));
                Ok(self.coerce(SsaVal::new("i32", w32), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingPushByte {
                target,
                cap,
                byte,
                boost,
            } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let bv = self.emit_expr(byte)?;
                let b8 = self.coerce(bv, "i8", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_ring_push_byte_{id}(ptr {rptr}, i8 {}, i32 {b})",
                    b8.repr
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::ByteRingGrantWrite { target, cap, len } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let lv = self.emit_expr(len)?;
                let l32 = self.coerce(lv, "i32", false);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty();
                let res_ty = format!("{{ {{ ptr, {usize_ty} }}, i32 }}");
                let rv = self.next_tmp();
                self.emit_line(format!(
                    "{rv} = {call_kw} {res_ty} @__syn_ring_grant_write_{id}(ptr {rptr}, i32 {})",
                    l32.repr
                ));
                Ok(SsaVal::new(res_ty, rv))
            }
            MirPrimReadOp::ByteRingReadSlice { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sl = self.next_tmp();
                self.emit_line(format!(
                    "{sl} = {call_kw} {sl_ty} @__syn_ring_read_slice_{id}(ptr {rptr})"
                ));
                Ok(SsaVal::new(sl_ty, sl))
            }
            MirPrimReadOp::ByteRingPop { target, cap, dst } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let dv = self.emit_expr(dst)?;
                let dc = self.coerce(dv, &sl_ty, false);
                let dp = self.next_tmp();
                let dl = self.next_tmp();
                self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", dc.repr));
                self.emit_line(format!("{dl} = extractvalue {sl_ty} {}, 1", dc.repr));
                let dl32 = self.coerce(SsaVal::new(usize_ty.clone(), dl), "i32", false);
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let n32 = self.next_tmp();
                self.emit_line(format!(
                    "{n32} = {call_kw} i32 @__syn_ring_pop_{id}(ptr {rptr}, ptr {dp}, i32 {}, ptr {task_arg})",
                    dl32.repr
                ));
                Ok(self.coerce(SsaVal::new("i32", n32), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingTryPopByte { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let rv = self.next_tmp();
                self.emit_line(format!(
                    "{rv} = {call_kw} {{ i8, i32 }} @__syn_ring_try_pop_byte_{id}(ptr {rptr}, ptr {task_arg})"
                ));
                Ok(SsaVal::new("{ i8, i32 }", rv))
            }
            MirPrimReadOp::ByteRingLen { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let rptr = self.resolve_prim_ptr(target)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {rptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", cnt), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingFreeSpace { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let fs = self.next_tmp();
                self.emit_line(format!(
                    "{fs} = {call_kw} i32 @__syn_ring_free_space_{id}(ptr {rptr})"
                ));
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", fs), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingIsEmpty { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let pty = format!("%prim.{}", sanitize_ident(&pk.mangle_name()));
                let rptr = self.resolve_prim_ptr(target)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds {pty}, ptr {rptr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_prim_family_contended(&pk);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let emp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{emp} = icmp eq i32 {cnt}, 0"));
                self.emit_line(format!("{b8} = zext i1 {emp} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::ByteRingIsFull { target, cap } => {
                let pk = MirPrimKind::ByteRing { cap: *cap };
                let id = sanitize_ident(&pk.mangle_name());
                let rptr = self.resolve_prim_ptr(target)?;
                let fs = self.next_tmp();
                let full = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!(
                    "{fs} = {call_kw} i32 @__syn_ring_free_space_{id}(ptr {rptr})"
                ));
                self.emit_line(format!("{full} = icmp eq i32 {fs}, 0"));
                self.emit_line(format!("{b8} = zext i1 {full} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::InboxViewTryRecv { view, elem_ty } => {
                let (hdr, items_base, cap32) = self.emit_inbox_view_parts(view)?;
                let key = sanitize_ident(&elem_ty.mangle_name());
                let ety = self.ctx.llvm_ty(elem_ty);
                let rty = format!("{{ {ety}, i32 }}");
                let (tmp, al) = self.emit_temp_alloca(&ety);
                self.emit_line(format!(
                    "store {ety} zeroinitializer, ptr {tmp}, align {al}"
                ));
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_q_view_pop_{key}(ptr {hdr}, ptr {items_base}, i32 {cap32}, ptr {tmp}, ptr {task_arg})"
                ));
                let val = self.next_tmp();
                let got = self.next_tmp();
                let tag = self.next_tmp();
                let r0 = self.next_tmp();
                let r1 = self.next_tmp();
                self.emit_line(format!("{val} = load {ety}, ptr {tmp}, align {al}"));
                self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
                self.emit_line(format!("{tag} = select i1 {got}, i32 0, i32 1"));
                self.emit_line(format!("{r0} = insertvalue {rty} poison, {ety} {val}, 0"));
                self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 {tag}, 1"));
                Ok(SsaVal::new(rty, r1))
            }
            MirPrimReadOp::InboxViewTrySend {
                view,
                elem_ty,
                value,
                boost,
            } => {
                let (hdr, items_base, cap32) = self.emit_inbox_view_parts(view)?;
                let key = sanitize_ident(&elem_ty.mangle_name());
                let ety = self.ctx.llvm_ty(elem_ty);
                let vv = self.emit_expr(value)?;
                let vc = self.coerce(vv, &ety, !value.ty.is_unsigned());
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_q_view_push_{key}(ptr {hdr}, ptr {items_base}, i32 {cap32}, {ety} {}, i32 {b})",
                    vc.repr
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::InboxViewLen { view, elem_ty } => {
                let (hdr, _, _) = self.emit_inbox_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_inbox_view_contended(elem_ty);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", cnt), &usize_ty, false))
            }
            MirPrimReadOp::InboxViewCap { view, .. } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let vty = format!("{{ ptr, ptr, {usize_ty} }}");
                let vv = self.emit_expr(view)?;
                let cap_u = self.next_tmp();
                self.emit_line(format!("{cap_u} = extractvalue {vty} {}, 2", vv.repr));
                Ok(SsaVal::new(usize_ty, cap_u))
            }
            MirPrimReadOp::InboxViewIsEmpty { view, elem_ty } => {
                let (hdr, _, _) = self.emit_inbox_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_inbox_view_contended(elem_ty);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let emp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{emp} = icmp eq i32 {cnt}, 0"));
                self.emit_line(format!("{b8} = zext i1 {emp} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::InboxViewIsFull { view, elem_ty } => {
                let (hdr, _, cap32) = self.emit_inbox_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.inbox_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_inbox_view_contended(elem_ty);
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let fl = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{fl} = icmp sge i32 {cnt}, {cap32}"));
                self.emit_line(format!("{b8} = zext i1 {fl} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::ByteRingViewPush { view, src, boost } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(src)?;
                let sc = self.coerce(sv, &sl_ty, false);
                let sp = self.next_tmp();
                let sl = self.next_tmp();
                self.emit_line(format!("{sp} = extractvalue {sl_ty} {}, 0", sc.repr));
                self.emit_line(format!("{sl} = extractvalue {sl_ty} {}, 1", sc.repr));
                let sl32 = self.coerce(SsaVal::new(usize_ty.clone(), sl), "i32", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let w32 = self.next_tmp();
                self.emit_line(format!(
                    "{w32} = {call_kw} i32 @__syn_ring_view_push(ptr {hdr}, i32 {cap32}, ptr {sp}, i32 {}, i32 {b})",
                    sl32.repr
                ));
                Ok(self.coerce(SsaVal::new("i32", w32), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingViewPushByte { view, byte, boost } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let bv = self.emit_expr(byte)?;
                let b8 = self.coerce(bv, "i8", false);
                let b = boost.map_or(-1i32, |p| p.as_u8() as i32);
                let ok = self.next_tmp();
                self.emit_line(format!(
                    "{ok} = {call_kw} i8 @__syn_ring_view_push_byte(ptr {hdr}, i32 {cap32}, i8 {}, i32 {b})",
                    b8.repr
                ));
                Ok(SsaVal::new("i8", ok))
            }
            MirPrimReadOp::ByteRingViewGrantWrite { view, len } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let res_slice_ty = format!("{{ {{ ptr, {usize_ty} }}, i32 }}");
                let lv = self.emit_expr(len)?;
                let l32 = self.coerce(lv, "i32", false);
                let rv = self.next_tmp();
                self.emit_line(format!(
                    "{rv} = {call_kw} {res_slice_ty} @__syn_ring_view_grant_write(ptr {hdr}, i32 {cap32}, i32 {})",
                    l32.repr
                ));
                Ok(SsaVal::new(res_slice_ty, rv))
            }
            MirPrimReadOp::ByteRingViewReadSlice { view } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.next_tmp();
                self.emit_line(format!(
                    "{sv} = {call_kw} {sl_ty} @__syn_ring_view_read_slice(ptr {hdr}, i32 {cap32})"
                ));
                Ok(SsaVal::new(sl_ty, sv))
            }
            MirPrimReadOp::ByteRingViewPop { view, dst } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let dv = self.emit_expr(dst)?;
                let dc = self.coerce(dv, &sl_ty, false);
                let dp = self.next_tmp();
                let dl = self.next_tmp();
                self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", dc.repr));
                self.emit_line(format!("{dl} = extractvalue {sl_ty} {}, 1", dc.repr));
                let dl32 = self.coerce(SsaVal::new(usize_ty.clone(), dl), "i32", false);
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let n32 = self.next_tmp();
                self.emit_line(format!(
                    "{n32} = {call_kw} i32 @__syn_ring_view_pop(ptr {hdr}, i32 {cap32}, ptr {dp}, i32 {}, ptr {task_arg})",
                    dl32.repr
                ));
                Ok(self.coerce(SsaVal::new("i32", n32), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingViewTryPopByte { view } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let task_arg = if self.coroutine.is_some() {
                    "%__task"
                } else {
                    "null"
                };
                let rv = self.next_tmp();
                self.emit_line(format!(
                    "{rv} = {call_kw} {{ i8, i32 }} @__syn_ring_view_try_pop_byte(ptr {hdr}, i32 {cap32}, ptr {task_arg})"
                ));
                Ok(SsaVal::new("{ i8, i32 }", rv))
            }
            MirPrimReadOp::ByteRingViewLen { view } => {
                let (hdr, _) = self.emit_byte_ring_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.byte_ring_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_byte_ring_view_contended();
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", cnt), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingViewCap { view } => {
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                let vty = format!("{{ ptr, {usize_ty} }}");
                let vv = self.emit_expr(view)?;
                let cap_u = self.next_tmp();
                self.emit_line(format!("{cap_u} = extractvalue {vty} {}, 1", vv.repr));
                Ok(SsaVal::new(usize_ty, cap_u))
            }
            MirPrimReadOp::ByteRingViewFreeSpace { view } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let fs = self.next_tmp();
                self.emit_line(format!(
                    "{fs} = {call_kw} i32 @__syn_ring_view_free_space(ptr {hdr}, i32 {cap32})"
                ));
                let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
                Ok(self.coerce(SsaVal::new("i32", fs), &usize_ty, false))
            }
            MirPrimReadOp::ByteRingViewIsEmpty { view } => {
                let (hdr, _) = self.emit_byte_ring_view_parts(view)?;
                let cnt_p = self.next_tmp();
                self.emit_line(format!(
                    "{cnt_p} = getelementptr inbounds %prim.byte_ring_hdr, ptr {hdr}, i32 0, i32 2"
                ));
                let contended = self.ctx.vis.is_byte_ring_view_contended();
                let cnt = self.emit_avr_safe_i32_load(&cnt_p, contended);
                let emp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!("{emp} = icmp eq i32 {cnt}, 0"));
                self.emit_line(format!("{b8} = zext i1 {emp} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
            MirPrimReadOp::ByteRingViewIsFull { view } => {
                let (hdr, cap32) = self.emit_byte_ring_view_parts(view)?;
                let fs = self.next_tmp();
                let full = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!(
                    "{fs} = {call_kw} i32 @__syn_ring_view_free_space(ptr {hdr}, i32 {cap32})"
                ));
                self.emit_line(format!("{full} = icmp eq i32 {fs}, 0"));
                self.emit_line(format!("{b8} = zext i1 {full} to i8"));
                Ok(SsaVal::new("i8", b8))
            }
        }
    }

    pub(super) fn emit_try_recv_expr(
        &mut self,
        target: &MirPrimTarget,
        pk: &MirPrimKind,
        elem_ty: &MirType,
    ) -> Result<SsaVal, CodegenError> {
        let id = sanitize_ident(&pk.mangle_name());
        let qptr = self.resolve_prim_ptr(target)?;
        let ety = self.ctx.llvm_ty(elem_ty);
        let rty = format!("{{ {ety}, i32 }}");
        let (tmp, al) = self.emit_temp_alloca(&ety);
        self.emit_line(format!(
            "store {ety} zeroinitializer, ptr {tmp}, align {al}"
        ));
        let task_arg = if self.coroutine.is_some() {
            "%__task"
        } else {
            "null"
        };
        let call_kw = self.call_kw();
        let ok = self.next_tmp();
        self.emit_line(format!(
            "{ok} = {call_kw} i8 @__syn_q_pop_{id}(ptr {qptr}, ptr {tmp}, ptr {task_arg})"
        ));
        let val = self.next_tmp();
        let got = self.next_tmp();
        let tag = self.next_tmp();
        let r0 = self.next_tmp();
        let r1 = self.next_tmp();
        self.emit_line(format!("{val} = load {ety}, ptr {tmp}, align {al}"));
        self.emit_line(format!("{got} = icmp ne i8 {ok}, 0"));
        self.emit_line(format!("{tag} = select i1 {got}, i32 0, i32 1"));
        self.emit_line(format!("{r0} = insertvalue {rty} poison, {ety} {val}, 0"));
        self.emit_line(format!("{r1} = insertvalue {rty} {r0}, i32 {tag}, 1"));
        Ok(SsaVal::new(rty, r1))
    }

    pub(super) fn coerce_str_ptr(&mut self, val: SsaVal) -> SsaVal {
        if val.ty.starts_with("{ ptr,") {
            let p = self.next_tmp();
            self.emit_line(format!("{p} = extractvalue {} {}, 0", val.ty, val.repr));
            SsaVal::new("ptr", p)
        } else {
            self.coerce(val, "ptr", false)
        }
    }

    pub(super) fn emit_runtime_assert(&mut self, cond_val: SsaVal) {
        let call_kw = self.call_kw();
        let c1 = self.coerce_to_i1(cond_val);
        let ok_lbl = self.next_label("assert_ok");
        let fail_lbl = self.next_label("assert_fail");
        self.emit_term(format!("br i1 {c1}, label %{ok_lbl}, label %{fail_lbl}"));
        self.start_block(&fail_lbl);
        let idx = self.ctx.intern_string("ASSERTION FAILED\n");
        self.emit_line(format!("{call_kw} void @__syn_uart_puts(ptr @.str.{idx})"));
        self.emit_line(format!("{call_kw} void @__syn_platform_exit(i32 1)"));
        self.emit_term("unreachable");
        self.start_block(&ok_lbl);
    }

    pub(super) fn emit_direct_call(
        &mut self,
        callee: &str,
        is_extern_c: bool,
        args: &[MirExpr],
        ret_mir_ty: &MirType,
    ) -> Result<SsaVal, CodegenError> {
        let call_kw = self.call_kw();
        // Indirect call through a local/parameter/temp `MirVar` of `MirType::FnPtr`
        if callee != "assert" {
            if let Some((vid, params, ret, error_spec)) =
                self.vars.iter().rev().find_map(|v| match &v.ty {
                    MirType::FnPtr {
                        params,
                        ret,
                        error_spec,
                    } if v.name == callee => Some((
                        v.id,
                        params.clone(),
                        (**ret).clone(),
                        (**error_spec).clone(),
                    )),
                    _ => None,
                })
            {
                let fp_ty = MirType::FnPtr {
                    params: params.clone(),
                    ret: Box::new(ret.clone()),
                    error_spec: Box::new(error_spec.clone()),
                };
                let fp_addr = self.var_ptr(vid);
                let fp_val = self.load_from_place_ptr(&fp_addr, &fp_ty, false);
                let fn_ptr_ty = self.ctx.spec.isa.fn_ptr_llvm_ty();
                let fp_ptr = self.coerce(fp_val, fn_ptr_ty, false);
                let env_mir = if error_spec.has_error() {
                    MirType::Result {
                        ok: Box::new(ret.clone()),
                        err: error_spec.err_type().map(Box::new),
                    }
                } else {
                    ret.clone()
                };
                let ret_lty = if env_mir.is_void() {
                    "void".to_string()
                } else {
                    self.ctx.llvm_ty(&env_mir)
                };
                let mut call_args = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    let av = self.emit_expr(a)?;
                    if let Some(pty_mir) = params.get(i) {
                        let pty = self.ctx.llvm_ty(&pty_mir.host_value_type());
                        let c = self.coerce(av, &pty, !a.ty.is_unsigned());
                        call_args.push(format!("{} {}", c.ty, c.repr));
                    } else {
                        call_args.push(format!("{} {}", av.ty, av.repr));
                    }
                }
                let joined = call_args.join(", ");
                if env_mir.is_void() || ret_lty == "void" {
                    self.emit_line(format!("{call_kw} void {}({joined})", fp_ptr.repr));
                    return Ok(SsaVal::new("void", "0"));
                } else {
                    let r = self.next_tmp();
                    self.emit_line(format!(
                        "{r} = {call_kw} {ret_lty} {}({joined})",
                        fp_ptr.repr
                    ));
                    return Ok(SsaVal::new(ret_lty, r));
                }
            }
        }

        if !is_extern_c {
            if let Some(sig) = self.ctx.fn_sigs.get(callee).cloned() {
                let mut call_args = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    let av = self.emit_expr(a)?;
                    if let Some(pty_mir) = sig.params.get(i) {
                        let pty = self.ctx.llvm_ty(pty_mir);
                        let c = self.coerce(av, &pty, !a.ty.is_unsigned());
                        call_args.push(format!("{} {}", c.ty, c.repr));
                    } else {
                        call_args.push(format!("{} {}", av.ty, av.repr));
                    }
                }
                let joined = call_args.join(", ");
                if sig.ret_llvm_ty == "void" {
                    self.emit_line(format!("{call_kw} void {}({joined})", sig.symbol));
                    return Ok(SsaVal::new("void", "0"));
                } else {
                    let r = self.next_tmp();
                    self.emit_line(format!(
                        "{r} = {call_kw} {} {}({joined})",
                        sig.ret_llvm_ty, sig.symbol
                    ));
                    return Ok(SsaVal::new(sig.ret_llvm_ty, r));
                }
            }
        }

        // Extern "C" call (if explicitly declared)
        if let Some(esig) = self.ctx.extern_sigs.get(callee).cloned() {
            let sym = format!("@{}", sanitize_ident(&esig.name));
            let ret_lty = self.ctx.llvm_ty(&esig.ret_ty);
            let mut call_args = Vec::new();
            for (i, a) in args.iter().enumerate() {
                let av = self.emit_expr(a)?;
                if let Some(pty_mir) = esig.params.get(i) {
                    let pty = self.ctx.llvm_ty(pty_mir);
                    let c = self.coerce(av, &pty, !a.ty.is_unsigned());
                    call_args.push(format!("{} {}", c.ty, c.repr));
                } else {
                    call_args.push(format!("{} {}", av.ty, av.repr));
                }
            }
            let sig_prefix = if esig.variadic {
                let fixed: Vec<String> = esig.params.iter().map(|t| self.ctx.llvm_ty(t)).collect();
                let mut fp = fixed;
                fp.push("...".to_string());
                format!("{ret_lty} ({})", fp.join(", "))
            } else {
                ret_lty.clone()
            };
            let joined = call_args.join(", ");
            if ret_lty == "void" {
                self.emit_line(format!("{call_kw} {sig_prefix} {sym}({joined})"));
                return Ok(SsaVal::new("void", "0"));
            } else {
                let r = self.next_tmp();
                self.emit_line(format!("{r} = {call_kw} {sig_prefix} {sym}({joined})"));
                return Ok(SsaVal::new(ret_lty, r));
            }
        }

        // Portable built-in functions (`print`, `println`, `assert`, `now_ms`, `now_ticks`)
        match callee {
            "print" | "puts" => {
                if let Some(a0) = args.first() {
                    let sv = self.emit_expr(a0)?;
                    let p = self.coerce_str_ptr(sv);
                    self.emit_line(format!("{call_kw} void @__syn_uart_puts(ptr {})", p.repr));
                }
                return Ok(SsaVal::new("void", "0"));
            }
            "println" => {
                if let Some(a0) = args.first() {
                    let sv = self.emit_expr(a0)?;
                    let p = self.coerce_str_ptr(sv);
                    self.emit_line(format!(
                        "{call_kw} void @__syn_uart_println(ptr {})",
                        p.repr
                    ));
                } else {
                    self.emit_line(format!("{call_kw} void @__syn_uart_putc(i8 10)"));
                }
                return Ok(SsaVal::new("void", "0"));
            }
            "assert" => {
                if let Some(a0) = args.first() {
                    let cv = self.emit_expr(a0)?;
                    self.emit_runtime_assert(cv);
                }
                return Ok(SsaVal::new("void", "0"));
            }
            "now_ms" | "now_ticks" => {
                let t = self.next_tmp();
                self.emit_line(format!("{t} = {call_kw} i32 @__syn_now_ms()"));
                let rlty = self.ctx.llvm_ty(ret_mir_ty);
                return Ok(self.coerce(SsaVal::new("i32", t), &rlty, false));
            }
            _ => {}
        }

        // Fallback if unknown pure helper
        let rlty = self.ctx.llvm_ty(ret_mir_ty);
        for a in args {
            let _ = self.emit_expr(a)?;
        }
        if rlty == "void" {
            Ok(SsaVal::new("void", "0"))
        } else {
            Ok(SsaVal::new(rlty, "zeroinitializer"))
        }
    }

    pub(super) fn emit_binary(
        &mut self,
        op: MirBinOp,
        lhs: &MirExpr,
        rhs: &MirExpr,
        res_ty: &MirType,
    ) -> Result<SsaVal, CodegenError> {
        match op {
            MirBinOp::LogicalAnd => {
                let lv = self.emit_expr(lhs)?;
                let l1 = self.coerce_to_i1(lv);
                let lhs_lbl = self.cur_label.clone();
                let rhs_lbl = self.next_label("and_rhs");
                let end_lbl = self.next_label("and_end");
                self.emit_term(format!("br i1 {l1}, label %{rhs_lbl}, label %{end_lbl}"));

                self.start_block(&rhs_lbl);
                let rv = self.emit_expr(rhs)?;
                let r1 = self.coerce_to_i1(rv);
                let r8 = self.next_tmp();
                self.emit_line(format!("{r8} = zext i1 {r1} to i8"));
                let rhs_done_lbl = self.cur_label.clone();
                self.emit_term(format!("br label %{end_lbl}"));

                self.start_block(&end_lbl);
                let phi = self.next_tmp();
                self.emit_line(format!(
                    "{phi} = phi i8 [ 0, %{lhs_lbl} ], [ {r8}, %{rhs_done_lbl} ]"
                ));
                return Ok(SsaVal::new("i8", phi));
            }
            MirBinOp::LogicalOr => {
                let lv = self.emit_expr(lhs)?;
                let l1 = self.coerce_to_i1(lv);
                let lhs_lbl = self.cur_label.clone();
                let rhs_lbl = self.next_label("or_rhs");
                let end_lbl = self.next_label("or_end");
                self.emit_term(format!("br i1 {l1}, label %{end_lbl}, label %{rhs_lbl}"));

                self.start_block(&rhs_lbl);
                let rv = self.emit_expr(rhs)?;
                let r1 = self.coerce_to_i1(rv);
                let r8 = self.next_tmp();
                self.emit_line(format!("{r8} = zext i1 {r1} to i8"));
                let rhs_done_lbl = self.cur_label.clone();
                self.emit_term(format!("br label %{end_lbl}"));

                self.start_block(&end_lbl);
                let phi = self.next_tmp();
                self.emit_line(format!(
                    "{phi} = phi i8 [ 1, %{lhs_lbl} ], [ {r8}, %{rhs_done_lbl} ]"
                ));
                return Ok(SsaVal::new("i8", phi));
            }
            _ => {}
        }

        let lv = self.emit_expr(lhs)?;
        let rv = self.emit_expr(rhs)?;
        let is_fp = lhs.ty.is_float()
            || rhs.ty.is_float()
            || lv.ty == "float"
            || lv.ty == "double"
            || rv.ty == "float"
            || rv.ty == "double";

        if is_fp {
            let res_lty = self.ctx.llvm_ty(&res_ty.host_value_type());
            let common_fty = if lv.ty == "double"
                || rv.ty == "double"
                || lhs.ty == MirType::F64
                || rhs.ty == MirType::F64
                || res_lty == "double"
            {
                "double"
            } else {
                "float"
            };
            let lc = self.coerce(lv, common_fty, !lhs.ty.is_unsigned());
            let rc = self.coerce(rv, common_fty, !rhs.ty.is_unsigned());

            let is_cmp = matches!(
                op,
                MirBinOp::Eq
                    | MirBinOp::Ne
                    | MirBinOp::Lt
                    | MirBinOp::Le
                    | MirBinOp::Gt
                    | MirBinOp::Ge
            );
            if is_cmp {
                let fpred = match op {
                    MirBinOp::Eq => "oeq",
                    MirBinOp::Ne => "une",
                    MirBinOp::Lt => "olt",
                    MirBinOp::Le => "ole",
                    MirBinOp::Gt => "ogt",
                    MirBinOp::Ge => "oge",
                    _ => unreachable!(),
                };
                let cmp = self.next_tmp();
                let b8 = self.next_tmp();
                self.emit_line(format!(
                    "{cmp} = fcmp {fpred} {common_fty} {}, {}",
                    lc.repr, rc.repr
                ));
                self.emit_line(format!("{b8} = zext i1 {cmp} to i8"));
                return Ok(SsaVal::new("i8", b8));
            }

            let finst = match op {
                MirBinOp::Add => "fadd",
                MirBinOp::Sub => "fsub",
                MirBinOp::Mul => "fmul",
                MirBinOp::Div => "fdiv",
                MirBinOp::Rem => "frem",
                _ => "fadd",
            };
            let t = self.next_tmp();
            self.emit_line(format!(
                "{t} = {finst} {common_fty} {}, {}",
                lc.repr, rc.repr
            ));
            return Ok(self.coerce(SsaVal::new(common_fty, t), &res_lty, true));
        }

        if matches!(op, MirBinOp::Shl | MirBinOp::Shr) {
            let lhs_host = lhs.ty.host_value_type();
            let shift_ty = self.ctx.llvm_ty(&lhs_host);
            let lc = self.coerce(lv, &shift_ty, !lhs.ty.is_unsigned());
            let rc = self.coerce(rv, &shift_ty, false);
            let mask = llvm_int_bits(&shift_ty).saturating_sub(1);
            let rc_masked = if let Ok(c) = rc.repr.parse::<i128>() {
                SsaVal::new(
                    shift_ty.clone(),
                    ((c as u128) & u128::from(mask)).to_string(),
                )
            } else {
                let m = self.next_tmp();
                self.emit_line(format!("{m} = and {shift_ty} {}, {mask}", rc.repr));
                SsaVal::new(shift_ty.clone(), m)
            };
            let inst = if matches!(op, MirBinOp::Shl) {
                "shl"
            } else if !lhs.ty.is_unsigned() {
                "ashr"
            } else {
                "lshr"
            };
            let t = self.next_tmp();
            self.emit_line(format!(
                "{t} = {inst} {shift_ty} {}, {}",
                lc.repr, rc_masked.repr
            ));
            let dst_lty = self.ctx.llvm_ty(&res_ty.host_value_type());
            return Ok(self.coerce(SsaVal::new(shift_ty, t), &dst_lty, !res_ty.is_unsigned()));
        }

        let signed = !lhs.ty.is_unsigned() || !rhs.ty.is_unsigned();

        // Pick common operand type
        let common_ty = if lv.ty == "ptr addrspace(1)" || rv.ty == "ptr addrspace(1)" {
            "ptr addrspace(1)".to_string()
        } else if lv.ty == "ptr" || rv.ty == "ptr" {
            "ptr".to_string()
        } else {
            let lb = llvm_int_bits(&lv.ty);
            let rb = llvm_int_bits(&rv.ty);
            let res_lty = self.ctx.llvm_ty(&res_ty.host_value_type());
            let mut max_b = lb.max(rb);
            if !matches!(
                op,
                MirBinOp::Eq
                    | MirBinOp::Ne
                    | MirBinOp::Lt
                    | MirBinOp::Le
                    | MirBinOp::Gt
                    | MirBinOp::Ge
            ) && res_lty.starts_with('i')
            {
                max_b = max_b.max(llvm_int_bits(&res_lty));
            }
            format!("i{max_b}")
        };

        let lc = self.coerce(lv, &common_ty, !lhs.ty.is_unsigned());
        let rc = self.coerce(rv, &common_ty, !rhs.ty.is_unsigned());

        let is_cmp = matches!(
            op,
            MirBinOp::Eq | MirBinOp::Ne | MirBinOp::Lt | MirBinOp::Le | MirBinOp::Gt | MirBinOp::Ge
        );
        if is_cmp {
            let pred = match (op, signed) {
                (MirBinOp::Eq, _) => "eq",
                (MirBinOp::Ne, _) => "ne",
                (MirBinOp::Lt, true) => "slt",
                (MirBinOp::Lt, false) => "ult",
                (MirBinOp::Le, true) => "sle",
                (MirBinOp::Le, false) => "ule",
                (MirBinOp::Gt, true) => "sgt",
                (MirBinOp::Gt, false) => "ugt",
                (MirBinOp::Ge, true) => "sge",
                (MirBinOp::Ge, false) => "uge",
                _ => unreachable!(),
            };
            let cmp = self.next_tmp();
            let b8 = self.next_tmp();
            self.emit_line(format!(
                "{cmp} = icmp {pred} {common_ty} {}, {}",
                lc.repr, rc.repr
            ));
            self.emit_line(format!("{b8} = zext i1 {cmp} to i8"));
            return Ok(SsaVal::new("i8", b8));
        }

        if matches!(op, MirBinOp::Div | MirBinOp::Rem) {
            let t = self.emit_checked_int_div_rem(
                &common_ty,
                &lc.repr,
                &rc.repr,
                signed,
                matches!(op, MirBinOp::Rem),
            );
            let dst_lty = self.ctx.llvm_ty(&res_ty.host_value_type());
            return Ok(self.coerce(SsaVal::new(common_ty, t), &dst_lty, !res_ty.is_unsigned()));
        }

        let inst = match (op, signed) {
            (MirBinOp::Add, _) => "add",
            (MirBinOp::Sub, _) => "sub",
            (MirBinOp::Mul, _) => "mul",
            (MirBinOp::BitOr, _) => "or",
            (MirBinOp::BitAnd, _) => "and",
            (MirBinOp::BitXor, _) => "xor",
            (MirBinOp::Shl, _) => "shl",
            (MirBinOp::Shr, true) => "ashr",
            (MirBinOp::Shr, false) => "lshr",
            _ => unreachable!(),
        };
        let t = self.next_tmp();
        self.emit_line(format!("{t} = {inst} {common_ty} {}, {}", lc.repr, rc.repr));
        let dst_lty = self.ctx.llvm_ty(&res_ty.host_value_type());
        Ok(self.coerce(SsaVal::new(common_ty, t), &dst_lty, !res_ty.is_unsigned()))
    }

    pub(super) fn emit_seq_ptr_and_len(
        &mut self,
        side: &MirExpr,
    ) -> Result<(String, String), CodegenError> {
        let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
        if let MirExprKind::StringLit(s) = &side.kind {
            let idx = self.ctx.intern_string(s);
            let slen = zev_ast::decoded_string_byte_len(s);
            return Ok((format!("@.str.{idx}"), format!("{slen}")));
        }
        match &side.ty {
            MirType::Slice(_) => {
                let sl_ty = format!("{{ ptr, {usize_ty} }}");
                let sv = self.emit_expr(side)?;
                let dp = self.next_tmp();
                let sl_len = self.next_tmp();
                self.emit_line(format!("{dp} = extractvalue {sl_ty} {}, 0", sv.repr));
                self.emit_line(format!("{sl_len} = extractvalue {sl_ty} {}, 1", sv.repr));
                Ok((dp, sl_len))
            }
            MirType::Array { len, .. } => {
                if *len == 0 {
                    return Ok(("null".to_string(), "0".to_string()));
                }
                let aty = self.ctx.llvm_ty(&side.ty);
                let base_ptr = if let MirExprKind::Var(vid) = &side.kind {
                    let ap = self.var_ptr(*vid);
                    let gp = self.next_tmp();
                    self.emit_line(format!(
                        "{gp} = getelementptr inbounds {aty}, ptr {ap}, {usize_ty} 0, {usize_ty} 0"
                    ));
                    gp
                } else if let Some((ap, aty_mir, _)) = self.try_emit_rodata_const_ptr(side)? {
                    let ro_aty = self.ctx.llvm_ty(&aty_mir);
                    let gp = self.next_tmp();
                    self.emit_line(format!(
                        "{gp} = getelementptr inbounds {ro_aty}, ptr {ap}, {usize_ty} 0, {usize_ty} 0"
                    ));
                    gp
                } else {
                    let bv = self.emit_expr(side)?;
                    let (tmp, al) = self.emit_temp_alloca(&bv.ty);
                    self.emit_line(format!(
                        "store {} {}, ptr {tmp}, align {al}",
                        bv.ty, bv.repr
                    ));
                    let gp = self.next_tmp();
                    self.emit_line(format!(
                        "{gp} = getelementptr inbounds {}, ptr {tmp}, {usize_ty} 0, {usize_ty} 0",
                        bv.ty
                    ));
                    gp
                };
                Ok((base_ptr, format!("{len}")))
            }
            _ => {
                let sv = self.emit_expr(side)?;
                Ok((sv.repr, "0".to_string()))
            }
        }
    }

    pub(super) fn emit_seq_eq(
        &mut self,
        eq: bool,
        lhs: &MirExpr,
        rhs: &MirExpr,
        elem_ty: &MirType,
    ) -> Result<SsaVal, CodegenError> {
        let usize_ty = self.ctx.spec.isa.usize_llvm_ty().to_string();
        let ety = self.ctx.llvm_ty(&elem_ty.host_value_type());
        let (pa, la) = self.emit_seq_ptr_and_len(lhs)?;
        let (pb, lb) = self.emit_seq_ptr_and_len(rhs)?;

        let (idx_p, idx_al) = self.emit_temp_alloca(&usize_ty);
        self.emit_line(format!("store {usize_ty} 0, ptr {idx_p}, align {idx_al}"));

        let len_eq = self.next_tmp();
        self.emit_line(format!("{len_eq} = icmp eq {usize_ty} {la}, {lb}"));
        let len_chk_lbl = self.cur_label.clone();

        let hdr_lbl = self.next_label("seqeq_hdr");
        let body_lbl = self.next_label("seqeq_body");
        let step_lbl = self.next_label("seqeq_step");
        let done_lbl = self.next_label("seqeq_done");
        self.emit_term(format!(
            "br i1 {len_eq}, label %{hdr_lbl}, label %{done_lbl}"
        ));

        self.start_block(&hdr_lbl);
        let i_val = self.next_tmp();
        let more = self.next_tmp();
        self.emit_line(format!(
            "{i_val} = load {usize_ty}, ptr {idx_p}, align {idx_al}"
        ));
        self.emit_line(format!("{more} = icmp ult {usize_ty} {i_val}, {la}"));
        self.emit_term(format!(
            "br i1 {more}, label %{body_lbl}, label %{done_lbl}"
        ));

        self.start_block(&body_lbl);
        let ep_a = self.next_tmp();
        let ep_b = self.next_tmp();
        let va = self.next_tmp();
        let vb = self.next_tmp();
        let el_eq = self.next_tmp();
        self.emit_line(format!(
            "{ep_a} = getelementptr {ety}, ptr {pa}, {usize_ty} {i_val}"
        ));
        self.emit_line(format!(
            "{ep_b} = getelementptr {ety}, ptr {pb}, {usize_ty} {i_val}"
        ));
        self.emit_line(format!("{va} = load {ety}, ptr {ep_a}, align 1"));
        self.emit_line(format!("{vb} = load {ety}, ptr {ep_b}, align 1"));
        self.emit_line(format!("{el_eq} = icmp eq {ety} {va}, {vb}"));
        self.emit_term(format!(
            "br i1 {el_eq}, label %{step_lbl}, label %{done_lbl}"
        ));

        self.start_block(&step_lbl);
        let next_i = self.next_tmp();
        self.emit_line(format!("{next_i} = add {usize_ty} {i_val}, 1"));
        self.emit_line(format!(
            "store {usize_ty} {next_i}, ptr {idx_p}, align {idx_al}"
        ));
        self.emit_term(format!("br label %{hdr_lbl}"));

        self.start_block(&done_lbl);
        let eq1 = self.next_tmp();
        self.emit_line(format!(
            "{eq1} = phi i1 [ false, %{len_chk_lbl} ], [ true, %{hdr_lbl} ], [ false, %{body_lbl} ]"
        ));
        let final1 = if eq {
            eq1
        } else {
            let neq1 = self.next_tmp();
            self.emit_line(format!("{neq1} = xor i1 {eq1}, true"));
            neq1
        };
        let b8 = self.next_tmp();
        self.emit_line(format!("{b8} = zext i1 {final1} to i8"));
        Ok(SsaVal::new("i8", b8))
    }
}

pub(super) fn crit_pair(contended: bool, call_kw: &str) -> (String, String) {
    if contended {
        (
            format!("  %pm = {call_kw} i32 @__syn_crit_enter()\n"),
            format!("  {call_kw} void @__syn_crit_exit(i32 %pm)\n"),
        )
    } else {
        (
            "  ; irq-mask elided (task-to-task)\n".to_string(),
            String::new(),
        )
    }
}
