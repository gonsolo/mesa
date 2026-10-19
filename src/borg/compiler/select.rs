use crate::state::ShaderState;
use crate::isel::borg_isel;
use crate::opt::classify_uniform;
use std::env;
use std::collections::HashMap;
use crate::{collect_cf_marks, decompose_vertex_offset, packed_layout, CfMark, resolve_vm, BorgInstr, Ubo, NO_DST, PUSH_WINDOW_WORDS};
use compiler::bindings::*;
use compiler::nir::AsDef;

impl ShaderState {
    pub unsafe fn load_const(&mut self, lc: &nir_load_const_instr) {
        let n = lc.def.num_components as usize;
        if n == 1 {
            self.consts.insert(lc.def.index, unsafe { lc.values()[0].u32_ });
        } else {
            let comps: Vec<(u32, u8)> = (0..n)
                .map(|c| {
                    let bits = unsafe { lc.values()[c].u32_ };
                    (self.konst(bits), 0u8)
                })
                .collect();
            self.sel.vec_map.insert(lc.def.index, comps);
        }
    }

    /// textureSize: TEX operation 3 at the level in TEXA's lod; the result block holds width,
    /// height, depth or layers and levels.
    unsafe fn tex_size(&mut self, tex: &nir_tex_instr) {
        let srcs = tex.srcs_as_slice();
        if let Some(t) = srcs.iter().find(|s| s.src_type == nir_tex_src_texture_deref) {
            self.tex_derefs.insert(t.src.as_def().index);
        }
        let ld = srcs.iter().find(|s| s.src_type == nir_tex_src_lod).map(|s| s.src.as_def().index);
        let wz = self.sel.next_vreg;
        self.sel.next_vreg += 1;
        self.sel.ubo.insert(wz, Ubo::Fixed(30));
        self.per_pixel_fixed.insert(wz);
        let (lv, lc) = match ld {
            Some(ld) => {
                if let Some(&bits) = self.consts.get(&ld) {
                    if !self.sel.vec_map.contains_key(&ld) {
                        let v = self.konst(bits);
                        self.sel.vec_map.insert(ld, vec![(v, 0u8)]);
                    }
                }
                resolve_vm(&self.sel.vec_map, ld, 0)
            }
            None => (wz, 0u8),
        };
        self.sel.prog.push(BorgInstr { mnem: "TEXA", dst: NO_DST, srcs: vec![wz, lv, wz], swz: vec![0, lc, 0] });
        let ctl = self.konst(3u32 << 16);
        let tex_v = self.sel.next_vreg;
        self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "TEX", dst: tex_v, srcs: vec![wz, wz, ctl], swz: vec![0, 0, 0] });
        self.tex_dsts.insert(tex_v);
        let n = tex.def.num_components as usize;
        // A 1D array's second size component is its layer count, the third word.
        let one_d_array = tex.sampler_dim == GLSL_SAMPLER_DIM_1D && tex.is_array;
        self.sel.vec_map.insert(
            tex.def.index,
            (0..n).map(|k| (tex_v, if one_d_array && k == 1 { 2 } else { k as u8 })).collect(),
        );
    }

    pub unsafe fn tex(&mut self, tex: &nir_tex_instr) {
        let srcs = tex.srcs_as_slice();
        if tex.op == nir_texop_txs && self.is_fragment {
            self.tex_size(tex);
            return;
        }
        let is_txl = tex.op == nir_texop_txl && self.is_fragment;
        let is_txb = tex.op == nir_texop_txb && self.is_fragment;
        let is_tg4 = tex.op == nir_texop_tg4 && self.is_fragment;
        if tex.op != nir_texop_tex && tex.op != nir_texop_txf && !is_txl && !is_txb && !is_tg4 {
            eprintln!("borgc: WARNING texture op {} not supported yet, dropped", tex.op);
            return;
        }
        if let Some(t) = srcs.iter().find(|s| s.src_type == nir_tex_src_texture_deref) {
            self.tex_derefs.insert(t.src.as_def().index);
            if self.tex_derefs.len() == 2 {
                eprintln!("borgc: WARNING more than one texture: all sample descriptor 0");
            }
        }
        let coord = srcs
            .iter()
            .find(|s| s.src_type == nir_tex_src_coord)
            .map(|s| s.src.as_def().index);
        if let Some(cd) = coord {
            let (ud, uc) = resolve_vm(&self.sel.vec_map, cd, 0);
            let is_fetch = tex.op == nir_texop_txf;
            let (vd, vc) = if is_fetch && tex.coord_components == 1 {
                (ud, uc)
            } else {
                resolve_vm(&self.sel.vec_map, cd, 1)
            };
            let is_cube = tex.sampler_dim == GLSL_SAMPLER_DIM_CUBE;
            let has_w = !is_fetch && tex.coord_components >= 3 && !is_cube && self.is_fragment;
            let has_bias = (is_txb || tex.op == nir_texop_tex) && !is_fetch && srcs.iter().any(|s| s.src_type == nir_tex_src_bias);
            if is_txl || has_w || has_bias {
                let ld = srcs.iter().find(|s| s.src_type == nir_tex_src_lod || s.src_type == nir_tex_src_bias).map(|s| s.src.as_def().index).unwrap_or(cd);
                if let Some(&bits) = self.consts.get(&ld).filter(|_| is_txl || has_bias) {
                    if !self.sel.vec_map.contains_key(&ld) {
                        let v = self.konst(bits);
                        self.sel.vec_map.insert(ld, vec![(v, 0u8)]);
                    }
                }
                let wz = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.ubo.insert(wz, Ubo::Fixed(30));
                self.per_pixel_fixed.insert(wz);
                let (lv, lc) = if is_txl || has_bias { resolve_vm(&self.sel.vec_map, ld, 0) } else { (wz, 0u8) };
                let (wv, wc) = if has_w { resolve_vm(&self.sel.vec_map, cd, 2) } else { (wz, 0u8) };
                self.sel.prog.push(BorgInstr { mnem: "TEXA", dst: NO_DST, srcs: vec![wv, lv, wz], swz: vec![wc, lc, 0] });
            }
            let ctl = if is_tg4 {
                // Operation 2 (gather) of the component in bits 19:18.
                let bits = 2u32 << 16 | (tex.component() & 3) << 18;
                self.konst(bits)
            } else if has_bias {
                // LOD mode 1: implicit LOD plus TEXA's bias.
                self.konst(1u32 << 21)
            } else if is_fetch || is_txl {
                self.konst(if is_txl { 2u32 << 21 } else { 1u32 << 16 })
            } else {
                match self.zero_ctl {
                Some(v) => v,
                None => {
                    let c = self.sel.next_vreg;
                    let neg = self.sel.next_vreg + 1;
                    let z = self.sel.next_vreg + 2;
                    self.sel.next_vreg += 3;
                    self.sel.ubo.insert(c, Ubo::Fixed(30));
                    self.per_pixel_fixed.insert(c);
                    self.sel.prog.push(BorgInstr { mnem: "FNEG", dst: neg, srcs: vec![c], swz: vec![0] });
                    self.sel.prog.push(BorgInstr { mnem: "FSTEP", dst: z, srcs: vec![neg], swz: vec![0] });
                    self.zero_ctl = Some(z);
                    z
                }
                }
            };
            let tex_v = self.sel.next_vreg;
            self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr {
                mnem: "TEX", dst: tex_v, srcs: vec![ud, vd, ctl], swz: vec![uc, vc, 0],
            });
            self.tex_dsts.insert(tex_v);
            self.sel.vec_map.insert(
                tex.def.index,
                vec![(tex_v, 0), (tex_v, 1), (tex_v, 2), (tex_v, 3)],
            );
        }
    }

    pub unsafe fn alu(&mut self, alu: &nir_alu_instr) {
        if self.is_compute && self.fold_const(alu) {
            return;
        }
        if self.is_compute || (matches!(alu.op, nir_op_fadd | nir_op_fmul | nir_op_ffma
            | nir_op_flt | nir_op_fge | nir_op_feq | nir_op_fneu | nir_op_bcsel | nir_op_fmin | nir_op_fmax
            | nir_op_slt | nir_op_sge | nir_op_seq | nir_op_sne
            | nir_op_ieq | nir_op_ine | nir_op_ilt | nir_op_ige | nir_op_ult | nir_op_uge)
            || (self.is_fragment && matches!(alu.op, nir_op_iadd | nir_op_imul | nir_op_ishl
                | nir_op_ishr | nir_op_ushr | nir_op_isub | nir_op_iand | nir_op_ior
                | nir_op_ixor))
            // Vertex bit operations are never address math (gl_VertexIndex & 2, say).
            || (self.is_vertex && matches!(alu.op, nir_op_ishr | nir_op_ushr | nir_op_iand | nir_op_ior
                | nir_op_ixor))
            || (self.is_vertex && self.flags.fetching() && matches!(alu.op, nir_op_iadd | nir_op_imul
                | nir_op_isub | nir_op_ishl | nir_op_umin | nir_op_umax | nir_op_imin | nir_op_imax)))
        {
            let first = if alu.op == nir_op_bcsel { 1 } else { 0 };
            for sr in &alu.srcs_as_slice()[first..] {
                let d = sr.src.as_def().index;
                let bits = match self.consts.get(&d) {
                    Some(&b) if !self.sel.vec_map.contains_key(&d) => b,
                    _ => continue,
                };
                let v = self.konst(bits);
                self.sel.vec_map.insert(d, vec![(v, 0u8)]);
            }
        }
        match alu.op {
            nir_op_vec2 | nir_op_vec3 | nir_op_vec4 => {
                let c: Vec<(u32, u8)> = alu
                    .srcs_as_slice()
                    .iter()
                    .map(|s| (s.src.as_def().index, s.swizzle[0]))
                    .collect();
                self.sel.vec_map.insert(alu.def.index, c);
            }
            // A graphics condition already is 1.0 or 0.0 (comparisons below), so b2f is a copy.
            nir_op_mov | nir_op_b2f16 | nir_op_b2f32 | nir_op_i2i16 | nir_op_i2i32 | nir_op_u2u16
            | nir_op_u2u32 => {
                let s = &alu.srcs_as_slice()[0];
                let n = alu.def.num_components as usize;
                let c: Vec<(u32, u8)> =
                    (0..n).map(|k| (s.src.as_def().index, s.swizzle[k])).collect();
                self.sel.vec_map.insert(alu.def.index, c);
            }
            nir_op_ineg => {
                let s = &alu.srcs_as_slice()[0];
                let x = resolve_vm(&self.sel.vec_map, s.src.as_def().index, s.swizzle[0]);
                let zero = self.sel.pair("ISUB", x, x);
                self.sel.prog.push(BorgInstr { mnem: "ISUB", dst: alu.def.index, srcs: vec![zero.0, x.0], swz: vec![0, x.1] });
            }
            // A condition is 1.0 or 0.0, so the integer 0 or 1 is its truncation.
            nir_op_b2i32 => {
                let s = &alu.srcs_as_slice()[0];
                let (v, c) = resolve_vm(&self.sel.vec_map, s.src.as_def().index, s.swizzle[0]);
                self.sel.prog.push(BorgInstr { mnem: "F2I", dst: alu.def.index, srcs: vec![v], swz: vec![c] });
            }
            nir_op_flt | nir_op_fge | nir_op_fneu | nir_op_feq | nir_op_slt | nir_op_sge | nir_op_sne | nir_op_seq
                if self.is_compute => self.exact_compare(alu),
            // slt and friends are the same compares with a float result, which a condition is here.
            nir_op_flt | nir_op_fge | nir_op_fneu | nir_op_feq | nir_op_slt | nir_op_sge | nir_op_sne | nir_op_seq => {
                let sl = alu.srcs_as_slice();
                let a = resolve_vm(&self.sel.vec_map, sl[0].src.as_def().index, sl[0].swizzle[0]);
                let b = resolve_vm(&self.sel.vec_map, sl[1].src.as_def().index, sl[1].swizzle[0]);

                // diff(hi, lo) = hi - lo, as a fresh vreg.

                match alu.op {
                    // a < b  ->  FSTEP(b - a)
                    nir_op_flt | nir_op_slt => {
                        let d = self.sel.fsub(b, a);
                        self.sel.prog.push(BorgInstr { mnem: "FSTEP", dst: alu.def.index, srcs: vec![d], swz: vec![0] });
                    }
                    nir_op_fge | nir_op_sge => {
                        let d = self.sel.fsub(b, a);
                        let st = self.sel.next_vreg; self.sel.next_vreg += 1;
                        let nst = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "FSTEP", dst: st, srcs: vec![d], swz: vec![0] });
                        self.sel.prog.push(BorgInstr { mnem: "FNEG", dst: nst, srcs: vec![st], swz: vec![0] });
                        let one = self.one();
                        self.sel.prog.push(BorgInstr { mnem: "FADD", dst: alu.def.index, srcs: vec![one, nst], swz: vec![0, 0] });
                    }
                    nir_op_fneu | nir_op_sne => {
                        let dab = self.sel.fsub(a, b);
                        let dba = self.sel.fsub(b, a);
                        let s1 = self.sel.next_vreg; self.sel.next_vreg += 1;
                        let s2 = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "FSTEP", dst: s1, srcs: vec![dab], swz: vec![0] });
                        self.sel.prog.push(BorgInstr { mnem: "FSTEP", dst: s2, srcs: vec![dba], swz: vec![0] });
                        self.sel.prog.push(BorgInstr { mnem: "FADD", dst: alu.def.index, srcs: vec![s1, s2], swz: vec![0, 0] });
                    }
                    // a == b  ->  1 - (a != b)
                    _ => {
                        let dab = self.sel.fsub(a, b);
                        let dba = self.sel.fsub(b, a);
                        let s1 = self.sel.next_vreg; self.sel.next_vreg += 1;
                        let s2 = self.sel.next_vreg; self.sel.next_vreg += 1;
                        let ne = self.sel.next_vreg; self.sel.next_vreg += 1;
                        let nne = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "FSTEP", dst: s1, srcs: vec![dab], swz: vec![0] });
                        self.sel.prog.push(BorgInstr { mnem: "FSTEP", dst: s2, srcs: vec![dba], swz: vec![0] });
                        self.sel.prog.push(BorgInstr { mnem: "FADD", dst: ne, srcs: vec![s1, s2], swz: vec![0, 0] });
                        self.sel.prog.push(BorgInstr { mnem: "FNEG", dst: nne, srcs: vec![ne], swz: vec![0] });
                        let one = self.one();
                        self.sel.prog.push(BorgInstr { mnem: "FADD", dst: alu.def.index, srcs: vec![one, nne], swz: vec![0, 0] });
                    }
                }
            }
            // fmax(0, x) → x · FSTEP(x)  (FSTEP(x)=1 if x>0 else 0).
            nir_op_fmax | nir_op_fmin if self.is_compute => self.exact_minmax(alu),
            nir_op_bcsel if self.is_compute => self.exact_bcsel(alu),
            nir_op_fmax | nir_op_fmin => {
                let s = alu.srcs_as_slice();
                let a_zero = s[0].comp_as_uint(0) == Some(0);
                let b_zero = s[1].comp_as_uint(0) == Some(0);
                if alu.op == nir_op_fmax && (a_zero || b_zero) {
                    let xs = if a_zero { &s[1] } else { &s[0] };
                    let x = resolve_vm(&self.sel.vec_map, xs.src.as_def().index, xs.swizzle[0]);
                    let vstep = self.sel.next_vreg;
                    self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FSTEP", dst: vstep, srcs: vec![x.0], swz: vec![x.1] });
                    self.sel.prog.push(BorgInstr { mnem: "FMUL", dst: alu.def.index, srcs: vec![x.0, vstep], swz: vec![x.1, 0] });
                } else {
                    // max(a, b) = b + (a >= b) * (a - b); min(a, b) = a + (a >= b) * (b - a)
                    let a = resolve_vm(&self.sel.vec_map, s[0].src.as_def().index, s[0].swizzle[0]);
                    let b = resolve_vm(&self.sel.vec_map, s[1].src.as_def().index, s[1].swizzle[0]);
                    let one = self.one();
                    let v = self.sel.next_vreg;
                    self.sel.next_vreg += 8;
                    self.sel.prog.push(BorgInstr { mnem: "FNEG", dst: v, srcs: vec![a.0], swz: vec![a.1] });          // -a
                    self.sel.prog.push(BorgInstr { mnem: "FADD", dst: v + 1, srcs: vec![b.0, v], swz: vec![b.1, 0] }); // b - a
                    self.sel.prog.push(BorgInstr { mnem: "FSTEP", dst: v + 2, srcs: vec![v + 1], swz: vec![0] });      // a < b
                    self.sel.prog.push(BorgInstr { mnem: "FNEG", dst: v + 3, srcs: vec![v + 2], swz: vec![0] });
                    self.sel.prog.push(BorgInstr { mnem: "FADD", dst: v + 4, srcs: vec![one, v + 3], swz: vec![0, 0] }); // a >= b
                    if alu.op == nir_op_fmax {
                        self.sel.prog.push(BorgInstr { mnem: "FNEG", dst: v + 5, srcs: vec![v + 1], swz: vec![0] });    // a - b
                        self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: alu.def.index, srcs: vec![v + 4, v + 5, b.0], swz: vec![0, 0, b.1] });
                    } else {
                        self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: alu.def.index, srcs: vec![v + 4, v + 1, a.0], swz: vec![0, 0, a.1] });
                    }
                }
            }
            nir_op_ieq | nir_op_ine | nir_op_ilt | nir_op_ige | nir_op_ult | nir_op_uge => {
                let sl = alu.srcs_as_slice();
                let a = resolve_vm(&self.sel.vec_map, sl[0].src.as_def().index, sl[0].swizzle[0]);
                let b = resolve_vm(&self.sel.vec_map, sl[1].src.as_def().index, sl[1].swizzle[0]);
                let lt = if matches!(alu.op, nir_op_ult | nir_op_uge) { "ISLTU" } else { "ISLT" };
                let bits = match alu.op {
                    nir_op_ieq => self.sel.pair("ISEQ", a, b),
                    nir_op_ilt | nir_op_ult => self.sel.pair(lt, a, b),
                    nir_op_ine => {
                        let l = self.sel.pair("ISLT", a, b);
                        let g = self.sel.pair("ISLT", b, a);
                        self.sel.pair("IADD", l, g)
                    }
                    _ => {
                        let g = self.sel.pair(lt, b, a);
                        let e = self.sel.pair("ISEQ", a, b);
                        self.sel.pair("IADD", g, e)
                    }
                };
                self.sel.prog.push(BorgInstr { mnem: "I2F", dst: alu.def.index, srcs: vec![bits.0], swz: vec![bits.1] });
            }
            nir_op_bcsel => {
                let s = alu.srcs_as_slice();
                let then_def = s[1].src.as_def().index;
                let mut x: Option<(u32, u8)> = None;
                if let Some((op, tsrcs)) = self.prod.get(&then_def) {
                    if *op == nir_op_fmul && tsrcs.len() == 2 {
                        // 12.92 is a NIR constant, or a window word once an arithmetic reader gave it one.
                        let is12 = |d: u32| {
                            let bits = self.consts.get(&d).copied().or_else(|| match self.sel.ubo.get(&d) {
                                Some(Ubo::Uniform(_)) => self.bits_of(d),
                                _ => None,
                            });
                            bits.map_or(false, |b| (f32::from_bits(b) - 12.92).abs() < 0.1)
                        };
                        x = if is12(tsrcs[0].0) {
                            Some(tsrcs[1])
                        } else if is12(tsrcs[1].0) {
                            Some(tsrcs[0])
                        } else {
                            None
                        };
                    }
                }
                if let Some(x) = x {
                    let v = self.sel.next_vreg;
                    self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FSRGB", dst: v, srcs: vec![x.0], swz: vec![x.1] });
                    self.sel.vec_map.insert(alu.def.index, vec![(v, 0)]);
                } else {
                    let n = alu.def.num_components as usize;
                    let mut comps = Vec::new();
                    for k in 0..n {
                        let c = resolve_vm(&self.sel.vec_map, s[0].src.as_def().index, s[0].swizzle[k]);
                        let t = resolve_vm(&self.sel.vec_map, s[1].src.as_def().index, s[1].swizzle[k]);
                        let e = resolve_vm(&self.sel.vec_map, s[2].src.as_def().index, s[2].swizzle[k]);
                        let ne = self.sel.next_vreg;
                        let d = self.sel.next_vreg + 1;
                        let r = self.sel.next_vreg + 2;
                        self.sel.next_vreg += 3;
                        self.sel.prog.push(BorgInstr { mnem: "FNEG", dst: ne, srcs: vec![e.0], swz: vec![e.1] });
                        self.sel.prog.push(BorgInstr { mnem: "FADD", dst: d, srcs: vec![t.0, ne], swz: vec![t.1, 0] });
                        self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: r, srcs: vec![c.0, d, e.0], swz: vec![c.1, 0, e.1] });
                        comps.push((r, 0u8));
                    }
                    self.sel.vec_map.insert(alu.def.index, comps);
                }
            }
            // max/min of integers: a + (a < b) * (b - a) for max, b + (a < b) * (a - b) for min.
            nir_op_umax | nir_op_umin | nir_op_imax | nir_op_imin => {
                let sl = alu.srcs_as_slice();
                let a = resolve_vm(&self.sel.vec_map, sl[0].src.as_def().index, sl[0].swizzle[0]);
                let b = resolve_vm(&self.sel.vec_map, sl[1].src.as_def().index, sl[1].swizzle[0]);
                let lt = if matches!(alu.op, nir_op_umax | nir_op_umin) { "ISLTU" } else { "ISLT" };
                let is_max = matches!(alu.op, nir_op_umax | nir_op_imax);
                let (p, q) = if is_max { (a, b) } else { (b, a) };
                let v = self.sel.next_vreg;
                self.sel.next_vreg += 3;
                self.sel.prog.push(BorgInstr { mnem: lt, dst: v, srcs: vec![a.0, b.0], swz: vec![a.1, b.1] });
                self.sel.prog.push(BorgInstr { mnem: "ISUB", dst: v + 1, srcs: vec![q.0, p.0], swz: vec![q.1, p.1] });
                self.sel.prog.push(BorgInstr { mnem: "IMUL", dst: v + 2, srcs: vec![v, v + 1], swz: vec![0, 0] });
                self.sel.prog.push(BorgInstr { mnem: "IADD", dst: alu.def.index, srcs: vec![p.0, v + 2], swz: vec![p.1, 0] });
            }
            // |x|: clear the sign bit.
            nir_op_fabs => {
                let sl = &alu.srcs_as_slice()[0];
                let (x, xc) = resolve_vm(&self.sel.vec_map, sl.src.as_def().index, sl.swizzle[0]);
                let m = self.konst(0x7fff_ffff);
                self.sel.prog.push(BorgInstr { mnem: "IAND", dst: alu.def.index, srcs: vec![x, m], swz: vec![xc, 0] });
            }
            // A condition is 1.0 or 0.0: its inverse is 1.0 - c.
            nir_op_inot if alu.def.bit_size == 1 => {
                let sl = &alu.srcs_as_slice()[0];
                let (x, xc) = resolve_vm(&self.sel.vec_map, sl.src.as_def().index, sl.swizzle[0]);
                let one = self.konst(1.0f32.to_bits());
                let n = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: "FNEG", dst: n, srcs: vec![x], swz: vec![xc] });
                self.sel.prog.push(BorgInstr { mnem: "FADD", dst: alu.def.index, srcs: vec![n, one], swz: vec![0, 0] });
            }
            nir_op_inot => {
                let sl = &alu.srcs_as_slice()[0];
                let x = resolve_vm(&self.sel.vec_map, sl.src.as_def().index, sl.swizzle[0]);
                let m = self.konst(u32::MAX);
                self.sel.prog.push(BorgInstr { mnem: "IXOR", dst: alu.def.index, srcs: vec![x.0, m], swz: vec![x.1, 0] });
            }
            nir_op_u2f16 | nir_op_u2f32 => {
                let s = &alu.srcs_as_slice()[0];
                let (x, xc) = resolve_vm(&self.sel.vec_map, s.src.as_def().index, s.swizzle[0]);
                let c1 = self.konst(1);
                let k = self.konst(2.0f32.to_bits());
                let v = self.sel.next_vreg;
                self.sel.next_vreg += 5;
                self.sel.prog.push(BorgInstr { mnem: "ISRL", dst: v, srcs: vec![x, c1], swz: vec![xc, 0] });
                self.sel.prog.push(BorgInstr { mnem: "IAND", dst: v + 2, srcs: vec![x, c1], swz: vec![xc, 0] });
                self.sel.prog.push(BorgInstr { mnem: "I2F", dst: v + 3, srcs: vec![v], swz: vec![0] });
                self.sel.prog.push(BorgInstr { mnem: "I2F", dst: v + 4, srcs: vec![v + 2], swz: vec![0] });
                self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: alu.def.index, srcs: vec![v + 3, k, v + 4], swz: vec![0, 0, 0] });
            }
            _ => {
                if let Some(mnem) = borg_isel(alu.op) {
                    if mnem != "mov" {
                        let mut srcs = Vec::new();
                        let mut swz = Vec::new();
                        for s in alu.srcs_as_slice() {
                            let (d, c) =
                                resolve_vm(&self.sel.vec_map, s.src.as_def().index, s.swizzle[0]);
                            srcs.push(d);
                            swz.push(c);
                        }
                        let pairs: Vec<(u32, u8)> =
                            srcs.iter().cloned().zip(swz.iter().cloned()).collect();
                        self.prod.insert(alu.def.index, (alu.op, pairs));
                        self.sel.prog.push(BorgInstr { mnem, dst: alu.def.index, srcs, swz });
                    }
                }
                else if !self.skipped.contains(&alu.info().name()) {
                    self.skipped.push(alu.info().name());
                }
            }
        }
    }

    pub unsafe fn intrinsic(&mut self, intr: &nir_intrinsic_instr) {
        if self.is_compute {
            self.compute_intrinsic(intr);
            return;
        }
        let deriv = match intr.intrinsic {
            nir_intrinsic_ddx | nir_intrinsic_ddx_coarse | nir_intrinsic_ddx_fine => Some("DDX"),
            nir_intrinsic_ddy | nir_intrinsic_ddy_coarse | nir_intrinsic_ddy_fine => Some("DDY"),
            _ => None,
        };
        if let Some(mnem) = deriv {
            let src = intr.srcs_as_slice()[0].as_def().index;
            let n = intr.def.num_components as usize;
            let raw: Vec<u32> = (0..n)
                .map(|c| {
                    let (sd, sc) = resolve_vm(&self.sel.vec_map, src, c as u8);
                    let v = self.sel.next_vreg;
                    self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem, dst: v, srcs: vec![sd], swz: vec![sc] });
                    v
                })
                .collect();
            let comps: Vec<(u32, u8)> = raw.into_iter().map(|v| (v, 0u8)).collect();
            self.sel.vec_map.insert(intr.def.index, comps);
        } else if (intr.intrinsic == nir_intrinsic_load_input || intr.intrinsic == nir_intrinsic_load_point_coord) && !self.is_vertex {
            let is_pc = intr.intrinsic == nir_intrinsic_load_point_coord;
            let loc = if is_pc { 25 } else { intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F };
            let base_index = if loc == 25 { self.flags.pntc_base() } else { 4 * loc.wrapping_sub(VARYING_SLOT_VAR0) + intr.get_const_index(NIR_INTRINSIC_COMPONENT) as u32 };
            let bary = [5u8, 6, 7].map(|r| {
                let v = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.ubo.insert(v, Ubo::Fixed(r));
                self.per_pixel_fixed.insert(v);
                v
            });
            let n = intr.def.num_components as usize;
            // Integer varyings are flat (Vulkan): the provoking vertex's bits, not a blend.
            let dtype = if is_pc { 0x80 } else { intr.get_const_index(NIR_INTRINSIC_DEST_TYPE) };
            let flat = dtype != 0 && dtype & 0x80 == 0;
            let comps: Vec<(u32, u8)> = (0..n)
                .map(|c| {
                    let fattr_rd = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.fattr_dsts.insert(fattr_rd);
                    self.sel.prog.push(BorgInstr {
                        mnem: "FATTR", dst: fattr_rd, srcs: vec![],
                        swz: vec![(base_index + c as u32) as u8],
                    });
                    if flat {
                        let res = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "IOR", dst: res, srcs: vec![fattr_rd, fattr_rd], swz: vec![0, 0] });
                        return (res, 0u8);
                    }
                    let t0 = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FMUL", dst: t0, srcs: vec![bary[0], fattr_rd], swz: vec![0, 0] });
                    let t1 = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: t1, srcs: vec![bary[1], fattr_rd, t0], swz: vec![0, 1, 0] });
                    let res = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: res, srcs: vec![bary[2], fattr_rd, t1], swz: vec![0, 2, 0] });
                    (res, 0u8)
                })
                .collect();
            self.sel.vec_map.insert(intr.def.index, comps);
        } else if intr.intrinsic == nir_intrinsic_load_input && !self.is_vertex {
            if self.weights.is_none() {
                let mut w = [0u32; 3];
                for (i, wi) in w.iter_mut().enumerate() {
                    let e = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.ubo.insert(e, Ubo::Fixed(i as u8));        // e_i in r0/r1/r2
                    self.per_pixel_fixed.insert(e); // NOT uniform: varies per pixel
                    let inv = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.ubo.insert(inv, Ubo::Uniform(12));         // inv_area = u12
                    let v = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FMUL", dst: v, srcs: vec![e, inv], swz: vec![0, 0] });
                    *wi = v;
                }
                self.weights = Some(w);
            }
            let w = self.weights.unwrap();
            let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
            let var = loc.wrapping_sub(VARYING_SLOT_VAR0); // 0 = texcoord, 1 = frag_pos
            let n = intr.def.num_components as usize;
            let comps: Vec<(u32, u8)> = (0..n)
                .map(|c| {
                    let base = ((if var == 0 { 13 } else { 19 }) + (c as u32) * 3) as u8;
                    // uniforms for (v0, v1, v2) = (base+2, base+1, base+0)
                    let uv: Vec<u32> = [2u8, 1, 0]
                        .iter()
                        .map(|&o| {
                            let v = self.sel.next_vreg; self.sel.next_vreg += 1;
                            self.sel.ubo.insert(v, Ubo::Uniform(base + o));
                            v
                        })
                        .collect();
                    let t0 = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FMUL", dst: t0, srcs: vec![w[0], uv[0]], swz: vec![0, 0] });
                    let t1 = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: t1, srcs: vec![w[1], uv[1], t0], swz: vec![0, 0, 0] });
                    let res = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: res, srcs: vec![w[2], uv[2], t1], swz: vec![0, 0, 0] });
                    (res, 0u8)
                })
                .collect();
            self.sel.vec_map.insert(intr.def.index, comps);
        } else if intr.intrinsic == nir_intrinsic_load_push_constant {
            let dyn_off = intr.get_src(0).comp_as_uint(0);
            if let Some(dyn_off) = dyn_off {
                let base = intr.base() + dyn_off as i32 + if self.push_window { PUSH_WINDOW_WORDS as i32 * 4 } else { 0 };
                let n = intr.def.num_components as usize;
                let comps: Vec<(u32, u8)> = (0..n)
                    .map(|c| {
                        let byte_off = base + 4 * (c as i32);
                        assert_eq!(byte_off % 4, 0,
                            "push constant field at byte {byte_off} is not word-aligned");
                        let word_idx = (byte_off / 4) as u32;
                        // A window word per run (a pinned GPR is clobbered by the fragment pass).
                        let idx_reg = match self.push_const_reg.get(&word_idx) {
                            Some(&r) => r,
                            None => {
                                let r = self.konst(word_idx);
                                self.push_const_reg.insert(word_idx, r);
                                r
                            }
                        };
                        let dst = self.sel.next_vreg;
                        self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr {
                            mnem: "LOAD", dst, srcs: vec![idx_reg], swz: vec![0],
                        });
                        (dst, 0u8)
                    })
                    .collect();
                self.sel.vec_map.insert(intr.def.index, comps);
            } else if self.is_vertex || self.is_fragment {
                                let o = resolve_vm(&self.sel.vec_map, intr.get_src(0).as_def().index, 0);
                let two = self.konst(2);
                let w = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: "ISRL", dst: w, srcs: vec![o.0, two], swz: vec![o.1, 0] });
                let base_word = (intr.base() / 4) as u32 + if self.push_window { PUSH_WINDOW_WORDS } else { 0 };
                let comps: Vec<(u32, u8)> = (0..intr.def.num_components as u32)
                    .map(|c| {
                        let k = self.konst(base_word + c);
                        let addr = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "IADD", dst: addr, srcs: vec![w, k], swz: vec![0, 0] });
                        let dst = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "LOAD", dst, srcs: vec![addr], swz: vec![0] });
                        (dst, 0u8)
                    })
                    .collect();
                self.sel.vec_map.insert(intr.def.index, comps);
            } else {
                eprintln!(
                    "borgc: WARNING push constant at base={} read with a \
                     non-constant dynamic offset -- not supported, dropped",
                    intr.base()
                );
            }
        }
        else if intr.intrinsic == nir_intrinsic_vulkan_resource_index {
            self.res_binding.insert(intr.def.index, intr.get_const_index(NIR_INTRINSIC_BINDING));
        } else if intr.intrinsic == nir_intrinsic_load_vulkan_descriptor {
            self.descriptor_defs.insert(intr.def.index);
            if let Some(&b) = self.res_binding.get(&intr.get_src(0).as_def().index) {
                self.res_binding.insert(intr.def.index, b);
            }
        } else if intr.intrinsic == nir_intrinsic_load_ssbo
                                || intr.intrinsic == nir_intrinsic_store_ssbo {
            const SSBO_SLOT_WORDS: u32 = 16384;
            let store = intr.intrinsic == nir_intrinsic_store_ssbo;
            let (bsrc, osrc) = if store { (1, 2) } else { (0, 1) };
            let mut bd = resolve_vm(&self.sel.vec_map, intr.get_src(bsrc).as_def().index, 0).0;
            for _ in 0..8 {
                if self.res_binding.contains_key(&bd) { break; }
                match self.prod.get(&bd) { Some((_, srcs)) if !srcs.is_empty() => bd = resolve_vm(&self.sel.vec_map, srcs[0].0, 0).0, _ => break }
            }
            let binding = self.res_binding.get(&bd).copied().unwrap_or(0);
            let slot = SSBO_SLOT_WORDS * (1 + binding);
            // A register holding 0, to copy window words through (an instruction reads one).
            let f = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.ubo.insert(f, Ubo::Fixed(30));
            let zero = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: "ISUB", dst: zero, srcs: vec![f, f], swz: vec![0, 0] });
            // The descriptor's own offset component (a dynamic buffer's) is the window base: 0.
            let off_is_descriptor = self.descriptor_defs.contains(&resolve_vm(&self.sel.vec_map, intr.get_src(osrc).as_def().index, 0).0);
            let word = match intr.get_src(osrc).comp_as_uint(0) {
                Some(_) => None,
                None if off_is_descriptor => None,
                None => {
                    let o = resolve_vm(&self.sel.vec_map, intr.get_src(osrc).as_def().index, 0);
                    let two = self.konst(2);
                    let w = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "ISRL", dst: w, srcs: vec![o.0, two], swz: vec![o.1, 0] });
                    Some(w)
                }
            };
            let const_word = intr.get_src(osrc).comp_as_uint(0).map(|o| o as u32 / 4);
            let n = if store { intr.get_src(0).num_components() as usize } else { intr.def.num_components as usize };
            let mut comps = Vec::with_capacity(n);
            for c in 0..n {
                if store && intr.write_mask() & (1 << c) == 0 {
                    continue;
                }
                let k = self.konst(slot + const_word.unwrap_or(0) + c as u32);
                let addr = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: "IADD", dst: addr, srcs: vec![word.unwrap_or(zero), k], swz: vec![0, 0] });
                if store {
                    let src = intr.get_src(0).as_def().index;
                    let d = resolve_vm(&self.sel.vec_map, src, c as u8);
                    let val = match self.consts.get(&d.0) {
                        Some(&bits) if !self.sel.vec_map.contains_key(&d.0) => {
                            let kv = self.konst(bits);
                            let t = self.sel.next_vreg; self.sel.next_vreg += 1;
                            self.sel.prog.push(BorgInstr { mnem: "IADD", dst: t, srcs: vec![zero, kv], swz: vec![0, 0] });
                            (t, 0u8)
                        }
                        _ => d,
                    };
                    self.sel.prog.push(BorgInstr { mnem: "STORE", dst: NO_DST, srcs: vec![addr, val.0], swz: vec![0, val.1] });
                } else {
                    let dst = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "LOAD", dst, srcs: vec![addr], swz: vec![0] });
                    comps.push((dst, 0u8));
                }
            }
            if !store {
                self.sel.vec_map.insert(intr.def.index, comps);
            }
        } else if self.is_fragment && intr.intrinsic == nir_intrinsic_load_frag_coord {
            let comps: Vec<(u32, u8)> = [30u8, 31, 29, 8]
                .iter()
                .map(|&r| {
                    let v = self.sel.next_vreg;
                    self.sel.next_vreg += 1;
                    self.sel.ubo.insert(v, Ubo::Fixed(r));
                    self.per_pixel_fixed.insert(v);
                    (v, 0u8)
                })
                .collect();
            self.frag_coord_w = true;
            self.sel.vec_map.insert(intr.def.index, comps);
        } else if intr.intrinsic == nir_intrinsic_load_vertex_id {
            self.sel.ubo.insert(intr.def.index, Ubo::Fixed(30));
            self.vertex_id_def = Some(intr.def.index);
        } else if intr.intrinsic == nir_intrinsic_load_instance_id {
            self.sel.ubo.insert(intr.def.index, Ubo::Fixed(31));
        } else if self.is_vertex && self.flags.fetching()
                  && intr.intrinsic == nir_intrinsic_load_input {
            const VERT_ATTRIB_GENERIC0: u32 = 15;
            let loc = (intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F)
                .wrapping_sub(VERT_ATTRIB_GENERIC0);
            let n = intr.def.num_components as usize;
            let ctl_bits = (1u32 << 16) | (128 + loc);
            let twelve = self.konst(12);
            let ctl = self.konst(ctl_bits);
            // The index register: r30 (VertexIndex) or r31 (InstanceIndex).
            let idx_reg = if loc < 16 && self.flags.instanced(loc) { 31 } else { 30 };
            let idx = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.ubo.insert(idx, Ubo::Fixed(idx_reg));
            // The element index i is texel (i & 4095, i >> 12) of a 4096-wide image.
            let ty = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: "ISRL", dst: ty, srcs: vec![idx, twelve], swz: vec![0, 0] });
            let hi = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: "ISHL", dst: hi, srcs: vec![ty, twelve], swz: vec![0, 0] });
            let tx = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: "ISUB", dst: tx, srcs: vec![idx, hi], swz: vec![0, 0] });
            let tex_v = self.sel.next_vreg; self.sel.next_vreg += 1;
            self.sel.prog.push(BorgInstr { mnem: "TEX", dst: tex_v, srcs: vec![tx, ty, ctl], swz: vec![0, 0, 0] });
            self.tex_dsts.insert(tex_v);
            let comp0 = intr.get_const_index(NIR_INTRINSIC_COMPONENT) as u8;
            let mut comps = Vec::with_capacity(n);
            for c in 0..n {
                let m = self.sel.next_vreg; self.sel.next_vreg += 1;
                self.sel.prog.push(BorgInstr { mnem: "IOR", dst: m, srcs: vec![tex_v, tex_v],
                                      swz: vec![comp0 + c as u8, comp0 + c as u8] });
                comps.push((m, 0u8));
            }
            self.sel.vec_map.insert(intr.def.index, comps);
        } else if intr.intrinsic == nir_intrinsic_load_ubo
                                || (self.is_vertex && intr.intrinsic == nir_intrinsic_load_input) {
            let is_input = intr.intrinsic == nir_intrinsic_load_input;
            let offset_def = if is_input { 0 } else { intr.get_src(1).as_def().index };
            let n = intr.def.num_components as usize;
            const VERT_ATTRIB_GENERIC0: u32 = 15;
            let input_slot = if is_input {
                let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
                Some(loc.wrapping_sub(VERT_ATTRIB_GENERIC0))
            } else {
                None
            };
            let (base_words, stride_words) = match input_slot
                .map(|l| Some((16 + 144 * l as i32, 4i32)))
                .unwrap_or_else(|| decompose_vertex_offset(
                    &self.sel.vec_map, &self.prod, &self.consts, &self.descriptor_defs,
                    self.vertex_id_def.unwrap_or(u32::MAX), offset_def,
                )) {
                Some(bs) => bs,
                None => {
                    eprintln!(
                        "borgc: WARNING load_ubo offset (def {offset_def}) is \
                         not a recognized draw-mode address shape -- dropped"
                    );
                    (-1, 0)
                }
            };
            let base_words = if base_words >= 0 && !is_input {
                let mut d = resolve_vm(&self.sel.vec_map, intr.get_src(0).as_def().index, 0).0;
                for _ in 0..8 {
                    if self.res_binding.contains_key(&d) { break; }
                    match self.prod.get(&d) { Some((_, srcs)) if !srcs.is_empty() => d = resolve_vm(&self.sel.vec_map, srcs[0].0, 0).0, _ => break }
                }
                let b = self.res_binding.get(&d).copied().unwrap_or(0);
                if b == 0 { base_words } else { base_words + 16384 * (1 + b as i32) }
            } else {
                base_words
            };
            if base_words >= 0 {
                let addr = match (stride_words, self.const_addr_chain) {
                    (0, Some((next, a))) if next == base_words => a,
                    (0, _) if !self.is_vertex && base_words > 0 => self.konst(base_words as u32),
                    (0, _) if !self.is_vertex => self.konst(base_words as u32),
                    // A single constant word: the address is a window word, no chain.
                    (0, _) if n == 1 => self.konst(base_words as u32),
                    _ => {
                        let vid = match self.vertex_id_def {
                            Some(v) => v,
                            None => {
                                let v = self.sel.next_vreg; self.sel.next_vreg += 1;
                                self.sel.ubo.insert(v, Ubo::Fixed(30));
                                v
                            }
                        };
                        let stride_v = self.konst(stride_words as u32);
                        let tmp = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "IMUL", dst: tmp, srcs: vec![vid, stride_v], swz: vec![0, 0] });
                        let base_v = self.konst(base_words as u32);
                        let a = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "IADD", dst: a, srcs: vec![tmp, base_v], swz: vec![0, 0] });
                        a
                    }
                };
                let mut cur = addr;
                let comps: Vec<(u32, u8)> = (0..n)
                    .map(|c| {
                        let dst = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "LOAD", dst, srcs: vec![cur], swz: vec![0] });
                        if c + 1 < n {
                            let one_v = self.konst(1);
                            let next_a = self.sel.next_vreg; self.sel.next_vreg += 1;
                            self.sel.prog.push(BorgInstr {
                                mnem: "IADD", dst: next_a, srcs: vec![cur, one_v], swz: vec![0, 0],
                            });
                            cur = next_a;
                        }
                        (dst, 0u8)
                    })
                    .collect();
                if stride_words == 0 && n > 1 {
                    let one_v = self.konst(1);
                    let past = self.sel.next_vreg; self.sel.next_vreg += 1;
                    self.sel.prog.push(BorgInstr { mnem: "IADD", dst: past, srcs: vec![cur, one_v], swz: vec![0, 0] });
                    self.const_addr_chain = Some((base_words + n as i32, past));
                }
                self.sel.vec_map.insert(intr.def.index, comps);
            }
        } else if intr.intrinsic == nir_intrinsic_store_output {
            let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
            if self.is_fragment && self.out_select >= 0 && (4..12).contains(&loc) && loc != 4 + self.out_select as u32 {
                return;
            }
            let src = intr.get_src(0).as_def().index;
            let ncomp = intr.get_src(0).num_components() as usize;
            // A scalar constant stored directly has no register yet: give it a window word.
            for c in 0..ncomp {
                let d = resolve_vm(&self.sel.vec_map, src, c as u8).0;
                let bits = match self.consts.get(&d) {
                    Some(&b) if !self.sel.vec_map.contains_key(&d) => b,
                    _ => continue,
                };
                let v = self.konst(bits);
                self.sel.vec_map.insert(d, vec![(v, 0u8)]);
            }
            let comps: Vec<(u32, u8)> = (0..ncomp).map(|c| resolve_vm(&self.sel.vec_map, src, c as u8)).collect();
            if self.is_vertex && loc == VARYING_SLOT_POS {
                for (c, &v0) in comps.iter().enumerate().take(4) {
                    let mut v = v0;
                    if matches!(self.sel.ubo.get(&v.0), Some(Ubo::Uniform(_))) {
                        let out = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "FMOV", dst: out, srcs: vec![v.0], swz: vec![0] });
                        v = (out, 0);
                    }
                    self.outs.pos[c] = Some(v);
                    self.outs.roots.push(v.0);
                }
            } else if self.is_fragment && self.flags.multi_target() && (4..8).contains(&loc) {
                self.outs.mrt[(loc - 4) as usize] = Some(comps[0]);
            } else if self.is_fragment {
                let ncomp2 = if self.frag_alpha || self.flags.bytes_packed() { 4 } else { 3 };
                for (c, &v) in comps.iter().enumerate().take(ncomp2) {
                    self.outs.frag[c] = Some(v);
                    self.outs.roots.push(v.0);
                }
            } else if self.is_vertex && loc == 12 {
                self.outs.psize = comps.first().copied();   // VARYING_SLOT_PSIZ
            } else if loc < VARYING_SLOT_VAR0 {
                // gl_PointSize and the other built-in outputs: triangles ignore them.
            } else {
                let base_index = 4 * loc.wrapping_sub(VARYING_SLOT_VAR0) + intr.get_const_index(NIR_INTRINSIC_COMPONENT) as u32;
                for (c, &(vd0, vc)) in comps.iter().enumerate() {
                    let mut vd = vd0;
                    // SOUT stores a register; a constant living in the uniform window needs a copy.
                    if matches!(self.sel.ubo.get(&vd), Some(Ubo::Uniform(_))) {
                        // An integer add keeps the bits of an integer constant (a float move changes a NaN or denormal).
                        let zero = self.cp.zero(&mut self.sel);
                        let out = self.sel.next_vreg; self.sel.next_vreg += 1;
                        self.sel.prog.push(BorgInstr { mnem: "IADD", dst: out, srcs: vec![zero, vd], swz: vec![0, 0] });
                        vd = out;
                    }
                    let index = base_index + c as u32;
                    assert!(index < 256, "borgc: SOUT index {index} does not fit a byte");
                    self.sel.prog.push(BorgInstr {
                        mnem: "SOUT", dst: NO_DST, srcs: vec![vd], swz: vec![index as u8],
                    });
                    self.outs.roots.push(vd);
                    let _ = vc;
                }
            }
            self.outs.stores.insert(loc, comps);
        } else if self.is_fragment && intr.intrinsic == nir_intrinsic_load_output
                  && intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) >> 14 & 1 != 0 {
            let desc = if self.flags.packed() && !self.flags.multi_target() {
                Some(packed_layout(self.flags))
            } else if self.flags.r8_unorm() {
                Some((0, vec![(0usize, 0u32, 8u32)]))
            } else {
                None
            };
            match desc {
                Some((kind, layout)) if layout.iter().all(|&(_, sh, b)| sh + b <= 32) => {
                    let w = self.sel.op0("TLD", vec![]);
                    let zero = self.sel.op0("ISUB", vec![w, w]);
                    let mut rgba: [Option<u32>; 4] = [None; 4];
                    for &(comp, sh, bits) in layout.iter() {
                        let mut f = w;
                        if sh != 0 {
                            let s = self.konst(sh);
                            f = self.sel.op0("ISRL", vec![f, s]);
                        }
                        if bits < 32 && (kind != 1) && sh + bits < 32 {
                            let m = self.konst((1u32 << bits) - 1);
                            f = self.sel.op0("IAND", vec![f, m]);
                        }
                        let v = match kind {
                            2 if comp != 3 => {
                                // sRGB to linear: a fetch from the driver's 256-entry table (slot 41).
                                let ctl = self.konst((1 << 16) | (128 + 41));
                                let t = self.sel.op0("TEX", vec![f, zero, ctl]);
                                self.tex_dsts.insert(t);
                                self.sel.op0("IOR", vec![t, t])
                            }
                            0 | 2 => {
                                let i = self.sel.op0("I2F", vec![f]);
                                let inv = self.konst((1.0f32 / ((1u64 << bits) - 1) as f32).to_bits());
                                self.sel.op0("FMUL", vec![i, inv])
                            }
                            1 => {
                                // Sign-extend the field from the word itself.
                                let up = self.konst(32 - bits - sh);
                                let a = if 32 - bits - sh != 0 { self.sel.op0("ISHL", vec![w, up]) } else { w };
                                let dn = self.konst(32 - bits);
                                let s = self.sel.op0("ISHR", vec![a, dn]);
                                let i = self.sel.op0("I2F", vec![s]);
                                let inv = self.konst((1.0f32 / ((1u64 << (bits - 1)) - 1) as f32).to_bits());
                                self.sel.op0("FMUL", vec![i, inv])
                            }
                            _ => f,
                        };
                        rgba[comp] = Some(v);
                    }
                    let one = self.konst(1.0f32.to_bits());
                    let comp0 = intr.get_const_index(NIR_INTRINSIC_COMPONENT) as usize;
                    let nc = intr.def.num_components as usize;
                    let comps: Vec<(u32, u8)> = (0..nc).map(|c| {
                        let c = c + comp0;
                        (rgba[c].unwrap_or(if c == 3 { one } else { zero }), 0u8)
                    }).collect();
                    self.sel.vec_map.insert(intr.def.index, comps);
                }
                _ if self.flags.half() => {
                    let nch = self.flags.half_channels();
                    let (rgba, zero) = if let Some(c) = self.dst_half { c } else {
                        let w0 = self.sel.op0("TLD", vec![]);
                        let w1 = if nch > 2 { self.sel.op0("TLD1", vec![]) } else { w0 };
                        let zero = self.sel.op0("ISUB", vec![w0, w0]);
                        let mut rgba: [Option<u32>; 4] = [None; 4];
                        for k in 0..nch {
                            let mut h = if k < 2 { w0 } else { w1 };
                            if k % 2 == 1 {
                                let c16 = self.konst(16);
                                h = self.sel.op0("ISRL", vec![h, c16]);
                            }
                            rgba[k] = Some(self.sel.op0("H2F", vec![h]));
                        }
                        self.dst_half = Some((rgba, zero));
                        (rgba, zero)
                    };
                    let one = self.konst(1.0f32.to_bits());
                    let comp0 = intr.get_const_index(NIR_INTRINSIC_COMPONENT) as usize;
                    let nc = intr.def.num_components as usize;
                    let comps: Vec<(u32, u8)> = (0..nc).map(|c| {
                        let c = c + comp0;
                        (rgba[c].unwrap_or(if c == 3 { one } else { zero }), 0u8)
                    }).collect();
                    self.sel.vec_map.insert(intr.def.index, comps);
                }
                _ => eprintln!("borgc: ERROR blending on this attachment format is not implemented"),
            }
        } else if intr.intrinsic == nir_intrinsic_load_output {
            let loc = intr.get_const_index(NIR_INTRINSIC_IO_SEMANTICS) & 0x7F;
            match self.outs.stores.get(&loc) {
                Some(v) => { self.sel.vec_map.insert(intr.def.index, v.clone()); }
                None => eprintln!(
                    "borgc: WARNING load_output of location {loc}, never stored earlier \
                     in program order -- dropped (draw mode resolves it from the store, \
                     not a real load; see decompose_vertex_offset's neighbor)"
                ),
            }
        }
    }
    /// A loop runs while EXANY sees an alive lane: `go` masks one trip, a jump clears it.
    fn loop_mark(&mut self, mnem: &str, id: usize, (alive, go, any): (u32, u32, u32)) {
        let one = if mnem == "LOOP_HEAD" { self.konst(1) } else { 0 };
        let i = |mnem: &'static str, dst: u32, srcs: Vec<u32>, swz: Vec<u8>| BorgInstr { mnem, dst, srcs, swz };
        let seq = match mnem {
            "LOOP_HEAD" => vec![
                i("IOR", alive, vec![one, one], vec![0, 0]),
                i("LABEL", NO_DST, vec![], vec![id as u8]),
                i("IOR", go, vec![alive, alive], vec![0, 0]),
                i("EXPUSH", NO_DST, vec![go], vec![0]),
            ],
            "LOOP_END" => vec![
                i("EXPOP", NO_DST, vec![], vec![]),
                i("EXPUSH", NO_DST, vec![alive], vec![0]),
                i("EXANY", any, vec![], vec![]),
                i("EXPOP", NO_DST, vec![], vec![]),
                // The branch names its LABEL in its second swizzle slot.
                i("BRNZ", NO_DST, vec![any], vec![0, id as u8]),
            ],
            _ => vec![i("EXPUSH", NO_DST, vec![go], vec![0])],
        };
        self.sel.prog.extend(seq);
    }

    /// Selects Borg instructions for the entry function in virtual registers, one NIR instruction
    /// at a time.
    pub unsafe fn select_entry(&mut self, entry: *mut nir_function_impl) {
        let mut cf_marks: std::collections::HashMap<usize, Vec<CfMark>> =
            std::collections::HashMap::new();
        let mut cf_unsupported: Vec<&'static str> = Vec::new();
        let mut loops = crate::Loops::default();
        collect_cf_marks((*entry).iter_body(), &mut cf_marks, &mut cf_unsupported, self.is_compute.then_some(&mut loops), None);
        if loops.count > 256 {
            loops.err = Some(format!("{} loops; labels are 8 bits", loops.count));
        }
        if let Some(e) = loops.err.take() {
            self.cs.err.get_or_insert(e);
        }
        // Per loop: lanes still looping, lanes going round this time, and whether any lane is.
        let lv: Vec<(u32, u32, u32)> = (0..loops.count).map(|_| (self.sel.fresh(), self.sel.fresh(), self.sel.fresh())).collect();
        if !cf_unsupported.is_empty() {
            eprintln!(
                "borgc: WARNING unsupported control flow ({}) -- the body will be \
                 emitted UNMASKED and run for every lane. See BorgCore.wireExecMask.",
                cf_unsupported.join(", ")
            );
        }

        let mut reg_vregs: HashMap<u32, Vec<u32>> = HashMap::new();
        for block in (*entry).iter_blocks() {
            if let Some(ms) = cf_marks.get(&(block as *const nir_block as usize)) {
                for m in ms {
                    if let Some(id) = m.lp {
                        self.loop_mark(m.mnem, id, lv[id]);
                        continue;
                    }
                    let cond = m.cond.map(|c| match self.consts.get(&c) {
                        Some(&b) if !self.sel.vec_map.contains_key(&c) => (self.konst(b), 0),
                        _ => resolve_vm(&self.sel.vec_map, c, 0),
                    });
                    self.sel.prog.push(BorgInstr {
                        mnem: m.mnem,
                        dst: NO_DST,
                        srcs: cond.map(|c| vec![c.0]).unwrap_or_default(),
                        swz: cond.map(|c| vec![c.1]).unwrap_or_default(),
                    });
                }
            }
            for instr in block.iter_instr_list() {
                if let Some(j) = instr.as_jump() {
                    if let Some(&id) = loops.jumps.get(&(block as *const nir_block as usize)) {
                        let (alive, go, _) = lv[id];
                        let zero = self.konst(0);
                        let clear = if j.type_ == nir_jump_break { vec![alive, go] } else { vec![go] };
                        for d in clear {
                            self.sel.prog.push(BorgInstr { mnem: "IOR", dst: d, srcs: vec![zero, zero], swz: vec![0, 0] });
                        }
                    }
                    continue;
                }
                if let Some(intr) = instr.as_intrinsic() {
                    if intr.intrinsic == nir_intrinsic_decl_reg {
                        let n = intr.get_const_index(NIR_INTRINSIC_NUM_COMPONENTS);
                        let v: Vec<u32> = (0..n).map(|_| { let r = self.sel.next_vreg; self.sel.next_vreg += 1; r }).collect();
                        reg_vregs.insert(intr.def.index, v);
                        continue;
                    }
                    if intr.intrinsic == nir_intrinsic_load_reg {
                        let h = intr.get_src(0).as_def().index;
                        let comps: Vec<(u32, u8)> = reg_vregs[&h].iter().map(|&r| (r, 0u8)).collect();
                        self.sel.vec_map.insert(intr.def.index, comps);
                        continue;
                    }
                    if intr.intrinsic == nir_intrinsic_store_reg {
                        let h = intr.get_src(1).as_def().index;
                        let src = intr.get_src(0).as_def().index;
                        for c in 0..intr.get_src(0).num_components() as usize {
                            if intr.write_mask() & (1 << c) == 0 {
                                continue;
                            }
                            let (v, comp) = self.operand(src, c as u8);
                            // A compute value is often an integer: copy its bits.
                            let (mnem, srcs, swz) = if self.is_compute { ("IOR", vec![v, v], vec![comp, comp]) } else { ("FMOV", vec![v], vec![comp]) };
                            self.sel.prog.push(BorgInstr { mnem, dst: reg_vregs[&h][c], srcs, swz });
                        }
                        continue;
                    }
                }
                if let Some(alu) = instr.as_alu() {
                    self.alu(alu);
                } else if let Some(intr) = instr.as_intrinsic() {
                    self.intrinsic(intr);
                } else if let Some(tex) = instr.as_tex() {
                    if self.is_compute {
                        self.skipped.push("tex");
                        continue;
                    }
                    self.tex(tex);
                } else if let Some(lc) = instr.as_load_const() {
                    self.load_const(lc);
                } else if let Some(u) = instr.as_undef().filter(|_| self.is_compute) {
                    self.undef(u);
                }
            }
        }
    
    }
    /// Gives each fragment output a register of its own where the value lives elsewhere (a
    /// uniform, a TEX or FATTR result block), so it can be placed in the output register.
    pub unsafe fn copy_frag_outputs(&mut self) {
    let mut seen = std::collections::HashSet::new();
    for c in 0..4 {
        let Some((v, comp)) = self.outs.frag[c] else { continue };
        // A TEX result block (r20-r23) or FATTR block is not the output register: copy that too;
        // so is a value that already is another channel's output.
        if seen.insert(v) && self.sel.prog.iter().any(|i| i.dst == v) && !self.tex_dsts.contains(&v) && !self.fattr_dsts.contains(&v) {
            continue;
        }
        let out = self.sel.next_vreg;
        self.sel.next_vreg += 1;
        if self.tex_dsts.contains(&v) || self.fattr_dsts.contains(&v) {
            self.sel.prog.push(BorgInstr { mnem: "IOR", dst: out, srcs: vec![v, v], swz: vec![comp, comp] });
        } else {
            self.sel.prog.push(BorgInstr { mnem: "FMOV", dst: out, srcs: vec![v], swz: vec![comp] });
        }
        self.outs.frag[c] = Some((out, 0));
        self.out_roots.push(out);
    }
    
    }
    /// Fragment depth as the barycentric-weighted sum of the three vertex depths, which arrive in
    /// uniform slots 30, 29 and 28. None when the shader has no weights.
    pub unsafe fn interpolate_depth(&mut self) -> Option<u32> {
    let mut frag_z: Option<u32> = None;
    if let Some(w) = self.weights {
        let uz: Vec<u32> = [30u8, 29, 28]
            .iter()
            .map(|&idx| {
                let v = self.sel.next_vreg;
                self.sel.next_vreg += 1;
                self.sel.ubo.insert(v, Ubo::Uniform(idx));
                v
            })
            .collect();
        let t0 = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "FMUL", dst: t0, srcs: vec![w[0], uz[0]], swz: vec![0, 0] });
        let t1 = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: t1, srcs: vec![w[1], uz[1], t0], swz: vec![0, 0, 0] });
        let zr = self.sel.next_vreg; self.sel.next_vreg += 1;
        self.sel.prog.push(BorgInstr { mnem: "FMADD", dst: zr, srcs: vec![w[2], uz[2], t1], swz: vec![0, 0, 0] });
        self.out_roots.push(zr);
        frag_z = Some(zr);
    }
    
    frag_z
    }
    pub fn log_uniformity(&self) {
    let uniform_roots: std::collections::HashSet<u32> = self.sel.ubo
        .iter()
        .filter_map(|(&v, u)| match u {
            Ubo::Uniform(_) => Some(v),
            Ubo::Fixed(_) if !self.per_pixel_fixed.contains(&v) => Some(v),
            _ => None,
        })
        .collect();
    let uniform = classify_uniform(&self.sel.prog, &uniform_roots);
    if env::var("BORGC_DUMP_ISA").is_ok() {
        let n_uniform = self.sel.prog.iter().filter(|i| uniform.contains(&i.dst)).count();
        eprintln!(
            "borgc: uniformity — {} of {} instr(s) are primitive-uniform (hoistable)",
            n_uniform,
            self.sel.prog.len()
        );
        for i in &self.sel.prog {
            if uniform.contains(&i.dst) {
                let s: Vec<String> = i.srcs.iter().map(|v| format!("v{v}")).collect();
                eprintln!("borgc:   [uniform] {} v{}, {}", i.mnem, i.dst, s.join(", "));
            }
        }
    }
    
    }
}

